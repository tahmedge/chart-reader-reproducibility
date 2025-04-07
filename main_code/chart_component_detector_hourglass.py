import json
import os
import math
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T # Keep standard transforms for basic image loading
import traceback # For detailed error printing
import argparse # For selecting train/infer mode


def gaussian_2d(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    # Avoid division by zero if sigma is too small
    sigma = max(sigma, 1e-6)
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    # Ensure diameter is positive
    if diameter <= 0:
        return
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1]) # Ensure integer coordinates
    height, width = heatmap.shape[0:2]

    # Determine the bounds for slicing the heatmap and the gaussian
    left, right = min(x, radius), min(width - 1 - x, radius) # Adjust right bound calculation
    top, bottom = min(y, radius), min(height - 1 - y, radius) # Adjust bottom bound calculation

    # Ensure bounds are non-negative
    left, right, top, bottom = max(0, left), max(0, right), max(0, top), max(0, bottom)

    # Calculate indices for slicing heatmap and gaussian
    heatmap_y_slice = slice(y - top, y + bottom + 1)
    heatmap_x_slice = slice(x - left, x + right + 1)

    # Gaussian slice needs adjustment based on potential heatmap clipping
    gaussian_y_start = radius - top
    gaussian_y_end = radius + bottom + 1
    gaussian_x_start = radius - left
    gaussian_x_end = radius + right + 1

    # Ensure gaussian indices are within bounds [0, diameter)
    gaussian_y_slice = slice(max(0, gaussian_y_start), min(diameter, gaussian_y_end))
    gaussian_x_slice = slice(max(0, gaussian_x_start), min(diameter, gaussian_x_end))

    # Get the slices
    try:
        masked_heatmap = heatmap[heatmap_y_slice, heatmap_x_slice]
        masked_gaussian = gaussian[gaussian_y_slice, gaussian_x_slice]
    except IndexError:
        return

    # Ensure shapes match
    if masked_heatmap.shape == masked_gaussian.shape and masked_heatmap.size > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)


def gaussian_radius(det_size, min_overlap):
    height, width = det_size
    height = max(1, height) # Ensure non-zero dims
    width = max(1, width)
    min_overlap = max(1e-6, min(min_overlap, 1 - 1e-6)) # Clamp overlap safely

    r1, r2, r3 = float('inf'), float('inf'), float('inf')

    # Eq 1
    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap) if abs(1 + min_overlap) > 1e-6 else float('inf')
    delta1 = b1 ** 2 - 4 * a1 * c1
    if delta1 >= 0:
        sq1 = np.sqrt(delta1)
        r1 = (b1 - sq1) / (2 * a1) if abs(2 * a1) > 1e-6 else float('inf')

    # Eq 2
    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    delta2 = b2 ** 2 - 4 * a2 * c2
    if delta2 >= 0:
        sq2 = np.sqrt(delta2)
        r2 = (b2 - sq2) / (2 * a2) if abs(2 * a2) > 1e-6 else float('inf')

    # Eq 3
    a3  = 4 * min_overlap
    if abs(a3) > 1e-8:
        b3  = -2 * min_overlap * (height + width)
        c3  = (min_overlap - 1) * width * height
        delta3 = b3 ** 2 - 4 * a3 * c3
        if delta3 >= 0:
            sq3 = np.sqrt(delta3)
            r3 = (b3 + sq3) / (2 * a3)

    valid_rs = [r for r in [r1, r2, r3] if r is not None and np.isreal(r) and np.isfinite(r) and r >= 0]
    if not valid_rs:
        return 1

    # Avoid overly large radius
    max_allowed_radius = min(height, width) / 2.0 if min(height, width) > 0 else 1.0
    valid_rs = [r for r in valid_rs if r <= max_allowed_radius]
    if not valid_rs:
        return 1

    # Return smallest valid radius
    return max(0, min(valid_rs))


def bad_p(x, y, output_size):
    # Check if point (x, y) is outside the heatmap bounds [0, H-1], [0, W-1]
    H, W = output_size
    return x < 0 or y < 0 or x >= W or y >= H

def _clip_coords(coords, shape):
    # Clip x coordinates to [0, width-1]
    coords[0::2] = np.clip(coords[0::2], 0, shape[1] - 1 if shape[1] > 0 else 0)
    # Clip y coordinates to [0, height-1]
    coords[1::2] = np.clip(coords[1::2], 0, shape[0] - 1 if shape[0] > 0 else 0)
    return coords


def normalize_(image, mean, std):
    std_safe = np.where(np.abs(std) < 1e-6, 1e-6, std)
    image -= mean
    image /= std_safe
    return image


def gather_feature(fmap, index, mask=None, use_index=True):
    """
    Gathers features from a feature map at specified indices.
    Args:
        fmap (torch.Tensor): Feature map of shape [B, C, H, W].
        index (torch.Tensor): LongTensor of indices, shape [B, N].
                                 Indices are flattened (y * W + x).
        mask (torch.Tensor, optional): BoolTensor mask, shape [B, N]. Defaults to None.
        use_index (bool, optional): Whether indices represent flattened spatial locations.
                                   Defaults to True.
    Returns:
        torch.Tensor: Gathered features, potentially masked, shape [Num_Valid, C] if mask else [B, N, C].
    """
    B, C, H, W = fmap.shape
    N = index.shape[1]
    index = index.unsqueeze(2).expand(B, N, C) # [B, N, C]
    fmap = fmap.view(B, C, H * W).permute(0, 2, 1) # [B, H*W, C]
    # Ensure index is within bounds
    index = torch.clamp(index, 0, H*W - 1)
    output = fmap.gather(1, index) # [B, N, C]

    if mask is not None:
        if mask.dim() == 1: # If mask is flattened
            raise ValueError("Flattened mask is not supported for batched feature gathering.")
        elif mask.shape != (B, N):
            raise ValueError(f"Mask shape {mask.shape} does not match index shape ({B}, {N})")

        mask_expanded = mask.unsqueeze(2).expand_as(output) # [B, N, C]
        output = output[mask_expanded]
        output = output.reshape(-1, C)
    return output

def smooth_l1_loss_masked(pred, target, mask, reduction='mean'):
    """Calculate Smooth L1 Loss only for masked elements."""
    if not mask.any(): # If no elements are masked, return 0 loss
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    mask = mask.unsqueeze(-1).expand_as(pred).float()
    loss = F.smooth_l1_loss(pred * mask, target * mask, reduction='sum')
    num_valid_elements = mask.sum()
    if reduction == 'mean':
        loss = loss / (num_valid_elements + 1e-6)
    elif reduction == 'sum':
        pass # Keep the sum
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")
    return loss

def focal_loss(preds, targets, alpha=2.0, beta=4.0):

    # Assumes targets are heatmaps with values potentially < 1 (e.g., Gaussian)
    preds = torch.clamp(torch.sigmoid(preds), min=1e-4, max=1 - 1e-4)
    pos_inds = targets.eq(1).float() # Strictly 1 for pos
    neg_inds = targets.lt(1).float() # Less than 1 for neg

    neg_weights = torch.pow(1 - targets, beta) # Down-weight negatives near positives

    pos_loss = -torch.log(preds) * torch.pow(1 - preds, alpha) * pos_inds
    neg_loss = -torch.log(1 - preds) * torch.pow(preds, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    # Avoid division by zero if no positive samples found
    if num_pos == 0:
        loss = neg_loss
    else:
        loss = (pos_loss + neg_loss) / num_pos
    return loss

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False), # Bias false common before BN
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return F.relu(x + self.conv(x))

class HourglassNet(nn.Module):
    def __init__(self, in_channels=3, num_centers=3, num_keypoints=3, hourglass_feature_dim=256): # num_centers/keypoints should match dataset categories=3
        super().__init__()
        self.hourglass_feature_dim = hourglass_feature_dim
        self.num_centers = num_centers
        self.num_keypoints = num_keypoints

        # Downsampling path
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)
        self.res1 = ResidualBlock(64)
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.res2 = ResidualBlock(128)
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.res3 = ResidualBlock(256)

        # Upsampling path
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.res4 = ResidualBlock(128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.res5 = ResidualBlock(64)

        # Output heads
        self.center_head = nn.Conv2d(64, self.num_centers, kernel_size=1)
        self.keypoint_head = nn.Conv2d(64, self.num_keypoints, kernel_size=1)
        # Adding regression heads for center/keypoint offsets
        self.center_regr_head = nn.Conv2d(64, 2, kernel_size=1)
        self.keypoint_regr_head = nn.Conv2d(64, 2, kernel_size=1)
        self.feature_map_conv = nn.Conv2d(64, self.hourglass_feature_dim, kernel_size=1) # For final feature map

    def forward(self, x):
        # Downsampling
        d1 = self.down1(x)
        d2 = self.pool1(d1)
        d2 = self.res1(d2)
        d3 = self.down2(d2)
        d3 = self.res2(d3)
        d4 = self.down3(d3)
        d4 = self.res3(d4)

        # Upsampling + skip
        u2 = self.up2(d4)
        if u2.shape[2:] != d3.shape[2:]:
            u2 = F.interpolate(u2, size=d3.shape[2:], mode='bilinear', align_corners=True)
        u2 = u2 + d3
        u2 = self.res4(u2)

        u1 = self.up1(u2)
        if u1.shape[2:] != d2.shape[2:]:
            u1 = F.interpolate(u1, size=d2.shape[2:], mode='bilinear', align_corners=True)
        u1 = u1 + d2
        final_feat = self.res5(u1)

        # Predictions
        center_heatmap = self.center_head(final_feat)
        keypoint_heatmap = self.keypoint_head(final_feat)
        center_regr = self.center_regr_head(final_feat) # Predict center offsets
        keypoint_regr = self.keypoint_regr_head(final_feat) # Predict keypoint offsets
        output_feature_map = self.feature_map_conv(final_feat)

        return {
            'center_heatmap': center_heatmap,           # [B, C, H, W]
            'keypoint_heatmap': keypoint_heatmap,       # [B, K, H, W]
            'center_regr': center_regr,                 # [B, 2, H, W]
            'keypoint_regr': keypoint_regr,             # [B, 2, H, W]
            'output_feature_map': output_feature_map    # [B, D_hg, H, W]
        }


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_h=128, max_w=128): # Max H/W should match heatmap size
        super().__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D sinusoidal encoding"
        self.d_model = d_model
        self.max_h = max_h
        self.max_w = max_w

        # Calculate the division term (frequencies) correctly
        d_half = d_model // 2
        if d_half == 0: # Avoid division by zero if d_model is small
             div_term = torch.zeros(0)
        else:
             div_term = torch.exp(torch.arange(0, d_half, 2).float() * (-math.log(10000.0) / d_half))

        # --- Precompute Positional Encodings for Y dimension ---
        pe_y = torch.zeros(max_h, d_half) # Shape: [max_h, d_model / 2]
        position_y = torch.arange(0, max_h, dtype=torch.float).unsqueeze(1) # Shape: [max_h, 1]

        if div_term.numel() > 0: # Check if div_term is not empty
            pe_y[:, 0::2] = torch.sin(position_y * div_term)
            if d_half % 2 == 0: # Even d_half
                pe_y[:, 1::2] = torch.cos(position_y * div_term)
            else: # Odd d_half - need to use same freqs for last cos
                 if d_half > 1:
                      pe_y[:, 1::2] = torch.cos(position_y * div_term[:-1]) # Use freqs matching sin

        # --- Precompute Positional Encodings for X dimension ---
        pe_x = torch.zeros(max_w, d_half) # Shape: [max_w, d_model / 2]
        position_x = torch.arange(0, max_w, dtype=torch.float).unsqueeze(1) # Shape: [max_w, 1]

        if div_term.numel() > 0:
            pe_x[:, 0::2] = torch.sin(position_x * div_term)
            if d_half % 2 == 0:
                pe_x[:, 1::2] = torch.cos(position_x * div_term)
            else:
                 if d_half > 1:
                      pe_x[:, 1::2] = torch.cos(position_x * div_term[:-1])


        self.register_buffer('pe_y', pe_y)
        self.register_buffer('pe_x', pe_x)

    def forward(self, coords):
        """
        Args:
            coords (torch.Tensor): Coordinates of shape [B, N, 2] (x, y).
        Returns:
            torch.Tensor: Positional encodings of shape [B, N, d_model].
        """
        B, N, _ = coords.shape
        # Clamp coordinates safely
        max_w_idx = self.max_w - 1
        max_h_idx = self.max_h - 1
        if max_w_idx < 0 or max_h_idx < 0: # Handle cases where max_h/w is 0 or 1
             return torch.zeros(B, N, self.d_model, device=coords.device, dtype=coords.dtype)

        # Ensure coordinates are within the bounds of precomputed encodings
        coords_x = torch.clamp(coords[..., 0].round(), 0, max_w_idx).long() # Shape [B, N]
        coords_y = torch.clamp(coords[..., 1].round(), 0, max_h_idx).long() # Shape [B, N]

        # Gather embeddings
        emb_x = self.pe_x[coords_x] # Indexing [B, N] into [max_w, d_half] -> [B, N, d_half]
        emb_y = self.pe_y[coords_y] # Indexing [B, N] into [max_h, d_half] -> [B, N, d_half]

        pos_encoding = torch.cat((emb_x, emb_y), dim=-1) # Shape [B, N, d_model]
        return pos_encoding


def soft_argmax2d(heatmap, temperature=1.0, normalize_coords=False):
    """
    Computes the soft argmax of a heatmap.
    Args:
        heatmap (torch.Tensor): Input heatmap tensor of shape [B, C, H, W].
        temperature (float): Temperature parameter for softmax. Default 1.0.
        normalize_coords (bool): If True, normalize coords to [-1, 1]. Default False.
    Returns:
        torch.Tensor: Coordinates tensor of shape [B, C, 2] (x, y).
    """
    B, C, H, W = heatmap.shape

    # Handle potential NaN/Inf in heatmap before softmax
    heatmap_float = torch.nan_to_num(heatmap.float())

    # Apply temperature and softmax over spatial dimensions
    heatmap_flat = heatmap_float.view(B, C, -1) # Flatten spatial dimensions (H*W)

    # Ensure temperature is positive
    temperature = max(temperature, 1e-6)
    softmax_probs = F.softmax(heatmap_flat / temperature, dim=-1) # Shape: [B, C, H*W]

    # Create coordinate grids
    device = heatmap.device
    dtype = heatmap_float.dtype
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij' # Use matrix indexing ('ij')
    )
    grid_y_flat = grid_y.reshape(-1) # Flatten the y grid [H*W]
    grid_x_flat = grid_x.reshape(-1) # Flatten the x grid [H*W]

    # Calculate the expected coordinates (weighted average)
    # Unsqueeze grid_x/y_flat to broadcast correctly: [1, 1, H*W]
    exp_x = torch.sum(softmax_probs * grid_x_flat.unsqueeze(0).unsqueeze(0), dim=-1) # Expected x-coordinate [B, C]
    exp_y = torch.sum(softmax_probs * grid_y_flat.unsqueeze(0).unsqueeze(0), dim=-1) # Expected y-coordinate [B, C]

    # Stack the expected x and y coordinates
    coords = torch.stack([exp_x, exp_y], dim=-1) # Shape: [B, C, 2]

    # Optional normalization to [-1, 1]
    if normalize_coords:
        coords[..., 0] = (coords[..., 0] / (W - 1)) * 2 - 1 if W > 1 else torch.zeros_like(coords[..., 0])
        coords[..., 1] = (coords[..., 1] / (H - 1)) * 2 - 1 if H > 1 else torch.zeros_like(coords[..., 1])

    return coords

class ChartComponentDetector(nn.Module):
    def __init__(self, num_types=7, num_centers=3, num_keypoints=3,
                 feature_dim=256, hourglass_feature_dim=256, mha_heads=4,
                 heatmap_h=128, heatmap_w=128, max_centers=64, max_keypoints=128):
        super().__init__()
        self.num_types = num_types # Final classification types (e.g., original 7)
        self.num_centers = num_centers # Heatmap channels (box, line, pie)
        self.num_keypoints = num_keypoints # Heatmap channels (box, line, pie)
        self.feature_dim = feature_dim # Dimension for attention/MLP
        self.hourglass_feature_dim = hourglass_feature_dim # Dimension from Hourglass
        self.heatmap_h = heatmap_h
        self.heatmap_w = heatmap_w
        self.max_centers = max_centers
        self.max_keypoints = max_keypoints

        self.hourglass = HourglassNet(
            in_channels=3, # Added this explicitly, assumed 3 before
            num_centers=self.num_centers,
            num_keypoints=self.num_keypoints,
            hourglass_feature_dim=self.hourglass_feature_dim
        )

        self.pos_encoder = PositionalEncoding2D(self.hourglass_feature_dim, max_h=heatmap_h, max_w=heatmap_w)

        self.center_type_emb = nn.Embedding(self.num_centers + 1, self.hourglass_feature_dim, padding_idx=0) # Assume 0 is padding
        self.keypoint_type_emb = nn.Embedding(self.num_keypoints + 1, self.hourglass_feature_dim, padding_idx=0)

        combined_feature_dim = self.hourglass_feature_dim

        # Linear layers for attention (Eq. 1) - Project to feature_dim
        self.Wc = nn.Linear(combined_feature_dim, self.feature_dim)
        self.Wk = nn.Linear(combined_feature_dim, self.feature_dim)

        # MLP for final type prediction (Eq. 6) - Input dim matches Wk output (feature_dim)
        self.type_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim // 2, self.num_types) # Predict final type (e.g., 7 original types)
        )

    def _predict_keypoint_locations(self, keypoint_heatmap_pred, keypoint_regr_pred_map):
        """
        Predicts keypoint locations using heatmap peaks and regression offsets.
        Args:
            keypoint_heatmap_pred: [B, K, H, W]
            keypoint_regr_pred_map: [B, 2, H, W] (Shared offset map for all K types)
        Returns:
            loc_pk_pred: [B, K, 2] Predicted float locations (x, y) for each keypoint type's peak.
        """
        B, K, H, W = keypoint_heatmap_pred.shape

        # Get integer locations using soft-argmax for each channel K
        # Use a low temperature to approximate hard argmax
        # soft_argmax2d input: [B, K, H, W], output: [B, K, 2]
        int_coords = soft_argmax2d(keypoint_heatmap_pred, temperature=0.1, normalize_coords=False) # [B, K, 2] (x, y)

        # Clamp coordinates to be within bounds for gathering offsets
        int_coords_x = torch.clamp(int_coords[..., 0].round(), 0, W - 1).long() # [B, K]
        int_coords_y = torch.clamp(int_coords[..., 1].round(), 0, H - 1).long() # [B, K]

        # Gather predicted offsets at these integer locations
        # Need to gather from shared offset map [B, 2, H, W] using indices [B, K]
        pred_offsets = torch.zeros(B, K, 2, device=keypoint_heatmap_pred.device, dtype=keypoint_heatmap_pred.dtype)
        for k in range(K):
            idx_x_k = int_coords_x[:, k] # [B]
            idx_y_k = int_coords_y[:, k] # [B]

            flat_indices_k = idx_y_k * W + idx_x_k # [B]

            regr_offset_map_flat = keypoint_regr_pred_map.permute(0, 2, 3, 1).reshape(B, H * W, 2) # [B, H*W, 2]

            # Ensure index is within bounds before expanding and gathering
            flat_indices_k_clamped = torch.clamp(flat_indices_k, 0, H*W -1) # Clamp here [B]
            flat_indices_k_expanded = flat_indices_k_clamped.unsqueeze(-1).expand(B, 2) # [B, 2] for gathering both dx, dy

            gathered_offsets_k = regr_offset_map_flat.gather(1, flat_indices_k_expanded.unsqueeze(1)).squeeze(1) # [B, 2]
            pred_offsets[:, k, :] = gathered_offsets_k

        # Calculate final float locations: integer location + predicted offset
        float_coords_x = int_coords_x.float() + pred_offsets[..., 0]
        float_coords_y = int_coords_y.float() + pred_offsets[..., 1]
        loc_pk_pred = torch.stack([float_coords_x, float_coords_y], dim=-1) # [B, K, 2]

        return loc_pk_pred

    def forward(self, image_tensor, key_tags_gt=None, center_tags_gt=None, tag_lens_keys=None, tag_lens_cens=None, center_heatmap_types_gt=None, key_heatmap_types_gt=None):
        """
        Forward pass for training. Uses GT tags/types to gather features for loss calculation.
        Args: (All tensors should be on the correct device)
            image_tensor: [B, 3, H_in, W_in]
            key_tags_gt: [B, max_keypoints] (flattened indices)
            center_tags_gt: [B, max_centers] (flattened indices)
            tag_lens_keys: [B]
            tag_lens_cens: [B]
            center_heatmap_types_gt: [B, max_centers] (Indices 0,1,2 for heatmap type)
            key_heatmap_types_gt: [B, max_keypoints] (Indices 0,1,2 for heatmap type)
        Returns:
            Dictionary containing predictions needed for loss calculation.
            OR Dictionary containing hourglass outputs if GT args are None (basic inference).
        """
        # --- Step 1: Get Heatmaps & Features from Hourglass ---
        outputs = self.hourglass(image_tensor)
        center_heatmap_pred = outputs['center_heatmap']     # [B, C, H_map, W_map] (C=3)
        keypoint_heatmap_pred = outputs['keypoint_heatmap'] # [B, K, H_map, W_map] (K=3)
        center_regr_pred_map = outputs['center_regr']       # [B, 2, H_map, W_map]
        keypoint_regr_pred_map = outputs['keypoint_regr']   # [B, 2, H_map, W_map]
        feature_map = outputs['output_feature_map']         # [B, D_hg, H_map, W_map]
        B, D_hg, H_map, W_map = feature_map.shape

        # --- If GT inputs are not provided, return hourglass outputs (for basic inference) ---
        # This allows using model.forward() for basic inference without needing a separate path
        if not all(x is not None for x in [center_tags_gt, key_tags_gt, center_heatmap_types_gt, key_heatmap_types_gt, tag_lens_cens, tag_lens_keys]):
             # print("Debug: Running in basic inference mode (forward pass)")
             return outputs # Return dict from hourglass


        # --- Step 2: Gather Features & Prepare Inputs for Grouping ---
        center_indices = torch.arange(self.max_centers, device=center_tags_gt.device).unsqueeze(0).expand(B, -1)
        center_masks_gt = (center_indices < tag_lens_cens.unsqueeze(1)).bool()

        key_indices = torch.arange(self.max_keypoints, device=key_tags_gt.device).unsqueeze(0).expand(B, -1)
        key_masks_gt = (key_indices < tag_lens_keys.unsqueeze(1)).bool()

        # Gather raw features at GT locations
        center_feat_gt_flat = gather_feature(feature_map, center_tags_gt, mask=center_masks_gt)
        key_feat_gt_flat = gather_feature(feature_map, key_tags_gt, mask=key_masks_gt)

        # Convert GT tags to coordinates for positional encoding
        center_coords_y_gt = center_tags_gt // W_map
        center_coords_x_gt = center_tags_gt % W_map
        center_coords_gt = torch.stack([center_coords_x_gt, center_coords_y_gt], dim=-1).float()

        key_coords_y_gt = key_tags_gt // W_map
        key_coords_x_gt = key_tags_gt % W_map
        key_coords_gt = torch.stack([key_coords_x_gt, key_coords_y_gt], dim=-1).float()

        # Calculate positional encodings
        center_pos_enc = self.pos_encoder(center_coords_gt)
        key_pos_enc = self.pos_encoder(key_coords_gt)

        # Get type embeddings (Add 1 to type indices if embedding uses 0 for padding)
        # Assuming GT types are 0, 1, 2 and embedding uses padding_idx=0
        center_type_indices = torch.clamp(center_heatmap_types_gt + 1, 0, self.num_centers) # Map 0,1,2 -> 1,2,3
        key_type_indices = torch.clamp(key_heatmap_types_gt + 1, 0, self.num_keypoints)   # Map 0,1,2 -> 1,2,3

        center_type_emb = self.center_type_emb(center_type_indices)
        key_type_emb = self.keypoint_type_emb(key_type_indices)

        # Reshape flat features back to batched tensors & apply mask before combination
        center_feat_gt_batched = torch.zeros(B, self.max_centers, D_hg, device=feature_map.device, dtype=feature_map.dtype)
        if center_masks_gt.any(): center_feat_gt_batched[center_masks_gt] = center_feat_gt_flat

        key_feat_gt_batched = torch.zeros(B, self.max_keypoints, D_hg, device=feature_map.device, dtype=feature_map.dtype)
        if key_masks_gt.any(): key_feat_gt_batched[key_masks_gt] = key_feat_gt_flat

        # Combine features: Raw features + Positional Encoding + Type Embedding (Element-wise Add)
        # Apply mask before projection
        center_combined_feat = (center_feat_gt_batched + center_pos_enc + center_type_emb) * center_masks_gt.unsqueeze(-1).float()
        key_combined_feat = (key_feat_gt_batched + key_pos_enc + key_type_emb) * key_masks_gt.unsqueeze(-1).float()

        # --- Step 3: Calculate Grouping Scores (Attention) ---
        center_proj = self.Wc(center_combined_feat) # [B, max_centers, D_feat]
        key_proj = self.Wk(key_combined_feat)       # [B, max_keypoints, D_feat]

        grouping_scores_G = torch.bmm(center_proj, key_proj.transpose(1, 2)) # [B, max_centers, max_keypoints]

        # Normalize with softmax over keypoints
        # Create a valid key mask for attention [B, 1, max_keypoints]
        attn_key_mask = key_masks_gt.unsqueeze(1)
        grouping_scores_G.masked_fill_(~attn_key_mask, -float('inf')) # Mask invalid keypoints

        attn_weights = F.softmax(grouping_scores_G, dim=2) # [B, max_centers, max_keypoints]
        attn_weights = attn_weights * center_masks_gt.unsqueeze(-1).float() # Zero weights for invalid centers
        # Handle cases where a center has no valid keypoints after masking (attn_weights sum to 0 -> NaN)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)


        # --- Step 4: Predict Keypoint Locations (\hat{loc}_pk) ---
        # Use GT keypoint integer locations + predicted offsets at those locations for training loss.
        keypoint_regr_pred_gathered_flat = gather_feature(keypoint_regr_pred_map, key_tags_gt, mask=key_masks_gt)
        keypoint_regr_pred_gathered = torch.zeros(B, self.max_keypoints, 2, device=feature_map.device, dtype=feature_map.dtype)
        if key_masks_gt.any(): # Avoid error if mask is all False
             keypoint_regr_pred_gathered[key_masks_gt] = keypoint_regr_pred_gathered_flat

        # loc_pk_pred_training = key_coords_gt + keypoint_regr_pred_gathered # [B, max_keypoints, 2] (Using float GT coord)
        # Use integer part of GT coords + predicted offset (as done in dataset gen usually)
        loc_pk_pred_training = key_coords_gt.round() + keypoint_regr_pred_gathered


        # --- Step 5: Calculate Intermediate Predictions for L_loc and L_CLS ---
        # Predict Center Location  using weighted predicted keypoint locations (Eq. 3)
        loc_pc_pred = torch.bmm(attn_weights, loc_pk_pred_training) # [B, max_centers, 2]

        # Predict Center Embedding using weighted keypoint features (Eq. 5)
        # key_proj is Wk applied to combined features
        h_pc_pred_embedding = torch.bmm(attn_weights, key_proj) # [B, max_centers, D_feat]

        # Predict Final Component Type using MLP (Eq. 6)
        type_logits_pred = torch.zeros(B, self.max_centers, self.num_types, device=feature_map.device, dtype=feature_map.dtype)
        if center_masks_gt.any():
            h_pc_pred_embedding_flat = h_pc_pred_embedding[center_masks_gt]
            # Only pass valid embeddings (non-zero due to masking/attn) to classifier
            valid_embed_mask = h_pc_pred_embedding_flat.abs().sum(dim=-1) > 1e-6
            if valid_embed_mask.any():
                type_logits_pred_flat = torch.zeros(h_pc_pred_embedding_flat.shape[0], self.num_types, device=h_pc_pred_embedding_flat.device, dtype=h_pc_pred_embedding_flat.dtype)
                type_logits_pred_flat[valid_embed_mask] = self.type_classifier(h_pc_pred_embedding_flat[valid_embed_mask])
                type_logits_pred[center_masks_gt] = type_logits_pred_flat

        # --- Step 6: Gather Regression Predictions at GT Locations for Auxiliary Loss ---
        center_regr_pred_gathered_flat = gather_feature(center_regr_pred_map, center_tags_gt, mask=center_masks_gt)

        # Return values needed for loss calculation during training
        return {
            "center_heatmap_pred": center_heatmap_pred,
            "keypoint_heatmap_pred": keypoint_heatmap_pred,
            "center_regr_pred_gathered": center_regr_pred_gathered_flat,
            "keypoint_regr_pred_gathered": keypoint_regr_pred_gathered_flat,
            "loc_pc_pred": loc_pc_pred,
            "type_logits_pred": type_logits_pred,
        }

    def compute_loss(self, preds, targets_ys, inputs_xs):
        """
        Computes loss based on Section 3.1 of the ChartReader paper.
        Args:
            preds (dict): Output dictionary from the forward pass (training mode).
            targets_ys (list): List of ground truth tensors from the dataset.
                               Order (10 items): key_hm, cen_hm, key_mask, cen_mask,
                               key_regr, cen_regr, group_target, len_cen, len_key, cen_final_type
            inputs_xs (list): List of input tensors (contains masks/tags/types).
                               Order (10 items): image, key_tags, cen_tags, len_keys, len_cens,
                               cen_mask, key_mask, cen_final_type, cen_hm_type, key_hm_type
        """
        # --- Unpack targets (ground truth for losses) ---
        (key_heatmaps_gt, center_heatmaps_gt, _, _, key_regrs_gt, center_regrs_gt,
         _, _, _, center_final_types_gt) = targets_ys # Unpack 10 items

        # --- Unpack necessary items from inputs_xs ---
        # image, key_tags, cen_tags, len_keys, len_cens, cen_mask, key_mask, cen_final_type_inp, cen_hm_type, key_hm_type = inputs_xs[:10]
        center_tags_gt = inputs_xs[2]
        center_masks_gt = inputs_xs[5].bool() # Ensure boolean
        key_masks_gt = inputs_xs[6].bool()    # Ensure boolean
        # center_final_types_gt = inputs_xs[7] # Use the one from targets_ys as it's aligned with max_centers


        # --- Unpack predictions ---
        center_heatmap_pred = preds.get("center_heatmap_pred")
        keypoint_heatmap_pred = preds.get("keypoint_heatmap_pred")
        loc_pc_pred = preds.get("loc_pc_pred")
        type_logits_pred = preds.get("type_logits_pred")
        center_regr_pred_gathered = preds.get("center_regr_pred_gathered")
        keypoint_regr_pred_gathered = preds.get("keypoint_regr_pred_gathered")

        if any(p is None for p in [center_heatmap_pred, keypoint_heatmap_pred, loc_pc_pred, type_logits_pred, center_regr_pred_gathered, keypoint_regr_pred_gathered]):
            print("Warning: Missing predictions in compute_loss. Returning zero loss.")
            return {"total": torch.tensor(0.0, device=targets_ys[0].device), "hm": 0, "loc": 0, "cls": 0, "regr": 0}


        B, _, H_map, W_map = center_heatmap_pred.shape

        # 1. Heatmap Loss (L_focal)
        loss_center_heatmap = focal_loss(center_heatmap_pred, center_heatmaps_gt)
        loss_keypoint_heatmap = focal_loss(keypoint_heatmap_pred, key_heatmaps_gt)
        loss_hm = loss_center_heatmap + loss_keypoint_heatmap

        # 2. Location Loss (L_loc - Eq. 4)
        # Convert GT center tags to integer coordinates
        center_coords_y_gt_int = (center_tags_gt // W_map).long()
        center_coords_x_gt_int = (center_tags_gt % W_map).long()

        # Ensure GT regression tensors match max length expected by mask/preds
        if center_regrs_gt.shape[1] != self.max_centers:
            # This case should ideally not happen if collation is correct
             print(f"Warning: GT center regr shape mismatch {center_regrs_gt.shape} vs max_centers {self.max_centers}")
             # Pad or truncate GT - padding is safer if loader didn't pad
             padded_regr = torch.zeros(B, self.max_centers, 2, device=center_regrs_gt.device, dtype=center_regrs_gt.dtype)
             valid_len = min(center_regrs_gt.shape[1], self.max_centers)
             padded_regr[:, :valid_len, :] = center_regrs_gt[:, :valid_len, :]
             center_regrs_gt = padded_regr
        if key_regrs_gt.shape[1] != self.max_keypoints:
             print(f"Warning: GT key regr shape mismatch {key_regrs_gt.shape} vs max_keypoints {self.max_keypoints}")
             padded_regr = torch.zeros(B, self.max_keypoints, 2, device=key_regrs_gt.device, dtype=key_regrs_gt.dtype)
             valid_len = min(key_regrs_gt.shape[1], self.max_keypoints)
             padded_regr[:, :valid_len, :] = key_regrs_gt[:, :valid_len, :]
             key_regrs_gt = padded_regr
        if center_final_types_gt.shape[1] != self.max_centers:
             print(f"Warning: GT final types shape mismatch {center_final_types_gt.shape} vs max_centers {self.max_centers}")
             padded_types = torch.full((B, self.max_centers), -1, device=center_final_types_gt.device, dtype=center_final_types_gt.dtype)
             valid_len = min(center_final_types_gt.shape[1], self.max_centers)
             padded_types[:, :valid_len] = center_final_types_gt[:, :valid_len]
             center_final_types_gt = padded_types


        # Reconstruct float GT center locations using integer coords + GT offsets
        loc_pc_gt_x = center_coords_x_gt_int.float() + center_regrs_gt[..., 0] # [B, max_centers]
        loc_pc_gt_y = center_coords_y_gt_int.float() + center_regrs_gt[..., 1] # [B, max_centers]
        loc_pc_gt = torch.stack([loc_pc_gt_x, loc_pc_gt_y], dim=-1) # [B, max_centers, 2]

        # Calculate loss using the mask derived from input lengths
        loss_loc = smooth_l1_loss_masked(loc_pc_pred, loc_pc_gt, center_masks_gt, reduction='mean')

        # 3. Classification Loss (L_CLS - Eq. 6)
        # Flatten predictions and GT labels, apply mask
        type_logits_pred_flat = type_logits_pred[center_masks_gt] # [N_valid_cens, num_types]
        y_pc_gt_flat = center_final_types_gt[center_masks_gt]     # [N_valid_cens]

        loss_cls = torch.tensor(0.0, device=type_logits_pred.device)
        if y_pc_gt_flat.numel() > 0: # Only compute if there are valid centers
             # Filter out ignored labels (-1) which might exist even within masked elements
             valid_gt_indices = (y_pc_gt_flat != -1)
             if valid_gt_indices.any():
                 type_logits_pred_flat_valid = type_logits_pred_flat[valid_gt_indices]
                 y_pc_gt_flat_valid = y_pc_gt_flat[valid_gt_indices]

                 # Ensure labels are in range [0, num_types-1] before cross entropy
                 if ((y_pc_gt_flat_valid >= 0) & (y_pc_gt_flat_valid < self.num_types)).all():
                     loss_cls = F.cross_entropy(type_logits_pred_flat_valid, y_pc_gt_flat_valid, reduction='mean')

        # 4. Auxiliary Regression Loss for Hourglass Offsets (L_regr)
        # GT offsets need to be masked correctly based on the input masks
        center_regrs_gt_flat = center_regrs_gt[center_masks_gt] # [N_valid_cens, 2]
        key_regrs_gt_flat = key_regrs_gt[key_masks_gt]       # [N_valid_keys, 2]

        loss_regr_cen = torch.tensor(0.0, device=center_heatmap_pred.device)
        # Check if number of predicted elements matches number of GT elements after masking
        if center_regr_pred_gathered.numel() > 0 and center_regrs_gt_flat.numel() > 0:
             if center_regr_pred_gathered.shape == center_regrs_gt_flat.shape:
                 loss_regr_cen = F.smooth_l1_loss(center_regr_pred_gathered, center_regrs_gt_flat, reduction='mean')

        loss_regr_key = torch.tensor(0.0, device=center_heatmap_pred.device)
        if keypoint_regr_pred_gathered.numel() > 0 and key_regrs_gt_flat.numel() > 0:
             if keypoint_regr_pred_gathered.shape == key_regrs_gt_flat.shape:
                 loss_regr_key = F.smooth_l1_loss(keypoint_regr_pred_gathered, key_regrs_gt_flat, reduction='mean')

        loss_regr = loss_regr_cen + loss_regr_key

        # --- Combine Losses ---
        # Weights can be adjusted or passed via config
        w_hm, w_loc, w_cls, w_regr = 1.0, 1.0, 1.0, 0.1 # Example weights from paper/code
        total_loss = (w_hm * loss_hm) + (w_loc * loss_loc) + (w_cls * loss_cls) + (w_regr * loss_regr)

        # Return dict for logging
        return {
            "total": total_loss,
            "hm": loss_hm.detach(),
            "loc": loss_loc.detach(),
            "cls": loss_cls.detach(),
            "regr": loss_regr.detach(),
        }

class EC400KDatasetWithTargets(Dataset):
    def __init__(self, root_dir, annotation_file, image_set='train2019', config=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images', image_set)
        self.annotation_path = os.path.join(root_dir, 'annotations', annotation_file)

        # --- Configuration (took help from their original code repository) ---
        self.cfg = {
            "input_size": [511, 511],
            "output_sizes": [[128, 128]],
            "categories": 3, # Number of heatmap channels (box, line, pie)
            "category_mapping": { 1: 0, 4: 0, 5: 0, 6: 0, 7: 0, 2: 1, 3: 2 }, # Map original ID -> heatmap channel index (0, 1, 2)
            "type_mapping": {i: i-1 for i in range(1, 8)}, # Map original ID -> final type index (0-6 for L_CLS)
            "num_final_types": 7, # Number of final types for L_CLS (0-6)
            "gaussian_bump": False, "gaussian_iou": 0.3, "gaussian_radius": -1,
            "max_centers": 64, "max_keypoints": 128, "max_group_len": 100, # Max annotations per image
            "rand_crop": False, "rand_color": True, "lighting": True, "border": 128, # Augmentation flags (TODO: implement if True)
            "mean": np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3),
            "std": np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3),
            "eig_val": np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32),
            "eig_vec": np.array([[-0.58752847, -0.69563484, 0.41340352], [-0.5832747, 0.00994535, -0.81221408], [-0.56089297, 0.71832671, 0.41158938]], dtype=np.float32)
        }
        if config: self.cfg.update(config)

        self.output_size = self.cfg["output_sizes"][0]
        self.input_size = self.cfg["input_size"]
        self.max_centers = self.cfg["max_centers"]
        self.max_keypoints = self.cfg["max_keypoints"]
        self.max_group_len = self.cfg["max_group_len"]
        self.heatmap_h, self.heatmap_w = self.output_size
        self.num_heatmap_channels = self.cfg["categories"] # Explicitly store this

        # --- Load Annotations ---
        try:
            with open(self.annotation_path, 'r') as f: data = json.load(f)
        except Exception as e: raise IOError(f"Error loading annotation file {self.annotation_path}: {e}")

        self.images_info = data.get('images', [])
        self.annotations_data = data.get('annotations', [])
        # Filter annotations based on valid *original* category IDs that we map
        valid_original_ids = set(self.cfg["category_mapping"].keys())
        self.annotations_data = [anno for anno in self.annotations_data if anno.get('category_id') in valid_original_ids]

        # Group annotations by image ID
        self.annotations_per_image = {}
        for anno in self.annotations_data:
            img_id = anno.get('image_id')
            if img_id is None: continue
            if img_id not in self.annotations_per_image: self.annotations_per_image[img_id] = []
            self.annotations_per_image[img_id].append(anno)

        # Create image ID lookup and filter image list to only those with annotations
        self.image_id_to_info = {img['id']: img for img in self.images_info if 'id' in img}
        self.image_ids = [img_id for img_id in self.image_id_to_info if img_id in self.annotations_per_image]

        if not self.image_ids: print(f"Warning: No valid annotated images found matching category mappings.")
        self.data_rng = np.random.RandomState(123) # For augmentations if implemented

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        if idx >= len(self.image_ids): raise IndexError("Index out of bounds")

        image_id = self.image_ids[idx]
        image_info = self.image_id_to_info[image_id]
        # Limit annotations per image *before* processing
        annotations = self.annotations_per_image.get(image_id, [])
        annotations = annotations[:self.max_group_len] # Limit number of components per image

        # --- Load Image ---
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        try:
            image = cv2.imread(image_path)
            if image is None: raise IOError(f"Image file not found or invalid: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            # Handle image loading errors gracefully by trying the next item
            print(f"Warning: Error loading image {image_path}: {e}. Attempting to skip.")
            return None

        original_height, original_width = image.shape[0:2]

        center_heatmaps = np.zeros((self.num_heatmap_channels, self.heatmap_h, self.heatmap_w), dtype=np.float32)
        key_heatmaps    = np.zeros((self.num_heatmap_channels, self.heatmap_h, self.heatmap_w), dtype=np.float32)
        center_regrs    = np.zeros((self.max_centers, 2), dtype=np.float32)
        key_regrs       = np.zeros((self.max_keypoints, 2), dtype=np.float32)
        center_tags     = np.zeros((self.max_centers), dtype=np.int64)    # Stores flattened index (y*W+x)
        key_tags        = np.zeros((self.max_keypoints), dtype=np.int64)   # Stores flattened index (y*W+x)
        center_masks    = np.zeros((self.max_centers), dtype=bool)     # Mask for valid centers in batch
        key_masks       = np.zeros((self.max_keypoints), dtype=bool)    # Mask for valid keypoints in batch
        center_final_types = np.full((self.max_centers), -1, dtype=np.int64) # GT for L_CLS (0-6), -1 for ignore/padding
        center_heatmap_types = np.full((self.max_centers), -1, dtype=np.int64) # GT heatmap channel index (0,1,2) for center type emb
        key_heatmap_types = np.full((self.max_keypoints), -1, dtype=np.int64) # GT heatmap channel index (0,1,2) for keypoint type emb
        group_target = np.zeros((self.max_centers, self.max_keypoints), dtype=np.int64) # GT grouping matrix (might not be needed if using attn loss?)
        tag_lens_keys = 0 # Number of valid keypoints added
        tag_lens_cens = 0 # Number of valid centers added

        # --- Image Preprocessing ---
        input_h, input_w = self.cfg["input_size"]
        if input_w <= 0 or input_h <= 0: raise ValueError("Invalid input size")
        resized_image = cv2.resize(image, (input_w, input_h))

        # Calculate scaling ratios for coordinate transforms
        height_ratio = input_h / original_height if original_height > 0 else 0 # Orig -> Input
        width_ratio = input_w / original_width if original_width > 0 else 0    # Orig -> Input
        hm_height_ratio = self.heatmap_h / input_h if input_h > 0 else 0        # Input -> Heatmap
        hm_width_ratio = self.heatmap_w / input_w if input_w > 0 else 0         # Input -> Heatmap

        center_mapping = self.cfg["category_mapping"]
        type_mapping = self.cfg["type_mapping"]

        # --- Process Annotations ---
        for anno_idx, anno in enumerate(annotations):
            # Check if we've reached the maximum number of centers or keypoints allowed
            # Need space for potential keypoints (e.g., pie needs 3, box needs 2)
            if tag_lens_cens >= self.max_centers or tag_lens_keys >= self.max_keypoints - 3: # Leave buffer
                 # print(f"Warning: Max centers ({self.max_centers}) or keypoints ({self.max_keypoints}) reached for image {image_id}. Skipping remaining annotations.")
                 break

            original_category_id = anno.get('category_id')
            bbox_data = anno.get('bbox')

            if original_category_id is None or bbox_data is None or original_category_id not in center_mapping:
                continue

            heatmap_category = center_mapping[original_category_id] # Heatmap channel index (0, 1, or 2)
            final_type_label = type_mapping.get(original_category_id, -1) # Final type index (0-6)
            if final_type_label == -1:
                 # print(f"Debug: Skipping annotation {anno.get('id')} due to unmapped final type for category {original_category_id}")
                 continue # Skip if final type cannot be determined

            # Parse coordinates based on original category type and resize to INPUT image space
            coords_resized = None
            try:
                if original_category_id in [1, 4, 5, 6, 7]: # Box [x, y, w, h]
                    if len(bbox_data)==4:
                        x1, y1, w, h = bbox_data
                        if w > 0 and h > 0:
                             x2, y2 = x1 + w, y1 + h
                             coords_resized = np.array([x1*width_ratio, y1*height_ratio, x2*width_ratio, y2*height_ratio])
                elif original_category_id == 2: # Line [x1, y1, ..., xn, yn]
                    if len(bbox_data) >= 2 and len(bbox_data)%2==0:
                        coords_resized = np.array(bbox_data).astype(float)
                        coords_resized[0::2] *= width_ratio
                        coords_resized[1::2] *= height_ratio
                elif original_category_id == 3: # Pie [cx, cy, e1x, e1y, e2x, e2y]
                    if len(bbox_data) == 6:
                        coords_resized = np.array(bbox_data).astype(float)
                        coords_resized[0::2] *= width_ratio
                        coords_resized[1::2] *= height_ratio
            except Exception as e:
                print(f"Warning: Error parsing bbox data {bbox_data} for category {original_category_id}: {e}. Skipping anno.")
                continue

            if coords_resized is None or len(coords_resized) == 0:
                 # print(f"Debug: Skipping annotation {anno.get('id')} due to invalid/empty coordinates after resize.")
                 continue

            # Transform coordinates from INPUT space to HEATMAP space
            coords_hm = coords_resized.copy()
            coords_hm[0::2] *= hm_width_ratio
            coords_hm[1::2] *= hm_height_ratio

            coords_hm_clipped = _clip_coords(coords_hm.copy(), (self.heatmap_h, self.heatmap_w))
            coords_int = coords_hm_clipped.astype(np.int32) # Integer coords for indexing heatmap

            # --- Calculate Gaussian Radius ---
            radius = self.cfg["gaussian_radius"]
            if radius == -1: # Auto-calculate radius based on object size in heatmap space
                try:
                     if heatmap_category == 0 and len(coords_hm) == 4: # Box type
                         h_hm = abs(coords_hm[3]-coords_hm[1]) # Height in heatmap pixels
                         w_hm = abs(coords_hm[2]-coords_hm[0]) # Width in heatmap pixels
                         radius = gaussian_radius((h_hm, w_hm), self.cfg["gaussian_iou"])
                     elif heatmap_category == 2 and len(coords_hm) == 6: # Pie type (use ellipse axes approx)
                         h1=np.sqrt(max(0,(coords_hm[0]-coords_hm[2])**2+(coords_hm[1]-coords_hm[3])**2))
                         h2=np.sqrt(max(0,(coords_hm[0]-coords_hm[4])**2+(coords_hm[1]-coords_hm[5])**2))
                         radius = gaussian_radius((max(h1,h2),max(h1,h2)), self.cfg["gaussian_iou"]) # Use max axis approx
                     else: # Default for lines or if calculation fails
                         radius = 2 # Smaller default radius
                except Exception as e:
                     print(f"Warning: Error calculating gaussian radius: {e}. Using default radius 2.")
                     radius = 2
                radius = max(0, int(radius)) # Ensure non-negative integer
            elif radius < 0: # Handle invalid config radius
                 radius = 2

            # --- Determine Center and Keypoints in Heatmap Space ---
            current_center_tag_ind = tag_lens_cens
            center_pt_hm, keypoints_hm = None, []

            try: # Wrap point calculation in try-except
                 if heatmap_category == 0: # Box (uses heatmap channel 0)
                     if len(coords_hm)<4: continue
                     fxk1, fyk1, fxk2, fyk2 = coords_hm[0], coords_hm[1], coords_hm[2], coords_hm[3]
                     fxce, fyce = (fxk1+fxk2)/2.0, (fyk1+fyk2)/2.0 # Center is midpoint
                     center_pt_hm = [fxce, fyce]
                     keypoints_hm = [[fxk1, fyk1], [fxk2, fyk2]] # Keypoints are corners
                 elif heatmap_category == 1: # Line (uses heatmap channel 1)
                     # Use integer coordinates to check validity *within heatmap bounds*
                     valid_pts_int = [[coords_int[2*k], coords_int[2*k+1]] for k in range(len(coords_int)//2)]
                     valid_pts_float = [[coords_hm[2*k], coords_hm[2*k+1]] for k in range(len(coords_hm)//2)]

                     # Filter based on whether integer version is inside heatmap
                     keypoints_hm = [pt_f for pt_f, pt_i in zip(valid_pts_float, valid_pts_int) if not bad_p(pt_i[0], pt_i[1], self.output_size)]

                     if not keypoints_hm: continue # Skip if no valid keypoints within bounds
                     n = len(keypoints_hm); mid = n//2
                     if n==0: continue # Should be caught by `if not keypoints_hm`
                     elif n % 2 == 0: # Even number of points, center is average of middle two
                         fxce = (keypoints_hm[mid-1][0] + keypoints_hm[mid][0]) / 2.0
                         fyce = (keypoints_hm[mid-1][1] + keypoints_hm[mid][1]) / 2.0
                     else: # Odd number of points, center is the middle point
                         fxce, fyce = keypoints_hm[mid][0], keypoints_hm[mid][1]
                     center_pt_hm = [fxce, fyce]
                 elif heatmap_category == 2: # Pie (uses heatmap channel 2)
                     if len(coords_hm)<6: continue
                     fxce, fyce = coords_hm[0], coords_hm[1] # Center provided directly
                     # Keypoints are center itself and the two ellipse points
                     fxk1,fyk1,fxk2,fyk2,fxk3,fyk3 = coords_hm[0],coords_hm[1],coords_hm[2],coords_hm[3],coords_hm[4],coords_hm[5]
                     center_pt_hm = [fxce, fyce]
                     keypoints_hm = [[fxk1, fyk1], [fxk2, fyk2], [fxk3, fyk3]] # KPs: center, ellipse pt1, ellipse pt2
            except Exception as e:
                 print(f"Warning: Error calculating center/keypoints for anno {anno.get('id')}: {e}. Skipping.")
                 continue

            if center_pt_hm is None or not keypoints_hm:
                 # print(f"Debug: Skipping anno {anno.get('id')} due to missing center or keypoints after processing.")
                 continue

            # --- Store Center Target ---
            fxce, fyce = center_pt_hm
            xce_int, yce_int = int(fxce), int(fyce)
            center_stored_successfully = False

            # Check if integer center coordinates are valid within heatmap bounds
            if not bad_p(xce_int, yce_int, self.output_size):
                # Draw center on the corresponding heatmap channel
                if self.cfg["gaussian_bump"]:
                    draw_gaussian(center_heatmaps[heatmap_category], [xce_int, yce_int], radius)
                else:
                    center_heatmaps[heatmap_category, yce_int, xce_int] = 1.0 # Use 1.0 for consistency

                # Store regression offset (float_coord - int_coord)
                center_regrs[current_center_tag_ind, :] = [fxce - xce_int, fyce - yce_int]
                # Store flattened integer coordinates as the 'tag'
                center_tags[current_center_tag_ind] = yce_int * self.heatmap_w + xce_int
                # Store the final classification type (0-6)
                center_final_types[current_center_tag_ind] = final_type_label
                # Store the heatmap category type (0, 1, or 2) used for type embedding lookup
                center_heatmap_types[current_center_tag_ind] = heatmap_category

                center_stored_successfully = True

            if center_stored_successfully:
                initial_key_tag_index = tag_lens_keys # Remember starting index for this group
                num_keys_added_for_this_center = 0

                for fxk, fyk in keypoints_hm:
                    if tag_lens_keys >= self.max_keypoints: # Check if max keypoints reached globally
                         # print(f"Warning: Reached max keypoints limit ({self.max_keypoints}). Cannot add more keypoints for center {current_center_tag_ind}.")
                         break

                    xk_int, yk_int = int(fxk), int(fyk)

                    # Check if integer keypoint coordinates are valid within heatmap bounds
                    if not bad_p(xk_int, yk_int, self.output_size):
                        # Draw keypoint on the corresponding heatmap channel
                        if self.cfg["gaussian_bump"]:
                            draw_gaussian(key_heatmaps[heatmap_category], [xk_int, yk_int], radius)
                        else:
                            key_heatmaps[heatmap_category, yk_int, xk_int] = 1.0

                        # Store regression offset
                        key_regrs[tag_lens_keys, :] = [fxk - xk_int, fyk - yk_int]
                        # Store flattened integer coordinates as the 'tag'
                        key_tags[tag_lens_keys] = yk_int * self.heatmap_w + xk_int
                        # Store the heatmap category type (0, 1, or 2)
                        key_heatmap_types[tag_lens_keys] = heatmap_category
                        # Mark this keypoint slot as valid (Set later based on tag_lens_keys)
                        # key_masks[tag_lens_keys] = True
                        tag_lens_keys += 1 # Increment count of valid keypoints added
                        num_keys_added_for_this_center += 1
                    # else:
                         # print(f"Debug: Keypoint ({xk_int},{yk_int}) for anno {anno.get('id')} is outside heatmap bounds. Skipping keypoint.")

                if num_keys_added_for_this_center > 0 or not keypoints_hm: # If keys were added OR the component type intrinsically has no separate keypoints to store
                    # Set the grouping target: mark the keypoints added [initial_key_tag_index:tag_lens_keys] as belonging to this center
                     if current_center_tag_ind < group_target.shape[0]:
                          group_target[current_center_tag_ind, initial_key_tag_index : tag_lens_keys] = 1

                     # Now confirm this center is valid and increment the count
                     center_masks[current_center_tag_ind] = True # Mark center as valid in the mask
                     tag_lens_cens += 1 # Increment count of valid centers processed

        # Set key masks based on final count
        key_masks[:tag_lens_keys] = True

        # --- Final Image Preprocessing ---
        processed_image = resized_image.astype(np.float32) / 255.
        processed_image = normalize_(processed_image, self.cfg["mean"], self.cfg["std"])
        image_tensor = torch.from_numpy(processed_image.transpose((2, 0, 1))) # HWC -> CHW

        # --- Convert Targets to Tensors ---
        targets = {
            "image_tensor": image_tensor,
            "key_heatmaps": torch.from_numpy(key_heatmaps),         # [K, H, W] GT heatmap for keypoints
            "center_heatmaps": torch.from_numpy(center_heatmaps),   # [C, H, W] GT heatmap for centers
            "key_regrs": torch.from_numpy(key_regrs),               # [max_keypoints, 2] GT offsets for keypoints
            "center_regrs": torch.from_numpy(center_regrs),         # [max_centers, 2] GT offsets for centers
            "key_tags": torch.from_numpy(key_tags),                 # [max_keypoints] GT flat indices for keys
            "center_tags": torch.from_numpy(center_tags),           # [max_centers] GT flat indices for centers
            "key_masks": torch.from_numpy(key_masks),               # [max_keypoints] Mask for valid keypoints
            "center_masks": torch.from_numpy(center_masks),         # [max_centers] Mask for valid centers
            "group_target": torch.from_numpy(group_target),         # [max_centers, max_keypoints] GT grouping matrix
            "center_final_types": torch.from_numpy(center_final_types), # [max_centers] GT final type (0-6) for L_CLS
            "center_heatmap_types": torch.from_numpy(center_heatmap_types), # [max_centers] GT heatmap type (0-2) for center emb
            "key_heatmap_types": torch.from_numpy(key_heatmap_types),     # [max_keypoints] GT heatmap type (0-2) for key emb
            "tag_lens_cens": torch.tensor(tag_lens_cens, dtype=torch.int32), # Scalar: number of valid centers
            "tag_lens_keys": torch.tensor(tag_lens_keys, dtype=torch.int32)  # Scalar: number of valid keypoints
        }

        # Inputs needed for model.forward() during training
        xs = [ targets["image_tensor"], targets["key_tags"], targets["center_tags"],
               targets["tag_lens_keys"], targets["tag_lens_cens"], targets["center_masks"],
               targets["key_masks"], targets["center_final_types"],
               targets["center_heatmap_types"], targets["key_heatmap_types"] ]

        # Outputs needed for model.compute_loss() during training
        ys = [ targets["key_heatmaps"], targets["center_heatmaps"], targets["key_masks"],
               targets["center_masks"], targets["key_regrs"], targets["center_regrs"],
               targets["group_target"], targets["tag_lens_cens"], targets["tag_lens_keys"],
               targets["center_final_types"] ]

        output_dict = {"xs": xs, "ys": ys}
        return output_dict

def custom_collate_fn(batch):
    # Filter out None items (e.g., from image loading errors)
    batch = [item for item in batch if item is not None]
    if not batch:
        # print("Warning: Collate function received an empty batch.")
        return None # Return None if the batch is empty after filtering

    # Determine the number of items in xs and ys from the first valid sample
    num_xs_items = len(batch[0]['xs'])
    num_ys_items = len(batch[0]['ys'])
    batched_xs = []
    batched_ys = []

    for i in range(num_xs_items):
        items_to_batch = [item['xs'][i] for item in batch]
        # Check if the item is a tensor (most items should be)
        if isinstance(items_to_batch[0], torch.Tensor):
            try:
                # Stack tensors along the batch dimension (dim=0)
                batched_xs.append(torch.stack(items_to_batch, dim=0))
            except RuntimeError as e:
                print(f"Collate Error xs[{i}]: {e}. Shapes: {[it.shape for it in items_to_batch]}")
                return None
        else:
            try:
                 batched_xs.append(torch.tensor(items_to_batch, dtype=torch.int32)) # Use int32 for lengths
            except Exception as e:
                 print(f"Collate Error xs[{i}] converting non-tensor: {e}. Items: {items_to_batch}")
                 return None


    for i in range(num_ys_items):
        items_to_batch = [item['ys'][i] for item in batch]
        if isinstance(items_to_batch[0], torch.Tensor):
            try:
                batched_ys.append(torch.stack(items_to_batch, dim=0))
            except RuntimeError as e:
                print(f"Collate Error ys[{i}]: {e}. Shapes: {[it.shape for it in items_to_batch]}")
                return None
        else:
            try:
                batched_ys.append(torch.tensor(items_to_batch, dtype=torch.int32))
            except Exception as e:
                 print(f"Collate Error ys[{i}] converting non-tensor: {e}. Items: {items_to_batch}")
                 return None

    # Return the batched data
    return {"xs": batched_xs, "ys": batched_ys}



def main_train(args):
    """ Main function to run the training loop. """
    # --- Configuration ---
    # Use dataset config defined below, update with specific training params
    db_config = {
         "input_size": [511, 511], "output_sizes": [[128, 128]],
         "gaussian_bump": False, "gaussian_iou": 0.3, "rand_crop": False, "rand_color": True,
         "categories": 3, "num_final_types": 7, "max_centers": 64, "max_keypoints": 128, "max_group_len": 100,
         "mean": np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3),
         "std": np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3),
    }

    system_config_train = {
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "num_epochs": args.epochs,
        "num_workers": args.num_workers
    }

    loss_weights = { "w_hm": 1.0, "w_loc": 1.0, "w_cls": 1.0, "w_regr": 0.1 }

    # --- Basic Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Dataset & DataLoader ---
    dataset_root = args.dataset_root
    annotation_fn = "chart_train2019.json" # Keep default or make it an arg
    image_set_name = 'train2019'          # Keep default or make it an arg

    annot_path = os.path.join(dataset_root, 'annotations', annotation_fn)
    img_dir_path = os.path.join(dataset_root, 'images', image_set_name)
    if not os.path.isfile(annot_path):
        print(f"ERROR: Annotation file not found: {annot_path}")
        return
    if not os.path.isdir(img_dir_path):
        print(f"ERROR: Image directory not found: {img_dir_path}")
        return
    print("Dataset paths seem valid.")

    try:
        # Pass the combined db_config here
        train_dataset = EC400KDatasetWithTargets(root_dir=dataset_root, annotation_file=annotation_fn, image_set=image_set_name, config=db_config)
        if len(train_dataset) == 0:
            print("ERROR: Dataset is empty after initialization. Check paths and category mappings.")
            return
        print(f"Dataset loaded: {len(train_dataset)} images.")
        train_loader = DataLoader( train_dataset, batch_size=system_config_train["batch_size"], shuffle=True,
                                   num_workers=system_config_train["num_workers"], collate_fn=custom_collate_fn,
                                   pin_memory=(device.type == 'cuda'), drop_last=True ) # drop_last=True helps avoid issues with incomplete batches
    except Exception as e:
        print(f"Error initializing dataset/loader: {e}")
        traceback.print_exc()
        return

    # --- Model Initialization ---
    heatmap_h, heatmap_w = db_config["output_sizes"][0]
    feature_dim = 256 # Should match ChartComponentDetector defaults or be configurable
    hourglass_feature_dim = 256 # Should match ChartComponentDetector defaults or be configurable

    model = ChartComponentDetector(
        num_types=db_config["num_final_types"],
        num_centers=db_config["categories"],
        num_keypoints=db_config["categories"],
        feature_dim=feature_dim,
        hourglass_feature_dim=hourglass_feature_dim,
        mha_heads=4, # Match default or make configurable
        heatmap_h=heatmap_h,
        heatmap_w=heatmap_w,
        max_centers=db_config["max_centers"],
        max_keypoints=db_config["max_keypoints"]
    ).to(device)

    # Optional: Load pre-trained weights if path provided
    if args.load_model and os.path.isfile(args.load_model):
        try:
            print(f"Loading weights for further training from: {args.load_model}")
            model.load_state_dict(torch.load(args.load_model, map_location=device))
        except Exception as e:
            print(f"Warning: Could not load weights from {args.load_model}. Starting from scratch. Error: {e}")


    optimizer = torch.optim.Adam(model.parameters(), lr=system_config_train["learning_rate"])
    # Optional: Learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(system_config_train["num_epochs"]):
        model.train() # Set model to training mode
        epoch_losses = {"total": 0.0, "hm": 0.0, "loc": 0.0, "cls": 0.0, "regr": 0.0}
        num_batches = 0
        optimizer.zero_grad() # Reset gradients once per epoch start

        for batch_idx, batch_data in enumerate(train_loader):
            # Check if collate_fn returned None (e.g., empty batch or error)
            if batch_data is None:
                print(f"Warning: Skipping batch {batch_idx} due to loading/collation errors.")
                continue

            try: # Move batch data to the target device
                # Ensure all list items are tensors before moving
                inputs_xs = [item.to(device, non_blocking=True) if isinstance(item, torch.Tensor) else torch.tensor(item, device=device) for item in batch_data['xs']]
                targets_ys = [item.to(device, non_blocking=True) if isinstance(item, torch.Tensor) else torch.tensor(item, device=device) for item in batch_data['ys']]
            except Exception as e:
                print(f"Error moving batch {batch_idx} to device: {e}. Skipping.")
                continue # Skip this batch

            (image_tensor, key_tags_gt, center_tags_gt, tag_lens_keys, tag_lens_cens,
             _, _, _, center_heatmap_types_gt, key_heatmap_types_gt) = inputs_xs[:10]


            try:
                # Pass all required arguments for training mode forward pass
                preds = model(image_tensor, key_tags_gt=key_tags_gt, center_tags_gt=center_tags_gt,
                              tag_lens_keys=tag_lens_keys, tag_lens_cens=tag_lens_cens,
                              center_heatmap_types_gt=center_heatmap_types_gt,
                              key_heatmap_types_gt=key_heatmap_types_gt)
            except Exception as e:
                print(f"Error during forward pass (Batch {batch_idx}): {e}. Skipping batch.")
                traceback.print_exc()
                optimizer.zero_grad()
                continue # Skip to next batch

            # --- Compute Loss ---
            try:
                # Pass predictions and ground truth lists
                loss_dict = model.compute_loss(preds, targets_ys, inputs_xs)
                if loss_dict is None or "total" not in loss_dict:
                    raise ValueError("Loss computation failed or returned invalid dict")
                loss = loss_dict["total"]
            except Exception as e:
                print(f"Error during loss computation (Batch {batch_idx}): {e}. Skipping batch.")
                traceback.print_exc()
                optimizer.zero_grad() # Clear potentially corrupted gradients
                continue # Skip to next batch

            # --- Backward Pass and Optimization ---
            try:
                # Check for NaN/Inf loss before backward pass
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss detected (Batch {batch_idx}). Value: {loss.item()}. Skipping optimizer step.")
                    optimizer.zero_grad() # Zero grads even if step is skipped
                    continue # Skip optimizer step

                loss.backward() # Calculate gradients

                optimizer.step() # Update weights
                optimizer.zero_grad(set_to_none=True) # Reset gradients for next iteration (more efficient)

            except Exception as e:
                print(f"Error during backward pass or optimizer step (Batch {batch_idx}): {e}. Skipping batch.")
                traceback.print_exc()
                optimizer.zero_grad() # Ensure grads are cleared
                continue # Skip to next batch


            # Accumulate losses for epoch average logging
            for k in epoch_losses:
                if k in loss_dict and isinstance(loss_dict[k], torch.Tensor):
                     epoch_losses[k] += loss_dict[k].item() # Use .item() to get float value
            num_batches += 1

            # Print progress periodically
            if (batch_idx + 1) % 50 == 0 or batch_idx == len(train_loader) - 1:
                print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Current Loss: {loss.item():.4f}")

        if num_batches > 0:
            log_str = f"Epoch [{epoch + 1}/{system_config_train['num_epochs']}] Avg Losses: "
            for k in epoch_losses:
                log_str += f"{k.upper()}: {epoch_losses[k]/num_batches:.4f} | "
            # Optional: Log learning rate
            # current_lr = optimizer.param_groups[0]['lr']
            # log_str += f"LR: {current_lr:.6f}"
            print(log_str.strip().strip('|').strip())
        else:
            print(f"Epoch [{epoch + 1}/{system_config_train['num_epochs']}] - No batches processed successfully.")

        # --- Save Model Periodically (or at the end) ---
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == system_config_train['num_epochs']:
             save_path = os.path.join(args.save_dir, f"chart_detector_epoch_{epoch+1}.pth")
             try:
                 os.makedirs(args.save_dir, exist_ok=True)
                 torch.save(model.state_dict(), save_path)
                 print(f"Model saved to {save_path}")
             except Exception as e:
                 print(f"Error saving model: {e}")

    print(f"\nTraining completed after {system_config_train['num_epochs']} epochs.")

def preprocess_image(image_path, input_size, mean, std):
    """Loads and preprocesses a single image for inference."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise IOError(f"Image file not found or invalid: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Ensure RGB
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None, None

    original_height, original_width = image.shape[0:2]
    input_h, input_w = input_size
    if input_w <= 0 or input_h <= 0:
        raise ValueError("Invalid input size specified in config")

    resized_image = cv2.resize(image, (input_w, input_h)) # Resize to model input size
    processed_image = resized_image.astype(np.float32) / 255. # Normalize to [0, 1]
    processed_image = normalize_(processed_image, mean, std) # Apply mean/std normalization
    image_tensor = torch.from_numpy(processed_image.transpose((2, 0, 1))).unsqueeze(0)

    return image_tensor, original_width, original_height

def find_heatmap_peaks(heatmap, threshold, nms_kernel=3):
    """
    Finds peaks in a heatmap above a threshold using NMS via max pooling.
    Args:
        heatmap (torch.Tensor): Heatmap tensor [B, C, H, W] (should be sigmoid output).
        threshold (float): Minimum score threshold for peaks.
        nms_kernel (int): Kernel size for max pooling NMS. Should be odd.
    Returns:
        List[Tuple[int, int, int, int, float]]: List of peaks (batch_idx, cat_idx, y, x, score).
    """
    B, C, H, W = heatmap.shape
    if nms_kernel <= 0: return [] # No NMS if kernel size is invalid
    pad = (nms_kernel - 1) // 2

    # Apply max pooling to find regions where center pixel is the max
    heatmap_max = F.max_pool2d(heatmap, nms_kernel, stride=1, padding=pad)
    # Keep only peaks where the original heatmap value equals the maxpooled value (local maxima)
    keep = (heatmap_max == heatmap).float()
    heatmap_peaks = heatmap * keep

    # Find indices where score > threshold
    peak_indices = torch.nonzero(heatmap_peaks > threshold, as_tuple=False) # Returns [N, 4] tensor [b, c, y, x]
    peaks = []
    for idx in peak_indices:
        b, c, y, x = idx.tolist() # Convert tensor indices to integers
        score = heatmap_peaks[b, c, y, x].item() # Get the score at the peak
        peaks.append((b, c, y, x, score))

    return peaks

def refine_peak_locations(peaks, regression_map):
    """
    Refines integer peak locations using predicted regression offsets.
    Args:
        peaks (List[Tuple[int, int, int, int, float]]): Output from find_heatmap_peaks (b, c, y_int, x_int, score).
        regression_map (torch.Tensor): Regression map [B, 2, H, W] (offsets dx, dy). Can be None.
    Returns:
        List[Tuple[int, int, float, float, float]]: List of refined peaks (b, c, y_refined, x_refined, score).
    """
    refined_peaks = []
    if regression_map is None:
        # print("Warning: Regression map is None, returning integer peak locations.")
        for b, c, y_int, x_int, score in peaks:
             refined_peaks.append((b, c, float(y_int), float(x_int), score)) # Return int coords as floats
        return refined_peaks

    B, _, H, W = regression_map.shape
    for b, c, y_int, x_int, score in peaks:
        # Ensure integer coordinates are within the bounds of the regression map
        if b >= B or y_int < 0 or y_int >= H or x_int < 0 or x_int >= W:
             print(f"Warning: Peak index ({b},{c},{y_int},{x_int}) out of bounds for regression map shape {regression_map.shape}. Skipping refinement for this peak.")
             refined_peaks.append((b, c, float(y_int), float(x_int), score)) # Add unrefined peak
             continue

        # Regression map has shape [B, 2, H, W], where channel 0 is dx, channel 1 is dy
        offset = regression_map[b, :, y_int, x_int] # Get the [dx, dy] tensor at the peak location
        dx, dy = offset[0].item(), offset[1].item()

        # Calculate refined coordinates: integer location + predicted offset
        x_refined = float(x_int) + dx
        y_refined = float(y_int) + dy
        refined_peaks.append((b, c, y_refined, x_refined, score))

    return refined_peaks

def map_to_original_coords(peaks_refined, heatmap_size, input_size, original_size):
    """
    Maps refined heatmap coordinates back to original image coordinates.
    Accounts for both heatmap->input and input->original scaling.
    Args:
        peaks_refined (List): List of refined peaks (b, c, y_hm, x_hm, score).
        heatmap_size (Tuple[int, int]): (H_hm, W_hm) of the heatmap.
        input_size (Tuple[int, int]): (H_in, W_in) the model input was resized to.
        original_size (Tuple[int, int]): (H_orig, W_orig) of the original image.
    Returns:
        List[Tuple[int, int, float, float, float]]: Peaks in original coords (b, c, y_orig, x_orig, score).
    """
    heatmap_h, heatmap_w = heatmap_size
    input_h, input_w = input_size
    original_h, original_w = original_size
    orig_coords_peaks = []

    if heatmap_w <= 0 or heatmap_h <= 0 or input_w <= 0 or input_h <= 0:
        print("Error: Invalid heatmap or input dimensions for coordinate mapping.")
        return []

    # Calculate scale factors
    # Heatmap coord -> Input coord scale
    scale_x_hm_to_in = input_w / heatmap_w
    scale_y_hm_to_in = input_h / heatmap_h
    # Input coord -> Original coord scale
    scale_x_in_to_orig = original_w / input_w
    scale_y_in_to_orig = original_h / input_h

    for b, c, y_hm, x_hm, score in peaks_refined:
        # Map heatmap coord to input coord
        x_in = x_hm * scale_x_hm_to_in
        y_in = y_hm * scale_y_hm_to_in
        # Map input coord to original coord
        x_orig = x_in * scale_x_in_to_orig
        y_orig = y_in * scale_y_in_to_orig

        # Clamp coordinates to stay within original image bounds [0, W-1] and [0, H-1]
        x_orig = max(0.0, min(x_orig, original_w - 1.0))
        y_orig = max(0.0, min(y_orig, original_h - 1.0))
        orig_coords_peaks.append((b, c, y_orig, x_orig, score)) # Keep batch and category index

    return orig_coords_peaks

def visualize_detections(image_path, centers_orig, keypoints_orig, output_path="output.png"):
    """Draws detected centers and keypoints on the image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image for visualization: {image_path}")
        return

    # Define colors for categories (e.g., Box=Red, Line=Green, Pie=Blue)
    # Assuming 3 heatmap categories (indices 0, 1, 2)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (128, 128, 128)] # Add more if categories > 3

    # Draw Centers (larger circles)
    for _, cat_idx, y, x, score in centers_orig:
        color = colors[cat_idx % len(colors)]
        center_coord = (int(round(x)), int(round(y))) # Round coords for drawing
        cv2.circle(image, center_coord, radius=7, color=color, thickness=-1) # Filled circle
        cv2.circle(image, center_coord, radius=7, color=(255,255,255), thickness=1) # White border

    # Draw Keypoints (smaller circles)
    for _, cat_idx, y, x, score in keypoints_orig:
        color = colors[cat_idx % len(colors)]
        keypoint_coord = (int(round(x)), int(round(y))) # Round coords
        cv2.circle(image, keypoint_coord, radius=4, color=color, thickness=-1) # Filled
        cv2.circle(image, keypoint_coord, radius=4, color=(0,0,0), thickness=1) # Black border

    try:
        cv2.imwrite(output_path, image)
        print(f"Visualization saved to {output_path}")
    except Exception as e:
        print(f"Error saving visualization to {output_path}: {e}")

def inference(model, image_path, device, config, score_threshold=0.3, nms_kernel=3):
    """
    Performs inference on a single image using the ChartComponentDetector model.
    Detects centers and keypoints based on heatmap peaks and refines with offsets.

    Args:
        model: The loaded ChartComponentDetector model (in eval mode).
        image_path (str): Path to the input image.
        device: torch.device ('cuda' or 'cpu').
        config (dict): Configuration dictionary used for training/preprocessing
                       (must contain input_size, output_sizes, mean, std).
        score_threshold (float): Minimum confidence score for detected heatmap peaks.
        nms_kernel (int): Kernel size for Non-Maximum Suppression on heatmaps (odd integer > 0).

    Returns:
        Tuple[List, List]: Tuple containing:
            - centers_orig: List of detected centers [(b, cat, y_orig, x_orig, score), ...]
            - keypoints_orig: List of detected keypoints [(b, cat, y_orig, x_orig, score), ...]
        Returns (None, None) if an error occurs during preprocessing or inference.
    """
    # --- 1. Preprocess Image ---
    input_tensor, orig_w, orig_h = preprocess_image(
        image_path, config['input_size'], config['mean'], config['std']
    )
    if input_tensor is None:
        print(f"Failed to preprocess image: {image_path}")
        return None, None
    input_tensor = input_tensor.to(device)
    original_size = (orig_h, orig_w) # H, W format

    # --- 2. Model Forward Pass (No Gradients) ---
    model.eval() # Ensure model is in evaluation mode
    with torch.no_grad():
        try:

            outputs = model(input_tensor)

        except Exception as e:
             print(f"Error during model forward pass for {image_path}: {e}")
             traceback.print_exc()
             return None, None

        center_heatmap = outputs.get('center_heatmap')
        keypoint_heatmap = outputs.get('keypoint_heatmap')
        center_regr = outputs.get('center_regr')       # May be None if model doesn't predict it
        keypoint_regr = outputs.get('keypoint_regr')   # May be None

        if center_heatmap is None or keypoint_heatmap is None:
             print(f"Error: Heatmaps not found in model output for {image_path}.")
             return None, None

    # --- 3. Post-process Heatmaps ---
    # Apply sigmoid to convert raw heatmap logits to probabilities [0, 1]
    center_heatmap = torch.sigmoid(center_heatmap)
    keypoint_heatmap = torch.sigmoid(keypoint_heatmap)

    # Get heatmap dimensions (H_hm, W_hm)
    try:
        heatmap_h, heatmap_w = center_heatmap.shape[2:]
        heatmap_size = (heatmap_h, heatmap_w) # H, W format
    except IndexError:
        print(f"Error: Invalid heatmap shape {center_heatmap.shape} for {image_path}.")
        return None, None

    # --- 4. Extract Peaks from Heatmaps ---
    center_peaks = find_heatmap_peaks(center_heatmap, score_threshold, nms_kernel)
    keypoint_peaks = find_heatmap_peaks(keypoint_heatmap, score_threshold, nms_kernel)

    # --- 5. Refine Peak Locations using Regression Offsets ---
    centers_refined = refine_peak_locations(center_peaks, center_regr)
    keypoints_refined = refine_peak_locations(keypoint_peaks, keypoint_regr)

    # --- 6. Map Coordinates back to Original Image Size ---
    input_size = tuple(config['input_size']) # Ensure tuple (H, W) - CHECK ORDER! Assume config is [H, W]
    if len(input_size) != 2: raise ValueError(f"Config 'input_size' must be [H, W], got {config['input_size']}")

    centers_orig = map_to_original_coords(centers_refined, heatmap_size, input_size, original_size)
    keypoints_orig = map_to_original_coords(keypoints_refined, heatmap_size, input_size, original_size)

    return centers_orig, keypoints_orig


def main_inference(args):
    """ Main function to run inference on images. """
    # Define the core config used during training here or load from a file
    db_config = {
         "input_size": [511, 511], # [H, W] format expected by preprocess
         "output_sizes": [[128, 128]], # Heatmap size [H, W]
         "categories": 3, # Num heatmap channels (box, line, pie etc.)
         "num_final_types": 7, # For model definition compatibility
         "max_centers": 64,    # For model definition compatibility
         "max_keypoints": 128, # For model definition compatibility
         "mean": np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3),
         "std": np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3),
         # Add other params from training config if they affect model structure
    }
    # Inference specific settings from args
    infer_config = {
        "model_path": args.load_model, # Use the --load_model arg
        "image_dir": args.image_dir,
        "output_dir": args.output_dir,
        "score_threshold": args.threshold,
        "nms_kernel": 3 # Keep fixed or add as arg
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not infer_config["model_path"] or not os.path.isfile(infer_config["model_path"]):
         print(f"ERROR: Model file not found at {infer_config['model_path']}. Please provide path using --load_model.")
         return
    if not infer_config["image_dir"] or not os.path.exists(infer_config["image_dir"]):
         print(f"ERROR: Image directory or file not found at {infer_config['image_dir']}. Please provide path using --image_dir.")
         return

    os.makedirs(infer_config["output_dir"], exist_ok=True)
    print(f"Output visualizations will be saved to: {infer_config['output_dir']}")


    print(f"Loading model structure definition...")
    heatmap_h, heatmap_w = db_config["output_sizes"][0]

    feature_dim = 256
    hourglass_feature_dim = 256


    model = ChartComponentDetector(
        num_types=db_config["num_final_types"],
        num_centers=db_config["categories"],
        num_keypoints=db_config["categories"],
        feature_dim=feature_dim,
        hourglass_feature_dim=hourglass_feature_dim,
        mha_heads=4,
        heatmap_h=heatmap_h,
        heatmap_w=heatmap_w,
        max_centers=db_config["max_centers"],
        max_keypoints=db_config["max_keypoints"]
    ).to(device)

    # Load the trained weights
    try:
        print(f"Loading trained weights from: {infer_config['model_path']}")
        model.load_state_dict(torch.load(infer_config["model_path"], map_location=device))
        model.eval() # Set model to evaluation mode *IMPORTANT*
        print(f"Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model state_dict from {infer_config['model_path']}: {e}")
        traceback.print_exc()
        return

    image_files = []
    if os.path.isdir(infer_config["image_dir"]):
         image_files = [os.path.join(infer_config["image_dir"], f)
                        for f in os.listdir(infer_config["image_dir"])
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    elif os.path.isfile(infer_config["image_dir"]): # Allow single image file input
         image_files = [infer_config["image_dir"]]

    if not image_files:
         print(f"No image files found in '{infer_config['image_dir']}'")
         return

    print(f"Found {len(image_files)} images for inference.")

    # --- Run Inference Loop ---
    total_centers = 0
    total_keypoints = 0
    for img_path in image_files:
        print(f"Processing: {os.path.basename(img_path)}")
        centers, keypoints = inference(model, img_path, device, db_config,
                                       score_threshold=infer_config["score_threshold"],
                                       nms_kernel=infer_config["nms_kernel"])

        if centers is not None and keypoints is not None:
            total_centers += len(centers)
            total_keypoints += len(keypoints)
            print(f" -> Detected {len(centers)} centers, {len(keypoints)} keypoints.")
            # Visualize results
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_vis_path = os.path.join(infer_config["output_dir"], f"{base_name}_detections.png")
            visualize_detections(img_path, centers, keypoints, output_vis_path)
        else:
            print(f" -> Inference failed for this image.")

    print(f"\nInference finished. Processed {len(image_files)} images.")
    print(f"Total centers detected: {total_centers}, Total keypoints detected: {total_keypoints}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or run inference for Chart Component Detector.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'infer'],
                        help="Select mode: 'train' or 'infer'.")

    # --- Training Arguments ---
    parser.add_argument('--dataset_root', type=str, default="ec400k/cls", # Default path, CHANGE IF NEEDED
                        help="Root directory of the EC400K dataset (or similar).")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=4, help="Training batch size.")
    parser.add_argument('--lr', type=float, default=2.5e-5, help="Learning rate for Adam optimizer.")
    parser.add_argument('--num_workers', type=int, default=2, help="Number of dataloader workers.")
    parser.add_argument('--save_dir', type=str, default="saved_models", help="Directory to save trained models.")
    parser.add_argument('--save_interval', type=int, default=5, help="Save model every N epochs.")
    # Add load_model also for training if you want to resume
    parser.add_argument('--load_model', type=str, default=None,
                        help="Path to a pre-trained model file (for resuming training or for inference).")

    # --- Inference Arguments ---
    parser.add_argument('--image_dir', type=str, default="path/to/inference/images", # Placeholder, MUST BE PROVIDED for inference
                        help="Path to the directory containing images for inference, or a single image file.")
    parser.add_argument('--output_dir', type=str, default="inference_outputs",
                        help="Directory to save inference visualization results.")
    parser.add_argument('--threshold', type=float, default=0.3,
                        help="Confidence threshold for detecting heatmap peaks during inference.")


    args = parser.parse_args()

    # Execute the chosen mode
    if args.mode == 'train':
        print("Selected mode: Training")
        main_train(args)
    elif args.mode == 'infer':
        print("Selected mode: Inference")
        # Check required inference args
        if not args.load_model:
             parser.error("--load_model is required for inference mode.")
        if not args.image_dir or args.image_dir == "path/to/inference/images":
             parser.error("--image_dir must be specified for inference mode.")

        main_inference(args)
    else:
        print(f"Error: Unknown mode '{args.mode}'")
