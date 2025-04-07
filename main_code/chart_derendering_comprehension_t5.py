# -*- coding: utf-8 -*-
import json
import os
import math
import random
import traceback
import argparse
import time
import re
import pandas as pd
import multiprocessing # Keep import

from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5Config, T5ForConditionalGeneration, AdamW,
    get_linear_schedule_with_warmup
)
from transformers import T5TokenizerFast as T5Tokenizer
# Suppress Hugging Face tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    nltk_available = True
except ImportError:
    nltk_available = False

class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return F.relu(x + self.conv(x))

class HourglassNet(nn.Module):

    def __init__(self, in_channels=3, num_centers=3, num_keypoints=3, hourglass_feature_dim=256):
        super().__init__()
        self.hourglass_feature_dim = hourglass_feature_dim
        self.num_centers = num_centers
        self.num_keypoints = num_keypoints
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
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.res4 = ResidualBlock(128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.res5 = ResidualBlock(64)
        self.center_head = nn.Conv2d(64, self.num_centers, kernel_size=1)
        self.keypoint_head = nn.Conv2d(64, self.num_keypoints, kernel_size=1)
        self.center_regr_head = nn.Conv2d(64, 2, kernel_size=1)
        self.keypoint_regr_head = nn.Conv2d(64, 2, kernel_size=1)
        self.feature_map_conv = nn.Conv2d(64, self.hourglass_feature_dim, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.pool1(d1)
        d2 = self.res1(d2)
        d3 = self.down2(d2)
        d3 = self.res2(d3)
        d4 = self.down3(d3)
        d4 = self.res3(d4)
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
        center_heatmap = self.center_head(final_feat)
        keypoint_heatmap = self.keypoint_head(final_feat)
        center_regr = self.center_regr_head(final_feat)
        keypoint_regr = self.keypoint_regr_head(final_feat)
        output_feature_map = self.feature_map_conv(final_feat)
        return {
            'center_heatmap': center_heatmap,
            'keypoint_heatmap': keypoint_heatmap,
            'center_regr': center_regr,
            'keypoint_regr': keypoint_regr,
            'output_feature_map': output_feature_map
        }

# --- Positional Encoding ---
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_h=128, max_w=128):
        super().__init__()
        assert d_model % 4 == 0
        self.d_model = d_model
        self.max_h = max_h
        self.max_w = max_w
        d_half = d_model // 2
        div_term = torch.zeros(0) if d_half == 0 else torch.exp(torch.arange(0, d_half, 2).float() * (-math.log(10000.0) / d_half))
        pe_y = torch.zeros(max_h, d_half)
        position_y = torch.arange(0, max_h, dtype=torch.float).unsqueeze(1)
        if div_term.numel() > 0:
            pe_y[:, 0::2] = torch.sin(position_y * div_term)
            pe_y[:, 1::2] = torch.cos(position_y * div_term[:len(pe_y[0, 1::2].T)]) # Handle odd d_half
        pe_x = torch.zeros(max_w, d_half)
        position_x = torch.arange(0, max_w, dtype=torch.float).unsqueeze(1)
        if div_term.numel() > 0:
            pe_x[:, 0::2] = torch.sin(position_x * div_term)
            pe_x[:, 1::2] = torch.cos(position_x * div_term[:len(pe_x[0, 1::2].T)]) # Handle odd d_half
        self.register_buffer('pe_y', pe_y)
        self.register_buffer('pe_x', pe_x)

    def forward(self, coords):
        B, N, _ = coords.shape
        max_w_idx = self.max_w - 1
        max_h_idx = self.max_h - 1
        if max_w_idx < 0 or max_h_idx < 0:
            return torch.zeros(B, N, self.d_model, device=coords.device, dtype=coords.dtype)
        coords_x = torch.clamp(coords[..., 0].round(), 0, max_w_idx).long()
        coords_y = torch.clamp(coords[..., 1].round(), 0, max_h_idx).long()
        emb_x = self.pe_x[coords_x]
        emb_y = self.pe_y[coords_y]
        pos_encoding = torch.cat((emb_x, emb_y), dim=-1)
        return pos_encoding

class ChartComponentDetector(nn.Module):
    def __init__(self, num_types=7, num_centers=3, num_keypoints=3,
                 feature_dim=256, hourglass_feature_dim=256, mha_heads=4,
                 heatmap_h=128, heatmap_w=128, max_centers=64, max_keypoints=128):
        super().__init__()
        self.num_types = num_types
        self.num_centers = num_centers
        self.num_keypoints = num_keypoints
        self.feature_dim = feature_dim
        self.hourglass_feature_dim = hourglass_feature_dim
        self.heatmap_h = heatmap_h
        self.heatmap_w = heatmap_w
        self.max_centers = max_centers
        self.max_keypoints = max_keypoints
        self.hourglass = HourglassNet(in_channels=3, num_centers=self.num_centers, num_keypoints=self.num_keypoints, hourglass_feature_dim=self.hourglass_feature_dim)
        self.pos_encoder = PositionalEncoding2D(self.hourglass_feature_dim, max_h=heatmap_h, max_w=heatmap_w)
        self.center_type_emb = nn.Embedding(self.num_centers + 1, self.hourglass_feature_dim, padding_idx=0)
        self.keypoint_type_emb = nn.Embedding(self.num_keypoints + 1, self.hourglass_feature_dim, padding_idx=0)
        combined_feature_dim = self.hourglass_feature_dim
        self.Wc = nn.Linear(combined_feature_dim, self.feature_dim)
        self.Wk = nn.Linear(combined_feature_dim, self.feature_dim)
        self.type_classifier = nn.Sequential(nn.Linear(self.feature_dim, self.feature_dim // 2), nn.ReLU(inplace=True), nn.Linear(self.feature_dim // 2, self.num_types))

    def forward(self, image_tensor, **kwargs):
        return self.hourglass(image_tensor)

def normalize_(image, mean, std):

    std_safe = np.where(np.abs(std) < 1e-6, 1e-6, std)
    image -= mean
    image /= std_safe
    return image

def preprocess_image(image_path, input_size, mean, std):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise IOError(f"Image file not found or invalid: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None, None
    original_height, original_width = image.shape[0:2]
    input_h, input_w = input_size
    if input_w <= 0 or input_h <= 0:
        raise ValueError("Invalid input size")
    resized_image = cv2.resize(image, (input_w, input_h))
    processed_image = resized_image.astype(np.float32) / 255.
    processed_image = normalize_(processed_image, mean, std)
    image_tensor = torch.from_numpy(processed_image.transpose((2, 0, 1))).unsqueeze(0)
    return image_tensor, original_width, original_height

def find_heatmap_peaks(heatmap, threshold, nms_kernel=3):
    B, C, H, W = heatmap.shape
    if nms_kernel <= 0:
        return []
    pad = (nms_kernel - 1) // 2
    heatmap_max = F.max_pool2d(heatmap, nms_kernel, stride=1, padding=pad)
    keep = (heatmap_max == heatmap).float()
    heatmap_peaks = heatmap * keep
    peak_indices = torch.nonzero(heatmap_peaks > threshold, as_tuple=False)
    peaks = [{'batch_idx': b, 'cat_idx': c, 'y': y, 'x': x, 'score': heatmap_peaks[b, c, y, x].item()}
             for b, c, y, x in peak_indices.tolist()]
    return peaks

def refine_peak_locations(peaks, regression_map):
    refined_peaks = []
    if regression_map is None:
        for p in peaks:
            p['y_refined'], p['x_refined'] = float(p['y']), float(p['x'])
            refined_peaks.append(p)
        return refined_peaks
    B, _, H, W = regression_map.shape
    for p in peaks:
        b, y_int, x_int = p['batch_idx'], p['y'], p['x']
        if b >= B or y_int < 0 or y_int >= H or x_int < 0 or x_int >= W:
            p['y_refined'], p['x_refined'] = float(y_int), float(x_int)
            refined_peaks.append(p)
            continue
        offset = regression_map[b, :, y_int, x_int]
        dx, dy = offset[0].item(), offset[1].item()
        p['x_refined'] = float(x_int) + dx
        p['y_refined'] = float(y_int) + dy
        refined_peaks.append(p)
    return refined_peaks

def gather_feature(fmap, index, mask=None):
    B, C, H, W = fmap.shape
    N = index.shape[1]
    index = torch.clamp(index, 0, H * W - 1)
    index = index.unsqueeze(2).expand(B, N, C)
    fmap_permuted = fmap.view(B, C, H * W).permute(0, 2, 1)
    output = fmap_permuted.gather(1, index)
    if mask is not None:
        mask_expanded = mask.unsqueeze(2).expand_as(output)
        output = output[mask_expanded].reshape(-1, C)
    return output

def run_detector_inference(detector_model, image_path, device, config, score_threshold=0.3, nms_kernel=3):
    input_tensor, orig_w, orig_h = preprocess_image(image_path, config['input_size'], config['mean'], config['std'])
    if input_tensor is None:
        return None
    input_tensor = input_tensor.to(device) # Move input tensor to device here
    detector_model.eval()
    with torch.no_grad():
        outputs = detector_model(input_tensor)
    center_heatmap = outputs.get('center_heatmap')
    keypoint_heatmap = outputs.get('keypoint_heatmap')
    center_regr = outputs.get('center_regr')
    keypoint_regr = outputs.get('keypoint_regr')
    feature_map = outputs.get('output_feature_map')
    if center_heatmap is None or keypoint_heatmap is None or feature_map is None:
        return None
    center_heatmap = torch.sigmoid(center_heatmap)
    keypoint_heatmap = torch.sigmoid(keypoint_heatmap)
    center_peaks = find_heatmap_peaks(center_heatmap, score_threshold, nms_kernel)
    keypoint_peaks = find_heatmap_peaks(keypoint_heatmap, score_threshold, nms_kernel)
    centers_refined = refine_peak_locations(center_peaks, center_regr)
    keypoints_refined = refine_peak_locations(keypoint_peaks, keypoint_regr)
    all_peaks = centers_refined + keypoints_refined
    hourglass_feature_dim = getattr(detector_model.hourglass, 'hourglass_feature_dim', 256)
    if not all_peaks:
        return {'centers': [], 'keypoints': [], 'appearance_features': torch.empty(0, hourglass_feature_dim, device=device), 'bboxes': []}
    bboxes = []
    heatmap_h, heatmap_w = config['output_sizes'][0]
    input_h, input_w = config['input_size']
    fixed_rel_w, fixed_rel_h = 0.1, 0.1
    for p in all_peaks:
        scale_x_hm_to_in = input_w / heatmap_w
        scale_y_hm_to_in = input_h / heatmap_h
        scale_x_in_to_orig = orig_w / input_w
        scale_y_in_to_orig = orig_h / input_h
        x_hm, y_hm = p['x_refined'], p['y_refined']
        x_orig = x_hm * scale_x_hm_to_in * scale_x_in_to_orig
        y_orig = y_hm * scale_y_hm_to_in * scale_y_in_to_orig
        box_w_orig = orig_w * fixed_rel_w
        box_h_orig = orig_h * fixed_rel_h
        x_tl_orig = x_orig - box_w_orig / 2.0
        y_tl_orig = y_orig - box_h_orig / 2.0
        x_br_orig = x_orig + box_w_orig / 2.0
        y_br_orig = y_orig + box_h_orig / 2.0
        x_tl_orig = max(0.0, x_tl_orig)
        y_tl_orig = max(0.0, y_tl_orig)
        x_br_orig = min(orig_w - 1.0, x_br_orig)
        y_br_orig = min(orig_h - 1.0, y_br_orig)
        norm_w = max(1, orig_w)
        norm_h = max(1, orig_h)
        rel_x_tl = x_tl_orig / norm_w
        rel_y_tl = y_tl_orig / norm_h
        rel_x_br = x_br_orig / norm_w
        rel_y_br = y_br_orig / norm_h
        p['relative_bbox'] = [rel_x_tl, rel_y_tl, rel_x_br, rel_y_br]
        bboxes.append(p['relative_bbox'])
    peak_tags = torch.tensor([[p['y'] * heatmap_w + p['x']] for p in all_peaks], dtype=torch.long, device=device).unsqueeze(0).squeeze(-1)
    appearance_features = gather_feature(feature_map, peak_tags, mask=None).squeeze(0)
    for i, p in enumerate(all_peaks):
        p['features'] = appearance_features[i].cpu() # Store features as CPU tensors
    centers_final = [p for p in all_peaks if p in centers_refined]
    keypoints_final = [p for p in all_peaks if p in keypoints_refined]
    return {'centers': centers_final, 'keypoints': keypoints_final, 'appearance_features': appearance_features.cpu(), 'bboxes': bboxes}

class DataVariableReplacement:
    def __init__(self, alpha=0.5, tokenizer=None):
        self.alpha = alpha
        if tokenizer is None: raise ValueError("Tokenizer required")
        self.tokenizer = tokenizer
        self.variable_tokens = [f"var{i}" for i in range(100)]
        self.variable_token_ids = set(tid for tid in self.tokenizer.convert_tokens_to_ids(self.variable_tokens) if tid != self.tokenizer.unk_token_id)

    def replace_text(self, text):
        pattern = r'(?:^|\s)(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|-?\.\d+%?|-?\d+(?:\.\d+)?%?)(?=\s|$|\.|\,)'
        matches = list(re.finditer(pattern, text))
        replaced_text = text
        variable_mapping = {}
        variable_count = 0
        offset_adjustment = 0
        for match in matches:
            if variable_count >= len(self.variable_tokens): break
            original_token = match.group(1)
            is_year = len(original_token) == 4 and original_token.isdigit() and 1900 < int(original_token) < 2100
            if is_year: continue
            variable_name = self.variable_tokens[variable_count]
            variable_mapping[variable_name] = original_token
            start = match.start(1) + offset_adjustment
            end = match.end(1) + offset_adjustment
            replaced_text = replaced_text[:start] + variable_name + replaced_text[end:]
            offset_adjustment += len(variable_name) - len(original_token)
            variable_count += 1
        return replaced_text, variable_mapping

    def revert_text(self, text, variable_mapping):
        output_text = text
        sorted_vars = sorted(variable_mapping.keys(), key=len, reverse=True)
        for var_name in sorted_vars: output_text = output_text.replace(var_name, variable_mapping[var_name])
        return output_text

    def get_input_variable_token_ids(self, variable_mapping):
        if not variable_mapping: return set()
        var_names = list(variable_mapping.keys())
        var_token_ids = self.tokenizer.convert_tokens_to_ids(var_names)
        return set(tid for tid in var_token_ids if tid != self.tokenizer.unk_token_id)

class T5ForChartUnderstanding(T5ForConditionalGeneration):

    def __init__(self, config: T5Config, num_chart_types=3, visual_feature_dim=256):
        super().__init__(config)
        self.config = config
        self.visual_feature_dim = visual_feature_dim
        self.type_embedding_layer = nn.Embedding(num_chart_types + 1, config.d_model, padding_idx=0)
        self.location_embedding_layer = nn.Linear(4, config.d_model)
        self.appearance_embedding_layer = nn.Linear(visual_feature_dim, config.d_model)
        self.fused_embedding_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def _create_fused_embeddings(self, tokens_ids, types, locations, appearances):
        token_embeddings = self.shared(tokens_ids)
        type_embeddings = self.type_embedding_layer(types)
        if locations.shape[-1] != 4: raise ValueError(f"Location tensor last dimension must be 4, got {locations.shape}")
        location_embeddings = self.location_embedding_layer(locations)
        appearance_embeddings = self.appearance_embedding_layer(appearances)
        fused = token_embeddings + type_embeddings + location_embeddings + appearance_embeddings
        fused_norm = self.fused_embedding_layer_norm(fused)
        return fused_norm


    def forward(self,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                head_mask=None,
                decoder_head_mask=None,
                cross_attn_head_mask=None,
                encoder_outputs=None,
                past_key_values=None,
                inputs_embeds=None, # Standard argument
                decoder_inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                # --- ADD cache_position TO SIGNATURE ---
                cache_position=None,
                # --- Custom args (won't be passed to super().forward) ---
                types=None,
                locations=None,
                appearances=None,
               ):

        if encoder_outputs is None:
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time for the encoder.")

            if input_ids is not None:
                if types is not None and locations is not None and appearances is not None:
                    inputs_embeds = self._create_fused_embeddings(input_ids, types, locations, appearances)
                    input_ids = None
                else:
                    pass # Use standard embeddings

            elif inputs_embeds is None:
                 raise ValueError("Encoder inputs are missing. Provide either input_ids or inputs_embeds.")

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

def compute_data_variable_loss(outputs, labels, input_var_token_ids_set):

    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1).clamp(min=0)).squeeze(-1)
    variable_mask = torch.zeros_like(labels, dtype=torch.bool, device=labels.device)
    for var_id in input_var_token_ids_set: variable_mask |= (labels == var_id)
    padding_mask = (labels != -100)
    final_mask = variable_mask & padding_mask
    variable_loss = -target_log_probs * final_mask
    num_variable_tokens = final_mask.sum()
    if num_variable_tokens == 0: return torch.tensor(0.0, device=logits.device)
    l_var = variable_loss.sum() / num_variable_tokens
    return l_var if not (torch.isnan(l_var) or torch.isinf(l_var)) else torch.tensor(0.0, device=logits.device)

class ChartQADatasetCSV(Dataset):
    def __init__(self, csv_path, image_root_dir, tokenizer, detector_model, detector_config, data_var_replacer, max_seq_length=512, max_target_length=128):
        self.image_root_dir = image_root_dir
        self.tokenizer = tokenizer
        self.detector_model = detector_model # Keep model reference
        self.detector_config = detector_config
        self.data_var_replacer = data_var_replacer
        self.max_seq_length = max_seq_length
        self.max_target_length = max_target_length
        self.comp_tokens = {0: "<BOX>", 1: "<LINE>", 2: "<PIE>"}
        self.hourglass_feature_dim = getattr(self.detector_model.hourglass, 'hourglass_feature_dim', 256)
        try:
            self.data = pd.read_csv(csv_path)
            required_cols = ['input', 'output', 'image_path']
            if not all(col in self.data.columns for col in required_cols):
                raise ValueError(f"CSV must contain columns: {required_cols}")
        except Exception as e:
            raise IOError(f"Error reading/parsing CSV {csv_path}: {e}")
        print(f"Loaded {len(self.data)} samples from {csv_path}.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            row = self.data.iloc[idx]
            input_col_text = str(row['input'])
            output_col_text = str(row['output'])
            image_filename = row['image_path']
            image_path = os.path.join(self.image_root_dir, image_filename)
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}. Skipping item {idx}.")
                return None

            # --- Run Chart Detector ---
            # Determine device from the model passed during init
            current_device = next(self.detector_model.parameters()).device
            detector_results = run_detector_inference(
                self.detector_model, image_path, current_device, self.detector_config
            )

            if detector_results is None:
                print(f"Warning: Detector failed for {image_path}. Skipping item {idx}.")
                return None

            detected_items_map = {}
            comp_limit = 20
            current_offset = 0
            all_detected = detector_results['centers'] + detector_results['keypoints']
            all_detected.sort(key=lambda p: p['score'], reverse=True)
            context_str_parts = []
            for i, peak in enumerate(all_detected[:comp_limit]):
                cat_idx = peak['cat_idx']
                comp_token = self.comp_tokens.get(cat_idx, "<UNK_COMP>")
                relative_bbox = peak.get('relative_bbox', [0.0]*4)
                appearance_feature = peak.get('features', torch.zeros(self.hourglass_feature_dim)).cpu() # Explicitly CPU
                comp_str = f"{comp_token}"
                context_str_parts.append(comp_str)
                start_offset = current_offset
                end_offset = start_offset + len(comp_str)
                detected_items_map[start_offset] = {'token': comp_token, 'type_id': cat_idx + 1, 'location': relative_bbox, 'appearance': appearance_feature, 'end_offset': end_offset}
                current_offset = end_offset + 1 if i < len(all_detected[:comp_limit]) - 1 else end_offset
            serialized_context = " ".join(context_str_parts)
            prefix = f"Input: {input_col_text} Context: "
            prefix_len = len(prefix)
            adjusted_items_map = {start + prefix_len: data for start, data in detected_items_map.items()}
            input_text_base = prefix + serialized_context
            input_text_replaced, variable_mapping = self.data_var_replacer.replace_text(input_text_base)
            tokenized_input = self.tokenizer(input_text_replaced, max_length=self.max_seq_length, padding='max_length', truncation=True, return_tensors="pt", return_offsets_mapping=True)
            tokenized_target = self.tokenizer(output_col_text, max_length=self.max_target_length, padding='max_length', truncation=True, return_tensors="pt")
            input_ids = tokenized_input['input_ids'].squeeze(0)
            attention_mask = tokenized_input['attention_mask'].squeeze(0)
            offset_mapping = tokenized_input['offset_mapping'].squeeze(0)
            labels = tokenized_target['input_ids'].squeeze(0)
            labels[labels == self.tokenizer.pad_token_id] = -100
            seq_len = input_ids.shape[0]
            visual_dim = self.hourglass_feature_dim
            types_tensor = torch.zeros(seq_len, dtype=torch.long)
            locations_tensor = torch.zeros(seq_len, 4, dtype=torch.float)
            appearances_tensor = torch.zeros(seq_len, visual_dim, dtype=torch.float)
            for token_idx, (start_char, end_char) in enumerate(offset_mapping):
                if start_char == end_char: continue
                for item_start_offset, item_data in adjusted_items_map.items():
                    item_end_offset = item_data['end_offset'] + prefix_len
                    if max(start_char, item_start_offset) < min(end_char, item_end_offset):
                        types_tensor[token_idx] = item_data['type_id']
                        locations_tensor[token_idx] = torch.tensor(item_data['location'], dtype=torch.float)
                        feat = item_data['appearance']
                        if feat.shape[0] == visual_dim: appearances_tensor[token_idx] = feat.to(dtype=torch.float)
                        else: print(f"Warning: Feature dim mismatch for item {idx}, token {token_idx}. Expected {visual_dim}, got {feat.shape}. Using zeros.")
                        break
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "types": types_tensor, "locations": locations_tensor, "appearances": appearances_tensor, "variable_mapping": variable_mapping}

        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            traceback.print_exc()
            return None

def t5_chart_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        items = [item[key] for item in batch]
        if key == "variable_mapping":
            collated[key] = items
        elif isinstance(items[0], torch.Tensor):
            try:
                collated[key] = torch.stack(items, dim=0)
            except RuntimeError as e:
                print(f"Collate Error key '{key}': {e}")
                return None
        else:
            collated[key] = items
    return collated


def train_t5_epoch(t5_model, dataloader, optimizer, scheduler, device, data_var_replacer, alpha_lvar):
    t5_model.train()
    total_loss = 0
    total_ce_loss = 0
    total_var_loss = 0
    start_time = time.time()
    for batch_idx, batch in enumerate(dataloader):
        if batch is None: continue
        try:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            types = batch['types'].to(device)
            locations = batch['locations'].to(device)
            appearances = batch['appearances'].to(device)
            variable_mappings = batch['variable_mapping']
        except Exception as e:
            print(f"Error moving batch {batch_idx} to device: {e}")
            continue
        try:
            outputs = t5_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, types=types, locations=locations, appearances=appearances, return_dict=True)
            ce_loss = outputs.loss
            batch_input_var_ids = set()
            for vm in variable_mappings: batch_input_var_ids.update(data_var_replacer.get_input_variable_token_ids(vm))
            l_var = compute_data_variable_loss(outputs, labels, batch_input_var_ids)
            loss = ce_loss + alpha_lvar * l_var
        except Exception as e:
            print(f"Error T5 forward/loss (Batch {batch_idx}): {e}")
            traceback.print_exc()
            optimizer.zero_grad()
            continue
        try:
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss (Batch {batch_idx}). Skip step.")
                optimizer.zero_grad()
                continue
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_var_loss += l_var.item()
        except Exception as e:
            print(f"Error backward/optimizer (Batch {batch_idx}): {e}")
            traceback.print_exc()
            optimizer.zero_grad()
            continue
        if (batch_idx + 1) % 50 == 0 or batch_idx == len(dataloader) - 1:
            elapsed_time = time.time() - start_time
            print(f"  Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f} (CE: {ce_loss.item():.4f}, Var: {l_var.item():.4f}) | LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed_time:.2f}s")
            start_time = time.time()
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    avg_ce = total_ce_loss / len(dataloader) if len(dataloader) > 0 else 0
    avg_var = total_var_loss / len(dataloader) if len(dataloader) > 0 else 0
    return avg_loss, avg_ce, avg_var

def evaluate_t5_model(t5_model, dataloader, tokenizer, device, data_var_replacer):

    t5_model.eval()
    predictions = []
    references = []
    total_eval_loss = 0
    alpha_lvar = data_var_replacer.alpha
    with ((torch.no_grad())):
        for batch in dataloader:
            if batch is None: continue
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                types = batch['types'].to(device)
                locations = batch['locations'].to(device)
                appearances = batch['appearances'].to(device)
                variable_mappings = batch['variable_mapping']
            except Exception as e:
                print(f"Error moving eval batch to device: {e}")
                continue
            try:
                outputs = t5_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, types=types, locations=locations, appearances=appearances, return_dict=True)
                ce_loss = outputs.loss
                batch_input_var_ids = set()
                for vm in variable_mappings: batch_input_var_ids.update(data_var_replacer.get_input_variable_token_ids(vm))
                l_var = compute_data_variable_loss(outputs, labels, batch_input_var_ids)
                loss = ce_loss + alpha_lvar * l_var
                total_eval_loss += loss.item()
            except Exception as e:
                print(f"Error during eval loss calc: {e}")
            try:
                inputs_embeds = t5_model._create_fused_embeddings(input_ids, types, locations, appearances)
                generated_ids = t5_model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_length=128, num_beams=4, early_stopping=True)
                preds_raw = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                labels[labels == -100] = tokenizer.pad_token_id
                refs_raw = tokenizer.batch_decode(labels, skip_special_tokens=True)
                final_preds = [data_var_replacer.revert_text(p, vm) for p, vm in zip(preds_raw, variable_mappings)]
                predictions.extend(final_preds)
                references.extend(refs_raw)
            except Exception as e:
                print(f"Error during generation/decoding: {e}")
                traceback.print_exc()
    avg_eval_loss = total_eval_loss / len(dataloader) if len(dataloader) > 0 else 0
    exact_match = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip()) / len(predictions) if predictions else 0
    bleu_score = 0
    if nltk_available and predictions and references:
        try:
            chencherry = SmoothingFunction()
            bleu_scores = [sentence_bleu([r.split()], p.split(), smoothing_function=chencherry.method1) for p, r in zip(predictions, references)]
            bleu_score = np.mean(bleu_scores)
        except Exception as e:
            print(f"Warning: Error calculating BLEU: {e}")
    return avg_eval_loss, exact_match, bleu_score, predictions, references

def prepare_t5_input_for_inference(input_text_user, image_path, tokenizer, detector_model, detector_config, device, data_var_replacer, max_seq_length):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None, None
    detector_results = run_detector_inference(detector_model, image_path, device, detector_config)
    hourglass_feature_dim = getattr(detector_model.hourglass, 'hourglass_feature_dim', 256)
    if detector_results is None:
        print(f"Warning: Detector failed for {image_path}.")
        all_detected = []
    else:
        all_detected = detector_results['centers'] + detector_results['keypoints']
        all_detected.sort(key=lambda p: p['score'], reverse=True)
    comp_tokens_map = {0: "<BOX>", 1: "<LINE>", 2: "<PIE>"}
    detected_items_map = {}
    comp_limit = 20
    current_offset = 0
    context_str_parts = []
    for i, peak in enumerate(all_detected[:comp_limit]):
        cat_idx = peak['cat_idx']
        comp_token = comp_tokens_map.get(cat_idx, "<UNK_COMP>")
        relative_bbox = peak.get('relative_bbox', [0.0] * 4)
        appearance_feature = peak.get('features', torch.zeros(hourglass_feature_dim)).cpu()
        comp_str = f"{comp_token}"
        context_str_parts.append(comp_str)
        start_offset = current_offset
        end_offset = start_offset + len(comp_str)
        detected_items_map[start_offset] = {'token': comp_token, 'type_id': cat_idx + 1, 'location': relative_bbox, 'appearance': appearance_feature, 'end_offset': end_offset}
        current_offset = end_offset + 1 if i < len(all_detected[:comp_limit]) - 1 else end_offset
    serialized_context = " ".join(context_str_parts)

    prefix = f"Input: {input_text_user} Context: "
    prefix_len = len(prefix)
    adjusted_items_map = {start + prefix_len: data for start, data in detected_items_map.items()}
    input_text_base = prefix + serialized_context
    input_text_replaced, variable_mapping = data_var_replacer.replace_text(input_text_base)
    tokenized_input = tokenizer(input_text_replaced, max_length=max_seq_length, padding='max_length', truncation=True, return_tensors="pt", return_offsets_mapping=True)
    input_ids = tokenized_input['input_ids']
    attention_mask = tokenized_input['attention_mask']
    offset_mapping = tokenized_input['offset_mapping'].squeeze(0)
    seq_len = input_ids.shape[1]
    visual_dim = hourglass_feature_dim
    types_tensor = torch.zeros(seq_len, dtype=torch.long)
    locations_tensor = torch.zeros(seq_len, 4, dtype=torch.float)
    appearances_tensor = torch.zeros(seq_len, visual_dim, dtype=torch.float)

    for token_idx, (start_char, end_char) in enumerate(offset_mapping):
        if start_char == end_char: continue
        for item_start_offset, item_data in adjusted_items_map.items():
            item_end_offset = item_data['end_offset'] + prefix_len
            if max(start_char, item_start_offset) < min(end_char, item_end_offset):
                types_tensor[token_idx] = item_data['type_id']
                locations_tensor[token_idx] = torch.tensor(item_data['location'], dtype=torch.float)
                feat = item_data['appearance']
                if feat.shape[0] == visual_dim: appearances_tensor[token_idx] = feat.to(dtype=torch.float)
                break
    types_tensor = types_tensor.unsqueeze(0)
    locations_tensor = locations_tensor.unsqueeze(0)
    appearances_tensor = appearances_tensor.unsqueeze(0)
    model_inputs = {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device), "types": types_tensor.to(device), "locations": locations_tensor.to(device), "appearances": appearances_tensor.to(device)}
    return model_inputs, variable_mapping

def run_t5_inference(args, detector_model, detector_config, t5_model, tokenizer, data_var_replacer, device):

    print(f"--- Running T5 Inference from CSV ---")
    detector_model.eval()
    t5_model.eval()
    if not args.input_csv:
        print("ERROR: --input_csv argument is required for infer_t5 mode.")
        return
    if not os.path.exists(args.input_csv):
        print(f"ERROR: Input CSV file not found: {args.input_csv}")
        return
    try:
        input_df = pd.read_csv(args.input_csv)
        required_cols = ['input', 'image_path']
        if not all(col in input_df.columns for col in required_cols):
            raise ValueError(f"Input CSV must contain columns: {required_cols}")
    except Exception as e:
        print(f"Error reading or parsing input CSV file {args.input_csv}: {e}")
        return
    print(f"Processing {len(input_df)} input items from {args.input_csv}...")
    results = []
    with torch.no_grad():
        for idx, row in input_df.iterrows():
            image_filename = row['image_path']
            input_text_user = str(row['input'])
            if args.image_root_dir and not os.path.isabs(image_filename):
                image_path = os.path.join(args.image_root_dir, image_filename)
            else:
                image_path = image_filename
            print(f"  Processing item {idx+1}/{len(input_df)}: Image='{os.path.basename(image_path)}', Input='{input_text_user[:50]}...'")
            model_inputs, variable_mapping = prepare_t5_input_for_inference(input_text_user, image_path, tokenizer, detector_model, detector_config, device, data_var_replacer, args.max_seq_length)
            if model_inputs is None:
                print(f"  -> Failed to prepare input for item {idx+1}. Skipping.")
                results.append({"input_row": row.to_dict(), "prediction": "[ERROR: Input preparation failed]"})
                continue
            try:
                inputs_embeds = t5_model._create_fused_embeddings(model_inputs['input_ids'], model_inputs['types'], model_inputs['locations'], model_inputs['appearances'])
                generated_ids = t5_model.generate(inputs_embeds=inputs_embeds, attention_mask=model_inputs['attention_mask'], max_length=args.max_target_length, num_beams=4, early_stopping=True)
                preds_raw = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                print(f" -> Output: {str(row['output'])}")
                final_prediction = data_var_replacer.revert_text(preds_raw, variable_mapping)
                print(f"  -> Prediction: {final_prediction}")
                results.append({"input_row": row.to_dict(), "prediction": final_prediction})
            except Exception as e:
                print(f"  -> Error during T5 generation for item {idx+1}: {e}")
                traceback.print_exc()
                results.append({"input_row": row.to_dict(), "prediction": f"[ERROR: Generation failed - {e}]"})
    output_file = args.output_file if args.output_file else "t5_inference_results.json"
    output_path = os.path.join(args.output_dir, output_file)
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
            print(f"\nT5 inference results saved to: {output_path}")
    except Exception as e:
        print(f"\nError saving T5 inference results: {e}")

if __name__ == '__main__':
    try:
        start_method = 'spawn'
        if torch.cuda.is_available():
             if hasattr(multiprocessing, 'get_all_start_methods') and 'forkserver' in multiprocessing.get_all_start_methods():
                  start_method = 'forkserver'
        multiprocessing.set_start_method(start_method, force=True)
        print(f"Set multiprocessing start method to: {start_method}")
    except RuntimeError as e:
         print(f"Warning: Could not set multiprocessing start method ('{start_method}'): {e}. Using default.")
    except Exception as e:
         print(f"Warning: Unexpected error setting multiprocessing start method: {e}")

    parser = argparse.ArgumentParser(description="Train models or run inference for Chart Understanding.")
    parser.add_argument('--mode', type=str, required=True, choices=['train_detector', 'train_t5', 'infer_t5'], help="Select mode.")
    parser.add_argument('--output_dir', type=str, default="output", help="Directory to save outputs.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--num_workers', type=int, default=2, help="Dataloader workers.")
    parser.add_argument('--image_root_dir', type=str, help="Root directory containing chart images.")
    parser.add_argument('--detector_model_path', type=str, help="Path to ChartComponentDetector model (.pth).")
    parser.add_argument('--dataset_root', type=str, default="ec400k/cls", help="Root directory for detector training.")
    parser.add_argument('--annotation_file', type=str, default="chart_train2019.json", help="Annotation filename for detector training.")
    parser.add_argument('--image_set', type=str, default="train2019", help="Image set subdirectory name for detector training.")
    parser.add_argument('--train_csv_path', type=str, help="Path to T5 training data CSV.")
    parser.add_argument('--val_csv_path', type=str, default=None, help="Path to T5 validation data CSV.")
    parser.add_argument('--t5_model_name', type=str, default='t5-base', help="Pretrained T5 model name.")
    parser.add_argument('--load_t5_checkpoint', type=str, default=None, help="Path to T5 checkpoint (.pth).")
    parser.add_argument('--max_seq_length', type=int, default=512, help="Max T5 input sequence length.")
    parser.add_argument('--max_target_length', type=int, default=128, help="Max T5 target sequence length.")
    parser.add_argument('--alpha_lvar', type=float, default=0.5, help="Weight for T5 data variable loss.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=4, help="Training batch size.")
    parser.add_argument('--lr', type=float, default=3e-5, help="Learning rate.")
    parser.add_argument('--warmup_steps', type=int, default=500, help="Warmup steps for T5 scheduler.")
    parser.add_argument('--save_interval', type=int, default=1, help="Save checkpoint every N epochs.")
    parser.add_argument('--input_csv', type=str, help="Path to CSV file for T5 inference.")
    parser.add_argument('--output_file', type=str, default="t5_inference_results.json", help="Filename for T5 inference results.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.mode == 'train_t5':
        if not args.train_csv_path or not args.image_root_dir or not args.detector_model_path:
            parser.error("--train_csv_path, --image_root_dir, and --detector_model_path are required for train_t5 mode.")
        # --- (Load models, datasets, run training loop) ---
        print("Loading Chart Detector Model for T5 training data processing...")
        detector_config = {'input_size': [511, 511], 'output_sizes': [[128, 128]], 'categories': 3, 'num_final_types': 7, 'max_centers': 64, 'max_keypoints': 128, 'mean': np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3), 'std': np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3), }
        try:
            detector_model = ChartComponentDetector(num_types=detector_config["num_final_types"],num_centers=detector_config["categories"],num_keypoints=detector_config["categories"],heatmap_h=detector_config["output_sizes"][0][0],heatmap_w=detector_config["output_sizes"][0][1],max_centers=detector_config["max_centers"],max_keypoints=detector_config["max_keypoints"]).to(device)
            state_dict = torch.load(args.detector_model_path, map_location=device)
            detector_model.load_state_dict(state_dict, strict=False)
            detector_model.eval()
            print(f"Detector loaded from {args.detector_model_path}")
            detector_visual_dim = detector_model.hourglass_feature_dim
        except Exception as e:
            print(f"Error loading detector model: {e}")
            exit(1)

        print(f"Loading T5 Tokenizer and Model ({args.t5_model_name})...")
        tokenizer = T5Tokenizer.from_pretrained(args.t5_model_name)
        t5_config = T5Config.from_pretrained(args.t5_model_name)
        special_comp_tokens = ["<BOX>", "<LINE>", "<PIE>", "<UNK_COMP>"]
        variable_tokens = [f"var{i}" for i in range(100)]
        num_added_toks = tokenizer.add_tokens(special_comp_tokens + variable_tokens, special_tokens=True)
        if num_added_toks > 0:
            print(f"Added {num_added_toks} special/variable tokens.")
        data_var_replacer = DataVariableReplacement(alpha=args.alpha_lvar, tokenizer=tokenizer)
        t5_model = T5ForChartUnderstanding(config=t5_config, num_chart_types=detector_config["categories"], visual_feature_dim=detector_visual_dim).to(device)
        t5_model.resize_token_embeddings(len(tokenizer))
        print(f"Resized T5 token embeddings to: {len(tokenizer)}")
        if args.load_t5_checkpoint:
            if os.path.exists(args.load_t5_checkpoint):
                try:
                    t5_model.load_state_dict(torch.load(args.load_t5_checkpoint, map_location=device))
                    print(f"Loaded T5 checkpoint: {args.load_t5_checkpoint}")
                except Exception as e:
                    print(f"Warning: Could not load T5 checkpoint: {e}.")
            else:
                print(f"Warning: T5 checkpoint not found: {args.load_t5_checkpoint}.")

        print("Loading T5 datasets from CSV...")
        train_dataset = None
        val_dataset = None
        try:
            train_dataset = ChartQADatasetCSV(
                csv_path=args.train_csv_path, image_root_dir=args.image_root_dir, tokenizer=tokenizer,
                detector_model=detector_model, detector_config=detector_config,
                data_var_replacer=data_var_replacer, max_seq_length=args.max_seq_length,
                max_target_length=args.max_target_length
            )
            if len(train_dataset) == 0:
                raise ValueError("Training dataset is empty.")
            if args.val_csv_path:
                if os.path.exists(args.val_csv_path):
                    val_dataset = ChartQADatasetCSV(
                        csv_path=args.val_csv_path, image_root_dir=args.image_root_dir, tokenizer=tokenizer,
                        detector_model=detector_model, detector_config=detector_config,
                        data_var_replacer=data_var_replacer, max_seq_length=args.max_seq_length,
                        max_target_length=args.max_target_length
                    )
                else: print(f"Warning: Validation CSV not found at {args.val_csv_path}.")
        except Exception as e:
            print(f"Error creating dataset(s): {e}")
            traceback.print_exc()
            exit(1)

        train_loader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,collate_fn=t5_chart_collate_fn,pin_memory=True)
        val_loader=None
        if val_dataset:
            val_loader=DataLoader(val_dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,collate_fn=t5_chart_collate_fn,pin_memory=True)
        optimizer=AdamW(t5_model.parameters(),lr=args.lr)
        total_steps=len(train_loader)*args.epochs if len(train_loader)>0 else 1
        scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=args.warmup_steps,num_training_steps=total_steps)
        print("\n--- Starting T5 Training ---")
        best_val_loss=float('inf')

        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            start_epoch_time=time.time()
            avg_loss,avg_ce,avg_var = train_t5_epoch(t5_model,train_loader,optimizer,scheduler,device,data_var_replacer,args.alpha_lvar)
            epoch_duration=time.time()-start_epoch_time
            print(f"Epoch {epoch+1} Train Summary | Avg Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, Var: {avg_var:.4f}) | Duration: {epoch_duration:.2f}s")

            if val_loader:
                print("Running Validation...")
                val_start_time=time.time()
                avg_val_loss,val_em,val_bleu,_,_ = evaluate_t5_model(t5_model,val_loader,tokenizer,device,data_var_replacer)
                val_duration=time.time()-val_start_time
                print(f"Epoch {epoch+1} Val Summary | Avg Loss: {avg_val_loss:.4f} | EM: {val_em:.4f} | BLEU: {val_bleu:.4f} | Duration: {val_duration:.2f}s")
                if avg_val_loss < best_val_loss: # Save Best Model
                    best_val_loss=avg_val_loss
                    save_path=os.path.join(args.output_dir,"t5_chart_model_best.pth")
                    try:
                        torch.save(t5_model.state_dict(),save_path)
                        print(f"New best val loss: {best_val_loss:.4f}. Model saved to {save_path}")
                    except Exception as e:
                        print(f"Error saving best model: {e}")
            elif (epoch+1)%args.save_interval==0:
                save_path=os.path.join(args.output_dir,f"t5_chart_model_epoch_{epoch+1}.pth")
                try:
                    os.makedirs(args.output_dir, exist_ok=True)
                    torch.save(t5_model.state_dict(),save_path)
                    print(f"Checkpoint saved to {save_path}")
                except Exception as e:
                    print(f"Error saving checkpoint: {e}")
        print("\n--- T5 Training Finished ---")
        final_save_path=os.path.join(args.output_dir,"t5_chart_model_final.pth")
        try:
            torch.save(t5_model.state_dict(),final_save_path)
            print(f"Final model saved to {final_save_path}")
        except Exception as e: 
            print(f"Error saving final model: {e}")

    elif args.mode == 'infer_t5':

        if not args.load_t5_checkpoint or not args.detector_model_path or not args.image_root_dir or not args.input_csv:
             parser.error("--load_t5_checkpoint, --detector_model_path, --image_root_dir and --input_csv are required for infer_t5 mode.")

        print("Loading Chart Detector Model for T5 inference...")
        detector_config={'input_size':[511,511],'output_sizes':[[128,128]],'categories':3,'num_final_types':7,'max_centers':64,'max_keypoints':128,'mean':np.array([0.485,0.456,0.406],dtype=np.float32).reshape(1,1,3),'std':np.array([0.229,0.224,0.225],dtype=np.float32).reshape(1,1,3),}
        try:
            detector_model = ChartComponentDetector(num_types=detector_config["num_final_types"],num_centers=detector_config["categories"],num_keypoints=detector_config["categories"],heatmap_h=detector_config["output_sizes"][0][0],heatmap_w=detector_config["output_sizes"][0][1],max_centers=detector_config["max_centers"],max_keypoints=detector_config["max_keypoints"]).to(device)
            state_dict = torch.load(args.detector_model_path, map_location=device)
            detector_model.load_state_dict(state_dict, strict=False)
            detector_model.eval()
            print(f"Detector loaded from {args.detector_model_path}")
            detector_visual_dim = detector_model.hourglass_feature_dim
        except Exception as e:
            print(f"Error loading detector model: {e}")
            exit(1)
        print(f"Loading T5 Tokenizer and Model ({args.t5_model_name})...")
        tokenizer = T5Tokenizer.from_pretrained(args.t5_model_name)
        t5_config = T5Config.from_pretrained(args.t5_model_name)
        special_comp_tokens=["<BOX>","<LINE>","<PIE>","<UNK_COMP>"]
        variable_tokens = [f"var{i}" for i in range(100)]
        num_added_toks = tokenizer.add_tokens(special_comp_tokens+variable_tokens,special_tokens=True)

        if num_added_toks>0:
            print(f"Added {num_added_toks} special/variable tokens.")

        data_var_replacer = DataVariableReplacement(alpha=args.alpha_lvar, tokenizer=tokenizer)
        t5_model = T5ForChartUnderstanding(config=t5_config, num_chart_types=detector_config["categories"], visual_feature_dim=detector_visual_dim).to(device)
        t5_model.resize_token_embeddings(len(tokenizer))

        if os.path.exists(args.load_t5_checkpoint):
            try:
                t5_model.load_state_dict(torch.load(args.load_t5_checkpoint, map_location=device))
                print(f"Loaded T5 checkpoint: {args.load_t5_checkpoint}")
            except Exception as e:
                print(f"Error loading T5 checkpoint: {e}")
                exit(1)
        else:
            print(f"Error: T5 checkpoint not found: {args.load_t5_checkpoint}")
            exit(1)
        t5_model.eval()

        run_t5_inference(args, detector_model, detector_config, t5_model, tokenizer, data_var_replacer, device)

    else:
        print(f"Error: Unknown mode '{args.mode}'")


