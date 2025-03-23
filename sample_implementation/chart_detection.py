import torch
import torch.nn as nn
import torchvision.models as models

# Simplified Hourglass Network for Keypoint Detection
class SimpleHourglass(nn.Module):
    def __init__(self, num_keypoints=100):
        super(SimpleHourglass, self).__init__()
        backbone = models.resnet50(pretrained=True)
        layers = list(backbone.children())[:-2]  # remove fc and avgpool
        self.encoder = nn.Sequential(*layers)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, num_keypoints, kernel_size=1)
        )

    def forward(self, x):
        feats = self.encoder(x)
        heatmaps = self.decoder(feats)
        return heatmaps

# Multi-head Attention for Grouping
class GroupingTransformer(nn.Module):
    def __init__(self, feature_dim=128, num_heads=4):
        super(GroupingTransformer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)
        self.fc_type = nn.Linear(feature_dim, 8)
        self.fc_bbox = nn.Linear(feature_dim, 4)

    def forward(self, center_feats, keypoint_feats):
        attn_output, _ = self.attention(center_feats, keypoint_feats, keypoint_feats)
        types = self.fc_type(attn_output)
        bboxes = self.fc_bbox(attn_output)
        return types, bboxes

# Chart Component Detection Module
class ChartComponentDetector(nn.Module):
    def __init__(self, num_keypoints=100, feature_dim=128, num_heads=4):
        super(ChartComponentDetector, self).__init__()
        self.hourglass = SimpleHourglass(num_keypoints)
        self.grouping_transformer = GroupingTransformer(feature_dim, num_heads)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_keypoints, feature_dim, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, images):
        heatmaps = self.hourglass(images)
        feats = self.feature_extractor(heatmaps).view(images.size(0), -1)

        center_feats = feats.unsqueeze(0)
        keypoint_feats = feats.unsqueeze(0)

        types_logits, bbox_preds = self.grouping_transformer(center_feats, keypoint_feats)

        types_logits = types_logits.squeeze(0)
        bbox_preds = bbox_preds.squeeze(0)

        return types_logits, bbox_preds, heatmaps


# Example Usage
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ChartComponentDetector().to(device)

    sample_images = torch.randn(2, 3, 512, 512).to(device)

    with torch.no_grad():
        types_logits, bbox_preds, heatmaps = model(sample_images)

    print("Types logits shape:", types_logits.shape)
    print("Bounding boxes shape:", bbox_preds.shape)
    print("Heatmaps shape:", heatmaps.shape)

