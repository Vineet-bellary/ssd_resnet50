import torch
import torch.nn as nn

# -----------------------------
# Per-feature-map head
# -----------------------------
class SSDHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()

        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.cls_conv = nn.Conv2d(
            in_channels,
            num_anchors * (num_classes + 1),
            kernel_size=3,
            padding=1
        )

        self.bbox_conv = nn.Conv2d(
            in_channels,
            num_anchors * 4,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        B, _, H, W = x.shape

        cls = self.cls_conv(x)
        cls = cls.permute(0, 2, 3, 1).contiguous()
        cls = cls.view(B, H * W * self.num_anchors, self.num_classes + 1)

        bbox = self.bbox_conv(x)
        bbox = bbox.permute(0, 2, 3, 1).contiguous()
        bbox = bbox.view(B, H * W * self.num_anchors, 4)

        return cls, bbox


# -----------------------------
# Full SSD Model
# -----------------------------
class SSDModel(nn.Module):
    def __init__(self, backbone, feature_channels, num_anchors, num_classes):
        super().__init__()

        self.backbone = backbone

        self.heads = nn.ModuleList([
            SSDHead(
                in_channels=ch,
                num_anchors=num_anchors,
                num_classes=num_classes
            )
            for ch in feature_channels
        ])

    def forward(self, x):
        features = self.backbone(x)

        cls_outputs = []
        bbox_outputs = []

        for feat, head in zip(features, self.heads):
            cls, bbox = head(feat)
            cls_outputs.append(cls)
            bbox_outputs.append(bbox)

        cls_logits = torch.cat(cls_outputs, dim=1)
        bbox_preds = torch.cat(bbox_outputs, dim=1)

        return cls_logits, bbox_preds
