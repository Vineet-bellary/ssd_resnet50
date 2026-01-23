import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSSDBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # Block 1 → 28x28
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 224 → 112
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 112 → 56
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 56 → 28
        )

        # Block 2 → 14x14
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 28 → 14
        )

        # Block 3 → 7x7
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 14 → 7
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        f1 = self.conv3(x)   # [B, 256, 28, 28]
        f2 = self.conv4(f1)  # [B, 256, 14, 14]
        f3 = self.conv5(f2)  # [B, 256, 7, 7]

        return [f1, f2, f3]
