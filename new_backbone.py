from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

class ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Using the latest weights parameter
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Layer 1-2: Output size (B, 512, 28, 28) for 224x224 input
        self.stage1 = nn.Sequential(*list(backbone.children())[:6]) 
        # Layer 3: Output size (B, 1024, 14, 14)
        self.stage2 = nn.Sequential(*list(backbone.children())[6:7])
        # Layer 4: Output size (B, 2048, 7, 7)
        self.stage3 = nn.Sequential(*list(backbone.children())[7:8])

    def forward(self, x):
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        return [f1, f2, f3]