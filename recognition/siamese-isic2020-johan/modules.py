import torch
import torch.nn as nn
import torchvision

class SiameseResNet18(nn.Module):
    """
    ResNet-18 backbone → 512-D embedding → pair head on |e1 - e2|.
    Returns logits for BCEWithLogitsLoss (apply sigmoid in eval).
    """
    def __init__(self, pretrained: bool = True, proj_dim: int = 512):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = torchvision.models.resnet18(weights=weights)
        self.encoder = nn.Sequential(*list(base.children())[:-1])  # [B,512,1,1]
        self.head = nn.Sequential(
            nn.Linear(proj_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).flatten(1)  # [B,512]

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        e1, e2 = self.embed(x1), self.embed(x2)
        d = torch.abs(e1 - e2)
        return self.head(d).squeeze(1)
