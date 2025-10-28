# recognition/siamese-isic2020-47525496/modules.py
import torch
import torch.nn as nn
import torchvision

__all__ = ["SiameseResNet18"]

class SiameseResNet18(nn.Module):
    """
    ResNet-18 backbone → embedding → pair head on |e1 - e2|.
    Forward returns logits for BCEWithLogitsLoss (apply sigmoid in eval).

    Args:
        pretrained: load ImageNet weights if available.
        proj_dim: override feature dim (auto-detected by default).
    """
    def __init__(self, pretrained: bool = True, proj_dim: int | None = None):
        super().__init__()

        # Handle torchvision versions with/without Weights enums
        weights = None
        if pretrained:
            try:
                weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1  # torchvision >= 0.13
            except Exception:
                weights = "IMAGENET1K_V1"  # older fallback; torchvision will ignore if unsupported

        base = torchvision.models.resnet18(weights=weights) if hasattr(torchvision.models, "resnet18") \
               else torchvision.models.ResNet18(weights=weights)  # Old API fallback

        # Strip the classification head, keep conv trunk
        self.encoder = nn.Sequential(*list(base.children())[:-1])  # -> [B, C=feat_dim, 1, 1]

        # Auto-detect feature dim (In case we swap backbones)
        feat_dim = getattr(base.fc, "in_features", 512)
        if proj_dim is None:
            proj_dim = feat_dim

        # Simple MLP pair head on absolute difference of embeddings
        self.head = nn.Sequential(
            nn.Linear(proj_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return encoder embedding as [B, D]."""
        return self.encoder(x).flatten(1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Pairwise logits: higher → more likely 'same-class'."""
        e1, e2 = self.embed(x1), self.embed(x2)
        d = torch.abs(e1 - e2)
        return self.head(d).squeeze(1)
