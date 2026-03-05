"""
EfficientNetV2-S for Chest X-Ray Classification
=================================================
Goal: Detect Cardiomegaly and Pulmonary Plethora.

Architecture:
  Input   : (B, 3, 224, 224)
  Backbone: EfficientNetV2-S (ImageNet pretrained)
  Pool    : Adaptive Average Pool → (B, 1280)
  Project : Linear → (B, 512)   — the embedding
  Head    : Linear → logit      — removed in surgery

The final conv feature map is also exposed for Grad-CAM.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class EfficientNetV2XRay(nn.Module):
    """
    EfficientNetV2-S fine-tuned for chest X-ray pathology detection.

    Parameters
    ----------
    embed_dim   : output embedding dimension (default 512)
    num_classes : 1 for binary; 0 to produce embedding only (surgery mode)
    freeze_bn   : freeze BatchNorm stats (useful when fine-tuning with small batches)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_classes: int = 1,
        freeze_bn: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Load pretrained EfficientNetV2-S
        base = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

        # ── Feature extractor (everything before the classifier) ──
        # base.features  : Sequential of MBConv/FusedMBConv blocks
        # base.avgpool   : AdaptiveAvgPool2d → (B, 1280, 1, 1)
        self.features = base.features
        self.avgpool = base.avgpool
        backbone_out = 1280

        if freeze_bn:
            for m in self.features.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

        # ── Projection head ──
        self.embed_proj = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(backbone_out, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # ── Classification head ──
        if num_classes > 0:
            self.classifier = nn.Linear(embed_dim, num_classes)

        # Hook storage for Grad-CAM
        self._gradients: torch.Tensor | None = None
        self._activations: torch.Tensor | None = None
        self._register_hooks()

    # ------------------------------------------------------------------
    # Grad-CAM hooks  (attached to the last conv stage)
    # ------------------------------------------------------------------
    def _register_hooks(self):
        """Attach forward/backward hooks to the last EfficientNetV2 stage."""
        last_block = self.features[-1]  # last Sequential block

        def forward_hook(module, inp, out):
            self._activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        last_block.register_forward_hook(forward_hook)
        last_block.register_full_backward_hook(backward_hook)

    def get_grad_cam(self) -> torch.Tensor | None:
        """
        Compute Grad-CAM heatmap from the last stored gradients and activations.
        Returns a tensor of shape (B, H, W) normalised to [0, 1], or None.
        """
        if self._gradients is None or self._activations is None:
            return None
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1)             # (B, H, W)
        cam = F.relu(cam)
        # Normalize per sample
        cam_min = cam.view(cam.shape[0], -1).min(dim=1).values.view(-1, 1, 1)
        cam_max = cam.view(cam.shape[0], -1).max(dim=1).values.view(-1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam  # (B, H, W)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, 224, 224)

        Returns
        -------
        embed : (B, embed_dim)
        """
        feat = self.features(x)     # (B, 1280, H', W')
        feat = self.avgpool(feat)   # (B, 1280, 1, 1)
        feat = feat.flatten(1)      # (B, 1280)
        embed = self.embed_proj(feat)  # (B, embed_dim)
        return embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed = self.forward_features(x)
        if self.num_classes > 0:
            return self.classifier(embed)
        return embed


# ---------------------------------------------------------------------------
# Utility: surgery
# ---------------------------------------------------------------------------

def efficientnet_without_head(checkpoint_path: str, device: str = "cpu") -> EfficientNetV2XRay:
    model = EfficientNetV2XRay(num_classes=1)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.num_classes = 0
    if hasattr(model, "classifier"):
        del model.classifier
    model.eval()
    return model.to(device)
