"""
NTS-Net for Ultrasound Still Image Classification
===================================================
Navigator-Teacher-Scrutinizer Network for fine-grained structural defect
detection (VSD, ASD, etc.) in paediatric echocardiography.

Architecture:
  Input       : (B, 3, 224, 224)
  Navigator   : ResNet-50 backbone → feature map (B, 2048, 7, 7)
                Proposal Layer → top-K region proposals
  Teacher     : Full-image global average pool → (B, 2048)
  Scrutinizer : Crop + resize top-K parts → per-part features → aggregated
  Fusion      : Concat(Teacher, Scrutinizer) → embed_proj → (B, 512)
  Head        : Linear → logit  (removed during surgery)

References
----------
Yang et al., "Learning to Navigate for Fine-grained Classification" (ECCV 2018)
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import roi_align


# ---------------------------------------------------------------------------
# Navigator: predicts attention maps → region proposals
# ---------------------------------------------------------------------------

class Navigator(nn.Module):
    """
    Takes the shared feature map and outputs K bounding-box proposals
    via a lightweight convolutional head.
    """

    def __init__(self, in_channels: int = 2048, num_parts: int = 6):
        super().__init__()
        self.num_parts = num_parts
        self.proposal_head = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_parts, 1),
        )

    def forward(
        self, feat_map: torch.Tensor, image_size: int = 224
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        feat_map   : (B, C, H, W)
        image_size : original image spatial resolution

        Returns
        -------
        attention  : (B, num_parts, H, W)  — soft attention maps
        boxes      : (B, num_parts, 4)     — (x1, y1, x2, y2) in image coords
        """
        attention = self.proposal_head(feat_map)  # (B, K, H, W)
        B, K, H, W = attention.shape
        scale_h = image_size / H
        scale_w = image_size / W

        # For each part: compute centre of mass → bounding box
        attn_flat = attention.view(B, K, -1)
        attn_softmax = torch.softmax(attn_flat, dim=-1).view(B, K, H, W)

        # Grid of coordinates
        ys = torch.arange(H, dtype=torch.float32, device=feat_map.device)
        xs = torch.arange(W, dtype=torch.float32, device=feat_map.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)

        # Expected coordinates
        cy = (attn_softmax * grid_y.unsqueeze(0).unsqueeze(0)).sum(dim=(-2, -1)) * scale_h
        cx = (attn_softmax * grid_x.unsqueeze(0).unsqueeze(0)).sum(dim=(-2, -1)) * scale_w

        # Fixed-size proposal box (40% of image)
        half = image_size * 0.2
        x1 = (cx - half).clamp(0, image_size)
        y1 = (cy - half).clamp(0, image_size)
        x2 = (cx + half).clamp(0, image_size)
        y2 = (cy + half).clamp(0, image_size)
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # (B, K, 4)

        return attention, boxes


# ---------------------------------------------------------------------------
# Scrutinizer: crops + re-processes the top-K parts
# ---------------------------------------------------------------------------

class Scrutinizer(nn.Module):
    """
    Crops top-K proposal regions from the *original image*,
    re-encodes them with a shared lightweight encoder,
    and aggregates into a fixed-size embedding.
    """

    def __init__(
        self,
        backbone_out: int = 2048,
        top_k: int = 3,
        part_size: int = 96,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.top_k = top_k
        self.part_size = part_size

        # Shared part encoder (small ResNet-like)
        self.part_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        part_feat_dim = 128 * 4 * 4  # 2048

        self.agg = nn.Sequential(
            nn.Linear(part_feat_dim * top_k, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, images: torch.Tensor, boxes: torch.Tensor, top_k_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        images        : (B, 3, H, W)  original images
        boxes         : (B, K, 4)     all proposals
        top_k_indices : (B, top_k)    indices of top-K parts

        Returns
        -------
        (B, embed_dim)
        """
        B = images.shape[0]
        # Gather top-K boxes
        selected = []
        for b in range(B):
            idx = top_k_indices[b]          # (top_k,)
            selected.append(boxes[b][idx])  # (top_k, 4)
        selected = torch.stack(selected, dim=0)  # (B, top_k, 4)

        # Crop and encode each part
        part_feats: List[torch.Tensor] = []
        for k in range(self.top_k):
            coords = selected[:, k, :]  # (B, 4) x1y1x2y2
            # Use RoI Align on the original image
            rois = torch.cat(
                [torch.arange(B, dtype=torch.float32, device=images.device).unsqueeze(1),
                 coords], dim=1
            )  # (B, 5)
            crops = roi_align(images, rois, output_size=self.part_size)  # (B, 3, ps, ps)
            feat = self.part_encoder(crops)  # (B, 128, 4, 4)
            feat = feat.view(B, -1)          # (B, 2048)
            part_feats.append(feat)

        concat = torch.cat(part_feats, dim=1)  # (B, 2048*top_k)
        return self.agg(concat)                # (B, embed_dim)


# ---------------------------------------------------------------------------
# NTS-Net
# ---------------------------------------------------------------------------

class NTSNet(nn.Module):
    """
    Navigator-Teacher-Scrutinizer Network for ultrasound structural defects.

    Parameters
    ----------
    embed_dim   : output embedding size (default 512)
    num_parts   : number of navigator proposals (default 6)
    top_k       : top proposals passed to Scrutinizer (default 3)
    num_classes : 1 for binary; set to 0 to strip head (surgery)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_parts: int = 6,
        top_k: int = 3,
        num_classes: int = 1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_parts = num_parts
        self.top_k = top_k

        # ── Teacher: ResNet-50 backbone ──
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        # outputs (B, 2048, 7, 7) for 224×224 input
        teacher_dim = 2048

        self.teacher_pool = nn.AdaptiveAvgPool2d(1)  # (B, 2048, 1, 1)
        self.teacher_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(teacher_dim, embed_dim),
            nn.GELU(),
        )

        # ── Navigator ──
        self.navigator = Navigator(in_channels=teacher_dim, num_parts=num_parts)

        # ── Scrutinizer ──
        self.scrutinizer = Scrutinizer(
            backbone_out=teacher_dim, top_k=top_k, embed_dim=embed_dim
        )

        # ── Fusion of Teacher + Scrutinizer ──
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # ── Classification Head ──
        self.num_classes = num_classes
        if num_classes > 0:
            self.classifier = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _top_k_parts(
        self, attention: torch.Tensor
    ) -> torch.Tensor:
        """
        Given attention maps (B, K, H, W), return indices of top-K parts
        ranked by max attention value.
        """
        B, K, H, W = attention.shape
        max_attn = attention.view(B, K, -1).max(dim=-1).values  # (B, K)
        top_k_idx = torch.topk(max_attn, self.top_k, dim=1).indices  # (B, top_k)
        return top_k_idx

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, 224, 224)

        Returns
        -------
        embed : (B, embed_dim)
        """
        # Teacher feature extraction
        feat_map = self.backbone(x)             # (B, 2048, 7, 7)

        # Teacher global embedding
        teacher_feat = self.teacher_pool(feat_map)   # (B, 2048, 1, 1)
        teacher_embed = self.teacher_proj(teacher_feat)  # (B, embed_dim)

        # Navigator proposals
        attention, boxes = self.navigator(feat_map, image_size=x.shape[-1])
        top_k_idx = self._top_k_parts(attention)    # (B, top_k)

        # Scrutinizer embedding
        scrutin_embed = self.scrutinizer(x, boxes, top_k_idx)  # (B, embed_dim)

        # Fuse
        fused = self.fusion(torch.cat([teacher_embed, scrutin_embed], dim=1))
        return fused  # (B, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed = self.forward_features(x)
        if self.num_classes > 0:
            return self.classifier(embed)
        return embed


# ---------------------------------------------------------------------------
# Utility: surgery
# ---------------------------------------------------------------------------

def ntsnet_without_head(checkpoint_path: str, device: str = "cpu") -> NTSNet:
    model = NTSNet(num_classes=1)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.num_classes = 0
    if hasattr(model, "classifier"):
        del model.classifier
    model.eval()
    return model.to(device)
