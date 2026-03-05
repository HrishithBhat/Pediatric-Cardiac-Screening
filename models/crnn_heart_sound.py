"""
2D-CRNN for Heart Sound Classification
========================================
Architecture:
  Input  : Log-Mel Spectrogram  (B, 1, 64, 256)
  Stage 1: ResNet-18 CNN Backbone  → feature map → global avg pool → (B, 512)
  Stage 2: Reshape to temporal sequence (B, T, 512/T_feat)
           Bidirectional LSTM  (2 layers, hidden=256)  → (B, T, 512)
  Stage 3: Temporal Attention Layer  → weighted pooling  → (B, 512)
  Head   : Linear → logit  (removed during "surgery", kept for Phase-1 training)

Embed dim = 512  (used by the GMU)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


# ---------------------------------------------------------------------------
# Temporal Attention Layer
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """
    Soft temporal attention over a sequence of hidden states.

    Input  : (B, T, H)
    Output : (B, H)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        scores = self.attention(x)           # (B, T, 1)
        weights = torch.softmax(scores, dim=1)  # (B, T, 1)
        context = (x * weights).sum(dim=1)   # (B, H)
        return context


# ---------------------------------------------------------------------------
# 2D-CRNN Model
# ---------------------------------------------------------------------------

class CRNN2D(nn.Module):
    """
    Convolutional Recurrent Neural Network for heart-sound classification.

    Parameters
    ----------
    embed_dim   : output embedding dimension (default 512)
    lstm_hidden : hidden units per direction of BiLSTM (default 256)
    lstm_layers : number of LSTM layers (default 2)
    num_classes : 1 for binary classification; set to 0 to skip head
    """

    def __init__(
        self,
        embed_dim: int = 512,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        num_classes: int = 1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # ── ResNet-18 backbone (modified for 1-channel input) ──
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Replace first conv: 1-channel spectrogram instead of RGB
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Remove the original avgpool + fc; keep up to layer4
        self.cnn = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        # backbone.layer4 outputs (B, 512, H/32, W/32)
        # For input (1, 64, 256): → (B, 512, 2, 8)  → temporal axis = 8

        # Adaptive pool: collapse frequency axis, keep time axis
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))  # (B, 512, 1, T_feat)

        # ── BiLSTM ──
        lstm_input_dim = 512
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * 2  # bidirectional

        # ── Temporal Attention ──
        self.attention = TemporalAttention(lstm_out_dim)

        # Project to embed_dim
        self.embed_proj = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Linear(lstm_out_dim, embed_dim),
            nn.GELU(),
        )

        # ── Classification Head (removed during surgery) ──
        self.num_classes = num_classes
        if num_classes > 0:
            self.classifier = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in [self.embed_proj, self.attention]:
            for layer in (m.modules() if hasattr(m, "modules") else [m]):
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embedding without the classification head.

        Parameters
        ----------
        x : (B, 1, 64, 256)

        Returns
        -------
        embed : (B, embed_dim)
        """
        # CNN → (B, 512, H', W')
        feat = self.cnn(x)
        # Collapse frequency → (B, 512, 1, T_feat)
        feat = self.freq_pool(feat)
        # (B, 512, T_feat)
        feat = feat.squeeze(2)
        # (B, T_feat, 512)
        feat = feat.permute(0, 2, 1)
        # BiLSTM → (B, T_feat, 512)
        lstm_out, _ = self.lstm(feat)
        # Temporal attention → (B, 512)
        context = self.attention(lstm_out)
        # Projection → (B, embed_dim)
        embed = self.embed_proj(context)
        return embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass.

        Returns
        -------
        If num_classes > 0 : logit (B, 1)
        Else               : embedding (B, embed_dim)
        """
        embed = self.forward_features(x)
        if self.num_classes > 0:
            return self.classifier(embed)
        return embed


# ---------------------------------------------------------------------------
# Utility: strip the classification head (Phase-1 → Phase-2 surgery)
# ---------------------------------------------------------------------------

def crnn_without_head(checkpoint_path: str, device: str = "cpu") -> CRNN2D:
    """
    Load a trained CRNN2D, strip the classifier head, and return an
    embedding-only model.

    Parameters
    ----------
    checkpoint_path : path to .pth saved by Phase-1 training
    device          : 'cuda' or 'cpu'

    Returns
    -------
    model : CRNN2D with num_classes=0  (produces embeddings)
    """
    model = CRNN2D(num_classes=1)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.num_classes = 0
    if hasattr(model, "classifier"):
        del model.classifier
    model.eval()
    return model.to(device)
