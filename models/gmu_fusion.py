"""
Gated Multimodal Unit (GMU) + Multimodal Model Wrapper
=======================================================
Level-1 Meta-Learner for the Stacked Multimodal Ensemble.

Design
------
  Inputs : embeddings from three specialist encoders
             e_audio  (B, D_a)  — CRNN
             e_us     (B, D_u)  — NTS-Net
             e_xray   (B, D_x)  — EfficientNetV2-S

  GMU    :
    1. Linear projection of each modality → shared dim H
    2. Sigmoid gating per modality  (autonomously weighs reliability)
    3. Element-wise gate × projected feature for each modality
    4. Concatenation of gated features  → (B, 3H)
    5. MLP with Dropout(0.5) → binary logit

  Modality Dropout :
    During training, each modality embedding is independently zeroed out
    with probability p_drop, teaching the GMU to be robust to missing inputs.

Outputs
-------
  logit : (B, 1)   raw (un-sigmoided) score for BCEWithLogitsLoss
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sigmoid Gating Module  (one per modality)
# ---------------------------------------------------------------------------

class SigmoidGate(nn.Module):
    """
    Computes a per-dimension gate vector in (0, 1) for one modality.

    z = σ( W_g · e + b_g )
    output = z ⊙ ( W_v · e + b_v )

    Parameters
    ----------
    in_dim  : input embedding dimension
    out_dim : hidden / projected dimension
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # Value branch
        self.value = nn.Linear(in_dim, out_dim)
        # Gate branch
        self.gate = nn.Linear(in_dim, out_dim)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        e : (B, in_dim)

        Returns
        -------
        gated : (B, out_dim)   — gate-weighted projected feature
        gate_weights : (B, out_dim) — the gate values (for interpretability)
        """
        z = torch.sigmoid(self.gate(e))   # (B, out_dim)
        v = self.value(e)                  # (B, out_dim)
        return z * v, z


# ---------------------------------------------------------------------------
# Gated Multimodal Unit  (Core GMU)
# ---------------------------------------------------------------------------

class GatedMultimodalUnit(nn.Module):
    """
    GMU that fuses three modality embeddings via per-modality Sigmoid Gates.

    Parameters
    ----------
    audio_dim : dimension of CRNN embedding
    us_dim    : dimension of NTS-Net embedding
    xray_dim  : dimension of EfficientNet embedding
    hidden_dim: projected dimension for each modality gate (default 512)
    """

    def __init__(
        self,
        audio_dim: int = 512,
        us_dim: int = 512,
        xray_dim: int = 512,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Per-modality Sigmoid gating layers
        self.gate_audio = SigmoidGate(audio_dim, hidden_dim)
        self.gate_us = SigmoidGate(us_dim, hidden_dim)
        self.gate_xray = SigmoidGate(xray_dim, hidden_dim)

        # Cross-modal interaction (optional learnable mix)
        self.cross_norm = nn.LayerNorm(hidden_dim * 3)

    def forward(
        self,
        e_audio: torch.Tensor,
        e_us: torch.Tensor,
        e_xray: torch.Tensor,
    ):
        """
        Returns
        -------
        fused       : (B, hidden_dim * 3)
        gate_weights: dict with per-modality gate vectors for interpretability
        """
        ga, w_a = self.gate_audio(e_audio)   # (B, H)
        gu, w_u = self.gate_us(e_us)
        gx, w_x = self.gate_xray(e_xray)

        fused = torch.cat([ga, gu, gx], dim=1)  # (B, 3H)
        fused = self.cross_norm(fused)

        gate_weights = {
            "audio": w_a.detach(),
            "ultrasound": w_u.detach(),
            "xray": w_x.detach(),
        }
        return fused, gate_weights


# ---------------------------------------------------------------------------
# MLP Classifier Head
# ---------------------------------------------------------------------------

class MLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron with Dropout for binary classification.

    Parameters
    ----------
    in_dim       : input dimension (typically hidden_dim * 3)
    hidden_dims  : list of hidden layer sizes
    num_classes  : output classes (1 for binary)
    dropout      : dropout probability (default 0.5)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int] = None,
        num_classes: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(p=dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, num_classes)


# ---------------------------------------------------------------------------
# Modality Dropout Utility
# ---------------------------------------------------------------------------

class ModalityDropout(nn.Module):
    """
    Randomly zeroes out an entire modality embedding during training.
    At inference time this is a no-op (equivalent to the modality being absent).

    If an embedding is intentionally missing at inference (e.g., no audio file
    provided), simply pass a zero tensor of the correct shape.

    Parameters
    ----------
    drop_p : probability that each modality is zeroed out (default 0.2)
    """

    def __init__(self, drop_p: float = 0.2):
        super().__init__()
        self.drop_p = drop_p

    def forward(
        self,
        e_audio: torch.Tensor,
        e_us: torch.Tensor,
        e_xray: torch.Tensor,
    ):
        if self.training:
            e_audio = self._maybe_drop(e_audio)
            e_us = self._maybe_drop(e_us)
            e_xray = self._maybe_drop(e_xray)
        return e_audio, e_us, e_xray

    def _maybe_drop(self, e: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.drop_p:
            return torch.zeros_like(e)
        return e


# ---------------------------------------------------------------------------
# Full Multimodal Model  (Level-0 specialists + Level-1 GMU)
# ---------------------------------------------------------------------------

class MultimodalModel(nn.Module):
    """
    Complete end-to-end multimodal model.

    In Phase-2 training, the specialist encoders are frozen and only
    the GMU + MLP weights are updated.

    Parameters
    ----------
    crnn_model    : CRNN2D instance (surgery-mode, no classifier head)
    nts_model     : NTSNet instance (surgery-mode)
    effnet_model  : EfficientNetV2XRay instance (surgery-mode)
    audio_dim     : CRNN embedding dimension
    us_dim        : NTS-Net embedding dimension
    xray_dim      : EfficientNet embedding dimension
    gmu_hidden    : GMU projected dimension per modality
    mlp_hidden    : hidden layer dims for MLP
    mlp_dropout   : MLP dropout probability
    modality_drop_p: modality dropout probability during training
    freeze_specialists: if True, freeze all specialist encoder parameters
    """

    def __init__(
        self,
        crnn_model: nn.Module,
        nts_model: nn.Module,
        effnet_model: nn.Module,
        audio_dim: int = 512,
        us_dim: int = 512,
        xray_dim: int = 512,
        gmu_hidden: int = 512,
        mlp_hidden: list[int] = None,
        mlp_dropout: float = 0.5,
        modality_drop_p: float = 0.2,
        freeze_specialists: bool = True,
    ):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = [512, 256]

        # ── Level-0 Specialist Encoders ──
        self.crnn = crnn_model
        self.nts_net = nts_model
        self.effnet = effnet_model

        if freeze_specialists:
            for model in [self.crnn, self.nts_net, self.effnet]:
                for p in model.parameters():
                    p.requires_grad = False

        # ── Modality Dropout ──
        self.modality_dropout = ModalityDropout(drop_p=modality_drop_p)

        # ── Level-1 GMU ──
        self.gmu = GatedMultimodalUnit(
            audio_dim=audio_dim,
            us_dim=us_dim,
            xray_dim=xray_dim,
            hidden_dim=gmu_hidden,
        )

        # ── MLP Classifier ──
        self.mlp = MLPClassifier(
            in_dim=gmu_hidden * 3,
            hidden_dims=mlp_hidden,
            num_classes=1,
            dropout=mlp_dropout,
        )

    def encode(
        self,
        audio_spec: Optional[torch.Tensor] = None,
        us_image: Optional[torch.Tensor] = None,
        xray_image: Optional[torch.Tensor] = None,
    ):
        """
        Extract embeddings from each specialist.
        Missing modalities should be passed as None (returns zero embedding).
        """
        B = next(
            t.shape[0] for t in [audio_spec, us_image, xray_image] if t is not None
        )
        dev = next(self.parameters()).device

        # Audio
        if audio_spec is not None:
            e_audio = self.crnn.forward_features(audio_spec.to(dev))
        else:
            e_audio = torch.zeros(B, self.gmu.gate_audio.value.in_features, device=dev)

        # Ultrasound
        if us_image is not None:
            e_us = self.nts_net.forward_features(us_image.to(dev))
        else:
            e_us = torch.zeros(B, self.gmu.gate_us.value.in_features, device=dev)

        # X-Ray
        if xray_image is not None:
            e_xray = self.effnet.forward_features(xray_image.to(dev))
        else:
            e_xray = torch.zeros(B, self.gmu.gate_xray.value.in_features, device=dev)

        return e_audio, e_us, e_xray

    def forward(
        self,
        audio_spec: Optional[torch.Tensor] = None,
        us_image: Optional[torch.Tensor] = None,
        xray_image: Optional[torch.Tensor] = None,
    ):
        """
        Full forward pass.

        Returns
        -------
        logit        : (B, 1)  raw logit for BCEWithLogitsLoss
        gate_weights : dict of per-modality gate tensors
        """
        e_audio, e_us, e_xray = self.encode(audio_spec, us_image, xray_image)

        # Modality dropout during training
        e_audio, e_us, e_xray = self.modality_dropout(e_audio, e_us, e_xray)

        # GMU fusion
        fused, gate_weights = self.gmu(e_audio, e_us, e_xray)

        # MLP classification
        logit = self.mlp(fused)  # (B, 1)

        return logit, gate_weights

    def predict_proba(
        self,
        audio_spec: Optional[torch.Tensor] = None,
        us_image: Optional[torch.Tensor] = None,
        xray_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return probability (0=Pass, 1=Refer)."""
        logit, _ = self.forward(audio_spec, us_image, xray_image)
        return torch.sigmoid(logit)

    def predict(
        self,
        audio_spec: Optional[torch.Tensor] = None,
        us_image: Optional[torch.Tensor] = None,
        xray_image: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Return binary label (0=Pass, 1=Refer)."""
        prob = self.predict_proba(audio_spec, us_image, xray_image)
        return (prob >= threshold).long()
