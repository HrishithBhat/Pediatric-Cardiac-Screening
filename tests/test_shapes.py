"""
Quick smoke-test to verify model shapes before real training.
Run: python tests/test_shapes.py
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from models.crnn_heart_sound import CRNN2D
from models.nts_net_ultrasound import NTSNet
from models.efficientnet_xray import EfficientNetV2XRay
from models.gmu_fusion import (
    GatedMultimodalUnit, MLPClassifier, ModalityDropout, MultimodalModel
)


def test_crnn():
    model = CRNN2D(embed_dim=512, lstm_hidden=256, lstm_layers=2, num_classes=1)
    x = torch.randn(2, 1, 64, 256)
    logit = model(x)
    embed = model.forward_features(x)
    assert logit.shape == (2, 1), f"CRNN logit shape: {logit.shape}"
    assert embed.shape == (2, 512), f"CRNN embed shape: {embed.shape}"
    print(f"  ✓ CRNN2D  logit={logit.shape}  embed={embed.shape}")


def test_ntsnet():
    model = NTSNet(embed_dim=512, num_parts=6, top_k=3, num_classes=1)
    x = torch.randn(2, 3, 224, 224)
    logit = model(x)
    embed = model.forward_features(x)
    assert logit.shape == (2, 1), f"NTSNet logit shape: {logit.shape}"
    assert embed.shape == (2, 512), f"NTSNet embed shape: {embed.shape}"
    print(f"  ✓ NTSNet   logit={logit.shape}  embed={embed.shape}")


def test_efficientnet():
    model = EfficientNetV2XRay(embed_dim=512, num_classes=1)
    x = torch.randn(2, 3, 224, 224)
    logit = model(x)
    embed = model.forward_features(x)
    assert logit.shape == (2, 1), f"EffNet logit shape: {logit.shape}"
    assert embed.shape == (2, 512), f"EffNet embed shape: {embed.shape}"
    print(f"  ✓ EfficientNetV2  logit={logit.shape}  embed={embed.shape}")


def test_gmu():
    gmu = GatedMultimodalUnit(512, 512, 512, hidden_dim=512)
    mlp = MLPClassifier(in_dim=512 * 3, hidden_dims=[512, 256], num_classes=1, dropout=0.5)
    e_a = torch.randn(2, 512)
    e_u = torch.randn(2, 512)
    e_x = torch.randn(2, 512)
    fused, gates = gmu(e_a, e_u, e_x)
    logit = mlp(fused)
    assert fused.shape == (2, 1536), f"GMU fused shape: {fused.shape}"
    assert logit.shape == (2, 1), f"MLP logit shape: {logit.shape}"
    assert set(gates.keys()) == {"audio", "ultrasound", "xray"}
    print(f"  ✓ GMU  fused={fused.shape}  logit={logit.shape}  gates={list(gates.keys())}")


def test_modality_dropout():
    md = ModalityDropout(drop_p=1.0)  # always drop
    md.train()
    e = torch.ones(2, 512)
    dropped, _, _ = md(e, e, e)
    assert dropped.sum() == 0.0, "Modality dropout not zeroing tensors"
    print("  ✓ ModalityDropout zeroes embeddings at p=1.0")


def test_multimodal_model():
    """Integration test: full pipeline with dummy weights."""
    crnn = CRNN2D(num_classes=0)  # no head — embedding-only
    nts = NTSNet(num_classes=0)
    eff = EfficientNetV2XRay(num_classes=0)
    model = MultimodalModel(
        crnn_model=crnn, nts_model=nts, effnet_model=eff, freeze_specialists=True
    )
    audio = torch.randn(1, 1, 64, 256)
    us = torch.randn(1, 3, 224, 224)
    xray = torch.randn(1, 3, 224, 224)
    logit, gates = model(audio, us, xray)
    assert logit.shape == (1, 1), f"MultimodalModel logit: {logit.shape}"
    prob = model.predict_proba(audio, us, xray)
    assert 0.0 <= prob.item() <= 1.0
    print(f"  ✓ MultimodalModel  logit={logit.shape}  prob={prob.item():.3f}")

    # Test missing modality
    logit_no_audio, _ = model(None, us, xray)
    assert logit_no_audio.shape == (1, 1)
    print(f"  ✓ Missing-modality  logit={logit_no_audio.shape}")


if __name__ == "__main__":
    print("\n=== Shape Smoke Tests ===\n")
    test_crnn()
    test_ntsnet()
    test_efficientnet()
    test_gmu()
    test_modality_dropout()
    test_multimodal_model()
    print("\n✅ All shape tests passed.\n")
