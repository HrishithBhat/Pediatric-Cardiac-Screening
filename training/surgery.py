"""
Surgery Script: Strip Classifier Heads from Specialist Models
==============================================================
Loads Phase-1 checkpoints, removes the fc layer, and re-saves
"surgery" checkpoints ready for Phase-2 GMU training.

Usage:
  python training/surgery.py \
      --audio_ckpt   checkpoints/audio_best.pth   \
      --us_ckpt      checkpoints/ultrasound_best.pth \
      --xray_ckpt    checkpoints/xray_best.pth    \
      --output_dir   checkpoints/surgery
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.crnn_heart_sound import CRNN2D
from models.nts_net_ultrasound import NTSNet
from models.efficientnet_xray import EfficientNetV2XRay


def perform_surgery(
    audio_ckpt: str,
    us_ckpt: str,
    xray_ckpt: str,
    output_dir: str = "checkpoints/surgery",
    device: str = "cpu",
):
    os.makedirs(output_dir, exist_ok=True)

    # ── CRNN ──
    print(f"Loading CRNN checkpoint: {audio_ckpt}")
    model = CRNN2D(num_classes=1)
    ckpt = torch.load(audio_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    # Remove classifier
    del model.classifier
    model.num_classes = 0
    out_path = os.path.join(output_dir, "audio_surgery.pth")
    torch.save({"model_state_dict": model.state_dict(), "num_classes": 0}, out_path)
    print(f"  Saved: {out_path}")

    # ── NTS-Net ──
    print(f"Loading NTSNet checkpoint: {us_ckpt}")
    model = NTSNet(num_classes=1)
    ckpt = torch.load(us_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    del model.classifier
    model.num_classes = 0
    out_path = os.path.join(output_dir, "ultrasound_surgery.pth")
    torch.save({"model_state_dict": model.state_dict(), "num_classes": 0}, out_path)
    print(f"  Saved: {out_path}")

    # ── EfficientNetV2 ──
    print(f"Loading EfficientNetV2 checkpoint: {xray_ckpt}")
    model = EfficientNetV2XRay(num_classes=1)
    ckpt = torch.load(xray_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    del model.classifier
    model.num_classes = 0
    out_path = os.path.join(output_dir, "xray_surgery.pth")
    torch.save({"model_state_dict": model.state_dict(), "num_classes": 0}, out_path)
    print(f"  Saved: {out_path}")

    print("Surgery complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_ckpt", required=True)
    parser.add_argument("--us_ckpt", required=True)
    parser.add_argument("--xray_ckpt", required=True)
    parser.add_argument("--output_dir", default="checkpoints/surgery")
    args = parser.parse_args()
    perform_surgery(args.audio_ckpt, args.us_ckpt, args.xray_ckpt, args.output_dir)
