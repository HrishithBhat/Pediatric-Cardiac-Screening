"""
Inference script — standalone prediction without a server.

Usage:
  python inference/infer.py \
      --audio    path/to/heart_sound.wav \
      --us       path/to/echo.jpg        \
      --xray     path/to/cxr.jpg         \
      --output   report_output/

Outputs:
  report_output/
    ├── report.json
    ├── audio_gradcam.png
    ├── ultrasound_gradcam.png
    └── xray_gradcam.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import CFG
from models.crnn_heart_sound import crnn_without_head
from models.nts_net_ultrasound import ntsnet_without_head
from models.efficientnet_xray import efficientnet_without_head
from models.gmu_fusion import MultimodalModel
from preprocessing.audio_preprocessing import AudioPreprocessor
from preprocessing.image_preprocessing import (
    get_ultrasound_preprocessor,
    get_xray_preprocessor,
)
from explainability.gradcam import generate_explainability_report


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUDIO_CKPT = os.getenv("AUDIO_CKPT", "checkpoints/audio_best.pth")
US_CKPT = os.getenv("US_CKPT", "checkpoints/ultrasound_best.pth")
XRAY_CKPT = os.getenv("XRAY_CKPT", "checkpoints/xray_best.pth")
GMU_CKPT = os.getenv("GMU_CKPT", "checkpoints/gmu_best.pth")


def load_multimodal_model():
    crnn = crnn_without_head(AUDIO_CKPT, device=DEVICE)
    nts = ntsnet_without_head(US_CKPT, device=DEVICE)
    eff = efficientnet_without_head(XRAY_CKPT, device=DEVICE)
    model = MultimodalModel(
        crnn_model=crnn,
        nts_model=nts,
        effnet_model=eff,
        audio_dim=CFG.fusion.audio_embed_dim,
        us_dim=CFG.fusion.us_embed_dim,
        xray_dim=CFG.fusion.xray_embed_dim,
        gmu_hidden=CFG.fusion.gmu_hidden_dim,
        mlp_hidden=CFG.fusion.mlp_hidden_dims,
        mlp_dropout=CFG.fusion.mlp_dropout,
        freeze_specialists=True,
    ).to(DEVICE)
    if os.path.exists(GMU_CKPT):
        ckpt = torch.load(GMU_CKPT, map_location=DEVICE)
        model.gmu.load_state_dict(ckpt["gmu_state_dict"])
        model.mlp.load_state_dict(ckpt["mlp_state_dict"])
    model.eval()
    return model


def run_inference(
    audio_path: str | None,
    us_path: str | None,
    xray_path: str | None,
    output_dir: str = "report_output",
    threshold: float = 0.5,
):
    os.makedirs(output_dir, exist_ok=True)
    model = load_multimodal_model()

    audio_pre = AudioPreprocessor()
    us_pre = get_ultrasound_preprocessor(augment=False)
    xray_pre = get_xray_preprocessor(augment=False)

    import numpy as np

    audio_spec, us_tensor, xray_tensor = None, None, None
    us_bgr, xray_bgr = None, None

    if audio_path and os.path.exists(audio_path):
        print(f"Processing audio: {audio_path}")
        audio_spec = audio_pre(audio_path).unsqueeze(0)

    if us_path and os.path.exists(us_path):
        print(f"Processing ultrasound: {us_path}")
        us_bgr = cv2.imread(us_path, cv2.IMREAD_COLOR)
        us_tensor = us_pre(us_bgr).unsqueeze(0)

    if xray_path and os.path.exists(xray_path):
        print(f"Processing X-ray: {xray_path}")
        xray_bgr = cv2.imread(xray_path, cv2.IMREAD_COLOR)
        xray_tensor = xray_pre(xray_bgr).unsqueeze(0)

    if audio_spec is None and us_tensor is None and xray_tensor is None:
        print("ERROR: No valid inputs found.")
        return

    print("Running inference...")
    report = generate_explainability_report(
        model,
        audio_spec=audio_spec,
        us_image=us_tensor,
        xray_image=xray_tensor,
        us_bgr=us_bgr,
        xray_bgr=xray_bgr,
        device=DEVICE,
    )

    probability = report["prediction"]
    decision = "REFER" if probability >= threshold else "PASS"

    # Save report JSON
    json_report = {
        "decision": decision,
        "probability_of_chd": round(probability, 4),
        "threshold_used": threshold,
        "modality_gate_weights": report["gate_weights"],
        "inputs": {
            "audio": audio_path,
            "ultrasound": us_path,
            "xray": xray_path,
        },
    }
    json_path = os.path.join(output_dir, "report.json")
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)
    print(f"Report saved: {json_path}")

    # Save Grad-CAM images
    if report["audio_cam_overlay"] is not None:
        p = os.path.join(output_dir, "audio_gradcam.png")
        cv2.imwrite(p, report["audio_cam_overlay"])
        print(f"Audio Grad-CAM: {p}")

    if report["us_cam_overlay"] is not None:
        p = os.path.join(output_dir, "ultrasound_gradcam.png")
        cv2.imwrite(p, report["us_cam_overlay"])
        print(f"Ultrasound Grad-CAM: {p}")

    if report["xray_cam_overlay"] is not None:
        p = os.path.join(output_dir, "xray_gradcam.png")
        cv2.imwrite(p, report["xray_cam_overlay"])
        print(f"X-Ray Grad-CAM: {p}")

    print(f"\n{'='*50}")
    print(f"DECISION: {decision}")
    print(f"CHD Probability: {probability*100:.2f}%")
    print(f"Gate Weights: {report['gate_weights']}")
    print(f"{'='*50}")
    return json_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone inference script")
    parser.add_argument("--audio", default=None, help="Path to WAV file")
    parser.add_argument("--us", default=None, help="Path to ultrasound image")
    parser.add_argument("--xray", default=None, help="Path to chest X-ray image")
    parser.add_argument("--output", default="report_output", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    run_inference(
        audio_path=args.audio,
        us_path=args.us,
        xray_path=args.xray,
        output_dir=args.output,
        threshold=args.threshold,
    )
