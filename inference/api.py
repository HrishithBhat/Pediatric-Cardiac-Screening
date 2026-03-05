"""
FastAPI Inference Interface
============================
Exposes a REST API for the doctor-facing referral dashboard.

Endpoints:
  POST /predict      — accepts WAV + up to 2 JPG files, returns JSON report
  GET  /health       — health check
  GET  /report/{id}  — retrieve a cached report with Grad-CAM images

Run:
  uvicorn inference.api:app --reload --host 0.0.0.0 --port 8000

Dependencies:
  fastapi, uvicorn, python-multipart, Pillow
"""

from __future__ import annotations

import base64
import io
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

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


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Pediatric Cardiac Screening API",
    description="Autonomous CHD detection via multimodal deep learning.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model singleton (loaded once on startup)
# ---------------------------------------------------------------------------

class ModelRegistry:
    model: Optional[MultimodalModel] = None
    device: str = "cpu"


registry = ModelRegistry()

# Environment-variable overrides for checkpoint paths
AUDIO_CKPT = os.getenv("AUDIO_CKPT", "checkpoints/audio_best.pth")
US_CKPT = os.getenv("US_CKPT", "checkpoints/ultrasound_best.pth")
XRAY_CKPT = os.getenv("XRAY_CKPT", "checkpoints/xray_best.pth")
GMU_CKPT = os.getenv("GMU_CKPT", "checkpoints/gmu_best.pth")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


@app.on_event("startup")
def load_model():
    registry.device = DEVICE
    dev = registry.device

    print(f"Loading models on {dev}...")
    crnn = crnn_without_head(AUDIO_CKPT, device=dev)
    nts = ntsnet_without_head(US_CKPT, device=dev)
    eff = efficientnet_without_head(XRAY_CKPT, device=dev)

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
    ).to(dev)

    # Load GMU weights
    if os.path.exists(GMU_CKPT):
        ckpt = torch.load(GMU_CKPT, map_location=dev)
        model.gmu.load_state_dict(ckpt["gmu_state_dict"])
        model.mlp.load_state_dict(ckpt["mlp_state_dict"])
        print("GMU weights loaded.")
    else:
        print(f"Warning: GMU checkpoint not found at {GMU_CKPT}. Using random weights.")

    model.eval()
    registry.model = model
    print("Model ready.")


# ---------------------------------------------------------------------------
# Preprocessors
# ---------------------------------------------------------------------------

audio_pre = AudioPreprocessor()
us_pre = get_ultrasound_preprocessor(augment=False)
xray_pre = get_xray_preprocessor(augment=False)

# In-memory report cache  {report_id: dict}
_report_cache: dict = {}


# ---------------------------------------------------------------------------
# Helper: encode numpy image to base64 PNG
# ---------------------------------------------------------------------------

def _encode_image(img: np.ndarray) -> str:
    """BGR uint8 → base64-encoded PNG string."""
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "device": registry.device}


@app.post("/predict")
async def predict(
    audio_file: Optional[UploadFile] = File(default=None, description="Heart sound WAV file"),
    us_file: Optional[UploadFile] = File(default=None, description="Ultrasound JPG/PNG"),
    xray_file: Optional[UploadFile] = File(default=None, description="Chest X-ray JPG/PNG"),
):
    """
    Accepts up to three files and returns a structured referral report.

    At least one file must be provided.
    """
    if audio_file is None and us_file is None and xray_file is None:
        raise HTTPException(status_code=400, detail="At least one input file is required.")

    if registry.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    dev = registry.device
    audio_spec, us_tensor, xray_tensor = None, None, None
    us_bgr, xray_bgr = None, None

    # ── Process Audio ──
    if audio_file is not None:
        try:
            audio_bytes = await audio_file.read()
            import tempfile, soundfile as sf
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            audio_spec = audio_pre(tmp_path).unsqueeze(0)  # (1, 1, 64, 256)
            os.unlink(tmp_path)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Audio processing error: {e}")

    # ── Process Ultrasound ──
    if us_file is not None:
        try:
            us_bytes = await us_file.read()
            us_np = np.frombuffer(us_bytes, dtype=np.uint8)
            us_bgr = cv2.imdecode(us_np, cv2.IMREAD_COLOR)
            us_tensor = us_pre(us_bgr).unsqueeze(0)  # (1, 3, 224, 224)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Ultrasound processing error: {e}")

    # ── Process X-Ray ──
    if xray_file is not None:
        try:
            xray_bytes = await xray_file.read()
            xray_np = np.frombuffer(xray_bytes, dtype=np.uint8)
            xray_bgr = cv2.imdecode(xray_np, cv2.IMREAD_COLOR)
            xray_tensor = xray_pre(xray_bgr).unsqueeze(0)  # (1, 3, 224, 224)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"X-ray processing error: {e}")

    # ── Inference + Grad-CAM ──
    report_data = generate_explainability_report(
        registry.model,
        audio_spec=audio_spec,
        us_image=us_tensor,
        xray_image=xray_tensor,
        us_bgr=us_bgr,
        xray_bgr=xray_bgr,
        device=dev,
    )

    probability = report_data["prediction"]
    label = "REFER" if probability >= 0.5 else "PASS"
    confidence = probability if label == "REFER" else 1.0 - probability

    # Build gate weights summary
    gate_info = report_data["gate_weights"] or {}

    # Build Grad-CAM images (base64)
    images = {}
    if report_data["audio_cam_overlay"] is not None:
        images["audio_gradcam"] = _encode_image(report_data["audio_cam_overlay"])
    if report_data["us_cam_overlay"] is not None:
        images["ultrasound_gradcam"] = _encode_image(report_data["us_cam_overlay"])
    if report_data["xray_cam_overlay"] is not None:
        images["xray_gradcam"] = _encode_image(report_data["xray_cam_overlay"])

    # Assemble report
    report_id = str(uuid.uuid4())
    report = {
        "report_id": report_id,
        "decision": label,
        "probability_of_chd": round(probability, 4),
        "confidence": round(confidence, 4),
        "modality_reliability": {
            k: round(v, 4) for k, v in gate_info.items()
        },
        "gradcam_images": images,
        "advice": (
            "Cardiology referral recommended. "
            "Please review Grad-CAM highlights for clinical correlation."
            if label == "REFER"
            else "No significant cardiac abnormality detected by AI screening. "
                 "Routine follow-up advised."
        ),
    }

    # Cache for retrieval
    _report_cache[report_id] = report

    # Return without base64 images inline (return as separate endpoint)
    response = {k: v for k, v in report.items() if k != "gradcam_images"}
    response["has_gradcam"] = bool(images)
    response["report_url"] = f"/report/{report_id}"
    return JSONResponse(content=response)


@app.get("/report/{report_id}")
def get_report(report_id: str):
    """Retrieve full cached report including base64 Grad-CAM images."""
    if report_id not in _report_cache:
        raise HTTPException(status_code=404, detail="Report not found.")
    return JSONResponse(content=_report_cache[report_id])
