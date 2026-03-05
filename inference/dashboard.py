"""
Streamlit Doctor Dashboard
============================
Interactive web UI for the Pediatric Cardiac Screening System.

Run:
  streamlit run inference/dashboard.py

Layout:
  ┌────────────────────────────────────────────────────┐
  │  🫀 Pediatric Cardiac Screening                    │
  ├──────────────┬─────────────────────────────────────┤
  │ Upload Panel │  Results Panel                      │
  │  - WAV       │  - PASS / REFER badge               │
  │  - US JPG    │  - Confidence bar                   │
  │  - CXR JPG   │  - Modality reliability bars        │
  │              │  - Grad-CAM images                  │
  │              │  - Clinical advice                  │
  └──────────────┴─────────────────────────────────────┘
"""

from __future__ import annotations

import base64
import io
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st
import torch
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
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Pediatric Cardiac Screening",
    page_icon="🫀",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached model loader
# ---------------------------------------------------------------------------

AUDIO_CKPT = os.getenv("AUDIO_CKPT", "checkpoints/audio_best.pth")
US_CKPT = os.getenv("US_CKPT", "checkpoints/ultrasound_best.pth")
XRAY_CKPT = os.getenv("XRAY_CKPT", "checkpoints/xray_best.pth")
GMU_CKPT = os.getenv("GMU_CKPT", "checkpoints/gmu_best.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource(show_spinner="Loading AI models...")
def load_model():
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


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main():
    st.title("🫀 Autonomous Pediatric Cardiac Screening")
    st.caption(
        "Upload patient data below. The AI will analyse all available modalities "
        "and generate a clinical referral report."
    )
    st.divider()

    # ── Sidebar ──
    with st.sidebar:
        st.header("📋 Patient Information")
        patient_id = st.text_input("Patient ID", value="PT-0001")
        patient_age = st.number_input("Age (months)", min_value=0, max_value=216, value=6)
        st.divider()
        threshold = st.slider(
            "Referral Threshold", min_value=0.3, max_value=0.8,
            value=0.5, step=0.05,
            help="Probability above which the patient is referred."
        )
        st.divider()
        st.info("ℹ️ At least one modality is required for inference.")

    col_upload, col_results = st.columns([1, 2])

    with col_upload:
        st.subheader("📂 Upload Modalities")

        audio_file = st.file_uploader(
            "🔊 Heart Sound (WAV)", type=["wav"],
            help="PCG recording. Will be resampled to 2 kHz automatically."
        )
        us_file = st.file_uploader(
            "🖥️ Echocardiogram Still (JPG/PNG)", type=["jpg", "jpeg", "png"],
            help="Parasternal or apical 4-chamber view preferred."
        )
        xray_file = st.file_uploader(
            "🩻 Chest X-Ray (JPG/PNG)", type=["jpg", "jpeg", "png"],
            help="PA or AP projection. Will be CLAHE-enhanced automatically."
        )

        run = st.button(
            "🔍 Run Screening", type="primary",
            disabled=(audio_file is None and us_file is None and xray_file is None)
        )

    with col_results:
        if not run:
            st.info("Upload files and click **Run Screening** to begin.")
            return

        with st.spinner("Running AI inference..."):
            model = load_model()
            audio_pre = AudioPreprocessor()
            us_pre = get_ultrasound_preprocessor(augment=False)
            xray_pre = get_xray_preprocessor(augment=False)

            audio_spec, us_tensor, xray_tensor = None, None, None
            us_bgr, xray_bgr = None, None

            # Process audio
            if audio_file is not None:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_file.read())
                    tmp_path = tmp.name
                try:
                    audio_spec = audio_pre(tmp_path).unsqueeze(0)
                except Exception as e:
                    st.warning(f"Audio processing failed: {e}")
                finally:
                    os.unlink(tmp_path)

            # Process ultrasound
            if us_file is not None:
                try:
                    us_arr = np.frombuffer(us_file.read(), dtype=np.uint8)
                    us_bgr = cv2.imdecode(us_arr, cv2.IMREAD_COLOR)
                    us_tensor = us_pre(us_bgr).unsqueeze(0)
                except Exception as e:
                    st.warning(f"Ultrasound processing failed: {e}")

            # Process X-ray
            if xray_file is not None:
                try:
                    xr_arr = np.frombuffer(xray_file.read(), dtype=np.uint8)
                    xray_bgr = cv2.imdecode(xr_arr, cv2.IMREAD_COLOR)
                    xray_tensor = xray_pre(xray_bgr).unsqueeze(0)
                except Exception as e:
                    st.warning(f"X-ray processing failed: {e}")

            # Run inference
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
        confidence = probability if decision == "REFER" else 1.0 - probability

        # ── Decision Badge ──
        if decision == "REFER":
            st.error(f"## 🚨 REFER  —  {probability*100:.1f}% CHD Probability")
        else:
            st.success(f"## ✅ PASS  —  {(1-probability)*100:.1f}% Normal Probability")

        st.progress(float(probability), text=f"CHD Probability: {probability*100:.1f}%")

        # ── Modality Reliability ──
        st.subheader("🎛️ Modality Reliability (Gate Weights)")
        if report["gate_weights"]:
            cols = st.columns(3)
            icons = {"audio": "🔊", "ultrasound": "🖥️", "xray": "🩻"}
            for col, (mod, val) in zip(cols, report["gate_weights"].items()):
                with col:
                    st.metric(label=f"{icons.get(mod, '')} {mod.title()}", value=f"{val:.3f}")
                    st.progress(float(min(val, 1.0)))

        st.divider()

        # ── Grad-CAM Images ──
        st.subheader("🔬 Grad-CAM Explainability")
        cam_cols = st.columns(3)
        cam_map = [
            ("audio_cam_overlay", "🔊 Heart Sound Saliency"),
            ("us_cam_overlay", "🖥️ Ultrasound Attention"),
            ("xray_cam_overlay", "🩻 X-Ray Attention"),
        ]
        for col, (key, title) in zip(cam_cols, cam_map):
            with col:
                st.caption(title)
                if report[key] is not None:
                    rgb = cv2.cvtColor(report[key], cv2.COLOR_BGR2RGB)
                    st.image(rgb, use_column_width=True)
                else:
                    st.info("Not provided")

        st.divider()

        # ── Clinical Advice ──
        st.subheader("📝 Clinical Advice")
        if decision == "REFER":
            st.warning(
                f"**Patient {patient_id}** (age {patient_age} months) — "
                "AI screening indicates a significant probability of Congenital Heart Disease. "
                "**Cardiology referral is recommended.** "
                "Please correlate with clinical examination and echocardiographic findings."
            )
        else:
            st.info(
                f"**Patient {patient_id}** (age {patient_age} months) — "
                "AI screening does not detect significant cardiac abnormality. "
                "Routine follow-up as per standard protocols advised."
            )

        st.caption(
            "⚠️ This is an AI-assisted screening tool. "
            "It does NOT replace clinical judgement or formal diagnostic evaluation."
        )


if __name__ == "__main__":
    main()
