# 🫀 Autonomous Pediatric Cardiac Screening System

> **Stacked Multimodal Ensemble for Congenital Heart Disease Detection**  
> Detects CHD (VSD, ASD, Cardiomegaly, Pulmonary Plethora) from Heart Sounds, Echocardiograms, and Chest X-Rays using a two-level ensemble.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     LEVEL-0 SPECIALIST ENCODERS                     │
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────┐ │
│  │   2D-CRNN        │  │   NTS-Net        │  │  EfficientNetV2-S │ │
│  │  Heart Sounds    │  │  Ultrasound      │  │  Chest X-Ray      │ │
│  │                  │  │                  │  │                   │ │
│  │ Log-Mel Spec →   │  │ ResNet-50 →      │  │ EfficientNetV2 →  │ │
│  │ ResNet-18 CNN →  │  │ Navigator →      │  │ Embed Proj        │ │
│  │ BiLSTM →         │  │ Scrutinizer →    │  │                   │ │
│  │ Temporal Attn    │  │ Fusion           │  │ (Grad-CAM hooked) │ │
│  │         ↓        │  │         ↓        │  │         ↓         │ │
│  │    e_audio        │  │    e_us          │  │    e_xray         │ │
│  │   (B, 512)       │  │   (B, 512)       │  │   (B, 512)        │ │
│  └──────────────────┘  └──────────────────┘  └───────────────────┘ │
└────────────────────────────┬────────────────────────────────────────┘
                             │  Modality Dropout (p=0.2)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LEVEL-1 META-LEARNER (GMU)                       │
│                                                                     │
│  SigmoidGate(audio) → z_a ⊙ v_a  (B, 512)                        │
│  SigmoidGate(us)    → z_u ⊙ v_u  (B, 512)                        │
│  SigmoidGate(xray)  → z_x ⊙ v_x  (B, 512)                        │
│                         │                                           │
│                  Concat  →  (B, 1536)                               │
│                         │                                           │
│            MLP [1536→512→256→1] + Dropout(0.5)                     │
│                         │                                           │
│              BCEWithLogitsLoss ←── label (0=Pass, 1=Refer)         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
pediatric_cardiac_screening/
├── configs/
│   └── config.py                 # All hyperparameters & paths
├── preprocessing/
│   ├── audio_preprocessing.py    # Resample→Filter→HSMM→Log-Mel
│   └── image_preprocessing.py   # CLAHE→Aspect-pad→Z-score
├── models/
│   ├── crnn_heart_sound.py       # 2D-CRNN (ResNet18+BiLSTM+Attention)
│   ├── nts_net_ultrasound.py     # NTS-Net (ResNet50+Navigator+Scrutinizer)
│   ├── efficientnet_xray.py      # EfficientNetV2-S + Grad-CAM hooks
│   └── gmu_fusion.py             # GMU + MLP + ModalityDropout + MultimodalModel
├── data/
│   └── dataset.py                # PediatricCardiacDataset + collate_fn
├── training/
│   ├── train_specialist.py       # Phase-1: train CRNN / NTSNet / EfficientNet
│   ├── surgery.py                # Strip classifier heads → embedding-only
│   └── train_gmu.py              # Phase-2: train GMU + MLP (specialists frozen)
├── explainability/
│   └── gradcam.py                # Grad-CAM for all three modalities
├── inference/
│   ├── api.py                    # FastAPI REST API
│   ├── dashboard.py              # Streamlit doctor dashboard
│   └── infer.py                  # Standalone CLI inference
└── tests/
    └── test_shapes.py            # Shape smoke tests
```

---

## Step-by-Step Build Guide

### Step 1 — Environment Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

# Install dependencies (CUDA build — adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Step 2 — Dataset Preparation

| Dataset | Modality | Source |
|---------|----------|--------|
| **ZCHSound** | Heart Sounds (WAV) | [PhysioNet / GitHub](https://github.com/yaseen21khan/Classification-of-Heart-Sound-Signal-Using-Multiple-Features-) |
| **VinDr-PCXR** | Pediatric Chest X-Ray | [PhysioNet](https://physionet.org/content/vindr-pcxr/1.0.0/) |
| **CARDIUM** | Echocardiography | CARDIUM dataset / local hospital data |

Prepare a CSV for each split:

```csv
patient_id,audio_path,us_path,xray_path,label
PT001,data/audio/pt001.wav,data/us/pt001.jpg,data/xray/pt001.jpg,1
PT002,,data/us/pt002.jpg,data/xray/pt002.jpg,0
PT003,data/audio/pt003.wav,,data/xray/pt003.jpg,1
```

> NaN in a path column = modality unavailable (handled automatically).

### Step 3 — Phase-1: Train Specialists

```bash
# Train CRNN on heart sounds
python training/train_specialist.py \
    --modality audio \
    --train_csv data/train.csv \
    --val_csv   data/val.csv   \
    --output_dir checkpoints

# Train NTS-Net on ultrasound
python training/train_specialist.py \
    --modality ultrasound \
    --train_csv data/train.csv \
    --val_csv   data/val.csv   \
    --output_dir checkpoints

# Train EfficientNetV2-S on chest X-rays
python training/train_specialist.py \
    --modality xray \
    --train_csv data/train.csv \
    --val_csv   data/val.csv   \
    --output_dir checkpoints
```

> **Target**: Each specialist must reach ≥ 85% validation accuracy before proceeding.

### Step 4 — The Surgery

```bash
python training/surgery.py \
    --audio_ckpt   checkpoints/audio_best.pth       \
    --us_ckpt      checkpoints/ultrasound_best.pth  \
    --xray_ckpt    checkpoints/xray_best.pth        \
    --output_dir   checkpoints/surgery
```

### Step 5 — Phase-2: Train GMU

```bash
python training/train_gmu.py \
    --audio_ckpt   checkpoints/surgery/audio_surgery.pth      \
    --us_ckpt      checkpoints/surgery/ultrasound_surgery.pth \
    --xray_ckpt    checkpoints/surgery/xray_surgery.pth       \
    --train_csv    data/train.csv \
    --val_csv      data/val.csv   \
    --output_dir   checkpoints
```

### Step 6 — Run Inference

**Standalone CLI:**
```bash
python inference/infer.py \
    --audio path/to/heart_sound.wav \
    --us    path/to/echo.jpg        \
    --xray  path/to/cxr.jpg         \
    --output report_output/
```

**FastAPI Server:**
```bash
uvicorn inference.api:app --host 0.0.0.0 --port 8000
# POST to http://localhost:8000/predict
```

**Streamlit Dashboard (Doctor UI):**
```bash
streamlit run inference/dashboard.py
```

### Step 7 — Smoke Test

```bash
python tests/test_shapes.py
```

---

## Key Design Decisions

### Modality Dropout (Missing Modality Handling)
During Phase-2 training, each modality embedding is independently zeroed with p=0.2. This forces the GMU to learn robust fusion even when a modality is absent. At inference, simply pass `None` for any missing modality — the model returns a zero-embedding for that branch, and the Sigmoid Gates will naturally down-weight it.

### Sigmoid Gating
Each modality has its own `SigmoidGate(in_dim → hidden_dim)`:
```
z = σ(W_g · e)    ← gate vector ∈ (0,1)
v = W_v · e       ← value projection
output = z ⊙ v    ← element-wise reliability weighting
```
The gate values are logged to TensorBoard during training, providing interpretability into which modality the model is trusting.

### Grad-CAM
All three specialists have Grad-CAM hooks:
- **EfficientNetV2-S**: hooks on `features[-1]` → heatmap on X-ray
- **NTSNet**: hooks on `backbone[-1]` (ResNet-50 layer4) → heatmap on echo image
- **CRNN2D**: hooks on `cnn[7]` (ResNet-18 layer4) → saliency map on spectrogram

### BCEWithLogitsLoss + Class Imbalance
A `pos_weight=4.0` is applied (assuming ~20% CHD prevalence in a screening population). Adjust based on your dataset statistics.

---

## Monitoring with TensorBoard

```bash
tensorboard --logdir logs/
```

Tracks per-modality loss, accuracy, sensitivity, specificity, F1, and mean gate weights.

---

## ⚠️ Clinical Disclaimer

> This is an AI-assisted **screening** tool. It does NOT replace clinical judgement, formal echocardiographic evaluation, or specialist diagnosis. All AI outputs must be reviewed by qualified healthcare professionals.
