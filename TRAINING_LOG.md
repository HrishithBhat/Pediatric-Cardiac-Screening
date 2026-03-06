# Training Log — Pediatric Cardiac Screening System

> **Project**: Autonomous Pediatric Cardiac Screening for CHD Detection  
> **Architecture**: Stacked Multimodal Ensemble (CRNN + NTS-Net + EfficientNetV2-S) fused by Gated Multimodal Unit (GMU)  
> **Date**: March 2026  

---

## System Setup

| Component | Details |
|---|---|
| OS | Windows 11 |
| Python | 3.12.5 (venv at `../MajorProject/.venv`) |
| PyTorch | 2.x + CUDA |
| GPU | NVIDIA (CUDA 12.5) |
| Project root | `MajorProject/pediatric_cardiac_screening/` |

---

## Datasets Used

### 1. Ultrasound — CARDIUM Dataset
| Property | Value |
|---|---|
| Source | `cardium_images(Ultrasound)/cardium_images/` |
| Structure | `fold_1,2,3 / train,test / CHD,Non_CHD / <patient_id> / <frame>.png` |
| Label encoding | Folder name (`CHD` → 1, `Non_CHD` → 0) |
| Frames per patient | 3–11 PNG frames |
| Total rows (per_image) | 19,674 across 3 folds |
| Class balance | CHD: 16.3% / Non-CHD: 83.7% |
| Preparation script | `data/prepare_cardium_ultrasound.py` |
| Mode used | `--fold all --mode per_image` |
| Split | Train: 17,528 rows (1,103 patients) / Val: 1,073 / Test: 1,073 |

### 2. X-Ray — Congenital Heart Disease Dataset
| Property | Value |
|---|---|
| Source | `congenital-heart-disease(xray)/` |
| Structure | `Normal/<CONTROL##.jpg>` and `CHD/ASD,PDA,VSD/<##.jpg>` |
| Label encoding | Folder name (`CHD` subtypes → 1, `Normal` → 0) |
| Files per patient | 1 image = 1 patient |
| Subtypes | ASD (194), PDA (216), VSD (210), Normal (208) |
| Total images | 828 |
| Class balance | CHD: 74.9% / Normal: 25.1% |
| Preparation script | `data/prepare_xray.py` |
| Split | Train: 582 / Val: 123 / Test: 123 (stratified by subtype) |

### 3. Audio — ZCHSound(CHD) Dataset
| Property | Value |
|---|---|
| Source | `ZCHSound(CHD)/ZCHSound/` |
| Structure | `clean Heartsound Data/*.wav` + `Noise Heartsound Data Details/*.wav` + 2 metadata CSVs |
| Label encoding | From CSV column `diagnosis` (NORMAL→0, ASD/PDA/VSD→1) |
| CSV format | Semicolon-delimited: `fileName;gender;age(days);diagnosis` |
| Total recordings | 1,259 WAV files; **105 skipped** (diagnosis=PFO, borderline condition) |
| Used recordings | 1,154 |
| Subtypes | ASD: 221, VSD: 201, PDA: 39, NORMAL: 693 |
| Class balance | CHD: 39.9% / Normal: 60.1% |
| Preparation script | `data/prepare_audio.py` |
| Split | Train: 812 / Val: 171 / Test: 171 (stratified by subtype) |

> **Note on PFO**: Patent Foramen Ovale (105 recordings, ZCH0685–ZCH0754 and ZCH1109–ZCH1143) was excluded. PFO is a borderline/minor defect, not a classic CHD — including it as CHD would introduce label noise.

---

## Phase 1 — Specialist Training

Each specialist was trained independently on its own single-modality dataset.

### Common Training Config
| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Loss | BCEWithLogitsLoss (pos_weight=4.0) |
| Scheduler | StepLR |
| AMP | Enabled (mixed precision) |
| Early stopping | `--patience 7` |
| TensorBoard logs | `logs/<modality>/` |

---

### Specialist 1: NTS-Net (Ultrasound)

| Parameter | Value |
|---|---|
| Model | `NTSNet` (37.7M parameters) |
| Batch size | 32 (default) |
| Input | `(B, 3, 224, 224)` PNG ultrasound frames |

**Training results:**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val Sen | Val Spe | Time |
|---|---|---|---|---|---|---|---|
| 1 | 0.7582 | 0.806 | 0.6841 | 0.826 | 0.678 | 0.855 | 721.6s |
| 2 | 0.5478 | 0.873 | 0.4921 | 0.856 | 0.893 | 0.849 | 289.7s |
| 3 | 0.3826 | 0.921 | 0.2818 | 0.920 | 0.898 | 0.924 | 252.6s |
| 4 | 0.2957 | 0.944 | 0.2995 | 0.959 | 0.864 | 0.978 | 322.0s |
| 5 | 0.2395 | 0.957 | 0.1644 | **0.976** | 0.938 | 0.983 | 302.9s |

**Stopped manually after epoch 5 (converged)**  
**Best checkpoint**: `checkpoints/ultrasound/ultrasound_best.pth`  
**Best Val Accuracy**: **97.6%**  ✅

---

### Specialist 2: EfficientNetV2-S (X-Ray)

| Parameter | Value |
|---|---|
| Model | `EfficientNetV2XRay` (20.7M parameters) |
| Batch size | 8 (reduced from 32 — CUDA OOM at 32) |
| Input | `(B, 3, 224, 224)` JPEG chest X-rays |

**Training results:**

| Epoch | Train Acc | Val Acc | Val Sen | Val Spe | Note |
|---|---|---|---|---|---|
| 1 | 0.729 | 0.740 | 0.978 | 0.032 | |
| 2 | 0.756 | 0.756 | 1.000 | 0.032 | |
| 3 | 0.802 | 0.797 | 0.978 | 0.258 | |
| 4 | 0.806 | 0.813 | 0.848 | 0.710 | |
| 7 | 0.869 | 0.837 | 0.913 | 0.613 | |
| 8 | 0.878 | 0.846 | 0.935 | 0.581 | |
| 10 | 0.899 | **0.894** | 0.978 | 0.645 | ✅ Best |
| 17 | 0.947 | 0.870 | 0.891 | 0.806 | ⏹ Early stop |

**Best checkpoint**: `checkpoints/xray/xray_best.pth`  
**Best Val Accuracy**: **89.4%**  ✅

> **Note**: Specificity was low early (model predicted CHD for almost everything due to class imbalance 75:25). Improved with training. Final model balances sensitivity and specificity well.

---

### Specialist 3: CRNN (Audio / Heart Sounds)

| Parameter | Value |
|---|---|
| Model | `CRNN2D` (14.7M parameters) |
| Batch size | 16 |
| Input | `(B, 1, 64, 256)` mel-spectrogram |

**Training results:**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val Sen | Val Spe |
|---|---|---|---|---|---|---|
| 1 | 1.0671 | 0.638 | 0.9570 | 0.620 | 0.971 | 0.388 |
| 2 | 0.8505 | 0.802 | 0.7923 | 0.795 | 0.897 | 0.728 |
| 3 | 0.6426 | 0.856 | 1.7205 | 0.795 | 0.647 | 0.893 |
| 4 | 0.4853 | 0.908 | 1.6794 | **0.801** | 0.721 | 0.854 | ✅ Best |
| 5–11 | ↑ train | — | ↑↑ val loss | ↓ val acc | — | — | Overfitting |
| 11 | 0.1118 | 0.988 | 3.3228 | 0.789 | — | — | ⏹ Early stop |

**Best checkpoint**: `checkpoints/audio/audio_best.pth`  
**Best Val Accuracy**: **80.1%**  ⚠️

> **Note**: Classic overfitting — training accuracy reached 98.8% while val loss exploded. Cause: small dataset (812 training samples). The GMU's ModalityDropout will compensate by down-weighting audio when unreliable. Audio modality still contributes useful signal for the fusion layer.

---

## Phase 1 Summary

| Specialist | Parameters | Best Val Acc | Status |
|---|---|---|---|
| NTS-Net (Ultrasound) | 37.7M | **97.6%** | ✅ Excellent |
| EfficientNetV2 (X-Ray) | 20.7M | **89.4%** | ✅ Good |
| CRNN (Audio) | 14.7M | **80.1%** | ⚠️ Overfits (small data) |

---

## Phase 2 — GMU Fusion Training

*(To be filled after run_phase2.ps1 completes)*

### Steps
1. **Multimodal CSV** built by `prepare_multimodal.py`:
   - Merges all 3 per-modality CSVs into one
   - Each row has ONE real modality path + `NaN` for the other two
   - `collate_fn` zero-pads missing modalities automatically

2. **Specialists frozen** — only GMU + MLP weights trained

3. **GMU architecture**:
   - `SigmoidGate` per modality → learned gating weights
   - Fused embedding: 1536-dim → MLP → binary logit
   - `ModalityDropout(p=0.2)` forces robustness to missing modalities

### Expected results
*(fill in after training)*

| Metric | Value |
|---|---|
| Best Val F1 | — |
| Best Val Accuracy | — |
| Best Val Sensitivity | — |
| Best Val Specificity | — |
| Epochs to converge | — |
| Checkpoint | `checkpoints/gmu/gmu_best.pth` |

---

## How to Reproduce

### Prerequisites
```powershell
# Activate venv
& "C:\Users\hrish\OneDrive\Documents\6th sem notes\MajorProject\.venv\Scripts\Activate.ps1"
cd "C:\Users\hrish\OneDrive\Documents\6th sem notes\MajorProject\pediatric_cardiac_screening"
```

### Phase 1 (datasets + specialist training)
```powershell
.\run_phase1.ps1
```

### Phase 2 (GMU fusion)
```powershell
.\run_phase2.ps1
```

### Manual specialist training (with custom args)
```powershell
$py = "C:\Users\hrish\OneDrive\Documents\6th sem notes\MajorProject\.venv\Scripts\python.exe"

& $py training\train_specialist.py `
    --modality   ultrasound `
    --train_csv  data\ultrasound\us_train.csv `
    --val_csv    data\ultrasound\us_val.csv `
    --output_dir checkpoints\ultrasound `
    --batch_size 32 `
    --patience   7

& $py training\train_specialist.py `
    --modality   xray `
    --train_csv  data\xray\xray_train.csv `
    --val_csv    data\xray\xray_val.csv `
    --output_dir checkpoints\xray `
    --batch_size 8 `
    --patience   7

& $py training\train_specialist.py `
    --modality   audio `
    --train_csv  data\audio\audio_train.csv `
    --val_csv    data\audio\audio_val.csv `
    --output_dir checkpoints\audio `
    --batch_size 16 `
    --patience   7
```

### Resume from checkpoint
```powershell
& $py training\train_specialist.py `
    --modality   ultrasound `
    --train_csv  data\ultrasound\us_train.csv `
    --val_csv    data\ultrasound\us_val.csv `
    --output_dir checkpoints\ultrasound `
    --resume     checkpoints\ultrasound\ultrasound_epoch005.pth
```

---

## Known Issues & Decisions

| Issue | Decision |
|---|---|
| CUDA OOM at batch_size=32 for EfficientNetV2 after NTS-Net run | Use `--batch_size 8` for xray, `--batch_size 16` for audio |
| CRNN overfitting on 812 samples | Acceptable — GMU gating will down-weight audio; future fix: data augmentation (SpecAugment) |
| PFO recordings (105 files) in ZCHSound | Skipped — PFO is borderline, not classic CHD |
| 3 datasets from different patients, no shared patients | GMU trained with single-modality rows + NaN padding — forces robust cross-modal gating |
| Ctrl+C kill shows multiprocessing traceback | Harmless on Windows — checkpoints are safe |
