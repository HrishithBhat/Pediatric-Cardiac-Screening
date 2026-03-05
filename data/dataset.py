"""
Dataset classes for the Pediatric Cardiac Screening System.

  - UltrasoundDataset   : single-modality for NTS-Net specialist training
                          CSV: patient_id | us_path | label  [+ fold, frame_idx]
  - XRayDataset         : single-modality for EfficientNetV2 specialist training
                          CSV: patient_id | xray_path | label
  - AudioDataset        : single-modality for CRNN specialist training
                          CSV: patient_id | audio_path | label
  - PediatricCardiacDataset : full multimodal dataset (all three modalities)
                          CSV: patient_id | audio_path | us_path | xray_path | label
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from preprocessing.audio_preprocessing import AudioPreprocessor
from preprocessing.image_preprocessing import (
    get_ultrasound_preprocessor,
    get_xray_preprocessor,
)


# ---------------------------------------------------------------------------
# Ultrasound-only Dataset  (for NTS-Net specialist)
# ---------------------------------------------------------------------------

class UltrasoundDataset(Dataset):
    """
    Single-modality dataset for the NTS-Net specialist.

    CSV must have columns:  us_path | label
    Optional columns that are ignored:  patient_id, fold, frame_idx, total_frames
    """

    def __init__(self, csv_path: str, augment: bool = False):
        self.df = pd.read_csv(csv_path)
        if "us_path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("CSV must have 'us_path' and 'label' columns.")
        # Drop rows where us_path is missing
        self.df = self.df.dropna(subset=["us_path"]).reset_index(drop=True)
        self.preprocessor = get_ultrasound_preprocessor(augment=augment)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        try:
            image = self.preprocessor(str(row["us_path"]))
        except Exception:
            image = torch.zeros(3, 224, 224)
        return {"image": image, "label": label}


# ---------------------------------------------------------------------------
# X-Ray-only Dataset  (for EfficientNetV2 specialist)
# ---------------------------------------------------------------------------

class XRayDataset(Dataset):
    """
    Single-modality dataset for the EfficientNetV2-S specialist.

    CSV must have columns:  xray_path | label
    """

    def __init__(self, csv_path: str, augment: bool = False):
        self.df = pd.read_csv(csv_path)
        if "xray_path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("CSV must have 'xray_path' and 'label' columns.")
        self.df = self.df.dropna(subset=["xray_path"]).reset_index(drop=True)
        self.preprocessor = get_xray_preprocessor(augment=augment)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        try:
            image = self.preprocessor(str(row["xray_path"]))
        except Exception:
            image = torch.zeros(3, 224, 224)
        return {"image": image, "label": label}


# ---------------------------------------------------------------------------
# Audio-only Dataset  (for CRNN specialist)
# ---------------------------------------------------------------------------

class AudioDataset(Dataset):
    """
    Single-modality dataset for the CRNN specialist.

    CSV must have columns:  audio_path | label
    """

    def __init__(self, csv_path: str, augment: bool = False,
                 audio_config: Optional[dict] = None):
        self.df = pd.read_csv(csv_path)
        if "audio_path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("CSV must have 'audio_path' and 'label' columns.")
        self.df = self.df.dropna(subset=["audio_path"]).reset_index(drop=True)
        self.preprocessor = AudioPreprocessor(**(audio_config or {}))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        try:
            spec = self.preprocessor(str(row["audio_path"]))
        except Exception:
            spec = torch.zeros(1, 64, 256)
        return {"image": spec, "label": label}


# ---------------------------------------------------------------------------
# Full Multimodal Dataset
# ---------------------------------------------------------------------------

class PediatricCardiacDataset(Dataset):
    """
    Multimodal dataset.  CSV columns expected:
      patient_id | audio_path | us_path | xray_path | label
                   (str/NaN)    (str/NaN) (str/NaN)   (0/1)

    A 'NaN' path means the modality is unavailable → returns zero tensor.
    """

    def __init__(
        self,
        csv_path: str,
        augment: bool = False,
        audio_config: Optional[dict] = None,
    ):
        self.df = pd.read_csv(csv_path)
        self._validate_columns()

        audio_cfg = audio_config or {}
        self.audio_pre = AudioPreprocessor(**audio_cfg)
        self.us_pre = get_ultrasound_preprocessor(augment=augment)
        self.xray_pre = get_xray_preprocessor(augment=augment)

    def _validate_columns(self):
        # At least one modality path column must be present
        modality_cols = {"audio_path", "us_path", "xray_path"}
        present = modality_cols & set(self.df.columns)
        if not present:
            raise ValueError(
                f"CSV must have at least one of: {modality_cols}. "
                f"Got columns: {list(self.df.columns)}"
            )
        if "label" not in self.df.columns:
            raise ValueError("CSV must have a 'label' column.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Optional[torch.Tensor]]:
        row = self.df.iloc[idx]
        label = torch.tensor(float(row["label"]), dtype=torch.float32)

        # Audio spectrogram
        audio_spec = self._load_audio(row.get("audio_path"))
        # Ultrasound image
        us_image = self._load_image(row.get("us_path"), self.us_pre)
        # X-ray image
        xray_image = self._load_image(row.get("xray_path"), self.xray_pre)

        return {
            "audio_spec": audio_spec,    # (1, 64, 256) or None
            "us_image": us_image,        # (3, 224, 224) or None
            "xray_image": xray_image,    # (3, 224, 224) or None
            "label": label,
        }

    def _load_audio(self, path) -> Optional[torch.Tensor]:
        if path is None or (isinstance(path, float) and pd.isna(path)):
            return None
        path = str(path)
        if not os.path.exists(path):
            return None
        try:
            return self.audio_pre(path)
        except Exception:
            return None

    def _load_image(self, path, preprocessor) -> Optional[torch.Tensor]:
        if path is None or (isinstance(path, float) and pd.isna(path)):
            return None
        path = str(path)
        if not os.path.exists(path):
            return None
        try:
            return preprocessor(path)
        except Exception:
            return None


def collate_fn(batch):
    """
    Custom collate: handles None entries (missing modalities) by replacing them
    with zero tensors of the correct shape.
    """
    AUDIO_SHAPE = (1, 64, 256)
    IMAGE_SHAPE = (3, 224, 224)

    def pad_none(items, shape):
        out = []
        for item in items:
            if item is None:
                out.append(torch.zeros(shape))
            else:
                out.append(item)
        return torch.stack(out, dim=0)

    audio_specs = pad_none([b["audio_spec"] for b in batch], AUDIO_SHAPE)
    us_images = pad_none([b["us_image"] for b in batch], IMAGE_SHAPE)
    xray_images = pad_none([b["xray_image"] for b in batch], IMAGE_SHAPE)
    labels = torch.stack([b["label"] for b in batch], dim=0)

    return {
        "audio_spec": audio_specs,
        "us_image": us_images,
        "xray_image": xray_images,
        "label": labels,
    }
