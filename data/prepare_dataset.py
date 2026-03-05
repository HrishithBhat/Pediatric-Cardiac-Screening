"""
Dataset Preparation Utility
============================
Scans your raw data folders and builds train.csv / val.csv / test.csv
automatically.

Folder layout expected (flexible — missing modalities are fine):
  data/raw/
    audio/          *.wav  (heart sounds)
    ultrasound/     *.jpg or *.png
    xray/           *.jpg or *.png
    labels.csv      patient_id, label  (1=CHD, 0=Normal)

  OR pass individual folders via CLI arguments.

Usage
-----
  python data/prepare_dataset.py \
      --audio_dir   data/raw/audio       \
      --us_dir      data/raw/ultrasound  \
      --xray_dir    data/raw/xray        \
      --labels_csv  data/raw/labels.csv  \
      --out_dir     data                 \
      --val_split   0.15                 \
      --test_split  0.10                 \
      --seed        42

Output: data/train.csv, data/val.csv, data/test.csv
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".aiff"}


def _scan_dir(folder: str | None, valid_exts: set) -> dict[str, str]:
    """
    Returns  { stem_id : absolute_path }  for all matching files in *folder*.
    stem_id = filename without extension (used for patient-ID matching).
    """
    if folder is None or not os.path.isdir(folder):
        return {}
    result = {}
    for f in Path(folder).rglob("*"):
        if f.suffix.lower() in valid_exts:
            result[f.stem] = str(f.resolve())
    return result


def prepare_csv(
    audio_dir: str | None,
    us_dir: str | None,
    xray_dir: str | None,
    labels_csv: str,
    out_dir: str,
    val_split: float = 0.15,
    test_split: float = 0.10,
    seed: int = 42,
):
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # Load labels
    labels_df = pd.read_csv(labels_csv)
    if "patient_id" not in labels_df.columns or "label" not in labels_df.columns:
        raise ValueError("labels.csv must have 'patient_id' and 'label' columns.")
    labels_df["patient_id"] = labels_df["patient_id"].astype(str).str.strip()
    labels_df["label"] = labels_df["label"].astype(int)

    # Scan modality folders
    audio_files = _scan_dir(audio_dir, AUDIO_EXTS)
    us_files = _scan_dir(us_dir, IMAGE_EXTS)
    xray_files = _scan_dir(xray_dir, IMAGE_EXTS)

    print(f"Found: {len(audio_files)} audio | {len(us_files)} ultrasound | {len(xray_files)} x-ray files")
    print(f"Label entries: {len(labels_df)}")

    # Build per-patient rows
    rows = []
    for _, row in labels_df.iterrows():
        pid = str(row["patient_id"])
        rows.append({
            "patient_id": pid,
            "audio_path": audio_files.get(pid, None),
            "us_path": us_files.get(pid, None),
            "xray_path": xray_files.get(pid, None),
            "label": int(row["label"]),
        })

    df = pd.DataFrame(rows)
    total = len(df)
    has_all = df[["audio_path", "us_path", "xray_path"]].notna().all(axis=1).sum()
    has_any = df[["audio_path", "us_path", "xray_path"]].notna().any(axis=1).sum()
    print(f"Patients: {total} total | {has_all} with all 3 modalities | {has_any} with ≥1 modality")
    print(f"Class balance: {df['label'].sum()} CHD / {(df['label']==0).sum()} Normal")

    # Remove patients with NO modality at all
    valid = df[df[["audio_path", "us_path", "xray_path"]].notna().any(axis=1)].copy()
    if len(valid) < len(df):
        print(f"Dropped {len(df)-len(valid)} patients with no modality data.")

    # Shuffle
    indices = list(range(len(valid)))
    random.shuffle(indices)

    n_test = int(len(valid) * test_split)
    n_val = int(len(valid) * val_split)
    n_train = len(valid) - n_test - n_val

    train_df = valid.iloc[indices[:n_train]].reset_index(drop=True)
    val_df = valid.iloc[indices[n_train:n_train + n_val]].reset_index(drop=True)
    test_df = valid.iloc[indices[n_train + n_val:]].reset_index(drop=True)

    train_path = os.path.join(out_dir, "train.csv")
    val_path = os.path.join(out_dir, "val.csv")
    test_path = os.path.join(out_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n✅ CSVs saved:")
    print(f"   Train : {train_path}  ({len(train_df)} samples)")
    print(f"   Val   : {val_path}  ({len(val_df)} samples)")
    print(f"   Test  : {test_path}  ({len(test_df)} samples)")
    return train_path, val_path, test_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build train/val/test CSV files.")
    parser.add_argument("--audio_dir",  default=None, help="Folder with .wav files")
    parser.add_argument("--us_dir",     default=None, help="Folder with ultrasound images")
    parser.add_argument("--xray_dir",   default=None, help="Folder with chest X-ray images")
    parser.add_argument("--labels_csv", required=True,
                        help="CSV with columns: patient_id, label")
    parser.add_argument("--out_dir",    default="data",
                        help="Output directory for train/val/test CSVs")
    parser.add_argument("--val_split",  type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.10)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    prepare_csv(
        audio_dir=args.audio_dir,
        us_dir=args.us_dir,
        xray_dir=args.xray_dir,
        labels_csv=args.labels_csv,
        out_dir=args.out_dir,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )
