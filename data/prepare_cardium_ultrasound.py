"""
CARDIUM Ultrasound Dataset Preparation
========================================
Scans the cardium_images folder structure and builds CSV files
ready for training the NTS-Net specialist.

Expected structure:
  <root>/
    fold_1/
      train/
        CHD/
          <patient_id>/
            0_<patient_id>.png
            1_<patient_id>.png
            ...
        Non_CHD/
          <patient_id>/
            ...
      test/
        CHD/  Non_CHD/  ...
    fold_2/  ...
    fold_3/  ...

Output CSV columns:
  patient_id | us_path | label | fold | split | frame_idx

Strategy options (--mode):
  "per_image"   : one row per PNG frame  (more data, recommended for training)
  "per_patient" : one row per patient using the middle frame  (cleaner, less noisy)

Usage
-----
  python data/prepare_cardium_ultrasound.py \
      --root   "C:/path/to/cardium_images" \
      --fold   1 \
      --out    data/ultrasound \
      --mode   per_image

  # Use all 3 folds combined (recommended for small datasets):
  python data/prepare_cardium_ultrasound.py \
      --root   "C:/path/to/cardium_images" \
      --fold   all \
      --out    data/ultrasound \
      --mode   per_image
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import pandas as pd


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
LABEL_MAP = {"CHD": 1, "Non_CHD": 0}


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

def scan_fold(fold_root: Path, fold_num: int | str, mode: str) -> list[dict]:
    """
    Scan one fold directory and return a list of row dicts.
    """
    rows = []
    for split in ("train", "test"):
        split_dir = fold_root / split
        if not split_dir.is_dir():
            continue

        for class_name, label in LABEL_MAP.items():
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                continue

            # Each sub-folder = one patient
            for patient_dir in sorted(class_dir.iterdir()):
                if not patient_dir.is_dir():
                    continue
                patient_id = patient_dir.name

                # Collect all PNG frames, sorted by frame index
                frames = sorted(
                    [f for f in patient_dir.iterdir()
                     if f.suffix.lower() in IMAGE_EXTS],
                    key=lambda f: _frame_index(f.name),
                )

                if not frames:
                    continue

                if mode == "per_patient":
                    # Use the middle frame as the representative image
                    mid = frames[len(frames) // 2]
                    rows.append({
                        "patient_id": patient_id,
                        "us_path": str(mid.resolve()),
                        "label": label,
                        "fold": fold_num,
                        "split": split,
                        "frame_idx": _frame_index(mid.name),
                        "total_frames": len(frames),
                    })
                else:  # per_image
                    for frame in frames:
                        rows.append({
                            "patient_id": patient_id,
                            "us_path": str(frame.resolve()),
                            "label": label,
                            "fold": fold_num,
                            "split": split,
                            "frame_idx": _frame_index(frame.name),
                            "total_frames": len(frames),
                        })
    return rows


def _frame_index(filename: str) -> int:
    """Extract the leading integer from filenames like '3_aedxf00003rrfgkl0.png'."""
    try:
        return int(filename.split("_")[0])
    except (ValueError, IndexError):
        return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare_cardium(
    root: str,
    fold: str | int,
    out_dir: str,
    mode: str = "per_image",
    seed: int = 42,
):
    random.seed(seed)
    root_path = Path(root)
    os.makedirs(out_dir, exist_ok=True)

    # Determine which folds to scan
    if str(fold).lower() == "all":
        fold_dirs = sorted([d for d in root_path.iterdir() if d.is_dir() and d.name.startswith("fold_")])
        fold_nums = [d.name for d in fold_dirs]
    else:
        fold_dirs = [root_path / f"fold_{fold}"]
        fold_nums = [int(fold)]
        if not fold_dirs[0].is_dir():
            raise FileNotFoundError(f"Fold directory not found: {fold_dirs[0]}")

    print(f"Scanning folds: {[d.name for d in fold_dirs]}")
    print(f"Mode           : {mode}")

    all_rows = []
    for fd, fn in zip(fold_dirs, fold_nums):
        rows = scan_fold(fd, fn, mode)
        all_rows.extend(rows)
        chd = sum(1 for r in rows if r["label"] == 1)
        non = sum(1 for r in rows if r["label"] == 0)
        print(f"  {fd.name}: {len(rows)} rows  |  CHD={chd}  Non_CHD={non}")

    df = pd.DataFrame(all_rows)
    total = len(df)
    print(f"\nTotal rows : {total}")
    print(f"CHD        : {df['label'].sum()}  ({df['label'].mean()*100:.1f}%)")
    print(f"Non-CHD    : {(df['label']==0).sum()}")

    # ---------------------------------------------------------------------------
    # Build train / val / test CSVs
    # ---------------------------------------------------------------------------
    if str(fold).lower() == "all":
        # Use fold_1 test as val+test, everything else as train
        # This respects the original cross-validation intent
        train_mask = ~((df["fold"] == "fold_1") & (df["split"] == "test"))
        val_mask   =  (df["fold"] == "fold_1") & (df["split"] == "test")

        train_df = df[train_mask].copy()
        # Split the fold_1 test evenly into val and test
        val_rows = df[val_mask].copy()
        val_rows = val_rows.sample(frac=1, random_state=seed).reset_index(drop=True)
        mid = len(val_rows) // 2
        val_df  = val_rows.iloc[:mid].reset_index(drop=True)
        test_df = val_rows.iloc[mid:].reset_index(drop=True)
    else:
        # Single fold: use provided train/test split
        # Reserve 15% of train as val (stratified by patient to avoid data leakage)
        train_all = df[df["split"] == "train"].copy()
        test_df   = df[df["split"] == "test"].copy()

        # Split by patient (not by image) to prevent leakage
        patient_ids = train_all["patient_id"].unique().tolist()
        random.shuffle(patient_ids)
        n_val_patients = max(1, int(len(patient_ids) * 0.15))
        val_patients   = set(patient_ids[:n_val_patients])
        train_patients = set(patient_ids[n_val_patients:])

        val_df   = train_all[train_all["patient_id"].isin(val_patients)].copy()
        train_df = train_all[train_all["patient_id"].isin(train_patients)].copy()

    # Shuffle rows
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df   = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df  = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Save
    train_path = os.path.join(out_dir, "us_train.csv")
    val_path   = os.path.join(out_dir, "us_val.csv")
    test_path  = os.path.join(out_dir, "us_test.csv")

    # Keep only the columns the training pipeline needs
    cols = ["patient_id", "us_path", "label", "fold", "frame_idx"]
    train_df[cols].to_csv(train_path, index=False)
    val_df[cols].to_csv(val_path,   index=False)
    test_df[cols].to_csv(test_path, index=False)

    print(f"\n✅ CSV files saved to '{out_dir}':")
    print(f"   Train : {train_path}  ({len(train_df)} rows, "
          f"{train_df['patient_id'].nunique()} patients)")
    print(f"   Val   : {val_path}  ({len(val_df)} rows, "
          f"{val_df['patient_id'].nunique()} patients)")
    print(f"   Test  : {test_path}  ({len(test_df)} rows, "
          f"{test_df['patient_id'].nunique()} patients)")

    _print_class_balance(train_df, val_df, test_df)
    return train_path, val_path, test_path


def _print_class_balance(train_df, val_df, test_df):
    print("\n  Class balance:")
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        chd = df["label"].sum()
        tot = len(df)
        print(f"    {name}: CHD={chd} ({chd/tot*100:.1f}%)  "
              f"Non-CHD={tot-chd} ({(tot-chd)/tot*100:.1f}%)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare CARDIUM ultrasound dataset CSVs."
    )
    parser.add_argument(
        "--root", required=True,
        help='Path to the "cardium_images" folder (contains fold_1, fold_2, fold_3)'
    )
    parser.add_argument(
        "--fold", default="1",
        help='Which fold to use: 1, 2, 3, or "all" (default: 1)'
    )
    parser.add_argument(
        "--out", default="data/ultrasound",
        help="Output directory for CSV files (default: data/ultrasound)"
    )
    parser.add_argument(
        "--mode", choices=["per_image", "per_patient"], default="per_image",
        help=(
            "per_image : one CSV row per PNG frame (more data, recommended)\n"
            "per_patient : one CSV row per patient using the middle frame"
        )
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_cardium(
        root=args.root,
        fold=args.fold,
        out_dir=args.out,
        mode=args.mode,
        seed=args.seed,
    )
