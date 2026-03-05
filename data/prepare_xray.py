"""
X-Ray Dataset Preparation  (congenital-heart-disease dataset)
===============================================================
Scans the folder structure and builds stratified train/val/test CSVs
ready for training the EfficientNetV2-S specialist.

Expected structure
------------------
  <root>/
    Normal/
      CONTROL01.jpg
      CONTROL02.jpg
      ...               ← 208 images  →  label = 0
    CHD/
      ASD/
        ASD01.jpg ...   ← 194 images  →  label = 1, subtype = ASD
      PDA/
        PDA01.jpg ...   ← 216 images  →  label = 1, subtype = PDA
      VSD/
        VSD01.jpg ...   ←  ?  images  →  label = 1, subtype = VSD

Output CSV columns
------------------
  patient_id | xray_path | label | subtype

  label   : 0 = Normal, 1 = CHD
  subtype : Normal | ASD | PDA | VSD   (useful for per-class metrics)

Split strategy
--------------
  Stratified by subtype  →  70% train / 15% val / 15% test
  No patient leakage (1 image = 1 patient).

Usage
-----
  python data/prepare_xray.py \\
      --root  "C:/path/to/congenital-heart-disease(xray)" \\
      --out   data/xray

  # Custom split ratios:
  python data/prepare_xray.py \\
      --root  "C:/path/to/congenital-heart-disease(xray)" \\
      --out   data/xray \\
      --val_frac 0.15 --test_frac 0.15 --seed 42
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import pandas as pd


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

def scan_dataset(root: str) -> pd.DataFrame:
    """
    Walk the dataset root and return a DataFrame with one row per image.
    """
    root_path = Path(root)
    rows = []

    # ── Normal (label = 0) ──────────────────────────────────────────────────
    normal_dir = root_path / "Normal"
    if not normal_dir.is_dir():
        # Try case-insensitive fallback
        for d in root_path.iterdir():
            if d.is_dir() and d.name.lower() == "normal":
                normal_dir = d
                break

    if normal_dir.is_dir():
        for img_file in sorted(normal_dir.iterdir()):
            if img_file.suffix.lower() in IMAGE_EXTS:
                rows.append({
                    "patient_id": img_file.stem,
                    "xray_path": str(img_file.resolve()),
                    "label": 0,
                    "subtype": "Normal",
                })
        print(f"  Normal : {sum(1 for r in rows if r['subtype'] == 'Normal')} images")
    else:
        print("  WARNING: 'Normal' folder not found.")

    # ── CHD subtypes (label = 1) ─────────────────────────────────────────────
    chd_dir = root_path / "CHD"
    if not chd_dir.is_dir():
        for d in root_path.iterdir():
            if d.is_dir() and d.name.upper() == "CHD":
                chd_dir = d
                break

    if chd_dir.is_dir():
        for subtype_dir in sorted(chd_dir.iterdir()):
            if not subtype_dir.is_dir():
                continue
            subtype = subtype_dir.name  # "ASD", "PDA", "VSD"
            before = len(rows)
            for img_file in sorted(subtype_dir.iterdir()):
                if img_file.suffix.lower() in IMAGE_EXTS:
                    rows.append({
                        "patient_id": img_file.stem,
                        "xray_path": str(img_file.resolve()),
                        "label": 1,
                        "subtype": subtype,
                    })
            count = len(rows) - before
            print(f"  CHD/{subtype:<8}: {count} images")
    else:
        print("  WARNING: 'CHD' folder not found.")

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def stratified_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split per subtype to keep class proportions consistent across splits.
    """
    random.seed(seed)
    train_rows, val_rows, test_rows = [], [], []

    for subtype, group in df.groupby("subtype"):
        indices = group.index.tolist()
        random.shuffle(indices)
        n = len(indices)
        n_test = max(1, int(n * test_frac))
        n_val  = max(1, int(n * val_frac))
        n_train = n - n_val - n_test

        train_rows.extend(indices[:n_train])
        val_rows.extend(indices[n_train: n_train + n_val])
        test_rows.extend(indices[n_train + n_val:])

    train_df = df.loc[train_rows].copy().reset_index(drop=True)
    val_df   = df.loc[val_rows].copy().reset_index(drop=True)
    test_df  = df.loc[test_rows].copy().reset_index(drop=True)

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare_xray(
    root: str,
    out_dir: str = "data/xray",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
):
    print(f"\nScanning: {root}")
    df = scan_dataset(root)

    total = len(df)
    chd   = df["label"].sum()
    print(f"\nTotal  : {total} images")
    print(f"CHD    : {chd}  ({chd/total*100:.1f}%)")
    print(f"Normal : {total - chd}  ({(total - chd)/total*100:.1f}%)")
    print(f"\nSubtype breakdown:")
    for st, grp in df.groupby("subtype"):
        print(f"  {st:<10}: {len(grp):>4} images")

    train_df, val_df, test_df = stratified_split(
        df, val_frac=val_frac, test_frac=test_frac, seed=seed
    )

    # Shuffle
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df   = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df  = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)

    cols = ["patient_id", "xray_path", "label", "subtype"]
    train_path = os.path.join(out_dir, "xray_train.csv")
    val_path   = os.path.join(out_dir, "xray_val.csv")
    test_path  = os.path.join(out_dir, "xray_test.csv")

    train_df[cols].to_csv(train_path, index=False)
    val_df[cols].to_csv(val_path,     index=False)
    test_df[cols].to_csv(test_path,   index=False)

    print(f"\n✅ CSV files saved to '{out_dir}':")
    print(f"   Train : {train_path}  ({len(train_df)} images)")
    print(f"   Val   : {val_path}  ({len(val_df)} images)")
    print(f"   Test  : {test_path}  ({len(test_df)} images)")

    _print_class_balance(train_df, val_df, test_df)


def _print_class_balance(train_df, val_df, test_df):
    print("\n  Class balance:")
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        chd = df["label"].sum()
        tot = len(df)
        print(f"    {name}: CHD={chd} ({chd/tot*100:.1f}%)  "
              f"Normal={tot-chd} ({(tot-chd)/tot*100:.1f}%)")
    print("\n  Subtype counts per split:")
    all_dfs = {"Train": train_df, "Val": val_df, "Test": test_df}
    subtypes = sorted(train_df["subtype"].unique().tolist() +
                      val_df["subtype"].unique().tolist() +
                      test_df["subtype"].unique().tolist())
    subtypes = sorted(set(subtypes))
    for st in subtypes:
        counts = {name: (df["subtype"] == st).sum() for name, df in all_dfs.items()}
        print(f"    {st:<10}: Train={counts['Train']:>3}  Val={counts['Val']:>3}  Test={counts['Test']:>3}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare congenital-heart-disease X-ray dataset CSVs."
    )
    parser.add_argument(
        "--root", required=True,
        help='Path to the "congenital-heart-disease(xray)" folder (contains CHD/ and Normal/)'
    )
    parser.add_argument(
        "--out", default="data/xray",
        help="Output directory for CSV files (default: data/xray)"
    )
    parser.add_argument(
        "--val_frac", type=float, default=0.15,
        help="Fraction of data for validation set (default: 0.15)"
    )
    parser.add_argument(
        "--test_frac", type=float, default=0.15,
        help="Fraction of data for test set (default: 0.15)"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_xray(
        root=args.root,
        out_dir=args.out,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
