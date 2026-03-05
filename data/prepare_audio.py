"""
ZCHSound Audio Dataset Preparation
====================================
Reads the two metadata CSVs, resolves WAV file paths, and builds
stratified train/val/test CSVs ready for CRNN specialist training.

Dataset structure
-----------------
  <root>/                                   (= the "ZCHSound" folder)
    clean Heartsound Data/
      ZCH0001.wav  …  ZCH0941.wav          (941 files, all NORMAL)
    Noise Heartsound Data Details/
      ZCH0942.wav  …  ZCH1259.wav          (318 files, CHD + some NORMAL)
    Clean Heartsound Data Details.csv       (semicolon-separated)
    Noise Heartsound Data Details.csv       (semicolon-separated)

CSV format (both files are identical in structure):
  fileName;gender;age(days);diagnosis
  ZCH0001.wav;M;2310;NORMAL
  ZCH1163.wav;M;2;ASD
  ...

Diagnosis values
  NORMAL  →  label = 0
  ASD     →  label = 1   (Atrial Septal Defect)
  PDA     →  label = 1   (Patent Ductus Arteriosus)
  VSD     →  label = 1   (Ventricular Septal Defect)

Output CSV columns
------------------
  patient_id | audio_path | label | subtype | noise_type | gender | age_days

  label      : 0 = Normal, 1 = CHD
  subtype    : NORMAL | ASD | PDA | VSD
  noise_type : clean | noisy   (which source folder the WAV came from)
  gender     : M | F
  age_days   : age in days (integer)

Split strategy
--------------
  Stratified by subtype  →  70% train / 15% val / 15% test
  No patient leakage (1 WAV file = 1 patient).

Usage
-----
  python data/prepare_audio.py \\
      --root  "C:/path/to/ZCHSound" \\
      --out   data/audio

  # With custom split ratios:
  python data/prepare_audio.py \\
      --root  "C:/path/to/ZCHSound" \\
      --out   data/audio \\
      --val_frac 0.15 --test_frac 0.15 --seed 42
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import pandas as pd


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}

# Map diagnosis strings → binary label
DIAGNOSIS_LABEL = {
    "NORMAL": 0,
    "ASD":    1,
    "PDA":    1,
    "VSD":    1,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_csv(root: Path, keyword: str) -> Path | None:
    """Case-insensitive search for a CSV containing `keyword` in its name."""
    for f in root.iterdir():
        if f.suffix.lower() == ".csv" and keyword.lower() in f.name.lower():
            return f
    return None


def _find_audio_dir(root: Path, keyword: str) -> Path | None:
    """Case-insensitive search for a subdirectory containing `keyword`."""
    for d in root.iterdir():
        if d.is_dir() and keyword.lower() in d.name.lower():
            return d
    return None


def _load_metadata_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load one of the ZCHSound metadata CSVs.
    Handles semicolon delimiter and strips whitespace from headers.
    """
    df = pd.read_csv(csv_path, sep=";", engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

def scan_dataset(root: str) -> pd.DataFrame:
    """
    Read both metadata CSVs and return a unified DataFrame with
    resolved absolute audio_path values.
    """
    root_path = Path(root)

    # Locate subdirectories
    clean_dir = _find_audio_dir(root_path, "clean")
    noise_dir = _find_audio_dir(root_path, "noise")

    if clean_dir is None:
        raise FileNotFoundError(
            f"Could not find 'clean Heartsound Data' folder under {root_path}"
        )
    if noise_dir is None:
        raise FileNotFoundError(
            f"Could not find 'Noise Heartsound Data Details' folder under {root_path}"
        )

    print(f"  Clean audio dir : {clean_dir.name}")
    print(f"  Noise audio dir : {noise_dir.name}")

    # Locate CSVs
    clean_csv_path = _find_csv(root_path, "clean")
    noise_csv_path = _find_csv(root_path, "noise")

    if clean_csv_path is None:
        raise FileNotFoundError(
            f"Could not find 'Clean Heartsound Data Details.csv' under {root_path}"
        )
    if noise_csv_path is None:
        raise FileNotFoundError(
            f"Could not find 'Noise Heartsound Data Details.csv' under {root_path}"
        )

    print(f"  Clean CSV       : {clean_csv_path.name}")
    print(f"  Noise CSV       : {noise_csv_path.name}")

    # Build a fast filename → absolute path index from both audio dirs
    file_index: dict[str, tuple[Path, str]] = {}  # fname → (abs_path, noise_type)
    for wav_file in clean_dir.iterdir():
        if wav_file.suffix.lower() in AUDIO_EXTS:
            file_index[wav_file.name.strip()] = (wav_file.resolve(), "clean")
    for wav_file in noise_dir.iterdir():
        if wav_file.suffix.lower() in AUDIO_EXTS:
            file_index[wav_file.name.strip()] = (wav_file.resolve(), "noisy")

    print(f"  WAV files found : {len(file_index)}")

    # Parse both CSVs
    rows = []
    missing = 0

    for csv_p, noise_type_hint in [(clean_csv_path, "clean"), (noise_csv_path, "noisy")]:
        df_meta = _load_metadata_csv(csv_p)

        # Normalise column names (handle slight variations)
        col_map = {}
        for col in df_meta.columns:
            cl = col.lower().replace(" ", "").replace("(", "").replace(")", "")
            if "filename" in cl:
                col_map[col] = "fileName"
            elif "gender" in cl:
                col_map[col] = "gender"
            elif "age" in cl:
                col_map[col] = "age_days"
            elif "diagnosis" in cl:
                col_map[col] = "diagnosis"
        df_meta = df_meta.rename(columns=col_map)

        for _, row in df_meta.iterrows():
            fname = str(row.get("fileName", "")).strip()
            diagnosis = str(row.get("diagnosis", "")).strip().upper()

            if fname not in file_index:
                missing += 1
                continue

            abs_path, noise_type = file_index[fname]
            label = DIAGNOSIS_LABEL.get(diagnosis, -1)
            if label == -1:
                # Unknown diagnosis — skip with a warning
                print(f"  WARNING: unknown diagnosis '{diagnosis}' for {fname}, skipping.")
                continue

            try:
                age_days = int(float(str(row.get("age_days", 0)).strip()))
            except (ValueError, TypeError):
                age_days = -1

            rows.append({
                "patient_id": Path(fname).stem,     # e.g. ZCH0001
                "audio_path": str(abs_path),
                "label":      label,
                "subtype":    diagnosis,             # NORMAL | ASD | PDA | VSD
                "noise_type": noise_type,            # clean | noisy
                "gender":     str(row.get("gender", "")).strip(),
                "age_days":   age_days,
            })

    if missing > 0:
        print(f"  WARNING: {missing} CSV entries had no matching WAV file (skipped).")

    df = pd.DataFrame(rows).drop_duplicates(subset=["patient_id"]).reset_index(drop=True)
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
    """Split per subtype to keep class proportions consistent."""
    random.seed(seed)
    train_idx, val_idx, test_idx = [], [], []

    for subtype, group in df.groupby("subtype"):
        indices = group.index.tolist()
        random.shuffle(indices)
        n = len(indices)
        n_test  = max(1, int(n * test_frac))
        n_val   = max(1, int(n * val_frac))
        n_train = n - n_val - n_test

        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train: n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])

    train_df = df.loc[train_idx].copy().reset_index(drop=True)
    val_df   = df.loc[val_idx].copy().reset_index(drop=True)
    test_df  = df.loc[test_idx].copy().reset_index(drop=True)
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare_audio(
    root: str,
    out_dir: str = "data/audio",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
):
    print(f"\nScanning: {root}")
    df = scan_dataset(root)

    total = len(df)
    chd   = df["label"].sum()
    print(f"\nTotal    : {total} recordings")
    print(f"CHD      : {int(chd)}  ({chd/total*100:.1f}%)")
    print(f"Normal   : {total - int(chd)}  ({(total - chd)/total*100:.1f}%)")
    print(f"\nSubtype breakdown:")
    for st, grp in df.groupby("subtype"):
        print(f"  {st:<10}: {len(grp):>4} recordings")
    print(f"\nNoise type:")
    for nt, grp in df.groupby("noise_type"):
        print(f"  {nt:<8}: {len(grp):>4} recordings")

    train_df, val_df, test_df = stratified_split(
        df, val_frac=val_frac, test_frac=test_frac, seed=seed
    )

    # Shuffle
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df   = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df  = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)

    cols = ["patient_id", "audio_path", "label", "subtype", "noise_type", "gender", "age_days"]
    train_path = os.path.join(out_dir, "audio_train.csv")
    val_path   = os.path.join(out_dir, "audio_val.csv")
    test_path  = os.path.join(out_dir, "audio_test.csv")

    train_df[cols].to_csv(train_path, index=False)
    val_df[cols].to_csv(val_path,     index=False)
    test_df[cols].to_csv(test_path,   index=False)

    print(f"\n✅ CSV files saved to '{out_dir}':")
    print(f"   Train : {train_path}  ({len(train_df)} recordings)")
    print(f"   Val   : {val_path}  ({len(val_df)} recordings)")
    print(f"   Test  : {test_path}  ({len(test_df)} recordings)")

    _print_class_balance(train_df, val_df, test_df)


def _print_class_balance(train_df, val_df, test_df):
    print("\n  Class balance:")
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        chd = df["label"].sum()
        tot = len(df)
        print(f"    {name}: CHD={int(chd)} ({chd/tot*100:.1f}%)  "
              f"Normal={tot - int(chd)} ({(tot - int(chd))/tot*100:.1f}%)")

    print("\n  Subtype counts per split:")
    all_dfs = {"Train": train_df, "Val": val_df, "Test": test_df}
    subtypes = sorted(set(
        train_df["subtype"].tolist() +
        val_df["subtype"].tolist() +
        test_df["subtype"].tolist()
    ))
    for st in subtypes:
        counts = {name: (df["subtype"] == st).sum() for name, df in all_dfs.items()}
        print(f"    {st:<10}: Train={counts['Train']:>3}  "
              f"Val={counts['Val']:>3}  Test={counts['Test']:>3}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare ZCHSound heart-sound dataset CSVs."
    )
    parser.add_argument(
        "--root", required=True,
        help=(
            'Path to the "ZCHSound" folder that contains '
            '"clean Heartsound Data/", "Noise Heartsound Data Details/", '
            'and the two .csv metadata files.'
        )
    )
    parser.add_argument(
        "--out", default="data/audio",
        help="Output directory for CSV files (default: data/audio)"
    )
    parser.add_argument(
        "--val_frac", type=float, default=0.15,
        help="Fraction for validation set (default: 0.15)"
    )
    parser.add_argument(
        "--test_frac", type=float, default=0.15,
        help="Fraction for test set (default: 0.15)"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_audio(
        root=args.root,
        out_dir=args.out,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
