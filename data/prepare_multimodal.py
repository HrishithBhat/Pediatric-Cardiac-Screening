"""
Multimodal CSV Builder for Phase-2 GMU Training
=================================================
Merges the three per-modality CSVs into a single multimodal CSV
where each row has ONE real modality path and NaN for the other two.

The GMU's ModalityDropout + collate_fn zero-pad missing modalities,
so the model learns robust gating even when modalities are absent —
exactly the right strategy when datasets are from different patients.

Output CSV columns:
  patient_id | audio_path | us_path | xray_path | label

Usage
-----
  python data/prepare_multimodal.py \\
      --audio_train  data/audio/audio_train.csv \\
      --audio_val    data/audio/audio_val.csv   \\
      --audio_test   data/audio/audio_test.csv  \\
      --us_train     data/ultrasound/us_train.csv \\
      --us_val       data/ultrasound/us_val.csv   \\
      --us_test      data/ultrasound/us_test.csv  \\
      --xray_train   data/xray/xray_train.csv \\
      --xray_val     data/xray/xray_val.csv   \\
      --xray_test    data/xray/xray_test.csv  \\
      --out          data/multimodal
"""

from __future__ import annotations

import argparse
import os

import pandas as pd


def _load(path: str, path_col: str, keep_cols: list[str]) -> pd.DataFrame:
    """Load a single-modality CSV, keep only needed columns, rename path col."""
    df = pd.read_csv(path)
    # Ensure patient_id exists (use index if absent)
    if "patient_id" not in df.columns:
        df["patient_id"] = df.index.astype(str)
    out = df[["patient_id", "label", path_col]].copy()
    out = out.rename(columns={path_col: path_col})  # keep original name
    # Add NaN columns for the other two modalities
    for col in keep_cols:
        if col not in out.columns:
            out[col] = float("nan")
    return out[["patient_id", "audio_path", "us_path", "xray_path", "label"]]


def merge_split(
    audio_csv: str,
    us_csv: str,
    xray_csv: str,
    seed: int = 42,
) -> pd.DataFrame:
    dfs = []
    if audio_csv and os.path.exists(audio_csv):
        dfs.append(_load(audio_csv, "audio_path",
                         ["audio_path", "us_path", "xray_path"]))
    if us_csv and os.path.exists(us_csv):
        dfs.append(_load(us_csv, "us_path",
                         ["audio_path", "us_path", "xray_path"]))
    if xray_csv and os.path.exists(xray_csv):
        dfs.append(_load(xray_csv, "xray_path",
                         ["audio_path", "us_path", "xray_path"]))

    if not dfs:
        raise ValueError("No valid CSV paths provided.")

    df = pd.concat(dfs, ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def prepare_multimodal(
    audio_train: str, audio_val: str, audio_test: str,
    us_train: str,    us_val: str,    us_test: str,
    xray_train: str,  xray_val: str,  xray_test: str,
    out_dir: str = "data/multimodal",
    seed: int = 42,
):
    os.makedirs(out_dir, exist_ok=True)

    for split, a, u, x in [
        ("train", audio_train, us_train, xray_train),
        ("val",   audio_val,   us_val,   xray_val),
        ("test",  audio_test,  us_test,  xray_test),
    ]:
        df = merge_split(a, u, x, seed=seed)
        out_path = os.path.join(out_dir, f"multimodal_{split}.csv")
        df.to_csv(out_path, index=False)

        total = len(df)
        chd   = df["label"].sum()
        has_audio = df["audio_path"].notna().sum()
        has_us    = df["us_path"].notna().sum()
        has_xray  = df["xray_path"].notna().sum()

        print(f"\n{split.upper()}  →  {out_path}")
        print(f"  Rows  : {total}   CHD={int(chd)} ({chd/total*100:.1f}%)  "
              f"Normal={total-int(chd)} ({(total-chd)/total*100:.1f}%)")
        print(f"  Audio : {has_audio} rows with real path  "
              f"({has_audio/total*100:.0f}%)")
        print(f"  US    : {has_us} rows with real path  "
              f"({has_us/total*100:.0f}%)")
        print(f"  X-Ray : {has_xray} rows with real path  "
              f"({has_xray/total*100:.0f}%)")

    print("\n✅ Multimodal CSVs ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge per-modality CSVs into a multimodal GMU training set."
    )
    parser.add_argument("--audio_train", required=True)
    parser.add_argument("--audio_val",   required=True)
    parser.add_argument("--audio_test",  required=True)
    parser.add_argument("--us_train",    required=True)
    parser.add_argument("--us_val",      required=True)
    parser.add_argument("--us_test",     required=True)
    parser.add_argument("--xray_train",  required=True)
    parser.add_argument("--xray_val",    required=True)
    parser.add_argument("--xray_test",   required=True)
    parser.add_argument("--out", default="data/multimodal",
                        help="Output directory (default: data/multimodal)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_multimodal(
        audio_train=args.audio_train, audio_val=args.audio_val,
        audio_test=args.audio_test,
        us_train=args.us_train,       us_val=args.us_val,
        us_test=args.us_test,
        xray_train=args.xray_train,   xray_val=args.xray_val,
        xray_test=args.xray_test,
        out_dir=args.out,
        seed=args.seed,
    )
