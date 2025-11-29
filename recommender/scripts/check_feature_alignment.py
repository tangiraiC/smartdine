"""
Quick alignment & integrity checks for Week-2 features.

Checks for each split:
  - Shapes of text_emb, img_emb, dense
  - Same N across all
  - Index alignment (row_id monotonic 0..N-1)
  - Basic stats (has_image %, dense feature means/stds)

Outputs:
  - Prints checks to console
  - You can also paste the summary into docs/feature_qc_week2.md
"""

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = BASE_DIR / "cache"
TEXT_EMB_DIR = CACHE_DIR / "text_emb"
IMG_EMB_DIR = CACHE_DIR / "img_emb"
DENSE_DIR = CACHE_DIR / "dense"
INDEX_DIR = CACHE_DIR / "index"


def check_split(split: str):
    print(f"\n=== Alignment check for split: {split} ===")

    # Load arrays
    text_path = TEXT_EMB_DIR / f"{split}.npy"
    img_path = IMG_EMB_DIR / f"{split}.npy"
    dense_path = DENSE_DIR / f"{split}.npy"
    idx_path = INDEX_DIR / f"{split}.parquet"

    text_emb = np.load(text_path)
    img_emb = np.load(img_path)
    dense = np.load(dense_path)
    idx_df = pd.read_parquet(idx_path)

    # Shapes
    print(f"  text_emb shape : {text_emb.shape}")
    print(f"  img_emb shape  : {img_emb.shape}")
    print(f"  dense shape    : {dense.shape}")
    print(f"  index rows     : {len(idx_df)}")

    # Alignment checks
    N = len(idx_df)
    assert text_emb.shape[0] == N, "text_emb rows != index rows"
    assert img_emb.shape[0] == N, "img_emb rows != index rows"
    assert dense.shape[0] == N, "dense rows != index rows"

    # Check row_id monotonic and aligned
    assert "row_id" in idx_df.columns, "'row_id' missing in index"
    expected_row_ids = list(range(N))
    assert list(idx_df["row_id"].tolist()) == expected_row_ids, "row_id not 0..N-1 in order"

    # Basic NaN checks
    assert not np.isnan(text_emb).any(), "NaNs in text_emb"
    assert not np.isnan(img_emb).any(), "NaNs in img_emb"
    assert not np.isnan(dense).any(), "NaNs in dense"

    # Image stats
    if "has_image" in idx_df.columns and "pic_count" in idx_df.columns:
        pct_has_image = idx_df["has_image"].mean() * 100.0
        print(f"  % rows with images: {pct_has_image:.2f}%")
        print(f"  pic_count summary:\n{idx_df['pic_count'].describe()}")

    # Dense stats (per-feature means/stds)
    dense_means = dense.mean(axis=0)
    dense_stds = dense.std(axis=0)
    print(f"  dense means (first 5): {dense_means[:5]}")
    print(f"  dense stds  (first 5): {dense_stds[:5]}")

    print("  ✔ Alignment OK")


def main():
    for split in ["train", "val", "test"]:
        check_split(split)

    print("\nAll splits passed alignment checks.")


if __name__ == "__main__":
    main()
