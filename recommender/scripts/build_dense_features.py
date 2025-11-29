"""
Build dense structured/history features using train-only statistics.

Inputs:
  - index/{split}.parquet   (must contain: row_id, business_id, user_id, rating,
                             text_len_words, sentiment)
  - manifests/image_manifest.parquet (business_id + split -> pic_ids, pic_count)

Outputs:
  - index/{split}.parquet   : updated with pic_count, has_image, user/item hist & bias
  - dense/{split}.npy       : (N_split, D) float32
  - manifests/dense.json    : metadata (feature order, means/stds if scaled)

Feature order (documented for your model):
  [pic_count,
   text_len_words,
   sentiment,
   user_hist_len,
   item_hist_len,
   user_bias,
   item_bias,
   has_image]
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = BASE_DIR / "cache"
INDEX_DIR = CACHE_DIR / "index"
DENSE_DIR = CACHE_DIR / "dense"
MANIFESTS_DIR = CACHE_DIR / "manifests"

IMAGE_MANIFEST_PATH = MANIFESTS_DIR / "image_manifest.parquet"

RANDOM_SEED = 42

FEATURE_ORDER = [
    "pic_count",
    "text_len_words",
    "sentiment",
    "user_hist_len",
    "item_hist_len",
    "user_bias",
    "item_bias",
    "has_image",
]


def compute_train_stats(train_idx: pd.DataFrame) -> dict:
    """
    Compute global mean rating, user bias, item bias, and history lengths on TRAIN ONLY.
    Returns a dict with all stats needed to transform val/test.
    """
    stats: dict = {}

    # Global mean rating
    mu = float(train_idx["rating"].mean())
    stats["mu"] = mu

    # User bias: mean_user_rating - mu
    user_means = train_idx.groupby("user_id")["rating"].mean()
    b_u = user_means - mu
    stats["user_bias"] = b_u.to_dict()

    # Item bias: mean_item_rating - mu
    item_means = train_idx.groupby("business_id")["rating"].mean()
    b_i = item_means - mu
    stats["item_bias"] = b_i.to_dict()

    # History lengths
    user_hist_len = train_idx.groupby("user_id")["rating"].size()
    item_hist_len = train_idx.groupby("business_id")["rating"].size()
    stats["user_hist_len"] = user_hist_len.to_dict()
    stats["item_hist_len"] = item_hist_len.to_dict()

    return stats


def add_bias_and_hist_features(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    Adds:
      - user_hist_len
      - item_hist_len
      - user_bias
      - item_bias

    Using TRAIN-derived stats; unknown users/items (in val/test) get 0.
    """
    df = df.copy()

    user_hist = stats["user_hist_len"]
    item_hist = stats["item_hist_len"]
    user_bias = stats["user_bias"]
    item_bias = stats["item_bias"]

    # History lengths (default 0 for new users/items)
    df["user_hist_len"] = df["user_id"].map(user_hist).fillna(0).astype("int32")
    df["item_hist_len"] = df["business_id"].map(item_hist).fillna(0).astype("int32")

    # Biases (default 0 → back to global mean)
    df["user_bias"] = df["user_id"].map(user_bias).fillna(0.0).astype("float32")
    df["item_bias"] = df["business_id"].map(item_bias).fillna(0.0).astype("float32")

    return df


def merge_image_features(df: pd.DataFrame, img_manifest: pd.DataFrame, split: str) -> pd.DataFrame:
    """
    For a given split's index df, merge in pic_count from image_manifest and derive has_image.
    """
    df = df.copy()

    manifest_split = img_manifest[img_manifest["split"] == split][
        ["business_id", "pic_count"]
    ].copy()

    # Merge by business_id (each split has its own manifest subset)
    df = df.merge(
        manifest_split,
        on="business_id",
        how="left",
        suffixes=("", "_img"),
    )

    # Rows with no entry in manifest get pic_count = 0
    df["pic_count"] = df["pic_count"].fillna(0).astype("int32")

    # has_image flag
    df["has_image"] = (df["pic_count"] > 0).astype("int32")

    return df


def build_dense_for_split(df: pd.DataFrame) -> np.ndarray:
    """
    Stack your FEATURE_ORDER columns into a 2D numpy array (N, D).
    """
    dense = df[FEATURE_ORDER].to_numpy(dtype="float32")
    return dense


def main():
    np.random.seed(RANDOM_SEED)

    # Load image manifest (business_id + split -> pic_count)
    print(f"Loading image manifest from {IMAGE_MANIFEST_PATH} ...")
    img_manifest = pd.read_parquet(IMAGE_MANIFEST_PATH)

    # Load train index to compute stats (we'll add image features to it first)
    train_idx_path = INDEX_DIR / "train.parquet"
    train_idx = pd.read_parquet(train_idx_path)

    required_core = [
        "business_id", "user_id", "rating",
        "text_len_words", "sentiment",
    ]
    missing = [c for c in required_core if c not in train_idx.columns]
    if missing:
        raise ValueError(f"Missing columns in train index: {missing}")

    # First add image features to TRAIN index (so pic_count & has_image exist there)
    train_idx = merge_image_features(train_idx, img_manifest, split="train")

    # Compute train-only stats for biases and history lengths
    stats = compute_train_stats(train_idx)

    # We'll store updated indices and dense features for each split
    dense_arrays: dict[str, np.ndarray] = {}

    for split in ["train", "val", "test"]:
        print(f"\n=== Building dense features for split: {split} ===")
        idx_path = INDEX_DIR / f"{split}.parquet"
        df = pd.read_parquet(idx_path)

        # Add image features (pic_count, has_image) from manifest
        df = merge_image_features(df, img_manifest, split=split)

        # Add bias + history columns (using train stats)
        df = add_bias_and_hist_features(df, stats)

        # Build dense matrix (no scaling yet)
        dense = build_dense_for_split(df)
        dense_arrays[split] = dense

        # Overwrite index with the new columns included
        df.to_parquet(idx_path, index=False)
        print(f"Updated index written to {idx_path}")

    # Apply z-score scaling based on TRAIN
    print("\nApplying z-score scaling based on TRAIN...")
    dense_train = dense_arrays["train"]
    means = dense_train.mean(axis=0)
    stds = dense_train.std(axis=0)
    stds_safe = np.where(stds == 0, 1.0, stds)  # avoid divide-by-zero

    for split in ["train", "val", "test"]:
        dense = dense_arrays[split]
        dense_arrays[split] = ((dense - means) / stds_safe).astype("float32")

    scaling_info = {
        "scaler": "zscore(train)",
        "means": means.tolist(),
        "stds": stds_safe.tolist(),
    }

    # Save dense arrays
    DENSE_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        out_path = DENSE_DIR / f"{split}.npy"
        np.save(out_path, dense_arrays[split])
        print(f"Saved dense features for '{split}' to {out_path}")

    # Write manifest
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFESTS_DIR / "dense.json"

    manifest = {
        "feature_order": FEATURE_ORDER,
        "fitted_on": "train",
        "scaler": scaling_info["scaler"],
        "means": scaling_info["means"],
        "stds": scaling_info["stds"],
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote dense feature manifest to {manifest_path}")
    print("Dense features ready for all splits.")


if __name__ == "__main__":
    main()
