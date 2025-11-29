"""
Build a per-business (item-level) feature cache for serving.

Inputs (from Week 2/3 pipeline):
  cache/index/{split}.parquet   (row_id, business_id, user_id, rating, text_len_words, sentiment, pic_count, has_image, ...)
  cache/text_emb/{split}.npy    (N_split, 768)
  cache/img_emb/{split}.npy     (N_split, 768)
  cache/dense/{split}.npy       (N_split, 8)

Assumptions:
  - row_id is 0..N_split-1 and matches the rows in the embedding arrays
  - splits = ["train", "val", "test"]

Outputs:
  recommender/serving/item_feature_cache.npz with:
    - business_ids: (M,) array of business_id (strings)
    - text_emb: (M, 768) float32  mean-pooled per business
    - img_emb:  (M, 768) float32  mean-pooled per business
    - dense:    (M, 8)   float32  mean-pooled per business
"""

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]  # recommender/
CACHE_DIR = BASE_DIR / "cache"
INDEX_DIR = CACHE_DIR / "index"
TEXT_EMB_DIR = CACHE_DIR / "text_emb"
IMG_EMB_DIR = CACHE_DIR / "img_emb"
DENSE_DIR = CACHE_DIR / "dense"
SERVING_DIR = BASE_DIR / "serving"


def main():
    splits = ["train", "val", "test"]

    # Accumulators: business_id -> (sum_vec, count)
    sum_text = {}
    sum_img = {}
    sum_dense = {}
    counts = defaultdict(int)

    for split in splits:
        print(f"Processing split: {split}")
        idx_path = INDEX_DIR / f"{split}.parquet"
        text_path = TEXT_EMB_DIR / f"{split}.npy"
        img_path = IMG_EMB_DIR / f"{split}.npy"
        dense_path = DENSE_DIR / f"{split}.npy"

        df = pd.read_parquet(idx_path)
        text_emb = np.load(text_path)  # (N, 768)
        img_emb = np.load(img_path)    # (N, 768)
        dense = np.load(dense_path)    # (N, 8)

        assert len(df) == text_emb.shape[0] == img_emb.shape[0] == dense.shape[0], \
            f"Shape mismatch in split {split}"

        # We trust row_id ordering already; if you want, enforce:
        df = df.sort_values("row_id").reset_index(drop=True)

        for i, row in df.iterrows():
            bid = row["business_id"]
            t = text_emb[i]
            im = img_emb[i]
            d = dense[i]

            if bid not in sum_text:
                sum_text[bid] = np.zeros_like(t, dtype=np.float64)
                sum_img[bid] = np.zeros_like(im, dtype=np.float64)
                sum_dense[bid] = np.zeros_like(d, dtype=np.float64)

            sum_text[bid] += t
            sum_img[bid] += im
            sum_dense[bid] += d
            counts[bid] += 1

    # Finalize: mean per business
    business_ids = sorted(sum_text.keys())  # deterministic order
    M = len(business_ids)
    print(f"Total unique businesses: {M}")

    text_out = np.zeros((M, sum_text[business_ids[0]].shape[0]), dtype=np.float32)
    img_out = np.zeros((M, sum_img[business_ids[0]].shape[0]), dtype=np.float32)
    dense_out = np.zeros((M, sum_dense[business_ids[0]].shape[0]), dtype=np.float32)

    for i, bid in enumerate(business_ids):
        c = counts[bid]
        text_out[i] = (sum_text[bid] / c).astype(np.float32)
        img_out[i] = (sum_img[bid] / c).astype(np.float32)
        dense_out[i] = (sum_dense[bid] / c).astype(np.float32)

    SERVING_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SERVING_DIR / "item_feature_cache.npz"
    np.savez(
        out_path,
        business_ids=np.array(business_ids, dtype=object),
        text_emb=text_out,
        img_emb=img_out,
        dense=dense_out,
    )
    print(f"Saved item feature cache to: {out_path}")


if __name__ == "__main__":
    main()
