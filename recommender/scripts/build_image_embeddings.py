"""
Build image embeddings for train/val/test.

Inputs
------
1) Image manifest (built from filter_all_t.json), e.g. cache/manifests/image_manifest.parquet
   Columns:
     - business_id : str
     - split       : "train" | "val" | "test"
     - pic_ids     : list[str] (image IDs for that business in that split)
     - pic_count   : int       (# of images)

2) Index files per split (already created by build_text_embeddings.py):
   cache/index/{split}.parquet
   Columns (at least):
     - row_id
     - business_id
     - user_id
     - rating
     - text_len_words
     - sentiment

3) Downloaded images on disk:
   We expect images under: ../data/images/{pic_id}.jpg
   This matches download_images.py where OUTDIR = "data/images".

Outputs
-------
For each split in ["train", "val", "test"]:

  cache/img_emb/{split}.npy
      - shape: (N_rows_split, IMG_EMB_DIM)  # usually 768 for ViT-L-14
      - dtype: float32
      - aligned row-by-row with cache/index/{split}.parquet

Design
------
We do NOT require images or pic_ids inside train/val/test.parquet.

Instead:
  - image_manifest gives: business_id + split -> list of pic_ids
  - for each row in index_{split}:
      * we look up the business_id in the manifest
      * we get its pic_ids
      * we compute (or reuse cached) image embedding for that business
      * fallback to zeros if no images found

This ensures:
  - all modalities (text_emb, img_emb, dense, index) share the same row order
  - we stay memory-friendly by caching one embedding per business_id
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import open_clip  # pip install open_clip_torch


# -------------------------
# Paths & constants
# -------------------------

# This script lives in recommender/scripts/
BASE_DIR = Path(__file__).resolve().parents[1]          # recommender/
CACHE_DIR = BASE_DIR / "cache"
INDEX_DIR = CACHE_DIR / "index"
IMG_EMB_DIR = CACHE_DIR / "img_emb"
MANIFESTS_DIR = CACHE_DIR / "manifests"

# Where download_images.py saved images: OUTDIR = "data/images"
# That was relative to the project root, which sits one level above `recommender/`.
IMG_ROOT = BASE_DIR.parent / "data" / "images"

# Image manifest created from filter_all_t.json (see our build_image_manifest.py)
IMAGE_MANIFEST_PATH = MANIFESTS_DIR / "image_manifest.parquet"

# Use a CLIP/OpenCLIP model with 768-dim output so it matches your 768-d assumption
# ViT-L-14 (laion2b_s32b_b82k) is a good default
IMG_MODEL_NAME = "ViT-L-14"
IMG_PRETRAINED = "laion2b_s32b_b82k"
IMG_EMB_DIM = 768

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# Model & preprocessing
# -------------------------

def load_image_model():
    """
    Load an OpenCLIP model + preprocessing transform.

    Returns
    -------
    model : torch.nn.Module (eval mode)
    preprocess : callable that maps PIL.Image -> tensor
    device : "cuda" or "cpu"
    """
    print(f"Loading OpenCLIP model: {IMG_MODEL_NAME} ({IMG_PRETRAINED}) on {DEVICE} ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        IMG_MODEL_NAME,
        pretrained=IMG_PRETRAINED,
    )
    model.to(DEVICE)
    model.eval()
    return model, preprocess, DEVICE


# -------------------------
# Core helpers
# -------------------------

def load_image(pic_id: str) -> Optional[Image.Image]:
    """
    Attempt to load a single image from disk given a pic_id.

    Expects files at:
        IMG_ROOT / "{pic_id}.jpg"

    Returns
    -------
    PIL.Image or None if file is missing/corrupt.
    """
    path = IMG_ROOT / f"{pic_id}.jpg"
    if not path.exists():
        return None
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        print(f"[warn] Failed to load image for {pic_id}: {e}")
        return None


@torch.no_grad()
def embed_images_for_business(
    model,
    preprocess,
    pic_ids: List[str],
) -> Optional[np.ndarray]:
    """
    Given a list of pic_ids for ONE business (or one row),
    load available images, run them through OpenCLIP, and
    aggregate into a single embedding (mean-pooled).

    Returns
    -------
    emb : (IMG_EMB_DIM,) float32, or None if no images found.
    """
    tensors = []

    for pid in pic_ids:
        img = load_image(pid)
        if img is None:
            continue
        tensor = preprocess(img).unsqueeze(0).to(DEVICE)  # shape: (1, 3, H, W)
        tensors.append(tensor)

    if not tensors:
        # No images actually loaded
        return None

    batch = torch.cat(tensors, dim=0)  # (K, 3, H, W)
    # Encode images with CLIP
    feats = model.encode_image(batch)          # (K, D)
    feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize

    # Aggregate all images for this business (mean-pooled)
    pooled = feats.mean(dim=0)                # (D,)
    pooled = pooled / pooled.norm()          # normalize again

    return pooled.cpu().numpy().astype("float32")  # (D,)


def build_business_embedding_cache(
    manifest_for_split: pd.DataFrame,
    model,
    preprocess,
) -> Dict[str, np.ndarray]:
    """
    Build a cache: business_id -> image embedding.

    We do this once per split so we don't recompute the same
    business images for every row.

    Parameters
    ----------
    manifest_for_split : DataFrame with columns:
        - business_id
        - pic_ids : list[str]
        - pic_count

    Returns
    -------
    cache : dict mapping business_id -> (IMG_EMB_DIM,) float32 embedding
    """
    cache: Dict[str, np.ndarray] = {}
    print(f"\nBuilding image embedding cache for {len(manifest_for_split)} businesses...")

    for i, row in manifest_for_split.iterrows():
        bid = row["business_id"]
        pic_ids = row["pic_ids"] or []
        if not isinstance(pic_ids, list):
            continue

        emb = embed_images_for_business(model, preprocess, pic_ids)
        if emb is None:
            # If images failed, we skip; they will be handled with zeros later
            continue

        cache[bid] = emb

        if (i + 1) % 1000 == 0:
            print(f"  processed {i + 1} businesses...")

    print(f"Finished cache: {len(cache)} businesses have non-empty image embeddings.")
    return cache


# -------------------------
# Per-split logic
# -------------------------

def process_split(split: str, model, preprocess, img_manifest: pd.DataFrame) -> int:
    """
    For a given split ("train", "val", "test"):

    1. Load index/{split}.parquet (row_id, business_id, user_id, rating, ...)
    2. Filter manifest to just this split
    3. Build a cache: business_id -> embedding
    4. For each row in index, map business_id to its embedding
       - If no embedding, use zeros
    5. Save the result to cache/img_emb/{split}.npy

    Returns
    -------
    N_rows_split
    """
    index_path = INDEX_DIR / f"{split}.parquet"
    print(f"\n=== Processing split: {split} ===")
    print(f"Loading index from {index_path} ...")
    idx = pd.read_parquet(index_path)

    n_rows = len(idx)
    print(f"Index has {n_rows} rows")

    # Filter manifest for this split
    manifest_split = img_manifest[img_manifest["split"] == split].copy()
    print(f"Manifest has {len(manifest_split)} businesses for split '{split}'")

    # Build business-level embedding cache
    business_cache = build_business_embedding_cache(manifest_split, model, preprocess)

    # Prepare output array
    img_emb = np.zeros((n_rows, IMG_EMB_DIM), dtype="float32")

    # For convenient lookups
    business_ids = idx["business_id"].tolist()

    missing_count = 0
    for i, bid in enumerate(business_ids):
        emb = business_cache.get(bid)
        if emb is None:
            # No image embedding for this business -> leave as zeros
            missing_count += 1
        else:
            img_emb[i] = emb

        if (i + 1) % 10000 == 0:
            print(f"  filled {i + 1}/{n_rows} rows...")

    print(f"Rows with NO image embedding for split '{split}': {missing_count} / {n_rows}")

    # Ensure output directory exists
    IMG_EMB_DIR.m
