"""
Build text embeddings and basic text features for train/val/test.

Outputs (under recommender/cache/):
  text_emb/{split}.npy        : (N_split, 768) float32
  index/{split}.parquet       : row_id, business_id, user_id, rating
  manifests/text_emb.json     : metadata about the embeddings
  feature_qc_week2.md         : you will append checks manually / from another script

Assumes:
  - Processed parquet files live under recommender/data/processed/
  - Each split parquet has: business_id, user_id, rating, review_text
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer


# -------------------------
# Paths & constants
# -------------------------

BASE_DIR = Path(__file__).resolve().parents[1]  # recommender/
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CACHE_DIR = BASE_DIR / "cache"
TEXT_EMB_DIR = CACHE_DIR / "text_emb"
INDEX_DIR = CACHE_DIR / "index"
MANIFESTS_DIR = CACHE_DIR / "manifests"
DOCS_DIR = BASE_DIR / "docs"

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMB_DIM = 768
RANDOM_SEED = 42


# -------------------------
# Utilities
# -------------------------

def clean_text(raw: str) -> str:
    """
    Deterministic text cleaning:
      - Strip HTML
      - Normalize whitespace
      - Lowercase
    """
    if raw is None:
        return ""
    # Remove HTML tags if present
    soup = BeautifulSoup(str(raw), "html.parser")
    text = soup.get_text(separator=" ")
    # Normalize whitespace and lowercase
    text = " ".join(text.split()).lower()
    return text


def compute_text_features(df: pd.DataFrame, sia: SentimentIntensityAnalyzer) -> pd.DataFrame:
    """
    Adds:
      - cleaned_text
      - text_len_words
      - sentiment (compound VADER score in [-1, 1])
    """
    df = df.copy()

    # Clean text
    df["cleaned_text"] = df["review_text"].astype(str).apply(clean_text)

    # Word length
    df["text_len_words"] = df["cleaned_text"].apply(lambda s: len(s.split()))

    # Sentiment (compound)
    def sentiment_score(s: str) -> float:
        if not s:
            return 0.0
        return float(sia.polarity_scores(s)["compound"])

    df["sentiment"] = df["cleaned_text"].apply(sentiment_score)

    return df


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """
    Encode a list of texts into a 2D numpy array (N, EMB_DIM).
    normalize_embeddings=True is good for cosine-based similarities later.
    """
    # model.encode returns a numpy array already
    emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype("float32")


# -------------------------
# Main per-split routine
# -------------------------

def process_split(split: str, model: SentenceTransformer, sia: SentimentIntensityAnalyzer) -> int:
    """
    Load {split}.parquet, compute text features + embeddings, and write:
      - text_emb/{split}.npy
      - index/{split}.parquet
    Returns the number of rows (N_split).
    """
    path = PROCESSED_DIR / f"{split}.parquet"
    print(f"\n=== Processing split: {split} ===")
    print(f"Loading {path} ...")

    df = pd.read_parquet(path)

    required_cols = ["business_id", "user_id", "rating", "review_text"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    # Compute cleaned_text, text_len_words, sentiment
    df = compute_text_features(df, sia)

    # Create deterministic row_id for alignment with other caches
    df["row_id"] = np.arange(len(df), dtype=np.int64)

    # Embed the cleaned text
    print(f"Encoding {len(df)} reviews for split '{split}' ...")
    texts = df["cleaned_text"].tolist()
    emb = embed_texts(model, texts)  # (N, EMB_DIM)

    # Sanity check
    assert emb.shape[0] == len(df)
    assert emb.shape[1] == EMB_DIM

    # Make sure output directories exist
    TEXT_EMB_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    emb_path = TEXT_EMB_DIR / f"{split}.npy"
    np.save(emb_path, emb)
    print(f"Saved text embeddings to {emb_path}")

    # Save index file (basic identity + text features we might reuse)
    index_cols = [
        "row_id",
        "business_id",
        "user_id",
        "rating",
        "text_len_words",
        "sentiment",
    ]
    index_df = df[index_cols].copy()
    index_path = INDEX_DIR / f"{split}.parquet"
    index_df.to_parquet(index_path, index=False)
    print(f"Saved index for split '{split}' to {index_path}")

    return len(df)


# -------------------------
# Manifest writer
# -------------------------

def write_manifest(split_counts: dict[str, int]) -> None:
    """
    Writes a JSON manifest describing the text embeddings.
    """
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFESTS_DIR / "text_emb.json"

    manifest = {
        "model_name": MODEL_NAME,
        "dim": EMB_DIM,
        "split_counts": split_counts,
        "seed": RANDOM_SEED,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_paths": {
            split: f"{PROCESSED_DIR}/{split}.parquet" for split in split_counts
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote text embedding manifest to {manifest_path}")


# -------------------------
# Entry point
# -------------------------

def main():
    # Fix random seed for any stochastic behavior (not crucial here but good practice)
    np.random.seed(RANDOM_SEED)

    # Make sure NLTK VADER is available:
    #   >>> import nltk; nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()

    print(f"Loading SentenceTransformer model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    split_counts: dict[str, int] = {}
    for split in ["train", "val", "test"]:
        n_rows = process_split(split, model, sia)
        split_counts[split] = n_rows

    write_manifest(split_counts)

    print("\nAll splits done. Text embeddings & index files are ready.")


if __name__ == "__main__":
    main()
