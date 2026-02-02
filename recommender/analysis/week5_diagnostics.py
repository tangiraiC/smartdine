# recommender/analysis/week5_diagnostics.py

"""
Week 5 diagnostics for SmartDine:

- Load per-item features & metadata
- Visualize item spaces (PCA)
- Inspect query steering (pizza/sushi/vegan/etc.)
- Modality ablation: text-only vs img-only vs dense-only scores
- Popularity vs model score

This version:
  - Uses a non-interactive matplotlib backend (Agg)
  - Saves plots to: recommender/analysis/figures_week5/
  - Skips t-SNE to avoid segfaults and GUI issues
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving images only
import matplotlib.pyplot as plt

from recommender.models.joint import JointRecModel
from recommender.serving.query_encoder import encode_query


BASE_DIR = Path(__file__).resolve().parents[1]  # recommender/
SERVING_DIR = BASE_DIR / "serving"
CACHE_DIR = BASE_DIR / "cache"
INDEX_DIR = CACHE_DIR / "index"
PLOT_DIR = BASE_DIR / "analysis" / "figures_week5"


def load_model_and_items():
    """
    Load trained joint model and per-item feature cache.
    """
    # ---- model ----
    cfg_path = SERVING_DIR / "joint_config.json"
    ckpt_path = BASE_DIR / "checkpoints" / "joint_best.pt"
    item_cache_path = SERVING_DIR / "item_feature_cache.npz"
    meta_path = SERVING_DIR / "item_metadata.json"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    n_users = int(cfg["n_users"])
    text_dim = cfg.get("text_dim", 768)
    img_dim = cfg.get("img_dim", 768)
    dense_dim = cfg.get("dense_dim", 8)
    user_emb_dim = cfg.get("user_emb_dim", 64)
    hidden_dims = tuple(cfg.get("hidden_dims", [512, 256]))

    model = JointRecModel(
        n_users=n_users,
        text_dim=text_dim,
        img_dim=img_dim,
        dense_dim=dense_dim,
        user_emb_dim=user_emb_dim,
        hidden_dims=hidden_dims,
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # ---- item cache ----
    cache = np.load(item_cache_path, allow_pickle=True)
    business_ids = cache["business_ids"]
    text_emb = torch.from_numpy(cache["text_emb"]).float()
    img_emb = torch.from_numpy(cache["img_emb"]).float()
    dense = torch.from_numpy(cache["dense"]).float()

    # ---- metadata ---- (optional for later extensions)
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_list = json.load(f)
        meta_by_id = {m["business_id"]: m for m in meta_list}
    except FileNotFoundError:
        meta_by_id = {}

    return model, business_ids, text_emb, img_emb, dense, meta_by_id


def load_item_popularity():
    """
    Compute per-business popularity and mean rating from train index.
    """
    train_idx = pd.read_parquet(INDEX_DIR / "train.parquet")
    grp = train_idx.groupby("business_id")["rating"]
    popularity = grp.size().rename("num_ratings")
    mean_rating = grp.mean().rename("mean_rating")
    stats = pd.concat([popularity, mean_rating], axis=1).reset_index()
    return stats


def pca_visualization(text_emb, img_emb, business_ids, item_stats, output_dir: Path):
    """
    PCA on text and image embeddings. Color points by mean rating.

    Saves:
      - pca_text.png
      - pca_image.png
    """
    print("Running PCA on text embeddings...")
    pca_text = PCA(n_components=2)
    X_text_2d = pca_text.fit_transform(text_emb.numpy())

    print("Running PCA on image embeddings...")
    pca_img = PCA(n_components=2)
    X_img_2d = pca_img.fit_transform(img_emb.numpy())

    # Join ratings
    stats = item_stats.set_index("business_id")
    mean_rating = np.array([
        stats.loc[bid, "mean_rating"] if bid in stats.index else np.nan
        for bid in business_ids
    ])

    # Text PCA plot
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(X_text_2d[:, 0], X_text_2d[:, 1], c=mean_rating, s=5, cmap="viridis")
    plt.colorbar(sc, label="Mean rating")
    plt.title("PCA of text embeddings (colored by mean rating)")
    plt.tight_layout()
    out_path = output_dir / "pca_text.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved text PCA plot to {out_path}")

    # Image PCA plot
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(X_img_2d[:, 0], X_img_2d[:, 1], c=mean_rating, s=5, cmap="viridis")
    plt.colorbar(sc, label="Mean rating")
    plt.title("PCA of image embeddings (colored by mean rating)")
    plt.tight_layout()
    out_path = output_dir / "pca_image.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved image PCA plot to {out_path}")


def query_steering_diagnostics(model, business_ids, text_emb, img_emb, dense):
    """
    Compare top items with and without query-text steering for several queries.

    Prints:
      - overlap of top-10
      - top-5 IDs for base vs query-steered scores
    """
    queries = ["pizza", "sushi", "vegan", "bbq", "brunch"]
    M = text_emb.shape[0]
    user_idx = torch.zeros(M, dtype=torch.long)

    with torch.no_grad():
        base_scores = model(user_idx, text_emb, img_emb, dense)  # (M,)

    text_emb_norm = F.normalize(text_emb, dim=1)

    for q in queries:
        with torch.no_grad():
            q_emb = encode_query(q)             # (1, 768)
            q_emb_norm = F.normalize(q_emb, 1)  # (1, 768)
            sim = torch.matmul(text_emb_norm, q_emb_norm.T).squeeze(-1)  # (M,)

        lambda_query = 0.5
        combined = base_scores + lambda_query * sim

        base_top = torch.topk(base_scores, k=10).indices.numpy()
        comb_top = torch.topk(combined, k=10).indices.numpy()

        base_bids = [business_ids[i] for i in base_top]
        comb_bids = [business_ids[i] for i in comb_top]

        overlap = len(set(base_bids) & set(comb_bids))
        print(f"\n=== Query: '{q}' ===")
        print(f"Overlap between base and combined top-10: {overlap}/10")
        print("Top-5 base IDs:    ", [str(b) for b in base_bids[:5]])
        print("Top-5 combined IDs:", [str(b) for b in comb_bids[:5]])


def modality_ablation(model, business_ids, text_emb, img_emb, dense):
    """
    Examine how each modality contributes by zeroing out others.

    Prints:
      - corr(full, text_only)
      - corr(full, img_only)
      - corr(full, dense_only)
    """
    M = text_emb.shape[0]
    user_idx = torch.zeros(M, dtype=torch.long)

    with torch.no_grad():
        full_scores = model(user_idx, text_emb, img_emb, dense)

        # Text-only: zero img + dense
        zero = torch.zeros_like
        scores_text_only = model(user_idx, text_emb, zero(img_emb), zero(dense))

        # Img-only
        scores_img_only = model(user_idx, zero(text_emb), img_emb, zero(dense))

        # Dense-only
        scores_dense_only = model(user_idx, zero(text_emb), zero(img_emb), dense)

    def corr(a, b):
        a_np = a.numpy()
        b_np = b.numpy()
        return np.corrcoef(a_np, b_np)[0, 1]

    print("\n=== Modality ablation correlations ===")
    print("corr(full, text_only): ", corr(full_scores, scores_text_only))
    print("corr(full, img_only):  ", corr(full_scores, scores_img_only))
    print("corr(full, dense_only):", corr(full_scores, scores_dense_only))


def popularity_vs_score(model, business_ids, text_emb, img_emb, dense, item_stats, output_dir: Path):
    """
    Scatter plot: popularity (num_ratings) vs model score.

    Saves:
      - popularity_vs_score.png
    """
    M = text_emb.shape[0]
    user_idx = torch.zeros(M, dtype=torch.long)

    with torch.no_grad():
        scores = model(user_idx, text_emb, img_emb, dense)  # (M,)

    stats = item_stats.set_index("business_id")
    num_ratings = np.array([
        stats.loc[bid, "num_ratings"] if bid in stats.index else 0
        for bid in business_ids
    ])

    scores_np = scores.numpy()
    plt.figure(figsize=(6, 5))
    plt.scatter(num_ratings, scores_np, s=5, alpha=0.5)
    plt.xlabel("Num ratings (popularity)")
    plt.ylabel("Model score")
    plt.title("Popularity vs model score")
    plt.tight_layout()
    out_path = output_dir / "popularity_vs_score.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved popularity vs score plot to {out_path}")


def main():
    # Ensure plot directory exists
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving Week 5 diagnostic figures under: {PLOT_DIR}")

    model, business_ids, text_emb, img_emb, dense, meta_by_id = load_model_and_items()
    item_stats = load_item_popularity()

    # 1) Visualize spaces (PCA only for stability)
    pca_visualization(text_emb, img_emb, business_ids, item_stats, PLOT_DIR)

    # 2) Query steering (prints only)
    query_steering_diagnostics(model, business_ids, text_emb, img_emb, dense)

    # 3) Modality ablation (prints only)
    modality_ablation(model, business_ids, text_emb, img_emb, dense)

    # 4) Popularity vs score (saved as PNG)
    popularity_vs_score(model, business_ids, text_emb, img_emb, dense, item_stats, PLOT_DIR)


if __name__ == "__main__":
    main()
