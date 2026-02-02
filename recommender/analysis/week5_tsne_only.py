# recommender/analysis/week5_tsne_only.py

"""
Standalone t-SNE visualization for SmartDine item text embeddings.

- Loads text_emb from serving/item_feature_cache.npz
- Runs t-SNE on a subset of items
- Saves:
    recommender/analysis/figures_week5/tsne_text_subset.png

Kept separate from week5_diagnostics.py so that any t-SNE instability
cannot crash the main diagnostics script.
"""

from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving images only
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]  # recommender/
SERVING_DIR = BASE_DIR / "serving"
PLOT_DIR = BASE_DIR / "analysis" / "figures_week5"


def run_tsne(sample_size: int = 1000, random_state: int = 42):
    # Ensure plot directory exists
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving t-SNE figure under: {PLOT_DIR}")

    # Load item feature cache
    item_cache_path = SERVING_DIR / "item_feature_cache.npz"
    print(f"Loading item feature cache from: {item_cache_path}")
    cache = np.load(item_cache_path, allow_pickle=True)

    text_emb = cache["text_emb"]  # shape: (M, 768)
    M = text_emb.shape[0]
    print(f"Total items (M): {M}")

    # Subsample for t-SNE
    n_sample = min(sample_size, M)
    np.random.seed(random_state)
    idx = np.random.choice(M, size=n_sample, replace=False)
    X = text_emb[idx]

    # Choose a safe-ish perplexity relative to sample size
    perplexity = min(30, max(5, n_sample // 3))
    print(f"Running t-SNE on n={n_sample} items with perplexity={perplexity}...")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    X_2d = tsne.fit_transform(X)

    # Plot
    plt.figure(figsize=(6, 5))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], s=5, alpha=0.8)
    plt.title(f"t-SNE of text embeddings (subset, n={n_sample})")
    plt.tight_layout()
    out_path = PLOT_DIR / "tsne_text_subset.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved t-SNE plot to {out_path}")


def main():
    try:
        run_tsne()
    except Exception as e:
        # If it errors, we at least see a clean message
        print(f"[ERROR] t-SNE failed: {e}")


if __name__ == "__main__":
    main()
