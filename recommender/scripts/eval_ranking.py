from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error  # not strictly needed here, but ok

# project paths
BASE_DIR = Path(__file__).resolve().parents[1]  # -> recommender
sys.path.append(str(BASE_DIR))

from datasets import SmartDineSplit
from models.mf import MatrixFactorization
from models.tower import TowerMLP
from eval.ranking import compute_ranking_metrics_for_scores


device = "cuda" if torch.cuda.is_available() else "cpu"


def make_split_frame(split: str):
    """
    Helper: get SmartDineSplit + DataFrame with user_idx, item_idx, rating.
    """
    ds = SmartDineSplit(split)
    df = ds.idx.copy()
    df["user_idx"] = ds.user_idx.numpy()
    df["item_idx"] = ds.item_idx.numpy()
    df["rating"] = ds.ratings.numpy()
    return ds, df


@torch.no_grad()
def scores_mf(split: str, n_factors: int = 64):
    train_ds, _ = make_split_frame("train")
    n_users = int(train_ds.user_idx.max().item() + 1)
    n_items = int(train_ds.item_idx.max().item() + 1)

    model = MatrixFactorization(n_users, n_items, n_factors=n_factors).to(device)
    ckpt = BASE_DIR / "checkpoints" / "mf_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))

    ds, df = make_split_frame(split)
    loader = DataLoader(ds, batch_size=4096, shuffle=False)

    scores = []
    for batch in loader:
        u = batch["user_idx"].to(device)
        i = batch["item_idx"].to(device)
        preds = model(u, i)
        scores.append(preds.cpu())
    scores = torch.cat(scores).numpy()
    return df, scores


@torch.no_grad()
def scores_unimodal(split: str, modality: str, input_dim: int):
    """
    modality in {"text", "img", "dense"}
    """
    ds, df = make_split_frame(split)

    model = TowerMLP(input_dim).to(device)
    ckpt = BASE_DIR / "checkpoints" / f"{modality}_tower_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))

    loader = DataLoader(ds, batch_size=4096, shuffle=False)
    scores = []

    for batch in loader:
        if modality == "text":
            x = batch["text"].to(device)
        elif modality == "img":
            x = batch["img"].to(device)
        elif modality == "dense":
            x = batch["dense"].to(device)
        else:
            raise ValueError(modality)

        preds = model(x)
        scores.append(preds.cpu())

    scores = torch.cat(scores).numpy()
    return df, scores


@torch.no_grad()
def scores_late_fusion(split: str):
    train_ds, _ = make_split_frame("train")
    text_dim = train_ds.text.shape[1]
    img_dim = train_ds.img.shape[1]
    dense_dim = train_ds.dense.shape[1]
    input_dim = text_dim + img_dim + dense_dim

    model = TowerMLP(input_dim).to(device)
    ckpt = BASE_DIR / "checkpoints" / "late_fusion_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))

    ds, df = make_split_frame(split)
    loader = DataLoader(ds, batch_size=4096, shuffle=False)

    scores = []
    for batch in loader:
        text = batch["text"].to(device)
        img = batch["img"].to(device)
        dense = batch["dense"].to(device)
        x = torch.cat([text, img, dense], dim=-1)
        preds = model(x)
        scores.append(preds.cpu())

    scores = torch.cat(scores).numpy()
    return df, scores


def main():
    rows = []
    k_values = (5, 10, 20)
    pos_threshold = 4.0  # rating >= 4 is "relevant"

    models = [
        ("mf", scores_mf),
        ("text", lambda split: scores_unimodal(split, "text", 768)),
        ("img", lambda split: scores_unimodal(split, "img", 768)),
        ("dense", lambda split: scores_unimodal(split, "dense", 8)),
        ("late_fusion", scores_late_fusion),
    ]

    for model_name, scorer in models:
        for split in ["val", "test"]:
            print(f"Evaluating ranking metrics for {model_name} on {split}...")

            df, scores = scorer(split)
            metrics_df = compute_ranking_metrics_for_scores(
                df,
                scores,
                k_values=k_values,
                pos_threshold=pos_threshold,
                user_col="user_idx",
                rating_col="rating",
            )

            metrics_df["model"] = model_name
            metrics_df["split"] = split
            rows.append(metrics_df)

    if rows:
        all_metrics = pd.concat(rows, ignore_index=True)
        reports_dir = BASE_DIR / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        out_path = reports_dir / "week3_ranking_metrics.csv"
        all_metrics.to_csv(out_path, index=False)
        print("Saved ranking metrics to", out_path)
    else:
        print("No metrics computed (no positives found?)")


if __name__ == "__main__":
    main()
