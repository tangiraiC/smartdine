from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

# Point to the recommender directory and add it to sys.path
BASE_DIR = Path(__file__).resolve().parents[1]   # .../SMARTDINE/recommender
sys.path.append(str(BASE_DIR))

from datasets import SmartDineSplit
from models.joint import JointRecModel
from eval.ranking import compute_ranking_metrics_for_scores  # from ranking.py


def eval_ndcg10(model, val_loader, device):
    """
    Evaluate the joint model using NDCG@10 on the given loader.
    """
    model.eval()
    all_scores = []
    all_users = []
    all_items = []
    all_ratings = []

    with torch.no_grad():
        for batch in val_loader:
            user_idx = batch["user_idx"].to(device)
            text = batch["text"].to(device)
            img = batch["img"].to(device)
            dense = batch["dense"].to(device)
            rating = batch["rating"].to(device)

            scores = model(user_idx, text, img, dense)

            all_scores.append(scores.cpu())
            all_users.append(user_idx.cpu())
            all_items.append(batch["item_idx"].cpu())
            all_ratings.append(rating.cpu())

    scores = torch.cat(all_scores).numpy()
    users = torch.cat(all_users).numpy()
    items = torch.cat(all_items).numpy()
    ratings = torch.cat(all_ratings).numpy()

    # Build a DataFrame like in other ranking evals
    df = pd.DataFrame(
        {
            "user_idx": users,
            "item_idx": items,
            "rating": ratings,
        }
    )

    metrics_df = compute_ranking_metrics_for_scores(
        df=df,
        scores=scores,
        k_values=(10,),       # only care about @10 here
        pos_threshold=4.0,    # rating >= 4 treated as "positive"
        user_col="user_idx",
        rating_col="rating",
    )

    # metrics_df has columns: k, precision, recall, ndcg
    ndcg10 = float(metrics_df.loc[metrics_df["k"] == 10, "ndcg"].iloc[0])
    return ndcg10


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = SmartDineSplit("train")
    val_ds = SmartDineSplit("val")

    n_users = int(train_ds.user_idx.max().item() + 1)

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2048)

    model = JointRecModel(n_users=n_users).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.MSELoss()

    best_ndcg10 = -1.0
    rows = []

    ckpt_dir = BASE_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, 21):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            user_idx = batch["user_idx"].to(device)
            text = batch["text"].to(device)
            img = batch["img"].to(device)
            dense = batch["dense"].to(device)
            rating = batch["rating"].to(device)

            opt.zero_grad()
            pred = model(user_idx, text, img, dense)
            loss = crit(pred, rating)
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(rating)

        train_loss = total_loss / len(train_ds)
        val_ndcg10 = eval_ndcg10(model, val_loader, device)

        print(f"Epoch {epoch} | train_loss={train_loss:.4f} | val_ndcg@10={val_ndcg10:.4f}")

        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_ndcg10": val_ndcg10,
            }
        )

        # keep best by NDCG@10
        if val_ndcg10 > best_ndcg10:
            best_ndcg10 = val_ndcg10
            torch.save(model.state_dict(), ckpt_dir / "joint_best.pt")

    pd.DataFrame(rows).to_csv(
        reports_dir / "week4_joint_training_curve.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
