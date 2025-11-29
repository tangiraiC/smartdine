from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
import pandas as pd

# Make sure we can import from the recommender package
BASE_DIR = Path(__file__).resolve().parents[1]   # -> .../SMARTDINE/recommender
sys.path.append(str(BASE_DIR))

from datasets import SmartDineSplit
from models.tower import TowerMLP


def train_late_fusion():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load splits
    train_ds = SmartDineSplit("train")
    val_ds = SmartDineSplit("val")
    test_ds = SmartDineSplit("test")

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2048)
    test_loader = DataLoader(test_ds, batch_size=2048)

    # Dynamically infer input_dim = text_dim + img_dim + dense_dim
    text_dim = train_ds.text.shape[1]
    img_dim = train_ds.img.shape[1]
    dense_dim = train_ds.dense.shape[1]
    input_dim = text_dim + img_dim + dense_dim
    print(f"Late fusion input_dim = {input_dim} (text={text_dim}, img={img_dim}, dense={dense_dim})")

    # MLP over concatenated features
    model = TowerMLP(input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.MSELoss()

    def eval_rmse(loader):
        model.eval()
        ys, preds = [], []
        with torch.no_grad():
            for batch in loader:
                # concat [text, img, dense] along feature dim
                text = batch["text"].to(device)
                img = batch["img"].to(device)
                dense = batch["dense"].to(device)
                x = torch.cat([text, img, dense], dim=-1)

                y = batch["rating"].to(device)
                pred = model(x)

                ys.append(y.cpu())
                preds.append(pred.cpu())

        ys_arr = torch.cat(ys).numpy()
        preds_arr = torch.cat(preds).numpy()
        mse = mean_squared_error(ys_arr, preds_arr)
        return mse ** 0.5

    best_val = 1e9
    ckpt_dir = BASE_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Train for 20 epochs (same as unimodal)
    for epoch in range(1, 21):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            text = batch["text"].to(device)
            img = batch["img"].to(device)
            dense = batch["dense"].to(device)
            x = torch.cat([text, img, dense], dim=-1)

            y = batch["rating"].to(device)

            opt.zero_grad()
            pred = model(x)
            loss = crit(pred, y)
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(y)

        train_loss = total_loss / len(train_ds)
        val_rmse = eval_rmse(val_loader)

        print(f"[late_fusion] epoch {epoch} | train_loss={train_loss:.4f} | val_rmse={val_rmse:.4f}")

        if val_rmse < best_val:
            best_val = val_rmse
            torch.save(model.state_dict(), ckpt_dir / "late_fusion_best.pt")

    # Final test RMSE with best model reloaded
    model.load_state_dict(torch.load(ckpt_dir / "late_fusion_best.pt", map_location=device))
    test_rmse = eval_rmse(test_loader)

    print(f"[late_fusion] best_val_rmse={best_val:.4f} | test_rmse={test_rmse:.4f}")
    return best_val, test_rmse


if __name__ == "__main__":
    best_val, test_rmse = train_late_fusion()

    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [("late_fusion_concat", best_val, test_rmse)],
        columns=["model", "best_val_rmse", "test_rmse"],
    )
    df.to_csv(reports_dir / "week3_late_fusion_rmse.csv", index=False)
    print("Saved metrics to", reports_dir / "week3_late_fusion_rmse.csv")
