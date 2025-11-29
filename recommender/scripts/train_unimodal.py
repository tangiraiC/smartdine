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


def get_feature(batch, modality):
    if modality == "text":
        return batch["text"]
    elif modality == "img":
        return batch["img"]
    elif modality == "dense":
        return batch["dense"]
    else:
        raise ValueError(modality)


def train_unimodal(modality: str, input_dim: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = SmartDineSplit("train")
    val_ds = SmartDineSplit("val")
    test_ds = SmartDineSplit("test")

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2048)
    test_loader = DataLoader(test_ds, batch_size=2048)

    model = TowerMLP(input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.MSELoss()

    def eval_rmse(loader):
        model.eval()
        ys, preds = [], []
        with torch.no_grad():
            for batch in loader:
                x = get_feature(batch, modality).to(device)
                y = batch["rating"].to(device)
                pred = model(x)
                ys.append(y.cpu())
                preds.append(pred.cpu())
        ys_arr = torch.cat(ys).numpy()
        preds_arr = torch.cat(preds).numpy()

        # Older sklearn: compute MSE then sqrt → RMSE
        mse = mean_squared_error(ys_arr, preds_arr)
        return mse ** 0.5

    best_val = 1e9
    ckpt_dir = BASE_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, 21):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            x = get_feature(batch, modality).to(device)
            y = batch["rating"].to(device)

            opt.zero_grad()
            pred = model(x)
            loss = crit(pred, y)
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(y)

        train_loss = total_loss / len(train_ds)
        val_rmse = eval_rmse(val_loader)
        print(f"[{modality}] epoch {epoch} | train_loss={train_loss:.4f} | val_rmse={val_rmse:.4f}")

        if val_rmse < best_val:
            best_val = val_rmse
            torch.save(
                model.state_dict(),
                ckpt_dir / f"{modality}_tower_best.pt"
            )

    # final test RMSE with best model (already in memory)
    test_rmse = eval_rmse(test_loader)
    print(f"[{modality}] best_val_rmse={best_val:.4f} | test_rmse={test_rmse:.4f}")
    return best_val, test_rmse


if __name__ == "__main__":
    results = []
    results.append(("text",) + train_unimodal("text", 768))
    results.append(("img",) + train_unimodal("img", 768))
    results.append(("dense",) + train_unimodal("dense", 8))

    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results, columns=["modality", "best_val_rmse", "test_rmse"])
    df.to_csv(reports_dir / "week3_unimodal_rmse.csv", index=False)
