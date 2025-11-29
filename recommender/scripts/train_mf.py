from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error

# Project root: .../smartdine/recommender
BASE_DIR = Path(__file__).resolve().parents[1]
# Make sure Python can find `datasets.py` and `models/` under this root
sys.path.append(str(BASE_DIR))

from datasets import SmartDineSplit
from models.mf import MatrixFactorization


def train_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    crit = torch.nn.MSELoss()
    for batch in loader:
        user = batch["user_idx"].to(device)
        item = batch["item_idx"].to(device)
        y = batch["rating"].to(device)

        opt.zero_grad()
        pred = model(user, item)
        loss = crit(pred, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_rmse(model, loader, device):
    model.eval()
    ys, preds = [], []
    for batch in loader:
        user = batch["user_idx"].to(device)
        item = batch["item_idx"].to(device)
        y = batch["rating"].to(device)
        pred = model(user, item)
        ys.append(y.cpu())
        preds.append(pred.cpu())
    ys = torch.cat(ys).numpy()
    preds = torch.cat(preds).numpy()

    # Older sklearn: no `squared` kwarg → compute MSE then sqrt it
    mse = mean_squared_error(ys, preds)
    rmse = mse ** 0.5
    return rmse


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = SmartDineSplit("train")
    val_ds = SmartDineSplit("val")

    n_users = train_ds.user_idx.max().item() + 1
    n_items = train_ds.item_idx.max().item() + 1

    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4096)

    model = MatrixFactorization(n_users, n_items, n_factors=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val = 1e9
    ckpt_dir = BASE_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, 21):
        train_loss = train_epoch(model, train_loader, opt, device)
        val_rmse = eval_rmse(model, val_loader, device)
        print(f"Epoch {epoch} | train_loss={train_loss:.4f} | val_rmse={val_rmse:.4f}")
        if val_rmse < best_val:
            best_val = val_rmse
            torch.save(model.state_dict(), ckpt_dir / "mf_best.pt")


if __name__ == "__main__":
    main()
