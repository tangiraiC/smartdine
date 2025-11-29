from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------------------------------------------------------------
# Paths and setup
# ---------------------------------------------------------------------

# BASE_DIR = project root: .../smartdine/recommender
BASE_DIR = Path(__file__).resolve().parents[1]

# IDX_DIR = folder where our indexed rating splits live (train/val/test parquet files)
IDX_DIR = BASE_DIR / "cache" / "index"


# ---------------------------------------------------------------------
# Step 1.1: Compute global mean and user/item biases
# ---------------------------------------------------------------------

def compute_biases():
    """
    Load the training ratings and compute:
      - mu: global average rating
      - user_bias: user-specific deviation from mu
      - item_bias: item-specific deviation from mu

    These are the parameters for the "global mean + user bias + item bias" baseline.
    """
    # Read the training split (must contain columns: user_id, business_id, rating)
    train = pd.read_parquet(IDX_DIR / "train.parquet")

    # Global mean rating across all user–item pairs
    mu = train["rating"].mean()

    # Mean rating per user and per item
    user_mean = train.groupby("user_id")["rating"].mean()
    item_mean = train.groupby("business_id")["rating"].mean()

    # User bias = how much a user is above/below the global mean on average
    user_bias = user_mean - mu

    # Item bias = how much an item is above/below the global mean on average
    item_bias = item_mean - mu

    # Convert to plain dicts so we can map quickly later
    return mu, user_bias.to_dict(), item_bias.to_dict()


# ---------------------------------------------------------------------
# Step 1.2: Prediction helper using the bias model
# ---------------------------------------------------------------------

def predict_bias(df, mu, user_bias, item_bias):
    """
    Given a DataFrame with user_id and business_id columns,
    produce predictions using:
        ŷ = μ + b_u + b_i

    Any user/item not seen in training gets bias 0.0 (i.e., falls back to μ).
    """
    # Look up user and item biases; unseen IDs → NaN → fill with 0.0
    ub = df["user_id"].map(user_bias).fillna(0.0)
    ib = df["business_id"].map(item_bias).fillna(0.0)

    # Global mean + user bias + item bias
    return mu + ub + ib


# ---------------------------------------------------------------------
# Step 1.3: Evaluate RMSE and MAE on a given split
# ---------------------------------------------------------------------

def eval_rmse_mae(split, mu, user_bias, item_bias):
    """
    Evaluate the bias baseline on a given split (train/val/test).

    Returns:
        rmse: root mean squared error
        mae: mean absolute error
    """
    # Load the corresponding split (expects rating column)
    df = pd.read_parquet(IDX_DIR / f"{split}.parquet")

    # Ground truth and predictions
    y_true = df["rating"].values
    y_pred = predict_bias(df, mu, user_bias, item_bias).values

    # Some sklearn versions don't support squared=False, so we:
    # 1) compute MSE
    # 2) take the square root to get RMSE
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # MAE directly from sklearn
    mae = mean_absolute_error(y_true, y_pred)

    return rmse, mae


# ---------------------------------------------------------------------
# Step 1.4: Driver – run bias baseline for all splits and save report
# ---------------------------------------------------------------------

def main():
    """
    Entry point:
      1) fit global mean + user/item biases on the training set
      2) evaluate RMSE/MAE on train, val, and test
      3) save results as reports/week3_bias_baseline.csv
    """
    # Fit bias model on train
    mu, u_b, i_b = compute_biases()

    rows = []
    for split in ["train", "val", "test"]:
        # Evaluate metrics for this split
        rmse, mae = eval_rmse_mae(split, mu, u_b, i_b)
        rows.append({
            "model": "bias",   # name of the baseline model
            "split": split,    # which split (train/val/test)
            "rmse": rmse,
            "mae": mae,
        })

    # Ensure reports directory exists
    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics table for inclusion in your Week 3 report
    pd.DataFrame(rows).to_csv(
        reports_dir / "week3_bias_baseline.csv",
        index=False
    )


if __name__ == "__main__":
    # Run everything when you call: python eval_baselines.py
    main()
