from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
IDX_DIR = BASE_DIR / "cache" / "index"

def main():
    train = pd.read_parquet(IDX_DIR / "train.parquet")
    val = pd.read_parquet(IDX_DIR / "val.parquet")
    test = pd.read_parquet(IDX_DIR / "test.parquet")

    train_users = set(train["user_id"].unique())
    train_items = set(train["business_id"].unique())

    for split_name, df in [("val", val), ("test", test)]:
        df["is_new_user"] = ~df["user_id"].isin(train_users)
        df["is_new_item"] = ~df["business_id"].isin(train_items)
        df.to_parquet(IDX_DIR / f"{split_name}.parquet", index=False)
        print(split_name, "new_user_pct:", df["is_new_user"].mean(),
                         "new_item_pct:", df["is_new_item"].mean())

if __name__ == "__main__":
    main()
