
import pandas as pd
from pathlib import Path

processed_dir = Path("recommender/data/processed")
img_manifest = list(processed_dir.glob("image_manifest.parquet*.parquet"))[0]
print(f"Reading: {img_manifest}")

df = pd.read_parquet(img_manifest)
print(df.head())
print(df.columns)

train_path = processed_dir / "train.parquet"
if train_path.exists():
    print(f"\nReading: {train_path}")
    df_train = pd.read_parquet(train_path)
    print(df_train.head())
    print(df_train.columns)
