from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# BASE_DIR should be the "recommender" directory
BASE_DIR = Path(__file__).resolve().parent      # or .parents[0]
CACHE_DIR = BASE_DIR / "cache"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"


class SmartDineSplit(Dataset):
    def __init__(self, split: str):
        self.split = split

        # ratings parquet lives in recommender/data/processed/{split}.parquet
        idx_path = DATA_PROCESSED_DIR / f"{split}.parquet"
        self.idx = pd.read_parquet(idx_path)

        # embeddings still in recommender/cache/...
        self.text = np.load(CACHE_DIR / "text_emb" / f"{split}.npy")
        self.img = np.load(CACHE_DIR / "img_emb" / f"{split}.npy")
        self.dense = np.load(CACHE_DIR / "dense" / f"{split}.npy")

        assert len(self.idx) == len(self.text) == len(self.img) == len(self.dense)

        self.user_ids = self.idx["user_id"].astype("category")
        self.item_ids = self.idx["business_id"].astype("category")

        self.user_map = dict(enumerate(self.user_ids.cat.categories))
        self.item_map = dict(enumerate(self.item_ids.cat.categories))

        self.user_idx = torch.tensor(self.user_ids.cat.codes.values, dtype=torch.long)
        self.item_idx = torch.tensor(self.item_ids.cat.codes.values, dtype=torch.long)

        self.ratings = torch.tensor(self.idx["rating"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return {
            "user_idx": self.user_idx[i],
            "item_idx": self.item_idx[i],
            "rating": self.ratings[i],
            "text": torch.from_numpy(self.text[i]),
            "img": torch.from_numpy(self.img[i]),
            "dense": torch.from_numpy(self.dense[i]),
        }
