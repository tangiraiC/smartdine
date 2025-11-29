from pathlib import Path
import json
import pandas as pd

# BASE_DIR = smartdine/recommender/
BASE_DIR = Path(__file__).resolve().parents[1]

RAW_PATH = BASE_DIR / "data" / "raw" / "filter_all_t.json"

OUTDIR = BASE_DIR / "cache" / "manifests"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUTDIR / "image_manifest.parquet"


def main():
    print(">>> build_image_manifest.py starting up")
    print("Loading:", RAW_PATH)

    with open(RAW_PATH, "r") as f:
        root = json.load(f)

    rows = []
    for split in ["train", "val", "test"]:
        for rec in root.get(split, []):
            business_id = rec.get("business_id")
            pic_ids = rec.get("pics", [])

            if isinstance(pic_ids, list):
                pic_ids = [p for p in pic_ids if isinstance(p, str) and p]
            else:
                pic_ids = []

            rows.append({
                "business_id": business_id,
                "split": split,
                "pic_ids": pic_ids,
                "pic_count": len(pic_ids),
            })

    df = pd.DataFrame(rows)

    # 🔴 IMPORTANT: collapse to ONE ROW per (business_id, split)
    # Aggregate pic_ids by concatenating lists, sum pic_count
    df = (
    df.groupby(["business_id", "split"], as_index=False)
      .agg({
          "pic_ids": lambda lists: sum(lists, []),
          "pic_count": "sum",
      })
)


    print(df.head())
    print(df["pic_count"].describe())

    df.to_parquet(OUT_PATH, index=False)
    print("Saved manifest to:", OUT_PATH)


if __name__ == "__main__":
    main()
