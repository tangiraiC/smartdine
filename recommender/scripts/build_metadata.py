import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]  # recommender/
RAW_PATH = BASE_DIR / "data" / "raw" / "filter_all_t.json"
OUT_PATH = BASE_DIR / "serving" / "item_metadata.json"

def main():
    print(f"Reading from {RAW_PATH}")
    try:
        with open(RAW_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: Raw data file not found.")
        return

    # data seems to be { "train": [...], "val": [...], "test": [...] }
    # specific structure: a dictionary with split keys
    
    metadata = {}
    
    # We want to cover all businesses in all splits
    all_splits = ["train", "val", "test"]
    count = 0
    
    for split in all_splits:
        items = data.get(split, [])
        print(f"Processing {split}: {len(items)} items")
        for item in items:
            bid = item.get("business_id")
            if not bid:
                continue
                
            # If we already have metadata for this bid, skip or overwrite?
            # Usually strict duplicates shouldn't exist across splits if partitioned by user/item?
            # Or maybe we just take the first one.
            if bid in metadata:
                continue

            # Extract fields
            # Check keys for first item
            if count == 0:
                print("Sample item keys:", item.keys())

            name = item.get("name")
            if not name:
                 name = f"Business {bid}" # Fallback
            rating = item.get("rating") # or stars?
            if rating is None: rating = item.get("stars")
            
            num_reviews = item.get("review_count") # or num_reviews?
            if num_reviews is None: num_reviews = item.get("num_reviews")

            # Image
            # "pics": ["id1", "id2"] or just strings?
            pics = item.get("pics", [])
            image_url = None
            if pics and isinstance(pics, list) and len(pics) > 0:
                first_pic = pics[0]
                if isinstance(first_pic, str) and first_pic:
                    # Construct local URL handled by Django media serving
                    # Assuming file is {id}.jpg inside MEDIA_ROOT/images/
                    image_url = f"/media/images/{first_pic}.jpg"

            metadata[bid] = {
                "business_id": bid,
                "name": name,
                "image_url": image_url,
                "avg_rating": rating,
                "num_reviews": num_reviews
            }
            count += 1

    # Save as list or dict? services.py expects a list?
    # services.py: 
    # with open(meta_path) as f: meta_list = json.load(f)
    # _item_meta_by_id = {m["business_id"]: m for m in meta_list}
    # So it expects a LIST.

    meta_list = list(metadata.values())
    
    print(f"Saving {len(meta_list)} items to {OUT_PATH}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, indent=2)

if __name__ == "__main__":
    main()
