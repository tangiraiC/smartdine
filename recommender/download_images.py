# save as: download_pics_ids.py
import json, requests
from pathlib import Path

INFILE = "data/filter_all_t.json"     # <-- set your path
OUTDIR = Path("data/images")          # images will be saved here
OUTDIR.mkdir(parents=True, exist_ok=True)

BASES = [
    "https://lh3.googleusercontent.com/p/",
    "https://lh5.googleusercontent.com/p/",
]

TIMEOUT = 10

def fetch_one(pid: str):
    for b in BASES:
        url = b + pid
        try:
            r = requests.get(url, timeout=TIMEOUT, stream=True)
            if r.status_code == 200 and r.headers.get("content-type","").startswith(("image/","application/octet-stream")):
                return r
        except Exception:
            pass
    return None

def collect_all_ids(obj):
    """obj has keys train/val/test -> list[dict], each dict has pics: list[str]."""
    seen = set()
    for split in ("train", "val", "test"):
        for rec in obj.get(split, []):
            for pid in rec.get("pics", []) or []:
                if isinstance(pid, str) and pid and pid not in seen:
                    seen.add(pid)
    return list(seen)

def main():
    with open(INFILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    ids = collect_all_ids(data)
    print(f"[info] unique pic IDs: {len(ids)}")

    saved = failed = 0
    for i, pid in enumerate(ids, 1):
        out = OUTDIR / f"{pid}.jpg"
        if out.exists() and out.stat().st_size > 0:
            saved += 1
        else:
            resp = fetch_one(pid)
            if resp:
                with open(out, "wb") as f:
                    for chunk in resp.iter_content(1 << 15):
                        if chunk: f.write(chunk)
                saved += 1
            else:
                failed += 1

        if i % 500 == 0:
            print(f"[progress] {i}/{len(ids)} | saved={saved} failed={failed}")

    print(f"[done] saved={saved} failed={failed} -> {OUTDIR}")

if __name__ == "__main__":
    # pip install requests
    main()
