# SmartDine — Multimodal Restaurant Recommender

SmartDine is a Django + DRF backend with a separate `recommender/` module for data and modeling (text, images, metadata). 
**Week 1 goal:** scaffold repo, define data contracts, produce processed files locally, and run stub API endpoints.

## Repo Structure
- `backend/` — Django + DRF project (API)
- `recommender/` — data, features, models, training
  - `recommender/data/` — **ignored** in git (raw + processed live here)
- `notebooks/` — light EDA and schema checks
- `ui/` — simple client (Streamlit or Vue later)
- `docs/` — data contract, API contracts, logs

## Environment
Copy `.env.example` → `.env` and set values.
- `DJANGO_SECRET_KEY` — any random string for dev
- `DATABASE_URL` — `sqlite:///db.sqlite3` for dev
- `API_DEMO_KEY` — simple header key for dev
- `RANDOM_SEED` — default `42`

## Data Locations (local only)
Place raw inputs under `recommender/data/raw/`.
Processed outputs (this week):
- `recommender/data/processed/train.parquet`
- `recommender/data/processed/val.parquet`
- `recommender/data/processed/test.parquet`
- `recommender/data/processed/image_manifest.parquet`

> NOTE: `recommender/data/` is **git-ignored**. Keep large files out of GitHub.

## Tasks (Week 1)
- **Data contract:** define schema/types for `business_id, user_id, rating, review_text, pics, history_reviews` in `docs/data_contract.md`.
- **Processed outputs:** write the four parquet files locally + `MANIFEST.json`, log counts in `docs/dataset_log.md`.
- **API skeleton:** spin up Django + DRF with `GET /api/health`, `POST /api/recommendations` (stub), `GET /api/restaurants/{id}` (stub).
- **Tracking:** simple CSV logs in `/docs` for this week. `RANDOM_SEED=42`.

## Make / Run (later weeks)
- Install deps: `pip install -r requirements.txt`
- (Week 2+) Run API: `cd backend && python manage.py runserver 0.0.0.0:8000`
