# backend/recs/services.py

from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F
from django.conf import settings

from recommender.models.joint import JointRecModel
from recommender.serving.query_encoder import encode_query

# --- Globals, lazy-loaded ---
_joint_model = None
_item_cache = None
_item_meta_by_id = None


def _get_recommender_root() -> Path:
    """
    Get the path to the top-level 'recommender' directory.
    settings.BASE_DIR is backend/, so parent is repo root.
    """
    backend_root = Path(settings.BASE_DIR)  # backend/
    repo_root = backend_root.parent         # smartdine/
    return repo_root / "recommender"        # smartdine/recommender


def _load_model_and_items():
    """
    Lazily load:
      - trained joint model
      - item feature cache (text/img/dense)
      - precomputed normalized text embeddings
      - item metadata (image URLs, etc.)
    """
    global _joint_model, _item_cache, _item_meta_by_id
    if _joint_model is not None and _item_cache is not None and _item_meta_by_id is not None:
        return

    rec_root = _get_recommender_root()
    ckpt_path = rec_root / "checkpoints" / "joint_best.pt"
    cfg_path = rec_root / "serving" / "joint_config.json"
    item_cache_path = rec_root / "serving" / "item_feature_cache.npz"
    meta_path = rec_root / "serving" / "item_metadata.json"

    # ----- Load config -----
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    n_users = int(cfg["n_users"])
    text_dim = cfg.get("text_dim", 768)
    img_dim = cfg.get("img_dim", 768)
    dense_dim = cfg.get("dense_dim", 8)
    user_emb_dim = cfg.get("user_emb_dim", 64)
    hidden_dims = tuple(cfg.get("hidden_dims", [512, 256]))

    # ----- Instantiate model -----
    model = JointRecModel(
        n_users=n_users,
        text_dim=text_dim,
        img_dim=img_dim,
        dense_dim=dense_dim,
        user_emb_dim=user_emb_dim,
        hidden_dims=hidden_dims,
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    _joint_model = model

    # ----- Load item cache -----
    cache = np.load(item_cache_path, allow_pickle=True)
    business_ids = cache["business_ids"]                  # (M,)
    text_emb = torch.from_numpy(cache["text_emb"]).float()   # (M, 768)
    img_emb = torch.from_numpy(cache["img_emb"]).float()     # (M, 768)
    dense = torch.from_numpy(cache["dense"]).float()         # (M, 8)

    # Precompute normalized text embeddings for cosine sim
    text_emb_norm = F.normalize(text_emb, dim=1)

    _item_cache = {
        "business_ids": business_ids,
        "text_emb": text_emb,
        "text_emb_norm": text_emb_norm,
        "img_emb": img_emb,
        "dense": dense,
    }

    # ----- Load metadata (image URLs, etc.) -----
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_list = json.load(f)
        _item_meta_by_id = {m["business_id"]: m for m in meta_list}
    except FileNotFoundError:
        _item_meta_by_id = {}


def rank_candidates(preferences: dict, k: int = 10) -> list[dict]:
    """
    Model-backed ranker using joint model + query-text similarity.

    preferences:
      {
        "query_text": "...",   # free text
        "k": 10
      }

    Returns a list of dicts:
      {
        "business_id": str,
        "score": float,
        "name": Optional[str],
        "representative_image_url": Optional[str],
        "avg_rating": Optional[float],
        "num_reviews": Optional[int],
      }
    """
    _load_model_and_items()
    model = _joint_model
    cache = _item_cache
    meta_by_id = _item_meta_by_id

    k = int(preferences.get("k", k))
    query_text = preferences.get("query_text", "") or ""

    business_ids = cache["business_ids"]         # (M,)
    text_emb = cache["text_emb"]                 # (M, 768)
    text_emb_norm = cache["text_emb_norm"]       # (M, 768)
    img_emb = cache["img_emb"]                   # (M, 768)
    dense = cache["dense"]                       # (M, 8)

    M = business_ids.shape[0]

    # 1) Base scores from joint model (dummy user 0 for all items)
    user_idx = torch.zeros(M, dtype=torch.long)  # (M,)
    with torch.no_grad():
        base_scores = model(user_idx, text_emb, img_emb, dense)  # (M,)

    # 2) Query-text similarity in the same SBERT space
    with torch.no_grad():
        q_emb = encode_query(query_text)         # (1, 768)
        q_emb_norm = F.normalize(q_emb, dim=1)   # (1, 768)
        # cosine sim = dot since both are normalized
        sim = torch.matmul(text_emb_norm, q_emb_norm.T).squeeze(-1)  # (M,)

    # 3) Combine: base_scores + λ * sim
    lambda_query = 0.5  # you can tune this offline
    combined = base_scores + lambda_query * sim  # (M,)

    scores_np = combined.detach().cpu().numpy()
    topk_idx = np.argsort(-scores_np)[:k]

    results = []
    for i in topk_idx:
        bid = str(business_ids[i])
        score = float(scores_np[i])
        meta = meta_by_id.get(bid, {})

        results.append({
            "business_id": bid,
            "score": score,
            "name": meta.get("name"),
            "representative_image_url": meta.get("image_url"),
            "avg_rating": meta.get("avg_rating"),
            "num_reviews": meta.get("num_reviews"),
        })

    return results
