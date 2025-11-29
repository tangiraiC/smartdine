from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[1]  # recommender/
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# lazy global
_model = None


def get_query_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def encode_query(text: str) -> torch.Tensor:
    """
    Encode a query string into a normalized embedding (1, 768) torch tensor.
    """
    if not text:
        text = ""  # handle None
    model = get_query_model()
    emb = model.encode(
        [text],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return torch.from_numpy(emb).float()  # shape: (1, 768)
