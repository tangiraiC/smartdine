# recommender/eval/ranking.py

from typing import Iterable, List, Dict
import numpy as np
import pandas as pd


def compute_ranking_metrics_for_scores(
    df: pd.DataFrame,
    scores: np.ndarray,
    k_values: Iterable[int] = (5, 10, 20),
    pos_threshold: float = 4.0,
    user_col: str = "user_idx",
    rating_col: str = "rating",
) -> pd.DataFrame:
    """
    df: DataFrame with at least [user_idx, rating]
    scores: 1D numpy array with a score per row in df
    k_values: list of k for @k metrics
    pos_threshold: rating >= threshold is considered a "positive" (relevant item)
    """

    df = df.copy()
    df["score"] = scores
    df["is_pos"] = df[rating_col] >= pos_threshold

    results: List[Dict] = []

    # group by user
    for user_id, user_df in df.groupby(user_col):
        num_pos = int(user_df["is_pos"].sum())
        if num_pos == 0:
            # no positives for this user in this split -> skip
            continue

        # sort items for this user by model score, descending
        user_sorted = user_df.sort_values("score", ascending=False)

        for k in k_values:
            topk = user_sorted.head(k)
            hits_k = int(topk["is_pos"].sum())

            precision = hits_k / float(k)
            recall = hits_k / float(num_pos)

            # NDCG@k
            gains = topk["is_pos"].astype(int).values
            discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
            dcg = float(np.sum(gains * discounts))

            # ideal DCG (best possible ordering)
            ideal_gains = np.sort(user_df["is_pos"].astype(int).values)[::-1][:k]
            ideal_discounts = discounts[: len(ideal_gains)]
            ideal_dcg = float(np.sum(ideal_gains * ideal_discounts))
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

            results.append(
                {
                    "user_idx": user_id,
                    "k": k,
                    "precision": precision,
                    "recall": recall,
                    "ndcg": ndcg,
                }
            )

    if not results:
        return pd.DataFrame(columns=["k", "precision", "recall", "ndcg"])

    res_df = pd.DataFrame(results)

    # average over users for each k
    agg = (
        res_df.groupby("k")[["precision", "recall", "ndcg"]]
        .mean()
        .reset_index()
    )
    return agg
