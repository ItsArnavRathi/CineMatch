"""Offline evaluation on historical data (lectures 28-31): split, RMSE/MAE, Precision/Recall/NDCG@K, coverage."""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from . import models as M


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def holdout_split(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42,
                  min_ratings: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user random hold-out: every user with >= min_ratings ratings gives test_frac of them to the test set."""
    rng = np.random.default_rng(seed)
    d = df[["u", "i", "r"]].copy()
    d["_rand"] = rng.random(len(d))
    d["_rank"] = d.groupby("u")["_rand"].rank(method="first")
    d["_n"] = d.groupby("u")["u"].transform("size")
    is_test = (d["_n"] >= min_ratings) & (d["_rank"] <= np.floor(d["_n"] * test_frac))
    cols = ["u", "i", "r"]
    return d.loc[~is_test, cols].reset_index(drop=True), d.loc[is_test, cols].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def rmse(P: np.ndarray, test: pd.DataFrame) -> float:
    err = P[test["u"].to_numpy(), test["i"].to_numpy()] - test["r"].to_numpy()
    return float(np.sqrt(np.mean(err ** 2)))


def mae(P: np.ndarray, test: pd.DataFrame) -> float:
    err = P[test["u"].to_numpy(), test["i"].to_numpy()] - test["r"].to_numpy()
    return float(np.mean(np.abs(err)))


def top_n_indices(scores: np.ndarray, allowed: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Top-k allowed items per row, best first. Returns (indices, scores); disallowed slots have score -inf."""
    s = np.where(allowed, scores, -np.inf)
    k = min(k, s.shape[1])
    idx = np.argpartition(-s, k - 1, axis=1)[:, :k]
    part = np.take_along_axis(s, idx, 1)
    order = np.argsort(-part, axis=1)
    return np.take_along_axis(idx, order, 1), np.take_along_axis(part, order, 1)


def ranking_metrics(scores: np.ndarray, rm_train: M.RatingMatrix, test: pd.DataFrame, k: int = 10,
                    threshold: float = 4.0, min_support: int = 20) -> Dict[str, float]:
    """Precision@K, Recall@K, NDCG@K, HitRate@K and catalogue coverage, over users with >= 1 relevant test item."""
    allowed = (rm_train.M == 0) & (rm_train.item_count >= min_support)[None, :]
    top, vals = top_n_indices(scores, allowed, k)
    valid = np.isfinite(vals)

    rel = np.zeros(rm_train.M.shape, dtype=bool)
    t = test[test["r"] >= threshold]
    rel[t["u"].to_numpy(), t["i"].to_numpy()] = True
    n_rel = rel.sum(1)
    users = n_rel > 0

    hits = np.take_along_axis(rel, top, 1) & valid
    n_hit = hits.sum(1)
    disc = 1.0 / np.log2(np.arange(top.shape[1]) + 2.0)
    dcg = (hits * disc[None, :]).sum(1)
    cum = np.concatenate([[0.0], np.cumsum(disc)])
    idcg = cum[np.minimum(n_rel, top.shape[1])]

    catalogue = int((rm_train.item_count >= min_support).sum())
    return {
        "Precision@K": float((n_hit / k)[users].mean()),
        "Recall@K": float((n_hit / np.maximum(n_rel, 1))[users].mean()),
        "NDCG@K": float((dcg / np.maximum(idcg, 1e-12))[users].mean()),
        "HitRate@K": float((n_hit > 0)[users].mean()),
        "Coverage": float(np.unique(top[valid]).size / max(catalogue, 1)),
    }


def evaluate_models(models: Dict[str, M.Recommender], rm_train: M.RatingMatrix, test: pd.DataFrame, k: int = 10,
                    threshold: float = 4.0, min_support: int = 20) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        row = {"Model": name, "RMSE": rmse(model.P, test), "MAE": mae(model.P, test)}
        row.update(ranking_metrics(model.rank_scores(), rm_train, test, k, threshold, min_support))
        rows.append(row)
    return pd.DataFrame(rows).rename(columns={
        "Precision@K": f"Precision@{k}", "Recall@K": f"Recall@{k}", "NDCG@K": f"NDCG@{k}",
        "HitRate@K": f"HitRate@{k}"})


# --------------------------------------------------------------------------- #
# Experiments used by the "Model Lab"
# --------------------------------------------------------------------------- #
def tune_hybrid_weights(train: pd.DataFrame, n_users: int, n_items: int, X: np.ndarray, names: Sequence[str],
                        step: float = 0.1, val_frac: float = 0.15, seed: int = 11, k: int = 40
                        ) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Grid-search blend weights on a validation split carved out of the *training* data (no test leakage)."""
    tr, val = holdout_split(train, val_frac, seed)
    rm = M.RatingMatrix(tr, n_users, n_items)
    cb = M.ContentBased(X, names).fit(rm)
    uk = M.UserKNN(k).fit(rm)
    ik = M.ItemKNN(k).fit(rm)

    rows = []
    grid = np.round(np.arange(0, 1 + 1e-9, step), 6)
    for wc in grid:
        for wu in grid:
            wi = 1.0 - wc - wu
            if wi < -1e-9:
                continue
            wi = max(float(wi), 0.0)
            P = wc * cb.P + wu * uk.P + wi * ik.P
            rows.append({M.CONTENT: wc, M.USER_CF: wu, M.ITEM_CF: wi, "Validation RMSE": rmse(P, val)})
    table = pd.DataFrame(rows).sort_values("Validation RMSE").reset_index(drop=True)
    best = table.iloc[0]
    return {M.CONTENT: float(best[M.CONTENT]), M.USER_CF: float(best[M.USER_CF]),
            M.ITEM_CF: float(best[M.ITEM_CF])}, table


def neighbor_sweep(ucf: M.UserKNN, icf: M.ItemKNN, test: pd.DataFrame,
                   ks: Sequence[int] = (5, 10, 20, 40, 80, 160)) -> pd.DataFrame:
    rows = []
    for k in ks:
        for model in (ucf, icf):
            rows.append({"k (neighbours)": k, "Model": model.name, "RMSE": rmse(model.predict_with_k(k), test)})
    return pd.DataFrame(rows)


def metric_comparison(rm_train: M.RatingMatrix, test: pd.DataFrame, k: int = 40) -> pd.DataFrame:
    rows = []
    for metric in ("pearson", "cosine"):
        for cls in (M.UserKNN, M.ItemKNN):
            model = cls(k, metric).fit(rm_train)
            rows.append({"Model": model.name, "Similarity": metric.capitalize(), "RMSE": rmse(model.P, test)})
    return pd.DataFrame(rows)


def run_full_evaluation(ratings: pd.DataFrame, n_users: int, n_items: int, X: np.ndarray, names: Sequence[str],
                        test_frac: float = 0.2, k: int = 10, threshold: float = 4.0, min_support: int = 20,
                        seed: int = 42) -> Dict[str, object]:
    """The whole offline protocol in one call (used by the CLI script and the app's Model Lab)."""
    train, test = holdout_split(ratings, test_frac, seed)
    rm_tr = M.RatingMatrix(train, n_users, n_items)
    models = M.fit_all(rm_tr, X, names)
    weights, grid = tune_hybrid_weights(train, n_users, n_items, X, names)
    models_h = dict(models)
    models_h[M.HYBRID] = M.make_hybrid(models, weights)
    table = evaluate_models(models_h, rm_tr, test, k, threshold, min_support)
    return {
        "table": table, "weights": weights, "weight_grid": grid.head(15),
        "k_sweep": neighbor_sweep(models[M.USER_CF], models[M.ITEM_CF], test),
        "metric_cmp": metric_comparison(rm_tr, test),
        "n_train": len(train), "n_test": len(test),
        "density": float(len(ratings) / (n_users * n_items)),
    }
