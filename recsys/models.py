"""Recommender algorithms: non-personalized baselines, content-based, user/item kNN CF, and a hybrid.

Everything works on dense numpy matrices (943 x 1682 for MovieLens 100K), which keeps the maths
visible and fast:   R = ratings (0 = missing),  M = rating mask,  C = user-mean-centred ratings.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

EPS = 1e-9

CONTENT = "Content-Based"
USER_CF = "User-based CF"
ITEM_CF = "Item-based CF"
POPULAR = "Most Popular"
TOP_RATED = "Top Rated"
HYBRID = "Hybrid"
DEFAULT_WEIGHTS = {CONTENT: 0.2, USER_CF: 0.3, ITEM_CF: 0.5}


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(a.shape, dtype=np.float32)
    np.divide(a, b, out=out, where=np.abs(b) > EPS)
    return out


# --------------------------------------------------------------------------- #
# Rating matrix
# --------------------------------------------------------------------------- #
class RatingMatrix:
    """Dense view of a ratings table (columns u, i, r) with the statistics every model needs."""

    def __init__(self, df: pd.DataFrame, n_users: int, n_items: int):
        self.df = df[["u", "i", "r"]].reset_index(drop=True)
        self.n_users, self.n_items = int(n_users), int(n_items)

        R = np.zeros((n_users, n_items), dtype=np.float32)
        R[self.df["u"].to_numpy(), self.df["i"].to_numpy()] = self.df["r"].to_numpy(dtype=np.float32)
        self.R = R
        self.M = (R > 0).astype(np.float32)

        self.user_count = self.M.sum(1)
        self.item_count = self.M.sum(0)
        self.global_mean = float(R.sum() / max(self.M.sum(), 1.0))
        self.user_mean = np.where(self.user_count > 0, R.sum(1) / np.maximum(self.user_count, 1),
                                  self.global_mean).astype(np.float32)
        item_sum = R.sum(0)
        self.item_mean_raw = np.where(self.item_count > 0, item_sum / np.maximum(self.item_count, 1),
                                      self.global_mean).astype(np.float32)
        self.item_mean = ((item_sum + 5.0 * self.global_mean) / (self.item_count + 5.0)).astype(np.float32)

        self.C = ((R - self.user_mean[:, None]) * self.M).astype(np.float32)   # centred ratings
        self.baseline = self._bias_baseline()

    def _bias_baseline(self, lam_i: float = 10.0, lam_u: float = 10.0) -> np.ndarray:
        """mu + b_u + b_i with shrinkage: the fallback prediction when no neighbours are available."""
        mu = self.global_mean
        resid = (self.R - mu) * self.M
        b_i = resid.sum(0) / (self.item_count + lam_i)
        resid_u = (self.R - mu - b_i[None, :]) * self.M
        b_u = resid_u.sum(1) / (self.user_count + lam_u)
        return (mu + b_u[:, None] + b_i[None, :]).astype(np.float32)

    def add_user(self, ratings: Dict[int, float]) -> "RatingMatrix":
        """New matrix with one extra user (index n_users) who gave the supplied item->rating ratings."""
        extra = pd.DataFrame({"u": self.n_users, "i": list(ratings.keys()), "r": list(ratings.values())})
        return RatingMatrix(pd.concat([self.df, extra], ignore_index=True), self.n_users + 1, self.n_items)


# --------------------------------------------------------------------------- #
# Similarity helpers
# --------------------------------------------------------------------------- #
def pearson_sim(C: np.ndarray, M: np.ndarray, shrink: float = 25.0) -> np.ndarray:
    """Pearson correlation over co-rated entries, between rows of C (already mean-centred).

    For items, pass C.T / M.T: centring by user mean then gives the *adjusted cosine* similarity.
    `shrink` applies significance weighting  n_common / (n_common + shrink).
    """
    num = C @ C.T
    A = (C * C) @ M.T                    # A[u, v] = sum_i C_ui^2 * M_vi  (squared norm restricted to v's items)
    sim = _safe_div(num, np.sqrt(A * A.T))
    if shrink:
        common = M @ M.T
        sim = sim * (common / (common + shrink))
    np.fill_diagonal(sim, 0.0)
    return sim.astype(np.float32)


def cosine_sim(R: np.ndarray, M: np.ndarray, shrink: float = 25.0) -> np.ndarray:
    """Plain cosine similarity on raw ratings (missing = 0)."""
    norms = np.sqrt((R * R).sum(1))
    sim = _safe_div(R @ R.T, np.outer(norms, norms))
    if shrink:
        common = M @ M.T
        sim = sim * (common / (common + shrink))
    np.fill_diagonal(sim, 0.0)
    return sim.astype(np.float32)


def topk_weights(S: np.ndarray, k: int) -> np.ndarray:
    """Keep only the k most similar *positive* neighbours in every row."""
    n = S.shape[0]
    k = int(max(1, min(k, n - 1)))
    P = np.where(S > 0, S, 0.0).astype(np.float32)
    idx = np.argpartition(-P, k - 1, axis=1)[:, :k]
    W = np.zeros_like(P)
    rows = np.arange(n)[:, None]
    W[rows, idx] = P[rows, idx]
    return W


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class Recommender:
    name = "Recommender"

    def fit(self, rm: RatingMatrix) -> "Recommender":
        self.rm = rm
        self._rank = None
        self._fit()
        self.P = np.clip(self._predict(), 1.0, 5.0).astype(np.float32)    # predicted ratings, all (user, item)
        return self

    def _fit(self) -> None:
        pass

    def _predict(self) -> np.ndarray:
        raise NotImplementedError

    def rank_scores(self) -> np.ndarray:
        """Scores used to order candidates. Defaults to predicted rating (+ a tiny popularity tie-break)."""
        if self._rank is None:
            self._rank = self.P + 1e-6 * np.log1p(self.rm.item_count)[None, :]
        return self._rank

    def recommend(self, u: int, n: int = 10, min_support: int = 0) -> List[Tuple[int, float]]:
        """Top-n unseen items for user u as (item, predicted_rating) pairs."""
        scores = self.rank_scores()[u].astype(np.float64).copy()
        scores[self.rm.M[u] > 0] = -np.inf
        scores[self.rm.item_count < min_support] = -np.inf
        n = min(n, int(np.isfinite(scores).sum()))
        if n <= 0:
            return []
        idx = np.argpartition(-scores, n - 1)[:n]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(self.P[u, i])) for i in idx]


# --------------------------------------------------------------------------- #
# Non-personalized baselines (lecture 3)
# --------------------------------------------------------------------------- #
class PopularityRecommender(Recommender):
    """mode='count': rank by number of ratings.  mode='rating': rank by damped (Bayesian) mean rating."""

    def __init__(self, mode: str = "count", m: float = 25.0):
        assert mode in ("count", "rating")
        self.mode, self.m = mode, m
        self.name = POPULAR if mode == "count" else TOP_RATED

    def _predict(self) -> np.ndarray:
        rm = self.rm
        wr = (rm.item_mean_raw * rm.item_count + self.m * rm.global_mean) / (rm.item_count + self.m)
        return np.broadcast_to(wr[None, :], (rm.n_users, rm.n_items)).copy()

    def rank_scores(self) -> np.ndarray:
        if self._rank is None:
            if self.mode == "count":
                base = np.broadcast_to(self.rm.item_count[None, :], self.P.shape) + 1e-3 * self.P
            else:
                base = self.P + 1e-6 * np.log1p(self.rm.item_count)[None, :]
            self._rank = np.ascontiguousarray(base, dtype=np.float32)
        return self._rank


# --------------------------------------------------------------------------- #
# Content-based filtering (lectures 5-13)
# --------------------------------------------------------------------------- #
ERA_EDGES = [1960, 1970, 1980, 1990, 1995, 1997]
ERA_NAMES = ["Era: pre-1960", "Era: 1960s", "Era: 1970s", "Era: 1980s", "Era: 1990-94", "Era: 1995-96", "Era: 1997+"]


def build_item_features(items: pd.DataFrame, genres: Sequence[str] | None = None) -> Tuple[np.ndarray, List[str]]:
    """Item profile = [genre one-hots | era one-hot], IDF-weighted and L2-normalised."""
    from .data import GENRES
    genres = list(genres or GENRES[1:])
    G = items[genres].to_numpy(dtype=np.float32)
    era = np.digitize(items["year"].fillna(1995).to_numpy(), ERA_EDGES)
    E = np.eye(len(ERA_NAMES), dtype=np.float32)[era]
    F = np.hstack([G, E])
    df = (F > 0).sum(0)
    idf = np.log((F.shape[0] + 1.0) / (df + 1.0)) + 1.0
    X = F * idf[None, :]
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), EPS)
    return X.astype(np.float32), list(genres) + ERA_NAMES


class ContentBased(Recommender):
    """Learns a ridge-regression user profile over item features; predicts mu_u + w_u . x_i."""
    name = CONTENT

    def __init__(self, X: np.ndarray, feature_names: Sequence[str], lam: float = 1.0):
        self.X, self.feature_names, self.lam = X, list(feature_names), lam
        self.n_genres = len(feature_names) - len(ERA_NAMES)

    def _fit(self) -> None:
        rm, X, d = self.rm, self.X, self.X.shape[1]
        Wu = np.zeros((rm.n_users, d), dtype=np.float32)
        eye = np.eye(d, dtype=np.float32)
        for u in range(rm.n_users):
            idx = np.flatnonzero(rm.M[u])
            if idx.size == 0:
                continue
            Xu = X[idx]
            Wu[u] = np.linalg.solve(Xu.T @ Xu + self.lam * eye, Xu.T @ rm.C[u, idx])
        self.Wu = Wu

    def _predict(self) -> np.ndarray:
        return self.rm.user_mean[:, None] + self.Wu @ self.X.T

    def taste_profile(self, u: int) -> pd.Series:
        """Learned genre affinities of a user (positive = likes, negative = dislikes)."""
        return pd.Series(self.Wu[u, :self.n_genres], index=self.feature_names[:self.n_genres])

    def similar_items(self, i: int, n: int = 10) -> List[Tuple[int, float]]:
        sim = self.X @ self.X[i] + 1e-4 * np.log1p(self.rm.item_count)
        sim[i] = -np.inf
        idx = np.argsort(-sim)[:n]
        return [(int(j), float(self.X[j] @ self.X[i])) for j in idx]

    def explain(self, u: int, i: int, n: int = 2) -> List[str]:
        contrib = self.Wu[u, :self.n_genres] * self.X[i, :self.n_genres]
        order = np.argsort(-contrib)[:n]
        return [self.feature_names[j] for j in order if contrib[j] > 0]


# --------------------------------------------------------------------------- #
# Neighbourhood collaborative filtering (lectures 14-18)
# --------------------------------------------------------------------------- #
class UserKNN(Recommender):
    """r_ui = mu_u + sum_v w_uv (r_vi - mu_v) / (sum |w_uv| + damp)   over the k most similar users v."""
    name = USER_CF

    def __init__(self, k: int = 40, metric: str = "pearson", shrink: float = 25.0, damp: float = 1.0):
        self.k, self.metric, self.shrink, self.damp = k, metric, shrink, damp

    def _fit(self) -> None:
        rm = self.rm
        self.S = (pearson_sim(rm.C, rm.M, self.shrink) if self.metric == "pearson"
                  else cosine_sim(rm.R, rm.M, self.shrink))
        self.W = topk_weights(self.S, self.k)

    def _predict_from(self, W: np.ndarray) -> np.ndarray:
        rm = self.rm
        num, den = W @ rm.C, np.abs(W) @ rm.M
        P = rm.user_mean[:, None] + num / (den + self.damp)
        return np.where(den > EPS, P, rm.baseline)

    def _predict(self) -> np.ndarray:
        return self._predict_from(self.W)

    def predict_with_k(self, k: int) -> np.ndarray:
        return np.clip(self._predict_from(topk_weights(self.S, k)), 1, 5)

    def rank_scores(self) -> np.ndarray:
        """Top-N scoring: similarity-weighted count of neighbours who *liked* (>= 4) the item.

        Ranking by predicted rating alone favours obscure items with one enthusiastic neighbour; neighbour
        support is the standard fix for top-N recommendation (rating prediction still uses P)."""
        if self._rank is None:
            liked = (self.rm.R >= 4).astype(np.float32)
            self._rank = (self.W @ liked + 1e-3 * self.P).astype(np.float32)
        return self._rank

    def explain(self, u: int, i: int) -> Tuple[int, int]:
        """(# neighbours who liked item i, # neighbours who rated it)."""
        nb = self.W[u] > 0
        rated = nb & (self.rm.M[:, i] > 0)
        liked = rated & (self.rm.R[:, i] >= 4)
        return int(liked.sum()), int(rated.sum())


class ItemKNN(Recommender):
    """r_ui = mu_u + sum_j w_ij (r_uj - mu_u) / (sum |w_ij| + damp)   over the k items j most similar to i."""
    name = ITEM_CF

    def __init__(self, k: int = 40, metric: str = "pearson", shrink: float = 25.0, damp: float = 1.0):
        self.k, self.metric, self.shrink, self.damp = k, metric, shrink, damp

    def _fit(self) -> None:
        rm = self.rm
        if self.metric == "pearson":      # adjusted cosine: user-mean-centred, over co-rating users
            self.S = pearson_sim(np.ascontiguousarray(rm.C.T), np.ascontiguousarray(rm.M.T), self.shrink)
        else:
            self.S = cosine_sim(np.ascontiguousarray(rm.R.T), np.ascontiguousarray(rm.M.T), self.shrink)
        self.W = topk_weights(self.S, self.k)

    def _predict_from(self, W: np.ndarray) -> np.ndarray:
        rm = self.rm
        num, den = rm.C @ W.T, rm.M @ np.abs(W).T
        P = rm.user_mean[:, None] + num / (den + self.damp)
        return np.where(den > EPS, P, rm.baseline)

    def _predict(self) -> np.ndarray:
        return self._predict_from(self.W)

    def predict_with_k(self, k: int) -> np.ndarray:
        return np.clip(self._predict_from(topk_weights(self.S, k)), 1, 5)

    def rank_scores(self) -> np.ndarray:
        """Top-N scoring: sum of similarities between the item and the (<= k) neighbours the user liked."""
        if self._rank is None:
            liked = (self.rm.R >= 4).astype(np.float32)
            self._rank = (liked @ self.W.T + 1e-3 * self.P).astype(np.float32)
        return self._rank

    def similar_items(self, i: int, n: int = 10) -> List[Tuple[int, float]]:
        idx = np.argsort(-self.S[i])[:n]
        return [(int(j), float(self.S[i, j])) for j in idx]

    def explain(self, u: int, i: int, n: int = 2) -> List[int]:
        """Items the user liked (>=4) that are most similar to item i."""
        rm = self.rm
        cand = np.flatnonzero((rm.M[u] > 0) & (rm.R[u] >= 4) & (self.W[i] > 0))
        return cand[np.argsort(-self.W[i, cand])][:n].tolist()


# --------------------------------------------------------------------------- #
# Hybrid (lectures 33-35): weighted / parallelized hybridization
# --------------------------------------------------------------------------- #
class Hybrid(Recommender):
    """Weighted blend of already-fitted component recommenders' predicted ratings."""
    name = HYBRID

    def __init__(self, parts: Dict[str, Recommender], weights: Dict[str, float]):
        self.parts = {k: parts[k] for k in weights if weights[k] > 0}
        self.weights = {k: float(weights[k]) for k in self.parts}

    def fit(self, rm: RatingMatrix | None = None) -> "Hybrid":
        self.rm = next(iter(self.parts.values())).rm
        self._rank = None
        total = sum(self.weights.values())
        self.P = (sum(w * self.parts[k].P for k, w in self.weights.items()) / total).astype(np.float32)
        return self

    def rank_scores(self) -> np.ndarray:
        """Blend of the components' ranking scores after per-user z-scoring (their raw scales differ)."""
        if self._rank is None:
            total = sum(self.weights.values())
            blended = 0.0
            for name, w in self.weights.items():
                s = self.parts[name].rank_scores()
                z = (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + EPS)
                blended = blended + (w / total) * z
            self._rank = np.asarray(blended, dtype=np.float32)
        return self._rank


def make_hybrid(models: Dict[str, Recommender], weights: Dict[str, float] | None = None) -> Hybrid:
    return Hybrid(models, weights or DEFAULT_WEIGHTS).fit()


def fit_all(rm: RatingMatrix, X: np.ndarray, feature_names: Sequence[str], k_user: int = 40, k_item: int = 40,
            metric: str = "pearson") -> Dict[str, Recommender]:
    return {
        POPULAR: PopularityRecommender("count").fit(rm),
        TOP_RATED: PopularityRecommender("rating").fit(rm),
        CONTENT: ContentBased(X, feature_names).fit(rm),
        USER_CF: UserKNN(k_user, metric).fit(rm),
        ITEM_CF: ItemKNN(k_item, metric).fit(rm),
    }
