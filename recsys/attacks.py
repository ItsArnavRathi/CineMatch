"""Shilling attacks, detection and a detect-and-filter defence (lectures 19 and 21-27)."""
from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np
import pandas as pd

from . import models as M
from .evaluation import top_n_indices

ATTACK_KINDS = ["Random", "Average", "Bandwagon"]
DETECTORS = ["RDMA", "WDMA"]


# --------------------------------------------------------------------------- #
# Attack generation
# --------------------------------------------------------------------------- #
def _item_std(rm: M.RatingMatrix) -> np.ndarray:
    cnt = np.maximum(rm.item_count, 1)
    mean = rm.R.sum(0) / cnt
    var = (rm.R ** 2).sum(0) / cnt - mean ** 2
    gstd = float(np.sqrt(max((rm.R ** 2).sum() / rm.M.sum() - rm.global_mean ** 2, 0.1)))
    return np.where(rm.item_count >= 3, np.sqrt(np.maximum(var, 0.05)), gstd)


def make_attack_profiles(rm: M.RatingMatrix, target: int, n_attackers: int, filler_frac: float, kind: str,
                         rng: np.random.Generator, push: bool = True) -> pd.DataFrame:
    """Fake users that all give `target` the max (push) / min (nuke) rating, camouflaged by filler ratings.

    random    : filler ratings ~ N(global mean, global std)
    average   : filler ratings ~ N(item mean, item std)     (stronger: looks like a typical user)
    bandwagon : random filler + max rating for the few most popular items (to look similar to many users)
    """
    kind = kind.lower()
    n_fill = max(1, int(round(filler_frac * rm.n_items)))
    pool = np.setdiff1d(np.arange(rm.n_items), [target])
    gstd = float(np.sqrt(max((rm.R ** 2).sum() / rm.M.sum() - rm.global_mean ** 2, 0.1)))
    istd = _item_std(rm)
    popular = np.argsort(-rm.item_count)[:3]
    popular = popular[popular != target]

    rows = []
    for a in range(n_attackers):
        filler = rng.choice(pool, size=n_fill, replace=False)
        if kind == "average":
            vals = rng.normal(rm.item_mean_raw[filler], istd[filler])
        else:
            vals = rng.normal(rm.global_mean, gstd, size=n_fill)
        profile = dict(zip(filler.tolist(), np.clip(np.rint(vals), 1, 5).tolist()))
        if kind == "bandwagon":
            profile.update({int(j): 5.0 for j in popular})
        profile[int(target)] = 5.0 if push else 1.0
        for item, rating in profile.items():
            rows.append((rm.n_users + a, item, rating))
    return pd.DataFrame(rows, columns=["u", "i", "r"])


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def detection_scores(rm: M.RatingMatrix, method: str = "RDMA") -> np.ndarray:
    """Higher = more suspicious. Both detectors measure how much a profile disagrees with the crowd,
    weighting disagreement on rarely-rated items (where shilling happens) more heavily.

    RDMA : rating deviation from mean agreement      (1/N_u) * sum_i |r_ui - mean_i| / n_i
    WDMA : weighted deviation from mean agreement    (1/N_u) * sum_i |r_ui - mean_i| / n_i^2
    """
    power = 1.0 if method == "RDMA" else 2.0
    n_i = np.maximum(rm.item_count, 1)[None, :] ** power
    dev = np.abs(rm.R - rm.item_mean_raw[None, :]) * rm.M / n_i
    return dev.sum(1) / np.maximum(rm.user_count, 1)


def flag_users(rm: M.RatingMatrix, method: str, flag_frac: float) -> np.ndarray:
    scores = detection_scores(rm, method)
    n_flag = int(round(flag_frac * rm.n_users))
    return np.argsort(-scores)[:n_flag]


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def _victim_stats(model: M.Recommender, victims: np.ndarray, target: int, clean: M.RatingMatrix, top_n: int,
                  min_support: int) -> Dict[str, float]:
    """Mean predicted rating of `target` for genuine users who have not rated it + hit ratio@top_n."""
    pred = float(model.P[victims, target].mean())
    scores = model.rank_scores()[victims]
    allowed = (clean.M[victims] == 0) & (clean.item_count >= min_support)[None, :]
    allowed[:, target] = True
    top, _ = top_n_indices(scores, allowed, top_n)
    return {"pred": pred, "hit": float((top == target).any(1).mean())}


def run_attack(rm: M.RatingMatrix, factory: Callable[[], M.Recommender], target: int, kind: str, attack_pct: float,
               filler_pct: float, detector: str = "RDMA", flag_pct: float = 8.0, seed: int = 0, top_n: int = 10,
               min_support: int = 20, base: M.Recommender | None = None) -> Dict[str, float]:
    """Inject attack_pct % fake users, then measure damage, detection quality and the detect-and-filter defence."""
    rng = np.random.default_rng(seed)
    victims = np.flatnonzero(rm.M[:, target] == 0)
    base = base or factory().fit(rm)
    before = _victim_stats(base, victims, target, rm, top_n, min_support)

    n_att = int(round(attack_pct / 100.0 * rm.n_users))
    out = {"attack_pct": attack_pct, "attackers": n_att, "pred_before": before["pred"], "hit_before": before["hit"]}
    if n_att == 0:
        out.update(pred_attack=before["pred"], hit_attack=before["hit"], pred_defended=before["pred"],
                   hit_defended=before["hit"], pred_reference=before["pred"], hit_reference=before["hit"],
                   det_precision=float("nan"), det_recall=float("nan"))
        return out

    atk = make_attack_profiles(rm, target, n_att, filler_pct / 100.0, kind, rng)
    rm_a = M.RatingMatrix(pd.concat([rm.df, atk], ignore_index=True), rm.n_users + n_att, rm.n_items)
    attacked = factory().fit(rm_a)
    after = _victim_stats(attacked, victims, target, rm, top_n, min_support)

    flagged = flag_users(rm_a, detector, flag_pct / 100.0)
    is_attacker = flagged >= rm.n_users
    out["det_precision"] = float(is_attacker.mean()) if len(flagged) else float("nan")
    out["det_recall"] = float(is_attacker.sum() / n_att)

    keep = ~rm_a.df["u"].isin(flagged)
    rm_d = M.RatingMatrix(rm_a.df[keep], rm_a.n_users, rm_a.n_items)
    defended = factory().fit(rm_d)
    after_d = _victim_stats(defended, victims, target, rm, top_n, min_support)

    # Fair reference for the defence: the clean system with the same *genuine* users removed, so the reported
    # residual shift measures the attack only, not the cost of discarding innocent users.
    genuine_flagged = flagged[flagged < rm.n_users]
    rm_ref = M.RatingMatrix(rm.df[~rm.df["u"].isin(genuine_flagged)], rm.n_users, rm.n_items)
    ref = _victim_stats(factory().fit(rm_ref), victims, target, rm, top_n, min_support)

    out.update(pred_attack=after["pred"], hit_attack=after["hit"], pred_defended=after_d["pred"],
               hit_defended=after_d["hit"], pred_reference=ref["pred"], hit_reference=ref["hit"])
    return out


def attack_sweep(rm: M.RatingMatrix, factory: Callable[[], M.Recommender], target: int, kind: str,
                 sizes_pct: Sequence[float], filler_pct: float, detector: str = "RDMA", flag_pct: float = 8.0,
                 seed: int = 0, min_support: int = 20) -> pd.DataFrame:
    base = factory().fit(rm)
    rows = [run_attack(rm, factory, target, kind, s, filler_pct, detector, flag_pct, seed,
                       min_support=min_support, base=base) for s in sizes_pct]
    df = pd.DataFrame(rows)
    df["shift_attack"] = df["pred_attack"] - df["pred_before"]
    df["shift_defended"] = df["pred_defended"] - df["pred_reference"]
    return df


def target_candidates(rm: M.RatingMatrix, lo: int = 15, hi: int = 60) -> np.ndarray:
    """Niche movies (lo..hi ratings), ordered by average rating: the usual targets of push attacks."""
    cand = np.flatnonzero((rm.item_count >= lo) & (rm.item_count <= hi))
    if cand.size == 0:
        cand = np.flatnonzero(rm.item_count > 0)
    return cand[np.argsort(rm.item_mean_raw[cand])]


def default_target(rm: M.RatingMatrix) -> int:
    """A mediocre niche movie: the median-rated one among the niche candidates."""
    cand = target_candidates(rm)
    return int(cand[len(cand) // 2])
