"""Data loading for MovieLens 100K, plus a clearly-labelled synthetic fallback.

All ids are converted to 0-based indices (u, i) so they can index numpy matrices directly.
"""
from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


class DatasetUnavailable(RuntimeError):
    """Raised when MovieLens 100K can neither be found locally nor downloaded."""


@dataclass
class Dataset:
    name: str
    ratings: pd.DataFrame   # columns: u, i, r, ts   (0-based ids)
    items: pd.DataFrame     # index = item idx; columns: title, year, genres (list), one 0/1 column per genre
    users: pd.DataFrame     # index = user idx; columns: age, gender, occupation, zip
    synthetic: bool = False

    @property
    def n_users(self) -> int:
        return int(self.users.shape[0])

    @property
    def n_items(self) -> int:
        return int(self.items.shape[0])

    def label(self, i: int) -> str:
        """Display label such as 'Toy Story (1995)'."""
        year = self.items.at[i, "year"]
        title = self.items.at[i, "title"]
        return f"{title} ({int(year)})" if pd.notna(year) else str(title)


# --------------------------------------------------------------------------- #
# Real data
# --------------------------------------------------------------------------- #
def download_movielens(data_dir: str | Path) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(ML100K_URL, timeout=60) as resp:
            payload = resp.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            zf.extractall(data_dir)
    except Exception as exc:  # network down, blocked, corrupt zip ...
        raise DatasetUnavailable(
            f"Could not download MovieLens 100K ({exc}). Download {ML100K_URL} manually and unzip it "
            f"so that '{data_dir / 'ml-100k' / 'u.data'}' exists."
        ) from exc


def _clean_title(raw: str) -> str:
    title = re.sub(r"\s*\(\d{4}\)\s*$", "", str(raw)).strip()
    m = re.match(r"^(.*), (The|A|An)$", title)       # 'Shawshank Redemption, The' -> 'The Shawshank Redemption'
    return f"{m.group(2)} {m.group(1)}" if m else title


def load_movielens(data_dir: str | Path = "data", download: bool = True) -> Dataset:
    root = Path(data_dir) / "ml-100k"
    if not (root / "u.data").exists():
        if not download:
            raise DatasetUnavailable(f"'{root / 'u.data'}' not found.")
        download_movielens(data_dir)

    ratings = pd.read_csv(root / "u.data", sep="\t", names=["u", "i", "r", "ts"])
    ratings["u"] -= 1
    ratings["i"] -= 1

    # u.item is pipe-separated latin-1: id | title | release | video release | url | 19 genre flags.
    # Parsed line by line (id, title, release date from the front; genre flags from the back) so a row with an
    # odd number of empty middle fields cannot shift the columns.
    rows = []
    with open(root / "u.item", encoding="latin-1") as fh:
        for line in fh:
            parts = line.rstrip("\r\n").split("|")
            if len(parts) < 5 + len(GENRES):
                continue
            rows.append([int(parts[0]), parts[1], parts[2]] + [int(x) for x in parts[-len(GENRES):]])
    items = pd.DataFrame(rows, columns=["item_id", "title_raw", "release"] + GENRES)
    items = items.sort_values("item_id").reset_index(drop=True)
    year = pd.to_datetime(items["release"], format="%d-%b-%Y", errors="coerce").dt.year
    title_year = items["title_raw"].str.extract(r"\((\d{4})\)\s*$")[0].astype(float)
    items["year"] = year.fillna(title_year)
    items["title"] = items["title_raw"].map(_clean_title)

    def genre_list(row) -> list:
        gs = [g for g in GENRES[1:] if row[g] == 1]
        return gs or ["unknown"]

    items["genres"] = items.apply(genre_list, axis=1)
    items = items[["title", "year", "genres"] + GENRES]

    users = pd.read_csv(root / "u.user", sep="|", names=["user_id", "age", "gender", "occupation", "zip"])
    users = users.sort_values("user_id").reset_index(drop=True).drop(columns="user_id")

    return Dataset("MovieLens 100K", ratings, items, users, synthetic=False)


# --------------------------------------------------------------------------- #
# Synthetic stand-in (only used when the real data cannot be loaded)
# --------------------------------------------------------------------------- #
def make_synthetic(n_users: int = 943, n_items: int = 1682, seed: int = 7) -> Dataset:
    """Fake data with MovieLens-100K's shape and a latent-factor structure, for offline demos/tests."""
    rng = np.random.default_rng(seed)
    d = 6
    n_g = len(GENRES)
    genre_vecs = rng.normal(0, 1, (n_g, d))

    G = np.zeros((n_items, n_g), dtype=np.int8)
    for i in range(n_items):
        k = rng.choice([1, 2, 3], p=[0.45, 0.40, 0.15])
        G[i, rng.choice(np.arange(1, n_g), size=k, replace=False)] = 1

    years = np.where(rng.random(n_items) < 0.75, rng.integers(1992, 1999, n_items), rng.integers(1930, 1992, n_items))
    q = (G @ genre_vecs) / np.sqrt(G.sum(1, keepdims=True)) + 0.4 * rng.normal(size=(n_items, d))
    b_i = rng.normal(0, 0.5, n_items)
    pop = rng.pareto(1.2, n_items) + 1.0
    pop = pop / pop.sum()

    p = rng.normal(0, 1, (n_users, d))
    b_u = rng.normal(0, 0.4, n_users)
    counts = np.clip(rng.lognormal(4.3, 0.9, n_users).astype(int), 20, 700)

    rows = []
    for u in range(n_users):
        chosen = rng.choice(n_items, size=min(counts[u], n_items), replace=False, p=pop)
        raw = 3.4 + b_u[u] + b_i[chosen] + 0.55 * (q[chosen] @ p[u]) / np.sqrt(d) + rng.normal(0, 0.6, len(chosen))
        r = np.clip(np.rint(raw), 1, 5)
        rows.append(np.column_stack([np.full(len(chosen), u), chosen, r, rng.integers(8.7e8, 8.9e8, len(chosen))]))
    ratings = pd.DataFrame(np.vstack(rows), columns=["u", "i", "r", "ts"]).astype(
        {"u": int, "i": int, "r": int, "ts": int})

    items = pd.DataFrame(G, columns=GENRES)
    items.insert(0, "genres", [[GENRES[j] for j in np.flatnonzero(G[i]) if j > 0] or ["unknown"] for i in range(n_items)])
    items.insert(0, "year", years.astype(float))
    items.insert(0, "title", [f"Synthetic Film {i + 1:04d}" for i in range(n_items)])

    users = pd.DataFrame({
        "age": rng.integers(15, 65, n_users),
        "gender": rng.choice(["M", "F"], n_users),
        "occupation": rng.choice(["student", "engineer", "educator", "artist", "writer", "other"], n_users),
        "zip": ["00000"] * n_users,
    })
    return Dataset("Synthetic demo data", ratings, items, users, synthetic=True)


def get_dataset(data_dir: str | Path = "data", download: bool = True,
                allow_synthetic: bool = True) -> Tuple[Dataset, Optional[str]]:
    """Return (dataset, warning). The warning is set when the synthetic fallback was used."""
    try:
        return load_movielens(data_dir, download=download), None
    except DatasetUnavailable as exc:
        if not allow_synthetic:
            raise
        return make_synthetic(), str(exc)
