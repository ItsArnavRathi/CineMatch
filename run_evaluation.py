"""Command-line version of the Model Lab: runs the full offline evaluation and saves the tables.

    python run_evaluation.py                 # defaults: 80/20 split, K=10, relevant if rating >= 4
    python run_evaluation.py --k 20 --test-frac 0.3
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from recsys import data as D
from recsys import evaluation as E
from recsys import models as M


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=4.0, help="rating at or above which a test item is relevant")
    ap.add_argument("--min-support", type=int, default=20, help="candidate movies need at least this many ratings")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    ds, warning = D.get_dataset(args.data_dir)
    if warning:
        print(f"WARNING: using SYNTHETIC data. {warning}\n")
    print(f"Dataset: {ds.name}  ({ds.n_users} users, {ds.n_items} movies, {len(ds.ratings):,} ratings)")

    X, names = M.build_item_features(ds.items)
    t0 = time.time()
    res = E.run_full_evaluation(ds.ratings, ds.n_users, ds.n_items, X, names, args.test_frac, args.k,
                                args.threshold, args.min_support, args.seed)
    print(f"Evaluation finished in {time.time() - t0:.1f}s  (train {res['n_train']:,} / test {res['n_test']:,})\n")

    pd.set_option("display.width", 200)
    print(res["table"].round(4).to_string(index=False))
    print("\nTuned hybrid weights:", {k: round(v, 2) for k, v in res["weights"].items()})
    print("\nRMSE vs neighbourhood size k:")
    print(res["k_sweep"].pivot(index="k (neighbours)", columns="Model", values="RMSE").round(4).to_string())
    print("\nPearson vs cosine similarity:")
    print(res["metric_cmp"].round(4).to_string(index=False))

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    res["table"].round(4).to_csv(out / "evaluation.csv", index=False)
    res["k_sweep"].round(4).to_csv(out / "k_sweep.csv", index=False)
    res["metric_cmp"].round(4).to_csv(out / "similarity_comparison.csv", index=False)
    print(f"\nSaved CSVs to {out}/")


if __name__ == "__main__":
    main()
