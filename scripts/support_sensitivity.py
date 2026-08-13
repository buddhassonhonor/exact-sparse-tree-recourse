#!/usr/bin/env python
"""Sensitivity of fixed-support Exact to the support-selection rule."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
MVE_DIR = ROOT / "script" / "mve"
if not MVE_DIR.exists():
    MVE_DIR = ROOT / "scripts" / "mve"
sys.path.insert(0, str(MVE_DIR))
import recourse_core as old  # noqa: E402
import run_recourse_benchmark as bench  # noqa: E402

RESULT_DIR = ROOT / "result"
RULES = ("nearest_positive", "random", "feature_importance")


def choose(rule: str, x0: np.ndarray, nn: np.ndarray | None, actionable: np.ndarray,
           importances: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    if rule == "nearest_positive":
        return old.choose_support(x0, nn, actionable, k)
    if rule == "feature_importance":
        order = np.argsort(-importances[actionable], kind="stable")
        return actionable[order[:k]].copy()
    return np.sort(rng.choice(actionable, size=min(k, len(actionable)), replace=False))


def run(max_queries: int, random_draws: int) -> pd.DataFrame:
    rows: list[dict] = []
    for dataset in bench.DATASET_NAMES:
        for seed in bench.DEFAULT_SEEDS:
            ds = bench.load_dataset(dataset, seed)
            tr, te = train_test_split(
                np.arange(len(ds.y)), test_size=0.35, stratify=ds.y, random_state=seed
            )
            Xtr = ds.X[tr].astype(np.float32, copy=True)
            Xte = ds.X[te].astype(np.float32, copy=True)
            scale = np.std(Xtr.astype(np.float64), axis=0)
            scale = np.where(scale > bench.EPS, scale, 1.0)
            clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=seed).fit(Xtr, ds.y[tr])
            boxes = old.get_leaf_boxes(clf)
            leaves = [n for n in boxes if int(np.argmax(clf.tree_.value[n][0])) == 1]
            lbs, ubs = bench.implementable_bound_matrices(boxes, leaves, Xtr.shape[1])
            immutable = bench.immutable_indices(ds, bench.PROTOCOLS[2])
            actionable = old.actionable_indices(Xtr.shape[1], immutable)
            pos_train = Xtr[clf.predict(Xtr) == 1]
            query_ids = np.where(clf.predict(Xte) == 0)[0]
            if len(query_ids) > max_queries:
                rng_q = np.random.default_rng(seed + 1009)
                query_ids = np.sort(rng_q.choice(query_ids, max_queries, replace=False))

            for rule in RULES:
                draws = random_draws if rule == "random" else 1
                for draw in range(draws):
                    valid = 0
                    costs: list[float] = []
                    changed: list[int] = []
                    rng = np.random.default_rng(seed * 1009 + draw)
                    for qi in query_ids:
                        x0 = Xte[int(qi)]
                        nn = old.nearest_positive_train(x0, pos_train)
                        support = choose(rule, x0, nn, actionable, clf.feature_importances_, 3, rng)
                        candidate = bench.project_exact_cached(
                            x0, clf, lbs, ubs, support, scale, "weighted_l1"
                        )
                        if candidate is None:
                            continue
                        valid += 1
                        costs.append(old.scaled_cost(x0, candidate, scale, "weighted_l1"))
                        changed.append(len(old.changed_feature_indices(x0, candidate)))
                    rows.append({
                        "dataset": dataset, "seed": seed, "support_rule": rule, "draw": draw,
                        "n_queries": len(query_ids), "coverage": valid / max(1, len(query_ids)),
                        "mean_cost_valid": float(np.mean(costs)) if costs else np.nan,
                        "mean_changed_valid": float(np.mean(changed)) if changed else np.nan,
                        "sparse_share_leq3": float(np.mean(np.asarray(changed) <= 3)) if changed else np.nan,
                    })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-queries", type=int, default=200)
    parser.add_argument("--random-draws", type=int, default=5)
    args = parser.parse_args()
    detail = run(args.max_queries, args.random_draws)
    summary = detail.groupby("support_rule", as_index=False).agg(
        n_units=("coverage", "size"),
        coverage=("coverage", "mean"),
        mean_cost_valid=("mean_cost_valid", "mean"),
        mean_changed_valid=("mean_changed_valid", "mean"),
        sparse_share_leq3=("sparse_share_leq3", "mean"),
    )
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(RESULT_DIR / "support_sensitivity_detail.csv", index=False)
    summary.to_csv(RESULT_DIR / "support_sensitivity_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
