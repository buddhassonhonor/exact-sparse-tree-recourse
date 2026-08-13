#!/usr/bin/env python
"""Evaluate both scaled L1 and scaled L2 under every actionability protocol."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
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
METHODS = ("metric_projection", "metric_repair", "fixed_support_exact", "native_exact_sparse")


@dataclass(frozen=True)
class Policy:
    protocol: str
    immutable: bool
    max_changes: int | None


POLICIES = (
    Policy("free", False, None),
    Policy("immutable", True, None),
    Policy("immutable_top3", True, 3),
)
METRICS = ("weighted_l1", "scaled_l2")


def repair(clf, x0, raw, support, nn_target):
    restricted = old.apply_support(x0, raw, support)
    if restricted is not None and int(clf.predict(restricted.reshape(1, -1))[0]) == 1:
        return restricted
    return old.constrained_repair(clf, restricted, nn_target, support, 8)


def task(dataset: str, seed: int, max_queries: int) -> list[dict]:
    ds = bench.load_dataset(dataset, seed)
    tr, te = train_test_split(
        np.arange(len(ds.y)), test_size=0.35, stratify=ds.y, random_state=seed
    )
    Xtr = ds.X[tr].astype(np.float32, copy=True)
    base_Xte = ds.X[te].astype(np.float32, copy=True)
    scale = np.std(Xtr.astype(np.float64), axis=0)
    scale = np.where(scale > bench.EPS, scale, 1.0)
    clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=seed).fit(Xtr, ds.y[tr])
    boxes = old.get_leaf_boxes(clf)
    leaves = [node for node in boxes if int(np.argmax(clf.tree_.value[node][0])) == 1]
    lbs, ubs = bench.implementable_bound_matrices(boxes, leaves, Xtr.shape[1])
    pos_train = Xtr[clf.predict(Xtr) == 1]
    rows: list[dict] = []
    rng = np.random.default_rng(seed + 1009)

    for displacement in (0.0, 1.0):
        Xte = base_Xte.copy()
        if displacement:
            count = min(3, Xte.shape[1])
            Xte[:, :count] = (Xte[:, :count].astype(np.float64) + scale[:count]).astype(np.float32)
        query_ids = np.where(clf.predict(Xte) == 0)[0]
        if len(query_ids) > max_queries:
            query_ids = np.sort(rng.choice(query_ids, max_queries, replace=False))

        for policy in POLICIES:
            immutable = bench.immutable_indices(
                ds, bench.Protocol("metric_check", policy.immutable, policy.max_changes, "scaled_l2")
            )
            actionable = old.actionable_indices(Xtr.shape[1], immutable)
            for metric in METRICS:
                counts = {method: 0 for method in METHODS}
                costs = {method: [] for method in METHODS}
                for qi in query_ids:
                    x0 = Xte[int(qi)]
                    nn = old.nearest_positive_train(x0, pos_train)
                    support = old.choose_support(x0, nn, actionable, policy.max_changes)
                    nn_target = old.build_nn_target(x0, nn, support)
                    raw = bench.select_projection(x0, lbs, ubs, scale, metric, matched_metric=True)
                    candidates = {
                        "metric_projection": old.apply_support(x0, raw, support),
                        "metric_repair": repair(clf, x0, raw, support, nn_target),
                        "fixed_support_exact": bench.project_exact_cached(
                            x0, clf, lbs, ubs, support, scale, metric
                        ),
                        "native_exact_sparse": bench.project_exact_cached(
                            x0, clf, lbs, ubs, actionable, scale, metric,
                            max_changes=policy.max_changes
                        ),
                    }
                    for method, candidate in candidates.items():
                        valid = candidate is not None and int(clf.predict(candidate.reshape(1, -1))[0]) == 1
                        if valid:
                            counts[method] += 1
                            costs[method].append(old.scaled_cost(x0, candidate, scale, metric))
                for method in METHODS:
                    rows.append({
                        "dataset": dataset, "seed": seed, "displacement": displacement,
                        "protocol": policy.protocol, "metric": metric, "method": method,
                        "n_queries": len(query_ids),
                        "coverage": counts[method] / max(1, len(query_ids)),
                        "conditional_cost": float(np.mean(costs[method])) if costs[method] else np.nan,
                    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-queries", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    parts = Parallel(n_jobs=args.workers, backend="loky", verbose=10)(
        delayed(task)(dataset, seed, args.max_queries)
        for dataset in bench.DATASET_NAMES for seed in bench.DEFAULT_SEEDS
    )
    detail = pd.DataFrame([row for part in parts for row in part])
    summary = detail.groupby(["protocol", "metric", "method"], as_index=False).agg(
        n_units=("coverage", "size"), coverage=("coverage", "mean"),
        conditional_cost=("conditional_cost", "mean"),
    )
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(RESULT_DIR / "metric_crosscheck_detail.csv", index=False)
    summary.to_csv(RESULT_DIR / "metric_crosscheck_summary.csv", index=False)
    metadata = {
        "datasets": list(bench.DATASET_NAMES), "seeds": list(bench.DEFAULT_SEEDS),
        "displacements": [0.0, 1.0], "metrics": list(METRICS),
        "policies": [policy.__dict__ for policy in POLICIES], "max_queries": args.max_queries,
        "dtype": "IEEE-754 binary32 for queries, bounds, candidates, and prediction validation",
    }
    (RESULT_DIR / "metric_crosscheck_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
