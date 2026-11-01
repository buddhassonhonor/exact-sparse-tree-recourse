#!/usr/bin/env python
"""Evaluate recourse under observed-sample covariate reweighting environments."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
MVE_DIR = ROOT / "script" / "mve"
sys.path.insert(0, str(MVE_DIR))
import recourse_core as core  # noqa: E402
import run_review_benchmark as bench  # noqa: E402


DATASETS = tuple(name for name in bench.DATASET_NAMES if name != "synthetic_overlap")
METHODS = ("clbr", "metric_clbr", "exact_leaf")
ENVIRONMENTS = ("iid", "pc1_low", "pc1_high", "mahalanobis_tail")


def environment_indices(Xtr: np.ndarray, Xneg: np.ndarray) -> dict[str, np.ndarray]:
    mean = Xtr.astype(np.float64).mean(axis=0)
    scale = Xtr.astype(np.float64).std(axis=0)
    scale = np.where(scale > bench.EPS, scale, 1.0)
    Ztr = (Xtr.astype(np.float64) - mean) / scale
    Zneg = (Xneg.astype(np.float64) - mean) / scale
    pca = PCA(n_components=1, random_state=0).fit(Ztr)
    pc1 = pca.transform(Zneg).ravel()
    distance = LedoitWolf().fit(Ztr).mahalanobis(Zneg)
    q25, q75 = np.quantile(pc1, [0.25, 0.75])
    d75 = np.quantile(distance, 0.75)
    return {
        "iid": np.arange(len(Xneg), dtype=int),
        "pc1_low": np.where(pc1 <= q25)[0],
        "pc1_high": np.where(pc1 >= q75)[0],
        "mahalanobis_tail": np.where(distance >= d75)[0],
    }


def covariance_distance(Xtr: np.ndarray, Xenv: np.ndarray) -> float:
    if len(Xenv) < 2:
        return float("nan")
    mean = Xtr.astype(np.float64).mean(axis=0)
    scale = Xtr.astype(np.float64).std(axis=0)
    scale = np.where(scale > bench.EPS, scale, 1.0)
    train_cov = np.cov(((Xtr.astype(np.float64) - mean) / scale), rowvar=False)
    env_cov = np.cov(((Xenv.astype(np.float64) - mean) / scale), rowvar=False)
    denominator = max(float(np.linalg.norm(train_cov, ord="fro")), bench.EPS)
    return float(np.linalg.norm(env_cov - train_cov, ord="fro") / denominator)


def run_task(dataset: str, seed: int, max_queries: int) -> tuple[list[dict], list[dict]]:
    ds = bench.load_dataset(dataset, seed)
    tr_idx, te_idx = train_test_split(
        np.arange(len(ds.y)), test_size=0.35, stratify=ds.y, random_state=seed
    )
    Xtr = ds.X[tr_idx].astype(np.float32, copy=True)
    Xte = ds.X[te_idx].astype(np.float32, copy=True)
    ytr = ds.y[tr_idx]
    scale = np.std(Xtr.astype(np.float64), axis=0)
    scale = np.where(scale > bench.EPS, scale, 1.0)
    clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=seed).fit(Xtr, ytr)
    boxes = core.get_leaf_boxes(clf)
    leaf_ids = clf.apply(Xtr)
    leaves = [node for node in boxes if int(np.argmax(clf.tree_.value[node][0])) == 1]
    lbs, ubs = bench.implementable_bound_matrices(boxes, leaves, Xtr.shape[1])
    pos_train = Xtr[clf.predict(Xtr) == 1]
    negative_test_ids = np.where(clf.predict(Xte) == 0)[0]
    Xneg = Xte[negative_test_ids]
    selections = environment_indices(Xtr, Xneg)
    immutable = bench.immutable_indices(ds, bench.PROTOCOLS[1])
    actionable = core.actionable_indices(Xtr.shape[1], immutable)
    rng = np.random.default_rng(seed + 3001)
    query_rows: list[dict] = []
    environment_rows: list[dict] = []

    for environment, local_ids in selections.items():
        if len(local_ids) > max_queries:
            local_ids = np.sort(rng.choice(local_ids, size=max_queries, replace=False))
        environment_rows.append({
            "dataset": dataset, "seed": seed, "environment": environment,
            "available_negative_queries": int(len(selections[environment])),
            "evaluated_queries": int(len(local_ids)),
            "covariance_distance": covariance_distance(Xtr, Xneg[local_ids]),
        })
        for local_id in local_ids:
            query_idx = int(negative_test_ids[int(local_id)])
            x0 = Xte[query_idx]
            nn = core.nearest_positive_train(x0, pos_train)
            nn_target = core.build_nn_target(x0, nn, actionable)
            for method in METHODS:
                start = time.perf_counter()
                candidate, _, _ = bench.candidate_for(
                    method, x0, clf, Xtr, pos_train, leaf_ids, boxes, leaves,
                    lbs, ubs, actionable, actionable, None, nn_target, scale,
                    "scaled_l2", 8,
                )
                elapsed = time.perf_counter() - start
                valid = candidate is not None and int(clf.predict(candidate.reshape(1, -1))[0]) == 1
                query_rows.append({
                    "dataset": dataset, "seed": seed, "environment": environment,
                    "query_idx": query_idx, "method": method, "valid": int(valid),
                    "cost": core.scaled_cost(x0, candidate, scale, "scaled_l2") if valid else np.nan,
                    "changed_count": len(core.changed_feature_indices(x0, candidate)) if valid else np.nan,
                    "runtime_ms": elapsed * 1000.0,
                })
    return query_rows, environment_rows


def hierarchical_interval(frame: pd.DataFrame, value: str, draws: int = 4000) -> tuple[float, float]:
    rng = np.random.default_rng(2026)
    datasets = frame.dataset.unique()
    grouped = {name: part[value].dropna().to_numpy(float) for name, part in frame.groupby("dataset")}
    samples = np.empty(draws)
    for draw in range(draws):
        selected = rng.choice(datasets, len(datasets), replace=True)
        means = []
        for name in selected:
            values = grouped[name]
            means.append(rng.choice(values, len(values), replace=True).mean())
        samples[draw] = np.mean(means)
    return tuple(np.quantile(samples, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(bench.DEFAULT_SEEDS))
    parser.add_argument("--max-queries", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    parts = Parallel(n_jobs=max(1, args.workers), backend="loky", verbose=10)(
        delayed(run_task)(dataset, seed, args.max_queries)
        for dataset in args.datasets for seed in args.seeds
    )
    detail = pd.DataFrame([row for queries, _ in parts for row in queries])
    environments = pd.DataFrame([row for _, envs in parts for row in envs])
    unit = detail.groupby(["dataset", "seed", "environment", "method"], as_index=False).agg(
        n_queries=("valid", "size"), coverage=("valid", "mean"),
        conditional_cost=("cost", "mean"), mean_changed=("changed_count", "mean"),
        runtime_ms=("runtime_ms", "mean"),
    )
    summary = unit.groupby(["environment", "method"], as_index=False).agg(
        n_units=("coverage", "size"), coverage=("coverage", "mean"),
        conditional_cost=("conditional_cost", "mean"), mean_changed=("mean_changed", "mean"),
    )
    paired = unit.pivot(
        index=["dataset", "seed", "environment"], columns="method", values="coverage"
    ).reset_index()
    paired["exact_minus_metric_repair"] = paired["exact_leaf"] - paired["metric_clbr"]
    intervals = []
    for environment, part in paired.groupby("environment"):
        low, high = hierarchical_interval(part, "exact_minus_metric_repair")
        intervals.append({
            "environment": environment,
            "mean_coverage_gain": float(part.exact_minus_metric_repair.mean()),
            "ci_low": low, "ci_high": high,
        })
    intervals = pd.DataFrame(intervals)

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "exp" / "empirical_shift" / stamp
    out_dir.mkdir(parents=True)
    detail.to_csv(out_dir / "query_results.csv", index=False)
    unit.to_csv(out_dir / "unit_summary.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    environments.to_csv(out_dir / "environment_diagnostics.csv", index=False)
    intervals.to_csv(out_dir / "coverage_gain_intervals.csv", index=False)
    metadata = {
        "generated_at": dt.datetime.now().isoformat(), "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "datasets": args.datasets, "seeds": args.seeds, "environments": list(ENVIRONMENTS),
        "environment_definition": {
            "iid": "all observed negative test predictions",
            "pc1_low": "lowest quartile of training-standardized PC1 among observed negative test predictions",
            "pc1_high": "highest quartile of training-standardized PC1 among observed negative test predictions",
            "mahalanobis_tail": "highest quartile of Ledoit-Wolf Mahalanobis distance among observed negative test predictions",
        },
        "sample_modification": "none; environments select unchanged observed test records",
        "aggregation": "query metrics within dataset-seed-environment, then equal-weight dataset-seed units",
        "bootstrap": "datasets resampled first, then dataset-seed units within sampled datasets; 4000 draws",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\n", intervals.to_string(index=False))
    print(out_dir)


if __name__ == "__main__":
    main()
