#!/usr/bin/env python
"""Benchmark for constrained single-tree recourse.

The runner writes only raw and aggregated experiment data under ``exp/``.
Paper figures are created by ``analyze_recourse_benchmark.py`` from saved CSV files.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import psutil
import scipy
import sklearn
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.datasets import load_breast_cancer, load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import recourse_core as old


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("RECOURSE_DATA_ROOT", ROOT / "data"))
EXP_ROOT = ROOT / "exp" / "recourse_benchmark"
DEFAULT_SEEDS = (7, 13, 19, 29, 37, 41, 53, 61, 73, 89)
EPS = 1e-8


@dataclass(frozen=True)
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    immutable_names: tuple[str, ...]
    domain: str


@dataclass(frozen=True)
class Protocol:
    name: str
    use_immutable: bool
    max_changes: int | None
    metric: str


PROTOCOLS = (
    Protocol("free_l2", False, None, "scaled_l2"),
    Protocol("immutable_l2", True, None, "scaled_l2"),
    Protocol("immutable_top3_l1", True, 3, "weighted_l1"),
)
METHODS = (
    "nn_positive_train",
    "leaf_centroid",
    "raw_leaf_projection",
    "clbr",
    "metric_leaf_projection",
    "metric_clbr",
    "exact_leaf",
    "exact_leaf_cached",
    "native_exact_k_sparse",
    "clbr_uncached",
)


def _subsample(X: np.ndarray, y: np.ndarray, seed: int, limit: int = 6000) -> tuple[np.ndarray, np.ndarray]:
    if len(y) <= limit:
        return X, y
    keep, _ = train_test_split(
        np.arange(len(y)), train_size=limit, stratify=y, random_state=seed
    )
    keep = np.sort(keep)
    return X[keep], y[keep]


def load_dataset(name: str, seed: int) -> Dataset:
    if name == "breast_cancer":
        ds = load_breast_cancer()
        return Dataset(name, ds.data.astype(float), ds.target.astype(int), tuple(ds.feature_names),
                       ("mean radius", "mean texture", "mean perimeter"), "diagnostic benchmark")
    if name == "wine_binary":
        ds = load_wine()
        return Dataset(name, ds.data.astype(float), (ds.target != 0).astype(int), tuple(ds.feature_names),
                       ("proline",), "product recognition")
    if name == "credit_default":
        # Use the catalogued local OpenML parquet directly; never trigger a download.
        path = DATA_ROOT / "openml" / "org" / "openml" / "www" / "datasets" / "42477" / "dataset_42477.pq"
        frame = pd.read_parquet(path)
        cols = [f"x{i}" for i in range(1, 24)]
        readable = ("limit_bal", "sex", "education", "marriage", "age", "pay_0", "pay_2", "pay_3",
                    "pay_4", "pay_5", "pay_6", "bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4",
                    "bill_amt5", "bill_amt6", "pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5",
                    "pay_amt6")
        X = frame.loc[:, cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
        y = (pd.to_numeric(frame["y"], errors="coerce").fillna(1).to_numpy(int) == 0).astype(int)
        X, y = _subsample(X, y, seed)
        return Dataset(name, X, y, readable,
                       ("sex", "education", "marriage", "age"), "credit risk")
    if name == "adult_income":
        cols = ["age", "workclass", "fnlwgt", "education", "education_num", "marital",
                "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
                "hours_per_week", "country", "income"]
        frame = pd.read_csv(DATA_ROOT / "adult" / "adult.data", names=cols, skipinitialspace=True)
        features = ("age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week")
        X = frame.loc[:, features].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
        y = frame["income"].astype(str).str.contains(">50K").astype(int).to_numpy()
        X, y = _subsample(X, y, seed)
        return Dataset(name, X, y, features, ("age", "fnlwgt"), "income and employment")
    if name == "german_credit":
        frame = pd.read_csv(DATA_ROOT / "german" / "german.data", sep=r"\s+", header=None)
        idx = (1, 4, 7, 10, 12, 15, 17)
        features = ("duration", "credit_amount", "installment_rate", "residence_years",
                    "age", "existing_credits", "dependents")
        X = frame.loc[:, idx].to_numpy(float)
        y = (frame.iloc[:, 20].to_numpy(int) == 1).astype(int)
        return Dataset(name, X, y, features, ("age", "residence_years", "dependents"), "credit approval")
    if name == "bank_marketing":
        frame = pd.read_csv(DATA_ROOT / "bank" / "bank-full.csv", sep=";")
        features = ("age", "balance", "day", "duration", "campaign", "pdays", "previous")
        X = frame.loc[:, features].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
        y = (frame["y"].astype(str) == "yes").astype(int).to_numpy()
        X, y = _subsample(X, y, seed)
        return Dataset(name, X, y, features, ("age", "day", "previous"), "marketing intervention")
    if name == "compas":
        frame = pd.read_csv(DATA_ROOT / "compas" / "compas-scores-two-years.csv")
        features = ("age", "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count",
                    "days_b_screening_arrest", "start", "end")
        num = frame.loc[:, features].apply(pd.to_numeric, errors="coerce")
        keep = num.notna().all(axis=1) & frame["two_year_recid"].notna()
        X = num.loc[keep].to_numpy(float)
        y = (frame.loc[keep, "two_year_recid"].to_numpy(int) == 0).astype(int)
        X, y = _subsample(X, y, seed)
        return Dataset(name, X, y, features,
                       ("age", "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count"),
                       "recidivism-risk audit")
    if name == "synthetic_overlap":
        X, y = make_classification(n_samples=1200, n_features=16, n_informative=10,
                                   n_redundant=3, class_sep=0.75, flip_y=0.08,
                                   weights=[0.45, 0.55], random_state=seed)
        features = tuple(f"feature_{i + 1}" for i in range(X.shape[1]))
        return Dataset(name, X.astype(float), y.astype(int), features, features[:4], "synthetic stress test")
    raise ValueError(f"Unknown dataset: {name}")


DATASET_NAMES = (
    "breast_cancer", "wine_binary", "credit_default", "adult_income",
    "german_credit", "bank_marketing", "compas", "synthetic_overlap",
)


def immutable_indices(ds: Dataset, protocol: Protocol) -> np.ndarray:
    if not protocol.use_immutable:
        return np.array([], dtype=int)
    wanted = set(ds.immutable_names)
    return np.array([i for i, n in enumerate(ds.feature_names) if n in wanted], dtype=int)


def implementable_bound_matrices(
    boxes: dict,
    positive_leaves: Sequence[int],
    n_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed float32 boxes matching sklearn's implementation-level routing."""
    lbs = np.full((len(positive_leaves), n_features), -np.inf, dtype=np.float32)
    ubs = np.full((len(positive_leaves), n_features), np.inf, dtype=np.float32)
    for row, leaf in enumerate(positive_leaves):
        for j, value in boxes[leaf]["lb"].items():
            bound = np.float32(value)
            if float(bound) <= float(value):
                bound = np.nextafter(bound, np.float32(math.inf), dtype=np.float32)
            lbs[row, int(j)] = bound
        for j, value in boxes[leaf]["ub"].items():
            bound = np.float32(value)
            if float(bound) > float(value):
                bound = np.nextafter(bound, np.float32(-math.inf), dtype=np.float32)
            ubs[row, int(j)] = bound
    if np.any(lbs > ubs):
        raise AssertionError("An implementation-level leaf box has no attainable float32 point")
    return lbs, ubs


def _all_leaf_projections(x0: np.ndarray, leaf_lbs: np.ndarray, leaf_ubs: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x0.astype(np.float32, copy=False), leaf_lbs), leaf_ubs)


def select_projection(
    x0: np.ndarray,
    leaf_lbs: np.ndarray,
    leaf_ubs: np.ndarray,
    scale: np.ndarray,
    metric: str,
    matched_metric: bool,
) -> np.ndarray | None:
    if len(leaf_lbs) == 0:
        return None
    candidates = _all_leaf_projections(x0, leaf_lbs, leaf_ubs)
    if matched_metric:
        delta = (candidates.astype(np.float64) - x0.astype(np.float64)) / scale
        costs = np.sum(np.abs(delta), axis=1) if metric == "weighted_l1" else np.linalg.norm(delta, axis=1)
    else:
        costs = np.linalg.norm(candidates.astype(np.float64) - x0.astype(np.float64), axis=1)
    return candidates[int(np.argmin(costs))].copy()


def select_projection_scalar(
    x0: np.ndarray,
    leaf_lbs: np.ndarray,
    leaf_ubs: np.ndarray,
    scale: np.ndarray,
    metric: str,
    matched_metric: bool,
) -> np.ndarray | None:
    best_cost = math.inf
    best: np.ndarray | None = None
    for lb, ub in zip(leaf_lbs, leaf_ubs):
        candidate = np.minimum(np.maximum(x0, lb), ub).astype(np.float32, copy=False)
        if matched_metric:
            cost = old.scaled_cost(x0, candidate, scale, metric)
        else:
            cost = float(np.linalg.norm(candidate.astype(np.float64) - x0.astype(np.float64)))
        if cost < best_cost:
            best_cost, best = cost, candidate.copy()
    return best


def project_exact(
    x0: np.ndarray,
    clf: DecisionTreeClassifier,
    leaf_lbs: np.ndarray,
    leaf_ubs: np.ndarray,
    support: np.ndarray,
    scale: np.ndarray,
    metric: str,
) -> np.ndarray | None:
    """Minimum-cost positive leaf projection on a fixed actionable support."""
    best_cost = math.inf
    best: np.ndarray | None = None
    for raw in _all_leaf_projections(x0, leaf_lbs, leaf_ubs):
        candidate = old.apply_support(x0, raw, support)
        if candidate is None or int(clf.predict(candidate.reshape(1, -1))[0]) != 1:
            continue
        cost = old.scaled_cost(x0, candidate, scale, metric)
        if cost < best_cost:
            best_cost, best = cost, candidate
    return best


def project_exact_cached(
    x0: np.ndarray,
    clf: DecisionTreeClassifier,
    leaf_lbs: np.ndarray,
    leaf_ubs: np.ndarray,
    support: np.ndarray,
    scale: np.ndarray,
    metric: str,
    max_changes: int | None = None,
) -> np.ndarray | None:
    raw = _all_leaf_projections(x0, leaf_lbs, leaf_ubs)
    candidates = np.repeat(x0.reshape(1, -1), len(raw), axis=0)
    candidates[:, support] = raw[:, support]
    changed = np.abs(candidates.astype(np.float64) - x0.astype(np.float64)) > old.ACTION_EPS
    valid = clf.predict(candidates) == 1
    if max_changes is not None:
        valid &= changed.sum(axis=1) <= max_changes
    if not np.any(valid):
        return None
    delta = (candidates.astype(np.float64) - x0.astype(np.float64)) / scale
    costs = np.sum(np.abs(delta), axis=1) if metric == "weighted_l1" else np.linalg.norm(delta, axis=1)
    costs[~valid] = np.inf
    return candidates[int(np.argmin(costs))].copy()


def candidate_for(
    method: str,
    x0: np.ndarray,
    clf: DecisionTreeClassifier,
    Xtr: np.ndarray,
    pos_train: np.ndarray,
    leaf_ids: np.ndarray,
    boxes: dict,
    positive_leaves: Sequence[int],
    leaf_lbs: np.ndarray,
    leaf_ubs: np.ndarray,
    support: np.ndarray,
    actionable: np.ndarray,
    max_changes: int | None,
    nn_target: np.ndarray | None,
    scale: np.ndarray,
    metric: str,
    repair_steps: int,
) -> tuple[np.ndarray | None, bool, bool]:
    """Return final candidate, restricted-projection validity, repair success."""
    if method == "nn_positive_train":
        raw = old.nearest_positive_train(x0, pos_train)
    elif method == "leaf_centroid":
        raw = old.leaf_centroid_candidate(x0, Xtr, leaf_ids, positive_leaves)
    elif method == "exact_leaf":
        final = project_exact(x0, clf, leaf_lbs, leaf_ubs, support, scale, metric)
        valid = final is not None and int(clf.predict(final.reshape(1, -1))[0]) == 1
        return final, valid, False
    elif method == "exact_leaf_cached":
        final = project_exact_cached(x0, clf, leaf_lbs, leaf_ubs, support, scale, metric)
        valid = final is not None and int(clf.predict(final.reshape(1, -1))[0]) == 1
        return final, valid, False
    elif method == "native_exact_k_sparse":
        final = project_exact_cached(
            x0, clf, leaf_lbs, leaf_ubs, actionable, scale, metric, max_changes=max_changes
        )
        valid = final is not None and int(clf.predict(final.reshape(1, -1))[0]) == 1
        return final, valid, False
    elif method == "clbr_uncached":
        raw = select_projection_scalar(x0, leaf_lbs, leaf_ubs, scale, metric, matched_metric=False)
    elif method in {"metric_leaf_projection", "metric_clbr"}:
        raw = select_projection(x0, leaf_lbs, leaf_ubs, scale, metric, matched_metric=True)
    else:
        raw = select_projection(x0, leaf_lbs, leaf_ubs, scale, metric, matched_metric=False)

    restricted = old.apply_support(x0, raw, support)
    direct_valid = restricted is not None and int(clf.predict(restricted.reshape(1, -1))[0]) == 1
    if method not in {"clbr", "clbr_uncached", "metric_clbr"} or direct_valid:
        return restricted, direct_valid, False
    repaired = old.constrained_repair(clf, restricted, nn_target, support, repair_steps)
    repaired_valid = repaired is not None and int(clf.predict(repaired.reshape(1, -1))[0]) == 1
    return repaired, direct_valid, repaired_valid


def confidence_interval(values: Iterable[float], seed: int = 2026) -> tuple[float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return math.nan, math.nan
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(4000, len(arr)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_task(
    dataset_name: str,
    seed: int,
    depth: int,
    max_queries: int,
    repair_steps: int,
    shift_strengths: Sequence[float],
) -> tuple[list[dict], list[dict]]:
    ds = load_dataset(dataset_name, seed)
    split = train_test_split(np.arange(len(ds.y)), test_size=0.35, stratify=ds.y, random_state=seed)
    tr_idx, te_idx = split
    # sklearn tree inference is float32.  Keep the full recourse pipeline on
    # the same representable domain so costs cannot exploit hidden float64
    # values that change only after sklearn's internal conversion.
    Xtr, ytr = ds.X[tr_idx].astype(np.float32, copy=True), ds.y[tr_idx]
    base_Xte, yte = ds.X[te_idx].astype(np.float32, copy=True), ds.y[te_idx]
    scale = np.std(Xtr.astype(np.float64), axis=0)
    scale = np.where(scale > EPS, scale, 1.0)
    clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=5, random_state=seed)
    clf.fit(Xtr, ytr)
    boxes = old.get_leaf_boxes(clf)
    leaf_ids = clf.apply(Xtr)
    positive_leaves = [n for n in boxes if int(np.argmax(clf.tree_.value[n][0])) == 1]
    leaf_lbs, leaf_ubs = implementable_bound_matrices(boxes, positive_leaves, Xtr.shape[1])
    pos_train = Xtr[clf.predict(Xtr) == 1]
    agg_rows: list[dict] = []
    query_rows: list[dict] = []
    rng = np.random.default_rng(seed + 1009)

    for shift_strength in shift_strengths:
        Xte = base_Xte.copy()
        if shift_strength > 0:
            shift_count = min(3, Xte.shape[1])
            Xte[:, :shift_count] += float(shift_strength) * scale[:shift_count]
        neg_idx = np.where(clf.predict(Xte) == 0)[0]
        if len(neg_idx) > max_queries:
            neg_idx = np.sort(rng.choice(neg_idx, size=max_queries, replace=False))
        for protocol in PROTOCOLS:
            immutable = immutable_indices(ds, protocol)
            actionable = old.actionable_indices(Xtr.shape[1], immutable)
            collectors = {m: {"valid": 0, "direct": 0, "repair": 0, "cost": [], "changed": [], "time": 0.0}
                          for m in METHODS}
            for qi in neg_idx:
                x0 = Xte[int(qi)]
                nn = old.nearest_positive_train(x0, pos_train)
                support = old.choose_support(x0, nn, actionable, protocol.max_changes)
                nn_target = old.build_nn_target(x0, nn, support)
                cached_candidate: np.ndarray | None = None
                scalar_exact_candidate: np.ndarray | None = None
                for method in METHODS:
                    start = time.perf_counter()
                    candidate, direct_valid, repaired = candidate_for(
                        method, x0, clf, Xtr, pos_train, leaf_ids, boxes, positive_leaves,
                        leaf_lbs, leaf_ubs, support, actionable, protocol.max_changes,
                        nn_target, scale, protocol.metric, repair_steps,
                    )
                    elapsed = time.perf_counter() - start
                    valid = candidate is not None and int(clf.predict(candidate.reshape(1, -1))[0]) == 1
                    cost = old.scaled_cost(x0, candidate, scale, protocol.metric) if valid else math.inf
                    changed_idx = old.changed_feature_indices(x0, candidate) if valid else np.array([], dtype=int)
                    c = collectors[method]
                    c["time"] += elapsed
                    c["valid"] += int(valid)
                    c["direct"] += int(direct_valid)
                    c["repair"] += int(repaired)
                    if valid:
                        c["cost"].append(float(cost))
                        c["changed"].append(int(len(changed_idx)))
                    if method == "clbr":
                        cached_candidate = candidate
                    if method == "clbr_uncached":
                        same = ((cached_candidate is None and candidate is None) or
                                (cached_candidate is not None and candidate is not None and
                                 np.allclose(cached_candidate, candidate, rtol=1e-9, atol=1e-9)))
                        if not same:
                            raise AssertionError("Cached and uncached CLBR produced different candidates")
                    if method == "exact_leaf":
                        scalar_exact_candidate = candidate
                    if method == "exact_leaf_cached":
                        same = ((scalar_exact_candidate is None and candidate is None) or
                                (scalar_exact_candidate is not None and candidate is not None and
                                 np.array_equal(scalar_exact_candidate, candidate)))
                        if not same:
                            raise AssertionError("Scalar and cached exact search produced different candidates")
                    query_rows.append({
                        "dataset": ds.name, "domain": ds.domain, "seed": seed, "depth": depth,
                        "shift_strength": shift_strength, "constraint": protocol.name,
                        "query_idx": int(qi), "method": method, "valid": int(valid),
                        "direct_valid": int(direct_valid), "repaired_success": int(repaired),
                        "cost": cost, "changed_features": int(len(changed_idx)) if valid else "",
                        "support_size": int(len(support)), "runtime_sec": elapsed,
                        "n_positive_leaves": int(len(positive_leaves)),
                        "changed_feature_deltas": old.format_changed_features(
                            x0=x0, xcf=candidate if valid else None, feature_names=ds.feature_names
                        ),
                    })
            n = max(1, len(neg_idx))
            for method in METHODS:
                c = collectors[method]
                agg_rows.append({
                    "dataset": ds.name, "domain": ds.domain, "seed": seed, "depth": depth,
                    "shift_strength": shift_strength, "constraint": protocol.name, "method": method,
                    "n_queries": len(neg_idx), "coverage": c["valid"] / n,
                    "direct_valid_rate": c["direct"] / n,
                    "repair_recovery_share": c["repair"] / n,
                    "mean_cost_valid": float(np.mean(c["cost"])) if c["cost"] else math.inf,
                    "mean_changed_valid": float(np.mean(c["changed"])) if c["changed"] else math.nan,
                    "sparse_share_leq3": float(np.mean(np.asarray(c["changed"]) <= 3)) if c["changed"] else math.nan,
                    "runtime_total_sec": c["time"], "runtime_ms_per_query": 1000.0 * c["time"] / n,
                    "n_positive_leaves": int(len(positive_leaves)),
                })
    return agg_rows, query_rows


def summarize(rows: list[dict]) -> list[dict]:
    frame = pd.DataFrame(rows)
    keys = ["dataset", "shift_strength", "constraint", "method"]
    out: list[dict] = []
    for key, part in frame.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        row["n_seeds"] = int(part["seed"].nunique())
        for metric in ("coverage", "mean_cost_valid", "mean_changed_valid", "sparse_share_leq3", "runtime_ms_per_query"):
            vals = pd.to_numeric(part[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            lo, hi = confidence_interval(vals)
            row[f"{metric}_mean"] = float(np.mean(vals)) if len(vals) else math.nan
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
        out.append(row)
    return out


def paired_tests(rows: list[dict], reference: str = "exact_leaf") -> list[dict]:
    frame = pd.DataFrame(rows)
    tests: list[dict] = []
    for metric in ("coverage", "mean_cost_valid", "runtime_ms_per_query"):
        index = ["dataset", "seed", "shift_strength", "constraint"]
        wide = frame.pivot_table(index=index, columns="method", values=metric, aggfunc="first")
        for method in METHODS:
            if method in {reference, "clbr_uncached"} or method not in wide or reference not in wide:
                continue
            pair = wide[[method, reference]].replace([np.inf, -np.inf], np.nan).dropna()
            delta = pair[method] - pair[reference]
            if len(delta) == 0:
                continue
            try:
                p = float(wilcoxon(delta, zero_method="zsplit", alternative="two-sided").pvalue)
            except ValueError:
                p = 1.0
            tests.append({"metric": metric, "method": method, "reference": reference,
                          "n_pairs": len(delta), "mean_delta": float(delta.mean()),
                          "median_delta": float(delta.median()), "wilcoxon_p": p})
    # Holm correction across all reported tests.
    order = np.argsort([r["wilcoxon_p"] for r in tests])
    adjusted = np.ones(len(tests))
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (len(tests) - rank) * tests[idx]["wilcoxon_p"])
        running = max(running, value)
        adjusted[idx] = running
    for row, adj in zip(tests, adjusted):
        row["holm_p"] = float(adj)
    return tests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=list(DATASET_NAMES))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--max-queries", type=int, default=200)
    parser.add_argument("--repair-steps", type=int, default=8)
    parser.add_argument("--shift-strengths", nargs="*", type=float, default=[0.0, 1.0])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    slug = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + (f"-{args.tag}" if args.tag else "")
    out_dir = EXP_ROOT / slug
    out_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    tasks = [(d, s) for d in args.datasets for s in args.seeds]
    results = Parallel(n_jobs=max(1, args.workers), backend="loky", verbose=10)(
        delayed(run_task)(d, s, args.depth, args.max_queries, args.repair_steps, args.shift_strengths)
        for d, s in tasks
    )
    aggregate = [row for a, _ in results for row in a]
    queries = [row for _, q in results for row in q]
    summary = summarize(aggregate)
    tests = paired_tests(aggregate)
    write_csv(out_dir / "results.csv", aggregate)
    write_csv(out_dir / "query_metrics.csv", queries)
    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "paired_tests.csv", tests)
    metadata = {
        "generated_at": dt.datetime.now().isoformat(), "elapsed_sec": time.time() - started,
        "arguments": vars(args), "protocols": [asdict(p) for p in PROTOCOLS], "methods": list(METHODS),
        "python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor(),
        "logical_cpus": psutil.cpu_count(), "memory_gib": psutil.virtual_memory().total / 2**30,
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__, "joblib_workers": args.workers,
        "train_test_split": "stratified 65/35", "tree_min_samples_leaf": 5,
        "timeout_policy": "no per-query timeout; failures recorded as invalid/infinite conditional cost",
        "timing_scope": "candidate generation only; excludes model fitting and dataset loading",
        "cache_equivalence": "asserted per query with rtol=atol=1e-9",
        "peak_process_rss_mib": psutil.Process(os.getpid()).memory_info().rss / 2**20,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out_dir / "RUN_COMPLETE").write_text("ok\n", encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
