#!/usr/bin/env python
"""Dataset-level effects, hierarchical intervals, and fair latency summaries."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "result"
PAIR = ("clbr", "exact_leaf")


def hierarchical_ci(frame: pd.DataFrame, value: str, n_boot: int = 4000) -> tuple[float, float]:
    rng = np.random.default_rng(2026)
    datasets = frame.dataset.unique()
    draws = np.empty(n_boot)
    grouped = {name: part[value].dropna().to_numpy(float) for name, part in frame.groupby("dataset")}
    for b in range(n_boot):
        sampled_datasets = rng.choice(datasets, len(datasets), replace=True)
        dataset_means = []
        for name in sampled_datasets:
            values = grouped[name]
            dataset_means.append(rng.choice(values, len(values), replace=True).mean())
        draws[b] = np.mean(dataset_means)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    raw = pd.read_csv(args.run_dir / "results.csv")
    for col in ("coverage", "mean_cost_valid", "mean_changed_valid", "runtime_ms_per_query"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    method_summary = raw.groupby(["constraint", "method"], as_index=False).agg(
        n_units=("coverage", "size"), coverage=("coverage", "mean"),
        cost=("mean_cost_valid", "mean"), changed=("mean_changed_valid", "mean"),
        latency_ms=("runtime_ms_per_query", "mean"),
    )

    index = ["dataset", "seed", "shift_strength", "constraint"]
    wide = raw[raw.method.isin(PAIR)].pivot(index=index, columns="method",
                                            values=["coverage", "mean_cost_valid", "mean_changed_valid"])
    paired = wide.reset_index()
    paired["coverage_gain"] = paired[("coverage", "exact_leaf")] - paired[("coverage", "clbr")]
    paired["cost_saving"] = paired[("mean_cost_valid", "clbr")] - paired[("mean_cost_valid", "exact_leaf")]
    paired["changed_reduction"] = paired[("mean_changed_valid", "clbr")] - paired[("mean_changed_valid", "exact_leaf")]
    paired.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in paired.columns]

    effects = paired.groupby(["dataset", "constraint"], as_index=False).agg(
        coverage_gain=("coverage_gain", "mean"), cost_saving=("cost_saving", "mean"),
        changed_reduction=("changed_reduction", "mean"),
    )
    wtl_rows = []
    for protocol, part in effects.groupby("constraint"):
        for metric in ("coverage_gain", "cost_saving", "changed_reduction"):
            values = part[metric].dropna().to_numpy()
            wtl_rows.append({"protocol": protocol, "metric": metric,
                             "win": int((values > 1e-12).sum()),
                             "tie": int((np.abs(values) <= 1e-12).sum()),
                             "loss": int((values < -1e-12).sum()),
                             "mean_dataset_effect": float(values.mean())})
    wtl = pd.DataFrame(wtl_rows)

    boot_rows = []
    for protocol, part in paired.groupby("constraint"):
        for metric in ("coverage_gain", "cost_saving", "changed_reduction"):
            finite = part[["dataset", metric]].dropna()
            lo, hi = hierarchical_ci(finite, metric)
            values = finite[metric].to_numpy()
            boot_rows.append({"protocol": protocol, "metric": metric,
                              "mean_effect": float(values.mean()), "ci_low": lo, "ci_high": hi,
                              "paired_effect_dz": float(values.mean() / values.std(ddof=1))
                              if values.std(ddof=1) > 0 else 0.0})
    boot = pd.DataFrame(boot_rows)

    latency_names = {"clbr_uncached": "Scalar Repair", "clbr": "Cached Repair",
                     "exact_leaf": "Scalar Exact", "exact_leaf_cached": "Cached Exact"}
    latency = raw[raw.method.isin(latency_names)].copy()
    latency["implementation"] = latency.method.map(latency_names)
    latency_2x2 = latency.groupby("implementation", as_index=False).agg(
        latency_ms=("runtime_ms_per_query", "mean"), n_units=("runtime_ms_per_query", "size")
    )
    latency_by_l = latency.groupby(["implementation", "n_positive_leaves"], as_index=False).agg(
        latency_ms=("runtime_ms_per_query", "mean"), n_units=("runtime_ms_per_query", "size")
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    method_summary.to_csv(RESULT_DIR / "priority_method_summary.csv", index=False)
    effects.to_csv(RESULT_DIR / "dataset_effects.csv", index=False)
    wtl.to_csv(RESULT_DIR / "dataset_win_tie_loss.csv", index=False)
    boot.to_csv(RESULT_DIR / "hierarchical_bootstrap.csv", index=False)
    latency_2x2.to_csv(RESULT_DIR / "latency_2x2.csv", index=False)
    latency_by_l.to_csv(RESULT_DIR / "latency_by_positive_leaves.csv", index=False)
    print(method_summary.to_string(index=False))
    print("\n", wtl.to_string(index=False))
    print("\n", boot.to_string(index=False))
    print("\n", latency_2x2.to_string(index=False))


if __name__ == "__main__":
    main()
