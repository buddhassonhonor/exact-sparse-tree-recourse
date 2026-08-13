#!/usr/bin/env python
"""Create paper-ready summaries and figures from saved benchmark CSVs."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = ROOT / "exp" / "recourse_benchmark"
FIG_ROOT = ROOT / "figures"
METHOD_LABELS = {
    "nn_positive_train": "Nearest positive",
    "leaf_centroid": "Leaf centroid",
    "raw_leaf_projection": "Boundary-aware projection",
    "clbr": "Projection + repair",
    "metric_clbr": "Metric-matched + repair",
    "exact_leaf": "Exact feasible leaf",
    "native_exact_k_sparse": "Native exact sparse",
}
COLORS = {
    "Nearest positive": "#4C78A8",
    "Leaf centroid": "#F58518",
    "Boundary-aware projection": "#B8B8B8",
    "Projection + repair": "#E45756",
    "Metric-matched + repair": "#B279A2",
    "Exact feasible leaf": "#54A24B",
    "Native exact sparse": "#72B7B2",
}


def ci(values: pd.Series, seed: int = 2026) -> tuple[float, float]:
    x = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if len(x) < 2:
        return (float(x[0]), float(x[0])) if len(x) else (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(4000, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(frame: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for key, part in frame.groupby(keys, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(keys, key))
        row["n_units"] = len(part)
        for metric in metrics:
            vals = pd.to_numeric(part[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            lo, hi = ci(vals)
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-run", required=True)
    parser.add_argument("--shift-run", required=True)
    parser.add_argument("--depth3-run", required=True)
    parser.add_argument("--depth7-run", required=True)
    args = parser.parse_args()

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = EXP_ROOT / f"analysis-{stamp}"
    out_dir.mkdir(parents=True)
    def run_path(value: str) -> Path:
        supplied = Path(value)
        return supplied if supplied.exists() else EXP_ROOT / supplied

    main_df = pd.read_csv(run_path(args.main_run) / "results.csv")
    main_df = main_df[~main_df.method.isin(["clbr_uncached", "exact_leaf_cached"])].copy()
    main_df["method_label"] = main_df.method.map(METHOD_LABELS)
    main_summary = summarize(
        main_df, ["constraint", "method", "method_label"],
        ["coverage", "mean_cost_valid", "mean_changed_valid", "runtime_ms_per_query"],
    )
    main_summary.to_csv(out_dir / "main_summary.csv", index=False)
    dataset_summary = summarize(
        main_df, ["dataset", "method", "method_label"],
        ["coverage", "mean_cost_valid", "runtime_ms_per_query"],
    )
    dataset_summary.to_csv(out_dir / "dataset_summary.csv", index=False)

    shift_df = pd.read_csv(run_path(args.shift_run) / "results.csv")
    shift_df = shift_df[shift_df.method.isin(["clbr", "exact_leaf"])].copy()
    shift_df["method_label"] = shift_df.method.map(METHOD_LABELS)
    shift_summary = summarize(
        shift_df, ["shift_strength", "method", "method_label"],
        ["coverage", "mean_cost_valid", "runtime_ms_per_query"],
    )
    shift_summary.to_csv(out_dir / "shift_summary.csv", index=False)

    depths = []
    for depth, run in [(3, args.depth3_run), (5, args.main_run), (7, args.depth7_run)]:
        frame = pd.read_csv(run_path(run) / "results.csv")
        frame = frame[(frame.shift_strength == 0) & frame.method.isin(["clbr", "exact_leaf"])].copy()
        frame["tree_depth"] = depth
        frame["method_label"] = frame.method.map(METHOD_LABELS)
        depths.append(frame)
    depth_df = pd.concat(depths, ignore_index=True)
    depth_summary = summarize(
        depth_df, ["tree_depth", "method", "method_label"],
        ["coverage", "mean_cost_valid", "runtime_ms_per_query"],
    )
    depth_summary.to_csv(out_dir / "depth_summary.csv", index=False)

    plt.rcParams.update({
        "font.size": 18, "axes.titlesize": 18, "axes.labelsize": 18,
        "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16,
    })
    constraints = ["free_l2", "immutable_l2", "immutable_top3_l1"]
    constraint_labels = ["Free", "Immutable", "Immutable + top-3"]
    methods = ["raw_leaf_projection", "clbr", "metric_clbr", "exact_leaf", "native_exact_k_sparse"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(len(constraints))
    width = 0.16
    for offset, method in enumerate(methods):
        part = main_summary[main_summary.method == method].set_index("constraint").reindex(constraints)
        label = METHOD_LABELS[method]
        axes[0].bar(x + (offset - 2) * width, part.coverage_mean, width, label=label, color=COLORS[label])
        axes[1].bar(x + (offset - 2) * width, part.mean_cost_valid_mean, width, label=label, color=COLORS[label])
    axes[0].set_ylabel("Coverage")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Valid recourse coverage")
    axes[1].set_ylabel("Conditional action cost")
    axes[1].set_title("Cost among valid recommendations")
    for ax in axes:
        ax.set_xticks(x, constraint_labels, rotation=12)
        ax.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(FIG_ROOT / f"recourse_quality_tradeoff.{suffix}", dpi=100, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for method in ["clbr", "exact_leaf"]:
        label = METHOD_LABELS[method]
        part = shift_summary[shift_summary.method == method].sort_values("shift_strength")
        axes[0].plot(part.shift_strength, part.coverage_mean, marker="o", linewidth=2.5,
                     label=label, color=COLORS[label])
        part = depth_summary[depth_summary.method == method].sort_values("tree_depth")
        axes[1].plot(part.tree_depth, part.coverage_mean, marker="o", linewidth=2.5,
                     label=label, color=COLORS[label])
    axes[0].set(xlabel="Shift strength (training SD)", ylabel="Coverage", title="Covariate-shift sensitivity")
    axes[1].set(xlabel="Maximum tree depth", ylabel="Coverage", title="Tree-depth sensitivity")
    axes[1].set_xticks([3, 5, 7])
    for ax in axes:
        ax.set_ylim(0.55, 0.95)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIG_ROOT / f"recourse_sensitivity.{suffix}", dpi=100, bbox_inches="tight")
    plt.close(fig)

    provenance = {"main_run": args.main_run, "shift_run": args.shift_run,
                  "depth3_run": args.depth3_run, "depth7_run": args.depth7_run,
                  "figures": ["recourse_quality_tradeoff", "recourse_sensitivity"]}
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
