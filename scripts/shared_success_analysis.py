#!/usr/bin/env python
"""Audit CLBR and exact feasible-leaf search on identical query keys.

The input is the raw ``query_metrics.csv`` produced by
``run_recourse_benchmark.py``.  The detail output contains one row per query and
the summary reports dominance counts on the jointly successful subset.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "results" / "main"
RESULT_DIR = ROOT / "result"
REPAIR_METHOD = "clbr"
EXACT_METHOD = "exact_leaf"
TOLERANCE = 1e-9
KEYS = ["dataset", "seed", "shift_strength", "constraint", "query_idx"]


def _load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = set(KEYS + ["method", "valid", "cost", "changed_features"])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    pair = frame[frame["method"].isin([REPAIR_METHOD, EXACT_METHOD])].copy()
    pair["valid"] = pd.to_numeric(pair["valid"], errors="raise").astype(bool)
    pair["cost"] = pd.to_numeric(pair["cost"], errors="coerce")
    duplicate = pair.duplicated(KEYS + ["method"], keep=False)
    if duplicate.any():
        raise AssertionError(
            "Method rows are not unique by query key:\n"
            + pair.loc[duplicate, KEYS + ["method"]].head(20).to_string(index=False)
        )
    return pair


def build_audit(frame: pd.DataFrame) -> pd.DataFrame:
    wide = frame.pivot(index=KEYS, columns="method", values=["valid", "cost", "changed_features"])
    expected = [(field, method) for field in ("valid", "cost", "changed_features")
                for method in (REPAIR_METHOD, EXACT_METHOD)]
    missing = [column for column in expected if column not in wide.columns]
    if missing:
        raise AssertionError(f"Incomplete method pairs: {missing}")
    out = wide.reset_index()
    out.columns = ["_".join(map(str, col)).rstrip("_") if isinstance(col, tuple) else str(col)
                   for col in out.columns]
    out = out.rename(columns={
        "shift_strength": "shift",
        "constraint": "protocol",
        "query_idx": "query_id",
        f"valid_{REPAIR_METHOD}": "repair_success",
        f"valid_{EXACT_METHOD}": "exact_success",
        f"cost_{REPAIR_METHOD}": "repair_cost",
        f"cost_{EXACT_METHOD}": "exact_cost",
        f"changed_features_{REPAIR_METHOD}": "repair_changed_features",
        f"changed_features_{EXACT_METHOD}": "exact_changed_features",
    })
    ordered = [
        "dataset", "seed", "shift", "protocol", "query_id",
        "repair_success", "exact_success", "repair_cost", "exact_cost",
        "repair_changed_features", "exact_changed_features",
    ]
    out = out[ordered].sort_values(ordered[:5]).reset_index(drop=True)
    out["repair_success"] = out["repair_success"].astype(bool)
    out["exact_success"] = out["exact_success"].astype(bool)
    return out


def summarise(audit: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    rows: list[dict] = []
    for protocol, group in audit.groupby("protocol", sort=True):
        shared = group[group.repair_success & group.exact_success].copy()
        diff = shared.repair_cost - shared.exact_cost
        violation = diff < -tolerance
        rows.append({
            "protocol": protocol,
            "n_queries": len(group),
            "n_repair_success": int(group.repair_success.sum()),
            "n_exact_success": int(group.exact_success.sum()),
            "n_shared": len(shared),
            "n_exact_strictly_better": int((diff > tolerance).sum()),
            "n_equal": int((diff.abs() <= tolerance).sum()),
            "n_violation": int(violation.sum()),
            "mean_exact_shared": float(shared.exact_cost.mean()),
            "mean_repair_shared": float(shared.repair_cost.mean()),
            "mean_paired_saving": float(diff.mean()),
            "median_paired_saving": float(diff.median()),
            "strict_improvement_share": float((diff > tolerance).mean()),
            "paired_effect_dz": float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0,
        })
        if violation.any():
            bad = shared.loc[violation, ordered_audit_columns()].head(20)
            raise AssertionError(
                f"Exact-cost dominance violated for {protocol} (tolerance={tolerance:g}):\n"
                + bad.to_string(index=False)
            )
        if protocol == "free_l2" and len(shared) != len(group):
            raise AssertionError(
                f"Free protocol must share every query: n_shared={len(shared)}, n_queries={len(group)}"
            )
    return pd.DataFrame(rows)


def ordered_audit_columns() -> list[str]:
    return [
        "dataset", "seed", "shift", "protocol", "query_id",
        "repair_success", "exact_success", "repair_cost", "exact_cost",
        "repair_changed_features", "exact_changed_features",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = parser.parse_args()
    source = args.run_dir / "query_metrics.csv"
    if not source.exists():
        raise FileNotFoundError(source)

    audit = build_audit(_load(source))
    summary = summarise(audit, args.tolerance)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(RESULT_DIR / "shared_success_detail.csv", index=False)
    summary.to_csv(RESULT_DIR / "shared_success_summary.csv", index=False)
    print(f"source={source}")
    print(summary.to_string(index=False))
    print("dominance_assertion=PASS")


if __name__ == "__main__":
    main()
