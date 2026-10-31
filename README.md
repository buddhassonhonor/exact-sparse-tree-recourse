# Exact Sparse Tree Recourse


Public artifact for *Diagnosing Projection-and-Repair for Actionability-Constrained Decision-Tree Recourse*.

The repository contains the complete single-tree recourse benchmark, binary32 boundary implementation, metric cross-check, actionability policy, statistical analyses, figure generator, aggregate results, and compressed per-query audit records.

## Install

Python 3.13.2 was used for the reported experiments.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## Data

Breast Cancer, Wine, and the synthetic benchmark are loaded or generated through scikit-learn. The remaining public datasets are not redistributed. Set `RECOURSE_DATA_ROOT` to a directory with this layout:

```text
adult/adult.data
bank/bank-full.csv
compas/compas-scores-two-years.csv
german/german.data
openml/org/openml/www/datasets/42477/dataset_42477.pq
```

Dataset URLs, target encodings, numeric features, immutable features, actionable features, top-k policy, and analysis sample sizes are recorded in `results/actionability_policy.csv` and the paper appendix.

## Reproduce

The main experiment command is recorded in `config/main.json`. A direct invocation is:

```bash
python scripts/mve/run_recourse_benchmark.py \
  --datasets breast_cancer wine_binary credit_default adult_income german_credit bank_marketing compas synthetic_overlap \
  --seeds 7 13 19 29 37 41 53 61 73 89 \
  --depth 5 --max-queries 200 --repair-steps 8 \
  --shift-strengths 0 1 --workers 4 --tag main
```

Run the metric cross-check and analyses with:

```bash
python scripts/metric_crosscheck.py --workers 4
python scripts/shared_success_analysis.py --run-dir results/main
python scripts/priority_analysis.py results/main
python scripts/support_sensitivity.py
```

`results/main/query_metrics.csv.zip` contains the per-query audit records. Extract it before rerunning analyses that consume `query_metrics.csv`.

The archive SHA-256 is `F03C3CBB202249F72638E6040F67CC896D616A568D285D4DA78D6A29B9B79BDA`.

Regenerate the paper figures from the aggregate result files with:

```bash
python scripts/mve/analyze_recourse_benchmark.py \
  --main-run results/main --shift-run results/shift \
  --depth3-run results/depth3 --depth7-run results/depth7
```

## Artifact contents

- `scripts/`: experiment, statistical, and figure code.
- `config/`: exact seeds, protocols, methods, and runtime configuration.
- `results/main/`: aggregate main results and compressed per-query records.
- `results/shift`, `results/depth3`, `results/depth7`: inputs for the sensitivity figure.
- `results/derived/`: shared-success, hierarchical-bootstrap, latency, support, metric, and actionability-policy tables.
- `figures/`: paper figures in PNG and PDF.
- `environment/`: captured software and hardware metadata.

The per-query data and generated results are released under CC BY 4.0; source code is released under the MIT License.

## Citation

The public artifact is archived on Zenodo under Concept DOI [10.5281/zenodo.21911117](https://doi.org/10.5281/zenodo.21911117) (resolving to the latest release). Citation metadata are provided in `CITATION.cff`.

## Version 1.1.0: controlled candidate-search diagnostics

This version distinguishes the original study-designed leaf-first project--restore--repair construction from complete path enumeration. Candidate budgets, support choice, restored-anchor filtering and precision can materially change coverage. In the shared 1,962-query subset per depth/protocol, filtering restored anchors before selection matches Exact coverage; Exact still provides the minimum-cost reference. Small dataset-cluster tests are reported without treating queries as independent datasets.

New results are in results/followup_v1_1_0. CSV files preserve dataset, seed, tree depth, support protocol and query identifiers. Large raw tables are gzip compressed; pandas.read_csv reads them directly. For the primary experiment there are 23,176 query-tree observations and three protocols. Mechanism studies use the first 25 queries from the same sampled, index-ordered lists; support/precision studies use the first 50. External comparisons use eight datasets, three seeds and 20 queries per tree.

To reproduce from provider datasets (RECOURSE_DATA_ROOT as above):

    python scripts/followup_study.py --mode study --workers 4
    python scripts/followup_study.py --mode dose --workers 4
    python scripts/followup_external.py
    python scripts/followup_precision.py

After those quality experiments finish, run timing alone:

    python scripts/followup_scaling.py
    python scripts/followup_analysis.py
    python scripts/followup_plot_scaling.py

The scripts write project-owned outputs under exp/followup_20261101. To analyze archived outputs without rerunning experiments, copy the CSVs to that folder and decompress the six large gzip tables there. Keep shared provider files separate from project-owned outputs.

DiCE uses the public generate_counterfactuals API. Genetic initialization has a two-second prediction-call wall-time bound to prevent unbounded random initialization; three timeout records are retained. Random search samples 2,000 points and genetic search has at most 100 iterations, each requesting five alternatives. Certified infeasible bounded supports are recorded as such. The constrained Growing Spheres implementation is an adaptation, not an official implementation claim. Its finite search uses 14 shells and 512 samples each.

Independent HiGHS checks use weighted L1 only on both support protocols, remove impossible leaf/range intersections, and compare optimization objectives to 1e-6. They do not independently validate the L2 solver. MILP tolerances are 1e-10, relative gap 1e-9, and time limit 10 seconds. Actual solver statuses and differences are in milp.csv.

The noise-tree scaling tests measure computation, not decision utility. Each fitted tree is measured at 20 distinct queries, with one warm-up and nine measured repeats in randomized method order. Timing includes support/anchor selection and excludes fitting and cached bound construction. The largest tree has 14,618 leaves and depth 59. Timing is hardware-dependent.

No manuscripts, response letters, review notes, API credentials or raw provider datasets are included in this artifact.

## Version 1.1.1 packaging correction

Adds scripts/mve/run_review_benchmark.py, the module imported by the follow-up experiment scripts. Scientific configurations and saved results are unchanged from v1.1.0. The public archive DOI is supplied by the archive landing page.
