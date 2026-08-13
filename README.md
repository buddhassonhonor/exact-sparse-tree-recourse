# Exact Sparse Tree Recourse

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21911118.svg)](https://doi.org/10.5281/zenodo.21911118)

Public artifact for *Auditable Recourse for Decision Trees under Actionability Constraints: From Projection-and-Repair to Exact Sparse Leaf Search*.

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

The immutable v1.0.1 artifact is archived at Zenodo: [10.5281/zenodo.21911118](https://doi.org/10.5281/zenodo.21911118). Citation metadata are provided in `CITATION.cff`.
