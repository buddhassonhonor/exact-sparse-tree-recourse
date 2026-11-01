# Diagnosing LF-PRR Tree Recourse — v1.3.0

This release extends the core mechanisms, pool audits, controlled constraint doses and external comparisons to **24 datasets**. Five splits support primary quality, mechanism and dose analyses. External methods use three splits; the longer DiCE budget sensitivity uses one split. Source code is MIT, generated results CC BY 4.0; the vendored CARLA kernel retains its upstream MIT license. The associated manuscript is *Diagnosing LF-PRR Decision-Tree Recourse through Visited Leaves, Candidate Cost, and Action Constraints*.

## Reproduce v1.3.0

Install requirements and obtain provider datasets under their own access conditions. Set `RECOURSE_DATA_ROOT` to the provider cache described below. Run from the repository root:

    python scripts/evidence_mechanisms.py
    python scripts/evidence_dose.py
    python scripts/evidence_main.py
    python scripts/evidence_solver.py
    python scripts/evidence_external.py
    python scripts/evidence_convergence.py
    python scripts/evidence_regression.py

The core benchmark's depth-five coverage and leaf-rank observations are inherited from v1.2.0; regenerate them with `scripts/mechanism_breadth.py` if needed. Run latency after other quality jobs have finished:

    python scripts/evidence_polish_timing.py

To analyze the archived observations without regenerating models:

    python scripts/expand_mechanism_results.py
    python scripts/evidence_analysis.py

Regenerating statistics uses archived query outcomes, not full returned vectors or provider records. It requires no manuscript source. The separate `scripts/evidence_regression.py` refits the Laplace-integrated binomial random-dataset-intercept models with 300 whole-dataset bootstrap resamples. Scripts resume completed per-dataset output units when rerunning models. Old eight-dataset iteration/grid/direction analyses retain their own designs; they are not pooled into the new 24-dataset mechanisms.

## Interpretation and corrections

Depth-five within-leaf choice accounts for 95.4%/97.5% of LF-PRR20 excess cost under Immutable/top-3. On the primary common queries, support optimization improves coverage by 0.292, versus 0.052 for completing the LF-PRR20 search. Single-output polishing preserves coverage. Polishing each repaired output before selection still exceeds Exact in 8 of 14,883 paired successes; polishing every restored endpoint equals Exact under complete-pool/fixed-support assumptions. The 159,773 identical first-compatible/cost ranks and filter-first coverage are partition consequences verified by assertions, not statistical discoveries.

The epsilon-box implementation now rounds bounds inward to binary32 before clipping, correcting false rejections from real-valued inward faces. Compatible-first and post-restoration versions are labeled study adaptations, not reproductions of Tolomei's threshold assignment. LIRE is region-distance search; its constrained full-training single-tree specialization equals Exact. Raw training-point nearest neighbor is separate.

The CARLA row executes the upstream Growing Spheres kernel with a sklearn/range adapter, not the full CARLA framework. Provenance and MIT license are in `scripts/vendor`. Five independent GS searches per fixed query yielded no Exact-feasible misses in this sample; finite shell sampling has no completeness guarantee. DiCE uses native constraints and explicit unconstrained-to-restored wrappers. Distance alignment leaves a soft class loss; the separate validity-first control assigns invalid points a dominating penalty. Budget caps 2/10/30 s retain native plateau stopping, which often stops early. Population-batch candidates and final returns are monitored, but one-dimensional initialization probes are not; initialization-only timeouts remain failures. Five alternatives favor DiCE relative to a single return. These are budgeted comparisons, not optimal-convergence claims or fair external runtime rankings.

The independent raw-node reference has 1,152 L1 MILP checks at depths 3/5/11 plus 384 L2 MISOCP checks at depths 3/5: 1,182 optimal and 354 infeasible. Maximum objective/reconstructed-cost error is 1.08e-7; 49 feasible raw L2 points require path reconstruction before boundary/support validation. Solver building and solving times are reported separately. This is a node-flow encoding in the OCEAN family, not execution of OCEAN software.

The 55-tree timing observations are retained from v1.2.0, with new tree-by-stratum summaries. Global interval timing covers only 392 Breast Cancer queries across five trees. Cached Exact is often faster than LF-PRR; polishing is useful for existing outputs and is not presented as faster than exact tree optimization. TOST tolerances are two/three extra misses per 100 as sensitivity scenarios, not validated domain utilities. Index-based immutable policies have no clinical/business semantics. No causal, deployment, or human-outcome claims follow.

## Version history

v1.2.0 remains available at https://doi.org/10.6084/m9.figshare.34875666.v1 and v1.1.1 at https://doi.org/10.6084/m9.figshare.34875612.v1. The current release is identified by its GitHub tag and manifest; these older DOIs do not cover v1.3.0. Historic instructions below apply only to their named version.

---

# Diagnosing LF-PRR Tree Recourse

Current software release: **v1.2.0**, archived at [Figshare DOI 10.6084/m9.figshare.34875666.v1](https://doi.org/10.6084/m9.figshare.34875666.v1). Associated paper: *Diagnosing LF-PRR Decision-Tree Recourse through Visited Leaves, Candidate Cost, and Action Constraints*.

This release adds 24-dataset coverage/depth/budget comparisons, anchor completeness stress tests, paired segment/within-leaf/leaf-choice cost decomposition, leaf polishing and a constructed counterexample, constraint-pruned DFS, aligned DiCE proximity and path tweaking, raw-tree MILP/MISOCP, directed grids, feature-specific direction sensitivity and isolated 55-tree timings. Five seeds are used for the broadened quality study; inference clusters datasets. Earlier v1.1.1 code/results and DOI are retained as the baseline, not claimed to contain these additions.

## Reproduce v1.2.0

Install requirements, set RECOURSE_DATA_ROOT to provider data, then run:

    python scripts/mechanism_breadth.py
    python scripts/mechanism_decomposition.py
    python scripts/mechanism_external.py
    python scripts/mechanism_solver.py
    python scripts/mechanism_practical.py
    python scripts/mechanism_polish_counterexample.py

After all quality jobs finish, run timing alone:

    python scripts/mechanism_latency.py
    python scripts/mechanism_large_budgets.py

For analysis of archived observations without rerunning models:

    python scripts/expand_mechanism_results.py
    python scripts/mechanism_assets.py

The analysis script regenerates statistics/figures and optional LaTeX tables; manuscript source is not distributed. The historical iteration table is kept in the original archive and is not needed for the new numeric analyses. New output files are in exp/mechanism_20261101. Query outcomes are compressed CSVs; read with pandas directly or use the expansion helper. Actual external loss values and solver statuses are included. Provider records and returned full feature vectors are not redistributed.

The 16 additional OpenML IDs and frozen numeric columns/labels are in config/mechanism_datasets.json. The loader uses the same OpenML cache convention under RECOURSE_DATA_ROOT: openml/org/openml/www/datasets/ID/dataset_ID.pq. These public provider datasets must be obtained separately under their terms. Numeric missing values use training medians; the new-task frozen-coordinate policy is index based and diagnostic, not a validated domain intervention. Class 1 is the frozen model target; it is not automatically favorable clinically or financially.

Genetic DiCE's adapted loss is exactly the shared weighted L1/scaled L2 cost with diversity and sparsity penalties disabled; defaults are retained separately. A two-second prediction-call deadline yields six failures in the recorded comparison. MISOCP uses ECOS_BB and MILP uses HiGHS, encoded from raw nodes independently of cached boxes. Raw solver values can be invalid at one-ulp boundaries; the selected path is reconstructed and validated, with raw validity disclosed. Global interval timing has the explicit A*N+ <= 10,000 workspace gate; larger configurations are unmeasured. KD-tree query complexity can degrade in high dimension. Coordinate-storage counts are not peak process RSS.

The real-data direction policies are declared sensitivities; retrospective credit payment values cannot be edited as a real intervention. Independent-label noise-tree scaling measures computation only. Causal effects and human outcomes are not evaluated.

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

The baseline v1.1.1 artifact is archived on Figshare at [10.6084/m9.figshare.34875612.v1](https://doi.org/10.6084/m9.figshare.34875612.v1). Citation metadata identify the current release. Historical Zenodo archives remain historical and do not cover the current mechanism study.

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

## Published Figshare archive

Version v1.1.1 is publicly archived at [10.6084/m9.figshare.34875612.v1](https://doi.org/10.6084/m9.figshare.34875612.v1). This DOI covers the code and generated results in the v1.1.1 release. Original provider datasets are accessed separately. The GitHub repository is public for reviewing and reusing the artifact.
