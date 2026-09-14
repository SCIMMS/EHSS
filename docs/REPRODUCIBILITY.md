# Reproduce the archived evidence

## 1. Verify this checkout

Run `python tools/verify_repository.py` from the repository root. It checks all
manifested files, including the immutable snapshot ZIP and numerical source.
Git line-ending conversion is disabled to preserve the recorded bytes.

## 2. Extract the immutable snapshot

```sh
python tools/prepare_evidence.py --output work/evidence
python -m pip install -r requirements-replay.txt
cd work/evidence
python verify_snapshot.py
```

The extraction helper requires a new output directory, validates member paths and
the original snapshot SHA256, then checks every extracted manifest entry. The
`work/` tree is ignored by Git. A standalone ZIP extraction is also possible.

## 3. Recompute statistics from recorded numerical results

From `work/evidence`, after the integrity check:

```sh
python scripts/analyze_imos113_stage_c.py
python scripts/analyze_ehss_native_ablation.py
python scripts/analyze_ehss_panel_own_holdout.py
python scripts/audit_ehss_panel_references.py
python scripts/analyze_imos113_stage_d.py
python scripts/analyze_imos113_stage_e.py
python scripts/audit_imos113_stage_f_completion.py
python scripts/analyze_imos113_stage_g.py
python scripts/analyze_ehss_sampling_convergence.py
```

These commands recompute summaries from archived data without a new scattering
simulation or vendor run. They may rewrite generated files inside the extracted
copy. The original offline validation (`data/evidence/validation_v4.json`) passed
integrity verification plus all nine commands; six principal JSON summaries
matched their originals exactly. It used the existing Windows dependency
environment, not a fresh OS. The historical pinned environment is retained for
reference, separately from the minimal CLI dependencies.

Only these offline workflows are validated as a self-contained analysis closure.
The archive also retains historical scripts, failed runs and execution launchers;
do not interpret their inclusion as an instruction to rerun them. Numerical IMoS
drag-tensor output was unavailable and is an accepted comparison limitation.

## 4. Check the executable distribution

From the repository root:

```sh
python -m pip install -e ".[test]"
python -m pytest
```

The six supplied core modules cover 72 previously recorded cases: analytical
sphere scattering, nearest collisions, trapped rays and caps, rigid-transform
covariance, native flat/BVH agreement, thread/backend checks and CLI validation.
These software checks are distinct from new benchmark replicates. In this
repository's distribution check, 71 passed and one optional C++/MSVC comparison
was skipped because its locally compiled DLL is deliberately not distributed.
The active Python/Numba path and CLI checks passed. The historical 72/72 receipt
used a locally built comparison DLL; it is not relabeled as the current result.

For exact publication sampling, consult the archived Stage G frozen selection,
worker, source hashes and per-run specifications. `data/benchmark/stage_g_final_rows.json`
contains 720 independent final estimates. `timing_blocks.csv` contains five paired
time values for each of 18 single cases and the actual 18-input batch. Timing
replays must not be added to the 20 independent final seeds.

## Data guide

- `final_accuracy.csv`: 36 method/structure cells, reference-relative RMSE, bias,
  standard deviation, pointwise run/reference bootstrap intervals and status.
- `process_summary.csv`: complete-process medians and ratios; `target_qualified`
  distinguishes the 15 jointly passing single cases from uncertain conditions.
- `timing_blocks.csv`: all five timing values per method/condition.
- `internal_components.csv`: internal timers from 20 accuracy runs; do not divide
  unlike timer scopes or add component medians to reconstruct a total median.
- `collision_tails.csv`: sampled bounce distribution and residual bounds; long-path
  contribution is not the error of truncating at that collision count.
- `frozen_selection.json`, `reference_summary.json`, `input_provenance.json`:
  calibration choices, finite-reference checks and exact coordinate provenance.
- `earlier_own_holdout.json`: stricter-target outcomes retained without relabeling.
- `sampling_convergence.json`: the archived finite-grid analysis with its original
  cohort exclusions; no universal asymptotic convergence order is claimed.

The local machine paths in archival metadata are provenance, not required install
locations. Analyses resolve their working root from the extracted scripts.
