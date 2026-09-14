# EHSS offline evidence snapshot

This snapshot retains source, molecular inputs and numerical records supporting the
September 13-14, 2026 development results. It is not a finished journal submission.
Stage D/E/F calibration is complete (1,056 estimates). The new practical 2% Stage G
cohort is complete: 720 independent final records, 360 separate timing replays,
five blocks, and 15/18 structures jointly passing 2%. All uncertain outcomes are
retained. An all-18 batch latency ratio is not an all-18 target-qualified speedup.
Vendor executables, MATLAB Runtime and compiled/JIT caches are excluded.

## Supported offline checks

Extract into a new writable directory. With Python and the dependencies in
requirements-publication-replay.txt installed, run from the extracted directory:

    python verify_snapshot.py
    python scripts/analyze_imos113_stage_c.py
    python scripts/analyze_ehss_native_ablation.py
    python scripts/analyze_ehss_panel_own_holdout.py
    python scripts/audit_ehss_panel_references.py
    python scripts/analyze_imos113_stage_d.py
    python scripts/analyze_imos113_stage_e.py
    python scripts/audit_imos113_stage_f_completion.py
    python scripts/analyze_imos113_stage_g.py
    python scripts/analyze_ehss_sampling_convergence.py
    python ccs.py --help

The analysis commands recompute statistics from archived numerical records; they
do not rerun vendor simulations. They may rewrite generated summaries/notes in
the extracted copy. Run integrity verification before these commands.

## Scientific and performance scope

Use the data ledger and Stage C results for exact sample counts, timing boundaries
and limitations. Cross-host replays and single/batch replays are intentionally
reused seeds and must not be pooled as new independent accuracy samples. The two
computers' times must not be divided to claim speedup. Failed/uncertain outcomes
and the original failed input-frame deployment attempt remain in the record.

The generic XYZ CLI differs from the frozen publication protocol: its repeat seed
stride is 104729; the publication worker pools three Sobol blocks with stride
10007 and applies a sampled-tail cap extension rule. Use the archived worker,
specifications and source hashes to reproduce the publication sampling protocol.
No edits to the frozen scientific source were made for this snapshot.

The snapshot contains historical experimental scripts for provenance. Only the
offline commands listed above are validated as a self-contained dependency set;
not every historical experiment or optional C++ build is a turnkey workflow.
IMoS runs require a separately obtained licensed/vendor installation and runtime.
No claim of an independently installed clean dependency environment is made:
validation uses the existing local Python environment with imports restricted to
the extracted source copy. See the adjacent validation JSON for actual checks.
