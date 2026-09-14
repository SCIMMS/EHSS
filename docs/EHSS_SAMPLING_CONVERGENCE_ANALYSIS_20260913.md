# EHSS sampling and convergence: archived-data analysis

2026-09-13. No new ray tracing or vendor executions. Stage F is not included because no returned Stage F observations were available for this analysis.

## Assessment: share with caveats

The clearest supported result is reduced finite-sample scatter and RMSE, not a universal increase in convergence order. At 16,384 rays, changing only the native incident sampler from IID to scrambled Sobol reduces estimated RMSE by 2.12–6.11-fold across all six development molecules; each pointwise bootstrap ratio interval lies above one. This is an empirical result at this budget, not simultaneous inference or superiority at every budget. Some lower-budget comparisons cross.

In Stage A at 196,608 rays, IMoS/own RMSE ratios are 4.47 (1CRN; 95% interval 2.31–8.11) and 8.15 (1LYZ; 4.20–12.96). IMoS sample-variance fractions of MSE are 96.1% and 96.7%, and its run SD is 4.67 and 8.74 times the own SD. Thus the measured error gap is mainly run-to-run dispersion, rather than a large sample mean offset at these two cells. This is consistent with sampling/estimator efficiency, but cannot isolate the IMoS random-number generator as the sole cause.

The 1CRN external empirical exponents are similar (own 0.388; IMoS 0.356); its advantage is mostly an error-level difference over this grid. Native 1AON similarly has exponents 0.479 (Sobol) and 0.486 (IID) while preserving a 2.12-fold high-budget RMSE advantage. Other native cases show steeper Sobol point estimates, with overlapping or wide pointwise intervals in several cases. No universal N^-1 claim follows.

The shallow Stage A IMoS 1LYZ exponent (0.135) is based on three budgets and eight seeds per budget. Its interval includes zero and does not establish a persistent error floor. Across the separate new panel, median descriptive exponents are 0.443/0.469 for IMoS s0/s1 (18/17 arms) over D-to-E, versus 0.543 for the 18 own initial grids. The different budget ranges, adaptive vendor extension and two-point vendor fits prohibit a causal comparison of these medians. They do show why the 1LYZ initial curve should not represent all IMoS cases.

At the largest fitted panel budget, median variance fractions of MSE are 91.7%/92.1% for IMoS s0/s1 and 93.4% for own. These are molecule/arm descriptive medians, not pooled per-ray statistics. Nonzero sample mean discrepancies still occur for individual cells, including own estimates; the analysis does not establish unbiasedness or rule out persistent discrepancies.

## Design and estimands

Native: 6 development molecules, 4 fixed budgets (256–16,384), 8 independent runs per sampler/cell. BVH rows only; large-molecule cap4096 replays replace cap64 rows. Stage A: 2 molecules, 3 fixed budgets (12,288–196,608); own 16 independent pooled runs and 8 per IMoS Simplify arm. Panel: 18 initial own grids (432 runs), vendor D/E (592 runs); no holdouts or adaptive own extensions pooled.

For relative errors e in percent, MSE = mean(e²) = mean(e)² + var(e, ddof=0). Reported SD uses ddof=1. Sample bias is not persistent estimator bias. Reference means are resampled once per bootstrap draw and shared across budgets/methods. Independent runs are resampled within cells; matched vendor Simplify seeds share indices. 20,000 bootstrap draws yield pointwise percentile intervals; no simultaneous or confirmatory hypothesis tests. Own Stage A pooling means 16 separate estimates at N rays each; they are not averaged into a single 16N-ray estimate. Simplify arms are never pooled as independent observations.

The empirical exponent alpha is the unweighted log–log slope in RMSE ~ N^(-alpha). It describes this finite grid, not an asymptotic order. SD slopes are reported separately to expose reference/mean-discrepancy effects. Vendor panel slopes use only the common D-to-E interval 24,576 to 98,304; the four downward E cells remain in the data table.

## Native control: sampler effect

| Molecule | IID alpha [95%] | Sobol alpha [95%] | RMSE ratio at 16,384, IID/Sobol [95%] | SD ratio |
|---|---|---|---|---:|
| 1CRN | 0.475 [0.276, 0.639] | 0.723 [0.562, 0.856] | 4.392 [2.135, 7.476] | 3.991 |
| 1BNA | 0.534 [0.381, 0.702] | 0.724 [0.587, 0.929] | 2.490 [1.168, 6.614] | 2.284 |
| 1LYZ | 0.460 [0.354, 0.590] | 0.600 [0.427, 0.720] | 6.110 [3.347, 9.830] | 6.175 |
| 4AKE | 0.351 [0.210, 0.511] | 0.583 [0.416, 0.694] | 2.619 [1.362, 4.272] | 2.584 |
| 1IGT | 0.432 [0.329, 0.537] | 0.714 [0.545, 0.838] | 2.451 [1.575, 4.060] | 2.596 |
| 1AON | 0.486 [0.364, 0.617] | 0.479 [0.228, 0.651] | 2.125 [1.165, 4.204] | 1.826 |

![Native convergence](../output/ehss_sampling_convergence_20260913/native_convergence.png)

## Stage A: external fixed-grid comparison

| Molecule | Method | RMSE alpha [95%] | SD alpha [95%] | High-budget RMSE (%) | Bias (%) | SD (%) | Variance/MSE (%) |
|---|---|---|---|---:|---:|---:|---:|
| 1CRN | imos_s0 | 0.356 [0.164, 0.586] | 0.349 [0.075, 0.600] | 0.651 | -0.128 | 0.682 | 96.1 |
| 1CRN | imos_s1 | 0.356 [0.164, 0.586] | 0.349 [0.075, 0.600] | 0.651 | -0.128 | 0.682 | 96.1 |
| 1CRN | own_sobol | 0.388 [0.214, 0.594] | 0.398 [0.227, 0.598] | 0.146 | -0.035 | 0.146 | 94.2 |
| 1LYZ | imos_s0 | 0.135 [-0.119, 0.390] | 0.044 [-0.328, 0.309] | 0.746 | -0.136 | 0.784 | 96.7 |
| 1LYZ | imos_s1 | 0.135 [-0.119, 0.390] | 0.044 [-0.328, 0.309] | 0.746 | -0.136 | 0.784 | 96.7 |
| 1LYZ | own_sobol | 0.702 [0.503, 0.867] | 0.720 [0.520, 0.881] | 0.092 | -0.029 | 0.090 | 90.0 |

![External convergence and MSE](../output/ehss_sampling_convergence_20260913/imos_convergence_decomposition.png)

## New panel: descriptive extension only

Each row is one molecule/arm, not an independent molecule if two arms share a molecule. Initial own grids and adaptive vendor grids cover different N ranges; differences of pooled medians are not causal sampler effects. No same-host timing ratio is inferred.

| Molecule | Method | Empirical RMSE alpha [95%] | Highest fitted N | RMSE (%) | Bias (%) | SD (%) | Variance/MSE (%) |
|---|---|---|---:|---:|---:|---:|---:|
| 1A11 | imos_s0 | 0.658 [0.317, 1.040] | 98,304 | 0.622 | 0.442 | 0.468 | 49.5 |
| 1A11 | imos_s1 | 0.658 [0.317, 1.040] | 98,304 | 0.622 | 0.442 | 0.468 | 49.5 |
| 1A11 | own_sobol | 0.586 [0.382, 0.868] | 196,608 | 0.113 | 0.033 | 0.116 | 91.6 |
| 1AL1 | imos_s0 | 0.587 [0.195, 1.148] | 98,304 | 0.617 | -0.075 | 0.654 | 98.5 |
| 1AL1 | imos_s1 | 0.587 [0.195, 1.148] | 98,304 | 0.617 | -0.075 | 0.654 | 98.5 |
| 1AL1 | own_sobol | 0.504 [0.291, 0.722] | 196,608 | 0.100 | -0.012 | 0.106 | 98.5 |
| 1AO6 | imos_s0 | 0.677 [0.272, 1.106] | 98,304 | 0.436 | 0.102 | 0.453 | 94.6 |
| 1AO6 | imos_s1 | 0.727 [0.313, 1.155] | 98,304 | 0.421 | 0.095 | 0.438 | 94.9 |
| 1AO6 | own_sobol | 0.528 [0.204, 0.783] | 196,608 | 0.154 | -0.031 | 0.161 | 95.9 |
| 1BMF | imos_s0 | 0.264 [-0.089, 0.712] | 98,304 | 0.475 | -0.084 | 0.500 | 96.8 |
| 1BMF | imos_s1 | 0.278 [-0.087, 0.728] | 98,304 | 0.473 | -0.082 | 0.498 | 97.0 |
| 1BMF | own_sobol | 0.734 [0.536, 0.947] | 196,608 | 0.055 | -0.040 | 0.041 | 48.0 |
| 1CLL | imos_s0 | 0.469 [0.005, 0.875] | 98,304 | 0.831 | -0.234 | 0.852 | 92.1 |
| 1CLL | imos_s1 | 0.469 [0.006, 0.875] | 98,304 | 0.831 | -0.233 | 0.852 | 92.1 |
| 1CLL | own_sobol | 0.574 [0.170, 0.779] | 196,608 | 0.131 | 0.000 | 0.140 | 100.0 |
| 1E0L | imos_s0 | 1.060 [0.654, 1.591] | 98,304 | 0.511 | 0.357 | 0.391 | 51.2 |
| 1E0L | imos_s1 | 1.060 [0.654, 1.591] | 98,304 | 0.511 | 0.357 | 0.391 | 51.2 |
| 1E0L | own_sobol | 0.458 [0.226, 0.791] | 196,608 | 0.153 | 0.047 | 0.155 | 90.7 |
| 1ENH | imos_s0 | 0.273 [-0.068, 0.654] | 98,304 | 0.802 | -0.202 | 0.830 | 93.6 |
| 1ENH | imos_s1 | 0.273 [-0.068, 0.654] | 98,304 | 0.802 | -0.202 | 0.830 | 93.6 |
| 1ENH | own_sobol | 0.570 [0.381, 0.738] | 196,608 | 0.116 | 0.071 | 0.097 | 62.1 |
| 1F88 | imos_s0 | 0.637 [0.208, 1.113] | 98,304 | 0.633 | 0.479 | 0.443 | 42.8 |
| 1F88 | imos_s1 | 0.637 [0.209, 1.115] | 98,304 | 0.633 | 0.480 | 0.441 | 42.5 |
| 1F88 | own_sobol | 0.258 [-0.006, 0.551] | 196,608 | 0.161 | 0.072 | 0.154 | 80.1 |
| 1GFL | imos_s0 | 0.491 [0.141, 0.930] | 98,304 | 0.900 | -0.382 | 0.871 | 82.0 |
| 1GFL | imos_s1 | 0.491 [0.141, 0.930] | 98,304 | 0.900 | -0.383 | 0.871 | 81.9 |
| 1GFL | own_sobol | 0.707 [0.409, 1.133] | 196,608 | 0.103 | -0.015 | 0.108 | 97.8 |
| 1HRC | imos_s0 | 0.036 [-0.375, 0.520] | 98,304 | 0.618 | -0.181 | 0.632 | 91.4 |
| 1HRC | imos_s1 | 0.041 [-0.367, 0.524] | 98,304 | 0.614 | -0.179 | 0.629 | 91.5 |
| 1HRC | own_sobol | 0.484 [0.222, 0.863] | 196,608 | 0.112 | -0.004 | 0.120 | 99.9 |
| 1MBN | imos_s0 | 0.296 [0.024, 0.574] | 98,304 | 0.802 | -0.120 | 0.848 | 97.8 |
| 1MBN | imos_s1 | 0.294 [0.022, 0.568] | 98,304 | 0.804 | -0.119 | 0.850 | 97.8 |
| 1MBN | own_sobol | 0.657 [0.496, 0.824] | 196,608 | 0.082 | -0.010 | 0.087 | 98.4 |
| 1PGB | imos_s0 | 0.764 [0.479, 1.140] | 98,304 | 0.628 | -0.063 | 0.668 | 99.0 |
| 1PGB | imos_s1 | 0.764 [0.479, 1.140] | 98,304 | 0.628 | -0.063 | 0.668 | 99.0 |
| 1PGB | own_sobol | 0.685 [0.333, 1.001] | 196,608 | 0.090 | -0.030 | 0.091 | 88.7 |
| 1RYP | imos_s0 | 0.418 [0.024, 1.026] | 98,304 | 0.586 | -0.295 | 0.541 | 74.7 |
| 1RYP | own_sobol | 0.472 [0.250, 0.633] | 196,608 | 0.116 | 0.059 | 0.106 | 73.6 |
| 1SU4 | imos_s0 | 0.409 [-0.083, 0.795] | 98,304 | 0.504 | 0.364 | 0.372 | 47.8 |
| 1SU4 | imos_s1 | 0.398 [-0.100, 0.776] | 98,304 | 0.509 | 0.372 | 0.372 | 46.7 |
| 1SU4 | own_sobol | 0.455 [0.243, 0.658] | 196,608 | 0.163 | 0.141 | 0.086 | 24.4 |
| 1TIM | imos_s0 | 0.311 [-0.203, 0.802] | 98,304 | 0.668 | 0.014 | 0.714 | 100.0 |
| 1TIM | imos_s1 | 0.313 [-0.203, 0.804] | 98,304 | 0.669 | 0.012 | 0.715 | 100.0 |
| 1TIM | own_sobol | 0.513 [0.271, 0.724] | 196,608 | 0.156 | 0.012 | 0.167 | 99.4 |
| 1VII | imos_s0 | 0.572 [0.134, 1.075] | 98,304 | 0.625 | 0.210 | 0.629 | 88.7 |
| 1VII | imos_s1 | 0.572 [0.134, 1.075] | 98,304 | 0.625 | 0.210 | 0.629 | 88.7 |
| 1VII | own_sobol | 0.566 [0.407, 0.740] | 196,608 | 0.103 | -0.000 | 0.110 | 100.0 |
| 1W0O | imos_s0 | 0.186 [-0.270, 0.527] | 98,304 | 0.932 | 0.429 | 0.885 | 78.9 |
| 1W0O | imos_s1 | 0.191 [-0.264, 0.528] | 98,304 | 0.930 | 0.428 | 0.883 | 78.8 |
| 1W0O | own_sobol | 0.559 [0.365, 0.773] | 196,608 | 0.175 | -0.088 | 0.161 | 74.4 |
| 2PTC | imos_s0 | 0.300 [-0.172, 1.128] | 98,304 | 0.763 | 0.071 | 0.812 | 99.1 |
| 2PTC | imos_s1 | 0.299 [-0.173, 1.128] | 98,304 | 0.763 | 0.071 | 0.812 | 99.1 |
| 2PTC | own_sobol | 0.458 [0.153, 0.680] | 196,608 | 0.162 | -0.036 | 0.169 | 95.1 |

## Mechanistic explanation and manuscript wording

The own estimator draws four scrambled-Sobol coordinates for an isotropic direction and a uniform point on the perpendicular enclosing impact disk. With elastic specular reflection and fixed geometry, all subsequent scattering events are deterministic functions of those four inputs. Multiple reflections therefore do not add independent random dimensions, although collider switching and grazing can introduce discontinuities. Balanced coverage of this input space is a plausible variance-reduction mechanism, directly supported by the native IID/Sobol control. The current data do not quantify how individual path discontinuities determine each molecular exponent.

IMoS literature and its 1.13 manual describe a drag-based construction with three perpendicular bulk-flow directions. These are not three allowed incoming-ray directions. Its incidence weighting and conversion/averaging of momentum-transfer responses differ from the own direct scalar estimator. Signed momentum components can affect variance in such a formulation, but their contribution to the observed binary-output variance was not isolated here. Do not claim an exact Rao–Blackwell improvement, unnecessary velocity sampling, or a measured cancellation penalty without additional evidence. Equivalence of all scalar definitions across arbitrary geometries also remains outside this analysis.

Suggested manuscript paragraph:

> The reduction in time to a specified error combines lower trajectory-evaluation cost with improved sampling efficiency. In controlled native calculations, replacing IID incidence samples with scrambled Sobol samples reduced RMSE by 2.12–6.11-fold at 16,384 trajectories across the six development structures, while changing the traversal from exhaustive search to BVH left the sampled scattering results unchanged. In the fixed-grid IMoS comparison, run-to-run variance accounted for approximately 96–97% of the observed mean-squared error for 1CRN and 1LYZ at 196,608 trajectories. These observations support a substantial sampling-efficiency contribution, while empirical convergence slopes varied with structure and budget. Since IMoS uses a different incidence and drag-estimation formulation, its full error difference cannot be attributed to the Sobol sequence alone.

Supporting definitions: own `src/support_exchange_transport/ehss_compact_source.py`; [IMoS 1.13 manual](https://www.imospedia.com/imos/explanation-of-the-code/) sections 3.2 and 8.4; Larriba and Hogan, [J. Computational Physics 251 (2013), 344–363](https://doi.org/10.1016/j.jcp.2013.05.038), sections 2.3 and 2.5; [SciPy scrambled Sobol documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html). The archived manual/paper texts were reviewed in the preceding discussion. The general published formulation is not a full source-level reconstruction of the distributed IMoS binary.

## Interpretation limits

- Pointwise percentile bootstrap, not simultaneous intervals; empirical finite-grid slopes are not asymptotic convergence orders.
- Bias is sample mean discrepancy, not proof of persistent algorithmic bias. MSE uses population variance; reported SD uses ddof=1.
- Panel vendor two-point slopes are especially noisy and subsequent budget was adaptive; no cross-method matched-N or speed ratio on this panel.
- Reference resampling captures run uncertainty, not unknown systematic model/kernel error or unseen path tails.
- Calibration seed replicates only; no change to original final selections. Same vendor seed does not guarantee identical rays.

## Validation

- 576 archived flat/BVH pairs match physical fields; only 384 primary BVH rows enter native statistics.
- Large-case cap4096 replaces cap64; no replay is counted as a new replicate.
- Stage A readout diagnostics and sphere excluded; own 16 distinct estimates and vendor 8 per arm/budget.
- Vendor D/E 592 rows match frozen jobs and both cohorts are complete; own initial grid has 432 rows.
- 175 frozen numerical source hashes unchanged; no numerical experiment invoked.
- All recomputed native and Stage A RMSE/bias/SD and initial own-panel metrics match archived summaries. MSE decomposition checked in every cell.

Machine-readable full results: `output/ehss_sampling_convergence_20260913/analysis.json`; normalized archived observations: `source_observations.json`; source SHA256 inventory: `provenance.json`. Reproduction: `.venv/Scripts/python.exe -X utf8 scripts/analyze_ehss_sampling_convergence.py`.
