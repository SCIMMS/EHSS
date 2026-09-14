# Stage G: independent 2% accuracy and same-PC timing

Completed on 2026-09-14 at 09:20:30 KST (PID 12792). Final archive, source/design hashes, 720 accuracy records, 360 timing replays, one excluded warmup and all 281 process records passed the local audit. There were no missing/duplicate requests, process failures, forced shutdowns or new calibration runs. All single/batch paired numerical outputs matched exactly.

Own method: 17/18 pass and 1/18 uncertain. IMoS: 16/18 pass and 2/18 uncertain. Neither method has a definite fail by the prespecified confidence-interval rule. Both methods pass on 15/18 structures. Pass means the run-level/reference bootstrap RMSE upper endpoint is <=2%; uncertain means the interval crosses 2%. It is not a guaranteed population error bound.

Across the 15 jointly passing structures, the ratio of median IMoS/own single-process wall times ranges from 2.58 to 14.55. This is a same-host comparison at a common validated error target, not exactly identical RMSE or a general algorithmic speedup claim.

The fixed all-18 batch takes 103.662 s for IMoS and 4.330 s for own, a latency ratio of 23.94. Three structures have an uncertain outcome in at least one method. Therefore this is an all-input workload latency observation, not an all-18 target-qualified 2% speedup. No synthetic passing-only batch time is obtained by subtracting per-input times from that observation.

The lab host is an Intel Core i5-12400F; one requested thread. Timing includes process creation, imports/runtime, input/output and fresh geometry, with code cache present after an excluded warmup. There are five blocks. Ratios are ratios of method medians; percentile 95% intervals enumerate all 3,125 paired five-block bootstrap resamples. With only five blocks, uncertainty resolution is limited and intervals are not simultaneous guarantees. The measured duration from the first final batch status to completion was 77 min 51 s, excluding preceding setup/warmup.

## Accuracy outcomes

Each method/structure has 20 independent new seeds. Timing seeds are separate; single/batch replays are not additional accuracy samples.

| Structure | Own rays | Own RMSE % [95%] | Own state | IMoS Simplify | IMoS rays | IMoS RMSE % [95%] | IMoS state |
|---|---:|---|---|---:|---:|---|---|
| 1A11 | 6,144 | 0.9179 [0.5294, 1.2731] | pass | 1 | 49,152 | 0.9356 [0.6957, 1.1561] | pass |
| 1AL1 | 3,072 | 1.3571 [0.8860, 1.7853] | pass | 1 | 24,576 | 1.6564 [1.0117, 2.2378] | uncertain |
| 1AO6 | 6,144 | 0.9332 [0.5739, 1.2346] | pass | 1 | 24,576 | 1.1774 [0.8656, 1.4430] | pass |
| 1BMF | 3,072 | 0.9857 [0.7571, 1.1965] | pass | 0 | 12,288 | 1.9230 [1.3254, 2.5006] | uncertain |
| 1CLL | 12,288 | 0.6998 [0.4423, 0.9640] | pass | 1 | 98,304 | 0.8279 [0.6123, 1.0292] | pass |
| 1E0L | 6,144 | 0.5538 [0.3914, 0.6952] | pass | 1 | 49,152 | 0.5746 [0.4124, 0.7133] | pass |
| 1ENH | 3,072 | 1.0141 [0.6558, 1.3792] | pass | 0 | 24,576 | 0.9486 [0.6566, 1.2014] | pass |
| 1F88 | 3,072 | 1.0514 [0.6647, 1.4316] | pass | 1 | 49,152 | 0.9835 [0.6907, 1.2823] | pass |
| 1GFL | 12,288 | 0.6037 [0.4053, 0.7935] | pass | 1 | 98,304 | 0.6838 [0.4938, 0.8549] | pass |
| 1HRC | 3,072 | 0.9907 [0.7416, 1.2204] | pass | 0 | 24,576 | 1.2695 [0.8273, 1.6656] | pass |
| 1MBN | 3,072 | 0.8850 [0.6642, 1.0902] | pass | 0 | 24,576 | 1.4438 [1.0555, 1.7881] | pass |
| 1PGB | 3,072 | 1.1859 [0.8334, 1.5262] | pass | 1 | 49,152 | 1.1830 [0.8531, 1.4896] | pass |
| 1RYP | 3,072 | 1.1392 [0.7900, 1.4463] | pass | 0 | 24,576 | 1.2214 [0.8627, 1.5552] | pass |
| 1SU4 | 3,072 | 2.0175 [1.5077, 2.4773] | uncertain | 0 | 24,576 | 1.3087 [0.8491, 1.7673] | pass |
| 1TIM | 6,144 | 0.7943 [0.5560, 1.0048] | pass | 0 | 24,576 | 1.1891 [0.8054, 1.5330] | pass |
| 1VII | 3,072 | 1.2166 [0.8960, 1.5340] | pass | 1 | 24,576 | 1.5632 [1.1593, 1.9291] | pass |
| 1W0O | 12,288 | 0.4991 [0.3625, 0.6246] | pass | 1 | 24,576 | 1.0095 [0.6720, 1.3387] | pass |
| 2PTC | 3,072 | 1.0915 [0.7582, 1.3860] | pass | 0 | 24,576 | 1.3391 [0.9431, 1.7074] | pass |

## Process timing

All measured conditions are retained. A ratio labelled unqualified is a latency observation only. No outcome was dropped or retuned.

| Structure / workload | IMoS median (s) | Own median (s) | Ratio [95% paired block interval] | Both pass 2%? |
|---|---:|---:|---|---|
| 1A11 | 9.3500 | 3.5417 | 2.640 [2.599, 2.661] | yes |
| 1AL1 | 9.1484 | 3.5408 | 2.584 [2.565, 2.684] | no; uncertain outcome included |
| 1AO6 | 13.5421 | 3.6051 | 3.756 [3.701, 3.813] | yes |
| 1BMF | 20.1000 | 3.7106 | 5.417 [5.378, 5.550] | no; uncertain outcome included |
| 1CLL | 10.6078 | 3.5448 | 2.992 [2.968, 3.206] | yes |
| 1E0L | 9.4481 | 3.5359 | 2.672 [2.602, 2.710] | yes |
| 1ENH | 10.2494 | 3.4754 | 2.949 [2.660, 2.977] | yes |
| 1F88 | 12.6285 | 3.5212 | 3.586 [3.545, 3.810] | yes |
| 1GFL | 14.7026 | 3.6069 | 4.076 [3.861, 4.268] | yes |
| 1HRC | 9.4385 | 3.5351 | 2.670 [2.665, 2.969] | yes |
| 1MBN | 9.5939 | 3.5382 | 2.711 [2.612, 2.767] | yes |
| 1PGB | 9.4854 | 3.5489 | 2.673 [2.609, 2.711] | yes |
| 1RYP | 55.3411 | 3.8033 | 14.551 [14.222, 14.712] | yes |
| 1SU4 | 11.6605 | 3.6115 | 3.229 [3.186, 3.315] | no; uncertain outcome included |
| 1TIM | 10.4575 | 3.6039 | 2.902 [2.852, 3.114] | yes |
| 1VII | 9.2452 | 3.5789 | 2.583 [2.556, 2.614] | yes |
| 1W0O | 11.4970 | 3.6016 | 3.192 [3.056, 3.294] | yes |
| 2PTC | 9.7854 | 3.5323 | 2.770 [2.735, 2.792] | yes |
| all_18_batch | 103.6617 | 4.3304 | 23.938 [23.751, 24.497] | no; uncertain outcome included |

## Provenance and interpretation

The practical 2% target was explicitly chosen after the earlier 1%/0.5% study. Existing calibration alone selected the settings; all final seeds are new. Earlier holdout failures and uncertainties remain unchanged. These results should be presented as a new cohort, not a retrospective relaxation of earlier failures.

IMoS 1RYP Simplify=1 remains a recorded technical failure; this cohort uses Simplify=0. Numerical drag-tensor output remains unavailable and is an accepted limitation. No further tensor acquisition, calibration, seed extension or timing repetition is triggered by these results.

Candidate selection used calibration internal medians on each calibration host (own: local, IMoS: lab). This selects a finite-grid candidate, not a global optimum in process latency. Reported speed ratios use only the joint lab measurements. Reference uncertainty is propagated from the existing eight high-budget replicates per structure; previous independent-boundary and collision-subset checks support those references.

All 267 manifest entries were accounted for: archived entries were checked directly; 175 omitted remote src files were checked against local frozen hashes and were not re-downloaded. Raw IMoS text/configuration, own worker payload/specification, coordinates, worker time sums, histogram/weighted CCS sums, original seed plan and the frozen selection all passed the audit.

Archive SHA256: `ef9cbabed5c0c423155b0941387d9664605bba614889ac638e63c9f35aabbb0b`. Selection SHA256: `781fdc5b7b49d234d06d70564c0f7de8fb31d971a356866d60626453d1a25d76`.

Raw archive: `output/imos113_stage_g_20260914/final_archive.zip`; extracted records: `returned/`; machine-readable audit and all five timing values: `stage_g_summary.json`. Offline audit: `python scripts/analyze_imos113_stage_g.py`.
