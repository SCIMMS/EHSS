# Supporting Information for Efficient CPU Evaluation of Molecular Exact Hard Sphere Scattering

{{AUTHORS}}

## S1 Molecular geometries and provenance

The numerical input is the retained coordinate/radius file, not a PDB identifier alone. The additional panel uses the first deposited model, all ATOM chains, alternate locations blank or A and positive occupancy; H/D, HETATM and waters are excluded. No assembly expansion or structural relaxation is performed. Each original PDB and metadata response, selected NPZ and XYZ, and centered comparison input has a retained SHA256. The original and centered XYZ hashes may differ because of translation and serialization; the audit checks the coordinate/radius roundtrip used in the comparison.

Bare C/N/O/S radii are 1.70/1.55/1.52/1.80 Å; the helium probe radius is 1.4 Å. The six development geometries are 1CRN (327 atoms), 1BNA (486), 1LYZ (1,001), 4AKE (3,312), 1IGT (10,196), and 1AON (58,674). Input preprocessing and radius files for the development controls remain in their archived bundles. Earlier MD-derived step-zero pilot inputs and current standardized comparison inputs must not be substituted for one another using only their common PDB identifiers.

The additional panel was frozen before its EHSS performance evaluation. Six entries occupy each size group; the observed axis-shape coverage is six elongated and twelve compact_axes cases. Axis shape uses gyration eigenvalue ratios, not a cavity calculation or biological family partition. Two small candidates were replaced by elongated peptides before performance evaluation, as recorded in the panel manifest. This is a deliberately varied test panel rather than a random population sample or a guaranteed sequence-disjoint dataset.

{{TABLE:panel}}

Original entry metadata, selection/exclusion counts, source URLs and full hashes are in input_provenance.json. Individual structure primary citations are transcribed from those archived metadata records below. They credit the deposited structures; they do not establish that the retained coordinates represent gas-phase ions.

{{ENTRY_REFERENCES}}

## S2 Algorithmic and numerical details

The enclosing sphere is centered at the arithmetic mean of atom coordinates, with R = maxⱼ(|cⱼ − c| + Rⱼ)(1 + 10⁻⁹). For sphere incidence, two Sobol coordinates generate uniform S² direction; two generate radius R√q and azimuth 2πq on the perpendicular impact disk. The starting plane lies a distance R upstream of the centroid. Three independent blocks use seeds s, s + 10,007, and s + 20,014. Each block contains a power-of-two number of points; its weight is divided by three when pooling. The total N = 3n is not itself treated as a single power-of-two digital net. IID controls use the same geometric mapping with independent uniform coordinates.

Static spatial grouping recursively sorts along the maximum center-span axis with stable mergesort, splits at the median, and stops at four atoms, depth 24, or zero span. Groups can therefore exceed four atoms in a degenerate case. Leaf bounds are padded by 10⁻¹⁰ max(1, maxⱼ Rⱼ) in input length units. The kernel stores traversal arrays and subtree-end indices; no triangle mesh is built. Sphere entries retain the computed direction norm, reject backward candidates and negative discriminants, and tolerate only a small negative entry distance before clamping to zero. Reflection restores the origin to the sphere surface and normalizes the direction.

The production comparison disables response caching and motion metadata. It starts at cap 512, checks for one more collision after the cap, and retraces unchanged incident samples at cap 1,024, 2,048, or 4,096 when the sampled missing-contribution bound exceeds 0.05% of computed CCS. All attempts remain in the record. The additional acceptance guard is a bound no larger than target/10 relative to the primary reference. A small observed bound does not certify unsampled long paths or general ensemble error.

Core validation comprises 72 passed tests without failures or skips: analytical sphere and impact-parameter controls; multiple collision order; near-gap, grazing, and trapped-ray cases; cap consistency; static/earlier path agreement; backend and thread consistency; native traversal agreement; CLI parsing and errors. Co-moving transformations test translation, rotation, atom reordering and length scales 0.01, 1, and 100. These are finite software tests rather than proofs for all contact geometries.

## S3 References and independent collision checks

For each additional structure, eight independent sphere-boundary estimates use 2²⁰ rays and eight box-boundary estimates use 2¹⁸ rays. Across the panel this is 288 estimates and 188,743,680 original incident rays; retracing at a larger cap is not counted as an independent sample. Reference caps start at 1,024 and extend through 4,096 when the observed residual bound exceeds 0.01%. Reference uncertainty uses the t interval of the primary eight repeats. The two boundaries share the production collision kernel; a separate exhaustive implementation compares histories, collision counts, escape states, final directions and CCS on 13,824 matched rays. Small/medium cases use 1,024 test rays and large cases use 256.

{{TABLE:references}}

All independent-kernel test subsets agreed. All primary half-widths satisfy the 0.1% precision criterion, and all primary and box intervals overlap. The box mean differences and primary residual bounds are reported explicitly rather than interpreting interval overlap as equivalence. Reference resampling represents measured run uncertainty; it does not remove a common systematic geometry, radius or kernel error. Stage C uses its separately archived development reference ensemble, not the older pilot CCS means.

## S4 Native search and sampling controls

The original 768-run grid includes six structures, four budgets, eight seeds, two samplers and two traversal modes. All 384 flat/BVH pairs agree on sampled physical fields. For 4AKE, 1IGT and 1AON, 384 cap-4,096 method replays yield 192 further matched traversal comparisons and replace their cap-64 records for the reported high-cap analysis. The replays are dependent on the original incident samples and do not increase replicate count. The maximum recorded complete path in this control has 1,160 collisions. Cap-induced CCS increases remain within the original sampled residual bounds.

Main-text Table 1 reports preparation-inclusive native times at 16,384 rays. These are warm in-process local i5-7500 observations, distinct from laboratory process times. Operation counters were disabled during timed runs; their separate collection is retained. Figure 2 plots native sampling error over the four budgets with eight independent replicates and 20,000 pointwise bootstrap draws. Large-case cap-4,096 results replace cap-64 points. No all-structure asymptotic convergence rate is fitted or claimed.

An earlier single-thread backend control at 16,384 rays reduced tracing-plus-readout time by factors of 4.17 for 1CRN and 3.71 for 1LYZ when changing the array-oriented Numba implementation to scalar Numba. The sphere geometry, source, BVH and scattering rules were shared. C++ scalar controls and threaded variants remain separate implementation evidence; they do not measure the effect of introducing a BVH. The complete preserved backend table is supplied as backend_controls.csv.

## S5 IMoS configuration and readout checks

IMoS 1.13 is run through its distributed Workmule interface with fixed seeds, one requested thread, elastic/specular helium scattering, and PA, potential and trajectory-method modes disabled. The regular CCS field is parsed directly from original text output. Each emitted clause and reported atom count is checked against the frozen job. Both Simplify = 0 and 1 appear in calibration. Their results are not pooled as independent replicates when they share a seed. Source-level behavior of the distributed binary is not inferred from the label of the setting alone.

Analytical sphere normalization, effective radii, regular CCS selection, repeat-seed behavior and true batch execution were checked. The three synthetic rotation shapes are rod9, plate9 and asym4; each has twelve SO(3) rotations and eight seeds, giving 288 paired jobs at 196,608 requested trajectories per method. The method-specific 36-cell Welch/Holm screening rule yielded no warning. The same-ray geometric covariance tests and the finite-seed rotation diagnostic answer different questions. Neither establishes exact equality of vendor and own estimators.

HandB/DragRG settings yielded regular and Happel-Brenner outputs, but no reconstructable numerical drag tensor in the tested text, stdout or changed sidecars. Our own momentum-moment readout recovers its scalar CCS by a trace identity, but this is not a reconstruction of IMoS tensor averaging. Numerical drag-tensor availability is an accepted limitation of the present study. A direct runtime ranking against other rotation-optimized software was not evaluated.

## S6 Calibration and frozen evaluation cohorts

The two-structure campaign is identified as Stage C in the archive. After a bounded 96-estimate calibration supplement, eight method/structure/target settings were frozen and evaluated with twenty new seeds, totaling 160 independent final estimates. Ten paired timing blocks compare actual single and two-structure batch execution. All settings passed their 1% or 0.5% target. Main-text Table 2 contains all four per-structure comparisons; full accuracy and timing summaries are supplied as stage_c_summary.json.

The new-panel vendor calibration comprises Stage D (280 estimates), E (312), and F (464), totaling 1,056 estimates across 132 cells. Existing own calibration supplies 816 estimates across 102 cells. These records select settings; they are not the final 20-seed cohort. D/E calibration supplied 1% candidates for 12/18 structures and no 0.5% candidates. After F, candidate coverage reached 18/18 at 1% and 13/18 at 0.5%. Candidate coverage is not independently validated performance.

Stage F retained a timeout and two completed outputs recovered without rerunning. Its 214 retained estimates plus 250 subsequently completed estimates total 464; the original anomaly is kept in provenance. The earlier 1RYP Simplify = 1 access violation excludes that arm, not the molecule. These technical records are not converted into normal timing observations. The final Stage G cohort itself had no process anomaly.

The earlier own-method final cohort contains 700 estimates from 35 unique settings and twenty new seeds. A setting shared by two targets was run only once. At 1%, 13 structures passed, four were uncertain and 2PTC failed; at 0.5%, 15 passed and three were uncertain. These outcomes remain intact in Table S7. A 2% target was subsequently selected for a new practical comparison. It was not the original target of the study, and its selection must not be represented as fully prespecified before earlier observations.

Stage G reused calibration records only, froze both methods before new seeds, and executed 18 × 2 × 20 = 720 independent final estimates. Its own budgets range from 3,072 to 12,288 rays and IMoS budgets from 12,288 to 98,304 rays. Eligible settings minimize internal calibration median time on each method's calibration host; this is a finite-grid selection proxy rather than globally optimal process latency. No old own holdout is incorporated into the new accuracy statistics. Seed inventories, selections and numerical-source hashes are retained. No uncertain final outcome triggers retuning or extra sampling.

{{TABLE:selection}}

## S7 Complete final accuracy results

For eᵣ = 100(Ω̂ᵣ/Ωref − 1), RMSE is √mean(eᵣ²). The error decomposition uses mean(e)² + var(e, ddof = 0); reported sample SD uses ddof = 1. Confidence intervals resample full run estimates and primary-reference replicates, not individual correlated Sobol points. Each of 20,000 bootstrap draws uses a reference mean shared by all resampled runs in that draw. Pass requires the interval upper endpoint at or below target; fail requires its lower endpoint above target; crossing the target is uncertain. All comparisons are pointwise. The finite-sample tail guard is reported separately and has no matching vendor certificate.

{{TABLE:accuracy}}

Our 1SU4 and IMoS 1AL1/1BMF are uncertain. All other new 2% outcomes pass. The complete run-level CCS, mean discrepancy, SD, rays and seed records are in the accompanying JSON; timing replays are excluded from these twenty-replicate calculations. Both methods satisfy the target on fifteen structures. This count is a statement about the tested cohort rather than an estimated success probability for arbitrary proteins.

## S8 Full process timing and internal components

All final external ratios use one laboratory i5-12400F computer and the same single-thread request. Popen-to-natural-exit intervals include startup, imports/runtime, applicable code-cache loading, internal input/output and fresh geometry. Prior PDB retrieval, conversion, installation and wrapper preparation, and subsequent result archiving are excluded. The code cache exists; the OS cache is not reset. Background load and CPU affinity are not fully controlled. No timing block is removed because its value is inconvenient.

Five Stage G blocks alternate method and single/batch order. Within each block, the same seed/specification is replayed as a single input and in the full eighteen-input batch; all corresponding numerical results agree exactly. There are 360 replayed per-input results. Ratios are ratios of method medians. The five paired blocks generate exactly 3,125 bootstrap resamples, giving pointwise percentile intervals with limited five-block resolution. Results in uncertain accuracy cases remain present but are not classified as target-qualified speedups.

{{TABLE:timing}}

The full-panel batch ratio includes three uncertain method/structure cases. Its interpretation is fixed-workload latency, not an all-18 validated 2% speedup. A jointly passing subset batch was not measured and is not reconstructed by subtraction. All five raw timing values per method and condition are supplied in timing_blocks.csv. The earlier Stage C batch ratios are 4.49 [4.43, 4.52] at 1% and 6.46 [6.43, 6.53] at 0.5%, with both structures passing in each batch.

{{TABLE:components}}

Component medians describe twenty final accuracy runs on the laboratory host and are not the five-block process medians. Total is the median of each run's measured geometry + source + trace + readout sum. It is not the sum of independently computed component medians. Input parsing is separate from this internal total and remains included in process wall time. The IMoS internal median is retained for scope transparency; its endpoints are not established to match the own internal total, so no comparative factor is formed from this table.

## S9 Earlier stricter targets and convergence analysis

{{TABLE:old_holdout}}

These earlier own-method results were produced on the local PC and are not divided by laboratory IMoS times. A stricter target may use a larger independently selected budget, explaining why a structure can pass at 0.5% but fail at a separately selected 1% setting. The new Stage G campaign is separate in target, settings and seeds.

The archived convergence analysis uses the corrected native grid, Stage A fixed-grid molecular calibration, and the initial own-panel grid with vendor D/E calibration. It intentionally does not pool later F, final C/G, old own holdout or adaptive own extensions into those fitted slopes. Native budgets span 256 to 16,384 with eight independent replicates; Stage A includes three budgets from 12,288 to 196,608 with sixteen own estimates and eight per vendor arm. Reference means are resampled jointly, and 20,000 bootstrap draws provide pointwise intervals.

At 16,384 native rays, IID/Sobol RMSE ratios are 4.392, 2.490, 6.110, 2.619, 2.451 and 2.125 for 1CRN, 1BNA, 1LYZ, 4AKE, 1IGT and 1AON. In Stage A at 196,608 rays, IMoS/own RMSE ratios are 4.47 [2.31, 8.11] and 8.15 [4.20, 12.96] for 1CRN and 1LYZ. The associated vendor variance fractions of MSE are 96.1% and 96.7%. These fixed-grid observations support a lower error level and reduced dispersion; they do not identify the random-number generator as the sole cause of the vendor difference.

Empirical exponents are unweighted log-log slopes over finite grids, not asymptotic orders. Vendor panel D/E fits use only the common two-budget interval and include downward extensions as recorded. Comparing their median exponent with the different own budget grid is descriptive, not a controlled causal sampler comparison. The complete archived convergence analysis and all cell/slopes/ratio records accompany this supplement.

## S10 Collision counts and supporting PA and reuse observations

{{TABLE:tails}}

Counts are pooled only across the twenty own Stage G accuracy runs per structure. Percentages use all launched rays; escaped CCS shares use the sum of completed contributions across runs. The last recorded collision bin can contain unresolved trajectories and is not necessarily a true final collision count. Contributions grouped by completed path length are not the error from imposing that length as a cap. These statistics identify observed rare paths and do not bound their probability outside the sampled ensemble.

The earlier PA controlled-error study uses a continuous projected-disk-union reference. At 0.1% error, adaptive spatial PA is about 1.16 and 2.44 times faster than BVH sampling for 1CRN and 1LYZ, including preparation; the ranking changes in some less stringent conditions. The preserved table pa_equal_target_excerpt.csv includes the actual budgets and RMSE values. These results are supporting PA evidence, not a comparison against octree software or an EHSS performance result.

The earlier geometry-update trajectories were generated with OpenMM 8.6 and serialized ff14SB/OBC2 systems at 300 K, with a 2 fs step, friction 1 ps⁻¹, NoCutoff nonbonded treatment, HBonds constraints, and solute/solvent dielectrics 1/78.5. The archived protocol records up to 2,000 minimization iterations at tolerance 10 kJ mol⁻¹ nm⁻¹, 1,000 equilibration steps and 8,192 production steps; only original heavy atoms enter EHSS. These short controls are not converged thermodynamic ensembles. Protocol, retained trajectories and preparation records accompany the data. The 67-frame, 16.384 ps observations show every support group moving at each successive frame: 146 groups for 1CRN and 480 for 1LYZ. At 16,384 rays, update-plus-direct versus rebuild-plus-direct ratios are 1.04 and 1.03 in the recorded cases. The timing control updates an already prepared scene to MD steps 1, 32 and 8,192, with five repeats per condition; source preparation is included, while initial scene creation and response training are excluded. These cap-64 local controls are distinct from the final static benchmark. Spatially widespread perturbations require broadly distributed geometry/response checks. Short trajectories on two structures cannot establish behavior of all MD, but they do not support a useful general acceleration claim for the tested reuse design. The preserved md_reuse_excerpt.csv records the direct and response variants. Restricted terminal motion remains an untested regime and is outside this paper's performance conclusions.

## S11 Reproducibility and scope closure

The immutable evidence snapshot v4 contains 11,655 manifest entries and has SHA256 22e9f66d61447cb20e02b28784c2eb0e35de652b1b23dbe2bde2665380784645. It integrates the original raw/calibration/final evidence through Stage G, input and source hashes, numerical checks and subsequent convergence analyses. Earlier immutable snapshots and failed-process provenance are retained. Vendor executable/runtime redistribution is excluded; a user wishing to rerun vendor calculations must obtain the compatible distribution separately.

A separate extraction passed verify_snapshot.py and nine offline analysis/audit commands. Stage C/D/E/F, convergence and G principal JSON outputs matched their originals exactly. Project imports were verified to originate from the extracted scripts and source. These validation runs used the existing local dependency environment and performed no new numerical simulation or vendor execution. Separate earlier wheel-installation and core-test evidence is also preserved. The new manuscript integration only transforms archived evidence into text, tables and figures.

For offline reproduction, extract the evidence archive, install the recorded Python dependencies, run verify_snapshot.py, and follow the packaged SNAPSHOT_README.md analysis commands. The paper bundle includes final tables, raw final accuracy observations, timing values and source inventories for convenience; it does not replace the larger raw evidence snapshot. The run-level data and covariance of replicated results should be preserved when computing new summaries.

The evidence is restricted to static elastic/specular helium EHSS and the stated finite workloads. It does not establish general MD acceleration, compressed response transport or experimental CCS accuracy. The external benchmark is limited to IMoS 1.13.
