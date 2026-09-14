# XYZ PA/EHSS calculator

연구실용 실행 파일은 `ccs.py`입니다. XYZ 파일을 읽어 PA와 EHSS를 계산하고,
`summary.csv`와 상세 `results.json`을 저장합니다. 아래 안내와 CLI 메시지는
영어로 제공하여 연구실 구성원이 함께 사용할 수 있도록 했습니다.

## Quick start

Use Python 3.11-3.13 (3.13 tested). No compiler, IMoS installation or GPU is required.
Keep `ccs.py` and the supplied `src` folder together. The script is a launcher for the
existing research kernels, not a self-contained file with those kernels duplicated.

Windows PowerShell, from the extracted folder:

```powershell
py -3.13 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-ccs.txt
.venv/Scripts/python.exe ccs.py examples/ccs/water.xyz --output results/water
```

If `python` already selects a compatible environment:

```sh
python -m pip install -r requirements-ccs.txt
python ccs.py molecule.xyz --output results/molecule
python ccs.py --help
```

Linux/macOS: use `python3 -m venv .venv` and `.venv/bin/python` in place of the Windows
interpreter path. This release was executed on Windows; the kernels are Python/Numba
and do not use a Windows DLL. A full project installation (`python -m pip install .`)
also provides the `ccs-xyz` command. In this dedicated EHSS repository, its default
dependencies match the minimal CLI requirements; testing and offline replay use
the optional dependency groups documented in the repository README.

The first calculation compiles native kernels and may take tens of seconds. A new
process can compile them again. This startup is recorded separately from computation.
Run multiple files in one invocation to share startup preparation.

## Examples

```sh
# PA and EHSS: default adaptive PA and native single-thread EHSS
python ccs.py molecule.xyz --output results/molecule

# EHSS only: 65,536 rays per replicate, five independent scrambles
python ccs.py molecule.xyz --method ehss --rays 65536 --repeats 5 --output results/ehss

# Adaptive PA: refine both directions and projection grid
python ccs.py molecule.xyz --method pa --directions 256 --grid-size 1024 --output results/pa

# BVH any-hit PA; impact rays are shared with EHSS when both are requested
python ccs.py molecule.xyz --pa-method bvh --rays 65536 --output results/bvh

# Batch input, including quoted patterns on Windows
python ccs.py "structures/*.xyz" --output results/batch

# Frame 10, or every frame calculated independently
python ccs.py trajectory.xyz --frame 10 --output results/frame10
python ccs.py trajectory.xyz --all-frames --output results/all_frames

# Coordinates in nm; radius options remain in angstrom
python ccs.py molecule_nm.xyz --units nm --output results/nm

# Explicit heavy-atom selection to match the earlier protein scenario
python ccs.py protein.xyz --exclude-hydrogen --output results/heavy

# Change atomic radii, or the added probe radius
python ccs.py molecule.xyz --radii-json examples/ccs/radii_override.json --output results/custom
python ccs.py molecule.xyz --probe-radius 0 --output results/geometric

# Optional parallel EHSS tracing; PA and geometry stay serial
python ccs.py molecule.xyz --threads 4 --output results/threads4
```

The output directory must not already exist. Existing results and XYZ files are never
overwritten. Use a new directory for each setting. Exit code 2 indicates invalid
input/options or a file error; 130 indicates interruption. An unresolved EHSS tail is
reported as a warning and in the saved results, not silently treated as escape.

## Input and physical model

Standard XYZ is required: atom count, **one comment line (which may be empty)**, then
exactly that many `Element x y z` rows. Coordinates default to angstrom. Concatenated
XYZ frames are accepted. The first frame is selected by default, with a message if
more frames exist. Unknown elements, nonfinite coordinates, incomplete frames and
coincident selected atom centers are rejected. Extended XYZ property columns and
periodic cells must first be exported as an isolated standard XYZ molecule.

```text
3
water in angstrom
O  0.000000  0.000000  0.000000
H  0.957200  0.000000  0.000000
H -0.239987  0.926627  0.000000
```

All supplied atoms, including H, are included unless `--exclude-hydrogen` is given.
Hydrogens are never added; charge, protonation, bonds and residues are not inferred.
Element symbols are required, not atomic numbers or atom names such as CA1. Isotopic
labels such as D need an explicit radius override (the H exclusion switch only removes H).
Coordinates are recentered internally; the translation is recorded.

Each physical collision radius is **atomic VDW radius + probe radius**. Default probe
radius is 1.40 Å, matching the earlier static He scenario. These are geometric
hard-sphere parameters, not fitted IMoS/EHSSrot-Siu parameters. Changing a probe radius
does not implement a nitrogen potential or energy accommodation model.

Built-in atomic radii in Å:

| H | He | C | N | O | F | Si | P | S | Cl | Se | Br | I |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.10 | 1.40 | 1.70 | 1.55 | 1.52 | 1.47 | 2.10 | 1.80 | 1.80 | 1.75 | 1.90 | 1.83 | 1.98 |

Source: [Mantina et al. (2009), Table 12](https://doi.org/10.1021/jp8111556),
cross-checked against the [MolSSI QCElemental transcription](https://github.com/MolSSI/QCElemental/blob/master/qcelemental/data/mantina_2009_vanderwaals_radii.py).
H uses 1.10 Å (Rowland/Taylor), not the older Bondi 1.20 Å; C/N/O/S match our earlier
heavy-atom benchmarks. These additional elements are input support, not a claim of
experimental CCS validation for those species.

Override/add individual atomic radii with a JSON object, for example `{"H": 1.20}`.
For an unsupported element supply the radius selected for your model. Values must
be finite positive numbers in Å. Do **not** supply probe-expanded radii here: the
probe is added afterward. The complete resulting table is saved in `results.json`.

## Algorithms and interpretation

- PA `adaptive`: empty/full/partial spatial blocks and uint64 boundary masks on a
  midpoint projection grid, averaged over scrambled Sobol directions on S².
- PA `bvh`: native BVH any-hit queries for isotropic incident directions and uniformly
  sampled enclosing impact disks.
- EHSS: the existing scalar native Numba kernel, AABB candidate rejection, exact
  sphere intersection and elastic specular reflection. No response cache or TM.

The workspace CLI now uses `PreparedStaticCPUEHSS`: compiled stable median
grouping and direct construction of the same padded EHSS boxes. Static runs do
not prepare foreign-overlap exclusivity, intrinsic response blocks, or motion
version metadata. Atom ordering within leaves and the physical tracing kernel
are preserved. A prepared scene owns a copy of its coordinates and is shared
across sampling replicates; changed geometry requires a new static scene.
`PreparedCPUEHSS` remains available for the existing motion-aware research API.
Previously exported ZIP files are snapshots and do not receive this change.

XYZ has no residue ownership. We use median-split spatial groups of up to four atoms
by default, and build the existing native EHSS hierarchy over those groups. This
changes the search partition from the chemical grouping used in the lab benchmark;
it does not change the collision spheres. Therefore the previous IMoS timing ratios
are **not** promised for arbitrary XYZ inputs or for this full CLI startup.

Defaults: 16,384 impact rays per replicate, 64 PA directions, 512×512 PA grid,
3 independent replicates, cap 64, one EHSS thread. These are starting settings,
not a guaranteed error tolerance. Rays/directions/grid must be powers of two.
Check convergence by increasing rays, directions, grid size and/or cap in a new
output directory. Adaptive PA's replicate SD does not capture common grid bias.

For multiple frames, each geometry is rebuilt and calculated independently; this
does not claim general MD acceleration. Replicate seed sequences are the same across
frames, which can help paired comparisons but does not create independent MD frames.

## Output

`summary.csv` (UTF-8 with BOM for Excel) contains one row per input frame:

- PA/EHSS mean and sample SD, in Å²; unrequested methods are blank.
- Mean EHSS omitted-tail bound and total unresolved rays across replicates.
- Compute time including setup once and all replicates, excluding warmup and IO.

`results.json` includes settings, input SHA-256, frame/selection, effective radii,
versions, each replicate's seed, values, stage timings and collision histogram.
`collision_histogram[k]` counts rays with k executed collisions; bin 0 includes
misses, and the cap bin can contain both escaped and unresolved rays.
`collision_contribution_angstrom2[k]` sums resolved EHSS contributions in that bin.

EHSS integrates `w * (1 - incoming · outgoing)` for escaped rays. If a ray remains
unresolved at the cap, its missing contribution lies in `[0, 2w]`. We save the bound
and the corresponding **sampled** interval `[resolved sum, resolved sum + bound]`.
This is not a bound on sampling error or on experimental/model disagreement.

SD and SEM are between-replicate summaries, not guaranteed error bars. They are null
with a single replicate. When both methods and BVH PA are requested, source generation
is shared and timed once. No process-startup speed comparisons are made.

## Sanity check and reproducibility

```sh
python ccs.py examples/ccs/single_carbon.xyz --output results/sphere
```

One carbon with default radii has effective radius 3.10 Å. Both the isotropic PA and
elastic EHSS have analytic value π×3.10² ≈ 30.1907 Å². The finite PA grid can differ
slightly; refine it if necessary. Full project tests are in `tests/test_ccs_cli.py`,
including independent direct-trace agreement on a multiple-scattering geometry.

The sharing ZIP contains the launcher, Python kernel sources, requirements, examples
and this guide. It excludes private molecular inputs, old results and the IMoS binary.

Release validation (2026-09-12): 46 CLI/core tests passed on Windows, Python 3.13.13,
NumPy 2.4.6, SciPy 1.18.0, Numba 0.65.1. The extracted sharing bundle was also
executed on the frozen 327-heavy-atom 1CRN input and on the supplied water example.
These smoke runs establish packaging/input operation, not a new speed comparison.
