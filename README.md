# EHSS: molecular PA and exact hard sphere scattering

CPU calculation of projection approximation (PA) and exact hard sphere scattering
(EHSS) from XYZ coordinates, with frozen evidence for the accompanying JCIM draft.
The active EHSS path combines analytical atom-sphere intersections, BVH traversal,
scrambled Sobol incidence and a reported bound for unresolved sampled paths.

Authors: Minsu Kim, Minwook Hwang and Jongcheol Seo (corresponding author,
jongcheol.seo@postech.ac.kr), POSTECH.

## Run an XYZ calculation

Python 3.11–3.13 is supported by this distribution; the recorded execution platform
is Windows/Python 3.13. From this checkout:

```sh
python -m pip install -r requirements-ccs.txt
python ccs.py examples/ccs/water.xyz --method ehss --rays 4096 --repeats 3 --output results/water
python ccs.py "data/inputs/evaluation18/*.xyz" --method ehss --output results/panel
python ccs.py --help
```

The output directory must be new. Output includes `summary.csv` and `results.json`.
First-use JIT compilation can dominate short calculations. Batch files in one
invocation to share startup. For an installed console command, use `python -m pip
install .`, then `ccs-xyz --help`. This repository is not a PyPI release.

The [CLI guide](docs/CCS_CLI.md) covers adaptive/BVH PA, radii, frames, units and
thread controls. Defaults are an elastic/specular hard-sphere model, not a
trajectory-method potential. No IMoS executable or GPU is required for our code.

**Publication protocol versus CLI:** the general CLI uses a different repeat-seed
stride and does not reproduce the publication's three-block pooling and adaptive
collision-cap wrapper by merely matching `--rays`. Use the frozen worker and
specifications in the evidence snapshot for exact benchmark reproduction.

## Results and their scope

The external comparator is IMoS 1.13 on the same laboratory i5-12400F computer with
one requested thread. Sampling methods differ: comparison is against a common
relative-RMSE target, not identical ray counts or exactly equal achieved RMSE.

| Evidence | Result |
| --- | --- |
| Six development structures | Matched native exhaustive/BVH and IID/Sobol controls separate traversal and sampling effects |
| 1CRN and 1LYZ, Stage C | Both methods pass independently evaluated 1% and 0.5% targets; repeated single and batch times retained |
| 18 additional structures, Stage G | 720 independent estimates, 20 per method/structure; five separate timing blocks |
| Practical 2% target | Our method passes 17/18, IMoS 16/18; 15 jointly pass; three method/structure outcomes are uncertain |
| Jointly passing single-input cases | IMoS/own median complete-process time ratios: 2.58–14.55 |
| Actual all-18 batch | 103.662 s IMoS versus 4.330 s own; 23.94 ratio includes uncertain accuracy cases and is a latency observation |

The 2% target was chosen after earlier stricter-target work; final settings used
calibration only and were frozen before new final seeds. Earlier failures and
uncertainties remain in the archive. Pointwise intervals do not establish a
population-wide guarantee. No general speed ranking over all EHSS software is
claimed. EHSSrot direct comparison is outside this version; it is not a pending
validation requirement. MD reuse, TM, GPU and experimental-CCS accuracy are not
performance claims of this release.

## Repository map

| Location | Contents |
| --- | --- |
| `ccs.py`, `src/`, `tests/`, `examples/` | Frozen numerical source, six core test modules and XYZ examples |
| [`data/inputs/evaluation18/`](data/inputs/evaluation18) | Exact standardized XYZ inputs for the 18-structure final panel |
| [`data/benchmark/`](data/benchmark) | Browseable final raw rows, accuracy/timing CSVs, selection and reference summaries, provenance |
| [`data/evidence/`](data/evidence) | Immutable full evidence ZIP, original verification record and historical environment |
| [`figures/`](figures) | Four figures and TOC graphic, PNG/PDF/SVG |
| [`manuscript/`](manuscript) | English main text and SI, Word/PDF and editable text sources |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | Integrity checks, offline analyses and test instructions |

The numerical namespace remains `support_exchange_transport` to preserve frozen
source bytes. Some dependency modules contain earlier research paths; their
presence is not a recommendation or a performance claim. The dedicated packaging
metadata and README are new; numerical code is unchanged from snapshot v4.

```sh
python tools/verify_repository.py
python tools/prepare_evidence.py --output work/evidence
```

The ZIP is retained byte-for-byte (about 96 MB) to preserve the original source,
inputs, calibration, final runs and audit trail. Download it through GitHub's Raw
or Download control if a browser preview is unavailable. It is below GitHub's
individual-file limit and uses ordinary Git, with no LFS service dependency.

## Manuscript and reuse status

This repository was populated as a private research repository; no visibility
change, public data deposit or DOI was created. The draft still contains marked
data/software-availability, author-contribution and acknowledgement fields.
Repository hosting does not settle those publication statements.

See [RIGHTS.md](RIGHTS.md) for reuse and third-party provenance. IMoS software,
MATLAB Runtime, JIT caches, credentials and unrelated private working materials
are not part of the intended repository payload. The archived evidence contains
historical execution paths and environment labels for provenance.
