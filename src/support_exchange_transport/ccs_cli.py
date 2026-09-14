"""XYZ front end for the validated PA and scalar CPU EHSS kernels.

No benchmark output files, molecular topology, IMoS installation or network
access are required. Distances are normalized to angstrom before calculation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import glob
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np

VERSION = "0.1.0"
# Mantina et al. (2009), Table 12, DOI 10.1021/jp8111556.
# H is the Rowland/Taylor 1.10 value, not Bondi's older 1.20 value.
# C/N/O/S exactly match the existing heavy-atom benchmark scenario.
VDW_RADII = {"H": 1.10, "He": 1.40, "C": 1.70, "N": 1.55,
             "O": 1.52, "F": 1.47, "Si": 2.10, "P": 1.80,
             "S": 1.80, "Cl": 1.75, "Se": 1.90, "Br": 1.83, "I": 1.98}


@dataclass
class XYZFrame:
    number: int
    comment: str
    elements: tuple[str, ...]
    coordinates: np.ndarray


def element_symbol(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z]{1,2}", value):
        raise ValueError(f"Expected an element symbol, got {value!r}")
    return value[0].upper() + value[1:].lower()


def read_xyz(path):
    """Strict standard XYZ; read every frame so malformed tails are not ignored."""
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    frames = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        start = i
        if not re.fullmatch(r"[1-9][0-9]*", lines[i].strip()):
            raise ValueError(f"{path}:{i+1}: expected a positive atom count")
        count = int(lines[i].strip())
        if i + count + 1 >= len(lines):
            raise ValueError(f"{path}:{i+1}: incomplete XYZ frame (comment and {count} atom rows required)")
        comment = lines[i+1]
        if "Lattice=" in comment or "Properties=" in comment:
            raise ValueError(f"{path}:{i+2}: extended/periodic XYZ is not supported; export a standard isolated molecule")
        elements, coordinates = [], []
        for j in range(i+2, i+2+count):
            fields = lines[j].split()
            if len(fields) != 4:
                raise ValueError(f"{path}:{j+1}: expected exactly 'Element x y z'")
            try:
                symbol = element_symbol(fields[0])
                xyz = [float(x.replace('D', 'E').replace('d', 'e')) for x in fields[1:]]
            except ValueError as exc:
                raise ValueError(f"{path}:{j+1}: invalid element or coordinates") from exc
            if not np.isfinite(xyz).all():
                raise ValueError(f"{path}:{j+1}: coordinates must be finite")
            elements.append(symbol); coordinates.append(xyz)
        frames.append(XYZFrame(len(frames)+1, comment, tuple(elements), np.array(coordinates)))
        i = start + count + 2
    if not frames:
        raise ValueError(f"{path}: no XYZ frames")
    return frames


def radius_table(path=None):
    result = VDW_RADII.copy()
    if path:
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(raw, dict) or not raw:
            raise ValueError("Radii JSON must be a nonempty object: {\"C\": 1.70, ...}")
        seen = set()
        for name, value in raw.items():
            symbol = element_symbol(name)
            if symbol in seen:
                raise ValueError(f"Duplicate normalized radius key: {symbol}")
            seen.add(symbol)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Radius for {symbol} must be a positive finite number in angstrom")
            result[symbol] = float(value)
    return result


def prepare_frame(frame, radii, probe_radius, units="angstrom", exclude_hydrogen=False):
    from .inside_out_pa import Spheres
    keep = [i for i,e in enumerate(frame.elements) if not (exclude_hydrogen and e == "H")]
    if not keep:
        raise ValueError(f"Frame {frame.number}: atom selection is empty")
    elements = [frame.elements[i] for i in keep]
    unknown = sorted(set(elements)-radii.keys())
    if unknown:
        raise ValueError(f"Frame {frame.number}: no radius for {', '.join(unknown)}; supply --radii-json")
    xyz = frame.coordinates[keep] * (10. if units == "nm" else 1.)
    if not np.isfinite(xyz).all():
        raise ValueError("Coordinates overflow after unit conversion")
    if len(np.unique(xyz, axis=0)) != len(xyz):
        raise ValueError(f"Frame {frame.number}: coincident selected atom centers; check duplicate atoms")
    # Translation avoids unnecessary loss of precision for offset geometries.
    offset = xyz[0] + np.mean(xyz-xyz[0], axis=0)
    effective = np.array([radii[e]+probe_radius for e in elements])
    spheres = Spheres(xyz-offset, effective, deduplicate=False)
    info = dict(frame=frame.number, comment=frame.comment, input_atoms=len(frame.elements),
                selected_atoms=len(keep), selected_atom_rows_1based=[i+1 for i in keep],
                element_counts=dict(Counter(elements)), recenter_offset_angstrom=offset.tolist(),
                effective_radii_angstrom={e:radii[e]+probe_radius for e in sorted(set(elements))})
    return spheres, info


def spatial_groups(spheres, leaf_size=4):
    """Reuse PA's median-split leaf order; XYZ carries no chemical ownership."""
    from .pa_spatial import SphereIndex
    tree = SphereIndex(spheres, leaf_size=leaf_size).tree
    groups = np.empty(len(spheres.radii), dtype=np.int64)
    group = 0
    for first, count in zip(tree.first, tree.count):
        if count:
            groups[tree.ids[first:first+count]] = group
            group += 1
    return groups


def adaptive_pa(spheres, directions, grid_size, seed):
    from scipy.stats import qmc
    from .inside_out_pa import unit_sphere
    from .pa_walsh_geometry import project_grid
    from .pa_sparse_blocks import block_disks, block_count
    tick = time.perf_counter()
    ds = unit_sphere(qmc.Sobol(2, scramble=True, seed=seed).random_base2(directions.bit_length()-1))
    source_s = time.perf_counter()-tick
    projection_s = query_s = readout_s = 0.
    values = []
    for direction in ds:
        tick = time.perf_counter(); xy, lower, spacing, *_ = project_grid(spheres, direction, grid_size)
        projection_s += time.perf_counter()-tick
        tick = time.perf_counter(); full, partial, _ = block_disks(xy, spheres.radii, lower, spacing, grid_size)
        query_s += time.perf_counter()-tick
        tick = time.perf_counter(); values.append(float(block_count(full, partial))*spacing**2)
        readout_s += time.perf_counter()-tick
    return dict(pa_angstrom2=float(np.mean(values)), source_s=source_s, projection_s=projection_s,
                query_s=query_s, readout_s=readout_s)


def summarize(values):
    a = np.asarray(values, dtype=float)
    sd = float(a.std(ddof=1)) if len(a)>1 else None
    return dict(mean=float(a.mean()), sd=sd, sem=sd/math.sqrt(len(a)) if sd is not None else None,
                replicates=len(a))


def solve_frame(spheres, args, progress=True):
    from .ehss_state import external_source
    from .inside_out_pa import Source
    from .pa_spatial import SphereIndex
    ehss = args.method in ("ehss", "both")
    pa = args.method in ("pa", "both")
    start = time.perf_counter()
    engine = None; index = None
    tick = time.perf_counter()
    if ehss:
        from .ehss_static import PreparedStaticCPUEHSS
        engine = PreparedStaticCPUEHSS(spheres, leaf_size=args.leaf_size,
                                      threads=args.threads).engine
    ehss_setup_s = time.perf_counter()-tick if ehss else 0.
    tick = time.perf_counter()
    if pa and args.pa_method == "bvh":
        index = SphereIndex(spheres, leaf_size=args.leaf_size)
    pa_setup_s = time.perf_counter()-tick if index else 0.
    rows=[]
    for repeat in range(args.repeats):
        seed = (args.seed+104729*repeat) % (2**32)
        row=dict(repeat=repeat+1, seed=seed)
        if progress:
            print(f"  replicate {repeat+1}/{args.repeats} (seed {seed})", file=sys.stderr, flush=True)
        if pa and args.pa_method == "adaptive":
            row['pa'] = adaptive_pa(spheres, args.directions, args.grid_size, seed)
        if ehss or index is not None:
            tick = time.perf_counter()
            source = external_source(spheres, args.rays.bit_length()-1, seed)
            row['impact_source_s'] = time.perf_counter()-tick
            if index is not None:
                tick = time.perf_counter()
                hits, boxes, tests = index.query(Source(source.origins, source.directions, source.last_atom,
                                                       float(source.weights.sum())))
                query_s = time.perf_counter()-tick; tick = time.perf_counter()
                area = float(source.weights[hits].sum())
                row['pa']=dict(pa_angstrom2=area, query_s=query_s, readout_s=time.perf_counter()-tick,
                               box_tests=int(boxes), sphere_tests=int(tests))
            if ehss:
                tick = time.perf_counter(); result=engine.trace(source, args.max_collisions)
                query_s = time.perf_counter()-tick; tick = time.perf_counter()
                contributions=result.contributions
                omega=float(contributions.sum()); tail=result.tail_bound
                counts=np.bincount(result.bounces, minlength=args.max_collisions+1)
                ledger=np.bincount(result.bounces, weights=contributions, minlength=args.max_collisions+1)
                row['ehss']=dict(ehss_angstrom2=omega, pa_from_same_rays_angstrom2=result.pa,
                    tail_bound_angstrom2=tail, sampled_interval_angstrom2=[omega,omega+tail],
                    unresolved_rays=int(result.unresolved.sum()), query_s=query_s,
                    collision_histogram=counts.tolist(), collision_contribution_angstrom2=ledger.tolist(),
                    box_tests=int(result.metrics['box_tests']), sphere_tests=int(result.metrics['sphere_tests']))
                row['ehss']['readout_s']=time.perf_counter()-tick
                if progress and row['ehss']['unresolved_rays']:
                    print(f"  WARNING: {row['ehss']['unresolved_rays']} unresolved rays; omitted contribution <= {tail:.6g} A^2. Increase --max-collisions.",file=sys.stderr)
        rows.append(row)
    total=time.perf_counter()-start
    summary={}
    if pa:summary['pa_angstrom2']=summarize([r['pa']['pa_angstrom2'] for r in rows])
    if ehss:
        summary['ehss_angstrom2']=summarize([r['ehss']['ehss_angstrom2'] for r in rows])
        summary['ehss_tail_bound_mean_angstrom2']=float(np.mean([r['ehss']['tail_bound_angstrom2'] for r in rows]))
        summary['ehss_unresolved_rays_total']=sum(r['ehss']['unresolved_rays'] for r in rows)
    return dict(summary=summary, runs=rows, timings=dict(ehss_setup_s=ehss_setup_s,
                pa_setup_s=pa_setup_s, all_replicates_with_setup_s=total))


def parser():
    p=argparse.ArgumentParser(description="PA/EHSS for isolated molecules in standard XYZ files (results in A^2).")
    p.add_argument('xyz', nargs='+', help='XYZ paths; quoted wildcard patterns also work on Windows')
    p.add_argument('--version', action='version', version='ccs-xyz '+VERSION)
    p.add_argument('--method', choices=['pa','ehss','both'], default='both')
    p.add_argument('--pa-method', choices=['adaptive','bvh'], default='adaptive')
    p.add_argument('--rays',type=int,default=16384,help='EHSS/BVH PA rays per replicate, power of two (default: 16384)')
    p.add_argument('--directions',type=int,default=64,help='Adaptive PA directions per replicate, power of two (default: 64)')
    p.add_argument('--grid-size',type=int,default=512,help='Adaptive PA square midpoint grid, power of two (default: 512)')
    p.add_argument('--repeats',type=int,default=3,help='Independent scrambled Sobol replicates (default: 3)')
    p.add_argument('--seed',type=int,default=190003)
    p.add_argument('--threads',type=int,default=1,help='EHSS trace threads; PA/geometry remain serial')
    p.add_argument('--leaf-size',type=int,default=4,help='Maximum spatial group size for XYZ BVH (default: 4)')
    p.add_argument('--max-collisions',type=int,default=64)
    p.add_argument('--probe-radius',type=float,default=1.4,help='Added probe radius in angstrom (default: 1.4, He scenario)')
    p.add_argument('--radii-json',type=Path,help='Overrides/additions to atomic VDW radii in angstrom, never effective radii')
    p.add_argument('--units',choices=['angstrom','nm'],default='angstrom',help='Input coordinate units only')
    p.add_argument('--exclude-hydrogen',action='store_true',help='Explicitly omit H; default includes every supplied atom')
    f=p.add_mutually_exclusive_group()
    f.add_argument('--frame',type=int,default=1,help='1-based frame selection (default: first frame)')
    f.add_argument('--all-frames',action='store_true',help='Calculate every frame independently, no MD cache reuse')
    p.add_argument('--output',type=Path,default=Path('ccs_results'),help='New result directory; existing directory is preserved')
    return p


def validate_args(args):
    for name,low,high in [('rays',8,2**22),('directions',1,2**16),('grid_size',8,8192)]:
        value=getattr(args,name)
        if not low<=value<=high or value&(value-1):
            raise ValueError(f"--{name.replace('_','-')} must be a power of two in {low}..{high}")
    for name in ['repeats','threads','leaf_size','max_collisions','frame']:
        if getattr(args,name)<1:raise ValueError(f"--{name.replace('_','-')} must be positive")
    if args.max_collisions>100000:raise ValueError('--max-collisions must be <= 100000')
    if not 0<=args.seed<2**32:raise ValueError('--seed must be in 0..4294967295')
    if not math.isfinite(args.probe_radius) or args.probe_radius<0:
        raise ValueError('--probe-radius must be finite and nonnegative')
    if args.output.exists():raise ValueError(f"Output already exists: {args.output}; choose a new --output directory")


def execute(args):
    validate_args(args)
    import numba
    import scipy
    from .ehss_cpu import check_threads
    check_threads(args.threads)
    started=time.perf_counter(); table=radius_table(args.radii_json)
    jobs=[];seen=set()
    for pattern in args.xyz:
        matches=[pattern] if Path(pattern).is_file() else sorted(glob.glob(pattern))
        if not matches:raise ValueError(f"No XYZ files match: {pattern}")
        for match in matches:
            path=Path(match).resolve()
            if path in seen:continue
            seen.add(path);frames=read_xyz(path)
            if not args.all_frames and args.frame>len(frames):
                raise ValueError(f"{path.name}: --frame {args.frame} exceeds {len(frames)} frames")
            selected=frames if args.all_frames else [frames[args.frame-1]]
            if len(frames)>1 and not args.all_frames:
                print(f"{path.name}: using frame {args.frame}/{len(frames)}; use --all-frames for all.",file=sys.stderr)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            for frame in selected:
                spheres,info=prepare_frame(frame,table,args.probe_radius,args.units,args.exclude_hydrogen)
                jobs.append((spheres,dict(input_file=str(path),input_sha256=digest,total_frames=len(frames),**info)))
    print(f"Validated {len(jobs)} frame(s). Preparing native kernels; the first launch may take a while...",file=sys.stderr,flush=True)
    # Compile the same signatures outside scientific compute timing.
    from .inside_out_pa import Spheres
    warm=argparse.Namespace(**vars(args));warm.rays=8;warm.directions=1;warm.grid_size=8;warm.repeats=1
    tick=time.perf_counter()
    solve_frame(Spheres(np.array([[0.,0.,0.],[2.,0.,0.]]),np.array([1.,1.])),warm,progress=False)
    warmup_s=time.perf_counter()-tick
    records=[]
    for spheres,info in jobs:
        print(f"{Path(info['input_file']).name}, frame {info['frame']}, {info['selected_atoms']} atoms",file=sys.stderr,flush=True)
        record=dict(**info,**solve_frame(spheres,args));records.append(record)
        for name in ['pa_angstrom2','ehss_angstrom2']:
            if name in record['summary']:
                v=record['summary'][name];sd='n/a' if v['sd'] is None else f"{v['sd']:.6f}"
                print(f"  {name.split('_')[0].upper()}: {v['mean']:.6f} A^2 (replicate SD {sd})")
        print(f"  Compute: {record['timings']['all_replicates_with_setup_s']:.4f} s (setup + all replicates; warmup excluded)")
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    payload=dict(schema_version=1,tool='ccs-xyz',version=VERSION,created_utc=datetime.now(timezone.utc).isoformat(),
        settings=config,atomic_radii_angstrom=table,radii_source='Mantina et al. 2009 Table 12, subset; DOI 10.1021/jp8111556; user overrides recorded in table',
        environment=dict(python=platform.python_version(),platform=platform.platform(),numpy=np.__version__,
                         scipy=scipy.__version__,numba=numba.__version__),
        warmup_s=warmup_s,elapsed_before_output_s=time.perf_counter()-started,
        notes=['Static elastic hard-sphere model; atomic VDW radius + probe radius. Not TM or fitted IMoS/EHSSrot parameters.',
               'XYZ spatial groups, no inferred bonds/residues; no support response reuse.',
               'Independent Sobol scrambles across replicates. SD/SEM measure replicate variation, not total error or a guaranteed confidence interval.',
               'Adaptive PA has finite-grid and directional error; replicate SD does not estimate grid bias.',
               'EHSS sampled interval covers omitted cap survivors only, not sampling error or physical model error.',
               'Warmup/import/IO excluded from per-frame compute timing; setup is once per frame, shared across replicates.',
               'For both + BVH PA, the impact source is shared with EHSS and timed once.'],records=records)
    # Reserve a fresh destination only after successful calculation, never overwrite inputs/results.
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    columns=['input_file','frame','atoms','pa_mean_A2','pa_sd_A2','ehss_mean_A2','ehss_sd_A2',
             'ehss_tail_bound_mean_A2','unresolved_rays_total','compute_all_replicates_s']
    with (args.output/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for r in records:
            s=r['summary'];pa=s.get('pa_angstrom2',{});eh=s.get('ehss_angstrom2',{})
            writer.writerow(dict(input_file=r['input_file'],frame=r['frame'],atoms=r['selected_atoms'],
                pa_mean_A2=pa.get('mean'),pa_sd_A2=pa.get('sd'),ehss_mean_A2=eh.get('mean'),ehss_sd_A2=eh.get('sd'),
                ehss_tail_bound_mean_A2=s.get('ehss_tail_bound_mean_angstrom2'),
                unresolved_rays_total=s.get('ehss_unresolved_rays_total'),
                compute_all_replicates_s=r['timings']['all_replicates_with_setup_s']))
    print(f"Saved {args.output.resolve() / 'summary.csv'}")
    print(f"Saved {args.output.resolve() / 'results.json'}")
    return payload


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    try:
        execute(args)
    except (ValueError,OSError) as exc:
        p.exit(2,f"Error: {exc}\n")
    except KeyboardInterrupt:
        p.exit(130,'Interrupted.\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
