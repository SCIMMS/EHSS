from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import time

import numpy as np

from .experiments import PrototypeConfig, run_bump_experiment
from .accelerated2d import warm_up_accelerated_kernels
from .diagnostics2d import MismatchDiagnosticsRow, run_mismatch_diagnostics
from .geometry2d import BumpSpec, SegmentMesh2D, make_circle_cavity
from .metrics import hit_mismatch_count, rasterize_first_hits, relative_frobenius_error, row_conservation_error
from .rays2d import make_lambertian_rays
from .raytrace2d import trace_first_hits
from .transport2d import transport_first_hits, transport_first_hits_event_tiled


@dataclass(frozen=True)
class BenchmarkRow:
    suite: str
    n_segments: int
    rays_per_source: int
    steps: int
    amplitude: float
    width: float
    centers: str
    support_count: int
    layout: str
    support_abs: bool
    profile_kind: str
    neighbor_radius: int
    tile_margin: float
    active_fraction: float
    changed_segment_fraction: float
    full_queries: int
    support_queries: int
    tile_queries: int
    changed_rays: int
    tile_changed_rays: int
    footprint_rays: int
    footprint_overlap_rays: int
    query_speedup: float
    tile_query_speedup: float
    full_time_s: float
    support_time_s: float
    tile_time_s: float
    wall_time_speedup: float
    tile_wall_time_speedup: float
    matrix_error: float
    tile_matrix_error: float
    mismatches: int
    tile_mismatches: int
    row_error_transport: float

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _row_from_config(suite: str, config: PrototypeConfig) -> BenchmarkRow:
    result = run_bump_experiment(config)
    final = result.steps[-1]
    return BenchmarkRow(
        suite=suite,
        n_segments=config.n_segments,
        rays_per_source=config.rays_per_source,
        steps=config.steps,
        amplitude=config.amplitude,
        width=config.width,
        centers=",".join(f"{center:.8g}" for center in config.centers),
        support_count=len(config.centers),
        layout="custom",
        support_abs=config.support_abs,
        profile_kind=config.profile_kind,
        neighbor_radius=config.neighbor_radius,
        tile_margin=config.tile_margin,
        active_fraction=final.active_source_count / config.n_segments,
        changed_segment_fraction=final.changed_segment_count / config.n_segments,
        full_queries=final.full_query_count,
        support_queries=final.transport_query_count,
        tile_queries=final.tile_query_count,
        changed_rays=final.event_count,
        tile_changed_rays=final.tile_event_count,
        footprint_rays=final.tile_footprint_ray_count,
        footprint_overlap_rays=final.tile_footprint_overlap_ray_count,
        query_speedup=final.query_speedup,
        tile_query_speedup=final.tile_query_speedup,
        full_time_s=final.full_wall_time_s,
        support_time_s=final.transport_wall_time_s,
        tile_time_s=final.tile_wall_time_s,
        wall_time_speedup=final.wall_time_speedup,
        tile_wall_time_speedup=final.tile_wall_time_speedup,
        matrix_error=final.matrix_error,
        tile_matrix_error=final.tile_matrix_error,
        mismatches=final.hit_mismatches,
        tile_mismatches=final.tile_hit_mismatches,
        row_error_transport=final.row_error_transport,
    )


def _support_only_row_from_config(suite: str, config: PrototypeConfig) -> BenchmarkRow:
    bumps = config.bumps()
    old_mesh = make_circle_cavity(
        config.n_segments,
        config.radius,
        bumps,
        alpha=0.0,
        support_abs=config.support_abs,
    )
    old_state = trace_first_hits(old_mesh, make_lambertian_rays(old_mesh, config.rays_per_source))
    transport_state = old_state

    final_full_state = old_state
    final_transport = None
    final_full_time = 0.0
    final_transport_time = 0.0
    final_full_matrix = rasterize_first_hits(old_state, old_mesh.n_segments)
    final_transport_matrix = final_full_matrix

    for alpha in np.linspace(1.0 / config.steps, 1.0, config.steps):
        new_mesh = make_circle_cavity(
            config.n_segments,
            config.radius,
            bumps,
            alpha=float(alpha),
            support_abs=config.support_abs,
        )

        t0 = time.perf_counter()
        full_state = trace_first_hits(new_mesh, make_lambertian_rays(new_mesh, config.rays_per_source))
        full_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        transport = transport_first_hits(
            transport_state,
            old_mesh,
            new_mesh,
            config.rays_per_source,
            neighbor_radius=config.neighbor_radius,
        )
        transport_time = time.perf_counter() - t0

        old_mesh = new_mesh
        transport_state = transport.state
        final_full_state = full_state
        final_transport = transport
        final_full_time = full_time
        final_transport_time = transport_time
        final_full_matrix = rasterize_first_hits(full_state, new_mesh.n_segments)
        final_transport_matrix = rasterize_first_hits(transport.state, new_mesh.n_segments)

    if final_transport is None:
        raise RuntimeError("support-only benchmark produced no transport state.")

    return BenchmarkRow(
        suite=suite,
        n_segments=config.n_segments,
        rays_per_source=config.rays_per_source,
        steps=config.steps,
        amplitude=config.amplitude,
        width=config.width,
        centers=",".join(f"{center:.8g}" for center in config.centers),
        support_count=len(config.centers),
        layout="custom",
        support_abs=config.support_abs,
        profile_kind=config.profile_kind,
        neighbor_radius=config.neighbor_radius,
        tile_margin=config.tile_margin,
        active_fraction=final_transport.active_source_count / config.n_segments,
        changed_segment_fraction=final_transport.changed_segment_count / config.n_segments,
        full_queries=final_full_state.query_count,
        support_queries=final_transport.query_count,
        tile_queries=0,
        changed_rays=final_transport.event_count,
        tile_changed_rays=0,
        footprint_rays=0,
        footprint_overlap_rays=0,
        query_speedup=final_full_state.query_count / final_transport.query_count,
        tile_query_speedup=float("nan"),
        full_time_s=final_full_time,
        support_time_s=final_transport_time,
        tile_time_s=float("nan"),
        wall_time_speedup=final_full_time / final_transport_time,
        tile_wall_time_speedup=float("nan"),
        matrix_error=relative_frobenius_error(final_transport_matrix, final_full_matrix),
        tile_matrix_error=float("nan"),
        mismatches=hit_mismatch_count(final_transport.state, final_full_state),
        tile_mismatches=-1,
        row_error_transport=row_conservation_error(final_transport_matrix),
    )


def support_size_rows(
    widths: Iterable[float],
    n_segments: int,
    rays_per_source: int,
    steps: int,
    amplitude: float,
    tile_margin: float,
    neighbor_radius: int,
    profile_kind: str,
) -> list[BenchmarkRow]:
    rows = []
    for width in widths:
        rows.append(
            _row_from_config(
                "support-size",
                PrototypeConfig(
                    n_segments=n_segments,
                    rays_per_source=rays_per_source,
                    steps=steps,
                    amplitude=amplitude,
                    width=float(width),
                    tile_margin=tile_margin,
                    neighbor_radius=neighbor_radius,
                    profile_kind=profile_kind,
                ),
            )
        )
    return rows


def scaling_rows(
    n_segments_values: Iterable[int],
    ray_values: Iterable[int],
    steps: int,
    amplitude: float,
    width: float,
    tile_margin: float,
    neighbor_radius: int,
    profile_kind: str,
) -> list[BenchmarkRow]:
    rows = []
    for n_segments in n_segments_values:
        for rays_per_source in ray_values:
            rows.append(
                _row_from_config(
                    "segment-ray-scaling",
                    PrototypeConfig(
                        n_segments=int(n_segments),
                        rays_per_source=int(rays_per_source),
                        steps=steps,
                        amplitude=amplitude,
                        width=width,
                        tile_margin=tile_margin,
                        neighbor_radius=neighbor_radius,
                        profile_kind=profile_kind,
                    ),
                )
            )
    return rows


def support_only_scaling_rows(
    n_segments_values: Iterable[int],
    ray_values: Iterable[int],
    steps: int,
    amplitude: float,
    width: float,
    tile_margin: float,
    neighbor_radius: int,
    profile_kind: str,
) -> list[BenchmarkRow]:
    rows = []
    for n_segments in n_segments_values:
        for rays_per_source in ray_values:
            rows.append(
                _support_only_row_from_config(
                    "scaling-support",
                    PrototypeConfig(
                        n_segments=int(n_segments),
                        rays_per_source=int(rays_per_source),
                        steps=steps,
                        amplitude=amplitude,
                        width=width,
                        tile_margin=tile_margin,
                        neighbor_radius=neighbor_radius,
                        profile_kind=profile_kind,
                    ),
                )
            )
    return rows


def footprint_padding_rows(
    padding_values: Iterable[float],
    n_segments: int,
    rays_per_source: int,
    steps: int,
    amplitude: float,
    width: float,
    neighbor_radius: int,
    profile_kind: str,
    centers: tuple[float, ...] = (0.0,),
) -> list[BenchmarkRow]:
    rows = []
    for padding in padding_values:
        rows.append(
            _row_from_config(
                "footprint-padding",
                PrototypeConfig(
                    n_segments=n_segments,
                    rays_per_source=rays_per_source,
                    steps=steps,
                    amplitude=amplitude,
                    width=width,
                    tile_margin=float(padding),
                    neighbor_radius=neighbor_radius,
                    profile_kind=profile_kind,
                    centers=centers,
                ),
            )
        )
    return rows


def _support_centers(count: int, layout: str) -> tuple[float, ...]:
    if count <= 0:
        raise ValueError("support count must be positive.")
    if count == 1:
        return (0.0,)
    if layout == "far":
        return tuple(np.linspace(0.0, 2.0 * np.pi, count, endpoint=False))
    if layout == "clustered":
        span = min(1.2, 0.18 * (count - 1))
        return tuple(np.linspace(-0.5 * span, 0.5 * span, count))
    raise ValueError(f"unknown support layout: {layout!r}")


def support_count_footprint_rows(
    support_counts: Iterable[int],
    layouts: Iterable[str],
    n_segments: int,
    rays_per_source: int,
    steps: int,
    amplitude: float,
    width: float,
    neighbor_radius: int,
    profile_kind: str,
    padding: float,
) -> list[BenchmarkRow]:
    rows = []
    for layout in layouts:
        for count in support_counts:
            centers = _support_centers(int(count), layout)
            row = _row_from_config(
                "support-count-footprint",
                PrototypeConfig(
                    n_segments=n_segments,
                    rays_per_source=rays_per_source,
                    steps=steps,
                    amplitude=amplitude,
                    width=width,
                    tile_margin=padding,
                    neighbor_radius=neighbor_radius,
                    profile_kind=profile_kind,
                    centers=centers,
                ),
            )
            row = BenchmarkRow(
                **{
                    **row.as_dict(),
                    "support_count": int(count),
                    "layout": layout,
                }
            )
            rows.append(row)
    return rows


def neighbor_radius_rows(
    radii: Iterable[int],
    n_segments: int,
    rays_per_source: int,
    steps: int,
    amplitude: float,
    width: float,
    tile_margin: float,
    profile_kind: str,
) -> list[BenchmarkRow]:
    rows = []
    for radius in radii:
        rows.append(
            _support_only_row_from_config(
                "neighbor-radius",
                PrototypeConfig(
                    n_segments=n_segments,
                    rays_per_source=rays_per_source,
                    steps=steps,
                    amplitude=amplitude,
                    width=width,
                    tile_margin=tile_margin,
                    neighbor_radius=int(radius),
                    profile_kind=profile_kind,
                ),
            )
        )
    return rows


def failure_rows(
    n_segments: int,
    rays_per_source: int,
    steps: int,
    amplitude: float,
    width: float,
    tile_margin: float,
    neighbor_radius: int,
    profile_kind: str,
) -> list[BenchmarkRow]:
    rows = [
        _row_from_config(
            "nonmonotone-dent",
            PrototypeConfig(
                n_segments=n_segments,
                rays_per_source=rays_per_source,
                steps=steps,
                amplitude=-abs(amplitude),
                width=width,
                tile_margin=tile_margin,
                neighbor_radius=neighbor_radius,
                support_abs=True,
                profile_kind=profile_kind,
            ),
        )
    ]
    rows.append(
        hidden_support_violation_row(
            n_segments=n_segments,
            rays_per_source=rays_per_source,
            amplitude=abs(amplitude),
            width=width,
            tile_margin=tile_margin,
            neighbor_radius=neighbor_radius,
        )
    )
    return rows


def multi_support_footprint_rows(
    padding_values: Iterable[float],
    n_segments: int,
    rays_per_source: int,
    steps: int,
    amplitude: float,
    width: float,
    neighbor_radius: int,
    profile_kind: str,
) -> list[BenchmarkRow]:
    centers = (-0.22, 0.22, np.pi)
    rows = []
    for padding in padding_values:
        rows.append(
            _row_from_config(
                "multi-support-footprint",
                PrototypeConfig(
                    n_segments=n_segments,
                    rays_per_source=rays_per_source,
                    steps=steps,
                    amplitude=amplitude,
                    width=width,
                    tile_margin=float(padding),
                    neighbor_radius=neighbor_radius,
                    profile_kind=profile_kind,
                    centers=centers,
                ),
            )
        )
    return rows


def hidden_support_violation_row(
    n_segments: int,
    rays_per_source: int,
    amplitude: float,
    width: float,
    tile_margin: float,
    neighbor_radius: int,
) -> BenchmarkRow:
    primary = (BumpSpec(center=0.0, amplitude=amplitude, width=width, label=1),)
    actual = (
        BumpSpec(center=0.0, amplitude=amplitude, width=width, label=1),
        BumpSpec(center=np.pi, amplitude=amplitude, width=width, label=2),
    )

    old_mesh = make_circle_cavity(n_segments, bumps=primary, alpha=0.0)
    actual_new = make_circle_cavity(n_segments, bumps=actual, alpha=1.0)
    reported_new = make_circle_cavity(n_segments, bumps=primary, alpha=1.0)
    new_mesh = SegmentMesh2D(
        actual_new.vertices,
        reported_new.support_mask,
        reported_new.support_labels,
        actual_new.vertex_displacement,
    )

    old_state = trace_first_hits(old_mesh, make_lambertian_rays(old_mesh, rays_per_source))

    import time

    t0 = time.perf_counter()
    full_state = trace_first_hits(new_mesh, make_lambertian_rays(new_mesh, rays_per_source))
    full_time_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    support = transport_first_hits(old_state, old_mesh, new_mesh, rays_per_source, neighbor_radius)
    support_time_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    tile = transport_first_hits_event_tiled(
        old_state,
        old_mesh,
        new_mesh,
        rays_per_source,
        neighbor_radius=neighbor_radius,
        tile_margin=tile_margin,
    )
    tile_time_s = time.perf_counter() - t0

    f_full = rasterize_first_hits(full_state, n_segments)
    f_support = rasterize_first_hits(support.state, n_segments)
    f_tile = rasterize_first_hits(tile.state, n_segments)

    return BenchmarkRow(
        suite="hidden-support-violation",
        n_segments=n_segments,
        rays_per_source=rays_per_source,
        steps=1,
        amplitude=amplitude,
        width=width,
        centers=f"0,{np.pi:.8g}",
        support_count=2,
        layout="hidden-support-violation",
        support_abs=False,
        profile_kind="gaussian",
        neighbor_radius=neighbor_radius,
        tile_margin=tile_margin,
        active_fraction=support.active_source_count / n_segments,
        changed_segment_fraction=support.changed_segment_count / n_segments,
        full_queries=full_state.query_count,
        support_queries=support.query_count,
        tile_queries=tile.query_count,
        changed_rays=support.event_count,
        tile_changed_rays=tile.event_count,
        footprint_rays=tile.footprint_ray_count,
        footprint_overlap_rays=tile.footprint_overlap_ray_count,
        query_speedup=full_state.query_count / support.query_count,
        tile_query_speedup=full_state.query_count / tile.query_count if tile.query_count else float("inf"),
        full_time_s=full_time_s,
        support_time_s=support_time_s,
        tile_time_s=tile_time_s,
        wall_time_speedup=full_time_s / support_time_s,
        tile_wall_time_speedup=full_time_s / tile_time_s if tile_time_s else float("inf"),
        matrix_error=relative_frobenius_error(f_support, f_full),
        tile_matrix_error=relative_frobenius_error(f_tile, f_full),
        mismatches=hit_mismatch_count(support.state, full_state),
        tile_mismatches=hit_mismatch_count(tile.state, full_state),
        row_error_transport=row_conservation_error(f_support),
    )


def write_csv(rows: list[BenchmarkRow], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].as_dict()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())
    return output_path


def print_rows(rows: list[BenchmarkRow]) -> None:
    print(
        "suite                 layout     nsup  pad    nbr  Nseg  Nrays  width   "
        "q_full/q_FP  t_full/t_FP  mm_FP  changed  fp_rays  fp/change  fp_overlap"
    )
    for row in rows:
        lower_bound_ratio = row.footprint_rays / row.tile_changed_rays if row.tile_changed_rays else float("inf")
        print(
            f"{row.suite:20s} "
            f"{row.layout:9s} "
            f"{row.support_count:4d} "
            f"{row.tile_margin:5.3f} "
            f"{row.neighbor_radius:3d} "
            f"{row.n_segments:5d} "
            f"{row.rays_per_source:6d} "
            f"{row.width:6.3f} "
            f"{row.tile_query_speedup:11.3f} "
            f"{row.tile_wall_time_speedup:11.3f} "
            f"{row.tile_mismatches:5d} "
            f"{row.tile_changed_rays:7d} "
            f"{row.footprint_rays:7d} "
            f"{lower_bound_ratio:9.2f} "
            f"{row.footprint_overlap_rays:10d}"
        )


def write_diagnostics_csv(rows: list[MismatchDiagnosticsRow], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].as_dict()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())
    return output_path


def print_diagnostics(rows: list[MismatchDiagnosticsRow]) -> None:
    print(
        "alpha  mismatch  matrix_err  outside_band  target_grazing  "
        "OO  OD  DO  DD  top old->full"
    )
    for row in rows:
        print(
            f"{row.alpha:5.2f} "
            f"{row.total_mismatches:9d} "
            f"{row.matrix_error:10.2e} "
            f"{row.full_target_outside_band:12d} "
            f"{row.target_grazing:14d} "
            f"{row.block_oo:3d} "
            f"{row.block_od:3d} "
            f"{row.block_do:3d} "
            f"{row.block_dd:3d} "
            f"{row.top_old_to_full}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run 2D support-exchange benchmark suites.")
    parser.add_argument(
        "suite",
        choices=(
            "support-size",
            "scaling",
            "scaling-support",
            "footprint-padding",
            "multi-support-footprint",
            "support-count-footprint",
            "neighbor-radius",
            "failure",
            "diagnose",
            "all",
        ),
        help="benchmark suite to run",
    )
    parser.add_argument("--segments", type=int, default=64, help="single Nseg for support/failure suites")
    parser.add_argument("--rays", type=int, default=24, help="single ray count for support/failure suites")
    parser.add_argument("--steps", type=int, default=3, help="continuation steps per case")
    parser.add_argument("--amplitude", type=float, default=0.14, help="bump amplitude magnitude")
    parser.add_argument("--width", type=float, default=0.24, help="single width for scaling/failure suites")
    parser.add_argument(
        "--profile",
        choices=("gaussian", "compact-cosine"),
        default="gaussian",
        help="radial bump profile",
    )
    parser.add_argument(
        "--widths",
        type=_parse_floats,
        default=(0.10, 0.16, 0.24, 0.34),
        help="comma-separated support widths for support-size suite",
    )
    parser.add_argument(
        "--segment-grid",
        type=_parse_ints,
        default=(64, 96, 128),
        help="comma-separated Nseg values for scaling suite",
    )
    parser.add_argument(
        "--ray-grid",
        type=_parse_ints,
        default=(24, 48),
        help="comma-separated ray counts for scaling suite",
    )
    parser.add_argument("--tile-margin", type=float, default=0.01, help="event/tile angular margin")
    parser.add_argument(
        "--padding-grid",
        type=_parse_floats,
        default=(0.0, 0.002, 0.005, 0.01, 0.02),
        help="comma-separated angular padding values for footprint sweeps",
    )
    parser.add_argument(
        "--support-counts",
        type=_parse_ints,
        default=(1, 2, 4, 8),
        help="comma-separated support counts for support-count footprint suite",
    )
    parser.add_argument(
        "--layouts",
        default="far,clustered",
        help="comma-separated support layouts: far,clustered",
    )
    parser.add_argument("--neighbor-radius", type=int, default=1, help="cyclic support expansion radius")
    parser.add_argument("--grazing-threshold", type=float, default=0.05, help="absolute cosine threshold")
    parser.add_argument(
        "--radii",
        type=_parse_ints,
        default=(0, 1, 2, 3, 4, 5),
        help="comma-separated neighbor radii for neighbor-radius suite",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark2d.csv"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows: list[BenchmarkRow] = []
    warm_up_accelerated_kernels()

    if args.suite == "diagnose":
        diagnostics = list(
            run_mismatch_diagnostics(
                PrototypeConfig(
                    n_segments=args.segments,
                    rays_per_source=args.rays,
                    steps=args.steps,
                    amplitude=args.amplitude,
                    width=args.width,
                    neighbor_radius=args.neighbor_radius,
                    tile_margin=args.tile_margin,
                    profile_kind=args.profile,
                ),
                grazing_threshold=args.grazing_threshold,
            )
        )
        print_diagnostics(diagnostics)
        path = write_diagnostics_csv(diagnostics, args.output)
        print(f"csv written: {path}")
        return 0

    if args.suite in {"support-size", "all"}:
        rows.extend(
            support_size_rows(
                args.widths,
                args.segments,
                args.rays,
                args.steps,
                args.amplitude,
                args.tile_margin,
                args.neighbor_radius,
                args.profile,
            )
        )

    if args.suite in {"scaling", "all"}:
        rows.extend(
            scaling_rows(
                args.segment_grid,
                args.ray_grid,
                args.steps,
                args.amplitude,
                args.width,
                args.tile_margin,
                args.neighbor_radius,
                args.profile,
            )
        )

    if args.suite in {"scaling-support", "all"}:
        rows.extend(
            support_only_scaling_rows(
                args.segment_grid,
                args.ray_grid,
                args.steps,
                args.amplitude,
                args.width,
                args.tile_margin,
                args.neighbor_radius,
                args.profile,
            )
        )

    if args.suite in {"footprint-padding", "all"}:
        rows.extend(
            footprint_padding_rows(
                args.padding_grid,
                args.segments,
                args.rays,
                args.steps,
                args.amplitude,
                args.width,
                args.neighbor_radius,
                args.profile,
            )
        )

    if args.suite in {"multi-support-footprint", "all"}:
        rows.extend(
            multi_support_footprint_rows(
                args.padding_grid,
                args.segments,
                args.rays,
                args.steps,
                args.amplitude,
                args.width,
                args.neighbor_radius,
                args.profile,
            )
        )

    if args.suite in {"support-count-footprint", "all"}:
        layouts = tuple(part.strip() for part in args.layouts.split(",") if part.strip())
        rows.extend(
            support_count_footprint_rows(
                args.support_counts,
                layouts,
                args.segments,
                args.rays,
                args.steps,
                args.amplitude,
                args.width,
                args.neighbor_radius,
                args.profile,
                args.tile_margin,
            )
        )

    if args.suite in {"neighbor-radius", "all"}:
        rows.extend(
            neighbor_radius_rows(
                args.radii,
                args.segments,
                args.rays,
                args.steps,
                args.amplitude,
                args.width,
                args.tile_margin,
                args.profile,
            )
        )

    if args.suite in {"failure", "all"}:
        rows.extend(
            failure_rows(
                args.segments,
                args.rays,
                args.steps,
                args.amplitude,
                args.width,
                args.tile_margin,
                args.neighbor_radius,
                args.profile,
            )
        )

    if not rows:
        raise RuntimeError("no benchmark rows were produced.")

    print_rows(rows)
    path = write_csv(rows, args.output)
    print(f"csv written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
