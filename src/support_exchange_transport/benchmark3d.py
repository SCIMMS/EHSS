from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from .bvh3d import BVH3D
from .accelerated3d import warm_up_accelerated_3d
from .geometry3d import SphericalCapBump, make_default_cap_bump, make_sphere_cavity, normalize_vectors
from .metrics import hit_mismatch_count, rasterize_first_hits, relative_frobenius_error, row_conservation_error
from .radiosity import make_radiosity_case, max_abs_error, relative_l2_error, solve_radiosity
from .rays3d import make_cosine_hemisphere_rays
from .raytrace3d import trace_first_hits_bvh
from .transport3d import transport_first_hits_cone_bvh


@dataclass(frozen=True)
class Benchmark3DRow:
    suite: str
    n_lat: int
    n_lon: int
    n_faces: int
    rays_per_source: int
    support_count: int
    layout: str
    amplitude: float
    width: float
    cone_padding: float
    changed_face_fraction: float
    full_triangle_tests: int
    full_bbox_tests: int
    transport_triangle_tests: int
    transport_bbox_tests: int
    changed_rays: int
    footprint_rays: int
    footprint_overlap_rays: int
    support_bvh_rays: int
    fallback_full_rays: int
    candidate_per_changed: float
    overlap_fraction: float
    fallback_fraction: float
    query_speedup: float
    bbox_speedup: float
    wall_time_speedup: float
    full_time_s: float
    transport_time_s: float
    mismatches: int
    matrix_error: float
    row_error_transport: float

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class RadiosityBenchmark3DRow:
    suite: str
    n_lat: int
    n_lon: int
    n_faces: int
    rays_per_source: int
    support_count: int
    layout: str
    amplitude: float
    width: float
    cone_padding: float
    changed_rays: int
    footprint_rays: int
    overlap_fraction: float
    fallback_fraction: float
    query_speedup: float
    wall_time_speedup: float
    mismatches: int
    matrix_error: float
    row_error_full: float
    row_error_transport: float
    radiosity_rel_error: float
    irradiation_rel_error: float
    net_flux_rel_error: float
    radiosity_max_abs_error: float
    net_flux_max_abs_error: float
    total_power_full: float
    total_power_transport: float
    total_power_abs_error: float
    total_power_rel_error: float
    full_residual_inf: float
    transport_residual_inf: float

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _parse_sphere_grid(value: str) -> tuple[tuple[int, int], ...]:
    pairs = []
    for part in value.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if "x" not in part:
            raise argparse.ArgumentTypeError("sphere grid entries must look like 12x24")
        lat, lon = part.split("x", 1)
        pairs.append((int(lat), int(lon)))
    return tuple(pairs)


def run_single_support_benchmark(
    n_lat: int,
    n_lon: int,
    rays_per_source: int,
    amplitude: float,
    width: float,
    cone_padding: float,
    leaf_size: int,
) -> Benchmark3DRow:
    bump = make_default_cap_bump(amplitude=amplitude, width=width)
    return run_support_benchmark(
        suite="single-support-cone-bvh",
        n_lat=n_lat,
        n_lon=n_lon,
        rays_per_source=rays_per_source,
        bumps=(bump,),
        support_count=1,
        layout="single",
        amplitude=amplitude,
        width=width,
        cone_padding=cone_padding,
        leaf_size=leaf_size,
    )


def run_support_benchmark(
    suite: str,
    n_lat: int,
    n_lon: int,
    rays_per_source: int,
    bumps: tuple[SphericalCapBump, ...],
    support_count: int,
    layout: str,
    amplitude: float,
    width: float,
    cone_padding: float,
    leaf_size: int,
) -> Benchmark3DRow:
    old_mesh = make_sphere_cavity(n_lat=n_lat, n_lon=n_lon, bumps=bumps, alpha=0.0)
    old_rays = make_cosine_hemisphere_rays(old_mesh, rays_per_source)
    old_state = trace_first_hits_bvh(old_mesh, old_rays, BVH3D(old_mesh, leaf_size=leaf_size))

    new_mesh = make_sphere_cavity(n_lat=n_lat, n_lon=n_lon, bumps=bumps, alpha=1.0)

    t0 = time.perf_counter()
    new_rays = make_cosine_hemisphere_rays(new_mesh, rays_per_source)
    full_state = trace_first_hits_bvh(new_mesh, new_rays, BVH3D(new_mesh, leaf_size=leaf_size))
    full_time_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    transport = transport_first_hits_cone_bvh(
        old_state,
        old_mesh,
        new_mesh,
        rays_per_source,
        cone_padding=cone_padding,
        leaf_size=leaf_size,
    )
    transport_time_s = time.perf_counter() - t0

    full_matrix = rasterize_first_hits(full_state, new_mesh.n_faces)
    transport_matrix = rasterize_first_hits(transport.state, new_mesh.n_faces)
    changed_rays = int(np.count_nonzero(old_state.hits != full_state.hits))
    footprint_rays = int(transport.footprint_ray_count)

    return Benchmark3DRow(
        suite=suite,
        n_lat=n_lat,
        n_lon=n_lon,
        n_faces=new_mesh.n_faces,
        rays_per_source=rays_per_source,
        support_count=support_count,
        layout=layout,
        amplitude=amplitude,
        width=width,
        cone_padding=cone_padding,
        changed_face_fraction=transport.changed_face_count / new_mesh.n_faces,
        full_triangle_tests=full_state.triangle_tests,
        full_bbox_tests=full_state.bbox_tests,
        transport_triangle_tests=transport.triangle_tests,
        transport_bbox_tests=transport.bbox_tests,
        changed_rays=changed_rays,
        footprint_rays=footprint_rays,
        footprint_overlap_rays=transport.footprint_overlap_ray_count,
        support_bvh_rays=transport.support_bvh_ray_count,
        fallback_full_rays=transport.fallback_full_ray_count,
        candidate_per_changed=footprint_rays / changed_rays if changed_rays else float("inf"),
        overlap_fraction=transport.footprint_overlap_ray_count / footprint_rays if footprint_rays else 0.0,
        fallback_fraction=transport.fallback_full_ray_count / footprint_rays if footprint_rays else 0.0,
        query_speedup=full_state.triangle_tests / transport.triangle_tests if transport.triangle_tests else float("inf"),
        bbox_speedup=full_state.bbox_tests / transport.bbox_tests if transport.bbox_tests else float("inf"),
        wall_time_speedup=full_time_s / transport_time_s if transport_time_s else float("inf"),
        full_time_s=full_time_s,
        transport_time_s=transport_time_s,
        mismatches=hit_mismatch_count(transport.state, full_state),
        matrix_error=relative_frobenius_error(transport_matrix, full_matrix),
        row_error_transport=row_conservation_error(transport_matrix),
    )


def mesh_ray_scaling_rows(
    sphere_grid: tuple[tuple[int, int], ...],
    ray_grid: tuple[int, ...],
    amplitude: float,
    width: float,
    cone_padding: float,
    leaf_size: int,
) -> list[Benchmark3DRow]:
    rows = []
    for n_lat, n_lon in sphere_grid:
        for rays_per_source in ray_grid:
            row = run_single_support_benchmark(
                n_lat=n_lat,
                n_lon=n_lon,
                rays_per_source=rays_per_source,
                amplitude=amplitude,
                width=width,
                cone_padding=cone_padding,
                leaf_size=leaf_size,
            )
            rows.append(Benchmark3DRow(**{**row.as_dict(), "suite": "single-support-mesh-ray-scaling"}))
    return rows


def support_size_rows(
    widths: tuple[float, ...],
    n_lat: int,
    n_lon: int,
    rays_per_source: int,
    amplitude: float,
    cone_padding: float,
    leaf_size: int,
) -> list[Benchmark3DRow]:
    rows = []
    for width in widths:
        row = run_single_support_benchmark(
            n_lat=n_lat,
            n_lon=n_lon,
            rays_per_source=rays_per_source,
            amplitude=amplitude,
            width=width,
            cone_padding=cone_padding,
            leaf_size=leaf_size,
        )
        rows.append(Benchmark3DRow(**{**row.as_dict(), "suite": "single-support-size-scaling"}))
    return rows


def _far_support_centers(count: int) -> tuple[np.ndarray, ...]:
    if count == 1:
        values = [(1.0, 0.0, 0.0)]
    elif count == 2:
        values = [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)]
    elif count == 4:
        values = [(1.0, 1.0, 1.0), (1.0, -1.0, -1.0), (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0)]
    elif count == 8:
        values = [
            (sx, sy, sz)
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]
    else:
        golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
        values = []
        for idx in range(count):
            z = 1.0 - 2.0 * (idx + 0.5) / count
            radius = np.sqrt(max(0.0, 1.0 - z * z))
            phi = 2.0 * np.pi * ((idx / golden_ratio) % 1.0)
            values.append((radius * np.cos(phi), radius * np.sin(phi), z))
    return tuple(normalize_vectors(np.asarray(values, dtype=float)))


def _clustered_support_centers(count: int) -> tuple[np.ndarray, ...]:
    if count == 1:
        return (np.array([1.0, 0.0, 0.0]),)
    max_angle = min(0.65, 0.10 * (count - 1))
    angles = np.linspace(-max_angle, max_angle, count)
    values = [(np.cos(angle), np.sin(angle), 0.0) for angle in angles]
    return tuple(normalize_vectors(np.asarray(values, dtype=float)))


def support_centers(count: int, layout: str) -> tuple[np.ndarray, ...]:
    if count <= 0:
        raise ValueError("support count must be positive.")
    if layout == "far":
        return _far_support_centers(count)
    if layout == "clustered":
        return _clustered_support_centers(count)
    raise ValueError(f"unknown support layout: {layout!r}")


def make_support_bumps(
    support_count: int,
    layout: str,
    amplitude: float,
    width: float,
) -> tuple[SphericalCapBump, ...]:
    return tuple(
        SphericalCapBump(center=center, amplitude=amplitude, width=width, label=idx + 1)
        for idx, center in enumerate(support_centers(support_count, layout))
    )


def multi_support_rows(
    support_counts: tuple[int, ...],
    layouts: tuple[str, ...],
    n_lat: int,
    n_lon: int,
    rays_per_source: int,
    amplitude: float,
    width: float,
    cone_padding: float,
    leaf_size: int,
) -> list[Benchmark3DRow]:
    rows = []
    for layout in layouts:
        for support_count in support_counts:
            bumps = make_support_bumps(support_count, layout, amplitude, width)
            rows.append(
                run_support_benchmark(
                    suite="multi-support-cone-bvh",
                    n_lat=n_lat,
                    n_lon=n_lon,
                    rays_per_source=rays_per_source,
                    bumps=bumps,
                    support_count=support_count,
                    layout=layout,
                    amplitude=amplitude,
                    width=width,
                    cone_padding=cone_padding,
                    leaf_size=leaf_size,
                )
            )
    return rows


def run_radiosity_benchmark(
    n_lat: int,
    n_lon: int,
    rays_per_source: int,
    bumps: tuple[SphericalCapBump, ...],
    support_count: int,
    layout: str,
    amplitude: float,
    width: float,
    cone_padding: float,
    leaf_size: int,
) -> RadiosityBenchmark3DRow:
    old_mesh = make_sphere_cavity(n_lat=n_lat, n_lon=n_lon, bumps=bumps, alpha=0.0)
    old_rays = make_cosine_hemisphere_rays(old_mesh, rays_per_source)
    old_state = trace_first_hits_bvh(old_mesh, old_rays, BVH3D(old_mesh, leaf_size=leaf_size))

    new_mesh = make_sphere_cavity(n_lat=n_lat, n_lon=n_lon, bumps=bumps, alpha=1.0)
    t0 = time.perf_counter()
    new_rays = make_cosine_hemisphere_rays(new_mesh, rays_per_source)
    full_state = trace_first_hits_bvh(new_mesh, new_rays, BVH3D(new_mesh, leaf_size=leaf_size))
    full_time_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    transport = transport_first_hits_cone_bvh(
        old_state,
        old_mesh,
        new_mesh,
        rays_per_source,
        cone_padding=cone_padding,
        leaf_size=leaf_size,
    )
    transport_time_s = time.perf_counter() - t0

    full_matrix = rasterize_first_hits(full_state, new_mesh.n_faces)
    transport_matrix = rasterize_first_hits(transport.state, new_mesh.n_faces)
    case = make_radiosity_case(new_mesh.centroids, new_mesh.support_labels)
    full_solution = solve_radiosity(full_matrix, case.emission, case.reflectivity)
    transport_solution = solve_radiosity(transport_matrix, case.emission, case.reflectivity)

    full_power = float(np.dot(new_mesh.areas, full_solution.net_flux))
    transport_power = float(np.dot(new_mesh.areas, transport_solution.net_flux))
    total_power_abs_error = abs(transport_power - full_power)
    total_power_rel_error = total_power_abs_error / abs(full_power) if full_power != 0.0 else total_power_abs_error
    footprint_rays = int(transport.footprint_ray_count)

    return RadiosityBenchmark3DRow(
        suite="radiosity-toy-solve",
        n_lat=n_lat,
        n_lon=n_lon,
        n_faces=new_mesh.n_faces,
        rays_per_source=rays_per_source,
        support_count=support_count,
        layout=layout,
        amplitude=amplitude,
        width=width,
        cone_padding=cone_padding,
        changed_rays=int(np.count_nonzero(old_state.hits != full_state.hits)),
        footprint_rays=footprint_rays,
        overlap_fraction=transport.footprint_overlap_ray_count / footprint_rays if footprint_rays else 0.0,
        fallback_fraction=transport.fallback_full_ray_count / footprint_rays if footprint_rays else 0.0,
        query_speedup=full_state.triangle_tests / transport.triangle_tests if transport.triangle_tests else float("inf"),
        wall_time_speedup=full_time_s / transport_time_s if transport_time_s else float("inf"),
        mismatches=hit_mismatch_count(transport.state, full_state),
        matrix_error=relative_frobenius_error(transport_matrix, full_matrix),
        row_error_full=row_conservation_error(full_matrix),
        row_error_transport=row_conservation_error(transport_matrix),
        radiosity_rel_error=relative_l2_error(transport_solution.radiosity, full_solution.radiosity),
        irradiation_rel_error=relative_l2_error(transport_solution.irradiation, full_solution.irradiation),
        net_flux_rel_error=relative_l2_error(transport_solution.net_flux, full_solution.net_flux),
        radiosity_max_abs_error=max_abs_error(transport_solution.radiosity, full_solution.radiosity),
        net_flux_max_abs_error=max_abs_error(transport_solution.net_flux, full_solution.net_flux),
        total_power_full=full_power,
        total_power_transport=transport_power,
        total_power_abs_error=total_power_abs_error,
        total_power_rel_error=total_power_rel_error,
        full_residual_inf=full_solution.residual_inf,
        transport_residual_inf=transport_solution.residual_inf,
    )


def radiosity_rows(
    support_counts: tuple[int, ...],
    layouts: tuple[str, ...],
    n_lat: int,
    n_lon: int,
    rays_per_source: int,
    amplitude: float,
    width: float,
    cone_padding: float,
    leaf_size: int,
) -> list[RadiosityBenchmark3DRow]:
    rows = []
    for layout in layouts:
        for support_count in support_counts:
            bumps = make_support_bumps(support_count, layout, amplitude, width)
            rows.append(
                run_radiosity_benchmark(
                    n_lat=n_lat,
                    n_lon=n_lon,
                    rays_per_source=rays_per_source,
                    bumps=bumps,
                    support_count=support_count,
                    layout=layout,
                    amplitude=amplitude,
                    width=width,
                    cone_padding=cone_padding,
                    leaf_size=leaf_size,
                )
            )
    return rows


def write_csv(rows: list[Benchmark3DRow] | list[RadiosityBenchmark3DRow], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].as_dict()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())
    return output_path


def print_rows(rows: list[Benchmark3DRow]) -> None:
    print(
        "suite                    layout      nsup  faces  rays  width  pad      "
        "q_full/q_T  t_full/t_T  mismatch  matrix_err  changed  footprint  fp/change  overlap  fallback"
    )
    for row in rows:
        print(
            f"{row.suite:24s} "
            f"{row.layout:10s} "
            f"{row.support_count:4d} "
            f"{row.n_faces:5d} "
            f"{row.rays_per_source:5d} "
            f"{row.width:6.3f} "
            f"{row.cone_padding:8.1e} "
            f"{row.query_speedup:10.3f} "
            f"{row.wall_time_speedup:10.3f} "
            f"{row.mismatches:8d} "
            f"{row.matrix_error:10.2e} "
            f"{row.changed_rays:8d} "
            f"{row.footprint_rays:9d} "
            f"{row.candidate_per_changed:9.2f} "
            f"{row.overlap_fraction:8.2f} "
            f"{row.fallback_full_rays:8d}"
        )


def print_radiosity_rows(rows: list[RadiosityBenchmark3DRow]) -> None:
    print(
        "suite                 layout      nsup  faces  rays  q_full/q_T  t_full/t_T  "
        "mismatch  matrix_err  J_rel      q_rel      Q_abs      residual"
    )
    for row in rows:
        residual = max(row.full_residual_inf, row.transport_residual_inf)
        print(
            f"{row.suite:21s} "
            f"{row.layout:10s} "
            f"{row.support_count:4d} "
            f"{row.n_faces:5d} "
            f"{row.rays_per_source:5d} "
            f"{row.query_speedup:10.3f} "
            f"{row.wall_time_speedup:10.3f} "
            f"{row.mismatches:8d} "
            f"{row.matrix_error:10.2e} "
            f"{row.radiosity_rel_error:10.2e} "
            f"{row.net_flux_rel_error:10.2e} "
            f"{row.total_power_abs_error:10.2e} "
            f"{residual:10.2e}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run 3D spherical cavity support-transport benchmarks.")
    parser.add_argument(
        "suite",
        nargs="?",
        choices=("single", "scaling", "support-size", "multi-support", "radiosity", "m1", "m2", "m3"),
        default="single",
        help="benchmark suite to run",
    )
    parser.add_argument("--lat", type=int, default=12, help="sphere polar subdivisions")
    parser.add_argument("--lon", type=int, default=24, help="sphere azimuth subdivisions")
    parser.add_argument("--rays", type=int, default=24, help="deterministic hemisphere rays per source face")
    parser.add_argument(
        "--sphere-grid",
        type=_parse_sphere_grid,
        default=((8, 16), (10, 20), (12, 24)),
        help="comma-separated sphere grids such as 8x16,10x20,12x24",
    )
    parser.add_argument(
        "--ray-grid",
        type=_parse_ints,
        default=(12, 24, 48),
        help="comma-separated ray counts for scaling suites",
    )
    parser.add_argument(
        "--width-grid",
        type=_parse_floats,
        default=(0.25, 0.35, 0.50),
        help="comma-separated support angular widths for support-size suite",
    )
    parser.add_argument(
        "--support-counts",
        type=_parse_ints,
        default=(1, 2, 4, 8),
        help="comma-separated support counts for multi-support suite",
    )
    parser.add_argument(
        "--layouts",
        default="far,clustered",
        help="comma-separated support layouts for multi-support suite",
    )
    parser.add_argument("--amplitude", type=float, default=0.12, help="inward compact cap amplitude")
    parser.add_argument("--width", type=float, default=0.35, help="compact cap angular radius in radians")
    parser.add_argument("--cone-padding", type=float, default=1e-6, help="extra cone half-angle padding in radians")
    parser.add_argument("--leaf-size", type=int, default=8, help="BVH leaf triangle count")
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark3d_single_support.csv"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    warm_up_accelerated_3d()
    rows = []
    if args.suite in {"single", "m1"}:
        rows.append(
            run_single_support_benchmark(
                n_lat=args.lat,
                n_lon=args.lon,
                rays_per_source=args.rays,
                amplitude=args.amplitude,
                width=args.width,
                cone_padding=args.cone_padding,
                leaf_size=args.leaf_size,
            )
        )
    if args.suite in {"scaling", "m1"}:
        rows.extend(
            mesh_ray_scaling_rows(
                sphere_grid=args.sphere_grid,
                ray_grid=args.ray_grid,
                amplitude=args.amplitude,
                width=args.width,
                cone_padding=args.cone_padding,
                leaf_size=args.leaf_size,
            )
        )
    if args.suite in {"support-size", "m1"}:
        rows.extend(
            support_size_rows(
                widths=args.width_grid,
                n_lat=args.lat,
                n_lon=args.lon,
                rays_per_source=args.rays,
                amplitude=args.amplitude,
                cone_padding=args.cone_padding,
                leaf_size=args.leaf_size,
            )
        )
    if args.suite in {"multi-support", "m2"}:
        layouts = tuple(part.strip() for part in args.layouts.split(",") if part.strip())
        rows.extend(
            multi_support_rows(
                support_counts=args.support_counts,
                layouts=layouts,
                n_lat=args.lat,
                n_lon=args.lon,
                rays_per_source=args.rays,
                amplitude=args.amplitude,
                width=args.width,
                cone_padding=args.cone_padding,
                leaf_size=args.leaf_size,
            )
        )
    if args.suite in {"radiosity", "m3"}:
        layouts = tuple(part.strip() for part in args.layouts.split(",") if part.strip())
        radio_rows = radiosity_rows(
            support_counts=args.support_counts,
            layouts=layouts,
            n_lat=args.lat,
            n_lon=args.lon,
            rays_per_source=args.rays,
            amplitude=args.amplitude,
            width=args.width,
            cone_padding=args.cone_padding,
            leaf_size=args.leaf_size,
        )
        print_radiosity_rows(radio_rows)
        path = write_csv(radio_rows, args.output)
        print(f"csv written: {path}")
        return 0
    print_rows(rows)
    path = write_csv(rows, args.output)
    print(f"csv written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
