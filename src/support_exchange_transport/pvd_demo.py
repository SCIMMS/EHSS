from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import os
import time

import numpy as np

from .raytrace2d import intersect_ray_segment


@dataclass(frozen=True)
class PVDSegment:
    start: np.ndarray
    end: np.ndarray
    normal: np.ndarray
    label: str

    def __post_init__(self) -> None:
        start = np.asarray(self.start, dtype=float)
        end = np.asarray(self.end, dtype=float)
        normal = np.asarray(self.normal, dtype=float)
        if start.shape != (2,) or end.shape != (2,) or normal.shape != (2,):
            raise ValueError("segment start, end, and normal must have shape (2,).")
        norm = np.linalg.norm(normal)
        if norm <= 0.0:
            raise ValueError("normal must be nonzero.")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "normal", normal / norm)


@dataclass(frozen=True)
class PVDGeometry:
    segments: tuple[PVDSegment, ...]
    perturbation_bbox: tuple[float, float, float, float] | None = None

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(segment.label for segment in self.segments)


@dataclass(frozen=True)
class PVDRayBundle:
    origins: np.ndarray
    directions: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        origins = np.asarray(self.origins, dtype=float)
        directions = np.asarray(self.directions, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        if origins.ndim != 2 or origins.shape[1] != 2:
            raise ValueError("origins must have shape (n_rays, 2).")
        if directions.shape != origins.shape:
            raise ValueError("directions must match origins.")
        if weights.shape != (len(origins),):
            raise ValueError("weights must have shape (n_rays,).")
        norms = np.linalg.norm(directions, axis=1)
        if np.any(norms <= 0.0):
            raise ValueError("directions must be nonzero.")
        object.__setattr__(self, "origins", origins)
        object.__setattr__(self, "directions", directions / norms[:, None])
        object.__setattr__(self, "weights", weights)

    @property
    def n_rays(self) -> int:
        return int(self.origins.shape[0])


@dataclass(frozen=True)
class PVDTrace:
    targets: np.ndarray
    taus: np.ndarray
    labels: tuple[str, ...]
    incidence: np.ndarray
    weights: np.ndarray
    query_count: int


def make_trench_pvd_geometry(
    trench_width: float = 1.2,
    trench_depth: float = 2.4,
    field_half_width: float = 2.8,
    bump_width: float = 0.0,
    bump_height: float = 0.0,
) -> PVDGeometry:
    if trench_width <= 0.0 or trench_depth <= 0.0 or field_half_width <= 0.5 * trench_width:
        raise ValueError("invalid trench dimensions.")
    if bump_width < 0.0 or bump_height < 0.0:
        raise ValueError("bump dimensions must be nonnegative.")
    if bump_width > trench_width:
        raise ValueError("bump_width must fit inside the trench.")
    if bump_height >= 0.9 * trench_depth:
        raise ValueError("bump_height is too large for this toy feature.")

    half = 0.5 * trench_width
    y_bottom = -trench_depth
    segments: list[PVDSegment] = [
        PVDSegment(np.array([-field_half_width, 0.0]), np.array([-half, 0.0]), np.array([0.0, 1.0]), "mask_left"),
        PVDSegment(np.array([half, 0.0]), np.array([field_half_width, 0.0]), np.array([0.0, 1.0]), "mask_right"),
        PVDSegment(np.array([-half, 0.0]), np.array([-half, y_bottom]), np.array([1.0, 0.0]), "left_wall"),
        PVDSegment(np.array([half, y_bottom]), np.array([half, 0.0]), np.array([-1.0, 0.0]), "right_wall"),
        PVDSegment(np.array([-half, y_bottom]), np.array([half, y_bottom]), np.array([0.0, 1.0]), "bottom"),
    ]

    bbox: tuple[float, float, float, float] | None = None
    if bump_width > 0.0 and bump_height > 0.0:
        bump_half = 0.5 * bump_width
        bump_top_y = y_bottom + bump_height
        segments.extend(
            [
                PVDSegment(np.array([-bump_half, y_bottom]), np.array([-bump_half, bump_top_y]), np.array([-1.0, 0.0]), "bump_left"),
                PVDSegment(np.array([-bump_half, bump_top_y]), np.array([bump_half, bump_top_y]), np.array([0.0, 1.0]), "bump_top"),
                PVDSegment(np.array([bump_half, bump_top_y]), np.array([bump_half, y_bottom]), np.array([1.0, 0.0]), "bump_right"),
            ]
        )
        pad = 1e-9
        bbox = (-bump_half - pad, bump_half + pad, y_bottom - pad, bump_top_y + pad)

    return PVDGeometry(segments=tuple(segments), perturbation_bbox=bbox)


def make_pvd_rays(
    source_samples: int = 161,
    angle_samples: int = 31,
    source_half_width: float = 3.2,
    source_center: float = 0.0,
    source_y: float = 1.2,
    max_angle_deg: float = 72.0,
    mean_angle_deg: float = 0.0,
    collimation_power: float = 1.0,
    quadrature_rule: str = "endpoint",
    quadrature_seed: int = 1729,
) -> PVDRayBundle:
    if source_samples <= 0 or angle_samples <= 0:
        raise ValueError("sample counts must be positive.")
    if source_half_width <= 0.0 or source_y <= 0.0:
        raise ValueError("source extent and height must be positive.")
    if not 0.0 < max_angle_deg < 90.0:
        raise ValueError("max_angle_deg must be in (0, 90).")
    if abs(mean_angle_deg) + max_angle_deg >= 90.0:
        raise ValueError(
            "the prescribed angular interval must remain inside (-90, 90) degrees."
        )
    if quadrature_rule not in {"endpoint", "midpoint", "sobol"}:
        raise ValueError(
            "quadrature_rule must be 'endpoint', 'midpoint', or 'sobol'."
        )

    max_angle = np.deg2rad(max_angle_deg)
    mean_angle = np.deg2rad(mean_angle_deg)
    if quadrature_rule == "sobol":
        from scipy.stats import qmc

        ray_count = source_samples * angle_samples
        sampler = qmc.Sobol(d=2, scramble=True, seed=quadrature_seed)
        if ray_count & (ray_count - 1) == 0:
            samples = sampler.random_base2(int(np.log2(ray_count)))
        else:
            samples = sampler.random(ray_count)
        x_values = source_center + source_half_width * (
            2.0 * samples[:, 0] - 1.0
        )
        relative_angles = max_angle * (2.0 * samples[:, 1] - 1.0)
        angles = mean_angle + relative_angles
        weights = (
            np.clip(np.cos(relative_angles), 0.0, None) ** collimation_power
        )
        weights /= np.sum(weights)
        origins = np.column_stack(
            (x_values, np.full(ray_count, source_y, dtype=float))
        )
        directions = np.column_stack((np.sin(angles), -np.cos(angles)))
        return PVDRayBundle(origins, directions, weights)
    if quadrature_rule == "endpoint":
        x_values = source_center + np.linspace(
            -source_half_width,
            source_half_width,
            source_samples,
        )
        relative_angles = np.linspace(-max_angle, max_angle, angle_samples)
    else:
        x_values = source_center + (
            -source_half_width
            + (np.arange(source_samples, dtype=float) + 0.5)
            * (2.0 * source_half_width / source_samples)
        )
        relative_angles = (
            -max_angle
            + (np.arange(angle_samples, dtype=float) + 0.5)
            * (2.0 * max_angle / angle_samples)
        )
    angles = mean_angle + relative_angles
    angular_weights = (
        np.clip(np.cos(relative_angles), 0.0, None) ** collimation_power
    )
    angular_weights /= np.sum(angular_weights)
    source_weights = np.full(source_samples, 1.0 / source_samples)

    origins: list[list[float]] = []
    directions: list[list[float]] = []
    weights: list[float] = []
    for x_idx, x in enumerate(x_values):
        for angle_idx, angle in enumerate(angles):
            origins.append([x, source_y])
            directions.append([np.sin(angle), -np.cos(angle)])
            weights.append(float(source_weights[x_idx] * angular_weights[angle_idx]))
    return PVDRayBundle(np.asarray(origins), np.asarray(directions), np.asarray(weights))


def trace_pvd_first_hits(geometry: PVDGeometry, rays: PVDRayBundle) -> PVDTrace:
    targets = np.full(rays.n_rays, -1, dtype=np.int64)
    taus = np.full(rays.n_rays, np.inf, dtype=float)
    incidence = np.zeros(rays.n_rays, dtype=float)
    query_count = 0

    for ray_idx in range(rays.n_rays):
        origin = rays.origins[ray_idx]
        direction = rays.directions[ray_idx]
        best_tau = float("inf")
        best_target = -1
        for target, segment in enumerate(geometry.segments):
            query_count += 1
            tau = intersect_ray_segment(origin, direction, segment.start, segment.end, min_t=1e-9)
            if tau < best_tau:
                best_tau = tau
                best_target = target
        if best_target >= 0:
            targets[ray_idx] = best_target
            taus[ray_idx] = best_tau
            incidence[ray_idx] = max(0.0, -float(np.dot(direction, geometry.segments[best_target].normal)))

    return PVDTrace(targets, taus, geometry.labels, incidence, rays.weights.copy(), query_count)


def _ray_intersects_bbox_before_tau(
    origin: np.ndarray,
    direction: np.ndarray,
    bbox: tuple[float, float, float, float],
    tau_limit: float,
) -> bool:
    xmin, xmax, ymin, ymax = bbox
    tmin = 0.0
    tmax = tau_limit if np.isfinite(tau_limit) else 1e6
    for axis, lo, hi in ((0, xmin, xmax), (1, ymin, ymax)):
        if abs(direction[axis]) <= 1e-14:
            if origin[axis] < lo or origin[axis] > hi:
                return False
            continue
        t1 = (lo - origin[axis]) / direction[axis]
        t2 = (hi - origin[axis]) / direction[axis]
        t_near = min(t1, t2)
        t_far = max(t1, t2)
        tmin = max(tmin, t_near)
        tmax = min(tmax, t_far)
        if tmax < tmin:
            return False
    return tmax > 1e-9 and tmin <= tau_limit


def pvd_candidate_mask(
    old_trace: PVDTrace,
    new_geometry: PVDGeometry,
    rays: PVDRayBundle,
) -> np.ndarray:
    if new_geometry.perturbation_bbox is None:
        return np.zeros(rays.n_rays, dtype=bool)
    bbox = new_geometry.perturbation_bbox
    mask = np.zeros(rays.n_rays, dtype=bool)
    for ray_idx in range(rays.n_rays):
        mask[ray_idx] = _ray_intersects_bbox_before_tau(
            rays.origins[ray_idx],
            rays.directions[ray_idx],
            bbox,
            old_trace.taus[ray_idx],
        )
    return mask


def _label_flux_from_trace(trace: PVDTrace, labels: tuple[str, ...] | None = None) -> dict[str, float]:
    out: dict[str, float] = {}
    if labels is not None:
        for label in labels:
            out[label] = 0.0
    for target, incidence, weight in zip(trace.targets, trace.incidence, trace.weights):
        if target < 0:
            continue
        label = trace.labels[int(target)]
        out[label] = out.get(label, 0.0) + float(weight * incidence)
    return out


def _label_flux_from_arrays(
    targets: np.ndarray,
    incidence: np.ndarray,
    weights: np.ndarray,
    labels: tuple[str, ...],
    all_labels: tuple[str, ...],
) -> dict[str, float]:
    out = {label: 0.0 for label in all_labels}
    for target, value, weight in zip(targets, incidence, weights):
        if target < 0:
            continue
        out[labels[int(target)]] = out.get(labels[int(target)], 0.0) + float(weight * value)
    return out


def _flux_vector(flux: dict[str, float], labels: tuple[str, ...]) -> np.ndarray:
    return np.asarray([flux.get(label, 0.0) for label in labels], dtype=float)


def transport_pvd_added_profile(
    old_trace: PVDTrace,
    new_geometry: PVDGeometry,
    rays: PVDRayBundle,
) -> tuple[PVDTrace, np.ndarray, int]:
    candidate = pvd_candidate_mask(old_trace, new_geometry, rays)
    targets = np.array(old_trace.targets, copy=True)
    taus = np.array(old_trace.taus, copy=True)
    incidence = np.array(old_trace.incidence, copy=True)
    query_count = rays.n_rays

    candidate_indices = np.flatnonzero(candidate)
    if len(candidate_indices):
        candidate_rays = PVDRayBundle(
            rays.origins[candidate_indices],
            rays.directions[candidate_indices],
            rays.weights[candidate_indices],
        )
        candidate_trace = trace_pvd_first_hits(new_geometry, candidate_rays)
        query_count += candidate_trace.query_count
        targets[candidate_indices] = candidate_trace.targets
        taus[candidate_indices] = candidate_trace.taus
        incidence[candidate_indices] = candidate_trace.incidence

    return (
        PVDTrace(targets=targets, taus=taus, labels=new_geometry.labels, incidence=incidence, weights=rays.weights.copy(), query_count=query_count),
        candidate,
        query_count,
    )


def _time_stats(elapsed: list[float]) -> dict[str, float]:
    values = np.asarray(elapsed, dtype=float)
    return {
        "median": float(np.median(values)),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
        "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
    }


def _timed_full_trace(geometry: PVDGeometry, rays: PVDRayBundle, repeats: int) -> tuple[PVDTrace, dict[str, float]]:
    elapsed: list[float] = []
    trace: PVDTrace | None = None
    for _ in range(repeats):
        start = time.perf_counter()
        trace = trace_pvd_first_hits(geometry, rays)
        elapsed.append(time.perf_counter() - start)
    assert trace is not None
    return trace, _time_stats(elapsed)


def _timed_transport(old_trace: PVDTrace, new_geometry: PVDGeometry, rays: PVDRayBundle, repeats: int) -> tuple[PVDTrace, np.ndarray, int, dict[str, float]]:
    elapsed: list[float] = []
    trace: PVDTrace | None = None
    candidate: np.ndarray | None = None
    query_count = 0
    for _ in range(repeats):
        start = time.perf_counter()
        trace, candidate, query_count = transport_pvd_added_profile(old_trace, new_geometry, rays)
        elapsed.append(time.perf_counter() - start)
    assert trace is not None and candidate is not None
    return trace, candidate, query_count, _time_stats(elapsed)


def evaluate_pvd_profile_update(
    bump_width: float,
    bump_height: float,
    source_samples: int = 161,
    angle_samples: int = 31,
    timing_repeats: int = 5,
) -> dict[str, float | int | str]:
    rays = make_pvd_rays(source_samples=source_samples, angle_samples=angle_samples)
    old_geometry = make_trench_pvd_geometry()
    new_geometry = make_trench_pvd_geometry(bump_width=bump_width, bump_height=bump_height)

    old_trace, cache_build_timing = _timed_full_trace(old_geometry, rays, max(1, timing_repeats // 2))
    full_new_trace, full_new_timing = _timed_full_trace(new_geometry, rays, timing_repeats)
    transport_trace, candidate, candidate_query_count, candidate_timing = _timed_transport(
        old_trace,
        new_geometry,
        rays,
        timing_repeats,
    )

    labels = tuple(dict.fromkeys(old_geometry.labels + new_geometry.labels))
    full_flux = _label_flux_from_trace(full_new_trace, labels)
    transport_flux = _label_flux_from_arrays(
        transport_trace.targets,
        transport_trace.incidence,
        transport_trace.weights,
        transport_trace.labels,
        labels,
    )
    old_flux = _label_flux_from_trace(old_trace, labels)
    full_vec = _flux_vector(full_flux, labels)
    transport_vec = _flux_vector(transport_flux, labels)
    old_vec = _flux_vector(old_flux, labels)

    active = (old_trace.targets != full_new_trace.targets) | (np.abs(old_trace.incidence - full_new_trace.incidence) > 1e-14)
    false_negative_rays = int(np.count_nonzero(active & ~candidate))
    target_mismatch_rays = int(np.count_nonzero(transport_trace.targets != full_new_trace.targets))
    incidence_max_abs_error = float(np.max(np.abs(transport_trace.incidence - full_new_trace.incidence)))
    flux_rel_error = float(np.linalg.norm(transport_vec - full_vec) / max(np.linalg.norm(full_vec), 1e-15))
    total_flux_error = abs(float(np.sum(transport_vec) - np.sum(full_vec))) / max(abs(float(np.sum(full_vec))), 1e-15)

    bottom_old = old_flux.get("bottom", 0.0)
    bottom_new = full_flux.get("bottom", 0.0)
    sidewall_new = full_flux.get("left_wall", 0.0) + full_flux.get("right_wall", 0.0)
    bump_flux = full_flux.get("bump_left", 0.0) + full_flux.get("bump_top", 0.0) + full_flux.get("bump_right", 0.0)

    return {
        "scenario": f"bump_w{bump_width:.2f}_h{bump_height:.2f}",
        "model_scope": "feature_scale_ballistic_pvd_line_of_sight",
        "source_samples": source_samples,
        "angle_samples": angle_samples,
        "n_rays": rays.n_rays,
        "old_segment_count": len(old_geometry.segments),
        "new_segment_count": len(new_geometry.segments),
        "bump_width": bump_width,
        "bump_height": bump_height,
        "candidate_rays": int(np.count_nonzero(candidate)),
        "active_rays": int(np.count_nonzero(active)),
        "false_negative_rays": false_negative_rays,
        "target_mismatch_rays": target_mismatch_rays,
        "incidence_max_abs_error": incidence_max_abs_error,
        "candidate_fraction": float(np.count_nonzero(candidate) / max(rays.n_rays, 1)),
        "active_fraction": float(np.count_nonzero(active) / max(rays.n_rays, 1)),
        "candidate_per_active": float(np.count_nonzero(candidate) / max(np.count_nonzero(active), 1)),
        "flux_rel_error": flux_rel_error,
        "total_flux_rel_error": total_flux_error,
        "old_total_flux": float(np.sum(old_vec)),
        "full_new_total_flux": float(np.sum(full_vec)),
        "transport_total_flux": float(np.sum(transport_vec)),
        "bottom_flux_old": bottom_old,
        "bottom_flux_new": bottom_new,
        "sidewall_flux_new": sidewall_new,
        "bump_flux_new": bump_flux,
        "step_coverage_new": sidewall_new / max(bottom_new + bump_flux, 1e-15),
        "cache_build_wall_time_s": cache_build_timing["median"],
        "cache_build_wall_time_q1_s": cache_build_timing["q1"],
        "cache_build_wall_time_q3_s": cache_build_timing["q3"],
        "cache_build_wall_time_iqr_s": cache_build_timing["iqr"],
        "full_new_wall_time_s": full_new_timing["median"],
        "full_new_wall_time_q1_s": full_new_timing["q1"],
        "full_new_wall_time_q3_s": full_new_timing["q3"],
        "full_new_wall_time_iqr_s": full_new_timing["iqr"],
        "candidate_route_wall_time_s": candidate_timing["median"],
        "candidate_route_wall_time_q1_s": candidate_timing["q1"],
        "candidate_route_wall_time_q3_s": candidate_timing["q3"],
        "candidate_route_wall_time_iqr_s": candidate_timing["iqr"],
        "candidate_route_wall_speedup_vs_full": full_new_timing["median"] / max(candidate_timing["median"], 1e-15),
        "candidate_route_wall_speedup_q1_vs_full": full_new_timing["q1"] / max(candidate_timing["q3"], 1e-15),
        "candidate_route_wall_speedup_q3_vs_full": full_new_timing["q3"] / max(candidate_timing["q1"], 1e-15),
        "full_new_query_count": full_new_trace.query_count,
        "candidate_route_query_count": candidate_query_count,
        "candidate_route_query_speedup_vs_full": full_new_trace.query_count / max(candidate_query_count, 1),
    }


def run_default_pvd_demo(
    source_samples: int = 161,
    angle_samples: int = 31,
    timing_repeats: int = 5,
) -> list[dict[str, float | int | str]]:
    cases = [(0.12, 0.08), (0.24, 0.12), (0.42, 0.18), (0.72, 0.28), (1.00, 0.38)]
    return [
        evaluate_pvd_profile_update(
            bump_width=width,
            bump_height=height,
            source_samples=source_samples,
            angle_samples=angle_samples,
            timing_repeats=timing_repeats,
        )
        for width, height in cases
    ]


def write_pvd_demo_csv(rows: list[dict[str, float | int | str]], output_csv: str | Path) -> None:
    if not rows:
        raise ValueError("rows must not be empty.")
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_pvd_demo_rows(
    rows: list[dict[str, float | int | str]],
    output_png: str | Path,
    output_pdf: str | Path | None = None,
) -> None:
    cache_dir = Path("artifacts/matplotlib").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib.pyplot as plt

    widths = np.array([float(row["bump_width"]) for row in rows])
    total_change = np.array(
        [
            100.0 * (float(row["full_new_total_flux"]) - float(row["old_total_flux"])) / max(float(row["old_total_flux"]), 1e-15)
            for row in rows
        ]
    )
    bottom_change = np.array(
        [
            100.0 * (float(row["bottom_flux_new"]) - float(row["bottom_flux_old"])) / max(float(row["bottom_flux_old"]), 1e-15)
            for row in rows
        ]
    )
    candidate = np.array([100.0 * float(row["candidate_fraction"]) for row in rows])
    active = np.array([100.0 * float(row["active_fraction"]) for row in rows])
    query_speedup = np.array([float(row["candidate_route_query_speedup_vs_full"]) for row in rows])
    wall_speedup = np.array([float(row["candidate_route_wall_speedup_vs_full"]) for row in rows])

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
    axes[0].plot(widths, total_change, marker="o", label="total flux")
    axes[0].plot(widths, bottom_change, marker="s", linestyle="--", label="bottom flux")
    axes[0].axhline(0.0, color="0.25", linewidth=0.8)
    axes[0].set_xlabel("Deposited ridge width")
    axes[0].set_ylabel("PVD flux change (%)")
    axes[0].set_title("Feature-scale PVD response")
    axes[0].legend(fontsize=8)

    axes[1].plot(widths, candidate, marker="o", label="candidate rays")
    axes[1].plot(widths, active, marker="s", linestyle="--", label="active rays")
    axes[1].set_xlabel("Deposited ridge width")
    axes[1].set_ylabel("Rays (%)")
    axes[1].set_title("Changed event frontier")
    axes[1].legend(fontsize=8)

    axes[2].plot(widths, query_speedup, marker="o", label="intersection tests")
    axes[2].plot(widths, wall_speedup, marker="s", linestyle="--", label="wall time")
    axes[2].axhline(1.0, color="0.25", linewidth=0.8)
    axes[2].set_xlabel("Deposited ridge width")
    axes[2].set_ylabel("Speedup vs full new trace")
    axes[2].set_title("Transport update work")
    axes[2].legend(fontsize=8)

    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.suptitle("Feature-scale ballistic PVD event-frontier transport")

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    if output_pdf is not None:
        output_pdf = Path(output_pdf)
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf)
    plt.close(fig)
