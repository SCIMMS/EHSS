from __future__ import annotations

from pathlib import Path
import csv
import os
import time

import numpy as np

from .pvd_demo import (
    PVDRayBundle,
    PVDSegment,
    PVDGeometry,
    PVDTrace,
    make_pvd_rays,
    pvd_candidate_mask,
    trace_pvd_first_hits,
)


def make_recess_etch_geometry(
    trench_width: float = 1.2,
    trench_depth: float = 2.4,
    field_half_width: float = 2.8,
    recess_width: float = 0.0,
    recess_depth: float = 0.0,
) -> PVDGeometry:
    if trench_width <= 0.0 or trench_depth <= 0.0 or field_half_width <= 0.5 * trench_width:
        raise ValueError("invalid trench dimensions.")
    if recess_width < 0.0 or recess_depth < 0.0:
        raise ValueError("recess dimensions must be nonnegative.")
    if recess_width > trench_width:
        raise ValueError("recess_width must fit inside the trench.")
    if recess_depth >= 0.5 * trench_depth:
        raise ValueError("recess_depth is too large for this toy feature.")

    half = 0.5 * trench_width
    y_bottom = -trench_depth
    recess_half = 0.5 * recess_width
    floor_y = y_bottom - recess_depth

    segments: list[PVDSegment] = [
        PVDSegment(np.array([-field_half_width, 0.0]), np.array([-half, 0.0]), np.array([0.0, 1.0]), "mask_left"),
        PVDSegment(np.array([half, 0.0]), np.array([field_half_width, 0.0]), np.array([0.0, 1.0]), "mask_right"),
        PVDSegment(np.array([-half, 0.0]), np.array([-half, y_bottom]), np.array([1.0, 0.0]), "left_wall"),
        PVDSegment(np.array([half, y_bottom]), np.array([half, 0.0]), np.array([-1.0, 0.0]), "right_wall"),
    ]

    if recess_width > 0.0:
        segments.extend(
            [
                PVDSegment(np.array([-half, y_bottom]), np.array([-recess_half, y_bottom]), np.array([0.0, 1.0]), "bottom_left"),
                PVDSegment(np.array([-recess_half, floor_y]), np.array([recess_half, floor_y]), np.array([0.0, 1.0]), "etch_floor"),
                PVDSegment(np.array([recess_half, y_bottom]), np.array([half, y_bottom]), np.array([0.0, 1.0]), "bottom_right"),
            ]
        )
    else:
        segments.extend(
            [
                PVDSegment(np.array([-half, y_bottom]), np.array([0.0, y_bottom]), np.array([0.0, 1.0]), "bottom_left"),
                PVDSegment(np.array([0.0, y_bottom]), np.array([0.0, y_bottom]), np.array([0.0, 1.0]), "etch_floor"),
                PVDSegment(np.array([0.0, y_bottom]), np.array([half, y_bottom]), np.array([0.0, 1.0]), "bottom_right"),
            ]
        )

    bbox: tuple[float, float, float, float] | None = None
    if recess_width > 0.0 and recess_depth > 0.0:
        segments.extend(
            [
                PVDSegment(np.array([-recess_half, y_bottom]), np.array([-recess_half, floor_y]), np.array([1.0, 0.0]), "recess_left"),
                PVDSegment(np.array([recess_half, floor_y]), np.array([recess_half, y_bottom]), np.array([-1.0, 0.0]), "recess_right"),
            ]
        )
        pad = 1e-9
        bbox = (-recess_half - pad, recess_half + pad, floor_y - pad, y_bottom + pad)

    return PVDGeometry(segments=tuple(segments), perturbation_bbox=bbox)


def make_old_flat_etch_geometry(
    trench_width: float = 1.2,
    trench_depth: float = 2.4,
    field_half_width: float = 2.8,
    recess_width: float = 0.2,
) -> PVDGeometry:
    return make_recess_etch_geometry(
        trench_width=trench_width,
        trench_depth=trench_depth,
        field_half_width=field_half_width,
        recess_width=recess_width,
        recess_depth=0.0,
    )


def directional_etch_yield(incidence: np.ndarray, yield_power: float = 2.0, threshold: float = 0.0) -> np.ndarray:
    if yield_power <= 0.0:
        raise ValueError("yield_power must be positive.")
    if threshold < 0.0 or threshold >= 1.0:
        raise ValueError("threshold must be in [0, 1).")
    values = np.clip(incidence, 0.0, 1.0)
    values = np.where(values >= threshold, values, 0.0)
    return values**yield_power


def _label_etch_from_arrays(
    targets: np.ndarray,
    incidence: np.ndarray,
    weights: np.ndarray,
    labels: tuple[str, ...],
    all_labels: tuple[str, ...],
    yield_power: float,
    threshold: float,
) -> dict[str, float]:
    out = {label: 0.0 for label in all_labels}
    etch_values = directional_etch_yield(incidence, yield_power=yield_power, threshold=threshold)
    for target, value, weight in zip(targets, etch_values, weights):
        if target < 0:
            continue
        out[labels[int(target)]] = out.get(labels[int(target)], 0.0) + float(weight * value)
    return out


def _label_etch_from_trace(
    trace: PVDTrace,
    all_labels: tuple[str, ...],
    yield_power: float,
    threshold: float,
) -> dict[str, float]:
    return _label_etch_from_arrays(
        trace.targets,
        trace.incidence,
        trace.weights,
        trace.labels,
        all_labels,
        yield_power,
        threshold,
    )


def _response_vector(response: dict[str, float], labels: tuple[str, ...]) -> np.ndarray:
    return np.asarray([response.get(label, 0.0) for label in labels], dtype=float)


def transport_directional_etch_recess(
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


def _timed_transport(
    old_trace: PVDTrace,
    new_geometry: PVDGeometry,
    rays: PVDRayBundle,
    repeats: int,
) -> tuple[PVDTrace, np.ndarray, int, dict[str, float]]:
    elapsed: list[float] = []
    trace: PVDTrace | None = None
    candidate: np.ndarray | None = None
    query_count = 0
    for _ in range(repeats):
        start = time.perf_counter()
        trace, candidate, query_count = transport_directional_etch_recess(old_trace, new_geometry, rays)
        elapsed.append(time.perf_counter() - start)
    assert trace is not None and candidate is not None
    return trace, candidate, query_count, _time_stats(elapsed)


def evaluate_directional_etch_recess_update(
    recess_width: float,
    recess_depth: float,
    source_samples: int = 161,
    angle_samples: int = 31,
    timing_repeats: int = 5,
    yield_power: float = 2.0,
    yield_threshold: float = 0.0,
    max_angle_deg: float = 84.0,
    collimation_power: float = 0.4,
) -> dict[str, float | int | str]:
    rays = make_pvd_rays(
        source_samples=source_samples,
        angle_samples=angle_samples,
        max_angle_deg=max_angle_deg,
        collimation_power=collimation_power,
    )
    old_geometry = make_old_flat_etch_geometry(recess_width=recess_width)
    new_geometry = make_recess_etch_geometry(recess_width=recess_width, recess_depth=recess_depth)

    old_trace, cache_build_timing = _timed_full_trace(old_geometry, rays, max(1, timing_repeats // 2))
    full_new_trace, full_new_timing = _timed_full_trace(new_geometry, rays, timing_repeats)
    transport_trace, candidate, candidate_query_count, candidate_timing = _timed_transport(
        old_trace,
        new_geometry,
        rays,
        timing_repeats,
    )

    labels = tuple(dict.fromkeys(old_geometry.labels + new_geometry.labels))
    old_response = _label_etch_from_trace(old_trace, labels, yield_power, yield_threshold)
    full_response = _label_etch_from_trace(full_new_trace, labels, yield_power, yield_threshold)
    transport_response = _label_etch_from_arrays(
        transport_trace.targets,
        transport_trace.incidence,
        transport_trace.weights,
        transport_trace.labels,
        labels,
        yield_power,
        yield_threshold,
    )
    old_vec = _response_vector(old_response, labels)
    full_vec = _response_vector(full_response, labels)
    transport_vec = _response_vector(transport_response, labels)

    old_etch_values = directional_etch_yield(old_trace.incidence, yield_power=yield_power, threshold=yield_threshold)
    full_etch_values = directional_etch_yield(full_new_trace.incidence, yield_power=yield_power, threshold=yield_threshold)
    transport_etch_values = directional_etch_yield(transport_trace.incidence, yield_power=yield_power, threshold=yield_threshold)
    active = (old_trace.targets != full_new_trace.targets) | (np.abs(old_etch_values - full_etch_values) > 1e-14)
    false_negative_rays = int(np.count_nonzero(active & ~candidate))
    target_mismatch_rays = int(np.count_nonzero(transport_trace.targets != full_new_trace.targets))
    yield_max_abs_error = float(np.max(np.abs(transport_etch_values - full_etch_values)))
    etch_rel_error = float(np.linalg.norm(transport_vec - full_vec) / max(np.linalg.norm(full_vec), 1e-15))
    total_etch_error = abs(float(np.sum(transport_vec) - np.sum(full_vec))) / max(abs(float(np.sum(full_vec))), 1e-15)

    floor_old = old_response.get("etch_floor", 0.0)
    floor_new = full_response.get("etch_floor", 0.0)
    recess_wall_new = full_response.get("recess_left", 0.0) + full_response.get("recess_right", 0.0)
    trench_wall_new = full_response.get("left_wall", 0.0) + full_response.get("right_wall", 0.0)

    return {
        "scenario": f"recess_w{recess_width:.2f}_d{recess_depth:.2f}",
        "model_scope": "feature_scale_directional_etch_line_of_sight",
        "source_samples": source_samples,
        "angle_samples": angle_samples,
        "n_rays": rays.n_rays,
        "old_segment_count": len(old_geometry.segments),
        "new_segment_count": len(new_geometry.segments),
        "recess_width": recess_width,
        "recess_depth": recess_depth,
        "yield_power": yield_power,
        "yield_threshold": yield_threshold,
        "max_angle_deg": max_angle_deg,
        "collimation_power": collimation_power,
        "candidate_rays": int(np.count_nonzero(candidate)),
        "active_rays": int(np.count_nonzero(active)),
        "false_negative_rays": false_negative_rays,
        "target_mismatch_rays": target_mismatch_rays,
        "yield_max_abs_error": yield_max_abs_error,
        "candidate_fraction": float(np.count_nonzero(candidate) / max(rays.n_rays, 1)),
        "active_fraction": float(np.count_nonzero(active) / max(rays.n_rays, 1)),
        "candidate_per_active": float(np.count_nonzero(candidate) / max(np.count_nonzero(active), 1)),
        "etch_rel_error": etch_rel_error,
        "total_etch_rel_error": total_etch_error,
        "old_total_etch": float(np.sum(old_vec)),
        "full_new_total_etch": float(np.sum(full_vec)),
        "transport_total_etch": float(np.sum(transport_vec)),
        "floor_etch_old": floor_old,
        "floor_etch_new": floor_new,
        "recess_wall_etch_new": recess_wall_new,
        "trench_wall_etch_new": trench_wall_new,
        "recess_wall_share_new": recess_wall_new / max(float(np.sum(full_vec)), 1e-15),
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


def run_default_directional_etch_demo(
    source_samples: int = 161,
    angle_samples: int = 31,
    timing_repeats: int = 5,
) -> list[dict[str, float | int | str]]:
    cases = [(0.18, 0.18), (0.30, 0.26), (0.48, 0.40), (0.72, 0.56), (1.00, 0.72)]
    return [
        evaluate_directional_etch_recess_update(
            recess_width=width,
            recess_depth=depth,
            source_samples=source_samples,
            angle_samples=angle_samples,
            timing_repeats=timing_repeats,
        )
        for width, depth in cases
    ]


def write_directional_etch_demo_csv(rows: list[dict[str, float | int | str]], output_csv: str | Path) -> None:
    if not rows:
        raise ValueError("rows must not be empty.")
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_directional_etch_demo_rows(
    rows: list[dict[str, float | int | str]],
    output_png: str | Path,
    output_pdf: str | Path | None = None,
) -> None:
    cache_dir = Path("artifacts/matplotlib").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib.pyplot as plt

    widths = np.array([float(row["recess_width"]) for row in rows])
    total_change = np.array(
        [
            100.0 * (float(row["full_new_total_etch"]) - float(row["old_total_etch"])) / max(float(row["old_total_etch"]), 1e-15)
            for row in rows
        ]
    )
    floor_change = np.array(
        [
            100.0 * (float(row["floor_etch_new"]) - float(row["floor_etch_old"])) / max(float(row["floor_etch_old"]), 1e-15)
            for row in rows
        ]
    )
    candidate = np.array([100.0 * float(row["candidate_fraction"]) for row in rows])
    active = np.array([100.0 * float(row["active_fraction"]) for row in rows])
    query_speedup = np.array([float(row["candidate_route_query_speedup_vs_full"]) for row in rows])
    wall_speedup = np.array([float(row["candidate_route_wall_speedup_vs_full"]) for row in rows])

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
    axes[0].plot(widths, total_change, marker="o", label="total etch")
    axes[0].plot(widths, floor_change, marker="s", linestyle="--", label="floor etch")
    axes[0].axhline(0.0, color="0.25", linewidth=0.8)
    axes[0].set_xlabel("Etched recess width")
    axes[0].set_ylabel("Directional etch change (%)")
    axes[0].set_title("Feature-scale etch response")
    axes[0].legend(fontsize=8)

    axes[1].plot(widths, candidate, marker="o", label="candidate rays")
    axes[1].plot(widths, active, marker="s", linestyle="--", label="active rays")
    axes[1].set_xlabel("Etched recess width")
    axes[1].set_ylabel("Rays (%)")
    axes[1].set_title("Local recompute frontier")
    axes[1].legend(fontsize=8)

    axes[2].plot(widths, query_speedup, marker="o", label="intersection tests")
    axes[2].plot(widths, wall_speedup, marker="s", linestyle="--", label="wall time")
    axes[2].axhline(1.0, color="0.25", linewidth=0.8)
    axes[2].set_xlabel("Etched recess width")
    axes[2].set_ylabel("Speedup vs full new trace")
    axes[2].set_title("Footprint fallback work")
    axes[2].legend(fontsize=8)

    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.suptitle("Feature-scale directional etch local-recompute transport")

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    if output_pdf is not None:
        output_pdf = Path(output_pdf)
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf)
    plt.close(fig)
