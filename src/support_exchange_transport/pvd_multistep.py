from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .pvd_demo import (
    PVDGeometry,
    PVDRayBundle,
    PVDSegment,
    PVDTrace,
    make_pvd_rays,
    trace_pvd_first_hits,
)
from .pvd_reference import trace_pvd_first_hits_independent


PROFILE_SEGMENT_OFFSET = 4


class SupportContainmentError(ValueError):
    pass


@dataclass(frozen=True)
class PVDProfile2D:
    x: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=float)
        y = np.asarray(self.y, dtype=float)
        if x.ndim != 1 or y.shape != x.shape or len(x) < 3:
            raise ValueError("profile x and y must be one-dimensional arrays of equal length.")
        if np.any(np.diff(x) <= 0.0):
            raise ValueError("profile x coordinates must be strictly increasing.")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("profile coordinates must be finite.")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)

    @property
    def n_segments(self) -> int:
        return len(self.x) - 1


@dataclass(frozen=True)
class EnclosingSupport2D:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def __post_init__(self) -> None:
        if not self.xmin < self.xmax or not self.ymin < self.ymax:
            raise ValueError("support bounds must be ordered and nondegenerate.")


@dataclass(frozen=True)
class PVDMultistepConfig:
    scenario: str = "centered_seed"
    steps: int = 20
    profile_nodes: int = 33
    trench_width: float = 1.2
    trench_depth: float = 2.4
    field_half_width: float = 2.8
    seed_height: float = 0.06
    seed_half_width: float = 0.16
    seed_center: float = 0.0
    growth_center: float = 0.0
    growth_window_half_width: float = 0.22
    support_padding_x: float = 0.08
    support_height: float = 0.75
    deposition_scale: float = 0.035
    max_step_displacement: float = 0.012
    source_samples: int = 80
    angle_samples: int = 20
    source_half_width: float = 3.2
    source_center: float = 0.0
    source_y: float = 1.2
    max_angle_deg: float = 72.0
    mean_angle_deg: float = 0.0
    collimation_power: float = 1.0
    quadrature_rule: str = "endpoint"
    quadrature_seed: int = 1729
    timing_repeats: int = 1
    timing_order: str = "route_full"
    independent_reference_interval: int = 1
    independent_reference_enabled: bool = True
    tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive.")
        if self.profile_nodes < 5:
            raise ValueError("profile_nodes must be at least 5.")
        if self.profile_nodes % 2 == 0:
            raise ValueError("profile_nodes must be odd so the profile has a center node.")
        if self.growth_window_half_width <= 0.0:
            raise ValueError("growth_window_half_width must be positive.")
        trench_half_width = 0.5 * self.trench_width
        if (
            abs(self.growth_center) + self.growth_window_half_width
            >= trench_half_width
        ):
            raise ValueError("the growth window must remain inside the trench.")
        if self.support_padding_x <= 0.0 or self.support_height <= 0.0:
            raise ValueError("support padding and height must be positive.")
        if self.deposition_scale <= 0.0 or self.max_step_displacement <= 0.0:
            raise ValueError("deposition controls must be positive.")
        if self.timing_repeats <= 0:
            raise ValueError("timing_repeats must be positive.")
        if self.timing_order not in {"route_full", "full_route"}:
            raise ValueError("timing_order must be 'route_full' or 'full_route'.")
        if self.independent_reference_interval <= 0:
            raise ValueError("independent_reference_interval must be positive.")


@dataclass(frozen=True)
class PVDMultistepResult:
    config: PVDMultistepConfig
    support: EnclosingSupport2D
    candidate_mask: np.ndarray
    rows: tuple[dict[str, float | int | str | bool], ...]
    route_profile: PVDProfile2D
    full_profile: PVDProfile2D


def make_initial_profile(config: PVDMultistepConfig) -> PVDProfile2D:
    half = 0.5 * config.trench_width
    x = np.linspace(-half, half, config.profile_nodes)
    y_bottom = -config.trench_depth
    distance = np.abs(x - config.seed_center)
    height = np.zeros_like(x)
    inside = distance < config.seed_half_width
    height[inside] = config.seed_height * 0.5 * (
        1.0 + np.cos(np.pi * distance[inside] / config.seed_half_width)
    )
    height[0] = 0.0
    height[-1] = 0.0
    return PVDProfile2D(x=x, y=y_bottom + height)


def make_enclosing_support(config: PVDMultistepConfig) -> EnclosingSupport2D:
    y_bottom = -config.trench_depth
    half_width = config.growth_window_half_width + config.support_padding_x
    return EnclosingSupport2D(
        xmin=config.growth_center - half_width,
        xmax=config.growth_center + half_width,
        ymin=y_bottom - 1e-8,
        ymax=y_bottom + config.support_height,
    )


def _upward_segment_normal(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    edge = end - start
    normal = np.array([-edge[1], edge[0]], dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        raise ValueError("profile contains a degenerate segment.")
    return normal / norm


def make_profile_trench_geometry(
    profile: PVDProfile2D,
    trench_width: float = 1.2,
    field_half_width: float = 2.8,
) -> PVDGeometry:
    half = 0.5 * trench_width
    if not np.isclose(profile.x[0], -half) or not np.isclose(profile.x[-1], half):
        raise ValueError("profile endpoints must match the trench width.")
    if field_half_width <= half:
        raise ValueError("field_half_width must exceed the trench half-width.")

    segments: list[PVDSegment] = [
        PVDSegment(
            np.array([-field_half_width, 0.0]),
            np.array([-half, 0.0]),
            np.array([0.0, 1.0]),
            "mask_left",
        ),
        PVDSegment(
            np.array([half, 0.0]),
            np.array([field_half_width, 0.0]),
            np.array([0.0, 1.0]),
            "mask_right",
        ),
        PVDSegment(
            np.array([-half, 0.0]),
            np.array([-half, profile.y[0]]),
            np.array([1.0, 0.0]),
            "left_wall",
        ),
        PVDSegment(
            np.array([half, profile.y[-1]]),
            np.array([half, 0.0]),
            np.array([-1.0, 0.0]),
            "right_wall",
        ),
    ]
    for segment_idx in range(profile.n_segments):
        start = np.array([profile.x[segment_idx], profile.y[segment_idx]])
        end = np.array([profile.x[segment_idx + 1], profile.y[segment_idx + 1]])
        segments.append(
            PVDSegment(
                start=start,
                end=end,
                normal=_upward_segment_normal(start, end),
                label=f"profile_{segment_idx:03d}",
            )
        )
    return PVDGeometry(segments=tuple(segments))


def _ray_intersects_support(
    origin: np.ndarray,
    direction: np.ndarray,
    support: EnclosingSupport2D,
    min_t: float = 1e-9,
) -> bool:
    t_entry = 0.0
    t_exit = float("inf")
    for axis, lower, upper in (
        (0, support.xmin, support.xmax),
        (1, support.ymin, support.ymax),
    ):
        component = float(direction[axis])
        if abs(component) <= 1e-14:
            if origin[axis] < lower or origin[axis] > upper:
                return False
            continue
        t0 = (lower - origin[axis]) / component
        t1 = (upper - origin[axis]) / component
        if t0 > t1:
            t0, t1 = t1, t0
        t_entry = max(t_entry, t0)
        t_exit = min(t_exit, t1)
        if t_exit < t_entry:
            return False
    return t_exit > min_t


def enclosing_support_candidate_mask(
    rays: PVDRayBundle,
    support: EnclosingSupport2D,
) -> np.ndarray:
    return np.asarray(
        [
            _ray_intersects_support(origin, direction, support)
            for origin, direction in zip(rays.origins, rays.directions)
        ],
        dtype=bool,
    )


def _changed_profile_segment_indices(
    old_profile: PVDProfile2D,
    new_profile: PVDProfile2D,
    tolerance: float,
) -> np.ndarray:
    if not np.array_equal(old_profile.x, new_profile.x):
        raise SupportContainmentError("profile x coordinates changed.")
    changed_nodes = np.abs(new_profile.y - old_profile.y) > tolerance
    return np.flatnonzero(changed_nodes[:-1] | changed_nodes[1:])


def validate_profile_change_contained(
    old_profile: PVDProfile2D,
    new_profile: PVDProfile2D,
    support: EnclosingSupport2D,
    tolerance: float = 1e-12,
) -> None:
    changed_segments = _changed_profile_segment_indices(old_profile, new_profile, tolerance)
    for segment_idx in changed_segments:
        x_values = old_profile.x[segment_idx : segment_idx + 2]
        y_values = np.concatenate(
            (
                old_profile.y[segment_idx : segment_idx + 2],
                new_profile.y[segment_idx : segment_idx + 2],
            )
        )
        if (
            float(np.min(x_values)) < support.xmin - tolerance
            or float(np.max(x_values)) > support.xmax + tolerance
            or float(np.min(y_values)) < support.ymin - tolerance
            or float(np.max(y_values)) > support.ymax + tolerance
        ):
            raise SupportContainmentError(
                f"changed profile segment {segment_idx} lies outside the enclosing support."
            )


def transport_pvd_enclosing_support(
    old_trace: PVDTrace,
    new_geometry: PVDGeometry,
    rays: PVDRayBundle,
    candidate_mask: np.ndarray,
) -> PVDTrace:
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if candidate_mask.shape != (rays.n_rays,):
        raise ValueError("candidate_mask must have one entry per ray.")
    if len(old_trace.labels) != len(new_geometry.labels):
        raise ValueError("enclosing-support transport requires stable segment topology.")

    targets = np.array(old_trace.targets, copy=True)
    taus = np.array(old_trace.taus, copy=True)
    incidence = np.array(old_trace.incidence, copy=True)
    candidate_indices = np.flatnonzero(candidate_mask)
    query_count = 0
    if len(candidate_indices):
        candidate_rays = PVDRayBundle(
            origins=rays.origins[candidate_indices],
            directions=rays.directions[candidate_indices],
            weights=rays.weights[candidate_indices],
        )
        candidate_trace = trace_pvd_first_hits(new_geometry, candidate_rays)
        targets[candidate_indices] = candidate_trace.targets
        taus[candidate_indices] = candidate_trace.taus
        incidence[candidate_indices] = candidate_trace.incidence
        query_count = candidate_trace.query_count

    return PVDTrace(
        targets=targets,
        taus=taus,
        labels=new_geometry.labels,
        incidence=incidence,
        weights=rays.weights.copy(),
        query_count=query_count,
    )


def trace_segment_flux(trace: PVDTrace, segment_count: int) -> np.ndarray:
    valid = trace.targets >= 0
    contributions = trace.weights[valid] * trace.incidence[valid]
    return np.bincount(
        trace.targets[valid],
        weights=contributions,
        minlength=segment_count,
    ).astype(float)


def profile_deposition_density(
    profile: PVDProfile2D,
    geometry: PVDGeometry,
    trace: PVDTrace,
) -> np.ndarray:
    segment_flux = trace_segment_flux(trace, len(geometry.segments))[
        PROFILE_SEGMENT_OFFSET:
    ]
    lengths = np.linalg.norm(
        np.diff(np.column_stack((profile.x, profile.y)), axis=0),
        axis=1,
    )
    segment_density = segment_flux / np.maximum(lengths, 1e-15)
    node_density = np.zeros_like(profile.y)
    node_density[0] = segment_density[0]
    node_density[-1] = segment_density[-1]
    node_density[1:-1] = 0.5 * (segment_density[:-1] + segment_density[1:])
    smoothed = node_density.copy()
    smoothed[1:-1] = (
        node_density[:-2] + 2.0 * node_density[1:-1] + node_density[2:]
    ) / 4.0
    return smoothed


def advance_profile(
    profile: PVDProfile2D,
    geometry: PVDGeometry,
    trace: PVDTrace,
    config: PVDMultistepConfig,
) -> PVDProfile2D:
    density = profile_deposition_density(profile, geometry, trace)
    distance = np.abs(profile.x - config.growth_center)
    window = np.zeros_like(profile.x)
    inside = distance < config.growth_window_half_width
    window[inside] = 0.5 * (
        1.0 + np.cos(np.pi * distance[inside] / config.growth_window_half_width)
    )
    displacement = config.deposition_scale * density * window
    displacement = np.minimum(displacement, config.max_step_displacement)
    displacement[0] = 0.0
    displacement[-1] = 0.0
    return PVDProfile2D(x=profile.x.copy(), y=profile.y + displacement)


def _relative_error(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(actual) - np.asarray(reference))
        / max(float(np.linalg.norm(reference)), 1e-15)
    )


def _timed_call(callable_, repeats: int):
    elapsed: list[float] = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = callable_()
        elapsed.append(time.perf_counter() - start)
    return result, float(np.median(np.asarray(elapsed, dtype=float)))


def _finite_tau_error(actual: np.ndarray, reference: np.ndarray) -> float:
    finite = np.isfinite(actual) & np.isfinite(reference)
    if not np.any(finite):
        return 0.0
    return float(np.max(np.abs(actual[finite] - reference[finite])))


def _tau_changed(
    old_taus: np.ndarray,
    new_taus: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    old_finite = np.isfinite(old_taus)
    new_finite = np.isfinite(new_taus)
    changed = old_finite != new_finite
    both_finite = old_finite & new_finite
    changed[both_finite] |= (
        np.abs(old_taus[both_finite] - new_taus[both_finite]) > tolerance
    )
    return changed


def _advance_profile_geometry(
    profile: PVDProfile2D,
    geometry: PVDGeometry,
    trace: PVDTrace,
    config: PVDMultistepConfig,
    support: EnclosingSupport2D,
) -> tuple[PVDProfile2D, PVDGeometry]:
    next_profile = advance_profile(profile, geometry, trace, config)
    next_geometry = make_profile_trench_geometry(
        next_profile,
        trench_width=config.trench_width,
        field_half_width=config.field_half_width,
    )
    validate_profile_change_contained(
        profile,
        next_profile,
        support,
        tolerance=config.tolerance,
    )
    return next_profile, next_geometry


def run_multistep_pvd(config: PVDMultistepConfig) -> PVDMultistepResult:
    rays = make_pvd_rays(
        source_samples=config.source_samples,
        angle_samples=config.angle_samples,
        source_half_width=config.source_half_width,
        source_center=config.source_center,
        source_y=config.source_y,
        max_angle_deg=config.max_angle_deg,
        mean_angle_deg=config.mean_angle_deg,
        collimation_power=config.collimation_power,
        quadrature_rule=config.quadrature_rule,
        quadrature_seed=config.quadrature_seed,
    )
    support = make_enclosing_support(config)
    discovery_start = time.perf_counter()
    candidate_mask = enclosing_support_candidate_mask(rays, support)
    candidate_discovery_wall_s = time.perf_counter() - discovery_start

    route_profile = make_initial_profile(config)
    full_profile = make_initial_profile(config)
    route_geometry = make_profile_trench_geometry(
        route_profile,
        trench_width=config.trench_width,
        field_half_width=config.field_half_width,
    )
    full_geometry = make_profile_trench_geometry(
        full_profile,
        trench_width=config.trench_width,
        field_half_width=config.field_half_width,
    )
    route_trace, cache_wall_s = _timed_call(
        lambda: trace_pvd_first_hits(route_geometry, rays),
        config.timing_repeats,
    )
    full_trace = route_trace

    route_cumulative_wall_s = cache_wall_s + candidate_discovery_wall_s
    full_cumulative_wall_s = cache_wall_s
    rows: list[dict[str, float | int | str | bool]] = []

    for step in range(1, config.steps + 1):
        def time_route_update():
            return _timed_call(
                lambda: _advance_profile_geometry(
                    route_profile,
                    route_geometry,
                    route_trace,
                    config,
                    support,
                ),
                config.timing_repeats,
            )

        def time_full_update():
            return _timed_call(
                lambda: _advance_profile_geometry(
                    full_profile,
                    full_geometry,
                    full_trace,
                    config,
                    support,
                ),
                config.timing_repeats,
            )

        if config.timing_order == "route_full":
            route_update, route_profile_update_wall_s = time_route_update()
            full_update, full_profile_update_wall_s = time_full_update()
        else:
            full_update, full_profile_update_wall_s = time_full_update()
            route_update, route_profile_update_wall_s = time_route_update()
        route_next, route_new_geometry = route_update
        full_next, full_new_geometry = full_update

        def time_route_trace():
            return _timed_call(
                lambda: transport_pvd_enclosing_support(
                    route_trace,
                    route_new_geometry,
                    rays,
                    candidate_mask,
                ),
                config.timing_repeats,
            )

        def time_full_trace():
            return _timed_call(
                lambda: trace_pvd_first_hits(full_new_geometry, rays),
                config.timing_repeats,
            )

        if config.timing_order == "route_full":
            route_new_trace, route_wall_s = time_route_trace()
            full_new_trace, full_wall_s = time_full_trace()
        else:
            full_new_trace, full_wall_s = time_full_trace()
            route_new_trace, route_wall_s = time_route_trace()

        active = (
            (full_trace.targets != full_new_trace.targets)
            | _tau_changed(
                full_trace.taus,
                full_new_trace.taus,
                config.tolerance,
            )
            | (np.abs(full_trace.incidence - full_new_trace.incidence) > config.tolerance)
        )
        false_negatives = int(np.count_nonzero(active & ~candidate_mask))
        target_mismatches = int(
            np.count_nonzero(route_new_trace.targets != full_new_trace.targets)
        )
        incidence_max_abs_error = float(
            np.max(np.abs(route_new_trace.incidence - full_new_trace.incidence))
        )
        tau_max_abs_error = _finite_tau_error(
            route_new_trace.taus,
            full_new_trace.taus,
        )
        route_flux = trace_segment_flux(
            route_new_trace,
            len(route_new_geometry.segments),
        )
        full_flux = trace_segment_flux(
            full_new_trace,
            len(full_new_geometry.segments),
        )
        flux_rel_error = _relative_error(route_flux, full_flux)
        profile_rel_error = _relative_error(
            route_next.y + config.trench_depth,
            full_next.y + config.trench_depth,
        )

        independent_checked = config.independent_reference_enabled and (
            step == 1
            or step == config.steps
            or step % config.independent_reference_interval == 0
        )
        independent_wall_s = 0.0
        independent_target_mismatches = 0
        independent_tau_max_abs_error = 0.0
        independent_incidence_max_abs_error = 0.0
        if independent_checked:
            reference_start = time.perf_counter()
            independent = trace_pvd_first_hits_independent(full_new_geometry, rays)
            independent_wall_s = time.perf_counter() - reference_start
            independent_target_mismatches = int(
                np.count_nonzero(independent.targets != full_new_trace.targets)
            )
            independent_tau_max_abs_error = _finite_tau_error(
                independent.taus,
                full_new_trace.taus,
            )
            independent_incidence_max_abs_error = float(
                np.max(np.abs(independent.incidence - full_new_trace.incidence))
            )

        route_step_wall_s = route_profile_update_wall_s + route_wall_s
        full_step_wall_s = full_profile_update_wall_s + full_wall_s
        route_cumulative_wall_s += route_step_wall_s
        full_cumulative_wall_s += full_step_wall_s
        profile_height = full_next.y + config.trench_depth
        total_flux = float(np.sum(full_flux))
        mask_flux = float(np.sum(full_flux[:2]))
        wall_flux = float(np.sum(full_flux[2:4]))
        profile_flux = float(np.sum(full_flux[PROFILE_SEGMENT_OFFSET:]))
        mask_length = 2.0 * (
            config.field_half_width - 0.5 * config.trench_width
        )
        profile_length = float(
            np.sum(
                np.linalg.norm(
                    np.diff(
                        np.column_stack((full_next.x, full_next.y)),
                        axis=0,
                    ),
                    axis=1,
                )
            )
        )
        deposited_volume = float(np.trapezoid(profile_height, full_next.x))
        if deposited_volume > 1e-15:
            deposition_centroid_x = float(
                np.trapezoid(full_next.x * profile_height, full_next.x)
                / deposited_volume
            )
            left_volume = float(
                np.trapezoid(
                    profile_height[full_next.x <= config.growth_center],
                    full_next.x[full_next.x <= config.growth_center],
                )
            )
            right_volume = float(
                np.trapezoid(
                    profile_height[full_next.x >= config.growth_center],
                    full_next.x[full_next.x >= config.growth_center],
                )
            )
            deposition_asymmetry = (
                right_volume - left_volume
            ) / deposited_volume
        else:
            deposition_centroid_x = config.growth_center
            deposition_asymmetry = 0.0
        changed_height = np.maximum(full_next.y - full_profile.y, 0.0)
        changed_volume = float(np.trapezoid(changed_height, full_next.x))
        if changed_volume > 1e-15:
            changed_profile_centroid_x = float(
                np.trapezoid(full_next.x * changed_height, full_next.x)
                / changed_volume
            )
        else:
            changed_profile_centroid_x = config.growth_center
        active_valid = active & np.isfinite(full_new_trace.taus)
        if np.any(active_valid):
            active_hit_x = (
                rays.origins[active_valid, 0]
                + full_new_trace.taus[active_valid]
                * rays.directions[active_valid, 0]
            )
            active_weights = rays.weights[active_valid]
            active_hit_x_centroid = float(
                np.average(active_hit_x, weights=active_weights)
            )
            active_hit_x_min = float(np.min(active_hit_x))
            active_hit_x_max = float(np.max(active_hit_x))
        else:
            active_hit_x_centroid = config.growth_center
            active_hit_x_min = config.growth_center
            active_hit_x_max = config.growth_center
        rows.append(
            {
                "scenario": config.scenario,
                "step": step,
                "model_scope": "2d_multistep_ballistic_pvd_height_field",
                "n_rays": rays.n_rays,
                "profile_nodes": config.profile_nodes,
                "segment_count": len(full_new_geometry.segments),
                "support_xmin": support.xmin,
                "support_xmax": support.xmax,
                "support_ymin": support.ymin,
                "support_ymax": support.ymax,
                "candidate_rays": int(np.count_nonzero(candidate_mask)),
                "candidate_fraction": float(np.mean(candidate_mask)),
                "active_rays": int(np.count_nonzero(active)),
                "active_fraction": float(np.mean(active)),
                "false_negative_rays": false_negatives,
                "target_mismatch_rays": target_mismatches,
                "tau_max_abs_error": tau_max_abs_error,
                "incidence_max_abs_error": incidence_max_abs_error,
                "flux_rel_error": flux_rel_error,
                "profile_rel_error": profile_rel_error,
                "max_profile_height": float(np.max(profile_height)),
                "integrated_profile_height": deposited_volume,
                "deposition_centroid_x": deposition_centroid_x,
                "deposition_asymmetry_index": deposition_asymmetry,
                "changed_profile_centroid_x": changed_profile_centroid_x,
                "active_hit_x_centroid": active_hit_x_centroid,
                "active_hit_x_min": active_hit_x_min,
                "active_hit_x_max": active_hit_x_max,
                "total_captured_flux": total_flux,
                "profile_capture_fraction": profile_flux
                / max(total_flux, 1e-15),
                "sidewall_capture_fraction": wall_flux
                / max(total_flux, 1e-15),
                "bottom_to_mask_step_coverage": (
                    profile_flux / max(profile_length, 1e-15)
                )
                / (mask_flux / max(mask_length, 1e-15)),
                "independent_reference_checked": independent_checked,
                "independent_target_mismatch_rays": independent_target_mismatches,
                "independent_tau_max_abs_error": independent_tau_max_abs_error,
                "independent_incidence_max_abs_error": independent_incidence_max_abs_error,
                "candidate_discovery_wall_time_s": candidate_discovery_wall_s,
                "cache_build_wall_time_s": cache_wall_s,
                "full_profile_update_wall_time_s": full_profile_update_wall_s,
                "candidate_profile_update_wall_time_s": route_profile_update_wall_s,
                "full_trace_wall_time_s": full_wall_s,
                "candidate_trace_wall_time_s": route_wall_s,
                "full_step_wall_time_s": full_step_wall_s,
                "candidate_step_wall_time_s": route_step_wall_s,
                "independent_reference_wall_time_s": independent_wall_s,
                "step_wall_speedup_vs_full": full_step_wall_s
                / max(route_step_wall_s, 1e-15),
                "full_cumulative_wall_time_s": full_cumulative_wall_s,
                "candidate_cumulative_wall_time_s": route_cumulative_wall_s,
                "cumulative_wall_speedup_vs_full": full_cumulative_wall_s
                / max(route_cumulative_wall_s, 1e-15),
                "full_step_query_count": full_new_trace.query_count,
                "candidate_step_query_count": route_new_trace.query_count,
                "query_speedup_vs_full": full_new_trace.query_count
                / max(route_new_trace.query_count, 1),
                "support_containment_pass": True,
            }
        )

        route_profile = route_next
        full_profile = full_next
        route_geometry = route_new_geometry
        full_geometry = full_new_geometry
        route_trace = route_new_trace
        full_trace = full_new_trace

    return PVDMultistepResult(
        config=config,
        support=support,
        candidate_mask=candidate_mask,
        rows=tuple(rows),
        route_profile=route_profile,
        full_profile=full_profile,
    )
