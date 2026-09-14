from __future__ import annotations

import numpy as np
from numba import njit

from .geometry2d import SegmentMesh2D, expand_segment_mask
from .raytrace2d import FirstHitMap2D
from .rays2d import RayBundle2D, make_lambertian_rays


@njit(cache=True)
def _intersect_tau(
    ox: float,
    oy: float,
    dx: float,
    dy: float,
    sx: float,
    sy: float,
    ex: float,
    ey: float,
) -> float:
    edge_x = ex - sx
    edge_y = ey - sy
    denom = dx * edge_y - dy * edge_x
    if abs(denom) <= 1e-10:
        return np.inf

    delta_x = sx - ox
    delta_y = sy - oy
    tau = (delta_x * edge_y - delta_y * edge_x) / denom
    u = (delta_x * dy - delta_y * dx) / denom
    if tau > 1e-9 and -1e-10 <= u <= 1.0 + 1e-10:
        return tau
    return np.inf


@njit(cache=True)
def _trace_full_kernel(
    origins: np.ndarray,
    directions: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    n_sources = origins.shape[0]
    n_rays = origins.shape[1]
    n_segments = starts.shape[0]
    hits = np.full((n_sources, n_rays), -1, dtype=np.int64)
    taus = np.full((n_sources, n_rays), np.inf, dtype=np.float64)
    query_count = 0

    for source in range(n_sources):
        for ray_idx in range(n_rays):
            best_target = -1
            best_tau = np.inf
            ox = origins[source, ray_idx, 0]
            oy = origins[source, ray_idx, 1]
            dx = directions[source, ray_idx, 0]
            dy = directions[source, ray_idx, 1]
            for target in range(n_segments):
                if target == source:
                    continue
                query_count += 1
                tau = _intersect_tau(
                    ox,
                    oy,
                    dx,
                    dy,
                    starts[target, 0],
                    starts[target, 1],
                    ends[target, 0],
                    ends[target, 1],
                )
                if tau < best_tau:
                    best_tau = tau
                    best_target = target
            hits[source, ray_idx] = best_target
            taus[source, ray_idx] = best_tau

    return hits, taus, query_count


@njit(cache=True)
def _trace_support_local_kernel(
    origins: np.ndarray,
    directions: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    old_hits: np.ndarray,
    active_sources: np.ndarray,
    changed_targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    n_sources = origins.shape[0]
    n_rays = origins.shape[1]
    n_segments = starts.shape[0]
    hits = np.full((n_sources, n_rays), -1, dtype=np.int64)
    taus = np.full((n_sources, n_rays), np.inf, dtype=np.float64)
    query_count = 0

    for source in range(n_sources):
        source_is_active = active_sources[source]
        for ray_idx in range(n_rays):
            best_target = -1
            best_tau = np.inf
            ox = origins[source, ray_idx, 0]
            oy = origins[source, ray_idx, 1]
            dx = directions[source, ray_idx, 0]
            dy = directions[source, ray_idx, 1]

            if source_is_active:
                for target in range(n_segments):
                    if target == source:
                        continue
                    query_count += 1
                    tau = _intersect_tau(
                        ox,
                        oy,
                        dx,
                        dy,
                        starts[target, 0],
                        starts[target, 1],
                        ends[target, 0],
                        ends[target, 1],
                    )
                    if tau < best_tau:
                        best_tau = tau
                        best_target = target
            else:
                old_target = old_hits[source, ray_idx]
                old_target_was_checked = False
                for k in range(changed_targets.shape[0]):
                    target = changed_targets[k]
                    if target == source:
                        continue
                    if target == old_target:
                        old_target_was_checked = True
                    query_count += 1
                    tau = _intersect_tau(
                        ox,
                        oy,
                        dx,
                        dy,
                        starts[target, 0],
                        starts[target, 1],
                        ends[target, 0],
                        ends[target, 1],
                    )
                    if tau < best_tau:
                        best_tau = tau
                        best_target = target

                if old_target >= 0 and old_target != source and not old_target_was_checked:
                    query_count += 1
                    tau = _intersect_tau(
                        ox,
                        oy,
                        dx,
                        dy,
                        starts[old_target, 0],
                        starts[old_target, 1],
                        ends[old_target, 0],
                        ends[old_target, 1],
                    )
                    if tau < best_tau:
                        best_tau = tau
                        best_target = old_target

            hits[source, ray_idx] = best_target
            taus[source, ray_idx] = best_tau

    return hits, taus, query_count


@njit(cache=True)
def _wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@njit(cache=True)
def _mark_interval(
    footprint_counts: np.ndarray,
    source: int,
    center: float,
    half_width: float,
    local_angles: np.ndarray,
) -> None:
    n_rays = local_angles.shape[0]
    if n_rays == 1:
        if abs(_wrap_angle(local_angles[0] - center)) <= half_width:
            footprint_counts[source, 0] += 1
        return

    angle_min = local_angles[0]
    angle_max = local_angles[n_rays - 1]
    lo = center - half_width
    hi = center + half_width

    if half_width >= np.pi or lo < -np.pi or hi > np.pi:
        for ray_idx in range(n_rays):
            if abs(_wrap_angle(local_angles[ray_idx] - center)) <= half_width:
                footprint_counts[source, ray_idx] += 1
        return

    if hi < angle_min or lo > angle_max:
        return

    step = (angle_max - angle_min) / (n_rays - 1)
    start = int(np.floor((lo - angle_min) / step))
    end = int(np.ceil((hi - angle_min) / step))
    if start < 0:
        start = 0
    if end >= n_rays:
        end = n_rays - 1

    for ray_idx in range(start, end + 1):
        if abs(_wrap_angle(local_angles[ray_idx] - center)) <= half_width:
            footprint_counts[source, ray_idx] += 1


@njit(cache=True)
def _mark_interval_bool(
    ray_mask: np.ndarray,
    center: float,
    half_width: float,
    local_angles: np.ndarray,
) -> None:
    n_rays = local_angles.shape[0]
    if n_rays == 1:
        if abs(_wrap_angle(local_angles[0] - center)) <= half_width:
            ray_mask[0] = True
        return

    angle_min = local_angles[0]
    angle_max = local_angles[n_rays - 1]
    lo = center - half_width
    hi = center + half_width

    if half_width >= np.pi or lo < -np.pi or hi > np.pi:
        for ray_idx in range(n_rays):
            if abs(_wrap_angle(local_angles[ray_idx] - center)) <= half_width:
                ray_mask[ray_idx] = True
        return

    if hi < angle_min or lo > angle_max:
        return

    step = (angle_max - angle_min) / (n_rays - 1)
    start = int(np.floor((lo - angle_min) / step))
    end = int(np.ceil((hi - angle_min) / step))
    if start < 0:
        start = 0
    if end >= n_rays:
        end = n_rays - 1

    for ray_idx in range(start, end + 1):
        if abs(_wrap_angle(local_angles[ray_idx] - center)) <= half_width:
            ray_mask[ray_idx] = True


@njit(cache=True)
def _build_shadow_footprint_mask_kernel(
    origins: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    tangents: np.ndarray,
    normals: np.ndarray,
    support_labels: np.ndarray,
    active_sources: np.ndarray,
    changed_targets: np.ndarray,
    local_angles: np.ndarray,
    tile_margin: float,
) -> tuple[np.ndarray, int, int]:
    n_sources = origins.shape[0]
    n_rays = origins.shape[1]
    footprint_counts = np.zeros((n_sources, n_rays), dtype=np.int64)
    footprint_ray_count = 0
    overlap_ray_count = 0
    max_label = 0
    for k in range(changed_targets.shape[0]):
        label = support_labels[changed_targets[k]]
        if label > max_label:
            max_label = label

    for source in range(n_sources):
        if active_sources[source]:
            for ray_idx in range(n_rays):
                footprint_counts[source, ray_idx] = 1
            footprint_ray_count += n_rays
            continue

        ox = origins[source, 0, 0]
        oy = origins[source, 0, 1]
        tangent_x = tangents[source, 0]
        tangent_y = tangents[source, 1]
        normal_x = normals[source, 0]
        normal_y = normals[source, 1]

        for support_label in range(1, max_label + 1):
            label_mask = np.zeros(n_rays, dtype=np.bool_)
            for k in range(changed_targets.shape[0]):
                target = changed_targets[k]
                if target == source or support_labels[target] != support_label:
                    continue

                v0x = starts[target, 0] - ox
                v0y = starts[target, 1] - oy
                v1x = ends[target, 0] - ox
                v1y = ends[target, 1] - oy

                a0 = np.arctan2(v0x * tangent_x + v0y * tangent_y, v0x * normal_x + v0y * normal_y)
                a1 = np.arctan2(v1x * tangent_x + v1y * tangent_y, v1x * normal_x + v1y * normal_y)

                center = np.arctan2(np.sin(a0) + np.sin(a1), np.cos(a0) + np.cos(a1))
                d0 = abs(_wrap_angle(a0 - center))
                d1 = abs(_wrap_angle(a1 - center))
                half_width = max(d0, d1) + tile_margin
                _mark_interval_bool(label_mask, center, half_width, local_angles)

            for ray_idx in range(n_rays):
                if label_mask[ray_idx]:
                    footprint_counts[source, ray_idx] += 1

        for k in range(changed_targets.shape[0]):
            target = changed_targets[k]
            if target == source or support_labels[target] > 0:
                continue

            v0x = starts[target, 0] - ox
            v0y = starts[target, 1] - oy
            v1x = ends[target, 0] - ox
            v1y = ends[target, 1] - oy

            a0 = np.arctan2(v0x * tangent_x + v0y * tangent_y, v0x * normal_x + v0y * normal_y)
            a1 = np.arctan2(v1x * tangent_x + v1y * tangent_y, v1x * normal_x + v1y * normal_y)

            center = np.arctan2(np.sin(a0) + np.sin(a1), np.cos(a0) + np.cos(a1))
            d0 = abs(_wrap_angle(a0 - center))
            d1 = abs(_wrap_angle(a1 - center))
            half_width = max(d0, d1) + tile_margin
            _mark_interval(footprint_counts, source, center, half_width, local_angles)

        for ray_idx in range(n_rays):
            if footprint_counts[source, ray_idx] > 0:
                footprint_ray_count += 1
                if footprint_counts[source, ray_idx] > 1:
                    overlap_ray_count += 1

    return footprint_counts > 0, footprint_ray_count, overlap_ray_count


@njit(cache=True)
def _trace_footprint_tiled_kernel(
    origins: np.ndarray,
    directions: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    old_hits: np.ndarray,
    old_taus: np.ndarray,
    active_sources: np.ndarray,
    changed_targets: np.ndarray,
    footprint_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    n_sources = origins.shape[0]
    n_rays = origins.shape[1]
    n_segments = starts.shape[0]
    hits = np.full((n_sources, n_rays), -1, dtype=np.int64)
    taus = np.full((n_sources, n_rays), np.inf, dtype=np.float64)
    query_count = 0

    for source in range(n_sources):
        source_is_active = active_sources[source]
        for ray_idx in range(n_rays):
            if not source_is_active and not footprint_mask[source, ray_idx]:
                hits[source, ray_idx] = old_hits[source, ray_idx]
                taus[source, ray_idx] = old_taus[source, ray_idx]
                continue

            best_target = -1
            best_tau = np.inf
            ox = origins[source, ray_idx, 0]
            oy = origins[source, ray_idx, 1]
            dx = directions[source, ray_idx, 0]
            dy = directions[source, ray_idx, 1]

            if source_is_active:
                for target in range(n_segments):
                    if target == source:
                        continue
                    query_count += 1
                    tau = _intersect_tau(
                        ox,
                        oy,
                        dx,
                        dy,
                        starts[target, 0],
                        starts[target, 1],
                        ends[target, 0],
                        ends[target, 1],
                    )
                    if tau < best_tau:
                        best_tau = tau
                        best_target = target
            else:
                old_target = old_hits[source, ray_idx]
                old_target_was_checked = False
                for k in range(changed_targets.shape[0]):
                    target = changed_targets[k]
                    if target == source:
                        continue
                    if target == old_target:
                        old_target_was_checked = True
                    query_count += 1
                    tau = _intersect_tau(
                        ox,
                        oy,
                        dx,
                        dy,
                        starts[target, 0],
                        starts[target, 1],
                        ends[target, 0],
                        ends[target, 1],
                    )
                    if tau < best_tau:
                        best_tau = tau
                        best_target = target

                if old_target >= 0 and old_target != source and not old_target_was_checked:
                    query_count += 1
                    tau = _intersect_tau(
                        ox,
                        oy,
                        dx,
                        dy,
                        starts[old_target, 0],
                        starts[old_target, 1],
                        ends[old_target, 0],
                        ends[old_target, 1],
                    )
                    if tau < best_tau:
                        best_tau = tau
                        best_target = old_target

            hits[source, ray_idx] = best_target
            taus[source, ray_idx] = best_tau

    return hits, taus, query_count


def trace_first_hits_accelerated(mesh: SegmentMesh2D, rays: RayBundle2D) -> FirstHitMap2D:
    if rays.n_sources != mesh.n_segments:
        raise ValueError("ray bundle source count must match mesh segment count.")
    hits, taus, query_count = _trace_full_kernel(rays.origins, rays.directions, mesh.starts, mesh.ends)
    return FirstHitMap2D(hits, taus, rays.weights.copy(), int(query_count))


def trace_support_local_accelerated(
    old_state: FirstHitMap2D,
    old_mesh: SegmentMesh2D,
    new_mesh: SegmentMesh2D,
    rays_per_source: int,
    neighbor_radius: int,
) -> tuple[FirstHitMap2D, int, int, int]:
    if old_mesh.n_segments != new_mesh.n_segments:
        raise ValueError("transport requires matching segment indexing.")
    if old_state.hits.shape != (old_mesh.n_segments, rays_per_source):
        raise ValueError("old_state shape does not match mesh/ray configuration.")

    changed_mask = expand_segment_mask(old_mesh.support_mask | new_mesh.support_mask, neighbor_radius)
    active_sources = changed_mask.copy()
    changed_targets = np.flatnonzero(changed_mask).astype(np.int64)
    new_rays = make_lambertian_rays(new_mesh, rays_per_source)
    hits, taus, query_count = _trace_support_local_kernel(
        new_rays.origins,
        new_rays.directions,
        new_mesh.starts,
        new_mesh.ends,
        old_state.hits.astype(np.int64, copy=False),
        active_sources.astype(np.bool_, copy=False),
        changed_targets,
    )
    state = FirstHitMap2D(hits, taus, new_rays.weights.copy(), int(query_count))
    return state, int(query_count), int(np.count_nonzero(active_sources)), int(len(changed_targets))


def trace_footprint_tiled_accelerated(
    old_state: FirstHitMap2D,
    old_mesh: SegmentMesh2D,
    new_mesh: SegmentMesh2D,
    rays_per_source: int,
    neighbor_radius: int,
    tile_margin: float,
) -> tuple[FirstHitMap2D, int, int, int, int]:
    if tile_margin < 0.0:
        raise ValueError("tile_margin must be non-negative.")
    if old_mesh.n_segments != new_mesh.n_segments:
        raise ValueError("transport requires matching segment indexing.")
    if old_state.hits.shape != (old_mesh.n_segments, rays_per_source):
        raise ValueError("old_state shape does not match mesh/ray configuration.")

    changed_mask = expand_segment_mask(old_mesh.support_mask | new_mesh.support_mask, neighbor_radius)
    active_sources = changed_mask.copy()
    changed_targets = np.flatnonzero(changed_mask).astype(np.int64)
    new_rays = make_lambertian_rays(new_mesh, rays_per_source)
    footprint_mask, footprint_ray_count, overlap_ray_count = _build_shadow_footprint_mask_kernel(
        new_rays.origins,
        new_mesh.starts,
        new_mesh.ends,
        new_mesh.tangents,
        new_mesh.normals,
        new_mesh.support_labels.astype(np.int64, copy=False),
        active_sources.astype(np.bool_, copy=False),
        changed_targets,
        new_rays.local_angles,
        tile_margin,
    )
    hits, taus, query_count = _trace_footprint_tiled_kernel(
        new_rays.origins,
        new_rays.directions,
        new_mesh.starts,
        new_mesh.ends,
        old_state.hits.astype(np.int64, copy=False),
        old_state.taus,
        active_sources.astype(np.bool_, copy=False),
        changed_targets,
        footprint_mask,
    )
    state = FirstHitMap2D(hits, taus, new_rays.weights.copy(), int(query_count))
    return (
        state,
        int(query_count),
        int(np.count_nonzero(active_sources)),
        int(len(changed_targets)),
        int(footprint_ray_count),
        int(overlap_ray_count),
    )


def warm_up_accelerated_kernels() -> None:
    from .geometry2d import BumpSpec, make_circle_cavity

    bumps = (BumpSpec(center=0.0, amplitude=0.05, width=0.2),)
    old_mesh = make_circle_cavity(16, bumps=bumps, alpha=0.0)
    new_mesh = make_circle_cavity(16, bumps=bumps, alpha=0.5)
    rays = make_lambertian_rays(old_mesh, 4)
    old_state = trace_first_hits_accelerated(old_mesh, rays)
    trace_support_local_accelerated(old_state, old_mesh, new_mesh, 4, 1)
    trace_footprint_tiled_accelerated(old_state, old_mesh, new_mesh, 4, 1, 0.01)
