from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry2d import SegmentMesh2D, expand_segment_mask
from .metrics import signed_delta_accumulator
from .rays2d import make_lambertian_rays
from .raytrace2d import FirstHitMap2D, trace_ray


@dataclass(frozen=True)
class TransportResult2D:
    state: FirstHitMap2D
    delta: dict[tuple[int, int], float]
    query_count: int
    event_count: int
    active_source_count: int
    changed_segment_count: int
    method: str = "support-local"
    footprint_ray_count: int = 0
    footprint_overlap_ray_count: int = 0


def _candidate_targets(old_target: int, changed_targets: np.ndarray) -> np.ndarray:
    if old_target >= 0:
        return np.unique(np.concatenate((changed_targets, np.array([old_target], dtype=int))))
    return changed_targets


def transport_first_hits(
    old_state: FirstHitMap2D,
    old_mesh: SegmentMesh2D,
    new_mesh: SegmentMesh2D,
    rays_per_source: int,
    neighbor_radius: int = 1,
    use_accelerated: bool = True,
) -> TransportResult2D:
    """Transport first-hit state using support-aware local correction.

    Source rows touching the deformation support are fully retraced because
    endpoint motion can create non-monotone visibility changes. Outside-source
    rows only test the previous target plus changed support targets, which is
    the 2D analogue of delta-BVH first-hit reassignment.
    """

    if old_mesh.n_segments != new_mesh.n_segments:
        raise ValueError("transport requires matching segment indexing.")
    if old_state.hits.shape != (old_mesh.n_segments, rays_per_source):
        raise ValueError("old_state shape does not match mesh/ray configuration.")

    if use_accelerated:
        from .accelerated2d import trace_support_local_accelerated

        state, query_count, active_source_count, changed_segment_count = trace_support_local_accelerated(
            old_state,
            old_mesh,
            new_mesh,
            rays_per_source,
            neighbor_radius,
        )
        delta = signed_delta_accumulator(old_state.hits, state.hits, state.weights)
        event_count = int(np.count_nonzero(old_state.hits != state.hits))
        return TransportResult2D(
            state=state,
            delta=delta,
            query_count=query_count,
            event_count=event_count,
            active_source_count=active_source_count,
            changed_segment_count=changed_segment_count,
            method="support-local",
        )

    changed_mask = expand_segment_mask(old_mesh.support_mask | new_mesh.support_mask, neighbor_radius)
    active_sources = changed_mask.copy()
    changed_targets = np.flatnonzero(changed_mask)
    new_rays = make_lambertian_rays(new_mesh, rays_per_source)

    hits = np.full_like(old_state.hits, -1)
    taus = np.full_like(old_state.taus, np.inf, dtype=float)
    query_count = 0

    for source in range(new_mesh.n_segments):
        source_is_active = bool(active_sources[source])
        for ray_idx in range(rays_per_source):
            candidates = None
            if not source_is_active:
                candidates = _candidate_targets(int(old_state.hits[source, ray_idx]), changed_targets)

            hit = trace_ray(
                new_rays.origins[source, ray_idx],
                new_rays.directions[source, ray_idx],
                new_mesh,
                source_index=source,
                candidates=candidates,
            )
            hits[source, ray_idx] = hit.target
            taus[source, ray_idx] = hit.tau
            query_count += hit.query_count

    state = FirstHitMap2D(hits, taus, new_rays.weights.copy(), query_count)
    delta = signed_delta_accumulator(old_state.hits, state.hits, state.weights)
    event_count = int(np.count_nonzero(old_state.hits != state.hits))

    return TransportResult2D(
        state=state,
        delta=delta,
        query_count=query_count,
        event_count=event_count,
        active_source_count=int(np.count_nonzero(active_sources)),
        changed_segment_count=int(len(changed_targets)),
        method="support-local",
    )


def _angle_in_segment_span(
    ray_angle: float,
    origin: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    margin: float,
) -> bool:
    a0 = np.arctan2(start[1] - origin[1], start[0] - origin[0])
    a1 = np.arctan2(end[1] - origin[1], end[0] - origin[0])
    center = np.arctan2(np.sin(a0) + np.sin(a1), np.cos(a0) + np.cos(a1))
    d0 = abs((a0 - center + np.pi) % (2.0 * np.pi) - np.pi)
    d1 = abs((a1 - center + np.pi) % (2.0 * np.pi) - np.pi)
    half_width = max(d0, d1) + margin
    distance = abs((ray_angle - center + np.pi) % (2.0 * np.pi) - np.pi)
    return bool(distance <= half_width)


def _ray_near_changed_support(
    origin: np.ndarray,
    direction: np.ndarray,
    mesh: SegmentMesh2D,
    changed_targets: np.ndarray,
    tile_margin: float,
) -> bool:
    ray_angle = float(np.arctan2(direction[1], direction[0]))
    for target in changed_targets:
        if _angle_in_segment_span(
            ray_angle,
            origin,
            mesh.starts[int(target)],
            mesh.ends[int(target)],
            tile_margin,
        ):
            return True
    return False


def transport_first_hits_event_tiled(
    old_state: FirstHitMap2D,
    old_mesh: SegmentMesh2D,
    new_mesh: SegmentMesh2D,
    rays_per_source: int,
    neighbor_radius: int = 1,
    tile_margin: float = 0.01,
) -> TransportResult2D:
    """Experimental angular-tile transport.

    This keeps unchanged outside-source rays without segment intersection tests
    unless their direction lies in the angular span of changed support targets.
    It is useful as a prototype comparison against the conservative
    support-local method; mismatch counts against full recomputation should be
    monitored closely.
    """

    if tile_margin < 0.0:
        raise ValueError("tile_margin must be non-negative.")
    if old_mesh.n_segments != new_mesh.n_segments:
        raise ValueError("transport requires matching segment indexing.")
    if old_state.hits.shape != (old_mesh.n_segments, rays_per_source):
        raise ValueError("old_state shape does not match mesh/ray configuration.")

    from .accelerated2d import trace_footprint_tiled_accelerated

    (
        state,
        query_count,
        active_source_count,
        changed_segment_count,
        footprint_ray_count,
        footprint_overlap_ray_count,
    ) = (
        trace_footprint_tiled_accelerated(
            old_state,
            old_mesh,
            new_mesh,
            rays_per_source,
            neighbor_radius,
            tile_margin,
        )
    )
    delta = signed_delta_accumulator(old_state.hits, state.hits, state.weights)
    event_count = int(np.count_nonzero(old_state.hits != state.hits))
    return TransportResult2D(
        state=state,
        delta=delta,
        query_count=query_count,
        event_count=event_count,
        active_source_count=active_source_count,
        changed_segment_count=changed_segment_count,
        method="shadow-footprint",
        footprint_ray_count=footprint_ray_count,
        footprint_overlap_ray_count=footprint_overlap_ray_count,
    )

    changed_mask = expand_segment_mask(old_mesh.support_mask | new_mesh.support_mask, neighbor_radius)
    active_sources = changed_mask.copy()
    changed_targets = np.flatnonzero(changed_mask)
    new_rays = make_lambertian_rays(new_mesh, rays_per_source)

    hits = np.full_like(old_state.hits, -1)
    taus = np.full_like(old_state.taus, np.inf, dtype=float)
    query_count = 0

    for source in range(new_mesh.n_segments):
        source_is_active = bool(active_sources[source])
        for ray_idx in range(rays_per_source):
            origin = new_rays.origins[source, ray_idx]
            direction = new_rays.directions[source, ray_idx]

            if source_is_active:
                candidates = None
            elif _ray_near_changed_support(origin, direction, new_mesh, changed_targets, tile_margin):
                candidates = _candidate_targets(int(old_state.hits[source, ray_idx]), changed_targets)
            else:
                hits[source, ray_idx] = old_state.hits[source, ray_idx]
                taus[source, ray_idx] = old_state.taus[source, ray_idx]
                continue

            hit = trace_ray(
                origin,
                direction,
                new_mesh,
                source_index=source,
                candidates=candidates,
            )
            hits[source, ray_idx] = hit.target
            taus[source, ray_idx] = hit.tau
            query_count += hit.query_count

    state = FirstHitMap2D(hits, taus, new_rays.weights.copy(), query_count)
    delta = signed_delta_accumulator(old_state.hits, state.hits, state.weights)
    event_count = int(np.count_nonzero(old_state.hits != state.hits))

    return TransportResult2D(
        state=state,
        delta=delta,
        query_count=query_count,
        event_count=event_count,
        active_source_count=int(np.count_nonzero(active_sources)),
        changed_segment_count=int(len(changed_targets)),
        method="event-tile",
    )
