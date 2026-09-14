from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pvd_demo import PVDGeometry, PVDRayBundle


@dataclass(frozen=True)
class IndependentPVDTrace:
    targets: np.ndarray
    taus: np.ndarray
    incidence: np.ndarray
    query_count: int


def _independent_ray_segment_parameter(
    origin: np.ndarray,
    direction: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    min_t: float = 1e-9,
    tolerance: float = 1e-10,
) -> float:
    """Solve origin + t*direction = start + u*(end-start) as a 2x2 system."""

    edge = end - start
    system = np.column_stack((direction, -edge))
    determinant = float(np.linalg.det(system))
    if abs(determinant) <= tolerance:
        return float("inf")
    try:
        t, u = np.linalg.solve(system, start - origin)
    except np.linalg.LinAlgError:
        return float("inf")
    if float(t) > min_t and -tolerance <= float(u) <= 1.0 + tolerance:
        return float(t)
    return float("inf")


def trace_pvd_first_hits_independent(
    geometry: PVDGeometry,
    rays: PVDRayBundle,
    tie_tolerance: float = 1e-12,
) -> IndependentPVDTrace:
    """Brute-force reference that does not call the production intersection kernel."""

    targets = np.full(rays.n_rays, -1, dtype=np.int64)
    taus = np.full(rays.n_rays, np.inf, dtype=float)
    incidence = np.zeros(rays.n_rays, dtype=float)
    query_count = 0

    for ray_idx in range(rays.n_rays):
        origin = rays.origins[ray_idx]
        direction = rays.directions[ray_idx]
        best_target = -1
        best_tau = float("inf")
        for target, segment in enumerate(geometry.segments):
            query_count += 1
            tau = _independent_ray_segment_parameter(
                origin,
                direction,
                segment.start,
                segment.end,
            )
            if tau < best_tau - tie_tolerance:
                best_tau = tau
                best_target = target
            elif (
                np.isfinite(tau)
                and abs(tau - best_tau) <= tie_tolerance
                and (best_target < 0 or target < best_target)
            ):
                best_tau = tau
                best_target = target

        if best_target >= 0:
            targets[ray_idx] = best_target
            taus[ray_idx] = best_tau
            normal = geometry.segments[best_target].normal
            incidence[ray_idx] = max(0.0, -float(np.dot(direction, normal)))

    return IndependentPVDTrace(
        targets=targets,
        taus=taus,
        incidence=incidence,
        query_count=query_count,
    )
