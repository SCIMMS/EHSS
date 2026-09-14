from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .geometry2d import SegmentMesh2D
from .rays2d import RayBundle2D


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def intersect_ray_segment(
    origin: np.ndarray,
    direction: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    min_t: float = 1e-9,
    tol: float = 1e-10,
) -> float:
    """Return ray parameter t for the first intersection, or inf."""

    edge = end - start
    denom = cross2(direction, edge)
    if abs(denom) <= tol:
        return float("inf")

    delta = start - origin
    t = cross2(delta, edge) / denom
    u = cross2(delta, direction) / denom
    if t > min_t and -tol <= u <= 1.0 + tol:
        return float(t)
    return float("inf")


@dataclass(frozen=True)
class RayHit:
    target: int
    tau: float
    query_count: int


@dataclass(frozen=True)
class FirstHitMap2D:
    hits: np.ndarray
    taus: np.ndarray
    weights: np.ndarray
    query_count: int


def trace_ray(
    origin: np.ndarray,
    direction: np.ndarray,
    mesh: SegmentMesh2D,
    source_index: int | None = None,
    candidates: Iterable[int] | None = None,
) -> RayHit:
    best_target = -1
    best_tau = float("inf")
    query_count = 0
    if candidates is None:
        candidate_iter = range(mesh.n_segments)
    else:
        candidate_iter = candidates

    for target in candidate_iter:
        target = int(target)
        if source_index is not None and target == source_index:
            continue
        query_count += 1
        tau = intersect_ray_segment(origin, direction, mesh.starts[target], mesh.ends[target])
        if tau < best_tau:
            best_tau = tau
            best_target = target

    return RayHit(best_target, best_tau, query_count)


def trace_first_hits(
    mesh: SegmentMesh2D,
    rays: RayBundle2D,
    use_accelerated: bool = True,
) -> FirstHitMap2D:
    if rays.n_sources != mesh.n_segments:
        raise ValueError("ray bundle source count must match mesh segment count.")

    if use_accelerated:
        from .accelerated2d import trace_first_hits_accelerated

        return trace_first_hits_accelerated(mesh, rays)

    hits = np.full((rays.n_sources, rays.rays_per_source), -1, dtype=int)
    taus = np.full_like(hits, np.inf, dtype=float)
    query_count = 0

    for source in range(rays.n_sources):
        for ray_idx in range(rays.rays_per_source):
            hit = trace_ray(
                rays.origins[source, ray_idx],
                rays.directions[source, ray_idx],
                mesh,
                source_index=source,
            )
            hits[source, ray_idx] = hit.target
            taus[source, ray_idx] = hit.tau
            query_count += hit.query_count

    return FirstHitMap2D(hits, taus, rays.weights.copy(), query_count)
