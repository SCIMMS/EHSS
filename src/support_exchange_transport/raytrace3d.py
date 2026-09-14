from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bvh3d import BVH3D
from .geometry3d import TriangleMesh3D
from .rays3d import RayBundle3D


@dataclass(frozen=True)
class FirstHitMap3D:
    hits: np.ndarray
    taus: np.ndarray
    weights: np.ndarray
    triangle_tests: int
    bbox_tests: int

    @property
    def query_count(self) -> int:
        return int(self.triangle_tests)


def trace_first_hits_bvh(
    mesh: TriangleMesh3D,
    rays: RayBundle3D,
    bvh: BVH3D | None = None,
    use_accelerated: bool = True,
) -> FirstHitMap3D:
    if rays.n_sources != mesh.n_faces:
        raise ValueError("ray bundle source count must match mesh face count.")
    if bvh is None:
        bvh = BVH3D(mesh)
    if use_accelerated:
        from .accelerated3d import trace_first_hits_bvh_accelerated

        return trace_first_hits_bvh_accelerated(bvh, rays.origins, rays.directions, rays.weights)

    hits = np.full((rays.n_sources, rays.rays_per_source), -1, dtype=np.int64)
    taus = np.full_like(hits, np.inf, dtype=float)
    triangle_tests = 0
    bbox_tests = 0

    for source in range(rays.n_sources):
        for ray_idx in range(rays.rays_per_source):
            hit = bvh.intersect_ray(
                rays.origins[source, ray_idx],
                rays.directions[source, ray_idx],
                skip_face=source,
            )
            hits[source, ray_idx] = hit.target
            taus[source, ray_idx] = hit.tau
            triangle_tests += hit.triangle_tests
            bbox_tests += hit.bbox_tests

    return FirstHitMap3D(hits, taus, rays.weights.copy(), triangle_tests, bbox_tests)
