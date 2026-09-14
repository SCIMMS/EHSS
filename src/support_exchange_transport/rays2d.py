from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry2d import SegmentMesh2D


@dataclass(frozen=True)
class RayBundle2D:
    origins: np.ndarray
    directions: np.ndarray
    weights: np.ndarray
    local_angles: np.ndarray

    def __post_init__(self) -> None:
        if self.origins.shape != self.directions.shape:
            raise ValueError("origins and directions must have matching shape.")
        if self.origins.ndim != 3 or self.origins.shape[2] != 2:
            raise ValueError("origins and directions must have shape (sources, rays, 2).")
        if self.weights.shape != self.origins.shape[:2]:
            raise ValueError("weights must have shape (sources, rays).")
        if self.local_angles.shape != (self.origins.shape[1],):
            raise ValueError("local_angles must have one value per ray.")

    @property
    def n_sources(self) -> int:
        return int(self.origins.shape[0])

    @property
    def rays_per_source(self) -> int:
        return int(self.origins.shape[1])


def make_lambertian_rays(
    mesh: SegmentMesh2D,
    rays_per_source: int = 48,
    start_offset: float = 1e-8,
    grazing_epsilon: float = 1e-3,
) -> RayBundle2D:
    """Build deterministic cosine-weighted rays over each inward hemisphere."""

    if rays_per_source <= 0:
        raise ValueError("rays_per_source must be positive.")
    if start_offset <= 0.0:
        raise ValueError("start_offset must be positive.")
    if not 0.0 <= grazing_epsilon < 0.25 * np.pi:
        raise ValueError("grazing_epsilon must be in [0, pi/4).")

    span = np.pi - 2.0 * grazing_epsilon
    local_angles = -0.5 * np.pi + grazing_epsilon
    local_angles = local_angles + (np.arange(rays_per_source) + 0.5) * span / rays_per_source

    cos_phi = np.cos(local_angles)
    sin_phi = np.sin(local_angles)
    base_weights = cos_phi / np.sum(cos_phi)

    origin_base = mesh.centroids[:, None, :] + start_offset * mesh.normals[:, None, :]
    directions = (
        cos_phi[None, :, None] * mesh.normals[:, None, :]
        + sin_phi[None, :, None] * mesh.tangents[:, None, :]
    )
    directions = directions / np.linalg.norm(directions, axis=2)[:, :, None]
    origins = np.broadcast_to(origin_base, directions.shape).copy()
    weights = np.broadcast_to(base_weights[None, :], (mesh.n_segments, rays_per_source)).copy()

    return RayBundle2D(origins, directions, weights, local_angles)
