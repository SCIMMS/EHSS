from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry3d import TriangleMesh3D, normalize_vectors


@dataclass(frozen=True)
class RayBundle3D:
    origins: np.ndarray
    directions: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        origins = np.asarray(self.origins, dtype=float)
        directions = normalize_vectors(np.asarray(self.directions, dtype=float))
        weights = np.asarray(self.weights, dtype=float)
        if origins.ndim != 3 or origins.shape[2] != 3:
            raise ValueError("origins must have shape (n_sources, n_rays, 3).")
        if directions.shape != origins.shape:
            raise ValueError("directions must match origins shape.")
        if weights.shape != origins.shape[:2]:
            raise ValueError("weights must have shape (n_sources, n_rays).")
        object.__setattr__(self, "origins", origins)
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "weights", weights)

    @property
    def n_sources(self) -> int:
        return int(self.origins.shape[0])

    @property
    def rays_per_source(self) -> int:
        return int(self.origins.shape[1])


def _basis_from_normal(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(normal, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    tangent = np.cross(helper, normal)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    bitangent /= np.linalg.norm(bitangent)
    return tangent, bitangent


def _cosine_hemisphere_samples(rays_per_source: int) -> np.ndarray:
    if rays_per_source <= 0:
        raise ValueError("rays_per_source must be positive.")

    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    samples = np.zeros((rays_per_source, 3), dtype=float)
    for ray_idx in range(rays_per_source):
        u = (ray_idx + 0.5) / rays_per_source
        v = (ray_idx / golden_ratio) % 1.0
        radius = np.sqrt(u)
        phi = 2.0 * np.pi * v
        samples[ray_idx] = [
            radius * np.cos(phi),
            radius * np.sin(phi),
            np.sqrt(max(0.0, 1.0 - u)),
        ]
    return samples


def make_cosine_hemisphere_rays(
    mesh: TriangleMesh3D,
    rays_per_source: int,
    origin_epsilon: float = 1e-6,
) -> RayBundle3D:
    if origin_epsilon <= 0.0:
        raise ValueError("origin_epsilon must be positive.")

    local_samples = _cosine_hemisphere_samples(rays_per_source)
    origins = np.zeros((mesh.n_faces, rays_per_source, 3), dtype=float)
    directions = np.zeros_like(origins)
    weights = np.full((mesh.n_faces, rays_per_source), 1.0 / rays_per_source, dtype=float)

    centroids = mesh.centroids
    inward_normals = mesh.inward_normals
    for source in range(mesh.n_faces):
        normal = inward_normals[source]
        tangent, bitangent = _basis_from_normal(normal)
        origins[source, :, :] = centroids[source] + origin_epsilon * normal
        directions[source, :, :] = (
            local_samples[:, 0:1] * tangent[None, :]
            + local_samples[:, 1:2] * bitangent[None, :]
            + local_samples[:, 2:3] * normal[None, :]
        )

    return RayBundle3D(origins, directions, weights)
