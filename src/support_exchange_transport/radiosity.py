from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RadiosityCase:
    emission: np.ndarray
    reflectivity: np.ndarray


@dataclass(frozen=True)
class RadiositySolution:
    radiosity: np.ndarray
    irradiation: np.ndarray
    net_flux: np.ndarray
    residual_inf: float


def make_radiosity_case(centroids: np.ndarray, support_labels: np.ndarray | None = None) -> RadiosityCase:
    """Build a deterministic nonuniform radiosity toy case for benchmarks."""

    centroids = np.asarray(centroids, dtype=float)
    if centroids.ndim != 2 or centroids.shape[1] != 3:
        raise ValueError("centroids must have shape (n, 3).")
    n_faces = centroids.shape[0]
    if support_labels is None:
        support_labels = np.zeros(n_faces, dtype=int)
    support_labels = np.asarray(support_labels, dtype=int)
    if support_labels.shape != (n_faces,):
        raise ValueError("support_labels must have one value per centroid.")

    x = centroids[:, 0]
    y = centroids[:, 1]
    z = centroids[:, 2]
    hot_support = (support_labels > 0).astype(float)
    emission = 1.0 + 0.25 * z + 0.12 * np.sin(3.0 * np.arctan2(y, x)) + 0.20 * hot_support
    emission = np.maximum(emission, 0.05)

    reflectivity = 0.42 + 0.12 * x - 0.08 * z + 0.05 * hot_support
    reflectivity = np.clip(reflectivity, 0.15, 0.78)
    return RadiosityCase(emission=emission, reflectivity=reflectivity)


def solve_radiosity(
    view_factors: np.ndarray,
    emission: np.ndarray,
    reflectivity: np.ndarray,
) -> RadiositySolution:
    """Solve J = E + diag(rho) F J for an opaque diffuse toy model."""

    view_factors = np.asarray(view_factors, dtype=float)
    emission = np.asarray(emission, dtype=float)
    reflectivity = np.asarray(reflectivity, dtype=float)
    if view_factors.ndim != 2 or view_factors.shape[0] != view_factors.shape[1]:
        raise ValueError("view_factors must be a square matrix.")
    n_faces = view_factors.shape[0]
    if emission.shape != (n_faces,):
        raise ValueError("emission must have shape (n_faces,).")
    if reflectivity.shape != (n_faces,):
        raise ValueError("reflectivity must have shape (n_faces,).")
    if np.any(reflectivity < 0.0) or np.any(reflectivity >= 1.0):
        raise ValueError("reflectivity values must be in [0, 1).")

    system = np.eye(n_faces) - reflectivity[:, None] * view_factors
    radiosity = np.linalg.solve(system, emission)
    irradiation = view_factors @ radiosity
    net_flux = radiosity - irradiation
    residual_inf = float(np.max(np.abs(system @ radiosity - emission)))
    return RadiositySolution(radiosity=radiosity, irradiation=irradiation, net_flux=net_flux, residual_inf=residual_inf)


def relative_l2_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    denom = float(np.linalg.norm(reference))
    if denom == 0.0:
        return 0.0 if np.linalg.norm(candidate) == 0.0 else float("inf")
    return float(np.linalg.norm(candidate - reference) / denom)


def max_abs_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    return float(np.max(np.abs(candidate - reference)))
