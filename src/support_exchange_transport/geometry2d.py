from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_float_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def angular_distance(theta: np.ndarray | float, center: float) -> np.ndarray:
    """Shortest signed angular distance in radians."""

    return (np.asarray(theta) - center + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class BumpSpec:
    """Radial support deformation on the circular reference boundary.

    Positive amplitude moves the boundary inward along the cavity-facing normal.
    For a circular cavity this creates a local protrusion into the free space,
    which is the 2D analogue used by the first-hit transport prototype.
    """

    center: float
    amplitude: float
    width: float
    label: int = 1
    profile_kind: str = "gaussian"

    def profile(self, theta: np.ndarray) -> np.ndarray:
        if self.width <= 0.0:
            raise ValueError("Bump width must be positive.")
        dist = angular_distance(theta, self.center)
        if self.profile_kind == "gaussian":
            return self.amplitude * np.exp(-0.5 * (dist / self.width) ** 2)
        if self.profile_kind == "compact-cosine":
            abs_dist = np.abs(dist)
            return np.where(
                abs_dist <= self.width,
                self.amplitude * 0.5 * (1.0 + np.cos(np.pi * dist / self.width)),
                0.0,
            )
        raise ValueError(f"unknown bump profile_kind: {self.profile_kind!r}")


@dataclass(frozen=True)
class SegmentMesh2D:
    """Closed 2D segment mesh with inward normals for cavity rays."""

    vertices: np.ndarray
    support_mask: np.ndarray
    support_labels: np.ndarray
    vertex_displacement: np.ndarray

    def __post_init__(self) -> None:
        vertices = _as_float_array(self.vertices)
        if vertices.ndim != 2 or vertices.shape[1] != 2:
            raise ValueError("vertices must have shape (n_vertices, 2).")
        if len(vertices) < 3:
            raise ValueError("a closed segment mesh needs at least three vertices.")

        support_mask = np.asarray(self.support_mask, dtype=bool)
        support_labels = np.asarray(self.support_labels, dtype=int)
        vertex_displacement = np.asarray(self.vertex_displacement, dtype=float)
        n = len(vertices)
        if support_mask.shape != (n,):
            raise ValueError("support_mask must have one value per segment.")
        if support_labels.shape != (n,):
            raise ValueError("support_labels must have one value per segment.")
        if vertex_displacement.shape != (n,):
            raise ValueError("vertex_displacement must have one value per vertex.")

        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "support_mask", support_mask)
        object.__setattr__(self, "support_labels", support_labels)
        object.__setattr__(self, "vertex_displacement", vertex_displacement)

    @property
    def n_segments(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def starts(self) -> np.ndarray:
        return self.vertices

    @property
    def ends(self) -> np.ndarray:
        return np.roll(self.vertices, -1, axis=0)

    @property
    def edges(self) -> np.ndarray:
        return self.ends - self.starts

    @property
    def lengths(self) -> np.ndarray:
        return np.linalg.norm(self.edges, axis=1)

    @property
    def tangents(self) -> np.ndarray:
        lengths = self.lengths
        if np.any(lengths <= 0.0):
            raise ValueError("mesh contains a zero-length segment.")
        return self.edges / lengths[:, None]

    @property
    def normals(self) -> np.ndarray:
        tangents = self.tangents
        # Vertices are generated counter-clockwise, so left normals point into
        # the cavity free space.
        return np.column_stack((-tangents[:, 1], tangents[:, 0]))

    @property
    def centroids(self) -> np.ndarray:
        return 0.5 * (self.starts + self.ends)


def make_polygon_mesh(
    vertices: np.ndarray,
    support_mask: np.ndarray | None = None,
    support_labels: np.ndarray | None = None,
    vertex_displacement: np.ndarray | None = None,
) -> SegmentMesh2D:
    vertices = _as_float_array(vertices)
    n = vertices.shape[0]
    if support_mask is None:
        support_mask = np.zeros(n, dtype=bool)
    if support_labels is None:
        support_labels = np.zeros(n, dtype=int)
    if vertex_displacement is None:
        vertex_displacement = np.zeros(n, dtype=float)
    return SegmentMesh2D(vertices, support_mask, support_labels, vertex_displacement)


def make_circle_cavity(
    n_segments: int = 96,
    radius: float = 1.0,
    bumps: tuple[BumpSpec, ...] | list[BumpSpec] = (),
    alpha: float = 0.0,
    support_threshold: float = 1e-3,
    support_abs: bool = False,
) -> SegmentMesh2D:
    """Build a circular cavity boundary with optional inward support bumps."""

    if n_segments < 8:
        raise ValueError("n_segments must be at least 8 for a usable cavity mesh.")
    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")

    theta = np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=False)
    target_displacement = np.zeros_like(theta)
    per_bump = []
    for bump in bumps:
        profile = bump.profile(theta)
        per_bump.append(profile)
        target_displacement += profile

    displacement = alpha * target_displacement
    if np.any(displacement >= 0.9 * radius):
        raise ValueError("bump displacement is too large for this circular toy mesh.")

    radial = np.column_stack((np.cos(theta), np.sin(theta)))
    vertices = (radius - displacement)[:, None] * radial

    mid_theta = theta + np.pi / n_segments
    mid_profiles = []
    mid_total = np.zeros_like(mid_theta)
    for bump in bumps:
        profile = bump.profile(mid_theta)
        mid_profiles.append(profile)
        mid_total += profile

    if bumps:
        max_amp = max(abs(bump.amplitude) for bump in bumps)
        if all(bump.profile_kind == "compact-cosine" for bump in bumps):
            threshold = 0.0
        else:
            threshold = support_threshold * max(max_amp, 1.0)
        vertex_stack = np.vstack(per_bump)
        mid_stack = np.vstack(mid_profiles)
        if support_abs:
            endpoint_stack = np.maximum(np.abs(vertex_stack), np.abs(np.roll(vertex_stack, -1, axis=1)))
            score_stack = np.maximum(np.abs(mid_stack), endpoint_stack)
        else:
            endpoint_stack = np.maximum(vertex_stack, np.roll(vertex_stack, -1, axis=1))
            score_stack = np.maximum(mid_stack, endpoint_stack)
        support_score = np.max(score_stack, axis=0)
        support_mask = support_score > threshold
        best = np.argmax(score_stack, axis=0)
        labels = np.zeros(n_segments, dtype=int)
        bump_labels = np.array([bump.label for bump in bumps], dtype=int)
        labels[support_mask] = bump_labels[best[support_mask]]
    else:
        support_mask = np.zeros(n_segments, dtype=bool)
        labels = np.zeros(n_segments, dtype=int)

    return SegmentMesh2D(vertices, support_mask, labels, displacement)


def expand_segment_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Expand a cyclic segment mask by a small integer neighborhood."""

    mask = np.asarray(mask, dtype=bool)
    if radius < 0:
        raise ValueError("radius must be non-negative.")
    expanded = mask.copy()
    for shift in range(1, radius + 1):
        expanded |= np.roll(mask, shift)
        expanded |= np.roll(mask, -shift)
    return expanded
