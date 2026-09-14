from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def normalize_vectors(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    norms = np.linalg.norm(values, axis=-1)
    if np.any(norms <= 0.0):
        raise ValueError("cannot normalize zero-length vectors.")
    return values / norms[..., None]


@dataclass(frozen=True)
class SphericalCapBump:
    """Compact radial deformation on a spherical cavity boundary.

    Positive amplitude moves the boundary inward, creating a local protrusion
    into the cavity free space.
    """

    center: np.ndarray
    amplitude: float
    width: float
    label: int = 1

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float)
        if center.shape != (3,):
            raise ValueError("center must have shape (3,).")
        if self.width <= 0.0:
            raise ValueError("width must be positive.")
        object.__setattr__(self, "center", normalize_vectors(center[None, :])[0])

    def profile(self, directions: np.ndarray) -> np.ndarray:
        directions = normalize_vectors(np.asarray(directions, dtype=float))
        cos_angle = np.clip(directions @ self.center, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        return np.where(
            angle <= self.width,
            self.amplitude * 0.5 * (1.0 + np.cos(np.pi * angle / self.width)),
            0.0,
        )


@dataclass(frozen=True)
class TriangleMesh3D:
    vertices: np.ndarray
    faces: np.ndarray
    support_mask: np.ndarray
    support_labels: np.ndarray
    vertex_displacement: np.ndarray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=float)
        faces = np.asarray(self.faces, dtype=np.int64)
        support_mask = np.asarray(self.support_mask, dtype=bool)
        support_labels = np.asarray(self.support_labels, dtype=np.int64)
        vertex_displacement = np.asarray(self.vertex_displacement, dtype=float)

        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (n_vertices, 3).")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape (n_faces, 3).")
        if np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ValueError("faces contain out-of-range vertex indices.")
        if support_mask.shape != (len(faces),):
            raise ValueError("support_mask must have one value per face.")
        if support_labels.shape != (len(faces),):
            raise ValueError("support_labels must have one value per face.")
        if vertex_displacement.shape != (len(vertices),):
            raise ValueError("vertex_displacement must have one value per vertex.")

        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "support_mask", support_mask)
        object.__setattr__(self, "support_labels", support_labels)
        object.__setattr__(self, "vertex_displacement", vertex_displacement)

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    @property
    def triangles(self) -> np.ndarray:
        return self.vertices[self.faces]

    @property
    def centroids(self) -> np.ndarray:
        return np.mean(self.triangles, axis=1)

    @property
    def inward_normals(self) -> np.ndarray:
        tris = self.triangles
        normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        normals = normalize_vectors(normals)
        centroids = self.centroids
        outward = np.einsum("ij,ij->i", normals, centroids) > 0.0
        normals[outward] *= -1.0
        return normals

    @property
    def areas(self) -> np.ndarray:
        tris = self.triangles
        return 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)


def _uv_sphere_directions(n_lat: int, n_lon: int) -> np.ndarray:
    directions = [[0.0, 0.0, 1.0]]
    for lat in range(1, n_lat):
        polar = np.pi * lat / n_lat
        sin_polar = np.sin(polar)
        cos_polar = np.cos(polar)
        for lon in range(n_lon):
            azimuth = 2.0 * np.pi * lon / n_lon
            directions.append([sin_polar * np.cos(azimuth), sin_polar * np.sin(azimuth), cos_polar])
    directions.append([0.0, 0.0, -1.0])
    return np.asarray(directions, dtype=float)


def _uv_sphere_faces(n_lat: int, n_lon: int) -> np.ndarray:
    north = 0
    south = 1 + (n_lat - 1) * n_lon

    def ring_index(lat: int, lon: int) -> int:
        return 1 + (lat - 1) * n_lon + (lon % n_lon)

    faces: list[list[int]] = []
    for lon in range(n_lon):
        faces.append([north, ring_index(1, lon + 1), ring_index(1, lon)])

    for lat in range(1, n_lat - 1):
        for lon in range(n_lon):
            a = ring_index(lat, lon)
            b = ring_index(lat, lon + 1)
            c = ring_index(lat + 1, lon)
            d = ring_index(lat + 1, lon + 1)
            faces.append([a, b, c])
            faces.append([b, d, c])

    for lon in range(n_lon):
        faces.append([ring_index(n_lat - 1, lon), ring_index(n_lat - 1, lon + 1), south])

    return np.asarray(faces, dtype=np.int64)


def make_sphere_cavity(
    n_lat: int = 12,
    n_lon: int = 24,
    radius: float = 1.0,
    bumps: tuple[SphericalCapBump, ...] | list[SphericalCapBump] = (),
    alpha: float = 0.0,
) -> TriangleMesh3D:
    if n_lat < 4:
        raise ValueError("n_lat must be at least 4.")
    if n_lon < 8:
        raise ValueError("n_lon must be at least 8.")
    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")

    directions = _uv_sphere_directions(n_lat, n_lon)
    faces = _uv_sphere_faces(n_lat, n_lon)

    target_displacement = np.zeros(len(directions), dtype=float)
    vertex_profiles = []
    for bump in bumps:
        profile = bump.profile(directions)
        vertex_profiles.append(profile)
        target_displacement += profile

    displacement = alpha * target_displacement
    if np.any(displacement >= 0.9 * radius):
        raise ValueError("bump displacement is too large for this spherical toy mesh.")

    vertices = (radius - displacement)[:, None] * directions

    if bumps:
        face_dirs = normalize_vectors(np.mean(directions[faces], axis=1))
        face_profiles = np.vstack([bump.profile(face_dirs) for bump in bumps])
        vertex_stack = np.vstack(vertex_profiles)
        face_vertex_scores = np.max(vertex_stack[:, faces], axis=2)
        score_stack = np.maximum(face_profiles, face_vertex_scores)
        support_score = np.max(score_stack, axis=0)
        support_mask = support_score > 0.0
        best = np.argmax(score_stack, axis=0)
        labels = np.zeros(len(faces), dtype=np.int64)
        bump_labels = np.array([bump.label for bump in bumps], dtype=np.int64)
        labels[support_mask] = bump_labels[best[support_mask]]
    else:
        support_mask = np.zeros(len(faces), dtype=bool)
        labels = np.zeros(len(faces), dtype=np.int64)

    return TriangleMesh3D(vertices, faces, support_mask, labels, displacement)


def make_default_cap_bump(
    amplitude: float = 0.12,
    width: float = 0.35,
    center: tuple[float, float, float] = (1.0, 0.0, 0.0),
    label: int = 1,
) -> SphericalCapBump:
    return SphericalCapBump(center=np.asarray(center, dtype=float), amplitude=amplitude, width=width, label=label)
