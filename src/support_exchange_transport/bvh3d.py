from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry3d import TriangleMesh3D


@dataclass(frozen=True)
class RayTriangleHit3D:
    target: int
    tau: float
    triangle_tests: int
    bbox_tests: int = 0


def intersect_ray_triangle(
    origin: np.ndarray,
    direction: np.ndarray,
    triangle: np.ndarray,
    min_t: float = 1e-9,
    tol: float = 1e-10,
) -> float:
    v0, v1, v2 = triangle
    edge1 = v1 - v0
    edge2 = v2 - v0
    pvec = np.cross(direction, edge2)
    det = float(np.dot(edge1, pvec))
    if abs(det) <= tol:
        return float("inf")

    inv_det = 1.0 / det
    tvec = origin - v0
    u = float(np.dot(tvec, pvec) * inv_det)
    if u < -tol or u > 1.0 + tol:
        return float("inf")

    qvec = np.cross(tvec, edge1)
    v = float(np.dot(direction, qvec) * inv_det)
    if v < -tol or u + v > 1.0 + tol:
        return float("inf")

    tau = float(np.dot(edge2, qvec) * inv_det)
    if tau > min_t:
        return tau
    return float("inf")


def _intersect_ray_aabb(
    origin: np.ndarray,
    direction: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    max_t: float,
    tol: float = 1e-12,
) -> bool:
    tmin = 0.0
    tmax = max_t
    for axis in range(3):
        d = float(direction[axis])
        if abs(d) <= tol:
            if origin[axis] < bbox_min[axis] or origin[axis] > bbox_max[axis]:
                return False
            continue
        inv_d = 1.0 / d
        t0 = (bbox_min[axis] - origin[axis]) * inv_d
        t1 = (bbox_max[axis] - origin[axis]) * inv_d
        if t0 > t1:
            t0, t1 = t1, t0
        tmin = max(tmin, t0)
        tmax = min(tmax, t1)
        if tmax < tmin:
            return False
    return tmax >= 0.0


class BVH3D:
    def __init__(
        self,
        mesh: TriangleMesh3D,
        face_indices: np.ndarray | None = None,
        leaf_size: int = 8,
    ) -> None:
        if leaf_size <= 0:
            raise ValueError("leaf_size must be positive.")
        self.mesh = mesh
        self.leaf_size = int(leaf_size)
        if face_indices is None:
            face_indices = np.arange(mesh.n_faces, dtype=np.int64)
        face_indices = np.asarray(face_indices, dtype=np.int64)
        if face_indices.ndim != 1:
            raise ValueError("face_indices must be one-dimensional.")
        if len(face_indices) == 0:
            raise ValueError("BVH needs at least one face.")
        if np.any(face_indices < 0) or np.any(face_indices >= mesh.n_faces):
            raise ValueError("face_indices contain out-of-range values.")

        self._triangles = mesh.triangles
        self._centroids = mesh.centroids
        self._ordered_faces: list[int] = []
        self._bbox_min: list[np.ndarray] = []
        self._bbox_max: list[np.ndarray] = []
        self._left: list[int] = []
        self._right: list[int] = []
        self._start: list[int] = []
        self._count: list[int] = []
        self._build_node(face_indices.copy())

        self.face_indices = np.asarray(self._ordered_faces, dtype=np.int64)
        self.bbox_min = np.asarray(self._bbox_min, dtype=float)
        self.bbox_max = np.asarray(self._bbox_max, dtype=float)
        self.left = np.asarray(self._left, dtype=np.int64)
        self.right = np.asarray(self._right, dtype=np.int64)
        self.start = np.asarray(self._start, dtype=np.int64)
        self.count = np.asarray(self._count, dtype=np.int64)

    @property
    def n_nodes(self) -> int:
        return int(len(self.left))

    def refit(self, mesh: TriangleMesh3D) -> None:
        """Update node bounds for a mesh with unchanged face indexing."""
        if mesh.n_faces != self.mesh.n_faces:
            raise ValueError("BVH refit requires an unchanged face count.")
        if np.any(self.face_indices < 0) or np.any(
            self.face_indices >= mesh.n_faces
        ):
            raise ValueError("BVH face indices are incompatible with the new mesh.")

        self.mesh = mesh
        self._triangles = mesh.triangles
        self._centroids = mesh.centroids
        for node in range(self.n_nodes - 1, -1, -1):
            if self.count[node] > 0:
                start = int(self.start[node])
                stop = start + int(self.count[node])
                faces = self.face_indices[start:stop]
                triangles = self._triangles[faces]
                self.bbox_min[node] = np.min(
                    triangles.reshape(-1, 3),
                    axis=0,
                )
                self.bbox_max[node] = np.max(
                    triangles.reshape(-1, 3),
                    axis=0,
                )
                continue

            left = int(self.left[node])
            right = int(self.right[node])
            child_indices = np.asarray(
                [child for child in (left, right) if child >= 0],
                dtype=np.int64,
            )
            if len(child_indices) == 0:
                raise RuntimeError("internal BVH node has no children.")
            self.bbox_min[node] = np.min(
                self.bbox_min[child_indices],
                axis=0,
            )
            self.bbox_max[node] = np.max(
                self.bbox_max[child_indices],
                axis=0,
            )

    def _build_node(self, face_indices: np.ndarray) -> int:
        node_index = len(self._left)
        tris = self._triangles[face_indices]
        bbox_min = np.min(tris.reshape(-1, 3), axis=0)
        bbox_max = np.max(tris.reshape(-1, 3), axis=0)

        self._bbox_min.append(bbox_min)
        self._bbox_max.append(bbox_max)
        self._left.append(-1)
        self._right.append(-1)
        self._start.append(-1)
        self._count.append(0)

        if len(face_indices) <= self.leaf_size:
            start = len(self._ordered_faces)
            self._ordered_faces.extend(int(idx) for idx in face_indices)
            self._start[node_index] = start
            self._count[node_index] = int(len(face_indices))
            return node_index

        extent = bbox_max - bbox_min
        axis = int(np.argmax(extent))
        order = np.argsort(self._centroids[face_indices, axis])
        sorted_indices = face_indices[order]
        split = len(sorted_indices) // 2
        left = self._build_node(sorted_indices[:split])
        right = self._build_node(sorted_indices[split:])
        self._left[node_index] = left
        self._right[node_index] = right
        return node_index

    def intersect_ray(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        skip_face: int = -1,
        max_t: float = float("inf"),
    ) -> RayTriangleHit3D:
        best_target = -1
        best_tau = float(max_t)
        triangle_tests = 0
        bbox_tests = 0
        stack = [0]

        while stack:
            node = stack.pop()
            bbox_tests += 1
            if not _intersect_ray_aabb(origin, direction, self.bbox_min[node], self.bbox_max[node], best_tau):
                continue

            if self.count[node] > 0:
                start = int(self.start[node])
                stop = start + int(self.count[node])
                for face in self.face_indices[start:stop]:
                    face = int(face)
                    if face == skip_face:
                        continue
                    triangle_tests += 1
                    tau = intersect_ray_triangle(origin, direction, self._triangles[face])
                    if tau < best_tau:
                        best_tau = tau
                        best_target = face
                continue

            left = int(self.left[node])
            right = int(self.right[node])
            if left >= 0:
                stack.append(left)
            if right >= 0:
                stack.append(right)

        if best_target < 0:
            return RayTriangleHit3D(-1, float("inf"), triangle_tests, bbox_tests)
        return RayTriangleHit3D(best_target, best_tau, triangle_tests, bbox_tests)


def trace_direct_candidates(
    mesh: TriangleMesh3D,
    origin: np.ndarray,
    direction: np.ndarray,
    candidates: np.ndarray,
    skip_face: int = -1,
) -> RayTriangleHit3D:
    best_target = -1
    best_tau = float("inf")
    triangle_tests = 0
    triangles = mesh.triangles
    for face in np.asarray(candidates, dtype=np.int64):
        face = int(face)
        if face == skip_face:
            continue
        triangle_tests += 1
        tau = intersect_ray_triangle(origin, direction, triangles[face])
        if tau < best_tau:
            best_tau = tau
            best_target = face
    if best_target < 0:
        return RayTriangleHit3D(-1, float("inf"), triangle_tests, 0)
    return RayTriangleHit3D(best_target, best_tau, triangle_tests, 0)
