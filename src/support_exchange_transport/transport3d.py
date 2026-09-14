from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bvh3d import BVH3D, RayTriangleHit3D, intersect_ray_triangle
from .geometry3d import TriangleMesh3D
from .metrics import signed_delta_accumulator
from .rays3d import make_cosine_hemisphere_rays
from .raytrace3d import FirstHitMap3D


@dataclass(frozen=True)
class TransportResult3D:
    state: FirstHitMap3D
    delta: dict[tuple[int, int], float]
    triangle_tests: int
    bbox_tests: int
    event_count: int
    active_source_count: int
    changed_face_count: int
    footprint_ray_count: int
    footprint_overlap_ray_count: int
    support_bvh_ray_count: int
    fallback_full_ray_count: int
    method: str = "cone-footprint-support-bvh"

    @property
    def query_count(self) -> int:
        return int(self.triangle_tests)


def changed_support_bounding_sphere(
    old_mesh: TriangleMesh3D,
    new_mesh: TriangleMesh3D,
    changed_faces: np.ndarray,
) -> tuple[np.ndarray, float]:
    changed_faces = np.asarray(changed_faces, dtype=np.int64)
    if len(changed_faces) == 0:
        raise ValueError("changed_faces must be non-empty.")
    old_vertices = np.unique(old_mesh.faces[changed_faces].ravel())
    new_vertices = np.unique(new_mesh.faces[changed_faces].ravel())
    points = np.vstack((old_mesh.vertices[old_vertices], new_mesh.vertices[new_vertices]))
    center = np.mean(points, axis=0)
    radius = float(np.max(np.linalg.norm(points - center[None, :], axis=1)))
    return center, radius + 1e-12


def changed_support_bounding_spheres_by_label(
    old_mesh: TriangleMesh3D,
    new_mesh: TriangleMesh3D,
    changed_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    changed_faces = np.asarray(changed_faces, dtype=np.int64)
    labels = np.union1d(
        old_mesh.support_labels[changed_faces],
        new_mesh.support_labels[changed_faces],
    )
    labels = labels[labels > 0]
    if len(labels) == 0:
        center, radius = changed_support_bounding_sphere(old_mesh, new_mesh, changed_faces)
        return np.array([0], dtype=np.int64), center[None, :], np.array([radius], dtype=float)

    centers = np.zeros((len(labels), 3), dtype=float)
    radii = np.zeros(len(labels), dtype=float)
    for out_idx, label in enumerate(labels):
        label_faces = changed_faces[
            (old_mesh.support_labels[changed_faces] == label) | (new_mesh.support_labels[changed_faces] == label)
        ]
        old_vertices = np.unique(old_mesh.faces[label_faces].ravel())
        new_vertices = np.unique(new_mesh.faces[label_faces].ravel())
        points = np.vstack((old_mesh.vertices[old_vertices], new_mesh.vertices[new_vertices]))
        center = np.mean(points, axis=0)
        radius = float(np.max(np.linalg.norm(points - center[None, :], axis=1)))
        centers[out_idx] = center
        radii[out_idx] = radius + 1e-12
    return labels.astype(np.int64), centers, radii


def ray_in_conservative_cone(
    origin: np.ndarray,
    direction: np.ndarray,
    center: np.ndarray,
    radius: float,
    padding: float = 0.0,
) -> bool:
    if padding < 0.0:
        raise ValueError("padding must be non-negative.")
    axis = center - origin
    distance = float(np.linalg.norm(axis))
    if distance <= radius:
        return True
    axis /= distance
    half_angle = np.arcsin(min(1.0, radius / distance)) + padding
    if half_angle >= np.pi:
        return True
    return bool(float(np.dot(direction, axis)) >= np.cos(half_angle))


def _intersect_single_face(
    mesh: TriangleMesh3D,
    face: int,
    origin: np.ndarray,
    direction: np.ndarray,
    skip_face: int,
) -> RayTriangleHit3D:
    if face < 0 or face == skip_face:
        return RayTriangleHit3D(-1, float("inf"), 0, 0)
    tau = intersect_ray_triangle(origin, direction, mesh.triangles[int(face)])
    if np.isfinite(tau):
        return RayTriangleHit3D(int(face), tau, 1, 0)
    return RayTriangleHit3D(-1, float("inf"), 1, 0)


def _choose_nearest(*hits: RayTriangleHit3D) -> RayTriangleHit3D:
    best = RayTriangleHit3D(-1, float("inf"), 0, 0)
    triangle_tests = 0
    bbox_tests = 0
    for hit in hits:
        triangle_tests += hit.triangle_tests
        bbox_tests += hit.bbox_tests
        if hit.tau < best.tau:
            best = hit
    if best.target < 0:
        return RayTriangleHit3D(-1, float("inf"), triangle_tests, bbox_tests)
    return RayTriangleHit3D(best.target, best.tau, triangle_tests, bbox_tests)


def transport_first_hits_cone_bvh(
    old_state: FirstHitMap3D,
    old_mesh: TriangleMesh3D,
    new_mesh: TriangleMesh3D,
    rays_per_source: int,
    cone_padding: float = 1e-6,
    leaf_size: int = 8,
    use_accelerated: bool = True,
) -> TransportResult3D:
    if old_mesh.n_faces != new_mesh.n_faces:
        raise ValueError("transport requires matching face indexing.")
    if old_state.hits.shape != (old_mesh.n_faces, rays_per_source):
        raise ValueError("old_state shape does not match mesh/ray configuration.")

    changed_mask = old_mesh.support_mask | new_mesh.support_mask
    changed_faces = np.flatnonzero(changed_mask)
    if len(changed_faces) == 0:
        raise ValueError("new_mesh/old_mesh contain no support faces.")

    active_sources = changed_mask.copy()
    _, support_centers, support_radii = changed_support_bounding_spheres_by_label(old_mesh, new_mesh, changed_faces)
    full_bvh = BVH3D(new_mesh, leaf_size=leaf_size)
    support_bvh = BVH3D(new_mesh, changed_faces, leaf_size=leaf_size)
    new_rays = make_cosine_hemisphere_rays(new_mesh, rays_per_source)

    if use_accelerated:
        from .accelerated3d import transport_cone_bvh_accelerated

        hits, taus, counts = transport_cone_bvh_accelerated(
            old_state.hits,
            old_state.taus,
            new_rays.origins,
            new_rays.directions,
            support_centers,
            support_radii,
            cone_padding,
            changed_mask,
            full_bvh,
            support_bvh,
        )
        state = FirstHitMap3D(
            hits,
            taus,
            new_rays.weights.copy(),
            int(counts[0]),
            int(counts[1]),
        )
        delta = signed_delta_accumulator(old_state.hits, state.hits, state.weights)
        event_count = int(np.count_nonzero(old_state.hits != state.hits))
        return TransportResult3D(
            state=state,
            delta=delta,
            triangle_tests=int(counts[0]),
            bbox_tests=int(counts[1]),
            event_count=event_count,
            active_source_count=int(np.count_nonzero(active_sources)),
            changed_face_count=int(len(changed_faces)),
            footprint_ray_count=int(counts[2]),
            footprint_overlap_ray_count=int(counts[3]),
            support_bvh_ray_count=int(counts[4]),
            fallback_full_ray_count=int(counts[5]),
        )

    hits = old_state.hits.copy()
    taus = old_state.taus.copy()
    triangle_tests = 0
    bbox_tests = 0
    footprint_ray_count = 0
    footprint_overlap_ray_count = 0
    support_bvh_ray_count = 0
    fallback_full_ray_count = 0

    for source in range(new_mesh.n_faces):
        source_is_active = bool(active_sources[source])
        for ray_idx in range(rays_per_source):
            origin = new_rays.origins[source, ray_idx]
            direction = new_rays.directions[source, ray_idx]

            if source_is_active:
                footprint_ray_count += 1
                hit = full_bvh.intersect_ray(origin, direction, skip_face=source)
                hits[source, ray_idx] = hit.target
                taus[source, ray_idx] = hit.tau
                triangle_tests += hit.triangle_tests
                bbox_tests += hit.bbox_tests
                continue

            cone_hits = 0
            for support_center, support_radius in zip(support_centers, support_radii, strict=True):
                if ray_in_conservative_cone(origin, direction, support_center, float(support_radius), cone_padding):
                    cone_hits += 1
            if cone_hits == 0:
                continue

            footprint_ray_count += 1
            if cone_hits > 1:
                footprint_overlap_ray_count += 1
            old_target = int(old_state.hits[source, ray_idx])
            if old_target < 0 or changed_mask[old_target]:
                fallback_full_ray_count += 1
                hit = full_bvh.intersect_ray(origin, direction, skip_face=source)
                hits[source, ray_idx] = hit.target
                taus[source, ray_idx] = hit.tau
                triangle_tests += hit.triangle_tests
                bbox_tests += hit.bbox_tests
                continue

            support_bvh_ray_count += 1
            support_hit = support_bvh.intersect_ray(origin, direction, skip_face=source)
            old_target_hit = _intersect_single_face(new_mesh, old_target, origin, direction, skip_face=source)
            hit = _choose_nearest(support_hit, old_target_hit)
            if hit.target < 0:
                fallback_full_ray_count += 1
                hit = full_bvh.intersect_ray(origin, direction, skip_face=source)
            hits[source, ray_idx] = hit.target
            taus[source, ray_idx] = hit.tau
            triangle_tests += hit.triangle_tests
            bbox_tests += hit.bbox_tests

    state = FirstHitMap3D(hits, taus, new_rays.weights.copy(), triangle_tests, bbox_tests)
    delta = signed_delta_accumulator(old_state.hits, state.hits, state.weights)
    event_count = int(np.count_nonzero(old_state.hits != state.hits))
    return TransportResult3D(
        state=state,
        delta=delta,
        triangle_tests=triangle_tests,
        bbox_tests=bbox_tests,
        event_count=event_count,
        active_source_count=int(np.count_nonzero(active_sources)),
        changed_face_count=int(len(changed_faces)),
        footprint_ray_count=footprint_ray_count,
        footprint_overlap_ray_count=footprint_overlap_ray_count,
        support_bvh_ray_count=support_bvh_ray_count,
        fallback_full_ray_count=fallback_full_ray_count,
    )
