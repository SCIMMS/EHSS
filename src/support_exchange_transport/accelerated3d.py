from __future__ import annotations

import numpy as np
from numba import njit

from .bvh3d import BVH3D
from .raytrace3d import FirstHitMap3D


@njit(cache=True)
def _ray_triangle_tau(
    origin: np.ndarray,
    direction: np.ndarray,
    triangle: np.ndarray,
    min_t: float = 1e-9,
    tol: float = 1e-10,
) -> float:
    v0 = triangle[0]
    v1 = triangle[1]
    v2 = triangle[2]
    edge1 = v1 - v0
    edge2 = v2 - v0

    pvec = np.empty(3, dtype=np.float64)
    pvec[0] = direction[1] * edge2[2] - direction[2] * edge2[1]
    pvec[1] = direction[2] * edge2[0] - direction[0] * edge2[2]
    pvec[2] = direction[0] * edge2[1] - direction[1] * edge2[0]
    det = edge1[0] * pvec[0] + edge1[1] * pvec[1] + edge1[2] * pvec[2]
    if abs(det) <= tol:
        return np.inf

    inv_det = 1.0 / det
    tvec = origin - v0
    u = (tvec[0] * pvec[0] + tvec[1] * pvec[1] + tvec[2] * pvec[2]) * inv_det
    if u < -tol or u > 1.0 + tol:
        return np.inf

    qvec = np.empty(3, dtype=np.float64)
    qvec[0] = tvec[1] * edge1[2] - tvec[2] * edge1[1]
    qvec[1] = tvec[2] * edge1[0] - tvec[0] * edge1[2]
    qvec[2] = tvec[0] * edge1[1] - tvec[1] * edge1[0]
    v = (direction[0] * qvec[0] + direction[1] * qvec[1] + direction[2] * qvec[2]) * inv_det
    if v < -tol or u + v > 1.0 + tol:
        return np.inf

    tau = (edge2[0] * qvec[0] + edge2[1] * qvec[1] + edge2[2] * qvec[2]) * inv_det
    if tau > min_t:
        return tau
    return np.inf


@njit(cache=True)
def _ray_aabb_hit(
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
        d = direction[axis]
        if abs(d) <= tol:
            if origin[axis] < bbox_min[axis] or origin[axis] > bbox_max[axis]:
                return False
            continue
        inv_d = 1.0 / d
        t0 = (bbox_min[axis] - origin[axis]) * inv_d
        t1 = (bbox_max[axis] - origin[axis]) * inv_d
        if t0 > t1:
            tmp = t0
            t0 = t1
            t1 = tmp
        if t0 > tmin:
            tmin = t0
        if t1 < tmax:
            tmax = t1
        if tmax < tmin:
            return False
    return tmax >= 0.0


@njit(cache=True)
def _intersect_bvh(
    origin: np.ndarray,
    direction: np.ndarray,
    skip_face: int,
    triangles: np.ndarray,
    face_indices: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    start: np.ndarray,
    count: np.ndarray,
) -> tuple[int, float, int, int]:
    best_target = -1
    best_tau = np.inf
    triangle_tests = 0
    bbox_tests = 0
    stack = np.empty(len(left), dtype=np.int64)
    stack_size = 1
    stack[0] = 0

    while stack_size > 0:
        stack_size -= 1
        node = stack[stack_size]
        bbox_tests += 1
        if not _ray_aabb_hit(origin, direction, bbox_min[node], bbox_max[node], best_tau):
            continue

        node_count = count[node]
        if node_count > 0:
            node_start = start[node]
            for local in range(node_count):
                face = face_indices[node_start + local]
                if face == skip_face:
                    continue
                triangle_tests += 1
                tau = _ray_triangle_tau(origin, direction, triangles[face])
                if tau < best_tau:
                    best_tau = tau
                    best_target = face
            continue

        left_node = left[node]
        right_node = right[node]
        if left_node >= 0:
            stack[stack_size] = left_node
            stack_size += 1
        if right_node >= 0:
            stack[stack_size] = right_node
            stack_size += 1

    if best_target < 0:
        return -1, np.inf, triangle_tests, bbox_tests
    return best_target, best_tau, triangle_tests, bbox_tests


@njit(cache=True)
def _trace_first_hits_bvh_kernel(
    origins: np.ndarray,
    directions: np.ndarray,
    triangles: np.ndarray,
    face_indices: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    start: np.ndarray,
    count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_sources, rays_per_source = origins.shape[:2]
    hits = np.full((n_sources, rays_per_source), -1, dtype=np.int64)
    taus = np.full((n_sources, rays_per_source), np.inf, dtype=np.float64)
    triangle_counts = np.zeros((n_sources, rays_per_source), dtype=np.int64)
    bbox_counts = np.zeros((n_sources, rays_per_source), dtype=np.int64)

    for source in range(n_sources):
        for ray_idx in range(rays_per_source):
            target, tau, tri_count, box_count = _intersect_bvh(
                origins[source, ray_idx],
                directions[source, ray_idx],
                source,
                triangles,
                face_indices,
                bbox_min,
                bbox_max,
                left,
                right,
                start,
                count,
            )
            hits[source, ray_idx] = target
            taus[source, ray_idx] = tau
            triangle_counts[source, ray_idx] = tri_count
            bbox_counts[source, ray_idx] = box_count

    return hits, taus, triangle_counts, bbox_counts


def trace_first_hits_bvh_accelerated(bvh: BVH3D, origins: np.ndarray, directions: np.ndarray, weights: np.ndarray) -> FirstHitMap3D:
    hits, taus, triangle_counts, bbox_counts = _trace_first_hits_bvh_kernel(
        origins,
        directions,
        bvh.mesh.triangles,
        bvh.face_indices,
        bvh.bbox_min,
        bvh.bbox_max,
        bvh.left,
        bvh.right,
        bvh.start,
        bvh.count,
    )
    return FirstHitMap3D(
        hits,
        taus,
        weights.copy(),
        int(np.sum(triangle_counts)),
        int(np.sum(bbox_counts)),
    )


@njit(cache=True)
def _ray_in_cone(origin: np.ndarray, direction: np.ndarray, center: np.ndarray, radius: float, padding: float) -> bool:
    ax0 = center[0] - origin[0]
    ax1 = center[1] - origin[1]
    ax2 = center[2] - origin[2]
    distance = np.sqrt(ax0 * ax0 + ax1 * ax1 + ax2 * ax2)
    if distance <= radius:
        return True
    ax0 /= distance
    ax1 /= distance
    ax2 /= distance
    half_angle = np.arcsin(min(1.0, radius / distance)) + padding
    if half_angle >= np.pi:
        return True
    return direction[0] * ax0 + direction[1] * ax1 + direction[2] * ax2 >= np.cos(half_angle)


@njit(cache=True)
def _cone_membership_count(
    origin: np.ndarray,
    direction: np.ndarray,
    support_centers: np.ndarray,
    support_radii: np.ndarray,
    padding: float,
) -> int:
    count = 0
    for support_idx in range(len(support_radii)):
        if _ray_in_cone(origin, direction, support_centers[support_idx], support_radii[support_idx], padding):
            count += 1
    return count


@njit(cache=True)
def _intersect_single_face(
    origin: np.ndarray,
    direction: np.ndarray,
    face: int,
    skip_face: int,
    triangles: np.ndarray,
) -> tuple[int, float, int]:
    if face < 0 or face == skip_face:
        return -1, np.inf, 0
    tau = _ray_triangle_tau(origin, direction, triangles[face])
    if np.isfinite(tau):
        return face, tau, 1
    return -1, np.inf, 1


@njit(cache=True)
def _transport_cone_bvh_kernel(
    old_hits: np.ndarray,
    old_taus: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
    support_centers: np.ndarray,
    support_radii: np.ndarray,
    cone_padding: float,
    changed_mask: np.ndarray,
    full_triangles: np.ndarray,
    full_face_indices: np.ndarray,
    full_bbox_min: np.ndarray,
    full_bbox_max: np.ndarray,
    full_left: np.ndarray,
    full_right: np.ndarray,
    full_start: np.ndarray,
    full_count: np.ndarray,
    support_face_indices: np.ndarray,
    support_bbox_min: np.ndarray,
    support_bbox_max: np.ndarray,
    support_left: np.ndarray,
    support_right: np.ndarray,
    support_start: np.ndarray,
    support_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_sources, rays_per_source = old_hits.shape
    hits = old_hits.copy()
    taus = old_taus.copy()
    counts = np.zeros(6, dtype=np.int64)

    for source in range(n_sources):
        source_is_active = changed_mask[source]
        for ray_idx in range(rays_per_source):
            origin = origins[source, ray_idx]
            direction = directions[source, ray_idx]

            if source_is_active:
                counts[2] += 1
                target, tau, tri_count, box_count = _intersect_bvh(
                    origin,
                    direction,
                    source,
                    full_triangles,
                    full_face_indices,
                    full_bbox_min,
                    full_bbox_max,
                    full_left,
                    full_right,
                    full_start,
                    full_count,
                )
                hits[source, ray_idx] = target
                taus[source, ray_idx] = tau
                counts[0] += tri_count
                counts[1] += box_count
                continue

            cone_count = _cone_membership_count(origin, direction, support_centers, support_radii, cone_padding)
            if cone_count == 0:
                continue

            counts[2] += 1
            if cone_count > 1:
                counts[3] += 1
            old_target = old_hits[source, ray_idx]
            if old_target < 0 or changed_mask[old_target]:
                counts[5] += 1
                target, tau, tri_count, box_count = _intersect_bvh(
                    origin,
                    direction,
                    source,
                    full_triangles,
                    full_face_indices,
                    full_bbox_min,
                    full_bbox_max,
                    full_left,
                    full_right,
                    full_start,
                    full_count,
                )
                hits[source, ray_idx] = target
                taus[source, ray_idx] = tau
                counts[0] += tri_count
                counts[1] += box_count
                continue

            counts[4] += 1
            support_target, support_tau, support_tri, support_box = _intersect_bvh(
                origin,
                direction,
                source,
                full_triangles,
                support_face_indices,
                support_bbox_min,
                support_bbox_max,
                support_left,
                support_right,
                support_start,
                support_count,
            )
            old_face, old_tau, old_tri = _intersect_single_face(origin, direction, old_target, source, full_triangles)
            counts[0] += support_tri + old_tri
            counts[1] += support_box

            if support_tau < old_tau:
                hits[source, ray_idx] = support_target
                taus[source, ray_idx] = support_tau
            elif old_face >= 0:
                hits[source, ray_idx] = old_face
                taus[source, ray_idx] = old_tau
            else:
                counts[5] += 1
                target, tau, tri_count, box_count = _intersect_bvh(
                    origin,
                    direction,
                    source,
                    full_triangles,
                    full_face_indices,
                    full_bbox_min,
                    full_bbox_max,
                    full_left,
                    full_right,
                    full_start,
                    full_count,
                )
                hits[source, ray_idx] = target
                taus[source, ray_idx] = tau
                counts[0] += tri_count
                counts[1] += box_count

    return hits, taus, counts


def transport_cone_bvh_accelerated(
    old_hits: np.ndarray,
    old_taus: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
    support_centers: np.ndarray,
    support_radii: np.ndarray,
    cone_padding: float,
    changed_mask: np.ndarray,
    full_bvh: BVH3D,
    support_bvh: BVH3D,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _transport_cone_bvh_kernel(
        old_hits,
        old_taus,
        origins,
        directions,
        support_centers,
        support_radii,
        cone_padding,
        changed_mask,
        full_bvh.mesh.triangles,
        full_bvh.face_indices,
        full_bvh.bbox_min,
        full_bvh.bbox_max,
        full_bvh.left,
        full_bvh.right,
        full_bvh.start,
        full_bvh.count,
        support_bvh.face_indices,
        support_bvh.bbox_min,
        support_bvh.bbox_max,
        support_bvh.left,
        support_bvh.right,
        support_bvh.start,
        support_bvh.count,
    )


def warm_up_accelerated_3d() -> None:
    from .bvh3d import BVH3D
    from .geometry3d import make_default_cap_bump, make_sphere_cavity
    from .rays3d import make_cosine_hemisphere_rays

    bump = make_default_cap_bump(amplitude=0.05, width=0.5)
    old_mesh = make_sphere_cavity(n_lat=4, n_lon=8, bumps=(bump,), alpha=0.0)
    new_mesh = make_sphere_cavity(n_lat=4, n_lon=8, bumps=(bump,), alpha=1.0)
    rays = make_cosine_hemisphere_rays(old_mesh, 2)
    bvh = BVH3D(old_mesh)
    state = trace_first_hits_bvh_accelerated(bvh, rays.origins, rays.directions, rays.weights)
    new_rays = make_cosine_hemisphere_rays(new_mesh, 2)
    changed = old_mesh.support_mask | new_mesh.support_mask
    changed_faces = np.flatnonzero(changed)
    support_bvh = BVH3D(new_mesh, changed_faces)
    full_bvh = BVH3D(new_mesh)
    support_centers = np.mean(new_mesh.vertices[np.unique(new_mesh.faces[changed_faces].ravel())], axis=0)[None, :]
    support_radii = np.array([1.0], dtype=float)
    transport_cone_bvh_accelerated(
        state.hits,
        state.taus,
        new_rays.origins,
        new_rays.directions,
        support_centers,
        support_radii,
        1e-6,
        changed,
        full_bvh,
        support_bvh,
    )
