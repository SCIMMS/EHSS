from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .bvh3d import BVH3D
from .geometry3d import TriangleMesh3D, normalize_vectors


@dataclass(frozen=True)
class PVDProfile3D:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=float)
        y = np.asarray(self.y, dtype=float)
        z = np.asarray(self.z, dtype=float)
        if x.ndim != 1 or y.ndim != 1 or z.shape != (len(x), len(y)):
            raise ValueError("z must have shape (len(x), len(y)).")
        if len(x) < 3 or len(y) < 3:
            raise ValueError("profile axes must contain at least three nodes.")
        if np.any(np.diff(x) <= 0.0) or np.any(np.diff(y) <= 0.0):
            raise ValueError("profile axes must be strictly increasing.")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "z", z)


@dataclass(frozen=True)
class PVDGeometry3D:
    mesh: TriangleMesh3D
    normals: np.ndarray
    labels: tuple[str, ...]
    profile_face_nodes: np.ndarray
    dynamic_faces: np.ndarray
    static_faces: np.ndarray

    def __post_init__(self) -> None:
        normals = np.asarray(self.normals, dtype=float)
        profile_face_nodes = np.asarray(self.profile_face_nodes, dtype=np.int64)
        dynamic_faces = np.asarray(self.dynamic_faces, dtype=np.int64)
        static_faces = np.asarray(self.static_faces, dtype=np.int64)
        if normals.shape != (self.mesh.n_faces, 3):
            raise ValueError("normals must have one vector per face.")
        if len(self.labels) != self.mesh.n_faces:
            raise ValueError("labels must have one value per face.")
        if profile_face_nodes.shape != (self.mesh.n_faces, 3, 2):
            raise ValueError("profile_face_nodes must have shape (n_faces, 3, 2).")
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "profile_face_nodes", profile_face_nodes)
        object.__setattr__(self, "dynamic_faces", dynamic_faces)
        object.__setattr__(self, "static_faces", static_faces)


@dataclass(frozen=True)
class PVDRayBundle3D:
    origins: np.ndarray
    directions: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        origins = np.asarray(self.origins, dtype=float)
        directions = np.asarray(self.directions, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        if origins.ndim != 2 or origins.shape[1] != 3:
            raise ValueError("origins must have shape (n_rays, 3).")
        if directions.shape != origins.shape:
            raise ValueError("directions must match origins.")
        if weights.shape != (len(origins),):
            raise ValueError("weights must have one value per ray.")
        object.__setattr__(self, "origins", origins)
        object.__setattr__(self, "directions", normalize_vectors(directions))
        object.__setattr__(self, "weights", weights)

    @property
    def n_rays(self) -> int:
        return len(self.origins)


@dataclass(frozen=True)
class PVDTrace3D:
    targets: np.ndarray
    taus: np.ndarray
    incidence: np.ndarray
    weights: np.ndarray
    triangle_tests: int
    bbox_tests: int


@dataclass(frozen=True)
class EnclosingSupport3D:
    minimum: np.ndarray
    maximum: np.ndarray

    def __post_init__(self) -> None:
        minimum = np.asarray(self.minimum, dtype=float)
        maximum = np.asarray(self.maximum, dtype=float)
        if minimum.shape != (3,) or maximum.shape != (3,):
            raise ValueError("support bounds must have shape (3,).")
        if np.any(maximum <= minimum):
            raise ValueError("support bounds must be ordered and nondegenerate.")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)


@dataclass(frozen=True)
class PVD3DConfig:
    scenario: str = "localized_square_trench_growth"
    steps: int = 6
    profile_nodes: int = 9
    source_x_samples: int = 16
    source_y_samples: int = 16
    directions_per_source: int = 5
    field_half_width: float = 1.5
    trench_half_width: float = 0.6
    trench_depth: float = 1.0
    source_half_width: float = 1.7
    source_center_x: float = 0.0
    source_center_y: float = 0.0
    source_z: float = 1.0
    source_tilt_deg: float = 5.0
    source_azimuth_deg: float = 17.0
    angular_spread_deg: float = 22.0
    direction_rule: str = "legacy_absolute"
    seed_height: float = 0.05
    seed_radius: float = 0.18
    growth_radius: float = 0.24
    support_padding_xy: float = 0.16
    support_height: float = 0.65
    deposition_scale: float = 0.018
    max_step_displacement: float = 0.01
    leaf_size: int = 8
    timing_repeats: int = 1
    timing_order: str = "route_full_refit"
    independent_reference_max_rays: int | None = None
    tolerance: float = 1e-11

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive.")
        if self.profile_nodes < 5 or self.profile_nodes % 2 == 0:
            raise ValueError("profile_nodes must be an odd integer of at least 5.")
        if self.source_x_samples <= 0 or self.source_y_samples <= 0:
            raise ValueError("source sample counts must be positive.")
        if self.directions_per_source not in {1, 5, 9}:
            raise ValueError("directions_per_source must be 1, 5, or 9.")
        if not 0.0 <= self.source_tilt_deg < 80.0:
            raise ValueError("source_tilt_deg must be in [0, 80).")
        if not 0.0 <= self.angular_spread_deg < 45.0:
            raise ValueError("angular_spread_deg must be in [0, 45).")
        if self.source_tilt_deg + self.angular_spread_deg >= 89.0:
            raise ValueError("tilt plus angular spread must remain below 89 degrees.")
        if self.direction_rule not in {"legacy_absolute", "local_cone"}:
            raise ValueError(
                "direction_rule must be 'legacy_absolute' or 'local_cone'."
            )
        if self.timing_repeats <= 0:
            raise ValueError("timing_repeats must be positive.")
        valid_timing_orders = {
            "route_full_refit",
            "route_refit_full",
            "full_route_refit",
            "full_refit_route",
            "refit_route_full",
            "refit_full_route",
        }
        if self.timing_order not in valid_timing_orders:
            raise ValueError(
                "timing_order must be one of the six route/full/refit permutations."
            )
        if (
            self.independent_reference_max_rays is not None
            and self.independent_reference_max_rays <= 0
        ):
            raise ValueError("independent_reference_max_rays must be positive.")


@dataclass(frozen=True)
class PVD3DResult:
    config: PVD3DConfig
    support: EnclosingSupport3D
    candidate_mask: np.ndarray
    rows: tuple[dict[str, float | int | str | bool], ...]
    route_profile: PVDProfile3D
    full_profile: PVDProfile3D


def make_initial_profile_3d(config: PVD3DConfig) -> PVDProfile3D:
    axis = np.linspace(
        -config.trench_half_width,
        config.trench_half_width,
        config.profile_nodes,
    )
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy)
    height = np.zeros_like(radius)
    inside = radius < config.seed_radius
    height[inside] = config.seed_height * 0.5 * (
        1.0 + np.cos(np.pi * radius[inside] / config.seed_radius)
    )
    height[0, :] = 0.0
    height[-1, :] = 0.0
    height[:, 0] = 0.0
    height[:, -1] = 0.0
    return PVDProfile3D(axis, axis.copy(), -config.trench_depth + height)


def make_enclosing_support_3d(config: PVD3DConfig) -> EnclosingSupport3D:
    half_width = config.growth_radius + config.support_padding_xy
    return EnclosingSupport3D(
        minimum=np.array(
            [-half_width, -half_width, -config.trench_depth - 1e-8]
        ),
        maximum=np.array(
            [
                half_width,
                half_width,
                -config.trench_depth + config.support_height,
            ]
        ),
    )


def _oriented_quad_faces(
    vertex_offset: int,
    vertices: list[list[float]],
    desired_normal: np.ndarray,
) -> tuple[list[list[int]], list[np.ndarray]]:
    trial = np.asarray(vertices[vertex_offset : vertex_offset + 4], dtype=float)
    first = [vertex_offset, vertex_offset + 1, vertex_offset + 2]
    second = [vertex_offset, vertex_offset + 2, vertex_offset + 3]
    normal = np.cross(trial[1] - trial[0], trial[2] - trial[0])
    if float(np.dot(normal, desired_normal)) < 0.0:
        first = [vertex_offset, vertex_offset + 2, vertex_offset + 1]
        second = [vertex_offset, vertex_offset + 3, vertex_offset + 2]
    return [first, second], [desired_normal.copy(), desired_normal.copy()]


def make_trench_geometry_3d(
    profile: PVDProfile3D,
    config: PVD3DConfig,
) -> PVDGeometry3D:
    field = config.field_half_width
    half = config.trench_half_width
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    labels: list[str] = []
    normals: list[np.ndarray] = []
    face_profile_nodes: list[np.ndarray] = []

    def add_quad(
        corners: list[list[float]],
        desired_normal: tuple[float, float, float],
        label: str,
    ) -> None:
        offset = len(vertices)
        vertices.extend(corners)
        quad_faces, quad_normals = _oriented_quad_faces(
            offset,
            vertices,
            np.asarray(desired_normal, dtype=float),
        )
        faces.extend(quad_faces)
        normals.extend(quad_normals)
        labels.extend((label, label))
        face_profile_nodes.extend(
            (
                np.full((3, 2), -1, dtype=np.int64),
                np.full((3, 2), -1, dtype=np.int64),
            )
        )

    add_quad(
        [[-field, -field, 0.0], [-half, -field, 0.0], [-half, field, 0.0], [-field, field, 0.0]],
        (0.0, 0.0, 1.0),
        "mask_left",
    )
    add_quad(
        [[half, -field, 0.0], [field, -field, 0.0], [field, field, 0.0], [half, field, 0.0]],
        (0.0, 0.0, 1.0),
        "mask_right",
    )
    add_quad(
        [[-half, -field, 0.0], [half, -field, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        (0.0, 0.0, 1.0),
        "mask_front",
    )
    add_quad(
        [[-half, half, 0.0], [half, half, 0.0], [half, field, 0.0], [-half, field, 0.0]],
        (0.0, 0.0, 1.0),
        "mask_back",
    )

    z_left = profile.z[0, :]
    z_right = profile.z[-1, :]
    z_front = profile.z[:, 0]
    z_back = profile.z[:, -1]
    if not (
        np.allclose(z_left, z_left[0])
        and np.allclose(z_right, z_right[0])
        and np.allclose(z_front, z_front[0])
        and np.allclose(z_back, z_back[0])
    ):
        raise ValueError("profile boundary nodes must remain fixed.")
    bottom = float(profile.z[0, 0])
    add_quad(
        [[-half, -half, 0.0], [-half, half, 0.0], [-half, half, bottom], [-half, -half, bottom]],
        (1.0, 0.0, 0.0),
        "left_wall",
    )
    add_quad(
        [[half, -half, bottom], [half, half, bottom], [half, half, 0.0], [half, -half, 0.0]],
        (-1.0, 0.0, 0.0),
        "right_wall",
    )
    add_quad(
        [[-half, -half, bottom], [half, -half, bottom], [half, -half, 0.0], [-half, -half, 0.0]],
        (0.0, 1.0, 0.0),
        "front_wall",
    )
    add_quad(
        [[-half, half, 0.0], [half, half, 0.0], [half, half, bottom], [-half, half, bottom]],
        (0.0, -1.0, 0.0),
        "back_wall",
    )

    grid_offset = len(vertices)
    for i, x_value in enumerate(profile.x):
        for j, y_value in enumerate(profile.y):
            vertices.append([x_value, y_value, profile.z[i, j]])

    def grid_vertex(i: int, j: int) -> int:
        return grid_offset + i * len(profile.y) + j

    dynamic_profile_faces: list[int] = []
    for i in range(len(profile.x) - 1):
        for j in range(len(profile.y) - 1):
            a = grid_vertex(i, j)
            b = grid_vertex(i + 1, j)
            c = grid_vertex(i + 1, j + 1)
            d = grid_vertex(i, j + 1)
            triangle_specs = (
                ([a, b, c], np.array([[i, j], [i + 1, j], [i + 1, j + 1]])),
                ([a, c, d], np.array([[i, j], [i + 1, j + 1], [i, j + 1]])),
            )
            cell_radius = min(
                np.hypot(profile.x[ii], profile.y[jj])
                for ii in (i, i + 1)
                for jj in (j, j + 1)
            )
            cell_dynamic = cell_radius < config.growth_radius
            for local_triangle, node_map in triangle_specs:
                triangle = np.asarray([vertices[idx] for idx in local_triangle])
                normal = np.cross(
                    triangle[1] - triangle[0],
                    triangle[2] - triangle[0],
                )
                normal = normalize_vectors(normal[None, :])[0]
                if normal[2] < 0.0:
                    local_triangle = [
                        local_triangle[0],
                        local_triangle[2],
                        local_triangle[1],
                    ]
                    node_map = node_map[[0, 2, 1]]
                    normal *= -1.0
                face_idx = len(faces)
                faces.append(local_triangle)
                labels.append(f"profile_{i:02d}_{j:02d}")
                normals.append(normal)
                face_profile_nodes.append(node_map)
                if cell_dynamic:
                    dynamic_profile_faces.append(face_idx)

    vertices_array = np.asarray(vertices, dtype=float)
    faces_array = np.asarray(faces, dtype=np.int64)
    mesh = TriangleMesh3D(
        vertices=vertices_array,
        faces=faces_array,
        support_mask=np.zeros(len(faces_array), dtype=bool),
        support_labels=np.zeros(len(faces_array), dtype=np.int64),
        vertex_displacement=np.zeros(len(vertices_array), dtype=float),
    )
    dynamic_faces = np.asarray(dynamic_profile_faces, dtype=np.int64)
    all_faces = np.arange(mesh.n_faces, dtype=np.int64)
    static_faces = all_faces[~np.isin(all_faces, dynamic_faces)]
    return PVDGeometry3D(
        mesh=mesh,
        normals=normalize_vectors(np.asarray(normals, dtype=float)),
        labels=tuple(labels),
        profile_face_nodes=np.asarray(face_profile_nodes, dtype=np.int64),
        dynamic_faces=dynamic_faces,
        static_faces=static_faces,
    )


def make_pvd_rays_3d(config: PVD3DConfig) -> PVDRayBundle3D:
    x_values = config.source_center_x + np.linspace(
        -config.source_half_width,
        config.source_half_width,
        config.source_x_samples,
    )
    y_values = config.source_center_y + np.linspace(
        -config.source_half_width,
        config.source_half_width,
        config.source_y_samples,
    )
    directions: list[np.ndarray] = []
    angular_weights: list[float] = []
    if config.direction_rule == "legacy_absolute":
        if config.directions_per_source == 1:
            angle_pairs = [(5.0, 17.0)]
        elif config.directions_per_source == 5:
            angle_pairs = [(5.0, 17.0)] + [
                (22.0, float(azimuth_deg))
                for azimuth_deg in (0, 90, 180, 270)
            ]
        else:
            angle_pairs = [(5.0, 17.0)] + [
                (26.0, float(azimuth_deg))
                for azimuth_deg in range(0, 360, 45)
            ]
        for polar_deg, azimuth_deg in angle_pairs:
            polar = np.deg2rad(polar_deg)
            azimuth = np.deg2rad(azimuth_deg)
            directions.append(
                np.array(
                    [
                        np.sin(polar) * np.cos(azimuth),
                        np.sin(polar) * np.sin(azimuth),
                        -np.cos(polar),
                    ]
                )
            )
            angular_weights.append(float(np.cos(polar)))
    else:
        tilt = np.deg2rad(config.source_tilt_deg)
        azimuth = np.deg2rad(config.source_azimuth_deg)
        center_direction = np.array(
            [
                np.sin(tilt) * np.cos(azimuth),
                np.sin(tilt) * np.sin(azimuth),
                -np.cos(tilt),
            ]
        )
        polar_tangent = np.array(
            [
                np.cos(tilt) * np.cos(azimuth),
                np.cos(tilt) * np.sin(azimuth),
                np.sin(tilt),
            ]
        )
        azimuth_tangent = np.array(
            [-np.sin(azimuth), np.cos(azimuth), 0.0]
        )
        if config.directions_per_source == 1:
            angular_offsets = [(0.0, 0.0)]
        else:
            azimuth_count = config.directions_per_source - 1
            angular_offsets = [(0.0, 0.0)] + [
                (
                    config.angular_spread_deg,
                    360.0 * index / azimuth_count,
                )
                for index in range(azimuth_count)
            ]
        for offset_deg, ring_azimuth_deg in angular_offsets:
            offset = np.deg2rad(offset_deg)
            ring_azimuth = np.deg2rad(ring_azimuth_deg)
            tangent = (
                np.cos(ring_azimuth) * polar_tangent
                + np.sin(ring_azimuth) * azimuth_tangent
            )
            direction = (
                np.cos(offset) * center_direction
                + np.sin(offset) * tangent
            )
            directions.append(direction / np.linalg.norm(direction))
            angular_weights.append(float(np.cos(offset)))
    angular_weights_array = np.asarray(angular_weights)
    angular_weights_array /= np.sum(angular_weights_array)

    origins: list[np.ndarray] = []
    ray_directions: list[np.ndarray] = []
    weights: list[float] = []
    source_weight = 1.0 / (len(x_values) * len(y_values))
    for x_value in x_values:
        for y_value in y_values:
            for direction, angular_weight in zip(
                directions,
                angular_weights_array,
            ):
                origins.append(np.array([x_value, y_value, config.source_z]))
                ray_directions.append(direction)
                weights.append(source_weight * float(angular_weight))
    return PVDRayBundle3D(
        origins=np.asarray(origins),
        directions=np.asarray(ray_directions),
        weights=np.asarray(weights),
    )


def _trace_with_bvh(
    geometry: PVDGeometry3D,
    rays: PVDRayBundle3D,
    bvh: BVH3D,
    ray_indices: np.ndarray | None = None,
) -> PVDTrace3D:
    if ray_indices is None:
        ray_indices = np.arange(rays.n_rays, dtype=np.int64)
    ray_indices = np.asarray(ray_indices, dtype=np.int64)
    targets = np.full(len(ray_indices), -1, dtype=np.int64)
    taus = np.full(len(ray_indices), np.inf, dtype=float)
    incidence = np.zeros(len(ray_indices), dtype=float)
    triangle_tests = 0
    bbox_tests = 0
    for local_idx, ray_idx in enumerate(ray_indices):
        hit = bvh.intersect_ray(
            rays.origins[ray_idx],
            rays.directions[ray_idx],
        )
        triangle_tests += hit.triangle_tests
        bbox_tests += hit.bbox_tests
        if hit.target >= 0:
            targets[local_idx] = hit.target
            taus[local_idx] = hit.tau
            incidence[local_idx] = max(
                0.0,
                -float(
                    np.dot(
                        rays.directions[ray_idx],
                        geometry.normals[hit.target],
                    )
                ),
            )
    return PVDTrace3D(
        targets=targets,
        taus=taus,
        incidence=incidence,
        weights=rays.weights[ray_indices].copy(),
        triangle_tests=triangle_tests,
        bbox_tests=bbox_tests,
    )


def trace_pvd3d_full_bvh(
    geometry: PVDGeometry3D,
    rays: PVDRayBundle3D,
    leaf_size: int,
) -> PVDTrace3D:
    return _trace_with_bvh(
        geometry,
        rays,
        BVH3D(geometry.mesh, leaf_size=leaf_size),
    )


def _ray_intersects_aabb(
    origin: np.ndarray,
    direction: np.ndarray,
    support: EnclosingSupport3D,
) -> bool:
    t_entry = 0.0
    t_exit = float("inf")
    for axis in range(3):
        component = float(direction[axis])
        if abs(component) <= 1e-14:
            if origin[axis] < support.minimum[axis] or origin[axis] > support.maximum[axis]:
                return False
            continue
        t0 = (support.minimum[axis] - origin[axis]) / component
        t1 = (support.maximum[axis] - origin[axis]) / component
        if t0 > t1:
            t0, t1 = t1, t0
        t_entry = max(t_entry, t0)
        t_exit = min(t_exit, t1)
        if t_exit < t_entry:
            return False
    return t_exit > 1e-9


def enclosing_support_candidate_mask_3d(
    rays: PVDRayBundle3D,
    support: EnclosingSupport3D,
) -> np.ndarray:
    return np.asarray(
        [
            _ray_intersects_aabb(origin, direction, support)
            for origin, direction in zip(rays.origins, rays.directions)
        ],
        dtype=bool,
    )


def _trace_candidate_split_bvh(
    geometry: PVDGeometry3D,
    rays: PVDRayBundle3D,
    candidate_indices: np.ndarray,
    static_bvh: BVH3D,
    dynamic_bvh: BVH3D,
) -> PVDTrace3D:
    targets = np.full(len(candidate_indices), -1, dtype=np.int64)
    taus = np.full(len(candidate_indices), np.inf, dtype=float)
    incidence = np.zeros(len(candidate_indices), dtype=float)
    triangle_tests = 0
    bbox_tests = 0
    for local_idx, ray_idx in enumerate(candidate_indices):
        origin = rays.origins[ray_idx]
        direction = rays.directions[ray_idx]
        static_hit = static_bvh.intersect_ray(origin, direction)
        dynamic_hit = dynamic_bvh.intersect_ray(origin, direction)
        triangle_tests += static_hit.triangle_tests + dynamic_hit.triangle_tests
        bbox_tests += static_hit.bbox_tests + dynamic_hit.bbox_tests
        hit = dynamic_hit if dynamic_hit.tau < static_hit.tau else static_hit
        if hit.target >= 0:
            targets[local_idx] = hit.target
            taus[local_idx] = hit.tau
            incidence[local_idx] = max(
                0.0,
                -float(np.dot(direction, geometry.normals[hit.target])),
            )
    return PVDTrace3D(
        targets=targets,
        taus=taus,
        incidence=incidence,
        weights=rays.weights[candidate_indices].copy(),
        triangle_tests=triangle_tests,
        bbox_tests=bbox_tests,
    )


def transport_pvd3d_enclosing_support(
    old_trace: PVDTrace3D,
    new_geometry: PVDGeometry3D,
    rays: PVDRayBundle3D,
    candidate_mask: np.ndarray,
    static_bvh: BVH3D,
    leaf_size: int,
) -> PVDTrace3D:
    candidate_indices = np.flatnonzero(candidate_mask)
    dynamic_bvh = BVH3D(
        new_geometry.mesh,
        face_indices=new_geometry.dynamic_faces,
        leaf_size=leaf_size,
    )
    candidate_trace = _trace_candidate_split_bvh(
        new_geometry,
        rays,
        candidate_indices,
        static_bvh,
        dynamic_bvh,
    )
    targets = np.array(old_trace.targets, copy=True)
    taus = np.array(old_trace.taus, copy=True)
    incidence = np.array(old_trace.incidence, copy=True)
    targets[candidate_indices] = candidate_trace.targets
    taus[candidate_indices] = candidate_trace.taus
    incidence[candidate_indices] = candidate_trace.incidence
    return PVDTrace3D(
        targets=targets,
        taus=taus,
        incidence=incidence,
        weights=rays.weights.copy(),
        triangle_tests=candidate_trace.triangle_tests,
        bbox_tests=candidate_trace.bbox_tests,
    )


def _profile_node_density_3d(
    profile: PVDProfile3D,
    geometry: PVDGeometry3D,
    trace: PVDTrace3D,
) -> np.ndarray:
    valid = trace.targets >= 0
    face_flux = np.bincount(
        trace.targets[valid],
        weights=trace.weights[valid] * trace.incidence[valid],
        minlength=geometry.mesh.n_faces,
    )
    face_density = face_flux / np.maximum(geometry.mesh.areas, 1e-15)
    density = np.zeros_like(profile.z)
    counts = np.zeros_like(profile.z)
    for face_idx in np.flatnonzero(geometry.profile_face_nodes[:, 0, 0] >= 0):
        for i, j in geometry.profile_face_nodes[face_idx]:
            density[i, j] += face_density[face_idx]
            counts[i, j] += 1.0
    density /= np.maximum(counts, 1.0)
    smoothed = density.copy()
    smoothed[1:-1, 1:-1] = (
        4.0 * density[1:-1, 1:-1]
        + density[:-2, 1:-1]
        + density[2:, 1:-1]
        + density[1:-1, :-2]
        + density[1:-1, 2:]
    ) / 8.0
    return smoothed


def advance_profile_3d(
    profile: PVDProfile3D,
    geometry: PVDGeometry3D,
    trace: PVDTrace3D,
    config: PVD3DConfig,
) -> PVDProfile3D:
    density = _profile_node_density_3d(profile, geometry, trace)
    xx, yy = np.meshgrid(profile.x, profile.y, indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy)
    window = np.zeros_like(radius)
    inside = radius < config.growth_radius
    window[inside] = 0.5 * (
        1.0 + np.cos(np.pi * radius[inside] / config.growth_radius)
    )
    displacement = np.minimum(
        config.deposition_scale * density * window,
        config.max_step_displacement,
    )
    displacement[0, :] = 0.0
    displacement[-1, :] = 0.0
    displacement[:, 0] = 0.0
    displacement[:, -1] = 0.0
    return PVDProfile3D(profile.x.copy(), profile.y.copy(), profile.z + displacement)


def validate_dynamic_faces_contained(
    old_geometry: PVDGeometry3D,
    new_geometry: PVDGeometry3D,
    support: EnclosingSupport3D,
    tolerance: float,
) -> None:
    old_triangles = old_geometry.mesh.triangles
    new_triangles = new_geometry.mesh.triangles
    changed = np.any(
        np.abs(old_triangles - new_triangles) > tolerance,
        axis=(1, 2),
    )
    changed_faces = np.flatnonzero(changed)
    if np.any(~np.isin(changed_faces, new_geometry.dynamic_faces)):
        raise ValueError("a changed face is missing from the dynamic face set.")
    if len(changed_faces):
        points = np.concatenate(
            (old_triangles[changed_faces], new_triangles[changed_faces]),
            axis=1,
        )
        if (
            np.any(points < support.minimum[None, None, :] - tolerance)
            or np.any(points > support.maximum[None, None, :] + tolerance)
        ):
            raise ValueError("changed 3D profile geometry lies outside the support.")


def _trace_flux_3d(trace: PVDTrace3D, face_count: int) -> np.ndarray:
    valid = trace.targets >= 0
    return np.bincount(
        trace.targets[valid],
        weights=trace.weights[valid] * trace.incidence[valid],
        minlength=face_count,
    ).astype(float)


def _relative_error(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(actual) - np.asarray(reference))
        / max(float(np.linalg.norm(reference)), 1e-15)
    )


def _finite_tau_error(actual: np.ndarray, reference: np.ndarray) -> float:
    finite = np.isfinite(actual) & np.isfinite(reference)
    if not np.any(finite):
        return 0.0
    return float(np.max(np.abs(actual[finite] - reference[finite])))


def _tau_changed(old: np.ndarray, new: np.ndarray, tolerance: float) -> np.ndarray:
    old_finite = np.isfinite(old)
    new_finite = np.isfinite(new)
    changed = old_finite != new_finite
    both = old_finite & new_finite
    changed[both] |= np.abs(old[both] - new[both]) > tolerance
    return changed


def _timed_call(callable_, repeats: int):
    elapsed: list[float] = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = callable_()
        elapsed.append(time.perf_counter() - start)
    return result, float(np.median(np.asarray(elapsed)))


def _intersect_triangle_independent(
    origin: np.ndarray,
    direction: np.ndarray,
    triangle: np.ndarray,
    tolerance: float = 1e-10,
) -> float:
    v0, v1, v2 = triangle
    system = np.column_stack((direction, -(v1 - v0), -(v2 - v0)))
    determinant = float(np.linalg.det(system))
    if abs(determinant) <= tolerance:
        return float("inf")
    try:
        tau, u, v = np.linalg.solve(system, v0 - origin)
    except np.linalg.LinAlgError:
        return float("inf")
    if (
        tau > 1e-9
        and u >= -tolerance
        and v >= -tolerance
        and u + v <= 1.0 + tolerance
    ):
        return float(tau)
    return float("inf")


def trace_pvd3d_independent(
    geometry: PVDGeometry3D,
    rays: PVDRayBundle3D,
    ray_indices: np.ndarray | None = None,
) -> PVDTrace3D:
    if ray_indices is None:
        ray_indices = np.arange(rays.n_rays, dtype=np.int64)
    ray_indices = np.asarray(ray_indices, dtype=np.int64)
    if ray_indices.ndim != 1:
        raise ValueError("ray_indices must be one-dimensional.")
    targets = np.full(len(ray_indices), -1, dtype=np.int64)
    taus = np.full(len(ray_indices), np.inf, dtype=float)
    incidence = np.zeros(len(ray_indices), dtype=float)
    triangles = geometry.mesh.triangles
    for local_idx, ray_idx in enumerate(ray_indices):
        best_target = -1
        best_tau = float("inf")
        for face_idx, triangle in enumerate(triangles):
            tau = _intersect_triangle_independent(
                rays.origins[ray_idx],
                rays.directions[ray_idx],
                triangle,
            )
            if tau < best_tau - 1e-12:
                best_tau = tau
                best_target = face_idx
        if best_target >= 0:
            targets[local_idx] = best_target
            taus[local_idx] = best_tau
            incidence[local_idx] = max(
                0.0,
                -float(
                    np.dot(
                        rays.directions[ray_idx],
                        geometry.normals[best_target],
                    )
                ),
            )
    return PVDTrace3D(
        targets=targets,
        taus=taus,
        incidence=incidence,
        weights=rays.weights[ray_indices].copy(),
        triangle_tests=len(ray_indices) * geometry.mesh.n_faces,
        bbox_tests=0,
    )


def _advance_profile_geometry_3d(
    profile: PVDProfile3D,
    geometry: PVDGeometry3D,
    trace: PVDTrace3D,
    config: PVD3DConfig,
    support: EnclosingSupport3D,
) -> tuple[PVDProfile3D, PVDGeometry3D]:
    next_profile = advance_profile_3d(profile, geometry, trace, config)
    next_geometry = make_trench_geometry_3d(next_profile, config)
    validate_dynamic_faces_contained(
        geometry,
        next_geometry,
        support,
        config.tolerance,
    )
    return next_profile, next_geometry


def run_pvd3d_case(config: PVD3DConfig) -> PVD3DResult:
    rays = make_pvd_rays_3d(config)
    support = make_enclosing_support_3d(config)
    discovery_start = time.perf_counter()
    candidate_mask = enclosing_support_candidate_mask_3d(rays, support)
    discovery_wall_s = time.perf_counter() - discovery_start

    route_profile = make_initial_profile_3d(config)
    full_profile = make_initial_profile_3d(config)
    route_geometry = make_trench_geometry_3d(route_profile, config)
    full_geometry = make_trench_geometry_3d(full_profile, config)
    initial_trace, cache_wall_s = _timed_call(
        lambda: trace_pvd3d_full_bvh(route_geometry, rays, config.leaf_size),
        config.timing_repeats,
    )
    refit_build_start = time.perf_counter()
    refit_bvh = BVH3D(full_geometry.mesh, leaf_size=config.leaf_size)
    refit_bvh_build_wall_s = time.perf_counter() - refit_build_start
    refit_initial_trace, refit_initial_trace_wall_s = _timed_call(
        lambda: _trace_with_bvh(full_geometry, rays, refit_bvh),
        config.timing_repeats,
    )
    if np.any(refit_initial_trace.targets != initial_trace.targets):
        raise RuntimeError("initial refit-BVH trace disagrees with the full rebuild.")
    static_build_start = time.perf_counter()
    static_bvh = BVH3D(
        route_geometry.mesh,
        face_indices=route_geometry.static_faces,
        leaf_size=config.leaf_size,
    )
    static_bvh_build_wall_s = time.perf_counter() - static_build_start
    route_trace = initial_trace
    full_trace = initial_trace
    route_cumulative = cache_wall_s + discovery_wall_s + static_bvh_build_wall_s
    full_cumulative = cache_wall_s
    refit_cumulative = refit_bvh_build_wall_s + refit_initial_trace_wall_s
    rows: list[dict[str, float | int | str | bool]] = []

    for step in range(1, config.steps + 1):
        def time_route_update():
            return _timed_call(
                lambda: _advance_profile_geometry_3d(
                    route_profile,
                    route_geometry,
                    route_trace,
                    config,
                    support,
                ),
                config.timing_repeats,
            )

        def time_full_update():
            return _timed_call(
                lambda: _advance_profile_geometry_3d(
                    full_profile,
                    full_geometry,
                    full_trace,
                    config,
                    support,
                ),
                config.timing_repeats,
            )

        if config.timing_order.split("_")[0] == "route":
            route_update, route_profile_update_wall_s = time_route_update()
            full_update, full_profile_update_wall_s = time_full_update()
        else:
            full_update, full_profile_update_wall_s = time_full_update()
            route_update, route_profile_update_wall_s = time_route_update()
        route_next, route_new_geometry = route_update
        full_next, full_new_geometry = full_update

        timed: dict[str, object] = {}

        def time_route_trace() -> None:
            timed["route"] = _timed_call(
                lambda: transport_pvd3d_enclosing_support(
                    route_trace,
                    route_new_geometry,
                    rays,
                    candidate_mask,
                    static_bvh,
                    config.leaf_size,
                ),
                config.timing_repeats,
            )

        def time_full_trace() -> None:
            timed["full"] = _timed_call(
                lambda: trace_pvd3d_full_bvh(
                    full_new_geometry,
                    rays,
                    config.leaf_size,
                ),
                config.timing_repeats,
            )

        def time_refit_trace() -> None:
            _, refit_wall_s_local = _timed_call(
                lambda: refit_bvh.refit(full_new_geometry.mesh),
                config.timing_repeats,
            )
            refit_trace_local, refit_trace_wall_s_local = _timed_call(
                lambda: _trace_with_bvh(full_new_geometry, rays, refit_bvh),
                config.timing_repeats,
            )
            timed["refit"] = (
                refit_trace_local,
                refit_wall_s_local,
                refit_trace_wall_s_local,
            )

        timers = {
            "route": time_route_trace,
            "full": time_full_trace,
            "refit": time_refit_trace,
        }
        for method in config.timing_order.split("_"):
            timers[method]()

        route_new_trace, route_wall_s = timed["route"]
        full_new_trace, full_wall_s = timed["full"]
        (
            refit_new_trace,
            refit_wall_s,
            refit_trace_wall_s,
        ) = timed["refit"]
        active = (
            (full_trace.targets != full_new_trace.targets)
            | _tau_changed(full_trace.taus, full_new_trace.taus, config.tolerance)
            | (
                np.abs(full_trace.incidence - full_new_trace.incidence)
                > config.tolerance
            )
        )
        route_flux = _trace_flux_3d(
            route_new_trace,
            route_new_geometry.mesh.n_faces,
        )
        full_flux = _trace_flux_3d(
            full_new_trace,
            full_new_geometry.mesh.n_faces,
        )
        route_step_wall_s = route_profile_update_wall_s + route_wall_s
        full_step_wall_s = full_profile_update_wall_s + full_wall_s
        refit_step_wall_s = (
            full_profile_update_wall_s + refit_wall_s + refit_trace_wall_s
        )
        route_cumulative += route_step_wall_s
        full_cumulative += full_step_wall_s
        refit_cumulative += refit_step_wall_s

        independent_checked = step == config.steps
        independent_reference_rays = 0
        independent_target_mismatch = 0
        independent_tau_error = 0.0
        independent_incidence_error = 0.0
        independent_wall_s = 0.0
        if independent_checked:
            if (
                config.independent_reference_max_rays is not None
                and config.independent_reference_max_rays < rays.n_rays
            ):
                independent_indices = np.unique(
                    np.linspace(
                        0,
                        rays.n_rays - 1,
                        config.independent_reference_max_rays,
                        dtype=np.int64,
                    )
                )
            else:
                independent_indices = np.arange(
                    rays.n_rays,
                    dtype=np.int64,
                )
            independent_reference_rays = len(independent_indices)
            independent_start = time.perf_counter()
            independent = trace_pvd3d_independent(
                full_new_geometry,
                rays,
                ray_indices=independent_indices,
            )
            independent_wall_s = time.perf_counter() - independent_start
            independent_target_mismatch = int(
                np.count_nonzero(
                    independent.targets
                    != full_new_trace.targets[independent_indices]
                )
            )
            independent_tau_error = _finite_tau_error(
                independent.taus,
                full_new_trace.taus[independent_indices],
            )
            independent_incidence_error = float(
                np.max(
                    np.abs(
                        independent.incidence
                        - full_new_trace.incidence[independent_indices]
                    )
                )
            )

        profile_height = full_next.z + config.trench_depth
        xx, yy = np.meshgrid(full_next.x, full_next.y, indexing="ij")
        deposited_volume = float(
            np.trapezoid(
                np.trapezoid(profile_height, full_next.y, axis=1),
                full_next.x,
            )
        )
        if deposited_volume > 1e-15:
            deposition_centroid_x = float(
                np.trapezoid(
                    np.trapezoid(
                        xx * profile_height,
                        full_next.y,
                        axis=1,
                    ),
                    full_next.x,
                )
                / deposited_volume
            )
            deposition_centroid_y = float(
                np.trapezoid(
                    np.trapezoid(
                        yy * profile_height,
                        full_next.y,
                        axis=1,
                    ),
                    full_next.x,
                )
                / deposited_volume
            )
        else:
            deposition_centroid_x = 0.0
            deposition_centroid_y = 0.0
        labels = np.asarray(full_new_geometry.labels, dtype=object)
        profile_face_mask = (
            full_new_geometry.profile_face_nodes[:, 0, 0] >= 0
        )
        mask_face_mask = np.asarray(
            [str(label).startswith("mask_") for label in labels],
            dtype=bool,
        )
        wall_face_mask = np.asarray(
            [str(label).endswith("_wall") for label in labels],
            dtype=bool,
        )
        total_flux = float(np.sum(full_flux))
        profile_flux = float(np.sum(full_flux[profile_face_mask]))
        mask_flux = float(np.sum(full_flux[mask_face_mask]))
        wall_flux = float(np.sum(full_flux[wall_face_mask]))
        profile_area = float(
            np.sum(full_new_geometry.mesh.areas[profile_face_mask])
        )
        mask_area = float(
            np.sum(full_new_geometry.mesh.areas[mask_face_mask])
        )
        refit_target_mismatch = int(
            np.count_nonzero(
                refit_new_trace.targets != full_new_trace.targets
            )
        )
        rows.append(
            {
                "scenario": config.scenario,
                "step": step,
                "model_scope": "3d_multistep_ballistic_pvd_square_trench",
                "source_tilt_deg": config.source_tilt_deg,
                "source_azimuth_deg": config.source_azimuth_deg,
                "angular_spread_deg": config.angular_spread_deg,
                "n_rays": rays.n_rays,
                "face_count": full_new_geometry.mesh.n_faces,
                "dynamic_face_count": len(full_new_geometry.dynamic_faces),
                "static_face_count": len(full_new_geometry.static_faces),
                "candidate_rays": int(np.count_nonzero(candidate_mask)),
                "candidate_fraction": float(np.mean(candidate_mask)),
                "active_rays": int(np.count_nonzero(active)),
                "active_fraction": float(np.mean(active)),
                "false_negative_rays": int(
                    np.count_nonzero(active & ~candidate_mask)
                ),
                "target_mismatch_rays": int(
                    np.count_nonzero(
                        route_new_trace.targets != full_new_trace.targets
                    )
                ),
                "tau_max_abs_error": _finite_tau_error(
                    route_new_trace.taus,
                    full_new_trace.taus,
                ),
                "incidence_max_abs_error": float(
                    np.max(
                        np.abs(
                            route_new_trace.incidence
                            - full_new_trace.incidence
                        )
                    )
                ),
                "flux_rel_error": _relative_error(route_flux, full_flux),
                "profile_rel_error": _relative_error(
                    route_next.z + config.trench_depth,
                    full_next.z + config.trench_depth,
                ),
                "max_profile_height": float(
                    np.max(profile_height)
                ),
                "integrated_profile_volume": deposited_volume,
                "deposition_centroid_x": deposition_centroid_x,
                "deposition_centroid_y": deposition_centroid_y,
                "total_captured_flux": total_flux,
                "profile_capture_fraction": profile_flux
                / max(total_flux, 1e-15),
                "sidewall_capture_fraction": wall_flux
                / max(total_flux, 1e-15),
                "bottom_to_mask_step_coverage": (
                    profile_flux / max(profile_area, 1e-15)
                )
                / max(
                    mask_flux / max(mask_area, 1e-15),
                    1e-15,
                ),
                "refit_target_mismatch_rays": refit_target_mismatch,
                "refit_tau_max_abs_error": _finite_tau_error(
                    refit_new_trace.taus,
                    full_new_trace.taus,
                ),
                "refit_incidence_max_abs_error": float(
                    np.max(
                        np.abs(
                            refit_new_trace.incidence
                            - full_new_trace.incidence
                        )
                    )
                ),
                "support_x_half_width": float(support.maximum[0]),
                "support_y_half_width": float(support.maximum[1]),
                "candidate_discovery_wall_time_s": discovery_wall_s,
                "cache_build_wall_time_s": cache_wall_s,
                "refit_bvh_initial_build_wall_time_s": refit_bvh_build_wall_s,
                "refit_initial_trace_wall_time_s": refit_initial_trace_wall_s,
                "static_bvh_build_wall_time_s": static_bvh_build_wall_s,
                "full_profile_update_wall_time_s": full_profile_update_wall_s,
                "candidate_profile_update_wall_time_s": route_profile_update_wall_s,
                "full_trace_wall_time_s": full_wall_s,
                "refit_wall_time_s": refit_wall_s,
                "refit_trace_wall_time_s": refit_trace_wall_s,
                "candidate_trace_wall_time_s": route_wall_s,
                "full_step_wall_time_s": full_step_wall_s,
                "full_refit_step_wall_time_s": refit_step_wall_s,
                "candidate_step_wall_time_s": route_step_wall_s,
                "step_wall_speedup_vs_full": full_step_wall_s
                / max(route_step_wall_s, 1e-15),
                "candidate_step_wall_speedup_vs_full_refit": refit_step_wall_s
                / max(route_step_wall_s, 1e-15),
                "full_refit_speedup_vs_full_rebuild": full_step_wall_s
                / max(refit_step_wall_s, 1e-15),
                "full_cumulative_wall_time_s": full_cumulative,
                "full_refit_cumulative_wall_time_s": refit_cumulative,
                "candidate_cumulative_wall_time_s": route_cumulative,
                "cumulative_wall_speedup_vs_full": full_cumulative
                / max(route_cumulative, 1e-15),
                "candidate_cumulative_wall_speedup_vs_full_refit": refit_cumulative
                / max(route_cumulative, 1e-15),
                "full_triangle_tests": full_new_trace.triangle_tests,
                "candidate_triangle_tests": route_new_trace.triangle_tests,
                "full_bbox_tests": full_new_trace.bbox_tests,
                "candidate_bbox_tests": route_new_trace.bbox_tests,
                "independent_reference_checked": independent_checked,
                "independent_reference_rays": independent_reference_rays,
                "independent_target_mismatch_rays": independent_target_mismatch,
                "independent_tau_max_abs_error": independent_tau_error,
                "independent_incidence_max_abs_error": independent_incidence_error,
                "independent_reference_wall_time_s": independent_wall_s,
                "support_containment_pass": True,
            }
        )

        route_profile = route_next
        full_profile = full_next
        route_geometry = route_new_geometry
        full_geometry = full_new_geometry
        route_trace = route_new_trace
        full_trace = full_new_trace

    return PVD3DResult(
        config=config,
        support=support,
        candidate_mask=candidate_mask,
        rows=tuple(rows),
        route_profile=route_profile,
        full_profile=full_profile,
    )
