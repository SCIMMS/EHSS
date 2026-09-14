"""Stable physical blocks, rigid poses and exact coordinate residuals.

Physical ownership is independent of the clipped transport partition. A small
fit residual permits sharing an intrinsic block, never snapping physical atoms:
residuals are applied before candidate construction and collision queries. The
world boundary response compiler still checks every geometric dependency.
"""
from dataclasses import dataclass
from copy import copy
import time
import numpy as np
from .inside_out_pa import Spheres
from .ehss_region_versions import RegionVersionCompiler, hash_arrays


def frozen(array, dtype=None):
    value = np.array(array, dtype=dtype, copy=True, order='C')
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class IntrinsicBlock:
    atoms: np.ndarray
    centers: np.ndarray
    radii: np.ndarray
    version: str


@dataclass(frozen=True)
class PosedBlock:
    intrinsic: IntrinsicBlock
    rotation: np.ndarray
    translation: np.ndarray
    residual_rows: np.ndarray
    residuals: np.ndarray
    override_rows: np.ndarray
    overrides: np.ndarray

    def centers(self):
        result = self.intrinsic.centers @ self.rotation.T + self.translation
        result[self.residual_rows] += self.residuals
        # A final sparse absolute override covers subtraction/addition rounding.
        result[self.override_rows] = self.overrides
        return result


@dataclass(frozen=True)
class GeometryFrame:
    atom_ids: tuple
    groups: np.ndarray
    blocks: tuple
    receipt: dict

    def spheres(self):
        centers = np.empty((len(self.atom_ids), 3))
        radii = np.empty(len(self.atom_ids))
        for block in self.blocks:
            centers[block.intrinsic.atoms] = block.centers()
            radii[block.intrinsic.atoms] = block.intrinsic.radii
        return Spheres(centers, radii, deduplicate=False)


def _posed(intrinsic, rotation, translation, target):
    prediction = intrinsic.centers @ rotation.T + translation
    delta = target-prediction
    rows = np.flatnonzero(np.any(delta != 0., axis=1))
    corrected = prediction.copy(); corrected[rows] += delta[rows]
    overrides = np.flatnonzero(np.any(corrected != target, axis=1))
    result = PosedBlock(intrinsic, frozen(rotation), frozen(translation),
        frozen(rows), frozen(delta[rows]), frozen(overrides), frozen(target[overrides]))
    np.testing.assert_array_equal(result.centers(), target)
    return result


def _new_block(atoms, target, radii):
    local = frozen(target-target[0])
    intrinsic = IntrinsicBlock(frozen(atoms), local, frozen(radii), hash_arrays(atoms, local, radii))
    return _posed(intrinsic, np.eye(3), target[0], target)


def initial_geometry(spheres, groups, atom_ids=None):
    identities = tuple(range(len(spheres.radii))) if atom_ids is None else tuple(atom_ids)
    group_array = np.asarray(groups)
    if len(identities) != len(spheres.radii) or len(set(identities)) != len(identities):
        raise ValueError('Unique stable identities required in physical atom order')
    if group_array.shape != spheres.radii.shape or not np.issubdtype(group_array.dtype, np.integer):
        raise ValueError('One integer physical block ID per atom required')
    unique = np.unique(group_array)
    if not len(unique) or not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError('Contiguous nonempty physical block IDs required')
    blocks = []
    for group in unique:
        atoms = np.flatnonzero(group_array == group)
        blocks.append(_new_block(atoms, spheres.centers[atoms], spheres.radii[atoms]))
    return GeometryFrame(identities, frozen(group_array), tuple(blocks), dict(initial=True, blocks=len(blocks)))


def advance_geometry(previous, spheres, atom_ids=None, rigid_tolerance=1e-10):
    """Pure/transactional update; tolerance controls cache identity, not geometry error.

    Kabsch fitting uses proper rotations, including rank-deficient point/bond
    blocks. Such blocks have nonunique poses; no unique orientation is claimed.
    Each fit is against the retained intrinsic geometry, avoiding incremental
    drift. Above-threshold strain or changed radii rebuild that block only.
    """
    start = time.perf_counter()
    identities = previous.atom_ids if atom_ids is None else tuple(atom_ids)
    if identities != previous.atom_ids or len(spheres.radii) != len(identities):
        raise ValueError('Physical atom identity and ordering must remain fixed')
    if not np.isfinite(rigid_tolerance) or rigid_tolerance < 0:
        raise ValueError('Finite nonnegative rigid fit tolerance required')
    blocks = []; unchanged = []; pose_only = []; rebuilt = []; changed_radii = []
    fit_residuals = []; fit_s = 0.; rebuild_s = 0.
    for group, old in enumerate(previous.blocks):
        atoms = old.intrinsic.atoms; target = spheres.centers[atoms]; radii = spheres.radii[atoms]
        same_radii = np.array_equal(radii, old.intrinsic.radii)
        if same_radii and np.array_equal(target, old.centers()):
            blocks.append(old); unchanged.append(group); fit_residuals.append(0.); continue
        tick = time.perf_counter()
        local = old.intrinsic.centers
        local_mean = local.mean(axis=0); target_mean = target.mean(axis=0)
        u, _, vh = np.linalg.svd((local-local_mean).T @ (target-target_mean))
        correction = np.eye(3); correction[2, 2] = 1. if np.linalg.det(u@vh) >= 0. else -1.
        rotation = (u@correction@vh).T
        translation = target_mean-local_mean@rotation.T
        residual = float(np.linalg.norm(local@rotation.T+translation-target, axis=1).max())
        fit_s += time.perf_counter()-tick; fit_residuals.append(residual)
        if same_radii and residual <= rigid_tolerance:
            blocks.append(_posed(old.intrinsic, rotation, translation, target)); pose_only.append(group)
        else:
            tick = time.perf_counter(); blocks.append(_new_block(atoms, target, radii))
            rebuild_s += time.perf_counter()-tick; rebuilt.append(group)
            if not same_radii: changed_radii.append(group)
    receipt = dict(blocks=len(blocks), unchanged_blocks=unchanged, pose_only_blocks=pose_only,
        rebuilt_intrinsic_blocks=rebuilt, radius_changed_blocks=changed_radii,
        reused_intrinsic_blocks=len(unchanged)+len(pose_only), rigid_tolerance=float(rigid_tolerance),
        fit_residuals=fit_residuals, max_fit_residual=max(fit_residuals, default=0.),
        residual_atoms=sum(len(b.residual_rows) for b in blocks),
        absolute_override_atoms=sum(len(b.override_rows) for b in blocks),
        fit_s=fit_s, intrinsic_rebuild_s=rebuild_s, total_s=time.perf_counter()-start,
        physical_coordinate_approximation=False, response_reuse_from_intrinsic_identity=False,
        identity_scan_is_global=True)
    return GeometryFrame(identities, previous.groups, tuple(blocks), receipt)


@dataclass(frozen=True)
class IntrinsicRegionFrame:
    geometry: GeometryFrame
    region: object
    receipt: dict

    @property
    def model(self):
        return self.region.model


class IntrinsicRegionCompiler:
    """Materialize physical blocks into clipped response dependencies atomically."""
    def __init__(self, seed, groups, atom_ids=None, rigid_tolerance=1e-10, max_cache_entries=256):
        if not np.isfinite(rigid_tolerance) or rigid_tolerance < 0:
            raise ValueError('Finite nonnegative rigid fit tolerance required')
        self.initial = initial_geometry(seed.spheres, groups, atom_ids)
        self.regions = RegionVersionCompiler(seed, self.initial.atom_ids, max_cache_entries)
        self.rigid_tolerance = rigid_tolerance
        self.current = None

    def compile(self, spheres, atom_ids=None, normal=None, edges=None, reuse=True):
        start = time.perf_counter()
        previous = self.initial if self.current is None else self.current.geometry
        geometry = advance_geometry(previous, spheres, atom_ids, self.rigid_tolerance)
        tick = time.perf_counter(); materialized = geometry.spheres()
        np.testing.assert_array_equal(materialized.centers, spheres.centers)
        np.testing.assert_array_equal(materialized.radii, spheres.radii)
        materialization_s = time.perf_counter()-tick
        # RegionVersionCompiler publishes a fresh cache only on success. A
        # shallow clone isolates that publication until the outer frame exists.
        candidate = copy(self.regions)
        region = candidate.compile(materialized, geometry.atom_ids, normal, edges, reuse)
        receipt = dict(geometry=geometry.receipt, materialization_s=materialization_s,
            regions=region.receipt, total_s=time.perf_counter()-start,
            physical_coordinates_identical=True, intrinsic_and_world_response_versions_separate=True)
        result = IntrinsicRegionFrame(geometry, region, receipt)
        if reuse: self.regions = candidate; self.current = result
        return result
