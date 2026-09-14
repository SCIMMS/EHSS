"""Parallel-ray packets with conservative box/cylinder support masks.

All methods test the original world-space atomic spheres after broad phase.
The cylinder axis is the exact packet direction; no direction binning occurs.
Caps are interval limits, never physical scatterers. This module handles one
free flight (any hit or nearest hit), not a multiple-scattering response.
"""
from dataclasses import dataclass
import time
import numpy as np
from numba import njit
from .inside_out_pa import tangent_frames
from .ehss_reference import entry_distance
from .ehss_support_tree import box_interval
from .ehss_guarded_intrinsic_response import nearest_world


@dataclass
class ParallelPacket:
    uv: np.ndarray
    z: np.ndarray
    origins: np.ndarray
    direction: np.ndarray
    grid: object = None


class DirectionalSupports:
    def __init__(self, spheres, groups, direction):
        tick = time.perf_counter()
        direction = np.asarray(direction, dtype=float)
        groups = np.asarray(groups)
        if direction.shape != (3,) or not np.isfinite(direction).all() or not np.isclose(np.linalg.norm(direction), 1., atol=1e-13, rtol=1e-13):
            raise ValueError('An exact unit packet direction is required')
        if groups.shape != spheres.radii.shape or not np.issubdtype(groups.dtype, np.integer):
            raise ValueError('One integer owner per atom required')
        unique = np.unique(groups)
        if not np.array_equal(unique, np.arange(len(unique))):
            raise ValueError('Contiguous nonempty support IDs required')
        self.spheres, self.groups = spheres, groups.copy()
        self.direction = np.ascontiguousarray(direction)
        self.center = spheres.centers.mean(axis=0)
        a, b = tangent_frames(self.direction[None])
        self.basis = np.column_stack((a[0], b[0], self.direction))
        self.ids = np.argsort(groups, kind='stable').astype(np.int64)
        self.offsets = np.r_[0, np.cumsum(np.bincount(groups))].astype(np.int64)
        self.pad = 1e-10*max(1., float(np.abs(spheres.centers).max()), float(spheres.radii.max()))
        start = time.perf_counter()
        projected = (spheres.centers-self.center)@self.basis
        self.projection_s = time.perf_counter()-start
        self.world_lo, self.world_hi, self.ray_lo, self.ray_hi, self.circle = [], [], [], [], []
        phase = dict(world_aabb_s=0., ray_box_s=0., cylinder_s=0.)
        for lo, hi in zip(self.offsets[:-1], self.offsets[1:]):
            atoms = self.ids[lo:hi]
            radii, c, q = spheres.radii[atoms], spheres.centers[atoms], projected[atoms]
            start = time.perf_counter()
            self.world_lo.append(np.min(c-radii[:, None], axis=0)-self.pad)
            self.world_hi.append(np.max(c+radii[:, None], axis=0)+self.pad)
            phase['world_aabb_s'] += time.perf_counter()-start
            start = time.perf_counter()
            lower = np.min(q-radii[:, None], axis=0)-self.pad
            upper = np.max(q+radii[:, None], axis=0)+self.pad
            self.ray_lo.append(lower)
            self.ray_hi.append(upper)
            phase['ray_box_s'] += time.perf_counter()-start
            start = time.perf_counter()
            center = q[:, :2].mean(axis=0)
            radius = np.max(np.linalg.norm(q[:, :2]-center, axis=1)+radii)+self.pad
            self.circle.append([center[0], center[1], radius, lower[2], upper[2]])
            phase['cylinder_s'] += time.perf_counter()-start
        for name in ['world_lo', 'world_hi', 'ray_lo', 'ray_hi', 'circle']:
            setattr(self, name, np.asarray(getattr(self, name)))
        self.receipt = dict(prepare_s=time.perf_counter()-tick, projection_s=self.projection_s,
            **phase, groups=len(unique), spheres=len(groups), minimal_enclosing_circle=False,
            full_atomic_radius_enclosure=True, axis_aligned_to_packet=True)

    def packet(self, uv, z):
        uv = np.asarray(uv, dtype=float)
        if uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all():
            raise ValueError('Finite (N,2) transverse samples required')
        z = np.broadcast_to(np.asarray(z, dtype=float), (len(uv),)).copy()
        if not np.isfinite(z).all(): raise ValueError('Finite axial origins required')
        origins = self.center+uv@self.basis[:, :2].T+z[:, None]*self.direction
        return ParallelPacket(uv, z, np.ascontiguousarray(origins), self.direction)

    def candidates(self, packet, kind='cylinder', limit=np.inf):
        if kind not in ['cylinder', 'ray_box', 'world_aabb', 'cylinder_grid', 'ray_box_grid']:
            raise ValueError('Unknown support mask')
        if limit < 0 or np.isnan(limit): raise ValueError('Nonnegative ray limit required')
        if not np.array_equal(packet.direction, self.direction): raise ValueError('Different packet direction')
        if kind.endswith('_grid'):
            if packet.grid is None: raise ValueError('A rectangular packet grid is required')
            xs, ys = packet.grid
            return grid_candidates(xs, ys, packet.z[0], self.ray_lo, self.ray_hi, self.circle,
                kind == 'cylinder_grid', limit, self.pad)
        u, v, z = packet.uv[:, 0][None], packet.uv[:, 1][None], packet.z[None]
        if kind == 'cylinder':
            c = self.circle
            mask = (u-c[:, 0, None])**2+(v-c[:, 1, None])**2 <= c[:, 2, None]**2
            mask &= (c[:, 4, None]-z >= 0.) & (c[:, 3, None]-z <= limit)
        elif kind == 'ray_box':
            lo, hi = self.ray_lo, self.ray_hi
            mask = (u >= lo[:, 0, None]) & (u <= hi[:, 0, None])
            mask &= (v >= lo[:, 1, None]) & (v <= hi[:, 1, None])
            mask &= (z <= hi[:, 2, None]) & (lo[:, 2, None]-z <= limit)
        else:
            near = np.zeros((len(self.circle), len(packet.z)))
            far = np.full_like(near, limit)
            mask = np.ones(near.shape, dtype=bool)
            for k in range(3):
                p = packet.origins[:, k][None]
                d = self.direction[k]
                if d == 0.:
                    mask &= (p >= self.world_lo[:, k, None]) & (p <= self.world_hi[:, k, None])
                else:
                    a = (self.world_lo[:, k, None]-p)/d
                    b = (self.world_hi[:, k, None]-p)/d
                    near = np.maximum(near, np.minimum(a, b))
                    far = np.minimum(far, np.maximum(a, b))
            mask &= far >= near
        return np.ascontiguousarray(mask)

    def grid_packet(self, xs, ys, z):
        xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
        if xs.ndim != 1 or ys.ndim != 1 or not len(xs) or not len(ys) or np.any(np.diff(xs) <= 0) or np.any(np.diff(ys) <= 0):
            raise ValueError('Nonempty increasing grid axes required')
        if np.ndim(z) != 0: raise ValueError('One common upstream plane required')
        u, v = np.meshgrid(xs, ys)
        packet = self.packet(np.column_stack((u.ravel(), v.ravel())), z)
        packet.grid = (xs.copy(), ys.copy())
        return packet


@njit(cache=True)
def grid_candidates(xs, ys, z, lower, upper, circles, cylinder, limit, pad):
    result = np.zeros((len(circles), len(xs)*len(ys)), dtype=np.bool_)
    for g in range(len(circles)):
        if upper[g, 2] < z or lower[g, 2]-z > limit: continue
        for row in range(len(ys)):
            if cylinder:
                remain = circles[g, 2]**2-(ys[row]-circles[g, 1])**2
                if remain < 0.: continue
                width = np.sqrt(remain)
                lo, hi = circles[g, 0]-width-pad, circles[g, 0]+width+pad
            else:
                if ys[row] < lower[g, 1] or ys[row] > upper[g, 1]: continue
                lo, hi = lower[g, 0], upper[g, 0]
            first, stop = np.searchsorted(xs, lo, side='left'), np.searchsorted(xs, hi, side='right')
            result[g, row*len(xs)+first:row*len(xs)+stop] = True
    return result


@njit(cache=True)
def query_candidates(origins, direction, centers, radii, offsets, ids, mask, any_hit, limit):
    hit = np.full(len(origins), -1, dtype=np.int64)
    distances = np.full(len(origins), np.inf)
    tests = groups_visited = 0
    for ray in range(len(origins)):
        done = False
        for g in range(len(offsets)-1):
            if not mask[g, ray]: continue
            groups_visited += 1
            for j in range(offsets[g], offsets[g+1]):
                atom = ids[j]
                tests += 1
                distance = entry_distance(origins[ray], direction, centers[atom], radii[atom])
                if np.isfinite(distance) and distance <= limit and (distance < distances[ray] or (distance == distances[ray] and atom < hit[ray])):
                    hit[ray], distances[ray] = atom, distance
                    if any_hit:
                        done = True
                        break
            if done: break
    return hit, distances, tests, groups_visited


@njit(cache=False)
def any_world(node, origin, direction, limit, lo, hi, left, right, leaf_group, offsets, ids, centers, radii, checks):
    checks[0] += 1
    near, far = box_interval(origin, direction, lo[node], hi[node])
    if far < 0. or near > limit: return -1, np.inf
    group = leaf_group[node]
    if group >= 0:
        for j in range(offsets[group], offsets[group+1]):
            atom = ids[j]
            checks[1] += 1
            distance = entry_distance(origin, direction, centers[atom], radii[atom])
            if np.isfinite(distance) and distance <= limit: return atom, distance
        return -1, np.inf
    atom, distance = any_world(left[node], origin, direction, limit, lo, hi, left, right, leaf_group, offsets, ids, centers, radii, checks)
    if atom >= 0: return atom, distance
    return any_world(right[node], origin, direction, limit, lo, hi, left, right, leaf_group, offsets, ids, centers, radii, checks)


@njit(cache=False)
def query_bvh(origins, direction, centers, radii, lo, hi, left, right, end, leaf_group, offsets, ids, any_hit, limit):
    hits = np.full(len(origins), -1, dtype=np.int64)
    distances = np.full(len(origins), np.inf)
    checks = np.zeros(2, dtype=np.int64)
    for ray in range(len(origins)):
        if any_hit:
            hits[ray], distances[ray] = any_world(0, origins[ray], direction, limit, lo, hi, left, right, leaf_group, offsets, ids, centers, radii, checks)
        else:
            hits[ray], distances[ray] = nearest_world(0, origins[ray], direction, -1, limit, -1,
                lo, hi, left, right, end, leaf_group, offsets, ids, centers, radii, checks)
    return hits, distances, checks[1], checks[0]
