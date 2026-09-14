"""Confined-change mesh transport experiments.

The immutable envelope bounds admissible geometry changes; it is NOT a surface.
Certificates concern the supplied finite ray states, never arbitrary lighting.
All routes use the same compiled intersection and refit kernels. The hierarchy
may defer derived geometry, but always retains authoritative current triangles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from numba import njit

from .accelerated3d import _ray_triangle_tau
from .bvh3d import BVH3D
from .geometry3d import TriangleMesh3D


@njit(cache=True)
def box_entry(o, d, lo, hi, limit):
    near, far = 0.0, limit
    for k in range(3):
        if d[k] == 0.0:
            if o[k] < lo[k] or o[k] > hi[k]:
                return np.inf
        else:
            a, b = (lo[k]-o[k])/d[k], (hi[k]-o[k])/d[k]
            if a > b:
                a, b = b, a
            near, far = max(near, a), min(far, b)
            if near > far:
                return np.inf
    return near if far >= 0 else np.inf


@njit(cache=True)
def certificate_mask(origins, directions, lo, hi, limits):
    result = np.zeros(len(origins), np.bool_)
    for i in range(len(origins)):
        # Inclusive endpoint preserves potential equal-distance competitors.
        result[i] = np.isfinite(box_entry(origins[i], directions[i], lo, hi, limits[i]))
    return result


@njit(cache=True)
def refit_arrays(triangles, ids, start, count, left, right, lo, hi):
    for node in range(len(start)-1, -1, -1):
        if count[node]:
            for k in range(3):
                a, b = np.inf, -np.inf
                for j in range(start[node], start[node]+count[node]):
                    for v in range(3):
                        x = triangles[ids[j], v, k]
                        a, b = min(a, x), max(b, x)
                lo[node, k], hi[node, k] = a, b
        else:
            for k in range(3):
                lo[node, k] = min(lo[left[node], k], lo[right[node], k])
                hi[node, k] = max(hi[left[node], k], hi[right[node], k])


@njit(cache=True)
def trace_batch(origins, directions, limits, triangles, ids, start, count, end, lo, hi):
    hits = np.full(len(origins), -1, np.int64)
    distances = limits.copy()
    boxes, tests = 0, 0
    for i in range(len(origins)):
        node = 0
        while node < len(start):
            boxes += 1
            if not np.isfinite(box_entry(origins[i], directions[i], lo[node], hi[node], distances[i])):
                node = end[node]
                continue
            for j in range(start[node], start[node]+count[node]):
                face = ids[j]
                tests += 1
                tau = _ray_triangle_tau(origins[i], directions[i], triangles[face])
                if np.isfinite(tau) and (tau < distances[i] or (tau == distances[i] and (hits[i] < 0 or face < hits[i]))):
                    distances[i], hits[i] = tau, face
            node += 1
    return hits, distances, boxes, tests


class MeshTree:
    def __init__(self, triangles):
        self.triangles = np.ascontiguousarray(triangles)
        n = len(triangles)
        mesh = TriangleMesh3D(self.triangles.reshape(-1, 3), np.arange(3*n).reshape(n, 3),
                              np.zeros(n, bool), np.zeros(n, np.int64), np.zeros(3*n))
        tree = BVH3D(mesh, leaf_size=4)
        self.ids, self.start, self.count = tree.face_indices, tree.start, tree.count
        self.left, self.right = tree.left, tree.right
        self.lo, self.hi = tree.bbox_min.copy(), tree.bbox_max.copy()
        self.end = np.arange(tree.n_nodes, dtype=np.int64)+1
        for j in range(tree.n_nodes-1, -1, -1):
            if self.right[j] >= 0:
                self.end[j] = self.end[self.right[j]]

    def refit(self, triangles):
        if triangles.shape != self.triangles.shape:
            raise ValueError('Fixed topology/face order required')
        self.triangles = np.ascontiguousarray(triangles)
        refit_arrays(self.triangles, self.ids, self.start, self.count, self.left, self.right, self.lo, self.hi)

    def trace(self, origins, directions, limits):
        return trace_batch(origins, directions, limits, self.triangles, self.ids,
                           self.start, self.count, self.end, self.lo, self.hi)


@dataclass
class Part:
    name: str
    triangles: np.ndarray
    envelope: np.ndarray
    reflectivity: float = 0.0
    epoch: int = 0
    prepared_epoch: int = -1
    tree: MeshTree | None = None
    offset: int = 0

    def __post_init__(self):
        self.triangles = checked_triangles(self.triangles)
        self.envelope = checked_box(self.envelope)
        contains(self.envelope, self.triangles)
        if not 0 <= self.reflectivity <= 1:
            raise ValueError('Reflectivity outside [0,1]')

    def prepare(self, stats):
        if self.epoch == self.prepared_epoch:
            return
        t = perf_counter()
        if self.tree is None:
            self.tree = MeshTree(self.triangles)
            stats['built_faces'] += len(self.triangles)
        else:
            self.tree.refit(self.triangles)
            stats['refit_faces'] += len(self.triangles)
        stats['prepared_parts'] += 1
        stats['prepare_s'] += perf_counter()-t
        self.prepared_epoch = self.epoch


def checked_box(bounds):
    b = np.array(bounds, dtype=float, copy=True)
    if b.shape != (2, 3) or not np.all(np.isfinite(b)) or np.any(b[1] < b[0]):
        raise ValueError('Invalid envelope')
    b.flags.writeable = False
    return b


def checked_triangles(triangles):
    t = np.array(triangles, dtype=float, order='C', copy=True)
    if t.ndim != 3 or t.shape[1:] != (3, 3) or not len(t) or not np.all(np.isfinite(t)):
        raise ValueError('Invalid triangles')
    if np.any(np.linalg.norm(np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]), axis=1) <= 1e-14):
        raise ValueError('Degenerate triangle')
    t.flags.writeable = False
    return t


def contains(bounds, triangles):
    if np.any(triangles < bounds[0]) or np.any(triangles > bounds[1]):
        raise ValueError('Geometry leaves the declared change envelope')


def empty_stats():
    return dict(built_faces=0, refit_faces=0, prepared_parts=0, prepare_s=0.0,
                certificate_tests=0, certificate_s=0.0, dynamic_queries=0,
                box_tests=0, triangle_tests=0, deferred_parts=0, root_candidates=0,
                static_built_faces=0, static_prepare_s=0.0)


@dataclass
class HitState:
    faces: np.ndarray
    distances: np.ndarray
    stats: dict = field(default_factory=empty_stats)
    outgoing: np.ndarray | None = None


class ConfinedScene:
    """Two spatial levels: immutable root E and declared child motion bounds.

    Child ordering is a performance heuristic only. Every possible nearer hit
    is checked, so correctness does not depend on the ordering. Static geometry
    includes exterior receivers and fixed opaque/reflecting surfaces.
    """
    def __init__(self, static, parts, envelope, static_reflectivity=None):
        self.static = checked_triangles(static)
        self.envelope = checked_box(envelope)
        self.parts = parts
        if len({p.name for p in parts}) != len(parts):
            raise ValueError('Duplicate part name')
        offset = len(static)
        for part in parts:
            contains(self.envelope, part.envelope)
            part.offset = offset
            offset += len(part.triangles)
        self.static_tree = None
        self.full_tree = None
        self.full_version = -1
        self.version = 0
        self.static_reflectivity = np.zeros(len(static)) if static_reflectivity is None else np.array(static_reflectivity, copy=True)
        if self.static_reflectivity.shape != (len(static),) or np.any(self.static_reflectivity < 0) or np.any(self.static_reflectivity > 1):
            raise ValueError('Invalid material')
        self._source_cache = None

    def update(self, changes):
        by_name = {p.name: p for p in self.parts}
        prepared = {}
        # Validate the WHOLE update before mutating any part.
        for name, triangles in changes.items():
            if name not in by_name:
                raise ValueError('Only declared dynamic children may change')
            p = by_name[name]
            t = checked_triangles(triangles)
            if t.shape != p.triangles.shape:
                raise ValueError('Fixed topology/face order required')
            contains(p.envelope, t)
            contains(self.envelope, t)
            prepared[name] = t
        for name, triangles in prepared.items():
            p = by_name[name]
            if not np.array_equal(p.triangles, triangles):
                p.triangles = triangles
                p.epoch += 1
                self.version += 1

    def all_triangles(self):
        return np.concatenate([self.static]+[p.triangles for p in self.parts])

    def reflectivities(self):
        return np.concatenate([self.static_reflectivity]+[np.full(len(p.triangles), p.reflectivity) for p in self.parts])

    def _certificate(self, o, d, bounds, limit, stats, repeats):
        t = perf_counter()
        for _ in range(repeats):
            mask = certificate_mask(o, d, bounds[0], bounds[1], limit)
        stats['certificate_tests'] += repeats*len(o)
        stats['certificate_s'] += perf_counter()-t
        return mask

    def query(self, origins, directions, route='certified', certificate_repeats=1, cache_static=False, exclude=None):
        if route not in ('full', 'local', 'certified') or certificate_repeats < 1:
            raise ValueError('Invalid route/certificate cost')
        o, d = np.ascontiguousarray(origins, dtype=float), np.ascontiguousarray(directions, dtype=float)
        if o.shape != d.shape or o.ndim != 2 or o.shape[1] != 3 or not np.all(np.isfinite(o)) or not np.all(np.isfinite(d)) or not np.allclose(np.linalg.norm(d, axis=1), 1, atol=1e-12, rtol=0):
            raise ValueError('Finite origins and unit directions required')
        stats = empty_stats()
        limits = np.full(len(o), np.inf)
        if route == 'full':
            if exclude is not None:
                raise ValueError('Full route cannot exclude an owner')
            if self.full_version != self.version:
                t = perf_counter()
                triangles = self.all_triangles()
                if self.full_tree is None:
                    self.full_tree = MeshTree(triangles)
                    stats['built_faces'] = len(triangles)
                else:
                    self.full_tree.refit(triangles)
                    stats['refit_faces'] = len(triangles)
                stats['prepare_s'] = perf_counter()-t
                self.full_version = self.version
            h, tau, b, n = self.full_tree.trace(o, d, limits)
            stats['box_tests'], stats['triangle_tests'] = b, n
            return HitState(h, tau, stats)
        cached = self._source_cache
        if cache_static and cached is not None and np.array_equal(o, cached[0]) and np.array_equal(d, cached[1]):
            h, tau = cached[2].copy(), cached[3].copy()
        else:
            if self.static_tree is None:
                t = perf_counter()
                self.static_tree = MeshTree(self.static)
                stats['static_built_faces'] = len(self.static)
                stats['static_prepare_s'] = perf_counter()-t
            h, tau, b, n = self.static_tree.trace(o, d, limits)
            stats['box_tests'], stats['triangle_tests'] = b, n
            if cache_static:
                self._source_cache = (o.copy(), d.copy(), h.copy(), tau.copy())
        root = self._certificate(o, d, self.envelope, limits, stats, 1)
        stats['root_candidates'] = int(root.sum())
        for p in self.parts:
            if p.name == exclude:
                continue
            if route == 'local':
                p.prepare(stats)
                active = root
            else:
                active = root & self._certificate(o, d, p.envelope, tau, stats, certificate_repeats)
                if not np.any(active):
                    stats['deferred_parts'] += int(p.prepared_epoch != p.epoch)
                    continue
                p.prepare(stats)
            idx = np.flatnonzero(active)
            if not len(idx):
                continue
            nh, nt, b, n = p.tree.trace(o[idx], d[idx], tau[idx])
            stats['dynamic_queries'] += len(idx)
            stats['box_tests'] += b
            stats['triangle_tests'] += n
            nh = np.where(nh >= 0, nh+p.offset, -1)
            keep = (nh >= 0) & ((nt < tau[idx]) | ((nt == tau[idx]) & ((h[idx] < 0) | (nh < h[idx]))))
            h[idx[keep]], tau[idx[keep]] = nh[keep], nt[keep]
        return HitState(h, tau, stats)


def panel(center, u, v, subdivisions=1):
    """Exact planar tessellation: increasing n changes representation only."""
    c, u, v = np.asarray(center), np.asarray(u), np.asarray(v)
    edges = np.linspace(-1, 1, subdivisions+1)
    triangles = []
    for i in range(subdivisions):
        for j in range(subdivisions):
            a = c+edges[i]*u+edges[j]*v
            b = c+edges[i+1]*u+edges[j]*v
            cc = c+edges[i+1]*u+edges[j+1]*v
            dd = c+edges[i]*u+edges[j+1]*v
            triangles.extend([[a, b, cc], [a, cc, dd]])
    return np.array(triangles)


def source_grid(n=32, width=1.8, seed=731):
    rng = np.random.default_rng(seed)
    xy = (np.indices((n, n)).reshape(2, -1).T+rng.uniform(.15, .85, (n*n, 2)))/n
    origins = np.column_stack(((xy-.5)*2*width, np.full(n*n, 3.0)))
    directions = np.tile([0., 0., -1.], (n*n, 1))
    return origins, directions


def independent_trace(triangles, origins, directions):
    """Independent 3x3 linear solve, outside production intersection code."""
    faces = np.full(len(origins), -1, np.int64)
    taus = np.full(len(origins), np.inf)
    for i, (o, d) in enumerate(zip(origins, directions)):
        for j, tri in enumerate(triangles):
            matrix = np.column_stack((d, tri[0]-tri[1], tri[0]-tri[2]))
            try:
                t, u, v = np.linalg.solve(matrix, tri[0]-o)
            except np.linalg.LinAlgError:
                continue
            if t > 1e-9 and u >= -1e-10 and v >= -1e-10 and u+v <= 1+1e-10 and t < taus[i]:
                faces[i], taus[i] = j, t
    return HitState(faces, taus)


@dataclass
class PathState:
    history: np.ndarray
    points: np.ndarray
    directions: np.ndarray
    absorbed: np.ndarray
    escaped: np.ndarray
    residual: np.ndarray
    envelope_entries: np.ndarray
    stats: dict


def trace_paths(scene, origins, directions, route='certified', cap=16, response_provider=None, certificate_repeats=1):
    """Passive specular transport; each flight is certified, including reentry.

    Absorbed/escaped/residual energy sum to input energy. A cap is not escape.
    Histories are diagnostics, not an assumption about future path membership.
    """
    if cap < 1:
        raise ValueError('Positive cap required')
    points, dirs = origins.copy(), directions.copy()
    n = len(points)
    weights = np.full(n, 1/n)
    escaped, residual = np.zeros(n), np.zeros(n)
    triangles, rho = scene.all_triangles(), scene.reflectivities()
    absorbed = np.zeros(len(triangles))
    history = np.full((n, cap), -1, np.int64)
    entries = np.zeros(n, np.int64)
    active = np.arange(n)
    stats = empty_stats()
    for bounce in range(cap+1):
        if not len(active):
            break
        if response_provider is None:
            state = scene.query(points[active], dirs[active], route, cache_static=bounce == 0, certificate_repeats=certificate_repeats)
        else:
            last = np.full(len(active), -1, np.int64) if bounce == 0 else history[active, bounce-1]
            state = response_provider.query(scene, points[active], dirs[active], last)
        for key, value in state.stats.items():
            stats[key] = stats.get(key, 0)+value
        outside = np.any((points[active] < scene.envelope[0]) | (points[active] > scene.envelope[1]), axis=1)
        enters = certificate_mask(points[active], dirs[active], *scene.envelope, state.distances)
        entries[active] += outside & enters
        miss = state.faces < 0
        escaped[active[miss]] = weights[active[miss]]
        hit = active[~miss]
        faces = state.faces[~miss]
        if bounce == cap:
            residual[hit] = weights[hit]
            break
        history[hit, bounce] = faces
        distances = state.distances[~miss]
        points[hit] += distances[:, None]*dirs[hit]
        np.add.at(absorbed, faces, weights[hit]*(1-rho[faces]))
        weights[hit] *= rho[faces]
        cached = np.zeros(len(hit), bool) if state.outgoing is None else np.all(np.isfinite(state.outgoing[~miss]), axis=1)
        direct_hit, direct_faces = hit[~cached], faces[~cached]
        normal = np.cross(triangles[direct_faces, 1]-triangles[direct_faces, 0], triangles[direct_faces, 2]-triangles[direct_faces, 0])
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        dirs[direct_hit] -= 2*np.sum(dirs[direct_hit]*normal, axis=1)[:, None]*normal
        if np.any(cached):
            dirs[hit[cached]] = state.outgoing[~miss][cached]
        dirs[hit] /= np.linalg.norm(dirs[hit], axis=1)[:, None]
        active = hit[weights[hit] > 0]
    return PathState(history, points, dirs, absorbed, escaped, residual, entries, stats)
