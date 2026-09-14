"""Physical owned/foreign search independent of the BVH leaf partition.

Ownership is immutable metadata. Bounds and atom coordinates come from the
current bound model. This supplies T/G arbitration, not response reuse itself.
"""
from dataclasses import dataclass
import time
import numpy as np
from numba import njit
from .ehss_support_tree import box_interval
from .ehss_reference import entry_distance
from .ehss_guarded_intrinsic_response import reflect
from .ehss_reuse_frames import partition
from .ehss_sparse_path_response import readonly
from .ehss_state import EHSSResult


@njit(cache=False)
def nearest_owner(node, origin, direction, last, limit, owner, mode,
                  lo, hi, left, right, leaf_group, offsets, ids,
                  owners, uniform, centers, radii, counters):
    # mode: 0 all, 1 only this owner, -1 complement of this owner.
    known = uniform[node]
    if known >= 0 and ((mode == 1 and known != owner) or (mode == -1 and known == owner)):
        counters[2] += 1
        return -1, np.inf
    counters[0] += 1
    near, far = box_interval(origin, direction, lo[node], hi[node])
    if near > limit or far < 0.:
        return -1, np.inf
    atom = -1
    best = limit
    group = leaf_group[node]
    if group >= 0:
        for j in range(offsets[group], offsets[group+1]):
            candidate = ids[j]
            if candidate == last:
                continue
            if mode == 1 and owners[candidate] != owner:
                continue
            if mode == -1 and owners[candidate] == owner:
                continue
            counters[1] += 1
            distance = entry_distance(origin, direction, centers[candidate], radii[candidate])
            if np.isfinite(distance) and (distance < best or (distance == best and (atom < 0 or candidate < atom))):
                atom = candidate
                best = distance
    else:
        atom, distance = nearest_owner(left[node], origin, direction, last, best, owner, mode,
            lo, hi, left, right, leaf_group, offsets, ids, owners, uniform, centers, radii, counters)
        if atom >= 0:
            best = distance
        other, distance = nearest_owner(right[node], origin, direction, last, best, owner, mode,
            lo, hi, left, right, leaf_group, offsets, ids, owners, uniform, centers, radii, counters)
        if other >= 0 and (atom < 0 or distance < best or (distance == best and other < atom)):
            atom = other
            best = distance
    return atom, best if atom >= 0 else np.inf


@njit(cache=False)
def query_owners(origins, directions, lasts, limits, selected, mode,
                 lo, hi, left, right, leaf_group, offsets, ids, owners, uniform, centers, radii):
    atoms = np.full(len(origins), -1, dtype=np.int64)
    distances = np.full(len(origins), np.inf)
    counters = np.zeros(3, dtype=np.int64)
    for i in range(len(origins)):
        atoms[i], distances[i] = nearest_owner(0, origins[i], directions[i], lasts[i], limits[i], selected[i], mode,
            lo, hi, left, right, leaf_group, offsets, ids, owners, uniform, centers, radii, counters)
    return atoms, distances, counters


@njit(cache=False)
def trace_owner_split(origins, directions, lasts, initial, cap,
                      lo, hi, left, right, leaf_group, offsets, ids, owners, uniform, centers, radii):
    points = origins.copy(); outgoing = directions.copy(); counts = initial.copy()
    escaped = np.zeros(len(points), dtype=np.bool_); unresolved = escaped.copy()
    history = np.full((len(points), cap), -1, dtype=np.int32)
    counters = np.zeros(3, dtype=np.int64); splits = 0
    for i in range(len(points)):
        last = lasts[i]
        if counts[i] == 1:
            history[i, 0] = last
        while True:
            if last < 0:
                atom, distance = nearest_owner(0, points[i], outgoing[i], last, np.inf, -1, 0,
                    lo, hi, left, right, leaf_group, offsets, ids, owners, uniform, centers, radii, counters)
            else:
                owner = owners[last]; splits += 1
                atom, distance = nearest_owner(0, points[i], outgoing[i], last, np.inf, owner, 1,
                    lo, hi, left, right, leaf_group, offsets, ids, owners, uniform, centers, radii, counters)
                foreign, before = nearest_owner(0, points[i], outgoing[i], last, distance, owner, -1,
                    lo, hi, left, right, leaf_group, offsets, ids, owners, uniform, centers, radii, counters)
                if foreign >= 0 and (atom < 0 or before < distance or (before == distance and foreign < atom)):
                    atom = foreign; distance = before
            if atom < 0:
                escaped[i] = True; break
            if counts[i] >= cap:
                unresolved[i] = True; break
            reflect(points[i], outgoing[i], atom, distance, centers, radii)
            history[i, counts[i]] = atom; counts[i] += 1; last = atom
    return points, outgoing, escaped, unresolved, counts, history, counters, splits


class OwnerSearchLayout:
    """Reusable owner metadata for a fixed topology and stable atom identities."""
    topology_names = ('left', 'right', 'leaf_group', 'offsets', 'ids')

    def __init__(self, model, owners):
        self.owners, self.atoms = partition(owners, len(model.spheres.radii), 'Response ownership')
        self.atom_ids = tuple(model.geometry.atom_ids)
        tree = model.tree
        self.topology = {name: readonly(np.array(getattr(tree, name), copy=True)) for name in self.topology_names}
        uniform = np.full(len(tree.left), -1, dtype=np.int64)
        for node in range(len(uniform)-1, -1, -1):
            group = tree.leaf_group[node]
            if group >= 0:
                labels = self.owners[tree.ids[tree.offsets[group]:tree.offsets[group+1]]]
                if np.all(labels == labels[0]): uniform[node] = labels[0]
            elif uniform[tree.left[node]] >= 0 and uniform[tree.left[node]] == uniform[tree.right[node]]:
                uniform[node] = uniform[tree.left[node]]
        self.uniform = readonly(uniform)

    def bind(self, model):
        if tuple(model.geometry.atom_ids) != self.atom_ids:
            raise ValueError('Stable atom identities and order required')
        for name, values in self.topology.items():
            if not np.array_equal(getattr(model.tree, name), values):
                raise ValueError('Search topology changed; build a new owner layout')
        return BoundOwnerSearch(self, model)


@dataclass(frozen=True)
class BoundOwnerSearch:
    layout: OwnerSearchLayout
    model: object

    def arguments(self):
        t = self.model.tree; s = self.model.spheres
        return (t.lo, t.hi, t.left, t.right, t.leaf_group, t.offsets, t.ids,
                self.layout.owners, self.layout.uniform, s.centers, s.radii)

    def query(self, origins, directions, owner=0, mode='all', last=-1, limit=np.inf):
        modes = {'all': 0, 'owned': 1, 'foreign': -1}
        if mode not in modes: raise ValueError('Mode must be all, owned, or foreign')
        p = np.ascontiguousarray(origins, dtype=float); d = np.ascontiguousarray(directions, dtype=float)
        if p.ndim != 2 or p.shape[1:] != (3,) or d.shape != p.shape or not np.isfinite(p).all() or not np.isfinite(d).all():
            raise ValueError('Finite (N,3) origins and directions required')
        norms = np.linalg.norm(d, axis=1)
        if np.any((norms == 0) | ~np.isfinite(norms)): raise ValueError('Finite nonzero direction norms required')
        def integers(value, name, low, high):
            array = np.asarray(value)
            if not np.issubdtype(array.dtype, np.integer): raise ValueError(name+' must be integer')
            array = np.broadcast_to(array, (len(p),))
            if np.any((array < low) | (array >= high)): raise ValueError('Invalid '+name)
            return np.array(array, dtype=np.int64, copy=True)
        lasts = integers(last, 'last atom', -1, len(self.layout.owners))
        selected = integers(owner, 'owner', 0, len(self.layout.atoms))
        limits = np.array(np.broadcast_to(np.asarray(limit, dtype=float), (len(p),)), copy=True)
        if np.any(np.isnan(limits) | (limits < 0)): raise ValueError('Nonnegative distance limits required')
        atoms, distances, counters = query_owners(p, d, lasts, limits, selected, modes[mode], *self.arguments())
        return dict(atoms=atoms, distances=distances, box_tests=int(counters[0]),
                    sphere_tests=int(counters[1]), owner_prunes=int(counters[2]))

    def trace(self, source, cap=64):
        if type(cap) is not int or cap < 1: raise ValueError('Positive collision cap required')
        if np.any(source.initial_bounces > 1) or np.any(source.initial_bounces > cap):
            raise ValueError('Uncollided or first-reflected inputs required')
        if np.any(source.last_atom < -1) or np.any(source.last_atom >= len(self.layout.owners)):
            raise ValueError('Invalid initial atom')
        tick = time.perf_counter()
        result = trace_owner_split(source.origins, source.directions, source.last_atom, source.initial_bounces, cap, *self.arguments())
        p,d,escaped,unresolved,counts,history,counters,splits = result
        return EHSSResult(p,d,escaped,unresolved,counts,history,source,
            dict(query_s=time.perf_counter()-tick,box_tests=int(counters[0]),sphere_tests=int(counters[1]),
                 owner_prunes=int(counters[2]),owned_foreign_splits=int(splits),response_reuse=False))
