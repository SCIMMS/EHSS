"""Fused native EHSS on the existing node-aware physical BVH.

No angular approximation or response merging. Optional union-preserving atom
filtering changes only leaf search lists, keeping the complete hierarchy and
physical atom identities. Light mode retains count/tail diagnostics without a
cap-sized collision history. Bind a new model after geometry changes.
"""
import time
import numpy as np
from numba import njit
from .ehss_guarded_intrinsic_response import nearest_world, reflect
from .ehss_support_tree import box_interval
from .ehss_reference import entry_distance
from .ehss_active_atoms import active_atoms
from .ehss_state import EHSSResult


@njit(cache=False)
def _nearest_stackless(o, d, last, lo, hi, end, leaf_group, offsets, ids, world, radii, counters):
    """Preorder traversal using existing subtree ends; keep the global best.

The recursive reference may return infinity from a child with no closer hit,
loosening a previous ancestor limit. A persistent best avoids that extra work.
The physical intersection formula, visit order and identity tie rule are kept.
"""
    node, atom, best = 0, -1, np.inf
    while node < len(end):
        counters[0] += 1
        near, far = box_interval(o, d, lo[node], hi[node])
        if near > best or far < 0.:
            node = end[node]
            continue
        group = leaf_group[node]
        if group >= 0:
            for k in range(offsets[group], offsets[group+1]):
                wanted = ids[k]
                if wanted == last:
                    continue
                counters[1] += 1
                distance = entry_distance(o, d, world[wanted], radii[wanted])
                if distance < best or (distance == best and np.isfinite(distance) and (atom < 0 or wanted < atom)):
                    atom, best = wanted, distance
        node += 1
    return atom, best


@njit(cache=False)
def _trace(origins, directions, lasts, initial, cap, record, stackless,
           lo, hi, left, right, end, leaf_group, offsets, ids, world, radii):
    p, d, counts = origins.copy(), directions.copy(), initial.copy()
    history = np.full((len(p), cap if record else 0), -1, dtype=np.int32)
    escaped = np.zeros(len(p), dtype=np.bool_)
    unresolved = escaped.copy()
    counters = np.zeros(2, dtype=np.int64)
    for ray in range(len(p)):
        last = lasts[ray]
        if record and counts[ray] == 1:
            history[ray, 0] = last
        while True:
            if stackless:
                atom, distance = _nearest_stackless(p[ray], d[ray], last,
                    lo, hi, end, leaf_group, offsets, ids, world, radii, counters)
            else:
                atom, distance = nearest_world(0, p[ray], d[ray], last, np.inf, -1,
                    lo, hi, left, right, end, leaf_group, offsets, ids, world, radii, counters)
            if atom < 0:
                escaped[ray] = True
                break
            if counts[ray] >= cap:
                unresolved[ray] = True
                break
            reflect(p[ray], d[ray], atom, distance, world, radii)
            if record:
                history[ray, counts[ray]] = atom
            counts[ray] += 1
            last = atom
    return p, d, escaped, unresolved, counts, history, counters


class OptimizedEHSSTransport:
    def __init__(self, model, simplify=False, coverage_depth=4, traversal='stackless'):
        start = time.perf_counter()
        self.model = model
        t, s = model.tree, model.spheres
        self.simplify = simplify
        if traversal not in ('stackless', 'recursive'):
            raise ValueError('Traversal must be stackless or recursive')
        self.traversal = traversal
        self.coverage = {}
        self.active = np.ones(len(s.radii), dtype=bool)
        if simplify:
            self.active, self.coverage = active_atoms(s, coverage_depth)
        if self.active.all():
            self.offsets, self.ids = t.offsets, t.ids
        else:
            groups = [t.ids[a:b][self.active[t.ids[a:b]]] for a, b in zip(t.offsets[:-1], t.offsets[1:])]
            self.offsets = np.r_[0, np.cumsum([len(g) for g in groups])].astype(np.int64)
            self.ids = np.concatenate(groups)
        self.certified_centers, self.certified_radii = s.centers.copy(), s.radii.copy()
        self.prepare_s = time.perf_counter()-start

    def trace(self, source, cap=64, record_history=False):
        if type(cap) is not int or cap < 1 or np.any(source.initial_bounces > 1):
            raise ValueError('Positive cap and zero/one initial collision required')
        s, t = self.model.spheres, self.model.tree
        if np.any(source.last_atom < -1) or np.any(source.last_atom >= len(s.radii)):
            raise ValueError('Invalid initial physical identity')
        start = time.perf_counter()
        if not (np.array_equal(s.centers, self.certified_centers) and np.array_equal(s.radii, self.certified_radii)):
            raise ValueError('Geometry changed: bind a new optimized transport and coverage certificate')
        result = _trace(source.origins, source.directions, source.last_atom, source.initial_bounces, cap,
            bool(record_history), self.traversal == 'stackless', t.lo, t.hi, t.left, t.right, t.end, t.leaf_group, self.offsets, self.ids,
            s.centers, s.radii)
        p, d, escaped, unresolved, counts, history, counters = result
        return EHSSResult(p, d, escaped, unresolved, counts, history, source,
            dict(query_s=time.perf_counter()-start, sphere_tests=int(counters[1]), box_tests=int(counters[0]),
                 history_recorded=bool(record_history), history_bytes=history.nbytes,
                 active_atoms=int(self.active.sum()), response_reuse=False, traversal=self.traversal,
                 backend='fused native node-aware BVH'))


class PreparedEHSS:
    """Reusable scene, boundary and native tracer with separated cost receipts.

Boundary kind and simplification are explicit: their benefits depend on shape
and the number of solves per geometry. Existing PhysicalScene refits preserve
node ownership. Coverage is rebuilt on changed frames, never silently reused.
"""
    def __init__(self, spheres, groups, atom_ids=None, boundary='sphere', simplify=False, coverage_depth=4):
        from .ehss_workload import PhysicalScene
        from .ehss_compact_source import CompactBoundary
        start = time.perf_counter()
        self.scene = PhysicalScene(spheres, groups, atom_ids)
        geometry_s = time.perf_counter()-start
        self.kind, self.simplify, self.depth = boundary, simplify, coverage_depth
        self.boundary = CompactBoundary.fit(spheres, boundary)
        self.engine = OptimizedEHSSTransport(self.scene.model, simplify, coverage_depth)
        self.setup = dict(total_s=time.perf_counter()-start, geometry_s=geometry_s,
                         boundary_s=self.boundary.prepare_s, engine_s=self.engine.prepare_s)

    def update(self, spheres):
        from .ehss_compact_source import CompactBoundary
        start = time.perf_counter()
        unchanged = (np.array_equal(spheres.centers, self.engine.certified_centers)
                     and np.array_equal(spheres.radii, self.engine.certified_radii))
        self.scene.update(spheres)
        if not unchanged:
            self.boundary = CompactBoundary.fit(spheres, self.kind)
            self.engine = OptimizedEHSSTransport(self.scene.model, self.simplify, self.depth)
        self.receipt = dict(total_s=time.perf_counter()-start, unchanged=unchanged,
            geometry_s=self.scene.receipt['total_s'], boundary_s=0. if unchanged else self.boundary.prepare_s,
            engine_s=0. if unchanged else self.engine.prepare_s, coverage_rebuilt=not unchanged and self.simplify)
        return self.receipt

    def solve(self, power=12, seed=0, cap=64, capture=False, record_history=False):
        from .ehss_workload import physical_readout
        start = time.perf_counter()
        source = self.boundary.source(power, seed)
        source_s = time.perf_counter()-start
        tick = time.perf_counter()
        result = self.engine.trace(source, cap, record_history)
        trace_s = time.perf_counter()-tick
        tick = time.perf_counter()
        output = physical_readout(result, cap)
        output.update(source_s=source_s, trace_s=trace_s, readout_s=time.perf_counter()-tick,
                      solve_s=time.perf_counter()-start, history_recorded=bool(record_history))
        if capture:
            output['_physical_result'] = result
        return output
