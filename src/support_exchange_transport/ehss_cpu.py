"""CPU-only scalar/array and serial/parallel EHSS ablation on one fixed BVH."""
import time
import numpy as np
import numba
from numba import njit, prange
from .ehss_optimized import OptimizedEHSSTransport, _nearest_stackless
from .ehss_support_tree import box_interval
from .ehss_guarded_intrinsic_response import reflect
from .ehss_state import EHSSResult


@njit(inline='always')
def entry_scalar(o, d, c, radius):
    x0, x1, x2 = o[0]-c[0], o[1]-c[1], o[2]-c[2]
    aa = (d[0]*d[0]+d[1]*d[1])+d[2]*d[2]
    projection = ((x0*d[0]+x1*d[1])+x2*d[2])/aa
    p0, p1, p2 = x0-projection*d[0], x1-projection*d[1], x2-projection*d[2]
    disc = radius*radius-((p0*p0+p1*p1)+p2*p2)
    if disc < 0. or projection >= 0.:
        return np.inf
    near = -projection-np.sqrt(max(0., disc)/aa)
    if near < -64*np.finfo(np.float64).eps*radius:
        return np.inf
    return max(0., near)


@njit(inline='always')
def reflect_scalar(o, d, atom, distance, world, radii):
    n0 = o[0]+distance*d[0]-world[atom, 0]
    n1 = o[1]+distance*d[1]-world[atom, 1]
    n2 = o[2]+distance*d[2]-world[atom, 2]
    norm = np.sqrt((n0*n0+n1*n1)+n2*n2)
    n0 /= norm; n1 /= norm; n2 /= norm
    o[0] = world[atom, 0]+radii[atom]*n0
    o[1] = world[atom, 1]+radii[atom]*n1
    o[2] = world[atom, 2]+radii[atom]*n2
    dot = (d[0]*n0+d[1]*n1)+d[2]*n2
    d[0] = d[0]-2*dot*n0; d[1] = d[1]-2*dot*n1; d[2] = d[2]-2*dot*n2
    norm = np.sqrt((d[0]*d[0]+d[1]*d[1])+d[2]*d[2])
    d[0] /= norm; d[1] /= norm; d[2] /= norm


@njit
def nearest_scalar(o, d, last, lo, hi, end, leaf_group, offsets, ids, world, radii):
    node, atom, best, boxes, spheres = 0, -1, np.inf, 0, 0
    while node < len(end):
        boxes += 1
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
                spheres += 1
                distance = entry_scalar(o, d, world[wanted], radii[wanted])
                if distance < best or (distance == best and np.isfinite(distance) and (atom < 0 or wanted < atom)):
                    atom, best = wanted, distance
        node += 1
    return atom, best, boxes, spheres


def _trace_loop(origins, directions, lasts, initial, cap, record, scalar,
                lo, hi, end, leaf_group, offsets, ids, world, radii):
    p, d, counts = origins.copy(), directions.copy(), initial.copy()
    escaped = np.zeros(len(p), dtype=np.bool_); unresolved = escaped.copy()
    history = np.full((len(p), cap if record else 0), -1, dtype=np.int32)
    boxes, spheres = 0, 0
    for ray in prange(len(p)):
        last = lasts[ray]
        local = np.zeros(2, dtype=np.int64)  # private per ray, never shared across workers
        if record and counts[ray] == 1:
            history[ray, 0] = last
        nb, ns = 0, 0
        while True:
            if scalar:
                atom, distance, b, s = nearest_scalar(p[ray], d[ray], last,
                    lo, hi, end, leaf_group, offsets, ids, world, radii)
                nb += b; ns += s
            else:
                atom, distance = _nearest_stackless(p[ray], d[ray], last,
                    lo, hi, end, leaf_group, offsets, ids, world, radii, local)
            if atom < 0:
                escaped[ray] = True
                break
            if counts[ray] >= cap:
                unresolved[ray] = True
                break
            if scalar:
                reflect_scalar(p[ray], d[ray], atom, distance, world, radii)
            else:
                reflect(p[ray], d[ray], atom, distance, world, radii)
            if record:
                history[ray, counts[ray]] = atom
            counts[ray] += 1; last = atom
        boxes += nb+local[0]; spheres += ns+local[1]
    return p, d, escaped, unresolved, counts, history, boxes, spheres


trace_serial = njit(_trace_loop)
trace_parallel = njit(parallel=True)(_trace_loop)


def check_threads(threads):
    if type(threads) is not int or not 1 <= threads <= numba.config.NUMBA_NUM_THREADS:
        raise ValueError(f'Threads must be 1..{numba.config.NUMBA_NUM_THREADS}')


class CPUEHSSTransport(OptimizedEHSSTransport):
    def __init__(self, model, threads=1, scalar=True, simplify=False, coverage_depth=4):
        check_threads(threads)
        super().__init__(model, simplify=simplify, coverage_depth=coverage_depth)
        self.threads, self.scalar = threads, bool(scalar)

    def trace(self, source, cap=64, record_history=False):
        if type(cap) is not int or cap < 1 or np.any(source.initial_bounces > 1):
            raise ValueError('Positive cap and zero/one initial collision required')
        s, t = self.model.spheres, self.model.tree
        if np.any(source.last_atom < -1) or np.any(source.last_atom >= len(s.radii)):
            raise ValueError('Invalid initial physical identity')
        start = time.perf_counter()
        if not (np.array_equal(s.centers, self.certified_centers) and np.array_equal(s.radii, self.certified_radii)):
            raise ValueError('Geometry changed: bind a new transport')
        previous_threads = numba.get_num_threads()
        try:
            if self.threads > 1:
                numba.set_num_threads(self.threads)
            kernel = trace_serial if self.threads == 1 else trace_parallel
            result = kernel(source.origins, source.directions, source.last_atom, source.initial_bounces,
                cap, bool(record_history), self.scalar, t.lo, t.hi, t.end, t.leaf_group, self.offsets, self.ids,
                s.centers, s.radii)
        finally:
            if numba.get_num_threads() != previous_threads:
                numba.set_num_threads(previous_threads)
        p, d, escaped, unresolved, counts, history, boxes, spheres = result
        return EHSSResult(p, d, escaped, unresolved, counts, history, source,
            dict(query_s=time.perf_counter()-start, sphere_tests=int(spheres), box_tests=int(boxes),
                 backend='Numba CPU', threads=self.threads, scalar=self.scalar, fastmath=False,
                 history_recorded=bool(record_history), response_reuse=False))
