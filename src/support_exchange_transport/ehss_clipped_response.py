"""Sparse boundary itineraries in a full clipped-domain response hierarchy.

Position on an oriented plane and two direction slopes form a 4D chart.
Empty beam cells have a conservative geometric THROUGH test. Reflecting cells
store a common nine-probe itinerary, replayed analytically on each new ray.
These itineraries are APPROXIMATE: probes do not exclude every possible rival
collider between them. A failed replay rolls back before exact child descent.
No query compiles cells, quantizes outgoing directions, or changes source tags.
"""
import hashlib
import time
import numpy as np
from numba import njit, types
from numba.typed import Dict
from .ehss_clipped_hierarchy import ClippedSlabHierarchy, clipped_node_response
from .ehss_clipped_slabs import slab_response, locate_slab
from .ehss_reference import entry_distance
from .ehss_state import EHSSResult


KEY_TYPE = types.UniTuple(types.int64, 7)


@njit(cache=True)
def boundary_key(node, domain, o, d, cursor, last, normal, edges, lower, upper, u, v, widths):
    empty = (np.int64(-1), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0))
    # Only a virtual incoming boundary is charted. Interior/physical-face
    # origins, exactly parallel and extremely grazing inputs use exact descent.
    dn = np.dot(d, normal)
    if node == 0 or cursor <= 0. or abs(dn) < 1e-8:
        return False, empty
    side = 1 if dn > 0. else -1
    if side == 1:
        if lower[node] == 0 or domain != lower[node]: return False, empty
        plane = edges[lower[node]-1]
    else:
        if upper[node] == len(edges)+1 or domain != upper[node]-1: return False, empty
        plane = edges[upper[node]-1]
    point = o + cursor*d
    tolerance = 256*np.finfo(np.float64).eps*(1.+np.sqrt(np.dot(point, point))+abs(plane))
    if abs(np.dot(point, normal)-plane) > tolerance: return False, empty
    coords = np.array([np.dot(point, u), np.dot(point, v), np.dot(d, u)/abs(dn), np.dot(d, v)/abs(dn)]) / widths
    if np.max(np.abs(coords)) > 2.**40: return False, empty
    bins = np.floor(coords).astype(np.int64)
    return True, (np.int64(node), np.int64(side), np.int64(last), bins[0], bins[1], bins[2], bins[3])


@njit(cache=True)
def node_face(node, o, d, normal, edges, lower, upper):
    dn = np.dot(d, normal)
    if dn > 0. and upper[node] <= len(edges):
        return (edges[upper[node]-1]-np.dot(o, normal))/dn, upper[node]
    if dn < 0. and lower[node] > 0:
        return (edges[lower[node]-1]-np.dot(o, normal))/dn, lower[node]-1
    return np.inf, -1


@njit(cache=True)
def replay_route(node, domain, o, d, cursor, last, count, cap, route,
                 normal, edges, lower, upper, centers, radii, history):
    """Try an analytic response transaction, committing only at node exit."""
    start_count = count
    p = o.copy(); out = d.copy(); nt = 0
    if len(route) and count+len(route) >= cap:
        return False, domain, cursor, last, count, 0, nt
    for atom in route:
        if atom == last: return False, domain, cursor, last, start_count, 0, nt
        face, _ = node_face(node, p, out, normal, edges, lower, upper)
        nt += 1
        distance = entry_distance(p, out, centers[atom], radii[atom])
        if not np.isfinite(distance) or distance < cursor or distance > face:
            return False, domain, cursor, last, start_count, 0, nt
        p[:] = p + distance*out
        n = p-centers[atom]; n /= np.sqrt(np.dot(n, n))
        p[:] = centers[atom]+radii[atom]*n
        out[:] = out-2*np.dot(out, n)*n
        out[:] /= np.sqrt(np.dot(out, out))
        count += 1; last = atom; cursor = 0.
        domain = locate_slab(p, out, normal, edges)
        if (domain < lower[node] or domain >= upper[node]) and count < start_count+len(route):
            return False, domain, cursor, last, start_count, 0, nt
    if domain < lower[node] or domain >= upper[node]:
        status = 0
    else:
        face, next_domain = node_face(node, p, out, normal, edges, lower, upper)
        if face < cursor: return False, domain, cursor, last, start_count, 0, nt
        if np.isfinite(face):
            domain = next_domain; cursor = face; status = 0
        else:
            status = 1
    # No earlier writes to the actual source trajectory or history.
    o[:] = p; d[:] = out
    for k in range(len(route)): history[start_count+k] = route[k]
    return True, domain, cursor, last, count, status, nt


@njit(cache=False)
def response_node(node, domain, o, d, cursor, last, bounces, cap,
                  normal, edges, offsets, ids, centers, radii, history, lower, upper, left, right,
                  u, v, widths, table, kinds, path_offsets, paths, mode, approximate,
                  calls, collisions, multiple, returns, stats):
    before = bounces; calls[node] += 1; tests = 0; used = False; status = 0
    if mode != 0:
        eligible, key = boundary_key(node, domain, o, d, cursor, last, normal, edges, lower, upper, u, v, widths)
        if eligible:
            stats[node, 0] += 1
            if mode == 1:
                if key not in table: table[key] = len(table)
            elif key not in table:
                stats[node, 1] += 1
            else:
                index = table[key]; kind = kinds[index]
                if kind == 0:
                    stats[node, 2] += 1
                elif kind == 2 and not approximate:
                    stats[node, 3] += 1
                else:
                    stats[node, 4] += 1
                    route = paths[path_offsets[index]:path_offsets[index+1]]
                    ok, nd, nc, nl, nb, ns, nt = replay_route(node, domain, o, d, cursor, last, bounces, cap, route,
                        normal, edges, lower, upper, centers, radii, history)
                    tests += nt; stats[node, 8] += nt
                    if ok:
                        domain = nd; cursor = nc; last = nl; bounces = nb; status = ns; used = True
                        stats[node, 5 if kind == 1 else 6] += 1
                        stats[node, 9] += len(route)
                    else:
                        stats[node, 7] += 1
    if not used:
        if left[node] < 0:
            domain, cursor, last, bounces, status, nt = slab_response(domain, o, d, cursor, last, bounces, cap,
                normal, edges, offsets, ids, centers, radii, history)
            tests += nt
        else:
            finished = False
            for _ in range((cap+1)*(len(edges)+2)+1):
                child = left[node] if domain < upper[left[node]] else right[node]
                domain, cursor, last, bounces, status, nt = response_node(child, domain, o, d, cursor, last, bounces, cap,
                    normal, edges, offsets, ids, centers, radii, history, lower, upper, left, right,
                    u, v, widths, table, kinds, path_offsets, paths, mode, approximate,
                    calls, collisions, multiple, returns, stats)
                tests += nt
                if status != 0 or domain < lower[node] or domain >= upper[node]: finished = True; break
            if not finished: raise ValueError('Response composition exceeded physical-flight bound')
    added = bounces-before; collisions[node] += added
    if added >= 2: multiple[node] += 1
    if status == 0: returns[node] += 1
    return domain, cursor, last, bounces, status, tests


@njit(cache=False)
def trace_responses(origins, directions, last_atoms, initial_bounces, cap,
                    normal, edges, offsets, ids, centers, radii, lower, upper, left, right,
                    u, v, widths, table, kinds, path_offsets, paths, mode, approximate):
    p = origins.copy(); d = directions.copy(); b = initial_bounces.copy()
    history = np.full((len(p), cap), -1, dtype=np.int32)
    escaped = np.zeros(len(p), dtype=np.bool_); unresolved = np.zeros(len(p), dtype=np.bool_)
    calls = np.zeros(len(left), dtype=np.int64); collisions = calls.copy(); multiple = calls.copy(); returns = calls.copy()
    stats = np.zeros((len(left), 10), dtype=np.int64); tests = 0
    for ray in range(len(p)):
        last = last_atoms[ray]
        if b[ray] == 1: history[ray, 0] = last
        domain = locate_slab(p[ray], d[ray], normal, edges)
        _, _, _, b[ray], status, nt = response_node(0, domain, p[ray], d[ray], 0., last, b[ray], cap,
            normal, edges, offsets, ids, centers, radii, history[ray], lower, upper, left, right,
            u, v, widths, table, kinds, path_offsets, paths, mode, approximate,
            calls, collisions, multiple, returns, stats)
        tests += nt; escaped[ray] = status == 1; unresolved[ray] = status == 2
        if status == 0: raise ValueError('Root response must escape or retain a real cap tail')
    return p, d, escaped, unresolved, b, history, tests, calls, collisions, multiple, returns, stats


@njit(cache=True)
def beam_empty(key, normal, edges, lower, upper, u, v, widths, centers, radii):
    """Enclose every straight chart ray over each candidate ball's axial span.

    Overlap of transverse rectangles is only a possible hit. Rounding padding
    expands both the beam and sphere tests. A positive result is THROUGH, never
    a probe-based miss. The preceding convex collider follows the reference's
    skip-last convention. No physical domain geometry is copied into the cell.
    """
    node, side, last = key[:3]
    plane = edges[lower[node]-1] if side == 1 else edges[upper[node]-1]
    opposite = (edges[upper[node]-1] if upper[node] <= len(edges) else np.inf) if side == 1 else (edges[lower[node]-1] if lower[node] > 0 else -np.inf)
    extent = side*(opposite-plane)
    low = np.empty(4); high = np.empty(4)
    for k in range(4): low[k] = key[k+3]*widths[k]; high[k] = low[k]+widths[k]
    checks = 0
    for atom in range(len(radii)):
        if atom == last: continue
        checks += 1
        center = centers[atom]; r = radii[atom]
        pad = 2048*np.finfo(np.float64).eps*(1.+np.sqrt(np.dot(center, center))+r+abs(plane))
        axis = side*(np.dot(center, normal)-plane)
        a = max(0., axis-r-pad); b = min(extent, axis+r+pad)
        if b < a: continue
        possible = True
        for k in range(2):
            s0 = low[k+2]; s1 = high[k+2]
            minimum = min(s0*a, s0*b, s1*a, s1*b)+low[k]
            maximum = max(s0*a, s0*b, s1*a, s1*b)+high[k]
            margin = pad*(1.+abs(s0)+abs(s1))+2048*np.finfo(np.float64).eps*(1.+abs(minimum)+abs(maximum))
            transverse = np.dot(center, u if k == 0 else v)
            if transverse+r+margin < minimum or transverse-r-margin > maximum:
                possible = False; break
        if possible: return False, checks
    return True, checks


@njit(cache=False)
def compile_cells(keys, normal, edges, offsets, ids, centers, radii, lower, upper, left, right, u, v, widths, max_route):
    kinds = np.zeros(len(keys), dtype=np.int8)
    routes = np.full((len(keys), max_route), -1, dtype=np.int32)
    counts = np.zeros(len(keys), dtype=np.int64)
    probe_calls = 0; probe_tests = 0; beam_checks = 0
    for index in range(len(keys)):
        key = keys[index]; node = key[0]; side = key[1]; last = key[2]
        empty, checks = beam_empty(key, normal, edges, lower, upper, u, v, widths, centers, radii)
        beam_checks += checks
        if empty: kinds[index] = 1; continue
        plane = edges[lower[node]-1] if side == 1 else edges[upper[node]-1]
        domain = lower[node] if side == 1 else upper[node]-1
        seed = (key[3:]+.5)*widths
        accepted = True; first_status = -1; first_domain = -1
        for probe in range(9):
            q = seed.copy()
            if probe: q[(probe-1)//2] += (.5 if probe%2 else -.5)*widths[(probe-1)//2]
            p = plane*normal+q[0]*u+q[1]*v
            d = side*normal+q[2]*u+q[3]*v; d /= np.sqrt(np.dot(d, d))
            # A probe inside the atom union is not a valid free incoming state.
            for atom in range(len(radii)):
                delta = p-centers[atom]
                if np.dot(delta, delta) < radii[atom]**2:
                    accepted = False; break
            if not accepted: break
            history = np.full(max_route+1, -1, dtype=np.int32)
            calls = np.zeros(len(left), dtype=np.int64); col = calls.copy(); multi = calls.copy(); ret = calls.copy()
            nd, _, _, bounces, status, nt = clipped_node_response(node, domain, p, d, 0., last, 0, max_route+1,
                normal, edges, offsets, ids, centers, radii, history, lower, upper, left, right, calls, col, multi, ret)
            probe_calls += 1; probe_tests += nt
            if status == 2 or bounces == 0 or bounces > max_route:
                accepted = False; break
            if probe == 0:
                counts[index] = bounces; routes[index, :bounces] = history[:bounces]
                first_status = status; first_domain = nd
            elif bounces != counts[index] or status != first_status or nd != first_domain or np.any(history[:bounces] != routes[index, :bounces]):
                accepted = False; break
        if accepted: kinds[index] = 2
        else: counts[index] = 0
    return kinds, routes, counts, probe_calls, probe_tests, beam_checks


class ClippedResponseHierarchy(ClippedSlabHierarchy):
    def __init__(self, spheres, normal, edges, position_step=2., slope_step=.5):
        super().__init__(spheres, normal, edges)
        if not np.isfinite([position_step, slope_step]).all() or min(position_step, slope_step) <= 0:
            raise ValueError('Positive finite chart steps required')
        helper = np.eye(3)[np.argmin(np.abs(self.normal))]
        self.u = np.cross(self.normal, helper); self.u /= np.linalg.norm(self.u)
        self.v = np.cross(self.normal, self.u)
        self.widths = np.array([position_step, position_step, slope_step, slope_step])
        self.table = Dict.empty(KEY_TYPE, types.int64)
        self.kinds = np.empty(0, dtype=np.int8); self.path_offsets = np.zeros(1, dtype=np.int64); self.paths = np.empty(0, dtype=np.int32)
        for a in (self.u, self.v, self.widths, self.kinds, self.path_offsets, self.paths): a.setflags(write=False)
        self.prepared = False; self.compile_receipt = {}

    def _trace(self, source, cap, mode, approximate):
        if not isinstance(cap, (int, np.integer)) or cap < 1 or np.any(source.initial_bounces > 1):
            raise ValueError('Positive cap and uncollided/first-reflected source required')
        if np.any(source.last_atom >= len(self.spheres.radii)) or np.any(source.last_atom < -1):
            raise ValueError('Invalid physical collider ID')
        tick = time.perf_counter()
        p, d, e, unresolved, b, h, tests, calls, collisions, multiple, returns, stats = trace_responses(
            source.origins, source.directions, source.last_atom, source.initial_bounces, cap,
            self.normal, self.edges, self.offsets, self.ids, self.spheres.centers, self.spheres.radii,
            self.lower, self.upper, self.left, self.right, self.u, self.v, self.widths,
            self.table, self.kinds, self.path_offsets, self.paths, mode, approximate)
        labels = ['eligible', 'missing', 'unknown', 'approximate_disabled', 'attempts', 'through', 'reflected', 'rollback', 'replay_sphere_tests', 'replayed_collisions']
        metrics = dict(query_s=time.perf_counter()-tick, sphere_tests=int(tests), node_calls=calls.tolist(), node_collisions=collisions.tolist(),
                       multi_collision_responses=multiple.tolist(), boundary_returns=returns.tolist(),
                       responses={label:stats[:, k].tolist() for k, label in enumerate(labels)},
                       backend='clipped hierarchy with sparse boundary itineraries', approximate_enabled=bool(approximate and mode == 2))
        return EHSSResult(p, d, e, unresolved, b, h, source, metrics)

    def prepare(self, sources, max_bounces=128, max_route=16):
        if self.prepared: raise ValueError('Prepared table is frozen; use a new model to compile new cells')
        if not isinstance(max_route, (int, np.integer)) or max_route < 1: raise ValueError('Positive route limit required')
        start = time.perf_counter(); packets = []
        for source in sources:
            result = self._trace(source, max_bounces, 1, False)
            packets.append(dict(count=len(source.weights), metrics=result.metrics))
        collection_s = time.perf_counter()-start
        keys = np.array(list(self.table), dtype=np.int64).reshape(-1, 7)
        # The packed rows must follow table indices, not an assumed hash order.
        if len(keys):
            keys = keys[np.argsort(np.array([self.table[tuple(key)] for key in keys]))]
        tick = time.perf_counter()
        kinds, routes, counts, probes, tests, beam_checks = compile_cells(keys, self.normal, self.edges, self.offsets, self.ids,
            self.spheres.centers, self.spheres.radii, self.lower, self.upper, self.left, self.right, self.u, self.v, self.widths, max_route)
        compile_s = time.perf_counter()-tick
        tick = time.perf_counter()
        self.kinds = kinds; self.path_offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
        self.paths = np.concatenate([routes[k, :counts[k]] for k in range(len(keys))]).astype(np.int32) if len(keys) else np.empty(0, dtype=np.int32)
        self.keys = keys
        for a in (self.keys, self.kinds, self.path_offsets, self.paths): a.setflags(write=False)
        self.prepared = True
        self.compile_receipt = dict(collection_s=collection_s, compile_s=compile_s, pack_s=time.perf_counter()-tick,
            cells=len(keys), through_cells=int(np.count_nonzero(kinds == 1)), reflecting_cells=int(np.count_nonzero(kinds == 2)),
            unknown_cells=int(np.count_nonzero(kinds == 0)), stored_collisions=len(self.paths), probe_calls=int(probes),
            probe_sphere_tests=int(tests), beam_ball_checks=int(beam_checks), calibration_packets=packets,
            packed_arrays_bytes=sum(a.nbytes for a in (self.keys, self.kinds, self.path_offsets, self.paths)),
            dictionary_storage_included=False, chart_widths=self.widths.tolist(), max_route=max_route,
            multi_collision_cells=int(np.count_nonzero((kinds == 2)&(counts >= 2))),
            cells_by_node=[dict(node=int(n), through=int(np.count_nonzero((keys[:, 0] == n)&(kinds == 1))),
                reflecting=int(np.count_nonzero((keys[:, 0] == n)&(kinds == 2)))) for n in range(len(self.left))])
        return self.compile_receipt

    def fingerprint(self):
        digest = hashlib.sha256()
        for a in (self.normal, self.edges, self.spheres.centers, self.spheres.radii, self.widths, self.keys, self.kinds, self.path_offsets, self.paths):
            digest.update(a.tobytes())
        return digest.hexdigest()

    def trace(self, source, max_bounces=128, use_responses=True, approximate=True):
        if use_responses and not self.prepared: raise ValueError('Prepare on calibration inputs before response queries')
        return self._trace(source, max_bounces, 2 if use_responses else 0, approximate)
