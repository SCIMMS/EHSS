"""Analytic empty passage and positive-only, multiscale boundary responses.

Node boxes exclude only physically impossible collisions before the next domain
boundary. Their gain is separate from response reuse. Reflecting regions retain
probe-accepted itineraries and remain APPROXIMATE. No empty/UNKNOWN atlas cells
are retained. Optional last-independent charts are compiled without skip-last;
runtime still preserves the physical previous collider and rollback contract.
"""
import hashlib
import time
import numpy as np
from numba import njit, types
from numba.typed import Dict
from .ehss_clipped_response import ClippedResponseHierarchy, KEY_TYPE, boundary_key, node_face, replay_route, compile_cells
from .ehss_clipped_slabs import slab_response, locate_slab
from .ehss_state import EHSSResult


REGION_KEY_TYPE = types.UniTuple(types.int64, 8)


@njit(cache=True)
def box_miss(node, o, d, cursor, end, frame, box_low, box_high):
    """Conservative slab intersection of the remaining physical flight."""
    if end < cursor: return False
    begin = cursor; finish = end
    for axis in range(3):
        low = box_low[node, axis]; high = box_high[node, axis]
        if low > high: return True
        p = np.dot(o, frame[axis]); velocity = np.dot(d, frame[axis])
        if velocity == 0.:
            if p < low or p > high: return True
        else:
            first = (low-p)/velocity; second = (high-p)/velocity
            if first > second: first, second = second, first
            begin = max(begin, first); finish = min(finish, second)
            if begin > finish: return True
    return False


@njit(cache=False)
def region_node(node, domain, o, d, cursor, last, bounces, cap,
                normal, edges, offsets, ids, centers, radii, history, lower, upper, left, right,
                frame, box_low, box_high, widths, table, path_offsets, paths, active_levels, shared_last,
                gate_enabled, responses_enabled, calls, collisions, multiple, returns, stats):
    before = bounces; calls[node] += 1; tests = 0; used = False; status = 0
    if gate_enabled:
        stats[node, 0] += 1
        face, after = node_face(node, o, d, normal, edges, lower, upper)
        if box_miss(node, o, d, cursor, face, frame, box_low, box_high):
            stats[node, 1] += 1; used = True
            if np.isfinite(face): domain = after; cursor = face; status = 0
            else: status = 1
    if not used and responses_enabled and np.any(active_levels[node]):
        key_last = -1 if shared_last else last
        eligible, key = boundary_key(node, domain, o, d, cursor, key_last, normal, edges, lower, upper,
                                     frame[1], frame[2], widths)
        if eligible:
            stats[node, 2] += 1; found = False
            for level in range(active_levels.shape[1]-1, -1, -1):
                if not active_levels[node, level]: continue
                scale = 1 << level
                region_key = (key[0], np.int64(level), key[1], key[2], key[3]//scale, key[4]//scale, key[5]//scale, key[6]//scale)
                stats[node, 3] += 1
                if region_key not in table: continue
                found = True; route_index = table[region_key]
                route = paths[path_offsets[route_index]:path_offsets[route_index+1]]
                stats[node, 4] += 1
                ok, nd, nc, nl, nb, ns, nt = replay_route(node, domain, o, d, cursor, last, bounces, cap, route,
                    normal, edges, lower, upper, centers, radii, history)
                tests += nt; stats[node, 8] += nt
                if ok:
                    domain = nd; cursor = nc; last = nl; bounces = nb; status = ns; used = True
                    stats[node, 5] += 1; stats[node, 9] += len(route)
                    if level > 0: stats[node, 10] += 1
                    if len(route) >= 2: stats[node, 11] += 1
                    break
                stats[node, 6] += 1
            if not found: stats[node, 7] += 1
    if not used:
        if left[node] < 0:
            domain, cursor, last, bounces, status, nt = slab_response(domain, o, d, cursor, last, bounces, cap,
                normal, edges, offsets, ids, centers, radii, history)
            tests += nt
        else:
            finished = False
            for _ in range((cap+1)*(len(edges)+2)+1):
                child = left[node] if domain < upper[left[node]] else right[node]
                domain, cursor, last, bounces, status, nt = region_node(child, domain, o, d, cursor, last, bounces, cap,
                    normal, edges, offsets, ids, centers, radii, history, lower, upper, left, right,
                    frame, box_low, box_high, widths, table, path_offsets, paths, active_levels, shared_last,
                    gate_enabled, responses_enabled, calls, collisions, multiple, returns, stats)
                tests += nt
                if status != 0 or domain < lower[node] or domain >= upper[node]: finished = True; break
            if not finished: raise ValueError('Region response exceeded the physical-flight bound')
    added = bounces-before; collisions[node] += added
    if added >= 2: multiple[node] += 1
    if status == 0: returns[node] += 1
    return domain, cursor, last, bounces, status, tests


@njit(cache=False)
def trace_regions(origins, directions, last_atoms, initial_bounces, cap,
                  normal, edges, offsets, ids, centers, radii, lower, upper, left, right,
                  frame, box_low, box_high, widths, table, path_offsets, paths, active_levels, shared_last,
                  gate_enabled, responses_enabled):
    p = origins.copy(); d = directions.copy(); b = initial_bounces.copy()
    history = np.full((len(p), cap), -1, dtype=np.int32)
    escaped = np.zeros(len(p), dtype=np.bool_); unresolved = np.zeros(len(p), dtype=np.bool_)
    calls = np.zeros(len(left), dtype=np.int64); collisions = calls.copy(); multiple = calls.copy(); returns = calls.copy()
    stats = np.zeros((len(left), 12), dtype=np.int64); tests = 0
    for ray in range(len(p)):
        last = last_atoms[ray]
        if b[ray] == 1: history[ray, 0] = last
        domain = locate_slab(p[ray], d[ray], normal, edges)
        _, _, _, b[ray], status, nt = region_node(0, domain, p[ray], d[ray], 0., last, b[ray], cap,
            normal, edges, offsets, ids, centers, radii, history[ray], lower, upper, left, right,
            frame, box_low, box_high, widths, table, path_offsets, paths, active_levels, shared_last,
            gate_enabled, responses_enabled, calls, collisions, multiple, returns, stats)
        tests += nt; escaped[ray] = status == 1; unresolved[ray] = status == 2
        if status == 0: raise ValueError('Root response cannot hand off outside all space')
    return p, d, escaped, unresolved, b, history, tests, calls, collisions, multiple, returns, stats


class ReflectingRegionHierarchy(ClippedResponseHierarchy):
    def __init__(self, spheres, normal, edges, position_step=1., slope_step=.25):
        super().__init__(spheres, normal, edges, position_step, slope_step)
        tick = time.perf_counter()
        self.frame = np.stack([self.normal, self.u, self.v])
        projected = self.spheres.centers@self.frame.T
        self.box_low = np.full((len(self.left), 3), np.inf)
        self.box_high = np.full((len(self.left), 3), -np.inf)
        # Temporary candidate unions, never retained as parent atom geometry.
        for node in range(len(self.left)):
            atoms = np.unique(self.ids[self.offsets[self.lower[node]]:self.offsets[self.upper[node]]])
            if not len(atoms): continue
            radius = self.spheres.radii[atoms, None]
            lo = np.min(projected[atoms]-radius, axis=0)
            hi = np.max(projected[atoms]+radius, axis=0)
            if self.lower[node] > 0: lo[0] = max(lo[0], self.edges[self.lower[node]-1])
            if self.upper[node] <= len(self.edges): hi[0] = min(hi[0], self.edges[self.upper[node]-1])
            pad = 2048*np.finfo(float).eps*(1.+np.max(np.abs(self.spheres.centers[atoms]))+np.max(radius)+np.maximum(np.abs(lo), np.abs(hi)))
            self.box_low[node] = lo-pad; self.box_high[node] = hi+pad
        self.region_table = Dict.empty(REGION_KEY_TYPE, types.int64)
        self.region_keys = np.empty((0, 8), dtype=np.int64)
        self.region_routes = np.empty(0, dtype=np.int64)
        self.active_levels = np.zeros((len(self.left), 1), dtype=np.bool_)
        self.shared_last = False
        for a in (self.frame, self.box_low, self.box_high, self.region_keys, self.region_routes, self.active_levels): a.setflags(write=False)
        self.receipt.update(gate_build_s=time.perf_counter()-tick,
                            gate_array_bytes=self.frame.nbytes+self.box_low.nbytes+self.box_high.nbytes)

    def prepare(self, sources, max_bounces=128, max_route=16, shared_last=True, growth_levels=3):
        if self.prepared: raise ValueError('Prepared reflecting regions are frozen')
        if not isinstance(growth_levels, (int, np.integer)) or not 0 <= growth_levels <= 12:
            raise ValueError('Growth level must be an integer from 0 to 12')
        if not isinstance(max_route, (int, np.integer)) or max_route < 1:
            raise ValueError('Positive route limit required')
        start = time.perf_counter(); packets = []
        for source in sources:
            # Same exact collector as the fixed-grid control. Gate acceleration
            # is not credited to this preparation time or hidden in queries.
            result = self._trace(source, max_bounces, 1, False)
            packets.append(dict(count=len(source.weights), metrics=result.metrics))
        keys = np.array(list(self.table), dtype=np.int64).reshape(-1, 7)
        observed_keys = len(keys); self.shared_last = bool(shared_last)
        if shared_last and len(keys):
            keys[:, 2] = -1
            keys = np.unique(keys, axis=0)
        collection_s = time.perf_counter()-start
        self.table = Dict.empty(KEY_TYPE, types.int64)
        tick = time.perf_counter()
        accepted = {}; levels = []; previous = {}; all_routes = {}; route_payloads = []
        for level in range(growth_levels+1):
            if level:
                candidates = {}
                for key, route in previous.items():
                    parent = tuple(key[:3])+tuple(x//2 for x in key[3:])
                    candidates.setdefault(parent, set()).add(route)
                candidates = {key:next(iter(routes)) for key, routes in candidates.items() if len(routes) == 1}
                keys = np.array(list(candidates), dtype=np.int64).reshape(-1, 7)
            kinds, routes, counts, probes, tests, beam_checks = compile_cells(keys, self.normal, self.edges, self.offsets, self.ids,
                self.spheres.centers, self.spheres.radii, self.lower, self.upper, self.left, self.right,
                self.u, self.v, self.widths*(2.**level), max_route)
            previous = {}
            for index in np.flatnonzero(kinds == 2):
                key = tuple(int(x) for x in keys[index]); route = tuple(int(x) for x in routes[index, :counts[index]])
                if level and route != candidates[key]: continue
                previous[key] = route
                if route not in all_routes:
                    all_routes[route] = len(route_payloads); route_payloads.append(route)
                region_key = (key[0], level, *key[1:])
                accepted[region_key] = all_routes[route]
            levels.append(dict(level=level, candidate_cells=len(keys), accepted_cells=len(previous), probe_calls=int(probes),
                               probe_sphere_tests=int(tests), beam_ball_checks=int(beam_checks)))
            if not previous: break
        compile_s = time.perf_counter()-tick
        tick = time.perf_counter()
        self.region_keys = np.array(list(accepted), dtype=np.int64).reshape(-1, 8)
        self.region_routes = np.array(list(accepted.values()), dtype=np.int64)
        for key, value in accepted.items(): self.region_table[key] = value
        self.path_offsets = np.r_[0, np.cumsum([len(p) for p in route_payloads])].astype(np.int64)
        self.paths = np.array([atom for p in route_payloads for atom in p], dtype=np.int32)
        self.active_levels = np.zeros((len(self.left), growth_levels+1), dtype=np.bool_)
        for key in accepted: self.active_levels[key[0], key[1]] = True
        for a in (self.region_keys, self.region_routes, self.path_offsets, self.paths, self.active_levels): a.setflags(write=False)
        self.prepared = True
        self.compile_receipt = dict(collection_s=collection_s, compile_s=compile_s, pack_s=time.perf_counter()-tick,
            observed_keys=observed_keys, levels=levels, regions=len(accepted), unique_routes=len(route_payloads), stored_collisions=len(self.paths),
            multi_collision_routes=sum(len(p) >= 2 for p in route_payloads), shared_last=self.shared_last, growth_levels=growth_levels,
            kept_empty_or_unknown_cells=0, retained_collector_keys=len(self.table),
            packed_arrays_bytes=sum(a.nbytes for a in (self.region_keys, self.region_routes, self.path_offsets, self.paths, self.active_levels)),
            dictionary_storage_included=False, calibration_packets=packets, max_route=max_route)
        return self.compile_receipt

    def fingerprint(self):
        digest = hashlib.sha256(bytes([self.shared_last]))
        for a in (self.normal, self.edges, self.spheres.centers, self.spheres.radii, self.frame, self.box_low, self.box_high,
                  self.widths, self.region_keys, self.region_routes, self.path_offsets, self.paths, self.active_levels):
            digest.update(a.tobytes())
        return digest.hexdigest()

    def trace(self, source, max_bounces=128, use_responses=True, use_geometry_gate=True):
        if use_responses and not self.prepared: raise ValueError('Prepare reflecting regions before response queries')
        if not isinstance(max_bounces, (int, np.integer)) or max_bounces < 1 or np.any(source.initial_bounces > 1):
            raise ValueError('Positive cap and uncollided/first-reflected source required')
        if np.any(source.last_atom >= len(self.spheres.radii)) or np.any(source.last_atom < -1):
            raise ValueError('Invalid physical collider ID')
        tick = time.perf_counter()
        p, d, e, unresolved, b, h, tests, calls, collisions, multiple, returns, stats = trace_regions(
            source.origins, source.directions, source.last_atom, source.initial_bounces, max_bounces,
            self.normal, self.edges, self.offsets, self.ids, self.spheres.centers, self.spheres.radii, self.lower, self.upper, self.left, self.right,
            self.frame, self.box_low, self.box_high, self.widths, self.region_table, self.path_offsets, self.paths, self.active_levels,
            self.shared_last, use_geometry_gate, use_responses)
        labels = ['gate_checks', 'analytic_empty', 'eligible', 'lookups', 'attempts', 'reflected', 'rollback', 'missing',
                  'replay_sphere_tests', 'replayed_collisions', 'grown_region_hits', 'multi_collision_hits']
        return EHSSResult(p, d, e, unresolved, b, h, source,
            dict(query_s=time.perf_counter()-tick, sphere_tests=int(tests), node_calls=calls.tolist(), node_collisions=collisions.tolist(),
                 multi_collision_responses=multiple.tolist(), boundary_returns=returns.tolist(),
                 responses={label:stats[:, k].tolist() for k, label in enumerate(labels)},
                 backend='clipped hierarchy: analytic empty gates and positive reflecting regions',
                 geometry_gate=bool(use_geometry_gate), responses_enabled=bool(use_responses)))
