"""Sparse physical-surface 2D x outward-direction 2D response atlas.

Position uses a cubed-sphere face and two slopes. Direction uses independent
outward cosine and azimuth in a body-frame tangent basis. Nine valid physical
probes select a common itinerary; this is sampled, not interval-certified.
Foreign arbitration remains current. Optional owned arbitration supplies an
exact control, and the terminal flight always returns to full nearest search.
"""
from dataclasses import dataclass
import time
import numpy as np
from numba import njit, types
from numba.typed import Dict
from .ehss_response import node_geometry
from .ehss_reference import nearest_all, entry_distance
from .ehss_guarded_intrinsic_response import nearest_world, reflect
from .ehss_state import EHSSResult

KEY = types.UniTuple(types.int64, 7)


@njit(cache=True)
def normal_chart(n):
    axis = int(np.argmax(np.abs(n))); sign = 1 if n[axis] >= 0. else -1
    face = 2*axis+(sign < 0)
    other = np.array([(axis+1)%3, (axis+2)%3])
    return face, n[other[0]]/abs(n[axis]), n[other[1]]/abs(n[axis])


@njit(cache=True)
def normal_from_chart(face, u, v):
    axis = face//2; n = np.zeros(3); n[axis] = 1. if face%2 == 0 else -1.
    n[(axis+1)%3] = u; n[(axis+2)%3] = v
    return n/np.sqrt(np.dot(n, n))


@njit(cache=True)
def tangent_basis(n, face):
    helper = np.zeros(3); helper[(face//2+1)%3] = 1.
    u = helper-n*np.dot(helper, n); u /= np.sqrt(np.dot(u, u))
    return u, np.cross(n, u)


@njit(cache=True)
def surface_coordinates(n, d):
    face, a, b = normal_chart(n); u, v = tangent_basis(n, face)
    mu = np.dot(n, d)
    phi = np.arctan2(np.dot(d, v), np.dot(d, u))/(2*np.pi)
    if phi < 0.: phi += 1.
    return face, np.array([(a+1.)*.5, (b+1.)*.5, mu, phi])


@njit(cache=True)
def surface_state(face, q, center, radius):
    n = normal_from_chart(face, 2*q[0]-1., 2*q[1]-1.); u, v = tangent_basis(n, face)
    mu = min(1., max(0., q[2])); phi = 2*np.pi*q[3]
    d = mu*n+np.sqrt(max(0., 1.-mu*mu))*(np.cos(phi)*u+np.sin(phi)*v)
    d /= np.sqrt(np.dot(d, d))
    return center+radius*n, d


@njit(cache=True)
def surface_key(node, last, n, d, bins):
    face, q = surface_coordinates(n, d)
    return (node, last, face, min(bins-1, max(0, int(np.floor(q[0]*bins)))),
        min(bins-1, max(0, int(np.floor(q[1]*bins)))),
        min(bins-1, max(0, int(np.floor(q[2]*bins)))),
        min(bins-1, max(0, int(np.floor(q[3]*bins)))))


@njit(cache=True)
def compile_surface_cells(keys, atoms, centers, radii, bins, max_route, min_valid):
    accepted = np.zeros(len(keys), dtype=np.bool_)
    routes = np.full((len(keys), max_route), -1, dtype=np.int32)
    counts = np.zeros(len(keys), dtype=np.int64); valid_counts = counts.copy()
    tests = 0; probe_calls = 0
    for row in range(len(keys)):
        key = keys[row]; local_last = np.searchsorted(atoms, key[1]); face = key[2]
        middle = (key[3:].astype(np.float64)+.5)/bins
        agreed = True; first_count = -1
        for probe in range(9):
            q = middle.copy()
            if probe: q[(probe-1)//2] += (.5 if probe%2 else -.5)/bins
            if q[2] <= 1e-10: continue
            p, d = surface_state(face, q, centers[local_last], radii[local_last])
            physical = True
            for atom in range(len(radii)):
                if atom != local_last:
                    delta = p-centers[atom]
                    if np.dot(delta, delta) < (radii[atom]-1e-10)**2: physical = False; break
            if not physical: continue
            probe_calls += 1; valid_counts[row] += 1
            path = np.full(max_route, -1, dtype=np.int32); last = local_last; count = 0; finished = False
            for _ in range(max_route+1):
                atom, distance, nt = nearest_all(p, d, last, centers, radii); tests += nt
                if atom < 0: finished = True; break
                if count == max_route: break
                reflect(p, d, atom, distance, centers, radii)
                path[count] = atoms[atom]; count += 1; last = atom
            if not finished or count == 0: agreed = False; break
            if first_count < 0:
                first_count = count; routes[row, :count] = path[:count]; counts[row] = count
            elif count != first_count or np.any(path[:count] != routes[row, :count]): agreed = False; break
        accepted[row] = agreed and valid_counts[row] >= min_valid and first_count > 0
    return accepted, routes, counts, valid_counts, probe_calls, tests


@njit(cache=False)
def surface_replay(node, o, d, last, count, cap, history, bins, table, starts, paths, rotations, translations,
                   lo, hi, left, right, end, leaf_group, offsets, ids, world, radii, counters, stats, verify_owned):
    local_n = (o-world[last])@rotations[node]; local_n /= np.sqrt(np.dot(local_n, local_n))
    local_d = d@rotations[node]
    if np.dot(local_n, local_d) <= 1e-10: return False, last, count
    key = surface_key(node, last, local_n, local_d, bins); stats[0] += 1
    if key not in table: return False, last, count
    index = table[key]; first = starts[index]; stop = starts[index+1]
    if count+stop-first > cap: stats[7] += 1; return False, last, count
    stats[1] += 1; p = o.copy(); v = d.copy(); previous = last; done = 0; interrupted = False
    for k in range(first, stop):
        wanted = paths[k]
        if wanted == previous: stats[5] += 1; return False, last, count
        distance = entry_distance(p, v, world[wanted], radii[wanted]); stats[6] += 1
        if not np.isfinite(distance): stats[5] += 1; return False, last, count
        if verify_owned:
            competitor, before = nearest_world(node, p, v, previous, distance, -1, lo, hi, left, right, end,
                leaf_group, offsets, ids, world, radii, counters)
            stats[11] += 1
            if competitor >= 0 and (before < distance or (before == distance and competitor < wanted)):
                stats[12] += 1; interrupted = True; break
        competitor, before = nearest_world(0, p, v, previous, distance, node, lo, hi, left, right, end,
            leaf_group, offsets, ids, world, radii, counters)
        stats[4] += 1
        if competitor >= 0 and (before < distance or (before == distance and competitor < wanted)):
            stats[3] += 1; interrupted = True; break
        reflect(p, v, wanted, distance, world, radii); previous = wanted; done += 1
    if done == 0: return False, last, count
    o[:] = p; d[:] = v
    for k in range(done): history[count+k] = paths[first+k]
    stats[2] += 1; stats[8] += done
    if done >= 2: stats[9] += 1
    if interrupted: stats[10] += 1
    return True, previous, count+done


@njit(cache=False)
def trace_surface(origins, directions, last_atoms, initial, cap, bins, table, starts, paths, active,
                  rotations, translations, lo, hi, left, right, end, parent, leaf_group, leaf_node, groups,
                  offsets, ids, world, radii, verify_owned):
    p = origins.copy(); d = directions.copy(); counts = initial.copy()
    history = np.full((len(p), cap), -1, dtype=np.int32)
    escaped = np.zeros(len(p), dtype=np.bool_); unresolved = escaped.copy()
    counters = np.zeros(2, dtype=np.int64); stats = np.zeros(13, dtype=np.int64)
    hits = np.zeros(len(left), dtype=np.int64); chain = np.empty(len(left), dtype=np.int64)
    for ray in range(len(p)):
        last = last_atoms[ray]; count = counts[ray]
        if initial[ray] == 1: history[ray, 0] = last
        while True:
            if last >= 0:
                length = 0; node = leaf_node[groups[last]]
                while node >= 0:
                    chain[length] = node; length += 1; node = parent[node]
                for k in range(length-1, -1, -1):
                    node = chain[k]
                    if not active[node]: continue
                    used, last, count = surface_replay(node, p[ray], d[ray], last, count, cap, history[ray],
                        bins, table, starts, paths, rotations, translations, lo, hi, left, right, end,
                        leaf_group, offsets, ids, world, radii, counters, stats, verify_owned)
                    if used: hits[node] += 1; break
            # Sampled agreement does not certify the final empty flight. Always
            # search the whole current hierarchy, including the just-used node.
            atom, distance = nearest_world(0, p[ray], d[ray], last, np.inf, -1, lo, hi, left, right, end,
                leaf_group, offsets, ids, world, radii, counters)
            if atom < 0: escaped[ray] = True; break
            if count >= cap: unresolved[ray] = True; break
            reflect(p[ray], d[ray], atom, distance, world, radii)
            history[ray, count] = atom; count += 1; last = atom
        counts[ray] = count
    return p, d, escaped, unresolved, counts, history, counters, stats, hits


@dataclass(frozen=True)
class SurfaceNodeGeometry:
    atoms: np.ndarray
    anchor: int
    centers: np.ndarray
    radii: np.ndarray


class SurfaceResponseAtlas:
    def __init__(self, bins=4, residual_tolerance=.001, max_atoms=24, max_route=16, min_valid=3):
        if not isinstance(bins, int) or bins < 2: raise ValueError('At least two angular bins required')
        if not np.isfinite(residual_tolerance) or residual_tolerance < 0.: raise ValueError('Nonnegative finite residual tolerance required')
        if not isinstance(min_valid, int) or not 1 <= min_valid <= 9: raise ValueError('One to nine valid probes required')
        if not isinstance(max_atoms, int) or max_atoms < 1 or not isinstance(max_route, int) or max_route < 1:
            raise ValueError('Positive primitive and route budgets required')
        self.bins = bins; self.residual_tolerance = residual_tolerance; self.max_atoms = max_atoms
        self.max_route = max_route; self.min_valid = min_valid; self.nodes = {}; self.prepared = False

    def prepare(self, model, sources):
        if self.prepared: raise ValueError('Prepared atlas is frozen')
        start = time.perf_counter(); tree = model.tree; observed = set(); records = 0
        from .ehss_response import ResponseLibrary
        self.topology = ResponseLibrary.topology_key(tree)
        for node in range(len(tree.left)):
            atoms, anchor, centers, _, _ = node_geometry(tree, node)
            if len(atoms) > self.max_atoms: continue
            radii = tree.radii[atoms].copy()
            for value in (atoms, centers, radii): value.setflags(write=False)
            self.nodes[node] = SurfaceNodeGeometry(atoms, anchor, centers, radii)
        for source in sources:
            model.trace(source, 128, record_capacity=50000)
            lasts, states = model.last_records; records += len(lasts)
            for last, state in zip(lasts, states):
                node = int(tree.leaf_node[tree.groups[last]])
                while node >= 0 and node in self.nodes:
                    group = self.nodes[node].anchor; rotation = tree.rotations[group]
                    normal = (state[:3]-model.spheres.centers[last])@rotation; normal /= np.linalg.norm(normal)
                    direction = state[3:]@rotation
                    if np.dot(normal, direction) > 1e-10:
                        observed.add(tuple(surface_key(node, int(last), normal, direction, self.bins)))
                    node = int(tree.parent[node])
        discovery_s = time.perf_counter()-start; tick = time.perf_counter()
        keys = np.array(sorted(observed), dtype=np.int64).reshape(-1, 7)
        kept = []; routes = []; valid_counts = []; probes = tests = 0
        for node, geometry in self.nodes.items():
            selected = keys[keys[:, 0] == node]
            if not len(selected): continue
            accepted, paths, counts, valid, pc, nt = compile_surface_cells(selected, geometry.atoms, geometry.centers,
                geometry.radii, self.bins, self.max_route, self.min_valid)
            probes += int(pc); tests += int(nt)
            for index in np.flatnonzero(accepted):
                kept.append(selected[index]); routes.append(tuple(int(a) for a in paths[index, :counts[index]])); valid_counts.append(int(valid[index]))
        self.keys = np.array(kept, dtype=np.int64).reshape(-1, 7); self.routes = tuple(routes)
        used_nodes = set(self.keys[:, 0].tolist())
        self.nodes = {node:geometry for node, geometry in self.nodes.items() if node in used_nodes}
        self.keys.setflags(write=False); self.prepared = True
        self.receipt = dict(discovery_s=discovery_s, classification_s=time.perf_counter()-tick,
            total_prepare_s=time.perf_counter()-start, records=records, observed_cells=len(keys), retained_cells=len(kept),
            positive_only=True, bins=self.bins, min_valid=self.min_valid, valid_probe_counts=valid_counts,
            probe_calls=probes, probe_sphere_tests=tests, parent_cells=sum(tree.leaf_group[k[0]] < 0 for k in kept),
            multi_collision_cells=sum(len(r) > 1 for r in routes), sampled_not_certified=True,
            retained_nodes=len(self.nodes),
            numeric_payload_bytes=self.keys.nbytes+8*sum(len(r) for r in routes)+sum(g.atoms.nbytes+g.centers.nbytes+g.radii.nbytes for g in self.nodes.values()),
            python_and_dictionary_storage_included=False)
        return self.receipt

    def bind(self, model):
        return BoundSurfaceResponse(model, self)


class BoundSurfaceResponse:
    def __init__(self, model, atlas):
        from .ehss_response import ResponseLibrary
        if not atlas.prepared: raise ValueError('Prepared atlas required')
        if ResponseLibrary.topology_key(model.tree) != atlas.topology: raise ValueError('Physical ownership topology changed')
        start = time.perf_counter(); self.model = model; self.atlas = atlas
        tree = model.tree; self.rotations = np.tile(np.eye(3), (len(tree.left), 1, 1)); self.translations = np.zeros((len(tree.left), 3))
        self.active = np.zeros(len(tree.left), dtype=bool); invalid = []
        for node, old in atlas.nodes.items():
            atoms, anchor, centers, rotation, translation = node_geometry(tree, node)
            valid = anchor == old.anchor and np.array_equal(atoms, old.atoms) and np.array_equal(tree.radii[atoms], old.radii)
            valid = valid and np.all(np.abs(centers-old.centers) <= atlas.residual_tolerance+1e-10)
            if valid: self.rotations[node] = rotation; self.translations[node] = translation
            else: invalid.append(node)
        self.table = Dict.empty(KEY, types.int64); routes = []; kept = 0
        for key, route in zip(atlas.keys, atlas.routes):
            node = int(key[0])
            if node in invalid: continue
            self.table[tuple(int(v) for v in key)] = kept; routes.append(route); kept += 1; self.active[node] = True
        self.starts = np.r_[0, np.cumsum([len(r) for r in routes])].astype(np.int64)
        self.paths = np.array([a for route in routes for a in route], dtype=np.int64)
        for value in (self.rotations, self.translations, self.active, self.starts, self.paths): value.setflags(write=False)
        self.receipt = dict(bind_s=time.perf_counter()-start, active_cells=kept, inactive_cells=len(atlas.keys)-kept,
            active_nodes=int(self.active.sum()), sampled_geometry_compatibility=True,
            arrays_bytes=sum(a.nbytes for a in (self.rotations, self.translations, self.active, self.starts, self.paths)),
            dictionary_storage_included=False)

    def trace(self, source, max_bounces=128, verify_owned=False):
        if not isinstance(max_bounces, (int, np.integer)) or max_bounces < 1 or np.any(source.initial_bounces > 1):
            raise ValueError('Positive cap and uncollided/first-reflected input required')
        if np.any(source.last_atom < -1) or np.any(source.last_atom >= len(self.model.spheres.radii)): raise ValueError('Invalid collider identity')
        tree = self.model.tree; spheres = self.model.spheres; tick = time.perf_counter()
        p, d, escaped, unresolved, counts, paths, counters, stats, hits = trace_surface(source.origins, source.directions,
            source.last_atom, source.initial_bounces, max_bounces, self.atlas.bins, self.table, self.starts, self.paths,
            self.active, self.rotations, self.translations, tree.lo, tree.hi, tree.left, tree.right, tree.end, tree.parent,
            tree.leaf_group, tree.leaf_node, tree.groups, tree.offsets, tree.ids, spheres.centers, spheres.radii, verify_owned)
        labels = ('lookups', 'attempts', 'accepted_prefixes', 'foreign_interruptions', 'foreign_searches', 'replay_failures',
            'replay_sphere_tests', 'cap_fallbacks', 'replayed_collisions', 'multiple_prefixes', 'partial_prefixes',
            'owned_checks', 'owned_interruptions')
        return EHSSResult(p, d, escaped, unresolved, counts, paths, source, dict(query_s=time.perf_counter()-tick,
            box_tests=int(counters[0]), sphere_tests=int(counters[1]), responses=dict(zip(labels, map(int, stats))),
            node_response_hits=hits.tolist(), verify_owned=bool(verify_owned),
            backend='surface 2D x direction 2D sampled body atlas with current foreign arbitration'))
