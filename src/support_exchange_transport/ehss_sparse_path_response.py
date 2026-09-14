"""Field-visited sparse path tries on physical surface/direction cells.

Several observed owned itineraries may coexist in a cell. A trie preserves
their prefix dependence. Current geometric intersections choose among its
next-collider candidates; terminal flights return to full nearest search.
This is an approximate candidate response, not an averaged angular operator.
"""
import time
import numpy as np
from numba import njit, types
from numba.typed import Dict
from .ehss_surface_response import KEY, SurfaceNodeGeometry, surface_key
from .ehss_response import node_geometry, ResponseLibrary
from .ehss_reference import nearest_all, entry_distance
from .ehss_guarded_intrinsic_response import nearest_world, reflect
from .ehss_state import EHSSResult


@njit(cache=True)
def record_from_paths(origins, directions, weights, initial, paths, counts, world, radii):
    total = int(counts.sum()); points = np.empty((total, 3)); directions_out = points.copy()
    lasts = np.empty(total, dtype=np.int64); mass = np.empty(total); cursor = 0
    for ray in range(len(origins)):
        p = origins[ray].copy(); d = directions[ray].copy()
        for k in range(counts[ray]):
            atom = paths[ray, k]
            if k >= initial[ray]:
                distance = entry_distance(p, d, world[atom], radii[atom])
                if not np.isfinite(distance): raise ValueError('Recorded reference path cannot be replayed')
                reflect(p, d, atom, distance, world, radii)
            points[cursor] = p; directions_out[cursor] = d; lasts[cursor] = atom
            mass[cursor] = weights[ray]; cursor += 1
    return points, directions_out, lasts, mass


@njit(cache=True)
def owned_paths(points, directions, lasts, atoms, centers, radii, max_route):
    routes = np.full((len(points), max_route), -1, dtype=np.int64)
    counts = np.zeros(len(points), dtype=np.int64); capped = 0; tests = 0
    for row in range(len(points)):
        p = points[row].copy(); d = directions[row].copy(); last = np.searchsorted(atoms, lasts[row])
        for k in range(max_route):
            atom, distance, nt = nearest_all(p, d, last, centers, radii); tests += nt
            if atom < 0: break
            reflect(p, d, atom, distance, centers, radii); routes[row, k] = atoms[atom]
            counts[row] += 1; last = atom
        if counts[row] == max_route: capped += 1
    return routes, counts, capped, tests


@njit(cache=False)
def replay_trie(node, o, d, last, count, cap, history, bins, table, roots, edge_starts, edge_atoms, edge_next,
                rotations, lo, hi, left, right, end, leaf_group, offsets, ids, world, radii, counters, stats, verify_owned):
    n = (o-world[last])@rotations[node]; n /= np.sqrt(np.dot(n, n)); local_d = d@rotations[node]
    if np.dot(n, local_d) <= 1e-10: return False, last, count
    key = surface_key(node, last, n, local_d, bins); stats[0] += 1
    if key not in table: return False, last, count
    stats[1] += 1; state = roots[table[key]]; done = 0; interrupted = False
    # Mutations are committed only after each actual reflection passes the
    # current foreign check (and optional owned control).
    while edge_starts[state] < edge_starts[state+1]:
        if count >= cap: stats[11] += 1; break
        wanted = -1; distance = np.inf; next_state = -1
        for edge in range(edge_starts[state], edge_starts[state+1]):
            atom = edge_atoms[edge]
            if atom == last: continue
            candidate = entry_distance(o, d, world[atom], radii[atom]); stats[2] += 1
            if np.isfinite(candidate) and (candidate < distance or (candidate == distance and (wanted < 0 or atom < wanted))):
                wanted = atom; distance = candidate; next_state = edge_next[edge]
        if wanted < 0: stats[10] += 1; break
        if verify_owned:
            other, before = nearest_world(node, o, d, last, distance, -1, lo, hi, left, right, end,
                leaf_group, offsets, ids, world, radii, counters); stats[8] += 1
            if other >= 0 and (before < distance or (before == distance and other < wanted)):
                stats[9] += 1; interrupted = True; break
        other, before = nearest_world(0, o, d, last, distance, node, lo, hi, left, right, end,
            leaf_group, offsets, ids, world, radii, counters); stats[6] += 1
        if other >= 0 and (before < distance or (before == distance and other < wanted)):
            stats[7] += 1; interrupted = True; break
        reflect(o, d, wanted, distance, world, radii); history[count] = wanted
        count += 1; done += 1; last = wanted; state = next_state
    if done:
        stats[3] += 1; stats[4] += done
        if done > 1: stats[5] += 1
        if interrupted: stats[13] += 1
    return done > 0, last, count


@njit(cache=False)
def trace_sparse_paths(origins, directions, last_atoms, initial, weights, cap, bins, table, roots,
                       edge_starts, edge_atoms, edge_next, active, rotations, lo, hi, left, right, end,
                       parent, leaf_group, leaf_node, groups, offsets, ids, world, radii, verify_owned, use_responses):
    p = origins.copy(); d = directions.copy(); counts = initial.copy()
    history = np.full((len(p), cap), -1, dtype=np.int32)
    escaped = np.zeros(len(p), dtype=np.bool_); unresolved = escaped.copy()
    counters = np.zeros(2, dtype=np.int64); stats = np.zeros(14, dtype=np.int64)
    hits = np.zeros(len(left), dtype=np.int64); chain = np.empty(len(left), dtype=np.int64); reused_mass = 0.
    for ray in range(len(p)):
        last = last_atoms[ray]; count = counts[ray]; reused = False
        if initial[ray] == 1: history[ray, 0] = last
        while True:
            if use_responses and last >= 0:
                length = 0; node = leaf_node[groups[last]]
                while node >= 0:
                    chain[length] = node; length += 1; node = parent[node]
                for k in range(length-1, -1, -1):
                    node = chain[k]
                    if not active[node]: continue
                    used, last, count = replay_trie(node, p[ray], d[ray], last, count, cap, history[ray], bins,
                        table, roots, edge_starts, edge_atoms, edge_next, rotations, lo, hi, left, right, end,
                        leaf_group, offsets, ids, world, radii, counters, stats, verify_owned)
                    if used: hits[node] += 1; reused = True; break
            atom, distance = nearest_world(0, p[ray], d[ray], last, np.inf, -1, lo, hi, left, right, end,
                leaf_group, offsets, ids, world, radii, counters)
            if atom < 0: escaped[ray] = True; break
            if count >= cap: unresolved[ray] = True; break
            reflect(p[ray], d[ray], atom, distance, world, radii)
            history[ray, count] = atom; count += 1; last = atom
        counts[ray] = count
        if reused: stats[12] += 1; reused_mass += weights[ray]
    return p, d, escaped, unresolved, counts, history, counters, stats, hits, reused_mass


def readonly(a):
    a = np.asarray(a); a.setflags(write=False); return a


class SparsePathAtlas:
    def __init__(self, bins=2, max_cells=2048, max_routes_per_cell=8, max_atoms=24, max_route=16, residual_tolerance=.001):
        for name, value in (('bins', bins), ('max_cells', max_cells), ('max_routes_per_cell', max_routes_per_cell),
                            ('max_atoms', max_atoms), ('max_route', max_route)):
            if not isinstance(value, int) or value < (2 if name == 'bins' else 1): raise ValueError('Positive integer budget required')
        if not np.isfinite(residual_tolerance) or residual_tolerance < 0.: raise ValueError('Finite nonnegative residual tolerance required')
        self.bins = bins; self.max_cells = max_cells; self.max_routes_per_cell = max_routes_per_cell
        self.max_atoms = max_atoms; self.max_route = max_route; self.residual_tolerance = residual_tolerance
        self.nodes = {}; self.prepared = False

    def prepare(self, model, sources):
        if self.prepared: raise ValueError('Prepared atlas is frozen')
        sources = list(sources)
        if not sources: raise ValueError('Calibration source required')
        start = time.perf_counter(); tree = model.tree; self.topology = ResponseLibrary.topology_key(tree)
        records = []; discovery_s = 0.; route_s = 0.; collection_s = 0.
        for source in sources:
            tick = time.perf_counter(); result = model.trace(source)
            records.append(record_from_paths(source.origins, source.directions, source.weights/len(sources),
                source.initial_bounces, result.collider_ids, result.bounces, model.spheres.centers, model.spheres.radii))
            discovery_s += time.perf_counter()-tick
        points, directions, lasts, mass = [np.concatenate([r[k] for r in records]) for k in range(4)]
        cells = {}; total_states = capped = tests = 0
        for node in range(len(tree.left)):
            atoms, anchor, centers, rotation, translation = node_geometry(tree, node)
            if len(atoms) > self.max_atoms: continue
            self.nodes[node] = SurfaceNodeGeometry(readonly(atoms), anchor, readonly(centers), readonly(tree.radii[atoms].copy()))
            selection = np.isin(lasts, atoms); last = lasts[selection]; w = mass[selection]
            if not len(last): continue
            chosen_points = points[selection]
            local_p = (chosen_points-translation)@rotation; local_d = directions[selection]@rotation
            tick = time.perf_counter()
            routes, lengths, cap_count, nt = owned_paths(local_p, local_d, last, atoms, centers, tree.radii[atoms], self.max_route)
            route_s += time.perf_counter()-tick; capped += int(cap_count); tests += int(nt); total_states += len(last)
            tick = time.perf_counter()
            for j in range(len(last)):
                n = (chosen_points[j]-model.spheres.centers[last[j]])@rotation; n /= np.linalg.norm(n)
                if n@local_d[j] <= 1e-10: continue
                key = tuple(surface_key(node, int(last[j]), n, local_d[j], self.bins))
                if key not in cells: cells[key] = [0., 0, {}]
                row = cells[key]; row[0] += float(w[j]); row[1] += 1
                if lengths[j]:
                    route = tuple(int(a) for a in routes[j, :lengths[j]])
                    previous = row[2].get(route, (0., 0)); row[2][route] = (previous[0]+float(w[j]), previous[1]+1)
            collection_s += time.perf_counter()-tick
        ranked = []
        for key, (visits_mass, visits, paths) in cells.items():
            if not paths: continue
            routes = sorted(paths, key=lambda r:(-paths[r][0]*len(r), r))[:self.max_routes_per_cell]
            weights = tuple(paths[r][0] for r in routes)
            score = len(self.nodes[key[0]].atoms)*sum(len(r)*w for r, w in zip(routes, weights))
            ranked.append((score, key, tuple(routes), weights, visits_mass, visits))
        ranked.sort(key=lambda row:(-row[0], row[1])); selected = sorted(ranked[:self.max_cells], key=lambda row:row[1])
        self.keys = readonly(np.array([r[1] for r in selected], dtype=np.int64).reshape(-1, 7))
        self.routes = tuple(r[2] for r in selected); self.route_weights = tuple(r[3] for r in selected)
        self.cell_visits = readonly(np.array([r[5] for r in selected], dtype=np.int64))
        self.cell_mass = readonly(np.array([r[4] for r in selected])); self.scores = readonly(np.array([r[0] for r in selected]))
        self.nodes = {node:g for node, g in self.nodes.items() if node in set(self.keys[:, 0])}
        self.prepared = True
        payload = sum(a.nbytes for a in (self.keys, self.cell_visits, self.cell_mass, self.scores))
        payload += sum(g.atoms.nbytes+g.centers.nbytes+g.radii.nbytes for g in self.nodes.values())
        payload += 8*sum(len(r)+1 for routes in self.routes for r in routes)
        self.receipt = dict(prepare_s=time.perf_counter()-start, discovery_s=discovery_s, owned_routes_s=route_s,
            collection_s=collection_s, physical_records=len(lasts), node_states=total_states, capped_owned_prefixes=capped,
            owned_sphere_tests=tests, observed_cells=len(cells), positive_cells_before_budget=len(ranked), cells=len(selected),
            multiple_route_cells=sum(len(r)>1 for r in self.routes), routes=sum(len(r) for r in self.routes),
            multiple_collision_routes=sum(len(r)>1 for rs in self.routes for r in rs),
            parent_cells=sum(tree.leaf_group[k[0]] < 0 for k in self.keys), numeric_payload_bytes=payload,
            dictionary_and_python_overhead_included=False, ranking='Weighted observed route length times owned primitive count; ancestors overlap, score is heuristic, not additive saved work.',
            source_weighting='Each supplied calibration ensemble contributes its physical ray weight divided by the number of ensembles; repeated visits remain repeated events.')
        return self.receipt

    def bind(self, model): return BoundSparsePaths(model, self)


class BoundSparsePaths:
    def __init__(self, model, atlas):
        if not atlas.prepared: raise ValueError('Prepared atlas required')
        if ResponseLibrary.topology_key(model.tree) != atlas.topology: raise ValueError('Physical ownership topology changed')
        start = time.perf_counter(); self.model = model; self.atlas = atlas; tree = model.tree
        self.rotations = np.tile(np.eye(3), (len(tree.left), 1, 1)); invalid = set()
        for node, old in atlas.nodes.items():
            atoms, anchor, centers, rotation, _ = node_geometry(tree, node)
            valid = anchor == old.anchor and np.array_equal(atoms, old.atoms) and np.array_equal(tree.radii[atoms], old.radii)
            valid = valid and np.all(np.abs(centers-old.centers) <= atlas.residual_tolerance+1e-10)
            if valid: self.rotations[node] = rotation
            else: invalid.add(node)
        self.active = np.zeros(len(tree.left), dtype=np.bool_); self.table = Dict.empty(KEY, types.int64)
        children = []; roots = []
        for key, routes in zip(atlas.keys, atlas.routes):
            node = int(key[0])
            if node in invalid: continue
            self.table[tuple(int(v) for v in key)] = len(roots); self.active[node] = True
            root = len(children); children.append({}); roots.append(root)
            for route in routes:
                state = root
                for atom in route:
                    if atom not in children[state]: children[state][atom] = len(children); children.append({})
                    state = children[state][atom]
        starts = [0]; atoms = []; next_states = []
        for edges in children:
            for atom in sorted(edges): atoms.append(atom); next_states.append(edges[atom])
            starts.append(len(atoms))
        self.roots = readonly(np.array(roots, dtype=np.int64)); self.edge_starts = readonly(np.array(starts, dtype=np.int64))
        self.edge_atoms = readonly(np.array(atoms, dtype=np.int64)); self.edge_next = readonly(np.array(next_states, dtype=np.int64))
        self.rotations = readonly(self.rotations); self.active = readonly(self.active)
        self.receipt = dict(bind_s=time.perf_counter()-start, active_cells=len(roots), inactive_cells=len(atlas.keys)-len(roots),
            active_nodes=int(self.active.sum()), trie_states=len(children), trie_edges=len(atoms),
            numeric_payload_bytes=sum(a.nbytes for a in (self.roots, self.edge_starts, self.edge_atoms, self.edge_next, self.rotations, self.active)),
            dictionary_and_python_overhead_included=False)

    def trace(self, source, max_bounces=128, verify_owned=False, use_responses=True):
        if not isinstance(max_bounces, (int, np.integer)) or max_bounces < 1 or np.any(source.initial_bounces > 1):
            raise ValueError('Positive cap and uncollided/first-reflected source required')
        if np.any(source.last_atom < -1) or np.any(source.last_atom >= len(self.model.spheres.radii)): raise ValueError('Invalid collider identity')
        t = self.model.tree; spheres = self.model.spheres; tick = time.perf_counter()
        p, d, escaped, unresolved, counts, paths, counters, stats, hits, reused_mass = trace_sparse_paths(
            source.origins, source.directions, source.last_atom, source.initial_bounces, source.weights, max_bounces,
            self.atlas.bins, self.table, self.roots, self.edge_starts, self.edge_atoms, self.edge_next, self.active,
            self.rotations, t.lo, t.hi, t.left, t.right, t.end, t.parent, t.leaf_group, t.leaf_node, t.groups,
            t.offsets, t.ids, spheres.centers, spheres.radii, verify_owned, use_responses)
        labels = ('lookups', 'cell_hits', 'candidate_sphere_tests', 'accepted_prefixes', 'replayed_collisions',
            'multiple_prefixes', 'foreign_searches', 'foreign_interruptions', 'owned_checks', 'owned_interruptions',
            'no_candidate', 'cap_stops', 'reused_rays', 'partial_prefixes')
        return EHSSResult(p, d, escaped, unresolved, counts, paths, source,
            dict(query_s=time.perf_counter()-tick, box_tests=int(counters[0]), sphere_tests=int(counters[1]),
                responses=dict(zip(labels, map(int, stats))), reused_ray_mass=reused_mass, node_response_hits=hits.tolist(),
                verify_owned=bool(verify_owned), use_responses=bool(use_responses), backend='field-visited owned path trie with current foreign arbitration'))
