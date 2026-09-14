"""Indexed dispatch of unchanged sparse path responses.

Physical collider -> prepared ancestor nodes -> occupied surface bins ->
occupied outward-cosine rows -> azimuth bin. No response approximation is
changed. The hierarchy, atom-list and indexed modes share one trace kernel.
"""
import time
import numpy as np
from numba import njit
from .ehss_sparse_path_response import readonly, replay_trie
from .ehss_surface_response import normal_chart, tangent_basis
from .ehss_reference import entry_distance
from .ehss_guarded_intrinsic_response import nearest_world, reflect
from .ehss_state import EHSSResult


@njit(cache=True)
def lookup_indexed(entry, node, o, d, last, bins, rotations, world, position_rows, cosine_live, direction_roots, dispatch):
    n = (o-world[last])@rotations[node]; n /= np.sqrt(np.dot(n, n))
    face, u, v = normal_chart(n)
    a = min(bins-1, max(0, int(np.floor((u+1.)*.5*bins))))
    b = min(bins-1, max(0, int(np.floor((v+1.)*.5*bins))))
    row = position_rows[entry, face, a, b]
    if row < 0: dispatch[3] += 1; return -1
    local_d = d@rotations[node]; mu = np.dot(n, local_d)
    if mu <= 1e-10: dispatch[4] += 1; return -1
    c = min(bins-1, max(0, int(np.floor(mu*bins))))
    if not cosine_live[row, c]: dispatch[4] += 1; return -1
    tangent, bitangent = tangent_basis(n, face)
    phi = np.arctan2(np.dot(local_d, bitangent), np.dot(local_d, tangent))/(2*np.pi)
    if phi < 0.: phi += 1.
    azimuth = min(bins-1, max(0, int(np.floor(phi*bins))))
    dispatch[5] += 1
    return direction_roots[row, c, azimuth]


@njit(cache=False)
def replay_root(node, state, o, d, last, count, cap, history, edge_starts, edge_atoms, edge_next,
                lo, hi, left, right, end, leaf_group, offsets, ids, world, radii, counters, stats, verify_owned):
    stats[1] += 1; done = 0; interrupted = False
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
def trace_dispatch(origins, directions, last_atoms, initial, weights, cap, bins, table, roots,
                   edge_starts, edge_atoms, edge_next, active, rotations, lo, hi, left, right, end,
                   parent, leaf_group, leaf_node, groups, offsets, ids, world, radii,
                   atom_starts, entry_nodes, position_rows, cosine_live, direction_roots, verify_owned, mode):
    p = origins.copy(); d = directions.copy(); counts = initial.copy()
    history = np.full((len(p), cap), -1, dtype=np.int32)
    escaped = np.zeros(len(p), dtype=np.bool_); unresolved = escaped.copy()
    counters = np.zeros(2, dtype=np.int64); stats = np.zeros(14, dtype=np.int64)
    dispatch = np.zeros(7, dtype=np.int64)
    hits = np.zeros(len(left), dtype=np.int64); chain = np.empty(len(left), dtype=np.int64); reused_mass = 0.
    for ray in range(len(p)):
        last = last_atoms[ray]; count = counts[ray]; reused = False
        if initial[ray] == 1: history[ray, 0] = last
        while True:
            if mode > 0 and last >= 0:
                dispatch[0] += 1
                if mode == 1:
                    length = 0; node = leaf_node[groups[last]]
                    while node >= 0:
                        chain[length] = node; length += 1; node = parent[node]
                    for k in range(length-1, -1, -1):
                        node = chain[k]
                        if not active[node]: continue
                        dispatch[2] += 1
                        used, last, count = replay_trie(node, p[ray], d[ray], last, count, cap, history[ray], bins,
                            table, roots, edge_starts, edge_atoms, edge_next, rotations, lo, hi, left, right, end,
                            leaf_group, offsets, ids, world, radii, counters, stats, verify_owned)
                        if used: hits[node] += 1; reused = True; break
                else:
                    start = atom_starts[last]; stop = atom_starts[last+1]
                    if start == stop: dispatch[1] += 1
                    for entry in range(start, stop):
                        node = entry_nodes[entry]; dispatch[2] += 1
                        if mode == 2:
                            used, last, count = replay_trie(node, p[ray], d[ray], last, count, cap, history[ray], bins,
                                table, roots, edge_starts, edge_atoms, edge_next, rotations, lo, hi, left, right, end,
                                leaf_group, offsets, ids, world, radii, counters, stats, verify_owned)
                        else:
                            state = lookup_indexed(entry, node, p[ray], d[ray], last, bins, rotations, world,
                                position_rows, cosine_live, direction_roots, dispatch)
                            if state < 0: continue
                            stats[0] += 1; dispatch[6] += 1
                            used, last, count = replay_root(node, state, p[ray], d[ray], last, count, cap, history[ray],
                                edge_starts, edge_atoms, edge_next, lo, hi, left, right, end,
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
    return p, d, escaped, unresolved, counts, history, counters, stats, hits, reused_mass, dispatch


class IndexedResponse:
    """An additional exact lookup index over an already-bound frozen atlas."""
    def __init__(self, bound, max_index_bytes=64*1024*1024):
        if not isinstance(max_index_bytes, int) or max_index_bytes < 1: raise ValueError('Positive index byte budget required')
        tick = time.perf_counter(); self.bound = bound; self.model = bound.model
        bins = bound.atlas.bins; physical_count = len(bound.model.spheres.radii); t = bound.model.tree
        entries = {}; occupied = set()
        for key in bound.table:
            node, last, face, u, v, mu, phi = key
            if not 0 <= last < physical_count: raise ValueError('Invalid collider in response key')
            ancestor = int(t.leaf_node[t.groups[last]])
            while ancestor >= 0 and ancestor != node: ancestor = int(t.parent[ancestor])
            if ancestor < 0: raise ValueError('Response owner is not an ancestor of collider')
            if not 0 <= face < 6 or any(not 0 <= k < bins for k in (u, v, mu, phi)): raise ValueError('Invalid response chart bin')
            entries[(last, node)] = 0; occupied.add((last, node, face, u, v))
        pairs = sorted(entries); entries = {pair:i for i, pair in enumerate(pairs)}
        estimated = 8*(physical_count+1)+4*len(pairs)+4*len(pairs)*6*bins*bins
        estimated += len(occupied)*(bins+4*bins*bins)
        if estimated > max_index_bytes: raise ValueError('Dispatch index exceeds byte budget')
        if len(bound.edge_starts) >= np.iinfo(np.int32).max: raise ValueError('Trie state index exceeds int32')
        counts = np.zeros(physical_count, dtype=np.int64)
        for last, _ in pairs: counts[last] += 1
        self.atom_starts = readonly(np.r_[0, np.cumsum(counts)].astype(np.int64))
        self.entry_nodes = readonly(np.array([node for _, node in pairs], dtype=np.int32))
        positions = np.full((len(pairs), 6, bins, bins), -1, dtype=np.int32)
        directions = np.full((len(occupied), bins, bins), -1, dtype=np.int32)
        position_indices = {key:i for i, key in enumerate(sorted(occupied))}
        for (last, node, face, u, v), row in position_indices.items(): positions[entries[(last, node)], face, u, v] = row
        for key, cell in bound.table.items():
            node, last, face, u, v, mu, phi = key
            directions[position_indices[(last, node, face, u, v)], mu, phi] = bound.roots[cell]
        self.position_rows = readonly(positions); self.direction_roots = readonly(directions)
        self.cosine_live = readonly(np.any(directions >= 0, axis=2))
        arrays = (self.atom_starts, self.entry_nodes, self.position_rows, self.cosine_live, self.direction_roots)
        self.receipt = dict(index_build_s=time.perf_counter()-tick, index_bytes=sum(a.nbytes for a in arrays),
            prepared_atom_node_pairs=len(pairs), physical_atoms_with_responses=int(np.count_nonzero(counts)),
            occupied_surface_bins=len(occupied), active_cells=len(bound.table),
            additional_index_over_existing_binding=True, parent_before_child=True, response_payload_unchanged=True)

    def trace(self, source, max_bounces=128, verify_owned=False, mode='indexed'):
        modes = {'off':0, 'hierarchy':1, 'atom':2, 'indexed':3}
        if mode not in modes: raise ValueError('Unknown dispatch mode')
        if not isinstance(max_bounces, (int, np.integer)) or max_bounces < 1 or np.any(source.initial_bounces > 1):
            raise ValueError('Positive cap and uncollided/first-reflected source required')
        if np.any(source.last_atom < -1) or np.any(source.last_atom >= len(self.model.spheres.radii)): raise ValueError('Invalid collider identity')
        b = self.bound; t = self.model.tree; spheres = self.model.spheres; tick = time.perf_counter()
        p, d, escaped, unresolved, counts, paths, counters, stats, hits, reused_mass, dispatch = trace_dispatch(
            source.origins, source.directions, source.last_atom, source.initial_bounces, source.weights, max_bounces,
            b.atlas.bins, b.table, b.roots, b.edge_starts, b.edge_atoms, b.edge_next, b.active, b.rotations,
            t.lo, t.hi, t.left, t.right, t.end, t.parent, t.leaf_group, t.leaf_node, t.groups, t.offsets, t.ids,
            spheres.centers, spheres.radii, self.atom_starts, self.entry_nodes, self.position_rows, self.cosine_live,
            self.direction_roots, verify_owned, modes[mode])
        labels = ('lookups', 'cell_hits', 'candidate_sphere_tests', 'accepted_prefixes', 'replayed_collisions',
            'multiple_prefixes', 'foreign_searches', 'foreign_interruptions', 'owned_checks', 'owned_interruptions',
            'no_candidate', 'cap_stops', 'reused_rays', 'partial_prefixes')
        dispatch_labels = ('physical_departures', 'no_atom_response', 'node_visits', 'surface_rejections',
            'cosine_rejections', 'azimuth_evaluations', 'indexed_cell_hits')
        return EHSSResult(p, d, escaped, unresolved, counts, paths, source,
            dict(query_s=time.perf_counter()-tick, box_tests=int(counters[0]), sphere_tests=int(counters[1]),
                responses=dict(zip(labels, map(int, stats))), dispatch=dict(zip(dispatch_labels, map(int, dispatch))),
                reused_ray_mass=reused_mass, node_response_hits=hits.tolist(), verify_owned=bool(verify_owned),
                dispatch_mode=mode, backend='unchanged sparse responses with collider and chart occupancy dispatch'))
