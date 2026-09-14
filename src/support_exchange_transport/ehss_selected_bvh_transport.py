"""Optional certified subtree responses on the existing physical BVH.

No field grid or second geometry hierarchy is built. Dispatch contains only
active response nodes for the preceding atom. Stored itineraries replace owned
search, not reflection arithmetic; current foreign geometry is checked on each
leg. The direct and enabled controls run the same physical traversal kernel.
"""
import time
from collections import Counter
import numpy as np
from numba import njit
from .ehss_guarded_intrinsic_response import (
    nearest_world, reflect, replay_guarded, GuardedIntrinsicLibrary)
from .ehss_response import empty_responses, node_geometry, seed_itinerary
from .ehss_state import EHSSResult


@njit(cache=False)
def trace_selected(origins, directions, last_atoms, initial, cap, enabled,
                   dispatch_starts, dispatch_nodes, responses,
                   lo, hi, left, right, end, leaf_group, offsets, ids, world, radii):
    p, d, counts = origins.copy(), directions.copy(), initial.copy()
    history = np.full((len(p), cap), -1, dtype=np.int32)
    escaped = np.zeros(len(p), dtype=np.bool_)
    unresolved = escaped.copy()
    counters, stats = np.zeros(2, dtype=np.int64), np.zeros(11, dtype=np.int64)
    hits = np.zeros(len(left), dtype=np.int64)
    dispatch_checks = 0
    used = np.zeros(len(p), dtype=np.bool_)
    for ray in range(len(p)):
        last, count = last_atoms[ray], counts[ray]
        if initial[ray] == 1:
            history[ray, 0] = last
        while True:
            skip = -1
            if enabled and last >= 0:
                for k in range(dispatch_starts[last], dispatch_starts[last+1]):
                    node = dispatch_nodes[k]
                    dispatch_checks += 1
                    hit, last, count, skip = replay_guarded(
                        node, p[ray], d[ray], last, count, cap, history[ray], responses,
                        lo, hi, left, right, end, leaf_group, offsets, ids, world, radii, counters, stats)
                    if hit:
                        hits[node] += 1
                        used[ray] = True
                        break
            atom, distance = nearest_world(0, p[ray], d[ray], last, np.inf, skip,
                lo, hi, left, right, end, leaf_group, offsets, ids, world, radii, counters)
            if atom < 0:
                escaped[ray] = True
                break
            if count >= cap:
                unresolved[ray] = True
                break
            reflect(p[ray], d[ray], atom, distance, world, radii)
            history[ray, count] = atom
            count += 1
            last = atom
        counts[ray] = count
    return p, d, escaped, unresolved, counts, history, counters, stats, hits, dispatch_checks, used


class SelectedBVHTransport:
    """Bind once per immutable model snapshot; enabled/off share that exact BVH."""
    def __init__(self, model, library=None):
        start = time.perf_counter()
        self.model = model
        tree = model.tree
        self.responses, packing = (empty_responses(len(tree.left)), dict(active_cells=0, inactive_cells=0)) \
            if library is None else library.pack(tree)
        starts, _, _, lasts = self.responses[:4]
        nodes_by_atom = [set() for _ in model.spheres.radii]
        for node in range(len(tree.left)):
            for cell in range(starts[node], starts[node+1]):
                atom = int(lasts[cell])
                if atom < 0:
                    raise ValueError('Selected dispatch requires a known preceding atom')
                leaf = tree.leaf_node[tree.groups[atom]]
                if not node <= leaf < tree.end[node]:
                    raise ValueError('Response node must own its preceding atom')
                nodes_by_atom[atom].add(node)
        offsets, nodes = [0], []
        for selected in nodes_by_atom:
            nodes.extend(sorted(selected))  # preorder: parents before children
            offsets.append(len(nodes))
        self.starts = np.asarray(offsets, dtype=np.int64)
        self.nodes = np.asarray(nodes, dtype=np.int64)
        self.receipt = dict(bind_s=time.perf_counter()-start, **packing,
            dispatch_entries=len(nodes), selected_nodes=len(set(nodes)),
            response_bytes=sum(a.nbytes for a in self.responses),
            dispatch_bytes=self.starts.nbytes+self.nodes.nbytes,
            same_physical_bvh=True, field_compiled=False, boundary_mesh=False)

    def trace(self, source, cap=64, enabled=True):
        if type(cap) is not int or cap < 1 or np.any(source.initial_bounces > 1):
            raise ValueError('Positive cap and uncollided/first-reflected source required')
        if np.any(source.last_atom < -1) or np.any(source.last_atom >= len(self.model.spheres.radii)):
            raise ValueError('Invalid source atom')
        tick = time.perf_counter()
        t, s = self.model.tree, self.model.spheres
        p, d, escaped, tail, counts, history, counters, stats, hits, checks, used = trace_selected(
            source.origins, source.directions, source.last_atom, source.initial_bounces, cap, enabled,
            self.starts, self.nodes, self.responses, t.lo, t.hi, t.left, t.right, t.end,
            t.leaf_group, t.offsets, t.ids, s.centers, s.radii)
        labels = ('cell_checks', 'attempts', 'accepted_prefixes', 'foreign_interruptions', 'foreign_searches',
                  'replay_failures', 'replay_sphere_tests', 'cap_fallbacks', 'replayed_collisions',
                  'multiple_prefixes', 'partial_prefixes')
        return EHSSResult(p, d, escaped, tail, counts, history, source, dict(
            query_s=time.perf_counter()-tick, box_tests=int(counters[0]),
            sphere_tests=int(counters[1]+stats[6]), search_sphere_tests=int(counters[1]),
            replay_sphere_tests=int(stats[6]), responses=dict(zip(labels, map(int, stats))),
            dispatch_checks=int(checks), node_response_hits=hits.tolist(),
            response_rays=int(used.sum()), response_mass=float(source.weights[used].sum()),
            same_physical_bvh=True, enabled=bool(enabled)))


def prepare_subtrees(model, sources, max_atoms=64, max_attempts=16, max_candidates=128):
    """Training-only finite budget; prefer longer owned routes in internal nodes.

The work score is a heuristic, not a measured speed gate. Empty routes are not
compiled. Fixed identity/geometry certificates govern later MD activation.
"""
    tick = time.perf_counter()
    t = model.tree
    library = GuardedIntrinsicLibrary(residual_allowance=1e-5)
    geometry = {}
    candidates = []
    records_seen = 0
    for source in sources:
        model.trace(source, 64, record_capacity=2048)
        lasts, states = model.last_records
        for last, state in zip(lasts, states):
            records_seen += 1
            node = int(t.parent[t.leaf_node[t.groups[last]]])
            while node >= 0:
                if node not in geometry:
                    geometry[node] = node_geometry(t, node)
                atoms, _, centers, rotation, translation = geometry[node]
                if len(atoms) > max_atoms:
                    break
                route = seed_itinerary((state[:3]-translation)@rotation, state[3:]@rotation,
                    int(np.searchsorted(atoms, last)), centers, t.radii[atoms], 16)
                if route is not None and len(route):
                    candidates.append((len(route)*(len(atoms)-1), node, int(last), state.copy(), len(route)))
                    break
                node = int(t.parent[node])
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break
    candidates.sort(key=lambda item: -item[0])
    attempts = 0
    selected = Counter()
    seen = set()
    for score, node, last, state, length in candidates:
        key = (node, last, state.tobytes())
        if key in seen or selected[node] >= 2:
            continue
        seen.add(key)
        cell = library.compile(t, node, state[:3], state[3:], last,
            position_width=.03, direction_width=.03, max_shrinks=8, max_bounces=16)
        attempts += 1
        if cell is not None:
            selected[node] += 1
        if attempts >= max_attempts:
            break
    receipt = dict(prepare_s=time.perf_counter()-tick, training_records=records_seen,
        candidate_routes=len(candidates), attempts=attempts, cells=len(library.cells),
        parent_cells=sum(t.leaf_group[c.node] < 0 for c in library.cells),
        selected_nodes=len(selected), stored_collisions=sum(len(c.itinerary) for c in library.cells),
        multi_collision_cells=sum(len(c.itinerary) >= 2 for c in library.cells),
        max_atoms=max_atoms, max_attempts=max_attempts, max_candidates=max_candidates,
        selection='training-only route length times owned candidate count; heuristic',
        projection=False, certificates=library.compile_receipts)
    return library, receipt
