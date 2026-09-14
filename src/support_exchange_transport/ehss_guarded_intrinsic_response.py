"""Body-frame owned responses with current foreign-collision arbitration.

An owned response need not have a foreign-free enclosing sphere. Its certified
internal itinerary is reusable under a common rigid pose. Each physical leg is
checked against the complementary hierarchy before commit. This preserves
physical anchors, full incoming direction tags, and cap survivors.
"""
from copy import copy
from dataclasses import replace
import time
import numpy as np
from numba import njit
from .ehss_support_tree import NodeAwareSupportTree, box_interval
from .ehss_reference import entry_distance
from .ehss_response import ResponseLibrary, empty_responses, node_geometry, seed_itinerary, certify_itinerary
from .ehss_state import EHSSResult


@njit(cache=False)
def nearest_world(node, o, d, last, limit, skip, lo, hi, left, right, end, leaf_group, offsets, ids, world, radii, counters):
    if skip >= 0 and skip <= node < end[skip]: return -1, np.inf
    counters[0] += 1
    near, far = box_interval(o, d, lo[node], hi[node])
    if near > limit or far < 0.: return -1, np.inf
    atom = -1; best = limit; group = leaf_group[node]
    if group >= 0:
        for k in range(offsets[group], offsets[group+1]):
            wanted = ids[k]
            if wanted == last: continue
            counters[1] += 1
            distance = entry_distance(o, d, world[wanted], radii[wanted])
            if distance < best or (distance == best and np.isfinite(distance) and (atom < 0 or wanted < atom)):
                atom = wanted; best = distance
    else:
        atom, best = nearest_world(left[node], o, d, last, best, skip, lo, hi, left, right, end, leaf_group,
            offsets, ids, world, radii, counters)
        other, distance = nearest_world(right[node], o, d, last, min(limit, best), skip, lo, hi, left, right, end,
            leaf_group, offsets, ids, world, radii, counters)
        if other >= 0 and (atom < 0 or distance < best or (distance == best and other < atom)):
            atom = other; best = distance
    return atom, best


@njit(cache=True)
def reflect(o, d, atom, distance, world, radii):
    o[:] = o+distance*d
    normal = o-world[atom]; normal /= np.sqrt(np.dot(normal, normal))
    o[:] = world[atom]+radii[atom]*normal
    d[:] = d-2*np.dot(d, normal)*normal
    d[:] /= np.sqrt(np.dot(d, d))


@njit(cache=False)
def replay_guarded(node, o, d, last, count, cap, history, responses,
                   lo, hi, left, right, end, leaf_group, offsets, ids, world, radii, counters, stats):
    starts, low, high, last_keys, path_offsets, paths, rotations, translations, charts = responses
    if starts[node] == starts[node+1]: return False, last, count, -1
    local_o = (o-translations[node])@rotations[node]; local_d = d@rotations[node]
    state = np.empty(6)
    for cell in range(starts[node], starts[node+1]):
        stats[0] += 1
        if last_keys[cell] != last: continue
        axis = int(charts[cell, 0])
        if axis >= 0:
            if charts[cell, 1]*local_d[axis] <= 1e-12: continue
            k = 0
            for coordinate in range(3):
                if coordinate != axis:
                    slope = local_d[coordinate]/local_d[axis]
                    state[k] = local_o[coordinate]+(charts[cell, 2]-local_o[axis])*slope
                    state[k+2] = slope; k += 1
            state[4:] = 0.
        else:
            state[:3] = local_o; state[3:] = local_d
        if np.any(state < low[cell]) or np.any(state > high[cell]): continue
        first = path_offsets[cell]; stop = path_offsets[cell+1]
        if stop == first: continue
        if count+stop-first > cap: stats[7] += 1; continue
        stats[1] += 1
        p = o.copy(); v = d.copy(); previous = last; done = 0; interrupted = False; failed = False
        for k in range(first, stop):
            atom = paths[k]
            if atom == previous: failed = True; break
            distance = entry_distance(p, v, world[atom], radii[atom]); stats[6] += 1
            if not np.isfinite(distance): failed = True; break
            stats[4] += 1
            competitor, before = nearest_world(0, p, v, previous, distance, node, lo, hi, left, right, end,
                leaf_group, offsets, ids, world, radii, counters)
            if competitor >= 0 and (before < distance or (before == distance and competitor < atom)):
                interrupted = True; stats[3] += 1; break
            reflect(p, v, atom, distance, world, radii)
            previous = atom; done += 1
        if failed: stats[5] += 1; continue
        if done == 0: continue
        # All committed legs passed owned itinerary validity and foreign order.
        # A foreign interruption commits only the prefix and returns to root.
        o[:] = p; d[:] = v
        for k in range(done): history[count+k] = paths[first+k]
        stats[2] += 1; stats[8] += done
        if done >= 2: stats[9] += 1
        if interrupted: stats[10] += 1
        return True, previous, count+done, -1 if interrupted else node
    return False, last, count, -1


@njit(cache=False)
def trace_guarded(origins, directions, last_atoms, initial, cap, lo, hi, left, right, end, parent,
                  leaf_group, leaf_node, groups, offsets, ids, world, radii, responses, record_capacity):
    p = origins.copy(); d = directions.copy(); counts = initial.copy()
    history = np.full((len(p), cap), -1, dtype=np.int32)
    escaped = np.zeros(len(p), dtype=np.bool_); unresolved = escaped.copy()
    counters = np.zeros(2, dtype=np.int64); stats = np.zeros(11, dtype=np.int64)
    hits = np.zeros(len(left), dtype=np.int64)
    record_last = np.empty(record_capacity, dtype=np.int64); record_states = np.empty((record_capacity, 6)); recorded = 0
    chain = np.empty(len(left), dtype=np.int64)
    for ray in range(len(p)):
        last = last_atoms[ray]; count = counts[ray]
        if initial[ray] == 1: history[ray, 0] = last
        while True:
            if last >= 0 and recorded < record_capacity:
                record_last[recorded] = last; record_states[recorded, :3] = p[ray]; record_states[recorded, 3:] = d[ray]
                recorded += 1
            skip = -1
            if last >= 0:
                length = 0; node = leaf_node[groups[last]]
                while node >= 0:
                    chain[length] = node; length += 1; node = parent[node]
                # Prefer a prepared parent response over its prepared children.
                for k in range(length-1, -1, -1):
                    node = chain[k]
                    hit, last, count, skip = replay_guarded(node, p[ray], d[ray], last, count, cap, history[ray], responses,
                        lo, hi, left, right, end, leaf_group, offsets, ids, world, radii, counters, stats)
                    if hit: hits[node] += 1; break
            atom, distance = nearest_world(0, p[ray], d[ray], last, np.inf, skip, lo, hi, left, right, end,
                leaf_group, offsets, ids, world, radii, counters)
            if atom < 0: escaped[ray] = True; break
            if count >= cap: unresolved[ray] = True; break
            reflect(p[ray], d[ray], atom, distance, world, radii)
            history[ray, count] = atom; count += 1; last = atom
        counts[ray] = count
    return p, d, escaped, unresolved, counts, history, counters, stats, hits, record_last[:recorded], record_states[:recorded]


class GuardedIntrinsicLibrary(ResponseLibrary):
    """Retain the owned-geometry certificate; replace foreign-free eligibility."""
    def __init__(self, residual_allowance=.001):
        super().__init__()
        if not np.isfinite(residual_allowance) or residual_allowance < 0:
            raise ValueError('Finite nonnegative body-coordinate residual allowance required')
        self.residual_allowance = float(residual_allowance)

    def compile(self, *args, **kwargs):
        if kwargs.get('input_chart', 'cartesian') != 'cartesian':
            raise ValueError('Guarded residual compiler currently uses physical Cartesian input enclosures')
        start = time.perf_counter(); cell = super().compile(*args, **kwargs)
        if cell is None: return None
        pad = max(cell.geometry_pad, 4*self.residual_allowance)
        width = cell.halfwidth.copy()
        mapped = np.flatnonzero(cell.atom_ids == cell.last)
        local_last = int(mapped[0]) if len(mapped) else -1
        route = np.array([int(np.flatnonzero(cell.atom_ids == a)[0]) for a in cell.itinerary], dtype=np.int64)
        record = self.compile_receipts[-1]; valid = False
        for shrink in range(7):
            valid, tests, reason = certify_itinerary(cell.seed[:3], cell.seed[3:], width, local_last,
                cell.centers, cell.radii, route, pad)
            record['interval_tests'] += tests
            if valid: break
            width *= .5
        guard = 1e-11*(1.+np.abs(cell.seed))
        if not valid or np.any(width <= 2*guard):
            self.cells.pop(); record.update(accepted=False, reason='residual envelope: '+reason,
                compile_s=time.perf_counter()-start, residual_allowance=self.residual_allowance)
            return None
        low = cell.seed-width+guard; high = cell.seed+width-guard
        for value in (low, high, width): value.setflags(write=False)
        cell = replace(cell, geometry_pad=pad, low=low, high=high, halfwidth=width,
            interval_tests=record['interval_tests'])
        self.cells[-1] = cell
        record.update(accepted=True, reason='owned response with residual envelope',
            compile_s=time.perf_counter()-start, residual_allowance=self.residual_allowance,
            residual_shrinks=shrink, halfwidth=width.tolist())
        return cell

    def pack(self, tree, mode='all'):
        view = copy(tree)
        view.exclusive = np.ones_like(tree.exclusive)
        arrays, receipt = super().pack(view, mode)
        receipt.update(foreign_geometry_checked_per_physical_leg=True,
                       nonexclusive_cells=sum(not tree.exclusive[c.node] for c in self.cells))
        return arrays, receipt


class GuardedSupportHierarchy:
    def __init__(self, geometry):
        tick = time.perf_counter(); self.geometry = geometry; spheres = geometry.spheres()
        rotations = np.array([b.rotation for b in geometry.blocks]); translations = np.array([b.translation for b in geometry.blocks])
        local = np.empty_like(spheres.centers)
        for group, block in enumerate(geometry.blocks):
            atoms = block.intrinsic.atoms
            local[atoms] = (spheres.centers[atoms]-translations[group])@rotations[group]
        self.tree = NodeAwareSupportTree(local, spheres.radii, geometry.groups, rotations, translations)
        # Reference events use the exact materialized coordinates, not a second
        # body/world roundtrip. Refit conservative world bounds from those.
        self.spheres = spheres; self.tree.world = spheres.centers; self.tree.spheres = spheres
        pad = 1e-10*max(1., float(spheres.radii.max()))
        for group in range(self.tree.n_groups):
            atoms = self.tree.ids[self.tree.offsets[group]:self.tree.offsets[group+1]]; node = self.tree.leaf_node[group]
            self.tree.lo[node] = np.min(spheres.centers[atoms]-spheres.radii[atoms, None], axis=0)-pad
            self.tree.hi[node] = np.max(spheres.centers[atoms]+spheres.radii[atoms, None], axis=0)+pad
        for node in range(len(self.tree.left)-1, -1, -1):
            if self.tree.left[node] >= 0:
                self.tree.lo[node] = np.minimum(self.tree.lo[self.tree.left[node]], self.tree.lo[self.tree.right[node]])
                self.tree.hi[node] = np.maximum(self.tree.hi[self.tree.left[node]], self.tree.hi[self.tree.right[node]])
        self.receipt = dict(build_s=time.perf_counter()-tick, nodes=len(self.tree.left), physical_blocks=len(geometry.blocks),
            exact_materialized_geometry=True, bounds_and_overlap_refresh_global=True)

    def trace(self, source, max_bounces=128, library=None, record_capacity=0):
        if not isinstance(max_bounces, (int, np.integer)) or max_bounces < 1 or np.any(source.initial_bounces > 1):
            raise ValueError('Positive cap and uncollided/first-reflected input required')
        if np.any(source.last_atom < -1) or np.any(source.last_atom >= len(self.spheres.radii)):
            raise ValueError('Invalid physical collider identity')
        if not isinstance(record_capacity, (int, np.integer)) or record_capacity < 0: raise ValueError('Nonnegative record capacity required')
        tree = self.tree
        responses, packing = (empty_responses(len(tree.left)), {}) if library is None else library.pack(tree)
        tick = time.perf_counter()
        p, d, escaped, unresolved, counts, paths, counters, stats, hits, last, states = trace_guarded(
            source.origins, source.directions, source.last_atom, source.initial_bounces, max_bounces,
            tree.lo, tree.hi, tree.left, tree.right, tree.end, tree.parent, tree.leaf_group, tree.leaf_node,
            tree.groups, tree.offsets, tree.ids, self.spheres.centers, self.spheres.radii, responses, record_capacity)
        labels = ('cell_checks', 'attempts', 'accepted_prefixes', 'foreign_interruptions', 'foreign_searches',
                  'replay_failures', 'replay_sphere_tests', 'cap_fallbacks', 'replayed_collisions', 'multiple_prefixes', 'partial_prefixes')
        result = EHSSResult(p, d, escaped, unresolved, counts, paths, source,
            dict(query_s=time.perf_counter()-tick, box_tests=int(counters[0]), sphere_tests=int(counters[1]),
                responses=dict(zip(labels, map(int, stats))), node_response_hits=hits.tolist(), packing=packing,
                backend='node-aware owned responses with current foreign hierarchy arbitration'))
        self.last_records = (last, states)
        return result

    def prepare(self, sources, cells_per_node=2, max_atoms=24, max_attempts=160, position_width=.01, direction_width=.01):
        library = GuardedIntrinsicLibrary(); selected = {}; attempts = 0; records = 0
        start = time.perf_counter()
        for source in sources:
            self.trace(source, 128, record_capacity=12000)
            lasts, states = self.last_records; records += len(lasts)
            for last, state in zip(lasts, states):
                node = int(self.tree.leaf_node[self.tree.groups[last]])
                while node >= 0 and attempts < max_attempts:
                    atoms, _, centers, rotation, translation = node_geometry(self.tree, node)
                    if len(atoms) > max_atoms: break
                    if selected.get(node, 0) < cells_per_node:
                        local_last = int(np.flatnonzero(atoms == last)[0])
                        route = seed_itinerary((state[:3]-translation)@rotation, state[3:]@rotation,
                            local_last, centers, self.tree.radii[atoms], 16)
                        if route is not None and len(route):
                            attempts += 1
                            cell = library.compile(self.tree, node, state[:3], state[3:], int(last),
                                position_width=position_width, direction_width=direction_width, max_shrinks=10, max_bounces=16,
                                input_chart='cartesian')
                            if cell is not None: selected[node] = selected.get(node, 0)+1
                    node = int(self.tree.parent[node])
                if attempts >= max_attempts: break
            if attempts >= max_attempts: break
        self.prepare_receipt = dict(prepare_s=time.perf_counter()-start, attempts=attempts, records=records,
            cells=len(library.cells), parent_cells=sum(self.tree.leaf_group[c.node] < 0 for c in library.cells),
            nonexclusive_cells=sum(not self.tree.exclusive[c.node] for c in library.cells),
            multiple_routes=sum(len(c.itinerary) >= 2 for c in library.cells),
            cells_per_node=cells_per_node, max_atoms=max_atoms, max_attempts=max_attempts,
            initial_state='Known preceding physical collision; Cartesian enclosure of the physical boundary state.',
            compile_receipts=library.compile_receipts)
        return library
