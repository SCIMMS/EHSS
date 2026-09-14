"""Replace owner-local compiled tries/charts in reusable storage arenas.

The physics and indexed lookup kernel are unchanged. Live entry gathering and
atlas validation remain global. Growing an arena copies its existing storage;
ordinary replacements only write selected owners. Sequential use only.
Prepared input atlases are borrowed and must not be mutated after replacement.
"""
from copy import copy, deepcopy
from types import SimpleNamespace
import time
import numpy as np
from numba import types
from numba.typed import Dict
from .ehss_surface_response import KEY
from .ehss_sparse_path_response import readonly
from .ehss_response import ResponseLibrary, node_geometry
from .ehss_response_dispatch import IndexedResponse
from .ehss_batched_node_response import BatchedNodeResponses, PreparedNodeFits


class Arena:
    """Best-fit free blocks with coalescing and explicit growth accounting."""
    def __init__(self, specs):
        self.specs = specs
        self.arrays = {k: np.zeros((0, *shape), dtype=dtype) for k, (shape, dtype) in specs.items()}
        self.free = []
        self.capacity = self.copied_bytes = self.growths = 0

    def allocate(self, size):
        if size == 0:
            return (0, 0)
        choices = [(length, start, i) for i, (start, length) in enumerate(self.free) if length >= size]
        if not choices:
            old = self.capacity
            capacity = max(16, old * 2, old + size)
            for key, array in self.arrays.items():
                new = np.zeros((capacity, *array.shape[1:]), dtype=array.dtype)
                new[:old] = array
                self.copied_bytes += array.nbytes
                self.arrays[key] = new
            self.capacity = capacity
            self.growths += 1
            self.release((old, capacity - old))
            return self.allocate(size)
        length, start, i = min(choices)
        self.free.pop(i)
        if length > size:
            self.free.append((start + size, length - size))
        return start, size

    def release(self, block):
        if not block[1]:
            return
        ordered = sorted(self.free + [block])
        merged = []
        for start, length in ordered:
            if merged and merged[-1][0] + merged[-1][1] == start:
                prev, count = merged.pop()
                merged.append((prev, count + length))
            else:
                merged.append((start, length))
        self.free = merged


def owner_cells(atlas):
    cells = {}
    for i, key in enumerate(atlas.keys):
        cells.setdefault(int(key[0]), []).append(i)
    return cells


def compile_owner(model, atlas, node, indices):
    """Compile one owner without creating a global table or dispatch index."""
    keys = [tuple(int(v) for v in atlas.keys[i]) for i in indices]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate response key')
    allowed = set(map(int, atlas.nodes[node].atoms))
    actual, _, _, _, _ = node_geometry(model.tree, node)
    if not np.array_equal(actual, atlas.nodes[node].atoms):
        raise ValueError('Stored atom ownership changed')
    bins = atlas.bins
    children = []
    roots = []
    for key, index in zip(keys, indices):
        if key[1] not in allowed:
            raise ValueError('Collider outside response owner')
        if not 0 <= key[2] < 6 or any(not 0 <= v < bins for v in key[3:]):
            raise ValueError('Invalid response chart bin')
        roots.append(len(children))
        children.append({})
        for route in atlas.routes[index]:
            state = roots[-1]
            for atom in route:
                if atom not in allowed:
                    raise ValueError('Route leaves its response owner')
                if atom not in children[state]:
                    children[state][atom] = len(children)
                    children.append({})
                state = children[state][atom]
    starts = [0]
    atoms = []
    successors = []
    for edges in children:
        for atom in sorted(edges):
            atoms.append(atom)
            successors.append(edges[atom])
        starts.append(len(atoms))
    lasts = np.array(sorted({k[1] for k in keys}), dtype=np.int64)
    entry = {int(atom): i for i, atom in enumerate(lasts)}
    occupied = sorted({k[1:5] for k in keys})
    rows = {key: i for i, key in enumerate(occupied)}
    positions = np.full((len(lasts), 6, bins, bins), -1, dtype=np.int32)
    directions = np.full((len(rows), bins, bins), -1, dtype=np.int32)
    for (last, face, u, v), row in rows.items():
        positions[entry[last], face, u, v] = row
    for key, root in zip(keys, roots):
        directions[rows[key[1:5]], key[5], key[6]] = root
    return SimpleNamespace(keys=keys, roots=np.array(roots, dtype=np.int64),
        starts=np.array(starts, dtype=np.int64), atoms=np.array(atoms, dtype=np.int64),
        successors=np.array(successors, dtype=np.int64), lasts=readonly(lasts),
        positions=positions, directions=directions)


class OwnerPayloadResponses(BatchedNodeResponses):
    """Local replacement with the established batched fit and trace policies."""
    def __init__(self, model, atlas):
        tick = time.perf_counter()
        self._validate(model, atlas)
        self.atlas = deepcopy(atlas)
        bins = atlas.bins
        self.states = Arena({'starts': ((), np.int64)})
        self.edges = Arena({'atoms': ((), np.int64), 'next': ((), np.int64)})
        self.rows = Arena({'directions': ((bins, bins), np.int32), 'cosine': ((bins,), np.bool_)})
        self.chunks = {}
        self.bound = SimpleNamespace(model=model, atlas=self.atlas,
            table=Dict.empty(KEY, types.int64),
            active=readonly(np.zeros(len(model.tree.left), dtype=bool)),
            rotations=readonly(np.tile(np.eye(3), (len(model.tree.left), 1, 1))))
        self.index = IndexedResponse.__new__(IndexedResponse)
        self.index.bound = self.bound
        self.index.model = model
        for node, indices in owner_cells(self.atlas).items():
            self._install(node, compile_owner(model, self.atlas, node, indices))
        self._publish()
        self._catalog(model)
        self.layout = 'compact'
        self.ever_active = np.zeros(len(model.tree.left), dtype=bool)
        self.fits = {}
        self.previous = None
        self.fit_plan = PreparedNodeFits(model, self.atlas)
        self.update(model)
        self.build_receipt = dict(total_initial_s=time.perf_counter()-tick, **self.storage())

    @staticmethod
    def _validate(model, atlas):
        if not atlas.prepared or len(atlas.keys) != len(atlas.routes):
            raise ValueError('Prepared atlas required')
        if ResponseLibrary.topology_key(model.tree) != atlas.topology:
            raise ValueError('Physical ownership topology changed')
        if not np.isfinite(atlas.residual_tolerance) or atlas.residual_tolerance < 0:
            raise ValueError('Invalid tolerance')
        if not isinstance(atlas.bins, (int, np.integer)) or atlas.bins < 1:
            raise ValueError('Positive bins required')
        if set(map(int, atlas.keys[:, 0])) != set(atlas.nodes):
            raise ValueError('Owner catalog must match cells')

    def _install(self, node, chunk):
        # All semantic validation and compilation precedes this mutation.
        old = self.chunks.pop(node, None)
        if old is not None:
            for key in old.keys:
                del self.bound.table[key]
            for arena, block in zip((self.states, self.edges, self.rows), old.blocks):
                arena.release(block)
        if chunk is None:
            return
        sb = self.states.allocate(len(chunk.starts))  # includes private sentinel
        eb = self.edges.allocate(len(chunk.atoms))
        rb = self.rows.allocate(len(chunk.directions))
        s, e, r = sb[0], eb[0], rb[0]
        if s + len(chunk.starts) >= np.iinfo(np.int32).max or r + len(chunk.directions) >= np.iinfo(np.int32).max:
            raise ValueError('Arena exceeds int32 chart index')
        self.states.arrays['starts'][s:s+sb[1]] = chunk.starts + e
        self.edges.arrays['atoms'][e:e+eb[1]] = chunk.atoms
        self.edges.arrays['next'][e:e+eb[1]] = chunk.successors + s
        directions = chunk.directions.copy()
        directions[directions >= 0] += s
        self.rows.arrays['directions'][r:r+rb[1]] = directions
        self.rows.arrays['cosine'][r:r+rb[1]] = np.any(directions >= 0, axis=2)
        positions = chunk.positions.copy()
        positions[positions >= 0] += r
        for key, root in zip(chunk.keys, chunk.roots):
            self.bound.table[key] = int(root + s)
        # Retain the installed payload, not another copy of local trie arrays.
        self.chunks[node] = SimpleNamespace(keys=tuple(chunk.keys), lasts=chunk.lasts,
            positions=readonly(positions), blocks=(sb, eb, rb))

    def _publish(self):
        def view(array):
            return readonly(array.view())
        self.bound.edge_starts = view(self.states.arrays['starts'])
        self.bound.edge_atoms = view(self.edges.arrays['atoms'])
        self.bound.edge_next = view(self.edges.arrays['next'])
        if not hasattr(self.bound, 'roots') or len(self.bound.roots) != self.states.capacity:
            self.bound.roots = readonly(np.arange(self.states.capacity, dtype=np.int64))
        self.index.direction_roots = view(self.rows.arrays['directions'])
        self.index.cosine_live = view(self.rows.arrays['cosine'])

    def _catalog(self, model):
        """Gather metadata only; trie/chart contents stay in their owner slots."""
        bins = self.atlas.bins
        ordered = [(int(last), node, i) for node, chunk in self.chunks.items() for i, last in enumerate(chunk.lasts)]
        ordered.sort()  # same parent-before-child ordering as IndexedResponse
        self.full_last = readonly(np.array([last for last, _, _ in ordered], dtype=np.int64))
        self.full_nodes = readonly(np.array([node for _, node, _ in ordered], dtype=np.int32))
        self.full_positions = readonly(np.array([self.chunks[node].positions[i] for _, node, i in ordered], dtype=np.int32).reshape(-1, 6, bins, bins))
        counts = np.bincount(self.full_last, minlength=len(model.spheres.radii))
        self.full_starts = readonly(np.r_[0, np.cumsum(counts)].astype(np.int64))
        self.node_entries = {node: np.flatnonzero(self.full_nodes == node) for node in self.atlas.nodes}
        self.node_cell_count = np.bincount(self.atlas.keys[:, 0].astype(int), minlength=len(model.tree.left))
        self.dependencies = [set() for _ in model.spheres.radii]
        for node, stored in self.atlas.nodes.items():
            for atom in stored.atoms:
                self.dependencies[atom].add(node)

    def storage(self):
        arenas = (self.states, self.edges, self.rows)
        return dict(arena_bytes=sum(a.nbytes for arena in arenas for a in arena.arrays.values()),
            growth_copied_bytes=sum(a.copied_bytes for a in arenas), arena_growths=sum(a.growths for a in arenas),
            arena_capacities=[a.capacity for a in arenas],
            arena_free_slots=[sum(length for _, length in a.free) for a in arenas])

    def replace(self, model, atlas, selected_nodes):
        tick = time.perf_counter()
        self._validate(model, atlas)
        if model is not self.bound.model:
            raise ValueError('Update current geometry before replacing responses')
        for attr in ('bins', 'max_cells', 'max_routes_per_cell', 'max_atoms', 'max_route', 'residual_tolerance'):
            if getattr(atlas, attr) != getattr(self.atlas, attr):
                raise ValueError('Response policy changed')
        selected = set(map(int, selected_nodes))
        if any(n < 0 or n >= len(model.tree.left) for n in selected):
            raise ValueError('Invalid selected owner')
        before_cells = owner_cells(self.atlas)
        after_cells = owner_cells(atlas)
        retained = (set(before_cells) | set(after_cells)) - selected
        # Full metadata validation is charged, without recompiling retained data.
        for node in retained:
            if node not in before_cells or node not in after_cells:
                raise ValueError('Unselected owner changed')
            a, b = before_cells[node], after_cells[node]
            old, new = self.atlas.nodes[node], atlas.nodes[node]
            if (not np.array_equal(self.atlas.keys[a], atlas.keys[b]) or
                tuple(self.atlas.routes[i] for i in a) != tuple(atlas.routes[i] for i in b) or
                old.anchor != new.anchor or any(not np.array_equal(getattr(old, field), getattr(new, field)) for field in ('atoms', 'centers', 'radii'))):
                raise ValueError('Unselected owner changed')
        validation_s = time.perf_counter() - tick
        if not selected:
            return dict(replace_s=time.perf_counter()-tick, selected_nodes=[], compiled_owners=0,
                retained_owners=len(retained), validation_s=validation_s, no_op=True)
        start = time.perf_counter()
        compiled = {node: compile_owner(model, atlas, node, after_cells[node]) if node in after_cells else None for node in selected}
        # Only size buckets containing changed references are prepared again.
        sizes = {len(a.nodes[n].atoms) for a in (self.atlas, atlas) for n in selected if n in a.nodes}
        subset = copy(atlas)
        subset.nodes = {n: stored for n, stored in atlas.nodes.items() if len(stored.atoms) in sizes}
        partial_plan = PreparedNodeFits(model, subset)
        batches = [b for b in self.fit_plan.batches if b.atoms.shape[1] not in sizes] + partial_plan.batches
        prepare_s = time.perf_counter()-start
        storage_before = self.storage()
        start = time.perf_counter()
        for node in sorted(selected):
            self._install(node, compiled[node])
        self.atlas = copy(atlas)
        self.atlas.nodes = atlas.nodes.copy()
        self.bound.atlas = self.atlas
        self.fit_plan.batches = sorted(batches, key=lambda b: b.atoms.shape[1])
        self._publish()
        write_s = time.perf_counter()-start
        start = time.perf_counter()
        self._catalog(model)
        catalog_s = time.perf_counter()-start
        # Force one compact gather, but fit only the selected new references.
        start = time.perf_counter()
        active = self.bound.active.copy()
        rotations = self.bound.rotations.copy()
        for node in selected:
            active[node] = False
            rotations[node] = np.eye(3)
            self.fits.pop(node, None)
        evaluated, _ = self.fit_plan.evaluate(model, selected & set(atlas.nodes))
        labels = ('native_already_aligned', 'stored_node_fit', 'native_lower_component_error')
        for node, (rotation, error, rank, method, native, match) in evaluated.items():
            valid = match and error <= atlas.residual_tolerance + 1e-10
            active[node] = valid and self.node_cell_count[node] > 0
            rotations[node] = rotation if valid else np.eye(3)
            self.fits[node] = dict(node=node, valid=bool(valid), radii_match=match, rank=rank,
                method=labels[method], native_component_error_A=native, fitted_component_error_A=error)
        self.bound.active = readonly(active)
        self.bound.rotations = readonly(rotations)
        live = active[self.full_nodes]
        counts = np.bincount(self.full_last[live], minlength=len(model.spheres.radii))
        self.index.atom_starts = readonly(np.r_[0, np.cumsum(counts)].astype(np.int64))
        self.index.entry_nodes = readonly(self.full_nodes[live])
        self.index.position_rows = readonly(self.full_positions[live])
        self.ever_active |= active
        self.receipt = dict(active_nodes=int(active.sum()), active_cells=int(self.node_cell_count @ active))
        fit_gather_s = time.perf_counter()-start
        after = self.storage()
        self.replace_receipt = dict(replace_s=time.perf_counter()-tick, selected_nodes=sorted(selected),
            compiled_owners=sum(v is not None for v in compiled.values()), retained_owners=len(retained),
            validation_s=validation_s, local_prepare_s=prepare_s, arena_write_s=write_s,
            catalog_s=catalog_s, fit_gather_s=fit_gather_s,
            growth_copy_bytes=after['growth_copied_bytes']-storage_before['growth_copied_bytes'],
            global_entry_gather=True, global_trie_recompiled=False, **after)
        return self.replace_receipt
