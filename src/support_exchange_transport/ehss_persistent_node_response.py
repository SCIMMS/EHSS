"""Persistent response payload with per-node validity and reactivation.

The atlas is frozen at construction. Update fits dirty owners and switches their
dispatch entries; it never retrains or repacks their collision-prefix tries.
This is a sequential current-frame engine, not a collection of frame snapshots.
"""
from copy import deepcopy
from types import SimpleNamespace
import time
import numpy as np
from numba import types
from numba.typed import Dict
from .ehss_surface_response import KEY
from .ehss_sparse_path_response import readonly
from .ehss_response import ResponseLibrary, node_geometry
from .ehss_fitted_node_response import fit_node
from .ehss_response_dispatch import IndexedResponse


def pack_payload(model, atlas):
    """Compile every frozen cell once; geometry validity is applied separately."""
    table = Dict.empty(KEY, types.int64); children = []; roots = []
    for key, routes in zip(atlas.keys, atlas.routes):
        k = tuple(int(v) for v in key)
        if k in table: raise ValueError('Duplicate response key')
        if k[0] not in atlas.nodes: raise ValueError('Missing response owner')
        allowed = set(int(a) for a in atlas.nodes[k[0]].atoms)
        if k[1] not in allowed: raise ValueError('Collider outside response owner')
        table[k] = len(roots); roots.append(len(children)); children.append({})
        for route in routes:
            state = roots[-1]
            for atom in route:
                if atom not in allowed: raise ValueError('Route leaves its response owner')
                if atom not in children[state]:
                    children[state][atom] = len(children); children.append({})
                state = children[state][atom]
    starts = [0]; atoms = []; successors = []
    for edges in children:
        for atom in sorted(edges): atoms.append(atom); successors.append(edges[atom])
        starts.append(len(atoms))
    return SimpleNamespace(model=model, atlas=atlas, table=table,
        roots=readonly(np.array(roots,dtype=np.int64)), edge_starts=readonly(np.array(starts,dtype=np.int64)),
        edge_atoms=readonly(np.array(atoms,dtype=np.int64)), edge_next=readonly(np.array(successors,dtype=np.int64)),
        active=readonly(np.zeros(len(model.tree.left),dtype=bool)),
        rotations=readonly(np.tile(np.eye(3),(len(model.tree.left),1,1))))


class PersistentNodeResponses:
    """Reuse immutable trie/chart arrays; update current geometry and live entries.

    ``compact`` gathers the live collider/node entries only when validity changes.
    ``gated`` keeps all entries and changes only invalidated/reactivated position
    rows. Both modes call the existing physical indexed query kernel unchanged.
    Global identity/dirty scans and current hierarchy construction remain costs.
    """
    def __init__(self, model, atlas, layout='compact', max_index_bytes=64*1024*1024):
        tick = time.perf_counter()
        if layout not in ('compact','gated'): raise ValueError('Unknown live-entry layout')
        if not atlas.prepared or len(atlas.keys) != len(atlas.routes): raise ValueError('Prepared atlas required')
        if not np.isfinite(atlas.residual_tolerance) or atlas.residual_tolerance < 0: raise ValueError('Invalid tolerance')
        if ResponseLibrary.topology_key(model.tree) != atlas.topology: raise ValueError('Physical ownership topology changed')
        self.atlas = deepcopy(atlas); self.layout = layout
        self.bound = pack_payload(model, self.atlas); self.index = IndexedResponse(self.bound,max_index_bytes)
        self.full_starts = self.index.atom_starts; self.full_nodes = self.index.entry_nodes
        self.full_positions = self.index.position_rows
        self.full_last = readonly(np.repeat(np.arange(len(model.spheres.radii)),np.diff(self.full_starts)))
        self.node_entries = {node: np.flatnonzero(self.full_nodes==node) for node in self.atlas.nodes}
        self.node_cell_count = np.bincount(self.atlas.keys[:,0].astype(int),minlength=len(model.tree.left)) if len(self.atlas.keys) else np.zeros(len(model.tree.left),dtype=int)
        self.dependencies = [set() for _ in model.spheres.radii]
        for node, stored in self.atlas.nodes.items():
            atoms,_,_,_,_ = node_geometry(model.tree,node)
            if not np.array_equal(atoms,stored.atoms): raise ValueError('Stored atom ownership changed')
            for atom in atoms: self.dependencies[atom].add(node)
        self.ever_active = np.zeros(len(model.tree.left),dtype=bool)
        self.fits = {}; self.previous = None
        self.build_receipt = dict(payload_build_s=time.perf_counter()-tick,
            cells=len(atlas.keys), trie_states=len(self.bound.edge_starts)-1, trie_edges=len(self.bound.edge_atoms),
            full_index_bytes=self.index.receipt['index_bytes'], layout=layout)
        self.update(model)
        self.build_receipt['total_initial_s'] = time.perf_counter()-tick

    def update(self, model):
        tick = time.perf_counter(); t = model.tree
        if ResponseLibrary.topology_key(t) != self.atlas.topology: raise ValueError('Physical ownership topology changed')
        if not np.isfinite(model.spheres.centers).all() or not np.isfinite(model.spheres.radii).all(): raise ValueError('Finite geometry required')
        current = (model.spheres.centers,model.spheres.radii,t.local,t.rotations,t.translations)
        initial = self.previous is None
        if initial:
            changed = np.ones(len(model.spheres.radii),dtype=bool); dirty = set(self.atlas.nodes)
        else:
            world,radii,local,rotations,translations = self.previous
            changed = np.any(current[0]!=world,axis=1) | (current[1]!=radii) | np.any(t.local!=local,axis=1)
            pose_changed = np.any(t.rotations!=rotations,axis=(1,2)) | np.any(t.translations!=translations,axis=1)
            changed |= pose_changed[t.groups]
            dirty = set()
            for atom in np.flatnonzero(changed): dirty.update(self.dependencies[atom])
        scan_s = time.perf_counter()-tick; fit_tick = time.perf_counter()
        active = self.bound.active.copy(); rotations = self.bound.rotations.copy(); fits = self.fits.copy()
        for node in sorted(dirty):
            stored = self.atlas.nodes[node]; atoms,_,centers,rotation,_ = node_geometry(t,node)
            if not np.array_equal(atoms,stored.atoms): raise ValueError('Stored atom ownership changed')
            valid_radii = np.array_equal(t.radii[atoms],stored.radii)
            native_error = float(np.max(np.abs(centers-stored.centers)))
            if native_error <= 1e-12:
                error = native_error; rank = int(np.linalg.matrix_rank(stored.centers-stored.centers.mean(axis=0)))
                method = 'native_already_aligned'
            else:
                fitted_rotation,_,error,rank = fit_node(stored.centers,model.spheres.centers[atoms])
                method = 'stored_node_fit'
                if native_error < error: error = native_error; method = 'native_lower_component_error'
                else: rotation = fitted_rotation
            valid = valid_radii and error <= self.atlas.residual_tolerance+1e-10
            active[node] = valid and self.node_cell_count[node] > 0
            rotations[node] = rotation if valid else np.eye(3)
            fits[node] = dict(node=int(node),valid=bool(valid),radii_match=bool(valid_radii),rank=rank,method=method,
                native_component_error_A=native_error,fitted_component_error_A=error)
        fit_s = time.perf_counter()-fit_tick
        switched = np.flatnonzero(active != self.bound.active)
        activated = np.flatnonzero(active & ~self.bound.active)
        deactivated = np.flatnonzero(~active & self.bound.active)
        reactivated = activated[self.ever_active[activated]]
        dispatch_tick = time.perf_counter(); changed_entries = sum(len(self.node_entries.get(int(n),())) for n in switched)
        if initial or len(switched):
            if self.layout == 'compact':
                live = active[self.full_nodes]; last = self.full_last[live]
                counts = np.bincount(last,minlength=len(model.spheres.radii))
                self.index.atom_starts = readonly(np.r_[0,np.cumsum(counts)].astype(np.int64))
                self.index.entry_nodes = readonly(self.full_nodes[live])
                self.index.position_rows = readonly(self.full_positions[live])
            else:
                positions = self.full_positions.copy() if initial else self.index.position_rows.copy()
                affected = self.atlas.nodes if initial else switched
                for node in affected:
                    entries = self.node_entries[int(node)]
                    positions[entries] = self.full_positions[entries] if active[node] else -1
                self.index.position_rows = readonly(positions)
        dispatch_s = time.perf_counter()-dispatch_tick
        self.bound.rotations = readonly(rotations); self.bound.active = readonly(active)
        self.bound.model = model; self.index.model = model
        self.ever_active |= active; self.fits = fits; self.previous = tuple(a.copy() for a in current)
        self.receipt = dict(update_s=time.perf_counter()-tick,scan_s=scan_s,fit_s=fit_s,dispatch_update_s=dispatch_s,
            dirty_atoms=int(changed.sum()),dirty_nodes=len(dirty),unchanged_nodes=len(self.atlas.nodes)-len(dirty),
            activated_nodes=activated.tolist(),deactivated_nodes=deactivated.tolist(),reactivated_nodes=reactivated.tolist(),
            switched_entries=changed_entries, active_nodes=int(active.sum()),active_cells=int(self.node_cell_count@active),
            live_pairs=int(np.count_nonzero(active[self.full_nodes])),dispatched_pairs=len(self.index.entry_nodes),
            payload_repacked=False,direction_chart_rebuilt=False,dispatch_gathered=bool((initial or len(switched)) and self.layout=='compact'),
            dirty_scan_global=True,current_geometry_unchanged=True)
        return self.receipt

    def trace(self, source, max_bounces=128, verify_owned=False, mode='indexed'):
        # The full table retains dormant cells. Only the live indexed path may
        # dispatch responses; hierarchy/atom modes would bypass validity gates.
        if mode not in ('indexed','off'): raise ValueError('Persistent engine supports indexed or off only')
        return self.index.trace(source,max_bounces,verify_owned,mode)
