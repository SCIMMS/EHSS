"""Prepare reference node geometry once and batch equally sized proper fits."""
from dataclasses import dataclass
import time
import numpy as np
from .ehss_persistent_node_response import PersistentNodeResponses
from .ehss_sparse_path_response import readonly
from .ehss_response import ResponseLibrary


@dataclass(frozen=True)
class NodeFitBatch:
    nodes: np.ndarray
    atoms: np.ndarray
    anchors: np.ndarray
    same_anchor: np.ndarray
    reference: np.ndarray
    mean: np.ndarray
    centered: np.ndarray
    radii: np.ndarray
    ranks: np.ndarray


class PreparedNodeFits:
    """Fixed ownership and centered reference arrays grouped by atom count.

    Reference nodes may overlap as ancestors; no independence is assumed.
    The current physical coordinates are read, never changed by fitting.
    """
    def __init__(self, model, atlas):
        tick=time.perf_counter();self.batches=[];self.topology=atlas.topology
        if ResponseLibrary.topology_key(model.tree)!=self.topology:raise ValueError('Physical ownership topology changed')
        sizes=sorted({len(stored.atoms) for stored in atlas.nodes.values()})
        for size in sizes:
            nodes=np.array(sorted(node for node,stored in atlas.nodes.items() if len(stored.atoms)==size),dtype=np.int64)
            atoms=np.stack([atlas.nodes[int(node)].atoms for node in nodes])
            anchors=model.tree.groups[atoms[:,0]]
            reference=np.stack([atlas.nodes[int(node)].centers for node in nodes])
            # These per-node reductions intentionally match the original
            # reference calculation and are performed only during preparation.
            means=np.stack([v.mean(axis=0) for v in reference])
            centered=reference-means[:,None,:]
            ranks=np.array([np.linalg.matrix_rank(v) for v in centered],dtype=np.int64)
            arrays=(nodes,atoms,anchors,model.tree.groups[atoms]==anchors[:,None],reference,means,centered,
                np.stack([atlas.nodes[int(node)].radii for node in nodes]),ranks)
            self.batches.append(NodeFitBatch(*(readonly(a) for a in arrays)))
        self.receipt=dict(prepare_s=time.perf_counter()-tick,nodes=len(atlas.nodes),batches=len(self.batches),
            numeric_bytes=sum(a.nbytes for batch in self.batches for a in vars(batch).values()),
            reference_centering_repeated=False,owned_atoms_preindexed=True)

    def evaluate(self,model,dirty):
        """Return the old fit policy using stacked matrix operations and SVD.

        Determinant correction restricts fits to proper rotations. Native pose
        wins if already aligned or if it has the smaller maximum component
        residual. This remains a sampled response compatibility policy.
        """
        t=model.tree;dirty=np.asarray(sorted(dirty),dtype=np.int64);result={};batch_calls=0;svd_nodes=0
        for batch in self.batches:
            take=np.flatnonzero(np.isin(batch.nodes,dirty))
            if not len(take):continue
            nodes=batch.nodes[take];atoms=batch.atoms[take];anchors=batch.anchors[take]
            reference=batch.reference[take];same=batch.same_anchor[take]
            rotation=t.rotations[anchors].copy();translation=t.translations[anchors]
            native=(t.world[atoms]-translation[:,None,:])@rotation
            native[same]=t.local[atoms][same]
            native_error=np.max(np.abs(native-reference),axis=(1,2));error=native_error.copy()
            ranks=batch.ranks[take].copy();methods=np.zeros(len(nodes),dtype=np.int8)
            matching_radii=np.all(t.radii[atoms]==batch.radii[take],axis=1)
            fit=np.flatnonzero(native_error>1e-12)
            if len(fit):
                world=model.spheres.centers[atoms[fit]];mean=world.mean(axis=1)
                x=batch.centered[take[fit]];y=world-mean[:,None,:]
                u,singular,vh=np.linalg.svd(np.swapaxes(x,1,2)@y)
                correction=np.tile(np.eye(3),(len(fit),1,1));correction[:,2,2]=np.where(np.linalg.det(u@vh)>=0.,1.,-1.)
                fitted=np.swapaxes(u@correction@vh,1,2)
                # Preserve the small vector/matrix translation operation used
                # by scalar fitting; the costly node loops are batched above.
                refmean=batch.mean[take[fit]]
                shifts=np.stack([m-r@rot.T for m,r,rot in zip(mean,refmean,fitted)])
                fit_error=np.max(np.abs((world-shifts[:,None,:])@fitted-reference[fit]),axis=(1,2))
                use_native=native_error[fit]<fit_error
                error[fit]=np.where(use_native,native_error[fit],fit_error)
                ranks[fit]=np.count_nonzero(singular>np.maximum(singular[:,0],1.)[:,None]*1e-12,axis=1)
                methods[fit]=np.where(use_native,2,1)
                rotation[fit[~use_native]]=fitted[~use_native]
                batch_calls+=1;svd_nodes+=len(fit)
            for i,node in enumerate(nodes):
                result[int(node)]=(rotation[i],float(error[i]),int(ranks[i]),int(methods[i]),float(native_error[i]),bool(matching_radii[i]))
        return result,dict(svd_batch_calls=batch_calls,svd_nodes=svd_nodes,prepared_batches=len(self.batches))


class BatchedNodeResponses(PersistentNodeResponses):
    """Persistent payload with prepared, batched node compatibility updates."""
    def __init__(self,model,atlas,layout='compact',max_index_bytes=64*1024*1024):
        self.fit_plan=None
        super().__init__(model,atlas,layout,max_index_bytes)
        self.build_receipt['fit_plan']=self.fit_plan.receipt

    def update(self,model):
        tick=time.perf_counter();t=model.tree
        if ResponseLibrary.topology_key(t)!=self.atlas.topology:raise ValueError('Physical ownership topology changed')
        if not np.isfinite(model.spheres.centers).all() or not np.isfinite(model.spheres.radii).all():raise ValueError('Finite geometry required')
        if self.fit_plan is None:self.fit_plan=PreparedNodeFits(model,self.atlas)
        current=(model.spheres.centers,model.spheres.radii,t.local,t.rotations,t.translations);initial=self.previous is None
        if initial:
            changed=np.ones(len(model.spheres.radii),dtype=bool);dirty=set(self.atlas.nodes)
        else:
            world,radii,local,rotations,translations=self.previous
            changed=np.any(current[0]!=world,axis=1)|(current[1]!=radii)|np.any(t.local!=local,axis=1)
            pose_changed=np.any(t.rotations!=rotations,axis=(1,2))|np.any(t.translations!=translations,axis=1)
            changed|=pose_changed[t.groups];dirty=set()
            for atom in np.flatnonzero(changed):dirty.update(self.dependencies[atom])
        scan_s=time.perf_counter()-tick;fit_tick=time.perf_counter()
        active=self.bound.active.copy();rotations=self.bound.rotations.copy();fits=self.fits.copy()
        evaluated,stats=self.fit_plan.evaluate(model,dirty)
        labels=('native_already_aligned','stored_node_fit','native_lower_component_error')
        for node,(rotation,error,rank,method,native_error,radii_match) in evaluated.items():
            valid=radii_match and error<=self.atlas.residual_tolerance+1e-10
            active[node]=valid and self.node_cell_count[node]>0;rotations[node]=rotation if valid else np.eye(3)
            fits[node]=dict(node=node,valid=bool(valid),radii_match=radii_match,rank=rank,method=labels[method],
                native_component_error_A=native_error,fitted_component_error_A=error)
        fit_s=time.perf_counter()-fit_tick
        switched=np.flatnonzero(active!=self.bound.active);activated=np.flatnonzero(active&~self.bound.active)
        deactivated=np.flatnonzero(~active&self.bound.active);reactivated=activated[self.ever_active[activated]]
        dispatch_tick=time.perf_counter();changed_entries=sum(len(self.node_entries.get(int(n),())) for n in switched)
        if initial or len(switched):
            if self.layout=='compact':
                live=active[self.full_nodes];counts=np.bincount(self.full_last[live],minlength=len(model.spheres.radii))
                self.index.atom_starts=readonly(np.r_[0,np.cumsum(counts)].astype(np.int64))
                self.index.entry_nodes=readonly(self.full_nodes[live]);self.index.position_rows=readonly(self.full_positions[live])
            else:
                positions=self.full_positions.copy() if initial else self.index.position_rows.copy()
                for node in self.atlas.nodes if initial else switched:
                    entries=self.node_entries[int(node)];positions[entries]=self.full_positions[entries] if active[node] else -1
                self.index.position_rows=readonly(positions)
        dispatch_s=time.perf_counter()-dispatch_tick
        self.bound.rotations=readonly(rotations);self.bound.active=readonly(active)
        self.bound.model=model;self.index.model=model;self.ever_active|=active;self.fits=fits
        self.previous=tuple(a.copy() for a in current)
        self.receipt=dict(update_s=time.perf_counter()-tick,scan_s=scan_s,fit_s=fit_s,dispatch_update_s=dispatch_s,
            dirty_atoms=int(changed.sum()),dirty_nodes=len(dirty),unchanged_nodes=len(self.atlas.nodes)-len(dirty),
            activated_nodes=activated.tolist(),deactivated_nodes=deactivated.tolist(),reactivated_nodes=reactivated.tolist(),
            switched_entries=changed_entries,active_nodes=int(active.sum()),active_cells=int(self.node_cell_count@active),
            live_pairs=int(np.count_nonzero(active[self.full_nodes])),dispatched_pairs=len(self.index.entry_nodes),
            payload_repacked=False,direction_chart_rebuilt=False,dispatch_gathered=bool((initial or len(switched)) and self.layout=='compact'),
            dirty_scan_global=True,current_geometry_unchanged=True,**stats)
        return self.receipt
