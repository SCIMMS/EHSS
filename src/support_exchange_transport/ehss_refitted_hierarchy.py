"""Transactional refit of fixed ownership trees with exact current metadata.

Topology is immutable. Bounds of changed leaves and their ancestors are updated;
foreign overlap eligibility is recomputed globally because a remote mover can
change an otherwise stationary node's eligibility. No geometric tolerance is used.
"""
from copy import copy
import time
import numpy as np
from numba import njit
from .ehss_guarded_intrinsic_response import GuardedSupportHierarchy
from .ehss_sparse_path_response import readonly


@njit(cache=True)
def refit_bounds(world,radii,changed,leaf_node,offsets,ids,left,right,parent,old_lo,old_hi,pad):
    lo=old_lo.copy(); hi=old_hi.copy(); affected=np.zeros(len(left),dtype=np.bool_)
    for group in range(len(changed)):
        if not changed[group]: continue
        node=leaf_node[group]
        for c in range(3):
            low=np.inf; high=-np.inf
            for j in range(offsets[group],offsets[group+1]):
                atom=ids[j]; low=min(low,world[atom,c]-radii[atom]); high=max(high,world[atom,c]+radii[atom])
            lo[node,c]=low-pad; hi[node,c]=high+pad
        while node>=0:
            affected[node]=True; node=parent[node]
    for node in range(len(left)-1,-1,-1):
        if affected[node] and left[node]>=0:
            for c in range(3):
                lo[node,c]=min(lo[left[node],c],lo[right[node],c]); hi[node,c]=max(hi[left[node],c],hi[right[node],c])
    return lo,hi,affected


@njit(cache=True)
def exact_exclusivity(world,radii,atom_leaf,end,lo,hi):
    exclusive=np.ones(len(end),dtype=np.bool_)
    for node in range(len(end)):
        for atom in range(len(radii)):
            if node<=atom_leaf[atom]<end[node]: continue
            overlap=True
            for c in range(3):
                if world[atom,c]+radii[atom]<lo[node,c] or world[atom,c]-radii[atom]>hi[node,c]: overlap=False; break
            if overlap: exclusive[node]=False; break
    return exclusive


class HierarchyRefitLayout:
    """Fixed atom identity, physical ownership, topology and box chart metadata."""
    def __init__(self,model):
        tick=time.perf_counter()
        if not isinstance(model,GuardedSupportHierarchy) or model.tree.port_kind!='box':
            raise ValueError('Guarded box-support hierarchy required')
        self.atom_ids=model.geometry.atom_ids; self.tree=copy(model.tree)
        for name,value in vars(model.tree).items():
            if isinstance(value,np.ndarray): setattr(self.tree,name,readonly(value.copy()))
        self.atom_leaf=readonly(self.tree.leaf_node[self.tree.groups])
        self.prepare_s=time.perf_counter()-tick

    def build(self,geometry,previous=None): return RefittedSupportHierarchy(geometry,self,previous)


class RefittedSupportHierarchy(GuardedSupportHierarchy):
    def __init__(self,geometry,layout,previous=None):
        tick=time.perf_counter(); template=layout.tree
        if geometry.atom_ids!=layout.atom_ids or not np.array_equal(geometry.groups,template.groups):
            raise ValueError('Fixed atom identity and physical ownership required')
        if previous is not None and getattr(previous,'refit_layout',None) is not layout:
            raise ValueError('Previous model belongs to another refit layout')
        self.refit_layout=layout; self.geometry=geometry
        start=time.perf_counter(); spheres=geometry.spheres(); materialize_s=time.perf_counter()-start
        if len(spheres.radii)!=len(template.groups): raise ValueError('Fixed atom count required')
        self.spheres=spheres; tree=copy(template); self.tree=tree
        start=time.perf_counter()
        tree.rotations=np.array([b.rotation for b in geometry.blocks]); tree.translations=np.array([b.translation for b in geometry.blocks])
        tree.local=np.empty_like(spheres.centers)
        for group,block in enumerate(geometry.blocks):
            atoms=block.intrinsic.atoms
            tree.local[atoms]=(spheres.centers[atoms]-tree.translations[group])@tree.rotations[group]
        tree.local=readonly(tree.local); tree.radii=spheres.radii; tree.world=spheres.centers; tree.spheres=spheres
        old=template if previous is None else previous.tree
        pose_changed=np.any(tree.rotations!=old.rotations,axis=(1,2))|np.any(tree.translations!=old.translations,axis=1)
        tree.pose_versions=old.pose_versions.copy()+pose_changed.astype(np.int64)
        local_s=time.perf_counter()-start; start=time.perf_counter()
        pad=1e-10*max(1.,float(spheres.radii.max()))
        if previous is None:
            changed=np.ones(tree.n_groups,dtype=bool)
        else:
            moved=np.any(spheres.centers!=previous.spheres.centers,axis=1)|(spheres.radii!=previous.spheres.radii)
            changed=np.bincount(tree.groups[moved],minlength=tree.n_groups)>0
            if pad!=previous.bound_pad: changed[:]=True
        tree.lo,tree.hi,affected=refit_bounds(tree.world,tree.radii,changed,tree.leaf_node,tree.offsets,tree.ids,
            tree.left,tree.right,tree.parent,old.lo,old.hi,pad)
        bounds_s=time.perf_counter()-start; start=time.perf_counter()
        tree.exclusive=exact_exclusivity(tree.world,tree.radii,layout.atom_leaf,tree.end,tree.lo,tree.hi)
        overlap_s=time.perf_counter()-start; self.bound_pad=pad
        self.receipt=dict(build_s=time.perf_counter()-tick,materialize_s=materialize_s,local_s=local_s,bounds_s=bounds_s,overlap_s=overlap_s,
            nodes=len(tree.left),physical_blocks=tree.n_groups,changed_supports=np.flatnonzero(changed).tolist(),affected_nodes=np.flatnonzero(affected).tolist(),
            shared_topology=True,exact_materialized_geometry=True,physical_coordinate_approximation=False,
            bounds_refreshed=int(affected.sum()),exclusivity_refreshed=len(tree.left),exclusivity_uses_current_world=True,
            box_chart_metadata_shared=True,old_snapshot_modified=False,intrinsic_response_implicitly_reused=False)
