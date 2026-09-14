"""Batch exact initial-frame geometry using a fixed molecular ownership layout.

This preserves the initial_geometry representation and content versions. It does
not substitute a rigid fit for deformation. Exact unchanged local blocks can
share intrinsic data; current coordinates remain explicit and immutable.
"""
from dataclasses import dataclass
import time
import numpy as np
from .inside_out_pa import Spheres
from .ehss_intrinsic_frames import GeometryFrame,IntrinsicBlock,PosedBlock,frozen
from .ehss_region_versions import hash_arrays


@dataclass(frozen=True)
class PreparedGeometryFrame(GeometryFrame):
    layout: object
    packed_local: np.ndarray
    packed_radii: np.ndarray
    world: np.ndarray
    radii: np.ndarray

    def spheres(self):
        # Return independent arrays, as GeometryFrame.spheres does. The cached
        # world was checked against the batch pose/correction reconstruction.
        return Spheres(self.world,self.radii,deduplicate=False)


class GeometryPreparationLayout:
    def __init__(self,spheres,groups,atom_ids=None):
        tick=time.perf_counter(); n=len(spheres.radii); groups=np.asarray(groups)
        identities=tuple(range(n)) if atom_ids is None else tuple(atom_ids)
        if len(identities)!=n or len(set(identities))!=n: raise ValueError('Unique atom identities required')
        if groups.shape!=(n,) or not np.issubdtype(groups.dtype,np.integer): raise ValueError('One integer group per atom required')
        unique=np.unique(groups)
        if not len(unique) or not np.array_equal(unique,np.arange(len(unique))): raise ValueError('Contiguous nonempty groups required')
        self.atom_ids=identities; self.groups=frozen(groups); self.order=frozen(np.argsort(groups,kind='stable'))
        self.starts=frozen(np.r_[0,np.cumsum(np.bincount(groups))]); self.packed_groups=frozen(groups[self.order])
        self.atoms=tuple(self.order[a:b] for a,b in zip(self.starts[:-1],self.starts[1:]))
        self.identity=frozen(np.eye(3)); self.empty_rows=frozen(np.empty(0,dtype=np.int64)); self.empty_points=frozen(np.empty((0,3)))
        self.prepare_s=time.perf_counter()-tick

    def build(self,spheres,previous=None,atom_ids=None):
        tick=time.perf_counter(); n=len(self.order); ng=len(self.atoms)
        if atom_ids is not None and tuple(atom_ids)!=self.atom_ids: raise ValueError('Fixed atom identities and order required')
        if previous is not None and (not isinstance(previous,PreparedGeometryFrame) or previous.layout is not self):
            raise ValueError('Previous geometry belongs to another layout')
        if spheres.centers.shape!=(n,3) or spheres.radii.shape!=(n,): raise ValueError('Fixed atom count required')
        if not np.isfinite(spheres.centers).all() or not np.isfinite(spheres.radii).all() or np.any(spheres.radii<=0):
            raise ValueError('Finite current geometry and positive radii required')
        start=time.perf_counter(); world=frozen(spheres.centers); radii=frozen(spheres.radii)
        packed=world[self.order]; packed_radii=frozen(radii[self.order]); translations=frozen(packed[self.starts[:-1]])
        local=frozen(packed-translations[self.packed_groups]); prediction=local@self.identity.T+translations[self.packed_groups]
        delta=packed-prediction; rows=np.flatnonzero(np.any(delta!=0.,axis=1))
        corrected=prediction.copy(); corrected[rows]+=delta[rows]
        overrides=np.flatnonzero(np.any(corrected!=packed,axis=1)); corrected[overrides]=packed[overrides]
        if not np.array_equal(corrected,packed): raise AssertionError('Batch pose corrections must preserve world coordinates')
        pack_s=time.perf_counter()-start; start=time.perf_counter()
        row_starts=np.r_[0,np.cumsum(np.bincount(self.packed_groups[rows],minlength=ng))]
        override_starts=np.r_[0,np.cumsum(np.bincount(self.packed_groups[overrides],minlength=ng))]
        relative_rows=frozen(rows-self.starts[self.packed_groups[rows]]); residuals=frozen(delta[rows])
        relative_overrides=frozen(overrides-self.starts[self.packed_groups[overrides]]); absolute=frozen(packed[overrides])
        same_intrinsic=np.zeros(ng,dtype=bool); same_world=np.zeros(ng,dtype=bool)
        if previous is not None:
            changes=np.any(local!=previous.packed_local,axis=1)|(packed_radii!=previous.packed_radii)
            same_intrinsic=np.bincount(self.packed_groups[changes],minlength=ng)==0
            changes=np.any(world!=previous.world,axis=1)|(radii!=previous.radii)
            same_world=np.bincount(self.groups[changes],minlength=ng)==0
        partition_s=time.perf_counter()-start; start=time.perf_counter(); blocks=[]; hashed=0
        for group,(a,b) in enumerate(zip(self.starts[:-1],self.starts[1:])):
            if same_world[group]: blocks.append(previous.blocks[group]); continue
            atoms=self.atoms[group]
            if same_intrinsic[group]: intrinsic=previous.blocks[group].intrinsic
            else:
                intrinsic=IntrinsicBlock(atoms,local[a:b],packed_radii[a:b],hash_arrays(atoms,local[a:b],packed_radii[a:b])); hashed+=1
            ra,rb=row_starts[group:group+2]; oa,ob=override_starts[group:group+2]
            blocks.append(PosedBlock(intrinsic,self.identity,translations[group],relative_rows[ra:rb],residuals[ra:rb],relative_overrides[oa:ob],absolute[oa:ob]))
        assemble_s=time.perf_counter()-start
        receipt=dict(initial=previous is None,blocks=ng,prepare_s=time.perf_counter()-tick,pack_s=pack_s,partition_s=partition_s,assemble_s=assemble_s,
            unchanged_blocks=np.flatnonzero(same_world).tolist(),reused_intrinsic_blocks=int(same_intrinsic.sum()),hashed_blocks=hashed,
            residual_atoms=len(rows),absolute_override_atoms=len(overrides),fixed_ownership_layout_reused=True,
            batch_reconstruction_verified=True,physical_coordinate_approximation=False,initial_geometry_representation_preserved=True,
            materialized_world_cached=True,rigid_fit_performed=False,global_current_geometry_scan=True)
        return PreparedGeometryFrame(self.atom_ids,self.groups,tuple(blocks),receipt,self,local,packed_radii,world,radii)
