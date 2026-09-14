"""Material sampling axes may refine response ownership without changing it.

Binding advects points, independent directions and incoming tags together.
observe_next is solely a coupling-row observation, not a full path response.
"""
from dataclasses import dataclass
import time
import numpy as np
from numba import njit
from .ehss_reuse_frames import ReuseFrames
from .ehss_material_collision_field import material_rotations
from .ehss_factored_local_samples import point_inside
from .ehss_guarded_intrinsic_response import nearest_world
from .ehss_sparse_path_response import readonly

@njit(cache=False)
def next_observations(points,directions,lasts,cap,world,radii,lo,hi,left,right,end,leaf_group,offsets,ids):
    counts=np.ones(len(points),dtype=np.int64); history=np.full((len(points),cap),-1,dtype=np.int64)
    terminal=np.full(len(points),-1,dtype=np.int64); counters=np.zeros(2,dtype=np.int64)
    for i in range(len(points)):
        last=lasts[i]; history[i,0]=last
        if point_inside(0,points[i],last,lo,hi,left,right,leaf_group,offsets,ids,world,radii,counters): terminal[i]=-3; continue
        atom,_=nearest_world(0,points[i],directions[i],last,np.inf,-1,lo,hi,left,right,end,leaf_group,offsets,ids,world,radii,counters)
        if atom>=0: counts[i]=2; history[i,1]=atom; terminal[i]=atom
    return counts,history,terminal,counters

@dataclass(frozen=True)
class AlignedMaterialBinding:
    bank: object
    model: object
    basis: object
    parent_rotations: np.ndarray
    atom_rotations: np.ndarray
    points: np.ndarray
    directions: np.ndarray
    tags: np.ndarray
    receipt: dict

    def observe_next(self):
        start=time.perf_counter(); t=self.model.tree; s=self.model.spheres
        counts,history,terminal,counters=next_observations(self.points,self.directions,self.bank.lasts,self.bank.cap,s.centers,s.radii,
            t.lo,t.hi,t.left,t.right,t.end,t.leaf_group,t.offsets,t.ids)
        return dict(counts=counts,history=history,terminal=terminal,query_s=time.perf_counter()-start,
            sphere_tests=int(counters[1]),box_tests=int(counters[0]),one_event_coupling_observation=True,full_trajectory_observation=False)

class AlignedMaterialBasis:
    def __init__(self,bank,frame_groups=None):
        if bank.cap<2: raise ValueError('At least two collisions needed for next-event observations')
        self.bank=bank; self.frames=ReuseFrames(bank.reference,bank.groups,bank.groups if frame_groups is None else frame_groups)

    def bind(self,model):
        start=time.perf_counter(); bank=self.bank; world=model.spheres.centers; radii=model.spheres.radii
        if world.shape!=bank.reference.shape: raise ValueError('Physical atom count changed')
        parent=material_rotations(bank,world); axes=np.empty((len(radii),3,3))
        if np.array_equal(world,bank.reference): axes[:]=np.eye(3)
        else:
            for g,atoms in enumerate(self.frames.atoms): axes[atoms]=self.frames.fit(g,world,parent)[0]
        row_axes=axes[bank.lasts]
        normals=np.einsum('nij,nj->ni',row_axes,bank.normals)
        points=world[bank.lasts]+radii[bank.lasts,None]*normals
        directions=np.einsum('nij,nj->ni',row_axes,bank.directions); tags=np.einsum('nij,nj->ni',row_axes,bank.tags)
        return AlignedMaterialBinding(bank,model,self,readonly(parent),readonly(axes),readonly(points),readonly(directions),readonly(tags),
            dict(bind_s=time.perf_counter()-start,frame_count=len(self.frames.atoms),response_owner_count=len(bank.atoms),
                material_axes_independent_of_ownership=True,local_itineraries_compiled=False))
