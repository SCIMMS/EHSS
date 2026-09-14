"""Reuse coordinate frames independent of response ownership.

Point blocks inherit the parent orientation. Bond blocks fit their axis while
inheriting the unobservable roll from the parent through a minimum rotation.
Noncollinear blocks use proper Kabsch fitting. No incident ray defines a frame.
"""
import numpy as np
from .ehss_sparse_path_response import readonly

def partition(groups,size,name):
    a=np.asarray(groups)
    if a.shape!=(size,) or not np.issubdtype(a.dtype,np.integer): raise ValueError(f'{name}: one integer group per atom required')
    unique=np.unique(a)
    if not len(unique) or not np.array_equal(unique,np.arange(len(unique))): raise ValueError(f'{name}: contiguous nonempty groups required')
    return readonly(a.astype(np.int64,copy=True)),tuple(readonly(np.flatnonzero(a==g)) for g in unique)

def axis_rotation(a,b,reference_axis,parent):
    cross=np.cross(a,b); sine=np.linalg.norm(cross); cosine=np.clip(np.dot(a,b),-1.,1.)
    if sine<=1e-14:
        if cosine>=0.: return np.eye(3)
        basis=np.eye(3)[int(np.argmin(np.abs(reference_axis)))]; axis=parent@np.cross(reference_axis,basis); axis/=np.linalg.norm(axis)
        return 2*np.outer(axis,axis)-np.eye(3)
    axis=cross/sine; x,y,z=axis
    skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    return cosine*np.eye(3)+(1-cosine)*np.outer(axis,axis)+sine*skew

class ReuseFrames:
    def __init__(self,reference,response_groups,frame_groups):
        reference=np.asarray(reference,dtype=float)
        if reference.ndim!=2 or reference.shape[1]!=3 or not np.isfinite(reference).all(): raise ValueError('Finite reference coordinates required')
        self.groups,self.atoms=partition(frame_groups,len(reference),'Frame partition')
        response,response_atoms=partition(response_groups,len(reference),'Response partition')
        self.parents=[]; self.local=[]; self.ranks=[]; self.axes=[]; self.same_parent=[]
        for atoms in self.atoms:
            parents=np.unique(response[atoms])
            if len(parents)!=1: raise ValueError('Frame partition must refine response ownership')
            parent=int(parents[0]); a=reference[atoms]; a=a-a.mean(axis=0)
            _,singular,vh=np.linalg.svd(a,full_matrices=True)
            rank=int(np.count_nonzero(singular>max(float(singular.max(initial=0))*1e-10,1e-12)))
            self.parents.append(parent); self.local.append(readonly(a)); self.ranks.append(rank); self.axes.append(readonly(vh[0]))
            self.same_parent.append(np.array_equal(atoms,response_atoms[parent]))
        self.parents=tuple(self.parents); self.local=tuple(self.local); self.ranks=tuple(self.ranks); self.axes=tuple(self.axes); self.same_parent=tuple(self.same_parent)

    def fit(self,g,world,parent_rotations):
        atoms=self.atoms[g]; b=world[atoms]; center=b.mean(axis=0); b=b-center
        parent=parent_rotations[self.parents[g]]; a=self.local[g]; rank=self.ranks[g]
        if self.same_parent[g] or rank==0: rotation=parent
        elif rank==1:
            axis=self.axes[g]; target=(a@axis)@b; length=np.linalg.norm(target)
            rotation=parent if length<=1e-14 else axis_rotation(parent@axis,target/length,axis,parent)@parent
        else:
            u,_,vh=np.linalg.svd(a.T@b); correction=np.eye(3); correction[2,2]=1. if np.linalg.det(u@vh)>=0 else -1.
            rotation=(u@correction@vh).T
        return rotation,center
