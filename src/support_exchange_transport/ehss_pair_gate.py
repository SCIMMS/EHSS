"""Bounded, lazy sphere-pair candidate cache for full-tree EHSS queries."""
import hashlib
import numpy as np
from numba import njit
from .ehss_spherical_coupling import angular_grid


def empty_coupling():
    return (np.empty((0,3)),np.empty(0),np.empty(0,dtype=np.int64),np.empty(0,dtype=np.int64),
            np.empty((0,3)),np.empty(0),np.empty((0,0),dtype=np.uint8),np.empty((0,3)),np.array([1,1],dtype=np.int64))


class SphereCouplingCache:
    """Only queried pairs are compiled; full tables fall back without eviction.

    Geometry changes conservatively rebuild bounds and clear all pair entries.
    This initial implementation does not claim sparse frame invalidation.
    """
    def __init__(self,mode="cone",capacity=256,grid=(16,32),bounds="ports"):
        if mode not in ["cone","mask"] or bounds not in ["ports","physical"]:
            raise ValueError("Choose cone/mask and ports/physical bounds")
        if not isinstance(capacity,int) or capacity<1:
            raise ValueError("Positive bounded cache capacity required")
        self.mode=mode; self.capacity=capacity; self.bounds=bounds; self.grid=tuple(grid)
        self.cell_centers,self.beta=angular_grid(*self.grid)
        self.signature=None; self.generation=0

    def prepare(self,tree):
        digest=hashlib.sha256()
        values=[tree.world,tree.radii,tree.groups,tree.parent,tree.leaf_group]
        if self.bounds=="ports":
            values += [tree.lo,tree.hi,tree.port_centers,tree.port_normals,tree.port_heights]
        for value in values:
            digest.update(np.asarray(value.shape,dtype=np.int64).tobytes()); digest.update(value.tobytes())
        signature=digest.hexdigest(); rebuilt=signature!=self.signature
        discarded=0
        if rebuilt:
            discarded=0 if self.signature is None else int(np.count_nonzero(self.arrays[3]>=0))
            n=len(tree.parent); centers=np.empty((n,3)); radii=np.empty(n)
            leaves=tree.leaf_node[tree.groups]
            for node in range(n):
                owned=(leaves>=node)&(leaves<tree.end[node]); points=tree.world[owned]
                if self.bounds=="physical":
                    center=points.mean(axis=0); extra=0.
                elif tree.port_kind=="box":
                    center=(tree.lo[node]+tree.hi[node])/2; extra=float(np.linalg.norm(tree.hi[node]-center))
                else:
                    center=tree.port_centers[node]
                    extra=float(np.max(np.linalg.norm(tree.port_vertices[node]-center,axis=1)))
                radius=max(extra,float(np.max(np.linalg.norm(points-center,axis=1)+tree.radii[owned])))
                centers[node]=center
                radii[node]=radius+1e-10*max(1.,radius,float(np.max(np.abs(center))))
            mask_bytes=(len(self.cell_centers)+7)//8 if self.mode=="mask" else 0
            self.arrays=(centers,radii,tree.end.copy(),np.full(self.capacity,-1,dtype=np.int64),
                         np.empty((self.capacity,3)),np.empty(self.capacity),np.zeros((self.capacity,mask_bytes),dtype=np.uint8),
                         self.cell_centers.copy(),np.array(self.grid,dtype=np.int64))
            self.signature=signature; self.generation+=1
        return (1 if self.mode=="cone" else 2),self.arrays,dict(mode=self.mode,bounds=self.bounds,generation=self.generation,
            bounds_rebuilt=len(tree.parent) if rebuilt else 0,pairs_discarded=discarded,
            capacity=self.capacity,array_bytes=sum(a.nbytes for a in self.arrays),
            pairs_before_query=int(np.count_nonzero(self.arrays[3]>=0)),grid=list(self.grid))


@njit(cache=True)
def source_for_gate(o,last,groups,leaf_node,gate,counters):
    if last<0:
        counters[9]+=1
        return -1
    node=leaf_node[groups[last]]; delta=o-gate[0][node]
    counters[7]+=1
    if np.dot(delta,delta)>gate[1][node]*gate[1][node]:
        counters[8]+=1
        return -1
    return node


@njit(cache=True)
def direction_bin(d,grid):
    theta=np.arccos(min(1.,max(-1.,d[2])))
    phi=np.arctan2(d[1],d[0])%(2*np.pi)
    row=min(grid[0]-1,int(theta*grid[0]/np.pi))
    col=min(grid[1]-1,int(phi*grid[1]/(2*np.pi)))
    return row*grid[1]+col


@njit(cache=True)
def pair_allows(node,source,d,bin_index,mode,gate,counters):
    centers,radii,ends,keys,axes,cosines,masks,cells,grid=gate
    if source<0 or node<=source<ends[node]:
        return True
    counters[0]+=1
    key=source*len(ends)+node; start=(key^(key>>16))%len(keys); slot=-1
    for probe in range(len(keys)):
        index=(start+probe)%len(keys); counters[6]+=1
        if keys[index]==key:
            slot=index; counters[1]+=1; break
        if keys[index]<0:
            slot=index; keys[index]=key; counters[2]+=1
            delta=centers[node]-centers[source]; distance=np.sqrt(np.dot(delta,delta)); radius=radii[node]+radii[source]
            guard=128*np.finfo(np.float64).eps*max(1.,distance,radius,np.max(np.abs(centers[node])),np.max(np.abs(centers[source])))
            if distance<=radius+guard:
                axes[index,:]=0.; cosines[index]=-1.
            else:
                axes[index,:]=delta/distance
                ratio=min(1.,(radius+guard)/(distance-guard))
                cosines[index]=max(-1.,np.sqrt(max(0.,1-ratio*ratio))-128*np.finfo(np.float64).eps)
            if mode==2:
                alpha=np.pi if cosines[index]<0. else np.arccos(cosines[index])
                beta=min(np.pi,.5*np.pi/grid[0]+np.pi/grid[1])
                threshold=np.cos(min(np.pi,alpha+beta))-256*np.finfo(np.float64).eps
                masks[index,:]=0
                for k in range(len(cells)):
                    if alpha+beta>=np.pi or np.dot(cells[k],axes[index])>=threshold:
                        masks[index,k//8] |= np.uint8(1<<(k%8))
            break
    if slot<0:
        counters[5]+=1
        return True
    if cosines[slot]<0.:
        counters[4]+=1
        return True
    allowed=np.dot(d,axes[slot])>=cosines[slot] if mode==1 else bool(masks[slot,bin_index//8] & np.uint8(1<<(bin_index%8)))
    if not allowed:
        counters[3]+=1
    return allowed
