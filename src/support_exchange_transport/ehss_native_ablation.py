"""Research controls: one native reflection loop, flat/BVH search, IID/Sobol source.

The production engine is unchanged. Timing kernels omit traversal counters;
instrumented specializations can be run separately on the identical rays.
"""
import time
import numpy as np
from numba import njit
from .ehss_cpu import entry_scalar, reflect_scalar
from .ehss_support_tree import box_interval
from .ehss_state import EHSSSource, EHSSResult, external_source
from .inside_out_pa import Spheres, unit_sphere, tangent_frames
from .ehss_static import PreparedStaticCPUEHSS


def incident_source(spheres, power, seed, sampler='sobol'):
    if sampler == 'sobol':
        return external_source(spheres,power,seed)
    if sampler != 'iid':
        raise ValueError('sampler must be iid or sobol')
    if type(power) is not int or power < 0:
        raise ValueError('Nonnegative integer power required')
    u=np.random.default_rng(seed).random((2**power,4))
    directions=unit_sphere(u[:,:2])
    tangent,bitangent=tangent_frames(directions)
    center=spheres.centers.mean(axis=0)
    radius=float(np.max(np.linalg.norm(spheres.centers-center,axis=1)+spheres.radii))*(1.+1e-9)
    rho=radius*np.sqrt(u[:,2]);phi=2*np.pi*u[:,3]
    origins=center-radius*directions+rho[:,None]*(np.cos(phi)[:,None]*tangent+np.sin(phi)[:,None]*bitangent)
    n=len(u)
    return EHSSSource(origins,directions,directions.copy(),np.full(n,np.pi*radius*radius/n),
        np.full(n,-1),np.zeros(n,dtype=np.int64),'IID external impact disk; same measure as Sobol')


def _make_nearest(use_bvh, instrument):
    @njit
    def nearest(o,d,last,lo,hi,end,leaf_group,offsets,ids,world,radii):
        atom,best,boxes,spheres=-1,np.inf,0,0
        if not use_bvh:
            for wanted in range(len(radii)):
                if wanted==last:
                    continue
                if instrument:
                    spheres+=1
                distance=entry_scalar(o,d,world[wanted],radii[wanted])
                if distance<best or (distance==best and np.isfinite(distance) and (atom<0 or wanted<atom)):
                    atom,best=wanted,distance
        else:
            node=0
            while node<len(end):
                if instrument:
                    boxes+=1
                near,far=box_interval(o,d,lo[node],hi[node])
                if near>best or far<0.:
                    node=end[node]
                    continue
                group=leaf_group[node]
                if group>=0:
                    for k in range(offsets[group],offsets[group+1]):
                        wanted=ids[k]
                        if wanted==last:
                            continue
                        if instrument:
                            spheres+=1
                        distance=entry_scalar(o,d,world[wanted],radii[wanted])
                        if distance<best or (distance==best and np.isfinite(distance) and (atom<0 or wanted<atom)):
                            atom,best=wanted,distance
                node+=1
        return atom,best,boxes,spheres
    return nearest


NEAREST={(mode,instrument):_make_nearest(mode=='bvh',instrument)
         for mode in ('flat','bvh') for instrument in (False,True)}


@njit
def _trace(nearest,origins,directions,lasts,initial,cap,record,
           lo,hi,end,leaf_group,offsets,ids,world,radii):
    positions,outgoing,counts=origins.copy(),directions.copy(),initial.copy()
    escaped=np.zeros(len(positions),dtype=np.bool_)
    unresolved=escaped.copy()
    history=np.full((len(positions),cap if record else 0),-1,dtype=np.int32)
    boxes,spheres=0,0
    for ray in range(len(positions)):
        last=lasts[ray]
        if record and counts[ray]==1:
            history[ray,0]=last
        while True:
            atom,distance,nb,ns=nearest(positions[ray],outgoing[ray],last,
                lo,hi,end,leaf_group,offsets,ids,world,radii)
            boxes+=nb;spheres+=ns
            if atom<0:
                escaped[ray]=True
                break
            if counts[ray]>=cap:
                unresolved[ray]=True
                break
            reflect_scalar(positions[ray],outgoing[ray],atom,distance,world,radii)
            if record:
                history[ray,counts[ray]]=atom
            counts[ray]+=1;last=atom
    return positions,outgoing,escaped,unresolved,counts,history,boxes,spheres


class NativeAblation:
    def __init__(self,spheres,mode='bvh',leaf_size=4):
        if mode not in ('flat','bvh'):
            raise ValueError('mode must be flat or bvh')
        tick=time.perf_counter()
        self.mode=mode
        if mode=='bvh':
            self.prepared=PreparedStaticCPUEHSS(spheres,leaf_size=leaf_size,threads=1)
            self.spheres=self.prepared.model.spheres
            tree=self.prepared.model.tree
            self.tree_args=tuple(getattr(tree,k) for k in ('lo','hi','end','leaf_group','offsets','ids'))
        else:
            self.prepared=None
            self.spheres=Spheres(spheres.centers,spheres.radii,deduplicate=False)
            self.tree_args=(np.empty((0,3)),np.empty((0,3)),*(np.empty(0,dtype=np.int64) for _ in range(4)))
        self.prepare_s=time.perf_counter()-tick

    def trace(self,source,cap=64,record_history=False,instrument=False):
        if type(cap) is not int or cap<1:
            raise ValueError('Positive integer cap required')
        if np.any(source.initial_bounces>1) or np.any(source.last_atom>=len(self.spheres.radii)) or np.any(source.last_atom < -1):
            raise ValueError('Invalid initial history')
        tick=time.perf_counter()
        values=_trace(NEAREST[(self.mode,instrument)],source.origins,source.directions,
            source.last_atom,source.initial_bounces,cap,record_history,*self.tree_args,
            self.spheres.centers,self.spheres.radii)
        positions,outgoing,escaped,unresolved,counts,history,boxes,spheres=values
        return EHSSResult(positions,outgoing,escaped,unresolved,counts,history,source,
            dict(trace_s=time.perf_counter()-tick,mode=self.mode,instrumented=instrument,
                box_tests=boxes if instrument else None,sphere_tests=spheres if instrument else None))
