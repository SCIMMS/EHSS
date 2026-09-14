"""Recorded flight/support broad phase for empirical motion-gated reuse.

Expanded support AABBs conservatively cover proximity to the recorded flights.
They do not bound a perturbed multiple-scattering trajectory or its EHSS error.
Both recorded and current bounds are considered to catch new entrants.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit
from .ehss_reference import entry_distance
from .ehss_guarded_intrinsic_response import reflect
from .ehss_sparse_path_response import readonly

@njit(cache=True)
def support_bounds(centers,radii,groups,count):
    lo=np.full((count,3),np.inf); hi=np.full((count,3),-np.inf)
    for atom in range(len(radii)):
        g=groups[atom]
        for k in range(3):
            lo[g,k]=min(lo[g,k],centers[atom,k]-radii[atom])
            hi[g,k]=max(hi[g,k],centers[atom,k]+radii[atom])
    return lo,hi

@njit(cache=True)
def flight_near_box(p,d,length,lo,hi,padding):
    enter=0.; leave=length
    for k in range(3):
        a=lo[k]-padding; b=hi[k]+padding
        if d[k]==0.:
            if p[k]<a or p[k]>b: return False
        else:
            t0=(a-p[k])/d[k]; t1=(b-p[k])/d[k]
            if t0>t1: t0,t1=t1,t0
            enter=max(enter,t0); leave=min(leave,t1)
            if enter>leave: return False
    return True

@njit(cache=True)
def build_flights(points,directions,history,counts,ports,world,radii,rotation,center,groups,n_groups,padding):
    offsets=np.zeros(len(points)+1,dtype=np.int64)
    for i in range(len(points)): offsets[i+1]=offsets[i]+counts[i]-1+int(ports[i]<0)
    origins=np.zeros((offsets[-1],3)); dirs=np.zeros_like(origins); lengths=np.zeros(offsets[-1])
    local_world=(world-center)@rotation; lo,hi=support_bounds(local_world,radii,groups,n_groups)
    near=np.zeros((len(points),n_groups),dtype=np.bool_); valid=np.ones(len(points),dtype=np.bool_)
    tests=0
    for i in range(len(points)):
        p=points[i].copy(); d=directions[i].copy(); cursor=offsets[i]
        near[i,groups[history[i,0]]]=True
        for j in range(1,counts[i]):
            atom=history[i,j]; near[i,groups[atom]]=True
            distance=entry_distance(p,d,world[atom],radii[atom])
            origins[cursor]=(p-center)@rotation; dirs[cursor]=d@rotation; lengths[cursor]=distance; cursor+=1
            if not np.isfinite(distance): valid[i]=False; break
            reflect(p,d,atom,distance,world,radii)
        if not valid[i]: near[i,:]=True; continue
        if ports[i]<0:
            origins[cursor]=(p-center)@rotation; dirs[cursor]=d@rotation; lengths[cursor]=np.inf
        for g in range(n_groups):
            if near[i,g]: continue
            for j in range(offsets[i],offsets[i+1]):
                tests+=1
                if flight_near_box(origins[j],dirs[j],lengths[j],lo[g],hi[g],padding): near[i,g]=True; break
    return offsets,origins,dirs,lengths,near,valid,tests

@dataclass(frozen=True)
class FlightDependencies:
    offsets: np.ndarray
    origins: np.ndarray
    directions: np.ndarray
    lengths: np.ndarray
    near_supports: np.ndarray
    valid: np.ndarray
    build_box_tests: int

def record_flights(points,directions,history,counts,ports,world,radii,rotation,center,groups,n_groups,padding):
    values=build_flights(points,directions,history,counts,ports,world,radii,rotation,center,groups,n_groups,padding)
    return FlightDependencies(*(readonly(v) for v in values[:-1]),int(values[-1]))

@njit(cache=True)
def support_motion(current,recorded,groups,n_groups):
    result=np.zeros(n_groups)
    for i in range(len(current)):
        delta=current[i]-recorded[i]; distance=np.sqrt(np.dot(delta,delta)); g=groups[i]
        result[g]=max(result[g],distance)
    return result

@njit(cache=True)
def reject_changed_nearby(indices,offsets,origins,directions,lengths,near,valid,motion,lo,hi,tolerance,padding):
    reject=np.zeros(len(indices),dtype=np.bool_); old_reject=new_reject=tests=0
    for k,i in enumerate(indices):
        if not valid[i]: reject[k]=True; old_reject+=1; continue
        for g in range(len(motion)):
            if motion[g]<=tolerance: continue
            if near[i,g]: reject[k]=True; old_reject+=1; break
        if reject[k]: continue
        for g in range(len(motion)):
            if motion[g]<=tolerance: continue
            for j in range(offsets[i],offsets[i+1]):
                tests+=1
                if flight_near_box(origins[j],directions[j],lengths[j],lo[g],hi[g],padding):
                    reject[k]=True; new_reject+=1; break
            if reject[k]: break
    return reject,old_reject,new_reject,tests
