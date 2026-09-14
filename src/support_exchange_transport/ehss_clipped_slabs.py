"""Exact local responses on disjoint parallel-plane domains.

A physical atom may be a candidate in several domains. Its coordinates and ID
are stored once; domain membership is spatial, not an atom ownership partition.
Each response runs until its domain boundary or a physical cap. Virtual passage
preserves direction, weight, last collider, and the physical flight anchor plus
cursor, avoiding coordinate nudges. This is a boundary-contract prototype, not
the full hierarchy compiler or a fitted molecular collision model.
"""
import time
import numpy as np
from numba import njit
from .inside_out_pa import Spheres
from .ehss_reference import entry_distance
from .ehss_state import EHSSResult


@njit(cache=True)
def locate_slab(point,direction,normal,edges):
    q=np.dot(point,normal)
    # On a face, the outgoing half-space owns the next state. Parallel rays
    # consistently use the positive side. Sphere candidates include face ties.
    if np.dot(direction,normal)<0.:
        return np.searchsorted(edges,q,side='left')
    return np.searchsorted(edges,q,side='right')


@njit(cache=True)
def slab_response(domain,o,d,cursor,last,bounces,cap,normal,edges,offsets,ids,centers,radii,history):
    """Return (domain,cursor,last,count,status,tests); 0=handoff,1=escape,2=cap."""
    tests=0
    while True:
        speed=np.dot(d,normal)
        face=np.inf
        next_domain=domain
        if speed>0. and domain<len(edges):
            face=(edges[domain]-np.dot(o,normal))/speed
            next_domain=domain+1
        elif speed<0. and domain>0:
            face=(edges[domain-1]-np.dot(o,normal))/speed
            next_domain=domain-1
        if face<cursor:
            raise ValueError('Domain cursor moved past its next boundary')
        best=np.inf;atom=-1
        for k in range(offsets[domain],offsets[domain+1]):
            j=ids[k]
            if j==last:continue
            tests+=1
            distance=entry_distance(o,d,centers[j],radii[j])
            if distance>=cursor and (distance<best or (distance==best and np.isfinite(distance) and (atom<0 or j<atom))):
                best=distance;atom=j
        # A physical entry exactly on a face wins before virtual passage.
        if atom>=0 and best<=face:
            if bounces>=cap:return domain,cursor,last,bounces,2,tests
            o[:]=o+best*d
            n=o-centers[atom];n/=np.sqrt(np.dot(n,n))
            o[:]=centers[atom]+radii[atom]*n
            d[:]=d-2*np.dot(d,n)*n
            d[:]/=np.sqrt(np.dot(d,d))
            history[bounces]=atom;bounces+=1;last=atom;cursor=0.
            after=locate_slab(o,d,normal,edges)
            if after!=domain:return after,cursor,last,bounces,0,tests
        elif np.isfinite(face):
            return next_domain,face,last,bounces,0,tests
        else:
            return domain,cursor,last,bounces,1,tests


@njit(cache=True)
def trace_slabs(origins,directions,last_atoms,initial_bounces,cap,normal,edges,offsets,ids,centers,radii):
    p,d=origins.copy(),directions.copy()
    bounces=initial_bounces.copy()
    history=np.full((len(origins),cap),-1,dtype=np.int32)
    escaped=np.zeros(len(origins),dtype=np.bool_)
    unresolved=np.zeros(len(origins),dtype=np.bool_)
    n_domains=len(edges)+1
    calls=np.zeros(n_domains,dtype=np.int64)
    collisions=np.zeros(n_domains,dtype=np.int64)
    multiple=np.zeros(n_domains,dtype=np.int64)
    handoffs=np.zeros(n_domains,dtype=np.int64)
    tests=0
    for ray in range(len(origins)):
        o=p[ray];out=d[ray];last=last_atoms[ray];cursor=0.
        if initial_bounces[ray]==1:history[ray,0]=last
        domain=locate_slab(o,out,normal,edges)
        finished=False
        # Between physical reflections a straight ray crosses each plane at
        # most once. This bound diagnoses transport cycles; it is not a cutoff.
        for _ in range((cap+1)*(n_domains+1)+1):
            previous=domain;before=bounces[ray];calls[previous]+=1
            domain,cursor,last,bounces[ray],status,nt=slab_response(domain,o,out,cursor,last,bounces[ray],cap,
                normal,edges,offsets,ids,centers,radii,history[ray])
            tests+=nt;added=bounces[ray]-before;collisions[previous]+=added
            if added>=2:multiple[previous]+=1
            if status==0:handoffs[previous]+=1
            else:
                escaped[ray]=status==1;unresolved[ray]=status==2;finished=True;break
        if not finished:raise ValueError('Virtual domain exchange exceeded physical-flight bound')
    return p,d,escaped,unresolved,bounces,history,tests,calls,collisions,multiple,handoffs


class ClippedSlabTransport:
    def __init__(self,spheres,normal,edges):
        tick=time.perf_counter()
        n=np.asarray(normal,dtype=float)
        e=np.asarray(edges,dtype=float)
        if n.shape!=(3,) or not np.isfinite(n).all() or np.linalg.norm(n)==0.:
            raise ValueError('Finite nonzero plane normal required')
        if e.ndim!=1 or not np.isfinite(e).all() or np.any(np.diff(e)<=0):
            raise ValueError('Strictly increasing finite plane coordinates required')
        self.normal=n/np.linalg.norm(n)
        self.edges=e.copy()
        self.spheres=Spheres(spheres.centers,spheres.radii,deduplicate=False)
        q=self.spheres.centers@self.normal
        # Pad candidate inclusion only; do not shift physical event distances
        # or ray origins. Retain both candidates at shared/tangent faces.
        guard=128*np.finfo(float).eps*(1.+np.abs(self.spheres.centers)@np.abs(self.normal)+self.spheres.radii)
        bounds=np.r_[-np.inf,e,np.inf]
        groups=[]
        for a,b in zip(bounds[:-1],bounds[1:]):
            groups.append(np.flatnonzero((q+self.spheres.radii+guard>=a)&(q-self.spheres.radii-guard<=b)))
        self.offsets=np.r_[0,np.cumsum([len(g) for g in groups])].astype(np.int64)
        self.ids=np.concatenate(groups).astype(np.int64)
        for x in (self.normal,self.edges,self.offsets,self.ids,self.spheres.centers,self.spheres.radii):x.setflags(write=False)
        self.receipt=dict(build_s=time.perf_counter()-tick,domains=len(groups),physical_atoms=len(q),candidate_references=len(self.ids),
                          candidate_counts=[len(g) for g in groups],duplicated_candidate_references=len(self.ids)-len(q),
                          physical_geometry_bytes=self.spheres.centers.nbytes+self.spheres.radii.nbytes,
                          partition_array_bytes=self.normal.nbytes+self.edges.nbytes+self.offsets.nbytes+self.ids.nbytes)

    def trace(self,source,max_bounces=128):
        if not isinstance(max_bounces,(int,np.integer)) or max_bounces<1 or np.any(source.initial_bounces>1):
            raise ValueError('Positive cap and uncollided/first-reflected source required')
        if np.any(source.last_atom>=len(self.spheres.radii)) or np.any(source.last_atom< -1):
            raise ValueError('Invalid physical collider ID')
        tick=time.perf_counter()
        result=trace_slabs(source.origins,source.directions,source.last_atom,source.initial_bounces,max_bounces,
                           self.normal,self.edges,self.offsets,self.ids,self.spheres.centers,self.spheres.radii)
        p,d,e,u,b,h,tests,calls,collisions,multiple,handoffs=result
        return EHSSResult(p,d,e,u,b,h,source,dict(query_s=time.perf_counter()-tick,backend='clipped parallel-plane local responses',
            sphere_tests=int(tests),domain_calls=calls.tolist(),domain_collisions=collisions.tolist(),
            multi_collision_responses=multiple.tolist(),boundary_handoffs=handoffs.tolist(),geometry_copies_per_atom=1))
