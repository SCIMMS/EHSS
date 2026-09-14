"""Wavefront phase profiler, checked against the production packet traversal.

Boundary advance, geometric any-hit and active-state update are timed in batches.
The executed segments and geometry counters match production; scheduling and
memory traffic differ, so these timings are NOT a decomposition of its wall time.
"""
import time
import numpy as np
from numba import njit
from .pa_concentric import ConcentricTransport,segment_tree,segment_sphere


@njit(cache=True)
def initialize(origins,directions,center,metric_map,boundaries,is_sphere):
    n=len(origins)
    coefficients=np.empty((n,4)); layers=np.empty(n,dtype=np.int64)
    inward=np.empty(n,dtype=np.bool_)
    for i in range(n):
        aa=0.; bb=0.; cc=0.
        if is_sphere:
            aa=1.
            for j in range(3):
                x=origins[i,j]-center[j]
                bb+=x*directions[i,j]; cc+=x*x
        else:
            for k in range(3):
                x=0.; v=0.
                for j in range(3):
                    x+=(origins[i,j]-center[j])*metric_map[j,k]
                    v+=directions[i,j]*metric_map[j,k]
                aa+=v*v; bb+=x*v; cc+=x*x
        coefficients[i,0]=aa; coefficients[i,1]=bb; coefficients[i,2]=cc
        coefficients[i,3]=max(0.,cc-bb*bb/aa)
        layers[i]=min(np.searchsorted(boundaries,np.sqrt(cc),side="right"),len(boundaries)-1)
        inward[i]=bb<0
    return coefficients,layers,inward


@njit(cache=True)
def advance(active,coef,layers,inward,begins,finishes,next_layers,boundaries,visits):
    for ray in active:
        layer=layers[ray]; aa=coef[ray,0]; bb=coef[ray,1]; minimum=coef[ray,3]
        inner=boundaries[layer-1] if layer>0 else 0.
        if inward[ray] and layer>0 and inner*inner>minimum:
            finish=max(begins[ray],(-bb-np.sqrt(max(0.,aa*(inner*inner-minimum))))/aa)
            next_layers[ray]=layer-1
        else:
            inward[ray]=False
            finish=max(begins[ray],(-bb+np.sqrt(max(0.,aa*(boundaries[layer]**2-minimum))))/aa)
            next_layers[ray]=layer+1
        finishes[ray]=finish
        if finish>begins[ray]:
            visits[layer]+=1


@njit(cache=True)
def collisions(active,origins,directions,owners,layers,begins,finishes,centers,radii,
               offsets,members,roots,lo,hi,end,first,count,ids,use_bvh,step_hits):
    boxes=0; tests=0
    for ray in active:
        step_hits[ray]=False
        if finishes[ray]<=begins[ray]:
            continue
        layer=layers[ray]
        if use_bvh and roots[layer]>=0:
            hit,nb,nt=segment_tree(origins[ray],directions[ray],owners[ray],begins[ray],finishes[ray],
                centers,radii,lo,hi,end,first,count,ids,roots[layer])
            step_hits[ray]=hit; boxes+=nb; tests+=nt
        elif not use_bvh:
            for p in range(offsets[layer],offsets[layer+1]):
                atom=members[p]
                if atom==owners[ray]:
                    continue
                tests+=1
                if segment_sphere(origins[ray],directions[ray],centers[atom],radii[atom],begins[ray],finishes[ray]):
                    step_hits[ray]=True; break
    return boxes,tests


@njit(cache=True)
def update(active,layers,next_layers,begins,finishes,step_hits,blocked,absorbed,n_shells):
    following=np.empty(len(active),dtype=np.int64); count=0
    for ray in active:
        if step_hits[ray]:
            blocked[ray]=True; absorbed[layers[ray]]+=1
        elif next_layers[ray]<n_shells:
            begins[ray]=finishes[ray]; layers[ray]=next_layers[ray]
            following[count]=ray; count+=1
    return following[:count]


def profile_query(index,source,enable_collisions=True):
    start=time.perf_counter(); stamp=start
    is_sphere=isinstance(index,ConcentricTransport)
    metric_map=np.eye(3) if is_sphere else index.metric_map
    coef,layers,inward=initialize(source.origins,source.directions,index.center,metric_map,index.boundaries,is_sphere)
    inward_count=int(inward.sum())
    n=len(source.owners); active=np.arange(n,dtype=np.int64)
    begins=np.zeros(n); finishes=np.empty(n); next_layers=np.empty(n,dtype=np.int64)
    step_hits=np.zeros(n,dtype=np.bool_); blocked=np.zeros(n,dtype=np.bool_)
    visits=np.zeros(len(index.boundaries),dtype=np.int64); absorbed=np.zeros_like(visits)
    metrics=dict(initialize_s=time.perf_counter()-stamp,boundary_s=0.,collision_s=0.,state_update_s=0.,
                 sphere_tests=0,boxes=0,passes=0)
    while len(active):
        metrics["passes"]+=1
        if metrics["passes"]>2*len(index.boundaries)+1:
            raise RuntimeError("Nonterminating interface traversal")
        stamp=time.perf_counter()
        advance(active,coef,layers,inward,begins,finishes,next_layers,index.boundaries,visits)
        metrics["boundary_s"]+=time.perf_counter()-stamp
        if enable_collisions:
            stamp=time.perf_counter()
            boxes,tests=collisions(active,source.origins,source.directions,source.owners,layers,begins,finishes,
                index.spheres.centers,index.spheres.radii,index.offsets,index.primitive_ids,index.roots,
                *index.tree.arrays,index.use_bvh,step_hits)
            metrics["collision_s"]+=time.perf_counter()-stamp
            metrics["boxes"]+=boxes; metrics["sphere_tests"]+=tests
        stamp=time.perf_counter()
        active=update(active,layers,next_layers,begins,finishes,step_hits,blocked,absorbed,len(index.boundaries))
        metrics["state_update_s"]+=time.perf_counter()-stamp
    stamp=time.perf_counter()
    pa=source.pa(blocked) if enable_collisions else None
    metrics["readout_s"]=time.perf_counter()-stamp
    total=time.perf_counter()-start
    metrics["other_s"]=total-sum(v for k,v in metrics.items() if k.endswith("_s"))
    metrics["total_s"]=total
    metrics.update(segments=int(visits.sum()),inward_sources=inward_count,
                   visits=visits.tolist(),absorbed=absorbed.tolist(),enable_collisions=enable_collisions)
    return blocked,pa,metrics
