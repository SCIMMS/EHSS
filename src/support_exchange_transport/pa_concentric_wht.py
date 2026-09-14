"""Any-hit-only Walsh-Hadamard experiments on concentric shell transport.

For a shell boundary state q, h_j(q) in {0,1} is the intersection with atom j
within the next interface segment. The unnormalized Hadamard DC coefficient
is sum_j h_j, so DC > 0 is exact OR even with multiple overlapping hits.
No collision identity decoding or atom bit code is used.

The implementation exposes geometry-field generation separately from FWHT:
FWHT cannot supply missing geometric hit fields. Full FWHT, its exact DC-only
pruning, and an ordinary sum/OR control operate on the same fields.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from numba import njit

from .pa_concentric import ConcentricTransport,segment_box,segment_sphere,segment_tree


@njit(cache=True)
def fwht_inplace(values,width):
    """Unnormalized Sylvester FWHT; input width is a power of two."""
    stride=1
    while stride<width:
        for begin in range(0,width,2*stride):
            for j in range(begin,begin+stride):
                a=values[j]; b=values[j+stride]
                values[j]=a+b; values[j+stride]=a-b
        stride*=2


def fwht(values):
    values=np.asarray(values)
    if values.ndim!=1 or len(values)<1 or len(values)&(len(values)-1):
        raise ValueError("A nonempty power-of-two vector is required.")
    result=np.asarray(values,dtype=np.float64).copy()
    fwht_inplace(result,len(result))
    return result


@njit(cache=True)
def hit_field(o,d,owner,begin,finish,centers,radii,
              lo,hi,end,first,count,ids,root,slots,field,write_field):
    """A BVH supplies certified zeros outside the segment; no any-hit exit.

    A slot number is just the array column for an atom contribution, not an
    identity code. The caller never decodes which sphere caused a collision.
    """
    if root<0:
        return 0,0,0
    node=root; stop=end[root]; boxes=0; tests=0; hits=0
    while node<stop:
        boxes+=1
        if not segment_box(o,d,lo[node],hi[node],begin,finish):
            node=end[node]; continue
        for k in range(first[node],first[node]+count[node]):
            atom=ids[k]
            if atom==owner:
                continue
            tests+=1
            if segment_sphere(o,d,centers[atom],radii[atom],begin,finish):
                hits+=1
                if write_field:
                    field[slots[atom]]=1.
        node+=1
    return hits,boxes,tests


@njit(cache=True)
def next_segment(origin,direction,center,boundaries,begin,layer,inward):
    x=origin[0]-center[0]; y=origin[1]-center[1]; z=origin[2]-center[2]
    b=x*direction[0]+y*direction[1]+z*direction[2]
    impact2=max(0.,x*x+y*y+z*z-b*b)
    inner=boundaries[layer-1] if layer>0 else 0.
    if inward and layer>0 and inner*inner>impact2:
        finish=max(begin,-b-np.sqrt(max(0.,inner*inner-impact2)))
        return finish,layer-1,True
    finish=max(begin,-b+np.sqrt(max(0.,boundaries[layer]**2-impact2)))
    return finish,layer+1,False


@njit(cache=True)
def flat_hit_field(o,d,owner,begin,finish,centers,radii,members,start,stop,
                   slots,field,write_field,early_exit):
    """Direct shell-member scan; no tree construction, boxes, or traversal."""
    hits=0; tests=0
    for k in range(start,stop):
        atom=members[k]
        if atom==owner:
            continue
        tests+=1
        if segment_sphere(o,d,centers[atom],radii[atom],begin,finish):
            hits+=1
            if write_field:
                field[slots[atom]]=1.
            if early_exit:
                break
    return hits,0,tests


@njit(cache=True)
def online_transport(origins,directions,owners,center,boundaries,centers,radii,
                     roots,lo,hi,end,first,count,ids,widths,slots,mode,
                     use_bvh,offsets,members):
    """mode 0: FWHT; 1: only Hadamard DC (fused sum); 2: explicit sum control.

    Mode 2 fills exactly the same field as mode 0 but uses an ordinary reduction.
    Mode 1 computes exactly the only coefficient used without a field buffer.
    All modes terminate packets after a positive DC and preserve nonradial paths.
    """
    blocked=np.zeros(len(origins),dtype=np.bool_)
    visits=np.zeros(len(boundaries),dtype=np.int64)
    absorbed=np.zeros(len(boundaries),dtype=np.int64)
    boxes=0; tests=0; transforms=0; additions=0; multi_hit=0
    field=np.empty(np.max(widths) if mode in (0,2) else 1,dtype=np.float64)
    for ray in range(len(origins)):
        o=origins[ray]; d=directions[ray]
        x=o[0]-center[0]; y=o[1]-center[1]; z=o[2]-center[2]
        b=x*d[0]+y*d[1]+z*d[2]
        layer=min(np.searchsorted(boundaries,np.sqrt(x*x+y*y+z*z),side="right"),len(boundaries)-1)
        inward=b<0; begin=0.
        while layer<len(boundaries):
            finish,nxt,inward=next_segment(o,d,center,boundaries,begin,layer,inward)
            if finish>begin:
                visits[layer]+=1
                width=widths[layer]
                if mode in (0,2):
                    field[:width]=0.
                if not use_bvh:
                    hits,nb,nt=flat_hit_field(o,d,owners[ray],begin,finish,centers,radii,
                        members,offsets[layer],offsets[layer+1],slots[layer],field,
                        mode==0 or mode==2,mode==3)
                elif mode==3:
                    hits,nb,nt=False,0,0
                    if roots[layer]>=0:
                        hits,nb,nt=segment_tree(o,d,owners[ray],begin,finish,centers,radii,
                            lo,hi,end,first,count,ids,roots[layer])
                else:
                    hits,nb,nt=hit_field(o,d,owners[ray],begin,finish,centers,radii,
                        lo,hi,end,first,count,ids,roots[layer],slots[layer],field,mode!=1)
                boxes+=nb; tests+=nt
                if hits>1:
                    multi_hit+=1
                if mode==0:
                    fwht_inplace(field,width)
                    transforms+=1
                    additions+=width*int(np.log2(width))
                    dc=field[0]
                elif mode==1 or mode==3:
                    dc=float(hits)
                else:
                    dc=0.
                    for j in range(width):
                        dc+=field[j]
                    additions+=width
                if dc>0:
                    blocked[ray]=True; absorbed[layer]+=1; break
            begin=finish; layer=nxt
    return blocked,boxes,tests,visits,absorbed,transforms,additions,multi_hit


@dataclass
class CompiledShellField:
    """A Boolean transport field on a REGISTERED discrete boundary-state set.

    No unseen origin/direction can query this table. Reweighting its existing
    quadrature states is allowed; geometry or boundary-state changes recompile.
    Every method stores Boolean hit only, never collider identities.
    """
    hits: np.ndarray
    valid: np.ndarray
    area: float
    metrics: dict

    def transport(self,weights=None):
        return replay_field(self.hits,self.valid,weights if weights is not None else np.ones(len(self.hits)))*(self.area/(4*len(self.hits)))

    @property
    def nbytes(self):
        return self.hits.nbytes+self.valid.nbytes


@njit(cache=True)
def replay_field(hits,valid,weights):
    if len(weights)!=len(hits):
        raise ValueError("Weights must belong to the registered boundary states.")
    total=0.
    for ray in range(len(hits)):
        weight=weights[ray]
        if not np.isfinite(weight) or weight<0:
            raise ValueError("Finite nonnegative flux weights required.")
        for step in range(hits.shape[1]):
            if valid[ray,step] and hits[ray,step]:
                weight=0.; break
        total+=weight
    return total


@njit(cache=True)
def build_paths(origins,directions,center,boundaries):
    steps=2*len(boundaries)+1
    layers=np.full((len(origins),steps),-1,dtype=np.int32)
    begins=np.zeros((len(origins),steps)); finishes=np.zeros_like(begins)
    for ray in range(len(origins)):
        rel=origins[ray]-center
        layer=min(np.searchsorted(boundaries,np.sqrt(np.dot(rel,rel)),side="right"),len(boundaries)-1)
        inward=np.dot(rel,directions[ray])<0; begin=0.; step=0
        while layer<len(boundaries):
            finish,nxt,inward=next_segment(origins[ray],directions[ray],center,boundaries,begin,layer,inward)
            if finish>begin:
                layers[ray,step]=layer; begins[ray,step]=begin; finishes[ray,step]=finish; step+=1
            begin=finish; layer=nxt
    return layers,begins,finishes


@njit(cache=True)
def batch_fields(ray_ids,step,origins,directions,owners,layers,begins,finishes,
                 centers,radii,roots,lo,hi,end,first,count,ids,widths,slots,mode,buffer,
                 use_bvh,offsets,members):
    boxes=0; tests=0; multiplicities=np.zeros(len(ray_ids),dtype=np.int32)
    dummy=np.empty(1)
    for row in range(len(ray_ids)):
        ray=ray_ids[row]; layer=layers[ray,step]
        if not use_bvh:
            if mode==0 or mode==2:
                buffer[row,:widths[layer]]=0.
            hit,nb,nt=flat_hit_field(origins[ray],directions[ray],owners[ray],
                begins[ray,step],finishes[ray,step],centers,radii,members,
                offsets[layer],offsets[layer+1],slots[layer],
                buffer[row] if mode==0 or mode==2 else dummy,
                mode==0 or mode==2,mode==3)
            multiplicities[row]=hit; tests+=nt
        elif mode==3:
            if roots[layer]>=0:
                hit,nb,nt=segment_tree(origins[ray],directions[ray],owners[ray],begins[ray,step],finishes[ray,step],
                    centers,radii,lo,hi,end,first,count,ids,roots[layer])
                multiplicities[row]=int(hit); boxes+=nb; tests+=nt
        else:
            if mode!=1:
                buffer[row,:widths[layer]]=0.
            hit,nb,nt=hit_field(origins[ray],directions[ray],owners[ray],begins[ray,step],finishes[ray,step],
                centers,radii,lo,hi,end,first,count,ids,roots[layer],slots[layer],dummy if mode==1 else buffer[row],mode!=1)
            multiplicities[row]=hit; boxes+=nb; tests+=nt
    return multiplicities,boxes,tests


@njit(cache=True)
def transform_rows(buffer,ray_ids,layers,widths,step):
    additions=0
    for row,ray in enumerate(ray_ids):
        width=widths[layers[ray,step]]
        fwht_inplace(buffer[row],width)
        additions+=width*int(np.log2(width))
    return additions


@njit(cache=True)
def read_rows(buffer,ray_ids,layers,widths,step,mode,multiplicities):
    result=np.empty(len(ray_ids),dtype=np.bool_)
    for row,ray in enumerate(ray_ids):
        if mode==0:
            dc=buffer[row,0]
        elif mode==1 or mode==3:
            dc=multiplicities[row]
        else:
            dc=0.
            for k in range(widths[layers[ray,step]]):
                dc+=buffer[row,k]
        result[row]=dc>0
    return result


class ConcentricWHT:
    modes={"fwht":0,"dc_only":1,"sum_control":2,"anyhit_control":3,"dc_predicate":3}

    def __init__(self,spheres,n_shells=8,use_bvh=True):
        self.shells=ConcentricTransport(spheres,n_shells,use_bvh=use_bvh)
        self.widths=np.array([1<<max(0,int(self.shells.offsets[i+1]-self.shells.offsets[i])-1).bit_length()
                              for i in range(n_shells)],dtype=np.int64)
        self.slots=np.full((n_shells,len(spheres.radii)),-1,dtype=np.int32)
        for layer in range(n_shells):
            atoms=self.shells.primitive_ids[self.shells.offsets[layer]:self.shells.offsets[layer+1]]
            self.slots[layer,atoms]=np.arange(len(atoms),dtype=np.int32)

    def query(self,source,method="fwht"):
        mode=self.modes[method]
        s=self.shells
        return online_transport(source.origins,source.directions,source.owners,s.center,s.boundaries,
            s.spheres.centers,s.spheres.radii,s.roots,*s.tree.arrays,self.widths,self.slots,mode,
            s.use_bvh,s.offsets,s.primitive_ids)

    def compile_field(self,source,method="fwht",chunk_size=512):
        """Wavefront compilation with early absorption and explicit phase costs.

        Only active packets advance. A bounded work buffer holds hit channels
        for one chunk, preventing a dense states-by-atoms-by-shells allocation.
        Full FWHT really executes every butterfly; only DC is read afterward.
        """
        if chunk_size<1:
            raise ValueError("Positive chunk size required.")
        mode=self.modes[method]; s=self.shells
        total_start=time.perf_counter()
        start=time.perf_counter()
        layers,begins,finishes=build_paths(source.origins,source.directions,s.center,s.boundaries)
        path_s=time.perf_counter()-start
        hits=np.zeros_like(layers,dtype=np.bool_); valid=np.zeros_like(hits)
        alive=np.ones(len(source.owners),dtype=np.bool_)
        width=int(np.max(self.widths)) if mode in [0,2] else 1
        buffer=np.empty((min(chunk_size,len(alive)),width))
        metrics=dict(path_s=path_s,field_s=0.,transform_s=0.,detect_s=0.,sphere_tests=0,boxes=0,
                     transformed_rows=0,butterfly_additions=0,multiple_hit_states=0,visited_states=0,
                     work_buffer_bytes=buffer.nbytes,path_buffer_bytes=layers.nbytes+begins.nbytes+finishes.nbytes,
                     compile_s=0.)
        for step in range(layers.shape[1]):
            ray_ids=np.flatnonzero(alive&(layers[:,step]>=0))
            if not len(ray_ids):
                continue
            for offset in range(0,len(ray_ids),chunk_size):
                subset=ray_ids[offset:offset+chunk_size]
                start=time.perf_counter()
                counts,boxes,tests=batch_fields(subset,step,source.origins,source.directions,source.owners,
                    layers,begins,finishes,s.spheres.centers,s.spheres.radii,s.roots,*s.tree.arrays,
                    self.widths,self.slots,mode,buffer,s.use_bvh,s.offsets,s.primitive_ids)
                metrics["field_s"]+=time.perf_counter()-start
                metrics["sphere_tests"]+=tests; metrics["boxes"]+=boxes
                metrics["visited_states"]+=len(subset)
                if mode!=3:
                    metrics["multiple_hit_states"]+=int(np.count_nonzero(counts>1))
                if mode==0:
                    start=time.perf_counter()
                    metrics["butterfly_additions"]+=transform_rows(buffer,subset,layers,self.widths,step)
                    metrics["transform_s"]+=time.perf_counter()-start
                    metrics["transformed_rows"]+=len(subset)
                start=time.perf_counter()
                blocked=read_rows(buffer,subset,layers,self.widths,step,mode,counts)
                metrics["detect_s"]+=time.perf_counter()-start
                hits[subset,step]=blocked; valid[subset,step]=True
                alive[subset[blocked]]=False
        metrics["compile_s"]=time.perf_counter()-total_start
        return CompiledShellField(hits,valid,source.area,metrics)

    @property
    def nbytes(self):
        return self.shells.nbytes+self.widths.nbytes+self.slots.nbytes
