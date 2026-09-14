"""Reuse previous material-observation paths as guarded current BVH limits.

Cached collider IDs are search hints only. Every candidate is intersected in
the current geometry, and the full hierarchy still arbitrates nearer hits and
ties. A cached escape or exhausted prefix never certifies current escape.
"""
from collections import defaultdict,deque
import time
import numpy as np
from numba import njit
from .ehss_reference import entry_distance
from .ehss_guarded_intrinsic_response import nearest_world,reflect
from .ehss_state import EHSSResult
from .ehss_material_collision_field import MaterialCollisionField,first_observed_successors
from .ehss_sparse_path_response import readonly


@njit(cache=False)
def trace_with_hints(origins,directions,lasts,initial,cap,hints,lengths,
                     lo,hi,left,right,end,leaf_group,offsets,ids,centers,radii):
    p=origins.copy(); d=directions.copy(); counts=initial.copy()
    history=np.full((len(p),cap),-1,dtype=np.int32)
    escaped=np.zeros(len(p),dtype=np.bool_); unresolved=escaped.copy()
    counters=np.zeros(2,dtype=np.int64); stats=np.zeros(5,dtype=np.int64)
    for ray in range(len(p)):
        last=lasts[ray]; count=initial[ray]; prefix=True
        if count==1:
            history[ray,0]=last
            prefix=lengths[ray]>0 and hints[ray,0]==last
        while True:
            candidate=-1; limit=np.inf
            if prefix and count<lengths[ray]:
                wanted=hints[ray,count]
                if wanted>=0 and wanted<len(radii) and wanted!=last:
                    stats[0]+=1
                    distance=entry_distance(p[ray],d[ray],centers[wanted],radii[wanted])
                    if np.isfinite(distance): candidate=wanted; limit=distance; stats[1]+=1
            atom,distance=nearest_world(0,p[ray],d[ray],last,limit,-1,lo,hi,left,right,end,
                leaf_group,offsets,ids,centers,radii,counters)
            stats[4]+=1
            if candidate>=0 and (atom<0 or limit<distance or (limit==distance and candidate<atom)):
                atom=candidate; distance=limit
            if atom<0: escaped[ray]=True; break
            if candidate>=0:
                if atom==candidate: stats[2]+=1
                else: stats[3]+=1
            if prefix and (count>=lengths[ray] or atom!=hints[ray,count]): prefix=False
            if count>=cap: unresolved[ray]=True; break
            reflect(p[ray],d[ray],atom,distance,centers,radii)
            history[ray,count]=atom; count+=1; last=atom
        counts[ray]=count
    return p,d,escaped,unresolved,counts,history,counters,stats


def hinted_trace(model,source,cap,hints,lengths):
    if not isinstance(cap,(int,np.integer)) or cap<1 or np.any(source.initial_bounces>1): raise ValueError('Positive cap and uncollided/first-reflected source required')
    if np.any(source.last_atom < -1) or np.any(source.last_atom>=len(model.spheres.radii)): raise ValueError('Invalid initial atom')
    hints=np.asarray(hints,dtype=np.int32); lengths=np.asarray(lengths,dtype=np.int64)
    if hints.shape!=(len(source.weights),cap) or lengths.shape!=(len(source.weights),) or np.any(lengths<0) or np.any(lengths>cap): raise ValueError('Invalid path hint shape or lengths')
    t=model.tree; s=model.spheres; tick=time.perf_counter()
    p,d,e,u,n,h,counters,stats=trace_with_hints(source.origins,source.directions,source.last_atom,source.initial_bounces,cap,hints,lengths,
        t.lo,t.hi,t.left,t.right,t.end,t.leaf_group,t.offsets,t.ids,s.centers,s.radii)
    return EHSSResult(p,d,e,u,n,h,source,dict(query_s=time.perf_counter()-tick,
        box_tests=int(counters[0]),sphere_tests=int(counters[1]+stats[0]),bvh_sphere_tests=int(counters[1]),
        hint_sphere_tests=int(stats[0]),finite_hints=int(stats[1]),winning_hints=int(stats[2]),overruled_hints=int(stats[3]),
        nearest_queries=int(stats[4]),backend='Current full BVH search bounded by previous material-path hints'))


class CouplingPathPool:
    def __init__(self,bank): self.bank=bank; self.paths={}
    def snapshot(self):
        result=CouplingPathPool(self.bank); result.paths=self.paths.copy(); return result


def _source_key(atom,point,direction): return int(atom),point.tobytes(),direction.tobytes()


class _CouplingTrace:
    def __init__(self,model,bound,observed,pool,use_hints):
        tick=time.perf_counter()
        if pool.bank is not bound.bank: raise ValueError('Path pool belongs to a different material bank')
        self.model=model; self.pool=pool; self.use_hints=use_hints; self.calls=[]
        s=model.spheres
        p,d,atoms=first_observed_successors(bound.points,bound.directions,observed['counts'],observed['history'],s.centers,s.radii)
        self.rows=defaultdict(deque)
        for row in np.flatnonzero((observed['terminal']!=-3)&(atoms>=0)):
            self.rows[_source_key(atoms[row],p[row],d[row])].append(int(row))
        self.prepare_s=time.perf_counter()-tick
    def __getattr__(self,name): return getattr(self.model,name)
    def trace(self,source,max_bounces=128):
        if source.convention!='missing material target ports': raise ValueError('Coupling-only trace adapter')
        tick=time.perf_counter(); rows=[]
        for atom,p,d in zip(source.last_atom,source.origins,source.directions):
            entries=self.rows.get(_source_key(atom,p,d))
            if not entries: raise ValueError('Coupling source is not a current material observation')
            rows.append(entries.popleft())
        hints=np.full((len(rows),max_bounces),-1,dtype=np.int32); lengths=np.zeros(len(rows),dtype=np.int64)
        reused=0
        if self.use_hints:
            for ray,row in enumerate(rows):
                old=self.pool.paths.get(row)
                if old is not None:
                    count=min(len(old),max_bounces); hints[ray,:count]=old[:count]; lengths[ray]=count; reused+=1
        lookup_s=time.perf_counter()-tick
        result=hinted_trace(self.model,source,max_bounces,readonly(hints),readonly(lengths))
        tick=time.perf_counter()
        for ray,row in enumerate(rows): self.pool.paths[row]=readonly(result.collider_ids[ray,:result.bounces[ray]].copy())
        self.calls.append(dict(rows=len(rows),reused_rows=reused,lookup_s=lookup_s,store_s=time.perf_counter()-tick,**result.metrics))
        return result


def build_hinted_coupling(bound,observed,model,pool,bins=2,use_hints=True):
    tick=time.perf_counter(); adapter=_CouplingTrace(model,bound,observed,pool,use_hints)
    field=MaterialCollisionField(bound,observed,adapter,bins=bins)
    # Query-time unknown-first-port suffixes keep the ordinary physical model;
    # only coupling preparation has stable material-observation row identities.
    field.model=model
    receipt=dict(total_prepare_s=time.perf_counter()-tick,source_registration_s=adapter.prepare_s,
        field_prepare_s=field.receipt['prepare_s'],calls=adapter.calls,use_hints=bool(use_hints),
        stored_paths=len(pool.paths),stored_path_bytes=sum(p.nbytes for p in pool.paths.values()),
        current_geometry_arbitration=True,cached_escape_never_accepted=True)
    return field,receipt
