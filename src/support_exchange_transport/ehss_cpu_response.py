"""Certified subtree response ON/OFF on the single-thread scalar CPU kernel.

Owned itineraries skip owned search only. Actual sphere entry and reflection
are evaluated with the same scalar functions as CPUEHSSTransport, and current
foreign geometry is checked on every leg. This is not an approximate field.
"""
import time
import numpy as np
from numba import njit
from .ehss_cpu import CPUEHSSTransport, entry_scalar, reflect_scalar
from .ehss_support_tree import box_interval
from .ehss_selected_bvh_transport import SelectedBVHTransport
from .ehss_state import EHSSResult


@njit
def nearest_skip(o,d,last,limit,skip,lo,hi,end,leaf,offsets,ids,world,radii,counters):
    node,atom,best=0,-1,limit
    while node<len(end):
        if node==skip:
            node=end[node]; continue
        counters[0]+=1
        near,far=box_interval(o,d,lo[node],hi[node])
        if near>best or far<0.:
            node=end[node]; continue
        group=leaf[node]
        if group>=0:
            for k in range(offsets[group],offsets[group+1]):
                candidate=ids[k]
                if candidate==last: continue
                counters[1]+=1
                distance=entry_scalar(o,d,world[candidate],radii[candidate])
                if distance<best or (distance==best and np.isfinite(distance) and (atom<0 or candidate<atom)):
                    atom,best=candidate,distance
        node+=1
    return atom,best


@njit
def replay(node,o,d,last,count,cap,history,record,responses,
           lo,hi,end,leaf,offsets,ids,world,radii,counters,stats):
    starts,low,high,last_keys,path_offsets,paths,rotations,translations,charts=responses
    local_o=(o-translations[node])@rotations[node]; local_d=d@rotations[node]
    for cell in range(starts[node],starts[node+1]):
        stats[0]+=1
        if last_keys[cell]!=last: continue
        matched=True
        for c in range(3):
            if local_o[c]<low[cell,c] or local_o[c]>high[cell,c] or local_d[c]<low[cell,c+3] or local_d[c]>high[cell,c+3]:
                matched=False; break
        if not matched: continue
        first,stop=path_offsets[cell],path_offsets[cell+1]
        if first==stop: continue
        if count+stop-first>cap: stats[7]+=1; continue
        stats[1]+=1
        p,v=o.copy(),d.copy(); previous=last; done=0; interrupted=False; failed=False
        for k in range(first,stop):
            atom=paths[k]
            if atom==previous: failed=True; break
            distance=entry_scalar(p,v,world[atom],radii[atom]); stats[6]+=1
            if not np.isfinite(distance): failed=True; break
            stats[4]+=1
            foreign,before=nearest_skip(p,v,previous,distance,node,lo,hi,end,leaf,offsets,ids,world,radii,counters)
            if foreign>=0 and (before<distance or (before==distance and foreign<atom)):
                interrupted=True; stats[3]+=1; break
            reflect_scalar(p,v,atom,distance,world,radii)
            previous=atom; done+=1
        if failed: stats[5]+=1; continue
        if done==0: continue
        o[:]=p; d[:]=v
        if record:
            for k in range(done): history[count+k]=paths[first+k]
        stats[2]+=1; stats[8]+=done
        if done>=2: stats[9]+=1
        if interrupted: stats[10]+=1
        return True,previous,count+done,-1 if interrupted else node
    return False,last,count,-1


@njit
def trace_response(origins,directions,lasts,initial,cap,record,enabled,dispatch_starts,dispatch_nodes,responses,
                   lo,hi,end,leaf,offsets,ids,world,radii):
    p,d,counts=origins.copy(),directions.copy(),initial.copy()
    history=np.full((len(p),cap if record else 0),-1,dtype=np.int32)
    escaped=np.zeros(len(p),dtype=np.bool_); unresolved=escaped.copy(); used=escaped.copy()
    counters=np.zeros(2,dtype=np.int64); stats=np.zeros(11,dtype=np.int64)
    checks=0
    for ray in range(len(p)):
        last,count=lasts[ray],counts[ray]
        if record and count==1: history[ray,0]=last
        while True:
            skip=-1
            if enabled and last>=0:
                for k in range(dispatch_starts[last],dispatch_starts[last+1]):
                    node=dispatch_nodes[k]; checks+=1
                    hit,last,count,skip=replay(node,p[ray],d[ray],last,count,cap,history[ray],record,responses,
                        lo,hi,end,leaf,offsets,ids,world,radii,counters,stats)
                    if hit: used[ray]=True; break
            atom,distance=nearest_skip(p[ray],d[ray],last,np.inf,skip,lo,hi,end,leaf,offsets,ids,world,radii,counters)
            if atom<0: escaped[ray]=True; break
            if count>=cap: unresolved[ray]=True; break
            reflect_scalar(p[ray],d[ray],atom,distance,world,radii)
            if record: history[ray,count]=atom
            count+=1; last=atom
        counts[ray]=count
    return p,d,escaped,unresolved,counts,history,counters,stats,checks,used


class CPUResponseTransport(SelectedBVHTransport):
    def __init__(self,model,library=None):
        super().__init__(model,library)
        if np.any(self.responses[-1][:,0]>=0):
            raise ValueError('CPU response comparison currently accepts Cartesian certificates only')
        self.direct=CPUEHSSTransport(model)

    def trace(self,source,cap=64,enabled=True,record_history=False):
        if type(cap) is not int or cap<1 or np.any(source.initial_bounces>1):
            raise ValueError('Positive cap and zero/one initial collision required')
        t,s=self.model.tree,self.model.spheres
        if np.any(source.last_atom < -1) or np.any(source.last_atom>=len(s.radii)):
            raise ValueError('Invalid initial atom')
        if not (np.array_equal(s.centers,self.direct.certified_centers) and np.array_equal(s.radii,self.direct.certified_radii)):
            raise ValueError('Geometry changed: bind current response certificates')
        start=time.perf_counter()
        result=trace_response(source.origins,source.directions,source.last_atom,source.initial_bounces,cap,
            bool(record_history),bool(enabled),self.starts,self.nodes,self.responses,
            t.lo,t.hi,t.end,t.leaf_group,t.offsets,t.ids,s.centers,s.radii)
        p,d,e,u,c,h,counters,stats,checks,used=result
        labels=('cell_checks','attempts','accepted_prefixes','foreign_interruptions','foreign_searches',
                'replay_failures','replay_sphere_tests','cap_fallbacks','replayed_collisions','multiple_prefixes','partial_prefixes')
        return EHSSResult(p,d,e,u,c,h,source,dict(query_s=time.perf_counter()-start,
            box_tests=int(counters[0]),sphere_tests=int(counters[1]+stats[6]),responses=dict(zip(labels,map(int,stats))),
            response_rays=int(used.sum()),response_mass=float(source.weights[used].sum()),
            dispatch_checks=int(checks),enabled=bool(enabled),backend='scalar CPU certified response',threads=1))
