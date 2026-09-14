"""Motion-gated reuse of material-row physical prefixes.

Positive tolerances are an empirical approximation, not a visibility or error
certificate. Gates compare to the original recorded epoch (never the previous
advected output), so small successive motions cannot hide accumulated drift.
"""
from dataclasses import dataclass
import time
import numpy as np
from .ehss_material_port_reentry import trace_to_material_port
from .ehss_state import EHSSSource
from .ehss_sparse_path_response import readonly

@dataclass(frozen=True)
class PrefixEpoch:
    local_world: np.ndarray
    world: np.ndarray
    radii: np.ndarray
    rotation: np.ndarray
    center: np.ndarray
    active: np.ndarray
    generation: int

@dataclass(frozen=True)
class PrefixRecord:
    owner: int
    atom: int
    input_point: np.ndarray
    input_direction: np.ndarray
    world_input_point: np.ndarray
    world_input_direction: np.ndarray
    point: np.ndarray
    direction: np.ndarray
    world_point: np.ndarray
    world_direction: np.ndarray
    history: np.ndarray
    count: int
    port: int
    escaped: bool
    unresolved: bool
    epoch: PrefixEpoch

class PrefixRefreshPool:
    def __init__(self,bank,geometry_tolerance=0.,point_tolerance=0.,direction_degrees=0.,max_age=4,reuse=True):
        if any(not np.isfinite(x) or x<0 for x in (geometry_tolerance,point_tolerance,direction_degrees)) or direction_degrees>180:
            raise ValueError('Finite nonnegative motion thresholds required')
        if type(max_age) is not int or max_age<1: raise ValueError('Positive maximum age required')
        self.bank=bank; self.geometry_tolerance=float(geometry_tolerance); self.point_tolerance=float(point_tolerance)
        self.direction_chord=2*np.sin(np.deg2rad(direction_degrees)/2); self.direction_degrees=float(direction_degrees)
        self.max_age=max_age; self.reuse=bool(reuse); self.records={}; self.generation=0; self.bins=None; self.cap=None; self.allow_reentry=None; self.receipt={}; self.last_result=None

    def snapshot(self):
        p=PrefixRefreshPool(self.bank,self.geometry_tolerance,self.point_tolerance,self.direction_degrees,self.max_age,self.reuse)
        p.records=self.records.copy(); p.generation=self.generation; p.bins=self.bins; p.cap=self.cap; p.allow_reentry=self.allow_reentry
        return p

    def trace(self,rows,model,source,keys,active,groups,rotations,bins,cap,allow_reentry=True):
        tick=time.perf_counter(); rows=np.asarray(rows,dtype=np.int64)
        if len(rows)!=len(source.weights) or len(np.unique(rows))!=len(rows) or np.any(rows<0) or np.any(rows>=len(self.bank.mass)):
            raise ValueError('Unique material bank rows required')
        if self.bins is not None and (self.bins!=bins or self.cap!=cap): raise ValueError('Pool bins or cap changed')
        if self.allow_reentry is not None and self.allow_reentry!=bool(allow_reentry): raise ValueError('Pool reentry policy changed')
        if not np.array_equal(groups,self.bank.groups): raise ValueError('Pool ownership changed')
        if np.any(source.initial_bounces!=1): raise ValueError('First-reflected prefix inputs required')
        self.bins=bins; self.cap=cap; self.allow_reentry=bool(allow_reentry); self.generation+=1
        world=model.spheres.centers; radii=model.spheres.radii; owner=groups[source.last_atom]
        n=len(rows); p=np.empty((n,3)); d=np.empty((n,3)); counts=np.zeros(n,dtype=np.int64)
        history=np.full((n,cap),-1,dtype=np.int32); ports=np.full(n,-1,dtype=np.int64)
        escaped=np.zeros(n,dtype=bool); unresolved=np.zeros(n,dtype=bool); reused=np.zeros(n,dtype=bool); exact=np.zeros(n,dtype=bool)
        current={}; geometry_distances={}; failures={k:0 for k in ('missing','disabled','age','identity','radii','terminal_port','geometry','point','direction','active')}
        strict=self.geometry_tolerance==0 and self.point_tolerance==0 and self.direction_chord==0
        # Current owner frames and full relative geometry are shared across rows.
        def frame(g):
            if g not in current:
                r=rotations[g]; center=world[self.bank.atoms[g]].mean(axis=0)
                current[g]=(r,center,(world-center)@r)
            return current[g]
        for i,row in enumerate(rows):
            old=self.records.get(int(row)); reason=None; g=int(owner[i]); r,center,local_world=frame(g)
            if old is None: reason='missing'
            elif not self.reuse: reason='disabled'
            elif self.generation-old.epoch.generation>self.max_age: reason='age'
            elif old.owner!=g or old.atom!=source.last_atom[i]: reason='identity'
            elif not np.array_equal(radii,old.epoch.radii): reason='radii'
            elif old.port>=0 and (not allow_reentry or old.port>=len(active) or not active[old.port]): reason='terminal_port'
            elif strict:
                if not np.array_equal(world,old.epoch.world): reason='geometry'
                elif not np.array_equal(active,old.epoch.active): reason='active'
                elif not np.array_equal(source.origins[i],old.world_input_point): reason='point'
                elif not np.array_equal(source.directions[i],old.world_input_direction): reason='direction'
                else: exact[i]=True
            else:
                key=(g,id(old.epoch))
                if key not in geometry_distances: geometry_distances[key]=float(np.linalg.norm(local_world-old.epoch.local_world,axis=1).max(initial=0))
                if geometry_distances[key]>self.geometry_tolerance: reason='geometry'
                elif np.linalg.norm((source.origins[i]-center)@r-old.input_point)>self.point_tolerance: reason='point'
                elif np.linalg.norm(source.directions[i]@r-old.input_direction)>self.direction_chord: reason='direction'
            if reason is not None: failures[reason]+=1; continue
            reused[i]=True; counts[i]=old.count; history[i,:old.count]=old.history; ports[i]=old.port
            escaped[i]=old.escaped; unresolved[i]=old.unresolved
            if exact[i]: p[i]=old.world_point; d[i]=old.world_direction
            else:
                p[i]=old.point@r.T+center; d[i]=old.direction@r.T; d[i]/=np.linalg.norm(d[i])
        gate_s=time.perf_counter()-tick; fresh=np.flatnonzero(~reused); start=time.perf_counter()
        metrics=dict(sphere_tests=0,box_tests=0,nearest_queries=0,port_lookups=0)
        if len(fresh):
            sub=EHSSSource(source.origins[fresh],source.directions[fresh],source.incoming[fresh],source.weights[fresh],source.last_atom[fresh],source.initial_bounces[fresh],source.convention)
            result=trace_to_material_port(model,sub,keys,active,groups,rotations,bins,cap,allow_reentry)
            for dest,name in ((p,'positions'),(d,'outgoing'),(counts,'bounces'),(history,'collider_ids'),(ports,'ports'),(escaped,'escaped'),(unresolved,'unresolved')): dest[fresh]=result[name]
            metrics=result['metrics']
        trace_s=time.perf_counter()-start; start=time.perf_counter(); epochs={}
        # Drop non-requested rows: next invocation compares consecutive current
        # missing-row sets, avoiding unbounded historic suffix accumulation.
        kept={int(rows[i]):self.records[int(rows[i])] for i in np.flatnonzero(reused)}
        for i in fresh:
            g=int(owner[i]); r,center,local_world=frame(g)
            if g not in epochs:
                epochs[g]=PrefixEpoch(readonly(local_world.copy()),readonly(world.copy()),readonly(radii.copy()),readonly(r.copy()),readonly(center.copy()),readonly(np.array(active,copy=True)),self.generation)
            ro=lambda a:readonly(np.array(a,copy=True))
            kept[int(rows[i])]=PrefixRecord(g,int(source.last_atom[i]),ro((source.origins[i]-center)@r),ro(source.directions[i]@r),
                ro(source.origins[i]),ro(source.directions[i]),ro((p[i]-center)@r),ro(d[i]@r),ro(p[i]),ro(d[i]),ro(history[i,:counts[i]]),
                int(counts[i]),int(ports[i]),bool(escaped[i]),bool(unresolved[i]),epochs[g])
        self.records=kept; store_s=time.perf_counter()-start
        assert np.all(escaped.astype(int)+unresolved+(ports>=0)==1)
        self.receipt=dict(rows=n,reused_rows=int(reused.sum()),exact_reused_rows=int(exact.sum()),refreshed_rows=len(fresh),
            gate_s=gate_s,trace_s=trace_s,store_s=store_s,total_s=time.perf_counter()-tick,failures=failures,
            recorded_epochs=len({id(v.epoch) for v in kept.values()}),generation=self.generation,max_age=self.max_age,
            geometry_tolerance=self.geometry_tolerance,point_tolerance=self.point_tolerance,direction_degrees=self.direction_degrees,
            heuristic=not strict,visibility_certificate=False,error_bound=False,active_mask_changed_reuse_allowed=not strict,
            retained_rows=len(kept),reused_mask=readonly(reused),fresh_rows=readonly(rows[fresh]))
        self.last_result=dict(positions=p,outgoing=d,escaped=escaped,unresolved=unresolved,bounces=counts,collider_ids=history,ports=ports,
            metrics=dict(query_s=time.perf_counter()-tick,**{k:int(metrics[k]) for k in ('sphere_tests','box_tests','nearest_queries','port_lookups')}))
        return self.last_result
