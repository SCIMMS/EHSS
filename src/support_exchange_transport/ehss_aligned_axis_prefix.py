"""Prefix reuse with independent chemical frames and dependency partitions."""
from copy import copy
from .ehss_reuse_frames import ReuseFrames,partition
import time
import numpy as np
from .ehss_prefix_refresh_pool import PrefixRefreshPool,PrefixEpoch
from .ehss_atomic_material_axes import trace_to_material_port
from .ehss_material_collision_field import material_rotations
from .ehss_state import EHSSSource
from .ehss_sparse_path_response import readonly

from .ehss_packed_prefix_pool import PrefixBatch
from .ehss_path_dependencies import (FlightDependencies,record_flights,support_motion,support_bounds,reject_changed_nearby)

from .ehss_local_dependency_pool import DependencyPrefixBatch

class AlignedAxisPrefixPool(PrefixRefreshPool):
    """Old/current boxes gate recorded flights, not perturbed-path certificates."""
    def __init__(self,bank,geometry_tolerance=0.,point_tolerance=0.,direction_degrees=0.,max_age=4,reuse=True,padding=2.,frame_groups=None,dependency_groups=None):
        super().__init__(bank,geometry_tolerance,point_tolerance,direction_degrees,max_age,reuse)
        if not np.isfinite(padding) or padding<0: raise ValueError('Finite nonnegative dependency padding required')
        self.padding=float(padding)
        self.frames=ReuseFrames(bank.reference,bank.groups,bank.groups if frame_groups is None else frame_groups)
        self.dependency_groups,self.dependency_atoms=partition(bank.groups if dependency_groups is None else dependency_groups,len(bank.reference),'Dependency partition')
    def snapshot(self):
        p=copy(self); p.records=self.records.copy(); p.receipt={}; p.last_result=None
        return p

    def trace(self,rows,model,source,keys,active,groups,rotations,bins,cap,allow_reentry=True,parent_rotations=None):
        tick=time.perf_counter(); rows=np.asarray(rows,dtype=np.int64)
        if len(rows)!=len(source.weights) or len(np.unique(rows))!=len(rows) or np.any(rows<0) or np.any(rows>=len(self.bank.mass)):
            raise ValueError('Unique material bank rows required')
        if self.bins is not None and (self.bins!=bins or self.cap!=cap): raise ValueError('Pool bins or cap changed')
        if self.allow_reentry is not None and self.allow_reentry!=bool(allow_reentry): raise ValueError('Pool reentry policy changed')
        if not np.array_equal(groups,self.bank.groups): raise ValueError('Pool ownership changed')
        if np.any(source.initial_bounces!=1): raise ValueError('First-reflected prefix inputs required')
        self.bins=bins; self.cap=cap; self.allow_reentry=bool(allow_reentry); self.generation+=1
        if parent_rotations is None: parent_rotations=material_rotations(self.bank,model.spheres.centers)
        world=model.spheres.centers; radii=model.spheres.radii; owner=self.frames.groups[source.last_atom]
        n=len(rows); p=np.empty((n,3)); d=np.empty((n,3)); counts=np.zeros(n,dtype=np.int64)
        history=np.full((n,cap),-1,dtype=np.int32); ports=np.full(n,-1,dtype=np.int64)
        escaped=np.zeros(n,dtype=bool); unresolved=np.zeros(n,dtype=bool); reused=np.zeros(n,dtype=bool); exact=np.zeros(n,dtype=bool)
        failures={k:0 for k in ('missing','disabled','age','identity','radii','terminal_port','geometry','point','direction','active')}
        strict=self.geometry_tolerance==0 and self.point_tolerance==0 and self.direction_chord==0
        current={}; old_batches={}; dependency_old_rejects=dependency_new_rejects=dependency_box_tests=0
        dependency_build_s=0.; dependency_build_box_tests=dependency_invalid_rows=0
        frame_fit_s=0.
        def frame(g):
            nonlocal frame_fit_s
            if g not in current:
                fit_start=time.perf_counter(); r,center=self.frames.fit(g,world,parent_rotations)
                current[g]=(r,center,(world-center)@r); frame_fit_s+=time.perf_counter()-fit_start
            return current[g]
        for i,row in enumerate(rows):
            old=self.records.get(int(row))
            if old is None: failures['missing']+=1
            elif not self.reuse: failures['disabled']+=1
            else:
                batch,j=old; key=id(batch)
                if key not in old_batches: old_batches[key]=(batch,[],[])
                old_batches[key][1].append(i); old_batches[key][2].append(j)
        for batch,qi,ci in old_batches.values():
            qi=np.asarray(qi,dtype=np.int64); ci=np.asarray(ci,dtype=np.int64)
            if self.generation-batch.epoch.generation>self.max_age: failures['age']+=len(qi); continue
            def reject(mask,reason):
                nonlocal qi,ci
                failures[reason]+=int(np.count_nonzero(mask)); qi=qi[~mask]; ci=ci[~mask]
            reject((owner[qi]!=batch.owner)|(source.last_atom[qi]!=batch.atoms[ci]),'identity')
            if not len(qi): continue
            if not np.array_equal(radii,batch.epoch.radii): failures['radii']+=len(qi); continue
            port=batch.ports[ci]; bad=(port>=0)&((not allow_reentry)|(port>=len(active)))
            valid_port=(port>=0)&(port<len(active))
            bad[valid_port]|=~active[port[valid_port]]; reject(bad,'terminal_port')
            if not len(qi): continue
            r,center,local_world=frame(batch.owner)
            if strict:
                if not np.array_equal(world,batch.epoch.world): failures['geometry']+=len(qi); continue
                if not np.array_equal(active,batch.epoch.active): failures['active']+=len(qi); continue
                reject(np.any(source.origins[qi]!=batch.world_input_points[ci],axis=1),'point')
                reject(np.any(source.directions[qi]!=batch.world_input_directions[ci],axis=1),'direction')
                exact[qi]=True
            else:
                motion=support_motion(local_world,batch.epoch.local_world,self.dependency_groups,len(self.dependency_atoms))
                dep=batch.dependencies
                if np.any(motion>self.geometry_tolerance):
                    lo,hi=support_bounds(local_world,radii,self.dependency_groups,len(self.dependency_atoms))
                    bad,old_rejects,new_rejects,tests=reject_changed_nearby(ci,dep.offsets,dep.origins,dep.directions,dep.lengths,
                        dep.near_supports,dep.valid,motion,lo,hi,self.geometry_tolerance,self.padding)
                    dependency_old_rejects+=old_rejects; dependency_new_rejects+=new_rejects; dependency_box_tests+=tests
                else: bad=~dep.valid[ci]
                reject(bad,'geometry')
                reject(np.linalg.norm((source.origins[qi]-center)@r-batch.input_points[ci],axis=1)>self.point_tolerance,'point')
                reject(np.linalg.norm(source.directions[qi]@r-batch.input_directions[ci],axis=1)>self.direction_chord,'direction')
            if not len(qi): continue
            reused[qi]=True; counts[qi]=batch.counts[ci]; ports[qi]=batch.ports[ci]
            escaped[qi]=batch.escaped[ci]; unresolved[qi]=batch.unresolved[ci]
            for i,j in zip(qi,ci): history[i,:counts[i]]=batch.history[batch.offsets[j]:batch.offsets[j+1]]
            if strict: p[qi]=batch.world_points[ci]; d[qi]=batch.world_directions[ci]
            else:
                p[qi]=batch.points[ci]@r.T+center; rotated=batch.directions[ci]@r.T
                d[qi]=rotated/np.linalg.norm(rotated,axis=1)[:,None]
        gate_s=time.perf_counter()-tick; fresh=np.flatnonzero(~reused); start=time.perf_counter()
        metrics=dict(sphere_tests=0,box_tests=0,nearest_queries=0,port_lookups=0)
        if len(fresh):
            sub=EHSSSource(source.origins[fresh],source.directions[fresh],source.incoming[fresh],source.weights[fresh],source.last_atom[fresh],source.initial_bounces[fresh],source.convention)
            result=trace_to_material_port(model,sub,keys,active,groups,rotations,bins,cap,allow_reentry)
            for dest,name in ((p,'positions'),(d,'outgoing'),(counts,'bounces'),(history,'collider_ids'),(ports,'ports'),(escaped,'escaped'),(unresolved,'unresolved')): dest[fresh]=result[name]
            metrics=result['metrics']
        trace_s=time.perf_counter()-start; start=time.perf_counter()
        kept={int(rows[i]):self.records[int(rows[i])] for i in np.flatnonzero(reused)}
        ro=lambda a:readonly(np.array(a,copy=True))
        for g in np.unique(owner[fresh]):
            q=fresh[owner[fresh]==g]; r,center,local_world=frame(int(g))
            epoch=PrefixEpoch(ro(local_world),ro(world),ro(radii),ro(r),ro(center),ro(active),self.generation)
            offsets=np.r_[0,np.cumsum(counts[q])]
            mask=np.arange(cap)[None,:]<counts[q,None]
            dep_start=time.perf_counter()
            dependencies=record_flights(source.origins[q],source.directions[q],history[q],counts[q],ports[q],world,radii,r,center,
                self.dependency_groups,len(self.dependency_atoms),self.padding)
            dependency_build_s+=time.perf_counter()-dep_start
            dependency_build_box_tests+=dependencies.build_box_tests; dependency_invalid_rows+=int(np.count_nonzero(~dependencies.valid))
            batch=DependencyPrefixBatch(int(g),ro(source.last_atom[q]),ro((source.origins[q]-center)@r),ro(source.directions[q]@r),
                ro(source.origins[q]),ro(source.directions[q]),ro((p[q]-center)@r),ro(d[q]@r),ro(p[q]),ro(d[q]),
                ro(history[q][mask]),ro(offsets),ro(counts[q]),ro(ports[q]),ro(escaped[q]),ro(unresolved[q]),epoch,dependencies)
            for j,i in enumerate(q): kept[int(rows[i])]=(batch,j)
        self.records=kept; store_s=time.perf_counter()-start
        assert np.all(escaped.astype(int)+unresolved+(ports>=0)==1)
        self.receipt=dict(rows=n,reused_rows=int(reused.sum()),exact_reused_rows=int(exact.sum()),refreshed_rows=len(fresh),
            gate_s=gate_s,trace_s=trace_s,store_s=store_s,total_s=time.perf_counter()-tick,failures=failures,
            recorded_epochs=len({id(v[0].epoch) for v in kept.values()}),retained_batches=len({id(v[0]) for v in kept.values()}),
            generation=self.generation,max_age=self.max_age,geometry_tolerance=self.geometry_tolerance,point_tolerance=self.point_tolerance,direction_degrees=self.direction_degrees,
            heuristic=not strict,visibility_certificate=False,error_bound=False,active_mask_changed_reuse_allowed=not strict,
            retained_rows=len(kept),reused_mask=readonly(reused),fresh_rows=readonly(rows[fresh]),packed_records=True,
            local_dependencies=True,dependency_padding=self.padding,dependency_old_rejects=dependency_old_rejects,
            dependency_new_rejects=dependency_new_rejects,dependency_box_tests=dependency_box_tests,
            dependency_build_s=dependency_build_s,dependency_build_box_tests=dependency_build_box_tests,dependency_invalid_rows=dependency_invalid_rows,
            frame_fit_s=frame_fit_s,frame_count=len(self.frames.atoms),fitted_frames=len(current),dependency_support_count=len(self.dependency_atoms),
            response_owner_count=len(self.bank.atoms),independent_reuse_frames=True)
        self.last_result=dict(positions=p,outgoing=d,escaped=escaped,unresolved=unresolved,bounces=counts,collider_ids=history,ports=ports,
            metrics=dict(query_s=time.perf_counter()-tick,**{k:int(metrics[k]) for k in ('sphere_tests','box_tests','nearest_queries','port_lookups')}))
        return self.last_result
