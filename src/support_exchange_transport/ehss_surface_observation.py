"""Separate atomic surface input offsets from center motion in next-event reuse.

Only the next collider identity (or escape) is cached. A retained hit is
intersected again in current geometry; reflection remains current and exact
for that chosen collider. The motion gate is not a nearest-hit certificate.
"""
from copy import copy
from dataclasses import dataclass
import time
import numpy as np
from numba import njit
from .ehss_factored_local_samples import point_inside
from .ehss_guarded_intrinsic_response import nearest_world
from .ehss_reference import entry_distance
from .ehss_path_dependencies import support_bounds,support_motion,flight_near_box
from .ehss_reuse_frames import partition
from .ehss_sparse_path_response import readonly

from .ehss_observation_refresh import (ObservationBatch,current_occlusion,current_successors,candidate_lengths,first_event_neighborhood,changed_support_gate)

@dataclass(frozen=True)
class SurfaceObservationBatch(ObservationBatch):
    surface_points: np.ndarray

class SurfaceObservationPool:
    def __init__(self,basis,geometry_tolerance=0.,point_tolerance=0.,direction_degrees=0.,max_age=4,padding=2.,dependency_groups=None,reuse=True,surface_chart=True):
        if any(not np.isfinite(x) or x<0 for x in (geometry_tolerance,point_tolerance,direction_degrees,padding)) or direction_degrees>180:
            raise ValueError('Finite nonnegative tolerances and angle at most 180 required')
        if type(max_age) is not int or max_age<1: raise ValueError('Positive integer age required')
        self.basis=basis; self.bank=basis.bank; self.geometry_tolerance=float(geometry_tolerance); self.point_tolerance=float(point_tolerance)
        self.direction_degrees=float(direction_degrees); self.chord=2*np.sin(np.deg2rad(direction_degrees)/2)
        self.max_age=max_age; self.padding=float(padding); self.reuse=bool(reuse)
        self.groups,self.atoms=partition(self.bank.groups if dependency_groups is None else dependency_groups,len(self.bank.reference),'Observation dependencies')
        self.surface_chart=bool(surface_chart)
        self.records={}; self.generation=0; self.receipt={}

    def snapshot(self):
        child=copy(self); child.records=self.records.copy(); child.receipt={}; return child

    def observe(self,bound):
        start=time.perf_counter()
        if bound.basis is not self.basis or bound.bank is not self.bank: raise ValueError('Observation pool belongs to another material basis')
        self.generation+=1; bank=self.bank; s=bound.model.spheres; t=bound.model.tree; world=s.centers; radii=s.radii
        occluded,occlusion_checks=current_occlusion(bound.points,bank.lasts,world,radii,t.lo,t.hi,t.left,t.right,t.leaf_group,t.offsets,t.ids)
        occlusion_s=time.perf_counter()-start; gate_start=time.perf_counter(); n=len(bank.mass)
        reused=np.zeros(n,dtype=bool); targets=np.full(n,-1,dtype=np.int64); lengths=np.full(n,np.inf)
        failures={k:0 for k in ('missing','disabled','age','radii','geometry','point','direction','candidate')}
        strict=self.geometry_tolerance==0 and self.point_tolerance==0 and self.chord==0
        grouped={}; frames={}; candidate_tests=old_rejects=new_rejects=gate_box_tests=0; frame_s=0.
        def frame(g):
            nonlocal frame_s
            if g not in frames:
                tick=time.perf_counter(); r,center=self.basis.frames.fit(g,world,bound.parent_rotations)
                local=(world-center)@r; lo,hi=support_bounds(local,radii,self.groups,len(self.atoms))
                frames[g]=(r,center,local,lo,hi); frame_s+=time.perf_counter()-tick
            return frames[g]
        for i in np.flatnonzero(~occluded):
            old=self.records.get(int(i))
            if old is None: failures['missing']+=1
            elif not self.reuse: failures['disabled']+=1
            else:
                batch,j=old
                if id(batch) not in grouped: grouped[id(batch)]=(batch,[],[])
                grouped[id(batch)][1].append(i); grouped[id(batch)][2].append(j)
        for batch,qi,ci in grouped.values():
            qi=np.asarray(qi,dtype=np.int64); ci=np.asarray(ci,dtype=np.int64)
            if self.generation-batch.generation>self.max_age: failures['age']+=len(qi); continue
            if not np.array_equal(radii,batch.radii): failures['radii']+=len(qi); continue
            def reject(mask,reason):
                nonlocal qi,ci
                failures[reason]+=int(mask.sum()); qi=qi[~mask]; ci=ci[~mask]
            r,center,local,lo,hi=frame(batch.frame)
            if strict:
                if not np.array_equal(world,batch.world): failures['geometry']+=len(qi); continue
                reject(np.any(bound.points[qi]!=batch.world_points[ci],axis=1),'point')
                reject(np.any(bound.directions[qi]!=batch.world_directions[ci],axis=1),'direction')
            else:
                if self.surface_chart:
                    displacement=(bound.points[qi]-world[bank.lasts[qi]])@r-batch.surface_points[ci]
                else: displacement=(bound.points[qi]-center)@r-batch.points[ci]
                reject(np.linalg.norm(displacement,axis=1)>self.point_tolerance,'point')
                reject(np.linalg.norm(bound.directions[qi]@r-batch.directions[ci],axis=1)>self.chord,'direction')
            if not len(qi): continue
            distance,valid,tests=candidate_lengths(bound.points[qi],bound.directions[qi],batch.targets[ci],world,radii); candidate_tests+=tests
            reject(~valid,'candidate'); distance=distance[valid]
            if not len(qi): continue
            if not strict:
                motion=support_motion(local,batch.local_world,self.groups,len(self.atoms))
                if np.any(motion>self.geometry_tolerance):
                    bad,old,new,tests=changed_support_gate(ci,batch.near,motion,(bound.points[qi]-center)@r,bound.directions[qi]@r,distance,
                        lo,hi,self.geometry_tolerance,self.padding)
                    old_rejects+=old; new_rejects+=new; gate_box_tests+=tests
                    reject(bad,'geometry'); distance=distance[~bad]
            reused[qi]=True; targets[qi]=batch.targets[ci]; lengths[qi]=distance
        gate_s=time.perf_counter()-gate_start; fresh=np.flatnonzero(~occluded&~reused); tick=time.perf_counter()
        target,distance,search_checks=current_successors(fresh,bound.points,bound.directions,bank.lasts,world,radii,t.lo,t.hi,t.left,t.right,t.end,t.leaf_group,t.offsets,t.ids)
        targets[fresh]=target; lengths[fresh]=distance; search_s=time.perf_counter()-tick; tick=time.perf_counter()
        kept={int(i):self.records[int(i)] for i in np.flatnonzero(reused)}; ro=lambda a:readonly(np.array(a,copy=True))
        saved_world=ro(world); saved_radii=ro(radii); owners=self.basis.frames.groups[bank.lasts]; build_tests=0
        for g in np.unique(owners[fresh]):
            rows=fresh[owners[fresh]==g]; r,center,local,lo,hi=frame(int(g))
            points=(bound.points[rows]-center)@r; directions=bound.directions[rows]@r
            near,tests=first_event_neighborhood(points,directions,lengths[rows],bank.lasts[rows],targets[rows],self.groups,lo,hi,self.padding); build_tests+=tests
            batch=SurfaceObservationBatch(int(g),ro(rows),ro(targets[rows]),ro(points),ro(directions),ro(bound.points[rows]),ro(bound.directions[rows]),
                ro(near),ro(local),saved_world,saved_radii,self.generation,ro((bound.points[rows]-world[bank.lasts[rows]])@r))
            for j,i in enumerate(rows): kept[int(i)]=(batch,j)
        self.records=kept; store_s=time.perf_counter()-tick
        counts=np.ones(n,dtype=np.int64); history=np.full((n,bank.cap),-1,dtype=np.int64); history[:,0]=bank.lasts
        hit=(targets>=0)&~occluded; counts[hit]=2; history[hit,1]=targets[hit]
        terminal=targets.copy(); terminal[occluded]=-3
        self.receipt=dict(rows=n,visible_rows=int((~occluded).sum()),occluded_rows=int(occluded.sum()),reused_rows=int(reused.sum()),refreshed_rows=len(fresh),
            failures=failures,reused_mask=readonly(reused),fresh_rows=readonly(fresh),occlusion_s=occlusion_s,gate_s=gate_s,search_s=search_s,store_s=store_s,frame_s=frame_s,
            old_rejects=old_rejects,new_rejects=new_rejects,gate_box_tests=gate_box_tests,build_box_tests=build_tests,candidate_tests=candidate_tests,
            nearest_queries=len(fresh),occlusion_sphere_tests=int(occlusion_checks[1]),search_sphere_tests=int(search_checks[1]),
            retained_batches=len({id(v[0]) for v in kept.values()}),generation=self.generation,max_age=self.max_age,heuristic=not strict,
            exact_current_occlusion=True,current_candidate_intersection=True,nearest_hit_certificate=False,geometry_tolerance=self.geometry_tolerance,
            point_tolerance=self.point_tolerance,direction_degrees=self.direction_degrees,padding=self.padding,surface_chart=self.surface_chart,
            center_motion_in_geometry_gate=True,point_metric='atom_surface_offset' if self.surface_chart else 'owner_frame_position')
        return dict(counts=counts,history=history,terminal=terminal,query_s=time.perf_counter()-start,
            sphere_tests=int(occlusion_checks[1]+search_checks[1]+candidate_tests),box_tests=int(occlusion_checks[0]+search_checks[0]),
            one_event_coupling_observation=True,full_trajectory_observation=False,observation_refresh=self.receipt.copy())
