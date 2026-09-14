"""Recursive node-aware EHSS with optional certified itinerary responses.

An exclusive node may resolve internal repeated collisions up to its boundary.
Where foreign geometry intersects its bounds, it returns after one globally
ordered collision so the parent can arbitrate again. Bounds are never scatterers.
"""
import time
import numpy as np
from numba import njit
from .inside_out_pa import Spheres
from .ehss_state import EHSSResult
from .ehss_reference import entry_distance
from .ehss_canonical import convex_interval,chart_boundary_roundtrip,update_tree_ports
from .ehss_response import empty_responses,replay_response
from .ehss_convex import update_convex_ports
from .ehss_pair_gate import empty_coupling,source_for_gate,direction_bin,pair_allows


@njit(cache=True)
def box_interval(o,d,lo,hi):
    near=0.; far=np.inf
    for k in range(3):
        if d[k]==0.:
            if o[k]<lo[k] or o[k]>hi[k]:
                return np.inf,-np.inf
        else:
            a=(lo[k]-o[k])/d[k]; b=(hi[k]-o[k])/d[k]
            near=max(near,min(a,b)); far=min(far,max(a,b))
    if far<near:
        return np.inf,-np.inf
    return near,far


@njit(cache=False)
def nearest_node(node,o,d,last,limit,lo,hi,left,right,leaf_group,offsets,ids,
                 local,radii,rotations,translations,counters,gate_mode,gate,gate_counts,gate_source,gate_bin):
    if gate_mode>0 and not pair_allows(node,gate_source,d,gate_bin,gate_mode,gate,gate_counts):
        return -1,np.inf
    counters[0]+=1
    near,far=box_interval(o,d,lo[node],hi[node])
    if near>limit or far<0.:
        return -1,np.inf
    group=leaf_group[node]; best=limit; identity=-1
    if group>=0:
        counters[2]+=1
        local_o=(o-translations[group])@rotations[group]
        local_d=d@rotations[group]
        for k in range(offsets[group],offsets[group+1]):
            atom=ids[k]
            if atom==last:
                continue
            counters[1]+=1
            distance=entry_distance(local_o,local_d,local[atom],radii[atom])
            if distance<best or (distance==best and np.isfinite(distance) and (identity<0 or atom<identity)):
                identity=atom; best=distance
    else:
        a,ta=nearest_node(left[node],o,d,last,best,lo,hi,left,right,leaf_group,offsets,ids,local,radii,rotations,translations,counters,gate_mode,gate,gate_counts,gate_source,gate_bin)
        if a>=0:
            identity=a; best=ta
        b,tb=nearest_node(right[node],o,d,last,best,lo,hi,left,right,leaf_group,offsets,ids,local,radii,rotations,translations,counters,gate_mode,gate,gate_counts,gate_source,gate_bin)
        if b>=0 and (tb<best or (tb==best and (identity<0 or b<identity))):
            identity=b; best=tb
    return identity,best


@njit(cache=True)
def finish_node(node,o,d,bounces,initial,lo,hi,counters,node_bounces,multi_exits,
                use_polyhedral,port_centers,port_normals,port_heights,port_offsets,chart_metrics):
    if use_polyhedral:
        first=port_offsets[node]; end=port_offsets[node+1]
        normals=port_normals[first:end]; heights=port_heights[first:end]
        _,far=convex_interval(o,d,port_centers[node],normals,heights)
        chart_metrics[3]+=1
    else:
        _,far=box_interval(o,d,lo[node],hi[node])
    if np.isfinite(far) and far>0.:
        # Child boundaries retain the last physical-collision anchor.
        endpoint=o+far*d
        if use_polyhedral:
            reconstructed,face,error=chart_boundary_roundtrip(endpoint,port_centers[node],normals,heights)
            if face<0:
                raise ValueError("No canonical port face for boundary event")
            chart_metrics[0]+=1; chart_metrics[1]=max(chart_metrics[1],error)
            chart_metrics[2]=max(chart_metrics[2],error/max(1.,np.max(heights)))
            if node==0:
                endpoint=reconstructed
        if node==0:
            o[:]=endpoint
        counters[3]+=1
    node_bounces[node]+=bounces-initial
    if bounces-initial>=2:
        multi_exits[node]+=1


@njit(cache=False)
def apply_node(node,o,d,last,bounces,cap,stop_after_one,lo,hi,left,right,end,leaf_group,
               leaf_node,groups,exclusive,offsets,ids,local,world,radii,rotations,translations,
               history,counters,node_calls,node_bounces,multi_exits,
               use_polyhedral,port_centers,port_normals,port_heights,port_offsets,chart_metrics,
               responses,cache_counts,cache_hits,record_meta,record_states,record_count,gate_mode,gate,gate_counts):
    node_calls[node]+=1; initial=bounces
    if use_polyhedral:
        a=port_offsets[node]; b=port_offsets[node+1]
        normals=port_normals[a:b]; heights=port_heights[a:b]
        near,far=convex_interval(o,d,port_centers[node],normals,heights)
        chart_metrics[3]+=1
        if not np.isfinite(near) or far<near:
            # A convex envelope contains every owned physical sphere. A straight
            # flight missing it cannot hit any owned atom. Preserve the anchor.
            chart_metrics[5]+=1
            return last,bounces,False
        if near>0.:
            endpoint=o+near*d
            _,face,error=chart_boundary_roundtrip(endpoint,port_centers[node],normals,heights)
            if face<0:
                raise ValueError("Missing convex entry chart face")
            chart_metrics[0]+=1; chart_metrics[1]=max(chart_metrics[1],error)
            chart_metrics[2]=max(chart_metrics[2],error/max(1.,np.max(heights)))
            chart_metrics[4]+=1
    if not stop_after_one and exclusive[node]:
        if len(record_states)>0:
            k=record_count[0]; record_count[0]+=1
            if k<len(record_states):
                record_meta[k,0]=node; record_meta[k,1]=last
                record_states[k,:3]=o; record_states[k,3:]=d
        hit,last,bounces=replay_response(node,o,d,last,bounces,cap,world,radii,history,responses,counters,cache_counts)
        if hit:
            cache_hits[node]+=1
            finish_node(node,o,d,bounces,initial,lo,hi,counters,node_bounces,multi_exits,
                use_polyhedral,port_centers,port_normals,port_heights,port_offsets,chart_metrics)
            return last,bounces,False
    while True:
        gate_source=-1; gate_bin=0
        if gate_mode>0:
            gate_source=source_for_gate(o,last,groups,leaf_node,gate,gate_counts)
            if gate_mode==2 and gate_source>=0:
                gate_bin=direction_bin(d,gate[8])
        atom,distance=nearest_node(node,o,d,last,np.inf,lo,hi,left,right,leaf_group,
            offsets,ids,local,radii,rotations,translations,counters,gate_mode,gate,gate_counts,gate_source,gate_bin)
        if atom<0:
            finish_node(node,o,d,bounces,initial,lo,hi,counters,node_bounces,multi_exits,
                use_polyhedral,port_centers,port_normals,port_heights,port_offsets,chart_metrics)
            return last,bounces,False
        if bounces>=cap:
            node_bounces[node]+=bounces-initial
            return last,bounces,True
        if leaf_group[node]>=0:
            # Refine the selected physical event in world coordinates. The
            # body-frame search selects candidates; virtual interfaces must not
            # perturb the physical flight or accumulate reflection-angle error.
            distance=entry_distance(o,d,world[atom],radii[atom])
            counters[5]+=1
            if not np.isfinite(distance):
                raise ValueError("Body/world collision disagreement requires refinement")
            o[:]=o+distance*d
            normal=o-world[atom]; normal/=np.sqrt(np.dot(normal,normal))
            o[:]=world[atom]+radii[atom]*normal
            d[:]=d-2*np.dot(d,normal)*normal
            d[:]/=np.sqrt(np.dot(d,d))
            history[bounces]=atom; bounces+=1; last=atom
        else:
            target=leaf_node[groups[atom]]
            child=left[node] if left[node]<=target<end[left[node]] else right[node]
            one=stop_after_one or not exclusive[child]
            last,bounces,unresolved=apply_node(child,o,d,last,bounces,cap,one,
                lo,hi,left,right,end,leaf_group,leaf_node,groups,exclusive,offsets,ids,local,world,radii,
                rotations,translations,history,counters,node_calls,node_bounces,multi_exits,
                use_polyhedral,port_centers,port_normals,port_heights,port_offsets,chart_metrics,
                responses,cache_counts,cache_hits,record_meta,record_states,record_count,gate_mode,gate,gate_counts)
            if unresolved:
                node_bounces[node]+=bounces-initial
                return last,bounces,True
        if stop_after_one:
            counters[4]+=1
            node_bounces[node]+=bounces-initial
            return last,bounces,False


@njit(cache=False)
def trace_tree(origins,directions,last_atoms,initial_bounces,cap,lo,hi,left,right,end,leaf_group,
               leaf_node,groups,exclusive,offsets,ids,local,world,radii,rotations,translations,
               use_polyhedral,port_centers,port_normals,port_heights,port_offsets,responses,record_capacity,gate_mode,gate):
    n=len(origins); positions=origins.copy(); outgoing=directions.copy()
    unresolved=np.zeros(n,dtype=np.bool_); bounces=initial_bounces.copy()
    ledger=np.full((n,cap),-1,dtype=np.int32); counters=np.zeros(6,dtype=np.int64)
    calls=np.zeros(len(lo),dtype=np.int64); node_bounces=np.zeros(len(lo),dtype=np.int64)
    multi_exits=np.zeros(len(lo),dtype=np.int64)
    chart_metrics=np.zeros(6)
    cache_counts=np.zeros(11,dtype=np.int64); cache_hits=np.zeros(len(lo),dtype=np.int64)
    cache_events=np.zeros((n,3),dtype=np.int64)
    record_meta=np.empty((record_capacity,2),dtype=np.int64)
    record_states=np.empty((record_capacity,6)); record_count=np.zeros(1,dtype=np.int64)
    gate_counts=np.zeros(10,dtype=np.int64)
    for ray in range(n):
        cache_events[ray,:]=-cache_counts[8:11]
        if initial_bounces[ray]==1:
            ledger[ray,0]=last_atoms[ray]
        _,bounces[ray],unresolved[ray]=apply_node(0,positions[ray],outgoing[ray],last_atoms[ray],bounces[ray],cap,False,
            lo,hi,left,right,end,leaf_group,leaf_node,groups,exclusive,offsets,ids,local,world,radii,
            rotations,translations,ledger[ray],counters,calls,node_bounces,multi_exits,
            use_polyhedral,port_centers,port_normals,port_heights,port_offsets,chart_metrics,
            responses,cache_counts,cache_hits,record_meta,record_states,record_count,gate_mode,gate,gate_counts)
        cache_events[ray,:]+=cache_counts[8:11]
    kept=min(record_capacity,record_count[0])
    return (positions,outgoing,~unresolved,unresolved,bounces,ledger,counters,calls,node_bounces,multi_exits,chart_metrics,
        cache_counts,cache_hits,record_meta[:kept],record_states[:kept],max(0,record_count[0]-kept),cache_events,gate_counts)


class NodeAwareSupportTree:
    def __init__(self,local_centers,radii,groups,rotations,translations,port_kind="box",coarse_supports=None,convex_direction_level=1):
        if port_kind not in ["box","tetra","convex"]:
            raise ValueError("Choose box, tetra or arbitrary-node convex ports")
        self.port_kind=port_kind
        self.convex_direction_level=convex_direction_level
        self.local=np.ascontiguousarray(local_centers,dtype=float).copy()
        self.radii=np.ascontiguousarray(radii,dtype=float).copy()
        self.groups=np.ascontiguousarray(groups,dtype=np.int64).copy()
        unique=np.unique(self.groups)
        if len(unique)==0 or not np.array_equal(unique,np.arange(len(unique))):
            raise ValueError("Contiguous nonempty physical support IDs required")
        if self.local.shape!=(len(self.radii),3) or self.groups.shape!=(len(self.radii),):
            raise ValueError("One local sphere and physical support owner per atom required")
        if not np.isfinite(self.local).all() or not np.isfinite(self.radii).all() or np.any(self.radii<=0):
            raise ValueError("Finite local geometry and positive radii required")
        self.n_groups=len(unique)
        # Refinement inserts children inside an original rigid support. Keep
        # the original grouping as a subtree rather than globally regrouping it.
        self.coarse_supports=np.arange(self.n_groups,dtype=np.int64) if coarse_supports is None else np.asarray(coarse_supports,dtype=np.int64).copy()
        coarse_unique=np.unique(self.coarse_supports)
        if self.coarse_supports.shape!=(self.n_groups,) or not np.array_equal(coarse_unique,np.arange(len(coarse_unique))) or np.any(np.diff(self.coarse_supports)<0):
            raise ValueError("Contiguous, ordered coarse ownership required per refined support")
        self.coarse_node=np.full(len(coarse_unique),-1,dtype=np.int64)
        self.coarse_supports.setflags(write=False)
        self.offsets=np.r_[0,np.cumsum(np.bincount(self.groups))].astype(np.int64)
        self.ids=np.argsort(self.groups,kind="stable").astype(np.int64)
        self.left=[]; self.right=[]; self.end=[]; self.parent=[]; self.leaf_group=[]
        self.leaf_node=np.empty(self.n_groups,dtype=np.int64); self.depth=[]
        def build(subset,parent,depth):
            node=len(self.parent)
            coarse=self.coarse_supports[subset]
            if coarse[0]==coarse[-1] and self.coarse_node[coarse[0]]<0:
                self.coarse_node[coarse[0]]=node
            self.parent.append(parent); self.left.append(-1); self.right.append(-1)
            self.end.append(-1); self.leaf_group.append(-1); self.depth.append(depth)
            if len(subset)==1:
                group=int(subset[0]); self.leaf_group[node]=group; self.leaf_node[group]=node
            else:
                coarse_ids=np.unique(coarse)
                split=len(subset)//2 if len(coarse_ids)==1 else int(np.searchsorted(coarse,coarse_ids[len(coarse_ids)//2]))
                self.left[node]=build(subset[:split],node,depth+1)
                self.right[node]=build(subset[split:],node,depth+1)
            self.end[node]=len(self.parent)
            return node
        build(unique,-1,0)
        for name in ["left","right","end","parent","leaf_group","depth"]:
            setattr(self,name,np.asarray(getattr(self,name),dtype=np.int64))
        self.lo=np.empty((len(self.parent),3)); self.hi=np.empty_like(self.lo)
        self.rotations=np.repeat(np.eye(3)[None,:,:],self.n_groups,axis=0)
        self.translations=np.zeros((self.n_groups,3)); self.pose_versions=np.zeros(self.n_groups,dtype=np.int64)
        self.local.setflags(write=False); self.radii.setflags(write=False); self.groups.setflags(write=False)
        self.set_poses(rotations,translations,initial=True)

    def set_poses(self,rotations,translations,initial=False):
        rotations=np.ascontiguousarray(rotations,dtype=float); translations=np.ascontiguousarray(translations,dtype=float)
        if rotations.shape!=(self.n_groups,3,3) or translations.shape!=(self.n_groups,3):
            raise ValueError("One rigid pose per physical support required")
        if not np.isfinite(rotations).all() or not np.isfinite(translations).all() or not np.allclose(
            np.swapaxes(rotations,1,2)@rotations,np.eye(3),atol=1e-12,rtol=1e-12) or not np.allclose(np.linalg.det(rotations),1.,atol=1e-12):
            raise ValueError("Finite proper rigid transforms required")
        changed=np.any(rotations!=self.rotations,axis=(1,2))|np.any(translations!=self.translations,axis=1)
        self.rotations=rotations.copy(); self.translations=translations.copy()
        self.pose_versions+=changed.astype(np.int64)
        world=np.empty_like(self.local)
        pad=1e-10*max(1.,float(self.radii.max()))
        for group in range(self.n_groups):
            atoms=self.ids[self.offsets[group]:self.offsets[group+1]]
            world[atoms]=self.local[atoms]@rotations[group].T+translations[group]
            node=self.leaf_node[group]
            self.lo[node]=(world[atoms]-self.radii[atoms,None]).min(axis=0)-pad
            self.hi[node]=(world[atoms]+self.radii[atoms,None]).max(axis=0)+pad
        for node in range(len(self.parent)-1,-1,-1):
            if self.leaf_group[node]<0:
                self.lo[node]=np.minimum(self.lo[self.left[node]],self.lo[self.right[node]])
                self.hi[node]=np.maximum(self.hi[self.left[node]],self.hi[self.right[node]])
        self.world=world
        self.spheres=Spheres(world,self.radii,deduplicate=False)
        atom_leaf=self.leaf_node[self.groups]
        self.exclusive=np.ones(len(self.parent),dtype=np.bool_)
        for node in range(len(self.parent)):
            foreign=(atom_leaf<node)|(atom_leaf>=self.end[node])
            overlaps=np.all((world+self.radii[:,None]>=self.lo[node])&(world-self.radii[:,None]<=self.hi[node]),axis=1)
            self.exclusive[node]=not np.any(foreign&overlaps)
        if self.port_kind=="tetra":
            was_built=hasattr(self,"leaf_charts")
            port_receipt=update_tree_ports(self)
            port_receipt["leaf_charts_rebuilt"]=0 if was_built else self.n_groups
            self.port_offsets=np.arange(len(self.parent)+1,dtype=np.int64)*4
        elif self.port_kind=="convex":
            port_receipt=update_convex_ports(self)
        else:
            self.port_centers=np.zeros((len(self.parent),3))
            self.port_normals=np.zeros((len(self.parent),4,3))
            self.port_heights=np.zeros((len(self.parent),4))
            self.port_offsets=np.arange(len(self.parent)+1,dtype=np.int64)*4
            port_receipt=dict(port_kind="box")
        affected=set()
        for group in np.flatnonzero(changed):
            node=int(self.leaf_node[group])
            while node>=0:
                affected.add(node); node=int(self.parent[node])
        # Conservative full bounds/certificate refresh for this exact first stage.
        # No sparse-update cost or compiled-response reuse is claimed here.
        return dict(changed_supports=np.flatnonzero(changed).tolist(),affected_ancestors=sorted(affected),
                    local_geometry_rebuilt=0,bounds_refreshed=len(self.parent),certificates_refreshed=len(self.parent),**port_receipt)

    def trace(self,source,max_bounces=64,response_library=None,cache_mode="all",record_capacity=0,coupling_cache=None):
        if not isinstance(max_bounces,(int,np.integer)) or max_bounces<1 or np.any(source.initial_bounces>1):
            raise ValueError("Positive cap and zero/one initial bounce required")
        if np.any(source.last_atom>=len(self.radii)) or np.any(source.last_atom< -1):
            raise ValueError("Invalid initial owner")
        if not isinstance(record_capacity,(int,np.integer)) or record_capacity<0:
            raise ValueError("Nonnegative integer record capacity required")
        preparation=time.perf_counter()
        if response_library is None:
            responses=empty_responses(len(self.parent)); receipt=dict(active_cells=0,inactive_cells=0,packed_bytes=0)
        else:
            responses,receipt=response_library.pack(self,cache_mode)
        prepare_s=time.perf_counter()-preparation
        gate_start=time.perf_counter()
        if coupling_cache is None:
            gate_mode=0; gate=empty_coupling(); gate_receipt=dict(mode="off",array_bytes=0)
        else:
            gate_mode,gate,gate_receipt=coupling_cache.prepare(self)
        gate_prepare_s=time.perf_counter()-gate_start
        start=time.perf_counter()
        result=trace_tree(source.origins,source.directions,source.last_atom,source.initial_bounces,max_bounces,
            self.lo,self.hi,self.left,self.right,self.end,self.leaf_group,self.leaf_node,self.groups,self.exclusive,
            self.offsets,self.ids,self.local,self.world,self.radii,self.rotations,self.translations,
            self.port_kind!="box",self.port_centers,self.port_normals.reshape(-1,3),self.port_heights.reshape(-1),self.port_offsets,responses,record_capacity,gate_mode,gate)
        p,d,e,u,b,h,counters,calls,node_bounces,multi_exits,chart_metrics,cache_counts,cache_hits,record_meta,record_states,record_dropped,cache_events,gate_counts=result
        output=EHSSResult(p,d,e,u,b,h,source,dict(query_s=time.perf_counter()-start,backend="recursive node-aware exact response",
            boxes=int(counters[0]),sphere_tests=int(counters[1]+counters[5]),local_sphere_tests=int(counters[1]),
            world_event_refinements=int(counters[5]),local_frame_queries=int(counters[2]),
            boundary_exits=int(counters[3]),one_collision_returns=int(counters[4]),
            node_calls=calls.tolist(),node_bounces=node_bounces.tolist(),exclusive_nodes=int(self.exclusive.sum()),
            multi_collision_exits=multi_exits.tolist(),
            tree_depth=int(self.depth.max()),nodes=len(self.parent),compiled_response_cache=response_library is not None,
            port_kind=self.port_kind,canonical_boundary_events=int(chart_metrics[0]),
            max_chart_roundtrip_error=float(chart_metrics[1]),max_relative_chart_error=float(chart_metrics[2]),
            polyhedral_interval_queries=int(chart_metrics[3]),canonical_entry_events=int(chart_metrics[4]),
            polyhedral_free_flight_misses=int(chart_metrics[5]),
            cache_prepare_s=prepare_s,cache_mode=cache_mode,cache_receipt=receipt,
            cache_lookups=int(cache_counts[0]),cache_cells_tested=int(cache_counts[1]),cache_hits=int(cache_counts[2]),
            cache_replayed_bounces=int(cache_counts[3]),cache_cap_fallbacks=int(cache_counts[4]),
            cache_numeric_fallbacks=int(cache_counts[5]),cache_primitive_tests=int(cache_counts[6]),
            cache_line_anchor_fallbacks=int(cache_counts[7]),
            cache_zero_bounce_hits=int(cache_counts[8]),cache_one_bounce_hits=int(cache_counts[9]),cache_multi_bounce_hits=int(cache_counts[10]),
            cache_unique_rays=int(np.count_nonzero(np.any(cache_events>0,axis=1))),
            cache_unique_collision_rays=int(np.count_nonzero(np.any(cache_events[:,1:]>0,axis=1))),
            cache_source_mass=float(source.weights[np.any(cache_events>0,axis=1)].sum()),
            cache_collision_source_mass=float(source.weights[np.any(cache_events[:,1:]>0,axis=1)].sum()),
            cache_hits_by_node=cache_hits.tolist(),training_records_dropped=int(record_dropped)))
        output.metrics.update(coupling_prepare_s=gate_prepare_s,coupling_receipt=gate_receipt,
            coupling_queries=int(gate_counts[0]),coupling_reused_queries=int(gate_counts[1]),
            coupling_pairs_compiled=int(gate_counts[2]),coupling_rejections=int(gate_counts[3]),
            coupling_overlap_queries=int(gate_counts[4]),coupling_capacity_fallbacks=int(gate_counts[5]),
            coupling_hash_probes=int(gate_counts[6]),coupling_source_checks=int(gate_counts[7]),
            coupling_outside_source=int(gate_counts[8]),coupling_external_source=int(gate_counts[9]),
            coupling_pairs_stored=int(np.count_nonzero(gate[3]>=0)))
        output.training_meta=record_meta; output.training_states=record_states
        output.cache_events=cache_events
        return output
