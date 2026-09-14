"""Exact endpoint sparse updates of body-attached discrete PA coupling.

Both blocked and previously unblocked target states are retained. Discovery
uses OLD UNION NEW bounds, so blocker motion can safely reveal old paths.
This is an endpoint updater, not swept collision detection during a frame.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
import numpy as np
from numba import njit

from .inside_out_pa import Source
from .pa_spatial import build_tree, box_hit, _one, pose_source


@dataclass
class RayTree:
    center: np.ndarray
    radius: np.ndarray
    axis: np.ndarray
    cosine: np.ndarray
    end: np.ndarray
    first: np.ndarray
    count: np.ndarray
    ids: np.ndarray

    @property
    def arrays(self):
        return (self.center,self.radius,self.axis,self.cosine,self.end,self.first,self.count,self.ids)

    @property
    def nbytes(self):
        return sum(a.nbytes for a in self.arrays)


def build_ray_tree(origins,directions,indices,leaf_size=32):
    """Direction BVH with conservative origin balls and direction cones."""
    tree=build_tree(directions[indices],directions[indices],leaf_size=leaf_size)
    ids=indices[tree.ids]
    n=len(tree.end)
    centers=np.empty((n,3)); radii=np.empty(n)
    axes=np.empty((n,3)); cosine=np.empty(n)
    for node in range(n):
        # Leaves are contiguous in preorder; derive descendants' primitive span.
        stop=tree.end[node]
        last=stop-1
        members=ids[tree.first[node]:tree.first[last]+tree.count[last]]
        points=origins[members]
        center=(points.min(axis=0)+points.max(axis=0))/2
        centers[node]=center
        radii[node]=np.linalg.norm(points-center,axis=1).max()*(1+1e-13)+1e-13
        axis=directions[members].sum(axis=0)
        norm=np.linalg.norm(axis)
        if norm<1e-12:
            axes[node]=[1,0,0]; cosine[node]=-1.
        else:
            axes[node]=axis/norm
            cosine[node]=max(-1.,np.min(directions[members]@axes[node])-1e-13)
    return RayTree(centers,radii,axes,cosine,tree.end,tree.first,tree.count,ids)


@njit(cache=True)
def cone_possible(origin_center,origin_radius,axis,cosine,target_center,target_radius):
    dx=target_center[0]-origin_center[0]
    dy=target_center[1]-origin_center[1]
    dz=target_center[2]-origin_center[2]
    distance2=dx*dx+dy*dy+dz*dz
    radius=target_radius+origin_radius
    if distance2<=radius*radius or cosine<=0:
        return True
    # cos(alpha+beta)*distance, beta=asin(inflated_radius/distance).
    threshold=cosine*np.sqrt(max(0.,distance2-radius*radius))-np.sqrt(max(0.,1-cosine*cosine))*radius
    dot=axis[0]*dx+axis[1]*dy+axis[2]*dz
    return dot>=threshold-1e-12*(1+np.sqrt(distance2))


@njit(cache=True)
def discover_tree(origins,directions,old_lo,old_hi,new_lo,new_hi,
                  centers,radii,axes,cosines,end,first,count,ids):
    candidates=np.empty(len(ids),dtype=np.int64)
    potential=np.empty(len(ids),dtype=np.bool_)
    old_center=(old_lo+old_hi)/2; new_center=(new_lo+new_hi)/2
    old_radius=np.sqrt(np.sum(((old_hi-old_lo)/2)**2))
    new_radius=np.sqrt(np.sum(((new_hi-new_lo)/2)**2))
    node=0; size=0; nodes=0; rays=0
    while node<len(end):
        nodes+=1
        if not (cone_possible(centers[node],radii[node],axes[node],cosines[node],old_center,old_radius)
                or cone_possible(centers[node],radii[node],axes[node],cosines[node],new_center,new_radius)):
            node=end[node]
            continue
        for k in range(first[node],first[node]+count[node]):
            i=ids[k]; rays+=1
            new=box_hit(origins[i],directions[i],new_lo,new_hi)
            if new or box_hit(origins[i],directions[i],old_lo,old_hi):
                candidates[size]=i; potential[size]=new; size+=1
        node+=1
    return candidates[:size],potential[:size],nodes,rays


@njit(cache=True)
def discover_flat(origins,directions,ids,old_lo,old_hi,new_lo,new_hi):
    candidates=np.empty(len(ids),dtype=np.int64); potential=np.empty(len(ids),dtype=np.bool_)
    size=0
    for i in ids:
        new=box_hit(origins[i],directions[i],new_lo,new_hi)
        if new or box_hit(origins[i],directions[i],old_lo,old_hi):
            candidates[size]=i; potential[size]=new; size+=1
    return candidates[:size],potential[:size],0,len(ids)


@njit(cache=True)
def update_pair(origins,directions,owners,candidates,potential,relative_rotation,relative_translation,
                centers,radii,lo,hi,end,first,count,ids,root,old_hits,hit_counts):
    o=np.empty(3); d=np.empty(3)
    boxes=0; tests=0; changed=0; cleared=0; set_hits=0
    for k in range(len(candidates)):
        i=candidates[k]; hit=False
        if potential[k]:
            for a in range(3):
                o[a]=0.; d[a]=0.
                for b in range(3):
                    o[a]+=relative_rotation[b,a]*(origins[i,b]-relative_translation[b])
                    d[a]+=relative_rotation[b,a]*directions[i,b]
            hit,nb,nt=_one(o,d,owners[i],centers,radii,lo,hi,end,first,count,ids,root,ids,-1)
            boxes+=nb; tests+=nt
        old=old_hits[i]
        if hit!=old:
            changed+=1
            if hit:
                hit_counts[i]+=1; set_hits+=1
            else:
                hit_counts[i]-=1; cleared+=1
            old_hits[i]=hit
    return boxes,tests,changed,cleared,set_hits


class SparseCoupling:
    """Per-target Boolean coupling, with immutable local source/geometry.

    The full target-by-active-source Boolean matrix is intentionally explicit.
    Its memory and initialization are charged; this is not an unbounded-size
    sparse matrix claim. Discovery mode isolates the ray hierarchy's benefit.
    """
    def __init__(self,index,local_source,internal_mask,discovery="tree",leaf_size=32):
        if discovery not in ("tree","flat"):
            raise ValueError("Choose tree or flat discovery.")
        self.index=index
        self.active=np.flatnonzero(~internal_mask)
        self.source=Source(local_source.origins[self.active].copy(),local_source.directions[self.active].copy(),
                           local_source.owners[self.active].copy(),local_source.area)
        self.original_count=len(local_source.owners)
        self.discovery=discovery
        g=index.groups[self.source.owners]
        self.group_ids=[np.flatnonzero(g==s) for s in range(len(index.roots))]
        self.trees=[build_ray_tree(self.source.origins,self.source.directions,ids,leaf_size) if len(ids) and discovery=="tree" else None
                    for ids in self.group_ids]
        self.hits=np.zeros((len(index.roots),len(self.active)),dtype=np.bool_)
        self.hit_counts=np.zeros(len(self.active),dtype=np.int32)
        self.previous=None
        self.previous_rotations=None
        self.previous_translations=None
        self.last_metrics=self.update(index.rotations,index.translations)

    def _relative(self,rotations,translations):
        values={} if self.previous is None else self.previous.copy()
        if self.previous is None:
            moved=np.ones(len(rotations),dtype=bool)
        else:
            moved=np.any(rotations!=self.previous_rotations,axis=(1,2))|np.any(translations!=self.previous_translations,axis=1)
        candidates=[]
        for s in range(len(rotations)):
            for t in range(len(rotations)):
                if s==t or not len(self.group_ids[s]) or not (moved[s] or moved[t]):
                    continue
                rotation=rotations[s].T@rotations[t]
                translation=rotations[s].T@(translations[t]-translations[s])
                a=self.index.local.lo[self.index.roots[t]]; b=self.index.local.hi[self.index.roots[t]]
                center=rotation@((a+b)/2)+translation
                half=np.abs(rotation)@((b-a)/2)
                pad=1e-12*(1+np.abs(center)+half)
                values[s,t]=(rotation,translation,center-half-pad,center+half+pad)
                candidates.append((s,t))
        return values,candidates

    def update(self,rotations,translations):
        start=time.perf_counter()
        current,pairs=self._relative(rotations,translations)
        metrics=dict(relative_s=time.perf_counter()-start,discovery_s=0.,coupling_s=0.,
                     changed_pairs=0,total_pairs=len(current),eligible_pair_rays=0,
                     candidate_pair_rays=0,new_bound_pair_rays=0,discovery_nodes=0,discovery_rays=0,
                     sphere_tests=0,boxes=0,changed_hits=0,cleared_hits=0,set_hits=0)
        for s,t in pairs:
            new=current[s,t]
            old=new if self.previous is None else self.previous[s,t]
            if self.previous is not None and np.array_equal(old[0],new[0]) and np.array_equal(old[1],new[1]):
                continue
            metrics["changed_pairs"]+=1
            metrics["eligible_pair_rays"]+=len(self.group_ids[s])
            start=time.perf_counter()
            if self.discovery=="tree":
                candidates,potential,nodes,rays=discover_tree(self.source.origins,self.source.directions,
                    old[2],old[3],new[2],new[3],*self.trees[s].arrays)
            else:
                candidates,potential,nodes,rays=discover_flat(self.source.origins,self.source.directions,
                    self.group_ids[s],old[2],old[3],new[2],new[3])
            metrics["discovery_s"]+=time.perf_counter()-start
            metrics["candidate_pair_rays"]+=len(candidates)
            metrics["new_bound_pair_rays"]+=int(np.count_nonzero(potential))
            metrics["discovery_nodes"]+=nodes; metrics["discovery_rays"]+=rays
            start=time.perf_counter()
            nb,nt,nc,nclear,nset=update_pair(self.source.origins,self.source.directions,self.source.owners,
                candidates,potential,new[0],new[1],self.index.centers,self.index.radii,*self.index.local.arrays,
                self.index.roots[t],self.hits[t],self.hit_counts)
            metrics["coupling_s"]+=time.perf_counter()-start
            metrics["sphere_tests"]+=nt; metrics["boxes"]+=nb
            metrics["changed_hits"]+=nc; metrics["cleared_hits"]+=nclear; metrics["set_hits"]+=nset
        self.previous=current
        self.previous_rotations=np.array(rotations,copy=True)
        self.previous_translations=np.array(translations,copy=True)
        self.last_metrics=metrics
        return metrics

    def blocked(self):
        result=np.ones(self.original_count,dtype=np.bool_)
        result[self.active]=self.hit_counts>0
        return result

    def pa(self):
        return self.source.area*np.count_nonzero(self.hit_counts==0)/(4*self.original_count)

    @property
    def nbytes(self):
        return self.hits.nbytes+self.hit_counts.nbytes+self.active.nbytes+sum(a.nbytes for a in
            [self.source.origins,self.source.directions,self.source.owners])+sum(t.nbytes for t in self.trees if t is not None)


class AdaptiveCoupling:
    """Use full coupling for broad pose changes; retain last valid sparse cache.

    Full evaluation does not relabel the old pair cache as current. A later
    sparse update compares against the LAST SPARSE pose, not the previous full
    evaluation, preserving safe discovery after fallback and re-entry.
    """
    def __init__(self,index,local_source,internal_mask):
        self.sparse=SparseCoupling(index,local_source,internal_mask,"flat")
        self.index=index
        self.full_hits=None
        self.last_rotations=np.array(index.rotations,copy=True)
        self.last_translations=np.array(index.translations,copy=True)
        self.value=self.sparse.pa()

    def update(self,rotations,translations):
        if np.array_equal(rotations,self.last_rotations) and np.array_equal(translations,self.last_translations):
            return dict(route="no_change",sphere_tests=0,boxes=0)
        changed=np.any(rotations!=self.sparse.previous_rotations,axis=(1,2))|np.any(translations!=self.sparse.previous_translations,axis=1)
        # Explicit workload heuristic, not an empirically universal crossover.
        broad=len(rotations)>=4 and np.count_nonzero(changed)>=len(rotations)/2
        if broad:
            self.index.refit(rotations,translations)
            posed=pose_source(self.sparse.source,self.index.groups,rotations,translations)
            hits,boxes,tests,transforms=self.index.query(posed,self.index.groups[posed.owners])
            self.full_hits=hits
            self.value=self.sparse.source.area*np.count_nonzero(~hits)/(4*self.sparse.original_count)
            result=dict(route="full",sphere_tests=int(tests),boxes=int(boxes),transforms=int(transforms))
        else:
            result=self.sparse.update(rotations,translations)
            result["route"]="sparse"
            self.full_hits=None
            self.value=self.sparse.pa()
        self.last_rotations=np.array(rotations,copy=True)
        self.last_translations=np.array(translations,copy=True)
        return result

    def pa(self):
        return self.value

    def blocked(self):
        if self.full_hits is None:
            return self.sparse.blocked()
        result=np.ones(self.sparse.original_count,dtype=np.bool_)
        result[self.sparse.active]=self.full_hits
        return result

    @property
    def nbytes(self):
        return self.sparse.nbytes+self.last_rotations.nbytes+self.last_translations.nbytes+(0 if self.full_hits is None else self.full_hits.nbytes)
