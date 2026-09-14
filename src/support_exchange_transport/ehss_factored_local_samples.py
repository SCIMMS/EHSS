"""Material sample columns: isolated owned itineraries T and current foreign G.

Columns start at a known first reflection and end on the first foreign physical
collision, escape, or cap. They are not a complete molecular CCS solver. All
candidate reflections use current physical spheres; retained itineraries may
miss a new owned collision after strain, which is an explicit approximation.
"""
from dataclasses import dataclass
import time
import numpy as np
from numba import njit
from .ehss_sparse_path_response import readonly,record_from_paths
from .ehss_reference import nearest_all,entry_distance
from .ehss_guarded_intrinsic_response import nearest_world,reflect
from .ehss_state import EHSSSource


@njit(cache=True)
def local_itineraries(points,directions,lasts,centers,radii,cap):
    paths=np.full((len(points),cap-1),-1,dtype=np.int64);lengths=np.zeros(len(points),dtype=np.int64)
    capped=np.zeros(len(points),dtype=np.bool_);tests=0
    for i in range(len(points)):
        p=points[i].copy();d=directions[i].copy();last=lasts[i]
        for j in range(cap):
            atom,distance,nt=nearest_all(p,d,last,centers,radii);tests+=nt
            if atom<0:break
            if j==cap-1:capped[i]=True;break
            paths[i,j]=atom;lengths[i]+=1;reflect(p,d,atom,distance,centers,radii);last=atom
    return paths,lengths,capped,tests


@njit(cache=False)
def point_inside(node,p,last,lo,hi,left,right,leaf_group,offsets,ids,centers,radii,counters):
    counters[0]+=1
    for k in range(3):
        if p[k]<lo[node,k] or p[k]>hi[node,k]:return False
    group=leaf_group[node]
    if group>=0:
        for j in range(offsets[group],offsets[group+1]):
            atom=ids[j]
            if atom==last:continue
            counters[1]+=1;delta=p-centers[atom]
            if np.dot(delta,delta)<(radii[atom]-1e-10)**2:return True
        return False
    return point_inside(left[node],p,last,lo,hi,left,right,leaf_group,offsets,ids,centers,radii,counters) or point_inside(right[node],p,last,lo,hi,left,right,leaf_group,offsets,ids,centers,radii,counters)


@njit(cache=False)
def apply_columns(points,directions,lasts,owners,paths,lengths,cap,factored,
                  lo,hi,left,right,end,leaf_group,leaf_node,offsets,ids,groups,centers,radii):
    p=points.copy();d=directions.copy();counts=np.ones(len(p),dtype=np.int64)
    history=np.full((len(p),cap),-1,dtype=np.int64);history[:,0]=lasts
    # -3 occluded material input, -2 cap unresolved, -1 global escape, >=0 foreign collider.
    terminal=np.full(len(p),-2,dtype=np.int64);counters=np.zeros(2,dtype=np.int64)
    stats=np.zeros(4,dtype=np.int64) # candidate tests, foreign searches, local fallback, foreign interruptions
    for i in range(len(p)):
        last=lasts[i];owner=owners[i];node=leaf_node[owner];cursor=0;fallback=False
        if point_inside(0,p[i],last,lo,hi,left,right,leaf_group,offsets,ids,centers,radii,counters):
            terminal[i]=-3;continue
        while True:
            if not factored or fallback:
                atom,distance=nearest_world(0,p[i],d[i],last,np.inf,-1,lo,hi,left,right,end,leaf_group,offsets,ids,centers,radii,counters)
            else:
                wanted=-1;distance=np.inf
                if cursor<lengths[i]:
                    wanted=paths[i,cursor];distance=entry_distance(p[i],d[i],centers[wanted],radii[wanted]);stats[0]+=1
                    if not np.isfinite(distance):fallback=True;stats[2]+=1;continue
                else:
                    # An old terminal flight is not an absence certificate after
                    # strain or accumulated reflection roundoff. Re-query all geometry.
                    fallback=True;stats[2]+=1;continue
                atom,before=nearest_world(0,p[i],d[i],last,distance,node,lo,hi,left,right,end,leaf_group,offsets,ids,centers,radii,counters)
                stats[1]+=1
                if atom>=0 and (before<distance or (before==distance and (wanted<0 or atom<wanted))):
                    distance=before
                    if wanted>=0:stats[3]+=1
                else:atom=wanted
            if atom<0:terminal[i]=-1;break
            if counts[i]>=cap:terminal[i]=-2;break
            reflect(p[i],d[i],atom,distance,centers,radii)
            history[i,counts[i]]=atom;counts[i]+=1;last=atom;cursor+=1
            if groups[atom]!=owner:terminal[i]=atom;break
    return p,d,counts,history,terminal,counters,stats


@dataclass(frozen=True)
class LocalColumn:
    centers: np.ndarray
    radii: np.ndarray
    paths: np.ndarray
    lengths: np.ndarray
    capped: np.ndarray
    tests: int


class MaterialSamples:
    def __init__(self,model,results,cap=128):
        if not results or not isinstance(cap,int) or cap<1:raise ValueError('Results and a positive cap required')
        self.cap=cap;self.groups=readonly(model.tree.groups.copy());self.reference=readonly(model.spheres.centers.copy())
        self.radii=readonly(model.spheres.radii.copy());self.atoms=tuple(readonly(np.flatnonzero(self.groups==g)) for g in range(model.tree.n_groups))
        records=[];tags=[]
        for r in results:
            s=r.source
            records.append(record_from_paths(s.origins,s.directions,s.weights/len(results),s.initial_bounces,r.collider_ids,r.bounces,model.spheres.centers,model.spheres.radii))
            tags.append(np.repeat(s.incoming,r.bounces,axis=0))
        p,d,lasts,mass=[np.concatenate([r[k] for r in records]) for k in range(4)]
        self.lasts=readonly(lasts);self.owners=readonly(self.groups[lasts]);self.mass=readonly(mass)
        normal=p-model.spheres.centers[lasts];normal/=np.linalg.norm(normal,axis=1)[:,None]
        # The reference material axes are the initial global axes; later poses are fitted against them.
        self.normals=readonly(normal);self.directions=readonly(d);self.tags=readonly(np.concatenate(tags))
        self.rows=tuple(readonly(np.flatnonzero(self.owners==g)) for g in range(len(self.atoms)))

    def bind(self,model,previous=None,tolerance=1e-10):
        if not np.isfinite(tolerance) or tolerance<0:raise ValueError('Nonnegative finite shape tolerance required')
        if not np.array_equal(model.tree.groups,self.groups):raise ValueError('Stable physical ownership required')
        if previous is not None and previous.bank is not self:raise ValueError('Response belongs to another material bank')
        tick=time.perf_counter();columns=[];reused=[];rebuilt=[];residuals=[];compile_s=0.;tests=0
        world=model.spheres.centers;radii=model.spheres.radii;n=len(self.lasts)
        points=np.empty((n,3));directions=points.copy();tags=points.copy();paths=np.full((n,self.cap-1),-1,dtype=np.int64);lengths=np.zeros(n,dtype=np.int64)
        for g,(atoms,rows) in enumerate(zip(self.atoms,self.rows)):
            reference=self.reference[atoms];target=world[atoms];a=reference-reference.mean(axis=0);b=target-target.mean(axis=0)
            if np.array_equal(target-target[0],reference-reference[0]):rotation=np.eye(3)
            else:
                u,_,vh=np.linalg.svd(a.T@b);correction=np.eye(3);correction[2,2]=1. if np.linalg.det(u@vh)>=0 else -1.
                rotation=(u@correction@vh).T
            local=(target-target[0])@rotation
            points[rows]=world[self.lasts[rows]]+radii[self.lasts[rows],None]*(self.normals[rows]@rotation.T)
            directions[rows]=self.directions[rows]@rotation.T;tags[rows]=self.tags[rows]@rotation.T
            old=None if previous is None else previous.columns[g]
            residual=np.inf if old is None else float(np.linalg.norm(local-old.centers,axis=1).max())
            residuals.append(residual if np.isfinite(residual) else None)
            if old is not None and residual<=tolerance and np.array_equal(radii[atoms],old.radii):column=old;reused.append(g)
            else:
                start=time.perf_counter();last=np.searchsorted(atoms,self.lasts[rows])
                local_points=local[last]+radii[self.lasts[rows],None]*self.normals[rows]
                routes,counts,capped,nt=local_itineraries(local_points,self.directions[rows],last,local,radii[atoms],self.cap)
                column=LocalColumn(readonly(local.copy()),readonly(radii[atoms].copy()),readonly(routes),readonly(counts),readonly(capped),int(nt))
                compile_s+=time.perf_counter()-start;tests+=int(nt);rebuilt.append(g)
            columns.append(column);valid=column.paths>=0
            mapped=np.full(column.paths.shape,-1,dtype=np.int64);mapped[valid]=atoms[column.paths[valid]]
            paths[rows]=mapped;lengths[rows]=column.lengths
        receipt=dict(bind_s=time.perf_counter()-tick,local_compile_s=compile_s,local_sphere_tests=tests,reused_groups=reused,rebuilt_groups=rebuilt,
            shape_residuals=residuals,tolerance=tolerance,samples=n,local_multi_samples=int(np.count_nonzero(lengths>=1)),
            reuse_is_sampled_itinerary=True,full_response_matrix_reused=False)
        return BoundSamples(self,model,tuple(columns),points,directions,tags,paths,lengths,receipt)


@dataclass
class BoundSamples:
    bank: MaterialSamples
    model: object
    columns: tuple
    points: np.ndarray
    directions: np.ndarray
    tags: np.ndarray
    paths: np.ndarray
    lengths: np.ndarray
    receipt: dict

    def apply(self,factored=True,weights=None):
        tick=time.perf_counter();b=self.bank;t=self.model.tree;s=self.model.spheres
        mass=b.mass if weights is None else np.asarray(weights,float)
        if mass.shape!=b.mass.shape or not np.isfinite(mass).all() or np.any(mass<0):raise ValueError('Nonnegative sample weights required')
        p,d,counts,history,terminal,counters,stats=apply_columns(self.points,self.directions,b.lasts,b.owners,self.paths,self.lengths,b.cap,factored,
            t.lo,t.hi,t.left,t.right,t.end,t.leaf_group,t.leaf_node,t.offsets,t.ids,t.groups,s.centers,s.radii)
        completed=(terminal>=-1);readout=mass*np.clip(1-np.sum(self.tags*d,axis=1),0.,2.)*completed
        return dict(points=p,directions=d,counts=counts,history=history,terminal=terminal,
            moment=float(readout.sum()),completed_mass=float(mass[completed].sum()),occluded_mass=float(mass[terminal==-3].sum()),
            unresolved_mass=float(mass[terminal==-2].sum()),input_mass=float(mass.sum()),
            foreign_mass=float(mass[terminal>=0].sum()),escape_mass=float(mass[terminal==-1].sum()),
            query_s=time.perf_counter()-tick,box_tests=int(counters[0]),sphere_tests=int(counters[1]+stats[0]),
            candidate_tests=int(stats[0]),foreign_searches=int(stats[1]),local_fallbacks=int(stats[2]),foreign_interruptions=int(stats[3]))
