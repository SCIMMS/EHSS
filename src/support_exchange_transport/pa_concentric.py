"""Exact flux packets transported across concentric centroid sphere shells.

Shell boundaries are numerical interfaces, not physical absorbers. Position,
direction and packet weight survive each crossing. Nonradial/inward paths are
handled explicitly; PA absorption still terminates each blocked packet once.
"""
from __future__ import annotations

import numpy as np
import time
from numba import njit
from .pa_spatial import build_tree,Tree


@njit(cache=True)
def segment_box(o,d,lo,hi,begin,finish):
    near,far=begin,finish
    for axis in range(3):
        if d[axis]==0:
            if o[axis]<lo[axis] or o[axis]>hi[axis]:
                return False
        else:
            a,b=(lo[axis]-o[axis])/d[axis],(hi[axis]-o[axis])/d[axis]
            near=max(near,min(a,b)); far=min(far,max(a,b))
            if far<near:
                return False
    return True


@njit(cache=True)
def segment_sphere(o,d,c,r,begin,finish):
    x=o[0]-c[0]; y=o[1]-c[1]; z=o[2]-c[2]
    b=x*d[0]+y*d[1]+z*d[2]
    cc=x*x+y*y+z*z-r*r
    discriminant=b*b-cc
    if discriminant<0:
        return False
    root=np.sqrt(max(0.,discriminant))
    near,far=-b-root,-b+root
    return far>0 and far>=begin and near<=finish


@njit(cache=True)
def segment_tree(o,d,owner,begin,finish,centers,radii,lo,hi,end,first,count,ids,root):
    node=root; stop=end[root]; boxes=0; tests=0
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
                return True,boxes,tests
        node+=1
    return False,boxes,tests


@njit(cache=True)
def shell_transport(origins,directions,owners,center,boundaries,centers,radii,
                    offsets,primitive_ids,roots,lo,hi,end,first,count,ids,use_bvh):
    blocked=np.zeros(len(origins),dtype=np.bool_)
    layer_visits=np.zeros(len(boundaries),dtype=np.int64)
    layer_absorbed=np.zeros(len(boundaries),dtype=np.int64)
    sphere_tests=0; box_tests=0; inward_sources=0; segments=0
    max_exit_error=0.
    times=np.empty(2*len(boundaries)+1)
    rel=np.empty(3)
    for ray in range(len(origins)):
        o=origins[ray]; d=directions[ray]
        rel[:]=o-center
        b=np.dot(rel,d); length2=np.dot(rel,rel)
        if b<0:
            inward_sources+=1
        times[0]=0.; n=1
        for radius in boundaries:
            discriminant=b*b+radius*radius-length2
            if discriminant<0:
                continue
            delta=np.sqrt(max(0.,discriminant))
            for t in (-b-delta,-b+delta):
                if t>0:
                    times[n]=t; n+=1
        ordered=np.sort(times[:n])
        for k in range(n-1):
            a,z=ordered[k],ordered[k+1]
            if z<=a:
                continue
            middle=(a+z)/2
            radius=np.sqrt(max(0.,length2+2*middle*b+middle*middle))
            layer=min(np.searchsorted(boundaries,radius),len(boundaries)-1)
            layer_visits[layer]+=1; segments+=1
            if use_bvh and roots[layer]>=0:
                hit,nb,nt=segment_tree(o,d,owners[ray],a,z,centers,radii,lo,hi,end,first,count,ids,roots[layer])
                box_tests+=nb; sphere_tests+=nt
                blocked[ray]=hit
            elif not use_bvh:
                for p in range(offsets[layer],offsets[layer+1]):
                    atom=primitive_ids[p]
                    if atom==owners[ray]:
                        continue
                    sphere_tests+=1
                    if segment_sphere(o,d,centers[atom],radii[atom],a,z):
                        blocked[ray]=True
                        break
            if blocked[ray]:
                layer_absorbed[layer]+=1
                break
        if not blocked[ray]:
            t=ordered[n-1]
            actual=np.sqrt(max(0.,length2+2*t*b+t*t))
            max_exit_error=max(max_exit_error,abs(actual-boundaries[-1]))
    return blocked,sphere_tests,box_tests,segments,inward_sources,layer_visits,layer_absorbed,max_exit_error


@njit(cache=True)
def shell_transport_stream(origins,directions,owners,center,boundaries,centers,radii,
                           offsets,primitive_ids,roots,lo,hi,end,first,count,ids,use_bvh):
    """Compute only the next interface; do no future work after absorption."""
    blocked=np.zeros(len(origins),dtype=np.bool_)
    layer_visits=np.zeros(len(boundaries),dtype=np.int64)
    layer_absorbed=np.zeros(len(boundaries),dtype=np.int64)
    sphere_tests=0; box_tests=0; inward_sources=0; segments=0; max_exit_error=0.
    for ray in range(len(origins)):
        o=origins[ray]; d=directions[ray]
        x=o[0]-center[0]; y=o[1]-center[1]; z=o[2]-center[2]
        b=x*d[0]+y*d[1]+z*d[2]
        length2=x*x+y*y+z*z
        impact2=max(0.,length2-b*b)
        inward=b<0
        if inward:
            inward_sources+=1
        layer=min(np.searchsorted(boundaries,np.sqrt(length2),side="right"),len(boundaries)-1)
        begin=0.; finish=0.
        while layer<len(boundaries):
            inner_radius=boundaries[layer-1] if layer>0 else 0.
            if inward and layer>0 and inner_radius*inner_radius>impact2:
                finish=max(begin,-b-np.sqrt(max(0.,inner_radius*inner_radius-impact2)))
                next_layer=layer-1
            else:
                inward=False
                finish=max(begin,-b+np.sqrt(max(0.,boundaries[layer]**2-impact2)))
                next_layer=layer+1
            if finish>begin:
                layer_visits[layer]+=1; segments+=1
                if use_bvh and roots[layer]>=0:
                    hit,nb,nt=segment_tree(o,d,owners[ray],begin,finish,centers,radii,
                                           lo,hi,end,first,count,ids,roots[layer])
                    box_tests+=nb; sphere_tests+=nt; blocked[ray]=hit
                elif not use_bvh:
                    for p in range(offsets[layer],offsets[layer+1]):
                        atom=primitive_ids[p]
                        if atom==owners[ray]:
                            continue
                        sphere_tests+=1
                        if segment_sphere(o,d,centers[atom],radii[atom],begin,finish):
                            blocked[ray]=True; break
                if blocked[ray]:
                    layer_absorbed[layer]+=1; break
            begin=finish; layer=next_layer
        if not blocked[ray]:
            actual=np.sqrt(max(0.,length2+2*finish*b+finish*finish))
            max_exit_error=max(max_exit_error,abs(actual-boundaries[-1]))
    return blocked,sphere_tests,box_tests,segments,inward_sources,layer_visits,layer_absorbed,max_exit_error


class ConcentricTransport:
    def __init__(self,spheres,n_shells=8,center=None,boundaries=None,use_bvh=True,leaf_size=4,traversal="stream",profile=None):
        start=time.perf_counter() if profile is not None else 0.
        if profile is not None:
            profile.clear()
        self.spheres=spheres
        self.center=np.asarray(spheres.centers.mean(axis=0) if center is None else center,dtype=float)
        if self.center.shape!=(3,) or not np.isfinite(self.center).all():
            raise ValueError("Finite 3D common center required.")
        radial=np.linalg.norm(spheres.centers-self.center,axis=1)
        outer=float(np.max(radial+spheres.radii))
        outer+=1e-10*max(1.,outer)
        if boundaries is None:
            if n_shells<1:
                raise ValueError("Positive shell count required.")
            self.boundaries=np.linspace(0,outer,n_shells+1)[1:]
        else:
            self.boundaries=np.asarray(boundaries,dtype=float)
            if self.boundaries.ndim!=1 or not len(self.boundaries) or np.any(np.diff(self.boundaries)<=0) or self.boundaries[0]<=0 or not np.isfinite(self.boundaries).all():
                raise ValueError("Strictly increasing positive finite shell radii required.")
            if self.boundaries[-1]<outer:
                raise ValueError("Outermost interface must strictly enclose all spheres.")
        self.use_bvh=use_bvh
        if traversal not in ("stream","sorted"):
            raise ValueError("Choose streaming or sorted interface traversal.")
        self.traversal=traversal
        if profile is not None:
            profile["sphere_geometry_s"]=time.perf_counter()-start
            profile["membership_s"]=0.; profile["bvh_s"]=0.
        offsets=[0]; primitive_ids=[]; trees=[]; roots=[]; nnode=0; nids=0
        lower=0.
        for upper in self.boundaries:
            stamp=time.perf_counter() if profile is not None else 0.
            pad=1e-12*max(1.,upper)
            members=np.flatnonzero((radial+spheres.radii>=lower-pad)&(np.maximum(0.,radial-spheres.radii)<=upper+pad))
            primitive_ids.extend(members.tolist()); offsets.append(len(primitive_ids))
            if profile is not None:
                profile["membership_s"]+=time.perf_counter()-stamp
                stamp=time.perf_counter()
            if use_bvh and len(members):
                t=build_tree(spheres.centers[members]-spheres.radii[members,None],
                             spheres.centers[members]+spheres.radii[members,None],leaf_size=leaf_size)
                roots.append(nnode)
                t.end+=nnode; t.first+=nids; t.ids=members[t.ids]
                nnode+=len(t.end); nids+=len(t.ids); trees.append(t)
            else:
                roots.append(-1)
            if profile is not None:
                profile["bvh_s"]+=time.perf_counter()-stamp
            lower=upper
        stamp=time.perf_counter() if profile is not None else 0.
        self.offsets=np.array(offsets,dtype=np.int64)
        self.primitive_ids=np.array(primitive_ids,dtype=np.int64)
        self.roots=np.array(roots,dtype=np.int64)
        self.tree=(Tree(*(np.concatenate([t.arrays[k] for t in trees]) for k in range(6))) if trees else
                   Tree(np.zeros((0,3)),np.zeros((0,3)),*[np.empty(0,dtype=np.int64) for _ in range(4)]))
        if profile is not None:
            profile["pack_s"]=time.perf_counter()-stamp
            total=time.perf_counter()-start
            profile["other_s"]=total-sum(profile.values())
            profile["total_s"]=total

    def query(self,source):
        transport=shell_transport_stream if self.traversal=="stream" else shell_transport
        return transport(source.origins,source.directions,source.owners,self.center,self.boundaries,
            self.spheres.centers,self.spheres.radii,self.offsets,self.primitive_ids,self.roots,*self.tree.arrays,self.use_bvh)

    @property
    def nbytes(self):
        return self.tree.nbytes+sum(a.nbytes for a in [self.center,self.boundaries,self.offsets,self.primitive_ids,self.roots])
