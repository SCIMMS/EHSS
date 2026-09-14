"""Conservative finite cylinders on the exact same physical BVH and scalar rays.

Cylinders are broad-phase boundaries only. Leaves contain the original atom
spheres; no envelope/ghost surface is a physical scatterer.
"""
import time
import numpy as np
from numba import njit
from .ehss_cpu import entry_scalar,reflect_scalar
from .ehss_support_tree import box_interval
from .ehss_state import EHSSResult


class CylinderBounds:
    def __init__(self,model,previous=None):
        tick=time.perf_counter(); t,s=model.tree,model.spheres; n=len(t.left)
        if previous is not None and (not np.array_equal(t.groups,previous.groups) or not np.array_equal(t.end,previous.end)):
            raise ValueError('Cylinder topology changed')
        self.groups=t.groups.copy(); self.end=t.end.copy(); self.world=s.centers.copy(); self.radii=s.radii.copy()
        self.centers=np.empty((n,3)); self.axes=np.empty((n,3)); self.low=np.empty(n); self.high=np.empty(n); self.radius=np.empty(n)
        if previous is not None:
            for key in ('centers','axes','low','high','radius'): getattr(self,key)[:]=getattr(previous,key)
        atom_leaf=t.leaf_node[t.groups]; rebuilt=[]
        moved=np.ones(len(s.radii),bool) if previous is None else np.any(s.centers!=previous.world,axis=1)|(s.radii!=previous.radii)
        for node in range(n):
            ids=np.flatnonzero((atom_leaf>=node)&(atom_leaf<t.end[node]))
            if previous is not None and not moved[ids].any(): continue
            x=s.centers[ids]; center=x.mean(axis=0); q=x-center
            if len(ids)==1: axis=np.array([1.,0.,0.])
            else:
                _,_,vh=np.linalg.svd(q,full_matrices=False); axis=vh[0]
            axis=axis/np.linalg.norm(axis); z=q@axis
            pad=1e-9*max(1.,float(np.max(np.abs(x))),float(s.radii[ids].max()))
            self.centers[node]=center; self.axes[node]=axis
            self.low[node]=np.min(z-s.radii[ids])-pad; self.high[node]=np.max(z+s.radii[ids])+pad
            self.radius[node]=np.max(np.linalg.norm(q-z[:,None]*axis,axis=1)+s.radii[ids])+pad
            rebuilt.append(node)
        self.receipt=dict(prepare_s=time.perf_counter()-tick,refreshed_nodes=rebuilt,nodes=n,
            bounds_bytes=sum(getattr(self,k).nbytes for k in ('centers','axes','low','high','radius')),
            PCA_axis=True,all_atom_spheres_enclosed=True,previous_used=previous is not None)


@njit(inline='always')
def cylinder_interval(o,d,c,u,low,high,radius):
    x0,x1,x2=o[0]-c[0],o[1]-c[1],o[2]-c[2]
    z=(x0*u[0]+x1*u[1])+x2*u[2]; dz=(d[0]*u[0]+d[1]*u[1])+d[2]*u[2]
    near,far=-np.inf,np.inf
    if dz==0.:
        if z<low or z>high: return np.inf,-np.inf
    else:
        a,b=(low-z)/dz,(high-z)/dz; near,far=min(a,b),max(a,b)
    p0,p1,p2=x0-z*u[0],x1-z*u[1],x2-z*u[2]
    v0,v1,v2=d[0]-dz*u[0],d[1]-dz*u[1],d[2]-dz*u[2]
    aa=(v0*v0+v1*v1)+v2*v2
    if aa==0.:
        if (p0*p0+p1*p1)+p2*p2>radius*radius: return np.inf,-np.inf
    else:
        projection=((p0*v0+p1*v1)+p2*v2)/aa
        q0,q1,q2=p0-projection*v0,p1-projection*v1,p2-projection*v2
        disc=radius*radius-((q0*q0+q1*q1)+q2*q2)
        if disc<0.: return np.inf,-np.inf
        width=np.sqrt(max(0.,disc)/aa)
        near=max(near,-projection-width); far=min(far,-projection+width)
    return near,far


@njit
def nearest(o,d,last,kind,any_hit,lo,hi,end,leaf,offsets,ids,world,radii,centers,axes,low,high,radius,counters):
    node,atom,best=0,-1,np.inf
    while node<len(end):
        counters[0]+=1
        if kind!=1:
            counters[1]+=1; a,b=box_interval(o,d,lo[node],hi[node])
            if a>best or b<0.: node=end[node]; continue
        if kind!=0 and (kind!=3 or high[node]-low[node]>4*radius[node]):
            counters[2]+=1; a,b=cylinder_interval(o,d,centers[node],axes[node],low[node],high[node],radius[node])
            if a>b or a>best or b<0.: node=end[node]; continue
        group=leaf[node]
        if group>=0:
            for k in range(offsets[group],offsets[group+1]):
                candidate=ids[k]
                if candidate==last: continue
                counters[3]+=1; distance=entry_scalar(o,d,world[candidate],radii[candidate])
                if distance<best or (distance==best and np.isfinite(distance) and (atom<0 or candidate<atom)):
                    atom,best=candidate,distance
                    if any_hit: return atom,best
        node+=1
    return atom,best


@njit
def trace(origins,directions,lasts,initial,cap,record,kind,pa,lo,hi,end,leaf,offsets,ids,world,radii,centers,axes,low,high,radius):
    p,d,counts=origins.copy(),directions.copy(),initial.copy()
    escaped=np.zeros(len(p),np.bool_); unresolved=escaped.copy(); hits=escaped.copy()
    history=np.full((len(p),cap if record else 0),-1,np.int32); counters=np.zeros(4,np.int64)
    for ray in range(len(p)):
        last=lasts[ray]
        if record and counts[ray]==1: history[ray,0]=last
        while True:
            atom,distance=nearest(p[ray],d[ray],last,kind,pa,lo,hi,end,leaf,offsets,ids,world,radii,centers,axes,low,high,radius,counters)
            if pa: hits[ray]=atom>=0; break
            if atom<0: escaped[ray]=True; break
            if counts[ray]>=cap: unresolved[ray]=True; break
            reflect_scalar(p[ray],d[ray],atom,distance,world,radii)
            if record: history[ray,counts[ray]]=atom
            counts[ray]+=1; last=atom
    return p,d,escaped,unresolved,counts,history,counters,hits


class CylinderBVHTransport:
    def __init__(self,model,bounds,kind='cylinder'):
        if kind not in ('box','cylinder','box_cylinder','selected_cylinder'): raise ValueError('Unknown boundary kind')
        self.model,self.bounds,self.kind=model,bounds,kind
        if not np.array_equal(model.tree.groups,bounds.groups): raise ValueError('Wrong bounds topology')

    def _run(self,source,cap,record,pa):
        if type(cap) is not int or cap<1 or np.any(source.initial_bounces>1): raise ValueError('Valid collision cap required')
        s,t,b=self.model.spheres,self.model.tree,self.bounds
        if not np.array_equal(s.centers,b.world) or not np.array_equal(s.radii,b.radii): raise ValueError('Stale cylinder geometry')
        if np.any(source.last_atom < -1) or np.any(source.last_atom>=len(s.radii)): raise ValueError('Invalid atom identity')
        if pa and np.any(source.initial_bounces): raise ValueError('Any-hit PA expects uncollided sources')
        start=time.perf_counter()
        r=trace(source.origins,source.directions,source.last_atom,source.initial_bounces,cap,record,
            ('box','cylinder','box_cylinder','selected_cylinder').index(self.kind),pa,t.lo,t.hi,t.end,t.leaf_group,t.offsets,t.ids,
            s.centers,s.radii,b.centers,b.axes,b.low,b.high,b.radius)
        metrics=dict(query_s=time.perf_counter()-start,boundary=self.kind,threads=1,
            node_visits=int(r[6][0]),box_tests=int(r[6][1]),cylinder_tests=int(r[6][2]),sphere_tests=int(r[6][3]))
        return r,metrics

    def trace(self,source,cap=64,record_history=False):
        r,m=self._run(source,cap,record_history,False)
        return EHSSResult(*r[:6],source,m)

    def any_hit(self,source):
        r,m=self._run(source,1,False,True)
        return r[-1],m
