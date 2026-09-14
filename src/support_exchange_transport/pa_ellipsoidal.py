"""Homothetic centroid ellipsoid interfaces for exact inside-out PA packets.

Only numerical interfaces change. Atom geometry and physical unit ray directions
stay in world coordinates, and neither source weights nor the PA measure change.
"""
from __future__ import annotations

import numpy as np
import time
from numba import njit
from .pa_spatial import Tree,build_tree
from .pa_concentric import segment_tree,segment_sphere


@njit(cache=True)
def ball_metric_bounds(points,radii,axes):
    """Conservative bounds on sqrt(sum((p+u)^2/a^2)), ||u|| <= r.

    For D=diag(1/a^2), b=Dp, base=p'Dp:
      max q <= base + lambda*r^2 + sum(b^2/(lambda-D)), lambda>max(D)
      min q >= base - lambda*r^2 - sum(b^2/(D+lambda)), lambda>=0.
    Bisection tightens these dual bounds, including the max hard case b_top=0.
    Outward rounding margins preserve membership near numerical boundaries.
    """
    dd=1./(axes*axes); dmax=np.max(dd)
    low=np.empty(len(points)); high=np.empty(len(points))
    for i in range(len(points)):
        p=points[i]; r=radii[i]; b=dd*p
        base=np.dot(p,b); bnorm=np.sqrt(np.dot(b,b))
        if bnorm==0.:
            low[i]=0.; high[i]=r*np.sqrt(dmax)*(1.+1e-12)
            continue
        left=dmax; right=np.nextafter(dmax+bnorm/r,np.inf)
        for _ in range(64):
            mid=left+(right-left)/2.
            if mid<=dmax or mid==left or mid==right:
                break
            length2=0.
            for k in range(3):
                length2+=(b[k]/(mid-dd[k]))**2
            if length2>r*r:
                left=mid
            else:
                right=mid
        lam=max(right,np.nextafter(dmax,np.inf))
        upper=base+lam*r*r
        for k in range(3):
            upper+=b[k]*b[k]/(lam-dd[k])
        high[i]=np.sqrt(max(0.,upper+1e-12*max(1.,upper)))
        if np.dot(p,p)<=r*r:
            low[i]=0.; continue
        left=0.; right=bnorm/r
        for _ in range(64):
            mid=(left+right)/2.
            length2=0.
            for k in range(3):
                length2+=(b[k]/(dd[k]+mid))**2
            if length2>r*r:
                left=mid
            else:
                right=mid
        lam=right; correction=lam*r*r
        for k in range(3):
            correction+=b[k]*b[k]/(dd[k]+lam)
        low[i]=np.sqrt(max(0.,base-correction-1e-12*max(1.,base+correction)))
    return low,high


def fit_centroid_ellipsoid(spheres,profile=None):
    """PCA shape with a sphere-radius floor, uniformly scaled to enclose atoms.

    Fixed centroid and PCA axis ratios; this is not a minimum-area optimizer.
    The isotropic radius term makes collinear/planar atom sets well defined.
    """
    start=time.perf_counter() if profile is not None else 0.
    center=spheres.centers.mean(axis=0)
    centered=spheres.centers-center
    covariance=centered.T@centered/len(centered)+np.eye(3)*np.mean(spheres.radii**2)/5.
    eigenvalues,rotation=np.linalg.eigh(covariance)
    axes=np.sqrt(eigenvalues)
    points=np.ascontiguousarray(centered@rotation)
    if profile is not None:
        profile["centroid_pca_s"]=time.perf_counter()-start
        start=time.perf_counter()
    _,maximum=ball_metric_bounds(points,spheres.radii,axes)
    axes*=np.max(maximum)*(1.+1e-10)
    if profile is not None:
        profile["fit_enclosure_s"]=time.perf_counter()-start
    return center,rotation,axes


def ellipsoid_area(axes,order=96):
    """Area quadrature: abc integral_S2 ||diag(1/a,1/b,1/c) n|| dOmega."""
    axes=np.asarray(axes,dtype=float)
    z,w=np.polynomial.legendre.leggauss(order)
    phi=(np.arange(2*order)+.5)*(np.pi/order)
    transverse=np.sqrt(1-z*z)[:,None]
    x=transverse*np.cos(phi); y=transverse*np.sin(phi)
    integrand=np.sqrt((x/axes[0])**2+(y/axes[1])**2+(z[:,None]/axes[2])**2)
    return float(np.prod(axes)*(np.pi/order)*np.sum(w[:,None]*integrand))


@njit(cache=True)
def ellipsoid_transport(origins,directions,owners,center,metric_map,boundaries,
                        centers,radii,offsets,members,roots,lo,hi,end,first,count,ids,use_bvh):
    blocked=np.zeros(len(origins),dtype=np.bool_)
    visits=np.zeros(len(boundaries),dtype=np.int64)
    absorbed=np.zeros(len(boundaries),dtype=np.int64)
    tests=0; boxes=0; inward_sources=0; segments=0; exit_error=0.
    for ray in range(len(origins)):
        o=origins[ray]; d=directions[ray]
        aa=0.; bb=0.; cc=0.
        for k in range(3):
            x=0.; v=0.
            for j in range(3):
                x+=(o[j]-center[j])*metric_map[j,k]
                v+=d[j]*metric_map[j,k]
            aa+=v*v; bb+=x*v; cc+=x*x
        minimum=max(0.,cc-bb*bb/aa)
        inward=bb<0
        if inward:
            inward_sources+=1
        layer=min(np.searchsorted(boundaries,np.sqrt(cc),side="right"),len(boundaries)-1)
        begin=0.; finish=0.
        while layer<len(boundaries):
            inner=boundaries[layer-1] if layer>0 else 0.
            if inward and layer>0 and inner*inner>minimum:
                finish=max(begin,(-bb-np.sqrt(max(0.,aa*(inner*inner-minimum))))/aa)
                nxt=layer-1
            else:
                inward=False
                finish=max(begin,(-bb+np.sqrt(max(0.,aa*(boundaries[layer]**2-minimum))))/aa)
                nxt=layer+1
            if finish>begin:
                visits[layer]+=1; segments+=1
                if use_bvh and roots[layer]>=0:
                    hit,nb,nt=segment_tree(o,d,owners[ray],begin,finish,centers,radii,
                        lo,hi,end,first,count,ids,roots[layer])
                    boxes+=nb; tests+=nt; blocked[ray]=hit
                elif not use_bvh:
                    for p in range(offsets[layer],offsets[layer+1]):
                        atom=members[p]
                        if atom==owners[ray]:
                            continue
                        tests+=1
                        if segment_sphere(o,d,centers[atom],radii[atom],begin,finish):
                            blocked[ray]=True; break
                if blocked[ray]:
                    absorbed[layer]+=1; break
            begin=finish; layer=nxt
        if not blocked[ray]:
            actual=np.sqrt(max(0.,cc+2*bb*finish+aa*finish*finish))
            exit_error=max(exit_error,abs(actual-boundaries[-1]))
    return blocked,tests,boxes,segments,inward_sources,visits,absorbed,exit_error


class EllipsoidalTransport:
    def __init__(self,spheres,n_shells=8,use_bvh=True,axes=None,rotation=None,center=None,leaf_size=4,profile=None):
        start=time.perf_counter() if profile is not None else 0.
        if profile is not None:
            profile.clear()
        if not isinstance(n_shells,(int,np.integer)) or n_shells<1:
            raise ValueError("Positive integer shell count required")
        if axes is None:
            if rotation is not None or center is not None:
                raise ValueError("Explicit center/rotation require explicit axes")
            center,rotation,axes=fit_centroid_ellipsoid(spheres,profile)
        validation_start=time.perf_counter() if profile is not None else 0.
        self.spheres=spheres
        self.center=np.asarray(spheres.centers.mean(axis=0) if center is None else center,dtype=float)
        self.rotation=np.asarray(np.eye(3) if rotation is None else rotation,dtype=float)
        self.axes=np.asarray(axes,dtype=float)
        if self.center.shape!=(3,) or not np.isfinite(self.center).all():
            raise ValueError("Finite center required")
        if self.axes.shape!=(3,) or not np.isfinite(self.axes).all() or np.any(self.axes<=0):
            raise ValueError("Three positive finite semiaxes required")
        if self.rotation.shape!=(3,3) or not np.isfinite(self.rotation).all() or not np.allclose(self.rotation.T@self.rotation,np.eye(3),atol=1e-12,rtol=1e-12):
            raise ValueError("Orthonormal rotation required")
        points=np.ascontiguousarray((spheres.centers-self.center)@self.rotation)
        if profile is not None:
            profile["validation_transform_s"]=time.perf_counter()-validation_start
            range_start=time.perf_counter()
        low,high=ball_metric_bounds(points,spheres.radii,self.axes)
        if np.max(high)>=1.:
            raise ValueError("Ellipsoid must strictly enclose full atom spheres")
        self.metric_map=np.ascontiguousarray(self.rotation/self.axes[None,:])
        self.boundaries=np.linspace(0.,1.,n_shells+1)[1:]
        self.use_bvh=use_bvh
        self.radial_low=low; self.radial_high=high
        if profile is not None:
            profile["atom_ranges_s"]=time.perf_counter()-range_start
            profile["membership_s"]=0.; profile["bvh_s"]=0.
        offsets=[0]; members_all=[]; trees=[]; roots=[]; nn=0; ni=0; lower=0.
        for upper in self.boundaries:
            stamp=time.perf_counter() if profile is not None else 0.
            members=np.flatnonzero((high>=lower-1e-12)&(low<=upper+1e-12))
            members_all.extend(members.tolist()); offsets.append(len(members_all))
            if profile is not None:
                profile["membership_s"]+=time.perf_counter()-stamp
                stamp=time.perf_counter()
            if use_bvh and len(members):
                tree=build_tree(spheres.centers[members]-spheres.radii[members,None],
                    spheres.centers[members]+spheres.radii[members,None],leaf_size=leaf_size)
                roots.append(nn); tree.end+=nn; tree.first+=ni; tree.ids=members[tree.ids]
                nn+=len(tree.end); ni+=len(tree.ids); trees.append(tree)
            else:
                roots.append(-1)
            if profile is not None:
                profile["bvh_s"]+=time.perf_counter()-stamp
            lower=upper
        stamp=time.perf_counter() if profile is not None else 0.
        self.offsets=np.array(offsets,dtype=np.int64)
        self.primitive_ids=np.array(members_all,dtype=np.int64)
        self.roots=np.array(roots,dtype=np.int64)
        self.tree=(Tree(*(np.concatenate([t.arrays[k] for t in trees]) for k in range(6))) if trees else
            Tree(np.zeros((0,3)),np.zeros((0,3)),*[np.empty(0,dtype=np.int64) for _ in range(4)]))
        if profile is not None:
            profile["pack_s"]=time.perf_counter()-stamp
            total=time.perf_counter()-start
            profile["other_s"]=total-sum(profile.values())
            profile["total_s"]=total

    def query(self,source):
        return ellipsoid_transport(source.origins,source.directions,source.owners,self.center,self.metric_map,
            self.boundaries,self.spheres.centers,self.spheres.radii,self.offsets,self.primitive_ids,
            self.roots,*self.tree.arrays,self.use_bvh)

    @property
    def nbytes(self):
        return self.tree.nbytes+sum(a.nbytes for a in [self.center,self.rotation,self.axes,self.metric_map,
            self.boundaries,self.radial_low,self.radial_high,self.offsets,self.primitive_ids,self.roots])
