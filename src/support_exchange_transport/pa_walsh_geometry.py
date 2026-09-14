"""Geometry-generated Walsh coefficients for projected hard-sphere PA.

This is an ACFO-inspired coefficient-operator pilot, not the SO(2)/Bessel ACFO
implementation. A direction is projected to disks and sampled on an N x N
midpoint grid. Disk/row intersections produce intervals analytically. Walsh
prefix sums integrate those intervals without building an atom x pixel mask.
Full coefficients recover the exact discrete hit COUNT; union requires a
nonlinear threshold. A finite grid is not a continuous-area certificate.
"""
from dataclasses import dataclass
import math
import numpy as np
from numba import njit


@dataclass
class WalshBasis:
    size: int

    def __post_init__(self):
        n=self.size
        if not isinstance(n,int) or n<2 or n&(n-1):
            raise ValueError('Grid size must be an integer power of two >=2.')
        bits=n.bit_length()-1
        self.ids=np.array([int(f'{k^(k>>1):0{bits}b}'[::-1],2) for k in range(n)])
        self.h=np.array([[1.-2.*((int(k)&x).bit_count()%2) for x in range(n)] for k in self.ids])
        self.prefix=np.column_stack((np.zeros(n),np.cumsum(self.h,axis=1)))

    def check_modes(self,m):
        if not isinstance(m,int) or m<1 or m>self.size or m&(m-1):
            raise ValueError('Modes per axis must be a power of two <= grid size.')

    def coefficients(self,intervals,m=None):
        m=self.size if m is None else m; self.check_modes(m)
        rows,adds=interval_walsh_rows(intervals,self.prefix,m)
        if m==self.size:
            transformed=fwht_axis0(rows)
            return transformed[self.ids]/(self.size*self.size),adds
        return self.h[:m]@rows/(self.size*self.size),adds

    def synthesize(self,coefficients):
        m=coefficients.shape[0]; self.check_modes(m)
        if coefficients.shape!=(m,m):raise ValueError('Expected square coefficient array.')
        if m==self.size:
            natural=np.empty_like(coefficients)
            natural[np.ix_(self.ids,self.ids)]=coefficients
            return fwht_axis0(fwht_axis0(natural).T).T
        return self.h[:m].T@coefficients@self.h[:m]

    def from_counts(self,counts):
        transformed=fwht_axis0(fwht_axis0(counts.astype(float)).T).T
        return transformed[np.ix_(self.ids,self.ids)]/self.size**2

    @property
    def nbytes(self):return self.ids.nbytes+self.h.nbytes+self.prefix.nbytes


@njit(cache=True)
def fwht_axis0(values):
    out=values.copy();n=out.shape[0];stride=1
    while stride<n:
        for begin in range(0,n,2*stride):
            for j in range(begin,begin+stride):
                for k in range(out.shape[1]):
                    a=out[j,k];b=out[j+stride,k]
                    out[j,k]=a+b;out[j+stride,k]=a-b
        stride*=2
    return out


@njit(cache=True)
def disk_intervals(xy,radii,lower,spacing,n):
    """Integer [start,stop) midpoint intervals; clipped domain is intentional."""
    out=np.zeros((len(radii),n,2),np.int64)
    active=0
    for a in range(len(radii)):
        for y in range(n):
            dy=lower[1]+(y+.5)*spacing-xy[a,1]
            rem=radii[a]*radii[a]-dy*dy
            if rem<0:continue
            half=math.sqrt(rem)
            first=max(0,min(n,int(math.ceil((xy[a,0]-half-lower[0])/spacing-.5))))
            stop=max(0,min(n,int(math.floor((xy[a,0]+half-lower[0])/spacing-.5))+1))
            # Refine endpoint rounding against the same disk inequality.
            while first>0 and (lower[0]+(first-.5)*spacing-xy[a,0])**2<=rem:first-=1
            while first<stop and (lower[0]+(first+.5)*spacing-xy[a,0])**2>rem:first+=1
            while stop<n and (lower[0]+(stop+.5)*spacing-xy[a,0])**2<=rem:stop+=1
            while stop>first and (lower[0]+(stop-.5)*spacing-xy[a,0])**2>rem:stop-=1
            if stop>first:
                out[a,y,0]=first;out[a,y,1]=stop;active+=1
    return out,active


@njit(cache=True)
def interval_walsh_rows(intervals,prefix,m):
    n=intervals.shape[1]; out=np.zeros((n,m)); additions=0
    for a in range(len(intervals)):
        for y in range(n):
            lo,hi=intervals[a,y]
            if hi>lo:
                for k in range(m):out[y,k]+=prefix[k,hi]-prefix[k,lo]
                additions+=m
    return out,additions


@njit(cache=True)
def interval_counts(intervals):
    """Strong non-WHT baseline: row difference accumulation, O(A*N+N^2)."""
    n=intervals.shape[1]; delta=np.zeros((n,n+1),np.int64)
    for a in range(len(intervals)):
        for y in range(n):
            lo,hi=intervals[a,y]
            if hi>lo:delta[y,lo]+=1;delta[y,hi]-=1
    result=np.zeros((n,n),np.int64)
    for y in range(n):
        count=0
        for x in range(n):count+=delta[y,x];result[y,x]=count
    return result


@njit(cache=True)
def pixel_counts(xy,radii,lower,spacing,n,early_exit=False):
    result=np.zeros((n,n),np.int64);tests=0
    for y in range(n):
        yy=lower[1]+(y+.5)*spacing
        for x in range(n):
            xx=lower[0]+(x+.5)*spacing
            for a in range(len(radii)):
                tests+=1
                if (xx-xy[a,0])**2+(yy-xy[a,1])**2<=radii[a]**2:
                    result[y,x]+=1
                    if early_exit:break
    return result,tests


@njit(cache=True)
def xor_product(a,b):
    """Normalized Walsh coefficients of pointwise product; 2D flattened."""
    x=a.ravel();y=b.ravel();out=np.zeros(len(x))
    for i in range(len(x)):
        if x[i]!=0:
            for j in range(len(y)):
                if y[j]!=0:out[i^j]+=x[i]*y[j]
    return out.reshape(a.shape)


def transmission_product(basis,intervals,m=None,keep=None):
    """Small-problem coefficient-only product; optional largest-|c| pruning.

    Only the final DC is used for a uniformly weighted area. Truncation is an
    approximation, not an independent-collision assumption or a certificate.
    """
    m=basis.size if m is None else m; basis.check_modes(m)
    out=np.zeros((m,m));out[0,0]=1.;history=[]
    for a in range(len(intervals)):
        h,_=basis.coefficients(intervals[a:a+1],m)
        t=-h;t[0,0]+=1
        out=xor_product(out,t)
        before=int(np.count_nonzero(np.abs(out)>1e-13))
        if keep is not None and keep<m*m:
            # Preserve DC plus the strongest remaining modes, without a
            # threshold fitted to an oracle or renormalization of the result.
            ids=np.argsort(np.abs(out.ravel()[1:]),kind='stable')[::-1][:max(0,keep-1)]+1
            mask=np.zeros(out.size,bool);mask[0]=True;mask[ids]=True
            out.ravel()[~mask]=0
        history.append(dict(atom=a,nonzero_before_pruning=before,
            nonzero_after_pruning=int(np.count_nonzero(np.abs(out)>1e-13))))
    return 1.-out[0,0],out,history


def project_grid(spheres,direction,n):
    """Common centroid and enclosing square; no normals-as-directions shortcut."""
    direction=np.asarray(direction,float);direction=direction/np.linalg.norm(direction)
    axis=np.array([0.,0.,1.]) if abs(direction[2])<.9 else np.array([1.,0.,0.])
    u=np.cross(axis,direction);u/=np.linalg.norm(u);v=np.cross(direction,u)
    centered=spheres.centers-spheres.centers.mean(axis=0)
    xy=np.column_stack((centered@u,centered@v))
    extent=float(np.max(np.abs(xy)+spheres.radii[:,None]))*(1.+1e-10)
    lower=np.array([-extent,-extent]);spacing=2*extent/n
    return xy,lower,spacing,u,v,direction


def grid_rays(spheres,grid,n):
    from .inside_out_pa import Source
    xy,lower,spacing,u,v,direction=grid
    coords=lower[0]+(np.arange(n)+.5)*spacing
    xx,yy=np.meshgrid(coords,coords)
    center=spheres.centers.mean(axis=0)
    radius=float(np.max(np.linalg.norm(spheres.centers-center,axis=1)+spheres.radii))+1.
    origins=center-radius*direction+xx.ravel()[:,None]*u+yy.ravel()[:,None]*v
    return Source(np.ascontiguousarray(origins),np.ascontiguousarray(np.tile(direction,(n*n,1))),
        np.full(n*n,-1,np.int64),float((n*spacing)**2))
