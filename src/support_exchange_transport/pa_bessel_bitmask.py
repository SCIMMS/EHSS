"""Sphere-volume Fourier slices -> projected thickness -> positional bitmask.

The complex translation factor shifts a REAL occupancy density. No optical
amplitudes are squared. Finite-band ringing makes thresholding approximate.
"""
import numpy as np
from scipy.special import spherical_jn
from scipy.fft import ifft2
from numba import njit
from .pa_dyadic_bitmask import row_bounds,row_span


def sphere_volume_form(q,radius):
    z=np.asarray(q)*radius
    result=np.empty_like(z,dtype=float)
    small=np.abs(z)<1e-4
    # j1(z)/z = 1/3-z^2/30+z^4/840+...
    result[small]=4*np.pi*radius**3*(1/3-z[small]**2/30+z[small]**4/840)
    result[~small]=4*np.pi*radius**3*spherical_jn(1,z[~small])/z[~small]
    return result


class BesselSlicePlan:
    def __init__(self,radii,n,extent):
        if n<8 or n%2 or not np.isfinite(extent) or extent<=0:raise ValueError('Positive extent and even n >=8 required')
        self.n=n;self.extent=float(extent);self.spacing=2*extent/n
        self.lower=np.array([-extent,-extent]);self.origin=-extent+self.spacing/2
        self.k=2*np.pi*np.fft.fftfreq(n,d=self.spacing)
        q=np.hypot(self.k[:,None],self.k[None,:])
        values=np.asarray(radii,float)
        self.groups=[np.flatnonzero(values==r) for r in np.unique(values)]
        self.forms=[sphere_volume_form(q,float(r)) for r in np.unique(values)]

    def coefficients(self,xy):
        spectrum=np.zeros((self.n,self.n),np.complex128)
        for ids,form in zip(self.groups,self.forms):
            # A separable phase sum on a Cartesian Fourier slice; native BLAS
            # does the contraction. Exact grouping by radius, no radius bins.
            px=np.exp(-1j*(xy[ids,0]-self.origin)[:,None]*self.k[None,:])
            py=np.exp(-1j*(xy[ids,1]-self.origin)[:,None]*self.k[None,:])
            spectrum+=(py.T@px)*form
        # Omit the ambiguous unpaired Nyquist modes. Remaining +/- frequencies
        # are conjugate pairs, so the inverse is a real trigonometric polynomial.
        spectrum[self.n//2,:]=0.;spectrum[:,self.n//2]=0.
        return spectrum

    def inverse(self,spectrum):
        return ifft2(spectrum,workers=1).real/self.spacing**2


class NufftSlicePlan(BesselSlicePlan):
    """Optional FINUFFT type-1 replacement for the identical translation sum."""
    def __init__(self,radii,n,extent,eps=1e-12):
        import finufft
        super().__init__(radii,n,extent)
        self.plan=finufft.Plan(1,(n,n),eps=eps,isign=-1,nthreads=1,modeord=1)
        self.strengths=[np.ones(len(ids),np.complex128) for ids in self.groups]

    def coefficients(self,xy):
        spectrum=np.zeros((self.n,self.n),np.complex128)
        for ids,form,strength in zip(self.groups,self.forms,self.strengths):
            point=(xy[ids]-self.origin)*(np.pi/self.extent)
            point=(point+np.pi)%(2*np.pi)-np.pi
            # Output axis 0 is ky (array rows), axis 1 is kx (array columns).
            self.plan.setpts(np.ascontiguousarray(point[:,1]),np.ascontiguousarray(point[:,0]))
            spectrum+=self.plan.execute(strength)*form
        spectrum[self.n//2,:]=0.;spectrum[:,self.n//2]=0.
        return spectrum


@njit(cache=True)
def threshold_bits(field,threshold):
    ny,nx=field.shape;words=np.zeros((ny,(nx+63)//64),np.uint64)
    for y in range(ny):
        for x in range(nx):
            if field[y,x]>threshold:words[y,x//64]|=np.uint64(1)<<np.uint64(x%64)
    return words


@njit(cache=True)
def direct_thickness(xy,radii,lower,spacing,n):
    field=np.zeros((n,n))
    for j in range(len(radii)):
        first,last=row_bounds(xy[j,1],radii[j],lower,spacing,n)
        for y in range(first,last):
            low,high=row_span(xy[j,0],xy[j,1],radii[j],lower,spacing,y,n)
            dy=lower[1]+(y+.5)*spacing-xy[j,1]
            for x in range(low,high):
                dx=lower[0]+(x+.5)*spacing-xy[j,0]
                field[y,x]+=2*np.sqrt(max(0.,radii[j]**2-dx*dx-dy*dy))
    return field
