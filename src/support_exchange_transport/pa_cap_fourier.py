"""Analytic cap coefficients and a Fourier-azimuth transport pilot.

SH cap coefficients are independently available. The transport implementation
uses continuous Fourier modes in azimuth and numerical quadrature in mu, NOT
full spherical-harmonic/Gaunt composition. No atom-by-ray mask is built.
"""
import numpy as np
from time import perf_counter
from scipy.special import eval_legendre, sph_harm_y
from scipy.fft import fft, ifft, next_fast_len
from numba import njit
from .inside_out_pa import tangent_frames


def cap_sh_coefficients(axis,cosine,degree):
    axis=np.asarray(axis,float); axis=axis/np.linalg.norm(axis)
    theta=np.arccos(np.clip(axis[2],-1,1)); phi=np.arctan2(axis[1],axis[0])
    values=[]; modes=[]
    for ell in range(degree+1):
        integral=1-cosine if ell==0 else (eval_legendre(ell-1,cosine)-eval_legendre(ell+1,cosine))/(2*ell+1)
        for m in range(-ell,ell+1):
            modes.append((ell,m))
            values.append(2*np.pi*integral*np.conj(sph_harm_y(ell,m,theta,phi)))
    return np.array(modes),np.array(values)


def local_caps(spheres,point,inward):
    a,b=tangent_frames(np.asarray(inward)[None,:]); a=a[0]; b=b[0]
    rel=spheres.centers-point; distance=np.linalg.norm(rel,axis=1)
    if np.any(distance<=spheres.radii): raise ValueError('Boundary point must be outside every atom')
    axis=rel/distance[:,None]
    local=np.column_stack((axis@a,axis@b,axis@inward))
    cosine=np.sqrt(np.maximum(0,1-(spheres.radii/distance)**2))
    return local,cosine


def ring_arcs(local,cosine,mu):
    phi=np.arctan2(local[:,1],local[:,0])
    amplitude=np.sqrt(np.maximum(0,1-np.asarray(mu)**2))[:,None]*np.hypot(local[:,0],local[:,1])[None,:]
    rhs=cosine[None,:]-np.asarray(mu)[:,None]*local[:,2][None,:]
    ratio=np.divide(rhs,amplitude,out=np.zeros_like(rhs),where=amplitude>1e-15)
    alpha=np.arccos(np.clip(ratio,-1,1))
    alpha=np.where(amplitude>1e-15,alpha,np.where(rhs<=0,np.pi,0.))
    return phi,alpha


@njit(cache=True)
def exact_ring_union(phi,alpha):
    result=np.zeros(len(alpha))
    for row in range(len(alpha)):
        starts=np.empty(2*len(phi)); ends=np.empty(2*len(phi)); count=0; full=False
        for j in range(len(phi)):
            half=alpha[row,j]
            if half>=np.pi: full=True; break
            if half<=0: continue
            low=(phi[j]-half)%(2*np.pi); high=low+2*half
            starts[count]=low; ends[count]=min(high,2*np.pi); count+=1
            if high>2*np.pi:
                starts[count]=0.; ends[count]=high-2*np.pi; count+=1
        if full: result[row]=1.; continue
        if count==0: continue
        order=np.argsort(starts[:count]); low=starts[order[0]]; high=ends[order[0]]; covered=0.
        for k in range(1,count):
            j=order[k]
            if starts[j]>high:
                covered+=high-low; low=starts[j]; high=ends[j]
            else: high=max(high,ends[j])
        result[row]=(covered+high-low)/(2*np.pi)
    return result


def arc_coefficients(phi,alpha,order):
    """h_m=(1/2pi) integral h(phi) exp(-im phi) dphi, m=-M..M."""
    m=np.arange(-order,order+1)
    coeff=alpha[:,None]/np.pi*np.sinc(alpha[:,None]*m[None,:]/np.pi)
    return coeff*np.exp(-1j*phi[:,None]*m[None,:])


def projected_products(left,right,order):
    """Zero-padded linear coefficient convolution, retain -M..M; no aliasing."""
    length=next_fast_len(4*order+1)
    product=ifft(fft(left,length,axis=-1,workers=1)*fft(right,length,axis=-1,workers=1),axis=-1,workers=1)
    return product[...,order:3*order+1]


def transmission_dc(coefficients,order,fejer=False):
    """Balanced projected product of transmission coefficients, final DC only."""
    count=len(coefficients)
    if count==0: return 1.,0
    work=coefficients.copy(); filt=1-np.abs(np.arange(-order,order+1))/(order+1)
    if fejer: work*=filt
    merges=0
    while len(work)>2:
        pairs=len(work)//2
        product=projected_products(work[:2*pairs:2],work[1:2*pairs:2],order)
        if fejer: product*=filt
        if len(work)%2: product=np.concatenate((product,work[-1:]),axis=0)
        work=product; merges+=pairs
    if len(work)==1: return float(work[0,order].real),merges
    # The final constant mode is the cross-mode inner product; no inverse field.
    return float(np.sum(work[0]*work[1,::-1]).real),merges+1


def fourier_ring_union(phi,alpha,order,fejer=False):
    result=np.zeros(len(alpha)); atoms=0; merges=0; coefficient_s=0.; composition_s=0.
    for row,half in enumerate(alpha):
        if np.any(half>=np.pi): result[row]=1.; continue
        active=half>0
        if not np.any(active): continue
        t=perf_counter(); h=arc_coefficients(phi[active],half[active],order)
        transmission=-h; transmission[:,order]+=1
        coefficient_s+=perf_counter()-t; t=perf_counter()
        dc,nmerge=transmission_dc(transmission,order,fejer)
        composition_s+=perf_counter()-t
        result[row]=1-dc; atoms+=int(np.count_nonzero(active)); merges+=nmerge
    return result,dict(active_cap_rows=atoms,merges=merges,coefficient_s=coefficient_s,composition_s=composition_s)


@njit(cache=True)
def native_fourier_union(phi,alpha,order):
    """Same balanced projected products with native direct convolution.

Intended for low M; O(M^2) arithmetic avoids many short FFT dispatches.
No positivity clipping, averaging shortcut or change of merge order.
"""
    result=np.zeros(len(alpha)); width=2*order+1
    for row in range(len(alpha)):
        count=0; full=False
        for j in range(len(phi)):
            if alpha[row,j]>=np.pi: full=True; break
            if alpha[row,j]>0: count+=1
        if full: result[row]=1.; continue
        if count==0: continue
        work=np.zeros((count,width),np.complex128); index=0
        for j in range(len(phi)):
            half=alpha[row,j]
            if half<=0: continue
            work[index,order]=1-half/np.pi
            for m in range(1,order+1):
                scale=-np.sin(m*half)/(np.pi*m)
                value=scale*(np.cos(m*phi[j])-1j*np.sin(m*phi[j]))
                work[index,order+m]=value;work[index,order-m]=np.conj(value)
            index+=1
        while count>2:
            pairs=count//2; nxt=np.zeros((pairs+count%2,width),np.complex128)
            for p in range(pairs):
                for k in range(width):
                    total=0j; target=k+order
                    for j in range(max(0,target-width+1),min(width,target+1)):
                        total+=work[2*p,j]*work[2*p+1,target-j]
                    nxt[p,k]=total
            if count%2:
                for k in range(width):nxt[pairs,k]=work[count-1,k]
            work=nxt;count=len(work)
        dc=work[0,order]
        if count==2:
            dc=0j
            for k in range(width):dc+=work[0,k]*work[1,width-1-k]
        result[row]=1-dc.real
    return result
