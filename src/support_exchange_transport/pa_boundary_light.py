"""Opaque equal-radiance atom spheres read at an enclosing spherical boundary.

Boundary point x is uniform in area. Reverse rays cover the inward hemisphere;
cosine sampling has constant PA weights, uniform-angle sampling needs 2*cos.
This is a sampling/readout change, not a compressed transport implementation.
"""
import numpy as np
from scipy.stats import qmc
from .inside_out_pa import Source, unit_sphere, tangent_frames


def enclosing_sphere(spheres, scale=1.):
    if not np.isfinite(scale) or scale < 1: raise ValueError('scale must be >= 1')
    center=spheres.centers.mean(axis=0)
    radius=np.max(np.linalg.norm(spheres.centers-center,axis=1)+spheres.radii)*(1+1e-9)*scale
    return center,float(radius)


def boundary_source(spheres,power,seed,angular='cosine',scale=1.):
    u=qmc.Sobol(4,scramble=True,seed=seed).random_base2(power)
    normal=unit_sphere(u[:,:2]); a,b=tangent_frames(normal)
    if angular=='cosine': mu=np.sqrt(u[:,2])
    elif angular=='uniform': mu=u[:,2]
    else: raise ValueError('angular must be cosine or uniform')
    rho=np.sqrt(np.maximum(0,1-mu*mu)); phi=2*np.pi*u[:,3]
    outward=mu[:,None]*normal+rho[:,None]*(np.cos(phi)[:,None]*a+np.sin(phi)[:,None]*b)
    center,radius=enclosing_sphere(spheres,scale)
    source=Source(np.ascontiguousarray(center+radius*normal),np.ascontiguousarray(-outward),
                  np.full(len(u),-1,np.int64),np.pi*radius*radius)
    weights=np.ones(len(u)) if angular=='cosine' else 2*mu
    return source,weights


def boundary_pa(source,hits,weights):
    return source.area*float(np.mean(weights*hits))


def paired_deficit(source,hits,weights):
    # Same sampled paths for empty and transmitted flux, normalized by 4*pi*L0.
    empty=source.area*float(np.mean(weights))
    transmitted=source.area*float(np.mean(weights*(~hits)))
    return empty-transmitted


def forward_exit(spheres,source,escaped,scale=1.):
    """Map independently sampled escaping atomic emission rays onto the shell."""
    center,radius=enclosing_sphere(spheres,scale)
    o=source.origins[escaped]; d=source.directions[escaped]; x=o-center
    b=np.einsum('ij,ij->i',x,d)
    disc=b*b-np.einsum('ij,ij->i',x,x)+radius*radius
    t=-b+np.sqrt(np.maximum(0,disc))
    return o+t[:,None]*d,d
