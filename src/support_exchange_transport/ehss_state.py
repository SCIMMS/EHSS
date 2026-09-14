"""Weighted hard-sphere scattering states and initial/final direction readout."""
from dataclasses import dataclass
import numpy as np
from scipy.stats import qmc
from .inside_out_pa import sample_source,inside_out,unit_sphere,tangent_frames


@dataclass
class EHSSSource:
    origins: np.ndarray
    directions: np.ndarray
    incoming: np.ndarray
    weights: np.ndarray
    last_atom: np.ndarray
    initial_bounces: np.ndarray
    convention: str

    def __post_init__(self):
        n=len(self.weights)
        for name in ["origins","directions","incoming"]:
            value=np.ascontiguousarray(getattr(self,name),dtype=float)
            if value.shape!=(n,3) or not np.isfinite(value).all():
                raise ValueError("Finite (N,3) positions and directions required")
            setattr(self,name,value)
        for name in ["directions","incoming"]:
            if not np.allclose(np.linalg.norm(getattr(self,name),axis=1),1.,atol=1e-12,rtol=1e-12):
                raise ValueError("Unit physical directions required")
        self.weights=np.ascontiguousarray(self.weights,dtype=float)
        if self.weights.shape!=(n,) or not np.isfinite(self.weights).all() or np.any(self.weights<0):
            raise ValueError("Nonnegative finite source weights required")
        for name in ["last_atom","initial_bounces"]:
            value=np.ascontiguousarray(getattr(self,name),dtype=np.int64)
            if value.shape!=(n,):
                raise ValueError("One initial identity and bounce count per state required")
            setattr(self,name,value)
        if np.any(self.initial_bounces<0):
            raise ValueError("Nonnegative initial bounce count required")
        if np.any((self.initial_bounces==1)&(self.last_atom<0)):
            raise ValueError("First-reflected source states require their collider identity")


def external_source(spheres,power,seed):
    """Independent isotropic incidence times a uniform enclosing impact disk."""
    u=qmc.Sobol(4,scramble=True,seed=seed).random_base2(power)
    directions=unit_sphere(u[:,:2]); t,b=tangent_frames(directions)
    center=spheres.centers.mean(axis=0)
    radius=float(np.max(np.linalg.norm(spheres.centers-center,axis=1)+spheres.radii))*(1.+1e-9)
    rho=radius*np.sqrt(u[:,2]); phi=2*np.pi*u[:,3]
    origins=center-radius*directions+rho[:,None]*(np.cos(phi)[:,None]*t+np.sin(phi)[:,None]*b)
    n=len(u)
    return EHSSSource(origins,directions,directions.copy(),np.full(n,np.pi*radius*radius/n),
        np.full(n,-1),np.zeros(n,dtype=np.int64),"external impact disk")


def reverse_pa_source(spheres,power,seed):
    """Reverse PA escape states; preserve pre-filter sample weight; reflect once."""
    source=sample_source(spheres,power,seed)
    pa,blocked,_=inside_out(spheres,source)
    keep=~blocked; owners=source.owners[keep]
    origins=source.origins[keep]; incoming=-source.directions[keep]
    normals=(origins-spheres.centers[owners])/spheres.radii[owners,None]
    normals/=np.linalg.norm(normals,axis=1)[:,None]
    outgoing=incoming-2*np.sum(incoming*normals,axis=1)[:,None]*normals
    outgoing/=np.linalg.norm(outgoing,axis=1)[:,None]
    result=EHSSSource(origins,outgoing,incoming,np.full(len(owners),source.area/(4*len(source.owners))),
        owners,np.ones(len(owners),dtype=np.int64),"PA reverse, first reflection already applied")
    if not np.isclose(result.weights.sum(),pa,rtol=1e-13,atol=1e-13):
        raise AssertionError("PA source measure changed during filtering")
    return result


@dataclass
class EHSSResult:
    positions: np.ndarray
    outgoing: np.ndarray
    escaped: np.ndarray
    unresolved: np.ndarray
    bounces: np.ndarray
    collider_ids: np.ndarray
    source: EHSSSource
    metrics: dict

    @property
    def contributions(self):
        # Numerical clamp only at the analytic [0,2] bounds.
        angle=np.clip(1-np.sum(self.source.incoming*self.outgoing,axis=1),0.,2.)
        return self.source.weights*angle*self.escaped

    @property
    def omega(self):
        return float(self.contributions.sum())

    @property
    def residual_mass(self):
        return float(self.source.weights[self.unresolved].sum())

    @property
    def tail_bound(self):
        return 2*self.residual_mass

    @property
    def pa(self):
        return float(self.source.weights[(self.bounces>0)|self.unresolved].sum())

    def order_ledger(self):
        return [dict(order=k,count=int(np.count_nonzero(self.escaped&(self.bounces==k))),
            mass=float(self.source.weights[self.escaped&(self.bounces==k)].sum()),
            omega=float(self.contributions[self.bounces==k].sum())) for k in range(int(self.bounces.max(initial=0))+1)]
