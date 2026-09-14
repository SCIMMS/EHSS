"""Conservative sphere-pair directional reachability, before physical tracing.

A True result is a candidate, never a physical collision or visibility claim.
The source position must lie in the declared source ball. No solver integration
or sparse-update speedup is implied by this standalone geometric component.
"""
from dataclasses import dataclass
import numpy as np


def unit_directions(directions):
    value=np.asarray(directions,dtype=float)
    if value.ndim!=2 or value.shape[1]!=3 or not np.isfinite(value).all() or not np.allclose(
            np.linalg.norm(value,axis=1),1.,rtol=1e-12,atol=1e-12):
        raise ValueError("Finite unit physical directions required")
    return value


@dataclass(frozen=True)
class SpherePairCone:
    source_center: np.ndarray
    target_center: np.ndarray
    source_radius: float
    target_radius: float
    axis: np.ndarray
    cosine: float
    unrestricted: bool

    @classmethod
    def enclosing(cls,source_center,source_radius,target_center,target_radius):
        a=np.asarray(source_center,dtype=float).copy(); b=np.asarray(target_center,dtype=float).copy()
        if a.shape!=(3,) or b.shape!=(3,) or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Finite sphere centers required")
        if not np.isfinite(source_radius) or not np.isfinite(target_radius) or min(source_radius,target_radius)<=0:
            raise ValueError("Positive finite enclosing radii required")
        delta=b-a; distance=float(np.linalg.norm(delta)); radius=float(source_radius+target_radius)
        guard=128*np.finfo(float).eps*max(1.,distance,radius,float(np.max(np.abs(a))),float(np.max(np.abs(b))))
        unrestricted=distance<=radius+guard
        axis=np.array([1.,0.,0.]) if distance==0 else delta/distance
        # Inflate the angular cap; uncertain near-tangent inputs remain candidates.
        ratio=min(1.,(radius+guard)/max(distance-guard,np.finfo(float).tiny))
        cosine=-1. if unrestricted else max(-1.,float(np.sqrt(max(0.,1-ratio*ratio)))-128*np.finfo(float).eps)
        for value in [a,b,axis]:
            value.setflags(write=False)
        return cls(a,b,float(source_radius),float(target_radius),axis,cosine,unrestricted)

    def may_reach(self,directions):
        directions=unit_directions(directions)
        return np.ones(len(directions),dtype=bool) if self.unrestricted else directions@self.axis>=self.cosine

    def filter_rays(self,origins,directions):
        """Apply the pair cone only where its source-ball assumption is valid."""
        directions=unit_directions(directions); origins=np.asarray(origins,dtype=float)
        if origins.shape!=directions.shape or not np.isfinite(origins).all():
            raise ValueError("One finite origin per ray required")
        # Outside origins stay candidates, including numerical boundary uncertainty.
        inside=np.linalg.norm(origins-self.source_center,axis=1)<=self.source_radius
        return ~inside | self.may_reach(directions)

    def angular_cells(self,centers,halfangles):
        """Conservative cap intersections for certified enclosing angular cells.

        Each actual cell must be contained in its center/halfangle cap. Merely
        testing representative rays would not certify an entire cell as empty.
        """
        centers=unit_directions(centers); halfangles=np.broadcast_to(np.asarray(halfangles,dtype=float),(len(centers),))
        if not np.isfinite(halfangles).all() or np.any(halfangles<0) or np.any(halfangles>np.pi):
            raise ValueError("Angular cell halfangles must lie in [0,pi]")
        alpha=np.pi if self.unrestricted else float(np.arccos(self.cosine))
        total=alpha+halfangles
        threshold=np.cos(np.minimum(np.pi,total))-256*np.finfo(float).eps
        return (total>=np.pi)|(centers@self.axis>=threshold)


def angular_grid(n_theta,n_phi):
    """Latitude/longitude cells with a conservative enclosing cap per cell."""
    if not isinstance(n_theta,int) or not isinstance(n_phi,int) or min(n_theta,n_phi)<1:
        raise ValueError("Positive integer grid dimensions required")
    theta=(np.arange(n_theta)+.5)*np.pi/n_theta
    phi=(np.arange(n_phi)+.5)*2*np.pi/n_phi
    t,p=np.meshgrid(theta,phi,indexing="ij")
    centers=np.c_[np.sin(t.ravel())*np.cos(p.ravel()),np.sin(t.ravel())*np.sin(p.ravel()),np.cos(t.ravel())]
    # A meridian-plus-parallel path bounds the geodesic distance to any cell point.
    beta=min(np.pi,.5*np.pi/n_theta+np.pi/n_phi)
    return centers,np.full(len(centers),beta)


def angular_grid_index(directions,n_theta,n_phi):
    directions=unit_directions(directions)
    theta=np.arccos(np.clip(directions[:,2],-1.,1.))
    phi=np.mod(np.arctan2(directions[:,1],directions[:,0]),2*np.pi)
    row=np.minimum(n_theta-1,(theta*n_theta/np.pi).astype(int))
    column=np.minimum(n_phi-1,(phi*n_phi/(2*np.pi)).astype(int))
    return row*n_phi+column
