"""Canonical free-flight field operator between two disjoint sphere boundaries.

Geometry is compiled on a fixed outgoing boundary quadrature. New flux and
initial-direction moment coefficients may be applied without retracing it.
Unseen ray coordinates and physical scattering inside supports are separate.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from scipy.stats import qmc
from .inside_out_pa import unit_sphere,tangent_frames
from .ehss_reference import entry_distance
from .ehss_spherical_coupling import unit_directions,angular_grid,angular_grid_index


@dataclass(frozen=True)
class SphereBoundaryBasis:
    positions: np.ndarray
    directions: np.ndarray
    flux_weights: np.ndarray

    @classmethod
    def sobol(cls,power,seed):
        q=qmc.Sobol(4,scramble=True,seed=seed).random_base2(power)
        positions=unit_sphere(q[:,:2]); t,b=tangent_frames(positions)
        radius=np.sqrt(q[:,2]); phi=2*np.pi*q[:,3]
        directions=np.sqrt(1-radius*radius)[:,None]*positions+radius[:,None]*(np.cos(phi)[:,None]*t+np.sin(phi)[:,None]*b)
        directions/=np.linalg.norm(directions,axis=1)[:,None]
        weights=np.full(len(q),4*np.pi*np.pi/len(q))
        for value in [positions,directions,weights]:
            value.setflags(write=False)
        return cls(positions,directions,weights)


@njit(cache=True)
def sphere_flights(origins,directions,center,radius):
    distances=np.full(len(origins),np.inf)
    for i in range(len(origins)):
        distances[i]=entry_distance(origins[i],directions[i],center,radius)
    return distances


@dataclass
class TransferredField:
    flux: np.ndarray
    initial_moment: np.ndarray
    missed_flux: float
    missed_initial_moment: np.ndarray
    direction_centers: np.ndarray
    direction_halfangle: float
    direction_error_mass: float

    @property
    def momentum_readout(self):
        return float(np.sum(self.flux-np.sum(self.initial_moment*self.direction_centers,axis=1)))

    @property
    def direction_readout_bound(self):
        # A bound for direction binning only, on this finite quadrature.
        return float(2*np.sin(self.direction_halfangle/2)*self.direction_error_mass)


class CompiledSphereTransfer:
    def __init__(self,basis,distance_ratio,radius_ratio,grid=(8,16)):
        if not np.isfinite(distance_ratio) or not np.isfinite(radius_ratio) or radius_ratio<=0 or distance_ratio<=1+radius_ratio:
            raise ValueError("This free-flight interface requires strictly disjoint positive-radius spheres")
        unit_directions(basis.positions); unit_directions(basis.directions)
        if basis.positions.shape!=basis.directions.shape or np.any(np.sum(basis.positions*basis.directions,axis=1)<-1e-12):
            raise ValueError("Outgoing source-sphere boundary quadrature required")
        self.basis=basis; self.distance_ratio=float(distance_ratio); self.radius_ratio=float(radius_ratio); self.grid=tuple(grid)
        center=np.array([0.,0.,distance_ratio])
        distances=sphere_flights(basis.positions,basis.directions,center,radius_ratio)
        self.source_indices=np.flatnonzero(np.isfinite(distances))
        self.distances=distances[self.source_indices]
        points=basis.positions[self.source_indices]+self.distances[:,None]*basis.directions[self.source_indices]
        normal=(points-center)/radius_ratio
        normal/=np.linalg.norm(normal,axis=1)[:,None]
        cell_centers,beta=angular_grid(*self.grid); self.direction_halfangle=float(beta[0])
        self.cell_count=len(cell_centers)
        position_bins=angular_grid_index(normal,*self.grid)
        direction_bins=angular_grid_index(basis.directions[self.source_indices],*self.grid)
        destination=position_bins*self.cell_count+direction_bins
        self.destination_ids,self.edge_rows=np.unique(destination,return_inverse=True)
        self.position_centers=cell_centers[self.destination_ids//self.cell_count]
        self.direction_centers=cell_centers[self.destination_ids%self.cell_count]
        self.operator=csr_matrix((np.ones(len(self.source_indices)),(self.edge_rows,self.source_indices)),
                                  shape=(len(self.destination_ids),len(basis.positions)))
        self.miss=np.ones(len(basis.positions),dtype=bool); self.miss[self.source_indices]=False
        for value in [self.source_indices,self.distances,self.destination_ids,self.edge_rows,self.position_centers,self.direction_centers,self.miss]:
            value.setflags(write=False)

    @property
    def payload_bytes(self):
        values=[self.source_indices,self.distances,self.destination_ids,self.edge_rows,self.position_centers,self.direction_centers,self.miss,
                self.operator.data,self.operator.indices,self.operator.indptr]
        return sum(v.nbytes for v in values)

    def apply_flux(self,flux,initial_moment=None):
        flux=np.asarray(flux,dtype=float)
        if flux.shape!=(len(self.basis.positions),) or not np.isfinite(flux).all() or np.any(flux<0):
            raise ValueError("One finite nonnegative flux coefficient per source state required")
        moment=np.zeros((len(flux),3)) if initial_moment is None else np.asarray(initial_moment,dtype=float)
        if moment.shape!=(len(flux),3) or not np.isfinite(moment).all() or np.any(np.linalg.norm(moment,axis=1)>flux+1e-12*np.maximum(1.,flux)):
            raise ValueError("Initial-direction moment norm cannot exceed flux")
        # Existing packet/flux weights require no additional target cosine or area.
        return TransferredField(self.operator@flux,self.operator@moment,float(flux[self.miss].sum()),
                                moment[self.miss].sum(axis=0),self.direction_centers,self.direction_halfangle,
                                float(np.linalg.norm(moment[self.source_indices],axis=1).sum()))

    def place_hits(self,source_center,source_radius,rotation):
        center=np.asarray(source_center,dtype=float); rotation=np.asarray(rotation,dtype=float)
        if center.shape!=(3,) or not np.isfinite(center).all() or not np.isfinite(source_radius) or source_radius<=0 or rotation.shape!=(3,3) or not np.isfinite(rotation).all() or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-12,rtol=1e-12) or not np.isclose(np.linalg.det(rotation),1.,atol=1e-12,rtol=1e-12):
            raise ValueError("Finite positive scale and proper rigid placement required")
        points=self.basis.positions[self.source_indices]+self.distances[:,None]*self.basis.directions[self.source_indices]
        return dict(positions=center+source_radius*(points@rotation.T),
                    directions=self.basis.directions[self.source_indices]@rotation.T,
                    source_indices=self.source_indices,
                    target_center=center+source_radius*(np.array([0.,0.,self.distance_ratio])@rotation.T),
                    target_radius=source_radius*self.radius_ratio)
