"""Arbitrary-node convex transport envelopes and canonical spherical charts.

Every plane is an outer support plane of the full finite-radius atom geometry.
Interior atoms remain physical primitives. Parent envelopes are constructed from
all child vertices, without fitting them back into a tetrahedron.
"""
from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy.spatial import ConvexHull,HalfspaceIntersection
from scipy.stats import qmc


def envelope_directions(points,level=1):
    """Body-local node hull/prism normals plus a bounded direction stencil."""
    if level not in [0,1]:
        raise ValueError("Use direction level 0 (axes) or 1 (26 directions)")
    directions=list(np.vstack([np.eye(3),-np.eye(3)]))
    if level==1:
        directions.extend(np.array([p for p in product([-1.,0.,1.],repeat=3) if any(p)]))
    unique=np.unique(points,axis=0); shifted=unique-unique.mean(axis=0)
    scale=max(float(np.linalg.norm(shifted,axis=1).max(initial=0.)),1e-300)
    _,s,vh=np.linalg.svd(shifted/scale,full_matrices=True)
    rank=int(np.count_nonzero(s>1e-11*max(1.,float(s.max(initial=0.)))))
    if rank==3:
        hull=ConvexHull(shifted/scale)
        directions.extend(hull.equations[:,:3])
    elif rank==2:
        plane=shifted@vh[:2].T/scale
        hull=ConvexHull(plane)
        directions.extend(hull.equations[:,:2]@vh[:2])
        directions.extend([vh[2],-vh[2]])
    elif rank==1:
        directions.extend(vh); directions.extend(-vh)
    directions=np.asarray(directions,dtype=float)
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    # Keeping one representative is safe: offsets are recomputed from all balls.
    _,ids=np.unique(np.round(directions,12),axis=0,return_index=True)
    return directions[np.sort(ids)],rank


@dataclass
class ConvexChart:
    center: np.ndarray
    normals: np.ndarray
    heights: np.ndarray
    vertices: np.ndarray
    triangles: np.ndarray
    triangle_owners: np.ndarray
    triangle_areas: np.ndarray
    node_ids: np.ndarray
    boundary_node_ids: np.ndarray
    interior_node_ids: np.ndarray
    envelope_active_atom_ids: np.ndarray
    node_rank: int
    volume: float
    direction_level: int

    @classmethod
    def enclosing(cls,centers,radii,center=None,extra_vertices=None,direction_level=1):
        centers=np.asarray(centers,dtype=float); radii=np.asarray(radii,dtype=float)
        if centers.ndim!=2 or centers.shape[1]!=3 or radii.shape!=(len(centers),) or not len(centers):
            raise ValueError("Nonempty finite-radius atom geometry required")
        if not np.isfinite(centers).all() or not np.isfinite(radii).all() or np.any(radii<=0):
            raise ValueError("Finite centers and strictly positive radii required")
        center=centers.mean(axis=0) if center is None else np.asarray(center,dtype=float)
        if center.shape!=(3,) or not np.isfinite(center).all():
            raise ValueError("A finite interior center is required")
        extras=np.empty((0,3)) if extra_vertices is None else np.asarray(extra_vertices,dtype=float)
        if extras.ndim!=2 or extras.shape[1]!=3 or not np.isfinite(extras).all():
            raise ValueError("Finite child vertices required")
        points=np.vstack([centers,extras])
        normals,rank=envelope_directions(points,direction_level)
        heights=np.max((centers-center)@normals.T+radii[:,None],axis=0)
        if len(extras):
            heights=np.maximum(heights,np.max((extras-center)@normals.T,axis=0))
        extent=max(float(np.max(np.linalg.norm(points-center,axis=1))),float(radii.max()))
        # Check the intended center before padding: do not silently move a bad one.
        if np.min(heights)<=1e-12*extent:
            raise ValueError("Center must be strictly inside the full-dimensional envelope")
        heights=heights+1e-10*extent
        halfspaces=np.c_[normals,-heights/extent]
        vertices=HalfspaceIntersection(halfspaces,np.zeros(3)).intersections*extent+center
        hull=ConvexHull((vertices-center)/extent)
        triangles=vertices[hull.simplices]
        areas=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/2
        centroids=triangles.mean(axis=1)
        slack=np.abs((centroids-center)@normals.T-heights)
        owners=np.argmin(slack,axis=1)
        if np.max(slack[np.arange(len(owners)),owners])>1e-7*extent:
            raise ValueError("Hull triangulation disagrees with supporting planes")
        # Distinguish relative boundary of the node-center hull from atoms
        # active in a sampled outer support plane (radii can differ).
        shifted=centers-centers.mean(axis=0)
        _,singular,basis=np.linalg.svd(shifted/extent,full_matrices=True)
        atom_rank=int(np.count_nonzero(singular>1e-11*max(1.,float(singular.max(initial=0.)))))
        if atom_rank>=2:
            coordinates=shifted@basis[:atom_rank].T/extent
            center_hull=ConvexHull(coordinates)
            residual=coordinates@center_hull.equations[:,:atom_rank].T+center_hull.equations[:,-1]
            boundary=np.any(np.abs(residual)<1e-9,axis=1)
        elif atom_rank==1:
            coordinate=shifted@basis[0]
            boundary=(np.abs(coordinate-coordinate.min())<1e-9*extent)|(np.abs(coordinate-coordinate.max())<1e-9*extent)
        else:
            boundary=np.ones(len(centers),dtype=bool)
        support_values=(centers-center)@normals.T+radii[:,None]
        active=np.any(np.abs(support_values-(heights-1e-10*extent))<=1e-9*extent,axis=1)
        return cls(center.copy(),normals,heights,vertices,triangles,owners,areas,
            np.arange(len(centers),dtype=np.int64),np.flatnonzero(boundary),np.flatnonzero(~boundary),np.flatnonzero(active),
            atom_rank,float(hull.volume*extent**3),direction_level)

    @property
    def area(self):
        return float(self.triangle_areas.sum())

    def decode(self,unit_position):
        u=np.asarray(unit_position,dtype=float)
        if u.ndim!=2 or u.shape[1]!=3 or not np.isfinite(u).all() or not np.allclose(np.linalg.norm(u,axis=1),1.,rtol=1e-12,atol=1e-12):
            raise ValueError("Unit sphere position coordinates required")
        denominator=u@self.normals.T
        distances=np.full_like(denominator,np.inf)
        np.divide(self.heights[None,:],denominator,out=distances,where=denominator>0)
        face=np.argmin(distances,axis=1); rho=distances[np.arange(len(u)),face]
        cosine=denominator[np.arange(len(u)),face]
        return self.center+rho[:,None]*u,face,rho*rho/cosine

    def encode(self,points):
        points=np.asarray(points,dtype=float)
        delta=points-self.center; lengths=np.linalg.norm(delta,axis=1)
        if np.any(lengths<=0):
            raise ValueError("Interior center is not a boundary point")
        u=delta/lengths[:,None]; recovered,face,jac=self.decode(u)
        if not np.allclose(points,recovered,atol=1e-9*np.max(self.heights),rtol=1e-12):
            raise ValueError("Points must lie on the convex boundary")
        return u,face,jac

    def pulled_back_outgoing_flux(self,u,directions,radiance=1.):
        _,face,jac=self.decode(u); directions=np.asarray(directions,dtype=float)
        if directions.shape!=u.shape:
            raise ValueError("One physical direction per chart position required")
        return np.asarray(radiance)*np.maximum(0.,np.sum(self.normals[face]*directions,axis=1))*jac

    def surface_quadrature(self,power,seed):
        """Exactly 2**power states, area-stratified over boundary triangles.

        Returns canonical solid-angle weights: use J for physical surface flux.
        Each triangle receives at least one point and its exact area weight.
        """
        total=2**power; count=len(self.triangles)
        if total<count:
            raise ValueError("At least one sample per boundary triangle required")
        shares=(total-count)*self.triangle_areas/self.area
        allocation=np.ones(count,dtype=int)+np.floor(shares).astype(int)
        remaining=total-int(allocation.sum())
        allocation[np.argsort(-(shares-np.floor(shares)),kind="stable")[:remaining]]+=1
        positions=[]; physical_weights=[]
        for i,(triangle,n) in enumerate(zip(self.triangles,allocation)):
            uv=qmc.Sobol(2,scramble=True,seed=seed+100003*i).random_base2(int(np.ceil(np.log2(n))))[:n]
            root=np.sqrt(uv[:,0]); a,b,c=triangle
            positions.append((1-root)[:,None]*a+(root*(1-uv[:,1]))[:,None]*b+(root*uv[:,1])[:,None]*c)
            physical_weights.append(np.full(n,self.triangle_areas[i]/n))
        x=np.vstack(positions); u,face,jac=self.encode(x)
        return u,np.concatenate(physical_weights)/jac,face


def update_convex_ports(tree):
    """Persistent body-local leaves, general convex parents, packed live planes."""
    count=len(tree.parent); was_built=hasattr(tree,"leaf_charts")
    if not was_built:
        tree.leaf_charts=[]
        for group in range(tree.n_groups):
            atoms=tree.ids[tree.offsets[group]:tree.offsets[group+1]]
            tree.leaf_charts.append(ConvexChart.enclosing(tree.local[atoms],tree.radii[atoms],direction_level=tree.convex_direction_level))
    tree.node_charts=[None]*count; tree.port_vertices=[None]*count
    tree.port_centers=np.empty((count,3)); rotations=np.empty((count,3,3)); translations=np.empty((count,3))
    normals=[None]*count; heights=[None]*count
    atom_leaf=tree.leaf_node[tree.groups]
    for node in range(count-1,-1,-1):
        owned=(atom_leaf>=node)&(atom_leaf<tree.end[node])
        group=int(tree.groups[np.flatnonzero(owned)[0]])
        rotation=tree.rotations[group]; translation=tree.translations[group]
        if tree.leaf_group[node]>=0:
            chart=tree.leaf_charts[group]
        else:
            children=[tree.left[node],tree.right[node]]
            vertices=np.vstack([tree.port_vertices[c] for c in children])
            local_vertices=(vertices-translation)@rotation
            local_atoms=(tree.world[owned]-translation)@rotation
            chart=ConvexChart.enclosing(local_atoms,tree.radii[owned],extra_vertices=local_vertices,direction_level=tree.convex_direction_level)
        tree.node_charts[node]=chart
        tree.port_centers[node]=chart.center@rotation.T+translation
        normals[node]=chart.normals@rotation.T; heights[node]=chart.heights
        tree.port_vertices[node]=chart.vertices@rotation.T+translation
        rotations[node]=rotation; translations[node]=translation
        tree.lo[node]=tree.port_vertices[node].min(axis=0); tree.hi[node]=tree.port_vertices[node].max(axis=0)
    tree.port_rotations=rotations; tree.port_translations=translations
    tree.port_offsets=np.r_[0,np.cumsum([len(n) for n in normals])].astype(np.int64)
    tree.port_normals=np.vstack(normals); tree.port_heights=np.concatenate(heights)
    tree.exclusive=np.ones(count,dtype=np.bool_)
    for node in range(count):
        foreign=(atom_leaf<node)|(atom_leaf>=tree.end[node])
        signed=(tree.world-tree.port_centers[node])@normals[node].T-tree.radii[:,None]
        possible=np.all(signed<=heights[node]+1e-12,axis=1)
        tree.exclusive[node]=not np.any(foreign&possible)
    return dict(port_kind="convex",leaf_charts_rebuilt=0 if was_built else tree.n_groups,
        internal_charts_built=int(np.count_nonzero(tree.leaf_group<0)),ghost_coordinates=0,
        primitive_count=len(tree.radii),plane_count=len(tree.port_heights),
        total_vertices=sum(len(v) for v in tree.port_vertices))
