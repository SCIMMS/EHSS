"""Well-centered node/ghost charts and conservative tetrahedral transport ports.

Raw anchor vertices define a canonical circumsphere. A homothetic tetrahedron
encloses finite-radius atoms/child ports; the two geometric roles are distinct.
Ghost coordinates are never appended to physical primitive arrays.
"""
from dataclasses import dataclass
from itertools import combinations
import numpy as np
from numba import njit
from scipy.stats import qmc


def complete_anchors(points,scale=1.):
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or not 1<=len(points)<=4 or not np.isfinite(points).all():
        raise ValueError("One to four finite real anchors required")
    if not np.isfinite(scale) or scale<=0:
        raise ValueError("Positive anchor scale required")
    k=len(points); a=points[0]
    if k<=2:
        length=scale if k==1 else np.linalg.norm(points[1]-a)
        if length<=np.finfo(float).eps*scale:
            raise ValueError("Distinct edge anchors required")
        e1=np.array([1.,0.,0.]) if k==1 else (points[1]-a)/length
        components=np.abs(e1)
        helper=np.eye(3)[np.flatnonzero(components<=components.min()+1e-10)[0]]
        e2=np.cross(e1,helper); e2/=np.linalg.norm(e2); e3=np.cross(e1,e2)
        vertices=np.array([a,a+length*e1,
            a+length*(.5*e1+np.sqrt(3)/2*e2),
            a+length*(.5*e1+np.sqrt(3)/6*e2+np.sqrt(2/3)*e3)])
        vertices[:k]=points
        center=vertices.mean(axis=0)
    elif k==3:
        b,c=points[1:]; ab=b-a; ac=c-a; cross=np.cross(ab,ac)
        diameter=max(np.linalg.norm(ab),np.linalg.norm(ac),np.linalg.norm(c-b))
        if np.linalg.norm(cross)<=1e-12*diameter**2:
            raise ValueError("Collinear or numerically degenerate triple")
        n=cross/np.linalg.norm(cross)
        offset=np.linalg.solve(np.stack([2*ab,2*ac,n]),np.array([ab@ab,ac@ac,0.]))
        circum=a+offset; rho=np.linalg.norm(offset)
        center=circum+rho*n; radius=np.sqrt(2.)*rho
        inside=points.mean(axis=0); delta=center-inside
        ghost=center+radius*delta/np.linalg.norm(delta)
        vertices=np.vstack([points,ghost])
    else:
        delta=points[1:]-a
        if np.linalg.matrix_rank(delta)<3:
            raise ValueError("Coplanar four-anchor geometry")
        center=a+np.linalg.solve(2*delta,np.sum(delta*delta,axis=1))
        vertices=points.copy()
    radius=float(np.linalg.norm(vertices[0]-center))
    bary=np.linalg.solve(np.vstack([(vertices-center).T/radius,np.ones(4)]),np.array([0.,0.,0.,1.]))
    if np.min(bary)<=1e-10 or not np.isfinite(bary).all():
        raise ValueError("Circumcenter not strictly inside a well-conditioned tetrahedron")
    if not np.allclose(np.linalg.norm(vertices-center,axis=1),radius,rtol=1e-9,atol=1e-12*scale):
        raise ValueError("Inconsistent circumsphere")
    normals=[]; heights=[]
    for excluded in range(4):
        face=vertices[np.arange(4)!=excluded]
        normal=np.cross(face[1]-face[0],face[2]-face[0]); normal/=np.linalg.norm(normal)
        if normal@(face[0]-center)<0:
            normal=-normal
        normals.append(normal); heights.append(normal@(face[0]-center))
    return vertices,center,radius,bary,np.asarray(normals),np.asarray(heights)


def select_anchors(points,scale):
    points=np.asarray(points,dtype=float)
    _,unique=np.unique(points,axis=0,return_index=True); unique.sort()
    if len(unique)<=4:
        selected=list(unique)
    else:
        selected=[int(unique[0])]
        selected.append(int(unique[np.argmax(np.linalg.norm(points[unique]-points[selected[0]],axis=1))]))
        axis=points[selected[1]]-points[selected[0]]
        selected.append(int(unique[np.argmax(np.linalg.norm(np.cross(points[unique]-points[selected[0]],axis),axis=1))]))
        normal=np.cross(points[selected[1]]-points[selected[0]],points[selected[2]]-points[selected[0]])
        selected.append(int(unique[np.argmax(np.abs((points[unique]-points[selected[0]])@normal))]))
        selected=list(dict.fromkeys(selected))
    for count in range(len(selected),0,-1):
        candidates=[]
        for ids in combinations(selected,count):
            try:
                data=complete_anchors(points[list(ids)],scale)
                # Avoid a practically enormous canonical sphere if a smaller
                # subset can express the same physical support as a chart.
                if data[2]>1e4*max(scale,float(np.ptp(points,axis=0).max())):
                    continue
                candidates.append((data[2],ids,data))
            except (ValueError,np.linalg.LinAlgError):
                continue
        if candidates:
            _,ids,data=min(candidates,key=lambda x:x[0])
            return np.asarray(ids,dtype=np.int64),data
    raise ValueError("No finite well-centered anchor chart")


@dataclass
class TetraChart:
    raw_vertices: np.ndarray
    center: np.ndarray
    radius: float
    barycentric: np.ndarray
    normals: np.ndarray
    heights: np.ndarray
    vertices: np.ndarray
    real_anchor_ids: np.ndarray
    ghost_mask: np.ndarray
    enclosure_scale: float

    @classmethod
    def enclosing(cls,anchor_candidates,centers,radii,scale=1.):
        centers=np.asarray(centers,dtype=float); radii=np.asarray(radii,dtype=float)
        if centers.ndim!=2 or centers.shape[1]!=3 or radii.shape!=(len(centers),) or not len(centers):
            raise ValueError("Nonempty balls/vertices required for port enclosure")
        if not np.isfinite(centers).all() or not np.isfinite(radii).all() or np.any(radii<0):
            raise ValueError("Finite centers and nonnegative enclosure radii required")
        ids,data=select_anchors(anchor_candidates,scale)
        vertices,center,radius,bary,normals,heights=data
        factors=((centers-center)@normals.T+radii[:,None])/heights
        dilation=max(1.,float(factors.max()))*(1.+1e-10)
        domain_vertices=center+dilation*(vertices-center)
        return cls(vertices,center,radius,bary,normals,heights*dilation,domain_vertices,
            ids,np.arange(4)>=len(ids),dilation)

    def decode(self,unit_position):
        u=np.asarray(unit_position,dtype=float)
        if u.ndim!=2 or u.shape[1]!=3 or not np.isfinite(u).all() or not np.allclose(np.linalg.norm(u,axis=1),1.,atol=1e-12,rtol=1e-12):
            raise ValueError("Unit sphere chart coordinates required")
        denominator=u@self.normals.T
        distances=np.full_like(denominator,np.inf)
        np.divide(self.heights[None,:],denominator,out=distances,where=denominator>0)
        face=np.argmin(distances,axis=1)
        distance=distances[np.arange(len(u)),face]
        points=self.center+distance[:,None]*u
        cosine=denominator[np.arange(len(u)),face]
        jacobian=distance**2/(cosine*self.radius**2)
        return points,face,jacobian

    def encode(self,points):
        points=np.asarray(points,dtype=float)
        delta=points-self.center; lengths=np.linalg.norm(delta,axis=1)
        if np.any(lengths<=0):
            raise ValueError("Chart center is not a boundary point")
        u=delta/lengths[:,None]
        recovered,face,jacobian=self.decode(u)
        if not np.allclose(points,recovered,atol=1e-9*np.max(self.heights),rtol=1e-12):
            raise ValueError("Points must lie on the tetrahedral port boundary")
        return u,face,jacobian

    def pulled_back_outgoing_flux(self,u,directions,radiance=1.):
        _,face,jac=self.decode(u)
        d=np.asarray(directions,dtype=float)
        if d.shape!=u.shape:
            raise ValueError("One physical direction for each chart position required")
        return np.asarray(radiance)*np.maximum(0.,np.sum(self.normals[face]*d,axis=1))*jac

    def face_stratified_quadrature(self,power,seed):
        """Canonical sphere points with weights pulled back from four faces.

        There are 4*2**power states. These weights integrate canonical dA;
        multiply by the physical-density Jacobian when evaluating physical flux.
        Position-chart importance sampling does not alter physical directions.
        """
        positions=[]; physical_weights=[]
        for face in range(4):
            a,b,c=self.vertices[np.arange(4)!=face]
            area=np.linalg.norm(np.cross(b-a,c-a))/2.
            uv=qmc.Sobol(2,scramble=True,seed=seed+face*100003).random_base2(power)
            root=np.sqrt(uv[:,0])
            points=(1-root)[:,None]*a+(root*(1-uv[:,1]))[:,None]*b+(root*uv[:,1])[:,None]*c
            positions.append(points); physical_weights.append(np.full(len(points),area/len(points)))
        points=np.vstack(positions)
        u,face,jac=self.encode(points)
        weights=np.concatenate(physical_weights)/jac
        return u,weights,face


@njit(cache=True)
def convex_interval(o,d,center,normals,heights):
    near=0.; far=np.inf
    x0=o[0]-center[0]; x1=o[1]-center[1]; x2=o[2]-center[2]
    for face in range(len(heights)):
        slack=heights[face]-(normals[face,0]*x0+normals[face,1]*x1+normals[face,2]*x2)
        slope=normals[face,0]*d[0]+normals[face,1]*d[1]+normals[face,2]*d[2]
        if slope==0.:
            if slack<0.:
                return np.inf,-np.inf
        elif slope>0.:
            far=min(far,slack/slope)
        else:
            near=max(near,slack/slope)
    if far<near:
        return np.inf,-np.inf
    return near,far


@njit(cache=True)
def chart_boundary_roundtrip(point,center,normals,heights):
    delta=point-center; length=np.sqrt(np.dot(delta,delta))
    unit=delta/length
    distance=np.inf; owner=-1
    for face in range(len(heights)):
        denominator=normals[face,0]*unit[0]+normals[face,1]*unit[1]+normals[face,2]*unit[2]
        if denominator>0. and heights[face]/denominator<distance:
            distance=heights[face]/denominator; owner=face
    reconstructed=center+distance*unit
    error=np.sqrt(np.dot(reconstructed-point,reconstructed-point))
    return reconstructed,owner,error


def project_physical_events(atom_ids,weights,groups):
    """Disjoint event/surface ownership, not a disjoint volume partition.

    Each physical collision is assigned once. Input uses the physical-atom
    namespace only; -1 denotes a no-collision sample retained separately.
    """
    atom_ids=np.asarray(atom_ids,dtype=np.int64); weights=np.asarray(weights,dtype=float)
    groups=np.asarray(groups,dtype=np.int64)
    if atom_ids.shape!=weights.shape or np.any(atom_ids< -1) or np.any(atom_ids>=len(groups)):
        raise ValueError("Physical atom IDs or -1 only; ghost IDs cannot scatter")
    if not np.isfinite(weights).all() or np.any(weights<0) or np.any(groups<0):
        raise ValueError("Finite nonnegative weights and physical owners required")
    active=atom_ids>=0
    mass=np.bincount(groups[atom_ids[active]],weights=weights[active],minlength=int(groups.max(initial=-1))+1)
    return mass,float(weights[~active].sum())


def update_tree_ports(tree):
    """Body-attached leaf charts, nested parent ports, conservative interference."""
    count=len(tree.parent)
    if not hasattr(tree,"leaf_charts"):
        tree.leaf_charts=[]
        for group in range(tree.n_groups):
            atoms=tree.ids[tree.offsets[group]:tree.offsets[group+1]]
            tree.leaf_charts.append(TetraChart.enclosing(tree.local[atoms],tree.local[atoms],tree.radii[atoms],
                scale=float(tree.radii[atoms].max())))
    tree.node_charts=[None]*count
    tree.port_centers=np.empty((count,3)); tree.port_normals=np.empty((count,4,3)); tree.port_heights=np.empty((count,4))
    tree.port_vertices=np.empty((count,4,3)); tree.port_rotations=np.empty((count,3,3))
    tree.port_translations=np.empty((count,3))
    atom_leaf=tree.leaf_node[tree.groups]
    for node in range(count-1,-1,-1):
        group=int(tree.leaf_group[node])
        owned=(atom_leaf>=node)&(atom_leaf<tree.end[node])
        if group<0:
            group=int(tree.groups[np.flatnonzero(owned)[0]])
        rotation=tree.rotations[group]; translation=tree.translations[group]
        if tree.leaf_group[node]>=0:
            chart=tree.leaf_charts[group]
        else:
            children=[tree.left[node],tree.right[node]]
            anchors=(tree.port_centers[children]-translation)@rotation
            child_vertices=(tree.port_vertices[children].reshape(-1,3)-translation)@rotation
            physical=(tree.world[owned]-translation)@rotation
            chart=TetraChart.enclosing(anchors,np.vstack([physical,child_vertices]),
                np.r_[tree.radii[owned],np.zeros(len(child_vertices))],scale=float(tree.radii[owned].max()))
        tree.node_charts[node]=chart
        tree.port_rotations[node]=rotation; tree.port_translations[node]=translation
        tree.port_centers[node]=chart.center@rotation.T+translation
        tree.port_normals[node]=chart.normals@rotation.T
        tree.port_heights[node]=chart.heights
        tree.port_vertices[node]=chart.vertices@rotation.T+translation
        tree.lo[node]=tree.port_vertices[node].min(axis=0)
        tree.hi[node]=tree.port_vertices[node].max(axis=0)
    tree.exclusive=np.ones(count,dtype=np.bool_)
    for node in range(count):
        foreign=(atom_leaf<node)|(atom_leaf>=tree.end[node])
        signed=(tree.world-tree.port_centers[node])@tree.port_normals[node].T-tree.radii[:,None]
        possible=np.all(signed<=tree.port_heights[node]+1e-12,axis=1)
        tree.exclusive[node]=not np.any(foreign&possible)
    return dict(leaf_charts_rebuilt=0,internal_charts_built=int(np.count_nonzero(tree.leaf_group<0)),
        ghost_coordinates=sum(int(c.ghost_mask.sum()) for c in tree.node_charts),
        primitive_count=len(tree.radii),port_kind="tetra")
