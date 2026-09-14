"""Sparse numerical internal responses with current foreign arbitration.

Each material column starts after a known reflection. It stores every owned
flight and reflected direction through local escape or the collision cap.
Rigidly posed columns reuse these numbers, not just collider identities.
Nonrigid blocks are rebuilt; this version does not approximate strain reuse.
The first foreign collision ends a column application, not the global path.
"""
from dataclasses import dataclass
import time
import numpy as np
from numba import njit
from .ehss_reference import nearest_all
from .ehss_guarded_intrinsic_response import reflect
from .ehss_owner_search import OwnerSearchLayout, nearest_owner
from .ehss_factored_local_samples import point_inside
from .ehss_sparse_path_response import readonly
from .ehss_state import EHSSSource


@njit(cache=True)
def compile_internal(points, directions, lasts, centers, radii, cap):
    # Lists hold only visited segments; no samples x cap dense allocation.
    starts=[0]; origins=[]; outgoing=[]; targets=[]; distances=[]; tests=0
    for i in range(len(points)):
        p=points[i].copy(); d=directions[i].copy(); last=lasts[i]; count=1
        while True:
            atom,distance,nt=nearest_all(p,d,last,centers,radii); tests+=nt
            origins.append(p.copy()); outgoing.append(d.copy()); targets.append(atom); distances.append(distance)
            if atom<0 or count>=cap: break
            reflect(p,d,atom,distance,centers,radii); last=atom; count+=1
        starts.append(len(targets))
    return starts,origins,outgoing,targets,distances,tests


@njit(cache=False)
def apply_internal(rows, starts, origins, directions, targets, distances, atoms, rotation, translation,
                   input_points, input_directions, lasts, owner, cap,
                   lo,hi,left,right,leaf_group,offsets,ids,owners,uniform,centers,radii):
    n=len(rows); p=input_points[rows].copy(); d=input_directions[rows].copy()
    counts=np.ones(n,dtype=np.int64); history=np.full((n,cap),-1,dtype=np.int64)
    terminal=np.full(n,-2,dtype=np.int64); counters=np.zeros(3,dtype=np.int64)
    local_events=0; foreign_queries=0; interruptions=0
    for i in range(n):
        row=rows[i]; last=lasts[row]; history[i,0]=last
        if point_inside(0,p[i],last,lo,hi,left,right,leaf_group,offsets,ids,centers,radii,counters):
            terminal[i]=-3; continue
        for segment in range(starts[i],starts[i+1]):
            if segment>starts[i]:
                p[i]=origins[segment]@rotation.T+translation
                d[i]=directions[segment]@rotation.T
            local_target=targets[segment]
            wanted=-1 if local_target<0 else atoms[local_target]
            foreign,before=nearest_owner(0,p[i],d[i],last,distances[segment],owner,-1,
                lo,hi,left,right,leaf_group,offsets,ids,owners,uniform,centers,radii,counters)
            foreign_queries+=1
            if foreign>=0 and (wanted<0 or before<distances[segment] or (before==distances[segment] and foreign<wanted)):
                if counts[i]>=cap: break
                reflect(p[i],d[i],foreign,before,centers,radii)
                history[i,counts[i]]=foreign; counts[i]+=1; terminal[i]=foreign
                interruptions+=int(wanted>=0); break
            if wanted<0:
                terminal[i]=-1; break
            if counts[i]>=cap: break
            # The next stored segment supplies the owned post-reflection state.
            history[i,counts[i]]=wanted; counts[i]+=1; last=wanted; local_events+=1
    return p,d,counts,history,terminal,counters,local_events,foreign_queries,interruptions


@dataclass(frozen=True)
class IntrinsicColumnBlock:
    centers: np.ndarray
    radii: np.ndarray
    starts: np.ndarray
    origins: np.ndarray
    directions: np.ndarray
    targets: np.ndarray
    distances: np.ndarray
    primitive_tests: int


class IntrinsicColumns:
    """Frozen first-reflected material samples, grouped by chemical owner.

The tiny pose tolerance is for numerical rigid fits. Larger shape changes
recompile the affected internal block. No foreign geometry enters a T key.
"""
    def __init__(self, model, owners, source, cap=64, rigid_tolerance=1e-10):
        if type(cap) is not int or cap<1: raise ValueError('Positive collision cap required')
        if not np.isfinite(rigid_tolerance) or not 0<=rigid_tolerance<=1e-8:
            raise ValueError('Rigid roundoff tolerance must be between 0 and 1e-8')
        if np.any(source.initial_bounces!=1) or np.any(source.last_atom<0) or np.any(source.last_atom>=len(model.spheres.radii)):
            raise ValueError('Known first-reflected material samples required')
        self.search_layout=OwnerSearchLayout(model,owners)
        self.owners=self.search_layout.owners; self.atoms=self.search_layout.atoms
        self.reference=readonly(model.spheres.centers.copy())
        self.lasts=readonly(source.last_atom.copy()); self.weights=readonly(source.weights.copy())
        delta=source.origins-self.reference[self.lasts]; lengths=np.linalg.norm(delta,axis=1)
        if not np.allclose(lengths,model.spheres.radii[self.lasts],rtol=1e-11,atol=1e-10):
            raise ValueError('Material input must be on its known atom surface')
        normals=delta/lengths[:,None]
        if np.any(np.sum(normals*source.directions,axis=1)<-1e-12):
            raise ValueError('Post-reflection directions must point out of the known atom')
        self.normals=readonly(normals); self.directions=readonly(source.directions.copy()); self.tags=readonly(source.incoming.copy())
        self.rows=tuple(readonly(np.flatnonzero(self.owners[self.lasts]==g)) for g in range(len(self.atoms)))
        self.cap=cap; self.rigid_tolerance=float(rigid_tolerance)

    def bind(self,model,previous=None):
        tick=time.perf_counter(); search=self.search_layout.bind(model)
        if previous is not None and previous.bank is not self: raise ValueError('Previous response belongs to another bank')
        world=model.spheres.centers; radii=model.spheres.radii
        rotations=[]; translations=[]; columns=[]; reused=[]; rebuilt=[]; residuals=[]
        points=np.empty((len(self.lasts),3)); directions=points.copy(); tags=points.copy(); tests=0
        for g,(atoms,rows) in enumerate(zip(self.atoms,self.rows)):
            a=self.reference[atoms]; b=world[atoms]; translation=b[0]
            if np.array_equal(a-a[0],b-b[0]): rotation=np.eye(3)
            else:
                u,_,vh=np.linalg.svd((a-a.mean(axis=0)).T@(b-b.mean(axis=0)))
                fix=np.eye(3); fix[2,2]=1. if np.linalg.det(u@vh)>=0 else -1.
                rotation=(u@fix@vh).T
            local=(b-translation)@rotation
            rotations.append(readonly(rotation)); translations.append(readonly(translation.copy()))
            points[rows]=world[self.lasts[rows]]+radii[self.lasts[rows],None]*(self.normals[rows]@rotation.T)
            directions[rows]=self.directions[rows]@rotation.T; tags[rows]=self.tags[rows]@rotation.T
            old=None if previous is None else previous.columns[g]
            residual=np.inf if old is None else float(np.linalg.norm(local-old.centers,axis=1).max())
            residuals.append(None if not np.isfinite(residual) else residual)
            if not len(rows):
                columns.append(None); continue
            if old is not None and residual<=self.rigid_tolerance and np.array_equal(radii[atoms],old.radii):
                columns.append(old); reused.append(g); continue
            lasts=np.searchsorted(atoms,self.lasts[rows])
            local_points=local[lasts]+radii[self.lasts[rows],None]*self.normals[rows]
            starts,origins,outgoing,targets,distances,nt=compile_internal(local_points,self.directions[rows],lasts,local,radii[atoms],self.cap)
            column=IntrinsicColumnBlock(readonly(local.copy()),readonly(radii[atoms].copy()),
                readonly(np.array(starts,dtype=np.int64)),readonly(np.array(origins).reshape(-1,3)),
                readonly(np.array(outgoing).reshape(-1,3)),readonly(np.array(targets,dtype=np.int64)),
                readonly(np.array(distances,dtype=float)),int(nt))
            columns.append(column); rebuilt.append(g); tests+=nt
        receipt=dict(bind_s=time.perf_counter()-tick,reused_owners=reused,rebuilt_owners=rebuilt,
            shape_residuals=residuals,local_primitive_tests=int(tests),material_rows=len(self.lasts),
            stored_segments=sum(len(c.targets) for c in columns if c is not None),
            full_numerical_local_columns=True,complete_boundary_operator=False,
            strain_approximation=False,rigid_tolerance=self.rigid_tolerance)
        return BoundIntrinsicColumns(self,search,tuple(columns),tuple(rotations),tuple(translations),
            readonly(points),readonly(directions),readonly(tags),receipt)


@dataclass(frozen=True)
class BoundIntrinsicColumns:
    bank: IntrinsicColumns
    search: object
    columns: tuple
    rotations: tuple
    translations: tuple
    points: np.ndarray
    directions: np.ndarray
    tags: np.ndarray
    receipt: dict

    def source(self):
        b=self.bank
        return EHSSSource(self.points,self.directions,self.tags,b.weights,b.lasts,np.ones(len(b.lasts),dtype=np.int64),
                          'material first-reflected internal response columns')

    def apply(self):
        tick=time.perf_counter(); b=self.bank; n=len(b.lasts)
        points=self.points.copy(); directions=self.directions.copy(); counts=np.ones(n,dtype=np.int64)
        history=np.full((n,b.cap),-1,dtype=np.int64); terminal=np.full(n,-2,dtype=np.int64)
        counters=np.zeros(3,dtype=np.int64); local_events=foreign_queries=interruptions=reused_events=0
        for g,(rows,column) in enumerate(zip(b.rows,self.columns)):
            if column is None: continue
            result=apply_internal(rows,column.starts,column.origins,column.directions,column.targets,column.distances,
                b.atoms[g],self.rotations[g],self.translations[g],self.points,self.directions,b.lasts,g,b.cap,*self.search.arguments())
            p,d,c,h,t,costs,events,queries,interrupted=result
            points[rows]=p; directions[rows]=d; counts[rows]=c; history[rows]=h; terminal[rows]=t
            counters+=costs; local_events+=events; foreign_queries+=queries; interruptions+=interrupted
            if g in self.receipt['reused_owners']: reused_events+=events
        completed=terminal>=-1
        result=dict(points=points,directions=directions,counts=counts,history=history,terminal=terminal,
            input_mass=float(b.weights.sum()),escape_mass=float(b.weights[terminal==-1].sum()),
            foreign_mass=float(b.weights[terminal>=0].sum()),occluded_mass=float(b.weights[terminal==-3].sum()),
            unresolved_mass=float(b.weights[terminal==-2].sum()),
            completed_direction_moment=float(np.sum(b.weights*np.clip(1-np.sum(self.tags*directions,axis=1),0,2)*completed)),
            local_events_from_stored_numbers=int(local_events),reused_local_events=int(reused_events),
            foreign_queries=int(foreign_queries),foreign_interruptions=int(interruptions),
            box_tests=int(counters[0]),sphere_tests=int(counters[1]),owner_prunes=int(counters[2]),
            query_s=time.perf_counter()-tick,global_ccs_readout=False)
        if not np.isclose(result['input_mass'],sum(result[k] for k in ['escape_mass','foreign_mass','occluded_mass','unresolved_mass']),rtol=1e-12,atol=1e-12):
            raise AssertionError('First-exit outcomes must partition input mass')
        return result
