"""Advected material collision ports connected to a full molecular field.

Known ports average occupation-weighted one-event observations. Missing target
ports get explicit physical suffix chains, preserving owner and collision cost
at every event. Each row consumes its current collision once, including escape.
The material quadrature is reused; geometry-dependent coupling is rebuilt.
This is a source-conditioned finite quadrature, not a universal intrinsic DtN.
"""
import time
import numpy as np
from numba import njit
from .ehss_empirical_field import geometry_key
from .ehss_empirical_bvh import first_collisions_bvh
from .ehss_field_condensation import ScatteringKernel,flat_field_solve
from .ehss_surface_response import surface_key
from .ehss_sparse_path_response import readonly,record_from_paths
from .ehss_guarded_intrinsic_response import reflect
from .ehss_reference import entry_distance
from .ehss_state import EHSSSource


def material_rotations(bank,centers):
    rotations=[]
    for atoms in bank.atoms:
        a=bank.reference[atoms]; b=centers[atoms]
        if np.array_equal(a-a[0],b-b[0]): rotation=np.eye(3)
        else:
            u,_,vh=np.linalg.svd((a-a.mean(axis=0)).T@(b-b.mean(axis=0)))
            correction=np.eye(3); correction[2,2]=1. if np.linalg.det(u@vh)>=0 else -1.
            rotation=(u@correction@vh).T
        rotations.append(rotation)
    return np.asarray(rotations)


@njit(cache=True)
def material_keys(points,directions,atoms,centers,groups,rotations,bins):
    keys=np.empty((len(atoms),7),dtype=np.int64)
    for i,atom in enumerate(atoms):
        normal=points[i]-centers[atom]; normal/=np.sqrt(np.dot(normal,normal))
        rotation=rotations[groups[atom]]
        key=surface_key(groups[atom],atom,normal@rotation,directions[i]@rotation,bins)
        for j in range(7): keys[i,j]=key[j]
    return keys


@njit(cache=True)
def first_observed_successors(points,directions,counts,history,centers,radii):
    p=points.copy(); d=directions.copy(); atoms=np.full(len(points),-1,dtype=np.int64)
    for i in range(len(points)):
        if counts[i]<2: continue
        atom=history[i,1]; distance=entry_distance(p[i],d[i],centers[atom],radii[atom])
        if not np.isfinite(distance): raise ValueError('Observed successor misses current physical sphere')
        reflect(p[i],d[i],atom,distance,centers,radii); atoms[i]=atom
    return p,d,atoms


class MaterialCollisionField:
    def __init__(self,bound,observed,physical_model,bins=2,checked_observations=False):
        tick=time.perf_counter(); bank=bound.bank; spheres=physical_model.spheres
        if not isinstance(bins,int) or bins<1: raise ValueError('Positive bins required')
        if geometry_key(bound.model.spheres)!=geometry_key(spheres): raise ValueError('Observation geometry differs from physical model')
        if observed['history'].shape!=(len(bank.mass),bank.cap) or not np.array_equal(observed['history'][:,0],bank.lasts):
            raise ValueError('Observation rows differ from material bank')
        self.bank=bank; self.model=physical_model; self.bins=bins; self.cap=bank.cap
        self.geometry=geometry_key(spheres); self.rotations=readonly(material_rotations(bank,spheres.centers))
        keys=[]
        for atom,n,d in zip(bank.lasts,bank.normals,bank.directions): keys.append(tuple(surface_key(bank.groups[atom],atom,n,d,bins)))
        self.keys=tuple(sorted(set(keys))); self.index={k:i for i,k in enumerate(self.keys)}; size=len(self.keys)
        ids=np.array([self.index[k] for k in keys]); rows=[{} for _ in range(size)]
        mass=np.zeros(size); visits=np.zeros(size,dtype=np.int64); escape=np.zeros((size,4)); residual=np.zeros(size)
        valid=observed['terminal']!=-3
        for row in np.flatnonzero(valid): mass[ids[row]]+=bank.mass[row]; visits[ids[row]]+=1
        self.active=readonly(mass>0); self.visits=readonly(visits); self.occupation_mass=readonly(mass)
        p,d,atoms=first_observed_successors(bound.points,bound.directions,observed['counts'],observed['history'],spheres.centers,spheres.radii)
        target=np.full(len(bank.mass),-1,dtype=np.int64); moving=np.flatnonzero(valid&(atoms>=0))
        next_keys=material_keys(p[moving],d[moving],atoms[moving],spheres.centers,bank.groups,self.rotations,bins)
        missing=[]
        for row,key in zip(moving,next_keys):
            state=self.index.get(tuple(key))
            if state is None or not self.active[state]: missing.append(int(row))
            else: target[row]=state
        # Missing target port response is completed by physical event chains,
        # never assigned wholesale to one owner or charged as a zero-cost escape.
        suffix_s=0.; suffix_tests=0; virtual_owners=[]; virtual_escape=[]; virtual_residual=[]; virtual_next=[]
        if missing:
            missing=np.asarray(missing,dtype=np.int64)
            sub=EHSSSource(p[missing],d[missing],d[missing],np.ones(len(missing)),atoms[missing],np.ones(len(missing),dtype=np.int64),'missing material target ports')
            start=time.perf_counter(); exact=physical_model.trace(sub,bank.cap); suffix_s=time.perf_counter()-start; suffix_tests=exact.metrics['sphere_tests']
            _,out,colliders,_=record_from_paths(sub.origins,sub.directions,sub.weights,sub.initial_bounces,exact.collider_ids,exact.bounces,spheres.centers,spheres.radii)
            cursor=0
            for ray,row in enumerate(missing):
                count=int(exact.bounces[ray]); target[row]=size+len(virtual_owners)
                for j in range(count):
                    virtual_owners.append(int(bank.groups[colliders[cursor+j]]))
                    last=j==count-1
                    virtual_next.append(-1 if last else size+len(virtual_owners))
                    virtual_escape.append(np.r_[1.,out[cursor+j]] if last and exact.escaped[ray] else np.zeros(4))
                    virtual_residual.append(float(last and exact.unresolved[ray]))
                cursor+=count
        for row in np.flatnonzero(valid):
            state=ids[row]; w=bank.mass[row]
            if atoms[row]>=0:
                following=int(target[row]); rows[state][following]=rows[state].get(following,0.)+w
            elif observed['terminal'][row]==-1: escape[state]+=w*np.r_[1.,bound.directions[row]]
            else: residual[state]+=w
        owners=[int(k[0]) for k in self.keys]+virtual_owners; starts=[0]; targets=[]; probabilities=[]
        for state in range(size):
            if mass[state]>0:
                for following,w in sorted(rows[state].items()): targets.append(following); probabilities.append(w/mass[state])
                escape[state]/=mass[state]; residual[state]/=mass[state]
            else: residual[state]=1.
            starts.append(len(targets))
        for following in virtual_next:
            if following>=0: targets.append(following); probabilities.append(1.)
            starts.append(len(targets))
        self.kernel=ScatteringKernel(owners,np.ones(len(owners),dtype=np.int64),starts,targets,probabilities,
            np.concatenate((escape,np.array(virtual_escape).reshape(-1,4))),np.r_[residual,virtual_residual])
        self.receipt=dict(prepare_s=time.perf_counter()-tick,material_ports=size,active_ports=int(self.active.sum()),
            observed_samples=len(bank.mass),valid_samples=int(valid.sum()),occluded_material_mass=float(bank.mass[~valid].sum()),
            known_next_samples=int(np.count_nonzero((target>=0)&(target<size))),missing_next_samples=len(missing),
            virtual_suffix_states=len(virtual_owners),suffix_s=suffix_s,suffix_sphere_tests=int(suffix_tests),
            branching_ports=sum(len(row)+(escape[i,0]>0)+(residual[i]>0)>1 for i,row in enumerate(rows)),
            order_conditioned=False,occupation_conditioned=True,checked_observations=bool(checked_observations),
            observed_apply_s=observed['query_s'],collision_cost_per_state=1,
            material_quadrature_reused=True,coupling_rebuilt=True,full_intrinsic_operator_reused=False)

    def solve(self,source,cap=128,engine=None,min_visits=1):
        tick=time.perf_counter(); spheres=self.model.spheres; tree=self.model.tree
        if geometry_key(spheres)!=self.geometry: raise ValueError('Geometry changed; rebuild coupling')
        if not isinstance(cap,int) or not 1<=cap<=self.cap: raise ValueError('Cap exceeds compiled physical suffix budget')
        if not isinstance(min_visits,int) or min_visits<1: raise ValueError('Positive visit threshold required')
        if np.any((source.initial_bounces!=0)&(source.initial_bounces!=1)): raise ValueError('Uncollided or first-reflected sources required')
        if np.any(source.last_atom>=len(spheres.radii)) or np.any(source.last_atom < -1): raise ValueError('Invalid source atom')
        if engine is not None and engine.kernel is not self.kernel: raise ValueError('Different field kernel')
        first_tick=time.perf_counter()
        p,d,atoms,counters=first_collisions_bvh(source.origins,source.directions,source.last_atom,source.initial_bounces,
            spheres.centers,spheres.radii,tree.lo,tree.hi,tree.left,tree.right,tree.end,tree.leaf_group,tree.offsets,tree.ids)
        first_s=time.perf_counter()-first_tick; hit=atoms>=0; indices=np.flatnonzero(hit)
        keys=material_keys(p[hit],d[hit],atoms[hit],spheres.centers,self.bank.groups,self.rotations,self.bins)
        selected=[]; states=[]; fallback=[]
        for ray,key in zip(indices,keys):
            state=self.index.get(tuple(key))
            if state is None or not self.active[state] or self.visits[state]<min_visits: fallback.append(int(ray))
            else: selected.append(int(ray)); states.append(state)
        seeds=[]; values=[]; initial_escape=np.zeros(2); k=self.kernel
        # Charge the exactly located first collision here. Its immediate escape
        # uses this ray's direction, preserving the known first-event correlation.
        for ray,state in zip(selected,states):
            value=np.r_[source.weights[ray],source.weights[ray]*source.incoming[ray]]
            initial_escape+=k.escape[state,0]*np.array([value[0],value[0]-np.dot(value[1:],d[ray])])
            for edge in range(k.starts[state],k.starts[state+1]):
                seeds.append(int(k.targets[edge])); values.append(k.probabilities[edge]*value)
        initial_tail=sum(source.weights[ray]*k.residual[state] for ray,state in zip(selected,states))
        values=np.array(values).reshape(-1,4); orders=np.ones(len(seeds),dtype=np.int64); seed_s=time.perf_counter()-tick
        result=flat_field_solve(k,seeds,orders,values,cap) if engine is None else engine.solve(seeds,orders,values,cap)
        ledger=np.array([[r['mass'],r['omega']] for r in result['order_ledger']]); ledger[1]+=initial_escape
        ledger[0,0]+=source.weights[~hit].sum(); ledger[0,1]+=np.sum(source.weights[~hit]*(1-np.sum(source.incoming[~hit]*d[~hit],axis=1)))
        tail=result['residual_mass']+initial_tail; fallback_s=0.; fallback_tests=0
        if fallback:
            rays=np.asarray(fallback,dtype=np.int64)
            sub=EHSSSource(p[rays],d[rays],source.incoming[rays],source.weights[rays],atoms[rays],np.ones(len(rays),dtype=np.int64),'unknown first material port')
            start=time.perf_counter(); exact=self.model.trace(sub,cap); fallback_s=time.perf_counter()-start; fallback_tests=exact.metrics['sphere_tests']
            for row in exact.order_ledger(): ledger[row['order']]+=(row['mass'],row['omega'])
            tail+=exact.residual_mass
        if not np.isclose(ledger[:,0].sum()+tail,source.weights.sum(),rtol=1e-11,atol=1e-11): raise AssertionError('Full material field mass changed')
        return dict(omega=float(ledger[:,1].sum()),pa=float(source.weights[hit].sum()),escaped_mass=float(ledger[:,0].sum()),
            residual_mass=float(tail),tail_bound=float(2*tail),source_mass=float(source.weights.sum()),
            learned_mass=float(source.weights[selected].sum()),fallback_mass=float(source.weights[fallback].sum()),
            learned_rays=len(selected),fallback_rays=len(fallback),
            order_ledger=[dict(order=i,mass=float(v[0]),omega=float(v[1])) for i,v in enumerate(ledger)],
            first_s=first_s,seed_s=seed_s,field_s=result['query_s'],fallback_s=fallback_s,
            sphere_tests=int(counters[1]+fallback_tests),query_s=time.perf_counter()-tick,field=result)
