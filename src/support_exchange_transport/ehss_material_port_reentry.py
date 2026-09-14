"""Stop missing-target physical suffixes at an active material collision port.

The arrived known collision becomes the next field state, not an escape and
not an additional virtual collision. This adds an empirical projection at
reentry; it does not claim per-path equivalence to a full physical suffix.
"""
import time
import numpy as np
from numba import njit
from .ehss_guarded_intrinsic_response import nearest_world,reflect
from .ehss_surface_response import surface_key
from .ehss_sparse_path_response import readonly
from .ehss_empirical_field import geometry_key
from .ehss_field_condensation import ScatteringKernel
from .ehss_state import EHSSSource
from .ehss_material_collision_field import (
    MaterialCollisionField,material_keys,material_rotations,first_observed_successors)


@njit(cache=True)
def current_material_port(point,direction,atom,world,groups,rotations,bins,table,active):
    normal=point-world[atom]; normal/=np.sqrt(np.dot(normal,normal))
    rotation=rotations[groups[atom]]
    key=surface_key(groups[atom],atom,normal@rotation,direction@rotation,bins)
    lo=0; hi=len(table)
    while lo<hi:
        mid=(lo+hi)//2; compare=0
        for c in range(7):
            if table[mid,c]<key[c]: compare=-1; break
            if table[mid,c]>key[c]: compare=1; break
        if compare<0: lo=mid+1
        else: hi=mid
    if lo>=len(table) or not active[lo]: return -1
    for c in range(7):
        if table[lo,c]!=key[c]: return -1
    return lo


@njit(cache=False)
def trace_port_prefix(origins,directions,last_atoms,cap,world,radii,
                      lo,hi,left,right,end,leaf_group,offsets,ids,
                      groups,rotations,bins,table,active,allow_reentry):
    p=origins.copy(); d=directions.copy(); counts=np.ones(len(p),dtype=np.int64)
    history=np.full((len(p),cap),-1,dtype=np.int32)
    ports=np.full(len(p),-1,dtype=np.int64)
    escaped=np.zeros(len(p),dtype=np.bool_); unresolved=escaped.copy()
    counters=np.zeros(2,dtype=np.int64); queries=lookups=0
    for ray in range(len(p)):
        last=last_atoms[ray]; count=1; history[ray,0]=last
        while True:
            if allow_reentry:
                lookups+=1
                state=current_material_port(p[ray],d[ray],last,world,groups,rotations,bins,table,active)
                if state>=0:
                    ports[ray]=state; break
            queries+=1
            atom,distance=nearest_world(0,p[ray],d[ray],last,np.inf,-1,
                lo,hi,left,right,end,leaf_group,offsets,ids,world,radii,counters)
            if atom<0: escaped[ray]=True; break
            if count>=cap: unresolved[ray]=True; break
            reflect(p[ray],d[ray],atom,distance,world,radii)
            history[ray,count]=atom; count+=1; last=atom
        counts[ray]=count
    return p,d,escaped,unresolved,counts,history,ports,counters,queries,lookups


def trace_to_material_port(model,source,keys,active,groups,rotations,bins,cap,allow_reentry=True):
    """Current physical first-entry search; handoff is a third terminal category."""
    if not isinstance(cap,int) or cap<1: raise ValueError('Positive collision budget required')
    if np.any(source.initial_bounces!=1) or np.any(source.last_atom<0): raise ValueError('First-reflected inputs required')
    if np.any(source.last_atom>=len(model.spheres.radii)): raise ValueError('Invalid physical collider identity')
    if not isinstance(bins,int) or bins<1: raise ValueError('Positive bins required')
    if tuple(keys)!=tuple(sorted(set(keys))) or len(active)!=len(keys): raise ValueError('Unique sorted keys and matching active mask required')
    table=readonly(np.array(keys,dtype=np.int64).reshape(-1,7))
    tree=model.tree; tick=time.perf_counter()
    result=trace_port_prefix(source.origins,source.directions,source.last_atom,cap,model.spheres.centers,model.spheres.radii,
        tree.lo,tree.hi,tree.left,tree.right,tree.end,tree.leaf_group,tree.offsets,tree.ids,
        groups,rotations,bins,table,active,allow_reentry)
    p,d,escaped,unresolved,counts,history,ports,counters,queries,lookups=result
    assert np.all(escaped.astype(int)+unresolved+(ports>=0)==1)
    return dict(positions=p,outgoing=d,escaped=escaped,unresolved=unresolved,bounces=counts,collider_ids=history,ports=ports,
        metrics=dict(query_s=time.perf_counter()-tick,sphere_tests=int(counters[1]),box_tests=int(counters[0]),
            nearest_queries=int(queries),port_lookups=int(lookups)))


class ReentryMaterialCollisionField(MaterialCollisionField):
    def __init__(self,bound,observed,physical_model,bins=2,checked_observations=False,allow_reentry=True):
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
        # Only the unknown prefix becomes virtual states. A known arrival is
        # represented by its existing material state, consuming that collision
        # once on the following field step.
        suffix_s=0.; suffix_tests=0; virtual_owners=[]; virtual_escape=[]; virtual_residual=[]; virtual_next=[]
        virtual_labels=[]; self.suffix_routes=dict(missing_rows=[],counts=[],ports=[],virtual_counts=[])
        suffix_metrics=dict(sphere_tests=0,box_tests=0,nearest_queries=0,port_lookups=0)
        reentered=escaped_suffixes=unresolved_suffixes=0
        if missing:
            missing=np.asarray(missing,dtype=np.int64)
            sub=EHSSSource(p[missing],d[missing],d[missing],np.ones(len(missing)),atoms[missing],np.ones(len(missing),dtype=np.int64),'missing material target ports')
            start=time.perf_counter()
            prefix=trace_to_material_port(physical_model,sub,self.keys,self.active,bank.groups,self.rotations,bins,bank.cap,allow_reentry)
            suffix_s=time.perf_counter()-start; suffix_metrics=prefix['metrics']; suffix_tests=suffix_metrics['sphere_tests']
            virtual_counts=[]
            for ray,row in enumerate(missing):
                port=int(prefix['ports'][ray]); count=int(prefix['bounces'][ray]); nvirtual=count-int(port>=0)
                # In this constructor the initial target key was already missing.
                if nvirtual<1: raise AssertionError('Missing initial target unexpectedly matched a port')
                target[row]=size+len(virtual_owners); virtual_counts.append(nvirtual)
                for j in range(nvirtual):
                    virtual_owners.append(int(bank.groups[prefix['collider_ids'][ray,j]])); virtual_labels.append(('suffix',int(row),j))
                    last=j==nvirtual-1
                    virtual_next.append(port if last else size+len(virtual_owners))
                    virtual_escape.append(np.r_[1.,prefix['outgoing'][ray]] if last and prefix['escaped'][ray] else np.zeros(4))
                    virtual_residual.append(float(last and prefix['unresolved'][ray]))
            reentered=int(np.count_nonzero(prefix['ports']>=0)); escaped_suffixes=int(prefix['escaped'].sum()); unresolved_suffixes=int(prefix['unresolved'].sum())
            self.suffix_routes=dict(missing_rows=readonly(missing),counts=readonly(prefix['bounces']),ports=readonly(prefix['ports']),virtual_counts=readonly(np.array(virtual_counts)))
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
        self.labels=tuple(('port',)+tuple(map(int,key)) for key in self.keys)+tuple(virtual_labels)
        if len(set(self.labels))!=len(owners): raise AssertionError('Unique labels must cover every current state')
        self.receipt=dict(prepare_s=time.perf_counter()-tick,material_ports=size,active_ports=int(self.active.sum()),
            observed_samples=len(bank.mass),valid_samples=int(valid.sum()),occluded_material_mass=float(bank.mass[~valid].sum()),
            known_next_samples=int(np.count_nonzero((target>=0)&(target<size))),missing_next_samples=len(missing),
            virtual_suffix_states=len(virtual_owners),suffix_s=suffix_s,suffix_sphere_tests=int(suffix_tests),
            suffix_metrics=suffix_metrics,reentry_enabled=bool(allow_reentry),reentered_suffixes=reentered,
            escaped_suffixes=escaped_suffixes,unresolved_suffixes=unresolved_suffixes,
            handoff_is_escape=False,arrived_collision_represented_by_material_state=True,
            branching_ports=sum(len(row)+(escape[i,0]>0)+(residual[i]>0)>1 for i,row in enumerate(rows)),
            order_conditioned=False,occupation_conditioned=True,checked_observations=bool(checked_observations),
            observed_apply_s=observed['query_s'],collision_cost_per_state=1,
            material_quadrature_reused=True,coupling_rebuilt=True,full_intrinsic_operator_reused=False)

