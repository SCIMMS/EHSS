"""Batch coupling assembly without changing physical tracing or response rows."""
import time
import numpy as np
from numba import njit
from .ehss_field_condensation import ScatteringKernel
from .ehss_sparse_path_response import readonly
from .ehss_surface_response import surface_key
from .ehss_material_collision_field import MaterialCollisionField,material_rotations,material_keys,first_observed_successors
from .ehss_material_port_reentry import trace_to_material_port
from .ehss_empirical_field import geometry_key
from .ehss_state import EHSSSource


class BatchedScatteringKernel(ScatteringKernel):
    """Same immutable arrays and physical constraints, vector row validation."""
    def __init__(self,owners,costs,starts,targets,probabilities,escape,residual):
        values=(owners,costs,starts,targets,probabilities,escape,residual)
        dtypes=(np.int64,np.int64,np.int64,np.int64,float,float,float)
        self.owners,self.costs,self.starts,self.targets,self.probabilities,self.escape,self.residual=[readonly(np.array(v,dtype=d,copy=True)) for v,d in zip(values,dtypes)]
        n=len(self.owners)
        if self.owners.shape!=(n,) or self.costs.shape!=(n,) or self.starts.shape!=(n+1,) or self.escape.shape!=(n,4) or self.residual.shape!=(n,):
            raise ValueError('Invalid kernel shapes')
        if self.targets.ndim!=1 or self.probabilities.ndim!=1: raise ValueError('Invalid sparse transition shape')
        if np.any(self.costs<1): raise ValueError('Positive collision costs required after through closure')
        if self.starts[0]!=0 or np.any(np.diff(self.starts)<0) or self.starts[-1]!=len(self.targets) or len(self.targets)!=len(self.probabilities):
            raise ValueError('Invalid sparse transition index')
        if np.any(self.targets<0) or np.any(self.targets>=n): raise ValueError('Invalid target state')
        for a in (self.probabilities,self.escape,self.residual):
            if not np.isfinite(a).all(): raise ValueError('Finite coefficients required')
        if np.any(self.probabilities<0) or np.any(self.escape[:,0]<0) or np.any(self.residual<0): raise ValueError('Nonnegative outcome probabilities required')
        if np.any(np.linalg.norm(self.escape[:,1:],axis=1)>self.escape[:,0]+1e-12): raise ValueError('Escape direction moment exceeds mass')
        sources=np.repeat(np.arange(n),np.diff(self.starts))
        total=np.bincount(sources,weights=self.probabilities,minlength=n)+self.escape[:,0]+self.residual
        if np.any(np.abs(total-1.)>1e-12): raise ValueError('Each row must conserve probability')


@njit(cache=True)
def reference_keys(lasts,normals,directions,groups,bins):
    keys=np.empty((len(lasts),7),dtype=np.int64)
    for i,atom in enumerate(lasts):
        key=surface_key(groups[atom],atom,normals[i],directions[i],bins)
        for j in range(7): keys[i,j]=key[j]
    return keys


class MaterialPortLayout:
    """Immutable reference-cell identities, reusable only with the same bank."""
    def __init__(self,bank,bins=2):
        tick=time.perf_counter()
        if type(bins) is not int or bins<1: raise ValueError('Positive bins required')
        self.bank=bank; self.bins=bins
        table,ids=np.unique(reference_keys(bank.lasts,bank.normals,bank.directions,bank.groups,bins),axis=0,return_inverse=True)
        self.table=readonly(table); self.ids=readonly(ids)
        self.keys=tuple(tuple(map(int,k)) for k in table)
        self.index={k:i for i,k in enumerate(self.keys)}
        self.labels=tuple(('port',)+k for k in self.keys)
        self.prepare_s=time.perf_counter()-tick


@njit(cache=True)
def lookup_targets(keys,table,active):
    result=np.full(len(keys),-1,dtype=np.int64)
    for i in range(len(keys)):
        lo=0; hi=len(table)
        while lo<hi:
            mid=(lo+hi)//2; compare=0
            for c in range(7):
                if table[mid,c]<keys[i,c]: compare=-1; break
                if table[mid,c]>keys[i,c]: compare=1; break
            if compare<0: lo=mid+1
            else: hi=mid
        if lo<len(table) and active[lo]:
            match=True
            for c in range(7):
                if table[lo,c]!=keys[i,c]: match=False; break
            if match: result[i]=lo
    return result


@njit(cache=True)
def virtual_arrays(size,missing,counts,ports,history,outgoing,escaped,unresolved,groups):
    lengths=counts-(ports>=0); offsets=np.zeros(len(missing)+1,dtype=np.int64)
    for i in range(len(missing)):
        if lengths[i]<1: raise ValueError('Missing target unexpectedly matched a port')
        offsets[i+1]=offsets[i]+lengths[i]
    n=offsets[-1]; owners=np.empty(n,dtype=np.int64); following=np.full(n,-1,dtype=np.int64)
    escape=np.zeros((n,4)); residual=np.zeros(n)
    for ray in range(len(missing)):
        for j in range(lengths[ray]):
            index=offsets[ray]+j; owners[index]=groups[history[ray,j]]
            if j<lengths[ray]-1: following[index]=size+index+1
            else:
                following[index]=ports[ray]
                if escaped[ray]:
                    escape[index,0]=1.
                    for k in range(3): escape[index,k+1]=outgoing[ray,k]
                if unresolved[ray]: residual[index]=1.
    return lengths,offsets,owners,following,escape,residual


@njit(cache=True)
def port_rows(ids,weights,valid,atoms,target,terminal,directions,mass,edge_order):
    size=len(mass); escape=np.zeros((size,4)); residual=np.zeros(size)
    for row in range(len(ids)):
        if valid[row] and atoms[row]<0:
            state=ids[row]; w=weights[row]
            if terminal[row]==-1:
                escape[state,0]+=w
                for k in range(3): escape[state,k+1]+=w*directions[row,k]
            else: residual[state]+=w
    for state in range(size):
        if mass[state]>0:
            for k in range(4): escape[state,k]/=mass[state]
            residual[state]/=mass[state]
        else: residual[state]=1.
    starts=np.zeros(size+1,dtype=np.int64); targets=np.empty(len(edge_order),dtype=np.int64); prob=np.empty(len(edge_order))
    cursor=count=0
    while cursor<len(edge_order):
        row=edge_order[cursor]; state=ids[row]; following=target[row]; value=0.
        while cursor<len(edge_order):
            row=edge_order[cursor]
            if ids[row]!=state or target[row]!=following: break
            value+=weights[row]; cursor+=1
        if mass[state]>0:
            starts[state+1]+=1; targets[count]=following; prob[count]=value/mass[state]; count+=1
    for i in range(size): starts[i+1]+=starts[i]
    return starts,targets[:count],prob[:count],escape,residual


class BatchedReentryMaterialCollisionField(MaterialCollisionField):
    def __init__(self,bound,observed,physical_model,bins=2,checked_observations=False,allow_reentry=True,layout=None,vector_validation=True):
        tick=time.perf_counter(); bank=bound.bank; spheres=physical_model.spheres; stages={}
        if type(bins) is not int or bins<1: raise ValueError('Positive bins required')
        if geometry_key(bound.model.spheres)!=geometry_key(spheres): raise ValueError('Observation geometry differs from physical model')
        if observed['history'].shape!=(len(bank.mass),bank.cap) or not np.array_equal(observed['history'][:,0],bank.lasts): raise ValueError('Observation rows differ from material bank')
        self.bank=bank; self.model=physical_model; self.bins=bins; self.cap=bank.cap; self.geometry=geometry_key(spheres)
        start=time.perf_counter(); self.rotations=readonly(material_rotations(bank,spheres.centers)); stages['rotation_s']=time.perf_counter()-start
        start=time.perf_counter(); reused=layout is not None
        if layout is None: layout=MaterialPortLayout(bank,bins)
        elif layout.bank is not bank or layout.bins!=bins: raise ValueError('Layout belongs to another material bank or bins')
        self.layout=layout; self.keys=layout.keys; self.index=layout.index; ids=layout.ids; size=len(self.keys)
        stages['layout_s']=time.perf_counter()-start; start=time.perf_counter()
        valid=observed['terminal']!=-3
        mass=np.bincount(ids[valid],weights=bank.mass[valid],minlength=size)
        visits=np.bincount(ids[valid],minlength=size)
        self.active=readonly(mass>0); self.visits=readonly(visits); self.occupation_mass=readonly(mass)
        p,d,atoms=first_observed_successors(bound.points,bound.directions,observed['counts'],observed['history'],spheres.centers,spheres.radii)
        moving=np.flatnonzero(valid&(atoms>=0)); target=np.full(len(bank.mass),-1,dtype=np.int64)
        keys=material_keys(p[moving],d[moving],atoms[moving],spheres.centers,bank.groups,self.rotations,bins)
        target[moving]=lookup_targets(keys,layout.table,self.active)
        missing=moving[target[moving]<0]; stages['successors_s']=time.perf_counter()-start
        suffix_metrics=dict(sphere_tests=0,box_tests=0,nearest_queries=0,port_lookups=0)
        reentered=escaped_suffixes=unresolved_suffixes=0; suffix_s=0.; start=time.perf_counter()
        prefix=None
        if len(missing):
            sub=EHSSSource(p[missing],d[missing],d[missing],np.ones(len(missing)),atoms[missing],np.ones(len(missing),dtype=np.int64),'missing material target ports')
            trace_start=time.perf_counter()
            prefix=trace_to_material_port(physical_model,sub,self.keys,self.active,bank.groups,self.rotations,bins,bank.cap,allow_reentry)
            suffix_s=time.perf_counter()-trace_start; suffix_metrics=prefix['metrics']
        stages['physical_prefix_s']=time.perf_counter()-start; start=time.perf_counter()
        if prefix is not None:
            lengths,offsets,vowners,vnext,vescape,vres=virtual_arrays(size,missing,prefix['bounces'],prefix['ports'],prefix['collider_ids'],prefix['outgoing'],prefix['escaped'],prefix['unresolved'],bank.groups)
            target[missing]=size+offsets[:-1]
            virtual_labels=tuple(('suffix',int(row),j) for row,n in zip(missing,lengths) for j in range(n))
            self.suffix_routes=dict(missing_rows=readonly(missing),counts=readonly(prefix['bounces']),ports=readonly(prefix['ports']),virtual_counts=readonly(lengths))
            reentered=int(np.count_nonzero(prefix['ports']>=0)); escaped_suffixes=int(prefix['escaped'].sum()); unresolved_suffixes=int(prefix['unresolved'].sum())
        else:
            vowners=np.empty(0,dtype=np.int64); vnext=vowners; vescape=np.empty((0,4)); vres=np.empty(0); virtual_labels=()
            self.suffix_routes=dict(missing_rows=[],counts=[],ports=[],virtual_counts=[])
        edge_order=moving[np.lexsort((moving,target[moving],ids[moving]))]
        starts,targets,prob,escape,residual=port_rows(ids,bank.mass,valid,atoms,target,observed['terminal'],bound.directions,mass,edge_order)
        branching=int(np.count_nonzero(np.diff(starts)+(escape[:,0]>0)+(residual>0)>1))
        full_starts=np.r_[starts,starts[-1]+np.cumsum(vnext>=0)]
        self.labels=layout.labels+virtual_labels
        stages['numeric_assembly_s']=time.perf_counter()-start; start=time.perf_counter()
        factory=BatchedScatteringKernel if vector_validation else ScatteringKernel
        self.kernel=factory(np.r_[layout.table[:,0],vowners],np.ones(size+len(vowners),dtype=np.int64),full_starts,
            np.r_[targets,vnext[vnext>=0]],np.r_[prob,np.ones(np.count_nonzero(vnext>=0))],np.concatenate((escape,vescape)),np.r_[residual,vres])
        stages['kernel_validation_s']=time.perf_counter()-start
        if len(self.labels)!=len(self.kernel.owners) or len(set(self.labels))!=len(self.labels): raise AssertionError('Unique semantic state labels required')
        self.receipt=dict(prepare_s=time.perf_counter()-tick,stages=stages,material_ports=size,active_ports=int(self.active.sum()),observed_samples=len(bank.mass),
            valid_samples=int(valid.sum()),occluded_material_mass=float(bank.mass[~valid].sum()),known_next_samples=int(np.count_nonzero((target>=0)&(target<size))),
            missing_next_samples=len(missing),virtual_suffix_states=len(vowners),suffix_s=suffix_s,suffix_sphere_tests=int(suffix_metrics['sphere_tests']),suffix_metrics=suffix_metrics,
            reentry_enabled=bool(allow_reentry),reentered_suffixes=reentered,escaped_suffixes=escaped_suffixes,unresolved_suffixes=unresolved_suffixes,
            handoff_is_escape=False,arrived_collision_represented_by_material_state=True,branching_ports=branching,
            order_conditioned=False,occupation_conditioned=True,checked_observations=bool(checked_observations),observed_apply_s=observed['query_s'],collision_cost_per_state=1,
            material_quadrature_reused=True,coupling_rebuilt=True,full_intrinsic_operator_reused=False,layout_reused=reused,batched_assembly=True,vector_validation=bool(vector_validation))
