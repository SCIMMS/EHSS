"""Batched coupling with an optional per-material-row prefix refresh pool.

Assembly remains the audited batched algorithm. Positive motion gates add a
separate response reuse approximation, measured against full current refresh.
"""
import time
import numpy as np
from .ehss_batched_coupling import (MaterialCollisionField,MaterialPortLayout,material_rotations,material_keys,
    first_observed_successors,lookup_targets,virtual_arrays,port_rows,BatchedScatteringKernel,ScatteringKernel,
    geometry_key,readonly,EHSSSource,trace_to_material_port)

from .ehss_atomic_material_axes import material_keys,trace_to_material_port

class AlignedMaterialCollisionField(MaterialCollisionField):
    def __init__(self,bound,observed,physical_model,bins=2,checked_observations=False,allow_reentry=True,layout=None,vector_validation=True,prefix_pool=None):
        tick=time.perf_counter(); bank=bound.bank; spheres=physical_model.spheres; stages={}
        if type(bins) is not int or bins<1: raise ValueError('Positive bins required')
        if geometry_key(bound.model.spheres)!=geometry_key(spheres): raise ValueError('Observation geometry differs from physical model')
        if observed['history'].shape!=(len(bank.mass),bank.cap) or not np.array_equal(observed['history'][:,0],bank.lasts): raise ValueError('Observation rows differ from material bank')
        self.bank=bank; self.model=physical_model; self.bins=bins; self.cap=bank.cap; self.geometry=geometry_key(spheres)
        if prefix_pool is not None and prefix_pool.bank is not bank: raise ValueError('Prefix pool belongs to another bank')
        start=time.perf_counter(); self.rotations=bound.atom_rotations; stages['rotation_s']=time.perf_counter()-start
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
            if prefix_pool is None:
                prefix=trace_to_material_port(physical_model,sub,self.keys,self.active,bank.groups,self.rotations,bins,bank.cap,allow_reentry)
            else:
                prefix=prefix_pool.trace(missing,physical_model,sub,self.keys,self.active,bank.groups,self.rotations,bins,bank.cap,allow_reentry,parent_rotations=bound.parent_rotations)
            suffix_s=time.perf_counter()-trace_start; suffix_metrics=prefix['metrics']
        if not len(missing) and prefix_pool is not None:
            prefix_pool.records={}; prefix_pool.last_result=None; prefix_pool.generation+=1; prefix_pool.receipt=dict(rows=0,reused_rows=0,refreshed_rows=0,gate_s=0.,trace_s=0.,store_s=0.,total_s=0.,failures={})
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

        self.receipt['prefix_refresh']=None if prefix_pool is None else prefix_pool.receipt.copy()
        self.receipt['current_prefix_reused']=prefix_pool is not None and prefix_pool.receipt.get('reused_rows',0)>0

    def solve(self,source,cap=128,engine=None,min_visits=1,allow_reentry=True):
        from .ehss_aligned_material_input import AlignedMaterialInput
        return AlignedMaterialInput(self).solve(source,cap,engine,min_visits,allow_reentry)
