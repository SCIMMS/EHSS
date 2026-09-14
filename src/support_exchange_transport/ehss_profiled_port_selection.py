"""Profile required physical suffixes and select missing entry ports by work.

The work score is measured box + sphere tests, not predicted wall-clock savings.
Only first ports can replace an unknown-first-port suffix at query entry; deeper
occupation events remain available as observations within selected cells.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import time
import numpy as np
from numba import njit
from .ehss_guarded_intrinsic_response import nearest_world,reflect
from .ehss_state import EHSSResult
from .ehss_material_collision_field import material_keys
from .ehss_material_port_learning import learn_missing_ports
from .ehss_sparse_path_response import readonly,record_from_paths


@njit(cache=False)
def profiled_trace(origins,directions,lasts,initial,cap,lo,hi,left,right,end,leaf_group,offsets,ids,centers,radii):
    p=origins.copy(); d=directions.copy(); counts=initial.copy(); history=np.full((len(p),cap),-1,dtype=np.int32)
    escaped=np.zeros(len(p),dtype=np.bool_); unresolved=escaped.copy()
    checks=np.zeros((len(p),3),dtype=np.int64); first_checks=checks.copy()
    for ray in range(len(p)):
        last=lasts[ray]; count=counts[ray]
        if initial[ray]==1: history[ray,0]=last
        while True:
            atom,distance=nearest_world(0,p[ray],d[ray],last,np.inf,-1,lo,hi,left,right,end,leaf_group,offsets,ids,centers,radii,checks[ray,:2])
            checks[ray,2]+=1
            if checks[ray,2]==1 and initial[ray]==0: first_checks[ray]=checks[ray]
            if atom<0: escaped[ray]=True; break
            if count>=cap: unresolved[ray]=True; break
            reflect(p[ray],d[ray],atom,distance,centers,radii)
            history[ray,count]=atom; count+=1; last=atom
        counts[ray]=count
    return p,d,escaped,unresolved,counts,history,checks,first_checks


@dataclass
class ProfiledResult(EHSSResult):
    ray_checks: np.ndarray
    first_checks: np.ndarray

    @property
    def suffix_checks(self): return self.ray_checks-self.first_checks


class ProfiledModel:
    def __init__(self,model): self.model=model
    def __getattr__(self,name): return getattr(self.model,name)
    def trace(self,source,max_bounces=128):
        if not isinstance(max_bounces,(int,np.integer)) or max_bounces<1 or np.any(source.initial_bounces>1): raise ValueError('Positive cap and first-reflected/unscattered sources required')
        if np.any(source.last_atom < -1) or np.any(source.last_atom>=len(self.spheres.radii)): raise ValueError('Invalid initial atom')
        t=self.tree; s=self.spheres; tick=time.perf_counter()
        p,d,e,u,n,h,checks,first=profiled_trace(source.origins,source.directions,source.last_atom,source.initial_bounces,max_bounces,
            t.lo,t.hi,t.left,t.right,t.end,t.leaf_group,t.offsets,t.ids,s.centers,s.radii)
        return ProfiledResult(p,d,e,u,n,h,source,dict(query_s=time.perf_counter()-tick,box_tests=int(checks[:,0].sum()),
            sphere_tests=int(checks[:,1].sum()),nearest_queries=int(checks[:,2].sum()),backend='Direct BVH with per-ray counters'),readonly(checks),readonly(first))


def rank_entry_ports(field,batch):
    if batch.parent_bank is not field.bank or batch.geometry!=field.geometry: raise ValueError('Capture does not belong to this field')
    table={}; total=np.zeros(3); mass_total=0.
    for result in batch.suffixes:
        if not isinstance(result,ProfiledResult): raise ValueError('Per-ray profiled suffixes required')
        source=result.source
        if np.any(source.initial_bounces!=1): raise ValueError('Captured suffixes must start after first reflection')
        keys=material_keys(source.origins,source.directions,source.last_atom,field.model.spheres.centers,field.bank.groups,field.rotations,field.bins)
        for ray,key in enumerate(keys):
            key=tuple(key); state=field.index.get(key)
            if state is not None and field.active[state]: raise ValueError('Capture contains a known first port')
            weight=float(source.weights[ray]/batch.source_count); work=result.ray_checks[ray].astype(float)*weight
            if key not in table: table[key]=dict(mass=0.,weighted_box_tests=0.,weighted_sphere_tests=0.,weighted_nearest_queries=0.,rays=0)
            row=table[key]; row['mass']+=weight; row['weighted_box_tests']+=work[0]; row['weighted_sphere_tests']+=work[1]; row['weighted_nearest_queries']+=work[2]; row['rays']+=1
            total+=work; mass_total+=weight
    return table,dict(entry_cells=len(table),captured_entry_mass=mass_total,weighted_box_tests=float(total[0]),weighted_sphere_tests=float(total[1]),weighted_nearest_queries=float(total[2]))


def learn_entry_ports(field,bound,observed,batch,max_ports=256,score='checks'):
    if score not in ('mass','checks'): raise ValueError('Entry mass or measured checks score required')
    if not isinstance(max_ports,int) or max_ports<0: raise ValueError('Nonnegative entry port budget required')
    tick=time.perf_counter(); table,totals=rank_entry_ports(field,batch)
    def priority(key):
        row=table[key]
        return row['mass'] if score=='mass' else row['weighted_box_tests']+row['weighted_sphere_tests']
    ranked=sorted(table,key=lambda key:(-priority(key),key)); selected=set(ranked[:max_ports]); dummy=len(field.active)
    # Eligibility view is used only by the existing learner, never by a solver.
    class EligibleIndex:
        def get(self,key): return field.index.get(key) if key in selected else dummy
    view=SimpleNamespace(bank=field.bank,geometry=field.geometry,bins=field.bins,rotations=field.rotations,index=EligibleIndex(),active=np.r_[field.active,True])
    learned,events,receipt=learn_missing_ports(view,bound,observed,batch,None)
    if receipt['selected_cells']!=len(selected): raise AssertionError('Selected entry cells were not preserved')
    picked={name:sum(table[key][name] for key in selected) for name in ('mass','weighted_box_tests','weighted_sphere_tests','weighted_nearest_queries')}
    receipt.update(entry_selection=score,entry_totals=totals,selected_entry_work=picked,
        ranked_entry_cells=[dict(key=list(key),score=priority(key),selected=key in selected,**table[key]) for key in ranked],
        selection_total_s=time.perf_counter()-tick,
        score_units='Input-mass weighted actual box + sphere tests with equal unit weights; not a wall-clock or net-savings model.' if score=='checks' else 'Captured first-port input mass; deeper occupation is not a selection score.')
    return learned,events,receipt


def covered_suffix_work(field,reference):
    """Held-out opportunity, counted once per physical ray, without new search."""
    if not isinstance(reference,ProfiledResult): raise ValueError('Profiled independent reference required')
    source=reference.source; sphere=field.model.spheres
    p,d,atoms,_=record_from_paths(source.origins,source.directions,source.weights,source.initial_bounces,
        reference.collider_ids,reference.bounces,sphere.centers,sphere.radii)
    hit=reference.bounces>0; starts=np.r_[0,np.cumsum(reference.bounces)[:-1]][hit]
    keys=material_keys(p[starts],d[starts],atoms[starts],sphere.centers,field.bank.groups,field.rotations,field.bins)
    known=np.zeros(len(source.weights),dtype=bool)
    for ray,key in zip(np.flatnonzero(hit),keys):
        state=field.index.get(tuple(key)); known[ray]=state is not None and field.active[state]
    work=reference.suffix_checks; weighted=work*source.weights[:,None]
    return dict(known_first_rays=int(known.sum()),known_mass=float(source.weights[known].sum()),
        total_weighted_suffix_checks=weighted.sum(axis=0).tolist(),covered_weighted_suffix_checks=weighted[known].sum(axis=0).tolist(),
        covered_unweighted_suffix_checks=work[known].sum(axis=0).tolist(),
        counts_are_opportunities_not_net_time_savings=True)
