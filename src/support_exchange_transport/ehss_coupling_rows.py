"""Exact selective material-row assembly with stable suffix-entry identities.

Current first reflections, active ports and physical suffix traces remain inputs.
Only aggregation of unchanged material-state contributions is skipped. A suffix
entry is identified by its material row, never by its transient dense index.
"""
from dataclasses import dataclass
import copy,time
import numpy as np
from numba import njit
from .ehss_batched_coupling import MaterialPortLayout,port_rows,readonly


@njit(cache=True)
def contribution_gate(ids,valid,atoms,terminal,target,missing,directions,old_kind,old_target,old_directions,size):
    kind=np.zeros(len(ids),dtype=np.int64); semantic=np.full(len(ids),-1,dtype=np.int64)
    changed=np.zeros(len(ids),dtype=np.bool_); dirty=np.zeros(size,dtype=np.bool_)
    for row in range(len(ids)):
        if valid[row]:
            if atoms[row]>=0:
                kind[row]=3; semantic[row]=size+row if missing[row] else target[row]
            elif terminal[row]==-1: kind[row]=1
            else: kind[row]=2
        delta=kind[row]!=old_kind[row] or semantic[row]!=old_target[row]
        if kind[row]==1:
            for c in range(3):
                if directions[row,c]!=old_directions[row,c]: delta=True
        changed[row]=delta
        if delta: dirty[ids[row]]=True
    return kind,semantic,changed,dirty


@njit(cache=True)
def merge_rows(dirty,old_starts,old_targets,old_prob,old_escape,old_res,new_starts,new_targets,new_prob,new_escape,new_res,virtual):
    size=len(dirty); starts=np.zeros(size+1,dtype=np.int64)
    for state in range(size):
        starts[state+1]=starts[state]+(new_starts[state+1]-new_starts[state] if dirty[state] else old_starts[state+1]-old_starts[state])
    targets=np.empty(starts[-1],dtype=np.int64); semantic=np.empty_like(targets); prob=np.empty(starts[-1]); escape=np.empty((size,4)); residual=np.empty(size)
    for state in range(size):
        source_starts=new_starts if dirty[state] else old_starts
        source_targets=new_targets if dirty[state] else old_targets
        source_prob=new_prob if dirty[state] else old_prob
        source_escape=new_escape if dirty[state] else old_escape
        source_res=new_res if dirty[state] else old_res
        for j in range(source_starts[state],source_starts[state+1]):
            out=starts[state]+j-source_starts[state]; token=source_targets[j]
            semantic[out]=token; prob[out]=source_prob[j]
            if token>=size:
                if virtual[token-size]<size: raise ValueError('Retained suffix entry disappeared')
                targets[out]=virtual[token-size]
            else: targets[out]=token
        for c in range(4): escape[state,c]=source_escape[state,c]
        residual[state]=source_res[state]
    return starts,targets,prob,escape,residual,semantic


@dataclass(frozen=True)
class CouplingRows:
    kind: np.ndarray
    target: np.ndarray
    directions: np.ndarray
    starts: np.ndarray
    targets: np.ndarray
    probabilities: np.ndarray
    escape: np.ndarray
    residual: np.ndarray


class CouplingRowPool:
    def __init__(self,layout,force_refresh=False):
        if not isinstance(layout,MaterialPortLayout): raise TypeError('MaterialPortLayout required')
        self.layout=layout; self.bank=layout.bank; self.force_refresh=bool(force_refresh)
        self.state=None; self.generation=0; self.receipt={}

    def snapshot(self):
        result=copy.copy(self); result.receipt={}; return result

    def assemble(self,layout,valid,atoms,target,terminal,directions,mass,missing):
        tick=time.perf_counter()
        if layout is not self.layout: raise ValueError('Coupling pool belongs to another layout')
        ids=layout.ids; weights=self.bank.mass; n=len(ids); size=len(mass)
        if size!=len(layout.keys) or any(np.shape(a)!=(n,) for a in (valid,atoms,target,terminal)) or np.shape(directions)!=(n,3):
            raise ValueError('Current contributions differ from material layout')
        missing_mask=np.zeros(n,dtype=bool); missing_mask[missing]=True
        if np.any(missing_mask&(~valid|(atoms<0))) or np.any(target[missing]<size): raise ValueError('Invalid current suffix entries')
        if not np.isfinite(directions).all(): raise ValueError('Finite directions required')
        old=self.state
        if old is None:
            old=CouplingRows(np.full(n,-1,dtype=np.int64),np.full(n,-1,dtype=np.int64),np.zeros((n,3)),
                np.zeros(size+1,dtype=np.int64),np.empty(0,dtype=np.int64),np.empty(0),np.zeros((size,4)),np.ones(size))
        kind,semantic,changed,dirty=contribution_gate(ids,valid,atoms,terminal,target,missing_mask,directions,old.kind,old.target,old.directions,size)
        if self.state is None: dirty[:]=True
        potential_dirty=dirty.copy()
        if self.force_refresh: dirty[:]=True
        gate_s=time.perf_counter()-tick; start=time.perf_counter()
        selected=np.flatnonzero(dirty[ids]); selected_ids=ids[selected]; selected_target=semantic[selected]
        moving=np.flatnonzero(valid[selected]&(atoms[selected]>=0))
        order=moving[np.lexsort((selected[moving],selected_target[moving],selected_ids[moving]))]
        fresh=port_rows(selected_ids,weights[selected],valid[selected],atoms[selected],selected_target,
            terminal[selected],directions[selected],mass,order)
        aggregate_s=time.perf_counter()-start; start=time.perf_counter()
        virtual=np.full(n,-1,dtype=np.int64); virtual[missing]=target[missing]
        starts,targets,prob,escape,residual,tokens=merge_rows(dirty,old.starts,old.targets,old.probabilities,old.escape,old.residual,*fresh,virtual)
        pack_s=time.perf_counter()-start; start=time.perf_counter()
        self.state=CouplingRows(*(readonly(a) for a in (kind,semantic,directions.copy(),starts,tokens,prob,escape,residual)))
        self.generation+=1; store_s=time.perf_counter()-start
        self.receipt=dict(generation=self.generation,material_rows=n,material_ports=size,active_ports=int(np.count_nonzero(mass>0)),
            changed_input_rows=int(changed.sum()),dirty_ports=int(dirty.sum()),reused_ports=int(np.count_nonzero(~dirty)),
            dirty_active_ports=int(np.count_nonzero(dirty&(mass>0))),reused_active_ports=int(np.count_nonzero(~dirty&(mass>0))),
            potential_dirty_ports=int(potential_dirty.sum()),aggregated_input_rows=len(selected),
            reused_edges=int(np.diff(old.starts)[~dirty].sum()),current_edges=len(targets),
            gate_s=gate_s,aggregate_s=aggregate_s,pack_s=pack_s,store_s=store_s,total_s=time.perf_counter()-tick,
            force_refresh=self.force_refresh,dirty_mask=dirty,changed_input_mask=changed,
            retained_array_bytes=sum(a.nbytes for a in vars(self.state).values()),
            stable_suffix_entries=True,current_occupation_recomputed=True,current_reflections_recomputed=True,
            physical_suffixes_recomputed=True,extra_approximation=False,global_packing_still_required=True)
        return starts,targets,prob,escape,residual
