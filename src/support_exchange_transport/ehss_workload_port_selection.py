"""Choose entry cells by raw measured workload without changing physics weights."""
from types import SimpleNamespace
import time
import numpy as np
from .ehss_profiled_port_selection import rank_entry_ports
from .ehss_material_collision_field import material_keys
from .ehss_material_port_learning import learn_missing_ports


def learn_workload_ports(field,bound,observed,batch,max_ports=256):
    if not isinstance(max_ports,int) or max_ports<0: raise ValueError('Nonnegative entry port budget required')
    tick=time.perf_counter(); table,weighted_totals=rank_entry_ports(field,batch)
    for row in table.values(): row.update(box_tests=0,sphere_tests=0,nearest_queries=0)
    for result in batch.suffixes:
        source=result.source
        keys=material_keys(source.origins,source.directions,source.last_atom,field.model.spheres.centers,field.bank.groups,field.rotations,field.bins)
        for key,checks in zip(keys,result.ray_checks):
            row=table[tuple(key)]
            for name,value in zip(('box_tests','sphere_tests','nearest_queries'),checks): row[name]+=int(value)
    score=lambda key:table[key]['box_tests']+table[key]['sphere_tests']
    ranked=sorted(table,key=lambda key:(-score(key),key)); selected=set(ranked[:max_ports]); dummy=len(field.active)
    class EligibleIndex:
        def get(self,key): return field.index.get(key) if key in selected else dummy
    view=SimpleNamespace(bank=field.bank,geometry=field.geometry,bins=field.bins,rotations=field.rotations,index=EligibleIndex(),active=np.r_[field.active,True])
    # Selection uses raw work; the original weighted observations enter the bank.
    learned,events,receipt=learn_missing_ports(view,bound,observed,batch,None)
    if receipt['selected_cells']!=len(selected): raise AssertionError('Selected entry cells were not preserved')
    metrics=('mass','weighted_box_tests','weighted_sphere_tests','weighted_nearest_queries','box_tests','sphere_tests','nearest_queries')
    receipt.update(entry_selection='raw_checks',entry_totals=dict(weighted_totals,**{name:sum(r[name] for r in table.values()) for name in ('box_tests','sphere_tests','nearest_queries')}),
        selected_entry_work={name:sum(table[key][name] for key in selected) for name in metrics},
        ranked_entry_cells=[dict(key=list(key),score=score(key),selected=key in selected,**table[key]) for key in ranked],
        selection_total_s=time.perf_counter()-tick,physical_observation_weights_unchanged=True,
        score_units='Raw box + sphere tests summed once per captured ray. Equal check weights; observed workload proxy, not net wall-clock savings.')
    return learned,events,receipt
