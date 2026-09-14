"""Finite-wave hierarchy transport with a remaining-contribution budget.

Positive transport bounds uncomputed angular contribution by twice its mass.
The pending mass includes both active states and every not-yet-injected seed.
Truncated mass is recorded separately from physical local and collision-cap
residuals. This bound is for additional truncation, not projection bias.
"""
import time
import numpy as np
from numba import njit
from .ehss_field_condensation import _field,_result
from .ehss_semantic_hierarchy import SemanticHierarchyField


@njit(cache=True)
def apply_mass_bounded(states,orders,values,cap,escape,residual,blocks,omega_budget):
    current=np.zeros((len(residual),4)); outgoing=np.zeros_like(current)
    mass=np.zeros(cap+1); omega=np.zeros(cap+1)
    # Reverse summation avoids subtracting a large early seed from a total
    # and accidentally erasing a much smaller late seed through cancellation.
    future=np.zeros(cap+2)
    for j in range(len(states)): future[orders[j]]+=values[j,0]
    for order in range(cap-1,-1,-1): future[order]+=future[order+1]
    local=cap_tail=truncated=0.; waves=applications=0; stop_order=cap
    stopped_active=stopped_future=0.; reason=0
    for order in range(cap+1):
        for j in range(len(states)):
            if orders[j]==order:
                for c in range(4): current[states[j],c]+=values[j,c]
        current_mass=0.
        for state in range(len(residual)): current_mass+=current[state,0]
        if order==cap:
            cap_tail=current_mass; break
        pending=current_mass+future[order+1]
        if pending==0.:
            stop_order=order; reason=1; break
        if omega_budget>0. and 2*pending<=omega_budget:
            truncated=pending; stopped_active=current_mass; stopped_future=future[order+1]
            stop_order=order; reason=2; break
        if current_mass==0.: continue
        waves+=1
        for state in range(len(residual)):
            m=current[state,0]*escape[state,0]; mass[order+1]+=m
            for c in range(1,4): m-=current[state,c]*escape[state,c]
            omega[order+1]+=m; local+=current[state,0]*residual[state]
            for c in range(4): outgoing[state,c]=0.
        for block in blocks:
            inputs,outputs,indptr,indices,probabilities=block
            active=False
            for state in inputs:
                if current[state,0]>0: active=True; break
            if not active: continue
            applications+=1
            for row in range(len(outputs)):
                for c in range(4):
                    v=0.
                    for edge in range(indptr[row],indptr[row+1]): v+=probabilities[edge]*current[inputs[indices[edge]],c]
                    outgoing[outputs[row],c]+=v
        current,outgoing=outgoing,current
    return mass,omega,local,cap_tail,truncated,waves,applications,stop_order,stopped_active,stopped_future,reason


class MassBoundedHierarchyField:
    def __init__(self,engine,omega_budget_fraction=0.):
        if not isinstance(engine,SemanticHierarchyField): raise ValueError('Prepared semantic hierarchy required')
        fraction=float(omega_budget_fraction)
        if not np.isfinite(fraction) or not 0<=fraction<=2: raise ValueError('Finite contribution budget fraction in [0,2] required')
        self.engine=engine; self.kernel=engine.kernel; self.omega_budget_fraction=fraction
        self.receipt=dict(shared_payloads=True,omega_budget_fraction=fraction,
            normalization='Field seed mass; not a fraction of exact EHSS',future_seed_mass_included=True)

    def solve(self,states,orders,values,max_bounces=64):
        tick=time.perf_counter(); k=self.kernel
        if self.engine.kernel is not k: raise ValueError('Prepared hierarchy kernel changed')
        field=_field(states,orders,values,len(k.owners),max_bounces)
        ids=np.array([s for s,o in field],dtype=np.int64); orders=np.array([o for s,o in field],dtype=np.int64)
        values=np.array(list(field.values()),dtype=float).reshape(-1,4); total=sum(v[0] for v in field.values())
        budget=self.omega_budget_fraction*total
        m,w,local,cap_tail,truncated,waves,applications,stop_order,active,future,reason=apply_mass_bounded(
            ids,orders,values,max_bounces,k.escape,k.residual,self.engine.blocks,budget)
        result=_result(m,w,local,cap_tail+truncated,total,query_s=time.perf_counter()-tick,
            hierarchy_nodes=self.engine.receipt['hierarchy_nodes'],active_blocks=len(self.engine.blocks),
            sparse_block_applications=applications,collision_waves=waves,physical_primitive_tests=0,
            semantic_hierarchy=True,mass_bounded=True,omega_budget_fraction=self.omega_budget_fraction,
            omega_budget=float(budget),additional_truncation_bound=float(2*truncated),truncation_tail_mass=float(truncated),
            stop_order=int(stop_order),stop_reason=('collision_cap','exhausted','contribution_budget')[reason],
            active_mass_at_stop=float(active),future_seed_mass_at_stop=float(future))
        result['bounce_tail_mass']=float(cap_tail)
        return result
