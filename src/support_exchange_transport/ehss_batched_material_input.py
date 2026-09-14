"""Batched material-port lookup, input assembly and physical suffix readout.

The original field supplies all geometry, material coefficients and physical
fallback queries. Sorted seven-dimensional keys replace Python tuple lookup;
every direction component and original incident weight remains intact.
"""
import time
import numpy as np
from numba import njit
from .ehss_empirical_field import geometry_key
from .ehss_empirical_bvh import first_collisions_bvh
from .ehss_material_collision_field import material_keys
from .ehss_field_condensation import flat_field_solve
from .ehss_sparse_path_response import readonly
from .ehss_state import EHSSSource


@njit(cache=True)
def assemble_input(keys, ray_ids, table, active, visits, min_visits,
                   incoming, directions, weights, starts, targets, probabilities, escape, residual):
    states=np.empty(len(keys),dtype=np.int64)
    selected=np.empty(len(keys),dtype=np.int64); fallback=selected.copy()
    selected_count=0; fallback_count=0; edges=0
    for row in range(len(keys)):
        lo=0; hi=len(table)
        while lo<hi:
            mid=(lo+hi)//2; compare=0
            for c in range(7):
                if table[mid,c]<keys[row,c]: compare=-1; break
                if table[mid,c]>keys[row,c]: compare=1; break
            if compare<0: lo=mid+1
            else: hi=mid
        found=lo<len(table)
        if found:
            for c in range(7):
                if table[lo,c]!=keys[row,c]: found=False; break
        if found: found=active[lo] and visits[lo]>=min_visits
        if found:
            selected[selected_count]=ray_ids[row]; states[selected_count]=lo; selected_count+=1
            edges+=starts[lo+1]-starts[lo]
        else: fallback[fallback_count]=ray_ids[row]; fallback_count+=1
    seeds=np.empty(edges,dtype=np.int64); values=np.empty((edges,4))
    first_escape=np.zeros(2); first_tail=0.; cursor=0
    for j in range(selected_count):
        ray=selected[j]; state=states[j]; weight=weights[ray]
        dot=0.
        for c in range(3): dot+=(weight*incoming[ray,c])*directions[ray,c]
        first_escape[0]+=escape[state,0]*weight
        first_escape[1]+=escape[state,0]*(weight-dot)
        first_tail+=weight*residual[state]
        for edge in range(starts[state],starts[state+1]):
            seeds[cursor]=targets[edge]; p=probabilities[edge]
            values[cursor,0]=p*weight
            for c in range(3): values[cursor,c+1]=p*(weight*incoming[ray,c])
            cursor+=1
    return selected[:selected_count],fallback[:fallback_count],seeds,values,first_escape,first_tail


@njit(cache=True)
def physical_ledger(weights, incoming, outgoing, escaped, bounces, cap):
    ledger=np.zeros((cap+1,2))
    for ray in range(len(weights)):
        if not escaped[ray]: continue
        order=bounces[ray]
        if order<0 or order>cap: raise ValueError('Physical collision count exceeds ledger budget')
        dot=0.
        for c in range(3): dot+=incoming[ray,c]*outgoing[ray,c]
        angle=min(2.,max(0.,1.-dot))
        ledger[order,0]+=weights[ray]; ledger[order,1]+=weights[ray]*angle
    return ledger


class BatchedMaterialInput:
    def __init__(self,field):
        tick=time.perf_counter(); self.field=field; self.kernel=field.kernel
        if field.keys!=tuple(sorted(field.keys)): raise ValueError('Sorted material keys required')
        self.table=readonly(np.asarray(field.keys,dtype=np.int64).reshape(-1,7))
        self.receipt=dict(prepare_s=time.perf_counter()-tick,key_array_bytes=self.table.nbytes,
            physical_weights_unchanged=True,material_projection_unchanged=True,
            first_collision_search_unchanged=True,jit_included_in_prepare=False)

    def solve(self,source,cap=128,engine=None,min_visits=1,native_readout=True):
        tick=time.perf_counter(); fit=self.field; spheres=fit.model.spheres; tree=fit.model.tree; k=fit.kernel
        if k is not self.kernel: raise ValueError('Field kernel changed; rebuild input adapter')
        if geometry_key(spheres)!=fit.geometry: raise ValueError('Geometry changed; rebuild coupling')
        if not isinstance(cap,int) or not 1<=cap<=fit.cap: raise ValueError('Cap exceeds compiled physical suffix budget')
        if not isinstance(min_visits,int) or min_visits<1: raise ValueError('Positive visit threshold required')
        if np.any((source.initial_bounces!=0)&(source.initial_bounces!=1)): raise ValueError('Uncollided or first-reflected sources required')
        if np.any(source.last_atom>=len(spheres.radii)) or np.any(source.last_atom < -1): raise ValueError('Invalid source atom')
        if engine is not None and engine.kernel is not k: raise ValueError('Different field kernel')
        start=time.perf_counter()
        p,d,atoms,counters=first_collisions_bvh(source.origins,source.directions,source.last_atom,source.initial_bounces,
            spheres.centers,spheres.radii,tree.lo,tree.hi,tree.left,tree.right,tree.end,tree.leaf_group,tree.offsets,tree.ids)
        first_s=time.perf_counter()-start; hit=atoms>=0; ids=np.flatnonzero(hit)
        keys=material_keys(p[hit],d[hit],atoms[hit],spheres.centers,fit.bank.groups,fit.rotations,fit.bins)
        selected,fallback,seeds,values,initial_escape,initial_tail=assemble_input(keys,ids,self.table,fit.active,fit.visits,min_visits,
            source.incoming,d,source.weights,k.starts,k.targets,k.probabilities,k.escape,k.residual)
        orders=np.ones(len(seeds),dtype=np.int64); seed_s=time.perf_counter()-tick
        result=flat_field_solve(k,seeds,orders,values,cap) if engine is None else engine.solve(seeds,orders,values,cap)
        ledger=np.array([[r['mass'],r['omega']] for r in result['order_ledger']]); ledger[1]+=initial_escape
        ledger[0,0]+=source.weights[~hit].sum(); ledger[0,1]+=np.sum(source.weights[~hit]*(1-np.sum(source.incoming[~hit]*d[~hit],axis=1)))
        tail=result['residual_mass']+initial_tail; fallback_s=0.; fallback_tests=0; readout_s=0.
        if len(fallback):
            sub=EHSSSource(p[fallback],d[fallback],source.incoming[fallback],source.weights[fallback],atoms[fallback],
                np.ones(len(fallback),dtype=np.int64),'unknown first material port')
            start=time.perf_counter(); exact=fit.model.trace(sub,cap); fallback_s=time.perf_counter()-start
            fallback_tests=exact.metrics['sphere_tests']; start=time.perf_counter()
            if native_readout: ledger+=physical_ledger(sub.weights,sub.incoming,exact.outgoing,exact.escaped,exact.bounces,cap)
            else:
                for row in exact.order_ledger(): ledger[row['order']]+=(row['mass'],row['omega'])
            tail+=exact.residual_mass; readout_s=time.perf_counter()-start
        if not np.isclose(ledger[:,0].sum()+tail,source.weights.sum(),rtol=1e-11,atol=1e-11): raise AssertionError('Full material field mass changed')
        return dict(omega=float(ledger[:,1].sum()),pa=float(source.weights[hit].sum()),escaped_mass=float(ledger[:,0].sum()),
            residual_mass=float(tail),tail_bound=float(2*tail),source_mass=float(source.weights.sum()),
            learned_mass=float(source.weights[selected].sum()),fallback_mass=float(source.weights[fallback].sum()),
            learned_rays=len(selected),fallback_rays=len(fallback),
            order_ledger=[dict(order=i,mass=float(v[0]),omega=float(v[1])) for i,v in enumerate(ledger)],
            first_s=first_s,seed_s=seed_s,field_s=result['query_s'],fallback_s=fallback_s,fallback_readout_s=readout_s,
            sphere_tests=int(counters[1]+fallback_tests),query_s=time.perf_counter()-tick,field=result,
            batched_input=True,native_fallback_readout=bool(native_readout))
