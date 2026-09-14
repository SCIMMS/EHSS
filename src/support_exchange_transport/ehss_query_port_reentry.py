"""Route unknown first-port queries through physical prefixes into the field.

Seeds carry their actual arrival collision order and original incident moment.
Known arrivals consume one field collision, and handoff never implies escape.
The existing current-geometry field, first-hit search and first-port response
are unchanged. Reentry adds an empirical projection later along the query.
"""
import time
import numpy as np
from .ehss_batched_material_input import BatchedMaterialInput,assemble_input,physical_ledger
from .ehss_material_port_reentry import trace_port_prefix
from .ehss_empirical_field import geometry_key
from .ehss_empirical_bvh import first_collisions_bvh
from .ehss_material_collision_field import material_keys
from .ehss_field_condensation import flat_field_solve
from .ehss_sparse_path_response import readonly


class ReentryMaterialInput(BatchedMaterialInput):
    def solve(self,source,cap=128,engine=None,min_visits=1,allow_reentry=True):
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
        fallback_s=readout_s=reentry_seed_s=0.; fallback_tests=0; reentered=0; reentered_mass=0.
        fallback_ledger=np.zeros((cap+1,2)); physical_tail=0.
        reentry_orders=np.zeros(cap+1,dtype=np.int64); reentry_mass_by_order=np.zeros(cap+1)
        prefix_metrics=dict(sphere_tests=0,box_tests=0,nearest_queries=0,port_lookups=0)
        if len(fallback):
            start=time.perf_counter()
            # The table was validated by the parent constructor. A query visit
            # threshold applies to both initial selection and later handoff.
            active=readonly(fit.active&(fit.visits>=min_visits))
            prefix=trace_port_prefix(p[fallback],d[fallback],atoms[fallback],cap,spheres.centers,spheres.radii,
                tree.lo,tree.hi,tree.left,tree.right,tree.end,tree.leaf_group,tree.offsets,tree.ids,
                fit.bank.groups,fit.rotations,fit.bins,self.table,active,allow_reentry)
            _,out,escaped,unresolved,counts,_,ports,checks,queries,lookups=prefix
            fallback_s=time.perf_counter()-start; fallback_tests=int(checks[1])
            prefix_metrics=dict(sphere_tests=int(checks[1]),box_tests=int(checks[0]),nearest_queries=int(queries),port_lookups=int(lookups))
            if not np.all(escaped.astype(int)+unresolved+(ports>=0)==1): raise AssertionError('Ambiguous physical prefix termination')
            start=time.perf_counter()
            fallback_ledger=physical_ledger(source.weights[fallback],source.incoming[fallback],out,escaped,counts,cap)
            physical_tail=float(source.weights[fallback[unresolved]].sum()); readout_s=time.perf_counter()-start
            start=time.perf_counter(); arrived=ports>=0; rays=fallback[arrived]
            reentered=len(rays); reentered_mass=float(source.weights[rays].sum())
            if reentered:
                # counts includes the reflected arrival in physical coordinates.
                # Logical field cost for that same collision is still pending.
                seeds=np.r_[seeds,ports[arrived]]
                orders=np.r_[orders,counts[arrived]-1]
                appended=np.column_stack((source.weights[rays],source.weights[rays,None]*source.incoming[rays]))
                values=np.concatenate((values,appended),axis=0)
                reentry_orders=np.bincount(counts[arrived],minlength=cap+1)
                reentry_mass_by_order=np.bincount(counts[arrived],weights=source.weights[rays],minlength=cap+1)
            reentry_seed_s=time.perf_counter()-start
        result=flat_field_solve(k,seeds,orders,values,cap) if engine is None else engine.solve(seeds,orders,values,cap)
        ledger=np.array([[r['mass'],r['omega']] for r in result['order_ledger']]); ledger+=fallback_ledger; ledger[1]+=initial_escape
        ledger[0,0]+=source.weights[~hit].sum(); ledger[0,1]+=np.sum(source.weights[~hit]*(1-np.sum(source.incoming[~hit]*d[~hit],axis=1)))
        tail=result['residual_mass']+initial_tail+physical_tail
        if not np.isclose(ledger[:,0].sum()+tail,source.weights.sum(),rtol=1e-11,atol=1e-11): raise AssertionError('Full material field mass changed')
        return dict(omega=float(ledger[:,1].sum()),pa=float(source.weights[hit].sum()),escaped_mass=float(ledger[:,0].sum()),
            residual_mass=float(tail),tail_bound=float(2*tail),source_mass=float(source.weights.sum()),
            learned_mass=float(source.weights[selected].sum()),fallback_mass=float(source.weights[fallback].sum()),
            learned_rays=len(selected),fallback_rays=len(fallback),reentered_rays=reentered,reentered_mass=reentered_mass,
            transported_mass=float(source.weights[selected].sum())+reentered_mass,
            remaining_physical_mass=float(source.weights[fallback].sum())-reentered_mass,
            order_ledger=[dict(order=i,mass=float(v[0]),omega=float(v[1])) for i,v in enumerate(ledger)],
            first_s=first_s,seed_s=seed_s,field_s=result['query_s'],fallback_s=fallback_s,fallback_readout_s=readout_s,reentry_seed_s=reentry_seed_s,
            sphere_tests=int(counters[1]+fallback_tests),query_s=time.perf_counter()-tick,field=result,prefix_metrics=prefix_metrics,
            reentry_orders=reentry_orders,reentry_mass_by_order=reentry_mass_by_order,
            query_reentry_enabled=bool(allow_reentry),batched_input=True,native_fallback_readout=True)
