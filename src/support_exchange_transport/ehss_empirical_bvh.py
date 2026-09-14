"""BVH search adapter for a frozen empirical molecular field.

The historical all-atom implementation remains an independent regression
control. This adapter changes only the first collision and unknown-cell suffix
search; it does not learn, refine, or alter the empirical coefficients.
"""
import time
import numpy as np
from numba import njit
from .ehss_empirical_field import geometry_key,collision_keys,first_collisions
from .ehss_field_condensation import flat_field_solve
from .ehss_reference import trace
from .ehss_guarded_intrinsic_response import nearest_world,reflect
from .ehss_state import EHSSSource


@njit(cache=False)
def first_collisions_bvh(origins,directions,last_atoms,initial,centers,radii,
                         lo,hi,left,right,end,leaf_group,offsets,ids):
    points=origins.copy();outgoing=directions.copy();atoms=last_atoms.copy();counters=np.zeros(2,dtype=np.int64)
    for i in range(len(points)):
        if initial[i]==1:continue
        atom,distance=nearest_world(0,points[i],outgoing[i],-1,np.inf,-1,lo,hi,left,right,end,leaf_group,
            offsets,ids,centers,radii,counters)
        atoms[i]=atom
        if atom>=0:reflect(points[i],outgoing[i],atom,distance,centers,radii)
    return points,outgoing,atoms,counters


class BVHEmpiricalField:
    def __init__(self,field,model):
        self.field=field;self.model=model
        if geometry_key(model.spheres)!=field.geometry:raise ValueError('Geometry differs from the learned field')
        if not np.array_equal(model.tree.groups,field.groups):raise ValueError('Physical ownership differs from the learned field')

    def first(self,source,bvh=True):
        model=self.model;s=model.spheres;t=model.tree
        if bvh:
            p,d,atoms,counters=first_collisions_bvh(source.origins,source.directions,source.last_atom,source.initial_bounces,
                s.centers,s.radii,t.lo,t.hi,t.left,t.right,t.end,t.leaf_group,t.offsets,t.ids)
            return p,d,atoms,int(counters[1]),int(counters[0])
        p,d,atoms,tests=first_collisions(source.origins,source.directions,source.last_atom,source.initial_bounces,s.centers,s.radii)
        return p,d,atoms,int(tests),0

    def solve(self,source,cap=128,engine=None,min_visits=1,anchor_first_direction=True,first_bvh=True,fallback_bvh=True):
        # Keep validation within full-query timing, including the geometry hash.
        tick=time.perf_counter();fit=self.field;model=self.model;spheres=model.spheres
        if geometry_key(spheres)!=fit.geometry:raise ValueError('Geometry changed; empirical response must be rebuilt')
        if not isinstance(cap,int) or cap<1 or not isinstance(min_visits,int) or min_visits<1:
            raise ValueError('Positive cap and visit threshold required')
        if np.any((source.initial_bounces!=0)&(source.initial_bounces!=1)):
            raise ValueError('Uncollided or first-reflected sources required')
        if np.any(source.last_atom>=len(spheres.radii)) or np.any(source.last_atom < -1):raise ValueError('Invalid source atom identity')
        if engine is not None and engine.kernel is not fit.kernel:raise ValueError('Engine uses a different empirical kernel')
        first_tick=time.perf_counter();p,d,atoms,tests,boxes=self.first(source,first_bvh);first_s=time.perf_counter()-first_tick
        hit=atoms>=0;ids=np.flatnonzero(hit)
        keys=collision_keys(p[hit],d[hit],atoms[hit],np.ones(len(ids),dtype=np.int64),spheres.centers,fit.groups,fit.bins)
        selected=[];states=[];fallback=[]
        for ray,key in zip(ids,keys):
            state=fit.index.get(tuple(key))
            if state is None or fit.visits[state]<min_visits:fallback.append(ray)
            else:selected.append(ray);states.append(state)
        selected=np.array(selected,dtype=np.int64);fallback=np.array(fallback,dtype=np.int64)
        values=np.column_stack((source.weights[selected],source.weights[selected,None]*source.incoming[selected]))
        initial_escape=np.zeros(2);initial_tail=0.;orders=np.zeros(len(states),dtype=np.int64)
        if anchor_first_direction:
            next_states=[];next_values=[];k=fit.kernel
            for ray,state,value in zip(selected,states,values):
                e=k.escape[state,0];initial_escape+=e*np.array([value[0],value[0]-np.dot(value[1:],d[ray])])
                initial_tail+=value[0]*k.residual[state]
                for edge in range(k.starts[state],k.starts[state+1]):
                    next_states.append(k.targets[edge]);next_values.append(k.probabilities[edge]*value)
            states=next_states;values=np.array(next_values).reshape(-1,4);orders=np.ones(len(states),dtype=np.int64)
        seed_s=time.perf_counter()-tick
        result=flat_field_solve(fit.kernel,states,orders,values,cap) if engine is None else engine.solve(states,orders,values,cap)
        ledger=np.array([[r['mass'],r['omega']] for r in result['order_ledger']]);ledger[1]+=initial_escape
        ledger[0,0]+=source.weights[~hit].sum()
        ledger[0,1]+=np.sum(source.weights[~hit]*(1-np.sum(source.incoming[~hit]*d[~hit],axis=1)))
        tail=result['residual_mass']+initial_tail;fallback_s=0.;fallback_tests=0;fallback_boxes=0
        if len(fallback):
            sub=EHSSSource(p[fallback],d[fallback],source.incoming[fallback],source.weights[fallback],atoms[fallback],
                np.ones(len(fallback),dtype=np.int64),'exact unknown-cell suffix')
            fallback_tick=time.perf_counter()
            exact=model.trace(sub,cap) if fallback_bvh else trace(spheres,sub,cap)
            fallback_s=time.perf_counter()-fallback_tick
            fallback_tests=exact.metrics['sphere_tests'];fallback_boxes=exact.metrics.get('box_tests',0)
            for row in exact.order_ledger():ledger[row['order']]+=(row['mass'],row['omega'])
            tail+=exact.residual_mass
        if not np.isclose(ledger[:,0].sum()+tail,source.weights.sum(),rtol=1e-11,atol=1e-11):raise AssertionError('Hybrid mass changed')
        return dict(omega=float(ledger[:,1].sum()),pa=float(source.weights[hit].sum()),residual_mass=float(tail),tail_bound=float(2*tail),
            escaped_mass=float(ledger[:,0].sum()),source_mass=float(source.weights.sum()),
            order_ledger=[dict(order=i,mass=float(v[0]),omega=float(v[1])) for i,v in enumerate(ledger)],
            learned_mass=float(source.weights[selected].sum()),fallback_mass=float(source.weights[fallback].sum()),
            learned_rays=len(selected),fallback_rays=len(fallback),seed_s=seed_s,first_s=first_s,
            classify_and_seed_s=seed_s-first_s,field_s=result['query_s'],fallback_s=fallback_s,
            first_sphere_tests=tests,fallback_sphere_tests=fallback_tests,sphere_tests=tests+fallback_tests,
            box_tests=boxes+fallback_boxes,query_s=time.perf_counter()-tick,field=result)
