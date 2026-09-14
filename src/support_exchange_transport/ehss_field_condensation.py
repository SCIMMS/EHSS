"""Positive boundary-field first-exit responses composed bottom-up.

The finite collision budget is part of every response key. Each elementary
response consumes at least one collision; zero-collision through closure must
be performed upstream. No new spatial/angular projection is introduced here.
"""
from dataclasses import dataclass
from types import MappingProxyType
import hashlib
import time
import numpy as np
from .ehss_sparse_path_response import readonly


class ScatteringKernel:
    """Row transition probabilities and escaped direction moments.

All outcomes of a row share its positive collision cost. Escape moments are
sum(probability * final physical direction), not normalized average normals.
Residual probability represents an unresolved local response, never escape.
"""
    def __init__(self, owners, costs, starts, targets, probabilities, escape, residual):
        values = (owners, costs, starts, targets, probabilities, escape, residual)
        dtypes = (np.int64, np.int64, np.int64, np.int64, float, float, float)
        self.owners, self.costs, self.starts, self.targets, self.probabilities, self.escape, self.residual = [readonly(np.array(v, dtype=d, copy=True)) for v,d in zip(values,dtypes)]
        n = len(self.owners)
        if self.costs.shape != (n,) or self.starts.shape != (n+1,) or self.escape.shape != (n,4) or self.residual.shape != (n,):
            raise ValueError('Invalid kernel shapes')
        if np.any(self.costs < 1): raise ValueError('Positive collision costs required after through closure')
        if self.starts[0] != 0 or np.any(np.diff(self.starts)<0) or self.starts[-1] != len(self.targets) or len(self.targets)!=len(self.probabilities):
            raise ValueError('Invalid sparse transition index')
        if np.any(self.targets<0) or np.any(self.targets>=n): raise ValueError('Invalid target state')
        for array in (self.probabilities,self.escape,self.residual):
            if not np.isfinite(array).all(): raise ValueError('Finite coefficients required')
        if np.any(self.probabilities<0) or np.any(self.escape[:,0]<0) or np.any(self.residual<0): raise ValueError('Nonnegative outcome probabilities required')
        if np.any(np.linalg.norm(self.escape[:,1:],axis=1)>self.escape[:,0]+1e-12): raise ValueError('Escape direction moment exceeds mass')
        for i in range(n):
            if not np.isclose(self.probabilities[self.starts[i]:self.starts[i+1]].sum()+self.escape[i,0]+self.residual[i],1.,rtol=0,atol=1e-12):
                raise ValueError('Each row must conserve probability')

    @classmethod
    def from_compiled(cls, compiled):
        if compiled.through_mode != 'preserve': raise ValueError('Continuous-through closure required')
        ids = np.flatnonzero((compiled.bounces>0)|compiled.stopped)
        inverse = np.full(compiled.states,-1,dtype=np.int64); inverse[ids]=np.arange(len(ids))
        targets=[]; starts=[0]; escape=np.zeros((len(ids),4)); residual=np.zeros(len(ids))
        for i,state in enumerate(ids):
            if compiled.stopped[state]: residual[i]=1.
            elif compiled.successor[state]<0: escape[i]=np.r_[1.,compiled.physical_outgoing[state]]
            else:
                target=inverse[compiled.successor[state]]
                if target<0: raise ValueError('Through closure targets a non-response state')
                targets.append(target)
            starts.append(len(targets))
        result=cls(ids//compiled.grid.size,compiled.bounces[ids],starts,targets,np.ones(len(targets)),escape,residual)
        result.original_ids=readonly(ids); result.original_to_state=readonly(inverse)
        return result


@dataclass(frozen=True)
class FieldNode:
    name: str
    owners: frozenset
    children: tuple = ()

    def __post_init__(self):
        object.__setattr__(self,'owners',frozenset(self.owners)); object.__setattr__(self,'children',tuple(self.children))
        if not self.owners and self.children: raise ValueError('Empty response domain cannot have children')
        if self.children:
            merged=set()
            for child in self.children:
                if merged & child.owners: raise ValueError('Children overlap in physical ownership')
                merged.update(child.owners)
            if merged != self.owners: raise ValueError('Children must partition parent ownership')


@dataclass(frozen=True)
class FirstExit:
    exits: object  # (target state, consumed collisions) -> probability
    escape: object  # consumed collisions -> [probability, physical direction moment]
    local_tail: float
    budget_tail: float


class ResponsePool:
    def __init__(self): self.cache={}


def _field(states, orders, values, count, cap):
    states=np.asarray(states,dtype=np.int64); orders=np.asarray(orders,dtype=np.int64); values=np.asarray(values,dtype=float)
    if states.ndim!=1 or orders.shape!=states.shape or values.shape!=(len(states),4): raise ValueError('Invalid boundary field shape')
    if not isinstance(cap,int) or cap<1 or np.any(orders<0) or np.any(orders>cap): raise ValueError('Invalid collision budget')
    if np.any(states<0) or np.any(states>=count): raise ValueError('Invalid field state')
    if not np.isfinite(values).all() or np.any(values[:,0]<0) or np.any(np.linalg.norm(values[:,1:],axis=1)>values[:,0]+1e-12):
        raise ValueError('Field must contain mass and its bounded incoming direction moment')
    merged={}
    for state,order,value in zip(states,orders,values):
        key=(int(state),int(order))
        if key not in merged: merged[key]=value.copy()
        else: merged[key]+=value
    return merged


def _result(mass,omega,local,budget,total,**metrics):
    if not np.isclose(mass.sum()+local+budget,total,rtol=1e-11,atol=1e-12): raise AssertionError('Field mass balance changed')
    if np.any(omega < -1e-10*max(1.,total)) or np.any(omega > 2*mass+1e-10*max(1.,total)): raise AssertionError('Physical direction readout bound violated')
    return dict(omega=float(omega.sum()),escaped_mass=float(mass.sum()),residual_mass=float(local+budget),
        local_tail_mass=float(local),bounce_tail_mass=float(budget),tail_bound=float(2*(local+budget)),source_mass=float(total),
        order_ledger=[dict(order=i,mass=float(m),omega=float(w)) for i,(m,w) in enumerate(zip(mass,omega))],**metrics)


class CondensedField:
    def __init__(self,kernel,root,pool=None):
        tick=time.perf_counter();self.kernel=kernel;self.root=root;self.pool=ResponsePool() if pool is None else pool
        if root.owners != frozenset(map(int,kernel.owners)): raise ValueError('Root must cover every physical owner')
        self.signatures={};self.child_for={};self.nodes={}
        def visit(node):
            if node.name in self.nodes: raise ValueError('Unique hierarchy node names required')
            self.nodes[node.name]=node;h=hashlib.sha256();rows=np.flatnonzero(np.isin(kernel.owners,list(node.owners)))
            h.update(np.array(sorted(node.owners),dtype=np.int64).tobytes())
            for state in rows:
                h.update(np.array([state,kernel.costs[state]],dtype=np.int64).tobytes())
                lo,hi=kernel.starts[state:state+2]
                h.update(np.array([hi-lo],dtype=np.int64).tobytes());h.update(kernel.targets[lo:hi].tobytes());h.update(kernel.probabilities[lo:hi].tobytes())
                h.update(kernel.escape[state].tobytes());h.update(kernel.residual[state:state+1].tobytes())
            self.signatures[node.name]=h.hexdigest()
            self.child_for[node.name]={owner:child for child in node.children for owner in child.owners}
            for child in node.children:visit(child)
        visit(root)
        self.stats={name:dict(cache_hits=0,compiled_columns=0,elementary_rows=0,child_applications=0) for name in self.nodes}
        self.prepare_s=time.perf_counter()-tick

    def response(self,node,state,budget):
        k=self.kernel
        if state<0 or state>=len(k.owners) or k.owners[state] not in node.owners or budget<0: raise ValueError('Invalid parent entry or budget')
        key=(self.signatures[node.name],int(state),int(budget));stats=self.stats[node.name]
        if key in self.pool.cache:stats['cache_hits']+=1;return self.pool.cache[key]
        pending=[{} for _ in range(budget+1)];pending[0][int(state)]=1.
        exits={};escape={};local_tail=budget_tail=0.
        for used,layer in enumerate(pending):
            for current,weight in layer.items():
                if node.children:
                    child=self.child_for[node.name][int(k.owners[current])]
                    sub=self.response(child,current,budget-used);stats['child_applications']+=1
                else:
                    stats['elementary_rows']+=1;cost=int(k.costs[current])
                    if cost>budget-used:sub=FirstExit({}, {}, 0., 1.)
                    else:
                        lo,hi=k.starts[current:current+2];edges={}
                        for target,p in zip(k.targets[lo:hi],k.probabilities[lo:hi]):
                            edge=(int(target),cost);edges[edge]=edges.get(edge,0.)+float(p)
                        escaped={cost:k.escape[current]} if k.escape[current,0]>0 else {}
                        sub=FirstExit(edges,escaped,float(k.residual[current]),0.)
                local_tail+=weight*sub.local_tail;budget_tail+=weight*sub.budget_tail
                for cost,moment in sub.escape.items():
                    order=used+cost
                    if order not in escape:escape[order]=weight*moment
                    else:escape[order]+=weight*moment
                for (target,cost),p in sub.exits.items():
                    if cost<1: raise AssertionError('Through closure did not consume a collision')
                    order=used+cost;prob=weight*p
                    if k.owners[target] in node.owners:
                        pending[order][target]=pending[order].get(target,0.)+prob
                    else:
                        port=(target,order);exits[port]=exits.get(port,0.)+prob
        total=sum(exits.values())+sum(v[0] for v in escape.values())+local_tail+budget_tail
        if not np.isclose(total,1.,rtol=1e-11,atol=1e-12): raise AssertionError('Parent response probability changed')
        result=FirstExit(MappingProxyType(exits),MappingProxyType({i:readonly(v) for i,v in escape.items()}),local_tail,budget_tail)
        self.pool.cache[key]=result;stats['compiled_columns']+=1
        return result

    def solve(self,states,orders,values,max_bounces=64):
        tick=time.perf_counter();field=_field(states,orders,values,len(self.kernel.owners),max_bounces)
        mass=np.zeros(max_bounces+1);omega=mass.copy();local=budget=0.
        for (state,order),v in field.items():
            response=self.response(self.root,state,max_bounces-order)
            if response.exits:raise AssertionError('Root response leaks to an unowned state')
            for used,moment in response.escape.items():
                mass[order+used]+=v[0]*moment[0]
                omega[order+used]+=v[0]*moment[0]-np.dot(v[1:],moment[1:])
            local+=v[0]*response.local_tail;budget+=v[0]*response.budget_tail
        return _result(mass,omega,local,budget,sum(v[0] for v in field.values()),query_s=time.perf_counter()-tick,
            root_columns_applied=len(field),stats={n:s.copy() for n,s in self.stats.items()},physical_primitive_tests=0)


def flat_field_solve(kernel,states,orders,values,max_bounces=64):
    """Direct forward propagation, independent of hierarchy and response pool."""
    tick=time.perf_counter();field=_field(states,orders,values,len(kernel.owners),max_bounces)
    pending=[{} for _ in range(max_bounces+1)]
    for (state,order),v in field.items():pending[order][state]=v.copy()
    mass=np.zeros(max_bounces+1);omega=mass.copy();local=budget=0.;visits=0
    for order,layer in enumerate(pending):
        for state,v in layer.items():
            visits+=1;after=order+int(kernel.costs[state])
            if after>max_bounces:budget+=v[0];continue
            e=kernel.escape[state];mass[after]+=v[0]*e[0];omega[after]+=v[0]*e[0]-np.dot(v[1:],e[1:]);local+=v[0]*kernel.residual[state]
            lo,hi=kernel.starts[state:state+2]
            for target,p in zip(kernel.targets[lo:hi],kernel.probabilities[lo:hi]):
                if target not in pending[after]:pending[after][target]=p*v
                else:pending[after][target]+=p*v
    return _result(mass,omega,local,budget,sum(v[0] for v in field.values()),query_s=time.perf_counter()-tick,
        elementary_rows=visits,physical_primitive_tests=0)


def seed_compiled_field(compiled,kernel,source,max_bounces=64):
    """Use the existing exact first response and continuous-through routing."""
    from .ehss_compiled_field import initial_exact_response
    tick=time.perf_counter();scene=compiled.scene;grid=compiled.grid
    exits,directions,orders,owners,tests=initial_exact_response(source.origins,source.directions,source.last_atom,
        source.initial_bounces,scene.spheres.centers,scene.spheres.radii,scene.groups,scene.packed_ids,
        scene.offsets,scene.centers,scene.radii,max_bounces)
    target=np.full(len(owners),-1,dtype=np.int64);active=owners>=0
    projected=grid.project((exits[active]-scene.centers[owners[active]])/scene.radii[owners[active],None],directions[active])
    target[active]=compiled.routing[owners[active]*grid.size+projected]
    escaped=(owners==-1)|(active&(target<0));pending=active&(target>=0)
    mass=np.bincount(orders[escaped],weights=source.weights[escaped],minlength=max_bounces+1)
    contribution=source.weights*np.clip(1-np.sum(source.incoming*directions,axis=1),0.,2.)
    omega=np.bincount(orders[escaped],weights=contribution[escaped],minlength=max_bounces+1)
    states=kernel.original_to_state[target[pending]]
    if np.any(states<0):raise AssertionError('Source routing bypassed through closure')
    return dict(states=states,orders=orders[pending],values=np.c_[source.weights[pending],source.weights[pending,None]*source.incoming[pending]],
        initial_mass=mass,initial_omega=omega,initial_tail_mass=float(source.weights[owners==-2].sum()),
        source_mass=float(source.weights.sum()),seed_s=time.perf_counter()-tick,seed_primitive_tests=int(tests))


def with_seed(seed,result):
    mass=seed['initial_mass']+np.array([r['mass'] for r in result['order_ledger']])
    omega=seed['initial_omega']+np.array([r['omega'] for r in result['order_ledger']])
    return _result(mass,omega,result['local_tail_mass'],result['bounce_tail_mass']+seed['initial_tail_mass'],seed['source_mass'],
        initial_tail_mass=seed['initial_tail_mass'],seed_s=seed['seed_s'],seed_primitive_tests=seed['seed_primitive_tests'],field_query_s=result['query_s'])
