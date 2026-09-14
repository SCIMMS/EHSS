"""Apply a full field through cached subtree exits without root columns.

The frontier partitions a hierarchy into reusable first-exit response blocks.
Transport between blocks is accumulated for the actual incoming field, with
the global collision budget retained. This is algebraically the same finite
kernel as fully compiled root responses; no new spatial/angular averaging.
"""
import time
import numpy as np
from .ehss_field_condensation import CondensedField,_field,_result


class StreamedField:
    def __init__(self,kernel,root,pool=None,max_owners=4):
        if not isinstance(max_owners,int) or max_owners<1: raise ValueError('Positive frontier owner limit required')
        tick=time.perf_counter(); self.kernel=kernel; self.root=root
        self.compiled=CondensedField(kernel,root,pool); self.pool=self.compiled.pool
        frontier=[]
        def visit(node):
            if len(node.owners)<=max_owners or not node.children: frontier.append(node)
            else:
                for child in node.children: visit(child)
        visit(root); self.frontier=tuple(frontier)
        self.owner_node={owner:node for node in frontier for owner in node.owners}
        if frozenset(self.owner_node)!=root.owners: raise AssertionError('Frontier must cover the root')
        self.prepare_s=time.perf_counter()-tick

    def solve(self,states,orders,values,max_bounces=64):
        tick=time.perf_counter(); seeds=_field(states,orders,values,len(self.kernel.owners),max_bounces)
        pending=[{} for _ in range(max_bounces+1)]
        for (state,order),value in seeds.items(): pending[order][state]=value.copy()
        mass=np.zeros(max_bounces+1); omega=mass.copy(); local=budget=0.; applications=transfers=0
        for order,layer in enumerate(pending):
            for state,value in layer.items():
                node=self.owner_node[int(self.kernel.owners[state])]
                response=self.compiled.response(node,state,max_bounces-order); applications+=1
                for cost,moment in response.escape.items():
                    mass[order+cost]+=value[0]*moment[0]
                    omega[order+cost]+=value[0]*moment[0]-np.dot(value[1:],moment[1:])
                local+=value[0]*response.local_tail; budget+=value[0]*response.budget_tail
                for (target,cost),probability in response.exits.items():
                    if cost<1 or self.kernel.owners[target] in node.owners: raise AssertionError('Invalid frontier first exit')
                    after=order+cost; contribution=probability*value
                    if target in pending[after]: pending[after][target]+=contribution
                    else: pending[after][target]=contribution
                    transfers+=1
            # Once processed, an earlier collision order cannot be revisited.
            pending[order]={}
        return _result(mass,omega,local,budget,sum(v[0] for v in seeds.values()),query_s=time.perf_counter()-tick,
            frontier_nodes=len(self.frontier),frontier_applications=applications,frontier_transfers=transfers,
            stats={n:s.copy() for n,s in self.compiled.stats.items()},physical_primitive_tests=0)
