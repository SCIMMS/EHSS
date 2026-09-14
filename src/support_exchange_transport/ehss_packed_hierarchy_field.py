"""Native execution of the existing full hierarchy's postorder CSR blocks.

Packing preserves each separator block and its application order. It changes
neither the material projection nor the finite collision budget. The pool and
its original blocks remain available for subsequent geometry updates.
"""
import time
import numpy as np
from numba import njit
from .ehss_field_condensation import _field, _result
from .ehss_sparse_path_response import readonly


@njit(cache=True)
def apply_packed(states, orders, values, cap, escape, residual,
                 input_starts, inputs, output_starts, outputs,
                 row_starts, indices, probabilities):
    current=np.zeros((len(residual),4)); outgoing=np.zeros_like(current)
    mass=np.zeros(cap+1); omega=np.zeros(cap+1)
    local=0.; budget=0.; waves=0; applications=0
    for order in range(cap+1):
        for j in range(len(states)):
            if orders[j]==order:
                for c in range(4): current[states[j],c]+=values[j,c]
        if order==cap:
            for state in range(len(residual)): budget+=current[state,0]
            break
        active=False
        for state in range(len(residual)):
            if current[state,0]>0: active=True; break
        if not active: continue
        waves+=1
        for state in range(len(residual)):
            m=current[state,0]*escape[state,0]
            mass[order+1]+=m
            for c in range(1,4): m-=current[state,c]*escape[state,c]
            omega[order+1]+=m
            local+=current[state,0]*residual[state]
            for c in range(4): outgoing[state,c]=0.
        for block in range(len(input_starts)-1):
            active=False
            for j in range(input_starts[block],input_starts[block+1]):
                if current[inputs[j],0]>0: active=True; break
            if not active: continue
            applications+=1
            for row in range(output_starts[block],output_starts[block+1]):
                target=outputs[row]
                for c in range(4):
                    value=0.
                    for edge in range(row_starts[row],row_starts[row+1]):
                        value+=probabilities[edge]*current[indices[edge],c]
                    outgoing[target,c]+=value
        current,outgoing=outgoing,current
    return mass,omega,local,budget,waves,applications


class PackedHierarchyField:
    def __init__(self,hierarchy):
        tick=time.perf_counter(); self.kernel=hierarchy.kernel; self.hierarchy=hierarchy
        names=[]
        def visit(node):
            for child in node.children: visit(child)
            if hierarchy.blocks[node.name] is not None: names.append(node.name)
        visit(hierarchy.root)
        inputs=[]; outputs=[]; indices=[]; probabilities=[]
        input_starts=[0]; output_starts=[0]; row_starts=[0]
        for name in names:
            block=hierarchy.blocks[name]; matrix=block.matrix
            inputs.extend(block.inputs); outputs.extend(block.outputs)
            input_starts.append(len(inputs)); output_starts.append(len(outputs))
            for row in range(len(block.outputs)):
                lo,hi=matrix.indptr[row:row+2]
                indices.extend(block.inputs[matrix.indices[lo:hi]])
                probabilities.extend(matrix.data[lo:hi]); row_starts.append(len(indices))
        self.names=tuple(names)
        self.arrays=tuple(readonly(np.asarray(a,dtype=np.int64)) for a in
            (input_starts,inputs,output_starts,outputs,row_starts,indices))+(readonly(np.asarray(probabilities,dtype=float)),)
        self.receipt=dict(pack_s=time.perf_counter()-tick,packed_blocks=len(names),
            packed_array_bytes=sum(a.nbytes for a in self.arrays),hierarchy_nodes=hierarchy.receipt['hierarchy_nodes'],
            postorder_block_names=list(names),full_hierarchy_preserved=True,
            no_projection_change=True,no_probability_pruning=True,jit_included_in_pack=False)

    def solve(self,states,orders,values,max_bounces=64):
        tick=time.perf_counter(); k=self.kernel
        seeds=_field(states,orders,values,len(k.owners),max_bounces)
        ids=np.array([s for s,o in seeds],dtype=np.int64)
        order=np.array([o for s,o in seeds],dtype=np.int64)
        vals=np.array(list(seeds.values()),dtype=float).reshape(-1,4)
        mass,omega,local,budget,waves,applications=apply_packed(
            ids,order,vals,max_bounces,k.escape,k.residual,*self.arrays)
        return _result(mass,omega,local,budget,sum(v[0] for v in seeds.values()),
            query_s=time.perf_counter()-tick,hierarchy_nodes=self.receipt['hierarchy_nodes'],
            active_blocks=len(self.names),sparse_block_applications=applications,
            collision_waves=waves,physical_primitive_tests=0,packed_hierarchy=True)
