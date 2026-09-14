"""One-collision field application via reusable hierarchy separator blocks.

Each edge belongs to the lowest hierarchy node containing both physical owners.
Within-owner edges live at leaves; cross-child transfers live at their parent.
Application visits this tree of compact sparse blocks. No dense root response
columns or path-length pruning are introduced. Positive one-collision rows are
iterated with a single global budget and original incoming direction moments.
"""
from dataclasses import dataclass
import hashlib,time
import numpy as np
from scipy.sparse import csr_matrix
from .ehss_field_condensation import _field,_result
from .ehss_sparse_path_response import readonly


@dataclass(frozen=True)
class TransferBlock:
    inputs: np.ndarray
    outputs: np.ndarray
    matrix: object


class SparseBlockPool:
    def __init__(self): self.cache={}


class SparseHierarchyField:
    def __init__(self,kernel,root,pool=None):
        tick=time.perf_counter(); self.kernel=kernel; self.root=root; self.pool=SparseBlockPool() if pool is None else pool
        if np.any(kernel.costs!=1): raise ValueError('One physical collision per row required')
        if root.owners!=frozenset(map(int,kernel.owners)): raise ValueError('Hierarchy must cover physical owners')
        nodes={}; paths={}; buckets={}
        def visit(node,path):
            if node.name in nodes: raise ValueError('Unique node names required')
            nodes[node.name]=node; buckets[node.name]=[]; path=path+(node.name,)
            if node.children:
                for child in node.children: visit(child,path)
            else:
                for owner in node.owners: paths[owner]=path
        visit(root,tuple())
        for source in range(len(kernel.owners)):
            a=paths[int(kernel.owners[source])]
            for edge in range(kernel.starts[source],kernel.starts[source+1]):
                target=int(kernel.targets[edge]); b=paths[int(kernel.owners[target])]
                common=a[0]
                for x,y in zip(a,b):
                    if x!=y: break
                    common=x
                buckets[common].append((source,target,float(kernel.probabilities[edge])))
        self.blocks={}; reused=[]; built=[]; stored_bytes=0; leaf_edges=parent_edges=0
        for name,edges in buckets.items():
            if not edges: self.blocks[name]=None; continue
            sources=np.array([e[0] for e in edges],dtype=np.int64); targets=np.array([e[1] for e in edges],dtype=np.int64); probs=np.array([e[2] for e in edges])
            inputs,ii=np.unique(sources,return_inverse=True); outputs,oo=np.unique(targets,return_inverse=True)
            matrix=csr_matrix((probs,(oo,ii)),shape=(len(outputs),len(inputs))); matrix.sum_duplicates(); matrix.sort_indices()
            h=hashlib.sha256()
            for array in (np.array(sorted(nodes[name].owners),dtype=np.int64),inputs,outputs,matrix.indptr,matrix.indices,matrix.data): h.update(array.tobytes())
            signature=h.hexdigest()
            if signature in self.pool.cache: block=self.pool.cache[signature]; reused.append(name)
            else:
                matrix.data.flags.writeable=False; matrix.indices.flags.writeable=False; matrix.indptr.flags.writeable=False
                block=TransferBlock(readonly(inputs),readonly(outputs),matrix); self.pool.cache[signature]=block; built.append(name)
            self.blocks[name]=block
            stored_bytes+=sum(a.nbytes for a in (block.inputs,block.outputs,block.matrix.data,block.matrix.indices,block.matrix.indptr))
            if nodes[name].children: parent_edges+=len(edges)
            else: leaf_edges+=len(edges)
        self.prepare_s=time.perf_counter()-tick
        self.receipt=dict(prepare_s=self.prepare_s,hierarchy_nodes=len(nodes),active_blocks=len(reused)+len(built),
            reused_blocks=reused,built_blocks=built,within_leaf_edges=leaf_edges,cross_child_edges=parent_edges,
            current_block_array_bytes=stored_bytes,pool_entries=len(self.pool.cache),unit_collision_cost=True,
            no_projection_change=True,no_probability_pruning=True)

    def solve(self,states,orders,values,max_bounces=64):
        tick=time.perf_counter(); k=self.kernel; seeds=_field(states,orders,values,len(k.owners),max_bounces)
        source_mass=sum(v[0] for v in seeds.values()); seeds_by_order={}
        for (state,order),value in seeds.items(): seeds_by_order.setdefault(order,[]).append((state,value))
        current=np.zeros((len(k.owners),4)); mass=np.zeros(max_bounces+1); omega=mass.copy(); local=budget=0.; applications=waves=0
        for order in range(max_bounces+1):
            for state,value in seeds_by_order.get(order,[]): current[state]+=value
            if order==max_bounces: budget=float(current[:,0].sum()); break
            if not np.any(current[:,0]>0): continue
            waves+=1; after=order+1
            mass[after]=np.dot(current[:,0],k.escape[:,0])
            omega[after]=mass[after]-np.einsum('ij,ij->',current[:,1:],k.escape[:,1:])
            local+=np.dot(current[:,0],k.residual); outgoing=np.zeros_like(current)
            def apply(node):
                nonlocal applications
                for child in node.children: apply(child)
                block=self.blocks[node.name]
                if block is not None:
                    incoming=current[block.inputs]
                    if np.any(incoming[:,0]>0):
                        outgoing[block.outputs]+=block.matrix@incoming; applications+=1
            apply(self.root); current=outgoing
        return _result(mass,omega,local,budget,source_mass,query_s=time.perf_counter()-tick,
            hierarchy_nodes=self.receipt['hierarchy_nodes'],active_blocks=self.receipt['active_blocks'],
            sparse_block_applications=applications,collision_waves=waves,physical_primitive_tests=0)
