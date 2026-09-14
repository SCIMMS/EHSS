"""Stable material identities and reusable hierarchy transition payloads.

Dense current state indices remain compact. Persistent identities identify
material ports and sampled suffix events, so renumbering alone cannot invalidate
a payload. Current escape moments and global index bindings stay outside it.
"""
from collections import defaultdict,deque
from dataclasses import dataclass
import hashlib,time
import numpy as np
from numba import njit,types
from numba.typed import List
from scipy.sparse import csr_matrix
from .ehss_material_collision_field import MaterialCollisionField,first_observed_successors
from .ehss_field_condensation import _field,_result
from .ehss_sparse_path_response import readonly


class _TaggedTrace:
    def __init__(self,model,bound,observed):
        self.model=model; self.labels=[]; self.rows=defaultdict(deque)
        s=model.spheres
        p,d,atoms=first_observed_successors(bound.points,bound.directions,observed['counts'],observed['history'],s.centers,s.radii)
        for row in np.flatnonzero((observed['terminal']!=-3)&(atoms>=0)):
            self.rows[(int(atoms[row]),p[row].tobytes(),d[row].tobytes())].append(int(row))
    def __getattr__(self,name): return getattr(self.model,name)
    def trace(self,source,max_bounces=128):
        if source.convention!='missing material target ports': raise ValueError('Coupling construction only')
        rows=[]
        for atom,p,d in zip(source.last_atom,source.origins,source.directions):
            q=self.rows.get((int(atom),p.tobytes(),d.tobytes()))
            if not q: raise ValueError('Unregistered material observation')
            rows.append(q.popleft())
        result=self.model.trace(source,max_bounces)
        for row,count in zip(rows,result.bounces):
            self.labels.extend(('suffix',row,int(j)) for j in range(count))
        return result


def build_semantic_field(bound,observed,model,bins=2):
    tick=time.perf_counter(); adapter=_TaggedTrace(model,bound,observed); registration_s=time.perf_counter()-tick
    field=MaterialCollisionField(bound,observed,adapter,bins=bins); field.model=model
    labels=tuple(('port',)+tuple(map(int,k)) for k in field.keys)+tuple(adapter.labels)
    if len(labels)!=len(field.kernel.owners) or len(set(labels))!=len(labels): raise AssertionError('Semantic identities do not cover current states uniquely')
    return field,labels,dict(total_prepare_s=time.perf_counter()-tick,registration_s=registration_s,
        material_states=len(field.keys),suffix_states=len(adapter.labels),physical_coupling_recomputed=True)


@dataclass(frozen=True)
class SemanticPayload:
    inputs: np.ndarray
    outputs: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    probabilities: np.ndarray


class SemanticBlockPool:
    def __init__(self,scope=None): self.scope=scope; self.identities={}; self.cache={}
    def snapshot(self):
        p=SemanticBlockPool(self.scope); p.identities=self.identities.copy(); p.cache=self.cache.copy(); return p


_I=types.Array(types.int64,1,'C',readonly=True)
_F=types.Array(types.float64,1,'C',readonly=True)
_BLOCK=types.Tuple((_I,_I,_I,_I,_F))


@njit(cache=True)
def apply_semantic(states,orders,values,cap,escape,residual,blocks):
    current=np.zeros((len(residual),4)); outgoing=np.zeros_like(current)
    mass=np.zeros(cap+1); omega=np.zeros(cap+1); local=0.; budget=0.; waves=0; applications=0
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
    return mass,omega,local,budget,waves,applications


class SemanticHierarchyField:
    def __init__(self,kernel,root,labels,pool=None,scope=None):
        tick=time.perf_counter(); self.kernel=kernel; self.root=root
        self.pool=SemanticBlockPool(scope) if pool is None else pool
        if self.pool.scope is not scope: raise ValueError('Different semantic identity scope')
        if len(labels)!=len(kernel.owners) or len(set(labels))!=len(labels): raise ValueError('Unique semantic label per current state required')
        if np.any(kernel.costs!=1): raise ValueError('One physical collision per row required')
        if root.owners!=frozenset(map(int,kernel.owners)): raise ValueError('Hierarchy must cover owners')
        tokens=[]
        for label in labels:
            if label not in self.pool.identities: self.pool.identities[label]=len(self.pool.identities)
            tokens.append(self.pool.identities[label])
        tokens=np.array(tokens,dtype=np.int64); permutation=np.argsort(tokens); sorted_tokens=tokens[permutation]
        nodes=[]; paths={}; names=set()
        def visit(node,path):
            if node.name in names: raise ValueError('Unique hierarchy names required')
            names.add(node.name); here=len(nodes); nodes.append(node); path=path+(here,)
            if node.children:
                for child in node.children: visit(child,path)
            else:
                for owner in node.owners: paths[owner]=path
        visit(root,tuple()); owners=sorted(root.owners); lca=np.zeros((len(owners),len(owners)),dtype=np.int64)
        for i,a in enumerate(owners):
            for j,b in enumerate(owners):
                for x,y in zip(paths[a],paths[b]):
                    if x!=y: break
                    lca[i,j]=x
        owner_index=np.searchsorted(np.array(owners,dtype=np.int64),kernel.owners)
        sources=np.repeat(np.arange(len(tokens)),np.diff(kernel.starts)); targets=kernel.targets
        buckets=lca[owner_index[sources],owner_index[targets]]
        self.payloads={}; bindings={}; built=[]; reused=[]; payload_bytes=0; reused_bytes=0; binding_bytes=0
        for index,node in enumerate(nodes):
            edges=np.flatnonzero(buckets==index)
            if not len(edges): continue
            it,ii=np.unique(tokens[sources[edges]],return_inverse=True); ot,oo=np.unique(tokens[targets[edges]],return_inverse=True)
            p=kernel.probabilities[edges]; order=np.lexsort((ii,oo)); ii=ii[order]; oo=oo[order]; p=p[order]
            h=hashlib.sha256()
            for a in (np.array(sorted(node.owners),dtype=np.int64),it,ot,ii,oo,p):
                h.update(np.array(a.shape,dtype=np.int64).tobytes()); h.update(a.tobytes())
            signature=h.hexdigest()
            if signature in self.pool.cache: payload=self.pool.cache[signature]; reused.append(node.name)
            else:
                matrix=csr_matrix((p,(oo,ii)),shape=(len(ot),len(it))); matrix.sum_duplicates(); matrix.sort_indices()
                payload=SemanticPayload(readonly(it),readonly(ot),readonly(matrix.indptr.astype(np.int64)),readonly(matrix.indices.astype(np.int64)),readonly(matrix.data.copy()))
                self.pool.cache[signature]=payload; built.append(node.name)
            size=sum(a.nbytes for a in (payload.inputs,payload.outputs,payload.indptr,payload.indices,payload.probabilities))
            payload_bytes+=size
            if node.name in reused: reused_bytes+=size
            ins=readonly(permutation[np.searchsorted(sorted_tokens,payload.inputs)]); outs=readonly(permutation[np.searchsorted(sorted_tokens,payload.outputs)])
            binding_bytes+=ins.nbytes+outs.nbytes; self.payloads[node.name]=payload
            bindings[node.name]=(ins,outs,payload.indptr,payload.indices,payload.probabilities)
        self.blocks=List.empty_list(_BLOCK); self.block_names=[]
        def bind(node):
            for child in node.children: bind(child)
            if node.name in bindings: self.blocks.append(bindings[node.name]); self.block_names.append(node.name)
        bind(root)
        self.receipt=dict(prepare_s=time.perf_counter()-tick,hierarchy_nodes=len(nodes),active_blocks=len(bindings),
            built_blocks=built,reused_blocks=reused,payload_array_bytes=payload_bytes,reused_payload_array_bytes=reused_bytes,
            current_binding_array_bytes=binding_bytes,registered_identities=len(self.pool.identities),pool_entries=len(self.pool.cache),
            current_state_count=len(tokens),global_numeric_renumbering_allowed=True,full_hierarchy_preserved=True,
            current_escape_moments_used=True,transition_scan_still_global=True,physical_coupling_recomputed=True)

    def solve(self,states,orders,values,max_bounces=64):
        tick=time.perf_counter(); k=self.kernel; field=_field(states,orders,values,len(k.owners),max_bounces)
        ids=np.array([s for s,o in field],dtype=np.int64); orders=np.array([o for s,o in field],dtype=np.int64)
        values=np.array(list(field.values()),dtype=float).reshape(-1,4)
        m,w,l,b,waves,applications=apply_semantic(ids,orders,values,max_bounces,k.escape,k.residual,self.blocks)
        return _result(m,w,l,b,sum(v[0] for v in field.values()),query_s=time.perf_counter()-tick,
            hierarchy_nodes=self.receipt['hierarchy_nodes'],active_blocks=len(self.blocks),sparse_block_applications=applications,
            collision_waves=waves,physical_primitive_tests=0,semantic_hierarchy=True)
