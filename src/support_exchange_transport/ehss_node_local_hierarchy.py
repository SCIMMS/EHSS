"""Finer transition payloads inside the unchanged physical support hierarchy.

Each node assigns its own append-only slots to persistent source IDs.
Slots survive dense renumbering, temporary disappearance and additions. They are storage/update units,
not additional physical supports or a change of material projection.
"""
import hashlib,time
import numpy as np
from numba.typed import List
from .ehss_semantic_hierarchy import SemanticHierarchyField,SemanticBlockPool,SemanticPayload,_BLOCK
from .ehss_sparse_path_response import readonly


class NodeLocalBlockPool(SemanticBlockPool):
    """Immutable payload sharing, with independently copied membership registries."""
    def __init__(self,scope=None):
        super().__init__(scope); self.node_slots={}

    def snapshot(self):
        p=NodeLocalBlockPool(self.scope)
        p.identities=self.identities.copy(); p.cache=self.cache.copy()
        p.node_slots={key:slots.copy() for key,slots in self.node_slots.items()}
        return p


class NodeLocalSemanticHierarchyField(SemanticHierarchyField):
    def __init__(self,kernel,root,labels,pool=None,scope=None,max_inputs=32):
        tick=time.perf_counter(); self.kernel=kernel; self.root=root
        if not isinstance(max_inputs,int) or max_inputs<1: raise ValueError('Positive integer input range required')
        self.pool=NodeLocalBlockPool(scope) if pool is None else pool
        if not isinstance(self.pool,NodeLocalBlockPool): raise ValueError('Node-local pool required')
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
        self.payloads={}; bindings={}; node_fragments={node.name:[] for node in nodes}
        built=[]; reused=[]; payload_bytes=reused_bytes=binding_bytes=0; reused_edges=0; active_nodes=0
        input_counts=[]
        for index,node in enumerate(nodes):
            node_edges=np.flatnonzero(buckets==index)
            if not len(node_edges): continue
            active_nodes+=1
            membership_key=(node.name,tuple(sorted(node.owners)))
            slots=self.pool.node_slots.setdefault(membership_key,{})
            node_tokens,inverse=np.unique(tokens[sources[node_edges]],return_inverse=True)
            local_slots=np.empty(len(node_tokens),dtype=np.int64)
            for i,token in enumerate(node_tokens):
                token=int(token)
                if token not in slots: slots[token]=len(slots)
                local_slots[i]=slots[token]
            cohorts=local_slots[inverse]//max_inputs
            for cohort in np.unique(cohorts):
                edges=node_edges[cohorts==cohort]; key=(node.name,int(cohort))
                it,ii=np.unique(tokens[sources[edges]],return_inverse=True); ot,oo=np.unique(tokens[targets[edges]],return_inverse=True)
                input_counts.append(len(it))
                p=kernel.probabilities[edges]; order=np.lexsort((ii,oo)); ii=ii[order]; oo=oo[order]; p=p[order]
                h=hashlib.sha256()
                for a in (np.array(sorted(node.owners),dtype=np.int64),it,ot,ii,oo,p):
                    h.update(np.array(a.shape,dtype=np.int64).tobytes()); h.update(a.tobytes())
                signature=h.hexdigest(); reused_here=signature in self.pool.cache
                if reused_here:
                    payload=self.pool.cache[signature]; reused.append(key); reused_edges+=len(payload.probabilities)
                else:
                    # Build CSR only for a changed fragment. The current COO
                    # scan/hash remains necessary; a hit keeps the old arrays.
                    heads=np.r_[True,(ii[1:]!=ii[:-1])|(oo[1:]!=oo[:-1])]
                    starts=np.flatnonzero(heads); p=np.add.reduceat(p,starts); ii=ii[starts]; oo=oo[starts]
                    indptr=np.r_[0,np.cumsum(np.bincount(oo,minlength=len(ot)))].astype(np.int64)
                    payload=SemanticPayload(readonly(it),readonly(ot),readonly(indptr),readonly(ii.astype(np.int64)),readonly(p))
                    self.pool.cache[signature]=payload; built.append(key)
                size=sum(a.nbytes for a in (payload.inputs,payload.outputs,payload.indptr,payload.indices,payload.probabilities))
                payload_bytes+=size
                if reused_here: reused_bytes+=size
                ins=readonly(permutation[np.searchsorted(sorted_tokens,payload.inputs)]); outs=readonly(permutation[np.searchsorted(sorted_tokens,payload.outputs)])
                binding_bytes+=ins.nbytes+outs.nbytes; self.payloads[key]=payload
                bindings[key]=(ins,outs,payload.indptr,payload.indices,payload.probabilities); node_fragments[node.name].append(key)
        self.blocks=List.empty_list(_BLOCK); self.block_names=[]
        def bind(node):
            for child in node.children: bind(child)
            for key in node_fragments[node.name]: self.blocks.append(bindings[key]); self.block_names.append(key)
        bind(root)
        self.receipt=dict(prepare_s=time.perf_counter()-tick,hierarchy_nodes=len(nodes),active_hierarchy_nodes=active_nodes,
            active_blocks=len(bindings),built_blocks=built,reused_blocks=reused,payload_array_bytes=payload_bytes,
            reused_payload_array_bytes=reused_bytes,reused_transition_entries=reused_edges,
            current_binding_array_bytes=binding_bytes,registered_identities=len(self.pool.identities),pool_entries=len(self.pool.cache),
            current_state_count=len(tokens),global_numeric_renumbering_allowed=True,full_hierarchy_preserved=True,
            current_escape_moments_used=True,transition_scan_still_global=True,physical_coupling_recomputed=True,
            max_inputs_per_fragment=max_inputs,fragment_input_counts=input_counts,
            node_slot_count=sum(len(v) for v in self.pool.node_slots.values()),
            active_node_input_count=sum(input_counts),
            partition='Append-only node-local input slots inside each original hierarchy node')
