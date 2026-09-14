"""Local chemical ownership independent of a possibly cyclic physical graph.

Retain residue-local cycles, multiple/aromatic bonds, terminal heavy links and
conjugated C-N. Other bonds may cross supports, including disulfides. Every
physical bond is retained in the dependency ledger; a spanning forest is only
an index of that ledger, never a replacement molecular topology.
"""
import numpy as np


class _DSU:
    def __init__(self,n): self.parent=list(range(n))
    def find(self,a):
        while self.parent[a]!=a:
            self.parent[a]=self.parent[self.parent[a]]; a=self.parent[a]
        return a
    def join(self,a,b):
        a,b=self.find(a),self.find(b)
        if a==b: return False
        self.parent[max(a,b)]=min(a,b); return True


def _graph(n,annotations):
    graph=[set() for _ in range(n)]; rows={}
    for row in annotations:
        a,b=int(row['a']),int(row['b']); pair=tuple(sorted((a,b)))
        if not 0<=a<n or not 0<=b<n or a==b or pair in rows:
            raise ValueError('Simple valid physical bond graph required')
        if row['order'] not in ('SING','DOUB','TRIP','AROM'):
            raise ValueError('Supported explicit chemical bond order required')
        rows[pair]=dict(row,a=pair[0],b=pair[1]); graph[a].add(b); graph[b].add(a)
    return graph,rows


def _bridges(graph):
    """Iterative Tarjan: long protein graphs do not depend on recursion limits."""
    discovery=[-1]*len(graph); low=discovery.copy(); parent=[-1]*len(graph)
    clock=0; bridges=set()
    for root in range(len(graph)):
        if discovery[root]>=0: continue
        discovery[root]=low[root]=clock; clock+=1
        stack=[(root,iter(sorted(graph[root])))]
        while stack:
            a,neighbors=stack[-1]
            try: b=next(neighbors)
            except StopIteration:
                stack.pop(); p=parent[a]
                if p>=0:
                    low[p]=min(low[p],low[a])
                    if low[a]>discovery[p]: bridges.add(tuple(sorted((a,p))))
                continue
            if b==parent[a]: continue
            if discovery[b]<0:
                parent[b]=a; discovery[b]=low[b]=clock; clock+=1
                stack.append((b,iter(sorted(graph[b]))))
            else: low[a]=min(low[a],discovery[b])
    return bridges


def dependency_ledger(groups,annotations):
    groups=np.asarray(groups)
    if groups.ndim!=1 or not np.issubdtype(groups.dtype,np.integer) or not len(groups):
        raise ValueError('Nonempty integer support ownership required')
    if not np.array_equal(np.unique(groups),np.arange(groups.max()+1)):
        raise ValueError('Contiguous nonempty support IDs required')
    graph,rows=_graph(len(groups),annotations)
    forest=_DSU(int(groups.max()+1)); physical=_DSU(len(groups))
    internal=[]; boundary=[]; forest_edges=[]; closure_edges=[]
    for (a,b),row in sorted(rows.items()):
        physical.join(a,b)
        if groups[a]==groups[b]: internal.append(dict(row)); continue
        entry=dict(row,support_a=int(groups[a]),support_b=int(groups[b]))
        candidates_a=sorted(graph[a]-{b}); candidates_b=sorted(graph[b]-{a})
        entry['dihedral_quad']=None if not candidates_a or not candidates_b else [candidates_a[0],a,b,candidates_b[0]]
        # A closure edge's identity depends on deterministic forest order; all
        # boundary edges remain constraints and none is an independent rotor.
        entry['dependency_kind']='forest' if forest.join(int(groups[a]),int(groups[b])) else 'closure'
        index=len(boundary); boundary.append(entry)
        (forest_edges if entry['dependency_kind']=='forest' else closure_edges).append(index)
    components=len({forest.find(i) for i in range(len(forest.parent))})
    assert len(closure_edges)==len(boundary)-len(forest.parent)+components
    return dict(internal_bonds=internal,boundary_bonds=boundary,forest_edge_indices=forest_edges,
                closure_edge_indices=closure_edges,quotient_cycle_rank=len(closure_edges),
                quotient_components=components,physical_components=len({physical.find(i) for i in range(len(groups))}),
                physical_bond_count=len(rows),all_physical_bonds_preserved=True,
                interpretation='Full boundary multigraph retained. Forest/closure labels are bookkeeping, not independent torsional degrees of freedom.')


def local_chemical_partition(atom_keys,elements,annotations):
    keys=list(map(tuple,atom_keys)); n=len(keys)
    if not n or len(elements)!=n or len(set(keys))!=n:
        raise ValueError('Unique atom identities and matching elements required')
    graph,rows=_graph(n,annotations)
    local_graph=[{b for b in graph[a] if keys[a][:4]==keys[b][:4]} for a in range(n)]
    local_bridges=_bridges(local_graph)
    conjugated={a for a in range(n) if elements[a]=='C' and any(
        elements[b] in ('O','N') and rows[tuple(sorted((a,b)))]['order']=='DOUB' for b in graph[a])}
    keep=[]; candidates=[]; ownership=_DSU(n)
    for (a,b),row in sorted(rows.items()):
        same_residue=keys[a][:4]==keys[b][:4]
        reason=None
        if same_residue and (a,b) not in local_bridges: reason='residue_local_cycle'
        elif row['order']!='SING' or row['aromatic']: reason='multiple_or_aromatic'
        elif len(graph[a])<2 or len(graph[b])<2: reason='terminal_heavy_link'
        elif (a in conjugated and elements[b]=='N') or (b in conjugated and elements[a]=='N'):
            reason='conjugated_C_N'
        if reason:
            ownership.join(a,b); keep.append(dict(a=a,b=b,reason=reason))
        else: candidates.append((a,b))
    labels={}; groups=np.array([labels.setdefault(ownership.find(a),len(labels)) for a in range(n)],dtype=np.int64)
    members=[np.flatnonzero(groups==g).tolist() for g in range(len(labels))]
    return dict(groups=groups,members=members,blocks=len(members),protected_bonds=keep,
                candidate_boundary_bonds=candidates,dependencies=dependency_ledger(groups,annotations),
                rule='Retain residue-local cycles, multiple/aromatic, terminal and conjugated C-N links; partition other links without changing physical topology.',
                rigidity_claim=False,independent_rotor_claim=False)


def spatial_partition(centers,block_sizes):
    """Initial-coordinate median partition with the exact supplied size multiset.

    Freeze the resulting ownership across MD; do not repartition every frame.
    """
    centers=np.asarray(centers,dtype=float); sizes=np.asarray(block_sizes)
    if centers.ndim!=2 or centers.shape[1]!=3 or not np.isfinite(centers).all():
        raise ValueError('Finite N by 3 coordinates required')
    if sizes.ndim!=1 or not np.issubdtype(sizes.dtype,np.integer) or np.any(sizes<=0) or sizes.sum()!=len(centers):
        raise ValueError('Positive integer sizes must cover every atom')
    sizes=np.sort(sizes)[::-1]; groups=np.full(len(centers),-1,dtype=np.int64)
    stack=[(np.arange(len(centers)),0,len(sizes))]
    while stack:
        ids,start,end=stack.pop()
        if end-start==1: groups[ids]=start; continue
        mid=(start+end)//2; count=int(sizes[start:mid].sum())
        axis=int(np.argmax(np.ptp(centers[ids],axis=0)))
        order=np.lexsort((ids,centers[ids,(axis+2)%3],centers[ids,(axis+1)%3],centers[ids,axis]))
        ids=ids[order]; stack.append((ids[count:],mid,end)); stack.append((ids[:count],start,mid))
    assert np.array_equal(np.bincount(groups),sizes)
    return groups


def fit_residuals(reference,current,groups):
    """Per-block proper-rigid-fit Euclidean residual; coordinates never snapped."""
    reference=np.asarray(reference,dtype=float); current=np.asarray(current,dtype=float); groups=np.asarray(groups)
    if reference.shape!=current.shape or reference.ndim!=2 or reference.shape[1]!=3 or not np.isfinite(reference).all() or not np.isfinite(current).all():
        raise ValueError('Matching finite N by 3 coordinate arrays required')
    if groups.shape!=(len(reference),) or not np.issubdtype(groups.dtype,np.integer) or not np.array_equal(np.unique(groups),np.arange(groups.max()+1)):
        raise ValueError('Contiguous support ownership required')
    rows=[]
    for group in range(int(groups.max()+1)):
        ids=np.flatnonzero(groups==group); x=reference[ids]; y=current[ids]
        xm=x.mean(axis=0); ym=y.mean(axis=0)
        u,_,vh=np.linalg.svd((x-xm).T@(y-ym))
        correction=np.eye(3); correction[2,2]=1. if np.linalg.det(u@vh)>=0 else -1.
        rotation=u@correction@vh; residual=np.linalg.norm((x-xm)@rotation+ym-y,axis=1)
        rows.append(dict(group=group,atoms=len(ids),max_angstrom=float(residual.max()),rms_angstrom=float(np.sqrt(np.mean(residual**2)))))
    return rows
