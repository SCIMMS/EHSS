"""Remove field paths with no possible escape output, preserving finite ledgers.

Reachability is a property of the current empirical kernel, not a geometric
certificate about the molecule. Positive edges are never thresholded. Mass
entering output-inaccessible states is unresolved, not escaping or discarded.
"""
from copy import copy
import time
import numpy as np
from numba import njit
from .ehss_field_condensation import ScatteringKernel
from .ehss_semantic_hierarchy import SemanticHierarchyField
from .ehss_mass_bounded_field import MassBoundedHierarchyField
from .ehss_sparse_path_response import readonly


@njit(cache=True)
def reverse_escape_reachability(starts,targets,probabilities,escape):
    n=len(escape); incoming=np.zeros(n,dtype=np.int64)
    for edge in range(len(targets)):
        if probabilities[edge]>0: incoming[targets[edge]]+=1
    offsets=np.zeros(n+1,dtype=np.int64)
    for state in range(n): offsets[state+1]=offsets[state]+incoming[state]
    predecessors=np.empty(offsets[-1],dtype=np.int64); cursors=offsets[:-1].copy()
    for source in range(n):
        for edge in range(starts[source],starts[source+1]):
            if probabilities[edge]>0:
                target=targets[edge]; predecessors[cursors[target]]=source; cursors[target]+=1
    reachable=np.zeros(n,dtype=np.bool_); queue=np.empty(n,dtype=np.int64); head=tail=0
    for state in range(n):
        # Keep even a roundoff-sized nonzero direction moment conservatively.
        if np.any(escape[state]!=0):
            reachable[state]=True; queue[tail]=state; tail+=1
    while head<tail:
        target=queue[head]; head+=1
        for j in range(offsets[target],offsets[target+1]):
            source=predecessors[j]
            if not reachable[source]: reachable[source]=True; queue[tail]=source; tail+=1
    return reachable


def reduce_non_escaping(kernel):
    tick=time.perf_counter()
    reachable=reverse_escape_reachability(kernel.starts,kernel.targets,kernel.probabilities,kernel.escape)
    graph_s=time.perf_counter()-tick; start=time.perf_counter(); n=len(kernel.owners)
    sources=np.repeat(np.arange(n,dtype=np.int64),np.diff(kernel.starts)); positive=kernel.probabilities>0
    if np.any(positive&~reachable[sources]&reachable[kernel.targets]): raise AssertionError('Reachability missed a positive predecessor')
    keep=positive&reachable[sources]&reachable[kernel.targets]
    redirected=positive&reachable[sources]&~reachable[kernel.targets]
    additions=np.bincount(sources[redirected],weights=kernel.probabilities[redirected],minlength=n)
    residual=kernel.residual.copy()+additions; residual[~reachable]=1.
    starts=np.r_[0,np.cumsum(np.bincount(sources[keep],minlength=n))].astype(np.int64)
    # Original shape, identity, moment and positivity checks already hold.
    # Validate the changed row totals together, without repeating a Python
    # constructor loop over every unchanged owner/cost/escape row.
    totals=np.bincount(sources[keep],weights=kernel.probabilities[keep],minlength=n)+residual+kernel.escape[:,0]
    if not np.isfinite(residual).all() or np.any(residual<0) or not np.allclose(totals,1.,rtol=0,atol=2e-12):
        raise AssertionError('Residualized kernel must conserve row mass')
    reduced=copy(kernel)
    reduced.starts=readonly(starts); reduced.targets=readonly(kernel.targets[keep]); reduced.probabilities=readonly(kernel.probabilities[keep])
    reduced.residual=readonly(residual)
    receipt=dict(prepare_s=time.perf_counter()-tick,graph_s=graph_s,kernel_build_s=time.perf_counter()-start,
        states=n,escape_reachable_states=int(reachable.sum()),non_escaping_states=int((~reachable).sum()),
        original_edges=len(kernel.targets),positive_original_edges=int(positive.sum()),retained_edges=int(keep.sum()),
        redirected_edges=int(redirected.sum()),redirected_source_rows=int(np.count_nonzero(additions)),
        positive_edges_thresholded=False,state_ids_and_owners_preserved=True,geometric_escape_certificate=False,
        residual_reclassification='Local versus cap tails may change; total residual and escaped order ledger are preserved without mass truncation.')
    return reduced,readonly(reachable),receipt


class EscapeReachableHierarchyField:
    def __init__(self,kernel,root,labels,pool=None,scope=None):
        tick=time.perf_counter(); self.kernel=kernel
        self.reduced,self.reachable,reduction=reduce_non_escaping(kernel)
        self.inner=SemanticHierarchyField(self.reduced,root,labels,pool,scope)
        self.solver=MassBoundedHierarchyField(self.inner,0.)
        self.receipt=dict(prepare_s=time.perf_counter()-tick,reduction=reduction,hierarchy=self.inner.receipt)

    def with_budget(self,fraction):
        view=copy(self); view.solver=MassBoundedHierarchyField(self.inner,fraction); return view

    def solve(self,states,orders,values,max_bounces=64):
        result=self.solver.solve(states,orders,values,max_bounces)
        result.update(escape_reduction=True,non_escaping_states=self.receipt['reduction']['non_escaping_states'],
            tail_partition_changed=True,geometric_escape_certificate=False)
        return result
