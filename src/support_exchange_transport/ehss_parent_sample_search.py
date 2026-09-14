"""Use selected parents as response owners without flattening the physical BVH."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from .ehss_sparse_path_response import readonly


def with_parent_search(bound,search_model,support_nodes):
    t=search_model.tree;nodes=np.array(support_nodes,dtype=np.int64,copy=True)
    if nodes.shape!=(len(bound.bank.atoms),) or np.any(nodes<0) or np.any(nodes>=len(t.left)):
        raise ValueError('One existing search-tree node per response owner required')
    if not np.array_equal(search_model.spheres.centers,bound.model.spheres.centers) or not np.array_equal(search_model.spheres.radii,bound.model.spheres.radii):
        raise ValueError('Response and search geometry differ')
    atom_leaves=t.leaf_node[t.groups]
    for group,node in enumerate(nodes):
        actual=np.flatnonzero((atom_leaves>=node)&(atom_leaves<t.end[node]))
        if not np.array_equal(actual,bound.bank.atoms[group]):raise ValueError('Selected search node has different physical ownership')
    # This is a kernel argument view, not a general NodeAwareSupportTree. Physical
    # leaves/offsets stay fine; only response owner -> skipped subtree is replaced.
    view=SimpleNamespace(**{name:getattr(t,name) for name in ('lo','hi','left','right','end','leaf_group','offsets','ids')},
        leaf_node=readonly(nodes),groups=bound.bank.groups)
    return replace(bound,model=SimpleNamespace(tree=view,spheres=search_model.spheres))
