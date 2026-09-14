"""Bind a stored response in its own fitted node frame, independent of rebasing.

Physical coordinates are never projected. The fit changes response coordinates
and its compatibility check, not the current collision geometry.
"""
import time
import numpy as np
from numba import types
from numba.typed import Dict
from .ehss_sparse_path_response import BoundSparsePaths, readonly
from .ehss_surface_response import KEY
from .ehss_response import node_geometry, ResponseLibrary


def fit_node(reference, world):
    x = reference-reference.mean(axis=0); y = world-world.mean(axis=0)
    u, singular, vh = np.linalg.svd(x.T@y); correction = np.eye(3)
    correction[2,2] = 1. if np.linalg.det(u@vh) >= 0. else -1.
    rotation = (u@correction@vh).T; translation = world.mean(axis=0)-reference.mean(axis=0)@rotation.T
    error = float(np.max(np.abs((world-translation)@rotation-reference)))
    return rotation, translation, error, int(np.count_nonzero(singular > max(singular[0],1.)*1e-12))


class FittedNodeResponses(BoundSparsePaths):
    def __init__(self, model, atlas):
        if not atlas.prepared: raise ValueError('Prepared atlas required')
        if ResponseLibrary.topology_key(model.tree) != atlas.topology: raise ValueError('Physical ownership topology changed')
        tick=time.perf_counter();self.model=model;self.atlas=atlas;t=model.tree
        self.rotations=np.tile(np.eye(3),(len(t.left),1,1));self.active=np.zeros(len(t.left),dtype=bool)
        invalid=set();fits=[]
        for node,old in atlas.nodes.items():
            atoms,anchor,centers,rotation,translation=node_geometry(t,node)
            if not np.array_equal(atoms,old.atoms) or not np.array_equal(t.radii[atoms],old.radii):
                invalid.add(node);continue
            native_error=float(np.max(np.abs(centers-old.centers)))
            if native_error <= 1e-12:
                error=native_error;rank=int(np.linalg.matrix_rank(old.centers-old.centers.mean(axis=0)));method='native_already_aligned'
            else:
                fitted_rotation,fitted_translation,error,rank=fit_node(old.centers,model.spheres.centers[atoms]);method='stored_node_fit'
                if native_error < error:
                    error=native_error;method='native_lower_component_error'
                else:rotation=fitted_rotation;translation=fitted_translation
            valid=error <= atlas.residual_tolerance+1e-10
            if valid:self.rotations[node]=rotation
            else:invalid.add(node)
            fits.append(dict(node=node,native_component_error_A=native_error,fitted_component_error_A=error,
                valid=valid,rank=rank,method=method))
        self.table=Dict.empty(KEY,types.int64);children=[];roots=[]
        for key,routes in zip(atlas.keys,atlas.routes):
            node=int(key[0])
            if node in invalid:continue
            self.table[tuple(int(v) for v in key)]=len(roots);self.active[node]=True
            root=len(children);children.append({});roots.append(root)
            for route in routes:
                state=root
                for atom in route:
                    if atom not in children[state]:children[state][atom]=len(children);children.append({})
                    state=children[state][atom]
        starts=[0];atoms=[];next_states=[]
        for edges in children:
            for atom in sorted(edges):atoms.append(atom);next_states.append(edges[atom])
            starts.append(len(atoms))
        self.roots=readonly(np.array(roots,dtype=np.int64));self.edge_starts=readonly(np.array(starts,dtype=np.int64))
        self.edge_atoms=readonly(np.array(atoms,dtype=np.int64));self.edge_next=readonly(np.array(next_states,dtype=np.int64))
        self.rotations=readonly(self.rotations);self.active=readonly(self.active)
        self.receipt=dict(bind_s=time.perf_counter()-tick,active_cells=len(roots),inactive_cells=len(atlas.keys)-len(roots),
            active_nodes=int(self.active.sum()),trie_states=len(children),trie_edges=len(atoms),node_fits=fits,
            numeric_payload_bytes=sum(a.nbytes for a in (self.roots,self.edge_starts,self.edge_atoms,self.edge_next,self.rotations,self.active)),
            physical_coordinates_unchanged=True,rank_deficient_pose_not_unique=True,
            compatibility='Proper fit of the stored whole-node template to current owned atoms; component-wise body residual threshold, not a continuous response certificate.')
