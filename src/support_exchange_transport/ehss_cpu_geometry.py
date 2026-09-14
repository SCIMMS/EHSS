"""Native construction/refit of the same ordered chemical box hierarchy.

Independent leaf bounds and foreign-overlap checks support CPU parallelism.
Parent reductions remain ordered. GeometryFrame ownership, intrinsic blocks,
content hashes, stable IDs and all response-facing box metadata are preserved.
"""
from copy import copy
import time
import numpy as np
import numba
from numba import njit, prange
from .ehss_cpu import check_threads
from .ehss_prepared_geometry import GeometryPreparationLayout
from .ehss_guarded_intrinsic_response import GuardedSupportHierarchy
from .ehss_support_tree import NodeAwareSupportTree
from .ehss_intrinsic_frames import frozen


@njit
def topology(groups):
    nodes = 2*groups-1
    left = np.full(nodes, -1, np.int64); right = left.copy(); parent = left.copy(); leaf = left.copy()
    end = np.empty(nodes, np.int64); depth = np.zeros(nodes, np.int64)
    leaf_node = np.empty(groups, np.int64)
    begin = np.zeros(nodes, np.int64); stop = np.zeros(nodes, np.int64); stop[0] = groups
    for node in range(nodes):
        a, b = begin[node], stop[node]
        end[node] = node+2*(b-a)-1
        if b-a == 1:
            leaf[node] = a; leaf_node[a] = node
        else:
            mid = a+(b-a)//2
            l, r = node+1, node+2*(mid-a)
            left[node], right[node] = l, r
            begin[l], stop[l], begin[r], stop[r] = a, mid, mid, b
            parent[l] = node; parent[r] = node
            depth[l] = depth[node]+1; depth[r] = depth[node]+1
    return left, right, end, parent, leaf, leaf_node, depth


def _leaves(world, radii, changed, leaf_node, offsets, ids, lo, hi, pad):
    for group in prange(len(changed)):
        if changed[group]:
            node = leaf_node[group]
            for c in range(3):
                low, high = np.inf, -np.inf
                for k in range(offsets[group], offsets[group+1]):
                    atom = ids[k]
                    low = min(low, world[atom, c]-radii[atom])
                    high = max(high, world[atom, c]+radii[atom])
                lo[node, c] = low-pad; hi[node, c] = high+pad


leaf_serial = njit(_leaves)
leaf_parallel = njit(parallel=True)(_leaves)


@njit
def parents(changed, leaf_node, parent, left, right, lo, hi):
    affected = np.zeros(len(left), dtype=np.bool_)
    for group in range(len(changed)):
        if changed[group]:
            node = leaf_node[group]
            while node >= 0:
                affected[node] = True; node = parent[node]
    for node in range(len(left)-1, -1, -1):
        if affected[node] and left[node] >= 0:
            for c in range(3):
                lo[node, c] = min(lo[left[node], c], lo[right[node], c])
                hi[node, c] = max(hi[left[node], c], hi[right[node], c])
    return affected


def _exclusive(world, radii, atom_leaf, end, lo, hi):
    result = np.ones(len(end), dtype=np.bool_)
    for node in prange(len(end)):
        for atom in range(len(radii)):
            if node <= atom_leaf[atom] < end[node]:
                continue
            overlap = True
            for c in range(3):
                if world[atom, c]+radii[atom] < lo[node, c] or world[atom, c]-radii[atom] > hi[node, c]:
                    overlap = False; break
            if overlap:
                result[node] = False; break
    return result


exclusive_serial = njit(_exclusive)
exclusive_parallel = njit(parallel=True)(_exclusive)


class CPUPhysicalScene:
    def __init__(self, spheres, groups, atom_ids=None, threads=1):
        check_threads(threads)
        start = time.perf_counter()
        self.threads = threads
        self.geometry_layout = GeometryPreparationLayout(spheres, groups, atom_ids)
        layout_s = time.perf_counter()-start
        tick = time.perf_counter(); self.geometry = self.geometry_layout.build(spheres)
        geometry_s = time.perf_counter()-tick
        self.model = self._model(self.geometry)
        self.setup = dict(total_s=time.perf_counter()-start, layout_s=layout_s, geometry_s=geometry_s,
                          **self.model.receipt)

    def snapshot(self):
        return copy(self)

    def _model(self, geometry, previous=None):
        start = time.perf_counter()
        model = GuardedSupportHierarchy.__new__(GuardedSupportHierarchy)
        model.geometry = geometry; model.spheres = geometry.spheres()
        s = model.spheres; groups = geometry.groups; ng = len(geometry.blocks)
        tick = time.perf_counter()
        if previous is None:
            tree = NodeAwareSupportTree.__new__(NodeAwareSupportTree)
            tree.port_kind = 'box'; tree.convex_direction_level = 1
            tree.groups = frozen(groups); tree.n_groups = ng
            tree.coarse_supports = frozen(np.arange(ng))
            tree.offsets = self.geometry_layout.starts.copy(); tree.ids = self.geometry_layout.order.copy()
            tree.left, tree.right, tree.end, tree.parent, tree.leaf_group, tree.leaf_node, tree.depth = topology(ng)
            tree.coarse_node = tree.leaf_node.copy()
            nodes = len(tree.left)
            tree.lo = np.empty((nodes, 3)); tree.hi = np.empty((nodes, 3))
            tree.port_centers = np.zeros((nodes, 3)); tree.port_normals = np.zeros((nodes, 4, 3))
            tree.port_heights = np.zeros((nodes, 4)); tree.port_offsets = np.arange(nodes+1, dtype=np.int64)*4
        else:
            tree = copy(previous.tree)
            tree.lo = previous.tree.lo.copy(); tree.hi = previous.tree.hi.copy()
        topology_s = time.perf_counter()-tick
        model.tree = tree
        tick = time.perf_counter()
        tree.rotations = np.array([b.rotation for b in geometry.blocks])
        tree.translations = np.array([b.translation for b in geometry.blocks])
        if not np.array_equal(tree.rotations, np.broadcast_to(np.eye(3), tree.rotations.shape)):
            raise ValueError('CPU preparation requires the exact identity-frame GeometryPreparationLayout')
        tree.local = frozen(s.centers-tree.translations[groups])
        tree.world = s.centers; tree.radii = s.radii; tree.spheres = s
        if previous is None:
            tree.pose_versions = np.any(tree.translations != 0., axis=1).astype(np.int64)
            changed = np.ones(ng, dtype=bool)
        else:
            tree.pose_versions = previous.tree.pose_versions+np.any(tree.translations != previous.tree.translations, axis=1)
            moved = np.any(s.centers != previous.spheres.centers, axis=1)|(s.radii != previous.spheres.radii)
            changed = np.bincount(groups[moved], minlength=ng) > 0
        pad = 1e-10*max(1., float(s.radii.max()))
        if previous is not None and pad != previous.bound_pad:
            changed[:] = True
        model.bound_pad = pad
        metadata_s = time.perf_counter()-tick
        old_threads = numba.get_num_threads()
        try:
            if self.threads > 1: numba.set_num_threads(self.threads)
            tick = time.perf_counter()
            (leaf_serial if self.threads == 1 else leaf_parallel)(s.centers, s.radii, changed,
                tree.leaf_node, tree.offsets, tree.ids, tree.lo, tree.hi, pad)
            leaf_s = time.perf_counter()-tick
            tick = time.perf_counter()
            affected = parents(changed, tree.leaf_node, tree.parent, tree.left, tree.right, tree.lo, tree.hi)
            parent_s = time.perf_counter()-tick
            tick = time.perf_counter()
            tree.exclusive = (exclusive_serial if self.threads == 1 else exclusive_parallel)(s.centers,
                s.radii, tree.leaf_node[groups], tree.end, tree.lo, tree.hi)
            overlap_s = time.perf_counter()-tick
        finally:
            if numba.get_num_threads() != old_threads: numba.set_num_threads(old_threads)
        model.receipt = dict(model_s=time.perf_counter()-start, topology_s=topology_s,
            metadata_s=metadata_s, leaf_s=leaf_s, parent_s=parent_s, overlap_s=overlap_s,
            threads=self.threads, bounds_refreshed=int(affected.sum()),
            changed_supports=np.flatnonzero(changed).tolist(), exclusivity_refreshed=len(tree.left),
            topology_reused=previous is not None, exact_world=True)
        return model

    def update(self, spheres):
        start = time.perf_counter()
        if np.array_equal(spheres.centers, self.geometry.world) and np.array_equal(spheres.radii, self.geometry.radii):
            self.receipt = dict(total_s=time.perf_counter()-start, unchanged=True)
            return self.model
        tick = time.perf_counter()
        geometry = self.geometry_layout.build(spheres, self.geometry)
        geometry_s = time.perf_counter()-tick
        model = self._model(geometry, self.model)
        self.geometry, self.model = geometry, model
        self.receipt = dict(total_s=time.perf_counter()-start, unchanged=False, geometry_s=geometry_s, **model.receipt)
        return model
