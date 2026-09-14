"""Comparable any-hit sphere indexes and body-frame support reuse.

The octree partitions centers into spatial octants, stores each sphere once,
and bounds the full spheres conservatively (a loose centroid octree). It is an
IMPACT-inspired control, not a reimplementation of the IMPACT software.
Node-aware means a BVH over physical groups plus persistent body-frame BVHs.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numba import njit

from .inside_out_pa import Spheres, Source, sphere_hit, sample_source


@dataclass
class Tree:
    lo: np.ndarray
    hi: np.ndarray
    end: np.ndarray
    first: np.ndarray
    count: np.ndarray
    ids: np.ndarray

    @property
    def arrays(self):
        return (self.lo, self.hi, self.end, self.first, self.count, self.ids)

    @property
    def nbytes(self):
        return sum(x.nbytes for x in self.arrays)

    def refit(self, primitive_lo, primitive_hi):
        _refit(*self.arrays, primitive_lo, primitive_hi)


def build_tree(lo, hi, kind="bvh", leaf_size=4):
    if kind not in ("bvh", "octree") or leaf_size < 1 or len(lo) == 0:
        raise ValueError("Valid kind, positive leaf size and nonempty primitives required.")
    centers = (lo+hi)/2
    lower, upper, end, first, count, ids = [], [], [], [], [], []
    root_center = (centers.min(axis=0)+centers.max(axis=0))/2
    half = max(float(np.ptp(centers, axis=0).max()/2), 1e-12)

    def visit(indices, depth, cell_lo, cell_hi):
        index = len(end)
        lower.append(lo[indices].min(axis=0))
        upper.append(hi[indices].max(axis=0))
        end.append(0)
        first.append(len(ids))
        count.append(0)
        if len(indices) <= leaf_size or depth >= 24 or np.all(np.ptp(centers[indices], axis=0) == 0):
            count[index] = len(indices)
            ids.extend(indices.tolist())
        elif kind == "bvh":
            axis = int(np.argmax(np.ptp(centers[indices], axis=0)))
            order = indices[np.argsort(centers[indices, axis], kind="stable")]
            middle = len(order)//2
            visit(order[:middle], depth+1, cell_lo, cell_hi)
            visit(order[middle:], depth+1, cell_lo, cell_hi)
        else:
            mid = (cell_lo+cell_hi)/2
            codes = ((centers[indices] >= mid)*np.array([1,2,4])).sum(axis=1)
            for octant in range(8):
                subset = indices[codes == octant]
                if not len(subset):
                    continue
                bits = (octant & np.array([1,2,4])) != 0
                visit(subset, depth+1, np.where(bits, mid, cell_lo), np.where(bits, cell_hi, mid))
        end[index] = len(end)

    visit(np.arange(len(lo), dtype=np.int64), 0, root_center-half, root_center+half)
    return Tree(np.array(lower), np.array(upper), np.array(end, dtype=np.int64),
                np.array(first, dtype=np.int64), np.array(count, dtype=np.int64), np.array(ids, dtype=np.int64))


@njit(cache=True)
def _refit(lo, hi, end, first, count, ids, primitive_lo, primitive_hi):
    for node in range(len(end)-1, -1, -1):
        for axis in range(3):
            lo[node, axis], hi[node, axis] = np.inf, -np.inf
        if count[node] > 0:
            for k in range(first[node], first[node]+count[node]):
                j = ids[k]
                for axis in range(3):
                    lo[node, axis] = min(lo[node, axis], primitive_lo[j, axis])
                    hi[node, axis] = max(hi[node, axis], primitive_hi[j, axis])
        else:
            child = node+1
            while child < end[node]:
                for axis in range(3):
                    lo[node, axis] = min(lo[node, axis], lo[child, axis])
                    hi[node, axis] = max(hi[node, axis], hi[child, axis])
                child = end[child]


@njit(cache=True)
def box_hit(o, d, lo, hi):
    near, far = 0., np.inf
    for axis in range(3):
        if d[axis] == 0:
            if o[axis] < lo[axis] or o[axis] > hi[axis]:
                return False
        else:
            a, b = (lo[axis]-o[axis])/d[axis], (hi[axis]-o[axis])/d[axis]
            near = max(near, min(a,b))
            far = min(far, max(a,b))
            if far < near:
                return False
    return True


@njit(cache=True)
def _one(o, d, owner, centers, radii, lo, hi, end, first, count, ids, root, primitive_groups, excluded_group):
    node, stop = root, end[root]
    boxes, tests = 0, 0
    while node < stop:
        boxes += 1
        if not box_hit(o, d, lo[node], hi[node]):
            node = end[node]
            continue
        for k in range(first[node], first[node]+count[node]):
            j = ids[k]
            if j != owner and (excluded_group < 0 or primitive_groups[j] != excluded_group):
                tests += 1
                if sphere_hit(o, d, centers[j], radii[j]):
                    return True, boxes, tests
        node += 1
    return False, boxes, tests


@njit(cache=True)
def tree_blocked(origins, directions, owners, centers, radii, lo, hi, end, first, count, ids):
    result = np.zeros(len(origins), dtype=np.bool_)
    boxes, tests = 0, 0
    for i in range(len(origins)):
        result[i], nb, nt = _one(origins[i], directions[i], owners[i], centers, radii,
                                 lo, hi, end, first, count, ids, 0, ids, -1)
        boxes += nb
        tests += nt
    return result, boxes, tests


@njit(cache=True)
def tree_other_blocked(origins, directions, owners, groups, centers, radii, lo, hi, end, first, count, ids):
    result = np.zeros(len(origins), dtype=np.bool_)
    boxes, tests = 0, 0
    for i in range(len(origins)):
        result[i], nb, nt = _one(origins[i], directions[i], owners[i], centers, radii,
                                 lo, hi, end, first, count, ids, 0, groups, groups[owners[i]])
        boxes += nb
        tests += nt
    return result, boxes, tests


class SphereIndex:
    def __init__(self, spheres, kind="bvh", leaf_size=4):
        self.spheres = spheres
        self.tree = build_tree(spheres.centers-spheres.radii[:,None], spheres.centers+spheres.radii[:,None], kind, leaf_size)

    def query(self, source):
        return tree_blocked(source.origins, source.directions, source.owners,
                            self.spheres.centers, self.spheres.radii, *self.tree.arrays)

    def refit(self, spheres):
        if len(spheres.radii) != len(self.spheres.radii):
            raise ValueError("Refit requires fixed primitive identities.")
        self.spheres = spheres
        self.tree.refit(spheres.centers-spheres.radii[:,None], spheres.centers+spheres.radii[:,None])


@njit(cache=True)
def node_blocked(origins, directions, owners, skip_groups, centers, radii,
                 top_lo, top_hi, top_end, top_first, top_count, top_ids,
                 lo, hi, end, first, count, ids, roots, rotations, translations):
    result = np.zeros(len(origins), dtype=np.bool_)
    boxes, tests, transforms = 0, 0, 0
    o, d = np.empty(3), np.empty(3)
    for i in range(len(origins)):
        node = 0
        while node < len(top_end) and not result[i]:
            boxes += 1
            if not box_hit(origins[i], directions[i], top_lo[node], top_hi[node]):
                node = top_end[node]
                continue
            for k in range(top_first[node], top_first[node]+top_count[node]):
                group = top_ids[k]
                if group == skip_groups[i]:
                    continue
                transforms += 1
                for a in range(3):
                    o[a], d[a] = 0., 0.
                    for b in range(3):
                        o[a] += rotations[group,b,a]*(origins[i,b]-translations[group,b])
                        d[a] += rotations[group,b,a]*directions[i,b]
                hit, nb, nt = _one(o, d, owners[i], centers, radii,
                                    lo, hi, end, first, count, ids, roots[group], ids, -1)
                boxes += nb
                tests += nt
                if hit:
                    result[i] = True
                    break
            node += 1
    return result, boxes, tests, transforms


class NodeAwareIndex:
    """Persistent local BVHs; only poses and top-level bounds change on refit."""
    def __init__(self, local_centers, radii, groups, rotations, translations, leaf_size=4):
        self.centers = np.ascontiguousarray(local_centers)
        self.radii = np.ascontiguousarray(radii)
        self.groups = np.asarray(groups, dtype=np.int64)
        if not np.array_equal(np.unique(self.groups), np.arange(len(rotations))):
            raise ValueError("Groups must be contiguous nonempty IDs.")
        trees, roots = [], []
        nnode, nids = 0, 0
        for group in range(len(rotations)):
            members = np.flatnonzero(self.groups == group)
            tree = build_tree(self.centers[members]-radii[members,None], self.centers[members]+radii[members,None], leaf_size=leaf_size)
            roots.append(nnode)
            tree.end += nnode
            tree.first += nids
            tree.ids = members[tree.ids]
            trees.append(tree)
            nnode += len(tree.end)
            nids += len(tree.ids)
        self.local = Tree(*(np.concatenate([t.arrays[k] for t in trees]) for k in range(6)))
        self.roots = np.array(roots, dtype=np.int64)
        self.rotations = np.ascontiguousarray(rotations)
        self.translations = np.ascontiguousarray(translations)
        lo, hi = self.world_bounds()
        self.top = build_tree(lo, hi, leaf_size=1)

    def world_bounds(self):
        a, b = self.local.lo[self.roots], self.local.hi[self.roots]
        center, half = (a+b)/2, (b-a)/2
        wc = np.einsum("gij,gj->gi", self.rotations, center)+self.translations
        wh = np.einsum("gij,gj->gi", np.abs(self.rotations), half)
        return wc-wh, wc+wh

    def refit(self, rotations, translations):
        self.rotations = np.ascontiguousarray(rotations)
        self.translations = np.ascontiguousarray(translations)
        self.top.refit(*self.world_bounds())

    def query(self, source, skip_groups=None):
        if skip_groups is None:
            skip_groups = np.full(len(source.owners), -1, dtype=np.int64)
        return node_blocked(source.origins, source.directions, source.owners, skip_groups,
                            self.centers, self.radii, *self.top.arrays, *self.local.arrays,
                            self.roots, self.rotations, self.translations)

    @property
    def nbytes(self):
        return self.local.nbytes+self.top.nbytes+sum(x.nbytes for x in
            [self.centers, self.radii, self.groups, self.roots, self.rotations, self.translations])


def world_spheres(local_centers, radii, groups, rotations, translations):
    # Preserve body/atom identity; caller excludes coincident world spheres.
    return Spheres(np.einsum("nij,nj->ni", rotations[groups], local_centers)+translations[groups], radii, deduplicate=False)


def pose_source(source, groups, rotations, translations):
    g = groups[source.owners]
    origins = np.einsum("nij,nj->ni", rotations[g], source.origins)+translations[g]
    dirs = np.einsum("nij,nj->ni", rotations[g], source.directions)
    return Source(np.ascontiguousarray(origins), np.ascontiguousarray(dirs), source.owners, source.area)


def sample_local_source(index, power, seed):
    world = world_spheres(index.centers, index.radii, index.groups, index.rotations, index.translations)
    if len(np.unique(np.column_stack((world.centers, world.radii)), axis=0)) != len(index.radii):
        raise ValueError("Coincident world spheres across groups require a dynamic ownership policy.")
    source = sample_source(world, power, seed)
    g = index.groups[source.owners]
    inverse = index.rotations[g].transpose(0,2,1)
    o = np.einsum("nij,nj->ni", inverse, source.origins-index.translations[g])
    d = np.einsum("nij,nj->ni", inverse, source.directions)
    return Source(np.ascontiguousarray(o), np.ascontiguousarray(d), source.owners, source.area)


@njit(cache=True)
def internal_blocked(source_o, source_d, owners, groups, centers, radii, lo, hi, end, first, count, ids, roots):
    result = np.zeros(len(owners), dtype=np.bool_)
    boxes, tests = 0, 0
    for i in range(len(owners)):
        result[i], nb, nt = _one(source_o[i], source_d[i], owners[i], centers, radii,
                                 lo, hi, end, first, count, ids, roots[groups[owners[i]]], groups, -1)
        boxes += nb
        tests += nt
    return result, boxes, tests


def compile_internal(index, local_source):
    return internal_blocked(local_source.origins, local_source.directions, local_source.owners,
                            index.groups, index.centers, index.radii, *index.local.arrays, index.roots)


def coupled_query(index, world_source, internal_mask):
    """Exact reuse of a fixed discrete local source's intrinsic escape mask.

    This is not a continuous boundary operator or a new-input field compiler.
    Every surviving sample is retested against other groups on every frame,
    allowing both shadowing and unshadowing.
    """
    active = np.flatnonzero(~internal_mask)
    subset = Source(world_source.origins[active], world_source.directions[active],
                    world_source.owners[active], world_source.area)
    other, boxes, tests, transforms = index.query(subset, index.groups[subset.owners])
    result = internal_mask.copy()
    result[active] = other
    return result, boxes, tests, transforms


def coupled_global_query(index, source, internal_mask, groups):
    """Give generic BVH/octree the same intrinsic sample reuse as node-aware."""
    active = np.flatnonzero(~internal_mask)
    other, boxes, tests = tree_other_blocked(source.origins[active], source.directions[active],
        source.owners[active], groups, index.spheres.centers, index.spheres.radii, *index.tree.arrays)
    result = internal_mask.copy()
    result[active] = other
    return result, boxes, tests, 0
