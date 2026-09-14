"""Minimal static EHSS preparation; no response or motion metadata.

The median leaf order, padded bounds and physical tracing kernel are identical
to the full preparation path. PreparedCPUEHSS remains the motion-aware API.
"""
from types import SimpleNamespace
import time
import numpy as np
from numba import njit
from .inside_out_pa import Spheres
from .ehss_cpu import CPUEHSSTransport, check_threads
from .ehss_cpu_geometry import topology, leaf_serial, parents
from .ehss_compact_source import CompactBoundary


@njit(cache=True)
def _median_groups(centers, leaf_size):
    n = len(centers)
    ids = np.arange(n)
    groups = np.empty(n, np.int64)
    # Depth-first stack: right is pushed before left, preserving leaf order.
    starts = np.empty(64, np.int64)
    stops = np.empty(64, np.int64)
    depths = np.empty(64, np.int64)
    starts[0], stops[0], depths[0] = 0, n, 0
    pending, group = 1, 0
    while pending:
        pending -= 1
        a, b, depth = starts[pending], stops[pending], depths[pending]
        span = np.empty(3)
        for c in range(3):
            low, high = np.inf, -np.inf
            for k in range(a, b):
                v = centers[ids[k], c]
                low, high = min(low, v), max(high, v)
            span[c] = high-low
        axis = np.argmax(span)
        if b-a <= leaf_size or depth >= 24 or span[axis] == 0:
            for k in range(a, b):
                groups[ids[k]] = group
            group += 1
            continue
        values = centers[ids[a:b], axis]
        order = np.argsort(values, kind='mergesort')
        ids[a:b] = ids[a:b][order]
        mid = a+(b-a)//2
        starts[pending], stops[pending], depths[pending] = mid, b, depth+1
        starts[pending+1], stops[pending+1], depths[pending+1] = a, mid, depth+1
        pending += 2
    return groups


def static_spatial_groups(spheres, leaf_size=4):
    if type(leaf_size) is not int or leaf_size < 1:
        raise ValueError('Positive integer leaf size required')
    # Preserve the original PA builder's floating-point center calculation.
    lo = spheres.centers-spheres.radii[:, None]
    hi = spheres.centers+spheres.radii[:, None]
    return _median_groups((lo+hi)/2, leaf_size)


class PreparedStaticCPUEHSS:
    """Owned geometry for repeated sampling of one unchanged structure."""
    def __init__(self, spheres, groups=None, leaf_size=4, threads=1,
                 boundary='sphere', simplify=False, coverage_depth=4):
        check_threads(threads)
        start = time.perf_counter()
        # Own the coordinates so caller mutations cannot invalidate bounds.
        s = Spheres(spheres.centers, spheres.radii, deduplicate=False)
        tick = time.perf_counter()
        groups = static_spatial_groups(s, leaf_size) if groups is None else np.asarray(groups)
        if groups.shape != (len(s.radii),) or not np.issubdtype(groups.dtype, np.integer):
            raise ValueError('One integer group per atom required')
        unique = np.unique(groups)
        if not np.array_equal(unique, np.arange(len(unique))):
            raise ValueError('Contiguous nonempty groups required')
        groups = groups.astype(np.int64, copy=True)
        group_s = time.perf_counter()-tick
        tick = time.perf_counter()
        ng = len(unique)
        left, right, end, parent, leaf_group, leaf_node, depth = topology(ng)
        offsets = np.r_[0, np.cumsum(np.bincount(groups))].astype(np.int64)
        ids = np.argsort(groups, kind='stable')
        lo, hi = np.empty((len(end), 3)), np.empty((len(end), 3))
        changed = np.ones(ng, dtype=np.bool_)
        pad = 1e-10*max(1., float(s.radii.max()))
        leaf_serial(s.centers, s.radii, changed, leaf_node, offsets, ids, lo, hi, pad)
        parents(changed, leaf_node, parent, left, right, lo, hi)
        tree = SimpleNamespace(lo=lo, hi=hi, left=left, right=right, end=end,
            leaf_group=leaf_group, offsets=offsets, ids=ids, groups=groups)
        self.model = SimpleNamespace(spheres=s, tree=tree)
        bounds_s = time.perf_counter()-tick
        tick = time.perf_counter()
        self.boundary = CompactBoundary.fit(s, boundary)
        boundary_s = time.perf_counter()-tick
        self.engine = CPUEHSSTransport(self.model, threads=threads,
            simplify=simplify, coverage_depth=coverage_depth)
        self.setup = dict(total_s=time.perf_counter()-start, group_s=group_s,
            bounds_s=bounds_s, boundary_s=boundary_s, engine_s=self.engine.prepare_s,
            threads=threads, geometry_threads=1, preparation='static',
            response_metadata=False, exclusivity_computed=False)
