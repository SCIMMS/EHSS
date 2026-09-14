"""Conservative whole-ball union-preserving removal with stable atom IDs.

Subdivide a box containing the target ball. A box outside that ball is irrelevant;
every other box must be wholly inside a retained neighbor or recursively covered.
Distance bounds certify whole volumes, never just sampled surface points.
Sequential deletion preserves the union. Failure to prove coverage keeps the
atom. This temporary certificate subdivision is not the transport hierarchy.
"""
import time
import numpy as np
from numba import njit
from scipy.spatial import cKDTree


@njit(cache=False)
def _covered(x, y, z, half, radius, candidates, relative, radii, margin, depth, checks):
    minimum2 = max(0., abs(x)-half)**2+max(0., abs(y)-half)**2+max(0., abs(z)-half)**2
    if minimum2 > (radius+margin)**2:
        return True
    for j in candidates:
        checks[0] += 1
        # Exact farthest corner distance from a sphere center to this box.
        maximum2 = ((abs(x-relative[j, 0])+half)**2+(abs(y-relative[j, 1])+half)**2
                    +(abs(z-relative[j, 2])+half)**2)
        if maximum2 < max(0., radii[j]-margin)**2:
            return True
    if depth == 0:
        return False
    small = half*.5
    for sx in (-1., 1.):
        for sy in (-1., 1.):
            for sz in (-1., 1.):
                if not _covered(x+sx*small, y+sy*small, z+sz*small, small, radius,
                                candidates, relative, radii, margin, depth-1, checks):
                    return False
    return True


@njit(cache=False)
def _simplify(world, radii, offsets, neighbors, depth, margin):
    active = np.ones(len(radii), dtype=np.bool_)
    checks = np.zeros(1, dtype=np.int64)
    order = np.argsort(radii)
    for i in order:
        candidates = []
        contained = False
        for k in range(offsets[i], offsets[i+1]):
            j = neighbors[k]
            if j == i or not active[j]:
                continue
            delta = world[i]-world[j]
            distance = np.sqrt(np.dot(delta, delta))
            if distance+radii[i]+margin < radii[j]:
                contained = True
                break
            if distance < radii[i]+radii[j]:
                candidates.append(j)
        if contained:
            active[i] = False
            continue
        if not candidates:
            continue
        ids = np.asarray(candidates, dtype=np.int64)
        relative = world-world[i]
        covered = _covered(0., 0., 0., radii[i], radii[i], ids, relative, radii, margin, depth, checks)
        if covered:
            active[i] = False
    return active, checks[0]


def active_atoms(spheres, depth=4):
    if type(depth) is not int or not 0 <= depth <= 8:
        raise ValueError('Coverage depth must be an integer from 0 through 8')
    start = time.perf_counter()
    world, radii = spheres.centers, spheres.radii
    margin = 1e-10*max(1., float(np.abs(world).max()), float(radii.max()))
    lists = cKDTree(world).query_ball_point(world, radii+radii.max())
    offsets = np.r_[0, np.cumsum([len(row) for row in lists])].astype(np.int64)
    neighbors = np.concatenate(lists).astype(np.int64)
    active, checks = _simplify(world, radii, offsets, neighbors, depth, margin)
    return active, dict(prepare_s=time.perf_counter()-start, active_atoms=int(active.sum()),
                       removed_atoms=int((~active).sum()), coverage_checks=int(checks), depth=depth,
                       guarantee='sequential whole-ball union coverage; unchanged original atom IDs')
