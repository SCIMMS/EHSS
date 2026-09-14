"""Sparse child-outgoing to parent-exit responses on a four-dimensional chart.

Cells are prepared explicitly from a calibration visit inventory, then frozen
for new queries. Certified THROUGH excludes all foreign physical balls for the
entire ray cell; own-child balls are behind an outward convex boundary ray.
APPROXIMATE passes nine probes, not a continuous error proof. Missing/unknown
cells use child-map composition starting AFTER the indicated child, never replay
the earlier first collision. Response preparation is separate from query costs.
"""
import time
import numpy as np
from .ehss_adaptive_response import chart_rays, ray_chart, certified_through, THROUGH, APPROXIMATE, UNKNOWN
from .ehss_hierarchical_response import unit_interval


class ChildExitResponse:
    def __init__(self, parent, child, shape=(8, 16, 8, 16), tolerance=.4, cap=32):
        if not parent.children or not 0 <= child < len(parent.children):
            raise ValueError('Existing child of a composed parent required')
        if len(shape) != 4 or any(not isinstance(v, int) or v < 1 for v in shape):
            raise ValueError('Four positive integer chart dimensions required')
        if not np.isfinite(tolerance) or tolerance <= 0 or not isinstance(cap, int) or cap < 1:
            raise ValueError('Positive tolerance and collision cap required')
        self.parent, self.child = parent, child
        self.pose = parent.children[child]
        self.shape, self.tolerance, self.cap = np.array(shape), tolerance, cap
        self.cells, self.visited = {}, {}
        self.collect = True
        self.counts = dict(requests=0, through=0, approximate=0, fallback=0, atom_tests=0)
        foreign = np.ones(len(parent.radii), dtype=bool)
        foreign[parent.offsets[child]:parent.offsets[child+1]] = False
        self.foreign_centers = (parent.centers[foreign]-self.pose.center)@self.pose.rotation/self.pose.radius
        self.foreign_radii = parent.radii[foreign]/self.pose.radius

    def coordinates(self, positions, directions):
        p = (positions-self.pose.center)@self.pose.rotation/self.pose.radius
        d = directions@self.pose.rotation
        if (not np.allclose(np.sum(p*p, axis=1), 1., rtol=0., atol=2e-10)
                or np.any(np.sum(p*d, axis=1) < -1e-11)):
            raise ValueError('An outward ray on this child boundary is required')
        u = ray_chart(p, d)
        indices = np.minimum(self.shape-1, np.floor(u*self.shape).astype(np.int64))
        return np.ravel_multi_index(indices.T, self.shape)

    def evaluate(self, positions, directions, cap):
        return self.parent.compose(positions, directions, cap, completed_child=self.child, use_ports=False)

    def prepare(self, max_cells=512):
        if not isinstance(max_cells, int) or max_cells < 0:
            raise ValueError('Nonnegative preparation budget required')
        start = time.perf_counter()
        chosen = sorted(self.visited, key=lambda i: (-self.visited[i], i))[:max_cells]
        tests = probes = 0
        new = 0
        for identity in chosen:
            if identity in self.cells:
                continue
            new += 1
            index = np.array(np.unravel_index(identity, self.shape))
            low, high = index/self.shape, (index+1)/self.shape
            if not len(self.foreign_radii) or certified_through(low, high, self.foreign_centers, self.foreign_radii):
                self.cells[identity] = dict(status=THROUGH)
                continue
            u = np.tile((low+high)/2, (9, 1))
            for axis in range(4):
                u[1+axis*2, axis] -= .375*(high[axis]-low[axis])
                u[2+axis*2, axis] += .375*(high[axis]-low[axis])
            incoming, d, _ = chart_rays(u)
            p = incoming+2*np.sqrt(1-u[:, 2])[:, None]*d
            p = self.pose.center+self.pose.radius*(p@self.pose.rotation.T)
            d = d@self.pose.rotation.T
            exits, outgoing, counts, stopped, paths, nt = self.evaluate(p, d, self.cap)
            probes += len(u)
            tests += nt
            same = np.all(paths == paths[0]) and not np.any(stopped)
            spread = max(np.linalg.norm(exits-exits[0], axis=1).max(), np.linalg.norm(outgoing-outgoing[0], axis=1).max())
            accept = same and counts[0] > 0 and spread <= self.tolerance
            self.cells[identity] = dict(status=APPROXIMATE if accept else UNKNOWN, count=int(counts[0]),
                                        position=exits[0].copy(), direction=outgoing[0].copy(), path=paths[0].copy())
        counts = {str(status): sum(c['status'] == status for c in self.cells.values()) for status in (THROUGH, APPROXIMATE, UNKNOWN)}
        payload = sum(sum(v.nbytes for v in c.values() if isinstance(v, np.ndarray)) for c in self.cells.values())
        return dict(new_cells=new, cells=len(self.cells), status_counts=counts, probe_rays=probes, atom_tests=int(tests),
                    prepare_s=time.perf_counter()-start, array_payload_bytes=payload)

    def apply(self, positions, directions, remaining):
        ids = self.coordinates(positions, directions)
        if self.collect:
            unique, counts = np.unique(ids, return_counts=True)
            for identity, count in zip(unique, counts):
                self.visited[int(identity)] = self.visited.get(int(identity), 0)+int(count)
        p, d = positions.copy(), directions.copy()
        c, stopped = np.zeros(len(ids), dtype=np.int64), np.zeros(len(ids), dtype=bool)
        paths = np.full((len(ids), int(remaining.max(initial=0))), -1, dtype=np.int32)
        through, approximate = np.zeros(len(ids), dtype=bool), np.zeros(len(ids), dtype=bool)
        for identity in np.unique(ids):
            cell = self.cells.get(int(identity))
            if cell is None:
                continue
            use = ids == identity
            if cell['status'] == THROUGH:
                through[use] = True
            elif cell['status'] == APPROXIMATE:
                use &= remaining >= cell['count']
                approximate[use] = True
                p[use], d[use], c[use] = cell['position'], cell['direction'], cell['count']
                width = min(self.cap, paths.shape[1])
                paths[use, :width] = cell['path'][:width]
        if np.any(through):
            _, far, valid = unit_interval(p[through], d[through])
            if not np.all(valid):
                raise RuntimeError('Child exit must be inside its parent')
            p[through] += far[:, None]*d[through]
        fallback = ~(through | approximate)
        tests = 0
        for cap in np.unique(remaining[fallback]):
            use = np.flatnonzero(fallback & (remaining == cap))
            ep, ed, ec, es, path, nt = self.evaluate(positions[use], directions[use], int(cap))
            p[use], d[use], c[use], stopped[use] = ep, ed, ec, es
            paths[use, :int(cap)] = path
            tests += nt
        self.counts['requests'] += len(ids)
        for name, mask in [('through', through), ('approximate', approximate), ('fallback', fallback)]:
            self.counts[name] += int(mask.sum())
        self.counts['atom_tests'] += tests
        return p, d, c, stopped, dict(through=int(through.sum()), approximate=int(approximate.sum()), fallback=int(fallback.sum()),
                                     primitive_tests=int(tests), collider_ids=paths)
