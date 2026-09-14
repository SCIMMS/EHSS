"""Ascend from a known first-reflected one-atom leaf through parent exit ports.

Only a validated physical sphere surface with outward direction can skip its
own leaf kernel. Multi-atom leaves and uncollided/mismatched states use the
ordinary root response. Parent continuations may return to the original atom
after another collision; only the already completed initial traversal is skipped.
"""
import numpy as np
from .ehss_hierarchical_response import HierarchicalResponseTransport, unit_interval
from .ehss_path_append import append_collider_paths


class InsideOutHierarchyTransport(HierarchicalResponseTransport):
    def __init__(self, root, center=None, radius=1.):
        super().__init__(root, center, radius)
        self.seed_paths = {}
        def visit(node, offset, chain):
            if not node.children:
                if len(node.radii) == 1:
                    self.seed_paths[offset] = (node, chain)
                return
            for index, pose in enumerate(node.children):
                visit(pose.node, offset+int(node.offsets[index]), chain+[(node, index, offset)])
        visit(root, 0, [])

    def source_response(self, source, selected, points, remaining):
        directions = source.directions[selected]
        p, d = points.copy(), directions.copy()
        counts = np.zeros(len(points), dtype=np.int64)
        stopped = np.zeros(len(points), dtype=bool)
        paths = np.full((len(points), int(remaining.max(initial=0))), -1, dtype=np.int32)
        seeded = np.zeros(len(points), dtype=bool)
        tests = 0
        for atom in np.unique(source.last_atom[selected]):
            if atom not in self.seed_paths:
                continue
            leaf, chain = self.seed_paths[atom]
            rows = np.flatnonzero((source.last_atom[selected] == atom) & (source.initial_bounces[selected] == 1))
            lp, ld = points[rows].copy(), directions[rows].copy()
            for parent, index, _ in chain:
                pose = parent.children[index]
                lp, ld = (lp-pose.center)@pose.rotation/pose.radius, ld@pose.rotation
            normal = (lp-leaf.centers[0])/leaf.radii[0]
            valid = np.isclose(np.sum(normal*normal, axis=1), 1., rtol=0., atol=2e-10) & (np.sum(normal*ld, axis=1) >= -1e-12)
            rows, lp, ld = rows[valid], lp[valid], ld[valid]
            if not len(rows):
                continue
            seeded[rows] = True
            _, far, hit = unit_interval(lp, ld)
            if not np.all(hit):
                raise RuntimeError('Physical sphere surface must lie inside its leaf envelope')
            lp += far[:, None]*ld
            for parent, index, offset in reversed(chain):
                pose = parent.children[index]
                lp, ld = pose.center+pose.radius*(lp@pose.rotation.T), ld@pose.rotation.T
                active = np.flatnonzero(~stopped[rows])
                ids = rows[active]
                budget = remaining[ids]-counts[ids]
                if index in parent.exit_ports:
                    ep, ed, ec, es, er = parent.exit_ports[index].apply(lp[active], ld[active], budget)
                    tests += er['primitive_tests']
                    cp = er['collider_ids']
                    append_collider_paths(paths,ids,counts,ec,cp,offset)
                    counts[ids] += ec
                    stopped[ids] = es
                    lp[active], ld[active] = ep, ed
                else:
                    for cap in np.unique(budget):
                        use = active[budget == cap]
                        target = rows[use]
                        ep, ed, ec, es, cp, nt = parent.compose(lp[use], ld[use], int(cap), completed_child=index)
                        tests += nt
                        append_collider_paths(paths,target,counts,ec,cp,offset)
                        counts[target] += ec
                        stopped[target] = es
                        lp[use], ld[use] = ep, ed
            p[rows], d[rows] = lp, ld
        rest = np.flatnonzero(~seeded)
        ep, ed, ec, es, receipt = self.root.apply(points[rest], directions[rest], remaining[rest])
        p[rest], d[rest], counts[rest], stopped[rest] = ep, ed, ec, es
        paths[rest, :receipt['collider_ids'].shape[1]] = receipt['collider_ids']
        tests += receipt['primitive_tests']
        return p, d, counts, stopped, dict(collider_ids=paths, primitive_tests=int(tests), native_source_rays=int(seeded.sum()))
