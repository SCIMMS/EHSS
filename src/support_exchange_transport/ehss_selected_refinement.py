"""Graft smaller response atlases onto selected leaves of an immutable atlas."""
import time
import numpy as np
from .ehss_adaptive_response import AdaptiveLocalResponse, APPROXIMATE


def leaf_regions(atlas, selected):
    wanted = set(map(int, selected))
    if any(i < 0 or i >= len(atlas.left) or atlas.left[i] >= 0 for i in wanted):
        raise ValueError("Refinement targets must be existing leaf identities")
    regions = {}
    pending = [(0, atlas.region[0].copy(), atlas.region[1].copy())]
    while pending:
        identity, low, high = pending.pop()
        if atlas.left[identity] < 0:
            if identity in wanted:
                regions[identity] = np.array([low, high])
            continue
        axis, split = atlas.axis[identity], atlas.split[identity]
        left_high, right_low = high.copy(), low.copy()
        left_high[axis] = right_low[axis] = split
        child = atlas.left[identity]
        pending.extend(((int(child), low.copy(), left_high), (int(child+1), right_low, high.copy())))
    return regions


class SelectedResponseRefinement:
    def __init__(self, atlas, selected, leaf_budget=32, tolerance=.1):
        self.atlas = atlas
        self.patches = {}
        self.refined_queries = 0
        start = time.perf_counter()
        for identity, region in leaf_regions(atlas, selected).items():
            if atlas.status[identity] != APPROXIMATE:
                raise ValueError("This graft refines approximate response leaves only")
            self.patches[identity] = AdaptiveLocalResponse(atlas.centers, atlas.radii, leaf_budget=leaf_budget,
                                                         tolerance=tolerance, cap=atlas.cap, region=region)
        self.compile_s = time.perf_counter()-start
        self.selected = np.array(sorted(self.patches), dtype=np.int64)

    def apply(self, origins, directions, remaining):
        p, d, counts, stopped, receipt = self.atlas.apply(origins, directions, remaining)
        ids = self.atlas.locate(origins, directions)
        eligible = (np.sum(origins*origins, axis=1) >= 1.-1e-12) & (np.sum(origins*directions, axis=1) < 0.)
        chosen = eligible & (self.atlas.count[ids] <= remaining) & np.isin(ids, self.selected)
        self.refined_queries += int(chosen.sum())
        for identity in np.unique(ids[chosen]):
            use = np.flatnonzero(chosen & (ids == identity))
            pp, pd, pc, ps, detail = self.patches[int(identity)].apply(origins[use], directions[use], remaining[use])
            p[use], d[use], counts[use], stopped[use] = pp, pd, pc, ps
            paths = detail.pop("collider_ids")
            receipt["collider_ids"][use] = -1
            receipt["collider_ids"][use, :paths.shape[1]] = paths
            receipt["approximate"] -= len(use)
            for key, value in detail.items():
                receipt[key] += value
        return p, d, counts, stopped, receipt

    def receipt(self):
        return dict(selected_leaves=len(self.patches), added_leaves=sum(p.receipt["leaves"] for p in self.patches.values()),
                    payload_bytes=sum(p.receipt["payload_bytes"] for p in self.patches.values())+self.selected.nbytes,
                    probe_tests=sum(p.receipt["probe_tests"] for p in self.patches.values()),
                    compile_s=self.compile_s)
