"""Calibrate response precision using paired exact downstream continuations.

The two local branches share the current incident state and initial-direction
tag. Subsequent transport is exact all-atom continuation of each branch. These
conditional differences rank cells; they are not additive global error bounds
for the original approximate trajectory or for simultaneous cell repairs.
"""
import numpy as np
from .ehss_adaptive_response import APPROXIMATE, probe_responses
from .ehss_reference import trace_all


def continued_readout(scene, group, positions, directions, counts, stopped, orders, incoming, cap):
    final_orders = orders + counts
    physical = directions.copy()
    escaped = np.zeros(len(counts), dtype=bool)
    residual = stopped.copy()
    active = np.flatnonzero(~stopped)
    world = scene.centers[group] + scene.radii[group]*positions[active]
    _, outgoing, leave, tail, bounce, _, tests = trace_all(
        world, directions[active], np.full(len(active), -1, dtype=np.int64),
        final_orders[active], scene.spheres.centers, scene.spheres.radii, cap)
    physical[active] = outgoing
    final_orders[active] = bounce
    escaped[active], residual[active] = leave, tail
    value = np.clip(1-np.sum(incoming*physical, axis=1), 0., 2.)*escaped
    return value, final_orders, residual, int(tests)


class DownstreamPriorityAudit:
    def __init__(self, atlas, scene, normalizer=1.):
        if not np.isfinite(normalizer) or normalizer <= 0:
            raise ValueError("Positive finite observable normalization required")
        self.atlas, self.scene, self.normalizer = atlas, scene, normalizer
        self.context = None
        self.cells = {}
        self.local_tests = self.continuation_tests = 0
        self.context_calls = self.context_rows = 0

    def set_context(self, *, group, incoming, weights, orders, ray_ids, max_bounces):
        self.context = dict(group=group, incoming=incoming, weights=weights, orders=orders,
                            ray_ids=ray_ids, max_bounces=max_bounces)
        self.context_calls += 1
        self.context_rows += len(weights)

    def apply(self, origins, directions, remaining):
        context = self.context
        if context is None or len(context["weights"]) != len(origins):
            raise ValueError("A fresh transport context is required for this query batch")
        self.context = None
        if not np.array_equal(remaining, context["max_bounces"]-context["orders"]):
            raise ValueError("Context and remaining physical collision budget differ")
        result = self.atlas.apply(origins, directions, remaining)
        p, d, counts, stopped, receipt = result
        ids = self.atlas.locate(origins, directions)
        eligible = (np.sum(origins*origins, axis=1) >= 1.-1e-12) & (np.sum(origins*directions, axis=1) < 0.)
        inspect = (self.atlas.status[ids] == APPROXIMATE) & eligible & (self.atlas.count[ids] <= remaining)
        for cap in np.unique(remaining[inspect]):
            query = np.flatnonzero(inspect & (remaining == cap))
            ep, ed, ec, es, paths, tests = probe_responses(origins[query], directions[query], self.atlas.centers, self.atlas.radii, int(cap))
            self.local_tests += tests
            mismatch = (ec != counts[query]) | (es != stopped[query]) | np.any(paths != receipt["collider_ids"][query, :int(cap)], axis=1)
            common = (context["orders"][query], context["incoming"][query], context["max_bounces"])
            av, ao, at, ant = continued_readout(self.scene, context["group"], p[query], d[query], counts[query], stopped[query], *common)
            ev, eo, et, ent = continued_readout(self.scene, context["group"], ep, ed, ec, es, *common)
            self.continuation_tests += ant+ent
            chord = np.linalg.norm(ed-d[query], axis=1)
            total_error = np.abs(av-ev)
            order_error = np.where(ao == eo, total_error, av+ev)
            uncertainty = 2.*(at.astype(float)+et.astype(float))
            weight = context["weights"][query]/self.normalizer
            for row, identity in enumerate(query):
                cell = int(ids[identity])
                entry = self.cells.setdefault(cell, dict(visits=0, weight=0., local_score=0., total_score=0.,
                                                        order_score=0., signed_difference=0., residual_allowance=0.,
                                                        local_branch_mismatches=0, final_order_mismatches=0))
                entry["visits"] += 1
                entry["weight"] += float(weight[row])
                entry["local_score"] += float(weight[row]*(2*int(mismatch[row])+chord[row]))
                entry["total_score"] += float(weight[row]*(total_error[row]+uncertainty[row]))
                entry["order_score"] += float(weight[row]*(order_error[row]+uncertainty[row]))
                entry["signed_difference"] += float(weight[row]*(av[row]-ev[row]))
                entry["residual_allowance"] += float(weight[row]*uncertainty[row])
                entry["local_branch_mismatches"] += int(mismatch[row])
                entry["final_order_mismatches"] += int(ao[row] != eo[row])
        return result
