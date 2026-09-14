"""Field-weighted local error audit and selected-cell exact response repair.

Scores are calibration indicators, not bounds on the final multiple-scattering
observable. The underlying atlas and its whole-cell THROUGH certificate remain
unchanged. Unselected approximate responses retain their original approximation.
"""
import numpy as np
from .ehss_adaptive_response import APPROXIMATE, probe_responses


class ResponsePriorityProxy:
    def __init__(self, atlas, selected=(), audit_weight=None, repair_all=False):
        self.atlas = atlas
        self.selected = np.asarray(sorted(selected), dtype=np.int64)
        self.audit_weight = audit_weight
        self.repair_all = repair_all
        self.cells = {}
        self.repaired = self.audit_tests = 0

    def apply(self, origins, directions, remaining):
        p, d, counts, stopped, receipt = self.atlas.apply(origins, directions, remaining)
        ids = self.atlas.locate(origins, directions)
        eligible = (np.sum(origins*origins, axis=1) >= 1.-1e-12) & (np.sum(origins*directions, axis=1) < 0.)
        approximate = (self.atlas.status[ids] == APPROXIMATE) & eligible & (self.atlas.count[ids] <= remaining)
        repair = approximate & (self.repair_all | np.isin(ids, self.selected))
        inspect = approximate if self.audit_weight is not None else repair
        for cap in np.unique(remaining[inspect]):
            query = np.flatnonzero(inspect & (remaining == cap))
            ep, ed, ec, es, paths, tests = probe_responses(origins[query], directions[query], self.atlas.centers, self.atlas.radii, int(cap))
            if self.audit_weight is not None:
                self.audit_tests += tests
                local_paths = receipt["collider_ids"][query, :int(cap)]
                mismatch = (ec != counts[query]) | (es != stopped[query]) | np.any(paths != local_paths, axis=1)
                chord = np.linalg.norm(ed-d[query], axis=1)
                for row, identity in enumerate(query):
                    cell = int(ids[identity])
                    value = self.cells.setdefault(cell, dict(visits=0, weight=0., score=0., branch_mismatches=0, max_direction_chord=0.))
                    value["visits"] += 1
                    value["weight"] += self.audit_weight
                    value["score"] += self.audit_weight * (2*int(mismatch[row]) + chord[row])
                    value["branch_mismatches"] += int(mismatch[row])
                    value["max_direction_chord"] = max(value["max_direction_chord"], float(chord[row]))
            use = repair[query]
            replace = query[use]
            p[replace], d[replace], counts[replace], stopped[replace] = ep[use], ed[use], ec[use], es[use]
            receipt["collider_ids"][replace, :int(cap)] = paths[use]
            # Audit cost is accounted separately. Production repair evaluates
            # only selected requests, so its tests belong in the solver receipt.
            if self.audit_weight is None:
                receipt["primitive_tests"] += int(tests)
        amount = int(repair.sum())
        self.repaired += amount
        receipt["approximate"] -= amount
        receipt["fallback"] += amount
        return p, d, counts, stopped, receipt
