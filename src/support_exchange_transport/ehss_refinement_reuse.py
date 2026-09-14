"""Reuse an unchanged intrinsic response and its selected refinement patches.

Scene binding verifies normalized owned geometry exactly. This supports the
spacing experiment's unchanged supports, not general MD strain or rotated
asymmetric support maps. Coupling/priority calibration is not cached here.
"""
import time
import numpy as np
from .ehss_adaptive_response import AdaptiveInterfaceTransport
from .ehss_selected_refinement import SelectedResponseRefinement, leaf_regions


def bind_intrinsic_response(scene, atlas):
    for group in range(len(scene.radii)):
        owned = scene.groups == group
        centers = (scene.spheres.centers[owned]-scene.centers[group])/scene.radii[group]
        radii = scene.spheres.radii[owned]/scene.radii[group]
        if not np.array_equal(centers, atlas.centers) or not np.array_equal(radii, atlas.radii):
            raise ValueError("Normalized physical geometry changed; this response cannot be reused")
    model = object.__new__(AdaptiveInterfaceTransport)
    model.scene = scene
    model.maps = [atlas]*len(scene.radii)
    model.library = {"verified_intrinsic_geometry": atlas}
    model.compile_s = 0.
    return model


class RefinementPatchPool:
    def __init__(self, atlas, leaf_budget=512, tolerance=.1):
        self.atlas, self.leaf_budget, self.tolerance = atlas, leaf_budget, tolerance
        self.patches = {}

    def select(self, identities):
        start = time.perf_counter()
        selected = np.array(sorted(set(map(int, identities))), dtype=np.int64)
        leaf_regions(self.atlas, selected)  # Validate identities before mutation.
        missing = [int(i) for i in selected if int(i) not in self.patches]
        compiled = SelectedResponseRefinement(self.atlas, missing, self.leaf_budget, self.tolerance)
        self.patches.update(compiled.patches)
        current = SelectedResponseRefinement(self.atlas, [])
        current.patches = {int(i): self.patches[int(i)] for i in selected}
        current.selected = selected
        current.compile_s = time.perf_counter()-start
        receipt = dict(selected=len(selected), new_patches=len(missing), reused_patches=len(selected)-len(missing),
                       prepare_s=current.compile_s, new_probe_tests=sum(p.receipt["probe_tests"] for p in compiled.patches.values()),
                       new_payload_bytes=sum(p.receipt["payload_bytes"] for p in compiled.patches.values()),
                       pooled_patches=len(self.patches), pooled_payload_bytes=sum(p.receipt["payload_bytes"] for p in self.patches.values()))
        return current, receipt
