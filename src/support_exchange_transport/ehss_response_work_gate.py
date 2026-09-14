"""Execution admission by observed local response length, without branch pruning.

The score uses preparation weights only. Local owned paths may be interrupted
by foreign geometry: this is a workload heuristic, not predicted time savings.
Rejected whole cells fall back to the unchanged physical nearest-hit query.
"""
from copy import copy
import time
import numpy as np
from .ehss_sparse_path_response import readonly


def admit_response_cells(atlas, min_mean_owned_collisions=2.):
    tick = time.perf_counter()
    threshold = float(min_mean_owned_collisions)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('Finite nonnegative admission threshold required')
    if not atlas.prepared:
        raise ValueError('Prepared response atlas required')
    count = len(atlas.keys)
    if any(len(getattr(atlas, name)) != count for name in ('routes', 'route_weights', 'cell_mass', 'cell_visits', 'scores')):
        raise ValueError('Complete preparation statistics required')
    means = np.zeros(count)
    for i, (routes, weights, mass) in enumerate(zip(atlas.routes, atlas.route_weights, atlas.cell_mass)):
        w = np.asarray(weights, dtype=float)
        if len(routes) != len(w) or not np.isfinite(w).all() or np.any(w < 0) or not np.isfinite(mass) or mass <= 0:
            raise ValueError('Positive cell mass and finite nonnegative route weights required')
        if w.sum() > mass + 1e-10*max(1., mass):
            raise ValueError('Stored route mass exceeds cell mass')
        # The denominator includes zero-collision visits and omitted routes.
        # The numerator counts only the stored positive candidate paths.
        means[i] = sum(float(weight)*len(route) for weight, route in zip(w, routes))/mass
    chosen = np.flatnonzero(means >= threshold)
    result = copy(atlas)
    result.keys = readonly(atlas.keys[chosen].copy())
    result.routes = tuple(atlas.routes[i] for i in chosen)
    result.route_weights = tuple(atlas.route_weights[i] for i in chosen)
    for name in ('cell_mass', 'cell_visits', 'scores'):
        setattr(result, name, readonly(np.asarray(getattr(atlas, name))[chosen].copy()))
    owners = set(map(int, result.keys[:,0]))
    result.nodes = {node: stored for node, stored in atlas.nodes.items() if node in owners}
    receipt = dict(admission_s=time.perf_counter()-tick, threshold=threshold,
        input_cells=count, admitted_cells=len(chosen), input_owners=len(atlas.nodes), admitted_owners=len(owners),
        admitted_indices=chosen.tolist(), mean_stored_owned_collisions=means.tolist(),
        whole_cells_only=True, retained_candidate_routes_unchanged=True,
        preparation_avoided=False, foreign_interruption_accounted_in_score=False,
        policy='Sum of stored route weight times owned collision count / all observed cell mass; no held-out query information.')
    result.receipt = dict(getattr(atlas, 'receipt', {}), execution_admission=receipt)
    return result, receipt
