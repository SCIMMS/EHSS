"""Isolate the first outgoing projection from later boundary-map errors.

The exact first owned-support response is shared with the compiled solver.
After its optional outgoing projection, all subsequent physical collisions use
the independent all-atom kernel. Numerical direction changes are not counted as
physical scattering: rays with no subsequent collision keep their last physical
direction for the EHSS readout. This is a diagnostic, not a compiled solver.
"""
import numpy as np

from .ehss_compiled_field import initial_exact_response
from .ehss_reference import trace_all


def prefix_projection_control(scene, source, grid=None, max_bounces=32):
    if not isinstance(max_bounces, int) or max_bounces < 1 or np.any(source.initial_bounces > 1):
        raise ValueError("Positive cap and uncollided/first-reflected source required")
    if np.any(source.last_atom >= len(scene.spheres.radii)) or np.any(source.last_atom < -1):
        raise ValueError("Invalid source atom identity")
    positions, outgoing, orders, owners, seed_tests = initial_exact_response(
        source.origins, source.directions, source.last_atom, source.initial_bounces,
        scene.spheres.centers, scene.spheres.radii, scene.groups, scene.packed_ids,
        scene.offsets, scene.centers, scene.radii, max_bounces)
    active = np.flatnonzero(owners >= 0)
    physical = outgoing.copy()
    if grid is not None:
        groups = owners[active]
        cells = grid.project((positions[active] - scene.centers[groups]) / scene.radii[groups, None], outgoing[active])
        positions[active] = scene.centers[groups] + scene.radii[groups, None] * grid.points(cells, outgoing=True)
        outgoing[active] = grid.directions[cells]
    escaped = owners == -1
    unresolved = owners == -2
    _, continued, leave, stop, counts, _, continuation_tests = trace_all(
        positions[active], outgoing[active], np.full(len(active), -1, dtype=np.int64),
        orders[active], scene.spheres.centers, scene.spheres.radii, max_bounces)
    reflected = counts > orders[active]
    physical[active[reflected]] = continued[reflected]
    escaped[active] = leave
    unresolved[active] = stop
    orders[active] = counts
    contribution = source.weights * np.clip(1 - np.sum(source.incoming * physical, axis=1), 0., 2.) * escaped
    mass = np.bincount(orders[escaped], weights=source.weights[escaped], minlength=max_bounces + 1)
    omega = np.bincount(orders, weights=contribution, minlength=max_bounces + 1)
    residual = float(source.weights[unresolved].sum())
    assert np.isclose(mass.sum() + residual, source.weights.sum(), rtol=1e-12, atol=1e-12)
    return dict(omega=float(contribution.sum()), residual_mass=residual, tail_bound=2 * residual,
                order_ledger=[dict(order=k, mass=float(mass[k]), omega=float(omega[k])) for k in range(max_bounces + 1)],
                bounces=orders, outgoing=physical, contributions=contribution,
                seed_primitive_tests=int(seed_tests), continuation_primitive_tests=int(continuation_tests))
