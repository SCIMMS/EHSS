"""Replay only unresolved approximate rays from their original source state.

Continuing an approximate endpoint exactly cannot undo earlier branch errors.
Original source replay preserves the initial-direction tag and PA first-hit
prefix. The exact collision cap is unchanged; genuine tails remain unresolved.
Escaped approximate rays remain approximate and are not certified by this guard.
"""
import time
import numpy as np
from .ehss_state import EHSSSource
from .ehss_reference import trace


def replay_residual(spheres, source, result):
    start = time.perf_counter()
    escaped = np.asarray(result['escaped'], dtype=bool)
    unresolved = np.asarray(result['unresolved'], dtype=bool)
    if escaped.shape != source.weights.shape or unresolved.shape != escaped.shape or not np.all(escaped ^ unresolved):
        raise ValueError('Every source ray must be escaped or unresolved, exclusively')
    cap = result['collider_ids'].shape[1]
    ids = np.flatnonzero(unresolved)
    receipt = dict(replayed_rays=len(ids), replayed_mass=float(source.weights[ids].sum()),
                   resolved_rays=0, still_unresolved_rays=0, sphere_tests=0, replay_s=0.)
    # Preserve input arrays and all existing preparation/approximate-query costs.
    corrected = dict(result)
    if not len(ids):
        receipt['replay_s'] = time.perf_counter()-start
        corrected['residual_replay'] = receipt
        return corrected
    subset = EHSSSource(source.origins[ids], source.directions[ids], source.incoming[ids], source.weights[ids],
                        source.last_atom[ids], source.initial_bounces[ids], 'original-source residual replay')
    exact = trace(spheres, subset, cap)
    replacement = dict(escaped=exact.escaped, unresolved=exact.unresolved, bounces=exact.bounces,
                       outgoing=exact.outgoing, collider_ids=exact.collider_ids, contributions=exact.contributions)
    for key, value in replacement.items():
        corrected[key] = result[key].copy()
        corrected[key][ids] = value
    # Every exchange-limited ray was replayed by a solver without exchange caps.
    corrected['exchange_unresolved'] = np.zeros(len(source.weights), dtype=bool)
    corrected['exchange_tail_mass'] = 0.
    orders, escaped = corrected['bounces'], corrected['escaped']
    mass = np.bincount(orders[escaped], weights=source.weights[escaped], minlength=cap+1)
    omega = np.bincount(orders, weights=corrected['contributions'], minlength=cap+1)
    corrected['omega'] = float(corrected['contributions'].sum())
    corrected['residual_mass'] = float(source.weights[corrected['unresolved']].sum())
    corrected['tail_bound'] = 2*corrected['residual_mass']
    corrected['order_ledger'] = [dict(order=k, mass=float(mass[k]), omega=float(omega[k])) for k in range(cap+1)]
    if not np.isclose(mass.sum()+corrected['residual_mass'], source.weights.sum(), rtol=1e-12, atol=1e-12):
        raise AssertionError('Replay changed the source mass balance')
    receipt.update(resolved_rays=int(exact.escaped.sum()), still_unresolved_rays=int(exact.unresolved.sum()),
                   sphere_tests=exact.metrics['sphere_tests'], replay_s=time.perf_counter()-start)
    corrected['residual_replay'] = receipt
    return corrected
