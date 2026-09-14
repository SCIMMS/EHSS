"""Finite-validation selection for candidate response maps, not certification.

Compare complete deployed solvers including residual replay. Each source family
must improve mean total-error envelope without increasing its worst validation
error, mean collision-order error envelope, or worst unresolved tail. A separate
locked test set is still required; these empirical checks imply no population
guarantee and do not enforce equal runtime or preparation cost.
"""
import numpy as np


def choose_validated_candidate(rows, baseline='base'):
    inventory = {}
    for row in rows:
        key = (str(row['source']), int(row['seed']))
        candidate = str(row['candidate'])
        entries = inventory.setdefault(candidate, {})
        if key in entries:
            raise ValueError('Duplicate candidate/source/seed record')
        values = np.array([row[k] for k in ('total_error', 'order_error', 'tail', 'atom_tests')], dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Finite nonnegative diagnostics required')
        entries[key] = values
    if baseline not in inventory:
        raise ValueError('Baseline validation records required')
    keys = set(inventory[baseline])
    if any(set(v) != keys for v in inventory.values()):
        raise ValueError('Every candidate must use the same source/seed inventory')
    sources = sorted({source for source, _ in keys})
    if any(sum(s == source for s, _ in keys) < 2 for source in sources):
        raise ValueError('At least two independent seeds per source required')
    summaries = {}
    for candidate, records in inventory.items():
        groups = {}
        for source in sources:
            data = np.array([records[k] for k in sorted(keys) if k[0] == source])
            total, order = data[:, 0]+data[:, 2], data[:, 1]+data[:, 2]
            groups[source] = dict(mean_total=float(total.mean()), worst_total=float(total.max()),
                                  mean_order=float(order.mean()), worst_tail=float(data[:, 2].max()),
                                  mean_atom_tests=float(data[:, 3].mean()))
        summaries[candidate] = groups
    decisions = {}
    slack = 64*np.finfo(float).eps
    for candidate, groups in summaries.items():
        reasons = []
        if candidate != baseline:
            for source, value in groups.items():
                ref = summaries[baseline][source]
                if value['mean_total'] >= ref['mean_total']-slack:
                    reasons.append(source+': mean total error did not improve')
                for key in ('worst_total', 'mean_order', 'worst_tail'):
                    if value[key] > ref[key]+slack:
                        reasons.append(source+': '+key+' increased')
        decisions[candidate] = dict(eligible=not reasons, reasons=reasons, sources=groups)
    eligible = [c for c, d in decisions.items() if d['eligible']]
    def score(candidate):
        groups = summaries[candidate].values()
        return (np.mean([g['mean_total'] for g in groups]), np.mean([g['mean_atom_tests'] for g in groups]), candidate)
    selected = min(eligible, key=score)
    return dict(selected=selected, baseline=baseline, candidates=decisions,
                policy='finite validation: per-source mean/worst total, order, and tail; cost only breaks ties')
