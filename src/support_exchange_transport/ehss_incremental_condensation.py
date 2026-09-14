"""Retain unchanged first-exit columns across compact state renumbering.

Semantic tokens identify response ports; current dense indices are bindings.
Exact row fingerprints include escape moments, residuals, collision costs and
target ownership. Child fingerprints propagate changes to ancestors. Geometry
coupling and the row comparison still run upstream/globally; this module only
updates hierarchy responses that depend on changed rows.
"""
from types import MappingProxyType
import hashlib
import time
import numpy as np
from .ehss_field_condensation import FirstExit, _field, _result
from .ehss_sparse_path_response import readonly


def fingerprint(parts):
    h = hashlib.sha256()
    for part in parts:
        h.update(len(part).to_bytes(8, 'little'))
        h.update(part)
    return h.hexdigest()


class IncrementalCondensedField:
    def __init__(self, kernel, root, labels, previous=None, scope=None):
        tick = time.perf_counter()
        self.kernel, self.root, self.scope = kernel, root, scope
        if previous is not None and previous.scope is not scope:
            raise ValueError('Different semantic response scope')
        labels = tuple(labels)
        if len(labels) != len(kernel.owners) or len(set(labels)) != len(labels):
            raise ValueError('Unique semantic label per current state required')
        if root.owners != frozenset(map(int, kernel.owners)):
            raise ValueError('Root must cover every physical owner')
        self.identities = {} if previous is None else previous.identities.copy()
        for label in labels:
            if label not in self.identities:
                self.identities[label] = len(self.identities)
        self.tokens = readonly(np.array([self.identities[label] for label in labels], dtype=np.int64))
        self.dense = {int(token): i for i, token in enumerate(self.tokens)}
        self.row_signatures = {}
        self.owner_tokens = {}
        for i, token in enumerate(self.tokens):
            owner = int(kernel.owners[i])
            self.owner_tokens.setdefault(owner, []).append(int(token))
            start, stop = kernel.starts[i:i+2]
            # Aggregate duplicate semantic edges in a stable token order.
            edges = {}
            for target, probability in zip(kernel.targets[start:stop], kernel.probabilities[start:stop]):
                key = (int(self.tokens[target]), int(kernel.owners[target]))
                edges[key] = edges.get(key, 0.) + float(probability)
            parts = [np.array([owner, kernel.costs[i]], dtype=np.int64).tobytes(),
                     kernel.escape[i].tobytes(), kernel.residual[i:i+1].tobytes()]
            for key, probability in sorted(edges.items()):
                parts.extend((np.array(key, dtype=np.int64).tobytes(), np.float64(probability).tobytes()))
            self.row_signatures[int(token)] = fingerprint(parts)
        self.nodes, self.child_for, self.signatures = {}, {}, {}

        def visit(node):
            if node.name in self.nodes:
                raise ValueError('Unique hierarchy names required')
            self.nodes[node.name] = node
            self.child_for[node.name] = {owner: child for child in node.children for owner in child.owners}
            parts = [np.array(sorted(node.owners), dtype=np.int64).tobytes()]
            if node.children:
                for child in node.children:
                    visit(child)
                    parts.extend((child.name.encode(), self.signatures[child.name].encode()))
            else:
                tokens = sorted(token for owner in node.owners for token in self.owner_tokens.get(owner, []))
                for token in tokens:
                    parts.extend((np.int64(token).tobytes(), self.row_signatures[token].encode()))
            self.signatures[node.name] = fingerprint(parts)

        visit(root)
        self.cache = {} if previous is None else {
            key: value for key, value in previous.cache.items()
            if self.signatures.get(key[0]) == key[1] and key[2] in self.dense}
        self.retained = frozenset(self.cache)
        self.stats = {name: dict(cache_hits=0, retained_hits=0, compiled_columns=0,
                                elementary_rows=0, child_applications=0) for name in self.nodes}
        old_rows = {} if previous is None else previous.row_signatures
        changed = [token for token, signature in self.row_signatures.items() if old_rows.get(token) != signature]
        removed = sorted(set(old_rows)-set(self.row_signatures))
        self.receipt = dict(prepare_s=time.perf_counter()-tick, changed_rows=len(changed), removed_rows=len(removed),
            unchanged_rows=len(self.row_signatures)-len(changed),
            rebuilt_nodes=[name for name, signature in self.signatures.items()
                           if previous is None or previous.signatures.get(name) != signature],
            unchanged_nodes=[name for name, signature in self.signatures.items()
                             if previous is not None and previous.signatures.get(name) == signature],
            retained_columns=len(self.cache), dropped_columns=0 if previous is None else len(previous.cache)-len(self.cache),
            compact_renumbering_allowed=True, ancestor_dependency_propagation=True,
            row_scan_still_global=True, geometry_coupling_updated_here=False,
            exact_coefficient_reuse=True, historical_invalid_columns_retained=False)

    def response(self, node, state, budget):
        """Public dense-state entry; returned exits use stable semantic tokens."""
        if state < 0 or state >= len(self.tokens) or self.kernel.owners[state] not in node.owners or budget < 0:
            raise ValueError('Invalid parent entry or budget')
        return self._response(node, int(self.tokens[state]), budget)

    def _response(self, node, token, budget):
        key = (node.name, self.signatures[node.name], token, int(budget))
        stats = self.stats[node.name]
        if key in self.cache:
            stats['cache_hits'] += 1
            stats['retained_hits'] += int(key in self.retained)
            return self.cache[key]
        k = self.kernel
        pending = [{} for _ in range(budget+1)]
        pending[0][token] = 1.
        exits, escape = {}, {}
        local_tail = budget_tail = 0.
        for used, layer in enumerate(pending):
            for current, weight in layer.items():
                index = self.dense[current]
                if node.children:
                    child = self.child_for[node.name][int(k.owners[index])]
                    sub = self._response(child, current, budget-used)
                    stats['child_applications'] += 1
                else:
                    stats['elementary_rows'] += 1
                    cost = int(k.costs[index])
                    if cost > budget-used:
                        sub = FirstExit({}, {}, 0., 1.)
                    else:
                        start, stop = k.starts[index:index+2]
                        edges = {}
                        for target, probability in zip(k.targets[start:stop], k.probabilities[start:stop]):
                            edge = (int(self.tokens[target]), cost)
                            edges[edge] = edges.get(edge, 0.) + float(probability)
                        escaped = {cost: k.escape[index]} if k.escape[index, 0] > 0 else {}
                        sub = FirstExit(edges, escaped, float(k.residual[index]), 0.)
                local_tail += weight*sub.local_tail
                budget_tail += weight*sub.budget_tail
                for cost, moment in sub.escape.items():
                    order = used+cost
                    escape[order] = escape.get(order, np.zeros(4)) + weight*moment
                for (target, cost), probability in sub.exits.items():
                    if cost < 1:
                        raise AssertionError('Positive physical collision cost required')
                    order, mass = used+cost, weight*probability
                    if k.owners[self.dense[target]] in node.owners:
                        pending[order][target] = pending[order].get(target, 0.)+mass
                    else:
                        port = (target, order)
                        exits[port] = exits.get(port, 0.)+mass
        total = sum(exits.values())+sum(v[0] for v in escape.values())+local_tail+budget_tail
        if not np.isclose(total, 1., rtol=1e-11, atol=1e-12):
            raise AssertionError('First-exit probability changed')
        result = FirstExit(MappingProxyType(exits), MappingProxyType({i: readonly(v) for i, v in escape.items()}),
                           local_tail, budget_tail)
        self.cache[key] = result
        stats['compiled_columns'] += 1
        return result

    def solve(self, states, orders, values, max_bounces=64):
        tick = time.perf_counter()
        field = _field(states, orders, values, len(self.kernel.owners), max_bounces)
        before = {name: data.copy() for name, data in self.stats.items()}
        mass, omega = np.zeros(max_bounces+1), np.zeros(max_bounces+1)
        local = budget = reused_root_mass = 0.
        for (state, order), value in field.items():
            key = (self.root.name, self.signatures[self.root.name], int(self.tokens[state]), max_bounces-order)
            if key in self.retained:
                reused_root_mass += value[0]
            response = self.response(self.root, state, max_bounces-order)
            if response.exits:
                raise AssertionError('Root leaks to an unowned state')
            for cost, moment in response.escape.items():
                mass[order+cost] += value[0]*moment[0]
                omega[order+cost] += value[0]*moment[0]-np.dot(value[1:], moment[1:])
            local += value[0]*response.local_tail
            budget += value[0]*response.budget_tail
        delta = {name: {key: value-before[name][key] for key, value in data.items()} for name, data in self.stats.items()}
        return _result(mass, omega, local, budget, sum(v[0] for v in field.values()),
            query_s=time.perf_counter()-tick, root_columns_applied=len(field), stats=delta,
            retained_root_application_mass=float(reused_root_mass), physical_primitive_tests=0,
            incremental_first_exit=True)
