"""Incremental first-exit probabilities with current terminal direction readout.

Escape columns retain a sparse distribution of (terminal state, collision
order), not a frozen world direction. Current direction moments contract with
that distribution at query time. This is exact for the current discrete
kernel; geometry, material projection and T/G construction stay upstream.
"""
from types import MappingProxyType
import time
import numpy as np
from .ehss_batched_coupling import BatchedScatteringKernel
from .ehss_field_condensation import FirstExit, _field, _result
from .ehss_incremental_condensation import IncrementalCondensedField


class FactoredCondensedField(IncrementalCondensedField):
    def __init__(self, kernel, root, labels, previous=None, scope=None):
        tick = time.perf_counter()
        if previous is not None and not isinstance(previous, FactoredCondensedField):
            raise ValueError('Previous response must use factored terminal readout')
        escape = np.zeros_like(kernel.escape)
        escape[:, 0] = kernel.escape[:, 0]
        signature_kernel = BatchedScatteringKernel(kernel.owners, kernel.costs, kernel.starts,
            kernel.targets, kernel.probabilities, escape, kernel.residual)
        super().__init__(signature_kernel, root, labels, previous, scope)
        self.kernel = kernel
        changed_directions = 0
        if previous is not None:
            for token, state in self.dense.items():
                old = previous.dense.get(token)
                if old is None or not np.array_equal(kernel.escape[state, 1:], previous.kernel.escape[old, 1:]):
                    changed_directions += 1
        self.receipt.update(prepare_s=time.perf_counter()-tick, current_terminal_readout=True,
            frozen_world_escape_moment=False, changed_direction_rows=changed_directions,
            exact_current_kernel=True, sparse_terminal_distribution=True)

    def _response(self, node, token, budget):
        # FirstExit.escape now maps (semantic terminal token, cost) to mass.
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
                            edges[edge] = edges.get(edge, 0.)+float(probability)
                        escaped = {(current, cost): float(k.escape[index, 0])} if k.escape[index, 0] > 0 else {}
                        sub = FirstExit(edges, escaped, float(k.residual[index]), 0.)
                local_tail += weight*sub.local_tail
                budget_tail += weight*sub.budget_tail
                for (terminal, cost), probability in sub.escape.items():
                    event = (terminal, used+cost)
                    escape[event] = escape.get(event, 0.)+weight*probability
                for (target, cost), probability in sub.exits.items():
                    if cost < 1:
                        raise AssertionError('Positive physical collision cost required')
                    order, mass = used+cost, weight*probability
                    if k.owners[self.dense[target]] in node.owners:
                        pending[order][target] = pending[order].get(target, 0.)+mass
                    else:
                        port = (target, order)
                        exits[port] = exits.get(port, 0.)+mass
        total = sum(exits.values())+sum(escape.values())+local_tail+budget_tail
        if not np.isclose(total, 1., rtol=1e-11, atol=1e-12):
            raise AssertionError('Factored first-exit probability changed')
        result = FirstExit(MappingProxyType(exits), MappingProxyType(escape), local_tail, budget_tail)
        self.cache[key] = result
        stats['compiled_columns'] += 1
        return result

    def solve(self, states, orders, values, max_bounces=64):
        tick = time.perf_counter()
        field = _field(states, orders, values, len(self.kernel.owners), max_bounces)
        before = {name: data.copy() for name, data in self.stats.items()}
        mass, omega = np.zeros(max_bounces+1), np.zeros(max_bounces+1)
        local = budget = reused_root_mass = 0.
        contractions = 0
        for (state, order), value in field.items():
            key = (self.root.name, self.signatures[self.root.name], int(self.tokens[state]), max_bounces-order)
            if key in self.retained:
                reused_root_mass += value[0]
            response = self.response(self.root, state, max_bounces-order)
            if response.exits:
                raise AssertionError('Root leaks to an unowned state')
            for (terminal, cost), probability in response.escape.items():
                escaped = self.kernel.escape[self.dense[terminal]]
                if escaped[0] <= 0:
                    raise AssertionError('Retained terminal is no longer an escape')
                mass[order+cost] += value[0]*probability
                omega[order+cost] += probability*(value[0]-np.dot(value[1:], escaped[1:]/escaped[0]))
                contractions += 1
            local += value[0]*response.local_tail
            budget += value[0]*response.budget_tail
        delta = {name: {key: value-before[name][key] for key, value in data.items()} for name, data in self.stats.items()}
        return _result(mass, omega, local, budget, sum(v[0] for v in field.values()), query_s=time.perf_counter()-tick,
            root_columns_applied=len(field), stats=delta, retained_root_application_mass=float(reused_root_mass),
            terminal_moment_contractions=contractions, physical_primitive_tests=0, factored_first_exit=True)
