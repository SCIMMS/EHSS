"""Collision-order-preserving molecular field from numerical internal T/current G.

Each material port consumes its known reflection once. Additional owned
reflections have separate cost-one states, so branches with different lengths
retain their order. A reflected foreign arrival is charged by the target port
or physical suffix, never by both. Port projection remains empirical.
"""
from types import SimpleNamespace
import time
import numpy as np
from .ehss_batched_coupling import MaterialPortLayout, lookup_targets, BatchedScatteringKernel
from .ehss_atomic_material_axes import material_keys, trace_to_material_port, trace_port_prefix
from .ehss_batched_material_input import physical_ledger
from .ehss_empirical_bvh import first_collisions_bvh
from .ehss_empirical_field import geometry_key
from .ehss_field_condensation import flat_field_solve
from .ehss_sparse_path_response import readonly
from .ehss_state import EHSSSource


class IntrinsicMaterialField:
    def __init__(self, bound, bins=2, layout=None, allow_reentry=True):
        tick = time.perf_counter()
        original = bound.bank
        self.model = bound.search.model
        spheres = self.model.spheres
        self.geometry = geometry_key(spheres)
        self.cap = original.cap
        self.bins = bins
        self.rotations = readonly(np.asarray(bound.rotations)[original.owners].copy())
        if layout is None:
            bank = SimpleNamespace(intrinsic_bank=original, groups=original.owners,
                lasts=original.lasts, normals=original.normals, directions=original.directions,
                mass=original.weights, cap=original.cap)
            layout = MaterialPortLayout(bank, bins)
        elif getattr(layout.bank, 'intrinsic_bank', None) is not original or layout.bins != bins:
            raise ValueError('Layout belongs to another intrinsic bank or bins')
        self.bank = bank = layout.bank
        self.layout = layout
        self.keys, self.index = layout.keys, layout.index
        observed = bound.apply()
        self.observed = observed
        counts, history, terminal = (observed[k] for k in ('counts', 'history', 'terminal'))
        ids, size = layout.ids, len(layout.keys)
        valid = terminal != -3
        mass = np.bincount(ids[valid], weights=bank.mass[valid], minlength=size)
        self.active = readonly(mass > 0)
        self.visits = readonly(np.bincount(ids[valid], minlength=size))
        self.occupation_mass = readonly(mass)
        moving = np.flatnonzero(terminal >= 0)
        target = np.full(len(ids), -1, dtype=np.int64)
        p, d = observed['points'], observed['directions']
        target[moving] = lookup_targets(material_keys(p[moving], d[moving], terminal[moving],
            spheres.centers, bank.groups, self.rotations, bins), layout.table, self.active)
        missing = moving[target[moving] < 0]
        suffix = None
        start = time.perf_counter()
        if len(missing):
            source = EHSSSource(p[missing], d[missing], d[missing], np.ones(len(missing)),
                terminal[missing], np.ones(len(missing), dtype=np.int64), 'foreign missing material ports')
            suffix = trace_to_material_port(self.model, source, self.keys, self.active,
                bank.groups, self.rotations, bins, self.cap, allow_reentry)
        suffix_s = time.perf_counter() - start
        rows = [{} for _ in range(size)]
        owners = list(map(int, layout.table[:, 0]))
        escape = [np.zeros(4) for _ in range(size)]
        residual = [float(not active) for active in self.active]
        labels = list(layout.labels)

        def append_chain(atom_ids, label, following, outgoing, stopped):
            """Return a chain entry; every listed atom is one physical collision."""
            first = len(rows)
            for j, atom in enumerate(atom_ids):
                state = len(rows)
                rows.append({state + 1: 1.} if j + 1 < len(atom_ids) else
                            ({int(following): 1.} if following >= 0 else {}))
                owners.append(int(bank.groups[atom]))
                escape.append(np.r_[1., outgoing] if j + 1 == len(atom_ids) and
                              following < 0 and not stopped else np.zeros(4))
                residual.append(float(j + 1 == len(atom_ids) and stopped))
                labels.append(label + (j,))
            return first

        suffix_states = 0
        if suffix is not None:
            lengths = suffix['bounces'] - (suffix['ports'] >= 0)
            if np.any(lengths < 1):
                raise AssertionError('A missing foreign port cannot immediately match itself')
            for i, row in enumerate(missing):
                atoms = suffix['collider_ids'][i, :lengths[i]]
                target[row] = append_chain(atoms, ('foreign_suffix', int(row)),
                    suffix['ports'][i], suffix['outgoing'][i], suffix['unresolved'][i])
                suffix_states += len(atoms)
        internal_states = 0
        branch_lengths = []
        for row in np.flatnonzero(valid):
            state = int(ids[row])
            weight = bank.mass[row] / mass[state] if mass[state] else 0.
            if weight == 0:
                continue
            # The foreign collision is already reflected in the observation,
            # but its logical cost remains pending in target[row].
            own_count = int(counts[row]) - int(terminal[row] >= 0)
            owned = history[row, :own_count]
            if own_count < 1 or np.any(bank.groups[owned] != bank.groups[bank.lasts[row]]):
                raise AssertionError('Internal T must stop at its first foreign arrival')
            following = int(target[row])
            if own_count > 1:
                following = append_chain(owned[1:], ('internal', int(row)), following,
                    d[row], terminal[row] == -2)
                internal_states += own_count - 1
            if following >= 0:
                rows[state][following] = rows[state].get(following, 0.) + weight
            elif terminal[row] == -1:
                escape[state] += weight * np.r_[1., d[row]]
            elif terminal[row] == -2:
                residual[state] += weight
            else:
                raise AssertionError('Foreign handoff has no destination')
            branch_lengths.append(own_count)
        starts, targets, probabilities = [0], [], []
        for row in rows:
            for destination, probability in sorted(row.items()):
                targets.append(destination)
                probabilities.append(probability)
            starts.append(len(targets))
        self.kernel = BatchedScatteringKernel(owners, np.ones(len(rows), dtype=np.int64),
            starts, targets, probabilities, np.asarray(escape).reshape(-1, 4), residual)
        self.labels = tuple(labels)
        if len(set(labels)) != len(labels):
            raise AssertionError('Unique semantic labels required')
        self.receipt = dict(prepare_s=time.perf_counter()-tick, observed_apply_s=observed['query_s'],
            material_ports=size, active_ports=int(self.active.sum()), internal_collision_states=internal_states,
            physical_suffix_states=suffix_states, missing_foreign_rows=len(missing), suffix_s=suffix_s,
            known_foreign_rows=len(moving)-len(missing),
            suffix_metrics={} if suffix is None else suffix['metrics'],
            reentered_suffixes=0 if suffix is None else int(np.count_nonzero(suffix['ports'] >= 0)),
            max_owned_collisions=max(branch_lengths, default=0),
            full_first_exit_used=True, arrival_collision_charged_once=True,
            empirical_port_projection=True, incoming_direction_moment_preserved=True,
            coupling_rebuilt=True, partial_hierarchy_update=False,
            reused_local_events=observed['reused_local_events'],
            occluded_material_mass=float(bank.mass[~valid].sum()))

    def solve(self, source, cap=None, engine=None, min_visits=1, allow_reentry=True):
        """Project at the actual first/reentry reflection, with its cost pending.

        Unlike the older one-collision shortcut, no first row is pre-applied
        with the query's outgoing direction: all branches use kernel moments.
        This makes initial and later applications the same linear response.
        """
        tick = time.perf_counter()
        if cap is None:
            cap = self.cap
        if type(cap) is not int or not 1 <= cap <= self.cap:
            raise ValueError('Cap exceeds compiled physical suffix budget')
        if type(min_visits) is not int or min_visits < 1:
            raise ValueError('Positive visit threshold required')
        if geometry_key(self.model.spheres) != self.geometry:
            raise ValueError('Geometry changed; rebuild coupling')
        if engine is not None and engine.kernel is not self.kernel:
            raise ValueError('Different field kernel')
        if np.any((source.initial_bounces != 0) & (source.initial_bounces != 1)):
            raise ValueError('Uncollided or first-reflected sources required')
        spheres, tree = self.model.spheres, self.model.tree
        if np.any(source.last_atom < -1) or np.any(source.last_atom >= len(spheres.radii)):
            raise ValueError('Invalid source atom')
        if np.any((source.initial_bounces == 1) & (source.last_atom < 0)):
            raise ValueError('First-reflected sources require a known atom')
        start = time.perf_counter()
        p, d, atoms, counters = first_collisions_bvh(source.origins, source.directions, source.last_atom,
            source.initial_bounces, spheres.centers, spheres.radii, tree.lo, tree.hi, tree.left, tree.right,
            tree.end, tree.leaf_group, tree.offsets, tree.ids)
        first_s = time.perf_counter() - start
        hit = atoms >= 0
        hits = np.flatnonzero(hit)
        active = readonly(self.active & (self.visits >= min_visits))
        targets = lookup_targets(material_keys(p[hit], d[hit], atoms[hit], spheres.centers,
            self.bank.groups, self.rotations, self.bins), self.layout.table, active)
        selected, missing = hits[targets >= 0], hits[targets < 0]
        seeds = targets[targets >= 0]
        orders = np.zeros(len(seeds), dtype=np.int64)
        values = np.column_stack((source.weights[selected], source.weights[selected, None]*source.incoming[selected]))
        ledger = np.zeros((cap+1, 2))
        ledger[0, 0] = source.weights[~hit].sum()
        ledger[0, 1] = np.sum(source.weights[~hit]*(1-np.sum(source.incoming[~hit]*d[~hit], axis=1)))
        tail = reentry_mass = 0.
        reentered = 0
        prefix_tests = 0
        reentry_orders = np.zeros(cap+1, dtype=np.int64)
        start = time.perf_counter()
        if len(missing):
            result = trace_port_prefix(p[missing], d[missing], atoms[missing], cap, spheres.centers, spheres.radii,
                tree.lo, tree.hi, tree.left, tree.right, tree.end, tree.leaf_group, tree.offsets, tree.ids,
                self.bank.groups, self.rotations, self.bins, self.layout.table, active, allow_reentry)
            _, out, escaped, unresolved, counts, _, ports, checks, _, _ = result
            ledger += physical_ledger(source.weights[missing], source.incoming[missing], out, escaped, counts, cap)
            tail = float(source.weights[missing[unresolved]].sum())
            arrived = ports >= 0
            rays = missing[arrived]
            seeds = np.r_[seeds, ports[arrived]]
            orders = np.r_[orders, counts[arrived]-1]
            values = np.concatenate((values, np.column_stack((source.weights[rays], source.weights[rays, None]*source.incoming[rays]))))
            reentered, reentry_mass = len(rays), float(source.weights[rays].sum())
            reentry_orders = np.bincount(counts[arrived], minlength=cap+1)
            prefix_tests = int(checks[1])
        fallback_s = time.perf_counter() - start
        result = (flat_field_solve(self.kernel, seeds, orders, values, cap) if engine is None
                  else engine.solve(seeds, orders, values, cap))
        ledger += np.array([[row['mass'], row['omega']] for row in result['order_ledger']])
        tail += result['residual_mass']
        if not np.isclose(ledger[:, 0].sum()+tail, source.weights.sum(), rtol=1e-11, atol=1e-11):
            raise AssertionError('Full intrinsic field mass changed')
        return dict(omega=float(ledger[:, 1].sum()), pa=float(source.weights[hit].sum()),
            escaped_mass=float(ledger[:, 0].sum()), residual_mass=tail, tail_bound=2*tail,
            source_mass=float(source.weights.sum()), learned_mass=float(source.weights[selected].sum()),
            fallback_mass=float(source.weights[missing].sum()), learned_rays=len(selected), fallback_rays=len(missing),
            reentered_rays=reentered, reentered_mass=reentry_mass, reentry_orders=reentry_orders,
            transported_mass=float(source.weights[selected].sum())+reentry_mass,
            remaining_physical_mass=float(source.weights[missing].sum())-reentry_mass,
            order_ledger=[dict(order=i, mass=float(v[0]), omega=float(v[1])) for i, v in enumerate(ledger)],
            first_s=first_s, fallback_s=fallback_s, field_s=result['query_s'],
            query_s=time.perf_counter()-tick, sphere_tests=int(counters[1])+prefix_tests, field=result)
