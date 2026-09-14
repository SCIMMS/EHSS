"""Occupation-weighted molecular scattering fields from physical trajectories.

A state labels one physical collision, its outgoing surface/direction cell,
and its absolute collision order. Applying its row charges that collision once.
This order-conditioned empirical closure is source dependent: it is not an
intrinsic, universally reusable material response. No envelope exclusion is
needed because calibration paths use the complete physical geometry.
"""
import hashlib
import time
import numpy as np
from numba import njit
from .ehss_field_condensation import ScatteringKernel, FieldNode, CondensedField, flat_field_solve
from .ehss_surface_response import surface_key
from .ehss_sparse_path_response import record_from_paths, readonly
from .ehss_reference import trace, nearest_all
from .ehss_guarded_intrinsic_response import reflect
from .ehss_state import EHSSSource


def geometry_key(spheres):
    h = hashlib.sha256()
    for a in (spheres.centers, spheres.radii):
        h.update(np.asarray(a.shape, dtype=np.int64).tobytes()); h.update(a.tobytes())
    return h.hexdigest()


@njit(cache=True)
def collision_keys(points, directions, atoms, orders, centers, groups, bins):
    keys = np.empty((len(atoms), 8), dtype=np.int64)
    for i in range(len(atoms)):
        atom = atoms[i]; n = points[i] - centers[atom]; n /= np.sqrt(np.dot(n, n))
        key = surface_key(groups[atom], atom, n, directions[i], bins)
        for j in range(7): keys[i, j] = key[j]
        keys[i, 7] = orders[i]
    return keys


@njit(cache=True)
def first_collisions(origins, directions, last_atoms, initial, centers, radii):
    points = origins.copy(); outgoing = directions.copy(); atoms = last_atoms.copy(); tests = 0
    for i in range(len(points)):
        if initial[i] == 1: continue
        atom, distance, count = nearest_all(points[i], outgoing[i], -1, centers, radii); tests += count
        atoms[i] = atom
        if atom >= 0: reflect(points[i], outgoing[i], atom, distance, centers, radii)
    return points, outgoing, atoms, tests


def owner_hierarchy(tree, owners):
    """Restrict the existing node-aware tree to observed chemical owners."""
    owners = frozenset(map(int, owners))
    def visit(node):
        group = int(tree.leaf_group[node])
        if group >= 0:
            return FieldNode(str(node), [group]) if group in owners else None
        children = tuple(c for c in (visit(int(tree.left[node])), visit(int(tree.right[node]))) if c is not None)
        if not children: return None
        return FieldNode(str(node), frozenset().union(*(c.owners for c in children)), children)
    root = visit(0)
    return FieldNode('empty', []) if root is None else root


class EmpiricalCollisionField:
    def __init__(self, spheres, groups, results, bins=2):
        if not isinstance(bins, int) or bins < 1: raise ValueError('Positive integer bins required')
        groups = np.array(groups, dtype=np.int64, copy=True)
        if groups.shape != spheres.radii.shape or np.any(groups < 0): raise ValueError('One nonnegative owner per atom required')
        if not results: raise ValueError('Calibration results required')
        tick = time.perf_counter(); self.bins = bins; self.geometry = geometry_key(spheres); self.groups = readonly(groups)
        records = []; all_keys = []
        for result in results:
            source = result.source
            if np.any((source.initial_bounces != 0) & (source.initial_bounces != 1)):
                raise ValueError('Uncollided or first-reflected sources required')
            if np.any(result.escaped == result.unresolved): raise ValueError('Every calibration ray must escape or remain unresolved')
            p, d, atoms, weights = record_from_paths(source.origins, source.directions, source.weights / len(results),
                source.initial_bounces, result.collider_ids, result.bounces, spheres.centers, spheres.radii)
            orders = np.concatenate([np.arange(1, int(n)+1) for n in result.bounces]) if len(result.bounces) else np.empty(0, dtype=np.int64)
            keys = collision_keys(p, d, atoms, orders, spheres.centers, groups, bins)
            records.append((result, d, weights, keys)); all_keys.extend(map(tuple, keys))
        self.keys = tuple(sorted(set(all_keys))); self.index = {key:i for i,key in enumerate(self.keys)}
        n = len(self.keys); rows = [{} for _ in range(n)]; mass = np.zeros(n); visits = np.zeros(n, dtype=np.int64)
        escape = np.zeros((n,4)); residual = np.zeros(n)
        for result, directions, weights, keys in records:
            cursor = 0
            for ray, count in enumerate(result.bounces):
                for j in range(int(count)):
                    state = self.index[tuple(keys[cursor+j])]; w = weights[cursor+j]
                    mass[state] += w; visits[state] += 1
                    if j+1 < count:
                        target = self.index[tuple(keys[cursor+j+1])]
                        rows[state][target] = rows[state].get(target, 0.) + w
                    elif result.escaped[ray]: escape[state] += w * np.r_[1., directions[cursor+j]]
                    else: residual[state] += w
                cursor += int(count)
        # Zero-weight observations do not define a probability law.
        if np.any(mass <= 0): raise ValueError('Each observed cell needs positive occupation mass')
        starts = [0]; targets = []; probability = []
        for state, row in enumerate(rows):
            for target, w in sorted(row.items()): targets.append(target); probability.append(w/mass[state])
            starts.append(len(targets))
        self.kernel = ScatteringKernel([k[0] for k in self.keys], np.ones(n, dtype=np.int64), starts, targets, probability,
            escape / mass[:,None], residual / mass)
        self.visits = readonly(visits); self.occupation_mass = readonly(mass)
        self.receipt = dict(prepare_s=time.perf_counter()-tick, states=n, edges=len(targets), physical_visits=int(visits.sum()),
            branching_rows=sum(len(row)+(escape[i,0]>0)+(residual[i]>0)>1 for i,row in enumerate(rows)),
            bins=bins, order_conditioned=True, source_conditioned=True,
            calibration_query_s=sum(r.metrics['query_s'] for r in results),
            calibration_sphere_tests=sum(r.metrics['sphere_tests'] for r in results))

    def solve(self, spheres, source, cap=128, engine=None, min_visits=1, anchor_first_direction=True):
        """Exact first collision, empirical field on known cells, exact unknown suffixes.

        The incoming source is never used to modify learned probabilities.
        The first reflection is geometrically computed for classification but
        its collision count/readout belongs to the learned row. By default the
        first row's escape branch retains the query's known physical direction;
        its successor probabilities remain empirical. Disabling this anchor is
        an explicit mean-direction closure control, not the production default.
        """
        if geometry_key(spheres) != self.geometry: raise ValueError('Geometry changed; empirical response must be rebuilt')
        if not isinstance(cap, int) or cap < 1 or not isinstance(min_visits, int) or min_visits < 1:
            raise ValueError('Positive cap and visit threshold required')
        if np.any((source.initial_bounces != 0) & (source.initial_bounces != 1)):
            raise ValueError('Uncollided or first-reflected sources required')
        if np.any(source.last_atom >= len(spheres.radii)) or np.any(source.last_atom < -1):
            raise ValueError('Invalid source atom identity')
        if engine is not None and engine.kernel is not self.kernel: raise ValueError('Engine uses a different empirical kernel')
        tick = time.perf_counter()
        p,d,atoms,tests = first_collisions(source.origins,source.directions,source.last_atom,source.initial_bounces,spheres.centers,spheres.radii)
        hit = atoms >= 0; ids = np.flatnonzero(hit)
        keys = collision_keys(p[hit],d[hit],atoms[hit],np.ones(len(ids),dtype=np.int64),spheres.centers,self.groups,self.bins)
        selected = []; states = []; fallback = []
        for ray,key in zip(ids,keys):
            state = self.index.get(tuple(key))
            if state is None or self.visits[state] < min_visits: fallback.append(ray)
            else: selected.append(ray); states.append(state)
        selected = np.array(selected,dtype=np.int64); fallback = np.array(fallback,dtype=np.int64)
        values = np.column_stack((source.weights[selected],source.weights[selected,None]*source.incoming[selected]))
        initial_escape = np.zeros(2); initial_tail = 0.
        orders = np.zeros(len(states),dtype=np.int64)
        if anchor_first_direction:
            next_states = []; next_values = []; k = self.kernel
            for ray,state,value in zip(selected,states,values):
                e = k.escape[state,0]
                initial_escape += e*np.array([value[0],value[0]-np.dot(value[1:],d[ray])])
                initial_tail += value[0]*k.residual[state]
                for edge in range(k.starts[state],k.starts[state+1]):
                    next_states.append(k.targets[edge]); next_values.append(k.probabilities[edge]*value)
            states = next_states; values = np.array(next_values).reshape(-1,4)
            orders = np.ones(len(states),dtype=np.int64)
        seed_s = time.perf_counter()-tick
        fn = flat_field_solve if engine is None else None
        field = fn(self.kernel,states,orders,values,cap) if fn else engine.solve(states,orders,values,cap)
        ledger = np.array([[r['mass'],r['omega']] for r in field['order_ledger']])
        ledger[1] += initial_escape
        ledger[0,0] += source.weights[~hit].sum()
        # External no-hit rays retain their original physical direction.
        ledger[0,1] += np.sum(source.weights[~hit]*(1-np.sum(source.incoming[~hit]*d[~hit],axis=1)))
        tail = field['residual_mass'] + initial_tail; fallback_s = 0.; fallback_tests = 0
        if len(fallback):
            sub = EHSSSource(p[fallback],d[fallback],source.incoming[fallback],source.weights[fallback],atoms[fallback],
                np.ones(len(fallback),dtype=np.int64),'exact unknown-cell suffix')
            exact = trace(spheres,sub,cap); fallback_s = exact.metrics['query_s']; fallback_tests = exact.metrics['sphere_tests']
            for row in exact.order_ledger(): ledger[row['order']] += (row['mass'],row['omega'])
            tail += exact.residual_mass
        if not np.isclose(ledger[:,0].sum()+tail,source.weights.sum(),rtol=1e-11,atol=1e-11): raise AssertionError('Hybrid mass changed')
        return dict(omega=float(ledger[:,1].sum()),pa=float(source.weights[hit].sum()),residual_mass=float(tail),tail_bound=float(2*tail),
            escaped_mass=float(ledger[:,0].sum()),source_mass=float(source.weights.sum()),
            order_ledger=[dict(order=i,mass=float(v[0]),omega=float(v[1])) for i,v in enumerate(ledger)],
            learned_mass=float(source.weights[selected].sum()),fallback_mass=float(source.weights[fallback].sum()),
            learned_rays=len(selected),fallback_rays=len(fallback),seed_s=seed_s,field_s=field['query_s'],fallback_s=fallback_s,
            sphere_tests=int(tests+fallback_tests),query_s=time.perf_counter()-tick,field=field)
