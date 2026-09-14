"""Incremental ordered spherical-support events, without new-event oracle input.

Uses the Phase 3 scalar membrane/feedback model. Spheres are interaction-region
proxies, not a triangle visibility solver. All supplied rays are unit length.
Only sequences and old input weights are cached; candidate contributions are
subtracted/reassembled. Global nonlinear solution is deliberately separate.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from numba import njit
from .occlusion_hierarchy import box_entry
from .occlusion_pockets import SupportTree, candidates as tree_candidates


@dataclass
class System:
    drive: np.ndarray
    operator: np.ndarray
    emission_operator: np.ndarray
    outgoing_const: float
    outgoing_operator: np.ndarray
    event_count: int
    incident: float


@dataclass
class Inputs:
    centers: np.ndarray
    radii: np.ndarray
    origins: np.ndarray
    directions: np.ndarray
    weights: np.ndarray

    def checked(self):
        arrays = [np.array(a, dtype=float, order='C', copy=True) for a in
                  (self.centers, self.radii, self.origins, self.directions, self.weights)]
        c, r, o, d, w = arrays
        if c.shape != (len(r), 3) or o.shape != d.shape or o.shape != (len(w), 3):
            raise ValueError('Invalid input shapes')
        if not all(np.isfinite(a).all() for a in arrays) or np.any(r <= 0) or np.any(w < 0):
            raise ValueError('Nonfinite geometry or invalid radius/weight')
        if not np.allclose(np.linalg.norm(d, axis=1), 1., atol=1e-12, rtol=0):
            raise ValueError('Unit ray directions required')
        return Inputs(*arrays)


@njit(cache=True)
def trace_sequences(origins, directions, centers, radii, ids):
    seq = np.zeros((len(ids), len(radii)), np.int64)
    for k in range(len(ids)):
        i = ids[k]
        entries = np.empty(len(radii))
        count = 0
        for j in range(len(radii)):
            offset = origins[i] - centers[j]
            b = np.dot(offset, directions[i])
            disc = b*b - (np.dot(offset, offset)-radii[j]*radii[j])
            if disc < 0:
                continue
            t = np.sqrt(disc)
            if -b+t <= 1e-9:
                continue
            enter = -b-t
            if enter < 1e-9:
                enter = 0.
            q = count
            # Stable label tie order matches the original extractor.
            while q > 0 and entries[q-1] > enter:
                entries[q] = entries[q-1]
                seq[k, q] = seq[k, q-1]
                q -= 1
            entries[q] = enter
            seq[k, q] = j+1
            count += 1
    return seq


@njit(cache=True)
def assemble(seq, weights, tau, gains):
    n = len(tau)
    drive, matrix = np.zeros(n), np.zeros((n, n))
    emission, outgoing = np.zeros((n, n)), np.zeros(n)
    escaped = 0.
    events = 0
    for i in range(len(seq)):
        constant = weights[i]
        coefficients = np.zeros(n)
        for label in seq[i]:
            if label == 0:
                break
            absorb = 1.-tau[label]
            drive[label] += absorb*constant
            matrix[label] += absorb*coefficients
            gain = gains[label]*weights[i]
            emission[label, label] += gain
            constant *= tau[label]
            coefficients *= tau[label]
            coefficients[label] += gain
            events += 1
        escaped += constant
        outgoing += coefficients
    return drive, matrix, emission, escaped, outgoing, events, np.sum(weights)


def coefficients(seq, weights, tau, gains):
    return System(*assemble(seq, weights, tau, gains))


def system_delta(base, old, new):
    return System(*(getattr(base, name)-getattr(old, name)+getattr(new, name)
                    for name in System.__dataclass_fields__))


@njit(cache=True)
def flat_mask(origins, directions, bounds):
    mask = np.zeros(len(origins), np.bool_)
    for i in range(len(origins)):
        for box in bounds:
            if np.isfinite(box_entry(origins[i], directions[i], box[0], box[1], np.inf)):
                mask[i] = True
                break
    return mask


def discover(old, new, hierarchy=False):
    # All rays whose input changes are candidates, including weights only.
    mask = ((old.origins != new.origins).any(axis=1)
            | (old.directions != new.directions).any(axis=1)
            | (old.weights != new.weights))
    changed = np.flatnonzero((old.centers != new.centers).any(axis=1) | (old.radii != new.radii))
    if len(changed):
        lower = np.minimum(old.centers[changed]-old.radii[changed, None],
                           new.centers[changed]-new.radii[changed, None])
        upper = np.maximum(old.centers[changed]+old.radii[changed, None],
                           new.centers[changed]+new.radii[changed, None])
        bounds = np.stack((lower-1e-12, upper+1e-12), axis=1)
        # Unchanged-input rays have identical old/new rays. Changed inputs are
        # already in mask, so only the current ray needs a geometric query.
        if hierarchy:
            tree = SupportTree(bounds)
            _, rays, _ = tree_candidates(new.origins, new.directions, np.full(len(mask), np.inf),
                                         tree.lo, tree.hi, tree.end, tree.owner, len(bounds), False)
            mask[rays] = True
        else:
            mask |= flat_mask(new.origins, new.directions, bounds)
    return np.flatnonzero(mask), len(changed)


class EventRuntime:
    def __init__(self, inputs, tau, gains, envelope=None):
        start = perf_counter()
        self.inputs = inputs.checked()
        self.tau, self.gains = np.array(tau, copy=True), np.array(gains, copy=True)
        n = len(self.inputs.radii)+1
        if self.tau.shape != (n,) or self.gains.shape != (n,) or not np.isfinite(self.tau).all() or not np.isfinite(self.gains).all():
            raise ValueError('Invalid coefficient shape or value')
        if np.any((self.tau < 0) | (self.tau > 1)) or np.any(self.gains < 0):
            raise ValueError('Invalid membrane coefficients')
        self.envelope = None if envelope is None else np.array(envelope, dtype=float, copy=True)
        self._check_envelope(self.inputs)
        self.sequences = trace_sequences(self.inputs.origins, self.inputs.directions,
                                         self.inputs.centers, self.inputs.radii,
                                         np.arange(len(self.inputs.weights)))
        self.system = coefficients(self.sequences, self.inputs.weights, self.tau, self.gains)
        self.preparation_s = perf_counter()-start

    def _check_envelope(self, inputs):
        if self.envelope is not None:
            if self.envelope.shape != (2, 3) or not np.isfinite(self.envelope).all():
                raise ValueError('Invalid enclosing support')
            if np.any(inputs.centers-inputs.radii[:, None] < self.envelope[0]) or np.any(inputs.centers+inputs.radii[:, None] > self.envelope[1]):
                raise ValueError('Interaction sphere left declared E')

    @property
    def payload_bytes(self):
        return int(sum(a.nbytes for a in [self.inputs.centers, self.inputs.radii, self.inputs.origins,
                   self.inputs.directions, self.inputs.weights, self.tau, self.gains, self.sequences])
                   + sum(a.nbytes for a in self.system.__dict__.values() if isinstance(a, np.ndarray)))

    def update(self, new_inputs, route='local'):
        if route not in ('full', 'local', 'hierarchy'):
            raise ValueError(route)
        start = perf_counter()
        new = new_inputs.checked()
        if new.weights.shape != self.inputs.weights.shape or new.radii.shape != self.inputs.radii.shape:
            raise ValueError('Topology/ray count changed')
        self._check_envelope(new)
        after_input = perf_counter()
        if route == 'full':
            ids, changed = np.arange(len(new.weights)), len(new.radii)
        else:
            ids, changed = discover(self.inputs, new, hierarchy=route == 'hierarchy')
        after_candidate = perf_counter()
        geometry_reused = route != 'full' and all(np.array_equal(getattr(new, key), getattr(self.inputs, key))
            for key in ('centers', 'radii', 'origins', 'directions'))
        if geometry_reused:
            seq = self.sequences[ids].copy()
        else:
            seq = trace_sequences(new.origins, new.directions, new.centers, new.radii, ids)
        after_trace = perf_counter()
        new_contribution = coefficients(seq, new.weights[ids], self.tau, self.gains)
        if route == 'full':
            total = new_contribution
        else:
            old_contribution = coefficients(self.sequences[ids], self.inputs.weights[ids], self.tau, self.gains)
            total = system_delta(self.system, old_contribution, new_contribution)
        after_operator = perf_counter()
        self.sequences[ids] = seq
        self.inputs, self.system = new, total
        end = perf_counter()
        return dict(input_s=after_input-start, candidate_s=after_candidate-after_input,
                    event_s=after_trace-after_candidate, operator_s=after_operator-after_trace,
                    commit_s=end-after_operator, update_s=end-start,
                    candidate_rays=len(ids), total_rays=len(new.weights), traced_rays=0 if geometry_reused else len(ids),
                    sphere_tests=0 if geometry_reused else len(ids)*len(new.radii), changed_supports=changed,
                    geometry_reused=geometry_reused,
                    candidate_ids=ids)
