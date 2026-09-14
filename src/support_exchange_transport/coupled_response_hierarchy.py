"""Two-level, finite planar-membrane response composition.

The cached map is conditional on the global nonlinear response f(U); it does
not eliminate internal nonlinear unknowns. Parent geometry discovers ordered
branches, then composes child numerical responses. No MD reuse claim.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from .nonlinear_event_runtime import System


@dataclass
class Response:
    t: float
    c: float
    v: np.ndarray
    h: np.ndarray
    b: np.ndarray
    a: np.ndarray
    source: float
    emission: np.ndarray


def identity(n):
    return Response(1., 0., np.zeros(n), np.zeros(n), np.zeros(n), np.zeros((n, n)), 0., np.zeros(n))


def compose(first, second):
    return Response(second.t*first.t, second.t*first.c+second.c,
        second.t*first.v+second.v, first.h+second.h*first.t,
        first.b+second.h*first.c+second.b,
        first.a+np.outer(second.h, first.v)+second.a,
        first.source+second.source, first.emission+second.emission)


def element(label, n, tau, gain, source):
    r = identity(n)
    r.t, r.c, r.source = float(tau), float(source), float(source)
    r.v[label] = gain
    r.h[label] = 1.-tau
    r.emission[label] = gain
    return r


@dataclass
class Scene:
    # row: x center, z, halfwidth; module origin: x,z. Labels 1..6.
    local: np.ndarray
    pose: np.ndarray
    owner: np.ndarray
    tau: np.ndarray
    gain: np.ndarray
    source: np.ndarray

    def validate(self):
        if self.local.shape != (6, 3) or self.pose.shape != (3, 2) or self.owner.shape != (6,):
            raise ValueError('Fixed six-panel, two-child topology required')
        if not all(np.isfinite(x).all() for x in (self.local, self.pose, self.tau, self.gain, self.source)):
            raise ValueError('Finite state required')
        if np.any((self.tau < 0) | (self.tau > 1)) or np.any(self.gain < 0) or np.any(self.source < 0):
            raise ValueError('Invalid response law')
        world = self.world()
        if np.any(self.local[:, 2] <= 0) or np.any(np.abs(world[:, 0])+world[:, 2] > 5) or np.any(np.abs(world[:, 1]) > 5):
            raise ValueError('Geometry left declared E')

    def world(self):
        w = self.local.copy()
        w[:, :2] += self.pose[self.owner]
        return w

    def signature(self, owner):
        ids = np.flatnonzero(self.owner == owner)
        # Rigid parent placement is deliberately excluded; internal shape and
        # all response laws invalidate the entire child's compiled branches.
        return tuple(x.tobytes() for x in (self.local[ids], self.tau[ids+1], self.gain[ids+1], self.source[ids+1]))


def ordered(scene, x, sign):
    world = scene.world()
    ids = np.flatnonzero(np.abs(x-world[:, 0]) <= world[:, 2])
    return tuple((ids[np.lexsort((ids, sign*world[ids, 1]))]+1).tolist())


class Hierarchy:
    def __init__(self):
        self.signatures = {}
        self.maps = {}
        self.compilations = 0
        self.hits = 0
        self.invalidated = []

    def sync(self, scene, stale=False):
        scene.validate()
        self.invalidated = []
        for owner in (1, 2):
            sig = scene.signature(owner)
            if owner in self.signatures and self.signatures[owner] != sig and not stale:
                self.maps = {key: val for key, val in self.maps.items() if key[0] != owner}
                self.invalidated.append(owner)
            if not stale or owner not in self.signatures:
                self.signatures[owner] = sig

    def branch(self, scene, owner, labels):
        key = (owner, labels)
        if owner and key in self.maps:
            self.hits += 1
            return self.maps[key]
        result = identity(len(scene.tau))
        for label in labels:
            result = compose(result, element(label, len(scene.tau), scene.tau[label], scene.gain[label], scene.source[label]))
        self.compilations += 1
        if owner:
            self.maps[key] = result
        return result

    def response(self, scene, sequence):
        result = identity(len(scene.tau))
        cursor = 0
        # Split at foreign events: a cached complete child traversal must not
        # jump over an intervening shutter/other child.
        while cursor < len(sequence):
            owner = scene.owner[sequence[cursor]-1]
            end = cursor+1
            while end < len(sequence) and scene.owner[sequence[end]-1] == owner:
                end += 1
            result = compose(result, self.branch(scene, int(owner), sequence[cursor:end]))
            cursor = end
        return result

    @property
    def payload_bytes(self):
        return sum(sum(v.nbytes if isinstance(v, np.ndarray) else 8 for v in r.__dict__.values()) for r in self.maps.values())


def direct_ray(scene, sequence, light):
    """Independent scalar symbolic propagation; never calls response composition."""
    n = len(scene.tau)
    b, a, emitted = np.zeros(n), np.zeros((n, n)), np.zeros(n)
    value, vector, source = float(light), np.zeros(n), 0.
    for label in sequence:
        absorption = 1.-scene.tau[label]
        b[label] += absorption*value
        a[label] += absorption*vector
        value = scene.tau[label]*value+scene.source[label]
        vector *= scene.tau[label]
        vector[label] += scene.gain[label]
        source += scene.source[label]
        emitted[label] += scene.gain[label]
    return b, a, value, vector, source, emitted


def assembled(scene, x, signs, light, weights, hierarchy=None, stale=False, first_hit_only=False):
    start = perf_counter()
    scene.validate()
    if not (len(x) == len(signs) == len(light) == len(weights)) or not np.isfinite(light).all() or np.any(light < 0):
        raise ValueError('Invalid source state')
    if hierarchy is not None:
        hierarchy.sync(scene, stale=stale)
    setup = perf_counter()
    sequences = [ordered(scene, xx, ss) for xx, ss in zip(x, signs)]
    traced = perf_counter()
    n = len(scene.tau)
    b, a, emitted, outv = np.zeros(n), np.zeros((n, n)), np.zeros(n), np.zeros(n)
    outc, sources, events = 0., 0., 0
    for seq, intensity, weight in zip(sequences, light, weights):
        if first_hit_only and seq:
            seq = seq[:1]  # intentionally wrong negative control, not a baseline
        if hierarchy is None:
            bb, aa, cc, vv, ss, ee = direct_ray(scene, seq, intensity)
        else:
            r = hierarchy.response(scene, seq)
            bb, aa, cc, vv, ss, ee = r.h*intensity+r.b, r.a, r.t*intensity+r.c, r.v, r.source, r.emission
        b += weight*bb
        a += weight*aa
        outc += weight*cc
        outv += weight*vv
        sources += weight*ss
        emitted += weight*ee
        events += len(seq)
    end = perf_counter()
    system = System(b, a, np.diag(emitted), outc, outv, events, float(np.dot(light, weights)))
    return system, sources, sequences, dict(input_response_sync_s=setup-start, parent_geometry_s=traced-setup,
        assembly_s=end-traced, update_s=end-start, cached_branches=0 if hierarchy is None else len(hierarchy.maps),
        response_payload_bytes=0 if hierarchy is None else hierarchy.payload_bytes)


class DeltaAssembly:
    """Flat contribution cache with geometric/input candidates, no new oracle."""
    def __init__(self):
        self.old = None

    def update(self, scene, x, signs, light, weights):
        import copy
        start = perf_counter()
        scene.validate()
        if self.old is None:
            self.records = [None]*len(x)
            self.sequences = [None]*len(x)
            self.b = np.zeros(len(scene.tau))
            self.a = np.zeros((len(scene.tau), len(scene.tau)))
            self.outv = np.zeros(len(scene.tau))
            self.emitted = np.zeros(len(scene.tau))
            self.outc = self.sources = 0.
            ids = np.arange(len(x))
        else:
            previous, px, ps, pl, pw = self.old
            if len(x) != len(px):
                raise ValueError('Fixed ray count required')
            mask = (x != px) | (signs != ps) | (light != pl) | (weights != pw)
            old_world, new_world = previous.world(), scene.world()
            changed = ((old_world != new_world).any(axis=1) | (previous.tau[1:] != scene.tau[1:])
                       | (previous.gain[1:] != scene.gain[1:]) | (previous.source[1:] != scene.source[1:]))
            for a, b in zip(old_world[changed], new_world[changed]):
                mask |= (x >= min(a[0]-a[2], b[0]-b[2])) & (x <= max(a[0]+a[2], b[0]+b[2]))
            ids = np.flatnonzero(mask)
        selected = perf_counter()
        for i in ids:
            old = self.records[i]
            if old is not None:
                bb, aa, cc, vv, ss, ee = old
                self.b -= bb; self.a -= aa; self.outc -= cc
                self.outv -= vv; self.sources -= ss; self.emitted -= ee
            seq = ordered(scene, x[i], signs[i])
            self.sequences[i] = seq
            record = tuple(v*weights[i] for v in direct_ray(scene, seq, light[i]))
            bb, aa, cc, vv, ss, ee = record
            self.b += bb; self.a += aa; self.outc += cc
            self.outv += vv; self.sources += ss; self.emitted += ee
            self.records[i] = record
        assembled_at = perf_counter()
        self.old = (copy.deepcopy(scene), x.copy(), signs.copy(), light.copy(), weights.copy())
        system = System(self.b.copy(), self.a.copy(), np.diag(self.emitted), self.outc,
                        self.outv.copy(), sum(map(len, self.sequences)), float(np.dot(light, weights)))
        end = perf_counter()
        return system, self.sources, list(self.sequences), dict(candidate_s=selected-start,
             assembly_s=assembled_at-selected, commit_s=end-assembled_at, update_s=end-start,
             candidate_rays=len(ids), total_rays=len(x))
