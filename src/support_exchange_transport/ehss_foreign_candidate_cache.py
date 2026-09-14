"""Sparse, buffered foreign-flight candidate reuse for numerical internal T.

For old ray p0+t*d0, build all foreign spheres within radii+slack of the
forward ray. A current hit has t <= L, where L bounds the current molecular
enclosure. If |p-p0| + L*|d-d0| + max_j(|c_j-c0_j|+|r_j-r0_j|) < slack,
every current foreign hit is in that old candidate set. The bound compares
to the retained epoch, never sums only adjacent-step motion. Candidates are
intersected in current geometry; this is not frozen hit/escape reuse.
"""
from copy import copy
from dataclasses import dataclass
import time
import numpy as np
from numba import njit
from .ehss_reference import entry_distance
from .ehss_guarded_intrinsic_response import reflect
from .ehss_observation_refresh import current_occlusion
from .ehss_sparse_path_response import readonly


@njit(cache=True)
def build_candidates(points, directions, selected, owners, centers, radii, slack):
    starts = [0]
    candidates = []
    checks = 0
    for i in range(len(points)):
        aa = np.dot(directions[i], directions[i])
        for atom in range(len(radii)):
            if owners[atom] == selected[i]:
                continue
            checks += 1
            x = centers[atom]-points[i]
            projection = max(0., np.dot(x, directions[i])/aa)
            perpendicular = x-projection*directions[i]
            guard = 256*np.finfo(np.float64).eps*(1.+np.linalg.norm(x)+radii[atom]+slack)
            if np.dot(perpendicular, perpendicular) <= (radii[atom]+slack+guard)**2:
                candidates.append(atom)
        starts.append(len(candidates))
    return np.array(starts, dtype=np.int64), np.array(candidates, dtype=np.int64), checks


@njit(cache=True)
def intersect_candidates(points, directions, limits, starts, ids, centers, radii):
    atoms = np.full(len(points), -1, dtype=np.int64)
    distances = np.full(len(points), np.inf)
    tests = 0
    for i in range(len(points)):
        best = limits[i]
        for j in range(starts[i], starts[i+1]):
            atom = ids[j]
            distance = entry_distance(points[i], directions[i], centers[atom], radii[atom])
            tests += 1
            if np.isfinite(distance) and (distance < best or (distance == best and (atoms[i] < 0 or atom < atoms[i]))):
                atoms[i], best = atom, distance
        if atoms[i] >= 0:
            distances[i] = best
    return atoms, distances, tests


@dataclass(frozen=True)
class Epoch:
    centers: np.ndarray
    radii: np.ndarray


@dataclass(frozen=True)
class FlightCandidates:
    epoch: Epoch
    origin: np.ndarray
    direction: np.ndarray
    atoms: np.ndarray


class ForeignCandidatePool:
    def __init__(self, bank, slack=.25, reuse=True):
        if not np.isfinite(slack) or slack <= 0:
            raise ValueError('Finite positive candidate slack required')
        self.bank, self.slack, self.reuse = bank, float(slack), bool(reuse)
        self.records = {}
        self.receipt = {}

    def snapshot(self):
        result = copy(self)
        result.records = self.records.copy()
        result.receipt = {}
        return result

    def begin(self, bound):
        if bound.bank is not self.bank:
            raise ValueError('Candidate cache belongs to another material bank')
        spheres = bound.search.model.spheres
        self.world, self.radii = spheres.centers, spheres.radii
        self.center = self.world.mean(axis=0)
        self.radius = float(np.max(np.linalg.norm(self.world-self.center, axis=1)+self.radii))
        self.epoch = Epoch(readonly(self.world.copy()), readonly(self.radii.copy()))
        self.motion, self.kept = {}, {}
        self.receipt = dict(candidate_queries=0, reused_flights=0, rebuilt_flights=0, empty_reused_flights=0,
            candidate_intersections=0, neighborhood_distance_checks=0, motion_atom_checks=0,
            gate_s=0., build_s=0., intersect_s=0., slack=self.slack,
            current_intersections=True, conservative_candidate_cover=True,
            fixed_foreign_collision_reused=False, active_flights_only=True)

    def query(self, keys, points, directions, selected, limits):
        tick = time.perf_counter()
        chunks = [None]*len(keys)
        missing = []
        for i, key in enumerate(keys):
            old = self.records.get(key)
            valid = False
            if self.reuse and old is not None:
                epoch = id(old.epoch)
                if epoch not in self.motion:
                    self.motion[epoch] = float(np.max(np.linalg.norm(self.world-old.epoch.centers, axis=1)
                        +np.abs(self.radii-old.epoch.radii)))
                    self.receipt['motion_atom_checks'] += len(self.radii)
                length = (np.linalg.norm(points[i]-self.center)+self.radius)/np.linalg.norm(directions[i])
                change = (np.linalg.norm(points[i]-old.origin)+length*np.linalg.norm(directions[i]-old.direction)
                          +self.motion[epoch])
                guard = 512*np.finfo(np.float64).eps*(1.+length+self.radius+np.linalg.norm(points[i]))
                valid = change+guard < self.slack
            if valid:
                chunks[i] = old.atoms
                self.kept[key] = old
                self.receipt['reused_flights'] += 1
                self.receipt['empty_reused_flights'] += int(len(old.atoms) == 0)
            else:
                missing.append(i)
        self.receipt['gate_s'] += time.perf_counter()-tick
        tick = time.perf_counter()
        if missing:
            rows = np.asarray(missing, dtype=np.int64)
            offsets, ids, checks = build_candidates(points[rows], directions[rows], selected[rows],
                self.bank.owners, self.world, self.radii, self.slack)
            self.receipt['neighborhood_distance_checks'] += int(checks)
            self.receipt['rebuilt_flights'] += len(missing)
            for j, i in enumerate(missing):
                record = FlightCandidates(self.epoch, readonly(points[i].copy()), readonly(directions[i].copy()),
                    readonly(ids[offsets[j]:offsets[j+1]].copy()))
                self.kept[keys[i]] = record
                chunks[i] = record.atoms
        starts = np.r_[0, np.cumsum([len(a) for a in chunks], dtype=np.int64)]
        ids = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
        self.receipt['build_s'] += time.perf_counter()-tick
        tick = time.perf_counter()
        atoms, distances, tests = intersect_candidates(points, directions, limits, starts, ids, self.world, self.radii)
        self.receipt['intersect_s'] += time.perf_counter()-tick
        self.receipt['candidate_intersections'] += int(tests)
        self.receipt['candidate_queries'] += len(keys)
        return atoms, distances

    def finish(self):
        self.records = self.kept
        self.receipt.update(stored_flights=len(self.records), stored_candidate_ids=sum(len(r.atoms) for r in self.records.values()),
            retained_epochs=len({id(r.epoch) for r in self.records.values()}))


@njit(cache=True)
def pose_column(rows, starts, origins, directions, rotation, translation, input_points, input_directions):
    """Preserve the original rotation layout and arithmetic before flattening.

    Stacking Fortran-layout rotations into a C-layout tensor changes the BLAS
    reduction path at roundoff level; a later material bin/suffix can amplify
    that change. Use exactly the original per-owner matrix and operations.
    """
    p, d = np.empty_like(origins), np.empty_like(directions)
    for i, row in enumerate(rows):
        anchor = input_points[row]-(origins[starts[i]]@rotation.T+translation)
        for segment in range(starts[i], starts[i+1]):
            if segment == starts[i]:
                p[segment], d[segment] = input_points[row], input_directions[row]
            else:
                p[segment] = origins[segment]@rotation.T+translation+anchor
                d[segment] = directions[segment]@rotation.T
    return p, d


def flatten(bound):
    """Pose numerical T before flattening; foreign candidates remain lazy."""
    bank = bound.bank
    starts, ends = np.zeros(len(bank.lasts), dtype=np.int64), np.zeros(len(bank.lasts), dtype=np.int64)
    origins, directions, targets, distances = [], [], [], []
    cursor = 0
    for owner, (rows, column) in enumerate(zip(bank.rows, bound.columns)):
        if column is None:
            continue
        starts[rows], ends[rows] = cursor+column.starts[:-1], cursor+column.starts[1:]
        p, d = pose_column(rows, column.starts, column.origins, column.directions,
            bound.rotations[owner], bound.translations[owner], bound.points, bound.directions)
        origins.append(p)
        directions.append(d)
        atoms = np.full(len(column.targets), -1, dtype=np.int64)
        valid = column.targets >= 0
        atoms[valid] = bank.atoms[owner][column.targets[valid]]
        targets.append(atoms)
        distances.append(column.distances)
        cursor += len(column.targets)
    return (starts, ends, np.concatenate(origins) if origins else np.empty((0, 3)),
        np.concatenate(directions) if directions else np.empty((0, 3)),
        np.concatenate(targets) if targets else np.empty(0, dtype=np.int64),
        np.concatenate(distances) if distances else np.empty(0))


@njit(cache=True)
def advance(rows, positions, directions, wanted, limit, foreign, before, cap,
            centers, radii, counts, history, terminal, points, outgoing, progress):
    following = []
    events = interruptions = 0
    for i, row in enumerate(rows):
        points[row], outgoing[row] = positions[i], directions[i]
        if foreign[i] >= 0 and (wanted[i] < 0 or before[i] < limit[i] or (before[i] == limit[i] and foreign[i] < wanted[i])):
            if counts[row] >= cap:
                continue
            reflect(points[row], outgoing[row], foreign[i], before[i], centers, radii)
            history[row, counts[row]] = foreign[i]
            counts[row] += 1
            terminal[row] = foreign[i]
            interruptions += int(wanted[i] >= 0)
        elif wanted[i] < 0:
            terminal[row] = -1
        elif counts[row] < cap:
            history[row, counts[row]] = wanted[i]
            counts[row] += 1
            progress[row] += 1
            events += 1
            following.append(row)
    return np.array(following, dtype=np.int64), events, interruptions


class CandidateStrainBinding:
    def __init__(self, bound, pool):
        self.bound, self.pool = bound, pool

    def __getattr__(self, name):
        return getattr(self.bound, name)

    def apply(self):
        tick = time.perf_counter()
        bound, pool, bank = self.bound, self.pool, self.bank
        pool.begin(bound)
        spheres, tree = self.search.model.spheres, self.search.model.tree
        occluded, counters = current_occlusion(bound.points, bank.lasts, spheres.centers, spheres.radii,
            tree.lo, tree.hi, tree.left, tree.right, tree.leaf_group, tree.offsets, tree.ids)
        starts, ends, world_p, world_d, wanted, limits = flatten(bound)
        progress = np.zeros(len(bank.lasts), dtype=np.int64)
        counts = np.ones(len(bank.lasts), dtype=np.int64)
        history = np.full((len(bank.lasts), bank.cap), -1, dtype=np.int64)
        history[:, 0] = bank.lasts
        terminal = np.full(len(bank.lasts), -2, dtype=np.int64)
        terminal[occluded] = -3
        points, directions = bound.points.copy(), bound.directions.copy()
        owners = bank.owners[bank.lasts]
        rows = np.flatnonzero(~occluded)
        events = interruptions = waves = 0
        while len(rows):
            segments = starts[rows]+progress[rows]
            if np.any(segments >= ends[rows]):
                raise AssertionError('Internal column ended without a terminal flight')
            p, d = world_p[segments], world_d[segments]
            keys = [(int(row), int(progress[row])) for row in rows]
            foreign, before = pool.query(keys, p, d, owners[rows], limits[segments])
            rows, added, interrupted = advance(rows, p, d, wanted[segments], limits[segments], foreign, before,
                bank.cap, spheres.centers, spheres.radii, counts, history, terminal, points, directions, progress)
            events += added
            interruptions += interrupted
            waves += 1
        pool.finish()
        reused = np.isin(owners, bound.receipt['reused_owners'])
        local_events = counts-1-(terminal >= 0)
        assert events == local_events.sum()
        last = history[np.arange(len(counts)), counts-1]
        drift = np.abs(np.linalg.norm(points-spheres.centers[last], axis=1)-spheres.radii[last])
        active = terminal != -3
        result = dict(points=points, directions=directions, counts=counts, history=history, terminal=terminal,
            input_mass=float(bank.weights.sum()), escape_mass=float(bank.weights[terminal == -1].sum()),
            foreign_mass=float(bank.weights[terminal >= 0].sum()), occluded_mass=float(bank.weights[terminal == -3].sum()),
            unresolved_mass=float(bank.weights[terminal == -2].sum()),
            completed_direction_moment=float(np.sum(bank.weights*np.clip(1-np.sum(bound.tags*directions, axis=1), 0, 2)*(terminal >= -1))),
            local_events_from_stored_numbers=int(events), reused_local_events=int(local_events[reused].sum()),
            foreign_queries=pool.receipt['candidate_queries'], foreign_interruptions=int(interruptions),
            box_tests=int(counters[0]), sphere_tests=int(counters[1])+pool.receipt['candidate_intersections'], owner_prunes=0,
            query_s=time.perf_counter()-tick, global_ccs_readout=False,
            max_internal_endpoint_surface_drift=float(drift[active].max(initial=0.)),
            weighted_endpoint_surface_drift=float(np.dot(bank.weights[active], drift[active])),
            candidate_refresh=pool.receipt.copy(), waves=waves)
        if not np.isclose(result['input_mass'], sum(result[k] for k in ['escape_mass', 'foreign_mass', 'occluded_mass', 'unresolved_mass']), rtol=1e-12, atol=1e-12):
            raise AssertionError('Candidate T/G mass balance changed')
        return result
