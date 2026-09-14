"""Direct sparse compilation of the existing finite sphere boundary model.

Projected atom disks exclude finite input representatives with no collision.
This is not a certificate for an entire continuous cell. All remaining candidates
are evaluated before publishing the model, so absent responses mean THROUGH,
not UNKNOWN. Geometry remains available for source seeding and recompilation.
"""
import time
import numpy as np

from .inside_out_pa import tangent_frames
from .ehss_spherical_coupling import angular_grid
from .ehss_compiled_field import RayBoundaryGrid, local_boundary_responses
from .ehss_response_sparse import (
    DirectionLookup, SparseIndexArray, SparseCompiledSupportField,
)


class ProceduralRayBoundaryGrid(RayBoundaryGrid):
    def __init__(self, shape=(4, 8, 16, 16)):
        if len(shape) != 4 or any(not isinstance(x, int) or x < 1 for x in shape):
            raise ValueError("Four positive integer grid sizes required")
        self.shape = tuple(shape)
        self.axes, _ = angular_grid(*shape[:2])
        self.t, self.b = tangent_frames(self.axes)
        self.disk_count = shape[2] * shape[3]
        self.size = len(self.axes) * self.disk_count
        self.directions = DirectionLookup(self.axes, self.disk_count, self.size)
        for array in (self.axes, self.t, self.b):
            array.setflags(write=False)

    def points(self, ids, outgoing=False):
        ids = np.asarray(ids)
        directions = self.directions[ids]  # Also validates identities.
        direction_ids = ids // self.disk_count
        disk_ids = ids % self.disk_count
        rho = np.sqrt((disk_ids // self.shape[3] + .5) / self.shape[2])
        phi = (disk_ids % self.shape[3] + .5) * 2 * np.pi / self.shape[3]
        transverse = (self.t[direction_ids] * (rho * np.cos(phi))[..., None]
                      + self.b[direction_ids] * (rho * np.sin(phi))[..., None])
        height = np.sqrt(1 - rho ** 2)
        return transverse + (1 if outgoing else -1) * height[..., None] * directions


def projected_candidates(grid, centers, radii):
    """Sorted union of line representatives intersecting projected sphere disks.

    One direction and one atom's radial band are visited at a time. Roundoff
    padding admits extra candidates; the physical/pair kernel decides actual hits.
    No full direction-by-disk mask or point array is allocated.
    """
    nr, na = grid.shape[2:]
    radial = np.sqrt((np.arange(nr) + .5) / nr)
    phi = (np.arange(na) + .5) * 2 * np.pi / na
    cosine, sine = np.cos(phi), np.sin(phi)
    parts = []
    for direction in range(len(grid.axes)):
        selected = []
        for center, radius in zip(centers, radii):
            x = np.dot(center, grid.t[direction])
            y = np.dot(center, grid.b[direction])
            distance = np.hypot(x, y)
            guard = 512 * np.finfo(float).eps * (1 + np.linalg.norm(center) + radius)
            padded = radius + guard
            rows = np.flatnonzero((radial >= max(0., distance - padded))
                                  & (radial <= distance + padded))
            if not len(rows):
                continue
            dx = radial[rows, None] * cosine - x
            dy = radial[rows, None] * sine - y
            ir, ia = np.nonzero(dx * dx + dy * dy <= padded * padded)
            selected.append(rows[ir] * na + ia)
        if selected:
            parts.append(direction * grid.disk_count + np.unique(np.concatenate(selected)))
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def membership(keys, requested):
    positions = np.searchsorted(keys, requested)
    present = positions < len(keys)
    present[present] = keys[positions[present]] == requested[present]
    return present


def concatenate(parts, dtype=np.int64, trailing=()):
    return np.concatenate(parts) if parts else np.empty((0,) + trailing, dtype=dtype)


class SparseBoundaryMapLibrary:
    """Exact-key sparse T and purely geometric G sharing, without dense maps."""
    def __init__(self):
        self.local = {}
        self.pairs = {}
        self.local_hits = self.pair_hits = 0

    def local_map(self, grid, centers, radii, cap, chunk_size):
        key = (grid.shape, cap, centers.shape, centers.tobytes(), radii.tobytes())
        if key in self.local:
            self.local_hits += 1
            return self.local[key]
        candidates = projected_candidates(grid, centers, radii)
        parts = {name: [] for name in ("ids", "output", "bounces", "stopped", "physical_outgoing")}
        tests = 0
        for start in range(0, len(candidates), chunk_size):
            ids = candidates[start:start + chunk_size]
            points, directions, counts, stopped, nt = local_boundary_responses(
                grid.points(ids), grid.directions[ids], centers, radii, cap)
            tests += nt
            active = (counts > 0) | stopped
            parts["ids"].append(ids[active])
            parts["output"].append(grid.project(points[active], directions[active]))
            parts["bounces"].append(counts[active])
            parts["stopped"].append(stopped[active])
            parts["physical_outgoing"].append(directions[active])
        result = {name: concatenate(values,
                   bool if name == "stopped" else float if name == "physical_outgoing" else np.int64,
                   (3,) if name == "physical_outgoing" else ()) for name, values in parts.items()}
        result.update(candidate_cells=len(candidates), primitive_tests=int(tests),
                      analytically_excluded=grid.size - len(candidates))
        self.local[key] = result
        return result

    def pair_map(self, grid, delta, radius_ratio, chunk_size):
        key = (grid.shape, tuple(delta), float(radius_ratio))
        if key in self.pairs:
            self.pair_hits += 1
            return self.pairs[key]
        candidates = projected_candidates(grid, np.asarray([delta]), np.asarray([radius_ratio]))
        parts = {name: [] for name in ("ids", "starts", "far", "target", "inside")}
        for start in range(0, len(candidates), chunk_size):
            ids = candidates[start:start + chunk_size]
            directions = grid.directions[ids]
            relative = grid.points(ids, outgoing=True) - delta
            projection = np.sum(relative * directions, axis=1)
            perpendicular = relative - projection[:, None] * directions
            disc = radius_ratio ** 2 - np.sum(perpendicular * perpendicular, axis=1)
            half = np.sqrt(np.maximum(0., disc))
            near, far = -projection - half, -projection + half
            hit = (disc >= 0.) & (far > 0.)
            parts["ids"].append(ids[hit])
            parts["starts"].append(np.maximum(0., near[hit]))
            parts["far"].append(far[hit])
            parts["target"].append(grid.project(relative[hit] / radius_ratio,
                                               direction_ids=ids[hit] // grid.disk_count))
            parts["inside"].append(near[hit] < 0.)
        result = {name: concatenate(values, bool if name == "inside" else
                  np.int64 if name in ("ids", "target") else float) for name, values in parts.items()}
        result["candidate_cells"] = len(candidates)
        self.pairs[key] = result
        return result

    def receipt(self):
        arrays = [value for entry in (*self.local.values(), *self.pairs.values())
                  for value in entry.values() if isinstance(value, np.ndarray)]
        return dict(local_maps=len(self.local), pair_maps=len(self.pairs),
                    local_hits=self.local_hits, pair_hits=self.pair_hits,
                    payload_bytes=sum(a.nbytes for a in arrays),
                    local_candidate_cells=sum(m["candidate_cells"] for m in self.local.values()),
                    local_domain_cells=sum(key[0][0] * key[0][1] * key[0][2] * key[0][3] for key in self.local),
                    local_primitive_tests=sum(m["primitive_tests"] for m in self.local.values()),
                    pair_candidate_cells=sum(m["candidate_cells"] for m in self.pairs.values()),
                    pair_hit_cells=sum(len(m["ids"]) for m in self.pairs.values()))


class DirectSparseSupportField(SparseCompiledSupportField):
    """Compile response-only T and continuous-through routing directly.

    The optional library belongs to the caller. The completed field does not
    retain it, or full-domain T/G/basis arrays. Every finite state is classified;
    this constructor does not implement lazy UNKNOWN or MD invalidation.
    """
    def __init__(self, scene, shape=(4, 8, 16, 16), library=None, local_cap=64, chunk_size=16384):
        if not isinstance(local_cap, int) or local_cap < 1 or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("Positive integer cap and chunk size required")
        start = time.perf_counter()
        self.scene = scene
        self.grid = grid = ProceduralRayBoundaryGrid(shape)
        self.phase_seconds = dict(basis=time.perf_counter() - start, local=0., pairs=0., closure=0.)
        self.through_mode = "preserve"
        n, m = grid.size, len(scene.radii)
        self.states = n * m
        library = SparseBoundaryMapLibrary() if library is None else library
        responses = []
        phase = time.perf_counter()
        for a in range(m):
            owned = scene.groups == a
            local = np.ascontiguousarray((scene.spheres.centers[owned] - scene.centers[a]) / scene.radii[a])
            radii = np.ascontiguousarray(scene.spheres.radii[owned] / scene.radii[a])
            responses.append(library.local_map(grid, local, radii, local_cap, chunk_size))
        self.phase_seconds["local"] = time.perf_counter() - phase
        self.response_ids = np.concatenate([a * n + r["ids"] for a, r in enumerate(responses)])
        self.response_ids.setflags(write=False)
        for name in ("output", "bounces", "stopped", "physical_outgoing"):
            values = np.concatenate([r[name] + (a * n if name == "output" else 0)
                                     for a, r in enumerate(responses)])
            if name == "stopped":
                values = values.astype(bool)
            setattr(self, name, SparseIndexArray(self.response_ids, values, self.states))
        self.through_skips = self.overlap_routes = 0
        route_ids, route_targets = [], []
        self.closure_candidate_cells = 0
        for a in range(m):
            pairs = []
            phase = time.perf_counter()
            for b in range(m):
                if a != b:
                    pair = library.pair_map(grid, (scene.centers[b] - scene.centers[a]) / scene.radii[a],
                                            scene.radii[b] / scene.radii[a], chunk_size)
                    pairs.append((b, pair))
            self.phase_seconds["pairs"] += time.perf_counter() - phase
            phase = time.perf_counter()
            union = np.unique(concatenate([p["ids"] for _, p in pairs]))
            self.closure_candidate_cells += len(union)
            for first in range(0, len(union), chunk_size):
                ids = union[first:first + chunk_size]
                starts = np.full((len(pairs), len(ids)), np.inf)
                ends = np.full_like(starts, -np.inf)
                targets = np.full(starts.shape, -1, dtype=np.int64)
                inside = np.zeros(starts.shape, dtype=bool)
                for row, (b, pair) in enumerate(pairs):
                    pos = np.searchsorted(pair["ids"], ids)
                    valid = membership(pair["ids"], ids)
                    starts[row, valid] = pair["starts"][pos[valid]]
                    ends[row, valid] = pair["far"][pos[valid]]
                    targets[row, valid] = b * n + pair["target"][pos[valid]]
                    inside[row, valid] = pair["inside"][pos[valid]]
                schedule = np.argsort(starts, axis=0, kind="stable")
                columns = np.arange(len(ids))
                self.overlap_routes += int(inside[schedule[0], columns].sum())
                cursor = np.zeros(len(ids))
                pending = np.ones(len(ids), dtype=bool)
                route = np.full(len(ids), -1, dtype=np.int64)
                for candidate in schedule:
                    entry, far = starts[candidate, columns], ends[candidate, columns]
                    target = targets[candidate, columns]
                    valid = pending & np.isfinite(entry) & (far > cursor)
                    response = membership(self.response_ids, target)
                    selected = valid & response
                    route[selected] = target[selected]
                    pending[selected] = False
                    through = valid & ~response
                    cursor[through] = far[through]
                    self.through_skips += int(through.sum())
                active = route >= 0
                route_ids.append(a * n + ids[active])
                route_targets.append(route[active])
            self.phase_seconds["closure"] += time.perf_counter() - phase
        self.route_ids = concatenate(route_ids)
        self.route_ids.setflags(write=False)
        self.routing = SparseIndexArray(self.route_ids, concatenate(route_targets), self.states, default=-1)
        self.successor = SparseIndexArray(self.response_ids, self.routing[self.output.values], self.states)
        self.compilation_counts = library.receipt()
        self.compile_s = time.perf_counter() - start
        self.phase_seconds["assembly_other"] = self.compile_s - sum(self.phase_seconds.values())
