"""Dependency-local revalidation of a fixed positive boundary-region catalog.

This updates existing sampled response regions; it does not discover new cells.
Candidate membership is conservatively rebuilt for every frame. Local version
keys include all possible competing atoms, not only the recorded itinerary.
Node payloads are immutable, published transactionally, and kept in a bounded
LRU cache. Frame assembly still repacks the complete active response table.
"""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import time
import numpy as np
from .ehss_reflecting_regions import ReflectingRegionHierarchy
from .ehss_clipped_response import compile_cells


def hash_arrays(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.asarray(array)
        digest.update(repr((array.shape, array.dtype.str)).encode('ascii'))
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class NodeRegionVersion:
    node: int
    version: str
    region_indices: tuple
    routes: tuple
    tested_regions: int
    probe_calls: int
    probe_sphere_tests: int


@dataclass(frozen=True)
class RegionFrame:
    model: ReflectingRegionHierarchy
    nodes: tuple
    receipt: dict


class RegionVersionCompiler:
    def __init__(self, seed, atom_ids=None, max_cache_entries=256):
        if not seed.prepared: raise ValueError('Prepared positive regions required as the fixed catalog')
        if not isinstance(max_cache_entries, (int, np.integer)) or max_cache_entries < 1:
            raise ValueError('Positive cache entry limit required')
        self.atom_ids = tuple(range(len(seed.spheres.radii))) if atom_ids is None else tuple(atom_ids)
        if len(self.atom_ids) != len(seed.spheres.radii) or len(set(self.atom_ids)) != len(self.atom_ids):
            raise ValueError('One unique stable identity per physical atom required')
        self.catalog = seed.region_keys.copy(); self.catalog.setflags(write=False)
        self.normal = seed.normal.copy(); self.edges = seed.edges.copy(); self.widths = seed.widths.copy()
        self.shared_last = seed.shared_last; self.level_count = seed.active_levels.shape[1]
        self.max_route = seed.compile_receipt.get('max_route', 16)
        self.node_count = len(seed.left); self.max_cache_entries = int(max_cache_entries)
        self.node_rows = tuple(np.flatnonzero(self.catalog[:, 0] == n) for n in range(self.node_count))
        for a in (self.normal, self.edges, self.widths, *self.node_rows): a.setflags(write=False)
        self._cache = OrderedDict(); self.current = None
        self.initial_versions = []
        for node, rows in enumerate(self.node_rows):
            version, _, _ = self._version(seed, node)
            routes = tuple(tuple(int(a) for a in seed.paths[seed.path_offsets[seed.region_routes[r]]:seed.path_offsets[seed.region_routes[r]+1]]) for r in rows)
            value = NodeRegionVersion(node, version, tuple(int(r) for r in rows), routes, 0, 0, 0)
            self._cache[node, version] = value; self.initial_versions.append(version)
        while len(self._cache) > self.max_cache_entries: self._cache.popitem(last=False)
        self.initial_versions = tuple(self.initial_versions)

    def _version(self, model, node):
        lower, upper = int(model.lower[node]), int(model.upper[node])
        plane_low = model.edges[lower-1] if lower else -np.inf
        plane_high = model.edges[upper-1] if upper <= len(model.edges) else np.inf
        centers, radii = model.spheres.centers, model.spheres.radii
        # Also cover roundoff guards in probe validity / beam filtering, which
        # are somewhat wider than the leaf's physical candidate inclusion.
        scale = max(1., abs(plane_low) if np.isfinite(plane_low) else 0., abs(plane_high) if np.isfinite(plane_high) else 0.)
        pad = 8192*np.finfo(float).eps*(1.+np.linalg.norm(centers, axis=1)+radii+scale)
        q = centers@model.normal
        dependencies = np.flatnonzero((q+radii+pad >= plane_low)&(q-radii-pad <= plane_high))
        physical = np.unique(model.ids[model.offsets[lower]:model.offsets[upper]])
        dependencies = np.union1d(dependencies, physical)
        cuts = model.edges[max(0, lower-1):min(len(model.edges), upper)]
        digest = hash_arrays(np.array([node, lower, upper, self.max_route, self.shared_last], dtype=np.int64),
            model.frame, cuts, model.widths, self.catalog[self.node_rows[node]], dependencies,
            centers[dependencies], radii[dependencies])
        return digest, dependencies, physical

    def _revalidate_node(self, model, node, version):
        kept = {}; rows = self.node_rows[node]; probes = 0; tests = 0
        for level in range(self.level_count):
            selected = rows[self.catalog[rows, 1] == level]
            if not len(selected): continue
            region_keys = self.catalog[selected]
            keys = region_keys[:, [0, 2, 3, 4, 5, 6, 7]].copy()
            kinds, paths, counts, pc, nt, _ = compile_cells(keys, model.normal, model.edges, model.offsets, model.ids,
                model.spheres.centers, model.spheres.radii, model.lower, model.upper, model.left, model.right,
                model.u, model.v, model.widths*(2.**level), self.max_route)
            probes += int(pc); tests += int(nt)
            for j, index in enumerate(selected):
                if kinds[j] != 2: continue
                route = tuple(int(a) for a in paths[j, :counts[j]])
                if level:
                    parent = self.catalog[index]
                    children = rows[(self.catalog[rows, 1] == level-1)&(self.catalog[rows, 2] == parent[2])&
                                    (self.catalog[rows, 3] == parent[3])&np.all(self.catalog[rows, 4:]//2 == parent[4:], axis=1)]
                    # Known smaller regions may expose a contradiction missed
                    # by the coarser probes. Do not cover those with a parent.
                    if not len(children) or any(int(c) not in kept or kept[int(c)] != route for c in children): continue
                kept[int(index)] = route
        indices = tuple(sorted(kept))
        return NodeRegionVersion(node, version, indices, tuple(kept[i] for i in indices), len(rows), probes, tests)

    def _assemble(self, model, values):
        kept = {index:route for value in values for index, route in zip(value.region_indices, value.routes)}
        rows = sorted(kept); unique = {}; routes = []; indices = []
        for row in rows:
            route = kept[row]
            if route not in unique: unique[route] = len(routes); routes.append(route)
            indices.append(unique[route])
        model.region_keys = self.catalog[rows].copy()
        model.region_routes = np.array(indices, dtype=np.int64)
        model.path_offsets = np.r_[0, np.cumsum([len(route) for route in routes])].astype(np.int64)
        model.paths = np.array([a for route in routes for a in route], dtype=np.int32)
        model.active_levels = np.zeros((self.node_count, self.level_count), dtype=np.bool_)
        for key, route_index in zip(model.region_keys, model.region_routes):
            model.region_table[tuple(int(v) for v in key)] = int(route_index)
            model.active_levels[key[0], key[1]] = True
        for a in (model.region_keys, model.region_routes, model.path_offsets, model.paths, model.active_levels): a.setflags(write=False)
        model.shared_last = self.shared_last; model.prepared = True
        model.compile_receipt = dict(max_route=self.max_route, regions=len(rows), fixed_catalog_regions=len(self.catalog),
                                    unique_routes=len(routes), new_region_discovery=False)

    def compile(self, spheres, atom_ids=None, normal=None, edges=None, reuse=True):
        identities = self.atom_ids if atom_ids is None else tuple(atom_ids)
        if identities != self.atom_ids or len(spheres.radii) != len(self.atom_ids):
            raise ValueError('Physical atom identity and ordering must remain fixed')
        new_edges = self.edges if edges is None else np.asarray(edges, dtype=float)
        if new_edges.shape != self.edges.shape: raise ValueError('Changed partition topology needs a new catalog')
        start = time.perf_counter()
        model = ReflectingRegionHierarchy(spheres, self.normal if normal is None else normal, new_edges, self.widths[0], self.widths[2])
        candidate_and_gate_s = time.perf_counter()-start
        tick = time.perf_counter(); keys = []; dependency_counts = []; membership_counts = []
        for node in range(self.node_count):
            version, dependencies, physical = self._version(model, node)
            keys.append((node, version)); dependency_counts.append(len(dependencies)); membership_counts.append(len(physical))
        dependency_scan_s = time.perf_counter()-tick
        tick = time.perf_counter(); values = []; cache_hits = []; rebuilt = []; pending = {}
        for node, key in enumerate(keys):
            if reuse and key in self._cache:
                value = self._cache[key]; cache_hits.append(node)
            else:
                value = self._revalidate_node(model, node, key[1]); pending[key] = value; rebuilt.append(node)
            values.append(value)
        revalidation_s = time.perf_counter()-tick
        tick = time.perf_counter(); self._assemble(model, values)
        assembly_s = time.perf_counter()-tick
        # No published state is mutated before candidate construction,
        # validation and complete response assembly all succeed.
        evicted = 0; new_cache = self._cache
        if reuse:
            new_cache = OrderedDict(self._cache)
            for key in keys:
                if key in pending: new_cache[key] = pending[key]
                new_cache.move_to_end(key)
            while len(new_cache) > self.max_cache_entries:
                new_cache.popitem(last=False); evicted += 1
        previous = self.current.nodes if self.current is not None else None
        geometry_changed = [n for n, key in enumerate(keys) if key[1] != (previous[n].version if previous else self.initial_versions[n])]
        receipt = dict(candidate_and_gate_s=candidate_and_gate_s, dependency_scan_s=dependency_scan_s,
            revalidation_s=revalidation_s, assembly_s=assembly_s, total_s=time.perf_counter()-start,
            reuse_enabled=bool(reuse), geometry_changed_nodes=geometry_changed, reused_nodes=cache_hits, rebuilt_nodes=rebuilt,
            revalidated_regions=sum(values[n].tested_regions for n in rebuilt), probe_calls=sum(values[n].probe_calls for n in rebuilt),
            probe_sphere_tests=sum(values[n].probe_sphere_tests for n in rebuilt), retained_regions=len(model.region_keys),
            fixed_catalog_regions=len(self.catalog), dependency_counts=dependency_counts, candidate_counts=membership_counts,
            cache_versions=len(new_cache), evicted_versions=evicted, cache_limit=self.max_cache_entries,
            candidate_scan_is_global=True, response_table_assembly_is_global=True, new_region_discovery=False)
        frame = RegionFrame(model, tuple(values), receipt)
        if reuse: self._cache = new_cache; self.current = frame
        return frame
