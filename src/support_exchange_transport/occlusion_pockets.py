"""First-hit transport in many cavities with genuinely volumetric interiors.

Accelerated routes reuse fixed-geometry hits. The default current-hit candidate
map preserves the original experiment; an explicit envelope policy supports
factorial comparisons independent of eager/lazy detail preparation. Certificates
are valid for fixed source, fixed motion bounds and CURRENT lid geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import numpy as np
from numba import njit

from .occlusion_hierarchy import (ConfinedScene, Part, MeshTree, HitState, panel,
                                  box_entry, empty_stats, checked_triangles, checked_box, contains)


@njit(cache=True)
def candidates(origins, directions, limits, lo, hi, end, owner, owner_count, flat):
    """Two-pass packed owner/ray map, with identical inclusive AABB arithmetic."""
    counts = np.zeros(owner_count, np.int64)
    visits = 0
    for i in range(len(origins)):
        node = 0
        while node < len(owner):
            visits += 1
            if flat and owner[node] < 0:
                node += 1
                continue
            if not np.isfinite(box_entry(origins[i], directions[i], lo[node], hi[node], limits[i])):
                node = node+1 if flat else end[node]
                continue
            if owner[node] >= 0:
                counts[owner[node]] += 1
            node += 1
    offsets = np.zeros(owner_count+1, np.int64)
    for j in range(owner_count):
        offsets[j+1] = offsets[j]+counts[j]
    rays = np.empty(offsets[-1], np.int64)
    cursor = offsets[:-1].copy()
    for i in range(len(origins)):
        node = 0
        while node < len(owner):
            visits += 1
            if flat and owner[node] < 0:
                node += 1
                continue
            if not np.isfinite(box_entry(origins[i], directions[i], lo[node], hi[node], limits[i])):
                node = node+1 if flat else end[node]
                continue
            k = owner[node]
            if k >= 0:
                rays[cursor[k]] = i
                cursor[k] += 1
            node += 1
    return offsets, rays, visits


class SupportTree:
    def __init__(self, bounds):
        lo, hi, end, owner = [], [], [], []

        def build(ids):
            node = len(owner)
            lower, upper = bounds[ids, 0].min(axis=0), bounds[ids, 1].max(axis=0)
            lo.append(lower); hi.append(upper); end.append(-1); owner.append(-1)
            if len(ids) == 1:
                owner[node] = ids[0]
            else:
                centers = bounds[ids].mean(axis=1)
                axis = np.argmax(np.ptp(centers, axis=0))
                ordered = ids[np.argsort(centers[:, axis], kind='stable')]
                cut = len(ids)//2
                build(ordered[:cut]); build(ordered[cut:])
            end[node] = len(owner)
        build(np.arange(len(bounds)))
        self.lo, self.hi = np.asarray(lo), np.asarray(hi)
        self.end, self.owner = np.asarray(end, dtype=np.int64), np.asarray(owner, dtype=np.int64)


def lobed_body(center, scale, phase):
    """Closed 48-face lobed ellipsoid: independent 3D bodies, no coplanar refinement."""
    vertices = [[0., 0., -1.]]
    for lat in [np.pi/4, np.pi/2, 3*np.pi/4]:
        for j in range(8):
            angle = 2*np.pi*j/8
            radius = np.sin(lat)*(1+.16*np.cos(3*angle+phase))
            vertices.append([radius*np.cos(angle), radius*np.sin(angle), -np.cos(lat)])
    vertices.append([0., 0., 1.])
    faces = []
    for j in range(8):
        k = (j+1)%8
        faces.extend([[0, 1+k, 1+j], [25, 17+j, 17+k]])
        for ring in range(2):
            a, b, c, d = 1+ring*8+j, 1+ring*8+k, 9+ring*8+k, 9+ring*8+j
            faces.extend([[a, b, c], [a, c, d]])
    return (np.asarray(vertices)*scale+center)[np.asarray(faces)]


def payload(center, layers, version):
    result = []
    cx, cy = center
    for tx in range(4):
        for ty in range(4):
            bodies = []
            for layer in range(layers):
                phase = .7*layer+.3*tx+.4*ty+.18*version
                x = cx-.75+.5*tx+.045*np.sin(phase+version*.4)
                y = cy-.75+.5*ty+.045*np.cos(phase-version*.3)
                z = -.8+(layer+.5)*1.4/layers+.025*np.sin(version+tx+ty+layer)
                scale = np.array([.15*(1+.06*np.sin(version+layer)), .15, .40/layers])
                bodies.append(lobed_body([x, y, z], scale, phase))
            result.append(np.concatenate(bodies))
    return result


def lid(center, opening):
    # A hinged physical lid; opening specifies the uncovered projected fraction.
    theta = np.arccos(1-opening)
    tri = panel([0., 0., 0.], [1., 0., 0.], [0., 1., 0.])
    x = tri[:, :, 0]+1
    tri[:, :, 0] = center[0]-1+x*np.cos(theta)
    tri[:, :, 1] += center[1]
    tri[:, :, 2] = .9+x*np.sin(theta)
    return tri


@dataclass
class PocketExperiment:
    static: np.ndarray
    parts: list
    envelope: np.ndarray
    frames: list
    origins: np.ndarray
    directions: np.ndarray
    lid_count: int
    groups: list
    metadata: dict

    def scene(self, grouped=False):
        specs = self.parts
        if grouped:
            specs = self.parts[:self.lid_count]+[
                dict(name=f'group_{i:03}', triangles=np.concatenate([specs[j]['triangles'] for j in group]),
                     envelope=np.array([np.min([specs[j]['envelope'][0] for j in group], axis=0),
                                        np.max([specs[j]['envelope'][1] for j in group], axis=0)]))
                for i, group in enumerate(self.groups)]
        return PocketScene(self.static, [Part(**p) for p in specs], self.envelope,
                           self.lid_count, self.groups if grouped else None,
                           self.parts)


def pocket_experiment(grid=4, layers=4, changed=1, hold=6, opening=.15,
                      rays_per_pocket_side=8, seed=731, oblique=False, always_open=False):
    if not 1 <= changed <= grid*grid or layers < 1:
        raise ValueError('Invalid population')
    centers = [(3.*x, 3.*y) for x in range(grid) for y in range(grid)]
    # Select separated changed pockets to avoid favoring one contiguous branch.
    chosen = np.linspace(0, len(centers)-1, changed, dtype=int).tolist()
    part_specs, groups, static = [], [], []
    for index in chosen:
        cx, cy = centers[index]
        part_specs.append(dict(name=f'lid_{index:03}', triangles=lid((cx, cy), opening if always_open else 0.),
                               envelope=np.array([[cx-1.01, cy-1.01, .89], [cx+1.01, cy+1.01, 2.91]])))
    lid_count = len(part_specs)
    for index, (cx, cy) in enumerate(centers):
        static.extend([panel([cx, cy, -1.3], [1., 0., 0.], [0., 1., 0.]),
                       panel([cx-1, cy, -.2], [0., 1., 0.], [0., 0., 1.1]),
                       panel([cx+1, cy, -.2], [0., 1., 0.], [0., 0., 1.1]),
                       panel([cx, cy-1, -.2], [1., 0., 0.], [0., 0., 1.1]),
                       panel([cx, cy+1, -.2], [1., 0., 0.], [0., 0., 1.1])])
        leaves = payload((cx, cy), layers, 0)
        if index not in chosen:
            static.extend(leaves)
            static.append(lid((cx, cy), .35))
        else:
            group = []
            for j, triangles in enumerate(leaves):
                tx, ty = divmod(j, 4)
                group.append(len(part_specs))
                part_specs.append(dict(name=f'body_{index:03}_{j:02}', triangles=triangles,
                                       envelope=np.array([[cx-1+.5*tx, cy-1+.5*ty, -1.25],
                                                          [cx-.5+.5*tx, cy-.5+.5*ty, .85]])))
            groups.append(group)
    frames = [('initial', {})]
    for version in range(1, hold+1):
        change = {}
        for index in chosen:
            for j, triangles in enumerate(payload(centers[index], layers, version)):
                change[f'body_{index:03}_{j:02}'] = triangles
        frames.append((f'hold_edit_{version:02}', change))
    set_lids = lambda fraction: {f'lid_{index:03}': lid(centers[index], fraction) for index in chosen}
    frames.append(('partial_open', set_lids(opening)))
    visible_changes = {}
    for index in chosen:
        for j, triangles in enumerate(payload(centers[index], layers, hold+1)):
            visible_changes[f'body_{index:03}_{j:02}'] = triangles
    frames.extend([('visible_edit', visible_changes), ('close_again', set_lids(opening if always_open else 0.))])
    hidden_changes = {}
    for index in chosen:
        for j, triangles in enumerate(payload(centers[index], layers, hold+2)):
            hidden_changes[f'body_{index:03}_{j:02}'] = triangles
    frames.extend([('hidden_edit_again', hidden_changes), ('reopen', set_lids(opening)),
                   ('open_all_changed', set_lids(1.))])
    rng = np.random.default_rng(seed)
    origins, directions = [], []
    n = rays_per_pocket_side
    direction = np.array([.28, .09, -1.] if oblique else [0., 0., -1.])
    direction /= np.linalg.norm(direction)
    for cx, cy in centers:
        uv = (np.indices((n, n)).reshape(2, -1).T+rng.uniform(.15, .85, (n*n, 2)))/n
        mouth = np.column_stack(((uv-.5)*1.96+[cx, cy], np.full(n*n, .9)))
        origins.extend(mouth-direction*(3.1/-direction[2]))
        directions.extend(np.tile(direction, (n*n, 1)))
    return PocketExperiment(np.concatenate(static), part_specs,
                            np.array([[-1.1, -1.1, -1.31], [3*(grid-1)+1.1, 3*(grid-1)+1.1, 2.92]]),
                            frames, np.asarray(origins), np.asarray(directions), lid_count, groups,
                            dict(grid=grid, pockets=grid*grid, layers=layers, changed=changed,
                                 changed_fraction=changed/(grid*grid), hold=hold, opening=opening,
                                 oblique=oblique, always_open=always_open, seed=seed,
                                 bodies=grid*grid*16*layers, interior_faces=grid*grid*16*layers*48,
                                 rays=len(origins)))


class PocketScene(ConfinedScene):
    def __init__(self, static, parts, envelope, lid_count, groups=None, original_specs=None):
        super().__init__(static, parts, envelope)
        self.lid_count = lid_count
        self.groups = groups
        self.original_names = [p['name'] for p in original_specs]
        self.original_specs = {p['name']: (p['triangles'].shape, checked_box(p['envelope'])) for p in original_specs}
        self.supports = SupportTree(np.array([p.envelope for p in parts[lid_count:]]))
        self.lid_tree = None
        self.lid_version = None
        self.map_cache = None

    def update(self, changes):
        if not changes:
            return
        if set(changes)-set(self.original_specs):
            raise ValueError('Unknown dynamic owner')
        names = [n for n in self.original_names if n in changes]
        for name in names:
            if np.shape(changes[name]) != self.original_specs[name][0]:
                raise ValueError('Fixed topology required')
        group_names = []
        if self.groups is not None:
            for group in self.groups:
                members = [self.original_names[j] for j in group]
                touched = [n in changes for n in members]
                if any(touched) and not all(touched):
                    raise ValueError('Grouped update requires every child in the group')
                group_names.append(members if all(touched) else [])
        # One atomic geometry validation/storage pass for EVERY method. Grouped
        # methods obey the original leaf bounds too, not just the larger pocket.
        triangles = checked_triangles(np.concatenate([changes[n] for n in names]))
        counts = np.array([len(changes[n]) for n in names])
        lower = np.repeat(np.array([self.original_specs[n][1][0] for n in names]), counts, axis=0)
        upper = np.repeat(np.array([self.original_specs[n][1][1] for n in names]), counts, axis=0)
        if np.any(triangles < lower[:, None, :]) or np.any(triangles > upper[:, None, :]):
            raise ValueError('Geometry leaves original child envelope')
        contains(self.envelope, triangles)
        offsets = np.r_[0, np.cumsum(counts)]
        slices = {n: (offsets[i], offsets[i+1]) for i, n in enumerate(names)}
        prepared = {n: triangles[a:b] for n, (a, b) in slices.items()}
        if self.groups is not None:
            prepared = {n: t for n, t in prepared.items() if n.startswith('lid_')}
            for i, members in enumerate(group_names):
                if members:
                    prepared[f'group_{i:03}'] = triangles[slices[members[0]][0]:slices[members[-1]][1]]
        by_name = {p.name: p for p in self.parts}
        for name, triangles in prepared.items():
            p = by_name[name]
            if not np.array_equal(p.triangles, triangles):
                p.triangles = triangles
                p.epoch += 1
                self.version += 1

    def prepare_all(self):
        stats = empty_stats()
        for p in self.parts[self.lid_count:]:
            p.prepare(stats)
        return stats

    def query(self, origins, directions, route='certified', flat=False, reuse_map=True, candidate_policy='current'):
        if candidate_policy not in ('current', 'envelope'):
            raise ValueError(candidate_policy)
        if route == 'full':
            return super().query(origins, directions, 'full')
        if route not in ['eager', 'certified']:
            raise ValueError(route)
        o, d = np.ascontiguousarray(origins), np.ascontiguousarray(directions)
        if o.shape != d.shape or o.ndim != 2 or o.shape[1] != 3 or not np.all(np.isfinite(o)) or not np.all(np.isfinite(d)) or not np.allclose(np.linalg.norm(d, axis=1), 1, atol=1e-12, rtol=0):
            raise ValueError('Finite rays with unit directions required')
        stats = empty_stats()
        stats.update(map_reused=0, candidate_map_reused=0, map_node_visits=0, active_owners=0, dirty_owners=0,
                     deferred_dirty_owners=0, lid_refit_faces=0, lid_build_faces=0, lid_prepare_s=0.)
        payloads = self.parts[self.lid_count:]
        stats['dirty_owners'] = sum(p.epoch != p.prepared_epoch for p in payloads)
        if route == 'eager':
            for p in payloads:
                p.prepare(stats)
        version = tuple(p.epoch for p in self.parts[:self.lid_count])
        cache = self.map_cache
        map_key = (flat, candidate_policy)
        if reuse_map and cache is not None and cache[0] == version and cache[1] == map_key and np.array_equal(o, cache[2]) and np.array_equal(d, cache[3]):
            h, tau, offsets, rays = cache[4].copy(), cache[5].copy(), cache[6], cache[7]
            stats['map_reused'] = 1
            stats['candidate_map_reused'] = 1
        else:
            static = self._source_cache
            if static is not None and np.array_equal(o, static[0]) and np.array_equal(d, static[1]):
                h, tau = static[2].copy(), static[3].copy()
            else:
                if self.static_tree is None:
                    started = perf_counter()
                    self.static_tree = MeshTree(self.static)
                    stats['static_prepare_s'] = perf_counter()-started
                    stats['static_built_faces'] = len(self.static)
                h, tau, b, t = self.static_tree.trace(o, d, np.full(len(o), np.inf))
                stats['box_tests'] += b; stats['triangle_tests'] += t
                self._source_cache = o.copy(), d.copy(), h.copy(), tau.copy()
            if version != self.lid_version:
                started = perf_counter()
                lids = np.concatenate([p.triangles for p in self.parts[:self.lid_count]])
                if self.lid_tree is None:
                    self.lid_tree = MeshTree(lids)
                    stats['lid_build_faces'] = len(lids)
                else:
                    self.lid_tree.refit(lids)
                    stats['lid_refit_faces'] = len(lids)
                stats['lid_prepare_s'] = perf_counter()-started
                self.lid_version = version
            nh, nt, b, t = self.lid_tree.trace(o, d, tau)
            nh = np.where(nh >= 0, nh+len(self.static), -1)
            take = (nh >= 0) & ((nt < tau) | ((nt == tau) & ((h < 0) | (nh < h))))
            h[take], tau[take] = nh[take], nt[take]
            stats['box_tests'] += b; stats['triangle_tests'] += t
            # Envelope candidates do not depend on current lids. Preserve that
            # valid cache even when current hit distances must be refreshed.
            if (reuse_map and candidate_policy == 'envelope' and cache is not None
                    and cache[1] == map_key and np.array_equal(o, cache[2]) and np.array_equal(d, cache[3])):
                offsets, rays = cache[6], cache[7]
                stats['candidate_map_reused'] = 1
            else:
                started = perf_counter()
                tree = self.supports
                candidate_limits = tau if candidate_policy == 'current' else np.full(len(o), np.inf)
                offsets, rays, visits = candidates(o, d, candidate_limits, tree.lo, tree.hi, tree.end,
                                                   tree.owner, len(payloads), flat)
                stats['certificate_s'] = perf_counter()-started
                stats['map_node_visits'] = visits
            self.map_cache = (version, map_key, o.copy(), d.copy(), h.copy(), tau.copy(), offsets, rays)
        stats['candidate_pairs'] = len(rays)
        stats['candidate_policy'] = candidate_policy
        for j in np.flatnonzero(np.diff(offsets)):
            p = payloads[j]
            p.prepare(stats)
            idx = rays[offsets[j]:offsets[j+1]]
            nh, nt, b, t = p.tree.trace(o[idx], d[idx], tau[idx])
            nh = np.where(nh >= 0, nh+p.offset, -1)
            take = (nh >= 0) & ((nt < tau[idx]) | ((nt == tau[idx]) & ((h[idx] < 0) | (nh < h[idx]))))
            h[idx[take]], tau[idx[take]] = nh[take], nt[take]
            stats['active_owners'] += 1
            stats['dynamic_queries'] += len(idx)
            stats['box_tests'] += b; stats['triangle_tests'] += t
        stats['deferred_dirty_owners'] = sum(p.epoch != p.prepared_epoch for p in payloads)
        return HitState(h, tau, stats)
