"""Conservative reflected-ray packets, preserving every physical ray.

A packet's origins lie in a ball and its normalized directions in a cone.
Inflating a target sphere by the origin-ball radius makes cone rejection
conservative. Node bounding spheres provide a shared BVH broad phase; actual
ray/sphere intersections and reflection arithmetic remain unchanged.
"""
import time
import numpy as np
from numba import njit
from .ehss_guarded_intrinsic_response import nearest_world, reflect
from .ehss_reference import entry_distance
from .ehss_state import EHSSResult


@njit(cache=True)
def cone_possible(origin, axis, cosine, sine, spread, center, radius):
    w = center-origin
    length2 = np.dot(w, w)
    expanded = radius+spread
    if length2 <= expanded*expanded: return True
    axial = np.dot(w, axis)
    radial = np.sqrt(max(0., length2-axial*axial))
    if axial >= 0. and radial*cosine <= axial*sine: return True
    if axial*cosine+radial*sine < 0.: return False
    return radial*cosine-axial*sine <= expanded


@njit(cache=False)
def collect(node, origin, axis, cosine, sine, spread, last,
            node_centers, node_radii, left, right, leaf_group, offsets, ids,
            centers, radii, scratch, count, counters):
    counters[0] += 1
    if not cone_possible(origin, axis, cosine, sine, spread, node_centers[node], node_radii[node]):
        return count
    group = leaf_group[node]
    if group >= 0:
        for j in range(offsets[group], offsets[group+1]):
            atom = ids[j]
            if atom == last: continue
            counters[1] += 1
            if cone_possible(origin, axis, cosine, sine, spread, centers[atom], radii[atom]):
                scratch[count] = atom
                count += 1
    else:
        count = collect(left[node], origin, axis, cosine, sine, spread, last, node_centers, node_radii,
            left, right, leaf_group, offsets, ids, centers, radii, scratch, count, counters)
        count = collect(right[node], origin, axis, cosine, sine, spread, last, node_centers, node_radii,
            left, right, leaf_group, offsets, ids, centers, radii, scratch, count, counters)
    return count


@njit(cache=False)
def candidate_lists(origins, axes, cosines, sines, spreads, lasts,
                    node_centers, node_radii, left, right, leaf_group, offsets, ids, centers, radii):
    scratch = np.empty(len(radii), dtype=np.int64)
    starts = [0]
    flat = []
    counters = np.zeros(2, dtype=np.int64)
    for g in range(len(origins)):
        count = collect(0, origins[g], axes[g], cosines[g], sines[g], spreads[g], lasts[g],
            node_centers, node_radii, left, right, leaf_group, offsets, ids, centers, radii, scratch, 0, counters)
        for j in range(count): flat.append(scratch[j])
        starts.append(len(flat))
    return np.asarray(starts, dtype=np.int64), np.asarray(flat, dtype=np.int64), counters


@njit(cache=False)
def query_packets(points, directions, lasts, packet_starts, packet_rays, candidate_starts, candidate_ids,
                  fallback, centers, radii, lo, hi, left, right, end, leaf_group, offsets, ids):
    hits = np.full(len(points), -1, dtype=np.int64)
    distances = np.full(len(points), np.inf)
    counters = np.zeros(2, dtype=np.int64)
    packet_tests = 0
    for g in range(len(packet_starts)-1):
        for j in range(packet_starts[g], packet_starts[g+1]):
            ray = packet_rays[j]
            for k in range(candidate_starts[g], candidate_starts[g+1]):
                atom = candidate_ids[k]
                packet_tests += 1
                distance = entry_distance(points[ray], directions[ray], centers[atom], radii[atom])
                if np.isfinite(distance) and (distance < distances[ray] or (distance == distances[ray] and atom < hits[ray])):
                    hits[ray], distances[ray] = atom, distance
    for ray in fallback:
        hits[ray], distances[ray] = nearest_world(0, points[ray], directions[ray], lasts[ray], np.inf, -1,
            lo, hi, left, right, end, leaf_group, offsets, ids, centers, radii, counters)
    return hits, distances, int(packet_tests+counters[1]), counters[0]


def group_reflections(points, directions, lasts, centers, radii, degrees=30., minimum=4):
    if not 0. < degrees < 90. or type(minimum) is not int or minimum < 2:
        raise ValueError('Cone angle must be in (0,90); minimum packet size >=2')
    starts, rays, fallback, origins, axes, widths, spreads, atoms = [0], [], [], [], [], [], [], []
    pad = 1e-9*(1.+float(np.abs(centers).max()))
    maximum = np.deg2rad(degrees)
    splits = 0

    def visit(selected, last):
        nonlocal splits
        if len(selected) < minimum:
            fallback.extend(selected)
            return
        p, d = points[selected], directions[selected]
        origin = p.mean(axis=0)
        spread = float(np.linalg.norm(p-origin, axis=1).max())+pad
        axis = d.mean(axis=0)
        norm = np.linalg.norm(axis)
        if norm > 1e-12:
            axis /= norm
            dots = (d/np.linalg.norm(d, axis=1)[:, None])@axis
            angle = float(np.arccos(np.clip(dots.min(), -1., 1.)))+1e-9
        else:
            angle = np.pi
        if angle <= maximum and spread <= .5*radii[last]+2*pad:
            rays.extend(selected)
            starts.append(len(rays))
            origins.append(origin)
            axes.append(axis)
            widths.append(angle)
            spreads.append(spread)
            atoms.append(last)
            return
        features = np.column_stack(((p-centers[last])/radii[last], d))
        coordinate = int(np.argmax(np.ptp(features, axis=0)))
        ordered = selected[np.argsort(features[:, coordinate], kind='stable')]
        middle = len(ordered)//2
        splits += 1
        visit(ordered[:middle], last)
        visit(ordered[middle:], last)

    order = np.argsort(lasts, kind='stable')
    unique, first, counts = np.unique(lasts[order], return_index=True, return_counts=True)
    for last, start, count in zip(unique, first, counts):
        selected = order[start:start+count]
        if last < 0: fallback.extend(selected)
        else: visit(selected, int(last))
    return dict(starts=np.array(starts, dtype=np.int64), rays=np.array(rays, dtype=np.int64),
        fallback=np.array(fallback, dtype=np.int64), origins=np.asarray(origins).reshape(-1, 3),
        axes=np.asarray(axes).reshape(-1, 3), widths=np.asarray(widths),
        spreads=np.asarray(spreads), lasts=np.asarray(atoms, dtype=np.int64), splits=splits)


@njit(cache=True)
def advance(points, directions, lasts, counts, history, escaped, unresolved, active, hits, distances, cap, centers, radii):
    keep = []
    for i in range(len(active)):
        ray, atom = active[i], hits[i]
        if atom < 0:
            escaped[ray] = True
        elif counts[ray] >= cap:
            unresolved[ray] = True
        else:
            reflect(points[ray], directions[ray], atom, distances[i], centers, radii)
            history[ray, counts[ray]] = atom
            counts[ray] += 1
            lasts[ray] = atom
            keep.append(ray)
    return np.asarray(keep, dtype=np.int64)


class ConePacketTransport:
    def __init__(self, model):
        tick = time.perf_counter()
        self.model = model
        t = model.tree
        self.node_centers = (t.lo+t.hi)*.5
        self.node_radii = np.linalg.norm(t.hi-t.lo, axis=1)*.5
        self.prepare_s = time.perf_counter()-tick

    def next_hits(self, points, directions, lasts, degrees=30., minimum=4, enabled=True):
        t, s = self.model.tree, self.model.spheres
        tick = time.perf_counter()
        if enabled:
            groups = group_reflections(points, directions, lasts, s.centers, s.radii, degrees, minimum)
        else:
            groups = dict(starts=np.zeros(1, dtype=np.int64), rays=np.empty(0, dtype=np.int64),
                fallback=np.arange(len(points), dtype=np.int64), origins=np.empty((0,3)), axes=np.empty((0,3)),
                widths=np.empty(0), spreads=np.empty(0), lasts=np.empty(0, dtype=np.int64), splits=0)
        grouping_s = time.perf_counter()-tick
        tick = time.perf_counter()
        starts, ids, checks = candidate_lists(groups['origins'], groups['axes'], np.cos(groups['widths']),
            np.sin(groups['widths']), groups['spreads'], groups['lasts'], self.node_centers, self.node_radii,
            t.left, t.right, t.leaf_group, t.offsets, t.ids, s.centers, s.radii)
        candidate_s = time.perf_counter()-tick
        tick = time.perf_counter()
        hits, distances, sphere_tests, box_tests = query_packets(points, directions, lasts,
            groups['starts'], groups['rays'], starts, ids, groups['fallback'], s.centers, s.radii,
            t.lo, t.hi, t.left, t.right, t.end, t.leaf_group, t.offsets, t.ids)
        intersection_s = time.perf_counter()-tick
        return hits, distances, dict(grouping_s=grouping_s, candidate_s=candidate_s, intersection_s=intersection_s,
            sphere_tests=sphere_tests, box_tests=int(box_tests), node_cone_tests=int(checks[0]),
            atom_cone_tests=int(checks[1]), packets=len(groups['lasts']), packet_rays=len(groups['rays']),
            fallback_rays=len(groups['fallback']), splits=groups['splits'], candidate_ids=len(ids),
            empty_packets=int(np.count_nonzero(np.diff(starts) == 0)),
            max_candidate_bytes=ids.nbytes+starts.nbytes, max_cone_degrees=float(np.rad2deg(groups['widths']).max(initial=0.)))

    def trace(self, source, cap=64, enabled=True, degrees=30., minimum=4):
        if type(cap) is not int or cap < 1 or np.any(source.initial_bounces > 1):
            raise ValueError('Positive cap and zero/one initial collision required')
        if np.any(source.last_atom < -1) or np.any(source.last_atom >= len(self.model.spheres.radii)):
            raise ValueError('Invalid initial physical identity')
        tick = time.perf_counter()
        points, directions = source.origins.copy(), source.directions.copy()
        lasts, counts = source.last_atom.copy(), source.initial_bounces.copy()
        history = np.full((len(points), cap), -1, dtype=np.int32)
        initial = counts == 1
        history[initial, 0] = lasts[initial]
        escaped = np.zeros(len(points), dtype=bool)
        unresolved = escaped.copy()
        active = np.arange(len(points), dtype=np.int64)
        metrics = dict(grouping_s=0., candidate_s=0., intersection_s=0., reflection_s=0.,
            sphere_tests=0, box_tests=0, node_cone_tests=0, atom_cone_tests=0, packets=0, packet_rays=0,
            fallback_rays=0, splits=0, candidate_ids=0, empty_packets=0, max_candidate_bytes=0,
            max_cone_degrees=0., waves=0)
        while len(active):
            hits, distances, current = self.next_hits(points[active], directions[active], lasts[active], degrees, minimum, enabled)
            for key, value in current.items():
                metrics[key] = max(metrics[key], value) if key.startswith('max_') else metrics[key]+value
            start = time.perf_counter()
            active = advance(points, directions, lasts, counts, history, escaped, unresolved, active, hits, distances,
                cap, self.model.spheres.centers, self.model.spheres.radii)
            metrics['reflection_s'] += time.perf_counter()-start
            metrics['waves'] += 1
        metrics.update(query_s=time.perf_counter()-tick, physical_direction_approximation=False,
            angular_merge=False, candidate_cover='origin ball plus measured outgoing direction cone')
        return EHSSResult(points, directions, escaped, unresolved, counts, history, source, metrics)
