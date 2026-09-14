"""Persistent, split-only reflected packets with an exact empirical distribution.

Membership survives scattering. A packet branches by its next collider and is
geometrically split only when its measured origin/direction envelope is too wide.
Individual positions, directions, incoming directions and weights are retained.
This module does not approximate or compress the distribution into moments.
"""
import time
import numpy as np
from numba import njit
from .ehss_cone_packets import ConePacketTransport, candidate_lists, query_packets, advance
from .ehss_state import EHSSResult


@njit(cache=True)
def partition_packets(points, directions, lasts, parents, centers, radii, maximum, minimum, persistent):
    n = len(points)
    starts = np.empty(n+1, dtype=np.int64)
    rays = np.empty(n, dtype=np.int64)
    fallback = np.empty(n, dtype=np.int64)
    origins, axes = np.empty((n, 3)), np.empty((n, 3))
    widths, spreads = np.empty(n), np.empty(n)
    atoms, labels = np.empty(n, dtype=np.int64), np.full(n, -1, dtype=np.int64)
    packet_count = ray_count = fallback_count = splits = collider_branches = retained = 0
    starts[0] = 0
    pad = 1e-9*(1.+np.max(np.abs(centers)))
    keys = np.empty(n, dtype=np.int64)
    for i in range(n):
        if lasts[i] < 0:
            keys[i] = -2  # First free flight stays eligible after its first hit.
        elif persistent and parents[i] < 0:
            keys[i] = -1  # Retired small packets are never globally regrouped.
        else:
            parent = parents[i] if persistent else 0
            keys[i] = parent*(len(radii)+1)+lasts[i]+1
    order = np.argsort(keys, kind='mergesort')
    cursor = 0
    previous_parent = -2
    while cursor < n:
        stop = cursor+1
        key = keys[order[cursor]]
        while stop < n and keys[order[stop]] == key: stop += 1
        if key < 0:
            for j in range(cursor, stop):
                ray = order[j]
                fallback[fallback_count] = ray
                fallback_count += 1
                if key == -2: labels[ray] = n  # Reserved seed, distinct from accepted packets.
            cursor = stop
            continue
        last = lasts[order[cursor]]
        parent = parents[order[cursor]] if persistent else 0
        if parent == previous_parent: collider_branches += 1
        previous_parent = parent
        stack_lo, stack_hi = np.empty(stop-cursor, dtype=np.int64), np.empty(stop-cursor, dtype=np.int64)
        stack_lo[0], stack_hi[0], pending = cursor, stop, 1
        while pending:
            pending -= 1
            low, high = stack_lo[pending], stack_hi[pending]
            size = high-low
            if size < minimum:
                for j in range(low, high):
                    fallback[fallback_count] = order[j]
                    fallback_count += 1
                continue
            origin, axis = np.zeros(3), np.zeros(3)
            for j in range(low, high):
                origin += points[order[j]]
                axis += directions[order[j]]
            origin /= size
            norm = np.sqrt(np.dot(axis, axis))
            spread, mindot = 0., 1.
            angle = np.pi
            if norm > 1e-12: axis /= norm
            for j in range(low, high):
                ray = order[j]
                offset = points[ray]-origin
                spread = max(spread, np.sqrt(np.dot(offset, offset)))
                if norm > 1e-12:
                    d = directions[ray]
                    mindot = min(mindot, np.dot(d, axis)/np.sqrt(np.dot(d,d)))
            spread += pad
            if norm > 1e-12: angle = np.arccos(max(-1., min(1., mindot)))+1e-9
            if angle <= maximum and spread <= .5*radii[last]+2*pad:
                for j in range(low, high):
                    ray = order[j]
                    rays[ray_count] = ray
                    ray_count += 1
                    labels[ray] = packet_count
                origins[packet_count], axes[packet_count] = origin, axis
                widths[packet_count], spreads[packet_count], atoms[packet_count] = angle, spread, last
                packet_count += 1
                starts[packet_count] = ray_count
                if low == cursor and high == stop: retained += 1
                continue
            # Split measured joint position/direction support; physical values stay unchanged.
            features = np.empty((size, 6))
            mins, maxs = np.full(6, np.inf), np.full(6, -np.inf)
            for j in range(size):
                ray = order[low+j]
                for k in range(3):
                    features[j,k] = (points[ray,k]-centers[last,k])/radii[last]
                    features[j,k+3] = directions[ray,k]
                for k in range(6):
                    mins[k] = min(mins[k],features[j,k])
                    maxs[k] = max(maxs[k],features[j,k])
            coordinate = np.argmax(maxs-mins)
            perm = np.argsort(features[:,coordinate], kind='mergesort')
            selected = order[low:high].copy()
            for j in range(size): order[low+j] = selected[perm[j]]
            middle = low+size//2
            stack_lo[pending], stack_hi[pending] = middle, high
            stack_lo[pending+1], stack_hi[pending+1] = low, middle
            pending += 2
            splits += 1
        cursor = stop
    return (starts[:packet_count+1], rays[:ray_count], fallback[:fallback_count],
        origins[:packet_count], axes[:packet_count], widths[:packet_count], spreads[:packet_count],
        atoms[:packet_count], labels, splits, collider_branches, retained)


class PersistentConeTransport(ConePacketTransport):
    def trace(self, source, cap=64, degrees=30., minimum=4, persistent=True):
        if type(cap) is not int or cap < 1 or np.any(source.initial_bounces > 1):
            raise ValueError('Positive cap and zero/one initial collision required')
        if not 0. < degrees < 90. or type(minimum) is not int or minimum < 2:
            raise ValueError('Cone angle must be in (0,90); minimum packet size >=2')
        t, s = self.model.tree, self.model.spheres
        if np.any(source.last_atom < -1) or np.any(source.last_atom >= len(s.radii)):
            raise ValueError('Invalid initial physical identity')
        tick = time.perf_counter()
        points, directions = source.origins.copy(), source.directions.copy()
        lasts, counts = source.last_atom.copy(), source.initial_bounces.copy()
        labels = np.zeros(len(points), dtype=np.int64)
        history = np.full((len(points), cap), -1, dtype=np.int32)
        initial = counts == 1
        history[initial, 0] = lasts[initial]
        escaped = np.zeros(len(points), dtype=bool)
        unresolved = escaped.copy()
        active = np.arange(len(points), dtype=np.int64)
        metrics = dict(grouping_s=0., candidate_s=0., intersection_s=0., reflection_s=0.,
            sphere_tests=0, box_tests=0, node_cone_tests=0, atom_cone_tests=0, packets=0, packet_rays=0,
            fallback_rays=0, splits=0, collider_branches=0, geometrically_retained_groups=0,
            candidate_ids=0, empty_packets=0, max_candidate_bytes=0, max_cone_degrees=0., waves=0)
        while len(active):
            p, d, last = points[active], directions[active], lasts[active]
            start = time.perf_counter()
            groups = partition_packets(p, d, last, labels[active], s.centers, s.radii,
                np.deg2rad(degrees), minimum, persistent)
            ps, pr, fallback, origins, axes, widths, spreads, atoms, new_labels, splits, branches, retained = groups
            labels[active] = new_labels
            metrics['grouping_s'] += time.perf_counter()-start
            start = time.perf_counter()
            cs, ids, checks = candidate_lists(origins, axes, np.cos(widths), np.sin(widths), spreads,
                atoms, self.node_centers, self.node_radii, t.left, t.right, t.leaf_group, t.offsets, t.ids,
                s.centers, s.radii)
            metrics['candidate_s'] += time.perf_counter()-start
            start = time.perf_counter()
            hits, distances, sphere_tests, box_tests = query_packets(p, d, last, ps, pr, cs, ids, fallback,
                s.centers, s.radii, t.lo, t.hi, t.left, t.right, t.end, t.leaf_group, t.offsets, t.ids)
            metrics['intersection_s'] += time.perf_counter()-start
            start = time.perf_counter()
            active = advance(points, directions, lasts, counts, history, escaped, unresolved,
                active, hits, distances, cap, s.centers, s.radii)
            metrics['reflection_s'] += time.perf_counter()-start
            for key, val in dict(sphere_tests=int(sphere_tests), box_tests=int(box_tests),
                node_cone_tests=int(checks[0]), atom_cone_tests=int(checks[1]), packets=len(atoms),
                packet_rays=len(pr), fallback_rays=len(fallback), splits=splits, collider_branches=branches,
                geometrically_retained_groups=retained, candidate_ids=len(ids),
                empty_packets=int(np.count_nonzero(np.diff(cs)==0))).items(): metrics[key] += val
            metrics['max_candidate_bytes'] = max(metrics['max_candidate_bytes'], ids.nbytes+cs.nbytes)
            metrics['max_cone_degrees'] = max(metrics['max_cone_degrees'], float(np.rad2deg(widths).max(initial=0.)))
            metrics['waves'] += 1
        metrics.update(query_s=time.perf_counter()-tick, persistent_membership=persistent,
            distribution='Exact weighted empirical joint position/direction distribution; incoming retained',
            physical_direction_approximation=False, angular_merge=False, distribution_compressed=False)
        return EHSSResult(points, directions, escaped, unresolved, counts, history, source, metrics)
