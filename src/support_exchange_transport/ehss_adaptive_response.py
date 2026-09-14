"""Sample-adaptive local boundary responses with explicit exact fallback.

THROUGH cells have a conservative whole-cell line-exclusion bound. APPROXIMATE
responses pass a probe test, not a continuous error certificate. UNKNOWN cells
and interior-anchor requests use geometry. This is an experimental local atlas
with exact sphere routing, not the compiled G/T field or a multilevel compiler.
"""
import heapq
import time
import numpy as np
from numba import njit

from .ehss_interface_transport import apply_local_response, sphere_interval, next_interface
from .ehss_reference import nearest_all

THROUGH, APPROXIMATE, UNKNOWN = 0, 1, 2


def chart_rays(coordinates):
    u = np.asarray(coordinates)
    theta, phi, alpha = u[:, 0] * np.pi, u[:, 1] * 2 * np.pi, u[:, 3] * 2 * np.pi
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    direction = np.c_[st * cp, st * sp, ct]
    tangent = np.c_[-sp, cp, np.zeros(len(u))]
    binormal = np.c_[-ct * cp, -ct * sp, st]
    transverse = np.sqrt(u[:, 2])[:, None] * (np.cos(alpha)[:, None] * tangent + np.sin(alpha)[:, None] * binormal)
    return transverse - np.sqrt(1 - u[:, 2])[:, None] * direction, direction, transverse


def ray_chart(origins, directions):
    d = np.asarray(directions)
    theta = np.arccos(np.clip(d[:, 2], -1., 1.))
    phi = np.mod(np.arctan2(d[:, 1], d[:, 0]), 2 * np.pi)
    tangent = np.c_[-np.sin(phi), np.cos(phi), np.zeros(len(phi))]
    binormal = np.c_[-np.cos(theta) * np.cos(phi), -np.cos(theta) * np.sin(phi), np.sin(theta)]
    x, y = np.sum(origins * tangent, axis=1), np.sum(origins * binormal, axis=1)
    return np.c_[theta / np.pi, phi / (2 * np.pi), np.clip(x*x + y*y, 0., 1.),
                 np.mod(np.arctan2(y, x), 2 * np.pi) / (2 * np.pi)]


def certified_through(low, high, centers, radii):
    """Sufficient exclusion of every infinite line in a 4D chart cell."""
    guard = 1024 * np.finfo(float).eps * (1 + np.max(np.linalg.norm(centers, axis=1) + radii))
    rmin, rmax = np.sqrt(low[2]), np.sqrt(high[2])
    if rmin > np.max(np.linalg.norm(centers, axis=1) + radii) + guard:
        return True
    midpoint = (low + high) / 2
    _, direction, transverse = chart_rays(midpoint[None, :])
    rho = np.sqrt(midpoint[2])
    dt, dp, da = (high - low)[[0, 1, 3]] * np.array([np.pi, 2*np.pi, 2*np.pi]) / 2
    # A ray frame is a product of rotations. Their angular differences bound
    # transverse-point motion; the rank-one direction projector changes by at
    # most min(1, angle). This bound also covers chart seams and polar limits.
    point_error = min(2*rmax, max(rho-rmin, rmax-rho) + rmax*(dt+dp+da))
    error = point_error + np.linalg.norm(centers, axis=1) * min(1., dt+dp)
    perpendicular = centers - transverse - (centers @ direction[0])[:, None] * direction
    return bool(np.all(np.linalg.norm(perpendicular, axis=1) > radii + error + guard))


@njit(cache=True)
def probe_responses(points, directions, centers, radii, cap):
    exits = points.copy()
    outgoing = directions.copy()
    counts = np.zeros(len(points), dtype=np.int64)
    stopped = np.zeros(len(points), dtype=np.bool_)
    itinerary = np.full((len(points), cap), -1, dtype=np.int32)
    ids = np.arange(len(radii))
    tests = 0
    for i in range(len(points)):
        _, count, stop, nt, _ = apply_local_response(exits[i], outgoing[i], -1, 0, itinerary[i], cap, ids, centers, radii)
        tests += nt
        counts[i], stopped[i] = count, stop
        if not stop:
            _, far = sphere_interval(exits[i], outgoing[i], np.zeros(3), 1.)
            exits[i] += far * outgoing[i]
    return exits, outgoing, counts, stopped, itinerary, tests


@njit(cache=True)
def initial_response_with_paths(origins, directions, last_atoms, initial_counts, centers, radii, groups,
                                packed, offsets, envelope_centers, envelope_radii, cap):
    exits, outgoing, counts = origins.copy(), directions.copy(), initial_counts.copy()
    owners = np.full(len(origins), -1, dtype=np.int64)
    paths = np.full((len(origins), cap), -1, dtype=np.int32)
    tests = 0
    for i in range(len(origins)):
        last = last_atoms[i]
        if initial_counts[i] == 1:
            group = groups[last]
            paths[i, 0] = last
        else:
            atom, _, nt = nearest_all(exits[i], outgoing[i], last, centers, radii)
            tests += nt
            if atom < 0:
                continue
            group = groups[atom]
        _, counts[i], stop, nt, _ = apply_local_response(exits[i], outgoing[i], last, counts[i], paths[i], cap,
            packed[offsets[group]:offsets[group+1]], centers, radii)
        tests += nt
        owners[i] = -2 if stop else group
        if not stop:
            _, far = sphere_interval(exits[i], outgoing[i], envelope_centers[group], envelope_radii[group])
            exits[i] += far*outgoing[i]
    return exits, outgoing, counts, owners, paths, tests


class AdaptiveLocalResponse:
    def __init__(self, centers, radii, leaf_budget=2048, tolerance=.15, strategy="adaptive", cap=32, region=None):
        if strategy not in ("adaptive", "uniform") or not isinstance(leaf_budget, int) or leaf_budget < 1:
            raise ValueError("Positive leaf budget and adaptive/uniform strategy required")
        if not np.isfinite(tolerance) or tolerance <= 0 or not isinstance(cap, int) or cap < 1:
            raise ValueError("Positive response tolerance and cap required")
        self.centers = np.ascontiguousarray(centers, dtype=float).copy()
        self.radii = np.ascontiguousarray(radii, dtype=float).copy()
        if (self.centers.shape != (len(self.radii), 3) or not len(self.radii)
                or not np.isfinite(self.centers).all() or not np.isfinite(self.radii).all()
                or np.any(self.radii <= 0) or np.any(np.linalg.norm(self.centers, axis=1) + self.radii >= 1)):
            raise ValueError("Finite positive atom balls strictly inside the unit envelope required")
        self.cap = cap
        self.region = np.array([np.zeros(4), np.ones(4)] if region is None else region, dtype=float)
        if (self.region.shape != (2, 4) or not np.isfinite(self.region).all() or np.any(self.region[0] < 0)
                or np.any(self.region[1] > 1) or np.any(self.region[0] >= self.region[1])):
            raise ValueError("A nonempty chart region within [0,1]^4 is required")
        self.tolerance, self.strategy = tolerance, strategy
        start = time.perf_counter()
        nodes, queue = [], []
        self.probe_rays = self.probe_tests = 0

        def add(low, high, depth):
            identity = len(nodes)
            center = (low + high) / 2
            node = dict(low=low, high=high, depth=depth, left=-1, axis=-1, split=0.,
                        status=THROUGH, count=0, position=np.zeros(3), direction=np.zeros(3), path=np.full(cap, -1, dtype=np.int32))
            nodes.append(node)
            if certified_through(low, high, self.centers, self.radii):
                return identity
            probes = np.tile(center, (9, 1))
            for axis in range(4):
                probes[1+2*axis, axis] -= .375 * (high[axis] - low[axis])
                probes[2+2*axis, axis] += .375 * (high[axis] - low[axis])
            points, directions, _ = chart_rays(probes)
            positions, outgoing, counts, stopped, paths, tests = self._probe(points, directions, cap)
            self.probe_rays += len(probes)
            self.probe_tests += tests
            same = bool(np.all(paths == paths[0]) and not np.any(stopped))
            spread = max(np.linalg.norm(positions-positions[0], axis=1).max(),
                         np.linalg.norm(outgoing-outgoing[0], axis=1).max())
            accept = same and counts[0] > 0 and spread <= tolerance
            node.update(status=APPROXIMATE if accept else UNKNOWN, count=int(counts[0]),
                        position=positions[0], direction=outgoing[0], path=paths[0])
            if accept and strategy == "adaptive":
                return identity
            # Never turn unobserved collisions into THROUGH: uncertain leaves
            # remaining at the budget limit are evaluated from geometry at query.
            scores = np.zeros(4)
            for axis in range(4):
                a, b = 1+2*axis, 2+2*axis
                scores[axis] = (4. if np.any(paths[a] != paths[b]) or stopped[a] != stopped[b] else 0.)
                scores[axis] += np.linalg.norm(positions[a]-positions[b]) + np.linalg.norm(outgoing[a]-outgoing[b])
            if not np.any(counts):
                extent = np.max(np.linalg.norm(self.centers, axis=1) + self.radii)
                if low[2] < extent*extent < high[2]:
                    axis = 2
                else:
                    axis = int(np.argmax((high-low) * np.array([1., 2., 1., 2.])))
            else:
                axis = int(np.argmax(scores))
            node["axis"] = depth % 4 if strategy == "uniform" else axis
            priority = -depth if strategy == "uniform" else ((4. if not same else 0.) + spread) * np.prod(high-low)**.25
            heapq.heappush(queue, (-priority, identity))
            return identity

        add(self.region[0].copy(), self.region[1].copy(), 0)
        leaves = 1
        while queue and leaves < leaf_budget:
            _, identity = heapq.heappop(queue)
            node = nodes[identity]
            axis = node["axis"]
            split = (node["low"][axis] + node["high"][axis]) / 2
            if split == node["low"][axis] or split == node["high"][axis]:
                continue
            left_high, right_low = node["high"].copy(), node["low"].copy()
            left_high[axis] = right_low[axis] = split
            left = add(node["low"].copy(), left_high, node["depth"]+1)
            right = add(right_low, node["high"].copy(), node["depth"]+1)
            assert right == left + 1
            node.update(left=left, split=split)
            leaves += 1
        self.left = np.array([n["left"] for n in nodes], dtype=np.int64)
        self.axis = np.array([n["axis"] for n in nodes], dtype=np.int64)
        self.split = np.array([n["split"] for n in nodes])
        self.status = np.array([n["status"] for n in nodes], dtype=np.int8)
        self.count = np.array([n["count"] for n in nodes], dtype=np.int64)
        self.positions = np.array([n["position"] for n in nodes])
        self.directions = np.array([n["direction"] for n in nodes])
        self.paths = np.array([n["path"] for n in nodes])
        leaf = self.left < 0
        self.receipt = dict(nodes=len(nodes), leaves=int(leaf.sum()),
                            through_leaves=int(np.count_nonzero(leaf & (self.status == THROUGH))),
                            approximate_leaves=int(np.count_nonzero(leaf & (self.status == APPROXIMATE))),
                            unknown_leaves=int(np.count_nonzero(leaf & (self.status == UNKNOWN))),
                            probe_rays=self.probe_rays, probe_tests=int(self.probe_tests), compile_s=time.perf_counter()-start,
                            payload_bytes=sum(a.nbytes for a in (self.left, self.axis, self.split, self.status, self.count,
                                                               self.positions, self.directions, self.paths, self.region)))
        for a in (self.centers, self.radii, self.left, self.axis, self.split, self.status, self.count, self.positions, self.directions, self.paths, self.region):
            a.setflags(write=False)

    def _probe(self, origins, directions, cap):
        """Overridable evaluator for compiling and resolving an unknown cell."""
        return probe_responses(origins, directions, self.centers, self.radii, cap)

    def locate(self, origins, directions):
        coordinates = ray_chart(origins, directions)
        guard = 256*np.finfo(float).eps
        if np.any(coordinates < self.region[0]-guard) or np.any(coordinates > self.region[1]+guard):
            raise ValueError("Requested ray lies outside this local chart region")
        nodes = np.zeros(len(coordinates), dtype=np.int64)
        pending = np.arange(len(nodes))
        while len(pending):
            current = nodes[pending]
            pending = pending[self.left[current] >= 0]
            current = nodes[pending]
            nodes[pending] = self.left[current] + (coordinates[pending, self.axis[current]] >= self.split[current])
        return nodes

    def apply(self, origins, directions, remaining):
        ids = self.locate(origins, directions)
        counts = np.zeros(len(ids), dtype=np.int64)
        stopped = np.zeros(len(ids), dtype=bool)
        paths = np.full((len(ids), int(remaining.max(initial=0))), -1, dtype=np.int32)
        positions, outgoing = origins.copy(), directions.copy()
        through = self.status[ids] == THROUGH
        # An interior anchor can be beyond a collision on the full incoming
        # chord. It must not replay the response of an earlier entry point.
        eligible = (np.sum(origins*origins, axis=1) >= 1. - 1e-12) & (np.sum(origins*directions, axis=1) < 0.)
        accept = (self.status[ids] == APPROXIMATE) & eligible & (self.count[ids] <= remaining)
        counts[accept] = self.count[ids[accept]]
        positions[accept] = self.positions[ids[accept]]
        outgoing[accept] = self.directions[ids[accept]]
        width = min(self.cap, paths.shape[1])
        paths[accept, :width] = self.paths[ids[accept], :width]
        fallback = ~(through | accept)
        tests = 0
        for cap in np.unique(remaining[fallback]):
            use = np.flatnonzero(fallback & (remaining == cap))
            p, d, c, s, local_paths, nt = self._probe(origins[use], directions[use], int(cap))
            positions[use], outgoing[use], counts[use], stopped[use] = p, d, c, s
            paths[use, :int(cap)] = local_paths
            tests += nt
        return positions, outgoing, counts, stopped, dict(through=int(through.sum()), approximate=int(accept.sum()),
                                                         fallback=int(fallback.sum()), primitive_tests=int(tests), collider_ids=paths)


@njit(cache=True)
def next_interfaces(origins, directions, cursors, completed, centers, radii):
    groups = np.full(len(origins), -1, dtype=np.int64)
    fars = np.zeros(len(origins))
    for i in range(len(origins)):
        group, _, _, _ = next_interface(origins[i], directions[i], cursors[i], completed[i], centers, radii)
        groups[i] = group
        if group >= 0:
            _, fars[i] = sphere_interval(origins[i], directions[i], centers[group], radii[group])
    return groups, fars


class AdaptiveInterfaceTransport:
    def __init__(self, scene, **options):
        self.scene = scene
        self.maps, self.library = [], {}
        start = time.perf_counter()
        for group in range(len(scene.radii)):
            owned = scene.groups == group
            centers = np.ascontiguousarray((scene.spheres.centers[owned]-scene.centers[group])/scene.radii[group])
            radii = np.ascontiguousarray(scene.spheres.radii[owned]/scene.radii[group])
            key = (centers.shape, centers.tobytes(), radii.tobytes())
            if key not in self.library:
                self.library[key] = AdaptiveLocalResponse(centers, radii, **options)
            self.maps.append(self.library[key])
        self.compile_s = time.perf_counter()-start

    def solve(self, source, max_bounces=32, max_exchanges=256):
        if not isinstance(max_bounces, int) or max_bounces < 1 or not isinstance(max_exchanges, int) or max_exchanges < 1:
            raise ValueError("Positive collision and exchange caps required")
        if np.any(source.initial_bounces > 1) or np.any(source.last_atom >= len(self.scene.spheres.radii)) or np.any(source.last_atom < -1):
            raise ValueError("Valid uncollided or first-reflected source required")
        start = time.perf_counter()
        s = self.scene
        positions, outgoing, orders, owners, paths, tests = initial_response_with_paths(
            source.origins, source.directions, source.last_atom, source.initial_bounces, s.spheres.centers, s.spheres.radii,
            s.groups, s.packed_ids, s.offsets, s.centers, s.radii, max_bounces)
        escaped, unresolved = owners == -1, owners == -2
        completed = owners.copy()
        cursors = np.zeros(len(owners))
        active = np.flatnonzero(owners >= 0)
        counters = dict(through=0, approximate=0, fallback=0, primitive_tests=0, seed_primitive_tests=int(tests), interface_visits=0)
        for iteration in range(max_exchanges):
            if not len(active):
                break
            groups, fars = next_interfaces(positions[active], outgoing[active], cursors[active], completed[active], s.centers, s.radii)
            escaped[active[groups < 0]] = True
            for group in np.unique(groups[groups >= 0]):
                selected = groups == group
                ids = active[selected]
                contextualize = getattr(self.maps[group], "set_context", None)
                if contextualize is not None:
                    contextualize(group=int(group), incoming=source.incoming[ids], weights=source.weights[ids],
                                  orders=orders[ids], ray_ids=ids, max_bounces=max_bounces)
                p, d, c, stop, receipt = self.maps[group].apply(
                    (positions[ids]-s.centers[group])/s.radii[group], outgoing[ids], max_bounces-orders[ids])
                counters["interface_visits"] += len(ids)
                local_paths = receipt.pop("collider_ids")
                atoms = s.packed_ids[s.offsets[group]:s.offsets[group+1]]
                for row, identity in enumerate(ids):
                    if c[row]:
                        paths[identity, orders[identity]:orders[identity]+c[row]] = atoms[local_paths[row, :c[row]]]
                for key, value in receipt.items():
                    counters[key] += value
                orders[ids] += c
                unresolved[ids[stop]] = True
                reflected = c > 0
                physical = ids[reflected]
                positions[physical] = s.centers[group] + s.radii[group]*p[reflected]
                outgoing[physical] = d[reflected]
                cursors[ids] = np.where(reflected, 0., fars[selected])
                completed[ids] = group
            active = active[~escaped[active] & ~unresolved[active]]
        exchange_mass = float(source.weights[active].sum())
        exchange_unresolved = np.zeros(len(owners), dtype=bool)
        exchange_unresolved[active] = True
        unresolved[active] = True
        contribution = source.weights * np.clip(1-np.sum(source.incoming*outgoing, axis=1), 0., 2.) * escaped
        mass = np.bincount(orders[escaped], weights=source.weights[escaped], minlength=max_bounces+1)
        omega = np.bincount(orders, weights=contribution, minlength=max_bounces+1)
        residual = float(source.weights[unresolved].sum())
        assert np.isclose(mass.sum()+residual, source.weights.sum(), rtol=1e-12, atol=1e-12)
        return dict(omega=float(contribution.sum()), residual_mass=residual, tail_bound=2*residual,
                    exchange_tail_mass=exchange_mass, bounces=orders, outgoing=outgoing, contributions=contribution,
                    collider_ids=paths, escaped=escaped, unresolved=unresolved, exchange_unresolved=exchange_unresolved,
                    order_ledger=[dict(order=k, mass=float(mass[k]), omega=float(omega[k])) for k in range(max_bounces+1)],
                    query_s=time.perf_counter()-start, **counters)
