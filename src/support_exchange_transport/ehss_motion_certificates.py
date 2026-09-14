"""Reuse fixed boundary responses inside prevalidated atom-motion envelopes.

Each certificate covers the original nine calibration probes for every atom
center in its Cartesian envelope. It preserves the existing sampled-region
contract, not a continuous 4D chart certificate. Foreign atoms participate too.
This is a world-boundary response certificate, not a pose-transformed atlas.
"""
from dataclasses import dataclass
import time
import numpy as np
from .ehss_response import Interval, normalized
from .ehss_region_versions import RegionVersionCompiler, NodeRegionVersion
from .ehss_clipped_response import compile_cells


def sum_last(value):
    result = Interval(np.zeros(value.lo.shape[:-1]))
    for axis in range(value.lo.shape[-1]):
        result = result+Interval(value.lo[..., axis], value.hi[..., axis])
    return result


def entries(o, d, centers, radii):
    """Vector form of conservative perpendicular-distance entry intervals."""
    x = o-centers; aa = d.square().sum()
    projection = sum_last(x*d)/aa
    perpendicular = x-Interval(projection.lo[:, None], projection.hi[:, None])*d
    disc = Interval(radii).square()-sum_last(perpendicular.square())
    quotient = Interval(np.maximum(0., disc.lo), np.maximum(0., disc.hi))/aa
    # For possible real hits this radicand is nonnegative. Rounding zero down
    # must not make an unrelated missed atom abort the entire vector batch.
    root = Interval.rounded(np.sqrt(np.maximum(0., quotient.lo)), np.sqrt(np.maximum(0., quotient.hi)))
    roots = -projection-root
    absent = (projection.lo >= 0.) | (disc.hi < 0.) | (roots.hi < -64*np.finfo(float).eps*radii)
    margin = 1e-10*np.maximum(1., radii)
    certain = (disc.lo > margin*radii) & (projection.hi < -margin) & (roots.lo > margin) & ~absent
    low = np.maximum(0., roots.lo); high = np.maximum(0., roots.hi)
    low[absent] = np.inf; high[absent] = np.inf
    return low, high, certain


def _probe_certificate(p, d, route, normal, lower_plane, upper_plane, centers, radii, padding):
    cs = Interval(centers-padding[:, None], centers+padding[:, None])
    o = Interval(p); direction = Interval(d)
    # The probe must stay outside every physical ball throughout the envelope.
    if np.any(sum_last((o-cs).square()).lo <= Interval(radii).square().hi):
        return False, -1, -1, 0
    tests = 0; last = -1; margin = 1e-10*max(1., float(radii.max()))
    try:
        for wanted in route:
            low, high, certain = entries(o, direction, cs, radii); tests += len(radii)-(last >= 0)
            if last >= 0: low[last] = high[last] = np.inf; certain[last] = False
            if wanted == last or not certain[wanted]: return False, -1, -1, tests
            others = low.copy(); others[wanted] = np.inf
            if np.any(others <= high[wanted]+margin): return False, -1, -1, tests
            point = o+Interval(low[wanted], high[wanted])*direction
            surface_normal = normalized(point-Interval(cs.lo[wanted], cs.hi[wanted]))
            o = Interval(cs.lo[wanted], cs.hi[wanted])+radii[wanted]*surface_normal
            q = (o*normal).sum()
            # A straight segment with both endpoints inside cannot cross an
            # outer plane first. Reject uncertain face collisions conservatively.
            if q.lo <= lower_plane+margin or q.hi >= upper_plane-margin:
                return False, -1, -1, tests
            direction = normalized(direction-2*(direction*surface_normal).sum()*surface_normal)
            last = int(wanted)
        low, _, _ = entries(o, direction, cs, radii); tests += len(radii)-(last >= 0)
        if last >= 0: low[last] = np.inf
        q = (o*normal).sum(); speed = (direction*normal).sum()
        if speed.lo > 0.:
            face = (Interval(upper_plane)-q)/speed if np.isfinite(upper_plane) else None
            side = 1
        elif speed.hi < 0.:
            face = (Interval(lower_plane)-q)/speed if np.isfinite(lower_plane) else None
            side = -1
        else:
            return False, -1, -1, tests
        if face is None:
            return bool(np.isinf(low).all()), 1, side, tests
        return bool(face.lo >= 0. and np.all(low > face.hi+margin)), 0, side, tests
    except (ValueError, FloatingPointError):
        return False, -1, -1, tests


@dataclass(frozen=True)
class MotionCertificates:
    accepted: np.ndarray
    routes: tuple
    lower_centers: np.ndarray
    upper_centers: np.ndarray
    radii: np.ndarray
    frame: np.ndarray
    edges: np.ndarray
    seed_fingerprint: str
    receipt: dict


def prepare_certificates(seed, displacement):
    """Displacement is a per-atom bound on each world-coordinate component."""
    start = time.perf_counter()
    bound = np.asarray(displacement, dtype=float)
    if bound.ndim == 0: bound = np.full(len(seed.spheres.radii), float(bound))
    if bound.shape != seed.spheres.radii.shape or not np.isfinite(bound).all() or np.any(bound < 0.):
        raise ValueError('Finite nonnegative displacement bound per physical atom required')
    # Add explicit guards to the compilation intervals, retaining the requested
    # smaller box for runtime admission. Fixed radii are part of the contract.
    guard = 1e-10*(1.+np.max(np.abs(seed.spheres.centers), axis=1)+seed.spheres.radii)
    padding = bound+guard
    accepted = np.zeros(len(seed.region_keys), dtype=bool); routes = []
    tests = 0; probes = 0
    for index, region in enumerate(seed.region_keys):
        node, level, side, last = region[:4]
        ri = seed.region_routes[index]
        route = tuple(int(a) for a in seed.paths[seed.path_offsets[ri]:seed.path_offsets[ri+1]])
        routes.append(route)
        # Shared-last cells are compiled as initially uncollided probes. For
        # last-specific catalogs use the ordinary compiler until implemented.
        if last != -1 or not route: continue
        lower = seed.edges[seed.lower[node]-1] if seed.lower[node] else -np.inf
        upper = seed.edges[seed.upper[node]-1] if seed.upper[node] <= len(seed.edges) else np.inf
        plane = lower if side == 1 else upper
        if not np.isfinite(plane): continue
        widths = seed.widths*(2.**level); center = (region[4:]+.5)*widths
        signature = None; valid = True
        for probe in range(9):
            q = center.copy()
            if probe: q[(probe-1)//2] += (.5 if probe%2 else -.5)*widths[(probe-1)//2]
            p = plane*seed.normal+q[0]*seed.u+q[1]*seed.v
            d = side*seed.normal+q[2]*seed.u+q[3]*seed.v; d /= np.sqrt(np.dot(d, d))
            ok, status, outgoing_side, nt = _probe_certificate(p, d, route, seed.normal, lower, upper,
                seed.spheres.centers, seed.spheres.radii, padding)
            probes += 1; tests += nt
            if not ok: valid = False; break
            current = (status, outgoing_side)
            if signature is not None and current != signature: valid = False; break
            signature = current
        accepted[index] = valid
    lo = seed.spheres.centers-bound[:, None]; hi = seed.spheres.centers+bound[:, None]
    arrays = [accepted, lo, hi, seed.spheres.radii.copy(), seed.frame.copy(), seed.edges.copy()]
    for array in arrays: array.setflags(write=False)
    receipt = dict(prepare_s=time.perf_counter()-start, catalog_regions=len(accepted),
        accepted_regions=int(accepted.sum()), probe_certifications=probes, interval_atom_tests=tests,
        displacement_bounds=bound.tolist(), scope='Nine probes per sampled region; fixed world boundary and radii.',
        continuous_chart_certification=False, fixed_center_envelope=True)
    return MotionCertificates(arrays[0], tuple(routes), *arrays[1:], seed.fingerprint(), receipt)


class MotionEnvelopeCompiler(RegionVersionCompiler):
    def __init__(self, seed, certificates, atom_ids=None, max_cache_entries=256):
        super().__init__(seed, atom_ids, max_cache_entries)
        if certificates.seed_fingerprint != seed.fingerprint(): raise ValueError('Certificate seed fingerprint mismatch')
        if len(certificates.accepted) != len(self.catalog): raise ValueError('Certificate catalog size mismatch')
        # The caller cannot attach a certificate from another seed catalog.
        for i, key in enumerate(self.catalog):
            route = tuple(int(a) for a in seed.paths[seed.path_offsets[seed.region_routes[i]]:seed.path_offsets[seed.region_routes[i]+1]])
            if route != certificates.routes[i]: raise ValueError('Certificate route mismatch')
        if not np.array_equal(seed.spheres.radii, certificates.radii) or not np.array_equal(seed.frame, certificates.frame) or not np.array_equal(seed.edges, certificates.edges):
            raise ValueError('Certificate geometry contract mismatch')
        self.certificates = certificates
        self._admitted = False; self._saved = 0; self._fallback = 0

    def _revalidate_node(self, model, node, version):
        kept = {}; rows = self.node_rows[node]; probes = 0; tests = 0; evaluated = 0
        for level in range(self.level_count):
            selected = rows[self.catalog[rows, 1] == level]
            covered = self.certificates.accepted[selected] if self._admitted else np.zeros(len(selected), dtype=bool)
            fallback = selected[~covered]; evaluated += len(fallback)
            if len(fallback):
                keys = self.catalog[fallback][:, [0, 2, 3, 4, 5, 6, 7]].copy()
                kinds, paths, counts, pc, nt, _ = compile_cells(keys, model.normal, model.edges, model.offsets, model.ids,
                    model.spheres.centers, model.spheres.radii, model.lower, model.upper, model.left, model.right,
                    model.u, model.v, model.widths*(2.**level), self.max_route)
                probes += int(pc); tests += int(nt)
                candidates = {int(index):tuple(int(a) for a in paths[j, :counts[j]]) for j, index in enumerate(fallback) if kinds[j] == 2}
            else: candidates = {}
            candidates.update({int(index):self.certificates.routes[index] for index in selected[covered]})
            self._saved += int(covered.sum()); self._fallback += len(fallback)
            for index, route in candidates.items():
                if level:
                    parent = self.catalog[index]
                    children = rows[(self.catalog[rows, 1] == level-1)&(self.catalog[rows, 2] == parent[2])&
                        (self.catalog[rows, 3] == parent[3])&np.all(self.catalog[rows, 4:]//2 == parent[4:], axis=1)]
                    if not len(children) or any(int(c) not in kept or kept[int(c)] != route for c in children): continue
                kept[index] = route
        indices = tuple(sorted(kept))
        return NodeRegionVersion(node, version, indices, tuple(kept[i] for i in indices), evaluated, probes, tests)

    def compile(self, spheres, atom_ids=None, normal=None, edges=None, reuse=True, use_certificates=True):
        certificate = self.certificates
        candidate_normal = self.normal if normal is None else np.asarray(normal, dtype=float)
        candidate_edges = self.edges if edges is None else np.asarray(edges, dtype=float)
        self._admitted = bool(use_certificates and spheres.centers.shape == certificate.lower_centers.shape and
            np.array_equal(spheres.radii, certificate.radii) and np.array_equal(candidate_normal, self.normal) and
            np.array_equal(candidate_edges, certificate.edges) and
            np.all(spheres.centers >= certificate.lower_centers) and np.all(spheres.centers <= certificate.upper_centers))
        self._saved = self._fallback = 0
        frame = super().compile(spheres, atom_ids, normal, edges, reuse)
        frame.receipt.update(motion_envelope_admitted=self._admitted, certified_regions_reused=self._saved,
            ordinary_region_revalidations=self._fallback, continuous_chart_certification=False)
        return frame
