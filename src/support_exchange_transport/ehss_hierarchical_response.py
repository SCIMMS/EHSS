"""Bottom-up boundary response compilation, with child-map fallback.

Every node owns a unit spherical interface. A parent is compiled by transporting
its probe rays through child responses, rather than tracing its concatenated atom
geometry. Unknown parent cells descend into the same composition. Only physical
leaves call the atom kernel. Approximate errors propagate between levels and are
not certified by the parent probe test. Sibling envelopes may overlap only when
they exclude each other's physical geometry; child envelopes fit in their parent.
"""
from dataclasses import dataclass
import time
import numpy as np
from .ehss_adaptive_response import AdaptiveLocalResponse, next_interfaces
from .ehss_interface_transport import SphereInterfaceScene
from .inside_out_pa import Spheres
from .ehss_path_append import append_collider_paths


@dataclass(frozen=True)
class ChildPose:
    node: object
    center: np.ndarray
    radius: float
    rotation: np.ndarray

    def __post_init__(self):
        center, rotation = np.array(self.center, float, copy=True), np.array(self.rotation, float, copy=True)
        if (center.shape != (3,) or rotation.shape != (3, 3) or not np.isfinite(center).all()
                or not np.isfinite(rotation).all() or not np.isfinite(self.radius) or self.radius <= 0
                or not np.allclose(rotation.T@rotation, np.eye(3), rtol=0., atol=1e-12)
                or not np.isclose(np.linalg.det(rotation), 1., rtol=0., atol=1e-12)):
            raise ValueError('Finite center, positive scale, and proper rigid rotation required')
        center.setflags(write=False)
        rotation.setflags(write=False)
        object.__setattr__(self, 'center', center)
        object.__setattr__(self, 'rotation', rotation)


def unit_interval(points, directions):
    aa = np.sum(directions*directions, axis=1)
    proj = np.sum(points*directions, axis=1)/aa
    perpendicular = points-proj[:, None]*directions
    disc = 1.-np.sum(perpendicular*perpendicular, axis=1)
    half = np.sqrt(np.maximum(0., disc)/aa)
    return -proj-half, -proj+half, disc >= 0.


class ChildComposedAtlas(AdaptiveLocalResponse):
    def __init__(self, owner, **options):
        self.owner = owner
        super().__init__(owner.centers, owner.radii, **options)

    def _probe(self, origins, directions, cap):
        return self.owner.compose(origins, directions, cap)


class CompiledResponseNode:
    def __init__(self, name, centers=None, radii=None, children=(), **options):
        self.name = name
        self.children = tuple(children)
        self.exit_ports = {}
        self.counts = dict(requests=0, approximate=0, through=0, descended=0, subtree_atom_tests=0,
                           composition_calls=0, composition_rays=0, child_visits=0)
        if self.children:
            if centers is not None or radii is not None:
                raise ValueError('Parent geometry must be derived from children')
            if any(np.linalg.norm(p.center)+p.radius >= 1.-1e-12 for p in self.children):
                raise ValueError('Child envelopes must be strictly within the parent unit sphere')
            self.child_centers = np.array([p.center for p in self.children])
            self.child_radii = np.array([p.radius for p in self.children])
            self.offsets = np.r_[0, np.cumsum([len(p.node.radii) for p in self.children])]
            centers = np.concatenate([p.center+p.radius*(p.node.centers@p.rotation.T) for p in self.children])
            radii = np.concatenate([p.radius*p.node.radii for p in self.children])
            groups = np.repeat(np.arange(len(self.children)), np.diff(self.offsets))
            # Validates finite balls, complete ownership, and foreign-ball exclusion.
            SphereInterfaceScene(Spheres(centers, radii, deduplicate=False), groups, self.child_centers, self.child_radii)
        self.centers = np.array(centers, dtype=float, copy=True)
        self.radii = np.array(radii, dtype=float, copy=True)
        self.centers.setflags(write=False)
        self.radii.setflags(write=False)
        before = self.snapshot()
        start = time.perf_counter()
        self.atlas = ChildComposedAtlas(self, **options) if self.children else AdaptiveLocalResponse(self.centers, self.radii, **options)
        self.compile_receipt = dict(name=name, kind='composed' if self.children else 'physical_leaf',
                                    compile_s=time.perf_counter()-start, atlas=self.atlas.receipt,
                                    descendant_work=self.difference(before))

    def nodes(self):
        result, seen = [], set()
        def visit(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            result.append(node)
            for child in node.children:
                visit(child.node)
        visit(self)
        if len({n.name for n in result}) != len(result):
            raise ValueError('Distinct response definitions require distinct names')
        return result

    def snapshot(self):
        return {n.name: n.counts.copy() for n in self.nodes()}

    def difference(self, before):
        return {n.name: {k: v-before[n.name][k] for k, v in n.counts.items()} for n in self.nodes()}

    def apply(self, origins, directions, remaining):
        result = self.atlas.apply(origins, directions, remaining)
        receipt = result[4]
        self.counts['requests'] += len(origins)
        for k in ('approximate', 'through'):
            self.counts[k] += receipt[k]
        self.counts['descended'] += receipt['fallback']
        self.counts['subtree_atom_tests'] += receipt['primitive_tests']
        return result

    def compose(self, origins, directions, cap, completed_child=-1, use_ports=True):
        self.counts['composition_calls'] += 1
        self.counts['composition_rays'] += len(origins)
        positions, outgoing = origins.copy(), directions.copy()
        counts = np.zeros(len(origins), dtype=np.int64)
        stopped = np.zeros(len(origins), dtype=bool)
        paths = np.full((len(origins), cap), -1, dtype=np.int32)
        cursors = np.zeros(len(origins))
        completed = np.full(len(origins), completed_child, dtype=np.int64)
        finished = np.zeros(len(origins), dtype=bool)
        active = np.arange(len(origins))
        tests = 0
        # Between reflections, each convex child can be traversed at most once.
        # Reflections consume the finite physical budget. This is a correctness
        # invariant, not an extra physical cutoff or silent escape rule.
        for _ in range((cap+1)*(len(self.children)+1)+1):
            if not len(active):
                break
            groups, fars = next_interfaces(positions[active], outgoing[active], cursors[active], completed[active],
                                           self.child_centers, self.child_radii)
            pending = active[groups >= 0]
            for group in np.unique(groups[groups >= 0]):
                mask = groups == group
                ids = active[mask]
                pose = self.children[group]
                p, d, c, stop, receipt = pose.node.apply((positions[ids]-pose.center)@pose.rotation/pose.radius,
                                                        outgoing[ids]@pose.rotation, cap-counts[ids])
                self.counts['child_visits'] += len(ids)
                tests += receipt['primitive_tests']
                append_collider_paths(paths,ids,counts,c,receipt['collider_ids'],self.offsets[group])
                counts[ids] += c
                stopped[ids[stop]] = True
                reflected = c > 0
                hit = ids[reflected]
                positions[hit] = pose.center+pose.radius*(p[reflected]@pose.rotation.T)
                outgoing[hit] = d[reflected]@pose.rotation.T
                cursors[ids] = np.where(reflected, 0., fars[mask])
                completed[ids] = group
                if use_ports and group in self.exit_ports:
                    port_ids = ids[~stop]
                    # Advancing to the child's exit is safe under foreign-ball
                    # exclusion, including when virtual sibling envelopes overlap.
                    local = (positions[port_ids]-pose.center)@pose.rotation/pose.radius
                    local_d = outgoing[port_ids]@pose.rotation
                    _, exit_distance, valid = unit_interval(local, local_d)
                    if not np.all(valid):
                        raise RuntimeError('Child exit port requires an intersected child envelope')
                    exit_points = positions[port_ids]+(pose.radius*exit_distance)[:, None]*outgoing[port_ids]
                    ep, ed, ec, es, er = self.exit_ports[group].apply(exit_points, outgoing[port_ids], cap-counts[port_ids])
                    append_collider_paths(paths,port_ids,counts,ec,er['collider_ids'])
                    counts[port_ids] += ec
                    positions[port_ids], outgoing[port_ids] = ep, ed
                    stopped[port_ids] = es
                    finished[port_ids] = True
                    tests += er['primitive_tests']
            active = pending[~stopped[pending] & ~finished[pending]]
        if len(active):
            raise RuntimeError('Child traversal violated the no-collision free-flight bound')
        leave = np.flatnonzero(~stopped)
        _, far, valid = unit_interval(positions[leave], outgoing[leave])
        if not np.all(valid):
            raise RuntimeError('Composed response left its enclosing parent sphere')
        positions[leave] += far[:, None]*outgoing[leave]
        return positions, outgoing, counts, stopped, paths, int(tests)


class HierarchicalResponseTransport:
    def __init__(self, root, center=None, radius=1.):
        self.root = root
        self.center = np.array(np.zeros(3) if center is None else center, dtype=float)
        self.radius = float(radius)
        if self.center.shape != (3,) or not np.isfinite(self.center).all() or not np.isfinite(radius) or radius <= 0:
            raise ValueError('Finite root center and positive root scale required')
        self.spheres = Spheres(self.center+radius*root.centers, radius*root.radii, deduplicate=False)

    def source_response(self, source, selected, points, remaining):
        return self.root.apply(points, source.directions[selected], remaining)

    def solve(self, source, max_bounces=64):
        if (not isinstance(max_bounces, int) or max_bounces < 1 or np.any(source.initial_bounces > 1)
                or np.any(source.last_atom < -1) or np.any(source.last_atom >= len(self.spheres.radii))):
            raise ValueError('Valid uncollided or first-reflected source and positive collision cap required')
        start, before = time.perf_counter(), self.root.snapshot()
        points = (source.origins-self.center)/self.radius
        near, far, valid = unit_interval(points, source.directions)
        selected = np.flatnonzero(valid & (far > 0.))
        positions, outgoing = source.origins.copy(), source.directions.copy()
        orders = source.initial_bounces.copy()
        paths = np.full((len(points), max_bounces), -1, dtype=np.int32)
        first = orders == 1
        paths[first, 0] = source.last_atom[first]
        unresolved = np.zeros(len(points), dtype=bool)
        p, d, c, stop, receipt = self.source_response(source, selected, points[selected], max_bounces-orders[selected])
        append_collider_paths(paths,selected,orders,c,receipt['collider_ids'])
        orders[selected] += c
        positions[selected] = self.center+self.radius*p
        outgoing[selected] = d
        unresolved[selected] = stop
        escaped = ~unresolved
        contribution = source.weights*np.clip(1-np.sum(source.incoming*outgoing, axis=1), 0., 2.)*escaped
        mass = np.bincount(orders[escaped], weights=source.weights[escaped], minlength=max_bounces+1)
        omega = np.bincount(orders, weights=contribution, minlength=max_bounces+1)
        residual = float(source.weights[unresolved].sum())
        if not np.isclose(mass.sum()+residual, source.weights.sum(), rtol=1e-12, atol=1e-12):
            raise RuntimeError('Hierarchy changed source mass')
        return dict(omega=float(contribution.sum()), residual_mass=residual, tail_bound=2*residual,
                    exchange_tail_mass=0., exchange_unresolved=np.zeros(len(points), dtype=bool),
                    escaped=escaped, unresolved=unresolved, bounces=orders, outgoing=outgoing, contributions=contribution,
                    collider_ids=paths, positions=positions,
                    order_ledger=[dict(order=k, mass=float(mass[k]), omega=float(omega[k])) for k in range(max_bounces+1)],
                    query_s=time.perf_counter()-start, primitive_tests=receipt['primitive_tests'], seed_primitive_tests=0,
                    node_work=self.root.difference(before), native_source_rays=receipt.get('native_source_rays', 0))
