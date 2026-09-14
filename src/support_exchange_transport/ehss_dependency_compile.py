"""Transactional, versioned compilation of a fixed-ownership response DAG.

Specs are JSON-compatible values. Child placement belongs to the parent; a
rigidly moved child's intrinsic response therefore keeps its version. A leaf
geometry/policy change propagates through every dependent ancestor. Cache keys
include response and calibration policies, not only physical geometry.

Compiled objects are managed snapshots: consumers may query them but must not
mutate their geometry, atlas, or ports. Updates must pass through this compiler.
All historical versions are retained; eviction and per-cell invalidation remain
separate work. Fingerprinting scans the definition DAG each update.
"""
import hashlib
import json
import time
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np
from .ehss_hierarchical_response import ChildPose, CompiledResponseNode, HierarchicalResponseTransport
from .ehss_child_exit_response import ChildExitResponse
from .ehss_packed_exit_response import PackedChildExitResponse
from .ehss_state import reverse_pa_source


@dataclass(frozen=True)
class CompiledFrame:
    root: CompiledResponseNode
    versions: object
    nodes: object
    receipt: dict


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


class DependencyCompiler:
    def __init__(self, port_backend='dictionary'):
        if port_backend not in ('dictionary','packed'):
            raise ValueError('Choose dictionary or packed exit port lookup')
        self.port_backend = port_backend
        self._cache = {}
        self._ownership = None
        self.active = None

    def update(self, definitions, root_name):
        start = time.perf_counter()
        # Copy caller input, reject nonfinite/non-JSON values before cache use.
        specs = json.loads(_canonical(definitions))
        visiting, order, topology = set(), [], {}

        def visit(name):
            if name in visiting:
                raise ValueError('Response dependencies must be acyclic')
            if name in topology:
                return
            if name not in specs:
                raise ValueError(f'Missing response definition: {name}')
            visiting.add(name)
            spec = specs[name]
            if set(spec)-{'children', 'centers', 'radii', 'atlas', 'ports'}:
                raise ValueError(f'Unknown definition field: {name}')
            children = spec.get('children', [])
            if children:
                if 'centers' in spec or 'radii' in spec:
                    raise ValueError('Parent geometry is derived from children')
                for child in children:
                    if set(child) != {'node', 'center', 'radius', 'rotation'}:
                        raise ValueError('Child requires node, center, radius, rotation')
                    visit(child['node'])
                topology[name] = ('children', tuple(p['node'] for p in children))
            else:
                if 'centers' not in spec or 'radii' not in spec or spec.get('ports') is not None:
                    raise ValueError('A physical leaf requires geometry and no exit ports')
                topology[name] = ('leaf', len(spec['radii']))
            visiting.remove(name)
            order.append(name)

        visit(root_name)
        if set(order) != set(specs):
            raise ValueError('All supplied definitions must be reachable from root')
        ownership = (root_name, topology)
        if self._ownership is not None and ownership != self._ownership:
            raise ValueError('Frame update must preserve ordered physical ownership and topology')
        stages = dict(node_build_s=0., calibration_source_s=0., calibration_query_s=0., port_prepare_s=0.)
        versions, nodes, pending, builds, reused = {}, {}, {}, [], []
        for name in order:
            spec = specs[name]
            dependencies = [versions[p['node']] for p in spec.get('children', [])]
            version = hashlib.sha256(_canonical(['ehss-dependency-v2', self.port_backend, name, spec, dependencies]).encode()).hexdigest()
            versions[name] = version
            if version in self._cache:
                nodes[name] = self._cache[version]
                reused.append(name)
                continue
            options = dict(leaf_budget=1, tolerance=.4, cap=32)
            options.update(spec.get('atlas', {}))
            node_start = time.perf_counter()
            poses = [ChildPose(nodes[p['node']], p['center'], p['radius'], p['rotation']) for p in spec.get('children', [])]
            node = CompiledResponseNode(name, children=poses, centers=spec.get('centers'), radii=spec.get('radii'), **options)
            elapsed = time.perf_counter()-node_start
            stages['node_build_s'] += elapsed
            build = dict(name=name, version=version, node_build_s=elapsed, port_builds=[])
            config = spec.get('ports')
            if config is not None:
                if set(config)-{'shape', 'tolerance', 'cap', 'max_cells', 'calibration_power', 'calibration_seeds'}:
                    raise ValueError('Unknown exit port policy')
                settings = dict(shape=(16,32,16,32), tolerance=.4, cap=options['cap'],
                                max_cells=4096, calibration_power=11, calibration_seeds=[99851,99852])
                settings.update(config)
                power, seeds = settings['calibration_power'], settings['calibration_seeds']
                if not isinstance(power, int) or power < 1 or not seeds or any(not isinstance(s, int) or s < 0 for s in seeds):
                    raise ValueError('Positive calibration power and nonnegative integer seeds required')
                node.exit_ports = {i: ChildExitResponse(node, i, **{k:settings[k] for k in ('shape','tolerance','cap')}) for i in range(len(poses))}
                model = HierarchicalResponseTransport(node)
                for seed in seeds:
                    tick = time.perf_counter()
                    source = reverse_pa_source(model.spheres, power, seed)
                    stages['calibration_source_s'] += time.perf_counter()-tick
                    tick = time.perf_counter()
                    model.solve(source, settings['cap'])
                    stages['calibration_query_s'] += time.perf_counter()-tick
                # Own-port probes bypass own ports; sibling preparation order
                # cannot make one table depend on another partially built table.
                for i, port in node.exit_ports.items():
                    result = port.prepare(settings['max_cells'])
                    stages['port_prepare_s'] += result['prepare_s']
                    port.collect = False
                    build['port_builds'].append(dict(child=i, **result))
                if self.port_backend == 'packed':
                    node.exit_ports = {i: PackedChildExitResponse(p) for i,p in node.exit_ports.items()}
                    build['packing'] = [dict(child=i, **p.pack_receipt) for i,p in node.exit_ports.items()]
            node.exit_ports = MappingProxyType(node.exit_ports)
            for port in node.exit_ports.values():
                for cell in port.cells.values():
                    for value in cell.values():
                        if isinstance(value, np.ndarray):
                            value.setflags(write=False)
                port.cells = MappingProxyType({i: MappingProxyType(c) for i,c in port.cells.items()})
            nodes[name], pending[version] = node, node
            builds.append(build)
        # No new version or active frame is published until all enclosure and
        # compilation checks succeed. Old frames keep their physical geometry.
        changed = [n for n in order if self.active is None or self.active.versions[n] != versions[n]]
        receipt = dict(changed=changed, rebuilt=[b['name'] for b in builds], reused=reused,
                       builds=builds, stages=stages, cached_versions=len(self._cache)+len(pending),
                       update_s=time.perf_counter()-start)
        frame = CompiledFrame(nodes[root_name], MappingProxyType(versions), MappingProxyType(nodes), receipt)
        self._cache.update(pending)
        self._ownership = ownership
        self.active = frame
        return frame
