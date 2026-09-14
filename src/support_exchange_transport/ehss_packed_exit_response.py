"""Frozen sparse arrays for outgoing-port lookup, with optional phase timing.

The chart, cells, acceptance policy and physical fallback are unchanged. Sorted
cell IDs replace per-unique-ID Python masks. Only APPROXIMATE cells carry exit
and collider payload; THROUGH/UNKNOWN need a status byte. No dense 4D grid is
allocated. Timing is opt-in and fallback timings are inclusive of child calls.
"""
import time
from types import MappingProxyType
import numpy as np
from .ehss_child_exit_response import ChildExitResponse
from .ehss_adaptive_response import THROUGH, APPROXIMATE, UNKNOWN
from .ehss_hierarchical_response import unit_interval


class ProfiledChildExitResponse(ChildExitResponse):
    """Identical dictionary path; separates chart and inclusive fallback time."""
    def __init__(self, source):
        if source.collect:
            raise ValueError('Finish calibration before profiling a frozen response')
        self.__dict__.update(source.__dict__)
        self.counts = dict.fromkeys(source.counts,0)
        self.profile = dict(calls=0,chart_s=0.,fallback_s=0.,dispatch_s=0.,total_s=0.)
        self.profile_enabled = False

    def coordinates(self, positions, directions):
        if not self.profile_enabled:
            return super().coordinates(positions,directions)
        tick = time.perf_counter()
        result = super().coordinates(positions,directions)
        self.profile['chart_s'] += time.perf_counter()-tick
        return result

    def evaluate(self, positions, directions, cap):
        if not self.profile_enabled:
            return super().evaluate(positions,directions,cap)
        tick = time.perf_counter()
        result = super().evaluate(positions,directions,cap)
        self.profile['fallback_s'] += time.perf_counter()-tick
        return result

    def apply(self, positions, directions, remaining):
        if not self.profile_enabled:
            return super().apply(positions,directions,remaining)
        chart,fallback = self.profile['chart_s'],self.profile['fallback_s']
        tick = time.perf_counter()
        result = super().apply(positions,directions,remaining)
        elapsed = time.perf_counter()-tick
        self.profile['calls'] += 1
        self.profile['total_s'] += elapsed
        self.profile['dispatch_s'] += elapsed-(self.profile['chart_s']-chart)-(self.profile['fallback_s']-fallback)
        return result


class PackedChildExitResponse(ChildExitResponse):
    def __init__(self, source):
        if source.collect:
            raise ValueError('Finish calibration before packing a frozen response')
        tick = time.perf_counter()
        super().__init__(source.parent,source.child,tuple(int(v) for v in source.shape),source.tolerance,source.cap)
        self.collect = False
        self.cells = MappingProxyType({})
        self.keys = np.array(sorted(source.cells),dtype=np.int64)
        self.status = np.array([source.cells[int(k)]['status'] for k in self.keys],dtype=np.int8)
        approx_keys = self.keys[self.status == APPROXIMATE]
        self.payload_index = np.full(len(self.keys),-1,dtype=np.int64)
        self.payload_index[self.status == APPROXIMATE] = np.arange(len(approx_keys))
        self.response_counts = np.array([source.cells[int(k)]['count'] for k in approx_keys],dtype=np.int64)
        self.response_positions = np.array([source.cells[int(k)]['position'] for k in approx_keys],dtype=float).reshape(-1,3)
        self.response_directions = np.array([source.cells[int(k)]['direction'] for k in approx_keys],dtype=float).reshape(-1,3)
        self.response_paths = np.array([source.cells[int(k)]['path'] for k in approx_keys],dtype=np.int32).reshape(len(approx_keys),self.cap)
        arrays = [self.keys,self.status,self.payload_index,self.response_counts,self.response_positions,self.response_directions,self.response_paths]
        for array in arrays:
            array.setflags(write=False)
        self.pack_receipt = dict(cells=len(self.keys),approximate_cells=len(approx_keys),array_payload_bytes=sum(a.nbytes for a in arrays),
                                 pack_s=time.perf_counter()-tick)
        self.profile_enabled = False
        self.profile = dict(calls=0,chart_s=0.,allocate_s=0.,lookup_s=0.,apply_s=0.,fallback_s=0.,accounting_s=0.,total_s=0.)
        self.coverage = dict(requests=0,missing=0,unknown=0,insufficient_cap=0,through=0,approximate=0)

    def prepare(self, max_cells=512):
        raise RuntimeError('Packed response is frozen; compile a new dependency version')

    def apply(self, positions, directions, remaining):
        enabled = self.profile_enabled
        start = tick = time.perf_counter() if enabled else 0.
        ids = self.coordinates(positions,directions)
        if enabled:
            now = time.perf_counter(); self.profile['chart_s'] += now-tick; tick = now
        p,d = positions.copy(),directions.copy()
        c,stopped = np.zeros(len(ids),dtype=np.int64),np.zeros(len(ids),dtype=bool)
        paths = np.full((len(ids),int(remaining.max(initial=0))),-1,dtype=np.int32)
        if enabled:
            now = time.perf_counter(); self.profile['allocate_s'] += now-tick; tick = now
        index = np.searchsorted(self.keys,ids)
        present = index < len(self.keys)
        if len(self.keys):
            present &= self.keys[np.minimum(index,len(self.keys)-1)] == ids
        status = np.full(len(ids),-1,dtype=np.int8)
        status[present] = self.status[index[present]]
        through = status == THROUGH
        candidates = np.flatnonzero(status == APPROXIMATE)
        payload = self.payload_index[index[candidates]]
        admissible = remaining[candidates] >= self.response_counts[payload]
        approximate = np.zeros(len(ids),dtype=bool)
        approximate[candidates[admissible]] = True
        use,chosen = candidates[admissible],payload[admissible]
        if enabled:
            now = time.perf_counter(); self.profile['lookup_s'] += now-tick; tick = now
        p[use],d[use],c[use] = self.response_positions[chosen],self.response_directions[chosen],self.response_counts[chosen]
        width = min(self.cap,paths.shape[1])
        paths[use,:width] = self.response_paths[chosen,:width]
        if np.any(through):
            _,far,valid = unit_interval(p[through],d[through])
            if not np.all(valid):
                raise RuntimeError('Child exit must be inside its parent')
            p[through] += far[:,None]*d[through]
        if enabled:
            now = time.perf_counter(); self.profile['apply_s'] += now-tick; tick = now
        fallback = ~(through|approximate)
        tests = 0
        for cap in np.unique(remaining[fallback]):
            selected = np.flatnonzero(fallback & (remaining == cap))
            ep,ed,ec,es,path,nt = self.evaluate(positions[selected],directions[selected],int(cap))
            p[selected],d[selected],c[selected],stopped[selected] = ep,ed,ec,es
            paths[selected,:int(cap)] = path
            tests += nt
        if enabled:
            now = time.perf_counter(); self.profile['fallback_s'] += now-tick; tick = now
        self.counts['requests'] += len(ids)
        for name,mask in [('through',through),('approximate',approximate),('fallback',fallback)]:
            self.counts[name] += int(mask.sum())
        self.counts['atom_tests'] += tests
        # Coverage diagnostics share the lookup already needed by the solver.
        self.coverage['requests'] += len(ids)
        self.coverage['missing'] += int((~present).sum())
        self.coverage['unknown'] += int((status == UNKNOWN).sum())
        self.coverage['insufficient_cap'] += int((~admissible).sum())
        self.coverage['through'] += int(through.sum())
        self.coverage['approximate'] += int(approximate.sum())
        receipt = dict(through=int(through.sum()),approximate=int(approximate.sum()),fallback=int(fallback.sum()),
                       primitive_tests=int(tests),collider_ids=paths)
        if enabled:
            now = time.perf_counter()
            self.profile['accounting_s'] += now-tick
            self.profile['total_s'] += now-start
            self.profile['calls'] += 1
        return p,d,c,stopped,receipt
