"""CPU workload with explicit native geometry, tracing backend and thread count."""
import time
import numpy as np
from .ehss_optimized import PreparedEHSS
from .ehss_cpu import CPUEHSSTransport, check_threads
from .ehss_cpu_geometry import CPUPhysicalScene
from .ehss_compact_source import CompactBoundary


class PreparedCPUEHSS(PreparedEHSS):
    def __init__(self, spheres, groups, atom_ids=None, boundary='sphere', simplify=False, coverage_depth=4,
                 threads=1, geometry_threads=1, backend='numba', dll_path=None):
        check_threads(threads); check_threads(geometry_threads)
        if backend not in ('numba', 'cpp'):
            raise ValueError('CPU backend must be numba or cpp')
        if backend == 'cpp' and dll_path is None:
            raise ValueError('Explicit built DLL path required for C++ backend')
        start = time.perf_counter()
        self.threads, self.backend, self.dll_path = threads, backend, dll_path
        self.scene = CPUPhysicalScene(spheres, groups, atom_ids, threads=geometry_threads)
        geometry_s = time.perf_counter()-start
        self.kind, self.simplify, self.depth = boundary, simplify, coverage_depth
        self.boundary = CompactBoundary.fit(spheres, boundary)
        self.engine = self._engine()
        self.setup = dict(total_s=time.perf_counter()-start, geometry_s=geometry_s,
                         boundary_s=self.boundary.prepare_s, engine_s=self.engine.prepare_s,
                         geometry_threads=geometry_threads, threads=threads, backend=backend)

    def _engine(self):
        if self.backend == 'cpp':
            from .ehss_cpu_cpp import CppEHSSTransport
            return CppEHSSTransport(self.scene.model, self.dll_path, self.threads, self.simplify, self.depth)
        return CPUEHSSTransport(self.scene.model, threads=self.threads, simplify=self.simplify, coverage_depth=self.depth)

    def update(self, spheres):
        start = time.perf_counter()
        unchanged = (np.array_equal(spheres.centers, self.engine.certified_centers)
                     and np.array_equal(spheres.radii, self.engine.certified_radii))
        self.scene.update(spheres)
        if not unchanged:
            self.boundary = CompactBoundary.fit(spheres, self.kind)
            self.engine = self._engine()
        self.receipt = dict(total_s=time.perf_counter()-start, unchanged=unchanged,
            geometry_s=self.scene.receipt['total_s'], boundary_s=0. if unchanged else self.boundary.prepare_s,
            engine_s=0. if unchanged else self.engine.prepare_s, coverage_rebuilt=not unchanged and self.simplify)
        return self.receipt
