"""Explicit-path ctypes adapter for the CPU-only C++ comparison kernel."""
import ctypes as ct
from pathlib import Path
import time
import numpy as np
from .ehss_cpu import CPUEHSSTransport
from .ehss_state import EHSSResult


class CppEHSSTransport(CPUEHSSTransport):
    def __init__(self, model, dll_path, threads=1, simplify=False, coverage_depth=4):
        super().__init__(model, threads=threads, simplify=simplify, coverage_depth=coverage_depth)
        self.dll = ct.CDLL(str(Path(dll_path).resolve()))
        self.function = self.dll.ehss_trace
        self.function.restype = None
        self.function.argtypes = [ct.c_int64,ct.c_int64,ct.c_int,ct.c_int,ct.c_int]+[ct.c_void_p]*17

    def trace(self, source, cap=64, record_history=False):
        if type(cap) is not int or cap < 1 or cap > np.iinfo(np.int32).max or np.any(source.initial_bounces > 1):
            raise ValueError('Positive int32 cap and zero/one initial collision required')
        s,t = self.model.spheres,self.model.tree
        if np.any(source.last_atom < -1) or np.any(source.last_atom >= len(s.radii)):
            raise ValueError('Invalid initial physical identity')
        start=time.perf_counter()
        if not (np.array_equal(s.centers,self.certified_centers) and np.array_equal(s.radii,self.certified_radii)):
            raise ValueError('Geometry changed: bind a new transport')
        p,d,counts = source.origins.copy(),source.directions.copy(),source.initial_bounces.copy()
        escaped=np.zeros(len(p),dtype=bool); unresolved=escaped.copy()
        history=np.full((len(p),cap if record_history else 0),-1,dtype=np.int32)
        counters=np.zeros(2,dtype=np.int64); observed=np.zeros(1,dtype=np.int32)
        arrays=[p,d,source.last_atom,counts,escaped,unresolved,history,t.lo,t.hi,t.end,t.leaf_group,
                self.offsets,self.ids,s.centers,s.radii,counters,observed]
        self.function(len(p),len(t.end),cap,self.threads,int(record_history),
                      *(ct.c_void_p(a.ctypes.data) for a in arrays))
        return EHSSResult(p,d,escaped,unresolved,counts,history,source,
            dict(query_s=time.perf_counter()-start,sphere_tests=int(counters[1]),box_tests=int(counters[0]),
                 backend='MSVC C++ CPU',threads=self.threads,observed_threads=int(observed[0]),
                 scalar=True,fastmath=False,history_recorded=bool(record_history),response_reuse=False))
