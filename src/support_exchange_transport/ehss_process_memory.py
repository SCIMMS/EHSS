"""Current-process Windows memory counters, in bytes, without a new dependency.

Peaks cover the entire worker lifetime, including imports and JIT. They cannot
be described as an isolated algorithm allocation peak by subtracting a startup
reading. Use a fresh worker per method and retain stage readings.

Definitions: https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters_ex
"""
import ctypes
from ctypes import wintypes
import os


class _Counters(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[
        (name,ctypes.c_size_t) for name in (
            'PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage',
            'QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage',
            'PagefileUsage','PeakPagefileUsage','PrivateUsage')]


def process_memory():
    """Read only this process; commit is not actual disk pagefile usage."""
    if os.name!='nt': raise RuntimeError('This counter adapter is Windows-only')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    current=kernel.GetCurrentProcess; current.argtypes=[]; current.restype=wintypes.HANDLE
    get_info=kernel.K32GetProcessMemoryInfo
    get_info.argtypes=[wintypes.HANDLE,ctypes.POINTER(_Counters),wintypes.DWORD]
    get_info.restype=wintypes.BOOL
    counters=_Counters(); counters.cb=ctypes.sizeof(counters)
    if not get_info(current(),ctypes.byref(counters),counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(pid=os.getpid(),working_set_bytes=int(counters.WorkingSetSize),
        peak_working_set_bytes=int(counters.PeakWorkingSetSize),private_commit_bytes=int(counters.PrivateUsage),
        peak_commit_bytes=int(counters.PeakPagefileUsage),page_fault_count=int(counters.PageFaultCount),
        peak_scope='Process lifetime, including interpreter/import/JIT and instrumentation; not an algorithm-only allocation peak.')
