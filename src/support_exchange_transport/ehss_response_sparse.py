"""Response-only storage of an already compiled discrete boundary operator.

Non-collision local states are implicit. Missing entries of the *fully compiled*
through-closure routing mean escape, not unknown geometry. This changes storage
only; it cannot certify the original discretization or reduce compilation peak.
"""
import numpy as np
from .ehss_compiled_field import RayBoundaryGrid,CompiledSupportField


class SparseIndexArray:
    def __init__(self,keys,values,size,default=None):
        self.keys=keys
        self.values=np.array(values,copy=True)
        self.values.setflags(write=False)
        self.size=size; self.default=default

    def __getitem__(self,index):
        requested=np.asarray(index)
        if not np.issubdtype(requested.dtype,np.integer) or np.any(requested<0) or np.any(requested>=self.size):
            raise IndexError("Boundary cell identity outside the compiled domain")
        flat=requested.reshape(-1)
        positions=np.searchsorted(self.keys,flat)
        valid=np.zeros(len(flat),dtype=bool)
        present=positions<len(self.keys)
        valid[present]=self.keys[positions[present]]==flat[present]
        if self.default is None and not np.all(valid):
            raise KeyError("No nontrivial response stored for this boundary cell")
        result=np.full((len(flat),)+self.values.shape[1:],0 if self.default is None else self.default,dtype=self.values.dtype)
        result[valid]=self.values[positions[valid]]
        return result.reshape(requested.shape+self.values.shape[1:])


class DirectionLookup:
    def __init__(self,axes,disk_count,size):
        self.axes=axes; self.disk_count=disk_count; self.size=size

    def __getitem__(self,index):
        index=np.asarray(index)
        if not np.issubdtype(index.dtype,np.integer) or np.any(index<0) or np.any(index>=self.size):
            raise IndexError("Direction state outside the boundary basis")
        return self.axes[index//self.disk_count]


class CompactRayBoundaryGrid(RayBoundaryGrid):
    """Procedural ray basis: retain only direction axes and transverse frames."""
    def __init__(self,grid):
        self.shape=grid.shape; self.size=grid.size; self.disk_count=grid.disk_count
        self.axes=grid.axes; self.t=grid.t; self.b=grid.b
        self.directions=DirectionLookup(self.axes,self.disk_count,self.size)


class SparseCompiledSupportField(CompiledSupportField):
    def __init__(self,compiled):
        if compiled.through_mode!="preserve":
            raise ValueError("Response-only storage requires completed continuous-through closure")
        self.scene=compiled.scene; self.grid=CompactRayBoundaryGrid(compiled.grid)
        self.states=compiled.states; self.through_mode=compiled.through_mode
        self.response_ids=np.flatnonzero((compiled.bounces>0)|compiled.stopped)
        self.route_ids=np.flatnonzero(compiled.routing>=0)
        self.response_ids.setflags(write=False); self.route_ids.setflags(write=False)
        for name in ["output","bounces","stopped","physical_outgoing","successor"]:
            setattr(self,name,SparseIndexArray(self.response_ids,getattr(compiled,name)[self.response_ids],self.states))
        self.routing=SparseIndexArray(self.route_ids,compiled.routing[self.route_ids],self.states,default=-1)
        self.through_skips=compiled.through_skips; self.overlap_routes=compiled.overlap_routes
        targets=compiled.routing[self.route_ids]
        if np.any(self.response_ids[np.searchsorted(self.response_ids,targets)]!=targets):
            raise AssertionError("Sparse closure targets must have stored responses")

    def receipt(self):
        # Every shared key array is counted once; no reference to the dense graph
        # or map library is retained by this object.
        arrays=[self.response_ids,self.route_ids,self.grid.axes,self.grid.t,self.grid.b,
                self.routing.values,*[getattr(self,name).values for name in ["output","bounces","stopped","physical_outgoing","successor"]]]
        return dict(full_label_count=self.states,response_cells=len(self.response_ids),routed_outgoing_cells=len(self.route_ids),
                    payload_bytes=sum(a.nbytes for a in arrays),
                    response_fraction=len(self.response_ids)/self.states,route_fraction=len(self.route_ids)/self.states)
