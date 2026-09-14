"""Finite-region, collision-itinerary response compilation for exact EHSS.

A cell stores an input box and an analytic sequence of sphere reflections, not
a previously evaluated outgoing ray. Interval propagation checks the entire
box against every owned primitive during compilation. Runtime replays only the
certified itinerary. Uncertain/grazing cells are rejected, never guessed.
"""
from dataclasses import dataclass
import hashlib
import time
import numpy as np
from numba import njit
from .ehss_reference import entry_distance,nearest_all


class Interval:
    """Elementwise float64 intervals with outward-rounded elementary steps."""
    def __init__(self,lo,hi=None):
        self.lo=np.asarray(lo,dtype=float)
        self.hi=np.asarray(lo if hi is None else hi,dtype=float)

    @staticmethod
    def rounded(lo,hi):
        return Interval(np.nextafter(lo,-np.inf),np.nextafter(hi,np.inf))

    def __add__(self,other):
        other=as_interval(other)
        return self.rounded(self.lo+other.lo,self.hi+other.hi)
    __radd__=__add__

    def __neg__(self):
        return Interval(-self.hi,-self.lo)

    def __sub__(self,other):
        return self+-as_interval(other)

    def __rsub__(self,other):
        return as_interval(other)+-self

    def __mul__(self,other):
        other=as_interval(other)
        terms=np.stack([self.lo*other.lo,self.lo*other.hi,self.hi*other.lo,self.hi*other.hi])
        return self.rounded(terms.min(axis=0),terms.max(axis=0))
    __rmul__=__mul__

    def __truediv__(self,other):
        other=as_interval(other)
        if np.any((other.lo<=0)&(other.hi>=0)):
            raise ValueError("Interval division crosses zero")
        return self*self.rounded(1/other.hi,1/other.lo)

    def square(self):
        low=np.minimum(self.lo*self.lo,self.hi*self.hi)
        low=np.where((self.lo<=0)&(self.hi>=0),0.,low)
        return self.rounded(low,np.maximum(self.lo*self.lo,self.hi*self.hi))

    def sum(self):
        result=Interval(0.)
        for low,high in zip(self.lo.flat,self.hi.flat):
            result=result+Interval(low,high)
        return result

    def sqrt(self):
        if np.any(self.lo<0):
            raise ValueError("Uncertain/nonpositive squared length")
        return self.rounded(np.sqrt(self.lo),np.sqrt(self.hi))


def as_interval(value):
    return value if isinstance(value,Interval) else Interval(value)


def normalized(value):
    return value/value.square().sum().sqrt()


def possible_entry(o,d,center,radius):
    """Lower/upper possible incoming entry and whether every state hits.

    Match the reference's perpendicular-distance formulation. A tangency or
    zero-distance uncertainty stays a possible event, preventing certification.
    """
    x=o-center; aa=d.square().sum(); projection=(x*d).sum()/aa
    if projection.lo>=0:
        return np.inf,np.inf,False
    perpendicular=x-projection*d
    disc=Interval(radius).square()-perpendicular.square().sum()
    if disc.hi<0:
        return np.inf,np.inf,False
    nonnegative=Interval(max(0.,float(disc.lo)),max(0.,float(disc.hi)))
    # sqrt(0) rounds one ulp below zero; the final root enclosure is still safe.
    root=-projection-(nonnegative/aa).sqrt()
    if root.hi < -64*np.finfo(float).eps*radius:
        return np.inf,np.inf,False
    margin=1e-10*max(1.,radius)
    certain=disc.lo>margin*radius and projection.hi< -margin and root.lo>margin
    return max(0.,float(root.lo)),max(0.,float(root.hi)),bool(certain)


def seed_itinerary(origin,direction,last,centers,radii,cap=32):
    o=np.asarray(origin,dtype=float).copy(); d=np.asarray(direction,dtype=float).copy()
    route=[]
    for _ in range(cap+1):
        atom,distance,_=nearest_all(o,d,last,centers,radii)
        if atom<0:
            return np.asarray(route,dtype=np.int64)
        if len(route)==cap:
            return None
        o=o+distance*d
        normal=o-centers[atom]; normal/=np.linalg.norm(normal)
        o=centers[atom]+radii[atom]*normal
        d=d-2*np.dot(d,normal)*normal; d/=np.linalg.norm(d)
        route.append(atom); last=atom
    return None


def certify_itinerary(origin,direction,halfwidth,last,centers,radii,route,geometry_pad):
    """Certify all intended hits, their ordering, and final no-further-hit.

    Domain is restricted to legitimate EHSS states: if last>=0 it denotes the
    preceding convex-sphere reflection, hence that one sphere may be skipped.
    Geometry padding also guards body/world representation and rigid-pose drift.
    This is a conservative numerical certificate, not a proof for arbitrary
    coordinate magnitudes or arbitrarily long chaotic trajectories.
    """
    o=Interval(origin-halfwidth[:3],origin+halfwidth[:3])
    d=Interval(direction-halfwidth[3:],direction+halfwidth[3:])
    return certify_interval_state(o,d,last,centers,radii,route,geometry_pad)


def certify_interval_state(o,d,last,centers,radii,route,geometry_pad):
    """Shared propagation for a Cartesian box or a four-dimensional line chart."""
    cs=[Interval(c-geometry_pad,c+geometry_pad) for c in centers]
    tests=0; margin=1e-10*max(1.,float(np.max(radii)))
    try:
        for wanted in list(route)+[-1]:
            entries=[]
            for atom in range(len(radii)):
                if atom==last:
                    entries.append((np.inf,np.inf,False))
                else:
                    tests+=1; entries.append(possible_entry(o,d,cs[atom],radii[atom]))
            if wanted<0:
                return all(np.isinf(e[0]) for e in entries),tests,"exit uncertainty"
            low,high,certain=entries[wanted]
            if not certain:
                return False,tests,"selected hit uncertainty"
            if any(e[0]<=high+margin for j,e in enumerate(entries) if j!=wanted):
                return False,tests,"collision ordering uncertainty"
            distance=Interval(low,high)
            point=o+distance*d; normal=normalized(point-cs[wanted])
            o=cs[wanted]+radii[wanted]*normal
            d=normalized(d-2*(d*normal).sum()*normal)
            last=wanted
    except (ValueError,FloatingPointError):
        return False,tests,"interval degeneracy"
    return False,tests,"unfinished route"


def line_coordinates(origin,direction,axis,sign,plane):
    """Two transverse coordinates on a fixed plane and two direction slopes."""
    other=[k for k in range(3) if k!=axis]
    if sign*direction[axis]<=0:
        raise ValueError("Direction outside oriented line chart")
    slopes=direction[other]/direction[axis]
    point=origin[other]+(plane-origin[axis])*slopes
    return np.r_[point,slopes]


def line_interval_state(state,width,axis,sign,plane):
    other=[k for k in range(3) if k!=axis]
    lo=np.zeros(3); hi=np.zeros(3); lo[axis]=hi[axis]=plane
    lo[other]=state[:2]-width[:2]; hi[other]=state[:2]+width[:2]
    dl=np.zeros(3); dh=np.zeros(3); dl[axis]=dh[axis]=sign
    dl[other]=np.minimum(sign*(state[2:]-width[2:]),sign*(state[2:]+width[2:]))
    dh[other]=np.maximum(sign*(state[2:]-width[2:]),sign*(state[2:]+width[2:]))
    return Interval(lo,hi),normalized(Interval(dl,dh))


@dataclass
class ResponseCell:
    node: int
    seed: np.ndarray
    low: np.ndarray
    high: np.ndarray
    last: int
    itinerary: np.ndarray
    anchor_group: int
    atom_ids: np.ndarray
    centers: np.ndarray
    radii: np.ndarray
    geometry_pad: float
    halfwidth: np.ndarray
    interval_tests: int
    chart_axis: int = -1
    chart_sign: int = 1
    chart_plane: float = 0.


def node_geometry(tree,node):
    atom_leaf=tree.leaf_node[tree.groups]
    atoms=np.flatnonzero((atom_leaf>=node)&(atom_leaf<tree.end[node]))
    anchor=int(tree.groups[atoms[0]])
    rotation=tree.rotations[anchor]; translation=tree.translations[anchor]
    centers=(tree.world[atoms]-translation)@rotation
    # A leaf's intrinsic centers have no pose-dependent subtraction roundoff.
    same=tree.groups[atoms]==anchor
    centers[same]=tree.local[atoms[same]]
    return atoms,anchor,centers,rotation,translation


class ResponseLibrary:
    def __init__(self):
        self.cells=[]; self.compile_receipts=[]; self.topology=None

    @staticmethod
    def topology_key(tree):
        digest=hashlib.sha256()
        for value in [tree.parent,tree.leaf_group,tree.groups,tree.coarse_supports]:
            digest.update(np.asarray(value.shape,dtype=np.int64).tobytes()); digest.update(value.tobytes())
        return digest.hexdigest()

    def compile(self,tree,node,origin,direction,last=-1,position_width=.05,direction_width=.025,max_shrinks=10,max_bounces=32,
                input_chart="cartesian",adaptive_growth=False):
        if not isinstance(node,(int,np.integer)) or not 0<=node<len(tree.parent):
            raise ValueError("Valid physical hierarchy node required")
        if not np.isfinite(position_width) or not np.isfinite(direction_width) or min(position_width,direction_width)<=0:
            raise ValueError("Positive finite input halfwidths required")
        if not isinstance(max_shrinks,(int,np.integer)) or max_shrinks<0:
            raise ValueError("Nonnegative integer shrink limit required")
        if input_chart not in ["cartesian","line"]:
            raise ValueError("Input chart must be cartesian or line")
        origin=np.asarray(origin,dtype=float); direction=np.asarray(direction,dtype=float)
        if origin.shape!=(3,) or direction.shape!=(3,) or not np.isfinite(origin).all() or not np.isfinite(direction).all() or not np.isclose(np.linalg.norm(direction),1.,rtol=1e-12,atol=1e-12):
            raise ValueError("Finite position and unit physical direction required")
        topology=self.topology_key(tree)
        if self.topology is not None and topology!=self.topology:
            raise ValueError("A different physical partition needs a separate response library")
        self.topology=topology
        start=time.perf_counter()
        atoms,anchor,centers,rotation,translation=node_geometry(tree,node)
        radii=tree.radii[atoms].copy()
        o=(np.asarray(origin)-translation)@rotation; d=np.asarray(direction)@rotation
        mapped=np.flatnonzero(atoms==last); local_last=int(mapped[0]) if len(mapped) else -1
        route=seed_itinerary(o,d,local_last,centers,radii,max_bounces)
        record=dict(node=int(node),accepted=False,interval_tests=0,shrinks=0,
            route_length=None if route is None else len(route),input_chart=input_chart,adaptive_growth=adaptive_growth)
        pad=1e-10*max(1.,float(np.max(radii)),float(np.max(np.abs(centers))))
        axis=-1; sign=1; plane=0.; seed=np.r_[o,d]
        if input_chart=="line":
            axis=int(np.argmax(np.abs(d))); sign=1 if d[axis]>0 else -1
            # The plane is upstream of every owned finite-radius sphere.
            # A finite clearance avoids placing the certificate seed within a
            # numerical/tiny-gap uncertainty band of the frontmost sphere.
            clearance=float(np.max(radii))+10*pad
            plane=float(np.min(centers[:,axis]-radii)-clearance if sign>0 else np.max(centers[:,axis]+radii)+clearance)
            state=line_coordinates(o,d,axis,sign,plane)
            canonical_o=o+(plane-o[axis])/d[axis]*d
            plane_route=seed_itinerary(canonical_o,d,local_last,centers,radii,max_bounces)
            if route is None or plane_route is None or not np.array_equal(route,plane_route):
                record.update(reason="upstream plane has a different itinerary",compile_s=time.perf_counter()-start)
                self.compile_receipts.append(record); return None
            seed=np.r_[state,0.,0.]
            width=np.r_[np.full(2,position_width),np.full(2,direction_width)]
        else:
            width=np.r_[np.full(3,position_width),np.full(3,direction_width)]
        maximum_width=width.copy()
        def certify(w):
            if input_chart=="line":
                io,idirection=line_interval_state(seed[:4],w,axis,sign,plane)
                return certify_interval_state(io,idirection,local_last,centers,radii,route,pad)
            return certify_itinerary(o,d,w,local_last,centers,radii,route,pad)
        if route is not None:
            for shrink in range(max_shrinks+1):
                valid,tests,reason=certify(width)
                record["interval_tests"]+=tests; record["shrinks"]=shrink; record["reason"]=reason
                if valid:
                    # Recover width lost to coupled shrinking, using training geometry only.
                    # Every accepted enlargement is re-certified for the full box.
                    record["growth_attempts"]=0; record["growth_accepts"]=0
                    if adaptive_growth:
                        for coordinate in range(len(width)):
                            while width[coordinate]<maximum_width[coordinate]:
                                trial=width.copy(); trial[coordinate]=min(2*trial[coordinate],maximum_width[coordinate])
                                ok,nt,_=certify(trial)
                                record["interval_tests"]+=nt; record["growth_attempts"]+=1
                                if not ok:
                                    break
                                width=trial; record["growth_accepts"]+=1
                    stored_width=np.r_[width,1.,1.] if input_chart=="line" else width
                    # Exclude numerical boundary ambiguity in body-frame lookup.
                    lookup_guard=1e-11*(1.+np.abs(seed))
                    if np.any(stored_width<=2*lookup_guard):
                        record["reason"]="input region too small"; break
                    cell=ResponseCell(int(node),seed,seed-stored_width+lookup_guard,seed+stored_width-lookup_guard,
                        int(last),atoms[route],anchor,atoms.copy(),centers.copy(),radii,pad,stored_width.copy(),record["interval_tests"],
                        axis,sign,plane)
                    for value in [cell.seed,cell.low,cell.high,cell.itinerary,cell.atom_ids,cell.centers,cell.radii,cell.halfwidth]:
                        value.setflags(write=False)
                    self.cells.append(cell); record["accepted"]=True; record["reason"]="certified"
                    record["halfwidth"]=width.tolist(); break
                width*=.5
        else:
            record["reason"]="unresolved seed trajectory"
        record["compile_s"]=time.perf_counter()-start
        self.compile_receipts.append(record)
        return self.cells[-1] if record["accepted"] else None

    def pack(self,tree,mode="all"):
        if mode not in ["all","leaf","parent","off"]:
            raise ValueError("Cache mode must be all, leaf, parent or off")
        if self.topology is not None and self.topology!=self.topology_key(tree):
            arrays=empty_responses(len(tree.parent))
            return arrays,dict(active_cells=0,inactive_cells=len(self.cells),packed_bytes=sum(a.nbytes for a in arrays),
                invalidation_reason="physical topology changed")
        count=len(tree.parent); offsets=[0]; ordered=[]; inactive=0
        frames_r=np.tile(np.eye(3),(count,1,1)); frames_t=np.zeros((count,3))
        by_node=[[] for _ in range(count)]
        for cell in self.cells:
            by_node[cell.node].append(cell)
        for node in range(count):
            is_leaf=tree.leaf_group[node]>=0
            eligible=mode!="off" and (mode=="all" or (mode=="leaf")==is_leaf) and tree.exclusive[node]
            if by_node[node]:
                atoms,anchor,centers,r,t=node_geometry(tree,node)
                frames_r[node]=r; frames_t[node]=t
                for cell in by_node[node]:
                    compatible=eligible and anchor==cell.anchor_group and np.array_equal(atoms,cell.atom_ids) and np.array_equal(tree.radii[atoms],cell.radii)
                    compatible=compatible and np.all(np.abs(centers-cell.centers)<=.25*cell.geometry_pad)
                    if compatible:
                        ordered.append(cell)
                    else:
                        inactive+=1
            offsets.append(len(ordered))
        low=np.array([c.low for c in ordered],dtype=float).reshape(-1,6)
        high=np.array([c.high for c in ordered],dtype=float).reshape(-1,6)
        last=np.array([c.last for c in ordered],dtype=np.int64)
        route_offsets=np.r_[0,np.cumsum([len(c.itinerary) for c in ordered])].astype(np.int64)
        routes=np.concatenate([c.itinerary for c in ordered]) if ordered else np.empty(0,dtype=np.int64)
        charts=np.array([[c.chart_axis,c.chart_sign,c.chart_plane] for c in ordered],dtype=float).reshape(-1,3)
        arrays=(np.asarray(offsets,dtype=np.int64),low,high,last,route_offsets,routes,frames_r,frames_t,charts)
        return arrays,dict(active_cells=len(ordered),inactive_cells=inactive,
            packed_bytes=sum(a.nbytes for a in arrays),compile_s=sum(r["compile_s"] for r in self.compile_receipts),
            interval_tests=sum(r["interval_tests"] for r in self.compile_receipts))


def empty_responses(nodes):
    return (np.zeros(nodes+1,dtype=np.int64),np.empty((0,6)),np.empty((0,6)),np.empty(0,dtype=np.int64),
        np.zeros(1,dtype=np.int64),np.empty(0,dtype=np.int64),np.tile(np.eye(3),(nodes,1,1)),np.zeros((nodes,3)),np.empty((0,3)))


@njit(cache=True)
def replay_response(node,o,d,last,bounces,cap,world,radii,history,responses,counters,cache_counts):
    offsets,low,high,last_keys,route_offsets,routes,frames_r,frames_t,charts=responses
    if offsets[node]==offsets[node+1]:
        return False,last,bounces
    cache_counts[0]+=1
    local_o=(o-frames_t[node])@frames_r[node]; local_d=d@frames_r[node]
    state=np.empty(6)
    for cell in range(offsets[node],offsets[node+1]):
        cache_counts[1]+=1
        if last!=last_keys[cell]:
            continue
        axis=int(charts[cell,0])
        if axis>=0:
            sign=charts[cell,1]; plane=charts[cell,2]
            if sign*local_d[axis]<=1e-12:
                continue
            k=0
            for coordinate in range(3):
                if coordinate!=axis:
                    slope=local_d[coordinate]/local_d[axis]
                    state[k]=local_o[coordinate]+(plane-local_o[axis])*slope
                    state[k+2]=slope; k+=1
            state[4]=0.; state[5]=0.
        else:
            state[:3]=local_o; state[3:]=local_d
        if last!=last_keys[cell] or np.any(state<low[cell]) or np.any(state>high[cell]):
            continue
        first=route_offsets[cell]; end=route_offsets[cell+1]
        if bounces+end-first>cap:
            cache_counts[4]+=1; continue
        p=o.copy(); v=d.copy(); failed=False
        for k in range(first,end):
            atom=routes[k]; distance=entry_distance(p,v,world[atom],radii[atom])
            counters[5]+=1; cache_counts[6]+=1
            if not np.isfinite(distance):
                failed=True; break
            p[:]=p+distance*v
            normal=p-world[atom]; normal/=np.sqrt(np.dot(normal,normal))
            p[:]=world[atom]+radii[atom]*normal
            v[:]=v-2*np.dot(v,normal)*normal
            v[:]/=np.sqrt(np.dot(v,v))
        if failed:
            # For a line chart the physical anchor may already be past its first
            # certified collision. It must fall back rather than replay history.
            if axis>=0 and k==first:
                cache_counts[7]+=1
            else:
                cache_counts[5]+=1
            continue
        o[:]=p; d[:]=v
        for k in range(first,end):
            history[bounces]=routes[k]; bounces+=1; last=routes[k]
        cache_counts[2]+=1; cache_counts[3]+=end-first
        cache_counts[8+min(2,end-first)]+=1
        return True,last,bounces
    return False,last,bounces
