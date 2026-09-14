"""All-atom, nearest-entry hard-sphere reference; no hierarchy or response cache."""
import time
import numpy as np
from numba import njit
from .ehss_state import EHSSResult


@njit(cache=True)
def entry_distance(o,d,center,radius):
    """Stable perpendicular-distance intersection; only the incoming surface.

    Negative near roots within a radius-relative roundoff band are snapped to
    zero. Positive gaps are never skipped by a global t_min or position offset.
    """
    x=o-center; aa=np.dot(d,d); projection=np.dot(x,d)/aa
    perpendicular=x-projection*d
    disc=radius*radius-np.dot(perpendicular,perpendicular)
    if disc<0. or projection>=0.:
        return np.inf
    near=-projection-np.sqrt(max(0.,disc)/aa)
    if near < -64*np.finfo(np.float64).eps*radius:
        return np.inf
    return max(0.,near)


@njit(cache=True)
def nearest_all(o,d,last,centers,radii):
    best=np.inf; identity=-1; tests=0
    for j in range(len(radii)):
        # Outgoing reflection cannot next re-hit its own convex sphere.
        if j==last:
            continue
        tests+=1
        distance=entry_distance(o,d,centers[j],radii[j])
        if distance<best:
            best=distance; identity=j
    return identity,best,tests


@njit(cache=True)
def trace_all(origins,directions,last_atoms,initial_bounces,centers,radii,cap):
    n=len(origins); positions=origins.copy(); outgoing=directions.copy()
    escaped=np.zeros(n,dtype=np.bool_); unresolved=np.zeros(n,dtype=np.bool_)
    bounces=initial_bounces.copy(); ledger=np.full((n,cap),-1,dtype=np.int32)
    tests=0
    for ray in range(n):
        last=last_atoms[ray]
        if initial_bounces[ray]==1:
            ledger[ray,0]=last
        o=positions[ray]; d=outgoing[ray]
        while True:
            atom,distance,nt=nearest_all(o,d,last,centers,radii); tests+=nt
            if atom<0:
                escaped[ray]=True; break
            if bounces[ray]>=cap:
                unresolved[ray]=True; break
            o[:]=o+distance*d
            normal=o-centers[atom]; normal/=np.sqrt(np.dot(normal,normal))
            # Place the point exactly on the selected surface, no free-flight jump.
            o[:]=centers[atom]+radii[atom]*normal
            d[:]=d-2*np.dot(d,normal)*normal
            d[:]/=np.sqrt(np.dot(d,d))
            ledger[ray,bounces[ray]]=atom; bounces[ray]+=1; last=atom
    return positions,outgoing,escaped,unresolved,bounces,ledger,tests


def trace(spheres,source,max_bounces=64):
    if not isinstance(max_bounces,(int,np.integer)) or max_bounces<1:
        raise ValueError("Positive integer bounce cap required")
    if np.any(source.initial_bounces>1) or np.any(source.initial_bounces>max_bounces):
        raise ValueError("Reference accepts uncollided or first-reflected source states")
    if np.any(source.last_atom>=len(spheres.radii)) or np.any(source.last_atom < -1):
        raise ValueError("Invalid source atom identity")
    start=time.perf_counter()
    result=trace_all(source.origins,source.directions,source.last_atom,source.initial_bounces,
        spheres.centers,spheres.radii,max_bounces)
    positions,outgoing,escaped,unresolved,bounces,ledger,tests=result
    return EHSSResult(positions,outgoing,escaped,unresolved,bounces,ledger,source,
        dict(sphere_tests=int(tests),query_s=time.perf_counter()-start,backend="all-atom reference"))
