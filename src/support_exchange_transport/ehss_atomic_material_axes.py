"""Stop missing-target physical suffixes at an active material collision port.

The arrived known collision becomes the next field state, not an escape and
not an additional virtual collision. This adds an empirical projection at
reentry; it does not claim per-path equivalence to a full physical suffix.
"""
import time
import numpy as np
from numba import njit
from .ehss_guarded_intrinsic_response import nearest_world,reflect
from .ehss_surface_response import surface_key
from .ehss_sparse_path_response import readonly
from .ehss_empirical_field import geometry_key
from .ehss_field_condensation import ScatteringKernel
from .ehss_state import EHSSSource
from .ehss_material_collision_field import (
    MaterialCollisionField,material_keys,material_rotations,first_observed_successors)


@njit(cache=True)
def material_keys(points,directions,atoms,centers,groups,rotations,bins):
    keys=np.empty((len(atoms),7),dtype=np.int64)
    for i,atom in enumerate(atoms):
        normal=points[i]-centers[atom]; normal/=np.sqrt(np.dot(normal,normal))
        rotation=rotations[atom]
        key=surface_key(groups[atom],atom,normal@rotation,directions[i]@rotation,bins)
        for j in range(7): keys[i,j]=key[j]
    return keys


@njit(cache=True)
def current_material_port(point,direction,atom,world,groups,rotations,bins,table,active):
    normal=point-world[atom]; normal/=np.sqrt(np.dot(normal,normal))
    rotation=rotations[atom]
    key=surface_key(groups[atom],atom,normal@rotation,direction@rotation,bins)
    lo=0; hi=len(table)
    while lo<hi:
        mid=(lo+hi)//2; compare=0
        for c in range(7):
            if table[mid,c]<key[c]: compare=-1; break
            if table[mid,c]>key[c]: compare=1; break
        if compare<0: lo=mid+1
        else: hi=mid
    if lo>=len(table) or not active[lo]: return -1
    for c in range(7):
        if table[lo,c]!=key[c]: return -1
    return lo


@njit(cache=False)
def trace_port_prefix(origins,directions,last_atoms,cap,world,radii,
                      lo,hi,left,right,end,leaf_group,offsets,ids,
                      groups,rotations,bins,table,active,allow_reentry):
    p=origins.copy(); d=directions.copy(); counts=np.ones(len(p),dtype=np.int64)
    history=np.full((len(p),cap),-1,dtype=np.int32)
    ports=np.full(len(p),-1,dtype=np.int64)
    escaped=np.zeros(len(p),dtype=np.bool_); unresolved=escaped.copy()
    counters=np.zeros(2,dtype=np.int64); queries=lookups=0
    for ray in range(len(p)):
        last=last_atoms[ray]; count=1; history[ray,0]=last
        while True:
            if allow_reentry:
                lookups+=1
                state=current_material_port(p[ray],d[ray],last,world,groups,rotations,bins,table,active)
                if state>=0:
                    ports[ray]=state; break
            queries+=1
            atom,distance=nearest_world(0,p[ray],d[ray],last,np.inf,-1,
                lo,hi,left,right,end,leaf_group,offsets,ids,world,radii,counters)
            if atom<0: escaped[ray]=True; break
            if count>=cap: unresolved[ray]=True; break
            reflect(p[ray],d[ray],atom,distance,world,radii)
            history[ray,count]=atom; count+=1; last=atom
        counts[ray]=count
    return p,d,escaped,unresolved,counts,history,ports,counters,queries,lookups


def trace_to_material_port(model,source,keys,active,groups,rotations,bins,cap,allow_reentry=True):
    """Current physical first-entry search; handoff is a third terminal category."""
    if not isinstance(cap,int) or cap<1: raise ValueError('Positive collision budget required')
    if np.any(source.initial_bounces!=1) or np.any(source.last_atom<0): raise ValueError('First-reflected inputs required')
    if np.any(source.last_atom>=len(model.spheres.radii)): raise ValueError('Invalid physical collider identity')
    if not isinstance(bins,int) or bins<1: raise ValueError('Positive bins required')
    if tuple(keys)!=tuple(sorted(set(keys))) or len(active)!=len(keys): raise ValueError('Unique sorted keys and matching active mask required')
    table=readonly(np.array(keys,dtype=np.int64).reshape(-1,7))
    if rotations.shape!=(len(model.spheres.radii),3,3): raise ValueError('One material rotation per atom required')
    tree=model.tree; tick=time.perf_counter()
    result=trace_port_prefix(source.origins,source.directions,source.last_atom,cap,model.spheres.centers,model.spheres.radii,
        tree.lo,tree.hi,tree.left,tree.right,tree.end,tree.leaf_group,tree.offsets,tree.ids,
        groups,rotations,bins,table,active,allow_reentry)
    p,d,escaped,unresolved,counts,history,ports,counters,queries,lookups=result
    assert np.all(escaped.astype(int)+unresolved+(ports>=0)==1)
    return dict(positions=p,outgoing=d,escaped=escaped,unresolved=unresolved,bounces=counts,collider_ids=history,ports=ports,
        metrics=dict(query_s=time.perf_counter()-tick,sphere_tests=int(counters[1]),box_tests=int(counters[0]),
            nearest_queries=int(queries),port_lookups=int(lookups)))
