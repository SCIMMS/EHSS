"""Exact packet routing through overlapping, foreign-geometry-free envelopes.

The envelope is a virtual interface. Each contains its owned atom balls and
excludes all foreign atom balls (or optional larger declared spherical regions).
Envelope/envelope overlap is allowed. Physical anchors survive virtual crossings.
This is an interface-composition reference, not a general hierarchy compiler.
"""
import hashlib
import time
import numpy as np
from numba import njit
from .ehss_reference import entry_distance
from .ehss_state import EHSSResult
from .inside_out_pa import Spheres


@njit(cache=True)
def sphere_interval(origin,direction,center,radius):
    delta=origin-center
    aa=np.dot(direction,direction)
    projection=np.dot(delta,direction)/aa
    perpendicular=delta-projection*direction
    disc=radius*radius-np.dot(perpendicular,perpendicular)
    if disc<0.:
        return np.inf,-np.inf
    half=np.sqrt(max(0.,disc)/aa)
    return -projection-half,-projection+half


@njit(cache=True)
def next_interface(origin,direction,cursor,completed,centers,radii):
    selected=-1; best=np.inf; inside=False; tests=0
    for group in range(len(radii)):
        if group==completed:
            continue
        tests+=1
        near,far=sphere_interval(origin,direction,centers[group],radii[group])
        # A convex envelope already behind the cursor cannot be entered again
        # until a physical reflection changes direction. No epsilon position jump.
        if far<=cursor:
            continue
        start=max(cursor,near)
        if start<best:
            selected=group; best=start; inside=near<cursor
    return selected,best,inside,tests


@njit(cache=True)
def compile_initial_routes(origins,directions,centers,radii):
    targets=np.full(len(origins),-1,dtype=np.int64)
    inside=np.zeros(len(origins),dtype=np.bool_)
    for i in range(len(origins)):
        group,_,within,_=next_interface(origins[i],directions[i],0.,-1,centers,radii)
        targets[i]=group; inside[i]=within
    return targets,inside


@njit(cache=True)
def apply_local_response(origin,direction,last,bounce,ledger,cap,ids,centers,radii):
    tests=0; added=0
    while True:
        atom=-1; best=np.inf
        for identity in ids:
            if identity==last:
                continue
            tests+=1
            distance=entry_distance(origin,direction,centers[identity],radii[identity])
            if distance<best:
                atom=identity; best=distance
        if atom<0:
            return last,bounce,False,tests,added
        if bounce>=cap:
            return last,bounce,True,tests,added
        origin[:]=origin+best*direction
        normal=origin-centers[atom]; normal/=np.sqrt(np.dot(normal,normal))
        origin[:]=centers[atom]+radii[atom]*normal
        direction[:]=direction-2*np.dot(direction,normal)*normal
        direction[:]/=np.sqrt(np.dot(direction,direction))
        ledger[bounce]=atom; bounce+=1; last=atom; added+=1


@njit(cache=True)
def trace_interfaces(origins,directions,last_atoms,initial_bounces,atom_centers,atom_radii,
                     packed_ids,offsets,envelope_centers,envelope_radii,cap,
                     cached,initial_targets,initial_inside):
    n=len(origins); positions=origins.copy(); outgoing=directions.copy()
    escaped=np.zeros(n,dtype=np.bool_); unresolved=np.zeros(n,dtype=np.bool_)
    bounces=initial_bounces.copy(); ledger=np.full((n,cap),-1,dtype=np.int32)
    # Per-ray: envelope tests, atom tests, visits, no-collision visits,
    # inside-envelope entries, return visits, local multi-collision calls,
    # inside entries after a prior envelope exit (excluding initial PA sources).
    counts=np.zeros((n,8),dtype=np.int64)
    for ray in range(n):
        origin=positions[ray]; direction=outgoing[ray]; last=last_atoms[ray]
        if bounces[ray]==1:
            ledger[ray,0]=last
        cursor=0.; completed=-1; first=True
        seen=np.zeros(len(envelope_radii),dtype=np.bool_)
        while True:
            was_first=first
            if first and cached:
                group=initial_targets[ray]; within=initial_inside[ray]
            else:
                group,_,within,nt=next_interface(origin,direction,cursor,completed,envelope_centers,envelope_radii)
                counts[ray,0]+=nt
            first=False
            if group<0:
                escaped[ray]=True; break
            counts[ray,2]+=1; counts[ray,4]+=int(within)
            counts[ray,7]+=int(within and not was_first)
            counts[ray,5]+=int(seen[group]); seen[group]=True
            last,bounces[ray],stop,nt,added=apply_local_response(
                origin,direction,last,bounces[ray],ledger[ray],cap,
                packed_ids[offsets[group]:offsets[group+1]],atom_centers,atom_radii)
            counts[ray,1]+=nt; counts[ray,3]+=int(added==0 and not stop)
            counts[ray,6]+=int(added>1)
            if stop:
                unresolved[ray]=True; break
            # The owned response has no further collision. No foreign atom can
            # lie before this exit because construction certifies its exclusion.
            # Keep origin at the last physical event; only advance a scalar cursor.
            _,cursor=sphere_interval(origin,direction,envelope_centers[group],envelope_radii[group])
            completed=group
    return positions,outgoing,escaped,unresolved,bounces,ledger,counts


class SphereInterfaceScene:
    def __init__(self,spheres,groups,envelope_centers,envelope_radii,region_centers=None,region_radii=None):
        self.spheres=Spheres(spheres.centers,spheres.radii,deduplicate=False)
        raw=np.asarray(groups)
        if raw.shape!=(len(spheres.radii),) or not np.issubdtype(raw.dtype,np.integer):
            raise ValueError("One integer support ownership per atom required")
        self.groups=np.array(raw,dtype=np.int64,copy=True)
        unique=np.unique(self.groups)
        if not np.array_equal(unique,np.arange(len(unique))):
            raise ValueError("Contiguous nonempty support identities required")
        self.centers=np.array(envelope_centers,dtype=float,copy=True)
        self.radii=np.array(envelope_radii,dtype=float,copy=True)
        if self.centers.shape!=(len(unique),3) or self.radii.shape!=(len(unique),) or not np.isfinite(self.centers).all() or not np.isfinite(self.radii).all() or np.any(self.radii<=0):
            raise ValueError("Finite positive spherical interfaces required")
        self.region_centers=None; self.region_radii=None
        if (region_centers is None)!=(region_radii is None):
            raise ValueError("Both declared region centers and radii required")
        if region_centers is not None:
            self.region_centers=np.array(region_centers,dtype=float,copy=True)
            self.region_radii=np.array(region_radii,dtype=float,copy=True)
            if self.region_centers.shape!=self.centers.shape or self.region_radii.shape!=self.radii.shape or not np.isfinite(self.region_centers).all() or not np.isfinite(self.region_radii).all() or np.any(self.region_radii<=0):
                raise ValueError("One finite positive declared region per support required")
        self.minimum_clearance=np.inf
        for group in unique:
            owned=self.groups==group
            distance=np.linalg.norm(self.spheres.centers-self.centers[group],axis=1)
            guard=128*np.finfo(float).eps*max(1.,self.radii[group],float(np.max(np.abs(self.centers[group]))))
            if np.any(distance[owned]+self.spheres.radii[owned]>=self.radii[group]-guard):
                raise ValueError("Each interface must strictly contain all owned physical atom balls")
            if np.any(~owned):
                clearance=float(np.min(distance[~owned]-self.spheres.radii[~owned]-self.radii[group]))
                self.minimum_clearance=min(self.minimum_clearance,clearance)
                if clearance<=guard:
                    raise ValueError("Interface intersects foreign physical support; use general hierarchy fallback")
            if self.region_centers is not None:
                region=self.region_centers[group]; radius=self.region_radii[group]
                if np.any(np.linalg.norm(self.spheres.centers[owned]-region,axis=1)+self.spheres.radii[owned]>radius):
                    raise ValueError("Declared support region must contain owned atom balls")
                rd=np.linalg.norm(self.region_centers-self.centers[group],axis=1)
                if rd[group]+radius>=self.radii[group]-guard or np.any(rd[unique!=group]-self.region_radii[unique!=group]-self.radii[group]<=guard):
                    raise ValueError("Interface violates declared support region containment or foreign exclusion")
        ids=[np.flatnonzero(self.groups==group) for group in unique]
        self.packed_ids=np.concatenate(ids); self.offsets=np.r_[0,np.cumsum([len(x) for x in ids])]
        arrays=[self.spheres.centers,self.spheres.radii,self.groups,self.centers,self.radii,self.packed_ids,self.offsets]
        if self.region_centers is not None:
            arrays += [self.region_centers,self.region_radii]
        digest=hashlib.sha256()
        for value in arrays:
            value.setflags(write=False); digest.update(str(value.shape).encode()); digest.update(value.tobytes())
        self.signature=digest.hexdigest()

    @classmethod
    def enclose(cls,spheres,groups,expansion=1.1,desired_radius=None):
        """Enlarge toward a common radius while retaining foreign-ball clearance.

        The achieved radii can differ if requested sizes exceed local clearance.
        Returned metadata records these limits. Strain/pose changes require rebuild.
        """
        if not np.isfinite(expansion) or expansion<1. or (desired_radius is not None and (not np.isfinite(desired_radius) or desired_radius<=0)):
            raise ValueError("Finite expansion >=1 and positive desired radius required")
        groups=np.asarray(groups)
        if groups.shape!=(len(spheres.radii),) or not np.issubdtype(groups.dtype,np.integer) or not np.array_equal(np.unique(groups),np.arange(len(np.unique(groups)))):
            raise ValueError("Contiguous integer support ownership required")
        centers=[]; radii=[]; receipts=[]
        for group in np.unique(groups):
            owned=groups==group; center=spheres.centers[owned].mean(axis=0)
            distance=np.linalg.norm(spheres.centers-center,axis=1)
            minimum=float(np.max(distance[owned]+spheres.radii[owned]))
            guard=1024*np.finfo(float).eps*max(1.,minimum,float(np.max(np.abs(center))))
            maximum=float(np.min(distance[~owned]-spheres.radii[~owned]))-guard if np.any(~owned) else np.inf
            requested=max(minimum*expansion,minimum+guard,0. if desired_radius is None else desired_radius)
            achieved=min(requested,maximum)
            if achieved<=minimum+guard/2:
                raise ValueError("No foreign-free spherical enclosure; use general hierarchy fallback")
            centers.append(center); radii.append(achieved)
            receipts.append(dict(group=int(group),minimum=minimum,requested=requested,achieved=achieved,clearance_limit=maximum,limited=achieved<requested))
        scene=cls(spheres,groups,centers,radii); scene.expansion_receipt=receipts
        return scene

    def trace(self,source,max_bounces=64,routing=None):
        if not isinstance(max_bounces,(int,np.integer)) or max_bounces<1 or np.any(source.initial_bounces>1) or np.any(source.initial_bounces>max_bounces):
            raise ValueError("Positive cap and uncollided/first-reflected source required")
        if np.any(source.last_atom>=len(self.spheres.radii)) or np.any(source.last_atom < -1):
            raise ValueError("Invalid source atom identity")
        start=time.perf_counter()
        targets=np.empty(0,dtype=np.int64); inside=np.empty(0,dtype=np.bool_)
        if routing is not None:
            routing.validate(self,source.origins,source.directions)
            targets=routing.targets; inside=routing.inside
        values=trace_interfaces(source.origins,source.directions,source.last_atom,source.initial_bounces,
            self.spheres.centers,self.spheres.radii,self.packed_ids,self.offsets,self.centers,self.radii,
            max_bounces,routing is not None,targets,inside)
        positions,outgoing,escaped,unresolved,bounces,ledger,counts=values
        names=["interface_tests","sphere_tests","interface_visits","empty_visits","inside_entries","return_visits","local_multiple_collision_calls","overlap_handoffs"]
        metrics={name:int(counts[:,i].sum()) for i,name in enumerate(names)}
        metrics.update(query_s=time.perf_counter()-start,backend="exact spherical interface composition",initial_routing_reused=0 if routing is None else len(source.weights))
        return EHSSResult(positions,outgoing,escaped,unresolved,bounces,ledger,source,metrics)


class CompiledInterfaceRouting:
    """Unique initial interface on common, exact world-space packet coordinates.

    Reusable for new flux/initial-direction coefficients on this fixed basis.
    Scattering and all later routes are still evaluated; this caches no itinerary.
    """
    def __init__(self,scene,origins,directions):
        start=time.perf_counter()
        self.origins=np.array(origins,dtype=float,copy=True); self.directions=np.array(directions,dtype=float,copy=True)
        if self.origins.ndim!=2 or self.origins.shape[1]!=3 or self.directions.shape!=self.origins.shape or not np.isfinite(self.origins).all() or not np.isfinite(self.directions).all() or not np.allclose(np.linalg.norm(self.directions,axis=1),1.,atol=1e-12,rtol=1e-12):
            raise ValueError("Finite common packet positions and unit directions required")
        self.signature=scene.signature
        self.targets,self.inside=compile_initial_routes(self.origins,self.directions,scene.centers,scene.radii)
        for value in [self.origins,self.directions,self.targets,self.inside]:
            value.setflags(write=False)
        self.compile_s=time.perf_counter()-start

    def validate(self,scene,origins,directions):
        if scene.signature!=self.signature or not np.array_equal(origins,self.origins) or not np.array_equal(directions,self.directions):
            raise ValueError("Compiled routing requires unchanged geometry and packet coordinates")

    @property
    def payload_bytes(self):
        return sum(x.nbytes for x in [self.origins,self.directions,self.targets,self.inside])
