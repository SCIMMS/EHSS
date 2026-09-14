"""Reuse finite owned prefixes using per-rival segment clearance bounds.

For a straight segment, moving both endpoints by at most e moves every point
by at most e. A rival sphere cannot meet the new segment if its old clearance
exceeds e + relative center motion + positive radius growth. A floating guard
is used; this is a numerical conservative check, not interval arithmetic.
Terminal empty flights are never certified here: the existing apply kernel
always returns to full nearest search after the stored prefix.
"""
from dataclasses import dataclass
import time
import numpy as np
from numba import njit
from .ehss_reference import nearest_all,entry_distance
from .ehss_guarded_intrinsic_response import reflect
from .ehss_sparse_path_response import readonly
from .ehss_factored_local_samples import BoundSamples


@njit(cache=True)
def compile_prefix(point,direction,last,centers,radii,cap):
    paths=np.full(cap-1,-1,dtype=np.int64);starts=np.empty((cap-1,3));ends=starts.copy()
    p=point.copy();d=direction.copy();previous=last;length=0;tests=0;capped=False
    for step in range(cap):
        atom,distance,nt=nearest_all(p,d,previous,centers,radii);tests+=nt
        if atom<0:break
        if step==cap-1:capped=True;break
        paths[length]=atom;starts[length]=p;ends[length]=p+distance*d
        reflect(p,d,atom,distance,centers,radii);previous=atom;length+=1
    gaps=np.full((length,len(radii)),np.inf);clearance_tests=0;previous=last
    for step in range(length):
        a=starts[step];b=ends[step];v=b-a;vv=np.dot(v,v)
        for atom in range(len(radii)):
            if atom==previous or atom==paths[step]:continue
            parameter=0. if vv==0 else min(1.,max(0.,np.dot(centers[atom]-a,v)/vv))
            delta=centers[atom]-(a+parameter*v)
            gaps[step,atom]=np.sqrt(np.dot(delta,delta))-radii[atom];clearance_tests+=1
        previous=paths[step]
    return paths[:length].copy(),starts[:length].copy(),ends[:length].copy(),gaps,capped,tests,clearance_tests


@njit(cache=True)
def prefix_is_clear(point,direction,last,path,starts,ends,gaps,old_radii,current_centers,current_radii,
                    rotation,translation,center_shifts):
    p=point.copy();d=direction.copy();anchor_shift=center_shifts[last];checks=0;gap_checks=0
    motion=np.empty(len(current_radii))
    for atom in range(len(current_radii)):
        delta=center_shifts[atom]-anchor_shift
        motion[atom]=np.sqrt(np.dot(delta,delta))+max(0.,current_radii[atom]-old_radii[atom])
    for step,atom in enumerate(path):
        distance=entry_distance(p,d,current_centers[atom],current_radii[atom]);checks+=1
        if not np.isfinite(distance):return False,checks,gap_checks
        end=p+distance*d
        old_start=starts[step]@rotation.T+translation+anchor_shift
        old_end=ends[step]@rotation.T+translation+anchor_shift
        a=p-old_start;b=end-old_end
        displacement=max(np.sqrt(np.dot(a,a)),np.sqrt(np.dot(b,b)))
        scale=max(1.,np.max(np.abs(current_centers)),np.max(current_radii),np.max(np.abs(old_start)),np.max(np.abs(old_end)))
        guard=512*np.finfo(np.float64).eps*scale
        for rival in range(len(current_radii)):
            if not np.isfinite(gaps[step,rival]):continue
            gap_checks+=1
            if gaps[step,rival]<=displacement+motion[rival]+guard:return False,checks,gap_checks
        reflect(p,d,atom,distance,current_centers,current_radii)
    return True,checks,gap_checks


@dataclass(frozen=True)
class PrefixGeometry:
    centers: np.ndarray
    radii: np.ndarray


@dataclass(frozen=True)
class PrefixCertificate:
    geometry: PrefixGeometry
    path: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    gaps: np.ndarray
    capped: bool


@dataclass
class PartialSampleResponse:
    bank: object
    bound: BoundSamples
    certificates: tuple
    receipt: dict


def bind_partial_samples(bank,model,previous=None,defer_empty=False):
    if not np.array_equal(model.tree.groups,bank.groups):raise ValueError('Stable physical ownership required')
    if previous is not None and previous.bank is not bank:raise ValueError('Different material bank')
    tick=time.perf_counter();world=model.spheres.centers;radii=model.spheres.radii;n=len(bank.lasts)
    points=np.empty((n,3));directions=points.copy();tags=points.copy()
    paths=np.full((n,bank.cap-1),-1,dtype=np.int64);lengths=np.zeros(n,dtype=np.int64);certificates=[None]*n
    reused=[];deferred=[];refreshed=[];same_path=[];checks=gap_checks=compile_tests=clearance_tests=0
    verify_s=compile_s=0.
    for group,(atoms,rows) in enumerate(zip(bank.atoms,bank.rows)):
        reference=bank.reference[atoms];target=world[atoms]
        if np.array_equal(target-target[0],reference-reference[0]):rotation=np.eye(3)
        else:
            u,_,vh=np.linalg.svd((reference-reference.mean(axis=0)).T@(target-target.mean(axis=0)))
            correction=np.eye(3);correction[2,2]=1. if np.linalg.det(u@vh)>=0 else -1.
            rotation=(u@correction@vh).T
        translation=target[0];local=(target-translation)@rotation
        geometry=PrefixGeometry(readonly(local.copy()),readonly(radii[atoms].copy()));shift_cache={}
        points[rows]=world[bank.lasts[rows]]+radii[bank.lasts[rows],None]*(bank.normals[rows]@rotation.T)
        directions[rows]=bank.directions[rows]@rotation.T;tags[rows]=bank.tags[rows]@rotation.T
        for row in rows:
            last=int(np.searchsorted(atoms,bank.lasts[row]));old=None if previous is None else previous.certificates[row]
            keep=False
            if old is not None and len(old.path)==0 and defer_empty:
                keep=True;deferred.append(int(row))
            elif old is not None and len(old.path)>0:
                start=time.perf_counter();key=id(old.geometry)
                if key not in shift_cache:shift_cache[key]=target-(old.geometry.centers@rotation.T+translation)
                keep,nt,ng=prefix_is_clear(points[row],directions[row],last,old.path,old.starts,old.ends,old.gaps,old.geometry.radii,
                    target,radii[atoms],rotation,translation,shift_cache[key])
                checks+=int(nt);gap_checks+=int(ng);verify_s+=time.perf_counter()-start
                if keep:reused.append(int(row))
            if keep:certificate=old
            else:
                start=time.perf_counter()
                route,a,b,gaps,capped,nt,ng=compile_prefix(points[row],directions[row],last,target,radii[atoms],bank.cap)
                if old is not None and np.array_equal(route,old.path):route=old.path;same_path.append(int(row))
                certificate=PrefixCertificate(geometry,readonly(route),readonly((a-translation)@rotation),readonly((b-translation)@rotation),readonly(gaps),bool(capped))
                compile_tests+=int(nt);clearance_tests+=int(ng);compile_s+=time.perf_counter()-start;refreshed.append(int(row))
            certificates[row]=certificate;lengths[row]=len(certificate.path)
            if len(certificate.path):paths[row,:len(certificate.path)]=atoms[certificate.path]
    receipt=dict(bind_s=time.perf_counter()-tick,verify_s=verify_s,compile_s=compile_s,certificate_candidate_tests=checks,
        certificate_gap_checks=gap_checks,compile_sphere_tests=compile_tests,compile_clearance_tests=clearance_tests,
        reused_positive_rows=reused,deferred_empty_rows=deferred,refreshed_rows=refreshed,refreshed_same_path_rows=same_path,
        positive_rows=int(np.count_nonzero(lengths)),samples=n,defer_empty=defer_empty,
        certificate_scope='Finite owned prefix only; current physical final search remains mandatory.',
        geometry_versions=len({id(c.geometry) for c in certificates}))
    bound=BoundSamples(bank,model,tuple(),points,directions,tags,paths,lengths,receipt)
    return PartialSampleResponse(bank,bound,tuple(certificates),receipt)
