"""Reuse completed query paths as causal input for a later response refresh."""
from dataclasses import dataclass
import time
import numpy as np
from .ehss_selected_node_refresh import FieldObservations,geometry_key,rank_missing_nodes
from .ehss_sparse_path_response import readonly,record_from_paths


@dataclass(frozen=True)
class CompletedQueries:
    results: tuple
    geometry: str
    cap: int
    mode: str
    checked_owned: bool
    receipt: dict


def capture_queries(engine,sources,cap=128,mode='indexed',verify_owned=False):
    """Return ordinary query results plus a geometry tag for later observation.

    Results are the original outputs and must not be mutated before extraction.
    Approximate query paths are explicitly distinguished from checked/off paths.
    """
    sources=tuple(sources)
    if not sources:raise ValueError('Query sources required')
    start=time.perf_counter();key=geometry_key(engine.bound.model);query_s=0.;results=[]
    for source in sources:
        tick=time.perf_counter();results.append(engine.trace(source,cap,verify_owned=verify_owned,mode=mode));query_s+=time.perf_counter()-tick
    elapsed=time.perf_counter()-start
    return CompletedQueries(tuple(results),key,cap,mode,bool(verify_owned),
        dict(batch_s=elapsed,query_s=query_s,capture_overhead_s=elapsed-query_s,source_ensembles=len(sources)))


def observations_from_queries(model,batch):
    """Replay recorded reflections, with no source generation or nearest search.

    A query-reported next collider need not be the true nearest collider if its
    response used an approximate candidate itinerary. That provenance is kept.
    Current output positions/directions must agree with the recorded path.
    """
    tick=time.perf_counter()
    if batch.geometry!=geometry_key(model):raise ValueError('Query observation geometry changed')
    records=[];following=[];tails=[];n=len(batch.results)
    if not n:raise ValueError('Completed queries required')
    for result in batch.results:
        source=result.source;counts=result.bounces
        if np.any(counts<source.initial_bounces) or np.any(counts>batch.cap):raise ValueError('Invalid recorded bounce count')
        p,d,last,mass=record_from_paths(source.origins,source.directions,source.weights/n,source.initial_bounces,
            result.collider_ids,counts,model.spheres.centers,model.spheres.radii)
        ends=np.cumsum(counts)[counts>0]-1
        if len(ends):
            if not np.array_equal(p[ends],result.positions[counts>0]) or not np.array_equal(d[ends],result.outgoing[counts>0]):
                raise ValueError('Query path and final physical state disagree')
        nexts=np.full(len(last),-1,dtype=np.int64)
        if len(last):nexts[:-1]=last[1:];nexts[ends]=-1
        records.append((p,d,last,mass));following.append(nexts);tails.append(result.tail_bound)
    points,directions,lasts,mass=[np.concatenate([r[k] for r in records]) for k in range(4)];nexts=np.concatenate(following)
    exact=batch.mode=='off' or batch.checked_owned
    return FieldObservations(*(readonly(a) for a in (points,directions,lasts,nexts,mass)),batch.geometry,
        dict(observation_s=time.perf_counter()-tick,physical_records=len(lasts),known_continuations=int(np.count_nonzero(nexts>=0)),
            source_ensembles=n,cap=batch.cap,source_tail_bounds=tails,cap_terminal_next_unknown=True,
            provenance='existing_checked_or_off_query' if exact else 'existing_approximate_query',
            nearest_order_verified=exact,additional_nearest_searches=0,additional_source_samples=0,
            existing_query_s=batch.receipt['query_s'],capture_overhead_s=batch.receipt['capture_overhead_s'],
            weighting='Original physical query source weights divided by number of source ensembles; repeated visits remain repeated events.'))


def rank_query_missing_nodes(model,bound,observations,max_atoms=24):
    ranked,receipt=rank_missing_nodes(model,bound,observations,max_atoms)
    receipt['observation_provenance']=observations.receipt.get('provenance','exact_calibration')
    receipt['nearest_order_verified']=observations.receipt.get('nearest_order_verified',True)
    if not receipt['nearest_order_verified']:
        receipt['policy']='Query-reported next collider inside owner; missing mass times size. Approximate branch may omit a physical nearest collision; scores overlap across ancestors.'
    return ranked,receipt
