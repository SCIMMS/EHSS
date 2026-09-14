"""Causal novel-port learning from exact suffixes of already desired queries.

Old material batches keep their own geometry epochs. Newly learned batches
store observed finite owned prefixes; prefix ends remain UNKNOWN at apply time.
Only currently missing/inactive port cells are added, retaining all observations
within selected cells. These are fallback-conditioned occupation samples, not
an unbiased universal boundary quadrature. Coupling rebuild costs are separate.
"""
from copy import copy
from dataclasses import dataclass
import time
import numpy as np
from .ehss_empirical_field import geometry_key
from .ehss_factored_local_samples import MaterialSamples,BoundSamples,LocalColumn
from .ehss_material_collision_field import material_keys
from .ehss_sparse_path_response import record_from_paths,readonly


@dataclass(frozen=True)
class FallbackBatch:
    parent_bank: object
    geometry: str
    source_count: int
    cap: int
    suffixes: tuple
    query_results: tuple
    receipt: dict


def capture_fallback_queries(field,sources,cap=128,engine=None):
    """Capture the suffixes field.solve already computes; perform no extra trace."""
    sources=tuple(sources)
    if not sources: raise ValueError('Desired query sources required')
    tick=time.perf_counter(); suffixes=[]; physical=field.model
    class Recorder:
        def __getattr__(self,name): return getattr(physical,name)
        def trace(self,source,max_bounces=128):
            result=physical.trace(source,max_bounces); suffixes.append(result); return result
    observer=copy(field); observer.model=Recorder()
    results=tuple(observer.solve(source,cap,engine) for source in sources)
    return FallbackBatch(field.bank,geometry_key(physical.spheres),len(sources),cap,tuple(suffixes),results,
        dict(batch_s=time.perf_counter()-tick,desired_query_s=sum(r['query_s'] for r in results),
            suffix_batches=len(suffixes),exact_suffix_rays=sum(len(r.bounces) for r in suffixes),
            additional_nearest_searches=0,additional_source_samples=0,
            provenance='Exact unknown-first-port suffixes from existing desired hybrid queries.'))


@dataclass
class BatchBoundSamples(BoundSamples):
    parts: tuple


class BatchedMaterialSamples(MaterialSamples):
    def __init__(self,components):
        self.components=tuple(components); base=self.components[0]
        for name in ('cap','groups','reference','radii','atoms'): setattr(self,name,getattr(base,name))
        for other in self.components:
            if other.cap!=self.cap or not np.array_equal(other.groups,self.groups) or not np.array_equal(other.reference,self.reference):
                raise ValueError('Material batches require common ownership and reference axes')
        for name in ('lasts','owners','mass','normals','directions','tags'):
            setattr(self,name,readonly(np.concatenate([getattr(p,name) for p in self.components])))
        self.rows=tuple(readonly(np.flatnonzero(self.owners==g)) for g in range(len(self.atoms)))

    def assemble(self,model,parts,started=None):
        tick=time.perf_counter() if started is None else started
        arrays=[np.concatenate([getattr(p,name) for p in parts]) for name in ('points','directions','tags','paths','lengths')]
        reused=[]
        for group in range(len(self.atoms)):
            active=[p for p in parts if len(p.bank.rows[group])]
            if active and all(group in p.receipt['reused_groups'] for p in active): reused.append(group)
        receipt=dict(bind_s=time.perf_counter()-tick,local_compile_s=sum(p.receipt['local_compile_s'] for p in parts),
            local_sphere_tests=sum(p.receipt['local_sphere_tests'] for p in parts),reused_groups=reused,
            batches=len(parts),samples=len(self.mass),separate_shape_epochs=True)
        return BatchBoundSamples(self,model,tuple(),*arrays,receipt,tuple(parts))

    def bind(self,model,previous=None,tolerance=1e-10):
        if previous is not None and previous.bank is not self: raise ValueError('Response belongs to another bank')
        tick=time.perf_counter(); old=(None,)*len(self.components) if previous is None else previous.parts
        if len(old)!=len(self.components): raise ValueError('Material batch count changed')
        parts=tuple(bank.bind(model,prior,tolerance) for bank,prior in zip(self.components,old))
        return self.assemble(model,parts,tick)


def learn_missing_ports(field,bound,observed,batch,max_ports=None):
    """Append selected novel cells using replayed observations, without discovery.

Return a bound bank, one-event observation dictionary and receipt. The returned
observations suffice for MaterialCollisionField; they are not first-exit query
results. Current foreign coupling is rebuilt by that constructor as usual.
"""
    tick=time.perf_counter(); bank=bound.bank; model=bound.model; spheres=model.spheres
    if field.bank is not bank or batch.parent_bank is not bank: raise ValueError('Learning batch uses a different parent bank')
    if geometry_key(spheres)!=batch.geometry or batch.geometry!=field.geometry: raise ValueError('Learning geometry changed')
    if batch.cap!=bank.cap: raise ValueError('Learning and material collision budgets differ')
    if max_ports is not None and (not isinstance(max_ports,int) or max_ports<0): raise ValueError('Nonnegative port limit required')
    records=[]; follow=[]; routes=[]; route_capped=[]; tails=[]; tags=[]
    for result in batch.suffixes:
        source=result.source
        p,d,last,mass=record_from_paths(source.origins,source.directions,source.weights/batch.source_count,
            source.initial_bounces,result.collider_ids,result.bounces,spheres.centers,spheres.radii)
        ends=np.cumsum(result.bounces)[result.bounces>0]-1
        if len(ends):
            if not np.array_equal(p[ends],result.positions[result.bounces>0]) or not np.array_equal(d[ends],result.outgoing[result.bounces>0]):
                raise ValueError('Captured path and physical endpoint disagree')
        cursor=0
        for ray,count in enumerate(result.bounces):
            count=int(count); path=result.collider_ids[ray,:count]
            for j,atom in enumerate(path):
                follow.append(int(path[j+1]) if j+1<count else -1)
                end=j+1
                while end<count and bank.groups[path[end]]==bank.groups[atom]: end+=1
                routes.append(path[j+1:end].copy())
                route_capped.append(bool(end==count and result.unresolved[ray]))
                tails.append(bool(j+1==count and result.unresolved[ray]))
            cursor+=count
        records.append((p,d,last,mass)); tags.append(np.repeat(source.incoming,result.bounces,axis=0))
    if records:
        p,d,last,mass=[np.concatenate([r[k] for r in records]) for k in range(4)]; incoming=np.concatenate(tags)
    else:
        p=np.empty((0,3)); d=p.copy(); incoming=p.copy(); last=np.empty(0,dtype=np.int64); mass=np.empty(0)
    keys=material_keys(p,d,last,spheres.centers,bank.groups,field.rotations,field.bins)
    novelty={}; candidates=[]
    for row,key in enumerate(keys):
        key=tuple(key); state=field.index.get(key)
        if state is None or not field.active[state]:
            novelty[key]=novelty.get(key,0.)+float(mass[row]); candidates.append((row,key))
    ranked=sorted(novelty,key=lambda key:(-novelty[key],key)); selected=set(ranked if max_ports is None else ranked[:max_ports])
    keep=np.array([row for row,key in candidates if key in selected],dtype=np.int64)
    receipt=dict(extraction_s=time.perf_counter()-tick,observed_events=len(last),novel_cells=len(novelty),selected_cells=len(selected),
        selected_events=len(keep),added_occupation_mass=float(mass[keep].sum()),additional_nearest_searches=0,
        additional_source_samples=0,policy='Missing active port cells ranked by observed fallback occupation mass; keep every observation within selected cells.',
        conditioning='Fallback-conditioned occupation quadrature, not unconditional boundary sampling.',
        old_bank_unchanged=True,legacy_geometry_epochs_preserved=True)
    if not len(keep): return bound,observed,receipt
    # The new batch uses the existing material axes, but its own current geometry
    # epoch. Old batches are never re-labelled as newly prepared by this append.
    part=MaterialSamples.__new__(MaterialSamples)
    for name in ('cap','groups','reference','radii','atoms'): setattr(part,name,getattr(bank,name))
    part.lasts=readonly(last[keep]); part.owners=readonly(bank.groups[part.lasts]); part.mass=readonly(mass[keep])
    normal=p[keep]-spheres.centers[part.lasts]; normal/=np.linalg.norm(normal,axis=1)[:,None]
    rotations=field.rotations[part.owners]
    part.normals=readonly(np.einsum('ni,nij->nj',normal,rotations)); part.directions=readonly(np.einsum('ni,nij->nj',d[keep],rotations)); part.tags=readonly(np.einsum('ni,nij->nj',incoming[keep],rotations))
    part.rows=tuple(readonly(np.flatnonzero(part.owners==g)) for g in range(len(bank.atoms)))
    paths=np.full((len(keep),bank.cap-1),-1,dtype=np.int64); lengths=np.zeros(len(keep),dtype=np.int64); columns=[]
    for i,row in enumerate(keep): paths[i,:len(routes[row])]=routes[row]; lengths[i]=len(routes[row])
    for group,(atoms,rows) in enumerate(zip(part.atoms,part.rows)):
        local=(spheres.centers[atoms]-spheres.centers[atoms[0]])@field.rotations[group]
        local_paths=np.full((len(rows),bank.cap-1),-1,dtype=np.int64); valid=paths[rows]>=0
        local_paths[valid]=np.searchsorted(atoms,paths[rows][valid])
        columns.append(LocalColumn(readonly(local),readonly(spheres.radii[atoms].copy()),readonly(local_paths),readonly(lengths[rows]),readonly(np.asarray(route_capped,dtype=bool)[keep][rows]),0))
    new_bound=BoundSamples(part,model,tuple(columns),p[keep],d[keep],incoming[keep],paths,lengths,
        dict(bind_s=0.,local_compile_s=0.,local_sphere_tests=0,reused_groups=[],rebuilt_groups=[],observed_prefix_rows=len(keep)))
    old_parts=bound.parts if isinstance(bound,BatchBoundSamples) else (bound,)
    merged=BatchedMaterialSamples(tuple(p.bank for p in old_parts)+(part,)); aggregate=merged.assemble(model,old_parts+(new_bound,))
    # Historical preparation receipts on retained parts are not current work.
    aggregate.receipt.update(local_compile_s=0.,local_sphere_tests=0,reused_legacy_batches=len(old_parts),
        reused_groups=[g for g in range(len(part.atoms)) if not len(part.rows[g])])
    history=np.full((len(keep),bank.cap),-1,dtype=np.int64); history[:,0]=part.lasts
    nexts=np.asarray(follow,dtype=np.int64)[keep]; has_next=nexts>=0
    if np.any(has_next) and bank.cap<2: raise ValueError('Known successor exceeds collision cap')
    history[has_next,1]=nexts[has_next]
    # -4 is a known one-event successor, not an escape or cap terminal.
    terminal=np.where(has_next,-4,np.where(np.asarray(tails)[keep],-2,-1)); counts=1+has_next.astype(np.int64)
    event_observations=dict(history=np.concatenate((observed['history'],history)),counts=np.r_[observed['counts'],counts],
        terminal=np.r_[observed['terminal'],terminal],query_s=0.,observation_kind='Existing legacy observations plus captured one-event successors; not a full first-exit result.')
    receipt.update(total_s=time.perf_counter()-tick,total_samples=len(merged.mass),material_batches=len(merged.components),observed_prefix_steps=int(lengths.sum()))
    return aggregate,event_observations,receipt
