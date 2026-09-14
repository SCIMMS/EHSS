"""Discover current field requests and rebuild only selected node responses.

Frozen unselected cells are retained verbatim. Selected owners get a current
native-frame template; their old chart keys are not mixed with the new frame.
"""
from dataclasses import dataclass
import hashlib
import time
import numpy as np
from .ehss_sparse_path_response import SparsePathAtlas,readonly,record_from_paths,owned_paths
from .ehss_surface_response import SurfaceNodeGeometry,surface_key
from .ehss_response import node_geometry,ResponseLibrary


def geometry_key(model):
    h=hashlib.sha256(ResponseLibrary.topology_key(model.tree).encode())
    for value in (model.spheres.centers,model.spheres.radii):h.update(value.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class FieldObservations:
    points: np.ndarray
    directions: np.ndarray
    lasts: np.ndarray
    nexts: np.ndarray
    mass: np.ndarray
    geometry: str
    receipt: dict


def observe_field(model,sources,cap=128):
    sources=list(sources)
    if not sources:raise ValueError('Calibration sources required')
    tick=time.perf_counter();records=[];nexts=[];tail=[]
    for source in sources:
        result=model.trace(source,cap)
        records.append(record_from_paths(source.origins,source.directions,source.weights/len(sources),
            source.initial_bounces,result.collider_ids,result.bounces,model.spheres.centers,model.spheres.radii))
        for path,count in zip(result.collider_ids,result.bounces):
            if count:nexts.append(np.r_[path[1:count],-1])
        tail.append(result.tail_bound)
    points,directions,lasts,mass=[np.concatenate([r[k] for r in records]) for k in range(4)]
    following=np.concatenate(nexts).astype(np.int64) if nexts else np.empty(0,dtype=np.int64)
    assert len(following)==len(lasts)
    return FieldObservations(*(readonly(a) for a in (points,directions,lasts,following,mass)),geometry_key(model),
        dict(observation_s=time.perf_counter()-tick,physical_records=len(lasts),known_continuations=int(np.count_nonzero(following>=0)),
            source_ensembles=len(sources),cap=cap,source_tail_bounds=tail,
            weighting='Physical source weights divided by number of calibration ensembles; repeated visits are repeated events.',
            cap_terminal_next_unknown=True))


def rank_missing_nodes(model,bound,observations,max_atoms=24):
    """Rank missing first continuations by observed physical mass times size.

    Ancestors overlap: scores are scheduling heuristics, not additive savings or
    CCS error estimates. Selection uses calibration only, before held-out rays.
    """
    tick=time.perf_counter()
    if observations.geometry!=geometry_key(model):raise ValueError('Observation geometry changed')
    if bound.model is not model:raise ValueError('Current bound model required')
    t=model.tree;ranked=[];reasons=('new_owner','inactive_owner','missing_cell','missing_route')
    for node in range(len(t.left)):
        atoms,_,_,_,_=node_geometry(t,node)
        if len(atoms)<2 or len(atoms)>max_atoms:continue
        chosen=np.flatnonzero(np.isin(observations.lasts,atoms)&np.isin(observations.nexts,atoms))
        if not len(chosen):continue
        counts={r:0 for r in reasons};masses={r:0. for r in reasons};covered=0
        for i in chosen:
            if node not in bound.atlas.nodes:reason='new_owner'
            elif not bound.active[node]:reason='inactive_owner'
            else:
                last=int(observations.lasts[i]);rotation=bound.rotations[node]
                n=(observations.points[i]-model.spheres.centers[last])@rotation;n/=np.linalg.norm(n)
                d=observations.directions[i]@rotation
                if n@d<=1e-10:continue
                key=tuple(int(v) for v in surface_key(node,last,n,d,bound.atlas.bins))
                if key not in bound.table:reason='missing_cell'
                else:
                    root=bound.roots[bound.table[key]];lo,hi=bound.edge_starts[root:root+2]
                    if observations.nexts[i] in bound.edge_atoms[lo:hi]:covered+=1;continue
                    reason='missing_route'
            counts[reason]+=1;masses[reason]+=float(observations.mass[i])
        missing_mass=sum(masses.values())
        if missing_mass:
            ranked.append(dict(node=node,atoms=len(atoms),score=len(atoms)*missing_mass,missing_mass=missing_mass,
                reasons=counts,reason_mass=masses,known_owned_next_events=len(chosen),already_covered_first_events=covered))
    ranked.sort(key=lambda r:(-r['score'],r['node']))
    return ranked,dict(ranking_s=time.perf_counter()-tick,candidate_nodes=len(ranked),
        policy='Known next physical collider lies in owner; missing first-response mass times owner size. Overlapping ancestor scores are not additive.')


def choose_nodes(ranked,budget,seed=None):
    if not isinstance(budget,int) or budget<0:raise ValueError('Nonnegative node budget required')
    nodes=np.array([r['node'] for r in ranked],dtype=np.int64)
    if seed is not None:nodes=np.random.default_rng(seed).permutation(nodes)
    return tuple(int(n) for n in nodes[:budget])


def refresh_nodes(model,base,observations,nodes):
    """Compile current observed cells only in selected owners, preserving others.

    Unselected cells reserve their existing slots. Remaining global cell budget
    is available to the new cells ranked by the original weighted-route policy.
    Old selected-owner templates are replaced even if no positive route remains.
    """
    tick=time.perf_counter()
    if not base.prepared or ResponseLibrary.topology_key(model.tree)!=base.topology:raise ValueError('Prepared matching topology required')
    if observations.geometry!=geometry_key(model):raise ValueError('Observation geometry changed')
    nodes=tuple(sorted(set(int(n) for n in nodes)));tree=model.tree
    if any(n<0 or n>=len(tree.left) for n in nodes):raise ValueError('Invalid selected node')
    if not nodes:return base,dict(refresh_s=time.perf_counter()-tick,selected_nodes=[],retained_cells=len(base.keys),new_cells=0,
        compiled_states=0,owned_sphere_tests=0,payload_changed=False)
    selected=set(nodes);retained=[i for i,key in enumerate(base.keys) if int(key[0]) not in selected]
    if len(retained)>base.max_cells:raise ValueError('Existing cells exceed budget')
    new=SparsePathAtlas(**{k:getattr(base,k) for k in ('bins','max_cells','max_routes_per_cell','max_atoms','max_route','residual_tolerance')})
    new.topology=base.topology;new.nodes={n:g for n,g in base.nodes.items() if n not in selected}
    cells={};compiled_states=tests=capped=0;route_s=collection_s=0.
    for node in nodes:
        atoms,anchor,centers,rotation,translation=node_geometry(tree,node)
        if len(atoms)>base.max_atoms:raise ValueError('Selected owner exceeds atom budget')
        new.nodes[node]=SurfaceNodeGeometry(readonly(atoms),anchor,readonly(centers),readonly(tree.radii[atoms].copy()))
        chosen=np.flatnonzero(np.isin(observations.lasts,atoms))
        if not len(chosen):continue
        points=observations.points[chosen];lasts=observations.lasts[chosen];mass=observations.mass[chosen]
        local_p=(points-translation)@rotation;local_d=observations.directions[chosen]@rotation
        start=time.perf_counter();routes,lengths,cap_count,nt=owned_paths(local_p,local_d,lasts,atoms,centers,tree.radii[atoms],base.max_route)
        route_s+=time.perf_counter()-start;compiled_states+=len(chosen);tests+=int(nt);capped+=int(cap_count)
        start=time.perf_counter()
        for j in range(len(chosen)):
            n=(points[j]-model.spheres.centers[lasts[j]])@rotation;n/=np.linalg.norm(n)
            if n@local_d[j]<=1e-10:continue
            key=tuple(int(v) for v in surface_key(node,int(lasts[j]),n,local_d[j],base.bins))
            row=cells.setdefault(key,[0.,0,{}]);row[0]+=float(mass[j]);row[1]+=1
            if lengths[j]:
                route=tuple(int(a) for a in routes[j,:lengths[j]])
                w,visits=row[2].get(route,(0.,0));row[2][route]=(w+float(mass[j]),visits+1)
        collection_s+=time.perf_counter()-start
    ranked=[]
    for key,(visits_mass,visits,paths) in cells.items():
        if not paths:continue
        routes=sorted(paths,key=lambda r:(-paths[r][0]*len(r),r))[:base.max_routes_per_cell]
        weights=tuple(paths[r][0] for r in routes);score=len(new.nodes[key[0]].atoms)*sum(len(r)*w for r,w in zip(routes,weights))
        ranked.append((score,key,tuple(routes),weights,visits_mass,visits))
    ranked.sort(key=lambda row:(-row[0],row[1]));capacity=base.max_cells-len(retained);chosen=ranked[:capacity]
    combined=[(float(base.scores[i]),tuple(int(v) for v in base.keys[i]),base.routes[i],base.route_weights[i],float(base.cell_mass[i]),int(base.cell_visits[i])) for i in retained]+chosen
    combined.sort(key=lambda row:row[1])
    new.keys=readonly(np.array([r[1] for r in combined],dtype=np.int64).reshape(-1,7));new.routes=tuple(r[2] for r in combined)
    new.route_weights=tuple(r[3] for r in combined);new.cell_mass=readonly(np.array([r[4] for r in combined]))
    new.cell_visits=readonly(np.array([r[5] for r in combined],dtype=np.int64));new.scores=readonly(np.array([r[0] for r in combined]))
    new.nodes={node:g for node,g in new.nodes.items() if node in set(new.keys[:,0])};new.prepared=True
    new.receipt=dict(refresh_s=time.perf_counter()-tick,selected_nodes=list(nodes),retained_cells=len(retained),new_cells=len(chosen),
        removed_selected_cells=len(base.keys)-len(retained),compiled_states=compiled_states,owned_sphere_tests=tests,capped_owned_prefixes=capped,
        owned_routes_s=route_s,collection_s=collection_s,positive_new_cells_before_budget=len(ranked),new_cell_capacity=capacity,
        cells=len(new.keys),parent_cells=sum(tree.leaf_group[k[0]]<0 for k in new.keys),payload_changed=True,
        preservation='Unselected keys, routes, weights and node templates retained; selected owner replaced in its current native frame.',
        budget_policy='Retained cells reserve capacity; rank new cells only within remaining global cell budget.',
        operator_scope='Observed candidate path prefixes, not averaged field response or continuous chart certificate.')
    return new,new.receipt
