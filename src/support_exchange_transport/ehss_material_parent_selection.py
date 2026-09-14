"""A nonoverlapping cut of the existing support tree, selected before held-out MD.

Score is observed internal collision-leg mass, not a predicted wall-clock gain.
Eligible parents must fit the observed initial material geometry tolerance.
Leaves always remain available, even when a leaf itself is strained.
"""
import time
import numpy as np
from .ehss_sparse_path_response import readonly


def anchored_shape_residual(reference,target):
    a=reference-reference[0];b=target-target[0]
    if np.array_equal(a,b):return 0.
    u,_,vh=np.linalg.svd((reference-reference.mean(axis=0)).T@(target-target.mean(axis=0)))
    correction=np.eye(3);correction[2,2]=1. if np.linalg.det(u@vh)>=0 else -1.
    rotation=(u@correction@vh).T
    return float(np.linalg.norm(b@rotation-a,axis=1).max())


def select_material_parents(tree,results,selection_frames,max_atoms=48,tolerance=.05,require_stable=True):
    tick=time.perf_counter();frames=np.asarray(selection_frames,float);n=len(tree.radii)
    if frames.ndim!=3 or frames.shape[1:]!=(n,3) or not len(frames) or not np.isfinite(frames).all():
        raise ValueError('Finite selection frames with fixed physical atom order required')
    if not results or not isinstance(max_atoms,int) or max_atoms<1 or not np.isfinite(tolerance) or tolerance<0:
        raise ValueError('Calibration results, positive size cap and nonnegative tolerance required')
    if not np.array_equal(frames[0],tree.world):raise ValueError('First selection frame must be the tree reference geometry')
    pairs={}
    for result in results:
        for ray,count in enumerate(result.bounces):
            w=float(result.source.weights[ray])/len(results)
            if not np.isfinite(w) or w<0:raise ValueError('Finite nonnegative calibration weights required')
            path=result.collider_ids[ray,:int(count)]
            if np.any(path<0) or np.any(path>=n):raise ValueError('Invalid physical calibration path')
            for a,b in zip(path[:-1],path[1:]):
                key=(int(a),int(b));pairs[key]=pairs.get(key,0.)+w
    pairs_array=np.array(list(pairs),dtype=np.int64).reshape(-1,2);weights=np.array(list(pairs.values()))
    atoms={};scores={};residuals={};eligible={};chosen={};objective={}
    for node in range(len(tree.left)-1,-1,-1):
        group=int(tree.leaf_group[node])
        if group>=0:ids=np.sort(tree.ids[tree.offsets[group]:tree.offsets[group+1]])
        else:ids=np.sort(np.r_[atoms[int(tree.left[node])],atoms[int(tree.right[node])]])
        atoms[node]=ids
        owned=np.zeros(n,dtype=bool);owned[ids]=True
        score=float(weights[owned[pairs_array[:,0]]&owned[pairs_array[:,1]]].sum())
        scores[node]=score
        if len(ids)<=max_atoms:
            residual=max(anchored_shape_residual(frames[0,ids],frame[ids]) for frame in frames)
        else:residual=None
        residuals[node]=residual;allowed=group>=0 or (len(ids)<=max_atoms and (not require_stable or residual<=tolerance))
        eligible[node]=allowed
        if group>=0:chosen[node]=(node,);objective[node]=score
        else:
            a=int(tree.left[node]);b=int(tree.right[node]);child_score=objective[a]+objective[b]
            # Preserve children when a merge has no observed additional internal work.
            if allowed and score>child_score+1e-12*max(1.,sum(pairs.values())):
                chosen[node]=(node,);objective[node]=score
            else:chosen[node]=chosen[a]+chosen[b];objective[node]=child_score
    selected=chosen[0];groups=np.full(n,-1,dtype=np.int64)
    for group,node in enumerate(selected):
        if np.any(groups[atoms[node]]>=0):raise AssertionError('Selected supports overlap')
        groups[atoms[node]]=group
    if np.any(groups<0):raise AssertionError('Selected supports fail to cover physical atoms')
    leaf_score=sum(scores[node] for node in scores if tree.leaf_group[node]>=0)
    receipt=dict(select_s=time.perf_counter()-tick,max_atoms=max_atoms,tolerance=tolerance,require_stable=require_stable,
        selection_frame_count=len(frames),selected_nodes=selected,groups=len(selected),parent_groups=sum(tree.leaf_group[node]<0 for node in selected),
        sizes=[len(atoms[node]) for node in selected],internal_leg_mass=objective[0],chemical_leaf_internal_leg_mass=leaf_score,
        added_internal_leg_mass=objective[0]-leaf_score,total_observed_leg_mass=float(weights.sum()),
        selected=[dict(node=node,atoms=len(atoms[node]),score=scores[node],max_shape_residual=residuals[node],leaf=bool(tree.leaf_group[node]>=0)) for node in selected],
        candidates=[dict(node=node,atoms=len(atoms[node]),score=scores[node],max_shape_residual=residuals[node],eligible=eligible[node]) for node in range(len(tree.left))],
        score_is_cost_prediction=False,held_out_frames_used=False)
    return readonly(groups),receipt
