"""One-time body-local refinement of poorly shaped three-anchor supports.

Refinement changes only computational ownership. Physical atoms, their indices,
and the original rigid units are retained. The proposal is a geometric policy,
not a protein chemistry partitioner or an automatic per-frame update policy.
"""
from dataclasses import dataclass
import numpy as np
from .ehss_canonical import select_anchors


@dataclass
class SupportRefinement:
    groups: np.ndarray
    parent_support: np.ndarray
    diagnostics: list

    def poses(self,rotations,translations):
        """Both children inherit their original support's pose and local frame."""
        return np.asarray(rotations)[self.parent_support],np.asarray(translations)[self.parent_support]


def refine_thin_supports(local_centers,radii,groups,threshold=.15):
    """Propose a stable 2+1-anchor split when 2*triangle_area/Lmax**2 is small.

    A support may own more than three atoms; it is considered here only when
    canonical anchor selection returns three. The closest anchor pair stays
    together. Other atoms follow their nearest selected anchor, with a stable
    tie rule. Degenerate triples already handled by two-anchor completion are
    not forced to split. Acceptance based on total cost is a separate decision.
    """
    local=np.asarray(local_centers,dtype=float); radii=np.asarray(radii,dtype=float)
    groups=np.asarray(groups,dtype=np.int64)
    if local.shape!=(len(radii),3) or groups.shape!=(len(radii),) or not len(radii):
        raise ValueError("One finite local center, radius and owner per atom required")
    if not np.isfinite(local).all() or not np.isfinite(radii).all() or np.any(radii<=0):
        raise ValueError("Finite centers and positive radii required")
    unique=np.unique(groups)
    if not np.array_equal(unique,np.arange(len(unique))) or not np.isfinite(threshold) or not 0<threshold<1:
        raise ValueError("Contiguous owners and a shape threshold in (0,1) required")
    refined=np.empty_like(groups); parents=[]; diagnostics=[]
    for group in unique:
        atoms=np.flatnonzero(groups==group)
        selected,_=select_anchors(local[atoms],float(radii[atoms].max()))
        anchor_atoms=atoms[selected]; pieces=[atoms]; quality=None; pair=None
        if len(selected)==3:
            p=local[anchor_atoms]
            pairs=[(0,1),(0,2),(1,2)]
            length2=np.array([np.sum((p[a]-p[b])**2) for a,b in pairs])
            quality=float(np.linalg.norm(np.cross(p[1]-p[0],p[2]-p[0]))/length2.max())
            if quality<threshold:
                pair=pairs[int(np.argmin(length2))]
                owner=np.argmin(np.sum((local[atoms,None,:]-p[None,:,:])**2,axis=2),axis=1)
                together=np.isin(owner,pair)
                pieces=[atoms[together],atoms[~together]]
                assert all(len(piece)>0 for piece in pieces)
        child_ids=[]
        for piece in pieces:
            child_id=len(parents); child_ids.append(child_id)
            refined[piece]=child_id; parents.append(int(group))
        diagnostics.append(dict(original_support=int(group),anchor_atoms=anchor_atoms.tolist(),
            triangle_quality=quality,split=len(pieces)>1,
            paired_anchor_atoms=None if pair is None else anchor_atoms[list(pair)].tolist(),
            children=child_ids,physical_atoms=[piece.tolist() for piece in pieces]))
    parents=np.asarray(parents,dtype=np.int64)
    assert np.array_equal(parents[refined],groups)
    refined.setflags(write=False); parents.setflags(write=False)
    return SupportRefinement(refined,parents,diagnostics)
