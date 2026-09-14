"""Explicit heavy-atom PDB selection and geometric hard-ball diagnostics.

Radii are a named van-der-Waals plus probe scenario, NOT fitted EHSSrot-Siu
parameters. No hydrogen addition, protonation, force field, or bond inference.
The residue grouping supplies stable ordered ownership, not a claim of rigidity.
"""
from dataclasses import dataclass
from collections import Counter
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
from .inside_out_pa import Spheres

# C/N/O/S values tabulated by Mantina et al., JPC A 2009, Table 12 (Bondi
# values for these elements); helium 1.40 A. Only used elements are admitted.
VDW_ANGSTROM={'C':1.70,'N':1.55,'O':1.52,'S':1.80}


@dataclass
class MolecularGeometry:
    centers: np.ndarray
    elements: tuple
    atom_keys: tuple
    residue_keys: tuple
    groups: np.ndarray
    selection: dict

    def spheres(self,probe_radius=1.4):
        if not np.isfinite(probe_radius) or probe_radius<0:
            raise ValueError('Nonnegative finite probe radius required')
        radii=np.array([VDW_ANGSTROM[e]+probe_radius for e in self.elements])
        return Spheres(self.centers,radii,deduplicate=False)


def read_pdb_heavy(path,chain='A',altloc='A'):
    rows=[];seen=set();residues={};skipped=Counter()
    for line in path.read_text(encoding='ascii').splitlines():
        record=line[:6].strip()
        if record=='ENDMDL':break
        if record!='ATOM':
            if record=='HETATM':skipped['hetero_atoms']+=1
            continue
        if line[21:22]!=chain:skipped['other_chain']+=1;continue
        alternate=line[16:17].strip()
        if alternate and alternate!=altloc:skipped['other_altloc']+=1;continue
        element=line[76:78].strip().upper()
        if element in ('H','D'):skipped['hydrogens']+=1;continue
        if element not in VDW_ANGSTROM:raise ValueError(f'Unsupported or missing explicit element: {element!r}')
        occupancy=float(line[54:60])
        if not np.isfinite(occupancy):raise ValueError('Finite atom occupancy required')
        if occupancy<=0:skipped['nonpositive_occupancy']+=1;continue
        residue=(chain,int(line[22:26]),line[26:27].strip(),line[17:20].strip())
        key=(*residue,line[12:16].strip())
        if key in seen:raise ValueError(f'Ambiguous selected atom identity: {key}')
        xyz=[float(line[a:a+8]) for a in (30,38,46)]
        if not np.isfinite(xyz).all():raise ValueError('Finite atom coordinates required')
        seen.add(key);residues.setdefault(residue,len(residues))
        rows.append((xyz,element,key,residues[residue]))
    if not rows:raise ValueError('Selection contains no supported heavy atoms')
    centers=np.array([r[0] for r in rows])
    if len(np.unique(centers,axis=0))!=len(centers):raise ValueError('Coincident selected atom centers require explicit ownership handling')
    return MolecularGeometry(centers,tuple(r[1] for r in rows),tuple(r[2] for r in rows),tuple(residues),
                             np.array([r[3] for r in rows],dtype=np.int64),
                             dict(chain=chain,model='first',altloc=altloc,record='ATOM',hydrogen_policy='exclude; do not add',
                                  atoms=len(rows),residues=len(residues),elements=dict(Counter(r[1] for r in rows)),skipped=dict(skipped)))


def boundary_diagnostics(molecule,spheres):
    distance=cdist(spheres.centers,spheres.centers)
    gap=distance-spheres.radii[:,None]-spheres.radii[None,:]
    pairs=np.triu_indices(len(spheres.radii),1)
    overlap=gap<0;np.fill_diagonal(overlap,False)
    components,labels=connected_components(overlap,directed=False)
    envelopes=[]
    for group in range(len(molecule.residue_keys)):
        own=molecule.groups==group;foreign=~own
        center=spheres.centers[own].mean(axis=0)
        radius=float(np.max(np.linalg.norm(spheres.centers[own]-center,axis=1)+spheres.radii[own]))*(1+1e-9)
        clearance=np.linalg.norm(spheres.centers[foreign]-center,axis=1)-radius-spheres.radii[foreign]
        envelopes.append(dict(group=group,center=center.tolist(),radius=radius,foreign_intersections=int(np.count_nonzero(clearance<=0)),
                              min_foreign_clearance=float(clearance.min(initial=np.inf))))
    return dict(pair_gap_quantiles=dict(zip(['min','p01','p05','median'],map(float,np.quantile(gap[pairs],[0,.01,.05,.5])))),
                overlapping_atom_pairs=int(overlap[pairs].sum()),overlap_components=int(components),
                largest_overlap_component=int(np.bincount(labels).max()),residue_envelopes=envelopes,
                foreign_free_residue_envelopes=sum(e['foreign_intersections']==0 for e in envelopes))


def collision_tail(result,cutoffs):
    """Weighted omission bound; cap survivors remain unresolved, never escaped.

For omitted paths the readout is in [0,2w]. These are bounds for the sampled
measure; absence of sampled survivors does not certify a population cutoff.
"""
    cap=result.collider_ids.shape[1]
    rows=[]
    for k in cutoffs:
        if not isinstance(k,int) or k<1 or k>cap:raise ValueError('Cutoffs must lie within the traced collision budget')
        tail=(result.bounces>k)|result.unresolved
        mass=float(result.source.weights[tail].sum())
        observed=float(result.contributions[tail].sum())
        rows.append(dict(cutoff=k,count=int(tail.sum()),mass=mass,omission_bound=2*mass,
                         escaped_tail_omega=observed,physical_updates_beyond_cutoff=int(np.maximum(result.bounces-k,0).sum()),
                         unresolved_count=int(result.unresolved.sum())))
    return rows
