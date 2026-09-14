"""Backbone supports chosen from connectivity and a calibration-only MD window.

The hydrogen-bond energy is a DSSP-style geometry proxy, not a force-field
energy or a FIRST rigidity certificate. No coordinate or physical bond is changed.
"""
import numpy as np
from .ehss_local_chemical_partition import dependency_ledger, spatial_partition

BACKBONE={'N','CA','C','O','OXT'}


def residues(keys,annotations):
    keys=list(map(tuple,keys)); lookup={}; members=[]; maps=[]; identities=[]
    for atom,key in enumerate(keys):
        rid=lookup.get(key[:4])
        if rid is None:
            rid=len(members); lookup[key[:4]]=rid; members.append([]); maps.append({}); identities.append(key[:4])
        members[rid].append(atom); maps[rid][key[-1]]=atom
    peptide=[]
    for row in annotations:
        a,b=int(row['a']),int(row['b']); ka,kb=keys[a],keys[b]
        if ka[:4]==kb[:4] or ka[0]!=kb[0]: continue
        if ka[-1]=='N' and kb[-1]=='C': a,b=b,a; ka,kb=kb,ka
        if ka[-1]=='C' and kb[-1]=='N': peptide.append((lookup[ka[:4]],lookup[kb[:4]]))
    return identities,members,maps,sorted(peptide)


def hydrogen_scores(frames,keys,annotations):
    identities,members,maps,peptide=residues(keys,annotations)
    nr=len(members); energies=np.full((len(frames),nr,nr),np.inf)
    previous={b:a for a,b in peptide}
    excluded={tuple(sorted(p)) for p in peptide}
    for f,x in enumerate(frames):
        for donor in range(nr):
            if donor not in previous or identities[donor][3]=='PRO': continue
            d=maps[donor]; p=maps[previous[donor]]
            if not {'N','CA'}<=d.keys() or 'C' not in p: continue
            n=x[d['N']]
            # Planar amide bisector estimate, N-H length 1.01 Angstrom.
            a=n-x[p['C']]; b=n-x[d['CA']]
            h=a/np.linalg.norm(a)+b/np.linalg.norm(b); h=n+1.01*h/np.linalg.norm(h)
            for acceptor,am in enumerate(maps):
                if donor==acceptor or tuple(sorted((donor,acceptor))) in excluded or not {'C','O'}<=am.keys(): continue
                o,c=x[am['O']],x[am['C']]
                distances=np.array([np.linalg.norm(n-o),np.linalg.norm(h-c),np.linalg.norm(h-o),np.linalg.norm(n-c)])
                if distances.min()<.5 or distances[0]>5.: continue
                energies[f,donor,acceptor]=27.888*np.dot(1/distances,[1.,1.,-1.,-1.])
    return energies


def rigid_residual(reference,current):
    a=reference-reference.mean(axis=0); b=current-current.mean(axis=0)
    u,_,vh=np.linalg.svd(a.T@b); fix=np.eye(3); fix[2,2]=np.sign(np.linalg.det(u@vh))
    return float(np.linalg.norm(a@(u@fix@vh)-b,axis=1).max(initial=0.))


def build_partition(keys,annotations,chemical,calibration,mode='chain4',max_atoms=64,tolerance=.05):
    if mode not in ('chain4','calibrated_chain','calibrated_hbond'): raise ValueError('Unknown structural partition')
    if len(calibration)<1: raise ValueError('Calibration frames required')
    identities,members,maps,peptide=residues(keys,annotations)
    backbone=[np.array([a for a in row if keys[a][-1] in BACKBONE],dtype=int) for row in members]
    parent=list(range(len(members))); clusters={i:{i} for i in range(len(members))}; merge_log=[]
    def root(i):
        while parent[i]!=i: i=parent[i]
        return i
    hbonds=[]
    if mode=='calibrated_hbond':
        energies=hydrogen_scores(calibration,keys,annotations)
        occupancy=np.mean(energies<-.5,axis=0)
        for a,b in np.argwhere(occupancy>=.75):
            hbonds.append(dict(donor=int(a),acceptor=int(b),occupancy=float(occupancy[a,b]),
                median_energy=float(np.median(energies[:,a,b]))))
    edges=[(h['donor'],h['acceptor'],'hbond',h['median_energy']) for h in sorted(hbonds,key=lambda h:h['median_energy'])]
    edges += [(a,b,'peptide',0.) for a,b in peptide]
    for a,b,kind,score in edges:
        ra,rb=root(a),root(b)
        if ra==rb: continue
        merged=clusters[ra]|clusters[rb]
        atoms=np.concatenate([backbone[r] for r in sorted(merged)])
        if mode=='chain4':
            accept=len(merged)<=4
            residual=None
        else:
            residual=max(rigid_residual(calibration[-1,atoms],x[atoms]) for x in calibration)
            accept=len(atoms)<=max_atoms and residual<=tolerance
        merge_log.append(dict(a=a,b=b,kind=kind,score=score,accepted=accept,atoms=len(atoms),calibration_residual=residual))
        if accept:
            lo,hi=sorted((ra,rb)); parent[hi]=lo; clusters[lo]=merged; del clusters[hi]
    blocks=[np.concatenate([backbone[r] for r in sorted(rs)]) for rs in clusters.values()]
    blocks=[b for b in blocks if len(b)]
    side=np.array([key[-1] not in BACKBONE for key in keys])
    blocks += [np.flatnonzero(side&(chemical==g)) for g in np.unique(chemical[side])]
    blocks.sort(key=lambda b:int(b.min()))
    groups=np.full(len(keys),-1,dtype=np.int64)
    for g,b in enumerate(blocks):
        if np.any(groups[b]>=0): raise AssertionError('Duplicate physical ownership')
        groups[b]=g
    if np.any(groups<0): raise AssertionError('Missing physical ownership')
    ledger=dependency_ledger(groups,annotations)
    return groups,dict(mode=mode,groups=len(blocks),block_sizes=np.bincount(groups).tolist(),
        backbone_groups=[g for g,b in enumerate(blocks) if not side[b].any()],
        backbone_atoms=int((~side).sum()),sidechain_atoms=int(side.sum()),
        residue_ids=identities,peptide_links=peptide,hbond_candidates=hbonds,merge_log=merge_log,
        calibration_frames=len(calibration),max_atoms=max_atoms,shape_tolerance=tolerance,
        full_physical_bond_count=ledger['physical_bond_count'],all_physical_bonds_preserved=True,
        rigidity_certificate=False,hbond_method='Backbone electrostatic proxy; virtual planar NH; threshold -0.5, occupancy >=0.75')


def matched_spatial(centers,groups):
    return spatial_partition(centers,np.bincount(groups))
