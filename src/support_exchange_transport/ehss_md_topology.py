"""Chemical bridge cuts for stable heavy-atom support ownership.

Connectivity comes from the MD topology, bond order/aromatic flags from CCD.
Only nonterminal single bridges are cut; rings and amide/guanidino C-N links
are retained. These are candidate nearly rigid blocks, not rigid constraints.
"""
from pathlib import Path
import shlex
import numpy as np


def ccd_annotations(path, component):
    headers = []; rows = []; collecting = False
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        value = line.strip()
        if value.startswith('_chem_comp_bond.'):
            headers.append(value); collecting = True; continue
        if collecting and (not value or value.startswith('#') or value == 'loop_' or value.startswith('_')):
            if rows: break
            continue
        if collecting:
            parts = shlex.split(value)
            if len(parts) != len(headers): raise ValueError('Unsupported CCD multiline bond record')
            rows.append(dict(zip(headers, parts)))
    if not rows: raise ValueError('CCD bond loop missing')
    result = {}
    for row in rows:
        if row['_chem_comp_bond.comp_id'] != component: raise ValueError('CCD component mismatch')
        key = tuple(sorted((row['_chem_comp_bond.atom_id_1'], row['_chem_comp_bond.atom_id_2'])))
        result[key] = (row['_chem_comp_bond.value_order'], row['_chem_comp_bond.pdbx_aromatic_flag'] == 'Y')
    return result


def annotate_heavy_bonds(topology, ccd_directory):
    keys = [tuple(k) for k in topology['heavy_keys']]; atoms = {key:i for i, key in enumerate(keys)}
    if len(atoms) != len(keys): raise ValueError('Unique heavy-atom identities required')
    names = {key[3] for key in keys}; templates = {name:ccd_annotations(Path(ccd_directory)/f'{name}.cif', name) for name in names}
    actual = {tuple(sorted(pair)) for pair in topology['heavy_bonds']}
    expected_internal = set()
    for residue in {key[:4] for key in keys}:
        for a, b in templates[residue[3]]:
            if (*residue, a) in atoms and (*residue, b) in atoms:
                expected_internal.add(tuple(sorted((atoms[(*residue, a)], atoms[(*residue, b)]))))
    actual_internal = {pair for pair in actual if keys[pair[0]][:4] == keys[pair[1]][:4]}
    if actual_internal != expected_internal: raise ValueError('MD and CCD internal heavy-atom connectivity differ')
    annotations = []
    for a, b in sorted(actual):
        ka, kb = keys[a], keys[b]
        if ka[:4] == kb[:4]: order, aromatic = templates[ka[3]][tuple(sorted((ka[-1], kb[-1])))]
        elif ka[0] == kb[0] and {ka[-1], kb[-1]} == {'C', 'N'}: order, aromatic = 'SING', False
        else: raise ValueError('Inter-residue link needs explicit chemical annotation')
        annotations.append(dict(a=a, b=b, order=order, aromatic=aromatic))
    return annotations


def chemical_partition(elements, annotations):
    n = len(elements); graph = [set() for _ in range(n)]; orders = {}
    for row in annotations:
        a, b = int(row['a']), int(row['b']); key = tuple(sorted((a, b)))
        if not 0 <= a < n or not 0 <= b < n or a == b or key in orders: raise ValueError('Simple valid heavy-atom graph required')
        graph[a].add(b); graph[b].add(a); orders[key] = (row['order'], bool(row['aromatic']))
    # Tarjan bridge discovery. Ring links remain within a support even if CCD
    # assigns an alternating single/double Kekule representation.
    visited = np.full(n, -1, dtype=int); low = visited.copy(); bridges = set(); clock = 0
    def visit(a, parent):
        nonlocal clock
        visited[a] = low[a] = clock; clock += 1
        for b in sorted(graph[a]):
            if b == parent: continue
            if visited[b] < 0:
                visit(b, a); low[a] = min(low[a], low[b])
                if low[b] > visited[a]: bridges.add(tuple(sorted((a, b))))
            else: low[a] = min(low[a], visited[b])
    for a in range(n):
        if visited[a] < 0: visit(a, -1)
    conjugated = {a for a in range(n) if elements[a] == 'C' and any(elements[b] in ('O', 'N') and orders[tuple(sorted((a,b)))][0] == 'DOUB' for b in graph[a])}
    cuts = []; protected = []; quads = []
    for pair in sorted(orders):
        a, b = pair; order, aromatic = orders[pair]
        amide_like = (a in conjugated and elements[b] == 'N') or (b in conjugated and elements[a] == 'N')
        reason = None
        if pair not in bridges: reason = 'ring'
        elif order != 'SING' or aromatic: reason = 'multiple_or_aromatic'
        elif len(graph[a]) < 2 or len(graph[b]) < 2: reason = 'terminal_heavy_link'
        elif amide_like: reason = 'conjugated_C_N'
        if reason: protected.append(dict(a=a, b=b, reason=reason)); continue
        cuts.append(pair); quads.append((min(graph[a]-{b}), a, b, min(graph[b]-{a})))
    cut_set = set(cuts); groups = np.full(n, -1, dtype=np.int64); members = []
    for a in range(n):
        if groups[a] >= 0: continue
        group = len(members); stack = [a]; groups[a] = group; component = []
        while stack:
            v = stack.pop(); component.append(v)
            for b in graph[v]:
                if groups[b] < 0 and tuple(sorted((v, b))) not in cut_set:
                    groups[b] = group; stack.append(b)
        members.append(sorted(component))
    sizes = np.array([len(m) for m in members])
    return dict(groups=groups, members=members, cut_bonds=np.array(cuts, dtype=np.int64).reshape(-1, 2),
        dihedral_quads=np.array(quads, dtype=np.int64).reshape(-1, 4), protected_bonds=protected,
        blocks=len(members), singleton_blocks=int(np.count_nonzero(sizes == 1)), pair_blocks=int(np.count_nonzero(sizes == 2)),
        nontrivial_blocks=int(np.count_nonzero(sizes >= 3)), nontrivial_atoms=int(sizes[sizes >= 3].sum()),
        rule='Cut nonterminal single graph bridges; retain rings and conjugated carbonyl/guanidino C-N links. No distance-based bonds or physical ghost scatterers.',
        nonunique_pose_for_small_blocks=True)


def dihedrals(centers, quads):
    points = np.asarray(centers)[np.asarray(quads)]
    axis = points[:, 2]-points[:, 1]; axis /= np.linalg.norm(axis, axis=1)[:, None]
    v = points[:, 0]-points[:, 1]; w = points[:, 3]-points[:, 2]
    v -= (v*axis).sum(axis=1)[:, None]*axis; w -= (w*axis).sum(axis=1)[:, None]*axis
    return np.arctan2((np.cross(axis, v)*w).sum(axis=1), (v*w).sum(axis=1))
