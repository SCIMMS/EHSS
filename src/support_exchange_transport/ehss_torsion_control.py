"""Explicit CCD bond-axis rotations for geometric update controls, not MD."""
import shlex
import numpy as np


def ccd_bonds(path, component):
    """Read the single-line chem_comp_bond loop used by the downloaded CCD."""
    lines = path.read_text(encoding='utf-8').splitlines(); headers = []; rows = []; collecting = False
    for line in lines:
        value = line.strip()
        if value.startswith('_chem_comp_bond.'):
            headers.append(value); collecting = True; continue
        if collecting and (not value or value.startswith('#') or value == 'loop_' or value.startswith('_')):
            if rows: break
            continue
        if collecting:
            parts = shlex.split(value)
            if len(parts) != len(headers): raise ValueError('Unsupported multiline CCD bond record')
            rows.append(dict(zip(headers, parts)))
    if not rows: raise ValueError('CCD bond loop missing')
    return tuple((r['_chem_comp_bond.atom_id_1'], r['_chem_comp_bond.atom_id_2']) for r in rows
                 if r['_chem_comp_bond.comp_id'] == component)


def bond_cut(molecule, residue, bonds, axis=('CA', 'CB')):
    names = {key[-1]:i for i, key in enumerate(molecule.atom_keys) if key[:-1] == residue}
    if any(name not in names for name in axis): raise ValueError('Selected bond axis atoms missing')
    selected = [(names[a], names[b]) for a, b in bonds if a in names and b in names]
    a, b = (names[name] for name in axis)
    if (a, b) not in selected and (b, a) not in selected: raise ValueError('Axis must be an explicit CCD bond')
    graph = {i:set() for i in names.values()}
    for x, y in selected:
        if {x, y} == {a, b}: continue
        graph[x].add(y); graph[y].add(x)
    component = {b}; stack = [b]
    while stack:
        for other in graph[stack.pop()]:
            if other not in component: component.add(other); stack.append(other)
    if a in component: raise ValueError('A ring bond cannot define this isolated rigid branch')
    moving = np.array(sorted(component-{b}), dtype=np.int64)
    if not len(moving): raise ValueError('No distal selected atoms to rotate')
    return (a, b), moving, np.array(selected, dtype=np.int64)


def rotate_branch(centers, axis, moving, degrees):
    centers = np.asarray(centers, dtype=float)
    if not np.isfinite(centers).all() or not np.isfinite(degrees): raise ValueError('Finite geometry and angle required')
    a, b = axis; moving = np.asarray(moving, dtype=np.int64)
    if a == b or a in moving or b in moving or len(np.unique(moving)) != len(moving):
        raise ValueError('Distinct fixed axis and unique distal atom IDs required')
    vector = centers[b]-centers[a]; norm = np.linalg.norm(vector)
    if norm == 0.: raise ValueError('Nonzero bond axis required')
    result = centers.copy()
    if degrees == 0.: return result
    n = vector/norm; x, y, z = n
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    angle = np.deg2rad(degrees)
    rotation = np.cos(angle)*np.eye(3)+(1.-np.cos(angle))*np.outer(n, n)+np.sin(angle)*cross
    result[moving] = centers[a]+(centers[moving]-centers[a])@rotation.T
    return result


def dihedral_degrees(points):
    p0, p1, p2, p3 = np.asarray(points, dtype=float)
    axis = p2-p1; axis /= np.linalg.norm(axis)
    before = p0-p1; after = p3-p2
    v = before-np.dot(before, axis)*axis; w = after-np.dot(after, axis)*axis
    return float(np.degrees(np.arctan2(np.dot(np.cross(axis, v), w), np.dot(v, w))))
