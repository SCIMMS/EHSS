"""Strict input QA for complete, unmodified, single-chain protein controls.

CCD supplies internal chemistry, ordered SEQRES-matched residues supply peptide
links, and explicit identity-image SSBOND records supply disulfides. Missing
terminal OXT is reported separately: raw geometry is usable but an uncapped MD
template is incomplete. No atoms, coordinates, or distance-inferred bonds are
added. Nonstandard links and ambiguous selections require another contract.
"""
from pathlib import Path
import shlex
import numpy as np
from .ehss_molecular_input import read_pdb_heavy
from .ehss_md_topology import ccd_annotations


def ccd_heavy_inventory(path, component):
    headers = []; rows = []; collecting = False
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        value = line.strip()
        if value.startswith('_chem_comp_atom.'):
            headers.append(value); collecting = True; continue
        if collecting and (not value or value.startswith(('#', '_')) or value == 'loop_'):
            if rows: break
            continue
        if collecting:
            parts = shlex.split(value)
            if len(parts) != len(headers):
                raise ValueError('Unsupported CCD multiline atom record')
            rows.append(dict(zip(headers, parts)))
    result = {}
    for row in rows:
        prefix = '_chem_comp_atom.'
        if row[prefix+'comp_id'] != component:
            raise ValueError('CCD component mismatch')
        element = row[prefix+'type_symbol']
        if element in ('H', 'D'): continue
        name = row[prefix+'atom_id']
        if name in result: raise ValueError('Duplicate CCD heavy atom')
        flag = row[prefix+'pdbx_leaving_atom_flag']
        if flag not in ('Y', 'N'): raise ValueError('Explicit CCD leaving flag required')
        result[name] = dict(element=element, leaving=flag == 'Y')
    if not result: raise ValueError('CCD heavy atom inventory missing')
    return result


def explicit_disulfides(lines, molecule):
    atoms = {k:i for i,k in enumerate(molecule.atom_keys)}
    chain = molecule.selection['chain']; used = set(); links = []; ignored = []
    for line in lines:
        if not line.startswith('SSBOND'): continue
        a = (line[15:16], int(line[17:21]), line[21:22].strip(), line[11:14].strip(), 'SG')
        b = (line[29:30], int(line[31:35]), line[35:36].strip(), line[25:28].strip(), 'SG')
        if a[0] != chain and b[0] != chain:
            ignored.append(line); continue
        if a[0] != chain or b[0] != chain:
            raise ValueError('SSBOND crosses selected chain boundary')
        if line[59:65].strip() != '1555' or line[66:72].strip() != '1555':
            raise ValueError('SSBOND needs an explicit identity symmetry image')
        if a[3] != 'CYS' or b[3] != 'CYS' or a not in atoms or b not in atoms:
            raise ValueError('SSBOND endpoint missing or not CYS SG')
        pair = tuple(sorted((atoms[a], atoms[b])))
        if pair[0] == pair[1] or any(i in used for i in pair):
            raise ValueError('Duplicate or ambiguous disulfide partner')
        used.update(pair)
        # SSBOND has no altloc field. Reject alternate SG on involved residues.
        for row in lines:
            if row.startswith(('ATOM  ', 'HETATM')) and row[12:16].strip() == 'SG':
                key = (row[21:22], int(row[22:26]), row[26:27].strip(), row[17:20].strip(), 'SG')
                if key in (a, b) and row[16:17].strip():
                    raise ValueError('Alternate SG requires explicit disulfide selection')
        links.append(dict(a=pair[0], b=pair[1], order='SING', aromatic=False,
                          source='SSBOND', atom_keys=[a,b],
                          distance_angstrom=float(np.linalg.norm(molecule.centers[pair[0]]-molecule.centers[pair[1]]))))
    return links, ignored


def compare_heavy_topology(expected_keys, annotations, actual_keys, actual_bonds):
    """Compare identities and bonds independently of the MD atom order."""
    expected_keys = list(map(tuple, expected_keys)); actual_keys = list(map(tuple, actual_keys))
    if len(set(actual_keys)) != len(actual_keys) or set(expected_keys) != set(actual_keys):
        raise ValueError('MD heavy atom identities differ')
    order = {key:i for i,key in enumerate(expected_keys)}
    actual = set()
    for a,b in actual_bonds:
        if a == b or not 0 <= a < len(actual_keys) or not 0 <= b < len(actual_keys):
            raise ValueError('Invalid MD heavy bond')
        pair = tuple(sorted((order[actual_keys[a]], order[actual_keys[b]])))
        if pair in actual: raise ValueError('Duplicate MD heavy bond')
        actual.add(pair)
    expected = {tuple(sorted((r['a'],r['b']))) for r in annotations}
    if actual != expected:
        raise ValueError(f'MD heavy connectivity differs: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}')
    return True


def validated_protein_graph(path, ccd_directories, chain='A'):
    path = Path(path); lines = path.read_text(encoding='ascii').splitlines()
    molecule = read_pdb_heavy(path, chain=chain, altloc='A')
    keys = molecule.atom_keys; residues = molecule.residue_keys; atoms = {k:i for i,k in enumerate(keys)}
    sequence = []; counts = set(); serials = []
    for line in lines:
        if line.startswith('SEQRES') and line[11:12] == chain:
            sequence.extend(line[19:70].split()); counts.add(int(line[13:17])); serials.append(int(line[7:10]))
    if counts != {len(sequence)} or serials != list(range(1,len(serials)+1)) or sequence != [r[3] for r in residues]:
        raise ValueError('Complete ordered SEQRES must match selected residues')
    # Insertion codes and numbering gaps need explicit sequence mapping, not a
    # guessed C-N link across neighboring ATOM records.
    if any(r[2] for r in residues) or any(b[1] != a[1]+1 for a,b in zip(residues,residues[1:])):
        raise ValueError('Residue numbering gap/insertion needs explicit sequence mapping')
    ended = False; selected_seen = False
    first_model = []
    for line in lines:
        if line.startswith('ENDMDL'): break
        first_model.append(line)
        if line.startswith('TER') and selected_seen and (line[21:22] == chain or not line[21:22].strip()): ended = True
        if line.startswith('ATOM  ') and line[21:22] == chain:
            if ended: raise ValueError('Internal TER requires explicit segment mapping')
            selected_seen = True
    for line in lines:
        if line.startswith('LINK  ') and chain in (line[21:22],line[51:52]):
            raise ValueError('Selected LINK requires explicit chemical annotation')
    templates = {}; inventories = {}; template_paths = {}
    for name in sorted(set(sequence)):
        candidates = [Path(d)/f'{name}.cif' for d in ccd_directories if (Path(d)/f'{name}.cif').exists()]
        if len(candidates) != 1: raise ValueError(f'Unique CCD template required: {name}')
        template_paths[name] = str(candidates[0].resolve())
        templates[name] = ccd_annotations(candidates[0], name)
        inventories[name] = ccd_heavy_inventory(candidates[0], name)
    annotations = []; missing_terminal = []
    for residue in residues:
        inventory = inventories[residue[3]]
        present = {k[-1] for k in keys if k[:4] == residue}
        required = {name for name,v in inventory.items() if not v['leaving']}
        if required-present: raise ValueError(f'Missing required heavy atoms: {residue}: {sorted(required-present)}')
        if present-set(inventory): raise ValueError(f'Unexpected heavy atoms: {residue}: {sorted(present-set(inventory))}')
        for name in present:
            if molecule.elements[atoms[(*residue,name)]] != inventory[name]['element']:
                raise ValueError('PDB and CCD elements differ')
        if 'OXT' in present and residue != residues[-1]: raise ValueError('OXT on internal residue')
        if residue == residues[-1] and 'OXT' in inventory and 'OXT' not in present:
            missing_terminal.append((*residue,'OXT'))
        for (a,b),(order,aromatic) in templates[residue[3]].items():
            if a in present and b in present:
                i,j = sorted((atoms[(*residue,a)],atoms[(*residue,b)]))
                annotations.append(dict(a=i,b=j,order=order,aromatic=aromatic,source='CCD'))
    peptide_lengths = []
    for a,b in zip(residues,residues[1:]):
        i,j = atoms[(*a,'C')], atoms[(*b,'N')]
        peptide_lengths.append(float(np.linalg.norm(molecule.centers[i]-molecule.centers[j])))
        i,j = sorted((i,j))
        annotations.append(dict(a=i,b=j,order='SING',aromatic=False,source='ordered_peptide'))
    disulfides,ignored = explicit_disulfides(first_model, molecule)
    annotations.extend(disulfides)
    expected = {(r['a'],r['b']) for r in annotations}
    if len(expected) != len(annotations): raise ValueError('Duplicate expected bonds')
    serial_atoms = {}
    for line in first_model:
        if line.startswith(('ATOM  ','HETATM')):
            key = (line[21:22],int(line[22:26]),line[26:27].strip(),line[17:20].strip(),line[12:16].strip())
            serial_atoms[int(line[6:11])] = (key,line[76:78].strip().upper())
    conect_checked = set()
    for line in lines:
        if not line.startswith('CONECT'): continue
        serial = int(line[6:11])
        for start in (11,16,21,26):
            if not line[start:start+5].strip(): continue
            partner = int(line[start:start+5])
            if serial not in serial_atoms or partner not in serial_atoms:
                known = serial_atoms.get(serial, serial_atoms.get(partner))
                if known is not None and known[0] in atoms:
                    raise ValueError('CONECT endpoint missing from input model')
                continue
            (a,ae),(b,be) = serial_atoms[serial],serial_atoms[partner]
            if a not in atoms and b not in atoms: continue
            if ae in ('H','D') or be in ('H','D'): continue
            if a not in atoms or b not in atoms: raise ValueError('CONECT crosses selected heavy atom boundary')
            pair = tuple(sorted((atoms[a],atoms[b])))
            if pair not in expected: raise ValueError('CONECT lacks expected chemical annotation')
            conect_checked.add(pair)
    return molecule, sorted(annotations,key=lambda r:(r['a'],r['b'])), dict(
        selection=molecule.selection, seqres_complete=True, residues=len(residues), atoms=len(keys),
        heavy_bonds=len(annotations), internal_bonds=sum(r['source']=='CCD' for r in annotations),
        peptide_bonds=len(peptide_lengths), peptide_distance_range_angstrom=[min(peptide_lengths),max(peptide_lengths)] if peptide_lengths else [],
        disulfides=disulfides, ignored_other_chain_ssbonds=ignored, conect_bonds_checked=len(conect_checked),
        missing_required_heavy_atoms=[], missing_uncapped_terminal_atoms=missing_terminal,
        md_uncapped_heavy_inventory_complete=not missing_terminal, ccd_paths=template_paths,
        scope='Complete standard single chain, blank insertion codes, contiguous numbering; raw coordinates unchanged. Graph cycle membership is not a rigidity certificate.')
