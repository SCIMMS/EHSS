import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from support_exchange_transport.ccs_cli import (
    adaptive_pa, execute, parser, prepare_frame, radius_table, read_xyz,
    solve_frame, spatial_groups, validate_args,
)
from support_exchange_transport.inside_out_pa import Spheres


def options(tmp_path, *extra):
    return parser().parse_args(['unused.xyz','--rays','1024','--directions','8',
                               '--grid-size','128','--repeats','2',
                               '--output',str(tmp_path/'result'),*extra])


def test_xyz_multiframe_comment_and_fortran_coordinates(tmp_path):
    p=tmp_path/'unicode molecule.xyz'
    p.write_text('2\n\nc 1D0 2 3\nCl 4 5 6\n\n1\nsecond\nO 0 0 0\n',encoding='utf-8-sig')
    frames=read_xyz(p)
    assert len(frames)==2 and frames[0].comment==''
    assert frames[0].elements==('C','Cl')
    np.testing.assert_array_equal(frames[0].coordinates,[[1,2,3],[4,5,6]])


@pytest.mark.parametrize('text',['','0\ncomment\n','2\ncomment\nC 0 0 0\n',
    '1\nC 0 0 0\n','1\nc\nC nan 0 0\n','1\nc\n6 0 0 0\n',
    '1\nc\nC 0 0 0 1\n','1\nProperties=species:S:1:pos:R:3\nC 0 0 0\n',
    '1\nc\nC 0 0 0\ntrailing garbage\n'])
def test_bad_xyz_rejected(tmp_path,text):
    p=tmp_path/'bad.xyz';p.write_text(text)
    with pytest.raises(ValueError):read_xyz(p)


def test_units_selection_and_explicit_radii(tmp_path):
    p=tmp_path/'input.xyz';p.write_text('3\nmixed\nC 10 20 30\nH 10.1 20 30\nZn 10 20.2 30\n')
    frame=read_xyz(p)[0]
    with pytest.raises(ValueError,match='Zn'):prepare_frame(frame,radius_table(),1.4)
    override=tmp_path/'radii.json';override.write_text('{"Zn":1.39,"H":1.2}')
    s,info=prepare_frame(frame,radius_table(override),1.4,'nm',True)
    assert len(s.radii)==2 and info['selected_atom_rows_1based']==[1,3]
    np.testing.assert_allclose(s.radii,[3.1,2.79])
    np.testing.assert_allclose(s.centers[1]-s.centers[0],[0,2,0],atol=1e-12)
    p.write_text('2\ncoincident\nC 0 0 0\nO 0 0 0\n')
    with pytest.raises(ValueError,match='coincident'):prepare_frame(read_xyz(p)[0],radius_table(),1.4)


@pytest.mark.parametrize('data',[[],{'H':-1},{'H':True},{'H':float('nan')},{'cl':1.7,'Cl':1.8}])
def test_bad_radii_rejected(tmp_path,data):
    p=tmp_path/'r.json';p.write_text(json.dumps(data))
    with pytest.raises(ValueError):radius_table(p)


def test_single_sphere_analytic_and_repeat_reproducibility(tmp_path):
    args=options(tmp_path)
    s=Spheres(np.zeros((1,3)),np.array([3.1]))
    a=solve_frame(s,args,False);b=solve_frame(s,args,False)
    truth=np.pi*3.1**2
    assert a['summary']['ehss_angstrom2']['mean']==pytest.approx(truth,rel=2e-6)
    assert a['summary']['pa_angstrom2']['mean']==pytest.approx(truth,rel=.003)
    assert a['summary']==b['summary']
    for r in a['runs']:
        assert sum(r['ehss']['collision_histogram'])==args.rays
        assert r['ehss']['unresolved_rays']==0
        assert r['ehss']['tail_bound_angstrom2']==0


def test_xyz_spatial_grouping_against_independent_direct_trace(tmp_path):
    from support_exchange_transport.ehss_reference import trace
    from support_exchange_transport.ehss_state import external_source
    rng=np.random.default_rng(667)
    s=Spheres(rng.normal(size=(23,3))*2,np.full(23,1.2))
    args=options(tmp_path,'--pa-method','bvh','--repeats','1','--max-collisions','1')
    result=solve_frame(s,args,False)['runs'][0]
    reference=trace(s,external_source(s,10,args.seed),1)
    assert reference.unresolved.any()  # specifically exercise the omitted-tail contract
    assert result['ehss']['ehss_angstrom2']==pytest.approx(reference.omega,abs=1e-9)
    assert result['ehss']['tail_bound_angstrom2']==pytest.approx(reference.tail_bound)
    assert result['pa']['pa_angstrom2']==pytest.approx(reference.pa)
    assert np.max(np.bincount(spatial_groups(s,4)))<=4


def test_batch_outputs_preserve_inputs_and_single_replicate_sd(tmp_path):
    p=tmp_path/'two frames.xyz';text='1\nfirst\nC 0 0 0\n1\nsecond\nC 10 20 30\n';p.write_text(text)
    args=options(tmp_path,'--method','pa','--pa-method','bvh','--all-frames','--repeats','1')
    args.xyz=[str(p)]
    payload=execute(args)
    saved=json.loads((args.output/'results.json').read_text(encoding='utf-8'))
    assert saved==payload and len(saved['records'])==2
    assert saved['records'][0]['summary']['pa_angstrom2']['sd'] is None
    assert saved['records'][0]['summary']==saved['records'][1]['summary']
    with (args.output/'summary.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==2 and rows[0]['ehss_mean_A2']==''
    assert p.read_text()==text
    before=(args.output/'results.json').read_bytes()
    with pytest.raises(ValueError,match='already exists'):execute(args)
    assert (args.output/'results.json').read_bytes()==before


@pytest.mark.parametrize('flag,value',[('--rays','1000'),('--grid-size','4'),('--directions','3'),
    ('--repeats','0'),('--probe-radius','nan'),('--probe-radius','-1'),('--seed','-1')])
def test_invalid_options(tmp_path,flag,value):
    with pytest.raises(ValueError):validate_args(options(tmp_path,flag,value))
