import numpy as np
import pytest
from support_exchange_transport.inside_out_pa import Spheres
from support_exchange_transport.ehss_state import EHSSSource,external_source,reverse_pa_source
from support_exchange_transport.ehss_reference import trace,nearest_all,entry_distance


@pytest.mark.parametrize("radius",[.01,1.,100.])
def test_single_sphere_analytic_and_reverse_initialization(radius):
    s=Spheres(np.array([[2.,-1.,4.]]),np.array([radius]))
    for factory in [external_source,reverse_pa_source]:
        source=factory(s,14,98); r=trace(s,source,4)
        assert r.escaped.all() and not r.unresolved.any()
        assert np.all(r.bounces==1)
        assert r.omega==pytest.approx(np.pi*radius*radius,rel=1e-6)
        assert r.pa==pytest.approx(np.pi*radius*radius,rel=1e-7)
        assert np.allclose(np.linalg.norm(r.outgoing,axis=1),1.,atol=1e-14)


def test_nearest_collision_is_distance_order_not_atom_order():
    centers=np.array([[5.,0.,0.],[0.,0.,0.]]); radii=np.ones(2)
    atom,t,_=nearest_all(np.array([-3.,0.,0.]),np.array([1.,0.,0.]),-1,centers,radii)
    assert atom==1 and t==2.
    assert np.isinf(entry_distance(np.array([-3.,1.01,0.]),np.array([1.,0.,0.]),centers[1],1.))


def test_reflection_matches_analytic_impact_parameter():
    s=Spheres(np.zeros((1,3)),np.ones(1))
    b=np.array([0.,.2,.8,.999]); o=np.column_stack((np.full(4,-3.),b,np.zeros(4)))
    d=np.tile([1.,0.,0.],(4,1))
    source=EHSSSource(o,d,d,np.ones(4),np.full(4,-1),np.zeros(4),"controlled incidence")
    r=trace(s,source)
    assert np.allclose(r.outgoing[:,0],2*b*b-1.,atol=2e-14)
    assert np.allclose(r.contributions,2*(1-b*b),atol=2e-14)


def test_exact_two_sphere_trapped_path_reports_tail_and_bounce_order():
    s=Spheres(np.array([[-2.,0.,0.],[2.,0.,0.]]),np.ones(2))
    source=EHSSSource(np.array([[0.,0.,0.]]),np.array([[1.,0.,0.]]),np.array([[1.,0.,0.]]),
        np.ones(1),np.array([-1]),np.array([0]),"internal diagnostic, not external CCS source")
    r=trace(s,source,7)
    assert r.unresolved[0] and not r.escaped[0]
    assert r.collider_ids[0].tolist()==[1,0,1,0,1,0,1]
    assert r.omega==0 and r.tail_bound==2.


def test_multi_scattering_and_cap_bound():
    s=Spheres(np.array([[2.,0.,0.],[-2.,0.,0.],[0.,2.,0.],[0.,-2.,0.],[0.,0.,2.],[0.,0.,-2.]]),np.full(6,1.25))
    source=reverse_pa_source(s,14,114)
    small=trace(s,source,1); full=trace(s,source,64)
    assert np.count_nonzero(full.bounces>=2)>100
    assert full.omega>=small.omega-1e-12
    assert full.omega<=small.omega+small.tail_bound+1e-12
    assert np.all(small.escaped|small.unresolved)
    assert not np.any(small.escaped&small.unresolved)
    assert full.source.weights.sum()==pytest.approx(full.source.weights[full.escaped].sum()+full.residual_mass)


def test_pa_filter_does_not_renormalize_exposed_states_and_overlap():
    s=Spheres(np.array([[0.,0.,0.],[1.,0.,0.]]),np.ones(2))
    source=reverse_pa_source(s,14,1)
    assert len(source.weights)<2**14
    assert np.all(source.weights==s.area/(4*2**14))
    result=trace(s,source)
    assert result.escaped.all()
    assert np.isfinite(result.omega)


def test_small_positive_gap_is_not_skipped():
    gap=1e-10
    s=Spheres(np.array([[0.,0.,0.],[2.+gap,0.,0.]]),np.ones(2))
    atom,t,_=nearest_all(np.array([1.,0.,0.]),np.array([1.,0.,0.]),0,s.centers,s.radii)
    assert atom==1 and t==pytest.approx(gap,rel=1e-5)
