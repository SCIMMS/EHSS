import numpy as np
import pytest
from support_exchange_transport.inside_out_pa import Spheres
from support_exchange_transport.ehss_native_ablation import NativeAblation,incident_source
from support_exchange_transport.ehss_static import PreparedStaticCPUEHSS
from support_exchange_transport.ehss_state import external_source,EHSSSource
from support_exchange_transport.ehss_reference import trace as independent_trace


def test_same_paths_across_native_search_and_instrumentation():
    rng=np.random.default_rng(317)
    spheres=Spheres(rng.normal(size=(37,3))*3,rng.uniform(.7,1.5,37))
    source=incident_source(spheres,10,321)
    production=PreparedStaticCPUEHSS(spheres).engine.trace(source,64,record_history=True)
    reference=independent_trace(spheres,source,64)
    for mode in ('flat','bvh'):
        engine=NativeAblation(spheres,mode)
        for instrument in (False,True):
            result=engine.trace(source,64,record_history=True,instrument=instrument)
            for field in ('escaped','unresolved','bounces','collider_ids','outgoing'):
                np.testing.assert_array_equal(getattr(result,field),getattr(production,field))
            np.testing.assert_allclose(result.outgoing,reference.outgoing,rtol=0,atol=1e-10)
            assert result.omega==production.omega
            if instrument:
                assert result.metrics['sphere_tests']>0
                assert (result.metrics['box_tests']>0)==(mode=='bvh')
            else:
                assert result.metrics['sphere_tests'] is None


def test_cap_and_known_repeated_path_preserved():
    spheres=Spheres([[-2,0,0],[2,0,0]],[1,1])
    source=EHSSSource(np.array([[0.,0,0]]),np.array([[1.,0,0]]),np.array([[1.,0,0]]),
        np.ones(1),np.array([-1]),np.zeros(1,dtype=np.int64),'internal diagnostic')
    for mode in ('flat','bvh'):
        result=NativeAblation(spheres,mode).trace(source,7,record_history=True)
        assert result.unresolved[0] and result.tail_bound==2
        assert result.collider_ids[0].tolist()==[1,0,1,0,1,0,1]


def test_iid_measure_and_sobol_compatibility():
    spheres=Spheres([[0,0,0]],[3.1])
    sobol=incident_source(spheres,12,91,'sobol')
    old=external_source(spheres,12,91)
    for field in ('origins','directions','weights'):
        np.testing.assert_array_equal(getattr(sobol,field),getattr(old,field))
    iid=incident_source(spheres,16,92,'iid')
    assert iid.weights.sum()==pytest.approx(np.pi*(3.1*(1+1e-9))**2,rel=1e-14)
    np.testing.assert_allclose(iid.directions.mean(axis=0),0,atol=.01)
    engine=NativeAblation(spheres,'flat')
    result=engine.trace(iid)
    assert result.omega==pytest.approx(np.pi*3.1**2,rel=.01)
    assert engine.trace(sobol).omega==pytest.approx(np.pi*3.1**2,rel=1e-6)
