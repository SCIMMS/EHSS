"""Publication geometry invariants for the frozen static tracing path."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from support_exchange_transport.inside_out_pa import Spheres
from support_exchange_transport.ehss_state import EHSSSource,external_source
from support_exchange_transport.ehss_static import PreparedStaticCPUEHSS


def fixture():
    return Spheres(np.array([[0.,0.,0.],[4.,0.,0.],[1.,5.,0.],[.5,1.,6.]]),np.full(4,3.1),deduplicate=False)


@pytest.mark.parametrize('scale',[.01,1.,100.])
def test_comoving_rays_rigid_pose_and_area_scaling(scale):
    spheres=fixture();source=external_source(spheres,12,170100001)
    original=PreparedStaticCPUEHSS(spheres).engine.trace(source,256,record_history=True)
    rotation=Rotation.from_rotvec([.31,-.44,.28]).as_matrix();shift=scale*np.array([9.,-4.,7.])
    moved=Spheres(scale*spheres.centers@rotation.T+shift,scale*spheres.radii,deduplicate=False)
    rays=EHSSSource(scale*source.origins@rotation.T+shift,source.directions@rotation.T,
        source.incoming@rotation.T,source.weights*scale**2,source.last_atom.copy(),
        source.initial_bounces.copy(),'co-moving similarity transform')
    result=PreparedStaticCPUEHSS(moved).engine.trace(rays,256,record_history=True)
    np.testing.assert_array_equal(result.escaped,original.escaped)
    np.testing.assert_array_equal(result.bounces,original.bounces)
    np.testing.assert_array_equal(result.collider_ids,original.collider_ids)
    np.testing.assert_allclose(result.outgoing,original.outgoing@rotation.T,rtol=1e-8,atol=1e-8)
    assert result.omega/scale**2==pytest.approx(original.omega,rel=1e-9)
    assert result.pa/scale**2==pytest.approx(original.pa,rel=1e-12)
    assert result.tail_bound/scale**2==pytest.approx(original.tail_bound,abs=1e-12)


def test_atom_permutation_preserves_static_scattering_and_mapped_history():
    spheres=fixture();source=external_source(spheres,12,170100002)
    original=PreparedStaticCPUEHSS(spheres).engine.trace(source,256,record_history=True)
    order=np.array([3,1,0,2])
    moved=Spheres(spheres.centers[order],spheres.radii[order],deduplicate=False)
    result=PreparedStaticCPUEHSS(moved).engine.trace(source,256,record_history=True)
    np.testing.assert_array_equal(result.outgoing,original.outgoing)
    np.testing.assert_array_equal(result.bounces,original.bounces)
    mapped=result.collider_ids.copy();valid=mapped>=0;mapped[valid]=order[mapped[valid]]
    np.testing.assert_array_equal(mapped,original.collider_ids)
    assert result.omega==original.omega
