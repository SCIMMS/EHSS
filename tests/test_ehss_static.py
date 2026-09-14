import numpy as np
import pytest
from support_exchange_transport.inside_out_pa import Spheres
from support_exchange_transport.ccs_cli import spatial_groups
from support_exchange_transport.ehss_static import PreparedStaticCPUEHSS, static_spatial_groups
from support_exchange_transport.ehss_cpu_workload import PreparedCPUEHSS
from support_exchange_transport.ehss_state import external_source


@pytest.mark.parametrize('kind', ['random', 'ties', 'coincident', 'single', 'translated'])
@pytest.mark.parametrize('leaf', [1, 4, 9])
def test_static_matches_full_geometry_and_physical_paths(kind, leaf):
    rng = np.random.default_rng(936)
    xyz = rng.normal(size=(47, 3))*4
    if kind == 'ties': xyz = np.round(xyz)
    if kind == 'coincident': xyz[:] = 0
    if kind == 'single': xyz = xyz[:1]
    if kind == 'translated': xyz += 1e6
    s = Spheres(xyz, rng.uniform(.8, 2.5, len(xyz)), deduplicate=False)
    groups = spatial_groups(s, leaf)
    np.testing.assert_array_equal(static_spatial_groups(s, leaf), groups)
    old = PreparedCPUEHSS(s, groups)
    new = PreparedStaticCPUEHSS(s, leaf_size=leaf)
    for key in ('lo', 'hi', 'left', 'right', 'end', 'leaf_group', 'offsets', 'ids'):
        np.testing.assert_array_equal(getattr(old.scene.model.tree, key), getattr(new.model.tree, key))
    src = external_source(s, 8, 572)
    a, b = old.engine.trace(src, 64, True), new.engine.trace(src, 64, True)
    for key in ('outgoing', 'escaped', 'unresolved', 'bounces', 'collider_ids', 'contributions'):
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
    assert a.omega == b.omega and a.tail_bound == b.tail_bound
    s.centers[:] += 100  # A prepared scene must own its coordinates.
    np.testing.assert_array_equal(new.engine.trace(src, 64).outgoing, b.outgoing)


@pytest.mark.parametrize('groups', [[0, 2], [-1, 0], [0.0, 1.0], [0]])
def test_invalid_group_ownership(groups):
    s = Spheres([[0, 0, 0], [2, 0, 0]], [1, 1])
    with pytest.raises(ValueError): PreparedStaticCPUEHSS(s, groups=groups)
