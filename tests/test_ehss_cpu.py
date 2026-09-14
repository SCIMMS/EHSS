import numpy as np
import numba
import pytest
from support_exchange_transport.inside_out_pa import Spheres
from support_exchange_transport.ehss_state import external_source, reverse_pa_source, EHSSSource
from support_exchange_transport.ehss_workload import PhysicalScene
from support_exchange_transport.ehss_optimized import OptimizedEHSSTransport
from support_exchange_transport.ehss_cpu import CPUEHSSTransport, entry_scalar
from support_exchange_transport.ehss_cpu_geometry import CPUPhysicalScene
from support_exchange_transport.ehss_reference import entry_distance


def fixture():
    rng = np.random.default_rng(88271)
    return Spheres(rng.normal(size=(29, 3))*3, rng.uniform(.8, 1.5, 29)), np.arange(29)//3


def equal(a, b, exact=True):
    for k in ('escaped', 'unresolved', 'bounces', 'collider_ids'):
        np.testing.assert_array_equal(getattr(a, k), getattr(b, k))
    for k in ('positions', 'outgoing', 'contributions'):
        if exact:
            np.testing.assert_array_equal(getattr(a, k), getattr(b, k))
        else:
            np.testing.assert_allclose(getattr(a, k), getattr(b, k), rtol=1e-9, atol=1e-8)


@pytest.mark.parametrize('factory', [external_source, reverse_pa_source])
@pytest.mark.parametrize('cap', [1, 4, 64])
def test_cpu_backends(factory, cap):
    s, groups = fixture(); model = PhysicalScene(s, groups).model
    source = factory(s, 9, 781)
    baseline = OptimizedEHSSTransport(model).trace(source, cap, True)
    scalar = CPUEHSSTransport(model).trace(source, cap, True)
    equal(scalar, baseline, exact=False)
    before = numba.get_num_threads()
    for threads in (1, min(2, numba.config.NUMBA_NUM_THREADS), min(4, numba.config.NUMBA_NUM_THREADS)):
        equal(CPUEHSSTransport(model, threads=threads, scalar=False).trace(source, cap, True), baseline)
        equal(CPUEHSSTransport(model, threads=threads).trace(source, cap, True), scalar)
    assert numba.get_num_threads() == before


def tree_equal(a, b):
    for key in ('groups', 'ids', 'offsets', 'left', 'right', 'end', 'parent', 'leaf_group', 'leaf_node',
                'depth', 'coarse_node', 'coarse_supports', 'local', 'world', 'radii', 'lo', 'hi',
                'rotations', 'translations', 'pose_versions', 'exclusive', 'port_offsets'):
        np.testing.assert_array_equal(getattr(a.tree, key), getattr(b.tree, key), err_msg=key)


def test_geometry_exact_metadata_refit_and_snapshot():
    s, groups = fixture(); baseline = PhysicalScene(s, groups)
    cpu = CPUPhysicalScene(s, groups, threads=min(4, numba.config.NUMBA_NUM_THREADS))
    tree_equal(cpu.model, baseline.model)
    snapshot = cpu.snapshot(); old_bounds = cpu.model.tree.lo.copy()
    for factor in (1., 1.01, .97):
        p = s.centers.copy(); p[3:6, 2] *= factor
        moved = Spheres(p, s.radii)
        baseline.update(moved); cpu.update(moved)
        tree_equal(cpu.model, baseline.model)
        source = external_source(moved, 7, 817)
        equal(CPUEHSSTransport(cpu.model).trace(source, 32, True),
              CPUEHSSTransport(baseline.model).trace(source, 32, True))
    np.testing.assert_array_equal(snapshot.model.tree.lo, old_bounds)
    assert cpu.model.tree.left is snapshot.model.tree.left


def test_scalar_boundary_rules_and_inside_start():
    center = np.zeros(3); direction = np.array([1., 0., 0.])
    for radius in (.001, 1., 100.):
        for y in (0., .9*radius, radius, np.nextafter(radius, np.inf)):
            for x in (-3*radius, -.5*radius, 2*radius):
                origin = np.array([x, y, 0.])
                actual = entry_scalar(origin, direction, center, radius)
                expected = entry_distance(origin, direction, center, radius)
                assert actual == pytest.approx(expected, rel=1e-13, abs=1e-13)


def test_parallel_trapped_tail_and_empty_input():
    s = Spheres(np.array([[-2., 0., 0.], [2., 0., 0.]]), np.ones(2))
    model = CPUPhysicalScene(s, np.arange(2)).model
    for n in (0, 8):
        source = EHSSSource(np.zeros((n, 3)), np.tile([1., 0., 0.], (n, 1)),
            np.tile([1., 0., 0.], (n, 1)), np.ones(n), np.full(n, -1), np.zeros(n, dtype=int), 'internal diagnostic')
        result = CPUEHSSTransport(model, threads=min(4, numba.config.NUMBA_NUM_THREADS)).trace(source, 7, True)
        assert result.unresolved.all() and result.tail_bound == 2*n
        if n:
            np.testing.assert_array_equal(result.collider_ids[0], [1, 0, 1, 0, 1, 0, 1])


def test_cpp_against_same_scalar_kernel():
    from pathlib import Path
    from support_exchange_transport.ehss_cpu_cpp import CppEHSSTransport
    dll = Path(__file__).resolve().parents[1]/'output/ehss_cpu_20260910/native/ehss_cpu.dll'
    if not dll.exists():
        pytest.skip('Build optional MSVC comparison DLL first')
    s, groups = fixture(); model = CPUPhysicalScene(s, groups).model
    for source in (external_source(s, 10, 881), reverse_pa_source(s, 10, 881)):
        expected = CPUEHSSTransport(model).trace(source, 64, True)
        for threads in (1, min(4, numba.config.NUMBA_NUM_THREADS)):
            actual = CppEHSSTransport(model, dll, threads=threads).trace(source, 64, True)
            equal(actual, expected, exact=False)
            assert actual.metrics['observed_threads'] == threads


def test_prepared_cpu_updates_and_reuses_unchanged_geometry():
    from support_exchange_transport.ehss_cpu_workload import PreparedCPUEHSS
    s, groups = fixture(); work = PreparedCPUEHSS(s, groups, threads=min(4, numba.config.NUMBA_NUM_THREADS))
    result = work.solve(power=8, seed=44, capture=True, record_history=True)['_physical_result']
    expected = CPUEHSSTransport(PhysicalScene(s, groups).model).trace(result.source, 64, True)
    equal(result, expected)
    engine = work.engine
    assert work.update(s)['unchanged'] and work.engine is engine
    moved = Spheres(s.centers+[.001, 0., 0.], s.radii)
    assert not work.update(moved)['unchanged'] and work.engine is not engine
    result = work.solve(power=8, seed=44, capture=True, record_history=True)['_physical_result']
    equal(result, CPUEHSSTransport(PhysicalScene(moved, groups).model).trace(result.source, 64, True))
