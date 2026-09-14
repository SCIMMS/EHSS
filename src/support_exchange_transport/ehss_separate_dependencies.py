"""Choose collision-search leaves independently of observation dependencies.

Response ownership, material axes, training, and the physical kernels retain the
existing workload contract. This separates next-event cache dependencies; it
does not compile or reuse an entire multi-collision intrinsic response.
"""
import time
import numpy as np
from .ehss_reuse_frames import partition
from .ehss_surface_observation import SurfaceObservationPool
from .ehss_workload import SupportFieldWorkload


class SeparateDependencyWorkload(SupportFieldWorkload):
    def __init__(self, spheres, response_groups, search_groups, dependency_groups,
                 atom_ids=None, **kwargs):
        start = time.perf_counter()
        search, _ = partition(search_groups, len(spheres.radii), 'Search partition')
        dependency, atoms = partition(dependency_groups, len(spheres.radii), 'Dependency partition')
        validation_s = time.perf_counter() - start
        super().__init__(spheres, response_groups, search, atom_ids, **kwargs)
        assert self.observations.generation == 0 and not self.observations.records
        assert self.adapter is None and self.engine is None
        old = self.observations
        changed = not np.array_equal(old.groups, dependency)
        tick = time.perf_counter()
        if changed:
            self.observations = SurfaceObservationPool(
                self.basis, dependency_groups=dependency,
                geometry_tolerance=old.geometry_tolerance,
                point_tolerance=old.point_tolerance,
                direction_degrees=old.direction_degrees,
                max_age=old.max_age, padding=old.padding,
                reuse=old.reuse, surface_chart=old.surface_chart)
        replacement_s = time.perf_counter() - tick
        self.search_groups = search
        self.dependency_groups = dependency
        self.setup.update(
            partition_validation_s=validation_s,
            dependency_pool_replacement_s=replacement_s,
            dependency_pool_replaced=changed,
            search_group_count=len(np.unique(search)), dependency_group_count=len(atoms),
            search_dependency_partitions_differ=changed,
            full_intrinsic_response_reuse=False,
            cache_s=self.setup['cache_s'] + replacement_s,
            total_s=time.perf_counter() - start)

