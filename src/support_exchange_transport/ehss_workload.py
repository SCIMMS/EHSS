"""Measured PA/EHSS workloads from fixed ownership and current coordinates.

Direct uses the same chemical physical hierarchy without material discovery or
field construction. Both return PA, EHSS, unresolved mass and full order ledgers.
JIT warmup, input trajectory generation and ownership design are external.
"""
from copy import copy
import time
import numpy as np
from .ehss_prepared_geometry import GeometryPreparationLayout
from .ehss_refitted_hierarchy import HierarchyRefitLayout
from .ehss_guarded_intrinsic_response import GuardedSupportHierarchy
from .ehss_factored_local_samples import MaterialSamples
from .ehss_material_bank_cap import material_bank_with_cap
from .ehss_batched_coupling import MaterialPortLayout
from .ehss_aligned_material_basis import AlignedMaterialBasis
from .ehss_aligned_material_field import AlignedMaterialCollisionField
from .ehss_aligned_material_input import AlignedMaterialInput
from .ehss_surface_observation import SurfaceObservationPool
from .ehss_semantic_hierarchy import SemanticBlockPool
from .ehss_escape_reachable_field import EscapeReachableHierarchyField
from .ehss_empirical_field import owner_hierarchy
from .ehss_state import external_source,reverse_pa_source
from .ehss_reference import trace


class PhysicalScene:
    def __init__(self,spheres,groups,atom_ids=None):
        tick=time.perf_counter(); start=tick
        self.geometry_layout=GeometryPreparationLayout(spheres,groups,atom_ids); layout_s=time.perf_counter()-start
        start=time.perf_counter(); self.geometry=self.geometry_layout.build(spheres); geometry_s=time.perf_counter()-start
        start=time.perf_counter(); self.model=GuardedSupportHierarchy(self.geometry); model_s=time.perf_counter()-start
        start=time.perf_counter(); self.refit_layout=HierarchyRefitLayout(self.model); refit_layout_s=time.perf_counter()-start
        self.setup=dict(total_s=time.perf_counter()-tick,geometry_layout_s=layout_s,geometry_s=geometry_s,seed_model_s=model_s,refit_layout_s=refit_layout_s)
        self.receipt={}

    def snapshot(self):
        other=copy(self); other.receipt={}; return other

    def update(self,spheres):
        tick=time.perf_counter()
        if np.array_equal(spheres.centers,self.geometry.world) and np.array_equal(spheres.radii,self.geometry.radii):
            self.receipt=dict(total_s=time.perf_counter()-tick,geometry_s=0.,model_s=0.,unchanged=True); return self.model
        start=time.perf_counter(); geometry=self.geometry_layout.build(spheres,self.geometry); geometry_s=time.perf_counter()-start
        start=time.perf_counter(); old=self.model if getattr(self.model,'refit_layout',None) is self.refit_layout else None
        model=self.refit_layout.build(geometry,old); model_s=time.perf_counter()-start
        self.geometry=geometry; self.model=model
        self.receipt=dict(total_s=time.perf_counter()-tick,geometry_s=geometry_s,model_s=model_s,unchanged=False)
        return model


def physical_readout(result,cap):
    """Vector readout; timing must include this, not only physical tracing."""
    s=result.source; w=s.weights; count=result.bounces; escaped=result.escaped; unresolved=result.unresolved
    contribution=w*np.clip(1.-np.sum(s.incoming*result.outgoing,axis=1),0.,2.)*escaped
    mass=np.bincount(count[escaped],weights=w[escaped],minlength=cap+1)
    omega=np.bincount(count,weights=contribution,minlength=cap+1)
    residual=float(w[unresolved].sum())
    return dict(omega=float(contribution.sum()),pa=float(w[(count>0)|unresolved].sum()),residual_mass=residual,tail_bound=2*residual,
        escaped_mass=float(w[escaped].sum()),source_mass=float(w.sum()),sphere_tests=result.metrics['sphere_tests'],
        order_ledger=[dict(order=i,mass=float(mass[i]),omega=float(omega[i])) for i in range(cap+1)])


def collision_diagnostics(result,cap):
    """Additional path-count study; kept outside query timing for both routes."""
    w=result.source.weights; count=result.bounces; contribution=result.contributions
    return dict(
        collision_counts=np.bincount(count,minlength=cap+1).tolist(),
        collision_mass=np.bincount(count,weights=w,minlength=cap+1).tolist(),
        collision_omega=np.bincount(count,weights=contribution,minlength=cap+1).tolist(),
        unresolved_count=int(result.unresolved.sum()),unresolved_mass=result.residual_mass,physical_cap=cap)


class DirectWorkload:
    def __init__(self,spheres,physical_groups,atom_ids=None,cap=64):
        if type(cap) is not int or cap<1: raise ValueError('Positive query cap required')
        tick=time.perf_counter(); self.scene=PhysicalScene(spheres,physical_groups,atom_ids); self.cap=cap
        self.setup=dict(total_s=time.perf_counter()-tick,physical_scene=self.scene.setup); self.receipt={}
    def snapshot(self):
        other=copy(self); other.scene=self.scene.snapshot(); other.receipt={}; return other
    def update(self,spheres):
        tick=time.perf_counter(); self.scene.update(spheres); self.receipt=dict(total_s=time.perf_counter()-tick,physical_scene=self.scene.receipt)
    def solve(self,source,capture=False):
        tick=time.perf_counter(); result=self.scene.model.trace(source,self.cap); trace_s=time.perf_counter()-tick
        start=time.perf_counter(); output=physical_readout(result,self.cap)
        output.update(trace_s=trace_s,readout_s=time.perf_counter()-start,query_s=time.perf_counter()-tick)
        if capture: output['_physical_result']=result
        return output


class SupportFieldWorkload:
    def __init__(self,spheres,response_groups,physical_groups,atom_ids=None,cap=64,training_cap=128,training_power=10,
                 training_seeds=(90631,90633),budget=.01):
        if type(cap) is not int or type(training_cap) is not int or not 2<=cap<=training_cap: raise ValueError('2 <= query cap <= training cap required')
        if not np.isfinite(budget) or not 0<=budget<=1: raise ValueError('Finite relative mass budget in [0,1] required')
        tick=time.perf_counter(); self.coarse=PhysicalScene(spheres,response_groups,atom_ids); self.fine=PhysicalScene(spheres,physical_groups,atom_ids)
        start=time.perf_counter(); sources=[f(spheres,training_power,s) for f in (external_source,reverse_pa_source) for s in training_seeds]; source_s=time.perf_counter()-start
        start=time.perf_counter(); training=[trace(spheres,s,training_cap) for s in sources]; discovery_s=time.perf_counter()-start
        start=time.perf_counter(); self.bank=material_bank_with_cap(MaterialSamples(self.coarse.model,training,training_cap),cap); bank_s=time.perf_counter()-start
        start=time.perf_counter(); self.layout=MaterialPortLayout(self.bank,2); material_layout_s=time.perf_counter()-start
        start=time.perf_counter(); self.basis=AlignedMaterialBasis(self.bank); basis_s=time.perf_counter()-start
        start=time.perf_counter(); self.observations=SurfaceObservationPool(self.basis,dependency_groups=physical_groups,max_age=4,
            geometry_tolerance=1.,point_tolerance=.25,direction_degrees=5.)
        self.hierarchy_pool=SemanticBlockPool(self.bank); cache_s=time.perf_counter()-start
        self.cap=cap; self.budget=budget; self.adapter=None; self.engine=None; self.receipt={}
        self.setup=dict(total_s=time.perf_counter()-tick,coarse_scene=self.coarse.setup,physical_scene=self.fine.setup,
            training_source_s=source_s,discovery_s=discovery_s,bank_s=bank_s,material_layout_s=material_layout_s,basis_s=basis_s,cache_s=cache_s,
            training_packets=len(sources),training_rows=sum(len(s.weights) for s in sources),material_rows=len(self.bank.mass),material_ports=len(self.layout.keys))
    def snapshot(self):
        other=copy(self); other.coarse=self.coarse.snapshot(); other.fine=self.fine.snapshot()
        other.observations=self.observations.snapshot(); other.hierarchy_pool=self.hierarchy_pool.snapshot(); other.receipt={}; return other
    def update(self,spheres):
        tick=time.perf_counter(); self.coarse.update(spheres); self.fine.update(spheres)
        start=time.perf_counter(); bound=self.basis.bind(self.fine.model); bind_s=time.perf_counter()-start
        start=time.perf_counter(); observed=self.observations.observe(bound); observed_s=time.perf_counter()-start
        start=time.perf_counter(); fit=AlignedMaterialCollisionField(bound,observed,self.fine.model,layout=self.layout); coupling_s=time.perf_counter()-start
        start=time.perf_counter(); root=owner_hierarchy(self.coarse.model.tree,fit.kernel.owners)
        engine=EscapeReachableHierarchyField(fit.kernel,root,fit.labels,self.hierarchy_pool,self.bank); hierarchy_s=time.perf_counter()-start
        start=time.perf_counter(); self.adapter=AlignedMaterialInput(fit); self.engine=engine.with_budget(self.budget); adapter_s=time.perf_counter()-start
        self.receipt=dict(total_s=time.perf_counter()-tick,coarse_scene=self.coarse.receipt,physical_scene=self.fine.receipt,
            bind_s=bind_s,observed_s=observed_s,coupling_s=coupling_s,hierarchy_s=hierarchy_s,adapter_s=adapter_s,
            observation=self.observations.receipt,field=fit.receipt)
    def solve(self,source):
        if self.adapter is None: raise ValueError('Update current field before querying')
        return self.adapter.solve(source,self.cap,self.engine)
