"""Same-tree cone traversal with finite-flight local box / 14-DOP node bounds.

Bounds enclose owned finite-radius spheres and are never physical colliders.
Local axes are selected once, then held for deforming-frame refits. An explicit
rigid pose rotates those axes and reuses the interval coefficients. No re-PCA
or chemical repartition is hidden in a query or MD update.
"""
import time
from copy import copy
import numpy as np
from numba import njit
from .ehss_cone_packets import ConePacketTransport, cone_possible, query_packets, advance
from .ehss_persistent_cone_packets import partition_packets
from .ehss_state import EHSSResult


@njit(cache=True)
def refit_intervals(starts, ids, anchors, normals, centers, radii, dirty, old_low, old_high):
    low,high=old_low.copy(),old_high.copy()
    for node in range(len(anchors)):
        if not dirty[node]: continue
        origin=centers[anchors[node]]
        for k in range(7):
            a,b=np.inf,-np.inf
            normal=normals[node,k]
            for j in range(starts[node],starts[node+1]):
                atom=ids[j]
                p=np.dot(centers[atom]-origin,normal)
                a=min(a,p-radii[atom])
                b=max(b,p+radii[atom])
            low[node,k],high[node,k]=a,b
    return low,high


class SupportBoundLayout:
    def __init__(self, model):
        tick=time.perf_counter()
        t,s=model.tree,model.spheres
        self.topology=tuple(a.copy() for a in [t.left,t.right,t.end,t.leaf_group,t.groups])
        atom_leaf=t.leaf_node[t.groups]
        members=[np.flatnonzero((atom_leaf>=n)&(atom_leaf<t.end[n])) for n in range(len(t.left))]
        self.starts=np.r_[0,np.cumsum([len(a) for a in members])].astype(np.int64)
        self.ids=np.concatenate(members)
        self.anchors=np.array([a[0] for a in members],dtype=np.int64)
        self.parent=t.parent.copy()
        self.atom_leaf=atom_leaf.copy()
        self.normals=np.empty((len(members),7,3))
        self.membership_s=time.perf_counter()-tick
        start=time.perf_counter()
        diagonals=np.array([[1.,1.,1.],[1.,1.,-1.],[1.,-1.,1.],[-1.,1.,1.]])/np.sqrt(3.)
        for node,atoms in enumerate(members):
            x=s.centers[atoms]-s.centers[atoms].mean(axis=0)
            _,basis=np.linalg.eigh(x.T@x)
            basis=basis[:,::-1].copy()
            if np.linalg.det(basis)<0: basis[:,2]*=-1
            self.normals[node,:3]=basis.T
            self.normals[node,3:]=diagonals@basis.T
        self.axis_s=time.perf_counter()-start
        self.prepare_s=time.perf_counter()-tick

    def check(self,model):
        t=model.tree
        for a,b in zip(self.topology,[t.left,t.right,t.end,t.leaf_group,t.groups]):
            if not np.array_equal(a,b): raise ValueError('Fixed atom ownership and tree required')

    def build(self,model,previous=None,full=False):
        tick=time.perf_counter()
        self.check(model)
        if previous is not None and previous.layout is not self: raise ValueError('Different support layout')
        s=model.spheres
        if previous is None:
            dirty=np.ones(len(self.anchors),dtype=bool)
            low=high=np.zeros((len(self.anchors),7))
            changed=np.ones(len(s.radii),dtype=bool)
        else:
            if not np.array_equal(previous.normals,self.normals):
                raise ValueError('Posed bounds need an explicit new layout for deforming refits')
            changed=np.any(s.centers!=previous.centers,axis=1)|(s.radii!=previous.radii)
            dirty=np.zeros(len(self.anchors),dtype=bool)
            for leaf in np.unique(self.atom_leaf[changed]):
                node=leaf
                while node>=0 and not dirty[node]:
                    dirty[node]=True
                    node=self.parent[node]
            if full: dirty[:]=True
            low,high=previous.low,previous.high
        dirty_s=time.perf_counter()-tick
        start=time.perf_counter()
        low,high=refit_intervals(self.starts,self.ids,self.anchors,self.normals,s.centers,s.radii,dirty,low,high)
        projection_s=time.perf_counter()-start
        result=SupportBounds(self,model,self.normals,low,high)
        result.receipt=dict(total_s=time.perf_counter()-tick,dirty_s=dirty_s,projection_s=projection_s,
            changed_atoms=int(changed.sum()),updated_nodes=int(dirty.sum()),reused_nodes=int((~dirty).sum()),
            projected_members=int(np.diff(self.starts)[dirty].sum()),nodes=len(dirty),full=full,
            axes_reoptimized=False,local_axes_reused=True)
        return result


class SupportBounds:
    def __init__(self,layout,model,normals,low,high):
        self.layout,self.model=layout,model
        self.normals,self.low,self.high=normals,low,high
        self.centers,self.radii=model.spheres.centers.copy(),model.spheres.radii.copy()
        self.origins=self.centers[layout.anchors]
        self.node_centers=(model.tree.lo+model.tree.hi)*.5
        self.node_radii=np.linalg.norm(model.tree.hi-model.tree.lo,axis=1)*.5
        self.receipt={}

    def rigid_pose(self,model,rotation,translation):
        tick=time.perf_counter()
        self.layout.check(model)
        r,t=np.asarray(rotation),np.asarray(translation)
        if r.shape!=(3,3) or t.shape!=(3,) or not np.allclose(r.T@r,np.eye(3),rtol=0,atol=1e-12) or np.linalg.det(r)<0:
            raise ValueError('Proper rigid rotation and translation required')
        tolerance=1e-10*(1.+np.abs(model.spheres.centers).max())
        if not np.allclose(self.centers@r.T+t,model.spheres.centers,rtol=0,atol=tolerance) or not np.array_equal(self.radii,model.spheres.radii):
            raise ValueError('Current geometry is not the supplied rigid pose')
        result=SupportBounds(self.layout,model,self.normals@r.T,self.low,self.high)
        result.receipt=dict(total_s=time.perf_counter()-tick,interval_arrays_shared=True,
            axes_rotated=True,updated_interval_nodes=0,verified_explicit_pose=True)
        return result


@njit(cache=True)
def slab_cone_possible(origin,axis,cosine,sine,spread,node_center,node_radius,
                      support_origin,normals,low,high,count,counters):
    # Every ray hitting the node sphere must have a flight length in [near,far].
    axial=np.dot(node_center-origin,axis)
    pad=1e-9*(1.+np.max(np.abs(origin))+np.max(np.abs(support_origin))+node_radius)
    near=max(0.,axial-node_radius-spread-pad)
    far=(axial+node_radius+spread+pad)/max(cosine,1e-12)
    if far<near: return False
    offset=origin-support_origin
    for k in range(count):
        counters[2]+=1
        mu=max(-1.,min(1.,np.dot(axis,normals[k])))
        perpendicular=np.sqrt(max(0.,1.-mu*mu))
        dmax=1. if mu>=cosine else mu*cosine+perpendicular*sine
        dmin=-1. if -mu>=cosine else mu*cosine-perpendicular*sine
        q=np.dot(offset,normals[k])
        raylo=q-spread+min(near*dmin,far*dmin)-pad
        rayhi=q+spread+max(near*dmax,far*dmax)+pad
        if rayhi<low[k] or raylo>high[k]: return False
    return True


@njit(cache=False)
def collect_support(node,origin,axis,cosine,sine,spread,last,kind,
                    node_centers,node_radii,support_origins,normals,low,high,
                    left,right,leaf_group,offsets,ids,centers,radii,scratch,count,counters):
    counters[0]+=1
    if not cone_possible(origin,axis,cosine,sine,spread,node_centers[node],node_radii[node]): return count
    if kind>0:
        if not slab_cone_possible(origin,axis,cosine,sine,spread,node_centers[node],node_radii[node],
            support_origins[node],normals[node],low[node],high[node],3 if kind==1 else 7,counters):
            counters[3]+=1
            return count
    group=leaf_group[node]
    if group>=0:
        for j in range(offsets[group],offsets[group+1]):
            atom=ids[j]
            if atom==last: continue
            counters[1]+=1
            if cone_possible(origin,axis,cosine,sine,spread,centers[atom],radii[atom]):
                scratch[count]=atom
                count+=1
    else:
        count=collect_support(left[node],origin,axis,cosine,sine,spread,last,kind,node_centers,node_radii,
            support_origins,normals,low,high,left,right,leaf_group,offsets,ids,centers,radii,scratch,count,counters)
        count=collect_support(right[node],origin,axis,cosine,sine,spread,last,kind,node_centers,node_radii,
            support_origins,normals,low,high,left,right,leaf_group,offsets,ids,centers,radii,scratch,count,counters)
    return count


@njit(cache=False)
def support_candidates(origins,axes,cosines,sines,spreads,lasts,kind,node_centers,node_radii,
                       support_origins,normals,low,high,left,right,leaf_group,offsets,ids,centers,radii):
    scratch=np.empty(len(radii),dtype=np.int64)
    starts=[0]
    flat=[]
    counters=np.zeros(4,dtype=np.int64)
    for g in range(len(origins)):
        count=collect_support(0,origins[g],axes[g],cosines[g],sines[g],spreads[g],lasts[g],kind,
            node_centers,node_radii,support_origins,normals,low,high,left,right,leaf_group,offsets,ids,
            centers,radii,scratch,0,counters)
        for j in range(count): flat.append(scratch[j])
        starts.append(len(flat))
    return np.asarray(starts,dtype=np.int64),np.asarray(flat,dtype=np.int64),counters


class SupportConeTransport(ConePacketTransport):
    def __init__(self,model,bounds):
        super().__init__(model)
        if bounds.model is not model: raise ValueError('Bounds must belong to the exact current model')
        self.bounds=bounds

    def trace(self,source,cap=64,degrees=30.,minimum=4,support='dop14'):
        if type(cap) is not int or cap<1 or np.any(source.initial_bounces>1): raise ValueError('Positive cap and zero/one initial bounce required')
        if not 0.<degrees<90. or type(minimum) is not int or minimum<2 or support not in ['sphere','box','dop14']:
            raise ValueError('Valid cone and support policy required')
        t,s,b=self.model.tree,self.model.spheres,self.bounds
        if np.any(source.last_atom < -1) or np.any(source.last_atom>=len(s.radii)): raise ValueError('Invalid collider identity')
        kind=['sphere','box','dop14'].index(support)
        tick=time.perf_counter()
        points,directions=source.origins.copy(),source.directions.copy()
        lasts,counts=source.last_atom.copy(),source.initial_bounces.copy()
        labels=np.zeros(len(points),dtype=np.int64)
        history=np.full((len(points),cap),-1,dtype=np.int32)
        initial=counts==1
        history[initial,0]=lasts[initial]
        escaped=np.zeros(len(points),dtype=bool)
        unresolved=escaped.copy()
        active=np.arange(len(points),dtype=np.int64)
        metrics=dict(grouping_s=0.,candidate_s=0.,intersection_s=0.,reflection_s=0.,sphere_tests=0,box_tests=0,
            node_cone_tests=0,atom_cone_tests=0,slab_tests=0,slab_rejected_nodes=0,packets=0,packet_rays=0,
            fallback_rays=0,splits=0,candidate_ids=0,candidate_pairs=0,empty_packets=0,waves=0)
        while len(active):
            p,d,last=points[active],directions[active],lasts[active]
            start=time.perf_counter()
            groups=partition_packets(p,d,last,labels[active],s.centers,s.radii,np.deg2rad(degrees),minimum,True)
            ps,pr,fallback,origins,axes,widths,spreads,atoms,new_labels,splits,branches,retained=groups
            labels[active]=new_labels
            metrics['grouping_s']+=time.perf_counter()-start
            start=time.perf_counter()
            cs,ids,checks=support_candidates(origins,axes,np.cos(widths),np.sin(widths),spreads,atoms,kind,
                b.node_centers,b.node_radii,b.origins,b.normals,b.low,b.high,t.left,t.right,t.leaf_group,t.offsets,t.ids,s.centers,s.radii)
            metrics['candidate_s']+=time.perf_counter()-start
            metrics['candidate_pairs']+=int(np.dot(np.diff(ps),np.diff(cs)))
            start=time.perf_counter()
            hits,distances,sphere_tests,box_tests=query_packets(p,d,last,ps,pr,cs,ids,fallback,s.centers,s.radii,
                t.lo,t.hi,t.left,t.right,t.end,t.leaf_group,t.offsets,t.ids)
            metrics['intersection_s']+=time.perf_counter()-start
            start=time.perf_counter()
            active=advance(points,directions,lasts,counts,history,escaped,unresolved,active,hits,distances,cap,s.centers,s.radii)
            metrics['reflection_s']+=time.perf_counter()-start
            for key,val in dict(sphere_tests=int(sphere_tests),box_tests=int(box_tests),node_cone_tests=int(checks[0]),
                atom_cone_tests=int(checks[1]),slab_tests=int(checks[2]),slab_rejected_nodes=int(checks[3]),
                packets=len(atoms),packet_rays=len(pr),fallback_rays=len(fallback),splits=splits,candidate_ids=len(ids),
                empty_packets=int(np.count_nonzero(np.diff(cs)==0))).items(): metrics[key]+=val
            metrics['waves']+=1
        metrics.update(query_s=time.perf_counter()-tick,support=support,physical_direction_approximation=False,
            distribution_compressed=False,angular_merge=False,node_level_culling=True)
        return EHSSResult(points,directions,escaped,unresolved,counts,history,source,metrics)
