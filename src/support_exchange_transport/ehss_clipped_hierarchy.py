"""Full binary hierarchy of clipped-domain boundary responses.

Parents compose child responses until leaving their contiguous spatial range.
Only leaves search physical atom candidates; shared candidates retain global
IDs. No response approximation/cache is used in this boundary-contract stage.
"""
import time
import numpy as np
from numba import njit
from .ehss_clipped_slabs import ClippedSlabTransport,slab_response,locate_slab
from .ehss_state import EHSSResult


@njit(cache=False)
def clipped_node_response(node,domain,o,d,cursor,last,bounces,cap,normal,edges,offsets,ids,centers,radii,history,
                          lower,upper,left,right,calls,node_collisions,multiple,returns):
    before=bounces;calls[node]+=1;tests=0
    if left[node]<0:
        domain,cursor,last,bounces,status,tests=slab_response(domain,o,d,cursor,last,bounces,cap,normal,edges,offsets,ids,centers,radii,history)
    else:
        status=0
        finished=False
        for _ in range((cap+1)*(len(edges)+2)+1):
            child=left[node] if domain<upper[left[node]] else right[node]
            domain,cursor,last,bounces,status,nt=clipped_node_response(child,domain,o,d,cursor,last,bounces,cap,
                normal,edges,offsets,ids,centers,radii,history,lower,upper,left,right,calls,node_collisions,multiple,returns)
            tests+=nt
            if status!=0 or domain<lower[node] or domain>=upper[node]:finished=True;break
        if not finished:raise ValueError('Parent composition exceeded physical-flight bound')
    added=bounces-before;node_collisions[node]+=added
    if added>=2:multiple[node]+=1
    if status==0:returns[node]+=1
    return domain,cursor,last,bounces,status,tests


@njit(cache=False)
def trace_clipped_hierarchy(origins,directions,last_atoms,initial_bounces,cap,normal,edges,offsets,ids,centers,radii,lower,upper,left,right):
    p,d=origins.copy(),directions.copy();b=initial_bounces.copy()
    history=np.full((len(origins),cap),-1,dtype=np.int32)
    escaped=np.zeros(len(origins),dtype=np.bool_);unresolved=np.zeros(len(origins),dtype=np.bool_)
    calls=np.zeros(len(left),dtype=np.int64);collisions=np.zeros(len(left),dtype=np.int64)
    multiple=np.zeros(len(left),dtype=np.int64);returns=np.zeros(len(left),dtype=np.int64);tests=0
    for ray in range(len(origins)):
        o=p[ray];out=d[ray];last=last_atoms[ray]
        if initial_bounces[ray]==1:history[ray,0]=last
        domain=locate_slab(o,out,normal,edges)
        _,_,_,b[ray],status,nt=clipped_node_response(0,domain,o,out,0.,last,b[ray],cap,normal,edges,offsets,ids,centers,radii,history[ray],
            lower,upper,left,right,calls,collisions,multiple,returns)
        tests+=nt;escaped[ray]=status==1;unresolved[ray]=status==2
        if status==0:raise ValueError('Root domain covers all space and cannot hand off outside')
    return p,d,escaped,unresolved,b,history,tests,calls,collisions,multiple,returns


class ClippedSlabHierarchy(ClippedSlabTransport):
    def __init__(self,spheres,normal,edges):
        tick=time.perf_counter()
        super().__init__(spheres,normal,edges)
        lower=[];upper=[];left=[];right=[];depth=[]
        def build(a,b,level):
            node=len(left);lower.append(a);upper.append(b);left.append(-1);right.append(-1);depth.append(level)
            if b-a>1:
                mid=(a+b)//2
                left[node]=build(a,mid,level+1);right[node]=build(mid,b,level+1)
            return node
        build(0,len(self.edges)+1,0)
        for name,value in [('lower',lower),('upper',upper),('left',left),('right',right),('depth',depth)]:
            array=np.array(value,dtype=np.int64);array.setflags(write=False);setattr(self,name,array)
        self.receipt.update(build_s=time.perf_counter()-tick,nodes=len(left),tree_depth=max(depth),
                            hierarchy_array_bytes=sum(getattr(self,n).nbytes for n in ('lower','upper','left','right','depth')))

    def trace(self,source,max_bounces=128):
        if not isinstance(max_bounces,(int,np.integer)) or max_bounces<1 or np.any(source.initial_bounces>1):
            raise ValueError('Positive cap and uncollided/first-reflected source required')
        if np.any(source.last_atom>=len(self.spheres.radii)) or np.any(source.last_atom< -1):raise ValueError('Invalid physical collider ID')
        tick=time.perf_counter()
        p,d,e,u,b,h,tests,calls,collisions,multiple,returns=trace_clipped_hierarchy(source.origins,source.directions,source.last_atom,
            source.initial_bounces,max_bounces,self.normal,self.edges,self.offsets,self.ids,self.spheres.centers,self.spheres.radii,
            self.lower,self.upper,self.left,self.right)
        return EHSSResult(p,d,e,u,b,h,source,dict(query_s=time.perf_counter()-tick,backend='hierarchical clipped-domain response composition',
            sphere_tests=int(tests),node_calls=calls.tolist(),node_collisions=collisions.tolist(),multi_collision_responses=multiple.tolist(),
            boundary_returns=returns.tolist(),nodes=len(self.left),tree_depth=int(self.depth.max()),geometry_copies_per_atom=1))
