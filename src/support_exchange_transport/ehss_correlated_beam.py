"""Finite-depth ray-space bounds enclosing every actual packet ray.

In a local frame q(z)=q0+z*p. Fit q0=qbar+B*(p-pbar)+e, then
bound p and the measured residual e. B is a shear, not a Gaussian assumption.
Any finite B is valid when every residual is enclosed; regularization affects
tightness only. Candidate spheres are tested over their whole axial interval.
"""
import time
import numpy as np
from numba import njit
from .ehss_cone_packets import candidate_lists, query_packets, advance
from .ehss_persistent_cone_packets import PersistentConeTransport, partition_packets
from .ehss_state import EHSSResult


@njit(cache=True)
def prepare_beams(points, directions, starts, ray_ids, origins, axes):
    gcount=len(origins)
    frames=np.empty((gcount,2,3))
    # qbar(2), pbar(2), B(4), dp lo/hi(4), e lo/hi(4), q0 lo/hi(4), min_z, pad, valid.
    data=np.zeros((gcount,23))
    for g in range(gcount):
        axis=axes[g]
        unit=np.zeros(3)
        unit[np.argmin(np.abs(axis))]=1.
        u=np.cross(axis,unit)
        u/=np.sqrt(np.dot(u,u))
        v=np.cross(axis,u)
        frames[g,0],frames[g,1]=u,v
        n=starts[g+1]-starts[g]
        q,p=np.empty((n,2)),np.empty((n,2))
        qbar,pbar=np.zeros(2),np.zeros(2)
        min_z=np.inf
        valid=True
        for j in range(n):
            ray=ray_ids[starts[g]+j]
            o=points[ray]-origins[g]
            d=directions[ray]
            dz=np.dot(d,axis)
            if dz <= 1e-8 or not np.isfinite(dz): valid=False
            if not valid: break
            z=np.dot(o,axis)
            min_z=min(min_z,z)
            p[j,0],p[j,1]=np.dot(d,u)/dz,np.dot(d,v)/dz
            q[j,0],q[j,1]=np.dot(o,u)-z*p[j,0],np.dot(o,v)-z*p[j,1]
            qbar+=q[j]
            pbar+=p[j]
        if not valid or n==0: continue  # Keep every candidate if chart is invalid.
        qbar/=n
        pbar/=n
        cpp,cqp=np.zeros((2,2)),np.zeros((2,2))
        for j in range(n):
            dq,dp=q[j]-qbar,p[j]-pbar
            for a in range(2):
                for b in range(2):
                    cpp[a,b]+=dp[a]*dp[b]
                    cqp[a,b]+=dq[a]*dp[b]
        ridge=1e-10*(cpp[0,0]+cpp[1,1])+1e-20
        c00,c11,c01=cpp[0,0]+ridge,cpp[1,1]+ridge,cpp[0,1]
        det=c00*c11-c01*c01
        B=np.zeros((2,2))
        if det>0 and np.isfinite(det):
            for a in range(2):
                B[a,0]=(cqp[a,0]*c11-cqp[a,1]*c01)/det
                B[a,1]=(cqp[a,1]*c00-cqp[a,0]*c01)/det
        if not np.isfinite(B).all(): B[:,:]=0.
        plo,phi=np.full(2,np.inf),np.full(2,-np.inf)
        elo,ehi=np.full(2,np.inf),np.full(2,-np.inf)
        qlo,qhi=np.full(2,np.inf),np.full(2,-np.inf)
        scale=1.
        for j in range(n):
            dp=p[j]-pbar
            e=q[j]-qbar-B@dp
            for a in range(2):
                plo[a],phi[a]=min(plo[a],dp[a]),max(phi[a],dp[a])
                elo[a],ehi[a]=min(elo[a],e[a]),max(ehi[a],e[a])
                qlo[a],qhi[a]=min(qlo[a],q[j,a]),max(qhi[a],q[j,a])
                scale=max(scale,abs(q[j,a]),abs(e[a]),abs(B[a,0]*dp[0]),abs(B[a,1]*dp[1]))
        # Explicit roundoff slack; the envelope is for the finite physical ray set.
        pad=1e-9*(1.+np.max(np.abs(origins[g])))+1e-10*scale
        data[g,0:2],data[g,2:4]=qbar,pbar
        data[g,4:8]=B.reshape(4)
        data[g,8:10],data[g,10:12]=plo,phi
        data[g,12:14],data[g,14:16]=elo,ehi
        data[g,16:18],data[g,18:20]=qlo,qhi
        data[g,20:23]=np.array([min_z,pad,1.])
    return frames,data


@njit(cache=True)
def beam_possible(origin, axis, frame, data, center, radius, correlated=True):
    if data[22]==0.: return True
    w=center-origin
    zc=np.dot(w,axis)
    pad=data[21]+1e-10*(1.+abs(zc)+radius)*(1.+np.max(np.abs(data[2:12])))
    lo=max(zc-radius-pad,data[20]-pad)
    hi=zc+radius+pad
    if hi<lo: return False
    target=np.array([np.dot(w,frame[0]),np.dot(w,frame[1])])
    # Four transverse slab directions; no ellipse/probability cutoff.
    for k in range(4):
        a,b=1.,0.
        if k==1: a,b=0.,1.
        elif k==2: a,b=1./np.sqrt(2.),1./np.sqrt(2.)
        elif k==3: a,b=1./np.sqrt(2.),-1./np.sqrt(2.)
        lower,upper=np.inf,-np.inf
        for z in (lo,hi):
            if correlated:
                base=a*(data[0]+z*data[2])+b*(data[1]+z*data[3])
                el=min(a*data[12],a*data[14])+min(b*data[13],b*data[15])
                eh=max(a*data[12],a*data[14])+max(b*data[13],b*data[15])
                c0=a*(data[4]+z)+b*data[6]
                c1=a*data[5]+b*(data[7]+z)
                low=base+el+min(c0*data[8],c0*data[10])+min(c1*data[9],c1*data[11])
                high=base+eh+max(c0*data[8],c0*data[10])+max(c1*data[9],c1*data[11])
            else:
                low=min(a*data[16],a*data[18])+min(b*data[17],b*data[19])
                high=max(a*data[16],a*data[18])+max(b*data[17],b*data[19])
                low+=min(z*a*(data[2]+data[8]),z*a*(data[2]+data[10]))
                low+=min(z*b*(data[3]+data[9]),z*b*(data[3]+data[11]))
                high+=max(z*a*(data[2]+data[8]),z*a*(data[2]+data[10]))
                high+=max(z*b*(data[3]+data[9]),z*b*(data[3]+data[11]))
            lower,upper=min(lower,low),max(upper,high)
        t=a*target[0]+b*target[1]
        if upper<t-radius-pad or lower>t+radius+pad: return False
    return True


@njit(cache=True)
def filter_candidates(starts, ids, origins, axes, frames, data, centers, radii, correlated):
    new_starts=np.zeros(len(starts),dtype=np.int64)
    kept=np.empty(len(ids),dtype=np.int64)
    count=0
    for g in range(len(starts)-1):
        for j in range(starts[g],starts[g+1]):
            atom=ids[j]
            if beam_possible(origins[g],axes[g],frames[g],data[g],centers[atom],radii[atom],correlated):
                kept[count]=atom
                count+=1
        new_starts[g+1]=count
    return new_starts,kept[:count]


class BeamFilteredTransport(PersistentConeTransport):
    def trace(self, source, cap=64, degrees=30., minimum=4, beam='shear'):
        if type(cap) is not int or cap<1 or np.any(source.initial_bounces>1):
            raise ValueError('Positive cap and zero/one initial collision required')
        if not 0.<degrees<90. or type(minimum) is not int or minimum<2 or beam not in ['none','box','shear']:
            raise ValueError('Valid cone policy and beam mode required')
        t,s=self.model.tree,self.model.spheres
        if np.any(source.last_atom < -1) or np.any(source.last_atom>=len(s.radii)):
            raise ValueError('Invalid initial physical identity')
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
        metrics=dict(grouping_s=0.,candidate_s=0.,beam_prepare_s=0.,beam_filter_s=0.,intersection_s=0.,reflection_s=0.,
            sphere_tests=0,box_tests=0,node_cone_tests=0,atom_cone_tests=0,packets=0,packet_rays=0,fallback_rays=0,
            splits=0,candidate_ids=0,cone_candidate_ids=0,candidate_pairs_before=0,candidate_pairs_after=0,
            beam_tests=0,invalid_charts=0,empty_packets=0,max_candidate_bytes=0,waves=0)
        while len(active):
            p,d,last=points[active],directions[active],lasts[active]
            start=time.perf_counter()
            groups=partition_packets(p,d,last,labels[active],s.centers,s.radii,np.deg2rad(degrees),minimum,True)
            ps,pr,fallback,origins,axes,widths,spreads,atoms,new_labels,splits,branches,retained=groups
            labels[active]=new_labels
            metrics['grouping_s']+=time.perf_counter()-start
            start=time.perf_counter()
            cs,ids,checks=candidate_lists(origins,axes,np.cos(widths),np.sin(widths),spreads,atoms,
                self.node_centers,self.node_radii,t.left,t.right,t.leaf_group,t.offsets,t.ids,s.centers,s.radii)
            metrics['candidate_s']+=time.perf_counter()-start
            metrics['cone_candidate_ids']+=len(ids)
            metrics['candidate_pairs_before']+=int(np.dot(np.diff(ps),np.diff(cs)))
            if beam!='none':
                start=time.perf_counter()
                frames,data=prepare_beams(p,d,ps,pr,origins,axes)
                metrics['beam_prepare_s']+=time.perf_counter()-start
                metrics['invalid_charts']+=int(np.count_nonzero(data[:,22]==0.))
                metrics['beam_tests']+=len(ids)
                start=time.perf_counter()
                cs,ids=filter_candidates(cs,ids,origins,axes,frames,data,s.centers,s.radii,beam=='shear')
                metrics['beam_filter_s']+=time.perf_counter()-start
            metrics['candidate_pairs_after']+=int(np.dot(np.diff(ps),np.diff(cs)))
            start=time.perf_counter()
            hits,distances,sphere_tests,box_tests=query_packets(p,d,last,ps,pr,cs,ids,fallback,
                s.centers,s.radii,t.lo,t.hi,t.left,t.right,t.end,t.leaf_group,t.offsets,t.ids)
            metrics['intersection_s']+=time.perf_counter()-start
            start=time.perf_counter()
            active=advance(points,directions,lasts,counts,history,escaped,unresolved,active,hits,distances,cap,s.centers,s.radii)
            metrics['reflection_s']+=time.perf_counter()-start
            for key,val in dict(sphere_tests=int(sphere_tests),box_tests=int(box_tests),node_cone_tests=int(checks[0]),
                atom_cone_tests=int(checks[1]),packets=len(atoms),packet_rays=len(pr),fallback_rays=len(fallback),
                splits=splits,candidate_ids=len(ids),empty_packets=int(np.count_nonzero(np.diff(cs)==0))).items(): metrics[key]+=val
            metrics['max_candidate_bytes']=max(metrics['max_candidate_bytes'],ids.nbytes+cs.nbytes)
            metrics['waves']+=1
        metrics.update(query_s=time.perf_counter()-tick,beam=beam,persistent_membership=True,
            distribution_compressed=False,physical_direction_approximation=False,angular_merge=False,
            bound='Finite ray-set shear/residual enclosure; full target-sphere axial interval')
        return EHSSResult(points,directions,escaped,unresolved,counts,history,source,metrics)
