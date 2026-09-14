"""Positive discrete boundary-field G/T composition on sphere ray coordinates.

Directions are shared globally; free flight and no-collision local responses
preserve their direction bin. Local reflection and transferred impact position
are quantized. Conservation/tail bounds are for this discrete model, not a bound
on its difference from continuous EHSS. Geometry must pass SphereInterfaceScene.
"""
import time
import numpy as np
from numba import njit
from .inside_out_pa import tangent_frames
from .ehss_spherical_coupling import angular_grid,angular_grid_index
from .ehss_interface_transport import apply_local_response,sphere_interval
from .ehss_reference import nearest_all


class RayBoundaryGrid:
    def __init__(self,n_theta=4,n_phi=8,n_radius=16,n_azimuth=16):
        if any(not isinstance(x,int) or x<1 for x in [n_theta,n_phi,n_radius,n_azimuth]):
            raise ValueError("Positive integer boundary grid sizes required")
        self.shape=(n_theta,n_phi,n_radius,n_azimuth)
        self.axes,_=angular_grid(n_theta,n_phi); self.t,self.b=tangent_frames(self.axes)
        self.disk_count=n_radius*n_azimuth
        squared=(np.arange(n_radius)+.5)/n_radius
        azimuth=(np.arange(n_azimuth)+.5)*2*np.pi/n_azimuth
        rr,pp=np.meshgrid(np.sqrt(squared),azimuth,indexing="ij")
        x=(rr*np.cos(pp)).ravel(); y=(rr*np.sin(pp)).ravel()
        self.direction_ids=np.repeat(np.arange(len(self.axes)),self.disk_count)
        self.directions=self.axes[self.direction_ids]
        transverse=(self.t[:,None,:]*x[None,:,None]+self.b[:,None,:]*y[None,:,None]).reshape(-1,3)
        height=np.tile(np.sqrt(1-rr.ravel()**2),len(self.axes))
        self.incoming=transverse-height[:,None]*self.directions
        self.outgoing=transverse+height[:,None]*self.directions
        self.size=len(self.directions)
        for a in [self.axes,self.t,self.b,self.direction_ids,self.directions,self.incoming,self.outgoing]:
            a.setflags(write=False)

    def project(self,positions,directions=None,direction_ids=None):
        positions=np.asarray(positions,float)
        if direction_ids is None:
            direction_ids=angular_grid_index(directions,*self.shape[:2])
        ids=np.asarray(direction_ids,dtype=np.int64)
        x=np.sum(positions*self.t[ids],axis=1); y=np.sum(positions*self.b[ids],axis=1)
        rho2=x*x+y*y
        ir=np.minimum(self.shape[2]-1,(np.clip(rho2,0.,1.)*self.shape[2]).astype(np.int64))
        ia=np.minimum(self.shape[3]-1,(np.mod(np.arctan2(y,x),2*np.pi)*self.shape[3]/(2*np.pi)).astype(np.int64))
        return ids*self.disk_count+ir*self.shape[3]+ia


@njit(cache=True)
def local_boundary_responses(points,directions,centers,radii,cap):
    exits=points.copy(); outgoing=directions.copy(); counts=np.zeros(len(points),dtype=np.int64)
    stopped=np.zeros(len(points),dtype=np.bool_); tests=0
    atom_ids=np.arange(len(radii)); ledger=np.empty(cap,dtype=np.int32)
    for i in range(len(points)):
        _,count,stop,nt,_=apply_local_response(exits[i],outgoing[i],-1,0,ledger,cap,atom_ids,centers,radii)
        counts[i]=count; stopped[i]=stop; tests+=nt
        if not stop:
            _,far=sphere_interval(exits[i],outgoing[i],np.zeros(3),1.)
            exits[i]+=far*outgoing[i]
    return exits,outgoing,counts,stopped,tests


@njit(cache=True)
def initial_exact_response(origins,directions,last_atoms,initial_counts,centers,radii,groups,
                           packed,offsets,envelope_centers,envelope_radii,cap):
    exits=origins.copy(); outgoing=directions.copy(); counts=initial_counts.copy()
    owners=np.full(len(origins),-1,dtype=np.int64); tests=0
    for i in range(len(origins)):
        last=last_atoms[i]
        if initial_counts[i]==1:
            group=groups[last]
        else:
            atom,_,nt=nearest_all(exits[i],outgoing[i],last,centers,radii); tests+=nt
            if atom<0:
                continue
            group=groups[atom]
        ledger=np.empty(cap,dtype=np.int32)
        _,counts[i],stop,nt,_=apply_local_response(exits[i],outgoing[i],last,counts[i],ledger,cap,
            packed[offsets[group]:offsets[group+1]],centers,radii)
        tests+=nt; owners[i]=-2 if stop else group
        if not stop:
            _,far=sphere_interval(exits[i],outgoing[i],envelope_centers[group],envelope_radii[group])
            exits[i]+=far*outgoing[i]
    return exits,outgoing,counts,owners,tests


class BoundaryMapLibrary:
    """Exact-key sharing; T depends on normalized atoms, G only on sphere pairs.

    Pair displacement is expressed in the common direction frame. Arbitrary
    pair-axis rotation reuse and distance interpolation are not implemented.
    """
    def __init__(self):
        self.local={}; self.pairs={}; self.local_hits=0; self.pair_hits=0

    def local_map(self,grid,centers,radii,cap):
        key=(grid.shape,cap,centers.shape,centers.tobytes(),radii.tobytes())
        if key in self.local:
            self.local_hits+=1; return self.local[key]
        t=time.perf_counter()
        positions,directions,counts,stopped,tests=local_boundary_responses(grid.incoming,grid.directions,centers,radii,cap)
        output=grid.project(positions,directions)
        # The analytic through branch preserves the complete incident ray cell.
        output[counts==0]=np.flatnonzero(counts==0)
        valid=~stopped
        angular=np.linalg.norm(grid.directions[output[valid]]-directions[valid],axis=1)
        spatial=np.linalg.norm(grid.outgoing[output[valid]]-positions[valid],axis=1)
        result=dict(output=output,bounces=counts,stopped=stopped,physical_outgoing=directions,compile_s=time.perf_counter()-t,
                    primitive_tests=int(tests),max_direction_chord=float(angular.max(initial=0.)),
                    max_exit_position_error=float(spatial.max(initial=0.)))
        self.local[key]=result; return result

    def pair_map(self,grid,delta,radius_ratio):
        key=(grid.shape,tuple(delta),float(radius_ratio))
        if key in self.pairs:
            self.pair_hits+=1; return self.pairs[key]
        t=time.perf_counter(); relative=grid.outgoing-delta
        projection=np.sum(relative*grid.directions,axis=1)
        perpendicular=relative-projection[:,None]*grid.directions
        disc=radius_ratio**2-np.sum(perpendicular*perpendicular,axis=1)
        half=np.sqrt(np.maximum(0.,disc)); near=-projection-half; far=-projection+half
        hit=(disc>=0.)&(far>0.)
        starts=np.where(hit,np.maximum(0.,near),np.inf)
        target=grid.project(relative/radius_ratio,direction_ids=grid.direction_ids)
        target[~hit]=-1
        result=dict(starts=starts,far=np.where(hit,far,-np.inf),target=target,inside=hit&(near<0.),compile_s=time.perf_counter()-t)
        self.pairs[key]=result; return result

    def receipt(self):
        arrays=[a for m in list(self.local.values())+list(self.pairs.values()) for a in m.values() if isinstance(a,np.ndarray)]
        return dict(local_maps=len(self.local),pair_maps=len(self.pairs),local_hits=self.local_hits,pair_hits=self.pair_hits,
                    payload_bytes=sum(a.nbytes for a in arrays))


def aggregate(indices,values):
    unique,inverse=np.unique(indices,return_inverse=True)
    merged=np.stack([np.bincount(inverse,weights=values[:,j],minlength=len(unique)) for j in range(values.shape[1])],axis=1)
    return unique,merged


class CompiledSupportField:
    def __init__(self,scene,grid,library=None,local_cap=64,through_mode="preserve"):
        if not isinstance(local_cap,int) or local_cap<1:
            raise ValueError("Positive local compilation cap required")
        if through_mode not in ("preserve","project"):
            raise ValueError("Through mode must be preserve or project")
        t=time.perf_counter(); self.scene=scene; self.grid=grid
        self.through_mode=through_mode
        self.library=BoundaryMapLibrary() if library is None else library
        n=grid.size; m=len(scene.radii); self.states=n*m
        self.output=np.empty(self.states,dtype=np.int64)
        self.physical_outgoing=np.empty((self.states,3))
        self.bounces=np.empty(self.states,dtype=np.int64); self.stopped=np.empty(self.states,dtype=bool)
        self.routing=np.full(self.states,-1,dtype=np.int64)
        self.raw_routing=np.full(self.states,-1,dtype=np.int64)
        self.overlap_routes=0; self.through_skips=0
        for a in range(m):
            owned=scene.groups==a
            local=np.ascontiguousarray((scene.spheres.centers[owned]-scene.centers[a])/scene.radii[a])
            radii=np.ascontiguousarray(scene.spheres.radii[owned]/scene.radii[a])
            response=self.library.local_map(grid,local,radii,local_cap)
            sl=slice(a*n,(a+1)*n)
            self.output[sl]=a*n+response["output"]; self.bounces[sl]=response["bounces"]; self.stopped[sl]=response["stopped"]
            self.physical_outgoing[sl]=response["physical_outgoing"]
        # All child response maps must exist before composing their through
        # branches. Raw pair geometry is independent of these response masks.
        for a in range(m):
            sl=slice(a*n,(a+1)*n)
            best=np.full(n,np.inf); inside=np.zeros(n,dtype=bool)
            starts=[]; ends=[]; targets=[]
            for b in range(m):
                if a==b:
                    continue
                pair=self.library.pair_map(grid,(scene.centers[b]-scene.centers[a])/scene.radii[a],scene.radii[b]/scene.radii[a])
                use=pair["starts"]<best
                self.raw_routing[sl][use]=b*n+pair["target"][use]; best[use]=pair["starts"][use]; inside[use]=pair["inside"][use]
                starts.append(pair["starts"]); ends.append(pair["far"]); targets.append(b*n+pair["target"])
            self.overlap_routes+=int(inside.sum())
            if through_mode=="project":
                self.routing[sl]=self.raw_routing[sl]
            elif starts:
                starts=np.asarray(starts); ends=np.asarray(ends); targets=np.asarray(targets)
                schedule=np.argsort(starts,axis=0,kind="stable")
                columns=np.arange(n); cursor=np.zeros(n); pending=np.ones(n,dtype=bool)
                for rank in range(m-1):
                    candidate=schedule[rank]
                    entry=starts[candidate,columns]; far=ends[candidate,columns]
                    target=targets[candidate,columns]
                    valid=pending&np.isfinite(entry)&(far>cursor)
                    response=(self.bounces[np.maximum(target,0)]>0)|self.stopped[np.maximum(target,0)]
                    selected=valid&response
                    self.routing[sl][selected]=target[selected]; pending[selected]=False
                    through=valid&~response
                    # Preserve this original outgoing ray, advancing only its
                    # exact scalar distance through the visited envelope. Never
                    # replace it with the projected through-cell's outgoing ray.
                    cursor[through]=far[through]; self.through_skips+=int(through.sum())
        if through_mode=="preserve":
            targets=self.routing[self.routing>=0]
            if not np.all((self.bounces[targets]>0)|self.stopped[targets]):
                raise AssertionError("Through closure must end at a physical-response cell or escape")
        # Sparse deterministic composition H=G T. Each input column has either
        # one successor or one escaped/residual outcome; no pairwise flux sum.
        self.successor=self.routing[self.output]
        self.compile_s=time.perf_counter()-t

    def solve(self,source,max_bounces=64,max_exchanges=128):
        if not isinstance(max_bounces,int) or max_bounces<1 or not isinstance(max_exchanges,int) or max_exchanges<1 or np.any(source.initial_bounces>1) or np.any(source.initial_bounces>max_bounces):
            raise ValueError("Positive caps and uncollided/first-reflected source required")
        if np.any(source.last_atom>=len(self.scene.spheres.radii)) or np.any(source.last_atom < -1):
            raise ValueError("Invalid source atom identity")
        start=time.perf_counter(); s=self.scene; grid=self.grid; n=grid.size
        exits,directions,orders,owners,tests=initial_exact_response(source.origins,source.directions,source.last_atom,
            source.initial_bounces,s.spheres.centers,s.spheres.radii,s.groups,s.packed_ids,s.offsets,s.centers,s.radii,max_bounces)
        # q=<weight * v_initial dot v_last_physical> is preserved through
        # geometric transfer and no-collision responses. Only a real local
        # reflection replaces q with m dot the map's physical outgoing direction.
        values=np.c_[source.weights,source.weights[:,None]*source.incoming,
                     source.weights*np.sum(source.incoming*directions,axis=1)]
        mass=np.zeros(max_bounces+1); omega=np.zeros(max_bounces+1); cell_omega=np.zeros(max_bounces+1)
        def collect(v,k,d):
            contribution=np.clip(v[:,0]-v[:,4],0.,2*v[:,0])
            cell_contribution=np.clip(v[:,0]-np.sum(v[:,1:4]*d,axis=1),0.,2*v[:,0])
            mass[:]+=np.bincount(k,weights=v[:,0],minlength=len(mass))
            omega[:]+=np.bincount(k,weights=contribution,minlength=len(omega))
            cell_omega[:]+=np.bincount(k,weights=cell_contribution,minlength=len(cell_omega))
        missed=owners==-1; collect(values[missed],orders[missed],directions[missed])
        bounce_tail=float(values[owners==-2,0].sum())
        active=owners>=0
        projected=grid.project((exits[active]-s.centers[owners[active]])/s.radii[owners[active],None],directions[active])
        outgoing=owners[active]*n+projected; target=self.routing[outgoing]
        v=values[active]; k=orders[active]; escape=target<0
        collect(v[escape],k[escape],grid.directions[projected[escape]])
        indices,field=aggregate(k[~escape]*self.states+target[~escape],v[~escape])
        seed_s=time.perf_counter()-start; history=[]
        for iteration in range(max_exchanges):
            if len(indices)==0:
                break
            state=indices%self.states; order=indices//self.states
            new_order=order+self.bounces[state]
            stop=self.stopped[state]|(new_order>max_bounces)
            bounce_tail+=float(field[stop,0].sum())
            current=state[~stop]; new_order=new_order[~stop]; v=field[~stop]
            reflected=self.bounces[current]>0
            v[reflected,4]=np.sum(v[reflected,1:4]*self.physical_outgoing[current[reflected]],axis=1)
            target=self.successor[current]; escaped=target<0
            collect(v[escaped],new_order[escaped],grid.directions[self.output[current[escaped]]%n])
            history.append(dict(iteration=iteration+1,active_states=len(indices),mass=float(field[:,0].sum()),
                                reflected_states=int(np.count_nonzero(self.bounces[state]>0))))
            indices,field=aggregate(new_order[~escaped]*self.states+target[~escaped],v[~escaped])
        exchange_tail=float(field[:,0].sum())
        # Diagnose projection-induced zero-collision cycles instead of silently
        # treating an interface-iteration cutoff as a physical collision tail.
        zero_cycle_mass=0.; cycles=set()
        for index,value in zip(indices,field):
            state=int(index%self.states); visited={}; path=[]
            while state>=0 and self.bounces[state]==0 and not self.stopped[state]:
                if state in visited:
                    cycle=path[visited[state]:]
                    pivot=cycle.index(min(cycle)); cycles.add(tuple(cycle[pivot:]+cycle[:pivot]))
                    zero_cycle_mass+=float(value[0]); break
                visited[state]=len(path); path.append(state); state=int(self.successor[state])
        residual=bounce_tail+exchange_tail
        total=float(source.weights.sum())
        if not np.isclose(mass.sum()+residual,total,rtol=1e-12,atol=1e-12):
            raise AssertionError("Discrete transport mass ledger changed")
        return dict(omega=float(omega.sum()),cell_direction_omega=float(cell_omega.sum()),
                    readout_projection_difference=float(omega.sum()-cell_omega.sum()),
                    escaped_mass=float(mass.sum()),residual_mass=residual,tail_bound=2*residual,
                    bounce_tail_mass=bounce_tail,exchange_tail_mass=exchange_tail,source_mass=total,
                    zero_collision_cycle_mass=zero_cycle_mass,zero_collision_cycles=[list(c) for c in sorted(cycles)],
                    order_ledger=[dict(order=i,mass=float(mass[i]),omega=float(omega[i])) for i in range(len(mass))],
                    history=history,seed_s=seed_s,query_s=time.perf_counter()-start,seed_primitive_tests=int(tests),
                    compiled_apply_primitive_tests=0,through_mode=self.through_mode)
