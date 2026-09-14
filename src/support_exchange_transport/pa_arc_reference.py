"""Continuous projected-circle union by merging covered boundary intervals.

Independent of pixel masks and BVH rays. Equivalent Green line integral to
inside_out_pa.circle_union_area, but avoids testing every arc midpoint against
every disk. Angular integration is still numerical and needs convergence QA.
"""
import math
import numpy as np
from numba import njit

@njit(cache=True)
def arc_piece(cx,cy,r,a,b):
    return .5*(r*r*(b-a)+r*cx*(math.sin(b)-math.sin(a))+r*cy*(math.cos(a)-math.cos(b)))

@njit(cache=True)
def merged_circle_area(xy,radii):
    n=len(radii);total=0.;tau=2*math.pi
    for i in range(n):
        starts=np.empty(2*n+2);ends=np.empty(2*n+2);k=0;covered=False;r=radii[i]
        for j in range(n):
            if j==i:continue
            dx=xy[j,0]-xy[i,0];dy=xy[j,1]-xy[i,1];d=math.sqrt(dx*dx+dy*dy)
            if d+r<=radii[j]:
                if d>0 or r<radii[j] or j<i:covered=True;break
            if abs(r-radii[j])<d<r+radii[j]:
                angle=math.atan2(dy,dx)
                half=math.acos(max(-1.,min(1.,(d*d+r*r-radii[j]**2)/(2*d*r))))
                a=(angle-half)%tau;b=(angle+half)%tau
                if a<=b:
                    starts[k]=a;ends[k]=b;k+=1
                else:
                    starts[k]=0.;ends[k]=b;k+=1
                    starts[k]=a;ends[k]=tau;k+=1
        if covered:continue
        order=np.argsort(starts[:k]);last=0.
        for q in order:
            a=starts[q];b=ends[q]
            if a>last:total+=arc_piece(xy[i,0],xy[i,1],r,last,a)
            last=max(last,b)
        if last<tau:total+=arc_piece(xy[i,0],xy[i,1],r,last,tau)
    return total

@njit(cache=True)
def continuous_views(centers,radii,directions):
    centered=centers.copy()
    for a in range(3):centered[:,a]-=np.mean(centered[:,a])
    out=np.empty(len(directions));xy=np.empty((len(radii),2))
    for k in range(len(directions)):
        d=directions[k]
        if abs(d[2])<.9:u=np.array([-d[1],d[0],0.])
        else:u=np.array([0.,-d[2],d[1]])
        u/=math.sqrt(np.dot(u,u));v=np.cross(d,u)
        for j in range(len(radii)):
            xy[j,0]=np.dot(centered[j],u);xy[j,1]=np.dot(centered[j],v)
        out[k]=merged_circle_area(xy,radii)
    return out
