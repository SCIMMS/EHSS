"""Radius-aware sampled onion patches with conservative analytic ray bounds.

Peeling uses extrema of sphere support functions on a fixed direction set, not
an exact weighted convex hull. All atoms remain assigned exactly once. The
shared top tree uses AABBs; optional leaf bounds isolate sphere/sector effects.
Directions must be unit length, as in the existing PA kernels.
"""
import numpy as np
from time import perf_counter
from numba import njit
from .pa_spatial import build_tree, box_hit
from .inside_out_pa import sphere_hit


def sampled_layers(centers, radii, directions=128):
    k = np.arange(directions)
    z = 1 - 2 * (k + .5) / directions
    phi = k * (np.pi * (3 - np.sqrt(5)))
    u = np.column_stack((np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z))
    remaining = np.arange(len(radii)); layers = []
    scores = (centers-centers.mean(axis=0)) @ u.T + radii[:, None]
    while len(remaining):
        picked = np.unique(np.argmax(scores[remaining], axis=0))
        layers.append(remaining[picked])
        keep = np.ones(len(remaining), bool); keep[picked] = False
        remaining = remaining[keep]
    return layers


def spatial_patches(centers, layers, size):
    patches = []
    def split(ids):
        if len(ids) <= size:
            patches.append(ids)
        else:
            axis = np.argmax(np.ptp(centers[ids], axis=0))
            order = ids[np.argsort(centers[ids, axis], kind='stable')]
            middle = len(order)//2
            split(order[:middle]); split(order[middle:])
    for layer in layers: split(layer)
    return patches


@njit(cache=True)
def cone_interval(x, d, axis, cosine, near, far):
    """Whether a forward cone meets a *given* closed t interval."""
    return _cone_terms(np.dot(x,axis),np.dot(d,axis),np.dot(x,d),np.dot(x,x),
                       np.dot(d,d),cosine,near,far)


@njit(cache=True)
def _cone_terms(z0, zd, xd, xx, dd, cosine, near, far):
    if far < near: return False
    if cosine < 0: return True  # intentionally disabled wide cone
    tol = 1e-12 * (1 + np.sqrt(xx) + abs(near) + abs(far))
    if zd == 0:
        if z0 < -tol: return False
    elif zd > 0: near = max(near, (-tol-z0)/zd)
    else: far = min(far, (-tol-z0)/zd)
    if far < near: return False
    cc = cosine*cosine
    a = zd*zd - cc*dd
    b = 2*(z0*zd - cc*xd)
    c = z0*z0 - cc*xx
    maximum = max((a*near+b)*near+c, (a*far+b)*far+c)
    if a < 0:
        vertex = -b/(2*a)
        if near <= vertex <= far: maximum = max(maximum, (a*vertex+b)*vertex+c)
    scale = 1 + abs(a)*max(near*near, far*far) + abs(b)*max(abs(near),abs(far)) + abs(c)
    return maximum >= -1e-12*scale


@njit(cache=True)
def sector_hit(o, d, apex, axis, cosine, inner, outer):
    x=o[0]-apex[0]; y=o[1]-apex[1]; z=o[2]-apex[2]
    b=x*d[0]+y*d[1]+z*d[2]; xx=x*x+y*y+z*z
    disc = b*b-xx+outer*outer
    tol = 1e-12*(1+b*b+xx+outer*outer)
    if disc < -tol: return False
    root = np.sqrt(max(0.,disc)); near = max(0.,-b-root); far = -b+root
    if far < near: return False
    z0=x*axis[0]+y*axis[1]+z*axis[2]
    zd=d[0]*axis[0]+d[1]*axis[1]+d[2]*axis[2]
    dd=d[0]*d[0]+d[1]*d[1]+d[2]*d[2]
    inner_disc = b*b-xx+inner*inner
    if inner <= 0 or inner_disc <= 0:
        return _cone_terms(z0,zd,b,xx,dd,cosine,near,far)
    root = np.sqrt(inner_disc)
    # The radial hole can leave two disjoint intervals. Test both explicitly.
    return (_cone_terms(z0,zd,b,xx,dd,cosine,near,min(far,-b-root)) or
            _cone_terms(z0,zd,b,xx,dd,cosine,max(near,-b+root),far))


@njit(cache=True)
def query_patches(origins, directions, owners, centers, radii,
                  lo, hi, end, first, count, ids, offsets, members,
                  balls, ball_radii, apex, axes, cosines, inner, outer, mode):
    result = np.zeros(len(origins), np.bool_)
    boxes = 0; bounds = 0; rejected = 0; tests = 0
    for i in range(len(origins)):
        node = 0
        while node < len(end) and not result[i]:
            boxes += 1
            if not box_hit(origins[i], directions[i], lo[node], hi[node]):
                node = end[node]; continue
            for k in range(first[node],first[node]+count[node]):
                p = ids[k]; bounds += 1; hit = True
                if mode == 1:
                    hit = sphere_hit(origins[i],directions[i],balls[p],ball_radii[p])
                elif mode == 2:
                    hit = sector_hit(origins[i],directions[i],apex,axes[p],cosines[p],inner[p],outer[p])
                if not hit: rejected += 1; continue
                for j in range(offsets[p],offsets[p+1]):
                    atom = members[j]
                    if atom == owners[i]: continue
                    tests += 1
                    if sphere_hit(origins[i],directions[i],centers[atom],radii[atom]):
                        result[i] = True; break
                if result[i]: break
            node += 1
    return result, boxes, bounds, rejected, tests


class OnionIndex:
    def __init__(self, spheres, size=8, partition='onion', mode='sector'):
        start=perf_counter()
        self.spheres = spheres
        c, r = spheres.centers, spheres.radii
        self.layers = sampled_layers(c,r) if partition == 'onion' else [np.arange(len(r))]
        self.patches = spatial_patches(c,self.layers,size)
        self.offsets = np.array([0]+list(np.cumsum([len(p) for p in self.patches])),np.int64)
        self.members = np.concatenate(self.patches)
        partition_done=perf_counter()
        self.apex = c.mean(axis=0)
        lows=[]; highs=[]; balls=[]; br=[]; axes=[]; cs=[]; ins=[]; outs=[]
        for ids in self.patches:
            xyz=c[ids]; rr=r[ids]
            low=(xyz-rr[:,None]).min(axis=0); high=(xyz+rr[:,None]).max(axis=0)
            lows.append(low); highs.append(high)
            ball=(low+high)/2; balls.append(ball)
            br.append(np.max(np.linalg.norm(xyz-ball,axis=1)+rr)*(1+1e-12) if mode=='sphere' else 0.)
            if mode!='sector':
                axes.append(np.array([0.,0.,1.])); cs.append(-1.); ins.append(0.); outs.append(0.)
                continue
            rel=xyz-self.apex; length=np.linalg.norm(rel,axis=1)
            axis=rel.mean(axis=0); norm=np.linalg.norm(axis)
            axis=axis/norm if norm>1e-14 else np.array([0.,0.,1.]); axes.append(axis)
            if np.any(length <= rr): cosine=-1.
            else:
                angle=np.max(np.arccos(np.clip(rel@axis/length,-1,1))+np.arcsin(rr/length))
                cosine=np.cos(angle+1e-12) if angle < np.pi/2-1e-12 else -1.
            cs.append(cosine)
            ins.append(max(0.,np.min(length-rr)-1e-10))
            outs.append(np.max(length+rr)+1e-10)
        self.balls=np.array(balls); self.ball_radii=np.array(br)
        self.axes=np.array(axes); self.cosines=np.array(cs)
        self.inner=np.array(ins); self.outer=np.array(outs)
        self.mode={'aabb':0,'sphere':1,'sector':2}[mode]
        bounds_done=perf_counter()
        self.tree=build_tree(np.array(lows),np.array(highs),leaf_size=1)
        self.preparation=dict(partition_s=partition_done-start,bounds_s=bounds_done-partition_done,
                              tree_s=perf_counter()-bounds_done)

    def query(self, source):
        return query_patches(source.origins,source.directions,source.owners,
            self.spheres.centers,self.spheres.radii,*self.tree.arrays,
            self.offsets,self.members,self.balls,self.ball_radii,self.apex,
            self.axes,self.cosines,self.inner,self.outer,self.mode)
