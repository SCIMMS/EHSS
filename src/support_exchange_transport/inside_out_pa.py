"""Inside-out PA reference and independent projected-circle integration.

Units are arbitrary but consistent. Radii already include the chosen gas radius.
No EHSS reflection, ghost construction or field compression is claimed here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numba import njit
from scipy.integrate import quad
from scipy.stats import qmc


@dataclass(frozen=True)
class Spheres:
    centers: np.ndarray
    radii: np.ndarray
    deduplicate: bool = True

    def __post_init__(self):
        c = np.ascontiguousarray(self.centers, dtype=np.float64)
        r = np.ascontiguousarray(self.radii, dtype=np.float64)
        if c.ndim != 2 or c.shape[1] != 3 or r.shape != (len(c),) or len(c) == 0:
            raise ValueError("Expected nonempty centers (N,3) and radii (N,).")
        if not np.isfinite(c).all() or not np.isfinite(r).all() or np.any(r <= 0):
            raise ValueError("Finite centers and positive finite radii required.")
        # Exact duplicates need a single surface owner. Keep first occurrence.
        if self.deduplicate:
            _, ids = np.unique(np.column_stack((c, r)), axis=0, return_index=True)
            ids.sort()
        else:
            # Fixed-identity motion callers guarantee no coincident surfaces.
            ids = np.arange(len(c))
        object.__setattr__(self, "centers", c[ids].copy())
        object.__setattr__(self, "radii", r[ids].copy())

    @property
    def area(self):
        return float(4 * np.pi * np.dot(self.radii, self.radii))


def unit_sphere(uv):
    z = 1 - 2 * uv[:, 0]
    phi = 2 * np.pi * uv[:, 1]
    rho = np.sqrt(np.maximum(0, 1 - z * z))
    return np.column_stack((rho * np.cos(phi), rho * np.sin(phi), z))


def tangent_frames(normals):
    axes = np.zeros_like(normals)
    axes[:, 2] = 1
    axes[np.abs(normals[:, 2]) > .9] = [1, 0, 0]
    t = np.cross(axes, normals)
    t /= np.linalg.norm(t, axis=1)[:, None]
    return t, np.cross(normals, t)


@dataclass
class Source:
    origins: np.ndarray
    directions: np.ndarray
    owners: np.ndarray
    area: float

    def pa(self, blocked):
        return self.area * float(np.mean(~blocked)) / 4


def sample_source(spheres: Spheres, power: int, seed: int) -> Source:
    """Area-selected full sphere surfaces and a cosine hemisphere (Sobol RQMC).

    Buried surface points score zero through their positive exit from a covering
    sphere; no explicit union meshing or surface-area estimate is needed.
    """
    u = qmc.Sobol(5, scramble=True, seed=seed).random_base2(power)
    weights = spheres.radii ** 2
    ids = np.searchsorted(np.cumsum(weights) / weights.sum(), u[:, 0], side="right")
    normals = unit_sphere(u[:, 1:3])
    t, b = tangent_frames(normals)
    phi = 2 * np.pi * u[:, 4]
    rho = np.sqrt(u[:, 3])
    directions = (np.sqrt(1 - u[:, 3])[:, None] * normals
                  + (rho * np.cos(phi))[:, None] * t
                  + (rho * np.sin(phi))[:, None] * b)
    origins = spheres.centers[ids] + spheres.radii[ids, None] * normals
    return Source(np.ascontiguousarray(origins), np.ascontiguousarray(directions),
                  ids.astype(np.int64), spheres.area)


@njit(cache=True)
def sphere_hit(o, d, c, r):
    # Source sphere is excluded by identity. Other spheres are tested including
    # rays starting inside them. Unit directions are part of the API contract.
    x, y, z = o[0]-c[0], o[1]-c[1], o[2]-c[2]
    b = x*d[0] + y*d[1] + z*d[2]
    cc = x*x + y*y + z*z - r*r
    disc = b*b - cc
    return disc >= 0 and -b + math.sqrt(max(0., disc)) > 0


@njit(cache=True)
def direct_blocked(origins, directions, owners, centers, radii):
    blocked = np.zeros(len(origins), dtype=np.bool_)
    tests = 0
    for i in range(len(origins)):
        for j in range(len(centers)):
            if j == owners[i]:
                continue
            tests += 1
            if sphere_hit(origins[i], directions[i], centers[j], radii[j]):
                blocked[i] = True
                break
    return blocked, tests


def inside_out(spheres, source):
    blocked, tests = direct_blocked(source.origins, source.directions, source.owners,
                                    spheres.centers, spheres.radii)
    return source.pa(blocked), blocked, tests


def outer_sphere_crossings(source, center, radius):
    """Readout locations only; flux packet weights are unchanged."""
    rel = source.origins - np.asarray(center)
    if np.any(np.linalg.norm(rel, axis=1) >= radius):
        raise ValueError("Collector must enclose every source point strictly.")
    b = np.sum(rel * source.directions, axis=1)
    t = -b + np.sqrt(b*b + radius*radius - np.sum(rel*rel, axis=1))
    return source.origins + t[:, None] * source.directions


@njit(cache=True)
def circle_union_area(centers, radii):
    """Exact exposed circular-arc line integral, independent of ray transport.

    Split each circle at all intersections; integrate its uncovered arcs using
    Green's theorem. Quadrature error occurs only in orientation integration.
    """
    total = 0.
    n = len(radii)
    for i in range(n):
        r = radii[i]
        angles = np.empty(2*n + 2)
        angles[0], angles[1] = 0., 2*math.pi
        count = 2
        covered = False
        for j in range(n):
            if i == j:
                continue
            dx, dy = centers[j, 0]-centers[i, 0], centers[j, 1]-centers[i, 1]
            dist = math.sqrt(dx*dx+dy*dy)
            if dist + r <= radii[j]:
                # Coincident equal disks: first index owns the boundary.
                if dist > 0 or r < radii[j] or j < i:
                    covered = True
                    break
            if abs(r-radii[j]) < dist < r+radii[j]:
                a = math.atan2(dy, dx)
                w = math.acos(max(-1., min(1., (dist*dist+r*r-radii[j]**2)/(2*dist*r))))
                angles[count] = (a-w) % (2*math.pi)
                angles[count+1] = (a+w) % (2*math.pi)
                count += 2
        if covered:
            continue
        angles = np.sort(angles[:count])
        for k in range(count-1):
            a, b = angles[k], angles[k+1]
            mid = (a+b)/2
            px = centers[i, 0] + r*math.cos(mid)
            py = centers[i, 1] + r*math.sin(mid)
            visible = True
            for j in range(n):
                if j != i and (px-centers[j, 0])**2+(py-centers[j, 1])**2 < radii[j]**2:
                    visible = False
                    break
            if visible:
                total += .5*(r*r*(b-a) + r*centers[i, 0]*(math.sin(b)-math.sin(a))
                             + r*centers[i, 1]*(math.cos(a)-math.cos(b)))
    return total


@njit(cache=True)
def _projected_areas(centers, radii, tangents, bitangents):
    result = np.empty(len(tangents))
    for k in range(len(tangents)):
        xy = np.empty((len(centers), 2))
        for j in range(len(centers)):
            xy[j, 0] = np.dot(centers[j], tangents[k])
            xy[j, 1] = np.dot(centers[j], bitangents[k])
        result[k] = circle_union_area(xy, radii)
    return result


def view_based_pa(spheres, power, seed):
    dirs = unit_sphere(qmc.Sobol(2, scramble=True, seed=seed).random_base2(power))
    t, b = tangent_frames(dirs)
    centers = np.ascontiguousarray(spheres.centers-spheres.centers.mean(axis=0))
    return float(_projected_areas(centers, spheres.radii, t, b).mean())


def two_sphere_mean_pa(radius_a, radius_b, separation):
    """Independent 1-D adaptive quadrature of the analytic disk-lens formula."""
    a, b, distance = float(radius_a), float(radius_b), float(separation)
    if a <= 0 or b <= 0 or distance < 0:
        raise ValueError("Positive radii and nonnegative separation required.")

    def area(mu):
        d = distance * math.sqrt(max(0., 1-mu*mu))
        if d <= abs(a-b):
            return math.pi*max(a,b)**2
        if d >= a+b:
            return math.pi*(a*a+b*b)
        ca = max(-1., min(1., (d*d+a*a-b*b)/(2*d*a)))
        cb = max(-1., min(1., (d*d+b*b-a*a)/(2*d*b)))
        lens = a*a*math.acos(ca)+b*b*math.acos(cb)
        lens -= .5*math.sqrt(max(0., (-d+a+b)*(d+a-b)*(d-a+b)*(d+a+b)))
        return math.pi*(a*a+b*b)-lens

    points = [math.sqrt(1-(x/distance)**2) for x in [abs(a-b), a+b]
              if distance > 0 and 0 < x < distance]
    return quad(area, 0., 1., points=points, epsabs=1e-11, epsrel=1e-11)
