"""Unbiased isotropic EHSS incidence on compact convex enclosing boundaries.

Directions are uniform on S2. Conditional positions are uniform in the
directional projection, and each ray carries projected_area / sample_count.
This changes sampling only; scattering remains in world coordinates.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.stats import qmc
from .inside_out_pa import unit_sphere, tangent_frames
from .pa_ellipsoidal import fit_centroid_ellipsoid
from .ehss_state import EHSSSource


@dataclass
class CompactBoundary:
    kind: str
    center: np.ndarray
    rotation: np.ndarray
    axes: np.ndarray
    prepare_s: float = 0.

    @classmethod
    def fit(cls, spheres, kind='ellipsoid'):
        start = time.perf_counter()
        center = spheres.centers.mean(axis=0)
        rotation = np.eye(3)
        if kind == 'sphere':
            radius = np.max(np.linalg.norm(spheres.centers-center, axis=1)+spheres.radii)*(1+1e-9)
            axes = np.full(3, radius)
        elif kind == 'ellipsoid':
            center, rotation, axes = fit_centroid_ellipsoid(spheres)
        elif kind == 'box':
            # Tight world-axis bounds of the complete collision balls.
            low = np.min(spheres.centers-spheres.radii[:, None], axis=0)
            high = np.max(spheres.centers+spheres.radii[:, None], axis=0)
            center = (low+high)/2
            axes = (high-low)/2*(1+1e-9)
        else:
            raise ValueError('Boundary must be sphere, ellipsoid, or box')
        return cls(kind, center, rotation, axes, time.perf_counter()-start)

    def source(self, power, seed):
        if not isinstance(power, (int, np.integer)) or power < 0:
            raise ValueError('Nonnegative integer Sobol power required')
        u = qmc.Sobol(5 if self.kind == 'box' else 4, scramble=True, seed=seed).random_base2(power)
        d = unit_sphere(u[:, :2])
        n = len(d)
        if self.kind == 'sphere':
            t, b = tangent_frames(d)
            radius = self.axes[0]
            rho, phi = radius*np.sqrt(u[:, 2]), 2*np.pi*u[:, 3]
            origins = self.center-radius*d+rho[:, None]*(np.cos(phi)[:, None]*t+np.sin(phi)[:, None]*b)
            area = np.full(n, np.pi*radius*radius)
        elif self.kind == 'box':
            local_d = d@self.rotation
            face_areas = 4*np.prod(self.axes)/self.axes
            projected = np.abs(local_d)*face_areas
            area = projected.sum(axis=1)
            face = (u[:, 2, None]*area[:, None] >= np.cumsum(projected, axis=1)).sum(axis=1)
            p = np.empty((n, 3))
            for axis in range(3):
                selected = face == axis
                p[selected, axis] = -np.sign(local_d[selected, axis])*self.axes[axis]
                other = [k for k in range(3) if k != axis]
                p[np.ix_(selected, other)] = (2*u[selected, 3:5]-1)*self.axes[other]
            # Start outside the numerical boundary, upstream on the same line.
            origins = self.center+p@self.rotation.T-1e-10*max(1., self.axes.max())*d
        else:
            t, b = tangent_frames(d)
            # Project the ellipsoid's linear image of the unit ball into d-perp.
            lt, lb = (t@self.rotation)*self.axes, (b@self.rotation)*self.axes
            g00 = np.sum(lt*lt, axis=1)
            g01 = np.sum(lt*lb, axis=1)
            g11 = np.sum(lb*lb, axis=1)
            l00 = np.sqrt(g00)
            l10 = g01/l00
            l11 = np.sqrt(np.maximum(0., g11-l10*l10))
            rho, phi = np.sqrt(u[:, 2]), 2*np.pi*u[:, 3]
            x, y = rho*np.cos(phi), rho*np.sin(phi)
            offset = (l00*x)[:, None]*t+(l10*x+l11*y)[:, None]*b
            extent = np.linalg.norm((d@self.rotation)*self.axes, axis=1)
            origins = self.center+offset-(extent*(1+1e-10))[:, None]*d
            area = np.pi*l00*l11
        return EHSSSource(origins, d, d.copy(), area/n, np.full(n, -1),
                          np.zeros(n, dtype=np.int64), f'isotropic projected {self.kind}')
