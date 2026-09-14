"""Connected concave cavity with confined internal edits and a sliding lid.

The fixed shell and deforming floor share their boundary vertices. Refinement
subdivides existing flat triangles, preserving the same continuous geometry.
The shell, source, materials, root and child envelopes are fixed per sequence.
"""
from __future__ import annotations

import numpy as np

from .occlusion_experiments import Experiment
from .occlusion_hierarchy import panel


def subdivide(triangles, n):
    """Barycentric subdivision with n*n faces per original face."""
    weights = []
    for i in range(n):
        for j in range(n-i):
            weights.append([[i/n, j/n], [(i+1)/n, j/n], [i/n, (j+1)/n]])
            if i+j < n-1:
                weights.append([[(i+1)/n, j/n], [(i+1)/n, (j+1)/n], [i/n, (j+1)/n]])
    uv = np.asarray(weights)
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    return (a[:, None, None, :] + uv[None, :, :, :1]*(b-a)[:, None, None, :]
            + uv[None, :, :, 1:]*(c-a)[:, None, None, :]).reshape(-1, 3, 3)


def floor_vertices(version):
    x, y = np.meshgrid(np.linspace(-1., 1., 9), np.linspace(-1., 1., 9), indexing='ij')
    bump = (1-x*x)*(1-y*y)
    deformation = [0*x, .14*bump, -.12*bump,
                   .13*bump*(.35+np.sin(np.pi*x)), -.14*bump*(.4+np.sin(np.pi*y))][version]
    z = -.85+.42*(x*x+y*y)+.045*np.sin(3*np.pi*x)*np.sin(3*np.pi*y)*bump+deformation
    return np.stack([x, y, z], axis=-1)


def floor_tiles(version, detail):
    vertices = floor_vertices(version)
    tiles = []
    for tx in range(4):
        for ty in range(4):
            faces = []
            for i in range(2*tx, 2*tx+2):
                for j in range(2*ty, 2*ty+2):
                    a, b, c, d = vertices[i, j], vertices[i+1, j], vertices[i+1, j+1], vertices[i, j+1]
                    faces.extend([[a, b, c], [a, c, d]])
            tiles.append(subdivide(np.asarray(faces), detail))
    return tiles


def fixed_shell():
    vertices = floor_vertices(0)
    faces = []
    for edge in [vertices[0], vertices[-1], vertices[:, 0], vertices[:, -1]]:
        for a, b in zip(edge[:-1], edge[1:]):
            c, d = b.copy(), a.copy()
            c[2] = d[2] = .8
            faces.extend([[a, b, c], [a, c, d]])
    return np.asarray(faces)


def concave_experiment(detail=4, ray_side=32, seed=731, oblique=False, split=True):
    if detail < 1:
        raise ValueError('Positive refinement required')
    versions = [floor_tiles(v, detail) for v in range(5)]
    if not split:
        versions = [[np.concatenate(v)] for v in versions]
    floor_names = [f'floor_{i:02}' for i in range(len(versions[0]))]
    shutter = panel([0., 0., .8], [1.05, 0., 0.], [0., 1.05, 0.])
    E = np.array([[-1.1, -1.1, -1.1], [3.3, 1.1, .9]])
    parts = [dict(name='shutter', triangles=shutter,
                  envelope=np.array([[-1.06, -1.06, .79], [3.26, 1.06, .81]]), reflectivity=0.)]
    for i, name in enumerate(floor_names):
        # All versions are declared before any query. This bound is never fit
        # to visibility results, and remains fixed when hidden geometry changes.
        verts = np.concatenate([v[i].reshape(-1, 3) for v in versions])
        bounds = np.array([verts.min(axis=0)-1e-9, verts.max(axis=0)+1e-9])
        parts.append(dict(name=name, triangles=versions[0][i], envelope=bounds, reflectivity=.65))
    changes = lambda v: dict(zip(floor_names, versions[v]))
    frames = [('closed_initial', {}), ('closed_edit_1', changes(1)), ('closed_edit_2', changes(2)),
              ('partial', {'shutter': shutter+[.85, 0., 0.]}), ('partial_edit', changes(3)),
              ('reshadow', {'shutter': shutter}), ('closed_edit_3', changes(4)),
              ('reopen_partial', {'shutter': shutter+[.85, 0., 0.]}),
              ('open_full', {'shutter': shutter+[2.2, 0., 0.]}),
              ('reshadow_final', {'shutter': shutter})]
    rng = np.random.default_rng(seed)
    uv = (np.indices((ray_side, ray_side)).reshape(2, -1).T+rng.uniform(.13, .87, (ray_side**2, 2)))/ray_side
    aperture_points = np.column_stack(((uv-.5)*1.96, np.full(ray_side**2, .8)))
    direction = np.array([.65, .17, -1.] if oblique else [0., 0., -1.])
    direction /= np.linalg.norm(direction)
    origins = aperture_points-direction*((3.-.8)/-direction[2])
    directions = np.tile(direction, (len(origins), 1))
    # First two static triangles are an exterior catcher. Other static faces
    # are the object's own fixed cavity shell, lying within E.
    receiver = panel([0., 0., -1.5], [4., 0., 0.], [0., 4., 0.])
    shell = fixed_shell()
    static = np.concatenate([receiver, shell])
    return Experiment(7, 'concave_self_occlusion', static, parts, E, frames,
                      origins, directions, np.r_[np.zeros(2), np.full(len(shell), .65)])


def floor_hit_mask(scene, faces):
    return faces >= scene.parts[1].offset


def floor_epoch_state(scene):
    return {p.name: [p.epoch, p.prepared_epoch] for p in scene.parts[1:]}


def path_errors(actual, reference):
    return dict(divergent_rays=int(np.count_nonzero(np.any(actual.history != reference.history, axis=1))),
                **{key+'_error': float(np.max(np.abs(getattr(actual, key)-getattr(reference, key)), initial=0))
                   for key in ['points', 'directions', 'absorbed', 'escaped', 'residual']})


class IndependentScene:
    """Reference adapter using only independent 3x3 triangle solves."""
    def __init__(self, scene):
        self.scene, self.envelope = scene, scene.envelope

    def all_triangles(self):
        return self.scene.all_triangles()

    def reflectivities(self):
        return self.scene.reflectivities()

    def query(self, origins, directions, *args, **kwargs):
        from .occlusion_hierarchy import independent_trace
        return independent_trace(self.all_triangles(), origins, directions)
