"""Fixtures and evidence for the six ordered confined-support experiments."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .occlusion_hierarchy import ConfinedScene, Part, panel, source_grid


@dataclass
class Experiment:
    number: int
    name: str
    static: np.ndarray
    parts: list
    envelope: np.ndarray
    frames: list
    origins: np.ndarray
    directions: np.ndarray
    static_reflectivity: np.ndarray | None = None

    def scene(self):
        return ConfinedScene(self.static, [Part(**p) for p in self.parts], self.envelope,
                             self.static_reflectivity)


def first_hit_experiment(number, detail=8, ray_side=32, width=1.8, seed=731, envelope_width=2.3):
    receiver = panel([0, 0, -2], [3, 0, 0], [0, 3, 0])
    origins, directions = source_grid(ray_side, width, seed)
    E = np.array([[-envelope_width, -1.5, -.8], [envelope_width, 1.5, 1.]])
    if number == 1:
        fin = panel([-.55, 0, 0], [.4, 0, 0], [0, .75, 0], detail)
        moved = fin+[.9, 0, 0]
        parts = [dict(name='fin', triangles=fin, envelope=np.array([[-1., -.8, -.1], [.9, .8, .1]]))]
        frames = [('initial', {}), ('move', {'fin': moved}), ('return', {'fin': fin})]
        name = 'same_envelope_changed_shadow'
    elif number == 2:
        shield = panel([0, 0, .8], [1., 0, 0], [0, 1., 0])
        receiver = np.concatenate([receiver, shield])
        inside = panel([-.25, 0, 0], [.35, 0, 0], [0, .6, 0], detail)
        parts = [dict(name='interior', triangles=inside, envelope=np.array([[-.85, -.8, -.4], [.85, .8, .4]]))]
        frames = [('initial', {}), ('hidden_change_1', {'interior': inside+[.4, 0, .2]}),
                  ('hidden_change_2', {'interior': inside+[.15, .1, -.2]})]
        name = 'fixed_shield_deferred_interior'
    elif number == 3:
        shutter = panel([0, 0, .7], [.9, 0, 0], [0, 1., 0])
        inside = panel([-.25, 0, 0], [.25, 0, 0], [0, .65, 0], detail)
        parts = [dict(name='shutter', triangles=shutter, envelope=np.array([[-.95, -1.05, .65], [2.3, 1.05, .75]])),
                 dict(name='interior', triangles=inside, envelope=np.array([[-.8, -.8, -.35], [.8, .8, .35]]))]
        frames = [('closed', {}), ('hidden_change', {'interior': inside+[.55, 0, .2]}),
                  ('open', {'shutter': shutter+[1.35, 0, 0]}), ('close', {'shutter': shutter}),
                  ('hidden_change_again', {'interior': inside+[.1, 0, -.2]}),
                  ('reopen', {'shutter': shutter+[1.35, 0, 0]})]
        name = 'shutter_hidden_edit_reactivation'
    else:
        raise ValueError(number)
    return Experiment(number, name, receiver, parts, E, frames, origins, directions)


def compare_hits(actual, expected):
    same = np.isfinite(actual.distances) & np.isfinite(expected.distances)
    return dict(target_mismatches=int(np.count_nonzero(actual.faces != expected.faces)),
                finite_mismatches=int(np.count_nonzero(np.isfinite(actual.distances) != np.isfinite(expected.distances))),
                distance_error=float(np.max(np.abs(actual.distances[same]-expected.distances[same]), initial=0)))


def run_first_hit(experiment):
    scenes = {r: experiment.scene() for r in ('full', 'local', 'certified')}
    rows = []
    previous = None
    for frame, changes in experiment.frames:
        for s in scenes.values():
            s.update(changes)
        reference = scenes['full'].query(experiment.origins, experiment.directions, 'full')
        changed = np.zeros(len(reference.faces), bool) if previous is None else (
            (reference.faces != previous.faces) | ~np.isclose(reference.distances, previous.distances, atol=1e-12, rtol=0))
        for route, scene in scenes.items():
            actual = reference if route == 'full' else scene.query(experiment.origins, experiment.directions, route, cache_static=True)
            row = dict(experiment=experiment.number, name=experiment.name, frame=frame, route=route,
                       rays=len(reference.faces), changed_rays=int(changed.sum()),
                       receiver_fraction=float(np.mean((actual.faces >= 0) & (actual.faces < 2))),
                       new_receiver_rays=0 if previous is None else int(np.count_nonzero((reference.faces >= 0) & (reference.faces < 2) & (previous.faces >= 2))),
                       missed_changed_rays=int(np.count_nonzero(changed & (actual.faces != reference.faces))),
                       part_epochs={p.name: [p.epoch, p.prepared_epoch] for p in scene.parts},
                       **compare_hits(actual, reference), **actual.stats)
            rows.append(row)
        previous = reference
    return rows, scenes


def reflecting_experiment(detail=8, ray_side=12, seed=51):
    # Four fixed mirrors lie entirely outside E. The original flight misses E;
    # the next flight enters it. The unperturbed path later exits and reenters E.
    mirrors = [panel([-1.5, 0, 1.8], [.2, 0, -.2], [0, .5, 0]),
               panel([2.5, 0, 0], [.25, 0, -.25], [0, .5, 0]),
               panel([2.5, 0, -1.8], [.25, 0, .25], [0, .5, 0]),
               panel([0, 0, -1.8], [.25, 0, -.25], [0, .5, 0])]
    receiver = panel([0, 0, 3.], [.5, 0, 0], [0, .5, 0])
    # Fixed enclosing cavity with a roof aperture, not just an open mirror chain.
    # The receiver lies beyond the aperture, outside both cavity and E.
    walls = [panel([-3.5, 0, .2], [0, 1.2, 0], [0, 0, 2.6]),
             panel([3.5, 0, .2], [0, 1.2, 0], [0, 0, 2.6]),
             panel([0, -1.2, .2], [3.5, 0, 0], [0, 0, 2.6]),
             panel([0, 1.2, .2], [3.5, 0, 0], [0, 0, 2.6]),
             panel([0, 0, -2.4], [3.5, 0, 0], [0, 1.2, 0]),
             panel([-2., 0, 2.8], [1.5, 0, 0], [0, 1.2, 0]),
             panel([2., 0, 2.8], [1.5, 0, 0], [0, 1.2, 0]),
             panel([0, -.85, 2.8], [.5, 0, 0], [0, .35, 0]),
             panel([0, .85, 2.8], [.5, 0, 0], [0, .35, 0])]
    static = np.concatenate([receiver]+mirrors+walls)
    initial = panel([-1.5, 0, 0], [.25, 0, -.25], [0, .45, 0], detail)
    angle = .15
    rot = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    changed = (initial-[-1.5, 0, 0])@rot.T+[-1.5, 0, 0]
    rng = np.random.default_rng(seed)
    origins = np.column_stack((np.full(ray_side**2, -3.), rng.uniform(-.2, .2, ray_side**2),
                               rng.uniform(1.75, 1.85, ray_side**2)))
    directions = np.tile([1., 0, 0], (len(origins), 1))
    return Experiment(5, 'indirect_entry_and_reentry', static,
        [dict(name='fin', triangles=initial, envelope=np.array([[-1.9, -.5, -.4], [-1.1, .5, .4]]), reflectivity=.8)],
        np.array([[-2., -1, -1], [2., 1, 1]]),
        [('initial', {}), ('rotate', {'fin': changed}), ('return', {'fin': initial})],
        origins, directions, np.r_[np.zeros(2), np.full(8, .8), np.full(18, .6)])
