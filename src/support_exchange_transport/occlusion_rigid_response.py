"""A restricted, exact planar two-reflection response for experiment 4.

This is deliberately NOT a general learned boundary field. On the certified
parallel-input branch of a two-mirror child, the remaining internal flight is
an affine translation and the outgoing direction is fixed. The immutable bank
stores those numerical quantities, replacing owned search AND reflection.
Current foreign geometry is checked before accepting the stored continuation.
"""
from dataclasses import dataclass
import numpy as np

from .occlusion_hierarchy import HitState, empty_stats, panel
from .occlusion_experiments import Experiment


def canonical_child(detail):
    return np.concatenate([panel([0., 0, 0], [.25, 0, -.25], [0, .45, 0], detail),
                           panel([.9, 0, 0], [.25, 0, -.25], [0, .45, 0], detail)])


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


@dataclass(frozen=True)
class PlanarResponse:
    detail: int
    canonical: np.ndarray
    displacement: np.ndarray
    outgoing: np.ndarray

    @classmethod
    def compile(cls, detail):
        arrays = [canonical_child(detail), np.array([.9, 0, 0]), np.array([0., 0, -1.])]
        for a in arrays:
            a.flags.writeable = False
        return cls(detail, *arrays)

    def proposals(self, points, directions, last, offset, rotation, translation):
        local_p = (points-translation)@rotation
        local_d = directions@rotation
        first = (last >= offset) & (last < offset+2*self.detail**2)
        # Roundoff tolerance only, not an ensemble approximation gate. Strictly
        # interior domain keeps the affine branch away from edge/path switching.
        eligible = first & (np.max(np.abs(local_d-[1, 0, 0]), axis=1) < 2e-12)
        eligible &= (np.abs(local_p[:, 0]+local_p[:, 2]) < 2e-12)
        eligible &= (np.abs(local_p[:, 0]) < .20) & (np.abs(local_p[:, 1]) < .40)
        idx = np.flatnonzero(eligible)
        point = local_p[idx]+self.displacement
        # Recover the original tessellation's face ID without a triangle query.
        uv = np.column_stack(((point[:, 0]-.9)/.25, point[:, 1]/.45))
        cell = (uv+1)*.5*self.detail
        ij = np.floor(cell).astype(int)
        frac = cell-ij
        face = offset+2*self.detail**2+2*(ij[:, 0]*self.detail+ij[:, 1])+(frac[:, 1] > frac[:, 0])
        return idx, face, np.tile(self.outgoing@rotation.T, (len(idx), 1))


class BoundResponses:
    def __init__(self, banks, poses, certificate_repeats=1):
        self.banks, self.poses = banks, poses
        self.checked = None
        self.valid = {}
        self.certificate_repeats = certificate_repeats

    def query(self, scene, origins, directions, last):
        if self.checked != (id(scene), scene.version):
            for p in scene.parts:
                bank = self.banks[p.name]
                rot, tr = self.poses[p.name]
                self.valid[p.name] = np.array_equal(bank.canonical@rot.T+tr, p.triangles)
            self.checked = (id(scene), scene.version)
        n = len(origins)
        faces, distances = np.full(n, -1, np.int64), np.full(n, np.inf)
        outgoing = np.full((n, 3), np.nan)
        pending = np.ones(n, bool)
        stats = empty_stats()
        stats.update(response_candidates=0, response_events=0, foreign_interruptions=0)
        for p in scene.parts:
            if not self.valid[p.name]:
                continue
            bank, (rotation, translation) = self.banks[p.name], self.poses[p.name]
            idx, proposed, out = bank.proposals(origins, directions, last, p.offset, rotation, translation)
            if not len(idx):
                continue
            stats['response_candidates'] += len(idx)
            foreign = scene.query(origins[idx], directions[idx], 'certified', exclude=p.name, certificate_repeats=self.certificate_repeats)
            for k, v in foreign.stats.items():
                stats[k] += v
            interrupted = (foreign.faces >= 0) & ((foreign.distances < .9) |
                ((foreign.distances == .9) & (foreign.faces < proposed)))
            faces[idx] = np.where(interrupted, foreign.faces, proposed)
            distances[idx] = np.where(interrupted, foreign.distances, .9)
            outgoing[idx[~interrupted]] = out[~interrupted]
            pending[idx] = False
            stats['response_events'] += int((~interrupted).sum())
            stats['foreign_interruptions'] += int(interrupted.sum())
        idx = np.flatnonzero(pending)
        if len(idx):
            direct = scene.query(origins[idx], directions[idx], 'certified', certificate_repeats=self.certificate_repeats)
            faces[idx], distances[idx] = direct.faces, direct.distances
            for k, v in direct.stats.items():
                stats[k] += v
        return HitState(faces, distances, stats, outgoing)


def rigid_experiment(detail=8, ray_side=12, seed=781):
    local = canonical_child(detail)
    A = (rotation_z(0), np.array([0., 0, .5]))
    poses = [dict(A=A, B=(rotation_z(a), np.array([.9, 0, z])))
             for a, z in [(0., -1.), (.65, -1.), (1.6, -1.), (np.pi, .8), (0., -1.)]]
    poses.insert(3, dict(A=A, B=(rotation_z(0), np.array([.45, 0, .5]))))
    receiver = panel([.8, 0, -3.5], [3, 0, 0], [0, 3, 0])
    E = np.array([[-.5, -1.5, -2.], [2.5, 1.5, 1.2]])
    parts = [dict(name=name, triangles=local@rot.T+tr, envelope=E.copy(), reflectivity=.85)
             for name, (rot, tr) in poses[0].items()]
    frames = [(name, {key: local@rot.T+tr for key, (rot, tr) in pose.items()})
              for name, pose in zip(['initial', 'relative_rotation', 'larger_rotation', 'foreign_interruption', 'self_occlusion', 'unshadow'], poses)]
    rng = np.random.default_rng(seed)
    origins = np.column_stack((rng.uniform(-.075, .075, ray_side**2), rng.uniform(-.075, .075, ray_side**2), np.full(ray_side**2, 2.)))
    directions = np.tile([0., 0, -1.], (len(origins), 1))
    return Experiment(4, 'rigid_child_T_current_G', receiver, parts, E, frames, origins, directions), poses
