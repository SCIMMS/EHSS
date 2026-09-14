"""Support-exchange first-hit transport prototypes."""

from .experiments import PrototypeConfig, run_bump_experiment
from .geometry2d import BumpSpec, SegmentMesh2D, make_circle_cavity
from .geometry3d import SphericalCapBump, TriangleMesh3D, make_sphere_cavity
from .metrics import rasterize_first_hits, relative_frobenius_error, row_conservation_error
from .radiosity import make_radiosity_case, solve_radiosity
from .transport2d import transport_first_hits
from .transport3d import transport_first_hits_cone_bvh

__all__ = [
    "BumpSpec",
    "PrototypeConfig",
    "SegmentMesh2D",
    "SphericalCapBump",
    "TriangleMesh3D",
    "make_circle_cavity",
    "make_radiosity_case",
    "make_sphere_cavity",
    "rasterize_first_hits",
    "relative_frobenius_error",
    "row_conservation_error",
    "run_bump_experiment",
    "solve_radiosity",
    "transport_first_hits",
    "transport_first_hits_cone_bvh",
]
