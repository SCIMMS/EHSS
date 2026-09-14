from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time

import numpy as np

from .geometry2d import BumpSpec, make_circle_cavity
from .metrics import (
    hit_mismatch_count,
    rasterize_first_hits,
    relative_frobenius_error,
    row_conservation_error,
    support_exchange_matrix,
)
from .rays2d import make_lambertian_rays
from .raytrace2d import FirstHitMap2D, trace_first_hits
from .transport2d import TransportResult2D, transport_first_hits, transport_first_hits_event_tiled


@dataclass(frozen=True)
class PrototypeConfig:
    n_segments: int = 96
    rays_per_source: int = 48
    steps: int = 6
    radius: float = 1.0
    amplitude: float = 0.18
    width: float = 0.28
    centers: tuple[float, ...] = (0.0,)
    neighbor_radius: int = 1
    tile_margin: float = 0.01
    support_abs: bool = False
    profile_kind: str = "gaussian"

    def bumps(self) -> tuple[BumpSpec, ...]:
        return tuple(
            BumpSpec(
                center=center,
                amplitude=self.amplitude,
                width=self.width,
                label=i + 1,
                profile_kind=self.profile_kind,
            )
            for i, center in enumerate(self.centers)
        )


@dataclass(frozen=True)
class StepMetrics:
    alpha: float
    matrix_error: float
    tile_matrix_error: float
    row_error_transport: float
    row_error_tile: float
    row_error_full: float
    hit_mismatches: int
    tile_hit_mismatches: int
    event_count: int
    tile_event_count: int
    tile_footprint_ray_count: int
    tile_footprint_overlap_ray_count: int
    full_query_count: int
    transport_query_count: int
    tile_query_count: int
    active_source_count: int
    changed_segment_count: int
    support_exchange: np.ndarray
    full_wall_time_s: float
    transport_wall_time_s: float
    tile_wall_time_s: float

    @property
    def query_ratio(self) -> float:
        return self.transport_query_count / self.full_query_count

    @property
    def query_speedup(self) -> float:
        return self.full_query_count / self.transport_query_count

    @property
    def tile_query_speedup(self) -> float:
        if self.tile_query_count == 0:
            return float("inf")
        return self.full_query_count / self.tile_query_count

    @property
    def wall_time_speedup(self) -> float:
        return self.full_wall_time_s / self.transport_wall_time_s

    @property
    def tile_wall_time_speedup(self) -> float:
        if self.tile_wall_time_s == 0.0:
            return float("inf")
        return self.full_wall_time_s / self.tile_wall_time_s


@dataclass(frozen=True)
class ExperimentResult:
    config: PrototypeConfig
    steps: tuple[StepMetrics, ...]
    final_full_state: FirstHitMap2D
    final_transport_state: FirstHitMap2D
    final_transport: TransportResult2D
    final_tile_transport: TransportResult2D


def run_bump_experiment(config: PrototypeConfig) -> ExperimentResult:
    if config.steps <= 0:
        raise ValueError("steps must be positive.")

    bumps = config.bumps()
    old_mesh = make_circle_cavity(
        config.n_segments,
        config.radius,
        bumps,
        alpha=0.0,
        support_abs=config.support_abs,
    )
    old_state = trace_first_hits(old_mesh, make_lambertian_rays(old_mesh, config.rays_per_source))
    transport_state = old_state
    step_metrics: list[StepMetrics] = []
    final_full_state = old_state
    final_transport: TransportResult2D | None = None
    final_tile_transport: TransportResult2D | None = None

    for alpha in np.linspace(1.0 / config.steps, 1.0, config.steps):
        new_mesh = make_circle_cavity(
            config.n_segments,
            config.radius,
            bumps,
            alpha=float(alpha),
            support_abs=config.support_abs,
        )
        t0 = time.perf_counter()
        full_state = trace_first_hits(new_mesh, make_lambertian_rays(new_mesh, config.rays_per_source))
        full_wall_time_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        transport = transport_first_hits(
            transport_state,
            old_mesh,
            new_mesh,
            config.rays_per_source,
            neighbor_radius=config.neighbor_radius,
        )
        transport_wall_time_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        tile_transport = transport_first_hits_event_tiled(
            transport_state,
            old_mesh,
            new_mesh,
            config.rays_per_source,
            neighbor_radius=config.neighbor_radius,
            tile_margin=config.tile_margin,
        )
        tile_wall_time_s = time.perf_counter() - t0

        f_full = rasterize_first_hits(full_state, new_mesh.n_segments)
        f_transport = rasterize_first_hits(transport.state, new_mesh.n_segments)
        f_tile = rasterize_first_hits(tile_transport.state, new_mesh.n_segments)
        exchange = support_exchange_matrix(
            transport_state.hits,
            transport.state.hits,
            transport.state.weights,
            new_mesh.support_labels,
        )

        step_metrics.append(
            StepMetrics(
                alpha=float(alpha),
                matrix_error=relative_frobenius_error(f_transport, f_full),
                tile_matrix_error=relative_frobenius_error(f_tile, f_full),
                row_error_transport=row_conservation_error(f_transport),
                row_error_tile=row_conservation_error(f_tile),
                row_error_full=row_conservation_error(f_full),
                hit_mismatches=hit_mismatch_count(transport.state, full_state),
                tile_hit_mismatches=hit_mismatch_count(tile_transport.state, full_state),
                event_count=transport.event_count,
                tile_event_count=tile_transport.event_count,
                tile_footprint_ray_count=tile_transport.footprint_ray_count,
                tile_footprint_overlap_ray_count=tile_transport.footprint_overlap_ray_count,
                full_query_count=full_state.query_count,
                transport_query_count=transport.query_count,
                tile_query_count=tile_transport.query_count,
                active_source_count=transport.active_source_count,
                changed_segment_count=transport.changed_segment_count,
                support_exchange=exchange,
                full_wall_time_s=full_wall_time_s,
                transport_wall_time_s=transport_wall_time_s,
                tile_wall_time_s=tile_wall_time_s,
            )
        )

        old_mesh = new_mesh
        transport_state = transport.state
        final_full_state = full_state
        final_transport = transport
        final_tile_transport = tile_transport

    assert final_transport is not None
    assert final_tile_transport is not None
    return ExperimentResult(
        config=config,
        steps=tuple(step_metrics),
        final_full_state=final_full_state,
        final_transport_state=transport_state,
        final_transport=final_transport,
        final_tile_transport=final_tile_transport,
    )


def plot_experiment(result: ExperimentResult, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mpl_config = output_path.parent / ".matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

    import matplotlib.pyplot as plt

    alphas = np.array([step.alpha for step in result.steps])
    matrix_errors = np.array([step.matrix_error for step in result.steps])
    query_ratios = np.array([step.query_ratio for step in result.steps])
    events = np.array([step.event_count for step in result.steps])
    mismatches = np.array([step.hit_mismatches for step in result.steps])

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes[0, 0].plot(alphas, matrix_errors, marker="o")
    axes[0, 0].set_title("Matrix error")
    axes[0, 0].set_xlabel("alpha")
    axes[0, 0].set_ylabel("relative Frobenius")

    axes[0, 1].plot(alphas, query_ratios, marker="o", color="tab:green")
    axes[0, 1].set_title("Transport / full query ratio")
    axes[0, 1].set_xlabel("alpha")
    axes[0, 1].set_ylabel("ratio")

    axes[1, 0].plot(alphas, events, marker="o", color="tab:orange")
    axes[1, 0].set_title("First-hit reassignment events")
    axes[1, 0].set_xlabel("alpha")
    axes[1, 0].set_ylabel("ray count")

    axes[1, 1].plot(alphas, mismatches, marker="o", color="tab:red")
    axes[1, 1].set_title("Mismatches vs full recomputation")
    axes[1, 1].set_xlabel("alpha")
    axes[1, 1].set_ylabel("ray count")

    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
