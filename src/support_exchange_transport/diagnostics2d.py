from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .experiments import PrototypeConfig
from .geometry2d import expand_segment_mask, make_circle_cavity
from .metrics import rasterize_first_hits, relative_frobenius_error, row_conservation_error
from .rays2d import RayBundle2D, make_lambertian_rays
from .raytrace2d import FirstHitMap2D, trace_first_hits
from .transport2d import transport_first_hits


@dataclass(frozen=True)
class MismatchDiagnosticsRow:
    alpha: float
    n_segments: int
    rays_per_source: int
    amplitude: float
    width: float
    neighbor_radius: int
    total_mismatches: int
    matrix_error: float
    row_error_transport: float
    full_target_in_band: int
    full_target_outside_band: int
    source_in_support: int
    source_outside_support: int
    source_in_active_band: int
    source_outside_active_band: int
    target_in_support: int
    target_outside_support: int
    block_oo: int
    block_od: int
    block_do: int
    block_dd: int
    target_grazing: int
    source_grazing: int
    min_abs_target_cos: float
    mean_abs_target_cos: float
    top_old_to_full: str
    top_transport_to_full: str

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def _top_pairs(counter: Counter[tuple[int, int]], limit: int = 5) -> str:
    if not counter:
        return ""
    return ";".join(f"{old}->{new}:{count}" for (old, new), count in counter.most_common(limit))


def _region(mask: np.ndarray, index: int) -> str:
    return "D" if index >= 0 and bool(mask[index]) else "O"


def summarize_mismatches(
    *,
    alpha: float,
    config: PrototypeConfig,
    old_state: FirstHitMap2D,
    transport_state: FirstHitMap2D,
    full_state: FirstHitMap2D,
    rays: RayBundle2D,
    old_support_mask: np.ndarray,
    new_support_mask: np.ndarray,
    new_normals: np.ndarray,
    neighbor_radius: int,
    grazing_threshold: float,
) -> MismatchDiagnosticsRow:
    if grazing_threshold < 0.0:
        raise ValueError("grazing_threshold must be non-negative.")
    if transport_state.hits.shape != full_state.hits.shape:
        raise ValueError("transport and full first-hit maps must have matching shapes.")

    band_mask = expand_segment_mask(old_support_mask | new_support_mask, neighbor_radius)
    mismatch_sources, mismatch_rays = np.nonzero(transport_state.hits != full_state.hits)
    total = int(len(mismatch_sources))

    full_target_in_band = 0
    source_in_support = 0
    source_in_active_band = 0
    target_in_support = 0
    block_counts = Counter({"OO": 0, "OD": 0, "DO": 0, "DD": 0})
    target_grazing = 0
    source_grazing = 0
    abs_target_cosines: list[float] = []
    old_to_full: Counter[tuple[int, int]] = Counter()
    transport_to_full: Counter[tuple[int, int]] = Counter()

    for source, ray_idx in zip(mismatch_sources, mismatch_rays, strict=True):
        source = int(source)
        ray_idx = int(ray_idx)
        old_target = int(old_state.hits[source, ray_idx])
        transport_target = int(transport_state.hits[source, ray_idx])
        full_target = int(full_state.hits[source, ray_idx])

        old_to_full[(old_target, full_target)] += 1
        transport_to_full[(transport_target, full_target)] += 1

        if bool(band_mask[source]):
            source_in_active_band += 1
        if bool(new_support_mask[source]):
            source_in_support += 1
        if full_target >= 0 and bool(band_mask[full_target]):
            full_target_in_band += 1
        if full_target >= 0 and bool(new_support_mask[full_target]):
            target_in_support += 1

        source_region = _region(new_support_mask, source)
        target_region = _region(new_support_mask, full_target)
        block_counts[f"{source_region}{target_region}"] += 1

        direction = rays.directions[source, ray_idx]
        source_cos = abs(float(np.dot(direction, new_normals[source])))
        if source_cos < grazing_threshold:
            source_grazing += 1

        if full_target >= 0:
            target_cos = abs(float(np.dot(direction, new_normals[full_target])))
            abs_target_cosines.append(target_cos)
            if target_cos < grazing_threshold:
                target_grazing += 1

    f_full = rasterize_first_hits(full_state, config.n_segments)
    f_transport = rasterize_first_hits(transport_state, config.n_segments)
    if abs_target_cosines:
        min_abs_target_cos = float(np.min(abs_target_cosines))
        mean_abs_target_cos = float(np.mean(abs_target_cosines))
    else:
        min_abs_target_cos = float("nan")
        mean_abs_target_cos = float("nan")

    return MismatchDiagnosticsRow(
        alpha=alpha,
        n_segments=config.n_segments,
        rays_per_source=config.rays_per_source,
        amplitude=config.amplitude,
        width=config.width,
        neighbor_radius=neighbor_radius,
        total_mismatches=total,
        matrix_error=relative_frobenius_error(f_transport, f_full),
        row_error_transport=row_conservation_error(f_transport),
        full_target_in_band=full_target_in_band,
        full_target_outside_band=total - full_target_in_band,
        source_in_support=source_in_support,
        source_outside_support=total - source_in_support,
        source_in_active_band=source_in_active_band,
        source_outside_active_band=total - source_in_active_band,
        target_in_support=target_in_support,
        target_outside_support=total - target_in_support,
        block_oo=int(block_counts["OO"]),
        block_od=int(block_counts["OD"]),
        block_do=int(block_counts["DO"]),
        block_dd=int(block_counts["DD"]),
        target_grazing=target_grazing,
        source_grazing=source_grazing,
        min_abs_target_cos=min_abs_target_cos,
        mean_abs_target_cos=mean_abs_target_cos,
        top_old_to_full=_top_pairs(old_to_full),
        top_transport_to_full=_top_pairs(transport_to_full),
    )


def run_mismatch_diagnostics(
    config: PrototypeConfig,
    grazing_threshold: float = 0.05,
) -> tuple[MismatchDiagnosticsRow, ...]:
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
    transport_state = trace_first_hits(old_mesh, make_lambertian_rays(old_mesh, config.rays_per_source))
    rows: list[MismatchDiagnosticsRow] = []

    for alpha in np.linspace(1.0 / config.steps, 1.0, config.steps):
        new_mesh = make_circle_cavity(
            config.n_segments,
            config.radius,
            bumps,
            alpha=float(alpha),
            support_abs=config.support_abs,
        )
        rays = make_lambertian_rays(new_mesh, config.rays_per_source)
        full_state = trace_first_hits(new_mesh, rays)
        transport = transport_first_hits(
            transport_state,
            old_mesh,
            new_mesh,
            config.rays_per_source,
            neighbor_radius=config.neighbor_radius,
        )
        rows.append(
            summarize_mismatches(
                alpha=float(alpha),
                config=config,
                old_state=transport_state,
                transport_state=transport.state,
                full_state=full_state,
                rays=rays,
                old_support_mask=old_mesh.support_mask,
                new_support_mask=new_mesh.support_mask,
                new_normals=new_mesh.normals,
                neighbor_radius=config.neighbor_radius,
                grazing_threshold=grazing_threshold,
            )
        )
        old_mesh = new_mesh
        transport_state = transport.state

    return tuple(rows)

