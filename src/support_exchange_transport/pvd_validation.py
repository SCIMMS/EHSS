from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pvd_multistep import (
    PVDMultistepConfig,
    PVDMultistepResult,
    run_multistep_pvd,
)


@dataclass(frozen=True)
class PVDConvergenceResult:
    rows: tuple[dict[str, float | int | str | bool], ...]
    results: tuple[PVDMultistepResult, ...]


def _relative_scalar_error(actual: float, reference: float) -> float:
    return abs(actual - reference) / max(abs(reference), 1e-15)


def _profile_relative_error(
    result: PVDMultistepResult,
    reference: PVDMultistepResult,
) -> float:
    actual_height = result.full_profile.y + result.config.trench_depth
    reference_height = (
        reference.full_profile.y + reference.config.trench_depth
    )
    actual_on_reference = np.interp(
        reference.full_profile.x,
        result.full_profile.x,
        actual_height,
    )
    return float(
        np.linalg.norm(actual_on_reference - reference_height)
        / max(float(np.linalg.norm(reference_height)), 1e-15)
    )


def run_pvd_physical_convergence(
    configs: tuple[PVDMultistepConfig, ...],
    *,
    convergence_family: str = "coupled_resolution",
) -> PVDConvergenceResult:
    if len(configs) < 3:
        raise ValueError("physical convergence requires at least three resolutions.")
    first = configs[0]
    for config in configs[1:]:
        if (
            config.steps != first.steps
            or config.trench_width != first.trench_width
            or config.trench_depth != first.trench_depth
            or config.field_half_width != first.field_half_width
        ):
            raise ValueError(
                "convergence cases must share the evolution horizon and geometry."
            )

    results = tuple(run_multistep_pvd(config) for config in configs)
    reference = results[-1]
    reference_row = reference.rows[-1]
    rows: list[dict[str, float | int | str | bool]] = []
    for index, result in enumerate(results):
        row = result.rows[-1]
        next_finer = results[index + 1] if index + 1 < len(results) else result
        rows.append(
            {
                "convergence_family": convergence_family,
                "resolution_rank": index + 1,
                "resolution_label": result.config.scenario,
                "steps": result.config.steps,
                "source_samples": result.config.source_samples,
                "angle_samples": result.config.angle_samples,
                "n_rays": int(row["n_rays"]),
                "profile_nodes": result.config.profile_nodes,
                "candidate_fraction": float(row["candidate_fraction"]),
                "max_false_negative_rays": max(
                    int(step["false_negative_rays"])
                    for step in result.rows
                ),
                "max_target_mismatch_rays": max(
                    int(step["target_mismatch_rays"])
                    for step in result.rows
                ),
                "max_flux_rel_error_vs_same_discrete_reference": max(
                    float(step["flux_rel_error"]) for step in result.rows
                ),
                "max_profile_rel_error_vs_same_discrete_reference": max(
                    float(step["profile_rel_error"]) for step in result.rows
                ),
                "max_profile_height": float(row["max_profile_height"]),
                "integrated_profile_height": float(
                    row["integrated_profile_height"]
                ),
                "deposition_centroid_x": float(row["deposition_centroid_x"]),
                "profile_capture_fraction": float(
                    row["profile_capture_fraction"]
                ),
                "sidewall_capture_fraction": float(
                    row["sidewall_capture_fraction"]
                ),
                "bottom_to_mask_step_coverage": float(
                    row["bottom_to_mask_step_coverage"]
                ),
                "profile_l2_relative_difference_to_next_finer": (
                    _profile_relative_error(result, next_finer)
                ),
                "profile_l2_relative_error_vs_finest": (
                    _profile_relative_error(result, reference)
                ),
                "max_height_relative_error_vs_finest": _relative_scalar_error(
                    float(row["max_profile_height"]),
                    float(reference_row["max_profile_height"]),
                ),
                "integrated_height_relative_error_vs_finest": (
                    _relative_scalar_error(
                        float(row["integrated_profile_height"]),
                        float(reference_row["integrated_profile_height"]),
                    )
                ),
                "capture_fraction_relative_error_vs_finest": (
                    _relative_scalar_error(
                        float(row["profile_capture_fraction"]),
                        float(reference_row["profile_capture_fraction"]),
                    )
                ),
                "step_coverage_relative_error_vs_finest": (
                    _relative_scalar_error(
                        float(row["bottom_to_mask_step_coverage"]),
                        float(reference_row["bottom_to_mask_step_coverage"]),
                    )
                ),
            }
        )
    return PVDConvergenceResult(tuple(rows), results)


def run_asymmetric_oblique_pvd(
    *,
    steps: int = 24,
    timing_repeats: int = 1,
    timing_order: str = "route_full",
    independent_reference_interval: int = 4,
) -> PVDMultistepResult:
    return run_multistep_pvd(
        PVDMultistepConfig(
            scenario="offcenter_seed_oblique_incidence",
            steps=steps,
            profile_nodes=49,
            seed_center=-0.10,
            seed_half_width=0.14,
            growth_center=-0.08,
            growth_window_half_width=0.30,
            support_padding_x=0.10,
            source_samples=96,
            angle_samples=24,
            source_half_width=3.0,
            source_center=-0.35,
            max_angle_deg=48.0,
            mean_angle_deg=24.0,
            collimation_power=2.0,
            quadrature_rule="midpoint",
            timing_repeats=timing_repeats,
            timing_order=timing_order,
            independent_reference_interval=independent_reference_interval,
        )
    )
