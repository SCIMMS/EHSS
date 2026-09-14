from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import os
import time

import numpy as np

from .raytrace2d import intersect_ray_segment


SPEED_OF_SOUND_M_S = 343.0


@dataclass(frozen=True)
class Segment2D:
    start: np.ndarray
    end: np.ndarray
    normal: np.ndarray
    label: str

    def __post_init__(self) -> None:
        start = np.asarray(self.start, dtype=float)
        end = np.asarray(self.end, dtype=float)
        normal = np.asarray(self.normal, dtype=float)
        if start.shape != (2,) or end.shape != (2,) or normal.shape != (2,):
            raise ValueError("segment coordinates and normal must have shape (2,).")
        norm = np.linalg.norm(normal)
        if norm <= 0.0:
            raise ValueError("normal must be nonzero.")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "normal", normal / norm)


@dataclass(frozen=True)
class AcousticRoom2D:
    walls: tuple[Segment2D, ...]
    source: np.ndarray
    receiver: np.ndarray
    receiver_radius: float
    wall_reflection: float = 0.86
    air_attenuation: float = 0.015
    sound_speed: float = SPEED_OF_SOUND_M_S

    def __post_init__(self) -> None:
        source = np.asarray(self.source, dtype=float)
        receiver = np.asarray(self.receiver, dtype=float)
        if source.shape != (2,) or receiver.shape != (2,):
            raise ValueError("source and receiver must have shape (2,).")
        if self.receiver_radius <= 0.0:
            raise ValueError("receiver_radius must be positive.")
        if not 0.0 < self.wall_reflection <= 1.0:
            raise ValueError("wall_reflection must be in (0, 1].")
        if self.air_attenuation < 0.0:
            raise ValueError("air_attenuation must be nonnegative.")
        if self.sound_speed <= 0.0:
            raise ValueError("sound_speed must be positive.")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "receiver", receiver)


@dataclass(frozen=True)
class AcousticLeg:
    ray_idx: int
    bounce_idx: int
    start: np.ndarray
    direction: np.ndarray
    length: float
    cumulative_start: float
    energy_start: float
    receiver_tau: float


@dataclass(frozen=True)
class AcousticArrival:
    ray_idx: int
    bounce_idx: int
    path_length: float
    time_s: float
    energy: float


@dataclass(frozen=True)
class AcousticTrace:
    legs: tuple[AcousticLeg, ...]
    arrivals: tuple[AcousticArrival, ...]
    wall_tests: int
    absorber_tests: int
    receiver_tests: int


def make_rectangular_room(
    width: float = 10.0,
    height: float = 6.0,
    source: tuple[float, float] = (-3.8, -1.7),
    receiver: tuple[float, float] = (3.4, 1.4),
    receiver_radius: float = 0.28,
    wall_reflection: float = 0.86,
) -> AcousticRoom2D:
    if width <= 0.0 or height <= 0.0:
        raise ValueError("width and height must be positive.")
    x = 0.5 * width
    y = 0.5 * height
    walls = (
        Segment2D(np.array([-x, -y]), np.array([x, -y]), np.array([0.0, 1.0]), "bottom"),
        Segment2D(np.array([x, -y]), np.array([x, y]), np.array([-1.0, 0.0]), "right"),
        Segment2D(np.array([x, y]), np.array([-x, y]), np.array([0.0, -1.0]), "top"),
        Segment2D(np.array([-x, y]), np.array([-x, -y]), np.array([1.0, 0.0]), "left"),
    )
    return AcousticRoom2D(
        walls=walls,
        source=np.asarray(source, dtype=float),
        receiver=np.asarray(receiver, dtype=float),
        receiver_radius=receiver_radius,
        wall_reflection=wall_reflection,
    )


def make_vertical_absorber(
    center_x: float = 0.35,
    center_y: float = 0.0,
    height: float = 1.0,
    label: str = "absorber",
) -> Segment2D:
    if height <= 0.0:
        raise ValueError("height must be positive.")
    half = 0.5 * height
    return Segment2D(
        start=np.array([center_x, center_y - half], dtype=float),
        end=np.array([center_x, center_y + half], dtype=float),
        normal=np.array([1.0, 0.0], dtype=float),
        label=label,
    )


def uniform_ray_directions(n_rays: int) -> np.ndarray:
    if n_rays <= 0:
        raise ValueError("n_rays must be positive.")
    theta = 2.0 * np.pi * (np.arange(n_rays, dtype=float) + 0.5) / n_rays
    return np.column_stack((np.cos(theta), np.sin(theta)))


def _reflect(direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
    reflected = direction - 2.0 * float(np.dot(direction, normal)) * normal
    norm = np.linalg.norm(reflected)
    if norm <= 0.0:
        raise ValueError("reflection produced a zero direction.")
    return reflected / norm


def _intersect_ray_circle(origin: np.ndarray, direction: np.ndarray, center: np.ndarray, radius: float) -> float:
    rel = origin - center
    b = 2.0 * float(np.dot(direction, rel))
    c = float(np.dot(rel, rel) - radius * radius)
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return float("inf")
    root = np.sqrt(disc)
    taus = [(-b - root) * 0.5, (-b + root) * 0.5]
    valid = [tau for tau in taus if tau > 1e-8]
    if not valid:
        return float("inf")
    return float(min(valid))


def _arrival_energy(ray_weight: float, energy_start: float, path_length: float, air_attenuation: float) -> float:
    return ray_weight * energy_start * np.exp(-air_attenuation * path_length) / (1.0 + path_length)


def trace_acoustic_paths(
    room: AcousticRoom2D,
    directions: np.ndarray,
    max_bounces: int = 8,
    absorbers: tuple[Segment2D, ...] = (),
) -> AcousticTrace:
    if max_bounces < 0:
        raise ValueError("max_bounces must be nonnegative.")
    directions = np.asarray(directions, dtype=float)
    if directions.ndim != 2 or directions.shape[1] != 2:
        raise ValueError("directions must have shape (n_rays, 2).")

    ray_weight = 1.0 / len(directions)
    legs: list[AcousticLeg] = []
    arrivals: list[AcousticArrival] = []
    wall_tests = 0
    absorber_tests = 0
    receiver_tests = 0

    for ray_idx, initial_direction in enumerate(directions):
        direction = initial_direction / np.linalg.norm(initial_direction)
        origin = np.array(room.source, copy=True)
        cumulative = 0.0
        energy = 1.0
        for bounce_idx in range(max_bounces + 1):
            wall_taus = [
                intersect_ray_segment(origin, direction, wall.start, wall.end, min_t=1e-7)
                for wall in room.walls
            ]
            wall_tests += len(room.walls)
            hit_wall = int(np.argmin(wall_taus))
            wall_tau = float(wall_taus[hit_wall])
            if not np.isfinite(wall_tau):
                break

            absorber_tau = float("inf")
            for absorber in absorbers:
                tau = intersect_ray_segment(origin, direction, absorber.start, absorber.end, min_t=1e-7)
                absorber_tests += 1
                absorber_tau = min(absorber_tau, tau)

            leg_length = min(wall_tau, absorber_tau)
            receiver_tau = _intersect_ray_circle(origin, direction, room.receiver, room.receiver_radius)
            receiver_tests += 1
            if receiver_tau < leg_length:
                path_length = cumulative + receiver_tau
                arrivals.append(
                    AcousticArrival(
                        ray_idx=ray_idx,
                        bounce_idx=bounce_idx,
                        path_length=path_length,
                        time_s=path_length / room.sound_speed,
                        energy=_arrival_energy(ray_weight, energy, path_length, room.air_attenuation),
                    )
                )

            legs.append(
                AcousticLeg(
                    ray_idx=ray_idx,
                    bounce_idx=bounce_idx,
                    start=np.array(origin, copy=True),
                    direction=np.array(direction, copy=True),
                    length=leg_length,
                    cumulative_start=cumulative,
                    energy_start=energy,
                    receiver_tau=receiver_tau if receiver_tau < leg_length else float("inf"),
                )
            )

            if absorber_tau < wall_tau:
                break

            origin = origin + wall_tau * direction + 1e-7 * room.walls[hit_wall].normal
            direction = _reflect(direction, room.walls[hit_wall].normal)
            cumulative += wall_tau
            energy *= room.wall_reflection

    return AcousticTrace(
        legs=tuple(legs),
        arrivals=tuple(arrivals),
        wall_tests=wall_tests,
        absorber_tests=absorber_tests,
        receiver_tests=receiver_tests,
    )


def transport_added_absorber_from_cached_paths(
    room: AcousticRoom2D,
    old_trace: AcousticTrace,
    absorbers: tuple[Segment2D, ...],
) -> AcousticTrace:
    legs: list[AcousticLeg] = []
    arrivals: list[AcousticArrival] = []
    absorber_tests = 0
    stopped_rays: set[int] = set()
    ray_weight = 1.0 / _infer_ray_count(old_trace)

    for leg in old_trace.legs:
        if leg.ray_idx in stopped_rays:
            continue

        absorber_tau = float("inf")
        for absorber in absorbers:
            tau = intersect_ray_segment(leg.start, leg.direction, absorber.start, absorber.end, min_t=1e-7)
            absorber_tests += 1
            absorber_tau = min(absorber_tau, tau)
        transported_length = min(leg.length, absorber_tau)

        if leg.receiver_tau < transported_length:
            path_length = leg.cumulative_start + leg.receiver_tau
            arrivals.append(
                AcousticArrival(
                    ray_idx=leg.ray_idx,
                    bounce_idx=leg.bounce_idx,
                    path_length=path_length,
                    time_s=path_length / room.sound_speed,
                    energy=_arrival_energy(ray_weight, leg.energy_start, path_length, room.air_attenuation),
                )
            )

        legs.append(
            AcousticLeg(
                ray_idx=leg.ray_idx,
                bounce_idx=leg.bounce_idx,
                start=leg.start,
                direction=leg.direction,
                length=transported_length,
                cumulative_start=leg.cumulative_start,
                energy_start=leg.energy_start,
                receiver_tau=leg.receiver_tau if leg.receiver_tau < transported_length else float("inf"),
            )
        )

        if absorber_tau < leg.length:
            stopped_rays.add(leg.ray_idx)

    return AcousticTrace(
        legs=tuple(legs),
        arrivals=tuple(arrivals),
        wall_tests=0,
        absorber_tests=absorber_tests,
        receiver_tests=0,
    )


def _infer_ray_count(trace: AcousticTrace) -> int:
    if not trace.legs:
        return 1
    return max(leg.ray_idx for leg in trace.legs) + 1


def _absorber_route_counts(old_trace: AcousticTrace, absorbers: tuple[Segment2D, ...]) -> tuple[int, int, int]:
    tested_legs = 0
    truncated_legs = 0
    stopped_rays: set[int] = set()
    for leg in old_trace.legs:
        if leg.ray_idx in stopped_rays:
            continue
        tested_legs += 1
        absorber_tau = float("inf")
        for absorber in absorbers:
            absorber_tau = min(
                absorber_tau,
                intersect_ray_segment(leg.start, leg.direction, absorber.start, absorber.end, min_t=1e-7),
            )
        if absorber_tau < leg.length:
            truncated_legs += 1
            stopped_rays.add(leg.ray_idx)
    return tested_legs, truncated_legs, len(stopped_rays)


def impulse_histogram(trace: AcousticTrace, max_time_s: float, n_bins: int = 80) -> np.ndarray:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")
    hist = np.zeros(n_bins, dtype=float)
    if max_time_s <= 0.0:
        return hist
    for arrival in trace.arrivals:
        idx = int(np.floor(arrival.time_s / max_time_s * n_bins))
        if 0 <= idx < n_bins:
            hist[idx] += arrival.energy
    return hist


def _trace_with_timing(
    room: AcousticRoom2D,
    directions: np.ndarray,
    max_bounces: int,
    absorbers: tuple[Segment2D, ...],
    repeats: int,
) -> tuple[AcousticTrace, float]:
    if repeats <= 0:
        raise ValueError("repeats must be positive.")
    best_trace: AcousticTrace | None = None
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        trace = trace_acoustic_paths(room, directions, max_bounces=max_bounces, absorbers=absorbers)
        elapsed.append(time.perf_counter() - start)
        best_trace = trace
    assert best_trace is not None
    return best_trace, float(np.median(elapsed))


def _transport_with_timing(
    room: AcousticRoom2D,
    old_trace: AcousticTrace,
    absorbers: tuple[Segment2D, ...],
    repeats: int,
) -> tuple[AcousticTrace, float]:
    elapsed = []
    best_trace: AcousticTrace | None = None
    for _ in range(repeats):
        start = time.perf_counter()
        trace = transport_added_absorber_from_cached_paths(room, old_trace, absorbers)
        elapsed.append(time.perf_counter() - start)
        best_trace = trace
    assert best_trace is not None
    return best_trace, float(np.median(elapsed))


def evaluate_added_absorber_case(
    absorber_height: float,
    n_rays: int = 288,
    max_bounces: int = 8,
    timing_repeats: int = 7,
    n_time_bins: int = 80,
) -> dict[str, float | int | str]:
    room = make_rectangular_room()
    directions = uniform_ray_directions(n_rays)
    absorber = make_vertical_absorber(height=absorber_height)

    old_trace, cache_build_wall_time_s = _trace_with_timing(room, directions, max_bounces, (), max(1, timing_repeats // 2))
    full_new_trace, full_new_wall_time_s = _trace_with_timing(room, directions, max_bounces, (absorber,), timing_repeats)
    transported_trace, candidate_route_wall_time_s = _transport_with_timing(room, old_trace, (absorber,), timing_repeats)

    max_time_s = 1.1 * max(
        [arrival.time_s for arrival in old_trace.arrivals + full_new_trace.arrivals + transported_trace.arrivals] or [1.0]
    )
    full_hist = impulse_histogram(full_new_trace, max_time_s=max_time_s, n_bins=n_time_bins)
    transport_hist = impulse_histogram(transported_trace, max_time_s=max_time_s, n_bins=n_time_bins)
    hist_error = float(np.linalg.norm(transport_hist - full_hist) / max(np.linalg.norm(full_hist), 1e-15))

    old_energy = float(sum(arrival.energy for arrival in old_trace.arrivals))
    full_energy = float(sum(arrival.energy for arrival in full_new_trace.arrivals))
    transport_energy = float(sum(arrival.energy for arrival in transported_trace.arrivals))
    tested_legs, truncated_legs, truncated_rays = _absorber_route_counts(old_trace, (absorber,))
    full_tests = full_new_trace.wall_tests + full_new_trace.absorber_tests + full_new_trace.receiver_tests
    route_tests = transported_trace.absorber_tests

    return {
        "scenario": f"absorber_height_{absorber_height:.2f}",
        "model_scope": "high_frequency_geometric_acoustics_energy_rays",
        "n_rays": n_rays,
        "max_bounces": max_bounces,
        "absorber_height": absorber_height,
        "old_leg_count": len(old_trace.legs),
        "full_new_leg_count": len(full_new_trace.legs),
        "transport_leg_count": len(transported_trace.legs),
        "candidate_tested_legs": tested_legs,
        "active_truncated_legs": truncated_legs,
        "truncated_rays": truncated_rays,
        "candidate_leg_fraction": tested_legs / max(len(old_trace.legs), 1),
        "truncated_leg_fraction": truncated_legs / max(len(old_trace.legs), 1),
        "truncated_ray_fraction": truncated_rays / max(n_rays, 1),
        "old_arrivals": len(old_trace.arrivals),
        "full_new_arrivals": len(full_new_trace.arrivals),
        "transport_arrivals": len(transported_trace.arrivals),
        "old_total_energy": old_energy,
        "full_new_total_energy": full_energy,
        "transport_total_energy": transport_energy,
        "energy_rel_error": abs(transport_energy - full_energy) / max(abs(full_energy), 1e-15),
        "impulse_hist_rel_error": hist_error,
        "cache_build_wall_time_s": cache_build_wall_time_s,
        "full_new_wall_time_s": full_new_wall_time_s,
        "candidate_route_wall_time_s": candidate_route_wall_time_s,
        "candidate_route_wall_speedup_vs_full": full_new_wall_time_s / max(candidate_route_wall_time_s, 1e-15),
        "full_new_trace_tests": full_tests,
        "candidate_route_absorber_tests": route_tests,
        "candidate_route_test_speedup_vs_full": full_tests / max(route_tests, 1),
    }


def run_default_acoustic_demo(
    n_rays: int = 288,
    max_bounces: int = 8,
    timing_repeats: int = 7,
) -> list[dict[str, float | int | str]]:
    return [
        evaluate_added_absorber_case(height, n_rays=n_rays, max_bounces=max_bounces, timing_repeats=timing_repeats)
        for height in (0.5, 1.0, 2.0, 4.0)
    ]


def write_acoustic_demo_csv(rows: list[dict[str, float | int | str]], output_csv: str | Path) -> None:
    if not rows:
        raise ValueError("rows must not be empty.")
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_acoustic_demo_rows(
    rows: list[dict[str, float | int | str]],
    output_png: str | Path,
    output_pdf: str | Path | None = None,
) -> None:
    cache_dir = Path("artifacts/matplotlib").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib.pyplot as plt

    heights = np.array([float(row["absorber_height"]) for row in rows])
    energy_change = np.array(
        [
            100.0 * (float(row["full_new_total_energy"]) - float(row["old_total_energy"])) / max(float(row["old_total_energy"]), 1e-15)
            for row in rows
        ]
    )
    truncated = np.array([100.0 * float(row["truncated_leg_fraction"]) for row in rows])
    tested = np.array([100.0 * float(row["candidate_leg_fraction"]) for row in rows])
    truncated_rays = np.array([100.0 * float(row["truncated_ray_fraction"]) for row in rows])
    speedup_wall = np.array([float(row["candidate_route_wall_speedup_vs_full"]) for row in rows])
    speedup_tests = np.array([float(row["candidate_route_test_speedup_vs_full"]) for row in rows])

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
    axes[0].plot(heights, energy_change, marker="o")
    axes[0].axhline(0.0, color="0.25", linewidth=0.8)
    axes[0].set_xlabel("Added absorber height")
    axes[0].set_ylabel("Impulse energy change (%)")
    axes[0].set_title("High-frequency acoustic response")

    axes[1].plot(heights, tested, marker="o", label="cached legs tested")
    axes[1].plot(heights, truncated, marker="s", label="legs truncated")
    axes[1].plot(heights, truncated_rays, marker="^", linestyle="--", label="rays stopped")
    axes[1].set_xlabel("Added absorber height")
    axes[1].set_ylabel("Cached path events (%)")
    axes[1].set_title("Multi-bounce event boundary")

    axes[2].plot(heights, speedup_tests, marker="o", label="event tests")
    axes[2].plot(heights, speedup_wall, marker="s", linestyle="--", label="wall time")
    axes[2].axhline(1.0, color="0.25", linewidth=0.8)
    axes[2].set_xlabel("Added absorber height")
    axes[2].set_ylabel("Speedup vs full new trace")
    axes[2].set_title("Cached path update work")

    for axis in axes:
        axis.grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    fig.suptitle("Finite-bounce high-frequency acoustic event transport")

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    if output_pdf is not None:
        output_pdf = Path(output_pdf)
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf)
    plt.close(fig)
