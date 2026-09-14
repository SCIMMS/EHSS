from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import os
import time

import numpy as np


AU_HARD_SPHERE_RADIUS_A = 1.66
AU_AU_EDGE_A = 2.88


@dataclass(frozen=True)
class AtomCluster:
    name: str
    symbols: tuple[str, ...]
    positions: np.ndarray
    radii: np.ndarray
    shell_start: int

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=float)
        radii = np.asarray(self.radii, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (n_atoms, 3).")
        if radii.shape != (len(positions),):
            raise ValueError("radii must have one value per atom.")
        if len(self.symbols) != len(positions):
            raise ValueError("symbols must have one value per atom.")
        if np.any(radii <= 0.0):
            raise ValueError("radii must be positive.")
        if not 0 <= self.shell_start <= len(positions):
            raise ValueError("shell_start is outside the atom list.")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "radii", radii)

    @property
    def n_atoms(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_shell_atoms(self) -> int:
        return int(self.n_atoms - self.shell_start)


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    norms = np.linalg.norm(values, axis=-1)
    if np.any(norms <= 0.0):
        raise ValueError("cannot normalize zero-length vectors.")
    return values / norms[..., None]


def fibonacci_sphere(n_points: int) -> np.ndarray:
    if n_points <= 0:
        raise ValueError("n_points must be positive.")
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    directions = np.zeros((n_points, 3), dtype=float)
    for idx in range(n_points):
        z = 1.0 - 2.0 * (idx + 0.5) / n_points
        radius = np.sqrt(max(0.0, 1.0 - z * z))
        phi = 2.0 * np.pi * ((idx / golden_ratio) % 1.0)
        directions[idx] = [radius * np.cos(phi), radius * np.sin(phi), z]
    return directions


def ih_au13_core(edge_length: float = AU_AU_EDGE_A) -> np.ndarray:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array(
        [
            [0.0, -1.0, -phi],
            [0.0, -1.0, phi],
            [0.0, 1.0, -phi],
            [0.0, 1.0, phi],
            [-1.0, -phi, 0.0],
            [-1.0, phi, 0.0],
            [1.0, -phi, 0.0],
            [1.0, phi, 0.0],
            [-phi, 0.0, -1.0],
            [phi, 0.0, -1.0],
            [-phi, 0.0, 1.0],
            [phi, 0.0, 1.0],
        ],
        dtype=float,
    )
    current_edge = float(np.min(np.linalg.norm(vertices[:, None, :] - vertices[None, :, :], axis=-1)[np.triu_indices(12, 1)]))
    vertices *= edge_length / current_edge
    return np.vstack([np.zeros((1, 3), dtype=float), vertices])


def make_au_cluster_with_ih_core(
    shell_count: int = 12,
    shell_gap: float = 2.65,
    atom_radius: float = AU_HARD_SPHERE_RADIUS_A,
) -> AtomCluster:
    if shell_count <= 0:
        raise ValueError("shell_count must be positive.")

    core = ih_au13_core()
    core_radius = float(np.max(np.linalg.norm(core, axis=1)))
    if shell_count == 12:
        shell_directions = _normalize(core[1:])
    else:
        shell_directions = fibonacci_sphere(shell_count)
    shell_positions = shell_directions * (core_radius + shell_gap)
    positions = np.vstack([core, shell_positions])
    symbols = tuple("Au" for _ in range(len(positions)))
    radii = np.full(len(positions), atom_radius, dtype=float)
    if shell_count == 12:
        name = "Au25_Ih_Au13_core_plus_12_surface"
    elif shell_count == 42:
        name = "Au55_like_Ih_Au13_core_plus_42_surface"
    else:
        name = f"Au{13 + shell_count}_Ih_Au13_core_plus_{shell_count}_surface"
    return AtomCluster(name=name, symbols=symbols, positions=positions, radii=radii, shell_start=13)


def move_local_shell_atoms(
    cluster: AtomCluster,
    moved_shell_count: int,
    radial_shift: float = 0.35,
    tangential_shift: float = 0.18,
    target_direction: tuple[float, float, float] = (1.0, 0.25, 0.15),
) -> tuple[AtomCluster, np.ndarray]:
    if moved_shell_count < 0 or moved_shell_count > cluster.n_shell_atoms:
        raise ValueError("moved_shell_count must fit inside the shell atom count.")
    if moved_shell_count == 0:
        return cluster, np.array([], dtype=np.int64)

    target = _normalize(np.asarray(target_direction, dtype=float)[None, :])[0]
    shell_indices = np.arange(cluster.shell_start, cluster.n_atoms, dtype=np.int64)
    shell_dirs = _normalize(cluster.positions[shell_indices])
    order = np.argsort(-(shell_dirs @ target))
    moved = shell_indices[order[:moved_shell_count]]

    new_positions = np.array(cluster.positions, copy=True)
    for local_rank, atom_idx in enumerate(moved):
        radial = _normalize(new_positions[atom_idx : atom_idx + 1])[0]
        helper = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(radial, helper))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        tangent = np.cross(helper, radial)
        tangent /= np.linalg.norm(tangent)
        sign = -1.0 if local_rank % 2 else 1.0
        scale = 1.0 + 0.15 * (local_rank % 3)
        new_positions[atom_idx] += radial_shift * radial + sign * tangential_shift * scale * tangent

    moved_cluster = AtomCluster(
        name=f"{cluster.name}_move_{moved_shell_count}",
        symbols=cluster.symbols,
        positions=new_positions,
        radii=cluster.radii,
        shell_start=cluster.shell_start,
    )
    return moved_cluster, moved


def _basis_from_direction(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = _normalize(np.asarray(direction, dtype=float)[None, :])[0]
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(direction, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    tangent = np.cross(helper, direction)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(direction, tangent)
    bitangent /= np.linalg.norm(bitangent)
    return tangent, bitangent


def _impact_grid(extent: float, grid_size: int) -> tuple[np.ndarray, float]:
    if extent <= 0.0:
        raise ValueError("extent must be positive.")
    if grid_size <= 1:
        raise ValueError("grid_size must be greater than one.")
    step = 2.0 * extent / grid_size
    values = np.linspace(-extent + 0.5 * step, extent - 0.5 * step, grid_size)
    xx, yy = np.meshgrid(values, values, indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel()]), step * step


def _event_contributions(
    cluster: AtomCluster,
    direction: np.ndarray,
    grid_uv: np.ndarray,
    tangent: np.ndarray,
    bitangent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    projected = np.column_stack([cluster.positions @ tangent, cluster.positions @ bitangent])
    depth = cluster.positions @ direction
    delta = grid_uv[:, None, :] - projected[None, :, :]
    impact_b2 = np.sum(delta * delta, axis=2)
    radii2 = cluster.radii * cluster.radii
    inside = impact_b2 <= radii2[None, :]
    front_depth = depth[None, :] + np.sqrt(np.maximum(radii2[None, :] - impact_b2, 0.0))
    front_depth[~inside] = -np.inf
    hit = np.any(inside, axis=1)
    first_atom = np.argmax(front_depth, axis=1).astype(np.int64)
    first_atom[~hit] = -1

    pa = hit.astype(float)
    ehss = np.zeros(len(grid_uv), dtype=float)
    hit_rows = np.flatnonzero(hit)
    if len(hit_rows):
        atom_idx = first_atom[hit_rows]
        rel_b2 = impact_b2[hit_rows, atom_idx] / radii2[atom_idx]
        ehss[hit_rows] = 2.0 * np.maximum(0.0, 1.0 - rel_b2)
    return first_atom, pa, ehss


def _moved_atom_footprint(
    old_cluster: AtomCluster,
    new_cluster: AtomCluster,
    moved_indices: np.ndarray,
    grid_uv: np.ndarray,
    tangent: np.ndarray,
    bitangent: np.ndarray,
) -> np.ndarray:
    if len(moved_indices) == 0:
        return np.zeros(len(grid_uv), dtype=bool)
    mask = np.zeros(len(grid_uv), dtype=bool)
    for cluster in (old_cluster, new_cluster):
        projected = np.column_stack([cluster.positions[moved_indices] @ tangent, cluster.positions[moved_indices] @ bitangent])
        radii2 = cluster.radii[moved_indices] ** 2
        delta = grid_uv[:, None, :] - projected[None, :, :]
        mask |= np.any(np.sum(delta * delta, axis=2) <= radii2[None, :], axis=1)
    return mask


def evaluate_pa_ehss_transport(
    old_cluster: AtomCluster,
    new_cluster: AtomCluster,
    moved_indices: np.ndarray,
    n_orientations: int = 32,
    grid_size: int = 96,
) -> dict[str, float | int | str]:
    if old_cluster.n_atoms != new_cluster.n_atoms:
        raise ValueError("old and new clusters must have the same atom count.")
    if n_orientations <= 0:
        raise ValueError("n_orientations must be positive.")

    max_extent = float(
        max(np.max(np.linalg.norm(old_cluster.positions, axis=1)), np.max(np.linalg.norm(new_cluster.positions, axis=1)))
        + max(np.max(old_cluster.radii), np.max(new_cluster.radii))
        + 0.25
    )
    grid_uv, cell_area = _impact_grid(max_extent, grid_size)
    directions = fibonacci_sphere(n_orientations)

    totals = {
        "pa_old": 0.0,
        "pa_new_full": 0.0,
        "pa_new_transport": 0.0,
        "ehss_old": 0.0,
        "ehss_new_full": 0.0,
        "ehss_new_transport": 0.0,
    }
    total_samples = 0
    candidate_samples = 0
    pa_active_samples = 0
    pa_false_negative_samples = 0
    ehss_active_samples = 0
    ehss_false_negative_samples = 0
    cache_build_wall_time_s = 0.0
    full_new_wall_time_s = 0.0
    candidate_discovery_wall_time_s = 0.0
    candidate_update_wall_time_s = 0.0

    for direction in directions:
        tangent, bitangent = _basis_from_direction(direction)

        start = time.perf_counter()
        _, old_pa, old_ehss = _event_contributions(old_cluster, direction, grid_uv, tangent, bitangent)
        old_pa_sum = float(np.sum(old_pa) * cell_area)
        old_ehss_sum = float(np.sum(old_ehss) * cell_area)
        cache_build_wall_time_s += time.perf_counter() - start

        start = time.perf_counter()
        _, new_pa, new_ehss = _event_contributions(new_cluster, direction, grid_uv, tangent, bitangent)
        new_pa_sum = float(np.sum(new_pa) * cell_area)
        new_ehss_sum = float(np.sum(new_ehss) * cell_area)
        full_new_wall_time_s += time.perf_counter() - start

        start = time.perf_counter()
        candidate = _moved_atom_footprint(old_cluster, new_cluster, moved_indices, grid_uv, tangent, bitangent)
        candidate_discovery_wall_time_s += time.perf_counter() - start

        start = time.perf_counter()
        if np.any(candidate):
            _, candidate_new_pa, candidate_new_ehss = _event_contributions(
                new_cluster,
                direction,
                grid_uv[candidate],
                tangent,
                bitangent,
            )
            transport_pa_sum = old_pa_sum + float(np.sum(candidate_new_pa - old_pa[candidate]) * cell_area)
            transport_ehss_sum = old_ehss_sum + float(np.sum(candidate_new_ehss - old_ehss[candidate]) * cell_area)
        else:
            transport_pa_sum = old_pa_sum
            transport_ehss_sum = old_ehss_sum
        candidate_update_wall_time_s += time.perf_counter() - start

        pa_active = np.abs(new_pa - old_pa) > 1e-14
        ehss_active = np.abs(new_ehss - old_ehss) > 1e-14

        totals["pa_old"] += old_pa_sum
        totals["pa_new_full"] += new_pa_sum
        totals["pa_new_transport"] += transport_pa_sum
        totals["ehss_old"] += old_ehss_sum
        totals["ehss_new_full"] += new_ehss_sum
        totals["ehss_new_transport"] += transport_ehss_sum

        total_samples += len(grid_uv)
        candidate_samples += int(np.count_nonzero(candidate))
        pa_active_samples += int(np.count_nonzero(pa_active))
        pa_false_negative_samples += int(np.count_nonzero(pa_active & ~candidate))
        ehss_active_samples += int(np.count_nonzero(ehss_active))
        ehss_false_negative_samples += int(np.count_nonzero(ehss_active & ~candidate))

    for key in totals:
        totals[key] /= n_orientations

    pa_den = max(abs(totals["pa_new_full"]), 1e-15)
    ehss_den = max(abs(totals["ehss_new_full"]), 1e-15)
    active_union = max(pa_active_samples, ehss_active_samples)
    full_new_atom_tests = total_samples * old_cluster.n_atoms
    candidate_discovery_atom_tests = total_samples * len(moved_indices) * 2
    candidate_update_atom_tests = candidate_samples * old_cluster.n_atoms
    candidate_total_atom_tests = candidate_discovery_atom_tests + candidate_update_atom_tests
    candidate_route_wall_time_s = candidate_discovery_wall_time_s + candidate_update_wall_time_s
    result: dict[str, float | int | str] = {
        "cluster": old_cluster.name,
        "n_atoms": old_cluster.n_atoms,
        "moved_atoms": int(len(moved_indices)),
        "n_orientations": n_orientations,
        "grid_size": grid_size,
        **totals,
        "pa_delta_abs": totals["pa_new_full"] - totals["pa_old"],
        "ehss_delta_abs": totals["ehss_new_full"] - totals["ehss_old"],
        "pa_transport_rel_error": abs(totals["pa_new_transport"] - totals["pa_new_full"]) / pa_den,
        "ehss_transport_rel_error": abs(totals["ehss_new_transport"] - totals["ehss_new_full"]) / ehss_den,
        "candidate_fraction": candidate_samples / max(total_samples, 1),
        "pa_active_fraction": pa_active_samples / max(total_samples, 1),
        "ehss_active_fraction": ehss_active_samples / max(total_samples, 1),
        "candidate_per_active_union": candidate_samples / max(active_union, 1),
        "pa_false_negative_samples": pa_false_negative_samples,
        "ehss_false_negative_samples": ehss_false_negative_samples,
        "cache_build_wall_time_s": cache_build_wall_time_s,
        "full_new_wall_time_s": full_new_wall_time_s,
        "candidate_discovery_wall_time_s": candidate_discovery_wall_time_s,
        "candidate_update_wall_time_s": candidate_update_wall_time_s,
        "candidate_route_wall_time_s": candidate_route_wall_time_s,
        "candidate_route_wall_speedup_vs_full": full_new_wall_time_s / max(candidate_route_wall_time_s, 1e-15),
        "full_new_atom_tests": full_new_atom_tests,
        "candidate_discovery_atom_tests": candidate_discovery_atom_tests,
        "candidate_update_atom_tests": candidate_update_atom_tests,
        "candidate_total_atom_tests": candidate_total_atom_tests,
        "candidate_update_work_speedup_vs_full": full_new_atom_tests / max(candidate_update_atom_tests, 1),
        "candidate_total_work_speedup_vs_full": full_new_atom_tests / max(candidate_total_atom_tests, 1),
    }
    return result


def run_default_pa_ehss_demo(
    n_orientations: int = 32,
    grid_size: int = 96,
    cluster_mode: str = "both",
) -> list[dict[str, float | int | str]]:
    cluster_specs: list[tuple[int, tuple[int, ...]]]
    if cluster_mode == "small":
        cluster_specs = [(12, (1, 3, 6, 12))]
    elif cluster_mode == "large":
        cluster_specs = [(42, (1, 6, 12, 24))]
    elif cluster_mode == "both":
        cluster_specs = [(12, (1, 3, 6, 12)), (42, (1, 6, 12, 24))]
    else:
        raise ValueError("cluster_mode must be 'small', 'large', or 'both'.")

    rows: list[dict[str, float | int | str]] = []
    for shell_count, moved_counts in cluster_specs:
        old_cluster = make_au_cluster_with_ih_core(shell_count=shell_count)
        for moved_count in moved_counts:
            new_cluster, moved_indices = move_local_shell_atoms(old_cluster, moved_count)
            row = evaluate_pa_ehss_transport(
                old_cluster,
                new_cluster,
                moved_indices,
                n_orientations=n_orientations,
                grid_size=grid_size,
            )
            row["scenario"] = f"move_{moved_count}_surface_atoms"
            rows.append(row)
    return rows


def write_demo_csv(rows: list[dict[str, float | int | str]], output_csv: str | Path) -> None:
    if not rows:
        raise ValueError("rows must not be empty.")
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_demo_rows(
    rows: list[dict[str, float | int | str]],
    output_png: str | Path,
    output_pdf: str | Path | None = None,
) -> None:
    cache_dir = Path("artifacts/matplotlib").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("rows must not be empty.")
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    clusters = list(dict.fromkeys(str(row["cluster"]) for row in rows))
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
    for cluster in clusters:
        group = [row for row in rows if row["cluster"] == cluster]
        x = np.array([float(row["moved_atoms"]) for row in group])
        pa_delta = np.array([100.0 * float(row["pa_delta_abs"]) / max(float(row["pa_old"]), 1e-15) for row in group])
        ehss_delta = np.array([100.0 * float(row["ehss_delta_abs"]) / max(float(row["ehss_old"]), 1e-15) for row in group])
        candidate = np.array([100.0 * float(row["candidate_fraction"]) for row in group])
        active = np.array([100.0 * max(float(row["pa_active_fraction"]), float(row["ehss_active_fraction"])) for row in group])
        total_work_speedup = np.array([float(row["candidate_total_work_speedup_vs_full"]) for row in group])
        update_work_speedup = np.array([float(row["candidate_update_work_speedup_vs_full"]) for row in group])
        wall_speedup = np.array([float(row["candidate_route_wall_speedup_vs_full"]) for row in group])
        if cluster.startswith("Au25_"):
            label = "Au25 shell12"
        elif cluster.startswith("Au55_like_"):
            label = "Au55-like shell42"
        else:
            label = cluster
        axes[0].plot(x, pa_delta, marker="o", label=f"{label} projected area")
        axes[0].plot(
            x,
            ehss_delta,
            marker="s",
            linestyle="--",
            label=f"{label} momentum transfer",
        )
        axes[1].plot(x, candidate, marker="o", label=f"{label} candidate")
        axes[1].plot(x, active, marker="s", linestyle="--", label=f"{label} active")
        axes[2].plot(x, update_work_speedup, marker="o", linestyle=":", label=f"{label} update tests")
        axes[2].plot(x, total_work_speedup, marker="s", label=f"{label} total tests")
        axes[2].plot(x, wall_speedup, marker="^", linestyle="--", label=f"{label} wall")

    axes[0].axhline(0.0, color="0.25", linewidth=0.8)
    axes[0].set_xlabel("Moved surface atoms")
    axes[0].set_ylabel("Full recompute change (%)")
    axes[0].set_title("Projected collision response")
    axes[1].set_xlabel("Moved surface atoms")
    axes[1].set_ylabel("Impact samples (%)")
    axes[1].set_title("Moved-atom footprint route")
    axes[2].axhline(1.0, color="0.25", linewidth=0.8)
    axes[2].set_xlabel("Moved surface atoms")
    axes[2].set_ylabel("Speedup vs full new")
    axes[2].set_title("Direct update work")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=6)
    fig.suptitle("Ih Au13 core with locally moved Au surface atoms")
    fig.savefig(output_png, dpi=180)
    if output_pdf is not None:
        output_pdf = Path(output_pdf)
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf)
    plt.close(fig)
