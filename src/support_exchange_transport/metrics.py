from __future__ import annotations

from collections import defaultdict

import numpy as np

from .raytrace2d import FirstHitMap2D


def rasterize_first_hits(state: FirstHitMap2D, n_targets: int) -> np.ndarray:
    """Rasterize ray first-hit state into a row-stochastic exchange matrix."""

    matrix = np.zeros((state.hits.shape[0], n_targets), dtype=float)
    for source in range(state.hits.shape[0]):
        for ray_idx, target in enumerate(state.hits[source]):
            if target >= 0:
                matrix[source, target] += state.weights[source, ray_idx]
    return matrix


def relative_frobenius_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    denom = float(np.linalg.norm(reference))
    if denom == 0.0:
        return 0.0 if np.linalg.norm(candidate) == 0.0 else float("inf")
    return float(np.linalg.norm(candidate - reference) / denom)


def row_conservation_error(matrix: np.ndarray, expected: float = 1.0) -> float:
    return float(np.max(np.abs(np.sum(matrix, axis=1) - expected)))


def hit_mismatch_count(candidate: FirstHitMap2D, reference: FirstHitMap2D) -> int:
    if candidate.hits.shape != reference.hits.shape:
        raise ValueError("first-hit maps must have the same shape.")
    return int(np.count_nonzero(candidate.hits != reference.hits))


def signed_delta_accumulator(
    old_hits: np.ndarray,
    new_hits: np.ndarray,
    weights: np.ndarray,
) -> dict[tuple[int, int], float]:
    """Sparse signed delta entries keyed by (source, target)."""

    if old_hits.shape != new_hits.shape or old_hits.shape != weights.shape:
        raise ValueError("old hits, new hits, and weights must have matching shapes.")

    delta: dict[tuple[int, int], float] = defaultdict(float)
    changed_sources, changed_rays = np.nonzero(old_hits != new_hits)
    for source, ray_idx in zip(changed_sources, changed_rays, strict=True):
        old_target = int(old_hits[source, ray_idx])
        new_target = int(new_hits[source, ray_idx])
        weight = float(weights[source, ray_idx])
        if old_target >= 0:
            delta[(int(source), old_target)] -= weight
        if new_target >= 0:
            delta[(int(source), new_target)] += weight
    return dict(delta)


def support_exchange_matrix(
    old_hits: np.ndarray,
    new_hits: np.ndarray,
    weights: np.ndarray,
    support_labels: np.ndarray,
) -> np.ndarray:
    """Aggregate changed ray weights by old/new target support label."""

    max_label = int(np.max(support_labels)) if support_labels.size else 0
    exchange = np.zeros((max_label + 1, max_label + 1), dtype=float)
    for source in range(old_hits.shape[0]):
        for ray_idx in range(old_hits.shape[1]):
            old_target = int(old_hits[source, ray_idx])
            new_target = int(new_hits[source, ray_idx])
            if old_target == new_target or old_target < 0 or new_target < 0:
                continue
            old_label = int(support_labels[old_target])
            new_label = int(support_labels[new_target])
            exchange[old_label, new_label] += float(weights[source, ray_idx])
    return exchange
