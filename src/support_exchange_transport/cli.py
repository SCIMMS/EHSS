from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .experiments import PrototypeConfig, plot_experiment, run_bump_experiment


def _parse_centers(value: str) -> tuple[float, ...]:
    centers = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        centers.append(float(item))
    if not centers:
        raise argparse.ArgumentTypeError("at least one center must be provided.")
    return tuple(centers)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the 2D support-exchange prototype.")
    parser.add_argument("--segments", type=int, default=96, help="number of boundary segments")
    parser.add_argument("--rays", type=int, default=48, help="deterministic rays per source segment")
    parser.add_argument("--steps", type=int, default=6, help="deformation continuation steps")
    parser.add_argument("--radius", type=float, default=1.0, help="base circle radius")
    parser.add_argument("--amplitude", type=float, default=0.18, help="positive inward bump amplitude")
    parser.add_argument("--width", type=float, default=0.28, help="Gaussian angular bump width in radians")
    parser.add_argument(
        "--profile",
        choices=("gaussian", "compact-cosine"),
        default="gaussian",
        help="radial bump profile",
    )
    parser.add_argument(
        "--centers",
        type=_parse_centers,
        default=(0.0,),
        help="comma-separated bump centers in radians, e.g. '0,3.14159'",
    )
    parser.add_argument("--neighbor-radius", type=int, default=1, help="support mask expansion radius")
    parser.add_argument(
        "--support-abs",
        action="store_true",
        help="mark support by absolute deformation amplitude, useful for signed dent tests",
    )
    parser.add_argument("--plot", type=Path, default=None, help="optional output PNG path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PrototypeConfig(
        n_segments=args.segments,
        rays_per_source=args.rays,
        steps=args.steps,
        radius=args.radius,
        amplitude=args.amplitude,
        width=args.width,
        centers=args.centers,
        neighbor_radius=args.neighbor_radius,
        support_abs=args.support_abs,
        profile_kind=args.profile,
    )
    result = run_bump_experiment(config)

    print("2D support-exchange first-hit transport")
    print(f"segments={config.n_segments} rays/source={config.rays_per_source} steps={config.steps}")
    print(f"centers={tuple(round(c, 6) for c in config.centers)} amplitude={config.amplitude} width={config.width}")
    print()
    print(
        "alpha   matrix_err      row_err_T      mismatches  events   "
        "q_full/q_T  q_full/q_tile  t_full/t_T  active/chg"
    )
    for step in result.steps:
        print(
            f"{step.alpha:5.2f}  "
            f"{step.matrix_error:12.4e}  "
            f"{step.row_error_transport:12.4e}  "
            f"{step.hit_mismatches:10d}  "
            f"{step.event_count:6d}  "
            f"{step.query_speedup:10.3f}  "
            f"{step.tile_query_speedup:13.3f}  "
            f"{step.wall_time_speedup:10.3f}  "
            f"{step.active_source_count:4d}/{step.changed_segment_count:<4d}"
        )

    final = result.steps[-1]
    print()
    print(f"final transport/full query fraction: {final.query_ratio:.3f}")
    print(f"final full/transport query speedup: {final.query_speedup:.3f}")
    print(f"final full/event-tile query speedup: {final.tile_query_speedup:.3f}")
    print(f"final full/transport wall-clock speedup: {final.wall_time_speedup:.3f}")
    print(f"final event-tile mismatches: {final.tile_hit_mismatches}")
    print(f"final support-exchange matrix shape: {final.support_exchange.shape}")
    print(f"final support-exchange total weight: {np.sum(final.support_exchange):.6f}")

    if args.plot is not None:
        path = plot_experiment(result, args.plot)
        print(f"plot written: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
