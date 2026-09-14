from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def _format_error(value: float) -> str:
    return "0" if value == 0.0 else f"{value:.1e}"


def make_summary_figure(
    scaling_csv: Path,
    gaussian_diag_csv: Path,
    compact_diag_csv: Path,
    output_png: Path,
    output_pdf: Path | None = None,
) -> Path:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    mpl_config = output_png.parent / ".matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

    import matplotlib.pyplot as plt

    scaling = _read_csv(scaling_csv)
    gaussian = _read_csv(gaussian_diag_csv)
    compact = _read_csv(compact_diag_csv)

    full_queries = np.array([_float(row, "full_queries") for row in scaling])
    full_times = np.array([_float(row, "full_time_s") for row in scaling])
    transport_times = np.array([_float(row, "support_time_s") for row in scaling])
    labels = [f"{_int(row, 'n_segments')}x{_int(row, 'rays_per_source')}" for row in scaling]

    full_slope = np.polyfit(np.log10(full_queries), np.log10(full_times), 1)[0]
    transport_slope = np.polyfit(np.log10(full_queries), np.log10(transport_times), 1)[0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    fig.suptitle("2D Support-Exchange Transport: Compact Support Scaling", fontsize=14)

    ax = axes[0, 0]
    ax.loglog(full_queries, full_times, "o-", label=f"Full recomputation (slope {full_slope:.2f})")
    ax.loglog(
        full_queries,
        transport_times,
        "s-",
        label=f"Support-local transport (slope {transport_slope:.2f})",
    )
    for x, y, label in zip(full_queries, full_times, labels, strict=True):
        ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_title("a  Log-log wall-clock scaling", fontweight="bold")
    ax.set_xlabel("Full query count ~ Nseg^2 x Nrays")
    ax.set_ylabel("Wall time per final step (s)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ray_counts = sorted({_int(row, "rays_per_source") for row in scaling})
    offsets = np.linspace(-6.0, 6.0, len(ray_counts)) if len(ray_counts) > 1 else np.array([0.0])
    for offset, rays in zip(offsets, ray_counts, strict=True):
        subset = [row for row in scaling if _int(row, "rays_per_source") == rays]
        subset = sorted(subset, key=lambda row: _int(row, "n_segments"))
        x_values = np.array([_int(row, "n_segments") for row in subset], dtype=float)
        ax.plot(
            x_values + offset,
            [_float(row, "query_speedup") for row in subset],
            "o-",
            label=f"{rays} rays/source",
        )
    ax.set_title("b  Query speedup vs Nseg", fontweight="bold")
    ax.set_xlabel("Nseg")
    ax.set_ylabel("Full queries / transport queries")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.axis("off")
    total_mismatches = sum(_int(row, "mismatches") for row in scaling)
    max_matrix_error = max(_float(row, "matrix_error") for row in scaling)
    summary = (
        f"{len(scaling)} workloads\n"
        f"{total_mismatches} target mismatches\n"
        f"maximum matrix error: {_format_error(max_matrix_error)}"
    )
    ax.text(
        0.5,
        0.5,
        summary,
        ha="center",
        va="center",
        fontsize=14,
        linespacing=1.65,
        bbox={
            "boxstyle": "round,pad=0.8",
            "facecolor": "#EDF3F7",
            "edgecolor": "#5B7180",
        },
    )
    ax.set_title("c  Discrete recovery summary", pad=12, fontweight="bold")

    ax = axes[1, 1]
    alpha_g = np.array([_float(row, "alpha") for row in gaussian])
    alpha_c = np.array([_float(row, "alpha") for row in compact])
    mismatch_g = np.array([_int(row, "total_mismatches") for row in gaussian])
    mismatch_c = np.array([_int(row, "total_mismatches") for row in compact])
    ax.plot(alpha_g, mismatch_g, "o-", label="Gaussian threshold")
    ax.plot(alpha_c, mismatch_c, "s-", label="Compact support")
    ax.set_title(
        "d  Thresholded Gaussian failure vs compact exactness",
        fontweight="bold",
    )
    ax.set_xlabel("Deformation alpha")
    ax.set_ylabel("First-hit mismatches")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")

    final_gaussian_error = _float(gaussian[-1], "matrix_error")
    final_compact_error = _float(compact[-1], "matrix_error")
    ax.text(
        0.02,
        0.72,
        f"Final matrix error:\nGaussian {final_gaussian_error:.2e}\nCompact {final_compact_error:.1e}",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )

    fig.savefig(output_png, dpi=180)
    if output_pdf is not None:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf)
    plt.close(fig)
    return output_png


def make_support_count_footprint_figure(
    support_count_csv: Path,
    output_png: Path,
    output_pdf: Path | None = None,
) -> Path:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    mpl_config = output_png.parent / ".matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

    import matplotlib.pyplot as plt

    rows = _read_csv(support_count_csv)
    layouts = list(dict.fromkeys(row["layout"] for row in rows))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    fig.suptitle("Shadow Footprint Prefilter: Support Count and Overlap", fontsize=14)

    ax = axes[0, 0]
    for layout in layouts:
        subset = sorted(
            [row for row in rows if row["layout"] == layout],
            key=lambda row: _int(row, "support_count"),
        )
        counts = np.array([_int(row, "support_count") for row in subset], dtype=float)
        total_rays = np.array(
            [_int(row, "n_segments") * _int(row, "rays_per_source") for row in subset],
            dtype=float,
        )
        footprint_fraction = np.array([_int(row, "footprint_rays") for row in subset]) / total_rays
        changed_fraction = np.array([_int(row, "tile_changed_rays") for row in subset]) / total_rays
        ax.plot(counts, footprint_fraction, "o-", label=f"{layout}: footprint candidates")
        ax.plot(counts, changed_fraction, "s--", label=f"{layout}: changed rays")
    ax.set_title("a  Footprint union vs actual events", fontweight="bold")
    ax.set_xlabel("Number of supports")
    ax.set_ylabel("Fraction of source-ray pairs")
    ax.set_xticks([_int(row, "support_count") for row in rows if row["layout"] == layouts[0]])
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for layout in layouts:
        subset = sorted(
            [row for row in rows if row["layout"] == layout],
            key=lambda row: _int(row, "support_count"),
        )
        counts = [_int(row, "support_count") for row in subset]
        overlap_fraction = [
            _int(row, "footprint_overlap_rays") / max(_int(row, "footprint_rays"), 1)
            for row in subset
        ]
        ax.plot(counts, overlap_fraction, "o-", label=layout)
    ax.set_title("b  Distinct-support footprint overlap", fontweight="bold")
    ax.set_xlabel("Number of supports")
    ax.set_ylabel("Overlap rays / footprint rays")
    ax.set_xticks([_int(row, "support_count") for row in rows if row["layout"] == layouts[0]])
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for layout in layouts:
        subset = sorted(
            [row for row in rows if row["layout"] == layout],
            key=lambda row: _int(row, "support_count"),
        )
        counts = [_int(row, "support_count") for row in subset]
        ax.plot(counts, [_float(row, "tile_query_speedup") for row in subset], "o-", label=f"{layout}: query")
        ax.plot(
            counts,
            [_float(row, "tile_wall_time_speedup") for row in subset],
            "s--",
            label=f"{layout}: wall",
        )
    ax.set_title(
        "c  Full recomputation / footprint transport",
        fontweight="bold",
    )
    ax.set_xlabel("Number of supports")
    ax.set_ylabel("Speedup")
    ax.set_xticks([_int(row, "support_count") for row in rows if row["layout"] == layouts[0]])
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for layout in layouts:
        subset = sorted(
            [row for row in rows if row["layout"] == layout],
            key=lambda row: _int(row, "support_count"),
        )
        counts = [_int(row, "support_count") for row in subset]
        candidate_per_event = [
            _int(row, "footprint_rays") / max(_int(row, "tile_changed_rays"), 1)
            for row in subset
        ]
        ax.plot(counts, candidate_per_event, "o-", label=layout)
    ax.axhline(1.0, color="0.2", linewidth=1.0, linestyle=":", label="ideal event lower bound")
    ax.set_title(
        "d  Candidate rays per actual changed ray",
        fontweight="bold",
    )
    ax.set_xlabel("Number of supports")
    ax.set_ylabel("Footprint candidates / changed rays")
    ax.set_xticks([_int(row, "support_count") for row in rows if row["layout"] == layouts[0]])
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    all_exact = all(_int(row, "tile_mismatches") == 0 and _float(row, "tile_matrix_error") == 0.0 for row in rows)
    axes[1, 1].text(
        0.03,
        0.93,
        f"Exactness: {'all rows exact' if all_exact else 'see CSV'}",
        transform=axes[1, 1].transAxes,
        fontsize=9,
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )

    fig.savefig(output_png, dpi=180)
    if output_pdf is not None:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf)
    plt.close(fig)
    return output_png


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build summary figures for 2D transport benchmarks.")
    parser.add_argument(
        "--figure",
        choices=("summary", "support-count", "all"),
        default="summary",
        help="which figure set to build",
    )
    parser.add_argument(
        "--scaling-csv",
        type=Path,
        default=Path("data_raw/2d_core/large_scaling_compact_support_384_768_r96_192.csv"),
    )
    parser.add_argument(
        "--gaussian-diagnostics-csv",
        type=Path,
        default=Path("data_raw/failure_diagnostics/mismatch_diagnostics_gaussian_384x96_width028_radius0_current.csv"),
    )
    parser.add_argument(
        "--compact-diagnostics-csv",
        type=Path,
        default=Path("data_raw/failure_diagnostics/mismatch_diagnostics_compact_384x96_width028_radius0_current.csv"),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("figures/supplementary/si_2d_large_scaling_summary.png"),
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("figures/supplementary/si_2d_large_scaling_summary.pdf"),
    )
    parser.add_argument(
        "--support-count-csv",
        type=Path,
        default=Path("data_raw/2d_core/support_count_footprint_far_vs_clustered_768x192.csv"),
    )
    parser.add_argument(
        "--support-count-output-png",
        type=Path,
        default=Path("figures/main/fig2_2d_support_count_footprint_overlap.png"),
    )
    parser.add_argument(
        "--support-count-output-pdf",
        type=Path,
        default=Path("figures/main/fig2_2d_support_count_footprint_overlap.pdf"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.figure in {"summary", "all"}:
        path = make_summary_figure(
            args.scaling_csv,
            args.gaussian_diagnostics_csv,
            args.compact_diagnostics_csv,
            args.output_png,
            args.output_pdf,
        )
        print(f"figure written: {path}")
        if args.output_pdf is not None:
            print(f"pdf written: {args.output_pdf}")
    if args.figure in {"support-count", "all"}:
        path = make_support_count_footprint_figure(
            args.support_count_csv,
            args.support_count_output_png,
            args.support_count_output_pdf,
        )
        print(f"figure written: {path}")
        if args.support_count_output_pdf is not None:
            print(f"pdf written: {args.support_count_output_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
