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


def make_m1_figure(
    input_csv: Path,
    output_png: Path,
    output_pdf: Path | None = None,
) -> Path:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    mpl_config = output_png.parent / ".matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

    import matplotlib.pyplot as plt

    rows = _read_csv(input_csv)
    scaling = [row for row in rows if row["suite"] == "single-support-mesh-ray-scaling"]
    support_size = [row for row in rows if row["suite"] == "single-support-size-scaling"]
    if not scaling:
        raise ValueError("input CSV does not contain mesh-ray scaling rows.")
    if not support_size:
        raise ValueError("input CSV does not contain support-size scaling rows.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    fig.suptitle("3D Single-Support Cone-Footprint BVH Transport", fontsize=14)

    ax = axes[0, 0]
    ray_counts = sorted({_int(row, "rays_per_source") for row in scaling})
    for rays in ray_counts:
        subset = sorted(
            [row for row in scaling if _int(row, "rays_per_source") == rays],
            key=lambda row: _int(row, "n_faces"),
        )
        faces = [_int(row, "n_faces") for row in subset]
        ax.plot(faces, [_float(row, "query_speedup") for row in subset], "o-", label=f"{rays} rays: query")
        ax.plot(faces, [_float(row, "wall_time_speedup") for row in subset], "s--", label=f"{rays} rays: wall")
    ax.set_title("a  Mesh/ray scaling speedup", fontweight="bold")
    ax.set_xlabel("Triangle faces")
    ax.set_ylabel("Full BVH / transport")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncols=2)

    ax = axes[0, 1]
    for rays in ray_counts:
        subset = sorted(
            [row for row in scaling if _int(row, "rays_per_source") == rays],
            key=lambda row: _int(row, "n_faces"),
        )
        faces = [_int(row, "n_faces") for row in subset]
        candidate_ratio = [
            _int(row, "footprint_rays") / max(_int(row, "changed_rays"), 1)
            for row in subset
        ]
        fallback_fraction = [
            _int(row, "fallback_full_rays") / max(_int(row, "footprint_rays"), 1)
            for row in subset
        ]
        ax.plot(faces, candidate_ratio, "o-", label=f"{rays} rays: fp/change")
        ax.plot(faces, fallback_fraction, "s--", label=f"{rays} rays: fallback/fp")
    ax.axhline(1.0, color="0.2", linestyle=":", linewidth=1.0, label="ideal event lower bound")
    ax.set_title("b  Footprint tightness and fallback", fontweight="bold")
    ax.set_xlabel("Triangle faces")
    ax.set_ylabel("Ratio")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncols=2)

    ax = axes[1, 0]
    support_size = sorted(support_size, key=lambda row: _float(row, "width"))
    widths = [_float(row, "width") for row in support_size]
    ax.plot(widths, [_float(row, "query_speedup") for row in support_size], "o-", label="query speedup")
    ax.plot(widths, [_float(row, "wall_time_speedup") for row in support_size], "s--", label="wall speedup")
    ax2 = ax.twinx()
    ax2.plot(
        widths,
        [_int(row, "footprint_rays") / max(_int(row, "changed_rays"), 1) for row in support_size],
        "^-.",
        color="tab:green",
        label="fp/change",
    )
    ax.set_title("c  Support-size scaling", fontweight="bold")
    ax.set_xlabel("Compact cap angular width (rad)")
    ax.set_ylabel("Speedup")
    ax2.set_ylabel("Candidate / changed rays")
    ax.grid(True, alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8)

    ax = axes[1, 1]
    ax.axis("off")
    table_rows = [
        [
            str(_int(row, "n_faces")),
            str(_int(row, "rays_per_source")),
            f"{_float(row, 'width'):.2f}",
            str(_int(row, "mismatches")),
            _format_error(_float(row, "matrix_error")),
            _format_error(_float(row, "row_error_transport")),
        ]
        for row in rows
    ]
    table = ax.table(
        cellText=table_rows,
        colLabels=["Faces", "Rays", "Width", "Mismatch", "Matrix err", "Row err"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.25)
    all_exact = all(_int(row, "mismatches") == 0 and _float(row, "matrix_error") == 0.0 for row in rows)
    ax.set_title(
        f"d  Exactness table ({'all rows exact' if all_exact else 'see CSV'})",
        pad=12,
        fontweight="bold",
    )

    fig.savefig(output_png, dpi=180)
    if output_pdf is not None:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf)
    plt.close(fig)
    return output_png


def make_m2_figure(
    input_csv: Path,
    output_png: Path,
    output_pdf: Path | None = None,
) -> Path:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    mpl_config = output_png.parent / ".matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

    import matplotlib.pyplot as plt

    rows = _read_csv(input_csv)
    if not rows:
        raise ValueError("input CSV is empty.")
    layouts = list(dict.fromkeys(row["layout"] for row in rows))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    fig.suptitle("3D Multi-Support Cone-Footprint BVH Transport", fontsize=14)

    ax = axes[0, 0]
    for layout in layouts:
        subset = sorted(
            [row for row in rows if row["layout"] == layout],
            key=lambda row: _int(row, "support_count"),
        )
        counts = [_int(row, "support_count") for row in subset]
        total_rays = np.array([_int(row, "n_faces") * _int(row, "rays_per_source") for row in subset], dtype=float)
        footprint_fraction = np.array([_int(row, "footprint_rays") for row in subset], dtype=float) / total_rays
        changed_fraction = np.array([_int(row, "changed_rays") for row in subset], dtype=float) / total_rays
        ax.plot(counts, footprint_fraction, "o-", label=f"{layout}: footprint")
        ax.plot(counts, changed_fraction, "s--", label=f"{layout}: changed")
    ax.set_title("a  Footprint union vs actual events", fontweight="bold")
    ax.set_xlabel("Number of supports")
    ax.set_ylabel("Fraction of source-ray pairs")
    ax.set_xticks(sorted({_int(row, "support_count") for row in rows}))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for layout in layouts:
        subset = sorted(
            [row for row in rows if row["layout"] == layout],
            key=lambda row: _int(row, "support_count"),
        )
        counts = [_int(row, "support_count") for row in subset]
        ax.plot(counts, [_float(row, "overlap_fraction") for row in subset], "o-", label=f"{layout}: overlap")
        ax.plot(counts, [_float(row, "fallback_fraction") for row in subset], "s--", label=f"{layout}: fallback")
    ax.set_title("b  Footprint overlap and fallback", fontweight="bold")
    ax.set_xlabel("Number of supports")
    ax.set_ylabel("Fraction of footprint rays")
    ax.set_xticks(sorted({_int(row, "support_count") for row in rows}))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for layout in layouts:
        subset = sorted(
            [row for row in rows if row["layout"] == layout],
            key=lambda row: _int(row, "support_count"),
        )
        counts = [_int(row, "support_count") for row in subset]
        ax.plot(counts, [_float(row, "query_speedup") for row in subset], "o-", label=f"{layout}: query")
        ax.plot(counts, [_float(row, "wall_time_speedup") for row in subset], "s--", label=f"{layout}: wall")
    ax.set_title("c  Full BVH / transport speedup", fontweight="bold")
    ax.set_xlabel("Number of supports")
    ax.set_ylabel("Speedup")
    ax.set_xticks(sorted({_int(row, "support_count") for row in rows}))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.axis("off")
    table_rows = [
        [
            row["layout"],
            str(_int(row, "support_count")),
            str(_int(row, "mismatches")),
            _format_error(_float(row, "matrix_error")),
            _format_error(_float(row, "row_error_transport")),
            f"{_float(row, 'candidate_per_changed'):.2f}",
        ]
        for row in rows
    ]
    table = ax.table(
        cellText=table_rows,
        colLabels=["Layout", "Nsup", "Mismatch", "Matrix err", "Row err", "FP/change"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)
    all_exact = all(_int(row, "mismatches") == 0 and _float(row, "matrix_error") == 0.0 for row in rows)
    ax.set_title(
        f"d  Exactness table ({'all rows exact' if all_exact else 'see CSV'})",
        pad=12,
        fontweight="bold",
    )

    fig.savefig(output_png, dpi=180)
    if output_pdf is not None:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf)
    plt.close(fig)
    return output_png


def make_m3_figure(
    input_csv: Path,
    output_png: Path,
    output_pdf: Path | None = None,
    response_csv: Path | None = None,
    response_summary_csv: Path | None = None,
) -> Path:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    mpl_config = output_png.parent / ".matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

    import matplotlib.pyplot as plt

    rows = _read_csv(input_csv)
    if not rows:
        raise ValueError("input CSV is empty.")
    layouts = list(dict.fromkeys(row["layout"] for row in rows))
    floor = 1e-18

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), constrained_layout=True)
    fig.suptitle(
        "Discrete 3D radiosity validation using transported view factors",
        fontsize=14,
    )

    ax = axes[0]
    for layout in layouts:
        subset = sorted(
            [row for row in rows if row["layout"] == layout],
            key=lambda row: _int(row, "support_count"),
        )
        counts = [_int(row, "support_count") for row in subset]
        ax.plot(counts, [_float(row, "query_speedup") for row in subset], "o-", label=f"{layout}: query")
        ax.plot(counts, [_float(row, "wall_time_speedup") for row in subset], "s--", label=f"{layout}: wall")
    ax.set_title("a  Transport speedup", fontweight="bold")
    ax.set_xlabel("Number of supports")
    ax.set_ylabel("Full BVH / transport")
    ax.set_xticks(sorted({_int(row, "support_count") for row in rows}))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    for layout in layouts:
        subset = sorted(
            [row for row in rows if row["layout"] == layout],
            key=lambda row: _int(row, "support_count"),
        )
        counts = [_int(row, "support_count") for row in subset]
        radiosity_error = [max(_float(row, "radiosity_rel_error"), floor) for row in subset]
        flux_error = [max(_float(row, "net_flux_rel_error"), floor) for row in subset]
        power_error = [max(_float(row, "total_power_rel_error"), floor) for row in subset]
        ax.semilogy(counts, radiosity_error, "o-", label=f"{layout}: J")
        ax.semilogy(counts, flux_error, "s--", label=f"{layout}: q")
        ax.semilogy(counts, power_error, "^-.", label=f"{layout}: Q")
    ax.set_title("b  Physical output relative error", fontweight="bold")
    ax.set_xlabel("Number of supports")
    ax.set_ylabel("Relative error, zeros plotted at 1e-18")
    ax.set_xticks(sorted({_int(row, "support_count") for row in rows}))
    ax.set_ylim(5e-19, 1e-12)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncols=2)

    ax = axes[2]
    if response_csv is not None:
        response_rows = _read_csv(response_csv)
        if not response_rows:
            raise ValueError("response CSV is empty.")
        longitude = np.asarray(
            [_float(row, "longitude_deg") for row in response_rows]
        )
        latitude = np.asarray(
            [_float(row, "latitude_deg") for row in response_rows]
        )
        delta_q = np.asarray(
            [_float(row, "delta_net_flux") for row in response_rows]
        )
        changed_source_fraction = np.asarray(
            [_float(row, "changed_source_fraction") for row in response_rows]
        )
        limit = float(np.max(np.abs(delta_q)))
        scatter = ax.scatter(
            longitude,
            latitude,
            c=delta_q,
            s=19,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            linewidths=0,
            rasterized=True,
        )
        changed_sources = changed_source_fraction > 0.0
        ax.scatter(
            longitude[changed_sources],
            latitude[changed_sources],
            s=24,
            facecolors="none",
            edgecolors="black",
            linewidths=0.45,
            label="source face with changed first hit",
        )
        summary_text = ""
        if response_summary_csv is not None:
            summary_rows = _read_csv(response_summary_csv)
            if len(summary_rows) != 1:
                raise ValueError("response summary CSV must contain one row.")
            changed_percent = 100.0 * _float(
                summary_rows[0],
                "changed_ray_fraction",
            )
            response_percent = 100.0 * _float(
                summary_rows[0],
                "response_face_fraction_for_95pct_area_weighted_abs_delta_q",
            )
            summary_text = (
                f"changed first hits: {changed_percent:.2f}%\n"
                f"95% of area-weighted |Δq|: {response_percent:.1f}% of faces"
            )
        if summary_text:
            ax.text(
                0.02,
                0.02,
                summary_text,
                transform=ax.transAxes,
                fontsize=7.5,
                va="bottom",
                ha="left",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88},
            )
        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.048, pad=0.03)
        colorbar.set_label("Signed Δq (normalized units)")
        ax.set_title("c  Coupled heat-flux redistribution", fontweight="bold")
        ax.set_xlabel("Surface longitude (deg)")
        ax.set_ylabel("Surface latitude (deg)")
        ax.set_xlim(-180.0, 180.0)
        ax.set_ylim(-90.0, 90.0)
        ax.grid(True, alpha=0.18)
        ax.legend(loc="upper right", fontsize=6.5, framealpha=0.9)
    else:
        for layout in layouts:
            subset = sorted(
                [row for row in rows if row["layout"] == layout],
                key=lambda row: _int(row, "support_count"),
            )
            counts = [_int(row, "support_count") for row in subset]
            residuals = [
                max(
                    max(
                        _float(row, "full_residual_inf"),
                        _float(row, "transport_residual_inf"),
                    ),
                    floor,
                )
                for row in subset
            ]
            ax.semilogy(counts, residuals, "o-", label=layout)
        ax.set_title("c  Linear solve residual", fontweight="bold")
        ax.set_xlabel("Number of supports")
        ax.set_ylabel("Infinity-norm residual")
        ax.set_xticks(sorted({_int(row, "support_count") for row in rows}))
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)

    fig.savefig(output_png, dpi=180)
    if output_pdf is not None:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf)
    plt.close(fig)
    return output_png


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build 3D benchmark figures.")
    parser.add_argument("--figure", choices=("m1", "m2", "m3"), default="m1")
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--output-png", type=Path, default=None)
    parser.add_argument("--output-pdf", type=Path, default=None)
    parser.add_argument("--response-csv", type=Path, default=None)
    parser.add_argument("--response-summary-csv", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    defaults = {
        "m1": (
            Path("data_raw/3d_m1_single_support/benchmark3d_m1_single_support_scaling.csv"),
            Path("figures/main/fig3_3d_m1_single_support_scaling.png"),
            Path("figures/main/fig3_3d_m1_single_support_scaling.pdf"),
        ),
        "m2": (
            Path("data_raw/3d_m2_multi_support/benchmark3d_m2_multi_support_far_clustered.csv"),
            Path("figures/main/fig4_3d_m2_multi_support_far_clustered.png"),
            Path("figures/main/fig4_3d_m2_multi_support_far_clustered.pdf"),
        ),
        "m3": (
            Path("data_raw/3d_m3_radiosity/benchmark3d_m3_radiosity_toy_solve.csv"),
            Path("figures/main/fig5_3d_m3_radiosity_toy_solve.png"),
            Path("figures/main/fig5_3d_m3_radiosity_toy_solve.pdf"),
        ),
    }
    default_input, default_png, default_pdf = defaults[args.figure]
    input_csv = args.input_csv or default_input
    output_png = args.output_png or default_png
    output_pdf = args.output_pdf or default_pdf
    if args.figure == "m1":
        path = make_m1_figure(input_csv, output_png, output_pdf)
    elif args.figure == "m2":
        path = make_m2_figure(input_csv, output_png, output_pdf)
    else:
        path = make_m3_figure(
            input_csv,
            output_png,
            output_pdf,
            response_csv=args.response_csv,
            response_summary_csv=args.response_summary_csv,
        )
    print(f"figure written: {path}")
    if output_pdf is not None:
        print(f"pdf written: {output_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
