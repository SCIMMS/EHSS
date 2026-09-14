from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .pvd_demo import (
    PVDGeometry,
    PVDRayBundle,
    PVDSegment,
    make_pvd_rays,
    make_trench_pvd_geometry,
    trace_pvd_first_hits,
)
from .pvd_reference import trace_pvd_first_hits_independent
from .radiosity import solve_radiosity


RTX3090_NAME = "NVIDIA GeForce RTX 3090"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def total_physical_memory_bytes() -> int | None:
    if os.name == "nt":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
    if hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_PAGE_SIZE")) * int(
                os.sysconf("SC_PHYS_PAGES")
            )
        except (OSError, ValueError):
            return None
    return None


def linux_os_release() -> dict[str, str]:
    if not sys.platform.startswith("linux"):
        return {}
    source = Path("/etc/os-release")
    if not source.is_file():
        return {}
    values: dict[str, str] = {}
    for line in source.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def linux_cpu_model() -> str | None:
    if not sys.platform.startswith("linux"):
        return None
    source = Path("/proc/cpuinfo")
    if not source.is_file():
        return None
    for line in source.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        if line.lower().startswith("model name") and ":" in line:
            return line.split(":", 1)[1].strip()
    return None


def nvidia_smi_receipt() -> dict[str, Any]:
    fields = (
        "name,uuid,memory.total,driver_version,pstate,temperature.gpu,"
        "clocks.current.graphics,clocks.current.memory"
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            encoding="utf-8",
            errors="replace",
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": type(exc).__name__}
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    parsed = [[part.strip() for part in row.split(",")] for row in rows]
    return {
        "available": completed.returncode == 0 and bool(parsed),
        "returncode": completed.returncode,
        "query_fields": fields.split(","),
        "rows": parsed,
        "stderr": completed.stderr.strip(),
    }


def build_environment_receipt(
    *,
    root: Path,
    source_paths: Iterable[str] = (),
) -> dict[str, Any]:
    gpu = nvidia_smi_receipt()
    machine = {
        "hostname": socket.gethostname(),
        "os_name": os.name,
        "sys_platform": sys.platform,
        "platform": platform.platform(),
        "linux_os_release": linux_os_release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "linux_cpu_model": linux_cpu_model(),
        "processor_identifier": os.environ.get("PROCESSOR_IDENTIFIER"),
        "cpu_count": os.cpu_count(),
        "total_physical_memory_bytes": total_physical_memory_bytes(),
        "nvidia_smi_rows": gpu.get("rows", []),
    }
    fingerprint_source = json.dumps(
        machine,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema": "support-event-external-environment-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine": machine,
        "machine_fingerprint_sha256": hashlib.sha256(
            fingerprint_source
        ).hexdigest(),
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "packages": {
                name: package_version(name)
                for name in ("numpy", "scipy", "pytest")
            },
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        "nvidia_smi": gpu,
        "compute_contract": {
            "target_environment": RTX3090_NAME,
            "kernel_backend": "NumPy/Python CPU",
            "gpu_used_by_numerical_kernels": False,
            "gpu_role": (
                "Machine provenance only. The present heat-transfer and PVD "
                "kernels do not execute CUDA work."
            ),
            "timing_clock": "time.perf_counter",
        },
        "source_sha256": {
            path: sha256(root / path)
            for path in source_paths
            if (root / path).is_file()
        },
    }


def _finite_tau_error(actual: np.ndarray, expected: np.ndarray) -> float:
    finite = np.isfinite(actual) & np.isfinite(expected)
    if np.any(np.isfinite(actual) != np.isfinite(expected)):
        return float("inf")
    if not np.any(finite):
        return 0.0
    return float(np.max(np.abs(actual[finite] - expected[finite])))


def analytical_pvd_reference(tolerance: float = 1e-12) -> dict[str, Any]:
    angles_deg = np.asarray([-57.0, -41.0, -23.0, 0.0, 19.0, 37.0, 55.0])
    angles = np.deg2rad(angles_deg)
    height = 2.3
    flat_rays = PVDRayBundle(
        origins=np.column_stack(
            (np.linspace(-1.0, 1.0, len(angles)), np.full(len(angles), height))
        ),
        directions=np.column_stack((np.sin(angles), -np.cos(angles))),
        weights=np.full(len(angles), 1.0 / len(angles)),
    )
    flat_geometry = PVDGeometry(
        segments=(
            PVDSegment(
                start=np.asarray([-10.0, 0.0]),
                end=np.asarray([10.0, 0.0]),
                normal=np.asarray([0.0, 1.0]),
                label="flat_receiver",
            ),
        )
    )
    flat_trace = trace_pvd_first_hits(flat_geometry, flat_rays)
    flat_independent = trace_pvd_first_hits_independent(
        flat_geometry,
        flat_rays,
    )
    expected_tau = height / np.cos(angles)
    expected_incidence = np.cos(angles)
    expected_flux = float(np.mean(expected_incidence))
    measured_flux = float(
        np.sum(flat_trace.weights * flat_trace.incidence)
    )
    flat_metrics = {
        "ray_count": flat_rays.n_rays,
        "target_mismatch_rays": int(np.count_nonzero(flat_trace.targets != 0)),
        "tau_max_abs_error": _finite_tau_error(
            flat_trace.taus,
            expected_tau,
        ),
        "incidence_max_abs_error": float(
            np.max(np.abs(flat_trace.incidence - expected_incidence))
        ),
        "flux_abs_error": abs(measured_flux - expected_flux),
        "independent_target_mismatch_rays": int(
            np.count_nonzero(
                flat_independent.targets != flat_trace.targets
            )
        ),
        "independent_tau_max_abs_error": _finite_tau_error(
            flat_independent.taus,
            flat_trace.taus,
        ),
    }

    trench = make_trench_pvd_geometry(
        trench_width=1.2,
        trench_depth=2.4,
        field_half_width=2.8,
    )
    trench_rays = make_pvd_rays(
        source_samples=112,
        angle_samples=1,
        source_half_width=2.8,
        source_y=1.2,
        max_angle_deg=1.0,
        quadrature_rule="midpoint",
    )
    trench_trace = trace_pvd_first_hits(trench, trench_rays)
    trench_independent = trace_pvd_first_hits_independent(
        trench,
        trench_rays,
    )
    x = trench_rays.origins[:, 0]
    expected_targets = np.where(x < -0.6, 0, np.where(x > 0.6, 1, 4))
    expected_trench_tau = np.where(np.abs(x) < 0.6, 3.6, 1.2)
    bottom_mask = trench_trace.targets == 4
    mask_mask = (trench_trace.targets == 0) | (trench_trace.targets == 1)
    bottom_flux = float(
        np.sum(
            trench_trace.weights[bottom_mask]
            * trench_trace.incidence[bottom_mask]
        )
    )
    mask_flux = float(
        np.sum(
            trench_trace.weights[mask_mask]
            * trench_trace.incidence[mask_mask]
        )
    )
    bottom_flux_density = bottom_flux / 1.2
    mask_flux_density = mask_flux / 4.4
    step_coverage = bottom_flux_density / mask_flux_density
    trench_metrics = {
        "ray_count": trench_rays.n_rays,
        "target_mismatch_rays": int(
            np.count_nonzero(trench_trace.targets != expected_targets)
        ),
        "tau_max_abs_error": _finite_tau_error(
            trench_trace.taus,
            expected_trench_tau,
        ),
        "incidence_max_abs_error": float(
            np.max(np.abs(trench_trace.incidence - 1.0))
        ),
        "bottom_ray_count": int(np.count_nonzero(bottom_mask)),
        "left_mask_ray_count": int(
            np.count_nonzero(trench_trace.targets == 0)
        ),
        "right_mask_ray_count": int(
            np.count_nonzero(trench_trace.targets == 1)
        ),
        "bottom_to_mask_step_coverage": step_coverage,
        "step_coverage_abs_error": abs(step_coverage - 1.0),
        "independent_target_mismatch_rays": int(
            np.count_nonzero(
                trench_independent.targets != trench_trace.targets
            )
        ),
        "independent_tau_max_abs_error": _finite_tau_error(
            trench_independent.taus,
            trench_trace.taus,
        ),
    }
    gates = {
        "flat_all_rays_hit": flat_metrics["target_mismatch_rays"] == 0,
        "flat_tau_matches_analytic": flat_metrics["tau_max_abs_error"]
        <= tolerance,
        "flat_incidence_matches_cosine": flat_metrics[
            "incidence_max_abs_error"
        ]
        <= tolerance,
        "flat_flux_matches_cosine_quadrature": flat_metrics["flux_abs_error"]
        <= tolerance,
        "flat_independent_kernel_matches": flat_metrics[
            "independent_target_mismatch_rays"
        ]
        == 0
        and flat_metrics["independent_tau_max_abs_error"] <= tolerance,
        "vertical_trench_labels_match_visibility": trench_metrics[
            "target_mismatch_rays"
        ]
        == 0,
        "vertical_trench_tau_matches_analytic": trench_metrics[
            "tau_max_abs_error"
        ]
        <= tolerance,
        "vertical_trench_incidence_is_unity": trench_metrics[
            "incidence_max_abs_error"
        ]
        <= tolerance,
        "vertical_trench_counts_are_44_24_44": (
            trench_metrics["left_mask_ray_count"],
            trench_metrics["bottom_ray_count"],
            trench_metrics["right_mask_ray_count"],
        )
        == (44, 24, 44),
        "vertical_trench_step_coverage_is_unity": trench_metrics[
            "step_coverage_abs_error"
        ]
        <= tolerance,
        "vertical_trench_independent_kernel_matches": trench_metrics[
            "independent_target_mismatch_rays"
        ]
        == 0
        and trench_metrics["independent_tau_max_abs_error"] <= tolerance,
    }
    return {
        "schema": "support-event-pvd-analytical-reference-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tolerance": tolerance,
        "flat_oblique_cosine_reference": flat_metrics,
        "vertical_trench_visibility_reference": trench_metrics,
        "gates": gates,
        "passed": all(gates.values()),
        "claim_boundary": [
            "The flat benchmark is an analytic ray-plane and cosine-incidence oracle.",
            (
                "The trench benchmark is a vertical line-of-sight visibility "
                "and step-coverage oracle for the stated discrete source grid."
            ),
            (
                "These checks validate ballistic geometry and flux assembly; "
                "they are not calibrated sputter-process experiments."
            ),
        ],
    }


def analytical_heat_transfer_reference(
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    emission = np.asarray([1.25, 0.70], dtype=float)
    reflectivity = np.asarray([0.35, 0.55], dtype=float)
    view_factors = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    denominator = 1.0 - reflectivity[0] * reflectivity[1]
    expected_radiosity = np.asarray(
        [
            (emission[0] + reflectivity[0] * emission[1]) / denominator,
            (emission[1] + reflectivity[1] * emission[0]) / denominator,
        ]
    )
    expected_irradiation = expected_radiosity[::-1].copy()
    expected_net_flux = expected_radiosity - expected_irradiation
    solution = solve_radiosity(view_factors, emission, reflectivity)
    metrics = {
        "surface_count": 2,
        "radiosity_max_abs_error": float(
            np.max(np.abs(solution.radiosity - expected_radiosity))
        ),
        "irradiation_max_abs_error": float(
            np.max(np.abs(solution.irradiation - expected_irradiation))
        ),
        "net_flux_max_abs_error": float(
            np.max(np.abs(solution.net_flux - expected_net_flux))
        ),
        "residual_inf": solution.residual_inf,
        "net_exchange_balance_abs_error": abs(
            float(np.sum(solution.net_flux))
        ),
    }
    gates = {
        "closed_form_radiosity_match": metrics["radiosity_max_abs_error"]
        <= tolerance,
        "closed_form_irradiation_match": metrics[
            "irradiation_max_abs_error"
        ]
        <= tolerance,
        "closed_form_net_flux_match": metrics["net_flux_max_abs_error"]
        <= tolerance,
        "linear_system_residual_small": metrics["residual_inf"] <= tolerance,
        "two_surface_exchange_balances": metrics[
            "net_exchange_balance_abs_error"
        ]
        <= tolerance,
    }
    return {
        "schema": "support-event-heat-transfer-analytical-reference-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tolerance": tolerance,
        "two_surface_closed_form": metrics,
        "gates": gates,
        "passed": all(gates.values()),
        "claim_boundary": [
            (
                "This oracle validates the opaque diffuse two-surface "
                "radiosity, irradiation, and net-flux solve."
            ),
            (
                "It is an analytical heat-transfer check, separate from the "
                "routed fixed-support geometry benchmarks."
            ),
        ],
    }


def analytical_external_reference(
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    heat_transfer = analytical_heat_transfer_reference(tolerance)
    pvd = analytical_pvd_reference(tolerance)
    gates = {
        "heat_transfer_analytical_reference": bool(heat_transfer["passed"]),
        "pvd_analytical_reference": bool(pvd["passed"]),
    }
    return {
        "schema": "support-event-external-analytical-reference-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "heat_transfer": heat_transfer,
        "pvd": pvd,
        "gates": gates,
        "passed": all(gates.values()),
    }


def distribution(values: Iterable[float]) -> dict[str, float | int]:
    data = np.asarray(list(values), dtype=float)
    if data.ndim != 1 or len(data) == 0 or not np.all(np.isfinite(data)):
        raise ValueError("timing distribution must contain finite values.")
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=1)) if len(data) > 1 else 0.0
    return {
        "count": int(len(data)),
        "min": float(np.min(data)),
        "p05": float(np.quantile(data, 0.05)),
        "p25": float(np.quantile(data, 0.25)),
        "median": float(np.median(data)),
        "mean": mean,
        "p75": float(np.quantile(data, 0.75)),
        "p95": float(np.quantile(data, 0.95)),
        "max": float(np.max(data)),
        "sample_std": std,
        "coefficient_of_variation": std / mean if mean > 0.0 else math.inf,
    }
