#!/usr/bin/env python3
"""Offline matmul split and phone-offload planner.

This is a screening model. It only recommends work whose transfer, residency,
backend qualification, median latency, and p90 latency pass the configured
policy. Every reported latency remains an estimate until the exact shape is
benchmarked end to end.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROFILE_SCHEMA = "s41-offload-hardware-profile-v1"
WORKLOAD_SCHEMA = "s41-offload-workload-v1"
OUTPUT_SCHEMA = "s41-offload-plan-v1"


class SolverError(ValueError):
    pass


@dataclass(frozen=True)
class Work:
    label: str
    m: int
    n_hint: int
    k_hint: int
    flops_g: float
    memory_mb: float
    dispatches: int = 1


def positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SolverError(f"{name} must be a positive integer")
    return value


def nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SolverError(f"{name} must be a nonnegative integer")
    return value


def nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise SolverError(f"{name} must be a nonnegative number")
    return float(value)


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SolverError(f"{name} must be an object")
    return value


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="ascii") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SolverError(f"cannot read {path}: {exc}") from exc
    return require_mapping(value, str(path))


def quant_format(profile: dict[str, Any], quant: str) -> dict[str, Any]:
    formats = require_mapping(profile.get("weight_formats"), "profile.weight_formats")
    if quant not in formats:
        raise SolverError(f"unknown weight format: {quant}")
    return require_mapping(formats[quant], f"profile.weight_formats.{quant}")


def matrix_weight_bytes(
    profile: dict[str, Any], n: int, k: int, quant: str
) -> int:
    fmt = quant_format(profile, quant)
    block_elements = positive_int(fmt.get("block_elements"), f"{quant}.block_elements")
    block_bytes = positive_int(fmt.get("block_bytes"), f"{quant}.block_bytes")
    return n * math.ceil(k / block_elements) * block_bytes


def scalar_bytes(profile: dict[str, Any], dtype: str) -> int:
    formats = require_mapping(profile.get("activation_formats"), "profile.activation_formats")
    if dtype not in formats:
        raise SolverError(f"unknown activation format: {dtype}")
    return positive_int(formats[dtype], f"profile.activation_formats.{dtype}")


def matmul_work(
    profile: dict[str, Any],
    *,
    label: str,
    m: int,
    n: int,
    k: int,
    quant: str,
    activation_type: str,
) -> Work:
    activation = scalar_bytes(profile, activation_type)
    weight = matrix_weight_bytes(profile, n, k, quant)
    input_bytes = m * k * activation
    output_bytes = m * n * activation
    return Work(
        label=label,
        m=m,
        n_hint=n,
        k_hint=k,
        flops_g=(2.0 * m * n * k) / 1.0e9,
        memory_mb=(weight + input_bytes + output_bytes) / 1.0e6,
    )


def ffn_work(
    profile: dict[str, Any],
    *,
    label: str,
    m: int,
    hidden: int,
    intermediate: int,
    gate_count: int,
    quant: str,
    activation_type: str,
) -> Work:
    activation = scalar_bytes(profile, activation_type)
    gate_weights = gate_count * matrix_weight_bytes(profile, intermediate, hidden, quant)
    down_weights = matrix_weight_bytes(profile, hidden, intermediate, quant)
    scratch_elements = m * (2 * hidden + (gate_count + 1) * intermediate)
    parameters = (gate_count + 1) * hidden * intermediate
    return Work(
        label=label,
        m=m,
        n_hint=max(hidden, intermediate),
        k_hint=max(hidden, intermediate),
        flops_g=(2.0 * m * parameters) / 1.0e9,
        memory_mb=(gate_weights + down_weights + activation * scratch_elements) / 1.0e6,
    )


def combine_work(label: str, works: list[Work]) -> Work:
    if not works:
        return Work(
            label=label,
            m=1,
            n_hint=1,
            k_hint=1,
            flops_g=0.0,
            memory_mb=0.0,
            dispatches=0,
        )
    return Work(
        label=label,
        m=max(work.m for work in works),
        n_hint=max(work.n_hint for work in works),
        k_hint=max(work.k_hint for work in works),
        flops_g=sum(work.flops_g for work in works),
        memory_mb=sum(work.memory_mb for work in works),
        dispatches=sum(work.dispatches for work in works),
    )


def shape_efficiency(
    device: dict[str, Any], work: Work
) -> tuple[float, float, str]:
    model = require_mapping(
        device.get("shape_model", {"kind": "flat"}), "device.shape_model"
    )
    if model.get("kind", "flat") == "flat":
        return 1.0, 1.0, "flat"

    floor = float(model.get("efficiency_floor", 0.2))
    max_profiled_batch = positive_int(
        model.get("max_profiled_batch", 1), "max_profiled_batch"
    )
    if work.m <= 1:
        saturation_n = positive_int(
            model.get("m1_saturation_n", 512), "m1_saturation_n"
        )
        width = max(floor, min(1.0, work.n_hint / saturation_n))
        confidence = "profiled_m1" if max_profiled_batch >= 1 else "extrapolated"
        return width, width, confidence

    saturation_n = positive_int(
        model.get("batched_saturation_n", 4096), "batched_saturation_n"
    )
    saturation_m = positive_int(
        model.get("batched_saturation_m", 64), "batched_saturation_m"
    )
    width = max(floor, min(1.0, work.n_hint / saturation_n))
    batch = max(floor, min(1.0, work.m / saturation_m))
    memory_efficiency = max(
        floor, min(1.0, work.n_hint / max(1, saturation_n // 2))
    )
    confidence = (
        "profiled_batch"
        if work.m <= max_profiled_batch
        else "extrapolated_batch"
    )
    return memory_efficiency, max(floor, width * batch), confidence


def estimate_device(
    device: dict[str, Any], work: Work, family: str | None = None
) -> dict[str, Any]:
    effective_device = dict(device)
    family_overrides = require_mapping(
        device.get("family_overrides", {}), "device.family_overrides"
    )
    if family in family_overrides:
        effective_device.update(
            require_mapping(
                family_overrides[family],
                f"device.family_overrides.{family}",
            )
        )
    memory_gbps = float(effective_device["memory_gbps"])
    compute_gflops = float(effective_device["compute_gflops"])
    if memory_gbps <= 0 or compute_gflops <= 0:
        raise SolverError("device bandwidth and compute rate must be positive")
    memory_efficiency, compute_efficiency, confidence = shape_efficiency(
        effective_device, work
    )
    memory_ms = work.memory_mb / (memory_gbps * memory_efficiency)
    compute_ms = 1000.0 * work.flops_g / (
        compute_gflops * compute_efficiency
    )
    dispatch_ms = (
        float(effective_device.get("dispatch_ms", 0.0))
        * work.dispatches
    )
    median_ms = dispatch_ms + max(memory_ms, compute_ms)
    p90_ms = median_ms * float(
        effective_device.get("p90_multiplier", 1.1)
    )
    return {
        "median_ms": median_ms,
        "p90_ms": p90_ms,
        "family": family,
        "effective_memory_gbps": memory_gbps,
        "effective_compute_gflops": compute_gflops,
        "dispatch_ms": dispatch_ms,
        "memory_floor_ms": memory_ms,
        "compute_floor_ms": compute_ms,
        "memory_efficiency": memory_efficiency,
        "compute_efficiency": compute_efficiency,
        "shape_confidence": confidence,
        "work": {
            "flops_g": work.flops_g,
            "memory_mb": work.memory_mb,
            "m": work.m,
            "n_hint": work.n_hint,
            "k_hint": work.k_hint,
            "dispatches": work.dispatches,
        },
    }


def estimate_linear_latency(
    model: dict[str, Any], payload_bytes: int
) -> tuple[float, float]:
    payload_kib = payload_bytes / 1024.0
    median = float(model["base_median_ms"]) + (
        float(model["per_kib_median_ms"]) * payload_kib
    )
    p90 = float(model["base_p90_ms"]) + (
        float(model["per_kib_p90_ms"]) * payload_kib
    )
    return median, p90


def estimate_transport(
    profile: dict[str, Any],
    input_bytes: int,
    output_bytes: int,
    protocol: str,
    inter_request_gap_ms: float,
) -> dict[str, Any]:
    transports = require_mapping(profile.get("transports"), "profile.transports")
    names = list(transports) if protocol == "auto" else [protocol]
    if not names:
        raise SolverError("profile has no transports")
    choices = []
    total_bytes = input_bytes + output_bytes
    for name in names:
        if name not in transports:
            raise SolverError(f"unknown transport: {name}")
        model = require_mapping(
            transports[name], f"profile.transports.{name}"
        )
        median, p90 = estimate_linear_latency(model, total_bytes)
        gap_threshold = float(model.get("gap_threshold_ms", math.inf))
        if inter_request_gap_ms >= gap_threshold:
            median += float(model.get("gap_penalty_median_ms", 0.0))
            p90 += float(model.get("gap_penalty_p90_ms", 0.0))
        choices.append((median, p90, name))
    median, p90, name = min(
        choices, key=lambda value: (value[0], value[1], value[2])
    )
    return {
        "protocol": name,
        "median_ms": median,
        "p90_ms": p90,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "total_bytes": total_bytes,
    }


def estimate_staging(
    host: dict[str, Any], input_bytes: int, output_bytes: int
) -> tuple[dict[str, Any], list[str]]:
    if host.get("activation_location", "ram") == "ram":
        return {
            "median_ms": 0.0,
            "p90_ms": 0.0,
            "complete": True,
        }, []

    staging = require_mapping(
        host.get("phone_staging", {}), "host.phone_staging"
    )
    if not staging.get("complete", False):
        return {
            "median_ms": 0.0,
            "p90_ms": 0.0,
            "complete": False,
        }, ["missing measured GPU D2H/H2D staging profile"]

    d2h = require_mapping(staging.get("d2h"), "host.phone_staging.d2h")
    h2d = require_mapping(staging.get("h2d"), "host.phone_staging.h2d")
    d2h_median, d2h_p90 = estimate_linear_latency(d2h, input_bytes)
    h2d_median, h2d_p90 = estimate_linear_latency(h2d, output_bytes)
    return {
        "median_ms": d2h_median + h2d_median,
        "p90_ms": d2h_p90 + h2d_p90,
        "complete": True,
        "d2h_median_ms": d2h_median,
        "h2d_median_ms": h2d_median,
    }, []


def estimate_merge(
    profile: dict[str, Any], bytes_moved: int, kind: str
) -> dict[str, Any]:
    if bytes_moved <= 0:
        return {
            "kind": kind,
            "median_ms": 0.0,
            "p90_ms": 0.0,
            "bytes": 0,
        }
    merge = require_mapping(profile.get("merge"), "profile.merge")
    passes = 3 if kind == "reduction" else 1
    median = float(merge.get("base_ms", 0.0)) + (
        passes * (bytes_moved / 1.0e6) / float(merge["memory_gbps"])
    )
    return {
        "kind": kind,
        "median_ms": median,
        "p90_ms": median * float(merge.get("p90_multiplier", 1.2)),
        "bytes": bytes_moved,
    }


def aligned_cuts(total: int, alignment: int, search_points: int) -> list[int]:
    if total <= alignment:
        return []
    units = total // alignment
    if units <= 1:
        return []
    if units - 1 <= search_points:
        return [alignment * unit for unit in range(1, units)]
    selected = {
        alignment
        * max(1, min(units - 1, round(index * units / search_points)))
        for index in range(1, search_points)
    }
    return sorted(selected)


def phone_checks(
    phone_name: str,
    phone: dict[str, Any],
    family: str,
    quant: str,
    resident_bytes: int,
    allow_unqualified: bool,
) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    if quant not in phone.get("supported_quantizations", []):
        blockers.append(f"{phone_name} does not support {quant}")

    qualification_issues = []
    if not phone.get("qualified", True):
        qualification_issues.append(f"{phone_name} kernel path is not qualified")
    if quant not in phone.get("qualified_quantizations", []):
        qualification_issues.append(f"{phone_name} {quant} path is not qualified")
    if family not in phone.get("qualified_workloads", []):
        qualification_issues.append(
            f"{phone_name} {family} path is not qualified"
        )
    if qualification_issues:
        warnings.extend(qualification_issues)
        if not allow_unqualified:
            blockers.extend(qualification_issues)

    budget_bytes = (
        float(phone.get("resident_budget_mib", 0.0)) * 1024.0 * 1024.0
    )
    if budget_bytes <= 0 or resident_bytes > budget_bytes:
        blockers.append(
            f"phone resident weights need "
            f"{resident_bytes / (1024.0 * 1024.0):.1f} MiB, "
            f"budget is {budget_bytes / (1024.0 * 1024.0):.1f} MiB"
        )
    return blockers, warnings


def make_parallel_candidate(
    *,
    profile: dict[str, Any],
    workload: dict[str, Any],
    host_name: str,
    host: dict[str, Any],
    phone_name: str,
    phone: dict[str, Any],
    family: str,
    quant: str,
    strategy: str,
    split: dict[str, Any],
    host_work: Work,
    phone_work: Work,
    input_bytes: int,
    output_bytes: int,
    resident_bytes: int,
    working_weight_bytes: int,
    merge_kind: str,
    merge_bytes: int,
    allow_unqualified: bool,
) -> dict[str, Any]:
    host_estimate = estimate_device(host, host_work, family)
    phone_estimate = estimate_device(phone, phone_work, family)
    protocol = str(workload.get("transport", "auto"))
    gap = nonnegative_float(
        workload.get("inter_request_gap_ms", 0.0),
        "inter_request_gap_ms",
    )
    transport = estimate_transport(
        profile, input_bytes, output_bytes, protocol, gap
    )
    staging, staging_blockers = estimate_staging(
        host, input_bytes, output_bytes
    )
    merge = estimate_merge(profile, merge_bytes, merge_kind)
    remote_median = (
        staging["median_ms"]
        + transport["median_ms"]
        + phone_estimate["median_ms"]
    )
    remote_p90 = (
        staging["p90_ms"]
        + transport["p90_ms"]
        + phone_estimate["p90_ms"]
    )
    median = max(host_estimate["median_ms"], remote_median) + merge["median_ms"]
    p90 = max(host_estimate["p90_ms"], remote_p90) + merge["p90_ms"]
    blockers, warnings = phone_checks(
        phone_name,
        phone,
        family,
        quant,
        resident_bytes,
        allow_unqualified,
    )
    blockers.extend(staging_blockers)
    if phone_estimate["shape_confidence"].startswith("extrapolated"):
        issue = (
            f"{phone_name} shape is outside the profiled batch regime"
        )
        warnings.append(issue)
        if workload.get("require_profiled_shape", False):
            blockers.append(issue)
    return {
        "strategy": strategy,
        "split": split,
        "host_backend": host_name,
        "phone_backend": phone_name,
        "median_ms": median,
        "p90_ms": p90,
        "host_branch": host_estimate,
        "remote_branch": {
            "median_ms": remote_median,
            "p90_ms": remote_p90,
            "staging": staging,
            "transport": transport,
            "phone_compute": phone_estimate,
        },
        "merge": merge,
        "phone_resident_bytes": resident_bytes,
        "phone_working_weight_bytes": working_weight_bytes,
        "preparation": {
            "mode": "preloaded",
            "one_time_ms": None,
            "included_in_request_latency": False,
        },
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(warnings)),
    }


def get_devices(
    profile: dict[str, Any], workload: dict[str, Any]
) -> tuple[str, dict[str, Any], dict[str, dict[str, Any]]]:
    hosts = require_mapping(profile.get("hosts"), "profile.hosts")
    phones = require_mapping(
        profile.get("phone_backends"), "profile.phone_backends"
    )
    host_name = str(workload.get("host_backend", "cpu_isolated"))
    if host_name not in hosts:
        raise SolverError(f"unknown host backend: {host_name}")
    requested = workload.get("phone_backends")
    if requested is None:
        names = list(phones)
    else:
        if not isinstance(requested, list) or not all(
            isinstance(name, str) for name in requested
        ):
            raise SolverError("phone_backends must be a list of names")
        names = requested
    selected = {}
    for name in names:
        if name not in phones:
            raise SolverError(f"unknown phone backend: {name}")
        selected[name] = require_mapping(
            phones[name], f"profile.phone_backends.{name}"
        )
    return (
        host_name,
        require_mapping(hosts[host_name], f"profile.hosts.{host_name}"),
        selected,
    )


def solve_matmul(
    profile: dict[str, Any],
    workload: dict[str, Any],
    allow_unqualified: bool,
) -> tuple[Work, list[dict[str, Any]], dict[str, Any]]:
    host_name, host, phones = get_devices(profile, workload)
    quant = str(workload.get("quantization", "q8_0"))
    activation_type = str(workload.get("activation_type", "f16"))
    m = positive_int(workload.get("m", workload.get("batch", 1)), "m")
    n = positive_int(workload.get("n"), "n")
    k = positive_int(workload.get("k"), "k")
    alignment = positive_int(
        workload.get("split_alignment", 32), "split_alignment"
    )
    search_points = positive_int(
        workload.get("search_points", 128), "search_points"
    )
    axes = workload.get(
        "split_axes", ["n", "k"] + (["m"] if m > 1 else [])
    )
    if not isinstance(axes, list) or any(
        axis not in {"m", "n", "k"} for axis in axes
    ):
        raise SolverError("split_axes must contain only m, n, or k")
    activation = scalar_bytes(profile, activation_type)
    full = matmul_work(
        profile,
        label="full matmul",
        m=m,
        n=n,
        k=k,
        quant=quant,
        activation_type=activation_type,
    )
    candidates = []
    for phone_name, phone in phones.items():
        full_weight = matrix_weight_bytes(profile, n, k, quant)
        candidates.append(
            make_parallel_candidate(
                profile=profile,
                workload=workload,
                host_name=host_name,
                host=host,
                phone_name=phone_name,
                phone=phone,
                family="matmul",
                quant=quant,
                strategy="full_serial_offload",
                split={"axis": "all", "phone_fraction": 1.0},
                host_work=combine_work("empty host branch", []),
                phone_work=full,
                input_bytes=m * k * activation,
                output_bytes=m * n * activation,
                resident_bytes=full_weight,
                working_weight_bytes=full_weight,
                merge_kind="none",
                merge_bytes=0,
                allow_unqualified=allow_unqualified,
            )
        )
        if "n" in axes:
            for phone_n in aligned_cuts(n, alignment, search_points):
                phone_work = matmul_work(
                    profile,
                    label="phone output rows",
                    m=m,
                    n=phone_n,
                    k=k,
                    quant=quant,
                    activation_type=activation_type,
                )
                host_work = matmul_work(
                    profile,
                    label="host output rows",
                    m=m,
                    n=n - phone_n,
                    k=k,
                    quant=quant,
                    activation_type=activation_type,
                )
                phone_weight = matrix_weight_bytes(
                    profile, phone_n, k, quant
                )
                candidates.append(
                    make_parallel_candidate(
                        profile=profile,
                        workload=workload,
                        host_name=host_name,
                        host=host,
                        phone_name=phone_name,
                        phone=phone,
                        family="matmul",
                        quant=quant,
                        strategy="output_row_parallel",
                        split={
                            "axis": "n",
                            "phone_n": phone_n,
                            "host_n": n - phone_n,
                            "phone_fraction": phone_n / n,
                        },
                        host_work=host_work,
                        phone_work=phone_work,
                        input_bytes=m * k * activation,
                        output_bytes=m * phone_n * activation,
                        resident_bytes=phone_weight,
                        working_weight_bytes=phone_weight,
                        merge_kind="concatenate",
                        merge_bytes=m * phone_n * activation,
                        allow_unqualified=allow_unqualified,
                    )
                )
        if "k" in axes:
            for phone_k in aligned_cuts(k, alignment, search_points):
                phone_work = matmul_work(
                    profile,
                    label="phone reduction columns",
                    m=m,
                    n=n,
                    k=phone_k,
                    quant=quant,
                    activation_type=activation_type,
                )
                host_work = matmul_work(
                    profile,
                    label="host reduction columns",
                    m=m,
                    n=n,
                    k=k - phone_k,
                    quant=quant,
                    activation_type=activation_type,
                )
                phone_weight = matrix_weight_bytes(
                    profile, n, phone_k, quant
                )
                candidates.append(
                    make_parallel_candidate(
                        profile=profile,
                        workload=workload,
                        host_name=host_name,
                        host=host,
                        phone_name=phone_name,
                        phone=phone,
                        family="matmul",
                        quant=quant,
                        strategy="reduction_parallel",
                        split={
                            "axis": "k",
                            "phone_k": phone_k,
                            "host_k": k - phone_k,
                            "phone_fraction": phone_k / k,
                        },
                        host_work=host_work,
                        phone_work=phone_work,
                        input_bytes=m * phone_k * activation,
                        output_bytes=m * n * activation,
                        resident_bytes=phone_weight,
                        working_weight_bytes=phone_weight,
                        merge_kind="reduction",
                        merge_bytes=m * n * activation,
                        allow_unqualified=allow_unqualified,
                    )
                )
        if "m" in axes and m > 1:
            for phone_m in aligned_cuts(m, 1, min(search_points, m)):
                phone_work = matmul_work(
                    profile,
                    label="phone batch columns",
                    m=phone_m,
                    n=n,
                    k=k,
                    quant=quant,
                    activation_type=activation_type,
                )
                host_work = matmul_work(
                    profile,
                    label="host batch columns",
                    m=m - phone_m,
                    n=n,
                    k=k,
                    quant=quant,
                    activation_type=activation_type,
                )
                candidates.append(
                    make_parallel_candidate(
                        profile=profile,
                        workload=workload,
                        host_name=host_name,
                        host=host,
                        phone_name=phone_name,
                        phone=phone,
                        family="matmul",
                        quant=quant,
                        strategy="batch_parallel",
                        split={
                            "axis": "m",
                            "phone_m": phone_m,
                            "host_m": m - phone_m,
                            "phone_fraction": phone_m / m,
                        },
                        host_work=host_work,
                        phone_work=phone_work,
                        input_bytes=phone_m * k * activation,
                        output_bytes=phone_m * n * activation,
                        resident_bytes=full_weight,
                        working_weight_bytes=full_weight,
                        merge_kind="concatenate",
                        merge_bytes=phone_m * n * activation,
                        allow_unqualified=allow_unqualified,
                    )
                )
    summary = {
        "m": m,
        "n": n,
        "k": k,
        "quantization": quant,
        "split_axes": axes,
    }
    return full, candidates, summary


def solve_ffn(
    profile: dict[str, Any],
    workload: dict[str, Any],
    allow_unqualified: bool,
) -> tuple[Work, list[dict[str, Any]], dict[str, Any]]:
    host_name, host, phones = get_devices(profile, workload)
    quant = str(workload.get("quantization", "q8_0"))
    activation_type = str(workload.get("activation_type", "f16"))
    m = positive_int(workload.get("m", workload.get("batch", 1)), "m")
    hidden = positive_int(workload.get("hidden"), "hidden")
    intermediate = positive_int(
        workload.get("intermediate"), "intermediate"
    )
    gate_count = positive_int(workload.get("gate_count", 2), "gate_count")
    alignment = positive_int(
        workload.get("split_alignment", 32), "split_alignment"
    )
    search_points = positive_int(
        workload.get("search_points", 128), "search_points"
    )
    activation = scalar_bytes(profile, activation_type)
    full = ffn_work(
        profile,
        label="full fused FFN",
        m=m,
        hidden=hidden,
        intermediate=intermediate,
        gate_count=gate_count,
        quant=quant,
        activation_type=activation_type,
    )
    cuts = set(aligned_cuts(intermediate, alignment, search_points))
    cuts.add(intermediate)
    candidates = []
    for phone_name, phone in phones.items():
        for phone_intermediate in sorted(cuts):
            host_intermediate = intermediate - phone_intermediate
            phone_work = ffn_work(
                profile,
                label="phone FFN columns",
                m=m,
                hidden=hidden,
                intermediate=phone_intermediate,
                gate_count=gate_count,
                quant=quant,
                activation_type=activation_type,
            )
            if host_intermediate > 0:
                host_work = ffn_work(
                    profile,
                    label="host FFN columns",
                    m=m,
                    hidden=hidden,
                    intermediate=host_intermediate,
                    gate_count=gate_count,
                    quant=quant,
                    activation_type=activation_type,
                )
            else:
                host_work = combine_work("empty host branch", [])
            phone_weight = (
                gate_count
                * matrix_weight_bytes(
                    profile, phone_intermediate, hidden, quant
                )
                + matrix_weight_bytes(
                    profile, hidden, phone_intermediate, quant
                )
            )
            candidates.append(
                make_parallel_candidate(
                    profile=profile,
                    workload=workload,
                    host_name=host_name,
                    host=host,
                    phone_name=phone_name,
                    phone=phone,
                    family="ffn",
                    quant=quant,
                    strategy="ffn_intermediate_parallel",
                    split={
                        "axis": "intermediate",
                        "phone_intermediate": phone_intermediate,
                        "host_intermediate": host_intermediate,
                        "phone_fraction": (
                            phone_intermediate / intermediate
                        ),
                    },
                    host_work=host_work,
                    phone_work=phone_work,
                    input_bytes=m * hidden * activation,
                    output_bytes=m * hidden * activation,
                    resident_bytes=phone_weight,
                    working_weight_bytes=phone_weight,
                    merge_kind="reduction",
                    merge_bytes=m * hidden * activation,
                    allow_unqualified=allow_unqualified,
                )
            )
    summary = {
        "m": m,
        "hidden": hidden,
        "intermediate": intermediate,
        "gate_count": gate_count,
        "quantization": quant,
    }
    return full, candidates, summary


def solve_moe(
    profile: dict[str, Any],
    workload: dict[str, Any],
    allow_unqualified: bool,
) -> tuple[Work, list[dict[str, Any]], dict[str, Any]]:
    host_name, host, phones = get_devices(profile, workload)
    quant = str(workload.get("quantization", "q8_0"))
    activation_type = str(workload.get("activation_type", "f16"))
    m = positive_int(workload.get("m", workload.get("batch", 1)), "m")
    hidden = positive_int(workload.get("hidden"), "hidden")
    expert_intermediate = positive_int(
        workload.get("expert_intermediate"), "expert_intermediate"
    )
    active_experts = positive_int(
        workload.get("active_experts"), "active_experts"
    )
    total_experts = positive_int(
        workload.get("total_experts"), "total_experts"
    )
    shared_intermediate = nonnegative_int(
        workload.get("shared_intermediate", 0), "shared_intermediate"
    )
    gate_count = positive_int(workload.get("gate_count", 2), "gate_count")
    if active_experts > total_experts:
        raise SolverError("active_experts cannot exceed total_experts")
    residency = str(workload.get("expert_residency", "all"))
    if residency not in {"all", "active"}:
        raise SolverError("expert_residency must be all or active")
    activation = scalar_bytes(profile, activation_type)

    shared_works = []
    if shared_intermediate > 0:
        shared_works.append(
            ffn_work(
                profile,
                label="shared expert",
                m=m,
                hidden=hidden,
                intermediate=shared_intermediate,
                gate_count=gate_count,
                quant=quant,
                activation_type=activation_type,
            )
        )
    one_expert = ffn_work(
        profile,
        label="one routed expert",
        m=m,
        hidden=hidden,
        intermediate=expert_intermediate,
        gate_count=gate_count,
        quant=quant,
        activation_type=activation_type,
    )
    full = combine_work(
        "shared plus active experts",
        shared_works + [one_expert] * active_experts,
    )
    one_expert_weight = (
        gate_count
        * matrix_weight_bytes(
            profile, expert_intermediate, hidden, quant
        )
        + matrix_weight_bytes(
            profile, hidden, expert_intermediate, quant
        )
    )
    candidates = []
    for phone_name, phone in phones.items():
        resident_experts = (
            total_experts if residency == "all" else active_experts
        )
        resident_bytes = resident_experts * one_expert_weight
        for phone_experts in range(1, active_experts + 1):
            phone_work = combine_work(
                f"{phone_experts} phone experts",
                [one_expert] * phone_experts,
            )
            remaining = active_experts - phone_experts
            host_work = combine_work(
                "shared plus remaining experts",
                shared_works + [one_expert] * remaining,
            )
            candidates.append(
                make_parallel_candidate(
                    profile=profile,
                    workload=workload,
                    host_name=host_name,
                    host=host,
                    phone_name=phone_name,
                    phone=phone,
                    family="moe",
                    quant=quant,
                    strategy="whole_expert_parallel",
                    split={
                        "axis": "expert",
                        "phone_active_experts": phone_experts,
                        "host_active_experts": remaining,
                        "phone_fraction": phone_experts / active_experts,
                        "resident_policy": residency,
                        "resident_experts": resident_experts,
                    },
                    host_work=host_work,
                    phone_work=phone_work,
                    input_bytes=m * hidden * activation,
                    output_bytes=m * hidden * activation,
                    resident_bytes=resident_bytes,
                    working_weight_bytes=(
                        phone_experts * one_expert_weight
                    ),
                    merge_kind="reduction",
                    merge_bytes=m * hidden * activation,
                    allow_unqualified=allow_unqualified,
                )
            )
    summary = {
        "m": m,
        "hidden": hidden,
        "expert_intermediate": expert_intermediate,
        "active_experts": active_experts,
        "total_experts": total_experts,
        "shared_intermediate": shared_intermediate,
        "quantization": quant,
        "expert_residency": residency,
    }
    return full, candidates, summary


def chain_island_cost(
    profile: dict[str, Any],
    workload: dict[str, Any],
    host: dict[str, Any],
    phone: dict[str, Any],
    works: list[Work],
    operations: list[dict[str, int]],
    start: int,
    end: int,
    activation: int,
) -> tuple[dict[str, Any], list[str]]:
    island_work = combine_work(
        f"phone operations {start}:{end}", works[start:end]
    )
    phone_estimate = estimate_device(phone, island_work, "chain")
    input_bytes = (
        operations[start]["m"] * operations[start]["k"] * activation
    )
    output_bytes = (
        operations[end - 1]["m"]
        * operations[end - 1]["n"]
        * activation
    )
    transport = estimate_transport(
        profile,
        input_bytes,
        output_bytes,
        str(workload.get("transport", "auto")),
        nonnegative_float(
            workload.get("inter_request_gap_ms", 0.0),
            "inter_request_gap_ms",
        ),
    )
    staging, blockers = estimate_staging(
        host, input_bytes, output_bytes
    )
    return {
        "start": start,
        "end": end,
        "median_ms": (
            staging["median_ms"]
            + transport["median_ms"]
            + phone_estimate["median_ms"]
        ),
        "p90_ms": (
            staging["p90_ms"]
            + transport["p90_ms"]
            + phone_estimate["p90_ms"]
        ),
        "staging": staging,
        "transport": transport,
        "phone_compute": phone_estimate,
    }, blockers


def solve_chain(
    profile: dict[str, Any],
    workload: dict[str, Any],
    allow_unqualified: bool,
) -> tuple[Work, list[dict[str, Any]], dict[str, Any]]:
    host_name, host, phones = get_devices(profile, workload)
    quant = str(workload.get("quantization", "q8_0"))
    activation_type = str(workload.get("activation_type", "f16"))
    activation = scalar_bytes(profile, activation_type)
    raw_operations = workload.get("operations")
    if not isinstance(raw_operations, list) or not raw_operations:
        raise SolverError("operations must be a nonempty list")
    operations: list[dict[str, int]] = []
    works = []
    for index, raw in enumerate(raw_operations):
        op = require_mapping(raw, f"operations[{index}]")
        if op.get("kind", "matmul") != "matmul":
            raise SolverError(f"operations[{index}] must be a matmul")
        m = positive_int(
            op.get("m", workload.get("m", workload.get("batch", 1))),
            f"operations[{index}].m",
        )
        n = positive_int(op.get("n"), f"operations[{index}].n")
        k = positive_int(op.get("k"), f"operations[{index}].k")
        if operations and (
            operations[-1]["n"] != k or operations[-1]["m"] != m
        ):
            raise SolverError(
                f"operations[{index - 1}] output shape does not feed "
                f"operations[{index}]"
            )
        operations.append({"m": m, "n": n, "k": k})
        works.append(
            matmul_work(
                profile,
                label=f"operation {index}",
                m=m,
                n=n,
                k=k,
                quant=quant,
                activation_type=activation_type,
            )
        )
    full = combine_work("full host chain", works)
    candidates = []
    beam_width = positive_int(
        workload.get("chain_beam_width", 64), "chain_beam_width"
    )
    for phone_name, phone in phones.items():
        host_estimates = [
            estimate_device(host, work, "matmul") for work in works
        ]
        island_cache: dict[
            tuple[int, int], tuple[dict[str, Any], list[str]]
        ] = {}
        paths: list[list[dict[str, Any]]] = [
            [] for _ in range(len(works) + 1)
        ]
        paths[0] = [
            {
                "median_ms": 0.0,
                "p90_ms": 0.0,
                "segments": [],
                "islands": [],
                "stage_blockers": [],
            }
        ]
        for index in range(len(works)):
            if not paths[index]:
                continue
            for path in paths[index]:
                host_segment = {
                    "backend": host_name,
                    "start": index,
                    "end": index + 1,
                    "median_ms": host_estimates[index]["median_ms"],
                    "p90_ms": host_estimates[index]["p90_ms"],
                }
                paths[index + 1].append(
                    {
                        "median_ms": (
                            path["median_ms"]
                            + host_segment["median_ms"]
                        ),
                        "p90_ms": (
                            path["p90_ms"] + host_segment["p90_ms"]
                        ),
                        "segments": path["segments"] + [host_segment],
                        "islands": path["islands"],
                        "stage_blockers": path["stage_blockers"],
                    }
                )
                if (
                    path["segments"]
                    and path["segments"][-1]["backend"] == phone_name
                ):
                    continue
                for end in range(index + 1, len(works) + 1):
                    key = (index, end)
                    if key not in island_cache:
                        island_cache[key] = chain_island_cost(
                            profile,
                            workload,
                            host,
                            phone,
                            works,
                            operations,
                            index,
                            end,
                            activation,
                        )
                    island, stage_blockers = island_cache[key]
                    phone_segment = {
                        "backend": phone_name,
                        "start": index,
                        "end": end,
                        "median_ms": island["median_ms"],
                        "p90_ms": island["p90_ms"],
                    }
                    paths[end].append(
                        {
                            "median_ms": (
                                path["median_ms"]
                                + island["median_ms"]
                            ),
                            "p90_ms": (
                                path["p90_ms"] + island["p90_ms"]
                            ),
                            "segments": (
                                path["segments"] + [phone_segment]
                            ),
                            "islands": path["islands"] + [island],
                            "stage_blockers": (
                                path["stage_blockers"]
                                + stage_blockers
                            ),
                        }
                    )
            for target in range(index + 1, len(works) + 1):
                paths[target] = sorted(
                    paths[target],
                    key=lambda path: (
                        path["median_ms"],
                        path["p90_ms"],
                    ),
                )[:beam_width]

        seen = set()
        for path in paths[-1]:
            phone_indices = tuple(
                op_index
                for segment in path["segments"]
                if segment["backend"] == phone_name
                for op_index in range(segment["start"], segment["end"])
            )
            if not phone_indices or phone_indices in seen:
                continue
            seen.add(phone_indices)
            resident_bytes = sum(
                matrix_weight_bytes(
                    profile,
                    operations[op_index]["n"],
                    operations[op_index]["k"],
                    quant,
                )
                for op_index in set(phone_indices)
            )
            blockers, warnings = phone_checks(
                phone_name,
                phone,
                "chain",
                quant,
                resident_bytes,
                allow_unqualified,
            )
            blockers.extend(path["stage_blockers"])
            if any(
                island["phone_compute"][
                    "shape_confidence"
                ].startswith("extrapolated")
                for island in path["islands"]
            ):
                issue = (
                    f"{phone_name} shape is outside the profiled "
                    "batch regime"
                )
                warnings.append(issue)
                if workload.get("require_profiled_shape", False):
                    blockers.append(issue)
            candidates.append(
                {
                    "strategy": "ordered_backend_schedule",
                    "split": {
                        "phone_operation_indices": list(phone_indices),
                        "segments": path["segments"],
                    },
                    "host_backend": host_name,
                    "phone_backend": phone_name,
                    "median_ms": path["median_ms"],
                    "p90_ms": path["p90_ms"],
                    "phone_islands": path["islands"],
                    "phone_resident_bytes": resident_bytes,
                    "phone_working_weight_bytes": resident_bytes,
                    "preparation": {
                        "mode": "preloaded",
                        "one_time_ms": None,
                        "included_in_request_latency": False,
                    },
                    "blockers": sorted(set(blockers)),
                    "warnings": sorted(set(warnings)),
                }
            )
    summary = {
        "operations": operations,
        "quantization": quant,
        "activation_type": activation_type,
    }
    return full, candidates, summary


def baseline_estimate(
    profile: dict[str, Any],
    workload: dict[str, Any],
    host: dict[str, Any],
    full_host_work: Work,
    summary: dict[str, Any],
) -> dict[str, Any]:
    if workload["kind"] != "chain":
        return estimate_device(
            host, full_host_work, str(workload["kind"])
        )
    quant = summary["quantization"]
    activation_type = summary["activation_type"]
    components = []
    for index, op in enumerate(summary["operations"]):
        components.append(
            estimate_device(
                host,
                matmul_work(
                    profile,
                    label=f"operation {index}",
                    m=op["m"],
                    n=op["n"],
                    k=op["k"],
                    quant=quant,
                    activation_type=activation_type,
                ),
                "matmul",
            )
        )
    return {
        "median_ms": sum(item["median_ms"] for item in components),
        "p90_ms": sum(item["p90_ms"] for item in components),
        "components": components,
    }


def finish_plan(
    profile: dict[str, Any],
    workload: dict[str, Any],
    full_host_work: Work,
    candidates: list[dict[str, Any]],
    summary: dict[str, Any],
    top: int,
) -> dict[str, Any]:
    host_name, host, _ = get_devices(profile, workload)
    baseline_device_estimate = baseline_estimate(
        profile, workload, host, full_host_work, summary
    )
    baseline = {
        "strategy": "all_desktop",
        "host_backend": host_name,
        "median_ms": baseline_device_estimate["median_ms"],
        "p90_ms": baseline_device_estimate["p90_ms"],
        "estimate": baseline_device_estimate,
    }
    minimum_speedup = float(
        workload.get("minimum_speedup_percent", 5.0)
    )
    for candidate in candidates:
        candidate["speedup_percent"] = (
            100.0
            * (baseline["median_ms"] - candidate["median_ms"])
            / baseline["median_ms"]
        )
        candidate["eligible"] = not candidate["blockers"]
        candidate["passes_policy"] = (
            candidate["eligible"]
            and candidate["speedup_percent"] >= minimum_speedup
            and candidate["p90_ms"] <= baseline["p90_ms"]
        )
        if candidate["speedup_percent"] < minimum_speedup:
            candidate.setdefault("policy_failures", []).append(
                f"median speedup is below {minimum_speedup:.1f}%"
            )
        if candidate["p90_ms"] > baseline["p90_ms"]:
            candidate.setdefault("policy_failures", []).append(
                "p90 latency regresses"
            )

    viable = sorted(
        (
            candidate
            for candidate in candidates
            if candidate["eligible"]
        ),
        key=lambda candidate: (
            candidate["median_ms"],
            candidate["p90_ms"],
            candidate["phone_backend"],
        ),
    )
    blocked = sorted(
        (
            candidate
            for candidate in candidates
            if not candidate["eligible"]
        ),
        key=lambda candidate: (
            candidate["median_ms"],
            candidate["p90_ms"],
            candidate["phone_backend"],
        ),
    )
    passing = [
        candidate
        for candidate in viable
        if candidate["passes_policy"]
    ]
    if passing:
        best = passing[0]
        recommendation = {
            "decision": "offload",
            "strategy": best["strategy"],
            "phone_backend": best["phone_backend"],
            "split": best["split"],
            "median_ms": best["median_ms"],
            "p90_ms": best["p90_ms"],
            "speedup_percent": best["speedup_percent"],
            "status": (
                "model_prediction_requires_exact_shape_validation"
            ),
            "phone_resident_bytes": best["phone_resident_bytes"],
            "phone_working_weight_bytes": (
                best["phone_working_weight_bytes"]
            ),
        }
        if "remote_branch" in best:
            recommendation["branches"] = {
                "host_median_ms": best["host_branch"]["median_ms"],
                "remote_median_ms": best["remote_branch"][
                    "median_ms"
                ],
                "merge_median_ms": best["merge"]["median_ms"],
            }
            recommendation["host_work"] = best["host_branch"][
                "work"
            ]
            recommendation["phone_work"] = best["remote_branch"][
                "phone_compute"
            ]["work"]
            recommendation["transfer"] = best["remote_branch"][
                "transport"
            ]
        else:
            islands = best["phone_islands"]
            recommendation["transfer"] = {
                "island_count": len(islands),
                "input_bytes": sum(
                    island["transport"]["input_bytes"]
                    for island in islands
                ),
                "output_bytes": sum(
                    island["transport"]["output_bytes"]
                    for island in islands
                ),
                "protocols": [
                    island["transport"]["protocol"]
                    for island in islands
                ],
            }
    else:
        recommendation = {
            "decision": "all_desktop",
            "strategy": "all_desktop",
            "median_ms": baseline["median_ms"],
            "p90_ms": baseline["p90_ms"],
            "reason": (
                "no eligible offload candidate passed both median "
                "and p90 policy"
            ),
        }
        if viable:
            best = viable[0]
            recommendation["best_eligible_candidate"] = {
                "strategy": best["strategy"],
                "phone_backend": best["phone_backend"],
                "split": best["split"],
                "median_ms": best["median_ms"],
                "p90_ms": best["p90_ms"],
                "speedup_percent": best["speedup_percent"],
                "policy_failures": best.get(
                    "policy_failures", []
                ),
            }

    def limit(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return values if top == 0 else values[:top]

    return {
        "schema": OUTPUT_SCHEMA,
        "workload_name": str(workload.get("name", "unnamed")),
        "workload_kind": workload["kind"],
        "profile": str(profile.get("name", "unnamed")),
        "summary": summary,
        "assumptions": [
            (
                "phone kernels and selected weights are prepared "
                "before timed inference"
            ),
            (
                "one request is optimized for latency; host and "
                "phone split branches overlap"
            ),
            (
                "roofline values are screening estimates, not "
                "exact-shape measurements"
            ),
            (
                "recommendations require the configured median gain "
                "and no modeled p90 regression"
            ),
        ],
        "baseline": baseline,
        "recommendation": recommendation,
        "candidate_count": {
            "total": len(candidates),
            "eligible": len(viable),
            "blocked": len(blocked),
            "passing_policy": len(passing),
        },
        "candidates": limit(viable),
        "diagnostic_blocked_candidates": limit(blocked),
    }


def solve_workload(
    profile: dict[str, Any],
    workload: dict[str, Any],
    *,
    allow_unqualified: bool = False,
    top: int = 10,
) -> dict[str, Any]:
    if profile.get("schema") != PROFILE_SCHEMA:
        raise SolverError(f"profile schema must be {PROFILE_SCHEMA}")
    if workload.get("schema") != WORKLOAD_SCHEMA:
        raise SolverError(f"workload schema must be {WORKLOAD_SCHEMA}")
    kind = workload.get("kind")
    solvers = {
        "matmul": solve_matmul,
        "chain": solve_chain,
        "ffn": solve_ffn,
        "moe": solve_moe,
    }
    if kind not in solvers:
        raise SolverError("kind must be matmul, chain, ffn, or moe")
    if top < 0:
        raise SolverError("top must be nonnegative")
    full, candidates, summary = solvers[kind](
        profile, workload, allow_unqualified
    )
    return finish_plan(
        profile, workload, full, candidates, summary, top
    )


def rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return [rounded(item) for item in value]
    if isinstance(value, dict):
        return {
            key: rounded(item) for key, item in value.items()
        }
    return value


def format_text(result: dict[str, Any]) -> str:
    baseline = result["baseline"]
    recommendation = result["recommendation"]
    lines = [
        (
            f"workload: {result['workload_name']} "
            f"({result['workload_kind']})"
        ),
        f"profile: {result['profile']}",
        (
            f"baseline: {baseline['strategy']} on "
            f"{baseline['host_backend']}, "
            f"median {baseline['median_ms']:.3f} ms, "
            f"p90 {baseline['p90_ms']:.3f} ms"
        ),
    ]
    if recommendation["decision"] == "offload":
        working_mib = (
            recommendation["phone_working_weight_bytes"]
            / (1024.0 * 1024.0)
        )
        resident_mib = (
            recommendation["phone_resident_bytes"]
            / (1024.0 * 1024.0)
        )
        lines.extend(
            [
                (
                    "decision: offload with "
                    f"{recommendation['phone_backend']}"
                ),
                f"strategy: {recommendation['strategy']}",
                (
                    "split: "
                    + json.dumps(
                        rounded(recommendation["split"]),
                        sort_keys=True,
                    )
                ),
                (
                    "estimated latency: median "
                    f"{recommendation['median_ms']:.3f} ms, "
                    f"p90 {recommendation['p90_ms']:.3f} ms"
                ),
                (
                    "estimated speedup: "
                    f"{recommendation['speedup_percent']:.1f}%"
                ),
                (
                    "phone weights: "
                    f"{working_mib:.1f} MiB working, "
                    f"{resident_mib:.1f} MiB resident"
                ),
                f"status: {recommendation['status']}",
            ]
        )
        transfer = recommendation["transfer"]
        lines.append(
            "wire: "
            f"{transfer['input_bytes'] / 1024.0:.1f} KiB input, "
            f"{transfer['output_bytes'] / 1024.0:.1f} KiB output"
        )
    else:
        lines.extend(
            [
                "decision: keep all work on desktop",
                f"reason: {recommendation['reason']}",
            ]
        )
        if "best_eligible_candidate" in recommendation:
            best = recommendation["best_eligible_candidate"]
            lines.append(
                "best eligible alternative: "
                f"{best['phone_backend']} {best['strategy']}, "
                f"median {best['median_ms']:.3f} ms, "
                f"p90 {best['p90_ms']:.3f} ms - "
                + "; ".join(best["policy_failures"])
            )
    count = result["candidate_count"]
    lines.append(
        f"candidates: {count['passing_policy']} pass, "
        f"{count['eligible']} eligible, "
        f"{count['blocked']} blocked"
    )
    if result["diagnostic_blocked_candidates"]:
        first = result["diagnostic_blocked_candidates"][0]
        lines.append(
            f"fastest blocked estimate: {first['phone_backend']} "
            f"{first['strategy']} - "
            + "; ".join(first["blockers"])
        )
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    default_profile = Path(__file__).with_name("current_profile.json")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workload", type=Path, help="workload JSON")
    parser.add_argument(
        "--profile", type=Path, default=default_profile
    )
    parser.add_argument(
        "--format", choices=("json", "text"), default="text"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="candidates per eligible/blocked list; 0 keeps all",
    )
    parser.add_argument(
        "--allow-unqualified",
        action="store_true",
        help="permit provisional backend paths to be recommended",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        result = solve_workload(
            load_json(args.profile),
            load_json(args.workload),
            allow_unqualified=args.allow_unqualified,
            top=args.top,
        )
    except SolverError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.format == "json":
        rendered = (
            json.dumps(rounded(result), indent=2, sort_keys=True)
            + "\n"
        )
    else:
        rendered = format_text(result) + "\n"
    if args.output:
        try:
            args.output.write_text(rendered, encoding="ascii")
        except OSError as exc:
            print(
                f"error: cannot write {args.output}: {exc}",
                file=sys.stderr,
            )
            return 2
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
