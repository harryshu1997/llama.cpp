#!/usr/bin/env python3
"""Create one strict real-device BurstGPT loop report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
import sys
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    SOURCE_TRACE_SCHEMA as SOURCE_SCHEMA,
    sha256_file,
)


SCHEMA = "s42-physical-iteration-v2"
ENERGY_SCHEMA = "s42-accounted-device-energy-receipt-v1"
TRACE_SHA256 = "sha256:ccde6e3e53dee4547e4eb80f9f090032afb04b1d3fec3fd07f0f961bed60cf8f"
EXPECTED_REQUESTS = 74
EXPECTED_OUTPUT_TOKENS = 2175
REQUIRED_ENERGY_KINDS = {
    "server_cpu_package",
    "server_gpu_board",
    "phone_system",
}


class IterationError(ValueError):
    pass


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IterationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="ascii") as source:
            value = json.load(source, object_pairs_hook=_no_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IterationError(f"cannot read {path}: {exc}") from exc
    if type(value) is not dict:
        raise IterationError(f"{path} must contain an object")
    return value


def load_trace(path: Path) -> dict[str, dict[str, Any]]:
    if sha256_file(path) != TRACE_SHA256:
        raise IterationError("trace hash differs from the frozen loop")
    rows: dict[str, dict[str, Any]] = {}
    try:
        with path.open("r", encoding="ascii") as source:
            for line_number, line in enumerate(source, 1):
                row = json.loads(line, object_pairs_hook=_no_duplicates)
                if type(row) is not dict or row.get("schema") != SOURCE_SCHEMA:
                    raise IterationError(f"trace line {line_number}: schema mismatch")
                event_id = row.get("event_id")
                if type(event_id) is not str or event_id in rows:
                    raise IterationError(f"trace line {line_number}: invalid identity")
                rows[event_id] = row
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IterationError(f"cannot parse trace: {exc}") from exc
    if len(rows) != EXPECTED_REQUESTS:
        raise IterationError("trace request count differs from the frozen loop")
    return rows


def result_sha256(path: Path) -> str:
    return sha256_file(path)


def validate_result(
    path: Path,
    value: dict[str, Any],
    expected_mode: str,
    trace: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if value.get("status") != "PASS" or value.get("mode") != expected_mode:
        raise IterationError(f"{path}: result status or mode mismatch")
    preflight = value.get("preflight")
    metrics = value.get("metrics")
    rows = value.get("request_results")
    if (
        type(preflight) is not dict
        or preflight.get("requests_sha256") != TRACE_SHA256.removeprefix("sha256:")
        or type(metrics) is not dict
        or type(rows) is not list
        or len(rows) != EXPECTED_REQUESTS
    ):
        raise IterationError(f"{path}: result does not bind the frozen work")
    seen: set[str] = set()
    output_tokens = 0
    cold_ids = []
    token_map: dict[str, tuple[int, ...]] = {}
    for row in rows:
        if type(row) is not dict:
            raise IterationError(f"{path}: request row is invalid")
        event_id = row.get("event_id")
        tokens = row.get("tokens")
        if (
            type(event_id) is not str
            or event_id in seen
            or event_id not in trace
            or type(tokens) is not list
            or any(type(token) is not int for token in tokens)
        ):
            raise IterationError(f"{path}: request work differs from trace")
        seen.add(event_id)
        expected_tokens = trace[event_id]["output_tokens"]
        if len(tokens) != expected_tokens:
            raise IterationError(f"{path}: output work differs from trace")
        output_tokens += len(tokens)
        token_map[event_id] = tuple(tokens)
        if row.get("role") == "cold":
            cold_ids.append(event_id)
    if output_tokens != EXPECTED_OUTPUT_TOKENS or len(cold_ids) != 17:
        raise IterationError(f"{path}: completed work differs from the frozen loop")
    duration_s = metrics.get("duration_s")
    throughput = metrics.get("throughput_tokens_s")
    paid_start_ns = value.get("paid_start_ns")
    paid_end_ns = value.get("paid_end_ns")
    if (
        metrics.get("completed") != EXPECTED_REQUESTS
        or type(metrics.get("slo_met")) is not int
        or not isinstance(duration_s, (int, float))
        or isinstance(duration_s, bool)
        or duration_s <= 0
        or not isinstance(throughput, (int, float))
        or isinstance(throughput, bool)
        or throughput <= 0
        or type(paid_start_ns) is not int
        or type(paid_end_ns) is not int
        or paid_end_ns <= paid_start_ns
        or not math.isclose(
            (paid_end_ns - paid_start_ns) / 1e9,
            float(duration_s),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise IterationError(f"{path}: aggregate metric is invalid")
    samples = value.get("resources", {}).get("samples", {})
    gpu_power_mw = samples.get("gpu_power_mean_mw")
    return {
        "path": str(path),
        "sha256": result_sha256(path),
        "makespan_s": float(duration_s),
        "throughput_tokens_s": float(throughput),
        "completed_requests": metrics["completed"],
        "completed_output_tokens": output_tokens,
        "slo_met": metrics["slo_met"],
        "cold_event_ids": tuple(cold_ids),
        "tokens": token_map,
        "gpu_power_mean_mw": gpu_power_mw,
        "paid_start_ns": paid_start_ns,
        "paid_end_ns": paid_end_ns,
        "raw": value,
    }


def average(values: Iterable[float]) -> float:
    rows = list(values)
    if not rows:
        raise IterationError("cannot average an empty vector")
    return statistics.fmean(rows)


def range_json(values: Iterable[float]) -> dict[str, float]:
    rows = list(values)
    return {"min": min(rows), "max": max(rows)}


def _finite_positive(name: str, value: object) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise IterationError(f"{name} must be finite and positive")
    return float(value)


def _finite_nonnegative(name: str, value: object) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise IterationError(f"{name} must be finite and nonnegative")
    return float(value)


def energy_component_signature(
    receipt: dict[str, Any],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted((component["kind"], component["device_id"])
               for component in receipt["components"])
    )


def load_energy_receipt(
    path: Path, result: dict[str, Any]
) -> dict[str, Any]:
    receipt = load_json(path)
    if (
        receipt.get("schema") != ENERGY_SCHEMA
        or receipt.get("valid") is not True
        or receipt.get("boundary_scope") != "accounted_devices"
        or receipt.get("result_sha256") != result["sha256"]
        or type(receipt.get("boundary_id")) is not str
        or not receipt["boundary_id"]
        or type(receipt.get("paid_start_ns")) is not int
        or type(receipt.get("paid_end_ns")) is not int
        or receipt["paid_end_ns"] <= receipt["paid_start_ns"]
        or receipt.get("paid_start_ns") != result.get("paid_start_ns")
        or receipt.get("paid_end_ns") != result.get("paid_end_ns")
        or type(receipt.get("components")) is not list
        or not receipt["components"]
        or type(receipt.get("excluded_components")) is not list
        or any(type(item) is not str or not item
               for item in receipt["excluded_components"])
    ):
        raise IterationError(f"{path}: invalid accounted-device energy receipt")
    interval_s = (receipt["paid_end_ns"] - receipt["paid_start_ns"]) / 1e9
    seen: set[tuple[str, str]] = set()
    kinds: set[str] = set()
    component_energy = 0.0
    for index, component in enumerate(receipt["components"]):
        if type(component) is not dict:
            raise IterationError(f"{path}: component {index} is invalid")
        kind = component.get("kind")
        device_id = component.get("device_id")
        source = component.get("source")
        if (
            type(kind) is not str
            or not kind
            or type(device_id) is not str
            or not device_id
            or type(source) is not str
            or not source
            or type(component.get("sample_count")) is not int
            or component["sample_count"] < 2
        ):
            raise IterationError(f"{path}: component {index} metadata is invalid")
        identity = (kind, device_id)
        if identity in seen:
            raise IterationError(f"{path}: duplicate energy component")
        seen.add(identity)
        kinds.add(kind)
        power_w = _finite_positive(
            f"{path}: component {index} average_power_w",
            component.get("average_power_w"),
        )
        duration_s = _finite_positive(
            f"{path}: component {index} duration_s",
            component.get("duration_s"),
        )
        energy_j = _finite_positive(
            f"{path}: component {index} energy_j", component.get("energy_j")
        )
        if not math.isclose(duration_s, interval_s, rel_tol=0.0, abs_tol=1e-6):
            raise IterationError(f"{path}: component duration differs from paid interval")
        if not math.isclose(
            energy_j, power_w * duration_s, rel_tol=0.005, abs_tol=0.05
        ):
            raise IterationError(f"{path}: component energy is not power times time")
        component_energy += energy_j
    if not REQUIRED_ENERGY_KINDS.issubset(kinds):
        raise IterationError(f"{path}: required device-energy component is missing")
    total_energy = _finite_positive(f"{path}: energy_j", receipt.get("energy_j"))
    if not math.isclose(
        total_energy, component_energy, rel_tol=0.005, abs_tol=0.05
    ):
        raise IterationError(f"{path}: component energy does not reconcile")
    return receipt


def aggregate_energy(
    paths: list[Path], results: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not paths:
        return None, []
    if len(paths) != len(results):
        raise IterationError("energy receipt count differs from result count")
    receipts = [
        load_energy_receipt(path, result)
        for path, result in zip(paths, results)
    ]
    boundary_ids = {receipt["boundary_id"] for receipt in receipts}
    if len(boundary_ids) != 1:
        raise IterationError("energy receipts use different boundaries")
    signatures = {energy_component_signature(receipt) for receipt in receipts}
    if len(signatures) != 1:
        raise IterationError("energy receipts use different device sets")
    excluded = {
        tuple(sorted(receipt["excluded_components"])) for receipt in receipts
    }
    if len(excluded) != 1:
        raise IterationError("energy receipts use different coverage exclusions")
    energies = [float(receipt["energy_j"]) for receipt in receipts]
    return {
        "boundary_id": receipts[0]["boundary_id"],
        "boundary_scope": "accounted_devices",
        "component_set": [
            {"kind": kind, "device_id": device_id}
            for kind, device_id in next(iter(signatures))
        ],
        "excluded_components": list(next(iter(excluded))),
        "average_j": average(energies),
        "range_j": range_json(energies),
        "average_j_per_output_token": average(energies) / EXPECTED_OUTPUT_TOKENS,
    }, receipts


def selected_prefill_columns(policy: list[dict[str, Any]], tokens: int) -> int:
    previous = 0
    for point in policy:
        if (
            type(point) is not dict
            or type(point.get("max_tokens")) is not int
            or point["max_tokens"] <= previous
            or type(point.get("columns")) is not int
            or point["columns"] <= 0
        ):
            raise IterationError("prefill policy is invalid")
        previous = point["max_tokens"]
        if tokens <= point["max_tokens"]:
            return point["columns"]
    raise IterationError("prefill shape exceeds the physical policy")


def phone_work(
    treatment: dict[str, Any], trace: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    raw = treatment["raw"]
    cold_ids = treatment["cold_event_ids"]
    offload = raw.get("preflight", {}).get("offload")
    resources = raw.get("resources", {})
    ffn = resources.get("cold_log_ffn")
    dmabuf = resources.get("dmabuf")
    if (
        type(offload) is not dict
        or type(ffn) is not dict
        or type(dmabuf) is not dict
        or ffn.get("status") != "FFN_OVERLAP_OK"
        or dmabuf.get("status") != "ok"
        or ffn.get("io") != "f16"
    ):
        raise IterationError("treatment lacks valid phone work counters")
    layer_count = ffn.get("layer_count")
    n_ff = ffn.get("n_ff")
    decode_columns = offload.get("decode_columns")
    policy = offload.get("prefill_policy")
    shapes = ffn.get("shapes")
    if (
        type(layer_count) is not int
        or layer_count <= 0
        or type(n_ff) is not int
        or n_ff <= 0
        or type(decode_columns) is not int
        or not 0 < decode_columns <= n_ff
        or type(policy) is not list
        or type(shapes) is not list
    ):
        raise IterationError("treatment phone geometry is invalid")
    prompt_rows = sum(trace[event_id]["input_tokens"] for event_id in cold_ids)
    decode_rows = sum(trace[event_id]["output_tokens"] - 1 for event_id in cold_ids)
    weighted_columns = sum(
        trace[event_id]["input_tokens"]
        * selected_prefill_columns(policy, trace[event_id]["input_tokens"])
        for event_id in cold_ids
    ) + decode_rows * decode_columns
    total_rows = prompt_rows + decode_rows
    paid_prefill_calls = len(cold_ids) * layer_count
    paid_decode_calls = decode_rows * layer_count
    paid_calls = paid_prefill_calls + paid_decode_calls
    raw_row_calls = 0
    for shape in shapes:
        if (
            type(shape) is not dict
            or type(shape.get("tokens")) is not int
            or type(shape.get("calls")) is not int
        ):
            raise IterationError("phone shape counter is invalid")
        raw_row_calls += shape["tokens"] * shape["calls"]
    upload_bytes = dmabuf.get("upload_bytes")
    download_bytes = dmabuf.get("download_bytes")
    if (
        type(upload_bytes) is not int
        or upload_bytes <= 0
        or download_bytes != upload_bytes
        or raw_row_calls <= 0
        or upload_bytes % raw_row_calls != 0
    ):
        raise IterationError("phone transfer counters do not reconcile")
    bytes_per_row_call = upload_bytes // raw_row_calls
    if bytes_per_row_call % 2:
        raise IterationError("f16 row payload is not aligned")
    hidden = bytes_per_row_call // 2
    paid_bytes = total_rows * layer_count * bytes_per_row_call
    phone_macs = 3 * hidden * layer_count * weighted_columns
    eligible_macs = 3 * hidden * layer_count * total_rows * n_ff
    host_p50 = _finite_positive(
        "host_branch_p50_ms", ffn.get("host_branch_p50_ms")
    )
    phone_rpc_p50 = _finite_positive("rpc_p50_ms", ffn.get("rpc_p50_ms"))
    phone_compute_p50 = _finite_positive(
        "phone_compute_p50_ms", ffn.get("phone_compute_p50_ms")
    )
    wait_p50 = _finite_nonnegative("wait_p50_ms", ffn.get("wait_p50_ms"))
    island_p50 = _finite_positive("overlap_p50_ms", ffn.get("overlap_p50_ms"))
    if (
        wait_p50 > min(phone_rpc_p50, island_p50)
        or island_p50 < max(host_p50, phone_rpc_p50)
    ):
        raise IterationError("treatment p50 overlap counters are inconsistent")
    sum_names = (
        "host_branch_sum_ns",
        "rpc_sum_ns",
        "phone_compute_sum_ns",
        "wait_sum_ns",
        "overlap_sum_ns",
    )
    overlap_count = ffn.get("overlap_sample_count")
    raw_sums = [ffn.get(name) for name in sum_names]
    supplied_mean_record = overlap_count is not None or any(
        value is not None for value in raw_sums
    )
    if supplied_mean_record and (
        type(overlap_count) is not int
        or overlap_count <= 0
        or not all(type(value) is int and value >= 0 for value in raw_sums)
    ):
        raise IterationError("treatment overlap sum/count record is invalid")
    overlap_mean: dict[str, Any]
    if supplied_mean_record:
        assert type(overlap_count) is int
        mean_host, mean_rpc, mean_compute, mean_wait, mean_island = (
            float(value) / overlap_count / 1e6 for value in raw_sums
        )
        if min(mean_host, mean_rpc, mean_compute, mean_island) <= 0:
            raise IterationError("treatment overlap mean must be positive")
        if (
            mean_wait > min(mean_rpc, mean_island)
            or mean_island < max(mean_host, mean_rpc)
        ):
            raise IterationError("treatment overlap means are inconsistent")
        overlap_mean = {
            "status": "MEASURED_ARITHMETIC_MEAN",
            "sample_count": overlap_count,
            "host_branch_ms": mean_host,
            "phone_rpc_ms": mean_rpc,
            "phone_compute_ms": mean_compute,
            "join_wait_ms": mean_wait,
            "island_ms": mean_island,
            "branch_imbalance_fraction": abs(mean_rpc - mean_host)
            / max(mean_rpc, mean_host),
            "exposed_join_wait_fraction": mean_wait / mean_island,
            "phone_work_hidden_fraction": max(0.0, 1.0 - mean_wait / mean_rpc),
            "phone_finishes_after_host_ms": mean_rpc - mean_host,
        }
    else:
        overlap_mean = {
            "status": "MISSING_SUM_COUNT_COUNTERS",
            "sample_count": None,
            "host_branch_ms": None,
            "phone_rpc_ms": None,
            "phone_compute_ms": None,
            "join_wait_ms": None,
            "island_ms": None,
            "branch_imbalance_fraction": None,
            "exposed_join_wait_fraction": None,
            "phone_work_hidden_fraction": None,
            "phone_finishes_after_host_ms": None,
        }
    return {
        "phone_requests": len(cold_ids),
        "all_trace_requests": EXPECTED_REQUESTS,
        "eligible_requests": len(cold_ids),
        "request_fraction_all_trace": len(cold_ids) / EXPECTED_REQUESTS,
        "request_fraction_eligible": 1.0,
        "paid_phone_calls": paid_calls,
        "paid_prefill_calls": paid_prefill_calls,
        "paid_decode_calls": paid_decode_calls,
        "warmup_inclusive_counter_calls": ffn.get("calls"),
        "phone_macs": phone_macs,
        "eligible_cold_ffn_macs": eligible_macs,
        "phone_fraction_eligible_cold_ffn_macs": phone_macs / eligible_macs,
        "phone_fraction_all_model_macs": None,
        "phone_fraction_all_model_macs_status": "UNCLAIMED_FULL_GRAPH_NOT_COUNTED",
        "paid_upload_bytes": paid_bytes,
        "paid_download_bytes": paid_bytes,
        "warmup_inclusive_upload_bytes": upload_bytes,
        "warmup_inclusive_download_bytes": download_bytes,
        "resident_max_columns": offload.get("max_columns"),
        "full_ffn_columns": n_ff,
        "decode_phone_columns": decode_columns,
        "prefill_policy": policy,
        "latency_ms": {
            "phone_compute_p50": phone_compute_p50,
            "phone_rpc_p50": phone_rpc_p50,
            "host_branch_p50": host_p50,
            "join_wait_p50": wait_p50,
            "overlapped_island_p50": island_p50,
        },
        "overlap_p50_diagnostic": {
            "status": "DIAGNOSTIC_P50_NOT_ARITHMETIC_MEAN",
            "branch_imbalance_fraction": abs(phone_rpc_p50 - host_p50)
            / max(phone_rpc_p50, host_p50),
            "exposed_join_wait_fraction": wait_p50 / island_p50,
            "phone_work_hidden_fraction": max(
                0.0, 1.0 - wait_p50 / phone_rpc_p50
            ),
            "phone_finishes_after_host_ms": phone_rpc_p50 - host_p50,
        },
        "overlap_mean": overlap_mean,
    }


def quality_comparison(
    controls: list[dict[str, Any]], treatments: list[dict[str, Any]]
) -> dict[str, Any]:
    reference = controls[0]["tokens"]
    cold_ids = controls[0]["cold_event_ids"]
    exact = 0
    matched_positions = 0
    positions = 0
    for event_id in cold_ids:
        control_tokens = reference[event_id]
        treatment_tokens = treatments[0]["tokens"][event_id]
        exact += int(control_tokens == treatment_tokens)
        positions += min(len(control_tokens), len(treatment_tokens))
        matched_positions += sum(
            left == right for left, right in zip(control_tokens, treatment_tokens)
        )
    return {
        "cold_requests_exact": exact,
        "cold_requests_total": len(cold_ids),
        "cold_token_positions_equal": matched_positions,
        "cold_token_positions_total": positions,
        "exact_pass": exact == len(cold_ids),
    }


def gpu_diagnostic(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    energies = []
    powers = []
    for result in results:
        power_mw = result["gpu_power_mean_mw"]
        if not isinstance(power_mw, (int, float)) or isinstance(power_mw, bool):
            return None
        powers.append(float(power_mw) / 1000.0)
        energies.append(float(power_mw) * result["makespan_s"] / 1000.0)
    return {
        "average_power_w": average(powers),
        "mean_power_times_makespan_j": average(energies),
        "status": "DIAGNOSTIC_INCOMPLETE_DEVICE_SET",
    }


def build_iteration(
    *,
    iteration_id: str,
    declared_change: str,
    trace_path: Path,
    control_paths: list[Path],
    treatment_paths: list[Path],
    control_evidence_paths: list[str],
    treatment_evidence_paths: list[str],
    control_energy_paths: list[Path],
    treatment_energy_paths: list[Path],
    quality_requirement: str,
    phone_serial: str,
) -> dict[str, Any]:
    if not iteration_id or not declared_change:
        raise IterationError("iteration id and declared change are required")
    if quality_requirement not in {"exact", "approximate"}:
        raise IterationError("quality requirement is invalid")
    if not control_paths or len(control_paths) != len(treatment_paths):
        raise IterationError("control and treatment repetitions must be paired")
    trace = load_trace(trace_path)
    controls = [
        validate_result(path, load_json(path), "cpu", trace)
        for path in control_paths
    ]
    treatments = [
        validate_result(path, load_json(path), "op15", trace)
        for path in treatment_paths
    ]
    if control_evidence_paths:
        if len(control_evidence_paths) != len(controls):
            raise IterationError("control evidence locator count differs")
        for result, locator in zip(controls, control_evidence_paths):
            if not locator:
                raise IterationError("control evidence locator is empty")
            result["path"] = locator
    if treatment_evidence_paths:
        if len(treatment_evidence_paths) != len(treatments):
            raise IterationError("treatment evidence locator count differs")
        for result, locator in zip(treatments, treatment_evidence_paths):
            if not locator:
                raise IterationError("treatment evidence locator is empty")
            result["path"] = locator
    control_energy, control_receipts = aggregate_energy(
        control_energy_paths, controls
    )
    treatment_energy, treatment_receipts = aggregate_energy(
        treatment_energy_paths, treatments
    )
    if (control_energy is None) != (treatment_energy is None):
        raise IterationError("both arms require accounted-device energy receipts")
    if (
        control_energy is not None
        and treatment_energy is not None
        and control_energy["boundary_id"] != treatment_energy["boundary_id"]
    ):
        raise IterationError("control and treatment energy boundaries differ")
    if (
        control_energy is not None
        and treatment_energy is not None
        and (
            control_energy["component_set"] != treatment_energy["component_set"]
            or control_energy["excluded_components"]
            != treatment_energy["excluded_components"]
        )
    ):
        raise IterationError("control and treatment device-energy coverage differs")
    control_makespans = [result["makespan_s"] for result in controls]
    treatment_makespans = [result["makespan_s"] for result in treatments]
    control_average = average(control_makespans)
    treatment_average = average(treatment_makespans)
    latency_change = 100.0 * (treatment_average / control_average - 1.0)
    control_slo = average(float(result["slo_met"]) for result in controls)
    treatment_slo = average(float(result["slo_met"]) for result in treatments)
    work = phone_work(treatments[0], trace)
    for treatment in treatments[1:]:
        if phone_work(treatment, trace) != work:
            raise IterationError("treatment phone work differs across repetitions")
    quality = quality_comparison(controls, treatments)
    energy_change: float | None = None
    if control_energy is not None and treatment_energy is not None:
        energy_change = 100.0 * (
            treatment_energy["average_j"] / control_energy["average_j"] - 1.0
        )
    complete_gate = all(
        result["completed_requests"] == EXPECTED_REQUESTS
        and result["completed_output_tokens"] == EXPECTED_OUTPUT_TOKENS
        for result in [*controls, *treatments]
    )
    quality_gate = quality["exact_pass"] if quality_requirement == "exact" else True
    overlap_mean = work["overlap_mean"]
    overlap_measured = overlap_mean["status"] == "MEASURED_ARITHMETIC_MEAN"
    overlap_gate = (
        overlap_measured
        and overlap_mean["exposed_join_wait_fraction"] <= 0.05
    )
    gates = {
        "work_complete": complete_gate,
        "accounted_device_energy_measured": control_energy is not None,
        "accounted_energy_saving_at_least_10_percent": (
            energy_change is not None and energy_change <= -10.0
        ),
        "no_makespan_regression": treatment_average <= control_average,
        "mean_overlap_measured": overlap_measured,
        "mean_exposed_join_wait_at_most_5_percent": overlap_gate,
        "no_slo_loss": treatment_slo >= control_slo,
        "quality": quality_gate,
        "minimum_three_pairs": len(controls) >= 3,
    }
    if not gates["accounted_device_energy_measured"]:
        verdict = "INCOMPLETE_MISSING_ACCOUNTED_DEVICE_ENERGY"
        blocker = "synchronized CPU, GPU, and phone energy receipts for both arms"
    elif not gates["minimum_three_pairs"]:
        verdict = "INCOMPLETE_FEWER_THAN_THREE_PAIRS"
        blocker = "at least three alternating physical pairs"
    elif not gates["mean_overlap_measured"]:
        verdict = "INCOMPLETE_MISSING_MEAN_OVERLAP_COUNTERS"
        blocker = "arithmetic-mean host, phone, join-wait, and island counters"
    elif all(gates.values()):
        verdict = "PASS"
        blocker = "none"
    else:
        verdict = "FAIL"
        blocker = next(name for name, passed in gates.items() if not passed)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "iteration_id": iteration_id,
        "declared_one_change": declared_change,
        "target": {
            "accounted_fleet_energy_change_pct_max": -10.0,
            "makespan_change_pct_max": 0.0,
            "mean_exposed_join_wait_fraction_max": 0.05,
            "slo_loss_max": 0.0,
            "completed_requests": EXPECTED_REQUESTS,
            "completed_output_tokens": EXPECTED_OUTPUT_TOKENS,
            "quality_requirement": quality_requirement,
            "minimum_alternating_pairs": 3,
        },
        "physical_binding": {
            "gpu_uuid": "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08",
            "phone_serial": phone_serial,
            "trace_path": str(trace_path),
            "trace_sha256": TRACE_SHA256,
        },
        "repetitions": len(controls),
        "control_results": [
            {key: result[key] for key in ("path", "sha256", "makespan_s", "throughput_tokens_s", "completed_requests", "completed_output_tokens", "slo_met")}
            for result in controls
        ],
        "treatment_results": [
            {key: result[key] for key in ("path", "sha256", "makespan_s", "throughput_tokens_s", "completed_requests", "completed_output_tokens", "slo_met")}
            for result in treatments
        ],
        "real_device_headline": {
            "latency": {
                "control_average_s": control_average,
                "treatment_average_s": treatment_average,
                "change_pct": latency_change,
                "control_range_s": range_json(control_makespans),
                "treatment_range_s": range_json(treatment_makespans),
            },
            "accounted_fleet_energy": {
                "control": control_energy,
                "treatment": treatment_energy,
                "change_pct": energy_change,
                "status": (
                    "MEASURED"
                    if energy_change is not None
                    else "MISSING_NO_ACCOUNTED_DEVICE_ENERGY_CLAIM"
                ),
            },
            "completed_work": {
                "control_requests": EXPECTED_REQUESTS,
                "treatment_requests": EXPECTED_REQUESTS,
                "control_output_tokens": EXPECTED_OUTPUT_TOKENS,
                "treatment_output_tokens": EXPECTED_OUTPUT_TOKENS,
            },
            "slo_met_average": {
                "control": control_slo,
                "treatment": treatment_slo,
            },
            "phone_work": work,
            "quality": quality,
        },
        "diagnostics_not_primary_energy": {
            "control_gpu_board": gpu_diagnostic(controls),
            "treatment_gpu_board": gpu_diagnostic(treatments),
            "predictions_in_headline": False,
        },
        "energy_receipt_count": {
            "control": len(control_receipts),
            "treatment": len(treatment_receipts),
        },
        "gates": gates,
        "verdict": verdict,
        "blocker": blocker,
        "next_one_change": (
            "add synchronized CPU, GPU, and phone power plus mean overlap counters without changing execution"
            if verdict.startswith("INCOMPLETE_MISSING_ACCOUNTED")
            else "declare after reviewing this physical iteration"
        ),
    }
    result["iteration_hash"] = "sha256:" + hashlib.sha256(
        canonical_bytes(result)
    ).hexdigest()
    return result


def validate_iteration_record(value: object) -> dict[str, Any]:
    if type(value) is not dict or value.get("schema") != SCHEMA:
        raise IterationError("physical iteration schema mismatch")
    supplied = value.get("iteration_hash")
    if type(supplied) is not str or not supplied.startswith("sha256:"):
        raise IterationError("physical iteration hash is missing")
    unhashed = dict(value)
    del unhashed["iteration_hash"]
    expected = "sha256:" + hashlib.sha256(canonical_bytes(unhashed)).hexdigest()
    if supplied != expected:
        raise IterationError("physical iteration hash mismatch")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration-id", required=True)
    parser.add_argument("--declared-change", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--control-result", type=Path, action="append", required=True)
    parser.add_argument("--treatment-result", type=Path, action="append", required=True)
    parser.add_argument("--control-evidence-path", action="append", default=[])
    parser.add_argument("--treatment-evidence-path", action="append", default=[])
    parser.add_argument("--control-energy", type=Path, action="append", default=[])
    parser.add_argument("--treatment-energy", type=Path, action="append", default=[])
    parser.add_argument("--quality-requirement", default="approximate")
    parser.add_argument("--phone-serial", default="3C15AU002CL00000")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        result = build_iteration(
            iteration_id=args.iteration_id,
            declared_change=args.declared_change,
            trace_path=args.trace,
            control_paths=args.control_result,
            treatment_paths=args.treatment_result,
            control_evidence_paths=args.control_evidence_path,
            treatment_evidence_paths=args.treatment_evidence_path,
            control_energy_paths=args.control_energy,
            treatment_energy_paths=args.treatment_energy,
            quality_requirement=args.quality_requirement,
            phone_serial=args.phone_serial,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_bytes(result))
    except IterationError as exc:
        parser.exit(2, f"physical iteration failed: {exc}\n")
    print(json.dumps({
        "output": str(args.output),
        "iteration_hash": result["iteration_hash"],
        "verdict": result["verdict"],
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
