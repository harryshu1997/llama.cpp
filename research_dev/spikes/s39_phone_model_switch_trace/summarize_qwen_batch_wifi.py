#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import re
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_EVIDENCE = HERE / "results" / "w0_qwen_batch_wifi"
DEFAULT_CUDA = HERE / "results" / "w0_route_screen" / "qwen_cuda_control.err"
DEFAULT_SHARDS = HERE / "SHARD_MANIFEST.json"
DEFAULT_OUTPUT = DEFAULT_EVIDENCE / "qwen_batch_certificate.json"

MODEL_ID = "qwen3-14b-q4_k_m"
COHORTS = (1, 8, 32)
PROMPT_TOKENS = [785, 6722, 315, 9625, 374]
GENERATED_TOKENS = [12095, 13, 3555, 374, 279, 6722, 315, 279]
CUT_LAYER = 30
N_LAYER = 40
N_EMBD = 5120
ROW_BYTES = N_EMBD * 4
ROWS_PER_REQUEST = len(PROMPT_TOKENS) + len(GENERATED_TOKENS) - 1

REPORT_KEYS = {
    "batch_events",
    "batches",
    "configuration",
    "head_name",
    "latency_ms",
    "requests",
    "route_id",
    "schema",
    "tokens_equal",
    "verdict",
    "workers",
}
CONFIG_KEYS = {
    "gather_us",
    "prefill_chunk",
    "prompt_tokens",
    "requests",
    "slo_ms",
    "steps",
    "token",
}
WORKER_KEYS = {
    "capabilities",
    "file_type",
    "layer_end",
    "layer_start",
    "max_streams",
    "model_sha256",
    "n_batch",
    "n_ctx_seq",
    "n_embd",
    "n_layer",
    "n_ubatch",
}
REQUEST_KEYS = {
    "elapsed_ms",
    "head",
    "prefill_chunk",
    "prompt_length",
    "request_id",
    "route_epoch",
    "slo_met",
    "slo_ms",
    "tokens",
}
BATCH_KEYS = {"batch_count", "batch_sizes", "max_batch", "mean_batch"}
EVENT_KEYS = {
    "batch_size",
    "compute_us",
    "max_queue_us",
    "priorities",
    "request_ids",
}
SESSION_KEYS = {
    "compute_by_op_and_buffer",
    "device_boot_id",
    "expected_backend",
    "layer_end",
    "layer_start",
    "missing_buffer_compute_nodes",
    "n_layer",
    "placement_status",
    "proto_version",
    "reset_applied",
    "schema",
    "session_end",
    "session_id",
    "steps_session",
    "steps_total",
    "worker_boot_nonce",
    "worker_pid",
}
PLACEMENT_KEYS = {
    "compute_by_buffer_type",
    "compute_by_op",
    "compute_by_op_and_buffer",
    "compute_nodes",
    "copy_by_buffer_type",
    "copy_nodes",
    "layer_end",
    "layer_start",
    "metadata_nodes",
    "missing_buffer_compute_nodes",
    "mode",
    "n_layer",
    "pid",
    "role",
    "run_rc",
    "schema",
    "status",
}


class BatchEvidenceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BatchEvidenceError(message)


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def require_int(value: Any, field: str, minimum: int = 0) -> int:
    require(is_int(value), f"{field}: expected integer")
    require(value >= minimum, f"{field}: expected >= {minimum}")
    return value


def require_number(value: Any, field: str, minimum: Decimal) -> Decimal:
    require(
        isinstance(value, (int, Decimal)) and not isinstance(value, bool),
        f"{field}: expected number",
    )
    result = Decimal(value)
    require(result.is_finite(), f"{field}: expected finite number")
    require(result >= minimum, f"{field}: expected >= {minimum}")
    return result


def require_keys(value: Any, expected: set[str], field: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{field}: expected object")
    actual = set(value)
    require(
        actual == expected,
        f"{field}: keys differ; missing={sorted(expected - actual)}, "
        f"unknown={sorted(actual - expected)}",
    )
    return value


def reject_constant(value: str) -> None:
    raise BatchEvidenceError(f"invalid JSON constant {value}")


def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_json(raw: bytes, field: str) -> Any:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise BatchEvidenceError(f"{field}: expected ASCII JSON") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=reject_constant,
            parse_float=Decimal,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise BatchEvidenceError(f"{field}: invalid JSON: {error}") from error


def read_bytes(path: Path, field: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise BatchEvidenceError(f"{field}: cannot read {path}: {error}") from error


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def decimal_ms_to_ns(value: Any, field: str) -> int:
    milliseconds = require_number(value, field, Decimal(0))
    nanoseconds = milliseconds * Decimal(1_000_000)
    require(nanoseconds == nanoseconds.to_integral_value(), f"{field}: sub-ns value")
    return int(nanoseconds)


def extract_prefixed_records(raw: bytes, prefix: bytes, field: str) -> list[dict[str, Any]]:
    records = []
    for line in raw.splitlines():
        if line.startswith(prefix):
            value = parse_json(line[len(prefix):], field)
            require(isinstance(value, dict), f"{field}: expected object")
            records.append(value)
    return records


def nearest_rank(values: list[Decimal], numerator: int, denominator: int) -> Decimal:
    ordered = sorted(values)
    rank = (numerator * len(ordered) + denominator - 1) // denominator
    return ordered[rank - 1]


def validate_cuda(raw: bytes) -> str:
    records = extract_prefixed_records(raw, b"ROUTEJSON ", "cuda_control")
    require(len(records) == 1, "cuda_control: expected one ROUTEJSON")
    record = records[0]
    expected = {
        "status": "ok",
        "route": "SERVER_ONLY",
        "prompt_tokens": len(PROMPT_TOKENS),
        "requested_tokens": len(GENERATED_TOKENS),
        "generated_tokens": len(GENERATED_TOKENS),
        "token_ids": GENERATED_TOKENS,
    }
    for name, value in expected.items():
        require(
            record.get(name) == value and type(record.get(name)) is type(value),
            f"cuda_control.{name}",
        )
    return sha256(raw)


def validate_worker(
    value: Any,
    *,
    field: str,
    layer_start: int,
    layer_end: int,
    model_sha256: str,
    cohort: int,
) -> None:
    worker = require_keys(value, WORKER_KEYS, field)
    expected = {
        "file_type": 15,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "model_sha256": model_sha256,
        "n_embd": N_EMBD,
        "n_layer": N_LAYER,
    }
    for name, expected_value in expected.items():
        require(
            worker[name] == expected_value
            and type(worker[name]) is type(expected_value),
            f"{field}.{name}",
        )
    for name in ("capabilities", "n_batch", "n_ctx_seq", "n_ubatch"):
        require_int(worker[name], f"{field}.{name}", 1)
    require_int(worker["max_streams"], f"{field}.max_streams", cohort)


def validate_batch_summary(
    value: Any,
    events: list[dict[str, Any]],
    field: str,
) -> None:
    summary = require_keys(value, BATCH_KEYS, field)
    sizes = [event["batch_size"] for event in events]
    require(summary["batch_count"] == len(sizes), f"{field}.batch_count")
    require(summary["batch_sizes"] == sizes, f"{field}.batch_sizes")
    require(summary["max_batch"] == max(sizes), f"{field}.max_batch")
    mean = require_number(summary["mean_batch"], f"{field}.mean_batch", Decimal(0))
    require(mean == Decimal(sum(sizes)) / Decimal(len(sizes)), f"{field}.mean_batch")


def validate_events(
    value: Any,
    *,
    field: str,
    cohort: int,
    request_ids: list[int],
) -> list[dict[str, Any]]:
    require(isinstance(value, list) and len(value) == 8, f"{field}: expected 8 events")
    events = []
    expected_sizes = [len(PROMPT_TOKENS) * cohort] + [cohort] * 7
    for index, (candidate, expected_size) in enumerate(zip(value, expected_sizes)):
        event = require_keys(candidate, EVENT_KEYS, f"{field}[{index}]")
        require(event["batch_size"] == expected_size, f"{field}[{index}].batch_size")
        require_int(event["compute_us"], f"{field}[{index}].compute_us", 1)
        require_int(event["max_queue_us"], f"{field}[{index}].max_queue_us")
        require(
            isinstance(event["priorities"], list)
            and event["priorities"] == [0] * expected_size,
            f"{field}[{index}].priorities",
        )
        ids = event["request_ids"]
        require(
            isinstance(ids, list)
            and len(ids) == expected_size
            and all(is_int(request_id) for request_id in ids),
            f"{field}[{index}].request_ids",
        )
        if index == 0:
            require(
                sorted(ids) == sorted(request_ids * len(PROMPT_TOKENS)),
                f"{field}[0].request_ids",
            )
        else:
            require(sorted(ids) == request_ids, f"{field}[{index}].request_ids")
        events.append(event)
    return events


def validate_report(raw: bytes, cohort: int, model_sha256: str) -> dict[str, Any]:
    report = require_keys(parse_json(raw, f"b{cohort}"), REPORT_KEYS, f"b{cohort}")
    require(report["schema"] == "s22-route-probe-v1", f"b{cohort}.schema")
    require(report["verdict"] == "PASS", f"b{cohort}.verdict")
    require(report["tokens_equal"] is True, f"b{cohort}.tokens_equal")
    require(report["route_id"] == f"s39-qwen-wifi-cut30-b{cohort}", f"b{cohort}.route_id")
    require(report["head_name"] == "op15", f"b{cohort}.head_name")

    config = require_keys(report["configuration"], CONFIG_KEYS, f"b{cohort}.configuration")
    expected_config = {
        "prefill_chunk": len(PROMPT_TOKENS),
        "prompt_tokens": PROMPT_TOKENS,
        "requests": cohort,
        "steps": len(GENERATED_TOKENS),
        "token": 2,
    }
    for name, value in expected_config.items():
        require(
            config[name] == value and type(config[name]) is type(value),
            f"b{cohort}.configuration.{name}",
        )
    require_int(config["gather_us"], f"b{cohort}.configuration.gather_us")
    require_number(config["slo_ms"], f"b{cohort}.configuration.slo_ms", Decimal(1))

    workers = require_keys(report["workers"], {"head", "tail"}, f"b{cohort}.workers")
    validate_worker(
        workers["head"],
        field=f"b{cohort}.workers.head",
        layer_start=0,
        layer_end=CUT_LAYER,
        model_sha256=model_sha256,
        cohort=cohort,
    )
    validate_worker(
        workers["tail"],
        field=f"b{cohort}.workers.tail",
        layer_start=CUT_LAYER,
        layer_end=N_LAYER,
        model_sha256=model_sha256,
        cohort=cohort,
    )

    requests = report["requests"]
    require(isinstance(requests, list) and len(requests) == cohort, f"b{cohort}.requests")
    request_ids = list(range(2001, 2001 + cohort))
    elapsed = []
    for index, request in enumerate(requests):
        request = require_keys(request, REQUEST_KEYS, f"b{cohort}.requests[{index}]")
        expected = {
            "head": "op15",
            "prefill_chunk": len(PROMPT_TOKENS),
            "prompt_length": len(PROMPT_TOKENS),
            "request_id": request_ids[index],
            "route_epoch": 1,
            "slo_met": True,
            "tokens": GENERATED_TOKENS,
        }
        for name, value in expected.items():
            require(
                request[name] == value and type(request[name]) is type(value),
                f"b{cohort}.requests[{index}].{name}",
            )
        request_slo = require_number(
            request["slo_ms"], f"b{cohort}.requests[{index}].slo_ms", Decimal(1)
        )
        require(
            request_slo == require_number(
                config["slo_ms"], f"b{cohort}.configuration.slo_ms", Decimal(1)
            ),
            f"b{cohort}.requests[{index}].slo_ms",
        )
        elapsed.append(
            require_number(
                request["elapsed_ms"],
                f"b{cohort}.requests[{index}].elapsed_ms",
                Decimal(0),
            )
        )

    latency = require_keys(report["latency_ms"], {"max", "p50", "p95"}, f"b{cohort}.latency")
    expected_latency = {
        "max": max(elapsed),
        "p50": nearest_rank(elapsed, 1, 2),
        "p95": nearest_rank(elapsed, 19, 20),
    }
    for name, value in expected_latency.items():
        require(
            require_number(latency[name], f"b{cohort}.latency.{name}", Decimal(0))
            == value,
            f"b{cohort}.latency.{name}",
        )

    event_groups = require_keys(report["batch_events"], {"head", "tail"}, f"b{cohort}.batch_events")
    summaries = require_keys(report["batches"], {"head", "tail"}, f"b{cohort}.batches")
    for role in ("head", "tail"):
        events = validate_events(
            event_groups[role],
            field=f"b{cohort}.batch_events.{role}",
            cohort=cohort,
            request_ids=request_ids,
        )
        validate_batch_summary(summaries[role], events, f"b{cohort}.batches.{role}")

    return {
        "artifact_sha256": sha256(raw),
        "cohort_requests": cohort,
        "generated_tokens": cohort * len(GENERATED_TOKENS),
        "gather_us": config["gather_us"],
        "head_compute_us": sum(row["compute_us"] for row in event_groups["head"]),
        "latency_max_ns": decimal_ms_to_ns(latency["max"], f"b{cohort}.latency.max"),
        "latency_p50_ns": decimal_ms_to_ns(latency["p50"], f"b{cohort}.latency.p50"),
        "latency_p95_ns": decimal_ms_to_ns(latency["p95"], f"b{cohort}.latency.p95"),
        "request_throughput_milli_per_s": (
            cohort * 1_000_000_000_000
            // decimal_ms_to_ns(latency["p50"], f"b{cohort}.latency.p50")
        ),
        "tail_compute_us": sum(row["compute_us"] for row in event_groups["tail"]),
        "token_rows": ROWS_PER_REQUEST * cohort,
    }


def validate_op_map(value: Any, field: str, allow_cpu_get_rows: bool) -> int:
    require(isinstance(value, dict) and bool(value), f"{field}: expected object")
    total = 0
    for op, backends in value.items():
        require(isinstance(op, str) and bool(op), f"{field}: invalid op")
        require(isinstance(backends, dict) and bool(backends), f"{field}.{op}")
        for backend, count in backends.items():
            require_int(count, f"{field}.{op}.{backend}")
            require(backend in {"CPU", "OpenCL"}, f"{field}.{op}.{backend}")
            if backend == "CPU":
                require(allow_cpu_get_rows and op == "GET_ROWS", f"{field}: CPU fallback")
            total += count
    return total


def validate_worker_log(
    raw: bytes,
    *,
    field: str,
    role: str,
    mode: str,
    layer_start: int,
    layer_end: int,
    remote_path: str,
    allow_cpu_get_rows: bool,
) -> dict[str, Any]:
    require(remote_path.encode("ascii") in raw, f"{field}: model path missing")
    sessions = extract_prefixed_records(raw, b"SESSIONCERT ", f"{field}.session")
    require(len(sessions) == 3, f"{field}: expected three session certificates")
    expected_steps = [ROWS_PER_REQUEST * cohort for cohort in COHORTS]
    expected_totals = []
    running = 0
    for steps in expected_steps:
        running += steps
        expected_totals.append(running)

    pid = None
    nonce = None
    boot_id = None
    for index, session in enumerate(sessions):
        session = require_keys(session, SESSION_KEYS, f"{field}.session[{index}]")
        expected = {
            "schema": "ls-stagenet-session-v2",
            "proto_version": 2,
            "session_id": index + 1,
            "session_end": "STOP" if index == 2 else "DETACH",
            "expected_backend": "GPUOpenCL",
            "layer_start": layer_start,
            "layer_end": layer_end,
            "n_layer": N_LAYER,
            "steps_session": expected_steps[index],
            "steps_total": expected_totals[index],
            "reset_applied": index < 2,
            "missing_buffer_compute_nodes": 0,
            "placement_status": "SCHEDULED_PLACEMENT_OK",
        }
        for name, value in expected.items():
            require(
                session[name] == value and type(session[name]) is type(value),
                f"{field}.session[{index}].{name}",
            )
        session_pid = require_int(session["worker_pid"], f"{field}.session[{index}].pid", 1)
        session_nonce = session["worker_boot_nonce"]
        session_boot = session["device_boot_id"]
        require(isinstance(session_nonce, str) and bool(session_nonce), f"{field}.session[{index}].nonce")
        require(isinstance(session_boot, str) and bool(session_boot), f"{field}.session[{index}].boot")
        pid = session_pid if pid is None else pid
        nonce = session_nonce if nonce is None else nonce
        boot_id = session_boot if boot_id is None else boot_id
        require(session_pid == pid, f"{field}: worker PID changed")
        require(session_nonce == nonce, f"{field}: worker nonce changed")
        require(session_boot == boot_id, f"{field}: device boot changed")
        require(
            validate_op_map(
                session["compute_by_op_and_buffer"],
                f"{field}.session[{index}].ops",
                allow_cpu_get_rows,
            )
            > 0,
            f"{field}.session[{index}]: no compute",
        )

    placements = extract_prefixed_records(raw, b"PLACEMENTCERT ", f"{field}.placement")
    require(len(placements) == 1, f"{field}: expected one placement certificate")
    placement = require_keys(placements[0], PLACEMENT_KEYS, f"{field}.placement")
    expected_placement = {
        "schema": "layersplit-scheduled-placement-v2",
        "role": role,
        "mode": mode,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "n_layer": N_LAYER,
        "pid": pid,
        "run_rc": 0,
        "missing_buffer_compute_nodes": 0,
        "status": "SCHEDULED_PLACEMENT_OK",
    }
    for name, value in expected_placement.items():
        require(
            placement[name] == value and type(placement[name]) is type(value),
            f"{field}.placement.{name}",
        )
    compute_nodes = require_int(placement["compute_nodes"], f"{field}.placement.compute_nodes", 1)
    by_buffer = placement["compute_by_buffer_type"]
    require(isinstance(by_buffer, dict) and bool(by_buffer), f"{field}.placement.backends")
    require(
        all(
            backend in {"CPU", "OpenCL"} and is_int(count) and count >= 0
            for backend, count in by_buffer.items()
        )
        and sum(by_buffer.values()) == compute_nodes
        and by_buffer.get("OpenCL", 0) > 0,
        f"{field}.placement.backends",
    )
    if "CPU" in by_buffer:
        require(allow_cpu_get_rows, f"{field}.placement: CPU fallback")
    op_total = validate_op_map(
        placement["compute_by_op_and_buffer"],
        f"{field}.placement.ops",
        allow_cpu_get_rows,
    )
    require(op_total == compute_nodes, f"{field}.placement.ops total")
    derived_by_backend: dict[str, int] = {}
    derived_by_op: dict[str, int] = {}
    for op, backends in placement["compute_by_op_and_buffer"].items():
        derived_by_op[op] = sum(backends.values())
        for backend, count in backends.items():
            derived_by_backend[backend] = derived_by_backend.get(backend, 0) + count
    require(by_buffer == derived_by_backend, f"{field}.placement.backend totals")
    require(
        placement["compute_by_op_and_buffer"]
        == sessions[-1]["compute_by_op_and_buffer"],
        f"{field}: final session/placement mismatch",
    )
    require(
        isinstance(placement["compute_by_op"], dict)
        and all(is_int(value) and value >= 0 for value in placement["compute_by_op"].values())
        and placement["compute_by_op"] == derived_by_op,
        f"{field}.placement.compute_by_op",
    )
    for name in ("copy_nodes", "metadata_nodes"):
        require_int(placement[name], f"{field}.placement.{name}")
    require(
        isinstance(placement["copy_by_buffer_type"], dict)
        and all(
            is_int(value) and value >= 0
            for value in placement["copy_by_buffer_type"].values()
        )
        and sum(placement["copy_by_buffer_type"].values()) == placement["copy_nodes"],
        f"{field}.placement.copy_by_buffer_type",
    )

    memory_patterns = {
        "cpu_mapped_model_centi_mib": rb"CPU_Mapped model buffer size =\s+([0-9]+\.[0-9]{2}) MiB",
        "opencl_compute_centi_mib": rb"sched_reserve:\s+OpenCL compute buffer size =\s+([0-9]+\.[0-9]{2}) MiB",
        "opencl_kv_centi_mib": rb"OpenCL KV buffer size =\s+([0-9]+\.[0-9]{2}) MiB",
        "opencl_model_centi_mib": rb"OpenCL model buffer size =\s+([0-9]+\.[0-9]{2}) MiB",
    }
    memory = {}
    for name, pattern in memory_patterns.items():
        matches = re.findall(pattern, raw)
        require(len(matches) == 1, f"{field}.{name}: expected one value")
        memory[name] = int(Decimal(matches[0].decode("ascii")) * 100)

    return {
        "artifact_sha256": sha256(raw),
        "device_boot_id": boot_id,
        "memory_log_reported": memory,
        "opencl_compute_nodes": by_buffer["OpenCL"],
        "persistent_sessions": len(sessions),
        "worker_boot_nonce": nonce,
        "worker_pid": pid,
    }


def load_shards(raw: bytes) -> tuple[str, dict[str, Any]]:
    root = parse_json(raw, "shard_manifest")
    require(isinstance(root, dict) and root.get("schema_version") == 1, "shard_manifest")
    model = root.get("models", {}).get(MODEL_ID)
    require(isinstance(model, dict), f"shard_manifest.{MODEL_ID}")
    source = model.get("source")
    placements = model.get("placements")
    require(isinstance(source, dict), "shard_manifest.source")
    require(isinstance(placements, dict), "shard_manifest.placements")
    model_sha = source.get("sha256")
    require(
        isinstance(model_sha, str) and re.fullmatch(r"[0-9a-f]{64}", model_sha) is not None,
        "shard_manifest.source.sha256",
    )
    require(
        is_int(source.get("block_count")) and source["block_count"] == N_LAYER,
        "shard_manifest.source.block_count",
    )
    require(
        is_int(placements.get("op15", {}).get("layer_start"))
        and is_int(placements.get("op15", {}).get("layer_end"))
        and placements["op15"]["layer_start"] == 0
        and placements["op15"]["layer_end"] >= CUT_LAYER,
        "shard_manifest.op15 coverage",
    )
    require(
        is_int(placements.get("op12", {}).get("layer_start"))
        and is_int(placements.get("op12", {}).get("layer_end"))
        and placements["op12"]["layer_start"] <= CUT_LAYER
        and placements["op12"]["layer_end"] == N_LAYER,
        "shard_manifest.op12 coverage",
    )
    for device in ("op12", "op15"):
        require(
            isinstance(placements[device].get("remote_path"), str)
            and bool(placements[device]["remote_path"]),
            f"shard_manifest.{device}.remote_path",
        )
    return model_sha, placements


def validate_run_context(raw: bytes) -> dict[str, Any]:
    value = parse_json(raw, "run_context")
    require(isinstance(value, dict), "run_context: expected object")
    require(value.get("schema") == "s39-qwen-batch-run-context-v1", "run_context.schema")
    runtime = value.get("runtime")
    weights = value.get("weights")
    require(isinstance(runtime, dict), "run_context.runtime")
    require(isinstance(weights, dict), "run_context.weights")
    require(runtime.get("direct_phone_to_phone") is False, "run_context.direct_phone_to_phone")
    require(runtime.get("path_provenance") == "posthoc_operator_record_no_interface_counter", "run_context.path")
    for role, device, host, port in (
        ("head_endpoint", "op15", "172.20.173.218", 39315),
        ("tail_endpoint", "op12", "172.20.59.72", 39312),
    ):
        endpoint = runtime.get(role)
        require(isinstance(endpoint, dict), f"run_context.{role}")
        require(endpoint.get("device") == device, f"run_context.{role}.device")
        require(endpoint.get("host") == host, f"run_context.{role}.host")
        require(endpoint.get("port") == port, f"run_context.{role}.port")
        require(endpoint.get("transport") == "wifi_tcp", f"run_context.{role}.transport")
    require(weights.get("provisioning_transport") == "usb_adb", "run_context.weights.transport")
    require(weights.get("runtime_source") == "phone_ufs", "run_context.weights.source")
    require(weights.get("timing_scope") == "before_paid_window", "run_context.weights.scope")
    require(weights.get("in_paid_window_bytes") == 0, "run_context.weights.bytes")
    return value


def build(
    evidence_dir: Path = DEFAULT_EVIDENCE,
    cuda_path: Path = DEFAULT_CUDA,
    shard_manifest_path: Path = DEFAULT_SHARDS,
) -> dict[str, Any]:
    shard_raw = read_bytes(shard_manifest_path, "shard_manifest")
    model_sha, placements = load_shards(shard_raw)
    cuda_raw = read_bytes(cuda_path, "cuda_control")
    cuda_digest = validate_cuda(cuda_raw)
    context_raw = read_bytes(evidence_dir / "RUN_CONTEXT.json", "run_context")
    context = validate_run_context(context_raw)

    cohorts = []
    for cohort in COHORTS:
        raw = read_bytes(evidence_dir / f"b{cohort}.json", f"b{cohort}")
        cohorts.append(validate_report(raw, cohort, model_sha))

    head_raw = read_bytes(evidence_dir / "op15_qwen_batch32_head.log", "op15_log")
    tail_raw = read_bytes(evidence_dir / "op12_qwen_batch32_tail.log", "op12_log")
    head = validate_worker_log(
        head_raw,
        field="op15_log",
        role="phone_stage",
        mode="stagenet",
        layer_start=0,
        layer_end=CUT_LAYER,
        remote_path=placements["op15"]["remote_path"],
        allow_cpu_get_rows=True,
    )
    tail = validate_worker_log(
        tail_raw,
        field="op12_log",
        role="host_tail_v3",
        mode="tailv3",
        layer_start=CUT_LAYER,
        layer_end=N_LAYER,
        remote_path=placements["op12"]["remote_path"],
        allow_cpu_get_rows=False,
    )

    baseline = cohorts[0]
    for cohort in cohorts:
        rows = cohort["token_rows"]
        cohort["activation_payload_bytes_per_wifi_leg"] = rows * ROW_BYTES
        cohort["activation_payload_bytes_total"] = rows * ROW_BYTES * 2
        cohort["request_throughput_gain_vs_b1_ppm"] = (
            cohort["cohort_requests"]
            * baseline["latency_p50_ns"]
            * 1_000_000
            // (cohort["latency_p50_ns"] * baseline["cohort_requests"])
        )

    return {
        "artifacts": {
            "cuda_control": cuda_digest,
            "op12_worker_log": tail["artifact_sha256"],
            "op15_worker_log": head["artifact_sha256"],
            "run_context": sha256(context_raw),
            "shard_manifest": sha256(shard_raw),
        },
        "cohorts": cohorts,
        "correctness": {
            "cuda_reference_tokens": GENERATED_TOKENS,
            "generated_token_checks": sum(
                cohort["generated_tokens"] for cohort in cohorts
            ),
            "status": "TOKEN_EXACT_SINGLE_PROMPT",
        },
        "cut_layer": CUT_LAYER,
        "model_id": MODEL_ID,
        "model_sha256": model_sha,
        "schema": "s39-qwen-batch-certificate-v1",
        "status": "PROVISIONAL_BATCH",
        "transport": {
            "activation_path": "WIFI_TCP_VIA_HOST_COORDINATOR",
            "declared_runtime_weight_bytes": 0,
            "direct_phone_to_phone": False,
            "path_provenance": context["runtime"]["path_provenance"],
            "weight_path": "USB_ADB_BEFORE_PAID_WINDOW_THEN_PHONE_UFS",
        },
        "workers": {"op12": tail, "op15": head},
    }


def atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as error:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise BatchEvidenceError(f"cannot write {path}: {error}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description="Certify the S39 Qwen WiFi batch screen")
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--cuda-control", type=Path, default=DEFAULT_CUDA)
    parser.add_argument("--shard-manifest", type=Path, default=DEFAULT_SHARDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        result = build(args.evidence, args.cuda_control, args.shard_manifest)
        atomic_write(args.output, canonical_bytes(result))
    except BatchEvidenceError as error:
        print(f"S39_BATCH_EVIDENCE_ERROR: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "cohorts": [row["cohort_requests"] for row in result["cohorts"]],
                "status": result["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
