#!/usr/bin/env python3
"""Validate the real OP15-to-OP12 direct activation-chain evidence."""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import summarize_qwen_batch_wifi as common


DEFAULT_EVIDENCE = HERE / "results" / "w1_direct_phone_chain"
DEFAULT_OUTPUT = DEFAULT_EVIDENCE / "direct_chain_certificate.json"

MODEL_SHA256 = "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0"
RELAY_BINARY_SHA256 = "1c809cb50cae6aa86869d61068a05173c719e4542c851e478ee1e033c5456929"
PROMPT_TOKENS = [785, 6722, 315, 9625, 374]
GENERATED_TOKENS = [12095, 13, 3555, 374, 279, 6722, 315, 279]
N_EMBD = 5120
ROW_BYTES = N_EMBD * 4
ROWS_PER_REQUEST = len(PROMPT_TOKENS) + len(GENERATED_TOKENS) - 1

DIRECT_REPORT_KEYS = {
    "batch_events",
    "batches",
    "configuration",
    "latency_ms",
    "relay_endpoint",
    "requests",
    "route_id",
    "schema",
    "tokens_exact",
    "transport",
    "verdict",
    "worker",
}
DIRECT_CONFIG_KEYS = {
    "expected_tokens",
    "gather_us",
    "prefill_chunk",
    "prompt_tokens",
    "requests",
    "slo_ms",
    "steps",
}
DIRECT_REQUEST_KEYS = {
    "elapsed_ms",
    "prefill_chunk",
    "prompt_length",
    "request_id",
    "route_epoch",
    "slo_met",
    "slo_ms",
    "tokens",
}
DIRECT_EVENT_KEYS = {
    "batch_size",
    "compute_us",
    "max_queue_us",
    "priorities",
    "request_ids",
}
DIRECT_CERT_KEYS = {
    "activation_payload_bytes",
    "batches",
    "cut_layer",
    "file_type",
    "head_endpoint",
    "host_activation_payload_bytes",
    "layer_end",
    "layer_start",
    "model_sha256",
    "n_embd",
    "n_layer",
    "rows",
    "run_rc",
    "schema",
    "status",
    "tail_endpoint",
}


def exact(value: Any, expected: Any, field: str) -> None:
    common.require(
        value == expected and type(value) is type(expected),
        field,
    )


def validate_event(
    candidate: Any,
    index: int,
    cohort: int,
    request_ids: list[int],
) -> dict[str, Any]:
    field = f"direct_b{cohort}.batch_events[{index}]"
    event = common.require_keys(candidate, DIRECT_EVENT_KEYS, field)
    expected_size = len(PROMPT_TOKENS) * cohort if index == 0 else cohort
    exact(event["batch_size"], expected_size, f"{field}.batch_size")
    common.require_int(event["compute_us"], f"{field}.compute_us", 1)
    common.require_int(event["max_queue_us"], f"{field}.max_queue_us")
    common.require(
        isinstance(event["priorities"], list)
        and len(event["priorities"]) == expected_size
        and all(common.is_int(value) and value == 0 for value in event["priorities"]),
        f"{field}.priorities",
    )
    ids = event["request_ids"]
    common.require(
        isinstance(ids, list)
        and all(common.is_int(value) for value in ids)
        and len(ids) == expected_size,
        f"{field}.request_ids",
    )
    expected_ids = request_ids * (len(PROMPT_TOKENS) if index == 0 else 1)
    common.require(sorted(ids) == sorted(expected_ids), f"{field}.request_ids")
    return event


def validate_direct_report(raw: bytes, cohort: int) -> dict[str, Any]:
    field = f"direct_b{cohort}"
    report = common.require_keys(
        common.parse_json(raw, field),
        DIRECT_REPORT_KEYS,
        field,
    )
    exact(report["schema"], "s39-direct-route-probe-v1", f"{field}.schema")
    exact(report["verdict"], "PASS", f"{field}.verdict")
    exact(report["tokens_exact"], True, f"{field}.tokens_exact")
    exact(
        report["route_id"],
        f"s39-qwen-direct-fixed-cut30-b{cohort}",
        f"{field}.route_id",
    )
    exact(
        report["relay_endpoint"],
        {"host": "172.20.173.218", "port": 39415},
        f"{field}.relay_endpoint",
    )

    config = common.require_keys(
        report["configuration"],
        DIRECT_CONFIG_KEYS,
        f"{field}.configuration",
    )
    expected_config = {
        "expected_tokens": GENERATED_TOKENS,
        "prefill_chunk": len(PROMPT_TOKENS),
        "prompt_tokens": PROMPT_TOKENS,
        "requests": cohort,
        "steps": len(GENERATED_TOKENS),
    }
    for name, value in expected_config.items():
        exact(config[name], value, f"{field}.configuration.{name}")
    common.require_int(config["gather_us"], f"{field}.configuration.gather_us")
    config_slo = common.require_number(
        config["slo_ms"],
        f"{field}.configuration.slo_ms",
        Decimal(1),
    )

    worker = common.require_keys(report["worker"], common.WORKER_KEYS, f"{field}.worker")
    expected_worker = {
        "file_type": 15,
        "layer_end": 40,
        "layer_start": 0,
        "model_sha256": MODEL_SHA256,
        "n_embd": N_EMBD,
        "n_layer": 40,
    }
    for name, value in expected_worker.items():
        exact(worker[name], value, f"{field}.worker.{name}")
    for name in ("capabilities", "n_batch", "n_ctx_seq", "n_ubatch"):
        common.require_int(worker[name], f"{field}.worker.{name}", 1)
    common.require_int(worker["max_streams"], f"{field}.worker.max_streams", cohort)
    common.require(
        worker["capabilities"] & 16,
        f"{field}.worker.terminal_capability",
    )

    requests = report["requests"]
    request_ids = list(range(3001, 3001 + cohort))
    common.require(
        isinstance(requests, list) and len(requests) == cohort,
        f"{field}.requests",
    )
    elapsed = []
    for index, candidate in enumerate(requests):
        request = common.require_keys(
            candidate,
            DIRECT_REQUEST_KEYS,
            f"{field}.requests[{index}]",
        )
        expected = {
            "prefill_chunk": len(PROMPT_TOKENS),
            "prompt_length": len(PROMPT_TOKENS),
            "request_id": request_ids[index],
            "route_epoch": 1,
            "slo_met": True,
            "tokens": GENERATED_TOKENS,
        }
        for name, value in expected.items():
            exact(request[name], value, f"{field}.requests[{index}].{name}")
        request_slo = common.require_number(
            request["slo_ms"],
            f"{field}.requests[{index}].slo_ms",
            Decimal(1),
        )
        common.require(request_slo == config_slo, f"{field}.requests[{index}].slo_ms")
        elapsed.append(
            common.require_number(
                request["elapsed_ms"],
                f"{field}.requests[{index}].elapsed_ms",
                Decimal(0),
            )
        )

    latency = common.require_keys(
        report["latency_ms"],
        {"max", "p50", "p95"},
        f"{field}.latency_ms",
    )
    expected_latency = {
        "max": max(elapsed),
        "p50": common.nearest_rank(elapsed, 1, 2),
        "p95": common.nearest_rank(elapsed, 19, 20),
    }
    for name, value in expected_latency.items():
        common.require(
            common.require_number(
                latency[name],
                f"{field}.latency_ms.{name}",
                Decimal(0),
            )
            == value,
            f"{field}.latency_ms.{name}",
        )

    events_value = report["batch_events"]
    common.require(
        isinstance(events_value, list) and len(events_value) == 8,
        f"{field}.batch_events",
    )
    events = [
        validate_event(candidate, index, cohort, request_ids)
        for index, candidate in enumerate(events_value)
    ]
    sizes = [event["batch_size"] for event in events]
    summary = common.require_keys(
        report["batches"],
        {"batch_count", "batch_sizes", "max_batch"},
        f"{field}.batches",
    )
    exact(summary["batch_count"], len(events), f"{field}.batches.batch_count")
    exact(summary["batch_sizes"], sizes, f"{field}.batches.batch_sizes")
    exact(summary["max_batch"], max(sizes), f"{field}.batches.max_batch")

    expected_rows = ROWS_PER_REQUEST * cohort
    expected_bytes = expected_rows * ROW_BYTES
    transport = common.require_keys(
        report["transport"],
        {
            "direct_activation_payload_bytes",
            "host_activation_payload_bytes",
            "mode",
        },
        f"{field}.transport",
    )
    exact(
        transport["direct_activation_payload_bytes"],
        expected_bytes,
        f"{field}.transport.direct_activation_payload_bytes",
    )
    exact(
        transport["host_activation_payload_bytes"],
        0,
        f"{field}.transport.host_activation_payload_bytes",
    )
    exact(
        transport["mode"],
        "OP15_TO_OP12_DIRECT_WIFI",
        f"{field}.transport.mode",
    )
    return {
        "artifact_sha256": common.sha256(raw),
        "cohort_requests": cohort,
        "compute_us": sum(event["compute_us"] for event in events),
        "direct_activation_payload_bytes": expected_bytes,
        "host_activation_payload_bytes": 0,
        "latency_p50_ns": common.decimal_ms_to_ns(
            latency["p50"],
            f"{field}.latency_ms.p50",
        ),
        "token_checks": cohort * len(GENERATED_TOKENS),
        "token_rows": expected_rows,
    }


def validate_direct_cert(raw: bytes, cohort: int) -> dict[str, Any]:
    field = f"relay_b{cohort}"
    records = common.extract_prefixed_records(raw, b"DIRECTCERT ", field)
    common.require(len(records) == 1, f"{field}: expected one DIRECTCERT")
    cert = common.require_keys(records[0], DIRECT_CERT_KEYS, field)
    expected_rows = ROWS_PER_REQUEST * cohort
    expected = {
        "activation_payload_bytes": expected_rows * ROW_BYTES,
        "batches": 8,
        "cut_layer": 30,
        "file_type": 15,
        "head_endpoint": "127.0.0.1:39315",
        "host_activation_payload_bytes": 0,
        "layer_end": 40,
        "layer_start": 0,
        "model_sha256": MODEL_SHA256,
        "n_embd": N_EMBD,
        "n_layer": 40,
        "rows": expected_rows,
        "run_rc": 0,
        "schema": "ls-stage-direct-relay-v1",
        "status": "DIRECT_RELAY_OK",
        "tail_endpoint": "172.20.59.72:39312",
    }
    for name, value in expected.items():
        exact(cert[name], value, f"{field}.{name}")
    return {
        "artifact_sha256": common.sha256(raw),
        "activation_payload_bytes": cert["activation_payload_bytes"],
        "host_activation_payload_bytes": 0,
    }


def validate_ops(value: Any, field: str, allow_cpu_get_rows: bool) -> int:
    return common.validate_op_map(value, field, allow_cpu_get_rows)


def validate_worker_log(
    raw: bytes,
    *,
    field: str,
    layer_start: int,
    layer_end: int,
    role: str,
    mode: str,
    allow_cpu_get_rows: bool,
) -> dict[str, Any]:
    sessions = common.extract_prefixed_records(raw, b"SESSIONCERT ", f"{field}.session")
    common.require(len(sessions) == 2, f"{field}: expected two sessions")
    expected_steps = [ROWS_PER_REQUEST, ROWS_PER_REQUEST * 32]
    expected_totals = [ROWS_PER_REQUEST, ROWS_PER_REQUEST * 33]
    pid = None
    nonce = None
    boot = None
    for index, candidate in enumerate(sessions):
        session = common.require_keys(
            candidate,
            common.SESSION_KEYS,
            f"{field}.session[{index}]",
        )
        expected = {
            "expected_backend": "GPUOpenCL",
            "layer_end": layer_end,
            "layer_start": layer_start,
            "missing_buffer_compute_nodes": 0,
            "n_layer": 40,
            "placement_status": "SCHEDULED_PLACEMENT_OK",
            "proto_version": 2,
            "reset_applied": index == 0,
            "schema": "ls-stagenet-session-v2",
            "session_end": "DETACH" if index == 0 else "STOP",
            "session_id": index + 1,
            "steps_session": expected_steps[index],
            "steps_total": expected_totals[index],
        }
        for name, value in expected.items():
            exact(session[name], value, f"{field}.session[{index}].{name}")
        current_pid = common.require_int(
            session["worker_pid"],
            f"{field}.session[{index}].worker_pid",
            1,
        )
        current_nonce = session["worker_boot_nonce"]
        current_boot = session["device_boot_id"]
        common.require(
            isinstance(current_nonce, str) and bool(current_nonce),
            f"{field}.session[{index}].worker_boot_nonce",
        )
        common.require(
            isinstance(current_boot, str) and bool(current_boot),
            f"{field}.session[{index}].device_boot_id",
        )
        pid = current_pid if pid is None else pid
        nonce = current_nonce if nonce is None else nonce
        boot = current_boot if boot is None else boot
        exact(current_pid, pid, f"{field}.session[{index}].worker_pid")
        exact(current_nonce, nonce, f"{field}.session[{index}].worker_boot_nonce")
        exact(current_boot, boot, f"{field}.session[{index}].device_boot_id")
        common.require(
            validate_ops(
                session["compute_by_op_and_buffer"],
                f"{field}.session[{index}].ops",
                allow_cpu_get_rows,
            )
            > 0,
            f"{field}.session[{index}].ops",
        )

    placements = common.extract_prefixed_records(raw, b"PLACEMENTCERT ", f"{field}.placement")
    common.require(len(placements) == 1, f"{field}: expected one PLACEMENTCERT")
    placement = common.require_keys(
        placements[0],
        common.PLACEMENT_KEYS,
        f"{field}.placement",
    )
    expected_placement = {
        "layer_end": layer_end,
        "layer_start": layer_start,
        "missing_buffer_compute_nodes": 0,
        "mode": mode,
        "n_layer": 40,
        "pid": pid,
        "role": role,
        "run_rc": 0,
        "schema": "layersplit-scheduled-placement-v2",
        "status": "SCHEDULED_PLACEMENT_OK",
    }
    for name, value in expected_placement.items():
        exact(placement[name], value, f"{field}.placement.{name}")
    compute_nodes = common.require_int(
        placement["compute_nodes"],
        f"{field}.placement.compute_nodes",
        1,
    )
    op_nodes = validate_ops(
        placement["compute_by_op_and_buffer"],
        f"{field}.placement.ops",
        allow_cpu_get_rows,
    )
    common.require(op_nodes == compute_nodes, f"{field}.placement.compute_nodes")
    exact(
        placement["compute_by_op_and_buffer"],
        sessions[-1]["compute_by_op_and_buffer"],
        f"{field}.placement.session_match",
    )
    backend_counts = placement["compute_by_buffer_type"]
    derived_backends: dict[str, int] = {}
    derived_ops: dict[str, int] = {}
    for op, backends in placement["compute_by_op_and_buffer"].items():
        derived_ops[op] = sum(backends.values())
        for backend, count in backends.items():
            derived_backends[backend] = derived_backends.get(backend, 0) + count
    common.require(
        isinstance(backend_counts, dict)
        and backend_counts.get("OpenCL", 0) > 0
        and backend_counts == derived_backends,
        f"{field}.placement.compute_by_buffer_type",
    )
    common.require(
        placement["compute_by_op"] == derived_ops,
        f"{field}.placement.compute_by_op",
    )
    if "CPU" in backend_counts:
        common.require(allow_cpu_get_rows, f"{field}.placement.CPU")
    common.require_int(
        placement["copy_nodes"],
        f"{field}.placement.copy_nodes",
    )
    common.require_int(
        placement["metadata_nodes"],
        f"{field}.placement.metadata_nodes",
    )
    common.require(
        isinstance(placement["copy_by_buffer_type"], dict)
        and all(
            isinstance(name, str)
            and bool(name)
            and common.is_int(count)
            and count >= 0
            for name, count in placement["copy_by_buffer_type"].items()
        )
        and sum(placement["copy_by_buffer_type"].values())
        == placement["copy_nodes"],
        f"{field}.placement.copy_by_buffer_type",
    )
    return {
        "artifact_sha256": common.sha256(raw),
        "device_boot_id": boot,
        "opencl_compile_fallback_observed": b"kernel compile error" in raw,
        "opencl_compute_nodes": backend_counts["OpenCL"],
        "persistent_sessions": 2,
        "worker_boot_nonce": nonce,
        "worker_pid": pid,
    }


def validate_host_control(raw: bytes) -> dict[str, Any]:
    report = common.require_keys(
        common.parse_json(raw, "host_control"),
        common.REPORT_KEYS,
        "host_control",
    )
    exact(report["schema"], "s22-route-probe-v1", "host_control.schema")
    exact(report["verdict"], "PASS", "host_control.verdict")
    exact(report["tokens_equal"], True, "host_control.tokens_equal")
    exact(
        report["route_id"],
        "s39-qwen-matched-host-relay-cut30-b1",
        "host_control.route_id",
    )
    requests = report["requests"]
    common.require(isinstance(requests, list) and len(requests) == 1, "host_control.requests")
    exact(requests[0]["tokens"], GENERATED_TOKENS, "host_control.requests[0].tokens")
    latency = common.require_keys(
        report["latency_ms"],
        {"max", "p50", "p95"},
        "host_control.latency_ms",
    )
    elapsed = common.require_number(
        requests[0]["elapsed_ms"],
        "host_control.requests[0].elapsed_ms",
        Decimal(0),
    )
    for name in ("max", "p50", "p95"):
        common.require(
            common.require_number(
                latency[name],
                f"host_control.latency_ms.{name}",
                Decimal(0),
            )
            == elapsed,
            f"host_control.latency_ms.{name}",
        )
    workers = common.require_keys(report["workers"], {"head", "tail"}, "host_control.workers")
    for role, start, end in (("head", 0, 30), ("tail", 30, 40)):
        worker = common.require_keys(
            workers[role],
            common.WORKER_KEYS,
            f"host_control.workers.{role}",
        )
        exact(worker["layer_start"], start, f"host_control.workers.{role}.layer_start")
        exact(worker["layer_end"], end, f"host_control.workers.{role}.layer_end")
        exact(worker["n_layer"], 40, f"host_control.workers.{role}.n_layer")
        exact(worker["n_embd"], N_EMBD, f"host_control.workers.{role}.n_embd")
        exact(worker["file_type"], 15, f"host_control.workers.{role}.file_type")
        exact(
            worker["model_sha256"],
            MODEL_SHA256,
            f"host_control.workers.{role}.model_sha256",
        )
    return {
        "artifact_sha256": common.sha256(raw),
        "latency_p50_ns": common.decimal_ms_to_ns(
            latency["p50"],
            "host_control.latency_ms.p50",
        ),
    }


def validate_relay_binary_hashes(raw: bytes) -> dict[str, Any]:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise common.BatchEvidenceError("relay_binary_sha256: non-ASCII") from error
    expected = (
        f"local {RELAY_BINARY_SHA256}\n"
        f"op15 {RELAY_BINARY_SHA256}\n"
    )
    common.require(text == expected, "relay_binary_sha256: mismatch")
    return {
        "artifact_sha256": common.sha256(raw),
        "binary_sha256": RELAY_BINARY_SHA256,
        "scope": "POST_RUN_LOCAL_AND_DEVICE_HASH_EQUALITY",
    }


def build(evidence_dir: Path = DEFAULT_EVIDENCE) -> dict[str, Any]:
    direct = {
        1: validate_direct_report(
            common.read_bytes(evidence_dir / "b1_fixed.json", "direct_b1"),
            1,
        ),
        32: validate_direct_report(
            common.read_bytes(evidence_dir / "b32_fixed.json", "direct_b32"),
            32,
        ),
    }
    relay = {
        1: validate_direct_cert(
            common.read_bytes(evidence_dir / "op15_relay_fixed_b1.log", "relay_b1"),
            1,
        ),
        32: validate_direct_cert(
            common.read_bytes(evidence_dir / "op15_relay_fixed_b32.log", "relay_b32"),
            32,
        ),
    }
    for cohort in (1, 32):
        exact(
            relay[cohort]["activation_payload_bytes"],
            direct[cohort]["direct_activation_payload_bytes"],
            f"b{cohort}.report_relay_activation_bytes",
        )
    op15 = validate_worker_log(
        common.read_bytes(evidence_dir / "op15_fixed_head.log", "op15_log"),
        field="op15_log",
        layer_start=0,
        layer_end=30,
        role="phone_stage",
        mode="stagenet",
        allow_cpu_get_rows=True,
    )
    op12 = validate_worker_log(
        common.read_bytes(evidence_dir / "op12_fixed_tail.log", "op12_log"),
        field="op12_log",
        layer_start=30,
        layer_end=40,
        role="host_tail_v3",
        mode="tailv3",
        allow_cpu_get_rows=False,
    )
    control = validate_host_control(
        common.read_bytes(
            evidence_dir / "matched_host_relay_b1.json",
            "host_control",
        )
    )
    relay_binary = validate_relay_binary_hashes(
        common.read_bytes(
            evidence_dir / "relay_binary_sha256.txt",
            "relay_binary_sha256",
        )
    )
    direct_b1_ns = direct[1]["latency_p50_ns"]
    control_ns = control["latency_p50_ns"]
    return {
        "comparison": {
            "direct_b1_latency_ns": direct_b1_ns,
            "direct_minus_host_relay_ppm": (
                (direct_b1_ns - control_ns) * 1_000_000 // control_ns
            ),
            "host_relay_b1_latency_ns": control_ns,
            "scope": "ONE_UNPAIRED_PROCESS_EACH_INFORMATIONAL_ONLY",
        },
        "correctness": {
            "expected_tokens": GENERATED_TOKENS,
            "token_checks": direct[1]["token_checks"] + direct[32]["token_checks"],
            "tokens_exact": True,
        },
        "direct_cohorts": [direct[1], direct[32]],
        "model_sha256": MODEL_SHA256,
        "relay_certificates": [relay[1], relay[32]],
        "relay_executable": relay_binary,
        "schema": "s39-direct-chain-certificate-v1",
        "status": "DIRECT_CHAIN_MECHANICS_PASS_REPEATS_PENDING",
        "transport": {
            "activation_path": "OP15_TO_OP12_WIFI_TCP",
            "host_activation_payload_bytes": 0,
            "host_role": "ADMISSION_BATCHING_RESULT_OWNERSHIP",
            "reservation_binding": "IMPLICIT_SINGLE_CLIENT_MECHANICS_ONLY",
            "route_interface_counters": "NOT_CAPTURED",
        },
        "workers": {"op12": op12, "op15": op15},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        result = build(args.evidence)
        common.atomic_write(args.output, common.canonical_bytes(result))
    except common.BatchEvidenceError as error:
        print(f"S39_DIRECT_EVIDENCE_ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
