#!/usr/bin/env python3
"""Validate the real mixed-phase direct phone-chain evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import summarize_qwen_batch_wifi as common


DEFAULT_EVIDENCE = HERE / "results" / "w2_direct_mixed"
DEFAULT_OUTPUT = DEFAULT_EVIDENCE / "direct_mixed_certificate.json"

MODEL_SHA256 = "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0"
PROMPT_TOKENS = [785, 6722, 315, 9625, 374]
GENERATED_TOKENS = [12095, 13, 3555, 374, 279, 6722, 315, 279]
COHORT_SIZE = 16
N_EMBD = 5120
N_LAYER = 40
CUT_LAYER = 30
TOTAL_ROWS = 384
ROW_BYTES = N_EMBD * 4
BATCH_SIZES = [80, 96, 32, 32, 32, 32, 32, 32, 16]

REPORT_KEYS = {
    "batch_events",
    "batch_summary",
    "configuration",
    "latency_ms",
    "requests",
    "route_id",
    "schema",
    "scope",
    "status",
    "token_checks",
    "tokens_exact",
    "transport",
    "verdict",
    "worker",
}
CONFIG_KEYS = {
    "batch_knee",
    "cohort_size",
    "expected_tokens",
    "gather_us",
    "prompt_tokens",
    "queue_depth",
    "slo_ms",
    "steps",
}
EVENT_KEYS = {
    "batch_size",
    "compute_us",
    "decode_rows",
    "max_queue_us",
    "mixed_phase",
    "phases",
    "positions",
    "prefill_rows",
    "priorities",
    "release_reason",
    "request_ids",
    "route_epochs",
    "sequence_ids",
}
REQUEST_KEYS = {
    "cohort",
    "elapsed_ms",
    "request_id",
    "route_epoch",
    "sequence_id",
    "slo_met",
    "tokens",
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


def expected_event(index: int) -> dict[str, Any]:
    seed_ids = [4001 + item for item in range(COHORT_SIZE)]
    new_ids = [5001 + item for item in range(COHORT_SIZE)]
    seed_seqs = list(range(COHORT_SIZE))
    new_seqs = list(range(COHORT_SIZE, 2 * COHORT_SIZE))
    if index == 0:
        return {
            "decode_rows": 0,
            "mixed_phase": False,
            "phases": ["prefill"] * 80,
            "positions": list(range(5)) * COHORT_SIZE,
            "prefill_rows": 80,
            "priorities": [2] * 80,
            "release_reason": "DEADLINE",
            "request_ids": [
                request_id
                for request_id in seed_ids
                for _ in range(5)
            ],
            "route_epochs": [1] * 80,
            "sequence_ids": [
                sequence_id
                for sequence_id in seed_seqs
                for _ in range(5)
            ],
        }
    if index == 1:
        return {
            "decode_rows": 16,
            "mixed_phase": True,
            "phases": ["decode"] * 16 + ["prefill"] * 80,
            "positions": [5] * 16 + list(range(5)) * 16,
            "prefill_rows": 80,
            "priorities": [0] * 16 + [2] * 80,
            "release_reason": "BATCH_KNEE",
            "request_ids": seed_ids
            + [
                request_id
                for request_id in new_ids
                for _ in range(5)
            ],
            "route_epochs": [1] * 96,
            "sequence_ids": seed_seqs
            + [
                sequence_id
                for sequence_id in new_seqs
                for _ in range(5)
            ],
        }
    if 2 <= index <= 7:
        offset = index - 2
        return {
            "decode_rows": 32,
            "mixed_phase": False,
            "phases": ["decode"] * 32,
            "positions": [6 + offset] * 16 + [5 + offset] * 16,
            "prefill_rows": 0,
            "priorities": [0] * 32,
            "release_reason": "DEADLINE",
            "request_ids": seed_ids + new_ids,
            "route_epochs": [1] * 32,
            "sequence_ids": seed_seqs + new_seqs,
        }
    return {
        "decode_rows": 16,
        "mixed_phase": False,
        "phases": ["decode"] * 16,
        "positions": [11] * 16,
        "prefill_rows": 0,
        "priorities": [0] * 16,
        "release_reason": "DEADLINE",
        "request_ids": new_ids,
        "route_epochs": [1] * 16,
        "sequence_ids": new_seqs,
    }


def validate_report(raw: bytes) -> dict[str, Any]:
    report = common.require_keys(
        common.parse_json(raw, "mixed_report"),
        REPORT_KEYS,
        "mixed_report",
    )
    expected_scalars = {
        "route_id": "s39-qwen-direct-cut30-mixed16x16",
        "schema": "s39-direct-mixed-route-v1",
        "status": "DIRECT_MIXED_BATCH_MECHANICS_PASS",
        "token_checks": 256,
        "tokens_exact": True,
        "verdict": "PASS",
    }
    for name, value in expected_scalars.items():
        exact(report[name], value, f"mixed_report.{name}")

    config = common.require_keys(
        report["configuration"],
        CONFIG_KEYS,
        "mixed_report.configuration",
    )
    expected_config = {
        "batch_knee": 96,
        "cohort_size": COHORT_SIZE,
        "expected_tokens": GENERATED_TOKENS,
        "gather_us": 50000,
        "prompt_tokens": PROMPT_TOKENS,
        "queue_depth": 256,
        "steps": len(GENERATED_TOKENS),
    }
    for name, value in expected_config.items():
        exact(config[name], value, f"mixed_report.configuration.{name}")
    common.require_number(
        config["slo_ms"],
        "mixed_report.configuration.slo_ms",
        Decimal(1),
    )

    worker = common.require_keys(
        report["worker"],
        common.WORKER_KEYS,
        "mixed_report.worker",
    )
    expected_worker = {
        "file_type": 15,
        "layer_end": N_LAYER,
        "layer_start": 0,
        "max_streams": 32,
        "model_sha256": MODEL_SHA256,
        "n_embd": N_EMBD,
        "n_layer": N_LAYER,
    }
    for name, value in expected_worker.items():
        exact(worker[name], value, f"mixed_report.worker.{name}")
    for name in ("capabilities", "n_batch", "n_ctx_seq", "n_ubatch"):
        common.require_int(worker[name], f"mixed_report.worker.{name}", 1)
    common.require(
        worker["n_batch"] >= 96
        and worker["n_ubatch"] >= 96
        and worker["capabilities"] & 16,
        "mixed_report.worker.capacity",
    )

    events_value = report["batch_events"]
    common.require(
        isinstance(events_value, list)
        and len(events_value) == len(BATCH_SIZES),
        "mixed_report.batch_events",
    )
    events = []
    for index, candidate in enumerate(events_value):
        field = f"mixed_report.batch_events[{index}]"
        event = common.require_keys(candidate, EVENT_KEYS, field)
        exact(event["batch_size"], BATCH_SIZES[index], f"{field}.batch_size")
        common.require_int(event["compute_us"], f"{field}.compute_us", 1)
        common.require_int(event["max_queue_us"], f"{field}.max_queue_us")
        expected = expected_event(index)
        for name, value in expected.items():
            exact(event[name], value, f"{field}.{name}")
        common.require(
            event["decode_rows"] + event["prefill_rows"]
            == event["batch_size"],
            f"{field}.row_conservation",
        )
        events.append(event)

    summary = common.require_keys(
        report["batch_summary"],
        {"batch_count", "batch_sizes", "max_batch", "mixed_batch"},
        "mixed_report.batch_summary",
    )
    exact(
        summary["batch_count"],
        len(BATCH_SIZES),
        "mixed_report.batch_summary.batch_count",
    )
    exact(
        summary["batch_sizes"],
        BATCH_SIZES,
        "mixed_report.batch_summary.batch_sizes",
    )
    exact(summary["max_batch"], 96, "mixed_report.batch_summary.max_batch")
    exact(
        summary["mixed_batch"],
        {
            "batch_index": 1,
            "decode_rows": 16,
            "prefill_rows": 80,
            "release_reason": "BATCH_KNEE",
        },
        "mixed_report.batch_summary.mixed_batch",
    )

    requests_value = report["requests"]
    common.require(
        isinstance(requests_value, list) and len(requests_value) == 32,
        "mixed_report.requests",
    )
    elapsed = []
    for index, candidate in enumerate(requests_value):
        field = f"mixed_report.requests[{index}]"
        request = common.require_keys(candidate, REQUEST_KEYS, field)
        seed = index < COHORT_SIZE
        expected = {
            "cohort": "seed_decode" if seed else "new_prefill",
            "request_id": (
                4001 + index
                if seed
                else 5001 + index - COHORT_SIZE
            ),
            "route_epoch": 1,
            "sequence_id": index,
            "slo_met": True,
            "tokens": GENERATED_TOKENS,
        }
        for name, value in expected.items():
            exact(request[name], value, f"{field}.{name}")
        elapsed.append(
            common.require_number(
                request["elapsed_ms"],
                f"{field}.elapsed_ms",
                Decimal(0),
            )
        )

    latency = common.require_keys(
        report["latency_ms"],
        {"max", "p50", "p95"},
        "mixed_report.latency_ms",
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
                f"mixed_report.latency_ms.{name}",
                Decimal(0),
            )
            == value,
            f"mixed_report.latency_ms.{name}",
        )

    exact(
        report["transport"],
        {
            "direct_activation_payload_bytes": TOTAL_ROWS * ROW_BYTES,
            "host_activation_payload_bytes": 0,
            "mode": "OP15_TO_OP12_DIRECT_WIFI",
            "weight_provisioning": "USB_BEFORE_SERVICE",
        },
        "mixed_report.transport",
    )
    exact(
        report["scope"],
        {
            "continuous_admission": "MECHANICS_ONLY",
            "inter_stage_overlap": "NOT_IMPLEMENTED",
            "phone_energy": "UNKNOWN",
            "throughput_gain": "NOT_CLAIMED",
        },
        "mixed_report.scope",
    )
    return {
        "artifact_sha256": common.sha256(raw),
        "batch_compute_us": [event["compute_us"] for event in events],
        "batch_sizes": BATCH_SIZES,
        "direct_activation_payload_bytes": TOTAL_ROWS * ROW_BYTES,
        "host_activation_payload_bytes": 0,
        "latency_max_ns": common.decimal_ms_to_ns(
            latency["max"],
            "mixed_report.latency_ms.max",
        ),
        "mixed_batch_index": 1,
        "mixed_decode_rows": 16,
        "mixed_prefill_rows": 80,
        "token_checks": 256,
    }


def validate_relay(raw: bytes) -> dict[str, Any]:
    records = common.extract_prefixed_records(
        raw,
        b"DIRECTCERT ",
        "relay",
    )
    common.require(len(records) == 1, "relay: expected one DIRECTCERT")
    cert = common.require_keys(records[0], DIRECT_CERT_KEYS, "relay")
    expected = {
        "activation_payload_bytes": TOTAL_ROWS * ROW_BYTES,
        "batches": len(BATCH_SIZES),
        "cut_layer": CUT_LAYER,
        "file_type": 15,
        "head_endpoint": "127.0.0.1:39315",
        "host_activation_payload_bytes": 0,
        "layer_end": N_LAYER,
        "layer_start": 0,
        "model_sha256": MODEL_SHA256,
        "n_embd": N_EMBD,
        "n_layer": N_LAYER,
        "rows": TOTAL_ROWS,
        "run_rc": 0,
        "schema": "ls-stage-direct-relay-v1",
        "status": "DIRECT_RELAY_OK",
        "tail_endpoint": "172.20.59.72:39312",
    }
    for name, value in expected.items():
        exact(cert[name], value, f"relay.{name}")
    return {
        "artifact_sha256": common.sha256(raw),
        "batches": cert["batches"],
        "rows": cert["rows"],
    }


def validate_worker(
    raw: bytes,
    *,
    field: str,
    layer_start: int,
    layer_end: int,
    role: str,
    mode: str,
    allow_cpu_get_rows: bool,
) -> dict[str, Any]:
    sessions = common.extract_prefixed_records(
        raw,
        b"SESSIONCERT ",
        f"{field}.session",
    )
    common.require(len(sessions) == 1, f"{field}: expected one session")
    session = common.require_keys(
        sessions[0],
        common.SESSION_KEYS,
        f"{field}.session",
    )
    expected_session = {
        "expected_backend": "GPUOpenCL",
        "layer_end": layer_end,
        "layer_start": layer_start,
        "missing_buffer_compute_nodes": 0,
        "n_layer": N_LAYER,
        "placement_status": "SCHEDULED_PLACEMENT_OK",
        "proto_version": 2,
        "reset_applied": False,
        "schema": "ls-stagenet-session-v2",
        "session_end": "STOP",
        "session_id": 1,
        "steps_session": TOTAL_ROWS,
        "steps_total": TOTAL_ROWS,
    }
    for name, value in expected_session.items():
        exact(session[name], value, f"{field}.session.{name}")
    worker_pid = common.require_int(
        session["worker_pid"],
        f"{field}.session.worker_pid",
        1,
    )
    for name in ("device_boot_id", "worker_boot_nonce"):
        common.require(
            isinstance(session[name], str) and bool(session[name]),
            f"{field}.session.{name}",
        )
    session_nodes = common.validate_op_map(
        session["compute_by_op_and_buffer"],
        f"{field}.session.compute",
        allow_cpu_get_rows,
    )
    common.require(session_nodes > 0, f"{field}.session.compute")

    placements = common.extract_prefixed_records(
        raw,
        b"PLACEMENTCERT ",
        f"{field}.placement",
    )
    common.require(
        len(placements) == 1,
        f"{field}: expected one placement",
    )
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
        "n_layer": N_LAYER,
        "pid": worker_pid,
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
    op_nodes = common.validate_op_map(
        placement["compute_by_op_and_buffer"],
        f"{field}.placement.compute",
        allow_cpu_get_rows,
    )
    exact(
        placement["compute_by_op_and_buffer"],
        session["compute_by_op_and_buffer"],
        f"{field}.session_placement_compute",
    )
    common.require(op_nodes == compute_nodes, f"{field}.compute_nodes")

    derived_backends: dict[str, int] = {}
    derived_ops: dict[str, int] = {}
    for op, backends in placement["compute_by_op_and_buffer"].items():
        derived_ops[op] = sum(backends.values())
        for backend, count in backends.items():
            derived_backends[backend] = (
                derived_backends.get(backend, 0) + count
            )
    exact(
        placement["compute_by_buffer_type"],
        derived_backends,
        f"{field}.placement.compute_by_buffer_type",
    )
    exact(
        placement["compute_by_op"],
        derived_ops,
        f"{field}.placement.compute_by_op",
    )
    common.require(
        derived_backends.get("OpenCL", 0) > 0,
        f"{field}.placement.OpenCL",
    )
    common.require_int(placement["copy_nodes"], f"{field}.copy_nodes")
    common.require_int(
        placement["metadata_nodes"],
        f"{field}.metadata_nodes",
    )
    copy_by_buffer = placement["copy_by_buffer_type"]
    common.require(
        isinstance(copy_by_buffer, dict)
        and all(
            isinstance(name, str)
            and bool(name)
            and common.is_int(count)
            and count >= 0
            for name, count in copy_by_buffer.items()
        ),
        f"{field}.copy_by_buffer_type",
    )
    common.require(
        sum(copy_by_buffer.values()) == placement["copy_nodes"],
        f"{field}.copy_nodes",
    )
    return {
        "artifact_sha256": common.sha256(raw),
        "device_boot_id": session["device_boot_id"],
        "opencl_compile_fallback_observed": b"kernel compile error" in raw,
        "opencl_compute_nodes": derived_backends["OpenCL"],
        "worker_boot_nonce": session["worker_boot_nonce"],
        "worker_pid": worker_pid,
    }


def build(evidence_dir: Path = DEFAULT_EVIDENCE) -> dict[str, Any]:
    report = validate_report(
        common.read_bytes(
            evidence_dir / "mixed16x16.json",
            "mixed_report",
        )
    )
    relay = validate_relay(
        common.read_bytes(evidence_dir / "op15_relay.log", "relay")
    )
    op15 = validate_worker(
        common.read_bytes(evidence_dir / "op15_head.log", "op15"),
        field="op15",
        layer_start=0,
        layer_end=CUT_LAYER,
        role="phone_stage",
        mode="stagenet",
        allow_cpu_get_rows=True,
    )
    op12 = validate_worker(
        common.read_bytes(evidence_dir / "op12_tail.log", "op12"),
        field="op12",
        layer_start=CUT_LAYER,
        layer_end=N_LAYER,
        role="host_tail_v3",
        mode="tailv3",
        allow_cpu_get_rows=False,
    )
    exact(
        relay["rows"],
        sum(report["batch_sizes"]),
        "report_relay.rows",
    )
    return {
        "correctness": {
            "expected_tokens": GENERATED_TOKENS,
            "token_checks": report["token_checks"],
            "tokens_exact": True,
        },
        "execution": report,
        "model_sha256": MODEL_SHA256,
        "relay": relay,
        "schema": "s39-direct-mixed-certificate-v1",
        "status": "DIRECT_MIXED_BATCH_MECHANICS_PASS",
        "scope": {
            "continuous_admission": "ONE_DETERMINISTIC_COHORT",
            "inter_stage_overlap": "NOT_IMPLEMENTED",
            "latency_comparison": "NOT_AUTHORIZED",
            "phone_energy": "UNKNOWN",
            "throughput_gain": "NOT_CLAIMED",
        },
        "transport": {
            "activation_path": "OP15_TO_OP12_WIFI_TCP",
            "host_activation_payload_bytes": 0,
            "weight_path": "USB_BEFORE_SERVICE",
        },
        "workers": {"op12": op12, "op15": op15},
    }


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    raw = common.canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        result = build(args.evidence)
        write_atomic(args.output, result)
    except (OSError, common.BatchEvidenceError, ValueError) as error:
        print(f"S39_DIRECT_MIXED_EVIDENCE_ERROR: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(result, sort_keys=True, separators=(",", ":")),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
