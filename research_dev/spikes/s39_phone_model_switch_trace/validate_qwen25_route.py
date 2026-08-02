#!/usr/bin/env python3
"""Validate the bounded Qwen2.5 B32 two-phone route gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


MODEL_SHA256 = "924a4c39ef9fc6c139875ab6771c2e8172a3b40ffec5c720eca69ad7a0edfae7"
OP15_SHARD_SHA256 = "4eb56f404eb8e4e64b63ba0870db3a41016486cc137ce7d65d51969902970a16"
OP12_SHARD_SHA256 = "f741cd302150adb0be9349d9ae257e77f4e7e18be8414e98797b32445ed42737"
PROMPT_TOKENS = [49000]
B1_GENERATED_TOKENS = [25, 576, 8585, 3033, 702, 7228, 429, 432]
GENERATED_TOKENS = [25, 576, 8585, 3033, 702, 7228, 6649, 311]
REQUESTS = 32
STEPS = 8
N_LAYER = 48
N_EMBD = 5120
CUT_LAYER = 32
ROWS = REQUESTS * STEPS
ACTIVATION_BYTES = ROWS * N_EMBD * 4


class GateError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GateError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(raw: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicates)
    except (json.JSONDecodeError, GateError) as error:
        raise GateError(f"{label}: invalid JSON: {error}") from error
    require(type(value) is dict, f"{label}: expected object")
    return value


def read_json(path: Path) -> dict[str, Any]:
    return parse_json(path.read_text(encoding="ascii"), str(path))


def extract_one(path: Path, prefix: str) -> dict[str, Any]:
    records = extract_many(path, prefix)
    require(len(records) == 1, f"{path}: expected one {prefix.strip()} record")
    return records[0]


def extract_many(path: Path, prefix: str) -> list[dict[str, Any]]:
    return [
        parse_json(line[len(prefix):], f"{path}:{index}")
        for index, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        )
        if line.startswith(prefix)
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def exact(value: Any, expected: Any, label: str) -> None:
    require(
        value == expected and type(value) is type(expected),
        f"{label}: expected {expected!r}, got {value!r}",
    )


def validate_context(context: dict[str, Any]) -> None:
    exact(context.get("schema"), "s39-qwen25-route-context-v1", "context.schema")
    exact(context.get("model_sha256"), MODEL_SHA256, "context.model_sha256")
    exact(
        context.get("op15_shard_sha256"),
        OP15_SHARD_SHA256,
        "context.op15_shard_sha256",
    )
    exact(
        context.get("op12_shard_sha256"),
        OP12_SHARD_SHA256,
        "context.op12_shard_sha256",
    )
    exact(context.get("op15_serial"), "3C15AU002CL00000", "context.op15_serial")
    exact(context.get("op12_serial"), "5ae7a43d", "context.op12_serial")
    exact(context.get("op15_layers"), [0, CUT_LAYER], "context.op15_layers")
    exact(context.get("op12_layers"), [CUT_LAYER, N_LAYER], "context.op12_layers")
    exact(context.get("backend"), "GPUOpenCL", "context.backend")
    exact(context.get("driver_batch"), REQUESTS, "context.driver_batch")
    exact(context.get("driver_context"), 64, "context.driver_context")
    exact(
        context.get("activation_path"),
        "OP15_TO_OP12_WIFI_TCP",
        "context.activation_path",
    )
    exact(
        context.get("weight_path"),
        "USB_ADB_BEFORE_SERVICE",
        "context.weight_path",
    )
    for key in (
        "acquisition_utc",
        "base_git_commit",
        "op15_boot_id",
        "op12_boot_id",
        "op15_wifi",
        "op12_wifi",
        "op15_adb_target",
        "op12_adb_target",
        "worker_sha256",
        "relay_sha256",
    ):
        require(
            type(context.get(key)) is str and bool(context[key]),
            f"context.{key}: missing string",
        )


def validate_cuda(records: list[dict[str, Any]], placement: dict[str, Any]) -> None:
    require(len(records) == REQUESTS, "cuda: expected 32 route records")
    for index, record in enumerate(records):
        exact(record.get("status"), "ok", f"cuda[{index}].status")
        exact(record.get("route"), "SERVER_ONLY", f"cuda[{index}].route")
        exact(record.get("request_index"), index, f"cuda[{index}].request_index")
        exact(record.get("batch_size"), REQUESTS, f"cuda[{index}].batch_size")
        exact(
            record.get("prompt_tokens"),
            len(PROMPT_TOKENS),
            f"cuda[{index}].prompt_tokens",
        )
        exact(record.get("generated_tokens"), STEPS, f"cuda[{index}].generated_tokens")
        exact(record.get("token_ids"), GENERATED_TOKENS, f"cuda[{index}].token_ids")

    exact(
        placement.get("schema"),
        "layersplit-scheduled-placement-v2",
        "cuda_placement.schema",
    )
    exact(placement.get("status"), "SCHEDULED_PLACEMENT_OK", "cuda_placement.status")
    exact(placement.get("layer_start"), 0, "cuda_placement.layer_start")
    exact(placement.get("layer_end"), N_LAYER, "cuda_placement.layer_end")
    exact(placement.get("n_layer"), N_LAYER, "cuda_placement.n_layer")
    exact(
        placement.get("missing_buffer_compute_nodes"),
        0,
        "cuda_placement.missing_buffer_compute_nodes",
    )
    require(
        type(placement.get("compute_nodes")) is int
        and placement["compute_nodes"] > 0,
        "cuda_placement.compute_nodes: expected positive integer",
    )
    by_buffer = placement.get("compute_by_buffer_type")
    require(type(by_buffer) is dict, "cuda_placement.compute_by_buffer_type")
    require(
        type(by_buffer.get("CUDA0")) is int and by_buffer["CUDA0"] > 0,
        "cuda_placement: CUDA0 did no work",
    )


def validate_event(event: dict[str, Any], index: int) -> None:
    phase = "prefill" if index == 0 else "decode"
    position = 0 if index == 0 else index
    exact(event.get("batch_size"), REQUESTS, f"event[{index}].batch_size")
    exact(event.get("decode_rows"), 0 if index == 0 else REQUESTS, f"event[{index}].decode_rows")
    exact(event.get("prefill_rows"), REQUESTS if index == 0 else 0, f"event[{index}].prefill_rows")
    exact(event.get("mixed_phase"), False, f"event[{index}].mixed_phase")
    exact(event.get("phases"), [phase] * REQUESTS, f"event[{index}].phases")
    exact(event.get("positions"), [position] * REQUESTS, f"event[{index}].positions")
    exact(event.get("sequence_ids"), list(range(REQUESTS)), f"event[{index}].sequence_ids")
    exact(
        event.get("request_ids"),
        list(range(6001, 6001 + REQUESTS)),
        f"event[{index}].request_ids",
    )
    exact(
        event.get("release_reason"),
        "BATCH_KNEE",
        f"event[{index}].release_reason",
    )


def validate_route_report(report: dict[str, Any]) -> str:
    exact(report.get("schema"), "s39-direct-order-route-v1", "route.schema")
    exact(report.get("order"), "sorted", "route.order")
    exact(report.get("token_checks"), REQUESTS * STEPS, "route.token_checks")
    exact(report.get("permutation"), list(range(REQUESTS)), "route.permutation")

    config = report.get("configuration")
    require(type(config) is dict, "route.configuration")
    exact(config.get("requests"), REQUESTS, "route.configuration.requests")
    exact(config.get("steps"), STEPS, "route.configuration.steps")
    exact(config.get("batch_knee"), REQUESTS, "route.configuration.batch_knee")
    exact(config.get("prompt_tokens"), PROMPT_TOKENS, "route.configuration.prompt_tokens")
    configured_oracle = config.get("expected_tokens")
    if configured_oracle == GENERATED_TOKENS:
        oracle_scope = "MATCHED_B32"
        expected_exact = True
    elif configured_oracle == B1_GENERATED_TOKENS:
        oracle_scope = "MISMATCHED_B1_DIAGNOSTIC"
        expected_exact = False
    else:
        raise GateError("route.configuration.expected_tokens: unknown oracle")
    exact(report.get("tokens_exact"), expected_exact, "route.tokens_exact")
    exact(report.get("verdict"), "PASS" if expected_exact else "FAIL", "route.verdict")
    exact(
        report.get("status"),
        "DIRECT_ORDER_MECHANICS_PASS" if expected_exact else "DIRECT_ORDER_FAIL",
        "route.status",
    )

    worker = report.get("worker")
    require(type(worker) is dict, "route.worker")
    exact(worker.get("layer_start"), 0, "route.worker.layer_start")
    exact(worker.get("layer_end"), N_LAYER, "route.worker.layer_end")
    exact(worker.get("n_layer"), N_LAYER, "route.worker.n_layer")
    exact(worker.get("n_embd"), N_EMBD, "route.worker.n_embd")
    exact(worker.get("file_type"), 2, "route.worker.file_type")
    exact(worker.get("model_sha256"), MODEL_SHA256, "route.worker.model_sha256")
    require(worker.get("max_streams", 0) >= REQUESTS, "route.worker.max_streams")

    events = report.get("batch_events")
    require(type(events) is list and len(events) == STEPS, "route.batch_events")
    for index, event in enumerate(events):
        require(type(event) is dict, f"event[{index}]: expected object")
        validate_event(event, index)

    requests = report.get("requests")
    require(type(requests) is list and len(requests) == REQUESTS, "route.requests")
    for index, request in enumerate(requests):
        require(type(request) is dict, f"request[{index}]: expected object")
        exact(request.get("request_id"), 6001 + index, f"request[{index}].request_id")
        exact(request.get("sequence_id"), index, f"request[{index}].sequence_id")
        exact(request.get("tokens"), GENERATED_TOKENS, f"request[{index}].tokens")
        exact(request.get("slo_met"), True, f"request[{index}].slo_met")

    transport = report.get("transport")
    require(type(transport) is dict, "route.transport")
    exact(
        transport.get("direct_activation_payload_bytes"),
        ACTIVATION_BYTES,
        "route.transport.direct_activation_payload_bytes",
    )
    exact(
        transport.get("host_activation_payload_bytes"),
        0,
        "route.transport.host_activation_payload_bytes",
    )
    exact(
        transport.get("mode"),
        "OP15_TO_OP12_DIRECT_WIFI",
        "route.transport.mode",
    )
    return oracle_scope


def cpu_ops(placement: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    by_op = placement.get("compute_by_op_and_buffer")
    require(type(by_op) is dict, "placement.compute_by_op_and_buffer")
    for op, buffers in by_op.items():
        require(type(buffers) is dict, f"placement.{op}: expected object")
        for buffer, count in buffers.items():
            require(
                type(count) is int and count >= 0,
                f"placement.{op}.{buffer}: invalid count",
            )
            if count > 0 and buffer in {"CPU", "CPU_Mapped"}:
                result.add(op)
    return result


def validate_phone_log(
    path: Path,
    expected_range: list[int],
    context: dict[str, Any],
    context_boot_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    session = extract_one(path, "SESSIONCERT ")
    placement = extract_one(path, "PLACEMENTCERT ")
    exact(session.get("schema"), "ls-stagenet-session-v2", f"{path}.session.schema")
    exact(session.get("session_end"), "STOP", f"{path}.session.session_end")
    exact(session.get("expected_backend"), "GPUOpenCL", f"{path}.session.backend")
    exact(session.get("layer_start"), expected_range[0], f"{path}.session.layer_start")
    exact(session.get("layer_end"), expected_range[1], f"{path}.session.layer_end")
    exact(session.get("n_layer"), N_LAYER, f"{path}.session.n_layer")
    exact(session.get("steps_session"), ROWS, f"{path}.session.steps_session")
    exact(session.get("reset_applied"), False, f"{path}.session.reset_applied")
    exact(
        session.get("device_boot_id"),
        context[context_boot_key],
        f"{path}.session.device_boot_id",
    )
    exact(
        session.get("placement_status"),
        "SCHEDULED_PLACEMENT_OK",
        f"{path}.session.placement_status",
    )
    exact(
        session.get("missing_buffer_compute_nodes"),
        0,
        f"{path}.session.missing_buffer_compute_nodes",
    )
    require(
        type(session.get("worker_pid")) is int and session["worker_pid"] > 0,
        f"{path}.session.worker_pid",
    )
    require(
        re.fullmatch(r"[0-9a-f]{16}", session.get("worker_boot_nonce", "")) is not None,
        f"{path}.session.worker_boot_nonce",
    )

    exact(
        placement.get("schema"),
        "layersplit-scheduled-placement-v2",
        f"{path}.placement.schema",
    )
    exact(
        placement.get("status"),
        "SCHEDULED_PLACEMENT_OK",
        f"{path}.placement.status",
    )
    exact(placement.get("layer_start"), expected_range[0], f"{path}.placement.layer_start")
    exact(placement.get("layer_end"), expected_range[1], f"{path}.placement.layer_end")
    exact(placement.get("n_layer"), N_LAYER, f"{path}.placement.n_layer")
    exact(
        placement.get("missing_buffer_compute_nodes"),
        0,
        f"{path}.placement.missing_buffer_compute_nodes",
    )
    by_buffer = placement.get("compute_by_buffer_type")
    require(type(by_buffer) is dict, f"{path}.placement.compute_by_buffer_type")
    require(
        type(by_buffer.get("OpenCL")) is int and by_buffer["OpenCL"] > 0,
        f"{path}.placement: OpenCL did no work",
    )
    require(
        cpu_ops(placement) <= {"GET_ROWS"},
        f"{path}.placement: undeclared CPU compute",
    )
    return session, placement


def validate_relay(record: dict[str, Any]) -> None:
    exact(record.get("schema"), "ls-stage-direct-relay-v1", "relay.schema")
    exact(record.get("status"), "DIRECT_RELAY_OK", "relay.status")
    exact(record.get("run_rc"), 0, "relay.run_rc")
    exact(record.get("layer_start"), 0, "relay.layer_start")
    exact(record.get("cut_layer"), CUT_LAYER, "relay.cut_layer")
    exact(record.get("layer_end"), N_LAYER, "relay.layer_end")
    exact(record.get("n_layer"), N_LAYER, "relay.n_layer")
    exact(record.get("n_embd"), N_EMBD, "relay.n_embd")
    exact(record.get("file_type"), 2, "relay.file_type")
    exact(record.get("model_sha256"), MODEL_SHA256, "relay.model_sha256")
    exact(record.get("batches"), STEPS, "relay.batches")
    exact(record.get("rows"), ROWS, "relay.rows")
    exact(
        record.get("activation_payload_bytes"),
        ACTIVATION_BYTES,
        "relay.activation_payload_bytes",
    )
    exact(
        record.get("host_activation_payload_bytes"),
        0,
        "relay.host_activation_payload_bytes",
    )


def validate_evidence(evidence: Path) -> dict[str, Any]:
    context = read_json(evidence / "RUN_CONTEXT.json")
    validate_context(context)
    report = read_json(evidence / "b32.json")
    probe_oracle_scope = validate_route_report(report)

    cuda_log = evidence / "cuda_b32.log"
    cuda_records = extract_many(cuda_log, "ROUTEJSON ")
    cuda_placement = extract_one(cuda_log, "PLACEMENTCERT ")
    validate_cuda(cuda_records, cuda_placement)

    head_session, head_placement = validate_phone_log(
        evidence / "op15_head.log",
        [0, CUT_LAYER],
        context,
        "op15_boot_id",
    )
    tail_session, tail_placement = validate_phone_log(
        evidence / "op12_tail.log",
        [CUT_LAYER, N_LAYER],
        context,
        "op12_boot_id",
    )
    relay = extract_one(evidence / "op15_relay.log", "DIRECTCERT ")
    validate_relay(relay)

    require(
        head_session["worker_pid"] != tail_session["worker_pid"]
        or context["op15_boot_id"] != context["op12_boot_id"],
        "phone worker identity collision",
    )

    artifacts = {}
    for name in (
        "RUN_CONTEXT.json",
        "b32.json",
        "cuda_b32.log",
        "op15_head.log",
        "op12_tail.log",
        "op15_relay.log",
    ):
        artifacts[name] = sha256_file(evidence / name)

    tail_text = (evidence / "op12_tail.log").read_text(encoding="utf-8")
    return {
        "schema": "s39-qwen25-route-certificate-v1",
        "status": "QWEN25_ROUTE_B32_EXACT_CORPUS_PENDING",
        "verdict": "PROVISIONAL",
        "scheduler_eligible": False,
        "model_sha256": MODEL_SHA256,
        "layer_route": {
            "op15": [0, CUT_LAYER],
            "op12": [CUT_LAYER, N_LAYER],
        },
        "batch": {
            "requests": REQUESTS,
            "physical_batches": STEPS,
            "rows": ROWS,
            "batch_sizes": [REQUESTS] * STEPS,
        },
        "correctness": {
            "cuda_tokens": GENERATED_TOKENS,
            "phone_checks": REQUESTS * STEPS,
            "phone_tokens_exact": True,
            "probe_oracle_scope": probe_oracle_scope,
            "scope": "ONE_PROMPT",
        },
        "placement": {
            "op15_compute_nodes": head_placement["compute_nodes"],
            "op12_compute_nodes": tail_placement["compute_nodes"],
            "op15_cpu_ops": sorted(cpu_ops(head_placement)),
            "op12_cpu_ops": sorted(cpu_ops(tail_placement)),
        },
        "transport": {
            "activation_bytes": ACTIVATION_BYTES,
            "host_activation_bytes": 0,
            "path": "OP15_TO_OP12_WIFI_TCP",
        },
        "known_limits": {
            "corpus_gate": "NOT_RUN",
            "fresh_process_repeats": "NOT_RUN",
            "energy": "NOT_MEASURED",
            "op12_optional_kernel_compile_failure": (
                "sub_group_shuffle_xor" in tail_text
            ),
        },
        "artifacts": artifacts,
    }


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="ascii", newline="\n") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        certificate = validate_evidence(args.evidence)
        if args.output is not None:
            write_atomic(args.output, certificate)
        print(json.dumps(certificate, sort_keys=True, separators=(",", ":")))
        return 0
    except (GateError, OSError, UnicodeError) as error:
        print(json.dumps({"error": str(error), "verdict": "FAIL"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
