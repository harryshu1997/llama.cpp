#!/usr/bin/env python3
"""Validate a provenance-bound S39 concurrent-delta report."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import phone_cuda_delta_probe as delta
import phone_cuda_handoff_probe as w5


CERT_SCHEMA = "s39-phone-cuda-delta-certificate-v1"
PHYSICAL_GATE_SCHEMA = "s39-phone-cuda-delta-physical-gate-v1"
SESSION_SCHEMA = "ls-stagenet-session-v2"
HERE = Path(__file__).resolve().parent
HEX16 = re.compile(r"[0-9a-f]{16}")


@dataclass(frozen=True)
class PhysicalGate:
    raw_sha256: str
    artifacts: dict[str, str]
    n_layer: int
    placements: dict[str, dict[str, object]]


def read_canonical(path: Path, field: str) -> tuple[dict[str, object], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=delta.strict_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise delta.DeltaError(f"{field}: invalid JSON") from exc
    delta.require(
        type(value) is dict and delta.canonical(value) == raw,
        f"{field}: not canonical",
    )
    return value, raw


def load_physical_gate(
    path: Path,
    contract: delta.DeltaContract,
    base_contract: w5.Contract,
) -> PhysicalGate:
    value, raw = read_canonical(path, "physical_gate")
    root = delta.exact_keys(
        value,
        {
            "artifacts",
            "delta_contract_sha256",
            "n_layer",
            "placements",
            "requirements",
            "schema",
            "status",
        },
        "physical_gate",
    )
    delta.require(
        root["schema"] == PHYSICAL_GATE_SCHEMA,
        "physical_gate: schema mismatch",
    )
    delta.require(
        root["status"] == "FROZEN_BEFORE_CERTIFIED_ACQUISITION",
        "physical_gate: status mismatch",
    )
    delta.require(
        root["delta_contract_sha256"] == contract.raw_sha256,
        "physical_gate: delta contract mismatch",
    )
    delta.require(
        root["n_layer"] == base_contract.n_layer,
        "physical_gate: layer count mismatch",
    )
    delta.require(
        root["requirements"] == [
            "EXACT_DEPLOYED_ARTIFACTS",
            "MATCHED_DEVICE_BOOT_IDENTITIES",
            "REALIZED_PRIMARY_BACKEND_PLACEMENT",
            "NO_UNDECLARED_CPU_COMPUTE",
            "NO_MISSING_COMPUTE_BUFFER",
            "EXECUTED_STEP_COUNT_MATCHES_REPORT",
        ],
        "physical_gate: requirements mismatch",
    )
    artifact_keys = {
        "host_relay_sha256",
        "host_worker_sha256",
        "model_sha256",
        "op12_shard_sha256",
        "op15_relay_sha256",
        "op15_shard_sha256",
        "phone_worker_sha256",
    }
    artifacts = delta.exact_keys(
        root["artifacts"],
        artifact_keys,
        "physical_gate.artifacts",
    )
    for name, digest in artifacts.items():
        delta.checked_digest(digest, f"physical_gate.artifacts.{name}")
    delta.require(
        artifacts["model_sha256"] == base_contract.model_sha256,
        "physical_gate: model mismatch",
    )

    expected = {
        "cuda_head": ("CUDA0", "CUDA0", 0, 30, {"GET_ROWS": ["CUDA_Host"]}),
        "cuda_tail": ("CUDA0", "CUDA0", 30, 48, {"GET_ROWS": ["CUDA_Host"]}),
        "op12": ("GPUOpenCL", "OpenCL", 30, 48, {"GET_ROWS": ["CPU"]}),
        "op15": ("GPUOpenCL", "OpenCL", 0, 30, {"GET_ROWS": ["CPU"]}),
    }
    placements = delta.exact_keys(
        root["placements"],
        set(expected),
        "physical_gate.placements",
    )
    normalized: dict[str, dict[str, object]] = {}
    placement_keys = {
        "allowed_auxiliary",
        "expected_backend",
        "layer_end",
        "layer_start",
        "primary_buffer",
    }
    for name, spec in placements.items():
        route = delta.exact_keys(
            spec,
            placement_keys,
            f"physical_gate.placements.{name}",
        )
        expected_backend, primary_buffer, start, end, auxiliary = expected[name]
        delta.require(
            route == {
                "allowed_auxiliary": auxiliary,
                "expected_backend": expected_backend,
                "layer_end": end,
                "layer_start": start,
                "primary_buffer": primary_buffer,
            },
            f"physical_gate: {name} route mismatch",
        )
        normalized[name] = dict(route)
    return PhysicalGate(
        delta.sha256(raw),
        dict(artifacts),
        base_contract.n_layer,
        normalized,
    )


def validate_run_context(
    value: object,
    contract: delta.DeltaContract,
    base_contract: w5.Contract,
    physical_gate: PhysicalGate,
) -> None:
    root = delta.exact_keys(
        value,
        {
            "acquisition_unix_s",
            "base_contract_sha256",
            "base_git_commit",
            "contract_sha256",
            "cuda",
            "model_sha256",
            "op12",
            "op15",
            "physical_gate_sha256",
            "run_id",
            "schema",
            "sources",
        },
        "run_context",
    )
    delta.require(
        root["schema"] == "s39-phone-cuda-delta-context-v1",
        "run_context: schema mismatch",
    )
    delta.require(
        delta.is_int(root["acquisition_unix_s"])
        and root["acquisition_unix_s"] > 0,
        "run_context: invalid acquisition time",
    )
    delta.require(
        type(root["base_git_commit"]) is str
        and len(root["base_git_commit"]) == 40
        and all(char in "0123456789abcdef" for char in root["base_git_commit"]),
        "run_context: invalid base commit",
    )
    delta.require(
        root["contract_sha256"] == contract.raw_sha256,
        "run_context: contract mismatch",
    )
    delta.require(
        root["base_contract_sha256"] == base_contract.raw_sha256,
        "run_context: base contract mismatch",
    )
    delta.require(
        root["model_sha256"] == base_contract.model_sha256,
        "run_context: model mismatch",
    )
    delta.require(
        root["physical_gate_sha256"] == physical_gate.raw_sha256,
        "run_context: physical gate mismatch",
    )
    delta.checked_digest(root["run_id"], "run_context.run_id")

    cuda = delta.exact_keys(
        root["cuda"],
        {
            "boot_id",
            "device",
            "head_port",
            "relay_port",
            "relay_sha256",
            "tail_port",
            "worker_sha256",
        },
        "run_context.cuda",
    )
    delta.require(cuda["device"] == "CUDA0", "run_context: CUDA device mismatch")
    delta.require(
        type(cuda["boot_id"]) is str and cuda["boot_id"] != "",
        "run_context: invalid CUDA boot ID",
    )
    for field in ("head_port", "relay_port", "tail_port"):
        delta.require(
            delta.is_int(cuda[field]) and 0 < cuda[field] <= 65535,
            f"run_context.cuda.{field}: invalid port",
        )
    for field in ("relay_sha256", "worker_sha256"):
        delta.checked_digest(cuda[field], f"run_context.cuda.{field}")
    delta.require(
        cuda["relay_sha256"]
        == physical_gate.artifacts["host_relay_sha256"]
        and cuda["worker_sha256"]
        == physical_gate.artifacts["host_worker_sha256"],
        "run_context: CUDA artifact mismatch",
    )

    expected_devices = {
        "op15": ([0, 30], True),
        "op12": ([30, 48], False),
    }
    for name, (layers, has_relay) in expected_devices.items():
        keys = {
            "adb_target",
            "boot_id",
            "layers",
            "shard_sha256",
            "wifi",
            "worker_sha256",
        }
        if has_relay:
            keys.add("relay_sha256")
        device = delta.exact_keys(
            root[name],
            keys,
            f"run_context.{name}",
        )
        delta.require(
            type(device["adb_target"]) is str and device["adb_target"] != "",
            f"run_context.{name}: invalid ADB target",
        )
        delta.require(
            type(device["boot_id"]) is str and device["boot_id"] != "",
            f"run_context.{name}: invalid boot ID",
        )
        delta.require(
            device["layers"] == layers,
            f"run_context.{name}: layer range mismatch",
        )
        delta.require(
            type(device["wifi"]) is str and device["wifi"] != "",
            f"run_context.{name}: invalid WiFi endpoint",
        )
        for field in ("shard_sha256", "worker_sha256"):
            delta.checked_digest(
                device[field],
                f"run_context.{name}.{field}",
            )
        if has_relay:
            delta.checked_digest(
                device["relay_sha256"],
                f"run_context.{name}.relay_sha256",
            )
    delta.require(
        root["op15"]["worker_sha256"]
        == physical_gate.artifacts["phone_worker_sha256"]
        and root["op12"]["worker_sha256"]
        == physical_gate.artifacts["phone_worker_sha256"]
        and root["op15"]["relay_sha256"]
        == physical_gate.artifacts["op15_relay_sha256"]
        and root["op15"]["shard_sha256"]
        == physical_gate.artifacts["op15_shard_sha256"]
        and root["op12"]["shard_sha256"]
        == physical_gate.artifacts["op12_shard_sha256"],
        "run_context: phone artifact mismatch",
    )

    expected_sources = {
        "async_pipeline.py": (
            HERE.parent / "s22_slo_overlap_pipeline" / "async_pipeline.py"
        ),
        "phone_cuda_delta_probe.py": HERE / "phone_cuda_delta_probe.py",
        "phone_cuda_handoff_probe.py": HERE / "phone_cuda_handoff_probe.py",
        "qwen25_quality_probe.py": HERE / "qwen25_quality_probe.py",
        "run_w6_delta_gate.sh": HERE / "run_w6_delta_gate.sh",
        "stage_v3_client.py": (
            HERE.parent / "s22_slo_overlap_pipeline" / "stage_v3_client.py"
        ),
        "validate_phone_cuda_delta.py": (
            HERE / "validate_phone_cuda_delta.py"
        ),
    }
    sources = delta.exact_keys(
        root["sources"],
        set(expected_sources),
        "run_context.sources",
    )
    for name, path in expected_sources.items():
        digest = delta.checked_digest(
            sources[name],
            f"run_context.sources.{name}",
        )
        delta.require(path.is_file(), f"run_context: missing source {name}")
        delta.require(
            delta.sha256(path.read_bytes()) == digest,
            f"run_context: source digest mismatch for {name}",
        )


def parse_session_certificate(path: Path, field: str) -> tuple[dict[str, object], str]:
    delta.require(
        path.is_file() and not path.is_symlink(),
        f"{field}: invalid log path",
    )
    raw = path.read_bytes()
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise delta.DeltaError(f"{field}: log is not UTF-8") from exc
    prefix = "SESSIONCERT "
    records = [line[len(prefix):] for line in lines if line.startswith(prefix)]
    delta.require(len(records) == 1, f"{field}: expected one session certificate")
    try:
        value = json.loads(records[0], object_pairs_hook=delta.strict_object)
    except json.JSONDecodeError as exc:
        raise delta.DeltaError(f"{field}: invalid session certificate") from exc
    keys = {
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
    return delta.exact_keys(value, keys, field), delta.sha256(raw)


def validate_placement_logs(
    paths: dict[str, Path],
    run_context: dict[str, object],
    report: dict[str, object],
    gate: PhysicalGate,
) -> dict[str, object]:
    delta.require(
        set(paths) == set(gate.placements),
        "placement: log set mismatch",
    )
    metrics = report["metrics"]
    expected_steps = {
        "op15": metrics["phone_snapshot"]["rows"]
        + metrics["phone_delta"]["rows"],
        "op12": metrics["phone_snapshot"]["rows"]
        + metrics["phone_delta"]["rows"],
        "cuda_head": (
            metrics["cuda_snapshot"]["rows"]
            + metrics["cuda_delta"]["rows"]
            + metrics["cuda_continuation"]["rows"]
            + metrics["cuda_control"]["rows"]
        ),
        "cuda_tail": (
            metrics["cuda_snapshot"]["rows"]
            + metrics["cuda_delta"]["rows"]
            + metrics["cuda_continuation"]["rows"]
            + metrics["cuda_control"]["rows"]
        ),
    }
    expected_boot = {
        "op15": run_context["op15"]["boot_id"],
        "op12": run_context["op12"]["boot_id"],
        "cuda_head": run_context["cuda"]["boot_id"],
        "cuda_tail": run_context["cuda"]["boot_id"],
    }
    summaries: dict[str, object] = {}
    for name in sorted(paths):
        certificate, log_sha256 = parse_session_certificate(
            paths[name],
            f"placement.{name}",
        )
        spec = gate.placements[name]
        for numeric in (
            "layer_end",
            "layer_start",
            "missing_buffer_compute_nodes",
            "n_layer",
            "proto_version",
            "session_id",
            "steps_session",
            "steps_total",
            "worker_pid",
        ):
            delta.require(
                delta.is_int(certificate[numeric]),
                f"placement.{name}: invalid {numeric}",
            )
        delta.require(
            certificate["schema"] == SESSION_SCHEMA
            and certificate["proto_version"] == 2
            and certificate["session_id"] == 1
            and certificate["session_end"] == "STOP"
            and certificate["reset_applied"] is False,
            f"placement.{name}: session identity mismatch",
        )
        delta.require(
            certificate["device_boot_id"] == expected_boot[name],
            f"placement.{name}: device boot mismatch",
        )
        delta.require(
            certificate["expected_backend"] == spec["expected_backend"]
            and certificate["layer_start"] == spec["layer_start"]
            and certificate["layer_end"] == spec["layer_end"]
            and certificate["n_layer"] == gate.n_layer,
            f"placement.{name}: route mismatch",
        )
        delta.require(
            certificate["placement_status"] == "SCHEDULED_PLACEMENT_OK"
            and certificate["missing_buffer_compute_nodes"] == 0,
            f"placement.{name}: placement failed",
        )
        delta.require(
            delta.is_int(certificate["worker_pid"])
            and certificate["worker_pid"] > 0
            and type(certificate["worker_boot_nonce"]) is str
            and HEX16.fullmatch(certificate["worker_boot_nonce"]) is not None,
            f"placement.{name}: worker identity mismatch",
        )
        delta.require(
            certificate["steps_session"] == expected_steps[name]
            and certificate["steps_total"] == expected_steps[name],
            f"placement.{name}: executed step count mismatch",
        )
        compute = certificate["compute_by_op_and_buffer"]
        delta.require(
            type(compute) is dict and bool(compute),
            f"placement.{name}: empty compute placement",
        )
        by_buffer: dict[str, int] = {}
        primary_nodes = 0
        auxiliary = spec["allowed_auxiliary"]
        for op, buffers in compute.items():
            delta.require(
                type(op) is str
                and op != ""
                and type(buffers) is dict
                and bool(buffers),
                f"placement.{name}: invalid operation placement",
            )
            for buffer, count in buffers.items():
                delta.require(
                    type(buffer) is str
                    and buffer != ""
                    and delta.is_int(count)
                    and count > 0,
                    f"placement.{name}: invalid compute count",
                )
                allowed = (
                    buffer == spec["primary_buffer"]
                    or buffer in auxiliary.get(op, [])
                )
                delta.require(
                    allowed,
                    f"placement.{name}: undeclared compute buffer",
                )
                by_buffer[buffer] = by_buffer.get(buffer, 0) + count
                if buffer == spec["primary_buffer"]:
                    primary_nodes += count
        delta.require(
            primary_nodes > 0,
            f"placement.{name}: primary backend was unused",
        )
        summaries[name] = {
            "compute_by_buffer": dict(sorted(by_buffer.items())),
            "layer_end": spec["layer_end"],
            "layer_start": spec["layer_start"],
            "log_sha256": log_sha256,
            "primary_buffer": spec["primary_buffer"],
            "steps": expected_steps[name],
            "worker_boot_nonce": certificate["worker_boot_nonce"],
            "worker_pid": certificate["worker_pid"],
        }
    return summaries


def write_atomic(path: Path, value: dict[str, object]) -> None:
    raw = delta.canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--journal-dir", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=delta.DEFAULT_CONTRACT,
    )
    parser.add_argument(
        "--base-contract",
        type=Path,
        default=delta.DEFAULT_BASE_CONTRACT,
    )
    parser.add_argument("--physical-gate", type=Path, required=True)
    parser.add_argument("--run-context", type=Path, required=True)
    parser.add_argument("--op15-log", type=Path, required=True)
    parser.add_argument("--op12-log", type=Path, required=True)
    parser.add_argument("--cuda-head-log", type=Path, required=True)
    parser.add_argument("--cuda-tail-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")

    base_contract = w5.load_contract(args.base_contract)
    contract = delta.load_contract(args.contract, base_contract)
    physical_gate = load_physical_gate(
        args.physical_gate,
        contract,
        base_contract,
    )
    report, report_raw = read_canonical(args.report, "report")
    run_context, context_raw = read_canonical(
        args.run_context,
        "run_context",
    )
    validate_run_context(
        run_context,
        contract,
        base_contract,
        physical_gate,
    )
    delta.validate_report(
        report,
        contract,
        base_contract,
        args.journal_dir,
        run_context["run_id"],
    )
    placement = validate_placement_logs(
        {
            "cuda_head": args.cuda_head_log,
            "cuda_tail": args.cuda_tail_log,
            "op12": args.op12_log,
            "op15": args.op15_log,
        },
        run_context,
        report,
        physical_gate,
    )
    certificate = {
        "base_contract_sha256": base_contract.raw_sha256,
        "contract_sha256": contract.raw_sha256,
        "journal_summary_sha256": delta.sha256(
            delta.canonical(report["journal"])
        ),
        "physical_gate_sha256": physical_gate.raw_sha256,
        "placement": placement,
        "report_sha256": delta.sha256(report_raw),
        "run_context_sha256": delta.sha256(context_raw),
        "scheduler_eligible": False,
        "schema": CERT_SCHEMA,
        "scope": "MECHANICS_ONLY",
        "status": report["status"],
    }
    write_atomic(args.output, certificate)
    print(delta.canonical(certificate).decode("ascii"), end="")
    return 0 if report["status"] == "CONCURRENT_DELTA_MECHANICS_PASS" else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        delta.DeltaError,
        w5.HandoffError,
        ValueError,
    ) as exc:
        print(delta.canonical({
            "error": str(exc),
            "status": "CONCURRENT_DELTA_VALIDATION_ERROR",
        }).decode("ascii"), end="")
        raise SystemExit(2)
