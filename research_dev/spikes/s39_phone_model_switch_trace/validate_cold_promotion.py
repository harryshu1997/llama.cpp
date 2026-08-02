#!/usr/bin/env python3
"""Validate the matched W7 cold-promotion treatment and control."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cuda_cold_control_probe as control
import phone_cuda_cold_promotion_probe as promotion
import phone_cuda_delta_probe as w6
import phone_cuda_handoff_probe as w5
import prepare_phone_route as preparation
import validate_phone_cuda_delta as physical


SCHEMA = "s39-cold-promotion-certificate-v1"
CONTEXT_SCHEMA = "s39-cold-promotion-context-v1"
HERE = Path(__file__).resolve().parent


def read_canonical(path: Path, field: str) -> tuple[dict[str, object], bytes]:
    return w6.read_canonical(path, field)


def validate_context(
    value: object,
    contract: promotion.PromotionContract,
    base_contract: w5.Contract,
    delta_contract: w6.DeltaContract,
    gate: physical.PhysicalGate,
) -> None:
    root = w6.exact_keys(
        value,
        {
            "acquisition_unix_s",
            "base_contract_sha256",
            "base_git_commit",
            "contract_sha256",
            "cuda",
            "delta_contract_sha256",
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
    promotion.require(root["schema"] == CONTEXT_SCHEMA, "run_context: schema")
    promotion.require(
        w6.is_int(root["acquisition_unix_s"])
        and root["acquisition_unix_s"] > 0,
        "run_context: acquisition time",
    )
    promotion.require(
        type(root["base_git_commit"]) is str
        and len(root["base_git_commit"]) == 40
        and all(char in "0123456789abcdef" for char in root["base_git_commit"]),
        "run_context: base commit",
    )
    promotion.require(
        root["contract_sha256"] == contract.raw_sha256
        and root["base_contract_sha256"] == base_contract.raw_sha256
        and root["delta_contract_sha256"] == delta_contract.raw_sha256
        and root["physical_gate_sha256"] == gate.raw_sha256
        and root["model_sha256"] == base_contract.model_sha256,
        "run_context: contract identity",
    )
    w6.checked_digest(root["run_id"], "run_context.run_id")

    cuda = w6.exact_keys(
        root["cuda"],
        {
            "boot_id",
            "control_ports",
            "device",
            "relay_sha256",
            "treatment_ports",
            "worker_sha256",
        },
        "run_context.cuda",
    )
    promotion.require(
        cuda["device"] == "CUDA0"
        and type(cuda["boot_id"]) is str
        and cuda["boot_id"] != "",
        "run_context: CUDA identity",
    )
    promotion.require(
        cuda["worker_sha256"] == gate.artifacts["host_worker_sha256"]
        and cuda["relay_sha256"] == gate.artifacts["host_relay_sha256"],
        "run_context: CUDA artifacts",
    )
    for phase in ("control_ports", "treatment_ports"):
        ports = w6.exact_keys(
            cuda[phase],
            {"head", "relay", "tail"},
            f"run_context.cuda.{phase}",
        )
        promotion.require(
            all(
                w6.is_int(port) and 0 < port <= 65535
                for port in ports.values()
            )
            and len(set(ports.values())) == 3,
            f"run_context.cuda.{phase}: ports",
        )

    device_keys = {
        "adb_target",
        "boot_id",
        "layers",
        "shard_sha256",
        "wifi",
        "worker_sha256",
    }
    op12 = w6.exact_keys(root["op12"], device_keys, "run_context.op12")
    op15 = w6.exact_keys(
        root["op15"],
        device_keys | {"relay_sha256"},
        "run_context.op15",
    )
    for name, device, layers in (
        ("op12", op12, [30, 48]),
        ("op15", op15, [0, 30]),
    ):
        promotion.require(
            type(device["adb_target"]) is str
            and device["adb_target"] != ""
            and type(device["boot_id"]) is str
            and device["boot_id"] != ""
            and type(device["wifi"]) is str
            and device["wifi"] != ""
            and device["layers"] == layers,
            f"run_context.{name}: identity",
        )
    promotion.require(
        op12["worker_sha256"] == gate.artifacts["phone_worker_sha256"]
        and op15["worker_sha256"] == gate.artifacts["phone_worker_sha256"]
        and op12["shard_sha256"] == gate.artifacts["op12_shard_sha256"]
        and op15["shard_sha256"] == gate.artifacts["op15_shard_sha256"]
        and op15["relay_sha256"] == gate.artifacts["op15_relay_sha256"],
        "run_context: phone artifacts",
    )

    expected_sources = {
        "async_pipeline.py": (
            HERE.parent / "s22_slo_overlap_pipeline" / "async_pipeline.py"
        ),
        "cuda_cold_control_probe.py": HERE / "cuda_cold_control_probe.py",
        "phone_cuda_cold_promotion_probe.py": (
            HERE / "phone_cuda_cold_promotion_probe.py"
        ),
        "phone_cuda_delta_probe.py": HERE / "phone_cuda_delta_probe.py",
        "phone_cuda_handoff_probe.py": HERE / "phone_cuda_handoff_probe.py",
        "prepare_phone_route.py": HERE / "prepare_phone_route.py",
        "qwen25_quality_probe.py": HERE / "qwen25_quality_probe.py",
        "run_w7_cold_promotion_gate.sh": (
            HERE / "run_w7_cold_promotion_gate.sh"
        ),
        "stage_v3_client.py": (
            HERE.parent / "s22_slo_overlap_pipeline" / "stage_v3_client.py"
        ),
        "validate_cold_promotion.py": HERE / "validate_cold_promotion.py",
        "validate_phone_cuda_delta.py": (
            HERE / "validate_phone_cuda_delta.py"
        ),
    }
    sources = w6.exact_keys(
        root["sources"],
        set(expected_sources),
        "run_context.sources",
    )
    for name, path in expected_sources.items():
        digest = w6.checked_digest(
            sources[name],
            f"run_context.sources.{name}",
        )
        promotion.require(path.is_file(), f"run_context: missing {name}")
        promotion.require(
            w6.sha256(path.read_bytes()) == digest,
            f"run_context: source digest mismatch for {name}",
        )


def validate_start_marker(
    value: object,
    run_id: str,
) -> int:
    root = w6.exact_keys(
        value,
        {"process_pid", "request_start_ns", "run_id", "schema"},
        "start_marker",
    )
    promotion.require(
        root["schema"] == promotion.START_SCHEMA
        and root["run_id"] == run_id
        and w6.is_int(root["process_pid"])
        and root["process_pid"] > 0
        and w6.is_int(root["request_start_ns"])
        and root["request_start_ns"] > 0,
        "start_marker: invalid",
    )
    return root["request_start_ns"]


def pseudo_placement_report(
    treatment: dict[str, object],
    cuda_rows: int | None = None,
) -> dict[str, object]:
    metrics = treatment["metrics"]
    if cuda_rows is None:
        cuda = {
            "cuda_continuation": {"rows": metrics["cuda_continuation"]["rows"]},
            "cuda_control": {"rows": metrics["cuda_warm_control"]["rows"]},
            "cuda_delta": {"rows": metrics["cuda_delta"]["rows"]},
            "cuda_snapshot": {"rows": metrics["cuda_replay"]["rows"]},
        }
    else:
        cuda = {
            "cuda_continuation": {"rows": 0},
            "cuda_control": {"rows": cuda_rows},
            "cuda_delta": {"rows": 0},
            "cuda_snapshot": {"rows": 0},
        }
    return {
        "metrics": {
            **cuda,
            "phone_delta": {"rows": metrics["phone_delta"]["rows"]},
            "phone_snapshot": {"rows": metrics["phone_service"]["rows"]},
        },
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


def parse_session_certificates(
    path: Path,
    field: str,
) -> tuple[list[dict[str, object]], str]:
    promotion.require(
        path.is_file() and not path.is_symlink(),
        f"{field}: invalid log path",
    )
    raw = path.read_bytes()
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise promotion.PromotionError(f"{field}: log is not UTF-8") from exc
    prefix = "SESSIONCERT "
    records = [line[len(prefix):] for line in lines if line.startswith(prefix)]
    certificates = []
    for index, record in enumerate(records):
        try:
            value = json.loads(record, object_pairs_hook=w6.strict_object)
        except json.JSONDecodeError as exc:
            raise promotion.PromotionError(
                f"{field}: invalid session certificate {index}",
            ) from exc
        certificates.append(
            w6.exact_keys(value, SESSION_KEYS, f"{field}.{index}")
        )
    promotion.require(bool(certificates), f"{field}: no session certificate")
    return certificates, w6.sha256(raw)


def validate_session_certificate(
    certificate: dict[str, object],
    *,
    field: str,
    spec: dict[str, object],
    boot_id: str,
    n_layer: int,
    session_id: int,
    session_end: str,
    reset_applied: bool,
    steps_session: int,
    steps_total: int,
) -> dict[str, object]:
    for name in (
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
        promotion.require(
            w6.is_int(certificate[name]),
            f"{field}: invalid {name}",
        )
    promotion.require(
        certificate["schema"] == physical.SESSION_SCHEMA
        and certificate["proto_version"] == 2
        and certificate["session_id"] == session_id
        and certificate["session_end"] == session_end
        and certificate["reset_applied"] is reset_applied,
        f"{field}: session identity",
    )
    promotion.require(
        certificate["device_boot_id"] == boot_id,
        f"{field}: device boot",
    )
    promotion.require(
        certificate["expected_backend"] == spec["expected_backend"]
        and certificate["layer_start"] == spec["layer_start"]
        and certificate["layer_end"] == spec["layer_end"]
        and certificate["n_layer"] == n_layer,
        f"{field}: route",
    )
    promotion.require(
        certificate["placement_status"] == "SCHEDULED_PLACEMENT_OK"
        and certificate["missing_buffer_compute_nodes"] == 0
        and certificate["steps_session"] == steps_session
        and certificate["steps_total"] == steps_total,
        f"{field}: placement or step count",
    )
    promotion.require(
        certificate["worker_pid"] > 0
        and type(certificate["worker_boot_nonce"]) is str
        and physical.HEX16.fullmatch(certificate["worker_boot_nonce"])
        is not None,
        f"{field}: worker identity",
    )
    compute = certificate["compute_by_op_and_buffer"]
    promotion.require(
        type(compute) is dict and bool(compute),
        f"{field}: empty compute placement",
    )
    by_buffer: dict[str, int] = {}
    primary_nodes = 0
    auxiliary = spec["allowed_auxiliary"]
    for op, buffers in compute.items():
        promotion.require(
            type(op) is str
            and op != ""
            and type(buffers) is dict
            and bool(buffers),
            f"{field}: invalid operation placement",
        )
        for buffer, count in buffers.items():
            promotion.require(
                type(buffer) is str
                and buffer != ""
                and w6.is_int(count)
                and count > 0,
                f"{field}: invalid compute count",
            )
            promotion.require(
                buffer == spec["primary_buffer"]
                or buffer in auxiliary.get(op, []),
                f"{field}: undeclared compute buffer",
            )
            by_buffer[buffer] = by_buffer.get(buffer, 0) + count
            if buffer == spec["primary_buffer"]:
                primary_nodes += count
    promotion.require(primary_nodes > 0, f"{field}: primary backend unused")
    return {
        "compute_by_buffer": dict(sorted(by_buffer.items())),
        "layer_end": spec["layer_end"],
        "layer_start": spec["layer_start"],
        "primary_buffer": spec["primary_buffer"],
        "steps": steps_session,
        "worker_boot_nonce": certificate["worker_boot_nonce"],
        "worker_pid": certificate["worker_pid"],
    }


def validate_placement_sessions(
    *,
    paths: dict[str, Path],
    context: dict[str, object],
    gate: physical.PhysicalGate,
    preparation_report: dict[str, object],
    treatment: dict[str, object],
    control_report: dict[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    promotion.require(
        set(paths)
        == {
            "control_cuda_head",
            "control_cuda_tail",
            "op12",
            "op15",
            "treatment_cuda_head",
            "treatment_cuda_tail",
        },
        "placement: log set",
    )
    warmup_steps = preparation_report["metrics"]["rows"]
    treatment_phone_steps = (
        treatment["metrics"]["phone_service"]["rows"]
        + treatment["metrics"]["phone_delta"]["rows"]
    )
    treatment_cuda_steps = sum(
        treatment["metrics"][name]["rows"]
        for name in (
            "cuda_continuation",
            "cuda_delta",
            "cuda_replay",
            "cuda_warm_control",
        )
    )
    control_cuda_steps = control_report["metrics"]["rows"]
    phone_summaries: dict[str, object] = {}
    treatment_summaries: dict[str, object] = {}
    control_summaries: dict[str, object] = {}

    for name in ("op12", "op15"):
        certificates, log_sha256 = parse_session_certificates(
            paths[name],
            f"placement.{name}",
        )
        promotion.require(
            len(certificates) == 2,
            f"placement.{name}: expected preparation and treatment sessions",
        )
        prepared = validate_session_certificate(
            certificates[0],
            field=f"placement.{name}.preparation",
            spec=gate.placements[name],
            boot_id=context[name]["boot_id"],
            n_layer=gate.n_layer,
            session_id=1,
            session_end="DETACH",
            reset_applied=True,
            steps_session=warmup_steps,
            steps_total=warmup_steps,
        )
        treatment_session = validate_session_certificate(
            certificates[1],
            field=f"placement.{name}.treatment",
            spec=gate.placements[name],
            boot_id=context[name]["boot_id"],
            n_layer=gate.n_layer,
            session_id=2,
            session_end="STOP",
            reset_applied=False,
            steps_session=treatment_phone_steps,
            steps_total=warmup_steps + treatment_phone_steps,
        )
        promotion.require(
            prepared["worker_pid"] == treatment_session["worker_pid"]
            and prepared["worker_boot_nonce"]
            == treatment_session["worker_boot_nonce"],
            f"placement.{name}: worker was not resident",
        )
        phone_summaries[name] = {
            "log_sha256": log_sha256,
            "preparation_steps": warmup_steps,
            "treatment_steps": treatment_phone_steps,
            "worker_boot_nonce": prepared["worker_boot_nonce"],
            "worker_pid": prepared["worker_pid"],
        }
        treatment_summaries[name] = {
            **treatment_session,
            "log_sha256": log_sha256,
        }

    boot_id = context["cuda"]["boot_id"]
    for phase, steps, target in (
        ("treatment", treatment_cuda_steps, treatment_summaries),
        ("control", control_cuda_steps, control_summaries),
    ):
        for role in ("head", "tail"):
            path_name = f"{phase}_cuda_{role}"
            gate_name = f"cuda_{role}"
            certificates, log_sha256 = parse_session_certificates(
                paths[path_name],
                f"placement.{path_name}",
            )
            promotion.require(
                len(certificates) == 1,
                f"placement.{path_name}: expected one session",
            )
            summary = validate_session_certificate(
                certificates[0],
                field=f"placement.{path_name}",
                spec=gate.placements[gate_name],
                boot_id=boot_id,
                n_layer=gate.n_layer,
                session_id=1,
                session_end="STOP",
                reset_applied=False,
                steps_session=steps,
                steps_total=steps,
            )
            target[gate_name] = {**summary, "log_sha256": log_sha256}

    for name in ("cuda_head", "cuda_tail"):
        promotion.require(
            treatment_summaries[name]["worker_boot_nonce"]
            != control_summaries[name]["worker_boot_nonce"]
            and treatment_summaries[name]["worker_pid"]
            != control_summaries[name]["worker_pid"],
            f"{name}: control reused treatment worker",
        )
    return phone_summaries, treatment_summaries, control_summaries


def validate_launch_timing(
    launch: dict[str, object],
    request_start_ns: int,
    ready_ns: int,
    contract: promotion.PromotionContract,
    phase: str,
) -> None:
    promotion.require(
        launch["request_start_ns"] == request_start_ns,
        f"{phase}: request-start mismatch",
    )
    promotion.require(
        launch["cuda_launch_ns"] - request_start_ns
        <= contract.max_launch_delay_us * 1000,
        f"{phase}: launch delay",
    )
    promotion.require(
        0 < ready_ns - launch["cuda_launch_ns"]
        <= contract.max_cuda_ready_us * 1000,
        f"{phase}: CUDA readiness bound",
    )


def validate_matched_tokens(
    treatment: dict[str, object],
    control_report: dict[str, object],
) -> None:
    treatment_sequences = treatment["sequences"]
    control_sequences = control_report["sequences"]
    promotion.require(
        type(treatment_sequences) is list
        and type(control_sequences) is list
        and len(treatment_sequences) == len(control_sequences),
        "matched_control: sequence count",
    )
    for index, (treatment_sequence, control_sequence) in enumerate(zip(
        treatment_sequences,
        control_sequences,
    )):
        promotion.require(
            type(treatment_sequence) is dict
            and type(control_sequence) is dict
            and treatment_sequence["prompt_tokens"]
            == control_sequence["prompt_tokens"]
            and treatment_sequence["final_published_tokens"]
            == control_sequence["generated_tokens"],
            f"matched_control: token mismatch for sequence {index}",
        )


def write_atomic(path: Path, value: dict[str, object]) -> None:
    raw = w6.canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--base-contract", type=Path, required=True)
    parser.add_argument("--delta-contract", type=Path, required=True)
    parser.add_argument("--physical-gate", type=Path, required=True)
    parser.add_argument("--preparation-report", type=Path, required=True)
    parser.add_argument("--treatment-report", type=Path, required=True)
    parser.add_argument("--control-report", type=Path, required=True)
    parser.add_argument("--journal-dir", type=Path, required=True)
    parser.add_argument("--run-context", type=Path, required=True)
    parser.add_argument("--treatment-start", type=Path, required=True)
    parser.add_argument("--treatment-launch", type=Path, required=True)
    parser.add_argument("--control-launch", type=Path, required=True)
    parser.add_argument("--op15-log", type=Path, required=True)
    parser.add_argument("--op12-log", type=Path, required=True)
    parser.add_argument("--treatment-cuda-head-log", type=Path, required=True)
    parser.add_argument("--treatment-cuda-tail-log", type=Path, required=True)
    parser.add_argument("--control-cuda-head-log", type=Path, required=True)
    parser.add_argument("--control-cuda-tail-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")

    base_contract = w5.load_contract(args.base_contract)
    delta_contract = w6.load_contract(args.delta_contract, base_contract)
    gate = physical.load_physical_gate(
        args.physical_gate,
        delta_contract,
        base_contract,
    )
    contract = promotion.load_contract(
        args.contract,
        base_contract,
        delta_contract,
        gate.raw_sha256,
    )
    treatment, treatment_raw = read_canonical(
        args.treatment_report,
        "treatment",
    )
    control_report, control_raw = read_canonical(
        args.control_report,
        "control",
    )
    preparation_report, preparation_raw = read_canonical(
        args.preparation_report,
        "preparation",
    )
    context, context_raw = read_canonical(args.run_context, "run_context")
    validate_context(
        context,
        contract,
        base_contract,
        delta_contract,
        gate,
    )
    run_id = context["run_id"]
    preparation.validate_report(
        preparation_report,
        contract,
        base_contract,
        run_id,
    )
    promotion.validate_report(
        treatment,
        contract,
        base_contract,
        delta_contract,
        args.journal_dir,
        run_id,
    )
    for index, (prepared, served) in enumerate(zip(
        preparation_report["sequences"],
        treatment["sequences"],
    )):
        promotion.require(
            prepared["prompt_tokens"] == served["prompt_tokens"],
            f"preparation: prompt mismatch for sequence {index}",
        )
    total_tokens = (
        treatment["cuda_ready"]["phone_service_tokens"]
        + contract.phone_delta_tokens
        + contract.cuda_continuation_tokens
    )
    control.validate_report(
        control_report,
        contract,
        base_contract,
        run_id,
        total_tokens,
    )
    validate_matched_tokens(treatment, control_report)

    start, start_raw = read_canonical(args.treatment_start, "treatment_start")
    _, treatment_launch_raw = read_canonical(
        args.treatment_launch,
        "treatment_launch",
    )
    _, control_launch_raw = read_canonical(
        args.control_launch,
        "control_launch",
    )
    treatment_start_ns = validate_start_marker(start, run_id)
    treatment_launch = control.load_launch_record(
        args.treatment_launch,
        run_id,
        "TREATMENT",
    )
    control_launch = control.load_launch_record(
        args.control_launch,
        run_id,
        "CONTROL",
    )
    promotion.require(
        treatment["cuda_ready"]["request_start_ns"] == treatment_start_ns,
        "treatment: start marker mismatch",
    )
    validate_launch_timing(
        treatment_launch,
        treatment_start_ns,
        treatment["cuda_ready"]["cuda_ready_ns"],
        contract,
        "treatment",
    )
    validate_launch_timing(
        control_launch,
        control_report["request_start_ns"],
        control_report["cuda_ready_ns"],
        contract,
        "control",
    )

    phone_preparation, treatment_placement, control_placement = (
        validate_placement_sessions(
            paths={
                "control_cuda_head": args.control_cuda_head_log,
                "control_cuda_tail": args.control_cuda_tail_log,
                "op12": args.op12_log,
                "op15": args.op15_log,
                "treatment_cuda_head": args.treatment_cuda_head_log,
                "treatment_cuda_tail": args.treatment_cuda_tail_log,
            },
            context=context,
            gate=gate,
            preparation_report=preparation_report,
            treatment=treatment,
            control_report=control_report,
        )
    )

    treatment_ttft_us = (
        treatment["cuda_ready"]["phone_first_token_ns"]
        - treatment_start_ns
    ) // 1000
    control_ttft_us = (
        control_report["first_token_ns"] - control_report["request_start_ns"]
    ) // 1000
    treatment_complete_us = (
        treatment["cuda_ready"]["request_complete_ns"]
        - treatment_start_ns
    ) // 1000
    control_complete_us = (
        control_report["request_complete_ns"]
        - control_report["request_start_ns"]
    ) // 1000
    diagnostics = {
        "control_complete_us": control_complete_us,
        "control_cuda_ready_us": (
            control_report["cuda_ready_ns"] - control_report["cuda_launch_ns"]
        ) // 1000,
        "control_ttft_us": control_ttft_us,
        "phone_tokens_before_cuda_ready": (
            treatment["cuda_ready"]["useful_phone_tokens_before_ready"]
        ),
        "phone_tokens_at_frontier": (
            treatment["cuda_ready"]["phone_service_tokens"]
        ),
        "treatment_complete_us": treatment_complete_us,
        "treatment_cuda_ready_us": (
            treatment["cuda_ready"]["cuda_ready_ns"]
            - treatment_launch["cuda_launch_ns"]
        ) // 1000,
        "treatment_ttft_us": treatment_ttft_us,
    }
    certificate = {
        "base_contract_sha256": base_contract.raw_sha256,
        "contract_sha256": contract.raw_sha256,
        "control_launch_sha256": w6.sha256(control_launch_raw),
        "control_placement": control_placement,
        "control_report_sha256": w6.sha256(control_raw),
        "delta_contract_sha256": delta_contract.raw_sha256,
        "diagnostics": diagnostics,
        "journal_summary_sha256": w6.sha256(
            w6.canonical(treatment["journal"])
        ),
        "physical_gate_sha256": gate.raw_sha256,
        "phone_preparation": phone_preparation,
        "preparation_report_sha256": w6.sha256(preparation_raw),
        "run_context_sha256": w6.sha256(context_raw),
        "scheduler_eligible": False,
        "schema": SCHEMA,
        "scope": "MECHANICS_ONLY",
        "status": "COLD_PROMOTION_MECHANICS_PASS",
        "treatment_launch_sha256": w6.sha256(treatment_launch_raw),
        "treatment_placement": treatment_placement,
        "treatment_report_sha256": w6.sha256(treatment_raw),
        "treatment_start_sha256": w6.sha256(start_raw),
    }
    write_atomic(args.output, certificate)
    print(w6.canonical(certificate).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        KeyError,
        OSError,
        ProtocolError,
        TypeError,
        ValueError,
        promotion.PromotionError,
        w5.HandoffError,
        w6.DeltaError,
    ) as exc:
        print(w6.canonical({
            "error": str(exc),
            "status": "COLD_PROMOTION_VALIDATION_ERROR",
        }).decode("ascii"), end="")
        raise SystemExit(2)
