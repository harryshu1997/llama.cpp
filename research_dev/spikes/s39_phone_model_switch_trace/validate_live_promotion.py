#!/usr/bin/env python3
"""Validate the W8 live-session promotion treatment and control."""

from __future__ import annotations

import argparse
from pathlib import Path

import cuda_cold_control_probe as cold_control
import cuda_live_control_probe as control
import phone_cuda_cold_promotion_probe as cold
import phone_cuda_delta_probe as w6
import phone_cuda_handoff_probe as w5
import phone_cuda_live_promotion_probe as live
import validate_cold_promotion as w7_validator
import validate_phone_cuda_delta as physical
from stage_v3_client import ProtocolError


SCHEMA = "s39-live-session-promotion-certificate-v1"
CONTEXT_SCHEMA = "s39-live-session-promotion-context-v1"
HERE = Path(__file__).resolve().parent
FAIL_CLOSED_ERRORS = (
    KeyError,
    OSError,
    ProtocolError,
    TypeError,
    ValueError,
    live.LivePromotionError,
    w5.HandoffError,
    w6.DeltaError,
)


def validate_context(
    value: object,
    contract: live.LiveContract,
    base_contract: w5.Contract,
    delta_contract: w6.DeltaContract,
    gate: physical.PhysicalGate,
    *,
    context_schema: str = CONTEXT_SCHEMA,
    expected_sources: dict[str, Path] | None = None,
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
    live.require(
        root["schema"] == context_schema
        and w6.is_int(root["acquisition_unix_s"])
        and root["acquisition_unix_s"] > 0
        and type(root["base_git_commit"]) is str
        and len(root["base_git_commit"]) == 40
        and all(
            character in "0123456789abcdef"
            for character in root["base_git_commit"]
        ),
        "run_context: identity",
    )
    live.require(
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
    live.require(
        cuda["device"] == "CUDA0"
        and type(cuda["boot_id"]) is str
        and cuda["boot_id"] != ""
        and cuda["worker_sha256"] == gate.artifacts["host_worker_sha256"]
        and cuda["relay_sha256"] == gate.artifacts["host_relay_sha256"],
        "run_context: CUDA identity",
    )
    all_ports = []
    for phase in ("control_ports", "treatment_ports"):
        ports = w6.exact_keys(
            cuda[phase],
            {"head", "relay", "tail"},
            f"run_context.cuda.{phase}",
        )
        live.require(
            all(
                w6.is_int(port) and 0 < port <= 65535
                for port in ports.values()
            )
            and len(set(ports.values())) == 3,
            f"run_context.cuda.{phase}: ports",
        )
        all_ports.extend(ports.values())
    live.require(len(set(all_ports)) == 6, "run_context: reused CUDA port")

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
        live.require(
            type(device["adb_target"]) is str
            and device["adb_target"] != ""
            and type(device["boot_id"]) is str
            and device["boot_id"] != ""
            and type(device["wifi"]) is str
            and device["wifi"] != ""
            and device["layers"] == layers,
            f"run_context.{name}: identity",
        )
    live.require(
        op12["worker_sha256"] == gate.artifacts["phone_worker_sha256"]
        and op15["worker_sha256"] == gate.artifacts["phone_worker_sha256"]
        and op12["shard_sha256"] == gate.artifacts["op12_shard_sha256"]
        and op15["shard_sha256"] == gate.artifacts["op15_shard_sha256"]
        and op15["relay_sha256"] == gate.artifacts["op15_relay_sha256"],
        "run_context: phone artifacts",
    )

    if expected_sources is None:
        expected_sources = {
            "async_pipeline.py": (
                HERE.parent / "s22_slo_overlap_pipeline" / "async_pipeline.py"
            ),
            "cuda_cold_control_probe.py": HERE / "cuda_cold_control_probe.py",
            "cuda_live_control_probe.py": HERE / "cuda_live_control_probe.py",
            "phone_cuda_cold_promotion_probe.py": (
                HERE / "phone_cuda_cold_promotion_probe.py"
            ),
            "phone_cuda_delta_probe.py": HERE / "phone_cuda_delta_probe.py",
            "phone_cuda_handoff_probe.py": (
                HERE / "phone_cuda_handoff_probe.py"
            ),
            "phone_cuda_live_promotion_probe.py": (
                HERE / "phone_cuda_live_promotion_probe.py"
            ),
            "qwen25_quality_probe.py": HERE / "qwen25_quality_probe.py",
            "run_w8_live_promotion_gate.sh": (
                HERE / "run_w8_live_promotion_gate.sh"
            ),
            "stage_v3_client.py": (
                HERE.parent
                / "s22_slo_overlap_pipeline"
                / "stage_v3_client.py"
            ),
            "validate_cold_promotion.py": HERE / "validate_cold_promotion.py",
            "validate_live_promotion.py": HERE / "validate_live_promotion.py",
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
        live.require(
            path.is_file() and w6.sha256(path.read_bytes()) == digest,
            f"run_context: source mismatch for {name}",
        )


def validate_start_marker(
    value: object,
    treatment: dict[str, object],
    contract: live.LiveContract,
    run_id: str,
) -> int:
    root = w6.exact_keys(
        value,
        {
            "phone_state_count",
            "preexisting_ended_ns",
            "process_pid",
            "request_start_ns",
            "run_id",
            "schema",
        },
        "start_marker",
    )
    live.require(
        root["schema"] == live.START_SCHEMA
        and root["run_id"] == run_id
        and w6.is_int(root["process_pid"])
        and root["process_pid"] > 0
        and root["phone_state_count"] == contract.batch
        and root["preexisting_ended_ns"]
        == treatment["preexisting"]["ended_ns"]
        and w6.is_int(root["request_start_ns"])
        and root["preexisting_ended_ns"] <= root["request_start_ns"],
        "start_marker: invalid",
    )
    return root["request_start_ns"]


def validate_matched_tokens(
    treatment: dict[str, object],
    control_report: dict[str, object],
) -> None:
    treatment_sequences = treatment["sequences"]
    control_sequences = control_report["sequences"]
    live.require(
        len(treatment_sequences) == len(control_sequences),
        "matched_control: sequence count",
    )
    for index, (served, baseline) in enumerate(zip(
        treatment_sequences,
        control_sequences,
    )):
        post_start = (
            served["phone_service"]
            + served["phone_delta"]
            + served["cuda_continuation"]
        )
        live.require(
            served["prompt_tokens"] == baseline["prompt_tokens"]
            and served["preexisting_tokens"]
            == baseline["preexisting_tokens"]
            and post_start == baseline["generated_tokens"],
            f"matched_control: token mismatch for sequence {index}",
        )


def require_passing_treatment(treatment: dict[str, object]) -> None:
    live.require(
        treatment["status"] == "LIVE_SESSION_PROMOTION_PASS",
        "treatment: internal exactness failed",
    )


def validate_placement(
    paths: dict[str, Path],
    context: dict[str, object],
    gate: physical.PhysicalGate,
    treatment: dict[str, object],
    control_report: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    expected = {
        "op12": (
            treatment["preexisting"]["metrics"]["rows"]
            + treatment["metrics"]["phone_service"]["rows"]
            + treatment["metrics"]["phone_delta"]["rows"]
        ),
        "op15": (
            treatment["preexisting"]["metrics"]["rows"]
            + treatment["metrics"]["phone_service"]["rows"]
            + treatment["metrics"]["phone_delta"]["rows"]
        ),
        "treatment_cuda_head": sum(
            treatment["metrics"][name]["rows"]
            for name in (
                "cuda_continuation",
                "cuda_delta",
                "cuda_replay",
                "cuda_warm_control",
            )
        ),
        "treatment_cuda_tail": sum(
            treatment["metrics"][name]["rows"]
            for name in (
                "cuda_continuation",
                "cuda_delta",
                "cuda_replay",
                "cuda_warm_control",
            )
        ),
        "control_cuda_head": control_report["metrics"]["rows"],
        "control_cuda_tail": control_report["metrics"]["rows"],
    }
    treatment_summary = {}
    control_summary = {}
    for name, path in paths.items():
        certificates, log_sha256 = w7_validator.parse_session_certificates(
            path,
            f"placement.{name}",
        )
        live.require(
            len(certificates) == 1,
            f"placement.{name}: expected one session",
        )
        gate_name = (
            name
            if name in ("op12", "op15")
            else name.removeprefix("treatment_").removeprefix("control_")
        )
        boot_id = (
            context[name]["boot_id"]
            if name in ("op12", "op15")
            else context["cuda"]["boot_id"]
        )
        summary = w7_validator.validate_session_certificate(
            certificates[0],
            field=f"placement.{name}",
            spec=gate.placements[gate_name],
            boot_id=boot_id,
            n_layer=gate.n_layer,
            session_id=1,
            session_end="STOP",
            reset_applied=False,
            steps_session=expected[name],
            steps_total=expected[name],
        )
        summary = {**summary, "log_sha256": log_sha256}
        if name.startswith("control_"):
            control_summary[gate_name] = summary
        else:
            treatment_summary[gate_name] = summary
    for name in ("cuda_head", "cuda_tail"):
        live.require(
            treatment_summary[name]["worker_pid"]
            != control_summary[name]["worker_pid"]
            and treatment_summary[name]["worker_boot_nonce"]
            != control_summary[name]["worker_boot_nonce"],
            f"{name}: control reused treatment worker",
        )
    return treatment_summary, control_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--base-contract", type=Path, required=True)
    parser.add_argument("--delta-contract", type=Path, required=True)
    parser.add_argument("--physical-gate", type=Path, required=True)
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
    contract = live.load_contract(
        args.contract,
        base_contract,
        delta_contract,
        gate.raw_sha256,
    )
    treatment, treatment_raw = w6.read_canonical(
        args.treatment_report,
        "live_treatment",
    )
    control_report, control_raw = w6.read_canonical(
        args.control_report,
        "live_control",
    )
    context, context_raw = w6.read_canonical(args.run_context, "run_context")
    validate_context(
        context,
        contract,
        base_contract,
        delta_contract,
        gate,
    )
    run_id = context["run_id"]
    live.validate_report(
        treatment,
        contract,
        base_contract,
        delta_contract,
        args.journal_dir,
        run_id,
    )
    require_passing_treatment(treatment)
    post_start_tokens = (
        treatment["cuda_ready"]["phone_service_tokens"]
        + contract.phone_delta_tokens
        + contract.cuda_continuation_tokens
    )
    control.validate_report(
        control_report,
        contract,
        base_contract,
        run_id,
        post_start_tokens,
    )
    validate_matched_tokens(treatment, control_report)

    start, start_raw = w6.read_canonical(
        args.treatment_start,
        "treatment_start",
    )
    _, treatment_launch_raw = w6.read_canonical(
        args.treatment_launch,
        "treatment_launch",
    )
    _, control_launch_raw = w6.read_canonical(
        args.control_launch,
        "control_launch",
    )
    treatment_start_ns = validate_start_marker(
        start,
        treatment,
        contract,
        run_id,
    )
    treatment_launch = cold_control.load_launch_record(
        args.treatment_launch,
        run_id,
        "TREATMENT",
    )
    control_launch = cold_control.load_launch_record(
        args.control_launch,
        run_id,
        "CONTROL",
    )
    live.require(
        treatment["cuda_ready"]["request_start_ns"] == treatment_start_ns,
        "treatment: start marker mismatch",
    )
    w7_validator.validate_launch_timing(
        treatment_launch,
        treatment_start_ns,
        treatment["cuda_ready"]["cuda_ready_ns"],
        contract,
        "treatment",
    )
    w7_validator.validate_launch_timing(
        control_launch,
        control_report["request_start_ns"],
        control_report["cuda_ready_ns"],
        contract,
        "control",
    )

    treatment_placement, control_placement = validate_placement(
        {
            "control_cuda_head": args.control_cuda_head_log,
            "control_cuda_tail": args.control_cuda_tail_log,
            "op12": args.op12_log,
            "op15": args.op15_log,
            "treatment_cuda_head": args.treatment_cuda_head_log,
            "treatment_cuda_tail": args.treatment_cuda_tail_log,
        },
        context,
        gate,
        treatment,
        control_report,
    )
    diagnostics = {
        "control_complete_us": (
            control_report["request_complete_ns"]
            - control_report["request_start_ns"]
        )
        // 1000,
        "control_ttft_us": (
            control_report["first_token_ns"]
            - control_report["request_start_ns"]
        )
        // 1000,
        "phone_tokens_before_cuda_ready": (
            treatment["cuda_ready"]["useful_phone_tokens_before_ready"]
        ),
        "phone_tokens_at_frontier": (
            treatment["cuda_ready"]["phone_service_tokens"]
        ),
        "treatment_complete_us": (
            treatment["cuda_ready"]["request_complete_ns"]
            - treatment_start_ns
        )
        // 1000,
        "treatment_ttft_us": (
            treatment["cuda_ready"]["phone_first_token_ns"]
            - treatment_start_ns
        )
        // 1000,
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
        "run_context_sha256": w6.sha256(context_raw),
        "scheduler_eligible": False,
        "schema": SCHEMA,
        "scope": "MECHANICS_ONLY",
        "status": "LIVE_SESSION_PROMOTION_MECHANICS_PASS",
        "treatment_launch_sha256": w6.sha256(treatment_launch_raw),
        "treatment_placement": treatment_placement,
        "treatment_report_sha256": w6.sha256(treatment_raw),
        "treatment_start_sha256": w6.sha256(start_raw),
    }
    w7_validator.write_atomic(args.output, certificate)
    print(w6.canonical(certificate).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FAIL_CLOSED_ERRORS as exc:
        print(w6.canonical({
            "error": str(exc),
            "status": "LIVE_SESSION_VALIDATION_ERROR",
        }).decode("ascii"), end="")
        raise SystemExit(2)
