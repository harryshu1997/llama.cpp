#!/usr/bin/env python3
"""Validate W8-R1 with a same-token fresh-CUDA trace control."""

from __future__ import annotations

import argparse
from pathlib import Path

import cuda_cold_control_probe as cold_control
import cuda_live_trace_control_probe as trace_control
import phone_cuda_delta_probe as w6
import phone_cuda_handoff_probe as w5
import phone_cuda_live_promotion_probe as live
import validate_cold_promotion as w7_validator
import validate_live_promotion as v1
import validate_phone_cuda_delta as physical
from stage_v3_client import ProtocolError


SCHEMA = "s39-live-session-promotion-r1-certificate-v1"
CONTEXT_SCHEMA = "s39-live-session-promotion-r1-context-v1"
HERE = Path(__file__).resolve().parent


def expected_sources() -> dict[str, Path]:
    return {
        "async_pipeline.py": (
            HERE.parent / "s22_slo_overlap_pipeline" / "async_pipeline.py"
        ),
        "cuda_cold_control_probe.py": HERE / "cuda_cold_control_probe.py",
        "cuda_live_control_probe.py": HERE / "cuda_live_control_probe.py",
        "cuda_live_trace_control_probe.py": (
            HERE / "cuda_live_trace_control_probe.py"
        ),
        "phone_cuda_cold_promotion_probe.py": (
            HERE / "phone_cuda_cold_promotion_probe.py"
        ),
        "phone_cuda_delta_probe.py": HERE / "phone_cuda_delta_probe.py",
        "phone_cuda_handoff_probe.py": HERE / "phone_cuda_handoff_probe.py",
        "phone_cuda_live_promotion_probe.py": (
            HERE / "phone_cuda_live_promotion_probe.py"
        ),
        "qwen25_quality_probe.py": HERE / "qwen25_quality_probe.py",
        "run_w8_live_promotion_gate.sh": (
            HERE / "run_w8_live_promotion_gate.sh"
        ),
        "stage_v3_client.py": (
            HERE.parent / "s22_slo_overlap_pipeline" / "stage_v3_client.py"
        ),
        "validate_cold_promotion.py": HERE / "validate_cold_promotion.py",
        "validate_live_promotion.py": HERE / "validate_live_promotion.py",
        "validate_live_promotion_r1.py": (
            HERE / "validate_live_promotion_r1.py"
        ),
        "validate_phone_cuda_delta.py": (
            HERE / "validate_phone_cuda_delta.py"
        ),
    }


def validate_matched_trace(
    treatment: dict[str, object],
    control_report: dict[str, object],
) -> None:
    treatment_sequences = treatment["sequences"]
    control_sequences = control_report["sequences"]
    live.require(
        type(treatment_sequences) is list
        and type(control_sequences) is list
        and len(treatment_sequences) == len(control_sequences),
        "matched_trace: sequence count",
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
            and post_start == baseline["replayed_tokens"],
            f"matched_trace: token mismatch for sequence {index}",
        )


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
    contract_value, _ = w6.read_canonical(args.contract, "live_contract")
    live.require(
        contract_value.get("schema") == live.CONTRACT_R1_SCHEMA,
        "live_contract: R1 schema required",
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
        "trace_control",
    )
    context, context_raw = w6.read_canonical(args.run_context, "run_context")
    v1.validate_context(
        context,
        contract,
        base_contract,
        delta_contract,
        gate,
        context_schema=CONTEXT_SCHEMA,
        expected_sources=expected_sources(),
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
    v1.require_passing_treatment(treatment)
    histories, trace = trace_control.load_treatment_trace(
        args.treatment_report,
        contract,
        base_contract,
        run_id,
    )
    live.require(
        len(histories) == contract.batch,
        "matched_trace: history count",
    )
    trace_control.validate_report(
        control_report,
        contract,
        base_contract,
        run_id,
        trace,
    )
    validate_matched_trace(treatment, control_report)

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
    treatment_start_ns = v1.validate_start_marker(
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

    treatment_placement, control_placement = v1.validate_placement(
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
    agreement = control_report["greedy_agreement"]
    diagnostics = {
        "control_complete_us": (
            control_report["request_complete_ns"]
            - control_report["request_start_ns"]
        )
        // 1000,
        "control_greedy_matching_tokens": agreement["matching_tokens"],
        "control_greedy_total_tokens": agreement["total_tokens"],
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
        "control_mode": trace_control.MODE,
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
        "status": "LIVE_SESSION_PROMOTION_TRACE_MECHANICS_PASS",
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
    except (
        KeyError,
        OSError,
        ProtocolError,
        TypeError,
        ValueError,
        live.LivePromotionError,
        w5.HandoffError,
        w6.DeltaError,
    ) as exc:
        print(w6.canonical({
            "error": str(exc),
            "status": "LIVE_SESSION_R1_VALIDATION_ERROR",
        }).decode("ascii"), end="")
        raise SystemExit(2)
