#!/usr/bin/env python3
"""Build the prospective W9 contract from frozen W8 evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

import phone_cuda_delta_probe as w6
import validate_profiled_cutover_pair as pair_validator
import w9_profiled_cutover as w9


HERE = Path(__file__).resolve().parent
W8_RUN = (
    HERE
    / "results"
    / "w8_live_promotion_r1"
    / "run_20260725T013802Z"
)


def build() -> dict[str, object]:
    w8_contract = HERE / "W8_LIVE_SESSION_CONTRACT_R1.json"
    base_contract = HERE / "W5_HANDOFF_CONTRACT.json"
    delta_contract = HERE / "W6_DELTA_CONTRACT.json"
    physical_gate = HERE / "W6_PHYSICAL_GATE.json"
    treatment_path = W8_RUN / "treatment_report.json"
    manifest_path = W8_RUN / "SHA256SUMS.txt"
    treatment, _ = w6.read_canonical(treatment_path, "w8_treatment")
    third_batch = treatment["metrics"]["phone_service"]["batch_timeline"][2]
    ready_ns = treatment["cuda_ready"]["cuda_ready_ns"]
    inflight_elapsed_us = (ready_ns - third_batch["started_ns"]) // 1000
    source_sha256 = {
        name: w6.sha256(path.read_bytes())
        for name, path in sorted(pair_validator.source_paths().items())
    }
    candidate_k = list(range(12))
    return {
        "dependencies": {
            "base_contract_sha256": w6.sha256(base_contract.read_bytes()),
            "delta_contract_sha256": w6.sha256(delta_contract.read_bytes()),
            "physical_gate_sha256": w6.sha256(physical_gate.read_bytes()),
            "w8_contract_sha256": w6.sha256(w8_contract.read_bytes()),
            "w8_manifest_sha256": w6.sha256(manifest_path.read_bytes()),
            "w8_treatment_report_sha256": w6.sha256(
                treatment_path.read_bytes()
            ),
        },
        "devices": {
            "cuda": {
                "name": "NVIDIA RTX A6000",
                "uuid": "GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f",
            },
            "op12": {
                "layers": [30, 48],
                "serial": "5ae7a43d",
                "wifi": "172.20.59.72",
            },
            "op15": {
                "layers": [0, 30],
                "serial": "3C15AU002CL00000",
                "wifi": "172.20.173.218",
            },
        },
        "execution": {
            "batch": 8,
            "idle_samples": 5,
            "idle_span_us": 1_000_000,
            "max_cuda_ready_us": 120_000_000,
            "max_inflight_tokens": 1,
            "max_launch_delay_us": 250_000,
            "min_cuda_continuation_tokens": 1,
            "output_tokens": 13,
            "paired_launch_delta_us": 20_000,
            "phone_thermal_range_millic": 3_000,
            "preexisting_committed_tokens": 2,
        },
        "pair_ordinals": ["P1", "P2", "P3", "P4"],
        "performance_gates": {
            "completion_ratio_den": 4,
            "completion_ratio_num": 5,
            "next_token_ratio_den": 4,
            "next_token_ratio_num": 3,
        },
        "policy": {
            "candidate_k": candidate_k,
            "cuda_replay_us": 180_000,
            "cuda_tokens_us": [50_000 * index for index in range(14)],
            "cutover_margin_us": 50_000,
            "delta_ingest_us": [25_000 * index for index in range(13)],
            "phone_batch_estimate_us": 1_500_000,
            "phone_extra_us": [
                1_500_000 * index for index in candidate_k
            ],
            "predicted_commit_us": 10_000,
        },
        "reference_profile": {
            "expected_k_extra": 0,
            "inflight_elapsed_us": inflight_elapsed_us,
            "inflight_present": True,
            "phone_tokens_at_f0": 2,
            "source_bindings": [
                {
                    "artifact_sha256": w6.sha256(treatment_path.read_bytes()),
                    "field": "metrics.cuda_replay.elapsed_us",
                    "rounding": "CEIL",
                    "source_kind": "W8_ARTIFACT",
                    "value_us": 180_000,
                },
                {
                    "artifact_sha256": w6.sha256(treatment_path.read_bytes()),
                    "field": (
                        "metrics.phone_service.batch_timeline[2]."
                        "(ended_ns - started_ns) / 1000"
                    ),
                    "rounding": "CEIL",
                    "source_kind": "W8_ARTIFACT",
                    "value_us": 1_500_000,
                },
                {
                    "artifact_sha256": w6.sha256(treatment_path.read_bytes()),
                    "field": "metrics.cuda_delta.elapsed_us / 2",
                    "rounding": "CONSERVATIVE_BOUND",
                    "source_kind": "W8_ARTIFACT",
                    "value_us": 25_000,
                },
                {
                    "artifact_sha256": w6.sha256(treatment_path.read_bytes()),
                    "field": "metrics.cuda_continuation.elapsed_us / 8",
                    "rounding": "CONSERVATIVE_BOUND",
                    "source_kind": "W8_ARTIFACT",
                    "value_us": 50_000,
                },
                {
                    "artifact_sha256": w9.ZERO_SHA256,
                    "field": "predicted_commit_us",
                    "rounding": "CONSERVATIVE_BOUND",
                    "source_kind": "PROSPECTIVE_BOUND",
                    "value_us": 10_000,
                },
            ],
        },
        "requirements": [
            "PROFILE_DERIVED_ZERO_EXTRA_DECISION",
            "OPTIONAL_SINGLE_INFLIGHT_BATCH",
            "CUDA_REPLAY_STARTS_FROM_F0_AT_READINESS",
            "EXACT_F1_MINUS_F0_INGESTION",
            "DURABLE_SINGLE_PUBLICATION_OWNER",
            "MATCHED_FRESH_CUDA_TRACE_CONTROL",
            "FOUR_NONREPLACEABLE_PAIRS",
            "REALIZED_BACKEND_PLACEMENT",
            "GPU_IDENTITY_IDLE_AND_READINESS_COMPARABILITY",
            "PHONE_THERMAL_START_RANGE",
            "ZERO_TERMINAL_SEQUENCE_STATE",
        ],
        "scheduler_eligible_on_pass": False,
        "schema": w9.CONTRACT_SCHEMA,
        "scope": "MECHANICS_ONLY",
        "source_sha256": source_sha256,
        "status": "FROZEN_BEFORE_ACQUISITION",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    value = build()
    w9.write_atomic(args.output, value)
    w9.load_contract(args.output)
    print(w6.canonical(value).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError, w6.DeltaError, w9.W9Error) as exc:
        print(w6.canonical({"error": str(exc), "status": "W9_CONTRACT_ERROR"}).decode("ascii"), end="")
        raise SystemExit(2)
