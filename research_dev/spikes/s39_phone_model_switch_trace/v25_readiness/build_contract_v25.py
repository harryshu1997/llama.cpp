#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
S39 = HERE.parent
V23 = S39 / "v23_readiness"
V24 = S39 / "v24_readiness"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import v25_common as common


DEFAULT_OUTPUT = HERE / "CP0_R1_EVIDENCE_CONTRACT_V2_5.json"
CANDIDATE = S39 / "CP0_R1_CANDIDATE.json"
V24_CONTRACT = V24 / "CP0_R1_EVIDENCE_CONTRACT_V2_4.json"
TOKEN_HISTORY = (
    V24 / "results" / "prephase_20260726T0915Z" / "token-history.json"
)
TOKENIZER_PLAN = (
    V24 / "results" / "prephase_20260726T0915Z" / "tokenizer-plan.json"
)
CORPUS = S39 / "CP0_R1_MMLU64_CORPUS_V2_2.jsonl"
ACQUISITION = V23 / "a_only_acquisition_driver_v1"
PRODUCERS = ACQUISITION / "producers_v1"
PRODUCTION_V2 = V23 / "production_v2"
PRODUCTION_V1 = V23 / "production_v1"
PLAN_V1 = V23 / "acquisition_plan_v1"
REPO = S39.parents[2]
REMOTE_REPO = Path("/home/zhihao/llama.cpp-s40")
REMOTE_PREPHASE = Path("/home/zhihao/s39-v25-a-only/prephase")

V23_PROGRAMS = {
    "acquisition_driver": ACQUISITION / "run_a_only_acquisition_v1.py",
    "acquisition_support": ACQUISITION / "acquisition_support_v1.py",
    "artifact_snapshot_driver": PRODUCTION_V2 / "artifact_snapshot_driver_v2.py",
    "cuda_monolithic": PRODUCERS / "cuda_monolithic_v1.py",
    "cuda_route_capture": PRODUCERS / "cuda_route_capture_v1.py",
    "driver_common_base": PRODUCTION_V1 / "driver_common_v1.py",
    "driver_common_runtime": PRODUCTION_V2 / "driver_common_v2.py",
    "fresh_readiness_driver": PRODUCTION_V2 / "fresh_readiness_driver_v2.py",
    "joint_phone_cuda": PRODUCERS / "joint_phone_cuda_v1.py",
    "managed_runtime_launcher": PRODUCERS / "managed_runtime_launcher_v1.py",
    "materializer": PLAN_V1 / "materialize_a_only_runtime_inputs_v1.py",
    "phone_route_capture": PRODUCERS / "phone_route_capture_v1.py",
    "phone_runtime_probe": PRODUCERS / "phone_runtime_probe_v1.py",
    "plan_common": PLAN_V1 / "plan_common_v1.py",
    "remote_android_process_probe": (
        PRODUCERS / "remote_android_process_probe_v1.py"
    ),
    "runtime_bundle_overlay": PRODUCTION_V2 / "runtime_bundle_overlay_v1.py",
    "source_entry": PRODUCTION_V2 / "source_entry_v2.py",
}
HISTORY_PROGRAMS = {
    "history_common": V24 / "history_v1" / "history_common_v1.py",
    "validator": V24 / "history_v1" / "validate_b8_history_v1.py",
}
V24_PROGRAMS = {
    "authority": V24 / "cp0_r1_evidence_v24.py",
    "common": V24 / "v24_common.py",
    "contract_builder": V24 / "build_contract_v24.py",
    "cuda_monolithic_producer": (
        V24 / "producers_v1" / "cuda_monolithic_v1.py"
    ),
    "joint_phone_cuda_producer": (
        V24 / "producers_v1" / "joint_phone_cuda_v1.py"
    ),
    "production_common": (
        V24 / "production_plan_v1" / "production_common_v1.py"
    ),
    "remote_fan_in": V24 / "production_plan_v1" / "fan_in_v1.py",
}
V25_PROGRAMS = {
    "authority": HERE / "cp0_r1_evidence_v25.py",
    "common": HERE / "v25_common.py",
    "contract_builder": Path(__file__).resolve(),
    "materializer": (
        HERE / "production_v1" / "materialize_a_only_v1.py"
    ),
    "remote_cuda_capture": HERE / "remote_cuda_capture_v1.py",
    "remote_fan_in_contract": HERE / "remote_fan_in_contract_v1.py",
    "remote_fan_in_execute": HERE / "remote_fan_in_execute_v1.py",
    "remote_history_driver": HERE / "remote_history_validate_v1.py",
    "remote_phone_guard": HERE / "remote_phone_guard_v1.py",
}


def artifact(path: Path) -> dict[str, Any]:
    raw = common.read_regular(path, 512 * 1024 * 1024)
    return {
        "bytes": len(raw),
        "path": str(path.relative_to(S39)),
        "sha256": common.sha256_bytes(raw),
    }


def remote_history_artifact(path: Path) -> dict[str, Any]:
    value = artifact(path)
    relative = path.relative_to(REPO)
    return {
        "remote_path": str(REMOTE_REPO / relative),
        "source": value,
    }


def remote_input_artifact(path: Path, remote_name: str) -> dict[str, Any]:
    raw = common.read_regular(path, 512 * 1024 * 1024)
    return {
        "bytes": len(raw),
        "path": str(REMOTE_PREPHASE / remote_name),
        "sha256": common.sha256_bytes(raw),
    }


def build_contract() -> dict[str, Any]:
    candidate, candidate_raw = common.read_canonical(CANDIDATE)
    v24, v24_raw = common.read_canonical(V24_CONTRACT)
    _, history_raw = common.read_canonical(TOKEN_HISTORY)
    _, tokenizer_plan_raw = common.read_canonical(TOKENIZER_PLAN)
    corpus_raw = common.read_regular(CORPUS)
    model = next(
        value
        for value in candidate["models"]
        if value["slot"] == "A"
    )
    return {
        "candidate": {
            "model_id": model["model_id"],
            "sha256": common.sha256_bytes(candidate_raw),
            "slot": "A",
        },
        "claim_boundary": {
            "acquisition_ready": False,
            "energy_claim": "NONE",
            "formal_claim": "NONE",
            "hardware_acquisition_run": False,
            "parent_status_is_not_evidence": True,
            "qualification_requires_v2_4_raw_predicate_reevaluation": True,
            "v2_4_phone_system_swap_zero_replaced_by": (
                "V2_5_LOCKED_BASELINE_NO_GROWTH"
            ),
        },
        "composition": {
            "history": {
                name: remote_history_artifact(path)
                for name, path in sorted(HISTORY_PROGRAMS.items())
            },
            "v23": {
                name: artifact(path)
                for name, path in sorted(V23_PROGRAMS.items())
            },
            "v24": {
                **{
                    name: artifact(path)
                    for name, path in sorted(V24_PROGRAMS.items())
                },
                "contract": artifact(V24_CONTRACT),
                "contract_sha256": common.sha256_bytes(v24_raw),
            },
            "v25": {
                name: artifact(path)
                for name, path in sorted(V25_PROGRAMS.items())
            },
        },
        "devices": v24["devices"],
        "gates": {
            "maximum_process_swap_bytes": 0,
            "maximum_system_swap_growth_bytes": 0,
            "phone_minimum_available_bytes": v24["gates"][
                "phone_minimum_available_bytes"
            ],
        },
        "phase": "A_ONLY",
        "phase_order": [
            "REBOOT_PREPARATION",
            "USB_POST_REBOOT_DISCOVERY",
            "WIFI_SELECTOR_ATTESTATION",
            "PHASE_BOUND_PLAN_MATERIALIZATION",
            "REMOTE_HISTORY_VALIDATION",
            "PHASE_LOCK",
            "RUNTIME_ACQUISITION",
            "REMOTE_RTX_FAN_IN",
            "CLEANUP",
            "V2_4_RAW_PREDICATE_REEVALUATION",
        ],
        "quality": {
            "continuation_tokens_per_request": 8,
            "corpus_items": 64,
            "corpus_sha256": common.sha256_bytes(corpus_raw),
            "cuda_minimum_correct_items": 25,
            "group_count": 8,
            "group_size": 8,
            "n_batch": 64,
            "n_ctx_seq": 512,
            "n_ubatch": 64,
            "token_history_sha256": common.sha256_bytes(history_raw),
            "tokenizer_plan_sha256": common.sha256_bytes(tokenizer_plan_raw),
            "remote_inputs": {
                "candidate": remote_input_artifact(
                    CANDIDATE,
                    "CP0_R1_CANDIDATE.json",
                ),
                "corpus": remote_input_artifact(
                    CORPUS,
                    "CP0_R1_MMLU64_CORPUS_V2_2.jsonl",
                ),
                "history": remote_input_artifact(
                    TOKEN_HISTORY,
                    "token-history.json",
                ),
                "tokenizer_plan": remote_input_artifact(
                    TOKENIZER_PLAN,
                    "tokenizer-plan.json",
                ),
            },
        },
        "required_artifact_roles": sorted(
            {
                "history.remote_plan",
                "history.remote_receipt",
                "capture.cuda_monolithic.wrapper",
                "capture.joint_phone_cuda.wrapper",
                "capture.remote_fan_in.wrapper",
                "inner.artifact_root",
                "inner.bound_root",
                "inner.cuda_route_launch",
                "inner.fresh_readiness",
                "inner.identity_binding_attestation",
                "inner.identity_binding_receipt",
                "inner.identity_binding_stage_receipt",
                "inner.joint_capture_plan",
                "inner.orchestration_plan",
                "inner.phase_lock",
                "inner.phone_route_launch",
                "inner.preparation",
                "inner.prospective_root",
                "inner.runtime_plan",
                "phase.discovery",
                "phase.execution_ledger",
                "phase.fresh_identity",
                "phase.inventory",
                "phase.lock",
                "phase.materialization",
                "phase.preparation",
                "plan.managed.cuda_monolithic",
                "plan.managed.joint_phone_cuda",
                "plan.managed.remote_fan_in",
                "plan.remote_fan_in",
                "plan.remote_phone_guard",
                "plan.remote_phone_guard_policy",
                "plan.wrapper.cuda_monolithic",
                "plan.wrapper.joint_phone_cuda",
                "runtime.remote_phone_guard.after",
                "runtime.remote_phone_guard.before",
            }
        ),
        "schema": "s39-cp0-r1-evidence-contract-v2.5",
        "scope": "POST_REBOOT_A_ONLY_EVIDENCE_SUCCESSOR",
        "status": "FROZEN_BEFORE_A_ONLY_ACQUISITION",
        "topology": {
            "controller_host": "FCHLLX01",
            "cuda_gpu_uuid": v24["devices"]["cuda"]["uuid"],
            "cuda_host": v24["devices"]["cuda"]["host"],
            "cuda_prephase_root": str(REMOTE_PREPHASE),
            "cuda_python_bytes": 7_481_192,
            "cuda_python_path": "/usr/bin/python3.14",
            "cuda_python_sha256": (
                "b8d8288faefdd300201f43fcf00f6f539a27218ee"
                "ed3a3dff5ab10b9c4c99700"
            ),
            "cuda_ssh_target": "zhihao@172.20.74.85",
        },
        "version": "2.5",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        common.write_exclusive(args.output, build_contract())
    except (OSError, common.EvidenceError) as error:
        print(f"V2_5_CONTRACT_REFUSED: {error}", file=sys.stderr)
        return 2
    print(f"V2_5_CONTRACT_WRITTEN: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
