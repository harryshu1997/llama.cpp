#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
S39 = HERE.parent
V23 = S39 / "v23_readiness"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import v24_common as common


DEFAULT_OUTPUT = HERE / "CP0_R1_EVIDENCE_CONTRACT_V2_4.json"
CANDIDATE = S39 / "CP0_R1_CANDIDATE.json"
V22_CONTRACT = S39 / "CP0_R1_EVIDENCE_CONTRACT_V2_2.json"
V2_CONTRACT = S39 / "CP0_R1_EVIDENCE_CONTRACT_V2.json"
V23_CONTRACT = V23 / "CP0_R1_EVIDENCE_CONTRACT_V2_3.json"
AUTHORITY = HERE / "cp0_r1_evidence_v24.py"
HISTORICAL_PRIMITIVES = {
    "artifact_snapshot_driver_v2": (
        V23 / "production_v2" / "artifact_snapshot_driver_v2.py"
    ),
    "driver_common_v2": V23 / "production_v2" / "driver_common_v2.py",
    "fresh_readiness_driver_v2": (
        V23 / "production_v2" / "fresh_readiness_driver_v2.py"
    ),
    "prepare_zero_swap_v1": V23 / "production_v2" / "prepare_zero_swap_v1.py",
    "runtime_bundle_overlay_v1": (
        V23 / "production_v2" / "runtime_bundle_overlay_v1.py"
    ),
    "source_entry_v2": V23 / "production_v2" / "source_entry_v2.py",
}
EVALUATOR_HELPERS = {
    "builder_v21": S39 / "build_cp0_r1_v21.py",
    "builder_v22": S39 / "build_cp0_r1_v22.py",
    "evaluator_v2": S39 / "cp0_r1_evidence_v2.py",
    "evaluator_v21": S39 / "cp0_r1_evidence_v21.py",
    "evaluator_v22": S39 / "cp0_r1_evidence_v22.py",
    "mmlu_builder_v22": S39 / "build_cp0_r1_mmlu64_v22.py",
}
PRODUCER_PROGRAMS = {
    "cuda_monolithic": HERE / "producers_v1" / "cuda_monolithic_v1.py",
    "cuda_monolithic_launch_builder": (
        HERE / "producers_v1" / "build_cuda_monolithic_launch_v1.py"
    ),
    "cuda_route": HERE / "producers_v1" / "cuda_route_v1.py",
    "joint_phone_cuda": HERE / "producers_v1" / "joint_phone_cuda_v1.py",
    "phone_route": HERE / "producers_v1" / "phone_route_v1.py",
}
ORCHESTRATION_PROGRAMS = {
    "fan_in": HERE / "production_plan_v1" / "fan_in_v1.py",
    "identity_binding": (
        HERE / "production_plan_v1" / "identity_binding_v1.py"
    ),
    "phase_lock": HERE / "production_plan_v1" / "phase_lock_v1.py",
    "preparation": HERE / "production_plan_v1" / "preparation_v1.py",
    "readiness_projection": (
        HERE / "production_plan_v1" / "readiness_projection_v1.py"
    ),
}
ORCHESTRATION_SUPPORT = {
    "orchestration": (
        HERE / "orchestration_v1" / "orchestration_v1.py"
    ),
    "production_common": (
        HERE / "production_plan_v1" / "production_common_v1.py"
    ),
}


def _artifact(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "bytes": len(raw),
        "path": str(path.relative_to(S39)),
        "sha256": common.sha256_bytes(raw),
    }


def build_raw_predicate_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    v22, _ = common.read_canonical(V22_CONTRACT)
    v2, _ = common.read_canonical(V2_CONTRACT)
    serving_envelope = {
        **v22["serving_envelope"],
        "n_ctx_seq": 512,
    }
    raw = copy.deepcopy(v22)
    raw["schema"] = "s39-cp0-r1-raw-predicate-contract-v2.4"
    raw["status"] = "FROZEN_BEFORE_A_ONLY_ACQUISITION"
    raw["serving_envelope"] = serving_envelope
    raw["claim_boundary"]["exit_authority"] = (
        "v24_readiness/cp0_r1_evidence_v24.py"
    )
    parent = copy.deepcopy(v2)
    parent["serving_envelope"] = serving_envelope
    return raw, parent


def build_contract() -> dict[str, Any]:
    candidate, candidate_raw = common.read_canonical(CANDIDATE)
    v22, v22_raw = common.read_canonical(V22_CONTRACT)
    v23, v23_raw = common.read_canonical(V23_CONTRACT)
    common.exact(candidate["schema"], "s39-cp0-r1-candidate-v1", "candidate")
    common.exact(v22["schema"], "s39-cp0-r1-evidence-contract-v2.2", "v22")
    common.exact(v23["schema"], "s39-cp0-r1-evidence-contract-v2.3", "v23")
    model_a = next(row for row in candidate["models"] if row["slot"] == "A")
    raw_predicate, raw_parent = build_raw_predicate_contract()
    raw_predicate_raw = common.canonical_bytes(raw_predicate)
    raw_parent_raw = common.canonical_bytes(raw_parent)
    authority_support = {
        "common": _artifact(HERE / "v24_common.py"),
        "contract_builder": _artifact(Path(__file__).resolve()),
        **{
            name: _artifact(path)
            for name, path in sorted(EVALUATOR_HELPERS.items())
        },
    }
    orchestration_support = {
        "authority": _artifact(AUTHORITY),
        **authority_support,
        **{
            name: _artifact(path)
            for name, path in sorted(ORCHESTRATION_SUPPORT.items())
        },
    }
    full_orchestration_support = sorted(orchestration_support)
    return {
        "candidate_lock": {
            "bytes": len(candidate_raw),
            "model_id": model_a["model_id"],
            "sha256": common.sha256_bytes(candidate_raw),
            "slot": "A",
        },
        "claim_boundary": {
            "acquisition_authorized_without_producers": False,
            "energy_claim": "NONE",
            "formal_claim": "NONE",
            "mechanics_status": "V2_4_MECHANICS_CONTRACT_PRODUCERS_BLOCKED",
            "mechanics_token_history_scope": "FIRST_B8_OF_CANONICAL_MMLU64",
            "quality_scope": "ALL_64_AS_EIGHT_B8_GROUPS",
            "qualification_requires_v2_4_raw_predicate_reevaluation": True,
        },
        "cuda_monolithic_identity": {
            "expected_file_type": 15,
            "launch_sha256_environment": "LAYERSPLIT_MODEL_SHA256",
            "maps_exact_rows": [
                {
                    "offset_bytes": 644108288,
                    "permissions": "r--s",
                },
                {
                    "offset_bytes": 9001750528,
                    "permissions": "r--s",
                },
            ],
            "maps_identity_required": True,
            "protocol_identity_required": True,
        },
        "devices": v23["devices"],
        "exit_authority": {
            "entrypoint": _artifact(AUTHORITY),
            "legacy_authorities_rejected": [
                "cp0_r1_evidence_v21.py",
                "cp0_r1_evidence_v22.py",
                "v23_readiness/cp0_r1_evidence_v23.py",
            ],
            "support": authority_support,
        },
        "gates": {
            "artifact_root_maximum_age_ns": 3_600_000_000_000,
            "fast_check_maximum_duration_ns": 5_000_000_000,
            "fresh_snapshot_maximum_age_ns": 5_000_000_000,
            "maximum_process_swap_bytes": 0,
            "maximum_system_swap_growth_bytes": 0,
            "phone_minimum_available_bytes": v23["gates"][
                "phone_minimum_available_bytes"
            ],
        },
        "historical_tested_primitives": {
            name: _artifact(path)
            for name, path in sorted(HISTORICAL_PRIMITIVES.items())
        },
        "incumbent_route_lock": v23["incumbent_route_lock"],
        "model_geometry": {
            model_a["model_id"]: v23["model_geometry"][model_a["model_id"]]
        },
        "orchestration_requirements": {
            "source_programs": {
                name: _artifact(path)
                for name, path in sorted(ORCHESTRATION_PROGRAMS.items())
            },
            "support": {
                name: record
                for name, record in sorted(orchestration_support.items())
            },
            "stage_support": {
                "fan_in": full_orchestration_support,
                "identity_binding": ["production_common"],
                "phase_lock": ["production_common"],
                "preparation": ["production_common"],
                "readiness_projection": full_orchestration_support,
            },
        },
        "parent": {
            "v2_2_contract_sha256": common.sha256_bytes(v22_raw),
            "v2_3_contract_sha256": common.sha256_bytes(v23_raw),
        },
        "phase_protocol": {
            "artifact_root_scope": "PRE_REBOOT_OUTSIDE_PHASE",
            "clock_id": "HOST_MONOTONIC_RAW",
            "order": [
                "ARTIFACT_ROOT",
                "REBOOT_PREPARATION",
                "PHASE_LOCK",
                "POST_REBOOT_IDENTITY_BINDING",
                "FAST_FRESH_READINESS",
                "ACQUISITION",
                "RUNTIME_IDENTITY",
                "V2_4_RAW_PREDICATE_REEVALUATION",
            ],
            "phase": "A_ONLY",
        },
        "producer_requirements": {
            "capture_receipt_roles": [
                "capture.cuda_monolithic",
                "capture.joint_phone_cuda",
            ],
            "capture_entrypoint_kinds": [
                "artifact_root",
                "cuda_monolithic",
                "fast_fresh_readiness",
                "joint_phone_cuda",
            ],
            "nested_capture_must_be_self_contained": True,
            "runtime_bundle_ids": [
                "cuda_monolithic",
                "cuda_route",
                "op12_stagenet",
                "op15_direct_relay",
                "op15_stagenet",
            ],
            "source_programs": {
                name: _artifact(path)
                for name, path in sorted(PRODUCER_PROGRAMS.items())
            },
            "status": "BLOCKED_UNTIL_BOUND_PRODUCTION_PLAN_EXISTS",
        },
        "quality_corpus": v23["quality_corpus"],
        "raw_predicate_contract": {
            "historical_v2_2_sha256": common.sha256_bytes(v22_raw),
            "manifest_name": "EVIDENCE_BUNDLE_V2_4.json",
            "parent_sha256": common.sha256_bytes(raw_parent_raw),
            "schema": raw_predicate["schema"],
            "sha256": common.sha256_bytes(raw_predicate_raw),
        },
        "schema": "s39-cp0-r1-evidence-contract-v2.4",
        "scope": "BOUNDED_EVIDENCE_CORRECTION_ONLY",
        "serving_envelope": {
            **v23["serving_envelope"],
            "n_ctx_seq": 512,
        },
        "status": "FROZEN_BEFORE_A_ONLY_ACQUISITION",
        "token_history_protocol": {
            "batch": 8,
            "continuation_tokens_per_request": 8,
            "decode_calls_after_prefill": 7,
            "n_batch": 64,
            "n_ctx_seq": 512,
            "n_ubatch": 64,
            "prefill_chunking": "WHOLE_POSITION_WAVES_MAX_64_ROWS",
            "prefill_row_order": "POSITION_MAJOR_THEN_ITEM_INDEX",
            "prompt_hash_encoding": "UTF-8",
            "request_id_mapping": "GROUP_LOCAL_ONE_BASED",
            "mechanics_item_indices": list(range(8)),
            "quality_group_count": 8,
            "quality_group_size": 8,
            "quality_items": 64,
            "seq_id_mapping": "GROUP_LOCAL_ZERO_BASED",
            "vocab_size": 151936,
        },
        "version": "2.4",
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
        print(f"V2_4_CONTRACT_REFUSED: {error}", file=sys.stderr)
        return 2
    print(f"V2_4_CONTRACT_WRITTEN: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
