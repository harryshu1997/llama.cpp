#!/usr/bin/env python3

import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "CP0_R1_EVIDENCE_CONTRACT_V2.json"
MANIFEST_PATH = HERE / "CP0_R1_V2_SHA256SUMS.txt"
CANDIDATE_PATH = HERE / "CP0_R1_CANDIDATE.json"

PARENT_FILES = {
    "CP0_R1_CANDIDATE.json":
        "ee3196ca660fa7eb7ea293260dc98dd6fdbf14571a4d4c5aece5343fb29b28d8",
    "CP0_R1_SHA256SUMS.txt":
        "480cf8836e0a83ea9102c19e82bf5ec01e2f78b0ea2a3d9061be71bed4b31275",
    "CP0_R1_TWO_ROUTE_ELIGIBILITY_CONTRACT.json":
        "ffb2abeb33e818477e8a7181e177a6d296ed3b021767bc83a8df0d2450e5d095",
    "RESULTS_CP0_R1.md":
        "ae9b0ca6cd73fc6bab9da6f9c186558499542b8ad07c087ae8ff57dc82c08798",
    "build_cp0_r1_contract.py":
        "3f900a93bff85743958c80f713156aa4f14b0a667f7c767177a1e111a15b3f9a",
    "cp0_r1_eligibility.py":
        "c215a9548c71760a8c060c4b0c331dd9da8b964ce97bf8fe6174e23d36a94a45",
    "tests/test_cp0_r1_eligibility.py":
        "49768e330ebd528cd04a631d0a858263f5a8bedd5881b1c7fa973a60bd2108e3",
}


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_parent_candidate() -> dict[str, Any]:
    for relative, expected in PARENT_FILES.items():
        actual = sha256_file(HERE / relative)
        if actual != expected:
            raise RuntimeError(
                f"frozen CP0-R1 file changed: {relative}: {actual} != {expected}"
            )
    raw = CANDIDATE_PATH.read_bytes()
    candidate = json.loads(raw)
    if canonical_bytes(candidate) != raw:
        raise RuntimeError("CP0_R1_CANDIDATE.json is not canonical")
    return candidate


def required_roles(model_ids: list[str]) -> list[str]:
    roles = ["quality.corpus"]
    for model_id in model_ids:
        prefix = f"model.{model_id}"
        roles.extend(
            [
                f"{prefix}.route_lock",
                f"{prefix}.mechanics.phone",
                f"{prefix}.oracle.cuda_route",
                f"{prefix}.oracle.cuda_monolithic",
                f"{prefix}.cuda_memory",
                f"{prefix}.quality.cuda",
                f"{prefix}.quality.phone",
                f"{prefix}.bridge",
                f"{prefix}.placement.op15",
                f"{prefix}.placement.op12",
                f"{prefix}.route_transfer",
            ]
        )
    roles.extend(
        [
            "pair.cuda_memory",
            "reprepare.A_to_B",
            "reprepare.B_to_A",
        ]
    )
    return roles


def build_contract() -> dict[str, Any]:
    candidate = load_parent_candidate()
    models = [
        {
            "architecture": model["architecture"],
            "artifact_bytes": model["artifact"]["bytes"],
            "artifact_sha256": model["artifact"]["sha256"],
            "model_id": model["model_id"],
            "n_layer": model["n_layer"],
            "quantization": model["quantization"],
            "slot": model["slot"],
        }
        for model in candidate["models"]
    ]
    model_ids = [model["model_id"] for model in models]
    return {
        "schema": "s39-cp0-r1-evidence-contract-v2",
        "status": "FROZEN_BEFORE_CP0_R1_V2_MODEL_ACQUISITION",
        "scope": "EVIDENCE_HARDENING_AND_NO_MODEL_PREFLIGHT_ONLY",
        "parent": {
            "candidate_sha256": PARENT_FILES["CP0_R1_CANDIDATE.json"],
            "contract_sha256":
                PARENT_FILES["CP0_R1_TWO_ROUTE_ELIGIBILITY_CONTRACT.json"],
            "manifest_sha256": PARENT_FILES["CP0_R1_SHA256SUMS.txt"],
            "files": PARENT_FILES,
        },
        "candidate_lock": {
            "candidate_attempt": 1,
            "maximum_new_candidates": 1,
            "models": models,
            "task_suite_sha256": sha256_bytes(
                canonical_bytes(candidate["task_suite"])
            ),
        },
        "devices": {
            "cuda": {
                "host": "zhihao-Z690-C-ac",
                "memory_total_bytes": 17_175_674_880,
                "name": "NVIDIA GeForce RTX 4060 Ti",
                "uuid": "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08",
            },
            "op12": {
                "device": "OP595DL1",
                "model": "CPH2583",
                "product": "CPH2583",
                "serial": "5ae7a43d",
            },
            "op15": {
                "device": "OP611FL1",
                "model": "CPH2749",
                "product": "CPH2749",
                "serial": "3C15AU002CL00000",
            },
        },
        "serving_envelope": {
            "batch": 8,
            "kv_type_k": "f16",
            "kv_type_v": "f16",
            "max_streams": 8,
            "n_batch": 64,
            "n_ctx_seq": 256,
            "n_ubatch": 64,
            "sampler": "greedy",
        },
        "gates": {
            "cuda_minimum_free_bytes": 536_870_912,
            "phone_minimum_available_bytes": 536_870_912,
            "quality_items": 64,
            "quality_maximum_new_errors": 1,
            "quality_maximum_score_regression_items": 1,
            "bridge_minimum_requests": 8,
            "bridge_minimum_tokens": 8,
            "reprepare_maximum_elapsed_us": 30_000_000,
            "maximum_process_swap_bytes": 0,
            "maximum_system_swap_growth_bytes": 0,
        },
        "raw_evidence": {
            "artifact_format": "CANONICAL_ASCII_JSONL",
            "common_clock": "HOST_MONOTONIC_RAW",
            "maximum_artifact_bytes": 67_108_864,
            "read_semantics": "OPEN_ONCE_HASH_AND_PARSE_SAME_BYTES",
            "required_roles": required_roles(model_ids),
            "role_count": len(required_roles(model_ids)),
            "role_reuse": "FORBIDDEN",
            "summary_input": "FORBIDDEN",
        },
        "derivations": [
            "INDEPENDENT_PATH_MATCHED_CUDA_ORACLE",
            "CUDA_B8_MEMORY_AND_PAIR_NON_CORESIDENCY",
            "PER_ITEM_TASK_QUALITY",
            "COMMON_CLOCK_LIVE_BRIDGE",
            "REALIZED_PHONE_PLACEMENT",
            "DIRECT_PHONE_ACTIVATION",
            "FULL_LOCAL_UFS_REPREPARE",
        ],
        "claim_boundary": {
            "mechanics_status": "CP0_R1_V2_EVIDENCE_MECHANICS_PASS",
            "eligibility_status": "TWO_ROUTE_ELIGIBILITY_PASS",
            "eligibility_requires_complete_raw_bundle": True,
            "preflight_pass_does_not_authorize_model_acquisition": True,
            "forbidden_before_eligibility": [
                "MODEL_SWITCH_CYCLE",
                "TRACE_REPLAY",
                "CONTROLLER_INTEGRATION",
                "ENERGY_ACQUISITION",
            ],
        },
    }


def main() -> None:
    CONTRACT_PATH.write_bytes(canonical_bytes(build_contract()))
    sources = [
        CONTRACT_PATH,
        HERE / "RESULTS_CP0_R1_V2.md",
        HERE / "build_cp0_r1_v2.py",
        HERE / "cp0_r1_evidence_v2.py",
        HERE / "cp0_r1_preflight_v2.py",
        HERE / "validate_cp0_r1_preflight_v2.py",
        HERE / "tests" / "test_cp0_r1_evidence_v2.py",
    ]
    MANIFEST_PATH.write_bytes(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(HERE)}\n"
            for path in sources
        ).encode("ascii")
    )
    print(f"{CONTRACT_PATH.name} sha256={sha256_file(CONTRACT_PATH)}")
    print(f"{MANIFEST_PATH.name} sha256={sha256_file(MANIFEST_PATH)}")


if __name__ == "__main__":
    main()
