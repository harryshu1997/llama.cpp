#!/usr/bin/env python3

import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "CP0_R1_TWO_ROUTE_ELIGIBILITY_CONTRACT.json"
CANDIDATE_PATH = HERE / "CP0_R1_CANDIDATE.json"
MANIFEST_PATH = HERE / "CP0_R1_SHA256SUMS.txt"


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_contract() -> dict[str, Any]:
    return {
        "schema": "s39-two-route-eligibility-contract-v1",
        "status": "FROZEN_BEFORE_CP0_R1_ACQUISITION",
        "scope": "CP0_R1_TWO_ROUTE_ELIGIBILITY_ONLY",
        "candidate_search": {
            "maximum_new_candidates": 1,
            "stop_after_first_candidate_failure": True,
            "cycle_authorized_only_after_two_route_pass": True,
        },
        "target_server": {
            "device_name": "NVIDIA GeForce RTX 4060 Ti",
            "device_uuid": "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08",
            "memory_total_bytes": 17_175_674_880,
            "minimum_free_vram_bytes": 536_870_912,
        },
        "serving_envelope": {
            "batch": 8,
            "max_streams": 8,
            "n_batch": 64,
            "n_ubatch": 64,
            "n_ctx_seq": 256,
            "kv_type_k": "f16",
            "kv_type_v": "f16",
            "sampler": "greedy",
        },
        "phone_memory": {
            "minimum_available_bytes_per_phone": 536_870_912,
            "maximum_process_swap_bytes": 0,
            "maximum_system_swap_growth_bytes": 0,
        },
        "task_quality": {
            "items": 64,
            "maximum_new_errors": 1,
            "maximum_score_regression_items": 1,
            "require_all_outputs_parseable": True,
            "cross_backend_greedy_agreement": "DIAGNOSTIC_ONLY",
            "cross_geometry_greedy_agreement": "DIAGNOSTIC_ONLY",
        },
        "live_bridge": {
            "minimum_useful_phone_tokens": 8,
            "minimum_requests_published_before_cuda_ready": 8,
        },
        "reprepare": {
            "maximum_elapsed_us": 30_000_000,
            "source": "PHONE_LOCAL_UFS_ONLY",
            "maximum_usb_weight_bytes": 0,
            "maximum_network_weight_bytes": 0,
        },
        "route_gates": [
            "ARTIFACT_AND_CONFIG_IDENTITY",
            "CUDA_B8_SERVE_WITH_HEADROOM",
            "FULL_TWO_PHONE_COVERAGE",
            "DIRECT_PHONE_ACTIVATION",
            "REALIZED_BACKEND_PLACEMENT",
            "POSITIVE_PHONE_MEMORY_HEADROOM",
            "ZERO_SWAP_GROWTH",
            "EXACT_HISTORY_POSITION_OWNERSHIP_CLEANUP",
            "INDEPENDENT_PATH_MATCHED_CUDA_ORACLE",
            "TASK_QUALITY_NONINFERIORITY",
            "USEFUL_PHONE_PUBLICATION_BEFORE_CUDA_READY",
        ],
        "pair_gates": [
            "MEASURED_CUDA_NON_CORESIDENCY",
            "LOCAL_UFS_REPREPARE_A_TO_B",
            "LOCAL_UFS_REPREPARE_B_TO_A",
        ],
        "claim_boundary": {
            "pass_status": "TWO_ROUTE_ELIGIBILITY_PASS",
            "pass_authorizes": "ONE_REDUCED_A_TO_B_TO_A_CYCLE",
            "pass_does_not_authorize": [
                "TRACE_REPLAY",
                "CONTROLLER_INTEGRATION",
                "ENERGY_ACQUISITION",
                "ARCHITECTURE_DIVERSITY_CLAIM",
            ],
        },
    }


def build_candidate() -> dict[str, Any]:
    return {
        "schema": "s39-cp0-r1-candidate-v1",
        "status": "CANDIDATE_SELECTED_ARTIFACT_NOT_ACQUIRED",
        "contract_sha256": hashlib.sha256(
            canonical_bytes(build_contract())
        ).hexdigest(),
        "candidate_attempt": 1,
        "candidate_attempt_limit": 1,
        "models": [
            {
                "slot": "A",
                "selection": "INCUMBENT_REQUIRES_FULL_QUALIFICATION",
                "model_id": "qwen3-14b-q4_k_m",
                "architecture": "qwen3",
                "quantization": "Q4_K_M",
                "n_layer": 40,
                "artifact": {
                    "file_name": "Qwen3-14B-Q4_K_M.gguf",
                    "bytes": 9_001_752_960,
                    "sha256": (
                        "500a8806e85ee9c83f3ae084202955924"
                        "51379b4f8cf2d0f41c15dffeb6b81f0"
                    ),
                },
                "route_binding": {
                    "backend": "GPUOpenCL",
                    "executed_cut_layer": 30,
                    "op15_stored_layers": [0, 32],
                    "op12_stored_layers": [24, 40],
                    "op15_shard_sha256": (
                        "ba56b9c5e19b3a4512777e6a47803cc0"
                        "3261c2d3c2734965cd5ec96b7c6c59fb"
                    ),
                    "op12_shard_sha256": (
                        "72e312af745160dc33a0ba39ba94fbbce"
                        "6112950d0409d39c42ddc3b25e756ab"
                    ),
                },
                "readiness_at_freeze": "PROVISIONAL_BATCH",
            },
            {
                "slot": "B",
                "selection": "ONLY_NEW_CANDIDATE",
                "model_id": "qwen3-8b-q8_0",
                "architecture": "qwen3",
                "quantization": "Q8_0",
                "n_layer": 36,
                "artifact": {
                    "file_name": "Qwen3-8B-Q8_0.gguf",
                    "bytes": 8_709_518_112,
                    "sha256": (
                        "408b955510e196121c1c375201744783b"
                        "5c9a43c7956d73fc78df54c66e883d6"
                    ),
                    "repository": "Qwen/Qwen3-8B-GGUF",
                    "revision": "6cfbfc7d8ab95bf485c79fcc40be60930d5b4c8c",
                    "upstream_model": "Qwen/Qwen3-8B",
                },
                "route_binding": None,
                "readiness_at_freeze": "NOT_ACQUIRED",
            },
        ],
        "task_suite": {
            "answer_mapping": "0=A,1=B,2=C,3=D",
            "dataset": "cais/mmlu",
            "revision": "bc5d09e5f0d160a95bcd36354bb5e16e50afe270",
            "split": "test",
            "items": 64,
            "selection": (
                "round-robin by zero-based row index over ASCII-ascending "
                "subject names; stop after 64 rows"
            ),
            "subject_order": "ASCII_ASCENDING_DATASET_CONFIG_NAME",
            "row_order": "SOURCE_PARQUET_ROW_ORDER",
            "text_normalization": "NONE",
            "prompt_format": (
                "Question: {question}\nA. {choice0}\nB. {choice1}\n"
                "C. {choice2}\nD. {choice3}\nAnswer with exactly one "
                "uppercase letter: A, B, C, or D.\nAnswer:"
            ),
            "maximum_output_tokens": 8,
            "answer_parser": "first non-whitespace standalone uppercase A-D",
            "few_shot_examples": 0,
            "chat_template": "NONE_RAW_COMPLETION",
        },
        "historical_routes": {
            "gemma-4-12b-it-q4_0": "FAIL_CORRECTNESS_UNCHANGED",
            "qwen2.5-14b-q4_0": "QUALITY_FAIL_UNCHANGED",
            "qwen2.5-14b-q8_0": "QUALITY_FAIL_UNCHANGED",
        },
    }


def main() -> None:
    CONTRACT_PATH.write_bytes(canonical_bytes(build_contract()))
    CANDIDATE_PATH.write_bytes(canonical_bytes(build_candidate()))
    paths = [
        CONTRACT_PATH,
        CANDIDATE_PATH,
        HERE / "RESULTS_CP0_R1.md",
        HERE / "build_cp0_r1_contract.py",
        HERE / "cp0_r1_eligibility.py",
        HERE / "tests" / "test_cp0_r1_eligibility.py",
    ]
    manifest = "".join(
        f"{sha256_file(path)}  {path.relative_to(HERE)}\n" for path in paths
    ).encode("ascii")
    MANIFEST_PATH.write_bytes(manifest)
    print(f"{CONTRACT_PATH.name} sha256={sha256_file(CONTRACT_PATH)}")
    print(f"{CANDIDATE_PATH.name} sha256={sha256_file(CANDIDATE_PATH)}")
    print(f"{MANIFEST_PATH.name} sha256={sha256_file(MANIFEST_PATH)}")


if __name__ == "__main__":
    main()
