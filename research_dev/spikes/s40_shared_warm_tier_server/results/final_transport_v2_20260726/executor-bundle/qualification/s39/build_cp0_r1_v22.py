#!/usr/bin/env python3

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "CP0_R1_EVIDENCE_CONTRACT_V2_2.json"
MANIFEST_PATH = HERE / "CP0_R1_V2_2_SHA256SUMS.txt"
CANDIDATE_PATH = HERE / "CP0_R1_CANDIDATE.json"
PARENT_CONTRACT_PATH = HERE / "CP0_R1_EVIDENCE_CONTRACT_V2_1.json"
PARENT_MANIFEST_PATH = HERE / "CP0_R1_V2_1_SHA256SUMS.txt"
CORPUS_PATH = HERE / "CP0_R1_MMLU64_CORPUS_V2_2.jsonl"
SOURCE_MANIFEST_PATH = HERE / "CP0_R1_MMLU64_SOURCES_V2_2.json"
CORPUS_BUILDER_PATH = HERE / "build_cp0_r1_mmlu64_v22.py"

PARENT_CONTRACT_SHA256 = (
    "cd3fb16c2b053ac0cf4ae699d6fcfbdd3ffc6a6050d9b96f8b1f15252bea7393"
)
PARENT_MANIFEST_SHA256 = (
    "4cc9f08b85793f3d5f93f3eff97879102e11f79c6b669dc21af234607e8a893f"
)
CANDIDATE_SHA256 = (
    "ee3196ca660fa7eb7ea293260dc98dd6fdbf14571a4d4c5aece5343fb29b28d8"
)
CORPUS_SHA256 = (
    "3ffafee1615ae2de690a2726b880823e167a3d9c210c5faed86d8f0e93ecff4f"
)
SOURCE_MANIFEST_SHA256 = (
    "424120b2a34531d13e57dc82e6838d814488d01fa452a8d8e578276ea0e02ef8"
)
CORPUS_BUILDER_SHA256 = (
    "2f052a61790c9714aad82f9f87e79d84230a84d7e68db74e2d02aaff09a84c32"
)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_canonical(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected_sha256:
        raise RuntimeError(f"frozen file changed: {path.name}")
    value = json.loads(raw)
    if canonical_bytes(value) != raw:
        raise RuntimeError(f"frozen file is not canonical: {path.name}")
    return value


def verify_parent() -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(PARENT_MANIFEST_PATH) != PARENT_MANIFEST_SHA256:
        raise RuntimeError("frozen V2.1 manifest changed")
    for line in PARENT_MANIFEST_PATH.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        if sha256_file(HERE / relative) != expected:
            raise RuntimeError(f"frozen V2.1 file changed: {relative}")
    parent = load_canonical(PARENT_CONTRACT_PATH, PARENT_CONTRACT_SHA256)
    candidate = load_canonical(CANDIDATE_PATH, CANDIDATE_SHA256)
    frozen = (
        (CORPUS_PATH, CORPUS_SHA256),
        (SOURCE_MANIFEST_PATH, SOURCE_MANIFEST_SHA256),
        (CORPUS_BUILDER_PATH, CORPUS_BUILDER_SHA256),
    )
    for path, expected in frozen:
        if sha256_file(path) != expected:
            raise RuntimeError(f"frozen V2.2 input changed: {path.name}")
    return parent, candidate


def build_contract() -> dict[str, Any]:
    parent, candidate = verify_parent()
    contract = copy.deepcopy(parent)
    model_a = next(model for model in candidate["models"] if model["slot"] == "A")
    route = model_a["route_binding"]
    if route is None:
        raise RuntimeError("incumbent route binding is absent")
    contract["schema"] = "s39-cp0-r1-evidence-contract-v2.2"
    contract["status"] = "FROZEN_BEFORE_QWEN3_14B_QUALIFICATION"
    contract["parent"] = {
        "candidate_sha256": CANDIDATE_SHA256,
        "contract_sha256": PARENT_CONTRACT_SHA256,
        "manifest_sha256": PARENT_MANIFEST_SHA256,
    }
    contract["quality_corpus"] = {
        "builder_sha256": CORPUS_BUILDER_SHA256,
        "bytes": CORPUS_PATH.stat().st_size,
        "dataset": candidate["task_suite"]["dataset"],
        "items": 64,
        "path": CORPUS_PATH.name,
        "revision": candidate["task_suite"]["revision"],
        "sha256": CORPUS_SHA256,
        "source_manifest_path": SOURCE_MANIFEST_PATH.name,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
    }
    contract["gates"]["cuda_quality_minimum_correct_items"] = 25
    contract["gates"]["continuation_tokens_per_request"] = 8
    contract["incumbent_route_lock"] = {
        "backend": route["backend"],
        "cut_layer": route["executed_cut_layer"],
        "model_id": model_a["model_id"],
        "op12_shard_sha256": route["op12_shard_sha256"],
        "op12_stored_layers": route["op12_stored_layers"],
        "op15_shard_sha256": route["op15_shard_sha256"],
        "op15_stored_layers": route["op15_stored_layers"],
    }
    contract["phase_protocol"]["distinct_phase_ids"] = True
    contract["closed_checks"] = [
        *contract["closed_checks"],
        "PINNED_CANONICAL_MMLU64_CORPUS",
        "CUDA_ABOVE_CHANCE_SANITY_FLOOR",
        "EXACT_INCUMBENT_ROUTE_BINDING",
        "EXACT_EIGHT_TOKEN_CONTINUATIONS",
        "MECHANICS_PUBLICATION_READY_CAUSAL_ORDER",
        "DISTINCT_PHASE_IDENTITIES",
        "ROOT_REEVALUATED_CYCLE_AUTHORIZATION",
    ]
    contract["claim_boundary"]["exit_authority"] = "cp0_r1_evidence_v22.py"
    contract["claim_boundary"]["cycle_authorization"] = {
        "legacy_status_results_accepted": False,
        "requires_mode": "AUTHORIZE_CYCLE",
        "requires_raw_bundle_roots": ["A_ONLY", "B_ONLY", "PAIR"],
        "status": "ONE_REDUCED_A_TO_B_TO_A_CYCLE_AUTHORIZED",
    }
    return contract


def main() -> None:
    CONTRACT_PATH.write_bytes(canonical_bytes(build_contract()))
    sources = [
        CONTRACT_PATH,
        CORPUS_PATH,
        SOURCE_MANIFEST_PATH,
        HERE / "RESULTS_CP0_R1_V2_2.md",
        HERE / "build_cp0_r1_mmlu64_v22.py",
        HERE / "build_cp0_r1_v22.py",
        HERE / "cp0_r1_evidence_v22.py",
        HERE / "cp0_r1_phase_preflight_v22.py",
        HERE / "tests" / "test_cp0_r1_evidence_v22.py",
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
