#!/usr/bin/env python3

import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "CP0_R1_EVIDENCE_CONTRACT_V2_1.json"
MANIFEST_PATH = HERE / "CP0_R1_V2_1_SHA256SUMS.txt"
CANDIDATE_PATH = HERE / "CP0_R1_CANDIDATE.json"
PARENT_CONTRACT_PATH = HERE / "CP0_R1_EVIDENCE_CONTRACT_V2.json"
PARENT_MANIFEST_PATH = HERE / "CP0_R1_V2_SHA256SUMS.txt"
SHARD_MANIFEST_PATH = HERE / "SHARD_MANIFEST.json"

PARENT_CONTRACT_SHA256 = (
    "a1ae7f58a63dfbfeae7e29619b63da875a4e411ce60c11973907b0831b48d534"
)
PARENT_MANIFEST_SHA256 = (
    "517c23a049f3378ca10d71d5397abbe4dd7730bdc251a179fd84e96968d15d8c"
)
CANDIDATE_SHA256 = (
    "ee3196ca660fa7eb7ea293260dc98dd6fdbf14571a4d4c5aece5343fb29b28d8"
)
SHARD_MANIFEST_SHA256 = (
    "738626a4d9292407631b97ca1b05ea027d94a54466dfa68dfab424840eb4ea39"
)


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


def load_canonical(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected_sha256:
        raise RuntimeError(f"frozen file changed: {path.name}")
    value = json.loads(raw)
    if canonical_bytes(value) != raw:
        raise RuntimeError(f"frozen file is not canonical: {path.name}")
    return value


def verify_parent() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256_file(PARENT_MANIFEST_PATH) != PARENT_MANIFEST_SHA256:
        raise RuntimeError("frozen V2 manifest changed")
    for line in PARENT_MANIFEST_PATH.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        if sha256_file(HERE / relative) != expected:
            raise RuntimeError(f"frozen V2 file changed: {relative}")
    parent = load_canonical(PARENT_CONTRACT_PATH, PARENT_CONTRACT_SHA256)
    candidate = load_canonical(CANDIDATE_PATH, CANDIDATE_SHA256)
    shard_raw = SHARD_MANIFEST_PATH.read_bytes()
    if sha256_bytes(shard_raw) != SHARD_MANIFEST_SHA256:
        raise RuntimeError("frozen file changed: SHARD_MANIFEST.json")
    shards = json.loads(shard_raw)
    return parent, candidate, shards


def model_roles(model_id: str) -> list[str]:
    prefix = f"model.{model_id}"
    return [
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


def build_contract() -> dict[str, Any]:
    parent, candidate, shards = verify_parent()
    models = {model["slot"]: model for model in candidate["models"]}
    model_a = models["A"]
    model_b = models["B"]
    a_shards = shards["models"][model_a["model_id"]]["placements"]
    phase_roles = {
        "A_ONLY": [
            "phase.lock",
            "phase.preflight",
            "quality.corpus",
            *model_roles(model_a["model_id"]),
        ],
        "B_ONLY": [
            "phase.lock",
            "phase.preflight",
            "quality.corpus",
            *model_roles(model_b["model_id"]),
        ],
        "PAIR": [
            "phase.lock",
            "phase.preflight",
            "pair.cuda_memory",
            "reprepare.A_to_B",
            "reprepare.B_to_A",
        ],
    }
    return {
        "schema": "s39-cp0-r1-evidence-contract-v2.1",
        "status": "FROZEN_BEFORE_QWEN3_14B_QUALIFICATION",
        "scope": "BOUNDED_EVIDENCE_CORRECTION_ONLY",
        "parent": {
            "candidate_sha256": CANDIDATE_SHA256,
            "contract_sha256": PARENT_CONTRACT_SHA256,
            "manifest_sha256": PARENT_MANIFEST_SHA256,
            "shard_manifest_sha256": SHARD_MANIFEST_SHA256,
        },
        "candidate_lock": parent["candidate_lock"],
        "devices": parent["devices"],
        "serving_envelope": parent["serving_envelope"],
        "gates": {
            **parent["gates"],
            "phase_preflight_maximum_age_ns": 5_000_000_000,
        },
        "model_geometry": {
            model_a["model_id"]: {
                "activation_dtype": "F32",
                "activation_element_bytes": 4,
                "cuda_model_path": "/home/zhihao/models/Qwen3-14B-Q4_K_M.gguf",
                "hidden_size": 5120,
                "known_shards": {
                    phone: {
                        "bytes": a_shards[phone]["bytes"],
                        "path": a_shards[phone]["remote_path"],
                        "sha256": a_shards[phone]["sha256"],
                    }
                    for phone in ("op15", "op12")
                },
            },
            model_b["model_id"]: {
                "activation_dtype": "F32",
                "activation_element_bytes": 4,
                "cuda_model_path": "/home/zhihao/models/Qwen3-8B-Q8_0.gguf",
                "hidden_size": 4096,
                "known_shards": None,
                "planned_phone_path": (
                    "/data/local/tmp/s39-active-warm/v2/models/"
                    "qwen3-8b-q8_0/weights.gguf"
                ),
            },
        },
        "phase_protocol": {
            "clock_id": "HOST_MONOTONIC_RAW",
            "order": ["A_ONLY", "B_ONLY", "PAIR"],
            "phase_roles": phase_roles,
            "lock_before_acquisition": True,
            "all_events_inside_phase_interval": True,
            "fresh_readiness_before_each_acquisition": True,
            "prior_results_rederived_from_raw_bundles": True,
        },
        "preflight": {
            "adb_ports": [5037, 5038],
            "phone_adb_port": 5038,
            "ssh_target": "zhihao@172.20.74.85",
            "commands": "EXACT_ARGV_DERIVED_FROM_PHASE_LOCK",
            "checks": [
                "CUDA_IDENTITY_AND_FULL_MODEL",
                "ADB_TOPOLOGY",
                "PHONE_IDENTITY_AND_FULL_SHARD",
            ],
        },
        "closed_checks": [
            "CORPUS_AND_PER_ITEM_QUALITY_LINKAGE",
            "EXACT_CUDA_MEMORY_ACCOUNTING",
            "EQUAL_ORACLE_LENGTH_AND_B8_CALL_GEOMETRY",
            "PHONE_PUBLICATION_TO_MECHANICS_AND_CUDA_READY_LINKAGE",
            "EXACT_ACTIVATION_TRANSFER_BYTES",
            "FULL_SHARD_LOCAL_UFS_REPREPARE",
        ],
        "claim_boundary": {
            "phase_status": {
                "A_ONLY": "MODEL_A_QUALIFICATION_PASS",
                "B_ONLY": "MODEL_B_QUALIFICATION_PASS",
                "PAIR": "TWO_ROUTE_ELIGIBILITY_PASS",
            },
            "exit_authority": "cp0_r1_evidence_v21.py",
            "forbidden_before_pair_pass": parent["claim_boundary"][
                "forbidden_before_eligibility"
            ],
        },
    }


def main() -> None:
    CONTRACT_PATH.write_bytes(canonical_bytes(build_contract()))
    sources = [
        CONTRACT_PATH,
        HERE / "RESULTS_CP0_R1_V2_1.md",
        HERE / "build_cp0_r1_v21.py",
        HERE / "cp0_r1_evidence_v21.py",
        HERE / "cp0_r1_phase_preflight_v21.py",
        HERE / "tests" / "test_cp0_r1_evidence_v21.py",
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
