#!/usr/bin/env python3

import copy
from pathlib import Path

import v23_common as common


HERE = Path(__file__).resolve().parent
S39 = HERE.parent
PARENT_CONTRACT = S39 / "CP0_R1_EVIDENCE_CONTRACT_V2_2.json"
PARENT_MANIFEST = S39 / "CP0_R1_V2_2_SHA256SUMS.txt"
CANDIDATE = S39 / "CP0_R1_CANDIDATE.json"
OUTPUT = HERE / "CP0_R1_EVIDENCE_CONTRACT_V2_3.json"

PARENT_CONTRACT_SHA256 = (
    "3d13dcf91e2ea0ad809518a1a19123874ff3612ceb1c25a88e842d4dd3e2292f"
)
PARENT_MANIFEST_SHA256 = (
    "933753871124c3d88ed3ec0ca5df3971fb7cd521b9e31782a793b65e2b67b76d"
)
CANDIDATE_SHA256 = (
    "ee3196ca660fa7eb7ea293260dc98dd6fdbf14571a4d4c5aece5343fb29b28d8"
)


def load_parent() -> tuple[dict, dict]:
    if common.sha256_file(PARENT_MANIFEST) != PARENT_MANIFEST_SHA256:
        raise RuntimeError("frozen V2.2 manifest changed")
    parent, parent_raw = common.read_canonical(PARENT_CONTRACT)
    candidate, candidate_raw = common.read_canonical(CANDIDATE)
    common.exact(
        common.sha256_bytes(parent_raw),
        PARENT_CONTRACT_SHA256,
        "parent contract",
    )
    common.exact(
        common.sha256_bytes(candidate_raw),
        CANDIDATE_SHA256,
        "candidate",
    )
    return parent, candidate


def build_contract() -> dict:
    parent, candidate = load_parent()
    result = copy.deepcopy(parent)
    result["schema"] = "s39-cp0-r1-evidence-contract-v2.3"
    result["status"] = "FROZEN_BEFORE_QWEN3_14B_QUALIFICATION"
    result["parent"] = {
        "candidate_sha256": CANDIDATE_SHA256,
        "contract_sha256": PARENT_CONTRACT_SHA256,
        "manifest_sha256": PARENT_MANIFEST_SHA256,
    }
    result["readiness_v2_3"] = {
        "artifact_snapshot_may_precede_phase_lock": True,
        "fresh_snapshot_must_follow_phase_lock": True,
        "fresh_snapshot_maximum_age_ns": 5_000_000_000,
        "runtime_boot_id_link_required": True,
        "runtime_digest_link_required": True,
        "phone_maximum_gpu_temp_millic": 85_000,
        "phone_minimum_available_bytes": parent["gates"][
            "phone_minimum_available_bytes"
        ],
        "maximum_process_swap_bytes": parent["gates"][
            "maximum_process_swap_bytes"
        ],
        "maximum_system_swap_growth_bytes": parent["gates"][
            "maximum_system_swap_growth_bytes"
        ],
        "stat_fields": [
            "device_id",
            "inode",
            "size",
            "mtime_ns",
            "ctime_ns",
            "mode",
        ],
        "cuda_identity": {
            "host": parent["devices"]["cuda"]["host"],
            "memory_total_bytes": parent["devices"]["cuda"][
                "memory_total_bytes"
            ],
            "name": parent["devices"]["cuda"]["name"],
            "uuid": parent["devices"]["cuda"]["uuid"],
        },
        "phone_identity": {
            phone: {
                key: parent["devices"][phone][key]
                for key in ("device", "model", "product", "serial")
            }
            for phone in ("op15", "op12")
        },
    }
    result["closed_checks"] = [
        *parent["closed_checks"],
        "LONG_HASH_SEPARATED_FROM_FRESH_IDENTITY",
        "FRESH_IDENTITY_AFTER_PHASE_LOCK",
        "RUNTIME_BOOT_ID_LINKAGE",
        "EXACT_CUDA_DEVICE_LINKAGE",
        "RUNTIME_ARTIFACT_DIGEST_LINKAGE",
        "WORKER_PROCESS_AND_EXECUTABLE_IDENTITY",
        "V2_2_RAW_BUNDLE_REEVALUATION",
        "ACTIVATION_BYTES_TO_INTERFACE_COUNTER_FLOOR",
    ]
    result["claim_boundary"]["exit_authority"] = (
        "v23_readiness/cp0_r1_evidence_v23.py"
    )
    result["claim_boundary"]["cycle_authorization"][
        "requires_readiness_version"
    ] = "V2.3"
    return result


def main() -> None:
    OUTPUT.write_bytes(common.canonical_bytes(build_contract()))
    print(f"{common.sha256_file(OUTPUT)}  {OUTPUT}")


if __name__ == "__main__":
    main()
