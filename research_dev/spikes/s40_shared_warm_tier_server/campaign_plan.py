#!/usr/bin/env python3
"""Build the prospectively ordered S40 physical campaign."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
from typing import Any

from evidence_common import (
    EvidenceError,
    canonical_bytes,
    digest_file,
    read_json,
    require,
    require_int,
    require_string,
    validate_digest,
)


PRIMARY_ROTATIONS = (
    (
        ("C1_GPU_ONLY_OPTIMIZED", "WARM_HOST_CACHE"),
        ("C2_GPU_PLUS_CPU_WARM_EXECUTOR", "WARM_HOST_CACHE"),
        ("C1_GPU_ONLY_OPTIMIZED", "COLD_NVME"),
        ("T1_PHONE_WARM_TIER", "WARM_HOST_CACHE"),
    ),
    (
        ("C2_GPU_PLUS_CPU_WARM_EXECUTOR", "WARM_HOST_CACHE"),
        ("C1_GPU_ONLY_OPTIMIZED", "WARM_HOST_CACHE"),
        ("T1_PHONE_WARM_TIER", "WARM_HOST_CACHE"),
        ("C1_GPU_ONLY_OPTIMIZED", "COLD_NVME"),
    ),
    (
        ("T1_PHONE_WARM_TIER", "WARM_HOST_CACHE"),
        ("C1_GPU_ONLY_OPTIMIZED", "COLD_NVME"),
        ("C2_GPU_PLUS_CPU_WARM_EXECUTOR", "WARM_HOST_CACHE"),
        ("C1_GPU_ONLY_OPTIMIZED", "WARM_HOST_CACHE"),
    ),
)
CAMPAIGN_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,95}")
CAMPAIGN_KEYS = {
    "campaign_id",
    "experiment_contract_sha256",
    "physical_runs_are_sequential",
    "primary",
    "schema",
    "schema_version",
    "software_lock",
    "t2_repetitions",
}
CAMPAIGN_ROW_KEYS = {
    "cache_regime",
    "mode",
    "order",
    "phase",
    "repeat_index",
    "run_id",
}
SOFTWARE_LOCK_KEYS = {
    "campaign_plan_sha256",
    "campaign_reduce_sha256",
    "controller_binary_sha256",
    "evidence_bundle_manifest_sha256",
    "executor_bundle_manifest_sha256",
    "ldd_sha256",
    "native_bench_binary_sha256",
    "nvidia_smi_sha256",
    "physical_orchestrator_sha256",
    "python_sha256",
    "run_manifest_sha256",
    "schema",
}
CAMPAIGN_TOOL_PATHS = {
    "campaign_plan": Path(__file__).resolve(),
    "campaign_reduce": Path(__file__).resolve().with_name("campaign_reduce.py"),
    "physical_orchestrator":
        Path(__file__).resolve().with_name("physical_orchestrator.py"),
    "run_manifest": Path(__file__).resolve().with_name("run_manifest.py"),
}


def _row(
        campaign_id: str,
        phase: str,
        order: int,
        mode: str,
        cache_regime: str,
        repeat_index: int) -> dict[str, Any]:
    return {
        "cache_regime": cache_regime,
        "mode": mode,
        "order": order,
        "phase": phase,
        "repeat_index": repeat_index,
        "run_id": f"{campaign_id}-{order:03d}",
    }


def _validate_software_lock(value: Any) -> dict[str, str]:
    require(
        isinstance(value, dict) and set(value) == SOFTWARE_LOCK_KEYS,
        "campaign software lock: key set mismatch",
    )
    require(
        value["schema"] == "s40-primary-software-lock-v2",
        "campaign software lock: unsupported schema",
    )
    result = {"schema": value["schema"]}
    for key in sorted(SOFTWARE_LOCK_KEYS - {"schema"}):
        result[key] = validate_digest(
            value[key], f"campaign software lock.{key}")
    return result


def validate_campaign_tool_lock(value: Any) -> None:
    lock = _validate_software_lock(value)
    for name, path in CAMPAIGN_TOOL_PATHS.items():
        require(path.is_file(), f"campaign tool lock: missing {name}")
        require(
            digest_file(path) == lock[f"{name}_sha256"],
            f"campaign tool lock: changed {name}",
        )


def build_campaign(
        t2_repetitions: int,
        *,
        campaign_id: str,
        experiment_contract_sha256: str,
        software_lock: Any,
) -> dict[str, Any]:
    require(t2_repetitions in {1, 3},
            "campaign: T2 repetitions must be 1 or 3")
    campaign_id = require_string(campaign_id, "campaign ID")
    require(
        CAMPAIGN_ID_RE.fullmatch(campaign_id) is not None,
        "campaign ID: expected lowercase ASCII identifier",
    )
    experiment_contract_sha256 = validate_digest(
        experiment_contract_sha256, "campaign experiment contract SHA-256")
    frozen_software = _validate_software_lock(software_lock)
    primary = []
    order = 0
    for repeat_index, rotation in enumerate(PRIMARY_ROTATIONS):
        for mode, cache in rotation:
            primary.append(_row(
                campaign_id,
                "PRIMARY",
                order,
                mode,
                cache,
                repeat_index,
            ))
            order += 1
    for repeat_index in range(t2_repetitions):
        primary.append(_row(
            campaign_id,
            "T2_ISOLATION",
            order,
            "T2_PHONE_NO_PROMOTION",
            "WARM_HOST_CACHE",
            repeat_index,
        ))
        order += 1
    return {
        "campaign_id": campaign_id,
        "experiment_contract_sha256": experiment_contract_sha256,
        "physical_runs_are_sequential": True,
        "primary": primary,
        "schema": "s40-physical-campaign-v3",
        "schema_version": 3,
        "software_lock": frozen_software,
        "t2_repetitions": t2_repetitions,
    }


def validate_campaign(value: Any) -> dict[str, Any]:
    require(
        isinstance(value, dict) and set(value) == CAMPAIGN_KEYS,
        "campaign: key set mismatch",
    )
    require(
        value["schema"] == "s40-physical-campaign-v3"
        and value["schema_version"] == 3,
        "campaign: unsupported identity",
    )
    campaign_id = require_string(value["campaign_id"], "campaign ID")
    require(
        CAMPAIGN_ID_RE.fullmatch(campaign_id) is not None,
        "campaign ID: expected lowercase ASCII identifier",
    )
    contract_sha256 = validate_digest(
        value["experiment_contract_sha256"],
        "campaign experiment contract SHA-256",
    )
    software_lock = _validate_software_lock(value["software_lock"])
    t2_repetitions = require_int(
        value["t2_repetitions"], "campaign.t2_repetitions", 1)
    expected = build_campaign(
        t2_repetitions,
        campaign_id=campaign_id,
        experiment_contract_sha256=contract_sha256,
        software_lock=software_lock,
    )
    require(value == expected, "campaign: prospective row set mismatch")
    for index, row in enumerate(value["primary"]):
        require(
            isinstance(row, dict) and set(row) == CAMPAIGN_ROW_KEYS,
            f"campaign.primary[{index}]: key set mismatch",
        )
    return value


def read_campaign(path: Path) -> dict[str, Any]:
    campaign = read_json(path, "campaign")
    require(
        path.read_bytes() == canonical_bytes(campaign),
        "campaign: noncanonical JSON",
    )
    result = validate_campaign(campaign)
    validate_campaign_tool_lock(result["software_lock"])
    return result


def write_new(path: Path, value: dict[str, Any]) -> None:
    require(path.is_absolute(), "campaign output: expected absolute path")
    require(not path.exists(), "campaign output: already exists")
    with path.open("xb", buffering=0) as sink:
        sink.write(canonical_bytes(value))
        sink.flush()
        os.fsync(sink.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--t2-repetitions", type=int, default=1)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--experiment-contract-sha256", required=True)
    parser.add_argument("--software-lock", type=Path, required=True)
    args = parser.parse_args()
    try:
        require_int(args.t2_repetitions, "t2_repetitions", 1)
        campaign = build_campaign(
            args.t2_repetitions,
            campaign_id=args.campaign_id,
            experiment_contract_sha256=args.experiment_contract_sha256,
            software_lock=read_json(
                args.software_lock, "campaign software lock"),
        )
        write_new(args.output, campaign)
    except (EvidenceError, OSError) as error:
        print(f"ERROR: {error}")
        return 2
    print(canonical_bytes(campaign).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
