#!/usr/bin/env python3
"""Compile a GGUF-derived Llama CPU/phone FFN policy with the unified VQ."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    MatmulOp,
    MatmulSystemProfile,
    ModelProgram,
    UnifiedScheduler,
    materialize_matmul_profile,
)


MANIFEST_SCHEMA = "s42-llama-dense-ffn-manifest-v1"
CAMPAIGN_SCHEMA = "s42-kernel-energy-profile-v1"
SCHEMA = "s42-llama-ffn-vq-compiled-policy-v1"
PHYSICAL_CALIBRATION_SCHEMA_V1 = (
    "s42-llama-ffn-physical-shape-calibration-v1"
)
PHYSICAL_CALIBRATION_SCHEMA = (
    "s42-llama-ffn-physical-shape-calibration-v2"
)


class CompileError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CompileError(message)


def canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(type(value) is dict, f"object expected: {path}")
    return value


def verify_manifest(value: dict[str, Any]) -> None:
    supplied = value.get("record_sha256")
    unsigned = {key: row for key, row in value.items() if key != "record_sha256"}
    require(
        value.get("schema") == MANIFEST_SCHEMA
        and supplied == hashlib.sha256(canonical(unsigned)).hexdigest(),
        "FFN manifest identity",
    )


def verify_record(value: dict[str, Any], schema: str, label: str) -> None:
    supplied = value.get("record_sha256")
    unsigned = {
        key: row for key, row in value.items() if key != "record_sha256"
    }
    require(
        value.get("schema") == schema
        and supplied == hashlib.sha256(canonical(unsigned)).hexdigest(),
        f"{label} record identity",
    )


def model_contract(manifest: dict[str, Any]) -> dict[str, int | str]:
    geometry = manifest["geometry"]
    split = manifest["split_contract"]
    return {
        "column_quantum": split["column_quantum"],
        "max_tokens": split["max_tokens"],
        "model_sha256": manifest["model"]["sha256"],
        "n_embd": geometry["n_embd"],
        "n_ff": geometry["n_ff"],
        "quantization": geometry["quantization"],
    }


def placement_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    geometry = manifest["geometry"]
    resident = manifest["resident_slice"]
    return {
        "model_sha256": manifest["model"]["sha256"],
        "resident_layer_ids": geometry["resident_layer_ids"],
        "resident_slice_raw_bytes": resident["raw_bytes"],
        "resident_slice_weight_sha256": resident["weight_sha256"],
        "split_contract": manifest["split_contract"],
    }


def calibrated_buckets(
    manifest: dict[str, Any],
    estimated_policy_text: str,
    calibration: dict[str, Any],
) -> list[dict[str, int]]:
    calibration_schema = calibration.get("schema")
    require(
        calibration_schema in {
            PHYSICAL_CALIBRATION_SCHEMA_V1,
            PHYSICAL_CALIBRATION_SCHEMA,
        },
        "physical calibration schema",
    )
    verify_record(calibration, calibration_schema, "physical calibration")
    contract = model_contract(manifest)
    rows = calibration.get("compiled_buckets")
    exact_placement = (
        calibration.get("evidence", {}).get("manifest_record_sha256")
            == manifest["record_sha256"]
        if calibration_schema == PHYSICAL_CALIBRATION_SCHEMA_V1
        else calibration.get("placement_contract")
            == placement_contract(manifest)
    )
    require(
        calibration.get("status") == "PASS"
        and calibration.get("model_contract") == contract
        and exact_placement
        and calibration.get("source_policy_text") == estimated_policy_text
        and calibration.get("qualification", {}).get("route_admission")
            == "SHADOW_ONLY_UNTIL_HELDOUT_PHYSICAL_PROFILE"
        and type(rows) is list
        and rows,
        "physical calibration contract",
    )
    previous = 0
    output = []
    for row in rows:
        maximum = row.get("max_tokens")
        phone = row.get("phone_columns")
        cpu = row.get("cpu_columns")
        require(
            type(maximum) is int
            and previous < maximum <= contract["max_tokens"]
            and type(phone) is int
            and type(cpu) is int
            and phone >= 0
            and cpu >= 0
            and phone + cpu == contract["n_ff"]
            and phone % contract["column_quantum"] == 0,
            "physical calibration bucket",
        )
        output.append({
            "cpu_columns": cpu,
            "max_tokens": maximum,
            "phone_columns": phone,
        })
        previous = maximum
    require(previous == contract["max_tokens"], "calibration coverage")
    return output


def compile_policy(
    manifest: dict[str, Any],
    campaign: dict[str, Any],
    *,
    phone_capacity_bytes: int,
    phone_available_bytes: int,
    phone_reserve_bytes: int,
    physical_calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    verify_manifest(manifest)
    require(campaign.get("schema") == CAMPAIGN_SCHEMA, "kernel campaign")
    require(
        0 < phone_reserve_bytes <= phone_available_bytes <= phone_capacity_bytes,
        "phone memory snapshot",
    )
    weight_bytes = manifest["resident_slice"]["raw_bytes"]
    occupied_including_slice = phone_capacity_bytes - phone_available_bytes
    occupied_before_slice = max(0, occupied_including_slice - weight_bytes)
    materializer_reserved = occupied_before_slice + phone_reserve_bytes
    require(
        materializer_reserved + weight_bytes <= phone_capacity_bytes,
        "phone split residency capacity",
    )
    profile_value = materialize_matmul_profile(
        campaign,
        generic_family=True,
        phone_capacity_bytes=phone_capacity_bytes,
        phone_reserved_bytes=materializer_reserved,
        gpu_reserved_bytes=16 * 1024**3,
    )
    profile = MatmulSystemProfile.from_json(profile_value)
    decisions = []
    buckets = []
    previous_m = 0
    for op_value in manifest["aggregate_ops"]:
        op = MatmulOp.from_json(op_value)
        require(op.m > previous_m, "ordered aggregate M buckets")
        previous_m = op.m
        scheduler = UnifiedScheduler((), "adaptive", matmul_profile=profile)
        program = ModelProgram(
            program_id=f"compile-{op.op_id}",
            model_id=manifest["model"]["id"],
            arrival_us=0,
            deadline_us=max(1, op.deadline_us or 600_000_000),
            ops=(op,),
        )
        scheduler.enqueue_matmul(program)
        decision = scheduler.schedule_next_matmul(0)
        require(decision is not None and decision.get("kind") == "matmul", "VQ decision")
        placement = decision.get("placement")
        require(type(placement) is dict, "VQ placement")
        phone_columns = placement.get("phone")
        cpu_columns = placement.get("cpu")
        require(
            type(phone_columns) is int
            and type(cpu_columns) is int
            and phone_columns + cpu_columns == op.n
            and phone_columns % op.split_quantum_n == 0,
            "physical split geometry",
        )
        buckets.append({
            "cpu_columns": cpu_columns,
            "max_tokens": op.m,
            "phone_columns": phone_columns,
        })
        decisions.append(decision)
    estimated_policy_text = ",".join(
        f"{row['max_tokens']}:{row['phone_columns']}" for row in buckets
    )
    if physical_calibration is not None:
        buckets = calibrated_buckets(
            manifest, estimated_policy_text, physical_calibration
        )
    policy_text = ",".join(
        f"{row['max_tokens']}:{row['phone_columns']}" for row in buckets
    )
    evidence = {
        "kernel_campaign_profile_id": campaign["profile_id"],
        "kernel_campaign_status": "generic_shape_transfer",
        "manifest_record_sha256": manifest["record_sha256"],
        "matmul_profile_sha256": hashlib.sha256(
            canonical(profile_value)
        ).hexdigest(),
    }
    if physical_calibration is not None:
        evidence["physical_calibration_record_sha256"] = (
            physical_calibration["record_sha256"]
        )
    result: dict[str, Any] = {
        "compiled_buckets": buckets,
        "compiler": "research_dev.scheduler.UnifiedScheduler.matmul",
        "evidence": evidence,
        "memory_accounting": {
            "occupied_before_slice_bytes": occupied_before_slice,
            "phone_available_after_slice_bytes": phone_available_bytes,
            "phone_capacity_bytes": phone_capacity_bytes,
            "phone_reserve_bytes": phone_reserve_bytes,
            "resident_slice_bytes": weight_bytes,
            "vq_reserved_bytes": materializer_reserved,
        },
        "policy_text": policy_text,
        "qualification": {
            "physical_execution_qualified": False,
            "physical_shape_calibrated": physical_calibration is not None,
            "route_admission": "SHADOW_ONLY_UNTIL_HELDOUT_PHYSICAL_PROFILE",
            "status": (
                "PHYSICALLY_CALIBRATED_SHADOW"
                if physical_calibration is not None else "ESTIMATED"
            ),
        },
        "schema": SCHEMA,
        "vq_decisions": decisions,
    }
    result["record_sha256"] = hashlib.sha256(canonical(result)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kernel-campaign", type=Path, required=True)
    parser.add_argument("--phone-capacity-bytes", type=int, required=True)
    parser.add_argument("--phone-available-bytes", type=int, required=True)
    parser.add_argument("--phone-reserve-bytes", type=int, required=True)
    parser.add_argument("--physical-calibration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.is_absolute() and not args.output.exists(), "new output")
    value = compile_policy(
        load(args.manifest),
        load(args.kernel_campaign),
        phone_capacity_bytes=args.phone_capacity_bytes,
        phone_available_bytes=args.phone_available_bytes,
        phone_reserve_bytes=args.phone_reserve_bytes,
        physical_calibration=(
            None
            if args.physical_calibration is None
            else load(args.physical_calibration)
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical(value))
    print(json.dumps({
        "output": str(args.output),
        "policy": value["policy_text"],
        "qualification": value["qualification"]["status"],
        "status": "PASS",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
