#!/usr/bin/env python3
"""Build the runtime-auto profile from disjoint physical evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import ProfileBundle  # noqa: E402


PHASE_AUDIT_SCHEMA = "s42-fp16-llama1b-contention-calibration-v1"
NATURAL_AUDIT_SCHEMA = "s42-fp16-llama1b-natural-route-calibration-v1"
OUTPUT_AUDIT_SCHEMA = "s42-fp16-llama1b-runtime-auto-profile-v1"


class BuildError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


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


def route(profile: dict[str, Any], route_id: str) -> dict[str, Any]:
    rows = profile.get("routes")
    require(type(rows) is list, "profile routes")
    matches = [
        row for row in rows
        if type(row) is dict and row.get("route_id") == route_id
    ]
    require(len(matches) == 1, f"profile route: {route_id}")
    return matches[0]


def build(
    phase_profile_path: Path,
    phase_audit_path: Path,
    natural_profile_path: Path,
    natural_audit_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    phase_profile = load(phase_profile_path)
    phase_audit = load(phase_audit_path)
    natural_profile = load(natural_profile_path)
    natural_audit = load(natural_audit_path)

    phase_rows = phase_audit.get("variants")
    require(
        phase_audit.get("schema") == PHASE_AUDIT_SCHEMA
        and phase_audit.get("status") == "PASS"
        and phase_audit.get("all_variants_measured") is True
        and type(phase_rows) is list
        and {row.get("class_id") for row in phase_rows} == set(range(1, 9))
        and all(
            row.get("measured") is True
            and row.get("holdout_upper_violations") == 0
            and type(row.get("train_count")) is int
            and row["train_count"] >= 7
            and type(row.get("holdout_count")) is int
            and row["holdout_count"] >= 3
            for row in phase_rows
        ),
        "eight-class phase audit",
    )
    natural_rows = natural_audit.get("variants")
    require(
        natural_audit.get("schema") == NATURAL_AUDIT_SCHEMA
        and natural_audit.get("status") == "PASS"
        and natural_audit.get("all_variants_measured") is True
        and natural_audit.get("profile_sha256")
            == digest(natural_profile_path)
        and type(natural_rows) is list,
        "natural route audit",
    )
    phone_audits = [
        row for row in natural_rows
        if row.get("route_id") == "phone-adreno"
    ]
    require(
        len(phone_audits) == 1
        and phone_audits[0].get("measured") is True
        and phone_audits[0].get("holdout_upper_violations") == 0
        and phone_audits[0].get("train_count", 0) >= 6
        and phone_audits[0].get("holdout_count", 0) >= 4,
        "natural phone audit",
    )

    cpu_latency = route(phase_profile, "desktop-cpu").get("latency")
    require(
        type(cpu_latency) is dict
        and cpu_latency.get("kind") == "conditioned_affine_features_v1"
        and cpu_latency.get("selector_feature") == "contention_class_id"
        and {
            row.get("selector_value")
            for row in cpu_latency.get("variants", [])
        } == set(range(1, 9))
        and all(
            row.get("measured") is True
            for row in cpu_latency.get("variants", [])
        ),
        "phase CPU route",
    )

    natural_phone = route(natural_profile, "phone-adreno")
    phone_latency = natural_phone.get("latency")
    phone_variants = (
        phone_latency.get("variants")
        if type(phone_latency) is dict
        else None
    )
    require(
        type(phone_variants) is list
        and len(phone_variants) == 1
        and phone_variants[0].get("selector_value") == 1
        and phone_variants[0].get("measured") is True,
        "natural phone latency variant",
    )
    phone_variant = phone_variants[0]
    phone_cost = phone_variant.get("cost_us")
    coefficients = (
        phone_cost.get("coefficients")
        if type(phone_cost) is dict
        else None
    )
    require(
        type(coefficients) is dict
        and set(coefficients) <= {"input_tokens", "output_tokens"}
        and type(phone_variant.get("ucb_add_us")) is int
        and phone_variant["ucb_add_us"] > 0,
        "phase-independent phone service model",
    )
    phone_leases = natural_phone.get("resource_leases")
    require(
        type(phone_leases) is list
        and phone_leases
        and all(
            lease.get("duration_us") == phone_cost
            and lease.get("duration_ucb_add_us")
                == phone_variant["ucb_add_us"]
            for lease in phone_leases
        ),
        "natural phone leases",
    )

    profile = copy.deepcopy(phase_profile)
    output_phone = route(profile, "phone-adreno")
    output_phone["latency"] = {
        "cost_us": copy.deepcopy(phone_cost),
        "measured": True,
        "sample_count": phone_variant["sample_count"],
        "ucb_add_us": phone_variant["ucb_add_us"],
    }
    output_phone["resource_leases"] = copy.deepcopy(phone_leases)
    output_phone["evidence_ids"] = sorted(set(
        output_phone.get("evidence_ids", [])
        + natural_phone.get("evidence_ids", [])
    ))
    profile["profile_id"] = (
        "s42-op15-whole-task-llama1b-runtime-auto-v1"
    )
    ProfileBundle.from_json(profile)

    audit = {
        "input_sha256": {
            "natural_audit": digest(natural_audit_path),
            "natural_profile": digest(natural_profile_path),
            "phase_audit": digest(phase_audit_path),
            "phase_profile": digest(phase_profile_path),
        },
        "phone_holdout_count": phone_audits[0]["holdout_count"],
        "phone_train_count": phone_audits[0]["train_count"],
        "phone_upper_violations": 0,
        "profile_sha256": hashlib.sha256(canonical(profile)).hexdigest(),
        "required_request_features": sorted({
            name
            for variant in cpu_latency["variants"]
            for name in variant["cost_us"]["coefficients"]
        } | {"contention_class_id"}),
        "schema": OUTPUT_AUDIT_SCHEMA,
        "status": "PASS",
        "supported_contention_classes": list(range(1, 9)),
    }
    return profile, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-profile", type=Path, required=True)
    parser.add_argument("--phase-audit", type=Path, required=True)
    parser.add_argument("--natural-profile", type=Path, required=True)
    parser.add_argument("--natural-audit", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    args = parser.parse_args()
    require(
        args.output_profile.is_absolute()
        and args.output_audit.is_absolute()
        and not args.output_profile.exists()
        and not args.output_audit.exists(),
        "new absolute output paths",
    )
    profile, audit = build(
        args.phase_profile,
        args.phase_audit,
        args.natural_profile,
        args.natural_audit,
    )
    args.output_profile.write_bytes(canonical(profile))
    args.output_audit.write_bytes(canonical(audit))
    print(json.dumps({
        "audit": str(args.output_audit),
        "profile": str(args.output_profile),
        "profile_sha256": audit["profile_sha256"],
        "status": audit["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
