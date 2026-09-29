#!/usr/bin/env python3
"""Add a held-out-qualified CPU/phone FFN route to a scheduler profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import fit_phase_contention_profile as contention


RESULT_SCHEMA = "s42-full-fp16-llama1b-combined-result-v2"
PHONE_SCHEMA = "s41-phone-energy-v3"
PROFILE_SCHEMA = "s42-general-scheduler-profile-v1"
FFN_MANIFEST_SCHEMA = "s42-llama-dense-ffn-manifest-v1"
FFN_POLICY_SCHEMA = "s42-llama-ffn-vq-compiled-policy-v1"
QUALIFICATION_SCHEMA = "s42-fp16-llama1b-physical-qualification-v1"
AUDIT_SCHEMA = "s42-llama1b-ffn-split-route-calibration-v1"
DIRECT_ENERGY_SCHEMA = "s42-llama1b-ffn-direct-energy-v1"
SPLIT_ROUTE = "cpu-phone-ffn-split"
BOUNDARY = (
    "incremental-cpu-package-gpu-board-whole-phone-above-resident-idle-v1"
)
DIRECT_ENERGY_METHOD = (
    "isolated-sequential-incremental-above-resident-idle-v1"
)
T95_DF1 = 12.706204736


class MaterializationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MaterializationError(message)


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


def policy_execution_sha256(policy: dict[str, Any]) -> str:
    return hashlib.sha256(canonical({
        "compiled_buckets": policy["compiled_buckets"],
        "policy_text": policy["policy_text"],
    })).hexdigest()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(type(value) is dict, f"object expected: {path}")
    return value


def verified_record(path: Path, schema: str) -> dict[str, Any]:
    value = load(path)
    supplied = value.get("record_sha256")
    unsigned = {
        key: row for key, row in value.items() if key != "record_sha256"
    }
    require(
        value.get("schema") == schema
        and supplied == hashlib.sha256(canonical(unsigned)).hexdigest(),
        f"record identity: {path}",
    )
    return value


def parse_split_log(path: Path) -> dict[str, Any]:
    summary = None
    shapes: list[dict[str, Any]] = []
    summary_prefix = "S41SERVERFFN "
    shape_prefix = "S41SERVERFFNSHAPE "
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        for prefix, kind in (
            (summary_prefix, "summary"),
            (shape_prefix, "shape"),
        ):
            position = line.find(prefix)
            if position < 0:
                continue
            try:
                value = json.loads(line[position + len(prefix):])
            except json.JSONDecodeError:
                break
            if type(value) is not dict:
                break
            if kind == "summary":
                summary = value
                shapes = []
            elif summary is not None:
                shapes.append(value)
            break
    require(
        type(summary) is dict
        and summary.get("status") == "ok"
        and type(summary.get("calls")) is int
        and summary["calls"] > 0,
        f"failed physical FFN summary: {path}",
    )
    require(
        shapes
        and all(
            type(row.get("tokens")) is int
            and row["tokens"] > 0
            and type(row.get("columns")) is int
            and row["columns"] > 0
            and type(row.get("calls")) is int
            and row["calls"] > 0
            and all(
                type(row.get(name)) in {int, float}
                and not isinstance(row[name], bool)
                and row[name] >= 0
                for name in ("overlap_mean_ms", "wait_mean_ms")
            )
            for row in shapes
        )
        and sum(row["calls"] for row in shapes) == summary["calls"],
        f"physical FFN shape summaries: {path}",
    )
    return {
        "shapes": sorted(
            shapes, key=lambda row: (row["tokens"], row["columns"])
        ),
        "summary": summary,
    }


def phone_columns_for_tokens(policy: dict[str, Any], tokens: int) -> int:
    for bucket in policy["compiled_buckets"]:
        if tokens <= bucket["max_tokens"]:
            return int(bucket["phone_columns"])
    raise MaterializationError(f"FFN tokens outside compiled policy: {tokens}")


def validate_split_contract(
    path: Path,
    manifest: dict[str, Any],
    policy: dict[str, Any],
) -> None:
    content = path.read_text(encoding="utf-8", errors="replace")
    layer_count = len(manifest["geometry"]["resident_layer_ids"])
    layer_mask = manifest["split_contract"]["layer_mask"]
    require(
        f"layers={layer_count}" in content
        and f"mask={layer_mask:016x}" in content
        and f"policy={policy['policy_text']}" in content,
        f"split log contract: {path}",
    )
    split_log = parse_split_log(path)
    max_columns = manifest["split_contract"]["max_columns"]
    quantum = manifest["split_contract"]["column_quantum"]
    require(
        all(
            row["columns"] <= max_columns
            and row["columns"] % quantum == 0
            and row["columns"] == phone_columns_for_tokens(
                policy, row["tokens"]
            )
            for row in split_log["shapes"]
        ),
        f"split shape/policy identity: {path}",
    )


def exposed_join_wait_upper_ppm(
    split_logs: list[dict[str, Any]],
    max_columns: int,
) -> int:
    partial_shapes = [
        row
        for split_log in split_logs
        for row in split_log["shapes"]
        if row["columns"] < max_columns
    ]
    if not partial_shapes:
        return 0
    ratios = [
        float(row["wait_mean_ms"])
        / max(0.001, float(row["overlap_mean_ms"]))
        for row in partial_shapes
    ]
    return min(1_000_000, math.ceil(max(ratios) * 1_000_000))


def validate_run(
    result_path: Path,
    phone_path: Path,
    qualification_path: Path,
    split_log_path: Path,
    expected_route: str,
    expected_large_policy: str = "cpu-overflow",
) -> dict[str, Any]:
    result = load(result_path)
    phone = load(phone_path)
    qualification = load(qualification_path)
    rows = result.get("request_results")
    require(
        result.get("schema") == RESULT_SCHEMA
        and result.get("status") == "PASS"
        and result.get("policy", {}).get("large_model_policy")
            == expected_large_policy
        and result.get("policy", {}).get("small_model_policy")
            == "static-cpu"
        and result.get("policy", {}).get("static_route") == expected_route
        and type(rows) is list
        and rows
        and result.get("scheduler_runtime", {}).get("route_counts")
            == {expected_route: len(rows)},
        f"static route identity: {result_path}",
    )
    require(
        phone.get("schema") == PHONE_SCHEMA
        and phone.get("status") == "PASS"
        and phone.get("boundary") == "paid_trace_interval"
        and abs(phone["duration_s"] - result["metrics"]["duration_s"])
            < 1e-6,
        f"phone energy identity: {phone_path}",
    )
    gates = qualification.get("gates", {})
    require(
        qualification.get("schema") == QUALIFICATION_SCHEMA
        and qualification.get("status") == "PASS"
        and qualification.get("input_sha256", {}).get("result")
            == digest(result_path)
        and qualification.get("policy") == result.get("policy")
        and all(
            value is True
            for name, value in gates.items()
            if name != "nonbaseline_physical_execution"
        ),
        f"physical qualification: {qualification_path}",
    )
    split_log = None
    if expected_route == SPLIT_ROUTE:
        require(
            gates.get("selected_split_route_has_phone_ffn_calls") is True
            and qualification.get("nonbaseline_physical_executions")
                == len(rows),
            "split physical execution gate",
        )
        split_log = parse_split_log(split_log_path)
    return {
        "name": result_path.parent.parent.name,
        "phone": phone,
        "qualification": qualification,
        "result": result,
        "split_log": split_log,
        "split_summary": None if split_log is None else split_log["summary"],
    }


def server_energy_j(run: dict[str, Any]) -> float:
    return float(
        run["result"]["server_energy"]["server_compute_device_energy_j"]
    )


def fleet_energy_j(run: dict[str, Any]) -> float:
    return server_energy_j(run) + float(
        run["phone"]["whole_phone_energy_j"]
    )


def paired_ci95(values: list[float]) -> dict[str, float | int]:
    require(len(values) == 2, "two paired values")
    mean = sum(values) / 2
    sample_sd = abs(values[0] - values[1]) / math.sqrt(2)
    half_width = T95_DF1 * sample_sd / math.sqrt(2)
    return {
        "ci95_high": mean + half_width,
        "ci95_low": mean - half_width,
        "mean": mean,
        "samples": 2,
    }


def exact_work_identity(run: dict[str, Any]) -> dict[str, Any]:
    result = run["result"]
    inputs = {
        name: value
        for name, value in result["input_sha256"].items()
        if name not in {
            "ffn_compiled_policy",
            "ffn_manifest",
            "resident_release",
            "runtime_profile",
        }
    }
    receipts = {}
    for name in ("large_model_outputs", "small_model_outputs"):
        source = result["work_receipts"][name]
        receipts[name] = {
            "actual_output_tokens": source.get(
                "actual_output_tokens", source["output_tokens"]
            ),
            "output_tokens": source["output_tokens"],
            "request_count": source["request_count"],
        }
    return {
        "inputs": inputs,
        "metrics": {
            name: result["metrics"][name]
            for name in ("completed", "input_tokens", "output_tokens")
        },
        "model_identities": result["model_identities"],
        "work_receipts": receipts,
    }


def direct_route_energy(
    path: Path,
    manifest: dict[str, Any],
    policy: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    value = verified_record(path, DIRECT_ENERGY_SCHEMA)
    binding = value.get("binding")
    qualification = value.get("qualification")
    ratio = (
        None
        if type(qualification) is not dict
        else qualification.get("split_over_cpu_ratio_ci95")
    )
    energy = value.get("route_energy")
    require(
        type(binding) is dict
        and binding.get("manifest_record_sha256")
            == manifest["record_sha256"]
        and binding.get("model_sha256") == manifest["model"]["sha256"]
        and binding.get("policy_execution_sha256")
            == policy_execution_sha256(policy)
        and binding.get("route_id") == SPLIT_ROUTE,
        "direct route energy binding",
    )
    require(
        type(qualification) is dict
        and qualification.get("status") == "PASS"
        and qualification.get("method") == DIRECT_ENERGY_METHOD
        and qualification.get("idle_baseline_subtracted") is True
        and qualification.get("request_windows_non_overlapping") is True
        and qualification.get("max_concurrent_measured_requests") == 1
        and type(qualification.get("idle_baseline_sample_count")) is int
        and qualification["idle_baseline_sample_count"] >= 3
        and type(qualification.get("repeat_count")) is int
        and qualification["repeat_count"] >= 2
        and qualification.get("heldout_upper_bound_violations") == 0,
        "isolated direct route energy qualification",
    )
    require(
        type(ratio) is dict
        and type(ratio.get("samples")) is int
        and ratio["samples"] >= 2
        and type(ratio.get("ci95_high")) in {int, float}
        and ratio["ci95_high"] < 0.95,
        "direct route energy margin",
    )
    require(
        type(energy) is dict
        and energy.get("status") == "measured"
        and energy.get("boundary_id") == BOUNDARY
        and type(energy.get("cost_uj")) is dict
        and type(energy.get("lower_error_ppm")) is int
        and 0 <= energy["lower_error_ppm"] <= 1_000_000
        and type(energy.get("upper_error_ppm")) is int
        and 0 <= energy["upper_error_ppm"] <= 1_000_000,
        "direct route energy model",
    )
    return json.loads(json.dumps(energy)), {
        "input_sha256": digest(path),
        "method": qualification["method"],
        "qualification": qualification,
        "record_sha256": value["record_sha256"],
    }


def fit_route_variants(
    calibration_runs: list[dict[str, Any]],
    natural_runs: list[dict[str, Any]],
    op15_idle_runs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [
        row
        for run in calibration_runs
        for row in contention.calibration_observations(
            Path(run["result_path"]), "cpu-overflow", SPLIT_ROUTE
        )
    ] + [
        row
        for run in natural_runs
        for row in contention.natural_observations(
            Path(run["result_path"]), "cpu-overflow", SPLIT_ROUTE
        )
    ] + [
        row
        for run in op15_idle_runs
        for row in contention.calibration_observations(
            Path(run["result_path"]), "op15-assistance", SPLIT_ROUTE
        )
    ]
    variants = []
    audits = []
    for class_id in (*range(1, 5), 8):
        selected = [row for row in rows if row["class_id"] == class_id]
        profile, audit = contention.variant(class_id, selected)
        variants.append(profile)
        audits.append(audit)
    return variants, audits


def materialize(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = verified_record(args.ffn_manifest, FFN_MANIFEST_SCHEMA)
    policy = verified_record(args.ffn_policy, FFN_POLICY_SCHEMA)
    require(
        policy.get("qualification", {}).get("route_admission")
            == "SHADOW_ONLY_UNTIL_HELDOUT_PHYSICAL_PROFILE"
        and policy.get("qualification", {}).get(
            "physical_shape_calibrated"
        ) is True
        and type(policy.get("evidence", {}).get(
            "physical_calibration_record_sha256"
        )) is str
        and policy.get("evidence", {}).get("manifest_record_sha256")
            == manifest["record_sha256"],
        "physically calibrated shadow split policy identity",
    )
    cpu_runs = []
    split_runs = []
    for index in range(2):
        cpu_runs.append(validate_run(
            args.cpu_natural_result[index],
            args.cpu_natural_phone[index],
            args.cpu_natural_qualification[index],
            args.split_natural_log[index],
            "desktop-cpu",
        ))
        split = validate_run(
            args.split_natural_result[index],
            args.split_natural_phone[index],
            args.split_natural_qualification[index],
            args.split_natural_log[index],
            SPLIT_ROUTE,
        )
        split["result_path"] = str(args.split_natural_result[index])
        split_runs.append(split)
    calibration_runs = []
    for index in range(2):
        split = validate_run(
            args.split_calibration_result[index],
            args.split_calibration_phone[index],
            args.split_calibration_qualification[index],
            args.split_calibration_log[index],
            SPLIT_ROUTE,
        )
        split["result_path"] = str(args.split_calibration_result[index])
        calibration_runs.append(split)
    op15_idle_runs = []
    for index in range(2):
        split = validate_run(
            args.op15_idle_result[index],
            args.op15_idle_phone[index],
            args.op15_idle_qualification[index],
            args.op15_idle_log[index],
            SPLIT_ROUTE,
            "op15-assistance",
        )
        split["result_path"] = str(args.op15_idle_result[index])
        op15_idle_runs.append(split)
    for path in (
        args.split_natural_log
        + args.split_calibration_log
        + args.op15_idle_log
    ):
        validate_split_contract(path, manifest, policy)
    identities = [exact_work_identity(run) for run in cpu_runs + split_runs]
    require(all(row == identities[0] for row in identities[1:]), "natural work")
    variants, variant_audits = fit_route_variants(
        calibration_runs, split_runs, op15_idle_runs
    )
    latency_gate = all(row["measured"] for row in variant_audits)
    energy, energy_audit = direct_route_energy(
        args.direct_energy_profile,
        manifest,
        policy,
    )
    savings = [
        100 * (fleet_energy_j(cpu) - fleet_energy_j(split))
            / fleet_energy_j(cpu)
        for cpu, split in zip(cpu_runs, split_runs)
    ]
    fleet_ci = paired_ci95(savings)
    fleet_gate = fleet_ci["ci95_low"] > 0
    split_logs = [
        run["split_log"]
        for run in calibration_runs + split_runs + op15_idle_runs
    ]
    summaries = [split_log["summary"] for split_log in split_logs]
    overlap_upper_ppm = exposed_join_wait_upper_ppm(
        split_logs, manifest["split_contract"]["max_columns"]
    )
    overlap_gate = overlap_upper_ppm <= 50_000
    qualified = (
        latency_gate
        and fleet_gate
        and overlap_gate
    )

    profile = load(args.base_profile)
    require(profile.get("schema") == PROFILE_SCHEMA, "base profile")
    baseline = next(
        (row for row in profile["routes"] if row.get("baseline") is True),
        None,
    )
    require(
        type(baseline) is dict
        and baseline.get("energy", {}).get("status") == "measured"
        and baseline["energy"].get("boundary_id") == BOUNDARY,
        "incremental baseline route energy",
    )
    require(
        not any(row.get("route_id") == SPLIT_ROUTE for row in profile["routes"]),
        "split route already exists",
    )
    if qualified:
        if not any(
            row.get("resource_id") == "op15-htp"
            for row in profile["resources"]
        ):
            profile["resources"].append({
                "capacity": 1,
                "identity": "op15-htp3-3c15au002cl00000",
                "kind": "npu",
                "ready": True,
                "resource_id": "op15-htp",
            })
        profile["routes"].append({
            "baseline": False,
            "energy": energy,
            "evidence_ids": sorted({
                "sha256:" + digest(path)
                for path in (
                    args.cpu_natural_result
                    + args.split_natural_result
                    + args.split_calibration_result
                    + args.op15_idle_result
                    + [
                        args.direct_energy_profile,
                        args.ffn_manifest,
                        args.ffn_policy,
                    ]
                )
            }),
            "granularity": "operator",
            "latency": {
                "kind": "conditioned_affine_features_v1",
                "selector_feature": "contention_class_id",
                "variants": variants,
            },
            "overlap": {
                "exposed_join_wait_ppm": overlap_upper_ppm,
                "sample_count": sum(summary["calls"] for summary in summaries),
                "status": "measured",
                "upper_error_ppm": overlap_upper_ppm,
            },
            "placement_verified": True,
            "quality_class": "bounded_numeric",
            "resident": True,
            "resource_leases": [],
            "resource_slots": {
                "desktop-cpu": 1,
                "desktop-usb-root": 1,
                "op15-htp": 1,
                "op15-ncm": 1,
            },
            "route_id": SPLIT_ROUTE,
            "server_busy_ppm": 1_000_000,
            "server_memory_bytes": manifest["model"]["size_bytes"],
            "workload_id": "llama-1b-resident-task",
        })
        profile["ffn_split_binding"] = {
            "compiled_buckets_sha256": hashlib.sha256(canonical(
                policy["compiled_buckets"]
            )).hexdigest(),
            "layer_mask": manifest["split_contract"]["layer_mask"],
            "manifest_record_sha256": manifest["record_sha256"],
            "policy_text": policy["policy_text"],
            "route_admission": "QUALIFIED_FOR_RUNTIME_SELECTION",
            "route_id": SPLIT_ROUTE,
        }
        profile["profile_id"] = (
            profile["profile_id"] + "-heldout-ffn-split-v2"
        )

    audit = {
        "gates": {
            "direct_incremental_route_energy_qualified": True,
            "fleet_energy_ci_positive": fleet_gate,
            "split_overlap_upper_ppm_at_most_50000": overlap_gate,
            "zero_latency_upper_violations": latency_gate,
        },
        "input_sha256": {
            "base_profile": digest(args.base_profile),
            "cpu_natural_results": [
                digest(path) for path in args.cpu_natural_result
            ],
            "direct_energy_profile": digest(args.direct_energy_profile),
            "ffn_manifest": digest(args.ffn_manifest),
            "ffn_policy": digest(args.ffn_policy),
            "op15_idle_results": [
                digest(path) for path in args.op15_idle_result
            ],
            "split_calibration_results": [
                digest(path) for path in args.split_calibration_result
            ],
            "split_natural_results": [
                digest(path) for path in args.split_natural_result
            ],
        },
        "latency_variants": variant_audits,
        "metrics": {
            "direct_route_energy": energy_audit,
            "fleet_energy_saving_pct": fleet_ci,
            "split_overlap_upper_ppm": overlap_upper_ppm,
        },
        "route_admission": (
            "QUALIFIED_FOR_RUNTIME_SELECTION"
            if qualified else "SHADOW_ONLY_UNTIL_GATES_PASS"
        ),
        "schema": AUDIT_SCHEMA,
        "status": "PASS" if qualified else "FAIL",
    }
    return profile, audit


def repeated(parser: argparse.ArgumentParser, name: str) -> None:
    parser.add_argument(name, type=Path, action="append", required=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-profile", type=Path, required=True)
    parser.add_argument("--direct-energy-profile", type=Path, required=True)
    parser.add_argument("--ffn-manifest", type=Path, required=True)
    parser.add_argument("--ffn-policy", type=Path, required=True)
    for prefix in ("cpu-natural", "split-natural"):
        repeated(parser, f"--{prefix}-result")
        repeated(parser, f"--{prefix}-phone")
        repeated(parser, f"--{prefix}-qualification")
    repeated(parser, "--split-natural-log")
    repeated(parser, "--split-calibration-result")
    repeated(parser, "--split-calibration-phone")
    repeated(parser, "--split-calibration-qualification")
    repeated(parser, "--split-calibration-log")
    repeated(parser, "--op15-idle-result")
    repeated(parser, "--op15-idle-phone")
    repeated(parser, "--op15-idle-qualification")
    repeated(parser, "--op15-idle-log")
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    args = parser.parse_args()
    repeated_values = [
        value
        for name, value in vars(args).items()
        if name.startswith((
            "cpu_natural_",
            "op15_idle_",
            "split_natural_",
            "split_calibration_",
        ))
    ]
    require(all(len(value) == 2 for value in repeated_values), "two repeats")
    require(
        args.output_profile.is_absolute()
        and args.output_audit.is_absolute()
        and not args.output_profile.exists()
        and not args.output_audit.exists(),
        "new absolute output paths",
    )
    profile, audit = materialize(args)
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    args.output_profile.write_bytes(canonical(profile))
    args.output_audit.write_bytes(canonical(audit))
    print(json.dumps({
        "audit": str(args.output_audit),
        "profile": str(args.output_profile),
        "route_admission": audit["route_admission"],
        "status": audit["status"],
    }, sort_keys=True))
    return 0 if audit["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
