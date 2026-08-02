#!/usr/bin/env python3
"""Independently validate the matched S26 physical control and treatment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
S24 = HERE.parent / "s24_overlap_handoff_poc"
for dependency in (HERE, S24):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

import validate_physical_session as s24_validation
from physical_adapter import canonical_bytes, load_trace, sha256_file
from priority_profiles import OUTPUT_TOKENS_BY_POINT, load_bundle
from priority_runtime import prepare_trace, validate_conservation


SCHEMA = "s26-priority-comparison-v1"
WORKERS = {
    "cuda-prefix": ("cuda-prefix.log", 1, 2),
    "cuda-mid": ("cuda-mid.log", 1, 2),
    "op12-prefix": ("OP12.log", 1, 2),
    "op15-mid": ("OP15.log", 1, 2),
    "cuda-tail": ("cuda-tail.log", 1, 2),
}


class ResultValidationError(RuntimeError):
    pass


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ResultValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="ascii"), object_pairs_hook=_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResultValidationError(f"cannot load {path}: {exc}") from exc
    if type(value) is not dict:
        raise ResultValidationError(f"{path} is not a JSON object")
    return value


def _logs(cuda_dir: Path, phone_dir: Path) -> dict[str, Path]:
    return {
        "cuda-prefix": cuda_dir / "cuda-prefix.log",
        "cuda-mid": cuda_dir / "cuda-mid.log",
        "op12-prefix": phone_dir / "OP12.log",
        "op15-mid": phone_dir / "OP15.log",
        "cuda-tail": cuda_dir / "cuda-tail.log",
    }


def _flatten_treatment(run: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(run)
    normalized["schema"] = "s24-fixed-diamond-physical-v1"
    normalized["control"] = "C2"
    flat = []
    for decision in run["runtime"]["decisions"]:
        for request_id, route_epoch, priority in zip(
            decision["request_ids"],
            decision["route_epochs"],
            decision["priorities"],
        ):
            flat.append({
                "request_id": request_id,
                "route_epoch": route_epoch,
                "route_id": decision["route_id"],
                "priority": priority,
            })
    normalized["runtime"]["decisions"] = flat
    batch_events = normalized["batch_events"]
    tail_events = []
    for name in ("cuda-tail-r0", "cuda-tail-r2"):
        tail_events.extend(batch_events.pop(name, []))
    tail_events.sort(key=lambda event: int(event["dispatch_ns"]))
    batch_events["cuda-tail"] = tail_events
    return normalized


def _validate_priority_session(
    run: dict[str, Any], logs: dict[str, Path], session_id: int,
) -> dict[str, Any]:
    normalized = _flatten_treatment(run)
    expected_rows, lineage = s24_validation.validate_event_lineage(normalized)
    session_end = run.get("configuration", {}).get("session_end")
    if session_end not in ("detach", "stop"):
        raise ResultValidationError("session end contract is invalid")
    workers = {}
    for worker, path in logs.items():
        matches = [
            certificate
            for certificate in s24_validation.parse_session_certificates(path)
            if certificate.get("session_id") == session_id
        ]
        if len(matches) != 1:
            raise ResultValidationError(
                f"{worker} session {session_id} certificate count is {len(matches)}"
            )
        workers[worker] = s24_validation.validate_certificate(
            worker, matches[0], expected_rows[worker], session_end,
        )
    return {
        "expected_rows": expected_rows,
        "lineage": lineage,
        "workers": workers,
        "session_end": session_end,
        "placement_gate": "PASS",
    }


def _p95(values: list[int]) -> int:
    if not values:
        raise ResultValidationError("cannot compute p95 of an empty set")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _cuda_work(run: dict[str, Any]) -> int:
    total = 0
    tail_workers = (
        ("cuda-tail",)
        if "cuda-tail" in run["batch_events"]
        else ("cuda-tail-r0", "cuda-tail-r2")
    )
    for worker in ("cuda-prefix", "cuda-mid", *tail_workers):
        events = run["batch_events"].get(worker)
        if type(events) is not list:
            raise ResultValidationError(f"missing {worker} events")
        for event in events:
            if event.get("status") != "OK" or type(event.get("compute_us")) is not int:
                raise ResultValidationError(f"invalid {worker} compute event")
            total += event["compute_us"]
    return total


def _validate_source_manifest(run_dir: Path) -> dict[str, str]:
    manifest = run_dir / "executed-sources.sha256"
    result = {}
    try:
        lines = manifest.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ResultValidationError(f"cannot read executed source manifest: {exc}") from exc
    for line in lines:
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64 or relative in result:
            raise ResultValidationError("executed source manifest is malformed")
        path = run_dir / relative
        if sha256_file(path) != digest:
            raise ResultValidationError(f"executed source changed: {relative}")
        result[relative] = "sha256:" + digest
    required = {
        "executed-sources/physical_adapter.py",
        "executed-sources/priority_policy.py",
        "executed-sources/priority_profiles.py",
        "executed-sources/priority_runtime.py",
        "executed-sources/profiles.json",
        "executed-sources/route_runtime.py",
    }
    if set(result) != required:
        raise ResultValidationError("executed source manifest set changed")
    return result


def validate_comparison(
    control_path: Path,
    treatment_path: Path,
    trace_path: Path,
    profiles_path: Path,
    cuda_dir: Path,
    phone_dir: Path,
) -> dict[str, Any]:
    control = load_json(control_path)
    treatment = load_json(treatment_path)
    if (
        control.get("schema") != "s26-priority-physical-v1"
        or control.get("status") != "RUN_COMPLETE"
        or control.get("configuration", {}).get("control_mode") != "all-cuda"
        or control.get("numeric_scope") != "Q8_CUDA_ONLY"
    ):
        raise ResultValidationError("control is not a matched all-CUDA run")
    if (
        treatment.get("schema") != "s26-priority-physical-v1"
        or treatment.get("status") != "RUN_COMPLETE"
        or treatment.get("configuration", {}).get("control_mode") != "priority"
    ):
        raise ResultValidationError("treatment is not a complete S26 run")
    trace = load_trace(trace_path)
    if (
        control.get("trace", {}).get("trace_hash") != trace["trace_hash"]
        or treatment.get("trace", {}).get("trace_hash") != trace["trace_hash"]
        or control["trace"]["sha256"] != "sha256:" + sha256_file(trace_path)
        or treatment["trace"]["sha256"] != "sha256:" + sha256_file(trace_path)
    ):
        raise ResultValidationError("control and treatment trace evidence differ")
    _routes, _capacities, _reserve, profile = load_bundle(profiles_path)
    if (
        control.get("profile", {}).get("sha256")
        != "sha256:" + sha256_file(profiles_path)
        or control["profile"].get("schema") != profile["schema"]
        or
        treatment.get("profile", {}).get("sha256")
        != "sha256:" + sha256_file(profiles_path)
        or treatment["profile"].get("schema") != profile["schema"]
    ):
        raise ResultValidationError("treatment profile binding changed")

    rows = prepare_trace(trace)
    validate_conservation(
        rows,
        treatment["runtime"]["decisions"],
        treatment["runtime"]["requests"],
        treatment["runtime"]["rejected"],
    )
    logs = _logs(cuda_dir, phone_dir)
    control_session = _validate_priority_session(control, logs, 1)
    treatment_session = _validate_priority_session(treatment, logs, 2)
    for worker in WORKERS:
        before = control_session["workers"][worker]
        after = treatment_session["workers"][worker]
        if (
            before["worker_pid"] != after["worker_pid"]
            or before["worker_boot_nonce"] != after["worker_boot_nonce"]
            or before["device_boot_id"] != after["device_boot_id"]
            or after["steps_total"] < before["steps_total"]
        ):
            raise ResultValidationError(f"{worker} persistence identity changed")

    control_requests = control["runtime"]["requests"]
    treatment_requests = treatment["runtime"]["requests"]
    if len(control_requests) != 12 or len(treatment_requests) != 12:
        raise ResultValidationError("comparison does not contain twelve requests")
    control_by_id = {row["request_id"]: row for row in control_requests}
    treatment_by_id = {row["request_id"]: row for row in treatment_requests}
    if set(control_by_id) != set(treatment_by_id):
        raise ResultValidationError("comparison request sets differ")
    control_point = {}
    for decision in control["runtime"]["decisions"]:
        point = (decision["route_id"], decision["batch_size"])
        for request_id in decision["request_ids"]:
            if request_id in control_point:
                raise ResultValidationError("control request has two route points")
            control_point[request_id] = point
    treatment_point = {}
    for decision in treatment["runtime"]["decisions"]:
        point = (decision["route_id"], decision["batch_size"])
        for request_id in decision["request_ids"]:
            if request_id in treatment_point:
                raise ResultValidationError("request has two route points")
            treatment_point[request_id] = point
    for request_id in control_by_id:
        left = control_by_id[request_id]
        right = treatment_by_id[request_id]
        if (
            left["prompt_length"] != right["prompt_length"]
            or left["output_steps"] != right["output_steps"]
            or left["priority"] != right["priority"]
            or left["slo_us"] != right["slo_us"]
        ):
            raise ResultValidationError(f"request {request_id} equal-work result changed")
        expected_control = OUTPUT_TOKENS_BY_POINT.get(control_point.get(request_id))
        if expected_control is None or left["output_tokens"] != expected_control:
            raise ResultValidationError(
                f"control request {request_id} differs from its route-point oracle"
            )
        point = treatment_point.get(request_id)
        expected_tokens = OUTPUT_TOKENS_BY_POINT.get(point)
        if expected_tokens is None or right["output_tokens"] != expected_tokens:
            raise ResultValidationError(
                f"request {request_id} differs from its frozen route-point oracle"
            )

    route_distribution = Counter(row["route_id"] for row in treatment_requests)
    if route_distribution != Counter({"R0": 4, "R2": 8}):
        raise ResultValidationError("treatment route distribution changed")
    expected_control_batches = {
        "cuda-prefix": Counter({4: 12}),
        "cuda-mid": Counter({4: 12}),
        "op12-prefix": Counter(),
        "op15-mid": Counter(),
        "cuda-tail-r0": Counter({4: 12}),
        "cuda-tail-r2": Counter(),
    }
    for worker, expected in expected_control_batches.items():
        observed = Counter(
            event["batch_size"] for event in control["batch_events"][worker]
        )
        if observed != expected:
            raise ResultValidationError(
                f"matched control did not preserve R0 B4 on {worker}"
            )
    expected_batches = {
        "cuda-prefix": Counter({4: 4}),
        "cuda-mid": Counter({4: 4}),
        "op12-prefix": Counter({4: 8}),
        "op15-mid": Counter({4: 8}),
        "cuda-tail-r0": Counter({4: 4}),
        "cuda-tail-r2": Counter({4: 8}),
    }
    for worker, expected in expected_batches.items():
        observed = Counter(
            event["batch_size"] for event in treatment["batch_events"][worker]
        )
        if observed != expected:
            raise ResultValidationError(f"{worker} did not preserve measured B4 execution")

    control_p0 = [row for row in control_requests if row["priority"] == 0]
    treatment_p0 = [row for row in treatment_requests if row["priority"] == 0]
    control_p95 = _p95([row["latency_us"] for row in control_p0])
    treatment_p95 = _p95([row["latency_us"] for row in treatment_p0])
    control_p0_misses = sum(not row["slo_met"] for row in control_p0)
    treatment_p0_misses = sum(not row["slo_met"] for row in treatment_p0)
    treatment_total_misses = sum(not row["slo_met"] for row in treatment_requests)
    control_cuda = _cuda_work(control)
    treatment_cuda = _cuda_work(treatment)
    gates = {
        "all_requests_conserved": True,
        "treatment_matches_route_oracle": True,
        "all_treatment_slos_met": treatment_total_misses == 0,
        "no_extra_priority0_miss": treatment_p0_misses <= control_p0_misses,
        "priority0_p95_within_1_05x": treatment_p95 * 100 <= control_p95 * 105,
        "cuda_compute_reduced": treatment_cuda < control_cuda,
        "physical_placement": True,
        "persistent_workers": True,
        "measured_profile_execution": True,
    }
    if not all(gates.values()):
        failed = sorted(name for name, passed in gates.items() if not passed)
        raise ResultValidationError("comparison gate failed: " + ",".join(failed))

    sources = _validate_source_manifest(treatment_path.parent)
    return {
        "schema": SCHEMA,
        "status": "S26_PRIORITY_PHYSICAL_PASS",
        "scope": "MECHANICS_AND_SELECTED_CUDA_COMPUTE_NOT_ENERGY",
        "artifacts": {
            "control": {"path": str(control_path), "sha256": "sha256:" + sha256_file(control_path)},
            "treatment": {"path": str(treatment_path), "sha256": "sha256:" + sha256_file(treatment_path)},
            "trace": {"path": str(trace_path), "sha256": "sha256:" + sha256_file(trace_path)},
            "profiles": {"path": str(profiles_path), "sha256": "sha256:" + sha256_file(profiles_path)},
            "executed_sources": sources,
        },
        "gates": gates,
        "control": {
            "priority0_p95_us": control_p95,
            "priority0_slo_misses": control_p0_misses,
            "cuda_compute_us": control_cuda,
            "makespan_us": control["runtime"]["duration_ns"] // 1000,
            "distinct_output_sequences": len({
                tuple(row["output_tokens"]) for row in control_requests
            }),
        },
        "treatment": {
            "priority0_p95_us": treatment_p95,
            "priority0_slo_misses": treatment_p0_misses,
            "total_slo_misses": treatment_total_misses,
            "cuda_compute_us": treatment_cuda,
            "makespan_us": treatment["runtime"]["duration_ns"] // 1000,
            "route_distribution": dict(sorted(route_distribution.items())),
        },
        "deltas": {
            "priority0_p95_percent": (treatment_p95 - control_p95) * 100.0 / control_p95,
            "cuda_compute_percent": (treatment_cuda - control_cuda) * 100.0 / control_cuda,
        },
        "sessions": {
            "control": control_session,
            "treatment": treatment_session,
        },
        "energy": {
            "gpu_board": "NOT_MEASURED",
            "phone": "UNKNOWN",
            "network": "UNKNOWN",
            "total_system": "UNKNOWN",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--cuda-session-dir", type=Path, required=True)
    parser.add_argument("--phone-session-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        report = validate_comparison(
            args.control,
            args.treatment,
            args.trace,
            args.profiles,
            args.cuda_session_dir,
            args.phone_session_dir,
        )
    except (OSError, ValueError, ResultValidationError, s24_validation.ValidationError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True, separators=(",", ":")))
        return 2
    args.output.write_bytes(canonical_bytes(report))
    print(json.dumps({
        "status": report["status"],
        "priority0_p95_percent": report["deltas"]["priority0_p95_percent"],
        "cuda_compute_percent": report["deltas"]["cuda_compute_percent"],
        "output": str(args.output),
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
