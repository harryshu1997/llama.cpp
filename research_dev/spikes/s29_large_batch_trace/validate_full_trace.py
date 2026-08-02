#!/usr/bin/env python3
"""Independently validate the S29 matched physical full-trace campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from large_batch_profiles import canonical_bytes, load_bundle, sha256_file


SCHEMA = "s29-large-batch-validation-v1"
REPORT_SCHEMA = "s29-large-batch-full-trace-v1"
WORKERS = {
    "OP12": ("OP12.log", "HTP0", 0, 6),
    "OP15": ("OP15.log", "HTP0", 6, 8),
    "cuda-prefix": ("cuda-prefix.log", "CUDA0", 0, 6),
    "cuda-mid": ("cuda-mid.log", "CUDA0", 6, 8),
    "cuda-tail": ("cuda-tail.log", "CUDA0", 8, 48),
}


class ValidationError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot load {path}: {exc}") from exc
    if type(value) is not dict:
        raise ValidationError(f"{path} must contain an object")
    return value


def verify_manifest(directory: Path) -> str:
    path = directory / "SHA256SUMS.txt"
    if not path.is_file():
        raise ValidationError(f"missing manifest: {path}")
    seen = set()
    for line in path.read_text(encoding="ascii").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or len(digest) != 64 or name in seen:
            raise ValidationError(f"invalid manifest line in {path}")
        seen.add(name)
        target = directory / name.removeprefix("./")
        if not target.is_file() or sha256_file(target) != digest:
            raise ValidationError(f"manifest mismatch: {target}")
    return "sha256:" + sha256_file(path)


def session_certs(path: Path) -> list[dict[str, Any]]:
    certs = []
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        marker = "SESSIONCERT "
        if marker not in line:
            continue
        try:
            cert = json.loads(line.split(marker, 1)[1])
        except json.JSONDecodeError as exc:
            raise ValidationError(f"invalid session certificate in {path}") from exc
        if type(cert) is not dict:
            raise ValidationError(f"invalid session certificate in {path}")
        certs.append(cert)
    return certs


def validate_sessions(phone_dir: Path, desktop_dir: Path) -> dict[str, Any]:
    result = {}
    for name, (filename, backend, start, end) in WORKERS.items():
        directory = phone_dir if name in ("OP12", "OP15") else desktop_dir
        path = directory / filename
        certs = session_certs(path)
        if len(certs) != 3 or [row.get("session_id") for row in certs] != [1, 2, 3]:
            raise ValidationError(f"{name} did not emit three contiguous sessions")
        identities = {
            (row.get("worker_pid"), row.get("worker_boot_nonce"), row.get("device_boot_id"))
            for row in certs
        }
        if len(identities) != 1:
            raise ValidationError(f"{name} was not resident across the campaign")
        if [row.get("session_end") for row in certs] != ["DETACH", "DETACH", "STOP"]:
            raise ValidationError(f"{name} session endings changed")
        for row in certs:
            if (
                row.get("schema") != "ls-stagenet-session-v2"
                or row.get("expected_backend") != backend
                or row.get("layer_start") != start
                or row.get("layer_end") != end
                or row.get("missing_buffer_compute_nodes") != 0
            ):
                raise ValidationError(f"{name} session identity or placement changed")
            status = row.get("placement_status")
            if status not in ("SCHEDULED_PLACEMENT_OK", "PLACEMENT_UNOBSERVED"):
                raise ValidationError(f"{name} placement failed")
        if not any(row.get("placement_status") == "SCHEDULED_PLACEMENT_OK" for row in certs):
            raise ValidationError(f"{name} never produced a placement certificate")
        result[name] = {
            "sessions": len(certs),
            "session_ids": [1, 2, 3],
            "placement": [row["placement_status"] for row in certs],
            "resident_identity": list(identities)[0],
        }
    return result


def nearest_rank(values: list[int], numerator: int, denominator: int) -> int:
    ordered = sorted(values)
    rank = (numerator * len(ordered) + denominator - 1) // denominator
    return ordered[max(0, rank - 1)]


def validate_report(
    report: dict[str, Any],
    mode: str,
    profile_path: Path,
    measured_batches: dict[str, set[int]],
) -> None:
    if report.get("schema") != REPORT_SCHEMA or report.get("status") != "RUN_COMPLETE":
        raise ValidationError(f"{mode} report did not complete")
    if report.get("configuration", {}).get("control_mode") != mode:
        raise ValidationError(f"{mode} mode changed")
    if report.get("profile", {}).get("sha256") != "sha256:" + sha256_file(profile_path):
        raise ValidationError(f"{mode} profile binding changed")
    summary = report.get("summary", {})
    runtime = report.get("runtime", {})
    if (
        summary.get("completed_requests") != 60
        or summary.get("rejected_requests") != 0
        or summary.get("slo_misses") != 0
        or runtime.get("completed_count") != 60
        or runtime.get("rejected_count") != 0
        or len(runtime.get("requests", [])) != 60
    ):
        raise ValidationError(f"{mode} request conservation or SLO gate failed")
    request_ids = [row.get("request_id") for row in runtime["requests"]]
    if len(set(request_ids)) != 60:
        raise ValidationError(f"{mode} request IDs are not unique")
    if any(row.get("slo_met") is not True for row in runtime["requests"]):
        raise ValidationError(f"{mode} contains an SLO miss")
    decisions = runtime.get("decisions")
    if type(decisions) is not list or not decisions:
        raise ValidationError(f"{mode} decisions are missing")
    for decision in decisions:
        route = decision.get("route_id")
        batch = decision.get("batch_size")
        priorities = decision.get("priorities")
        if route not in measured_batches or batch not in measured_batches[route]:
            raise ValidationError(f"{mode} selected an unmeasured batch")
        if 0 in priorities and any(value != 0 for value in priorities):
            raise ValidationError(f"{mode} mixed urgent and background priorities")
        if route == "R2" and batch < 24:
            raise ValidationError("treatment sent a sub-B24 batch to the phones")
    for worker, events in report.get("batch_events", {}).items():
        for event in events:
            route_ids = set(event.get("routes", []))
            if event.get("status") != "OK":
                raise ValidationError(f"{mode} contains a failed physical batch")
            if worker in ("op12-prefix", "op15-mid") and event.get("batch_size", 0) < 24:
                raise ValidationError("phone physical batch is below B24")
            if "R2" in route_ids and event.get("batch_size") not in measured_batches["R2"]:
                raise ValidationError("phone route emitted an unmeasured physical batch")
    state = report.get("final_software_state", {})
    if (
        state.get("runner_pins") != {}
        or any(
            state.get("software_leases", {}).get(name) != {}
            for name in (
                "cuda-prefix", "cuda-mid", "op12-prefix", "op15-mid", "cuda-tail",
            )
        )
        or state.get("priority_pending") != []
        or state.get("priority_active") != {}
    ):
        raise ValidationError(f"{mode} retained software state")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--phone-session", type=Path, required=True)
    parser.add_argument("--desktop-session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        routes, _capacities, _reserve, bundle = load_bundle(args.profiles)
        measured = {
            route.route_id: {point.batch_size for point in route.points}
            for route in routes
        }
        control = load_json(args.control)
        treatment = load_json(args.treatment)
        validate_report(control, "all-cuda", args.profiles, measured)
        validate_report(treatment, "priority", args.profiles, measured)
        if control["summary"]["route_distribution"] != {"R0": 60}:
            raise ValidationError("control route distribution changed")
        if treatment["summary"]["route_distribution"].get("R2", 0) < 24:
            raise ValidationError("treatment did not execute a useful phone cohort")
        phone_batches = [
            int(event["batch_size"])
            for name in ("op12-prefix", "op15-mid")
            for event in treatment["batch_events"].get(name, [])
        ]
        if not phone_batches or max(phone_batches) < 32:
            raise ValidationError("treatment did not reach the B32 phone target")
        sessions = validate_sessions(args.phone_session, args.desktop_session)
        control_p0 = [
            row["latency_us"] for row in control["runtime"]["requests"]
            if row["priority"] == 0
        ]
        treatment_p0 = [
            row["latency_us"] for row in treatment["runtime"]["requests"]
            if row["priority"] == 0
        ]
        control_p95 = nearest_rank(control_p0, 95, 100)
        treatment_p95 = nearest_rank(treatment_p0, 95, 100)
        cuda_control = control["summary"]["summed_cuda_island_compute_us"]
        cuda_treatment = treatment["summary"]["summed_cuda_island_compute_us"]
        token_matches = sum(
            left["output_tokens"] == right["output_tokens"]
            for left, right in zip(
                sorted(control["runtime"]["requests"], key=lambda row: row["request_id"]),
                sorted(treatment["runtime"]["requests"], key=lambda row: row["request_id"]),
            )
        )
        relief_pass = cuda_treatment < cuda_control
        priority_pass = treatment_p95 <= (control_p95 * 11) // 10
        result = {
            "schema": SCHEMA,
            "status": (
                "S29_LARGE_BATCH_FULL_TRACE_PASS"
                if relief_pass and priority_pass
                else "S29_LARGE_BATCH_MECHANICS_PASS_BENEFIT_GATE_FAIL"
            ),
            "artifacts": {
                "control": "sha256:" + sha256_file(args.control),
                "treatment": "sha256:" + sha256_file(args.treatment),
                "profiles": "sha256:" + sha256_file(args.profiles),
                "phone_manifest": verify_manifest(args.phone_session),
                "desktop_manifest": verify_manifest(args.desktop_session),
            },
            "profile_schema": bundle["schema"],
            "completed": 60,
            "slo_misses": 0,
            "phone_batches": {
                "minimum": min(phone_batches),
                "maximum": max(phone_batches),
                "values": phone_batches,
            },
            "route_distribution": treatment["summary"]["route_distribution"],
            "cuda_compute_us": {
                "control": cuda_control,
                "treatment": cuda_treatment,
                "relief_percent": 100.0 * (cuda_control - cuda_treatment) / cuda_control,
                "pass": relief_pass,
            },
            "p0_p95_us": {
                "control": control_p95,
                "treatment": treatment_p95,
                "ratio": treatment_p95 / control_p95,
                "pass": priority_pass,
            },
            "makespan_us": {
                "control": control["summary"]["makespan_us"],
                "treatment": treatment["summary"]["makespan_us"],
            },
            "same_request_token_matches": token_matches,
            "numeric_scope": "UNCERTIFIED_UNLESS_60_OF_60_MATCH",
            "sessions": sessions,
            "energy": {
                "phone": "UNKNOWN",
                "network": "UNKNOWN",
                "total_system": "UNKNOWN",
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_bytes(result))
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0 if result["status"] == "S29_LARGE_BATCH_FULL_TRACE_PASS" else 3
    except BaseException as exc:
        print(json.dumps({
            "schema": SCHEMA,
            "status": "S29_VALIDATION_FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    import sys

    raise SystemExit(main())
