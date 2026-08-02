#!/usr/bin/env python3
"""Independently validate the S31 selected-cut matched campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cut_selector import canonical_bytes, load_json, nearest_rank, sha256_file
from selected_profiles import load_bundle


SCHEMA = "s31-selected-campaign-validation-v1"
REPORT_SCHEMA = "s31-selected-full-trace-v1"


class ValidationError(RuntimeError):
    pass


def verify_manifest(directory: Path) -> str:
    manifest = directory / "SHA256SUMS.txt"
    if not manifest.is_file():
        raise ValidationError(f"missing manifest: {manifest}")
    expected_files = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path != manifest
    }
    seen = set()
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, separator, name = line.partition("  ")
        name = name.removeprefix("./")
        if not separator or len(digest) != 64 or name in seen:
            raise ValidationError(f"invalid manifest line: {manifest}")
        target = directory / name
        if not target.is_file() or sha256_file(target) != digest:
            raise ValidationError(f"manifest mismatch: {target}")
        seen.add(name)
    if seen != expected_files:
        raise ValidationError(f"manifest coverage mismatch: {manifest}")
    return "sha256:" + sha256_file(manifest)


def session_certs(path: Path) -> list[dict[str, Any]]:
    result = []
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValidationError(f"cannot read worker log: {path}") from exc
    for line in lines:
        if "SESSIONCERT " not in line:
            continue
        try:
            value = json.loads(line.split("SESSIONCERT ", 1)[1])
        except json.JSONDecodeError as exc:
            raise ValidationError(f"invalid session certificate: {path}") from exc
        if type(value) is not dict:
            raise ValidationError(f"invalid session certificate: {path}")
        result.append(value)
    return result


def validate_sessions(phone_dir: Path, desktop_dir: Path, cut: int) -> dict[str, Any]:
    workers = {
        "OP12": (phone_dir / "OP12.log", "HTP0", 0, cut),
        "OP15": (phone_dir / "OP15.log", "HTP0", cut, 8),
        "cuda-prefix": (desktop_dir / "cuda-prefix.log", "CUDA0", 0, 6),
        "cuda-mid": (desktop_dir / "cuda-mid.log", "CUDA0", 6, 8),
        "cuda-tail": (desktop_dir / "cuda-tail.log", "CUDA0", 8, 48),
    }
    result = {}
    for name, (path, backend, start, end) in workers.items():
        certs = session_certs(path)
        if len(certs) != 3 or [row.get("session_id") for row in certs] != [1, 2, 3]:
            raise ValidationError(f"{name} session sequence changed")
        if [row.get("session_end") for row in certs] != ["DETACH", "DETACH", "STOP"]:
            raise ValidationError(f"{name} session endings changed")
        identities = {
            (row.get("worker_pid"), row.get("worker_boot_nonce"), row.get("device_boot_id"))
            for row in certs
        }
        if len(identities) != 1:
            raise ValidationError(f"{name} was not resident across the campaign")
        for row in certs:
            if (
                row.get("schema") != "ls-stagenet-session-v2"
                or row.get("expected_backend") != backend
                or row.get("layer_start") != start
                or row.get("layer_end") != end
                or row.get("missing_buffer_compute_nodes") != 0
                or row.get("placement_status") not in {
                    "SCHEDULED_PLACEMENT_OK", "PLACEMENT_UNOBSERVED",
                }
            ):
                raise ValidationError(f"{name} session identity or placement changed")
        if not any(row.get("placement_status") == "SCHEDULED_PLACEMENT_OK" for row in certs):
            raise ValidationError(f"{name} never computed on its expected backend")
        result[name] = {
            "sessions": 3,
            "placement": [row["placement_status"] for row in certs],
            "resident_identity": list(identities)[0],
        }
    return result


def validate_report(
    report: dict[str, Any], mode: str, profile_path: Path,
    measured: dict[str, set[int]], cut: int,
) -> None:
    if report.get("schema") != REPORT_SCHEMA or report.get("status") != "RUN_COMPLETE":
        raise ValidationError(f"{mode} report did not complete")
    config = report.get("configuration", {})
    if (
        config.get("control_mode") != mode
        or config.get("selected_cut") != cut
        or config.get("gather_us") != 5000
    ):
        raise ValidationError(f"{mode} configuration changed")
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
        raise ValidationError(f"{mode} conservation or SLO gate failed")
    request_ids = [row.get("request_id") for row in runtime["requests"]]
    if len(set(request_ids)) != 60 or any(row.get("slo_met") is not True for row in runtime["requests"]):
        raise ValidationError(f"{mode} request outcomes are invalid")
    for decision in runtime.get("decisions", []):
        route = decision.get("route_id")
        batch = decision.get("batch_size")
        priorities = decision.get("priorities")
        if route not in measured or batch not in measured[route]:
            raise ValidationError(f"{mode} selected an unmeasured batch")
        if 0 in priorities and any(value != 0 for value in priorities):
            raise ValidationError(f"{mode} mixed urgent and background priorities")
        if route == "R2" and batch < 24:
            raise ValidationError("phone dispatch fell below B24")
    for name, events in report.get("batch_events", {}).items():
        for event in events:
            if event.get("status") != "OK":
                raise ValidationError(f"{mode} contains a failed batch")
            if name in ("op12-prefix", "op15-mid") and event.get("batch_size", 0) < 24:
                raise ValidationError("phone physical batch fell below B24")
    state = report.get("final_software_state", {})
    if (
        state.get("runner_pins") != {}
        or any(state.get("software_leases", {}).get(name) != {} for name in (
            "cuda-prefix", "cuda-mid", "op12-prefix", "op15-mid", "cuda-tail",
        ))
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
    parser.add_argument("--s29-baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        routes, _capacities, _reserve, profile, cut = load_bundle(args.profiles)
        measured = {
            route.route_id: {point.batch_size for point in route.points}
            for route in routes
        }
        control = load_json(args.control)
        treatment = load_json(args.treatment)
        validate_report(control, "all-cuda", args.profiles, measured, cut)
        validate_report(treatment, "priority", args.profiles, measured, cut)
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
            raise ValidationError("treatment did not reach B32")
        sessions = validate_sessions(args.phone_session, args.desktop_session, cut)
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
        baseline = load_json(args.s29_baseline)
        baseline_makespan = baseline.get("makespan_us", {}).get("treatment")
        if (
            baseline.get("schema") != "s29-large-batch-validation-v1"
            or baseline.get("status") != "S29_LARGE_BATCH_FULL_TRACE_PASS"
            or type(baseline_makespan) is not int
        ):
            raise ValidationError("S29 baseline is invalid")
        token_matches = sum(
            left["output_tokens"] == right["output_tokens"]
            for left, right in zip(
                sorted(control["runtime"]["requests"], key=lambda row: row["request_id"]),
                sorted(treatment["runtime"]["requests"], key=lambda row: row["request_id"]),
            )
        )
        relief_pass = cuda_treatment < cuda_control
        priority_pass = treatment_p95 <= (control_p95 * 11) // 10
        makespan = treatment["summary"]["makespan_us"]
        balance_pass = makespan < baseline_makespan
        result = {
            "schema": SCHEMA,
            "status": (
                "S31_BALANCED_FULL_TRACE_PASS"
                if relief_pass and priority_pass and balance_pass
                else "S31_BALANCED_MECHANICS_PASS_BENEFIT_GATE_FAIL"
            ),
            "selected_cut": cut,
            "route_layers": profile["route_layers"],
            "artifacts": {
                "control": "sha256:" + sha256_file(args.control),
                "treatment": "sha256:" + sha256_file(args.treatment),
                "profiles": "sha256:" + sha256_file(args.profiles),
                "s29_baseline": "sha256:" + sha256_file(args.s29_baseline),
                "phone_manifest": verify_manifest(args.phone_session),
                "desktop_manifest": verify_manifest(args.desktop_session),
            },
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
                "s29_cut6": baseline_makespan,
                "s31_cut1": makespan,
                "reduction_percent": 100.0 * (baseline_makespan - makespan) / baseline_makespan,
                "pass": balance_pass,
            },
            "same_request_token_matches": token_matches,
            "numeric_scope": "UNCERTIFIED_UNLESS_60_OF_60_MATCH",
            "sessions": sessions,
            "energy": {"phone": "UNKNOWN", "network": "UNKNOWN", "total_system": "UNKNOWN"},
        }
        args.output.write_bytes(canonical_bytes(result))
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0 if result["status"] == "S31_BALANCED_FULL_TRACE_PASS" else 3
    except BaseException as exc:
        print(json.dumps({
            "schema": SCHEMA,
            "status": "S31_VALIDATION_FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, sort_keys=True), file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
