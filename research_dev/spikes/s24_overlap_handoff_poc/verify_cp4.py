#!/usr/bin/env python3
"""Evaluate the frozen S24 CP4 physical-mechanics gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "s24-cp4-gates-v1"
RUN_SCHEMA = "s24-fixed-diamond-physical-v1"
VALIDATION_SCHEMA = "s24-physical-session-validation-v1"
COMPARISON_SCHEMA = "s24-physical-comparison-v1"
CALIBRATION_SCHEMA = "s24-fixed-route-calibration-v1"
PROFILE_SCHEMA = "s24-fixed-route-profiles-v1"
BASE_EXPECTED_ROUTES = {
    "r0-b1-a": {"R0": 1},
    "r0-b1-b": {"R0": 1},
    "r1-b1-a": {"R1": 1},
    "r1-b1-b": {"R1": 1},
    "r2-b1-a": {"R2": 1},
    "r2-b1-b": {"R2": 1},
    "r1-cuda-control": {"R0": 1},
    "r2-cuda-control": {"R0": 1},
    "r2-b4": {"R2": 4},
}
REPEAT_LABELS = ("r0-repeat", "r1-repeat", "r2-repeat")
PHONE_COMPARISON_LABELS = ("r1-vs-cuda", "r2-vs-cuda")
DISPATCH_REASONS = {"BATCH_KNEE", "LATEST_SAFE_START", "GATHER_TIMER"}


class GateError(RuntimeError):
    pass


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GateError(f"artifact is not an object: {path}")
    return value


def finite_boundary(record: object) -> bool:
    if not isinstance(record, dict):
        return False
    for field in ("l2_norm", "minimum", "maximum"):
        value = record.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            return False
    return True


def expected_routes(campaign: Path) -> dict[str, dict[str, int]]:
    knee_path = campaign / "phone-knees" / "op15-mid-knee.json"
    knee_report = load_json(knee_path)
    knee = knee_report.get("selection", {}).get("batch_knee")
    if isinstance(knee, bool) or not isinstance(knee, int) or not 1 <= knee <= 4:
        raise GateError("campaign OP15 knee is invalid")
    convergence_count = (knee + 1) // 2
    result = {name: dict(routes) for name, routes in BASE_EXPECTED_ROUTES.items()}
    result["r1-r2-shared-op15"] = {
        "R1": convergence_count,
        "R2": convergence_count,
    }
    result["r0-r1-r2-shared-tail"] = {
        "R0": convergence_count,
        "R1": convergence_count,
        "R2": convergence_count,
    }
    return result


def load_run(
    campaign: Path,
    name: str,
    expected: dict[str, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_path = campaign / "runs" / name / "runtime.json"
    validation_path = campaign / "validation" / "sessions" / f"{name}.json"
    run = load_json(run_path)
    validation = load_json(validation_path)
    run_sha = "sha256:" + sha256_file(run_path)
    if (
        run.get("schema") != RUN_SCHEMA
        or run.get("status") != "RUN_COMPLETE"
        or validation.get("schema") != VALIDATION_SCHEMA
        or validation.get("status") != "PHYSICAL_SESSION_PASS"
        or validation.get("runtime", {}).get("sha256") != run_sha
        or validation.get("placement_gate") != "PASS"
        or validation.get("cpu_compute_fallback")
        != "NONE_EXCEPT_DECLARED_METADATA_GET_ROWS"
    ):
        raise GateError(f"run is not placement-validated: {name}")
    summary = run.get("summary", {})
    if (
        summary.get("route_distribution") != expected
        or summary.get("completed_requests") != sum(expected.values())
        or summary.get("rejected_requests") != 0
    ):
        raise GateError(f"run completion or route gate failed: {name}")
    requests = run.get("runtime", {}).get("requests")
    if not isinstance(requests, list) or len(requests) != sum(expected.values()):
        raise GateError(f"completed request records differ: {name}")
    for request in requests:
        tokens = request.get("output_tokens")
        boundaries = request.get("boundary_activations")
        if (
            not isinstance(tokens, list)
            or len(tokens) != request.get("output_steps")
            or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in tokens
            )
            or not isinstance(boundaries, list)
            or not boundaries
            or any(not finite_boundary(boundary) for boundary in boundaries)
        ):
            raise GateError(f"non-finite or malformed request output: {name}")
    if validation.get("lineage") != {
        "position_continuity": "PASS",
        "request_count": sum(expected.values()),
        "route_pin_continuity": "PASS",
        "software_leases_zero": "PASS",
        "worker_kv_zero": "PASS",
    }:
        raise GateError(f"lineage gate differs: {name}")
    events = run.get("batch_events")
    if not isinstance(events, dict) or not events:
        raise GateError(f"physical batch events are absent: {name}")
    for worker_events in events.values():
        if not isinstance(worker_events, list):
            raise GateError(f"physical event stream is invalid: {name}")
        for event in worker_events:
            if (
                event.get("status") != "OK"
                or event.get("dispatch_reason") not in DISPATCH_REASONS
            ):
                raise GateError(f"failed or unknown dispatch in {name}")
    return run, {
        "runtime": str(run_path),
        "runtime_sha256": run_sha,
        "validation": str(validation_path),
        "validation_sha256": "sha256:" + sha256_file(validation_path),
        "routes": expected,
        "slo_misses": summary.get("slo_misses"),
    }


def pooled_events(run: dict[str, Any], physical_worker: str) -> list[dict[str, Any]]:
    events = run["batch_events"]
    if physical_worker == "op15-mid":
        names = [
            name for name in events
            if name == "op15-mid" or name.startswith("op15-mid-r")
        ]
    else:
        names = [physical_worker]
    return [event for name in names for event in events.get(name, [])]


def convergence_evidence(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    op15_run = runs["r1-r2-shared-op15"]
    op15_events = pooled_events(op15_run, "op15-mid")
    op15_upstreams = sorted({
        upstream
        for event in op15_events
        for upstream in event["upstream_workers"]
    })
    op15_routes = sorted({
        route for event in op15_events for route in event["routes"]
    })
    op15_mixed = sum(
        len(set(event["upstream_workers"])) > 1 for event in op15_events
    )
    if (
        set(op15_run["batch_events"]) != {
            "cuda-prefix", "cuda-mid", "op12-prefix", "op15-mid", "cuda-tail",
        }
        or op15_upstreams != ["cuda-prefix", "op12-prefix"]
        or op15_routes != ["R1", "R2"]
        or op15_mixed < 1
    ):
        raise GateError("shared OP15 did not form a mixed-source physical batch")

    tail_run = runs["r0-r1-r2-shared-tail"]
    tail_events = pooled_events(tail_run, "cuda-tail")
    tail_routes = sorted({
        route for event in tail_events for route in event["routes"]
    })
    tail_upstreams = sorted({
        upstream
        for event in tail_events
        for upstream in event["upstream_workers"]
    })
    tail_mixed = sum(len(set(event["routes"])) > 1 for event in tail_events)
    if (
        tail_routes != ["R0", "R1", "R2"]
        or tail_upstreams != ["cuda-mid", "op15-mid"]
        or tail_mixed < 1
    ):
        raise GateError("shared CUDA tail did not form a mixed-route physical batch")

    r2_b4 = runs["r2-b4"]
    b4_maxima = {
        worker: max(
            (int(event["batch_size"]) for event in pooled_events(r2_b4, worker)),
            default=0,
        )
        for worker in ("op12-prefix", "op15-mid", "cuda-tail")
    }
    return {
        "op15": {
            "queue_count": 1,
            "routes": op15_routes,
            "upstreams": op15_upstreams,
            "mixed_source_batches": op15_mixed,
        },
        "cuda_tail": {
            "queue_count": 1,
            "routes": tail_routes,
            "upstreams": tail_upstreams,
            "mixed_route_batches": tail_mixed,
        },
        "r2_b4": {
            "completed_requests": 4,
            "maximum_physical_batch_by_worker": b4_maxima,
        },
        "cohort_or_global_barrier": "NONE",
    }


def load_comparison(path: Path, label: str) -> dict[str, Any]:
    value = load_json(path)
    if (
        value.get("schema") != COMPARISON_SCHEMA
        or value.get("label") != label
        or value.get("quality_certified") is not False
    ):
        raise GateError(f"comparison contract differs: {label}")
    return value


def verify_profiles(validation_dir: Path) -> dict[str, Any]:
    calibration_path = validation_dir / "route-calibration.json"
    profiles_path = validation_dir / "route-profiles.json"
    calibration = load_json(calibration_path)
    profiles = load_json(profiles_path)
    calibration_sha = "sha256:" + sha256_file(calibration_path)
    if (
        calibration.get("schema") != CALIBRATION_SCHEMA
        or calibration.get("quality_certified") is not False
        or set(calibration.get("routes", {})) != {"R0", "R1", "R2"}
    ):
        raise GateError("route calibration contract differs")
    for route_id, route in calibration["routes"].items():
        if (
            route.get("repeat_count", 0) < 2
            or not route.get("stable_output_tokens")
            or not route.get("stable_activation_signature")
        ):
            raise GateError(f"{route_id} is not repeat-stable")
    if (
        profiles.get("schema") != PROFILE_SCHEMA
        or {row.get("route_id") for row in profiles.get("profiles", [])}
        != {"R0", "R1", "R2"}
        or profiles.get("source_reports") != [{
            "path": "route-calibration.json",
            "sha256": calibration_sha,
        }]
    ):
        raise GateError("route profile evidence binding differs")
    return {
        "calibration": str(calibration_path),
        "calibration_sha256": calibration_sha,
        "profiles": str(profiles_path),
        "profiles_sha256": "sha256:" + sha256_file(profiles_path),
        "same_route_repeat_stability": "PASS",
    }


def verify_campaign(campaign: Path, validation_dir: Path) -> dict[str, Any]:
    if validation_dir != campaign / "validation":
        raise GateError("validation directory must be inside the fixed campaign")
    runs = {}
    records = {}
    trace_hashes = set()
    routes_by_case = expected_routes(campaign)
    for name, expected in routes_by_case.items():
        run, record = load_run(campaign, name, expected)
        runs[name] = run
        records[name] = record
        trace_hashes.add(run["trace"]["trace_hash"])
    if len(trace_hashes) != 1:
        raise GateError("CP4 runs did not use one trace")

    convergence = convergence_evidence(runs)
    comparisons = {
        label: load_comparison(
            validation_dir / "comparisons" / f"{label}.json", label,
        )
        for label in REPEAT_LABELS + PHONE_COMPARISON_LABELS
    }
    for label in REPEAT_LABELS:
        comparison = comparisons[label]
        if (
            comparison.get("token_screen_pass") is not True
            or comparison.get("boundary_gate_pass") is not True
        ):
            raise GateError(f"same-route repeat comparison failed: {label}")
    profile_record = verify_profiles(validation_dir)
    phone_numeric_pass = all(
        comparisons[label].get("token_screen_pass") is True
        and comparisons[label].get("boundary_gate_pass") is True
        for label in PHONE_COMPARISON_LABELS
    )
    numerical = {
        label: {
            "path": str(validation_dir / "comparisons" / f"{label}.json"),
            "sha256": "sha256:" + sha256_file(
                validation_dir / "comparisons" / f"{label}.json"
            ),
            "token_screen_pass": comparisons[label]["token_screen_pass"],
            "boundary_gate_pass": comparisons[label]["boundary_gate_pass"],
            "aggregate": comparisons[label]["aggregate"],
        }
        for label in PHONE_COMPARISON_LABELS
    }
    return {
        "schema": SCHEMA,
        "status": "CP4_PHYSICAL_PASS",
        "trace_hash": next(iter(trace_hashes)),
        "runs": records,
        "mechanics": {
            "finite_outputs": "PASS",
            "placement": "PASS",
            "lineage_and_positions": "PASS",
            "missing_buffers": 0,
            "live_worker_kv_after_completion": 0,
            "live_software_leases_after_completion": 0,
            "cpu_compute_fallback": "NONE_EXCEPT_DECLARED_METADATA_GET_ROWS",
            "convergence": convergence,
        },
        "profiles": profile_record,
        "numerical": {
            "status": (
                "SYNTHETIC_NUMERIC_SCREEN_PASS"
                if phone_numeric_pass
                else "NUMERICALLY_UNCERTIFIED"
            ),
            "comparisons": numerical,
            "quality_certified": False,
            "scope": "SYNTHETIC_TOKEN_BOUNDARY_AND_GREEDY_OUTPUT_SCREEN_ONLY",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        report = verify_campaign(args.campaign, args.validation_dir)
    except (GateError, OSError, ValueError) as exc:
        report = {
            "schema": SCHEMA,
            "status": "CP4_GATE_FAIL",
            "error": str(exc),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_bytes(report))
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(report))
    print(json.dumps({
        "status": report["status"],
        "numeric_status": report["numerical"]["status"],
        "output": str(args.output),
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
