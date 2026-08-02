#!/usr/bin/env python3
"""Validate the S24 dense-mechanics and observed-length physical runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from workloads import (
    MECHANICS_SCOPE,
    OBSERVED_SCOPE,
    WorkloadError,
    validate as validate_trace,
)


SCHEMA = "s24-cp6-workload-gates-v1"
RUN_SCHEMA = "s24-fixed-diamond-physical-v1"
VALIDATION_SCHEMA = "s24-physical-session-validation-v1"
PINNED = {
    "dense-mechanics": {
        "file_sha256": "324133ce4e2fc95c1a9dd34292797f95796e029099b2d8af9f48277340f49ab6",
        "trace_hash": "sha256:9079e3f939068d17fcd356ed05173910f23dead75e879e439d4ff85ea65f832c",
        "scope": MECHANICS_SCOPE,
        "count": 60,
        "control": "C3",
    },
    "observed-context-600": {
        "file_sha256": "5ae1691fbe50f497344546be22e2a6a4cf1f17df277ac6b37bbbf6098db58222",
        "trace_hash": "sha256:2891c8b17bece17b08a8cf03676ed4afaf3bdfc23eb0cc72735c80cda7b8cf83",
        "scope": OBSERVED_SCOPE,
        "count": 28,
        "control": "C2",
    },
}


class WorkloadGateError(RuntimeError):
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
        raise WorkloadGateError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkloadGateError(f"artifact is not an object: {path}")
    return value


def request_source(trace: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["request_id"]): row for row in trace["requests"]}


def validate_completed_work(
    name: str,
    trace: dict[str, Any],
    run: dict[str, Any],
) -> dict[str, Any]:
    source = request_source(trace)
    completed = {
        int(row["request_id"]): row for row in run["runtime"]["requests"]
    }
    rejected = run["runtime"]["rejected"]
    if set(completed) != set(source) or rejected:
        raise WorkloadGateError(f"{name} did not complete its exact denominator")
    route_counts = Counter()
    slo_misses = 0
    for request_id, expected in source.items():
        observed = completed[request_id]
        tokens = observed.get("output_tokens")
        if (
            observed.get("prompt_length") != expected["input_tokens"]
            or observed.get("output_steps") != expected["output_steps"]
            or observed.get("priority") != expected["priority"]
            or observed.get("observed_input_tokens")
            != expected["observed_input_tokens"]
            or observed.get("observed_output_tokens")
            != expected["observed_output_tokens"]
            or observed.get("scheduled_arrival_ns")
            != expected["arrival_us"] * 1000
            or not isinstance(tokens, list)
            or len(tokens) != expected["output_steps"]
            or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in tokens
            )
        ):
            raise WorkloadGateError(f"{name} request work changed: {request_id}")
        if name == "observed-context-600" and (
            observed.get("route_id") != expected["route_hint"]
        ):
            raise WorkloadGateError("observed cohort changed a pinned route hint")
        route_counts[str(observed["route_id"])] += 1
        slo_misses += not bool(observed["slo_met"])
    return {
        "completed": len(completed),
        "rejected": 0,
        "route_distribution": dict(sorted(route_counts.items())),
        "slo_misses": slo_misses,
        "makespan_us": run["summary"]["makespan_us"],
        "op15_batches": run["summary"]["op15_pooled"],
        "cuda_tail_batches": run["summary"]["cuda_tail"],
        "cuda_island_compute_us": run["summary"]["cuda_island_compute_us"],
    }


def load_validated_run(
    campaign: Path,
    name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = PINNED[name]
    run_path = campaign / "runs" / name / "runtime.json"
    validation_path = campaign / "validation" / "sessions" / f"{name}.json"
    run = load_json(run_path)
    validation = load_json(validation_path)
    run_sha = "sha256:" + sha256_file(run_path)
    if (
        run.get("schema") != RUN_SCHEMA
        or run.get("status") != "RUN_COMPLETE"
        or run.get("control") != expected["control"]
        or run.get("numeric_scope")
        != "MECHANICS_ONLY_F16_PHONE_Q8_SERVER_NUMERICALLY_UNCERTIFIED"
        or validation.get("schema") != VALIDATION_SCHEMA
        or validation.get("status") != "PHYSICAL_SESSION_PASS"
        or validation.get("runtime", {}).get("sha256") != run_sha
        or validation.get("placement_gate") != "PASS"
        or validation.get("cpu_compute_fallback")
        != "NONE_EXCEPT_DECLARED_METADATA_GET_ROWS"
    ):
        raise WorkloadGateError(f"{name} is not placement-validated")
    trace_path = Path(run["trace"]["path"])
    trace = load_json(trace_path)
    try:
        validate_trace(trace)
    except (ValueError, WorkloadError) as exc:
        raise WorkloadGateError(f"{name} trace validation failed: {exc}") from exc
    if (
        sha256_file(trace_path) != expected["file_sha256"]
        or trace.get("trace_hash") != expected["trace_hash"]
        or trace.get("scope") != expected["scope"]
        or len(trace.get("requests", [])) != expected["count"]
        or run["trace"]["sha256"] != "sha256:" + expected["file_sha256"]
        or run["trace"]["trace_hash"] != expected["trace_hash"]
        or run["configuration"]["arrival_scale"] != 1.0
        or any(run["runtime"]["route_delays_us"].values())
        or (
            name == "dense-mechanics"
            and run["configuration"]["prefill_chunk"] is not None
        )
        or (
            name == "observed-context-600"
            and run["configuration"]["prefill_chunk"] != 64
        )
    ):
        raise WorkloadGateError(f"{name} did not use the pinned unshifted trace")
    endpoints = run["configuration"]["endpoints"]
    if (
        endpoints.get("op12-prefix") != ["192.168.1.193", 24280]
        or endpoints.get("op15-mid") != ["192.168.1.97", 24281]
    ):
        raise WorkloadGateError(f"{name} phone traffic was not direct desktop WiFi")
    work = validate_completed_work(name, trace, run)
    return trace, {
        "runtime": str(run_path),
        "runtime_sha256": run_sha,
        "validation": str(validation_path),
        "validation_sha256": "sha256:" + sha256_file(validation_path),
        "trace": str(trace_path),
        "trace_file_sha256": "sha256:" + expected["file_sha256"],
        "trace_hash": expected["trace_hash"],
        "work": work,
    }


def arrival_histogram(trace: dict[str, Any]) -> dict[str, int]:
    return {
        str(arrival): count
        for arrival, count in sorted(Counter(
            int(row["arrival_us"]) for row in trace["requests"]
        ).items())
    }


def verify_campaign(campaign: Path) -> dict[str, Any]:
    dense_trace, dense = load_validated_run(campaign, "dense-mechanics")
    observed_trace, observed = load_validated_run(
        campaign, "observed-context-600",
    )
    dense_arrivals = arrival_histogram(dense_trace)
    if dense_arrivals != {"0": 21, "1000000": 17, "2000000": 22}:
        raise WorkloadGateError("dense 21/17/22 arrival distribution changed")
    if any(
        row["input_tokens"] != 1 or row["output_steps"] != 4
        for row in dense_trace["requests"]
    ):
        raise WorkloadGateError("dense mechanics proxy changed")
    if any(
        row["input_tokens"] != row["observed_input_tokens"]
        or row["output_steps"] != row["observed_output_tokens"]
        or row["input_tokens"] + row["output_steps"] > 600
        for row in observed_trace["requests"]
    ):
        raise WorkloadGateError("observed cohort length contract changed")
    return {
        "schema": SCHEMA,
        "status": "WORKLOAD_GATES_PASS",
        "dense_mechanics": {
            **dense,
            "arrival_histogram_us": dense_arrivals,
            "representation": "ONE_TOKEN_FOUR_STEP_MECHANICS_ONLY",
        },
        "observed_context_600": {
            **observed,
            "arrival_histogram_us": arrival_histogram(observed_trace),
            "representation": "OBSERVED_LENGTHS_SYNTHETIC_TOKEN_VALUES",
        },
        "quality_certified": False,
        "numeric_scope": "F16_PHONE_Q8_SERVER_NUMERICALLY_UNCERTIFIED",
        "runtime_activation_path": "DESKTOP_DIRECT_WIFI",
        "a6000_runtime_relay": "NONE",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        report = verify_campaign(args.campaign)
    except (OSError, ValueError, WorkloadGateError) as exc:
        report = {
            "schema": SCHEMA,
            "status": "WORKLOAD_GATE_FAIL",
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
        "output": str(args.output),
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
