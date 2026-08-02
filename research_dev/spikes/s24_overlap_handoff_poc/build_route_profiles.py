#!/usr/bin/env python3
"""Build conservative C3 route profiles from validated physical B1 repeats."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence


RUN_SCHEMA = "s24-fixed-diamond-physical-v1"
VALIDATION_SCHEMA = "s24-physical-session-validation-v1"
CALIBRATION_SCHEMA = "s24-fixed-route-calibration-v1"
PROFILE_SCHEMA = "s24-fixed-route-profiles-v1"
ROUTE_IDS = ("R0", "R1", "R2")
ROUTE_RESOURCES = {
    "R0": ("cuda-prefix", "cuda-mid", "cuda-tail"),
    "R1": ("cuda-prefix", "op15-mid", "cuda-tail"),
    "R2": ("op12-prefix", "op15-mid", "cuda-tail"),
}


class ProfileError(RuntimeError):
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
        raise ProfileError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProfileError(f"artifact is not an object: {path}")
    return value


def activation_signature(request: dict[str, Any]) -> list[tuple[int, int, str]]:
    records = request.get("boundary_activations")
    if not isinstance(records, list) or not records:
        raise ProfileError("profile run did not capture boundary activations")
    signature = []
    for record in records:
        digest = record.get("sha256")
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise ProfileError("profile activation has no digest")
        signature.append((
            int(record["layer_end"]),
            int(record["position"]),
            digest,
        ))
    return sorted(signature)


def load_validated_run(
    run_path: Path,
    validation_path: Path,
    route_id: str,
) -> dict[str, Any]:
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
    ):
        raise ProfileError(f"run is not placement-validated: {run_path}")
    requests = run.get("runtime", {}).get("requests")
    if not isinstance(requests, list) or len(requests) != 1:
        raise ProfileError(f"{route_id} calibration must contain exactly one request")
    request = requests[0]
    if request.get("route_id") != route_id:
        raise ProfileError(f"{route_id} calibration selected another route")
    route_distribution = run.get("summary", {}).get("route_distribution")
    if route_distribution != {route_id: 1}:
        raise ProfileError(f"{route_id} calibration route distribution differs")
    for worker in ROUTE_RESOURCES[route_id]:
        summary = run["summary"]["workers"].get(worker)
        if summary is None and worker == "op15-mid":
            summary = run["summary"]["op15_pooled"]
        if not isinstance(summary, dict) or summary.get("max_batch") != 1:
            raise ProfileError(f"{route_id} calibration is not physical B1 on {worker}")
    if not request.get("slo_met"):
        raise ProfileError(f"{route_id} B1 calibration missed its SLO")
    return {
        "run_path": str(run_path),
        "run_sha256": run_sha,
        "validation_path": str(validation_path),
        "validation_sha256": "sha256:" + sha256_file(validation_path),
        "latency_us": int(request["latency_us"]),
        "ttft_us": int(request["ttft_us"]),
        "prompt_length": int(request["prompt_length"]),
        "output_steps": int(request["output_steps"]),
        "output_tokens": list(request["output_tokens"]),
        "activation_signature": activation_signature(request),
        "worker_capacities": {
            name: int(hello["max_streams"])
            for name, hello in run["workers"].items()
        },
    }


def build_calibration(
    inputs: dict[str, Sequence[tuple[Path, Path]]],
) -> dict[str, Any]:
    routes = {}
    common_capacities = None
    common_work = None
    for route_id in ROUTE_IDS:
        pairs = inputs.get(route_id, ())
        if len(pairs) < 2:
            raise ProfileError(f"{route_id} requires at least two validated B1 repeats")
        runs = [
            load_validated_run(run, validation, route_id)
            for run, validation in pairs
        ]
        capacities = runs[0]["worker_capacities"]
        if any(run["worker_capacities"] != capacities for run in runs[1:]):
            raise ProfileError(f"{route_id} worker capacities changed across repeats")
        if common_capacities is None:
            common_capacities = capacities
        elif common_capacities != capacities:
            raise ProfileError("worker capacities changed across routes")
        work = (runs[0]["prompt_length"], runs[0]["output_steps"])
        if any(
            (run["prompt_length"], run["output_steps"]) != work
            for run in runs[1:]
        ):
            raise ProfileError(f"{route_id} B1 repeats do not have equal work")
        if common_work is None:
            common_work = work
        elif common_work != work:
            raise ProfileError("route calibration work differs")
        if any(run["output_tokens"] != runs[0]["output_tokens"] for run in runs[1:]):
            raise ProfileError(f"{route_id} output tokens are not repeat-stable")
        if any(
            run["activation_signature"] != runs[0]["activation_signature"]
            for run in runs[1:]
        ):
            raise ProfileError(f"{route_id} boundary activations are not byte-stable")
        service_rows = work[0] + work[1] - 1
        service_us = max(run["latency_us"] for run in runs)
        routes[route_id] = {
            "repeat_count": len(runs),
            "service_rows": service_rows,
            "conservative_service_us": service_us,
            "conservative_row_us": math.ceil(service_us / service_rows),
            "latency_samples_us": [run["latency_us"] for run in runs],
            "ttft_samples_us": [run["ttft_us"] for run in runs],
            "stable_output_tokens": runs[0]["output_tokens"],
            "stable_activation_signature": [
                list(item) for item in runs[0]["activation_signature"]
            ],
            "runs": runs,
        }
    if common_capacities is None or common_work is None:
        raise ProfileError("calibration is empty")
    return {
        "schema": CALIBRATION_SCHEMA,
        "scope": "CONSERVATIVE_VALIDATED_B1_FIXED_ROUTE_MECHANICS_PROFILE",
        "estimator": (
            "maximum repeated B1 end-to-end latency divided by physical rows; "
            "the same conservative row cost is used for prefill and decode"
        ),
        "profiled_batch": 1,
        "work": {
            "prompt_tokens": common_work[0],
            "output_steps": common_work[1],
            "physical_rows": common_work[0] + common_work[1] - 1,
        },
        "resource_capacities": common_capacities,
        "routes": routes,
        "quality_certified": False,
        "numeric_scope": "F16_PHONE_Q8_SERVER_MECHANICS_ONLY",
    }


def build_profiles(
    calibration_path: Path,
    calibration: dict[str, Any],
    gather_cap_us: int,
    output_path: Path,
) -> dict[str, Any]:
    if gather_cap_us < 0:
        raise ProfileError("gather cap cannot be negative")
    digest = "sha256:" + sha256_file(calibration_path)
    capacities = calibration["resource_capacities"]
    profiles = []
    for rank, route_id in enumerate(ROUTE_IDS):
        route = calibration["routes"][route_id]
        resources = ROUTE_RESOURCES[route_id]
        profiles.append({
            "route_id": route_id,
            "offload_rank": rank,
            "fixed_us": 0,
            "prefill_token_us": route["conservative_row_us"],
            "decode_step_us": route["conservative_row_us"],
            "profiled_batch": 1,
            "max_active": min(capacities[name] for name in resources),
            "gather_cap_us": gather_cap_us,
            "resources": list(resources),
            "evidence_sha256": digest,
        })
    relative = os.path.relpath(calibration_path, output_path.parent)
    return {
        "schema": PROFILE_SCHEMA,
        "profiles": profiles,
        "resource_capacities": capacities,
        "source_reports": [{
            "path": relative,
            "sha256": digest,
        }],
    }


def pair_paths(
    runs: Sequence[Path], validations: Sequence[Path], route_id: str,
) -> list[tuple[Path, Path]]:
    if len(runs) != len(validations):
        raise ProfileError(f"{route_id} run and validation counts differ")
    return list(zip(runs, validations))


def main() -> int:
    parser = argparse.ArgumentParser()
    for route_id in ROUTE_IDS:
        lower = route_id.lower()
        parser.add_argument(f"--{lower}-run", type=Path, action="append", default=[])
        parser.add_argument(
            f"--{lower}-validation", type=Path, action="append", default=[],
        )
    parser.add_argument("--gather-cap-us", type=int, default=5000)
    parser.add_argument("--calibration-output", type=Path, required=True)
    parser.add_argument("--profiles-output", type=Path, required=True)
    args = parser.parse_args()
    if args.calibration_output.exists() or args.profiles_output.exists():
        parser.error("calibration or profile output already exists")
    try:
        inputs = {
            route_id: pair_paths(
                getattr(args, f"{route_id.lower()}_run"),
                getattr(args, f"{route_id.lower()}_validation"),
                route_id,
            )
            for route_id in ROUTE_IDS
        }
        calibration = build_calibration(inputs)
        args.calibration_output.parent.mkdir(parents=True, exist_ok=True)
        args.calibration_output.write_bytes(canonical_bytes(calibration))
        profiles = build_profiles(
            args.calibration_output,
            calibration,
            args.gather_cap_us,
            args.profiles_output,
        )
        args.profiles_output.parent.mkdir(parents=True, exist_ok=True)
        args.profiles_output.write_bytes(canonical_bytes(profiles))
    except (OSError, ProfileError, ValueError) as exc:
        print(json.dumps({
            "status": "FAIL", "error": str(exc),
        }, sort_keys=True, separators=(",", ":")))
        return 2
    print(json.dumps({
        "status": "PROFILES_READY",
        "calibration": str(args.calibration_output),
        "calibration_sha256": "sha256:" + sha256_file(args.calibration_output),
        "profiles": str(args.profiles_output),
        "profiles_sha256": "sha256:" + sha256_file(args.profiles_output),
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
