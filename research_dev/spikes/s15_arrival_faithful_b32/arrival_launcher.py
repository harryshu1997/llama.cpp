#!/usr/bin/env python3
"""Prepared OP15 B32 launcher that reveals the prompt only after EXECUTE."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
LIVE = HERE.parent / "s15_live_launcher"
RUNTIME = HERE.parent / "s15_runtime_dispatch"
for path in (LIVE, RUNTIME, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import physical_launcher as base  # noqa: E402
import run_recertification as recert_run  # noqa: E402
import validate_recertification as recert_validate  # noqa: E402
from route_fixtures import op15_post_load_b32_snapshot  # noqa: E402


RAW_ROOT = HERE / "physical_results/raw"
EXPECTED_SNAPSHOT = op15_post_load_b32_snapshot()
EXPECTED_PROFILE = EXPECTED_SNAPSHOT.profile_id
EXPECTED_ROUTE_EPOCH = 13
EXPECTED_TIMEOUT_US = 4_000_000


class ArrivalLauncherError(RuntimeError):
    pass


def validate_request(value: dict, boot_id: str) -> None:
    required = {
        "schema", "command", "protocol_version", "launch_id", "route_id",
        "profile_id", "device_id", "route_epoch", "residency_epoch",
        "lease_epoch", "device_boot_epoch", "registry_generation",
        "compatibility_key", "request_ids", "cohort_sha256",
        "input_manifest_sha256", "timeout_us", "expected_boundary_schema",
        "worker_binary_sha256", "worker_generation", "device_boot_id",
        "layer_range",
    }
    if set(value) != required:
        raise ArrivalLauncherError("request has missing or unknown fields")
    expected = {
        "schema": base.REQUEST_SCHEMA,
        "command": "EXECUTE",
        "protocol_version": 1,
        "route_id": base.EXPECTED_ROUTE,
        "profile_id": EXPECTED_PROFILE,
        "device_id": base.EXPECTED_DEVICE,
        "route_epoch": EXPECTED_ROUTE_EPOCH,
        "residency_epoch": 1,
        "lease_epoch": 1,
        "device_boot_epoch": 1,
        "registry_generation": 1,
        "compatibility_key": "gemma-4-12b-it-f16|decode|gemma-head-0-8",
        "cohort_sha256": base.EXPECTED_COHORT,
        "input_manifest_sha256": base.EXPECTED_INPUT,
        "timeout_us": EXPECTED_TIMEOUT_US,
        "expected_boundary_schema": base.BOUNDARY_SCHEMA,
        "worker_binary_sha256": base.EXPECTED_WORKER,
        "worker_generation": 1,
        "device_boot_id": boot_id,
        "layer_range": [0, 8],
    }
    for key, expected_value in expected.items():
        if type(value.get(key)) is not type(expected_value) or value.get(key) != expected_value:
            raise ArrivalLauncherError(f"request identity mismatch: {key}")
    if type(value.get("launch_id")) is not int or value["launch_id"] < 1:
        raise ArrivalLauncherError("request launch_id must be positive")
    request_ids = value.get("request_ids")
    if type(request_ids) is not list or len(request_ids) != 32 \
            or any(type(item) is not str or not item for item in request_ids) \
            or len(set(request_ids)) != 32:
        raise ArrivalLauncherError("request IDs are not an exact B32 set")


def session_record(request: dict, placement: dict | None, outcome: str) -> dict:
    boundaries = [] if outcome != "completed" else [
        {
            "request_id": request_id,
            "identity_ok": True,
            "epoch_ok": True,
            "correctness_ok": True,
            "d2h_complete": True,
        }
        for request_id in request["request_ids"]
    ]
    return {
        "schema": base.SESSION_SCHEMA,
        "protocol_version": request["protocol_version"],
        "launch_id": request["launch_id"],
        "route_id": request["route_id"],
        "profile_id": request["profile_id"],
        "device_id": request["device_id"],
        "route_epoch": request["route_epoch"],
        "residency_epoch": request["residency_epoch"],
        "lease_epoch": request["lease_epoch"],
        "device_boot_epoch": request["device_boot_epoch"],
        "registry_generation": request["registry_generation"],
        "compatibility_key": request["compatibility_key"],
        "request_ids": request["request_ids"],
        "cohort_sha256": request["cohort_sha256"],
        "input_manifest_sha256": request["input_manifest_sha256"],
        "worker_binary_sha256": request["worker_binary_sha256"],
        "worker_generation": request["worker_generation"],
        "device_boot_id": request["device_boot_id"],
        "layer_range": request["layer_range"],
        "session_id": 1,
        "outcome": outcome,
        "boundary_schema": request["expected_boundary_schema"],
        "placement": placement,
        "boundaries": boundaries,
    }


def cleanup(host, phone, thermal) -> None:
    recert_run.clean_route(host, phone)
    if thermal is not None:
        thermal.stop()


def main() -> int:
    raw_dir = RAW_ROOT / "launch-1"
    if raw_dir.exists():
        print("launcher raw directory already exists", file=sys.stderr)
        return 2
    raw_dir.mkdir(parents=True)
    host = None
    phone = None
    thermal = None
    request = None
    try:
        recert_report = recert_validate.validate()
        if recert_report["profile_id"] != \
                "sha256:9fa84921ec2f90d2887d9ae0be6f785addd2ca64f2d99d98f4b446cd5e281a2d":
            raise ArrivalLauncherError("post-load profile changed")
        frozen = base.verify_artifacts()
        boot_id, deployed = base.deploy(frozen)
        thermal_paths, thermal_start = base.discover_thermal_paths()
        thermal = base.ThermalMonitor(
            raw_dir / "thermal_stream.log", raw_dir / "thermal_stream.stderr.bin", thermal_paths,
        )
        thermal.wait_ready()
        host, phone, host_command, phone_command = recert_run.prepare_route(raw_dir)
        pre_prompt = (raw_dir / "host.stderr.bin").read_bytes()
        if pre_prompt.count(b"DRIVER_INPUT_READY ") != 1 \
                or b"DRIVER_INPUT_ACCEPTED " in pre_prompt:
            raise ArrivalLauncherError("host was not input-blind at readiness")
        (raw_dir / "host.pre_prompt.stderr.bin").write_bytes(pre_prompt)
        preflight = {
            "schema": "s15-arrival-preflight-v1",
            "scope": "MODELS_RESIDENT_HOST_PROMPT_UNSUBMITTED",
            "cohort_sha256": base.EXPECTED_COHORT,
            "input_manifest_sha256": base.EXPECTED_INPUT,
            "profile_id": EXPECTED_PROFILE,
            "profile_evidence_id": recert_report["profile_id"],
            "host_artifacts": recert_report["host_artifacts"],
            "worker_binary_sha256": base.EXPECTED_WORKER,
            "device_boot_id": boot_id,
            "deployed": deployed,
            "thermal_paths": thermal_paths,
            "thermal_start": thermal_start,
            "host_command": host_command,
            "phone_command": phone_command,
            "prompt_in_host_argv": False,
        }
        (raw_dir / "preflight.json").write_bytes(base.canonical(preflight))
        print("LAUNCHER_READY " + json.dumps({
            "cohort_sha256": base.EXPECTED_COHORT,
            "input_manifest_sha256": base.EXPECTED_INPUT,
            "device_boot_id": boot_id,
        }, sort_keys=True), file=sys.stderr, flush=True)

        payload = sys.stdin.buffer.read(4 * 1024 * 1024 + 1)
        request = base.strict_object(payload, "request")
        validate_request(request, boot_id)
        (raw_dir / "request.json").write_bytes(payload)

        paid_start_ns = time.monotonic_ns()
        cohort, inputs, request_ids, prompt = base.load_workload()
        if request["request_ids"] != list(request_ids):
            raise ArrivalLauncherError("request IDs do not match the admitted cohort")
        (raw_dir / "cohort.json").write_bytes(base.COHORT_PATH.read_bytes())
        (raw_dir / "input_manifest.json").write_bytes(base.INPUT_PATH.read_bytes())
        reference = recert_report["reference_tokens"]
        host_exit = recert_run.ExitWatch(host)
        phone_exit = recert_run.ExitWatch(phone)
        if host.process.stdin is None:
            raise ArrivalLauncherError("prepared host stdin is unavailable")
        host.process.stdin.write((prompt + "\n").encode("utf-8"))
        host.process.stdin.flush()
        host.process.stdin.close()
        prompt_submitted_ns = time.monotonic_ns()
        timeout_s = request["timeout_us"] / 1_000_000
        host_rc, host_exit_ns = host_exit.finish(timeout_s)
        phone_rc, phone_exit_ns = phone_exit.finish(timeout_s)
        paid_end_ns = max(host_exit_ns, phone_exit_ns)
        end_thermal = thermal.snapshot()
        if end_thermal.get("sample_age_us", 1_000_001) > 1_000_000 \
                or not base.thermal_ok(end_thermal, base.THERMAL_END_MAX_MILLIC):
            raise ArrivalLauncherError("end thermal gate failed")
        if host_rc != 0 or phone_rc != 0:
            raise ArrivalLauncherError("host or phone process failed")
        placement, rows = base.validate_completion(
            (raw_dir / "host.stderr.bin").read_bytes(),
            (raw_dir / "phone.stdout.bin").read_bytes()
            + (raw_dir / "phone.stderr.bin").read_bytes(),
            reference,
        )
        paid = {
            "schema": "s15-arrival-paid-window-v1",
            "start_monotonic_ns": paid_start_ns,
            "prompt_submitted_ns": prompt_submitted_ns,
            "host_exit_observed_ns": host_exit_ns,
            "phone_exit_observed_ns": phone_exit_ns,
            "end_monotonic_ns": paid_end_ns,
            "elapsed_us": (paid_end_ns - paid_start_ns) // 1000,
            "route_wall_us_max": max(row["request_wall_us"] for row in rows),
            "thermal_end": end_thermal,
            "prompt_visible_during_preflight": False,
        }
        if paid["elapsed_us"] > EXPECTED_TIMEOUT_US:
            raise ArrivalLauncherError("arrival-faithful route missed the remaining deadline")
        (raw_dir / "paid_window.json").write_bytes(base.canonical(paid))
        sys.stdout.buffer.write(base.canonical(session_record(request, placement, "completed")))
        sys.stdout.buffer.flush()
    except Exception as exc:
        (raw_dir / "launcher_error.txt").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="ascii", errors="backslashreplace",
        )
        if request is None:
            return 2
        sys.stdout.buffer.write(base.canonical(session_record(request, None, "error")))
        sys.stdout.buffer.flush()
    finally:
        cleanup(host, phone, thermal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
