#!/usr/bin/env python3
"""S14 CP1 static resident mixed runtime.

Drives the S12-V2 reducer (two_level_vq.Simulator, unchanged) over the frozen
CP0-c catalog + real mix-v1 trace, comparing:

  C0  server_only         optimized server-only (no phone)
  C1  fixed_static_phone  fixed all-phone placement, static READY residency
  C2  static_two_phone    mixed virtual queue, static READY residency
  C3p dynamic_two_level   dynamic slow loop (preview; CP3 formalizes it)

The offline default remains fail-closed and fully deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
S12_DIR = HERE.parent / "s12_trace_vq"
sys.path.insert(0, str(S12_DIR))
sys.path.insert(0, str(HERE))

import catalog_common as cc  # noqa: E402
import cp1_adapter as ad  # noqa: E402
from two_level_vq import Simulator  # noqa: E402
from s12lib import parse_jsonl_bytes, read_bytes_once, sha256_bytes  # noqa: E402

TRACE_PATH = HERE / "fixtures" / "mix_v1.trace.jsonl"
MIX_V1_OUTPUT_SHA256 = "sha256:567d4af180f65f16db345817b15cf1abcd24ac534b930bd4267f542759c4712d"

C0, C1, C2, C3P = "server_only", "fixed_static_phone", "static_two_phone", "dynamic_two_level"
SIM_POLICIES = (C0, C1, C2, C3P)
CONTROL_TO_SIM_POLICY = {C0: C0, C1: C1, C2: C2, C3P: C3P}
HBM_RELIEF_BYTES = 860 * 1024 * 1024  # catalog gemma_head_0_2 server_relief.hbm_bytes_freed

# Arrival-compression sweep (SYNTHETIC load stress): arrival_us = original / den.
SWEEP_DENS = (1, 10, 50, 100, 200, 500)

LIVE_FLEET_VERDICTS = {
    "REAL_FLEET_FFN_PASS",
    "REAL_DEVICE_FFN_PASS",
    "REAL_TWO_PHONE_FFN_RUNTIME_MECHANICS_PASS",
}
LIVE_FLEET_SCHEME = 2
LIVE_FLEET_MIN_TIMEOUT_MS = 10_000


class FleetDeviceConfig:
    """Validated host-side device descriptor for phone-pim-fleet command line."""

    def __init__(self, text: str) -> None:
        parts = [part.strip() for part in text.split(",")]
        if len(parts) != 4:
            raise ad.CP1Error(
                f"invalid --live-device {text!r}; expected id,host,port,serial"
            )
        self.device_id, self.host, port_text, self.serial = parts
        if not self.device_id or not self.host or not self.serial:
            raise ad.CP1Error(
                f"invalid --live-device {text!r}; all fields must be non-empty"
            )
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.host):
            raise ad.CP1Error(f"invalid host in --live-device: {self.host!r}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.device_id):
            raise ad.CP1Error(f"invalid device id in --live-device: {self.device_id!r}")
        if not re.fullmatch(r"[A-Za-z0-9]+", self.serial):
            raise ad.CP1Error(f"invalid serial in --live-device: {self.serial!r}")
        if not re.fullmatch(r"[0-9]+", port_text):
            raise ad.CP1Error(f"invalid port in --live-device: {port_text!r}")
        self.port = int(port_text)
        if self.port <= 0 or self.port > 65535:
            raise ad.CP1Error(f"invalid port in --live-device: {self.port}")

    def to_arg(self) -> str:
        return f"{self.device_id},{self.host},{self.port},{self.serial}"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ad.CP1Error(f"duplicate JSON key: {key!r}")
        out[key] = value
    return out


def _reject_nonfinite(token: str) -> Any:
    raise ad.CP1Error(f"invalid JSON constant: {token!r}")


def _safe_load_json(text: str, label: str) -> dict[str, Any]:
    value = json.loads(
        text,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite,
    )
    if not isinstance(value, dict):
        raise ad.CP1Error(f"{label}: expected object")
    return value


def _require_int(label: str, value: Any, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ad.CP1Error(f"{label}: expected int, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ad.CP1Error(f"{label}: expected >= {minimum}, got {value}")
    return value


def _require_float(label: str, value: Any, minimum: float | None = None, *, integer: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ad.CP1Error(f"{label}: expected number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ad.CP1Error(f"{label}: expected finite number")
    if number < 0:
        raise ad.CP1Error(f"{label}: expected non-negative, got {number}")
    if minimum is not None and number < minimum:
        raise ad.CP1Error(f"{label}: expected >= {minimum}, got {number}")
    if integer and not float(number).is_integer():
        raise ad.CP1Error(f"{label}: expected integer value, got {number}")
    return number


def _require_str(
    label: str,
    value: Any,
    *,
    allowed: set[str] | None = None,
) -> str:
    if type(value) is not str:
        raise ad.CP1Error(f"{label}: expected string")
    if allowed is not None and value not in allowed:
        raise ad.CP1Error(f"{label}: unexpected value {value!r}")
    return value


def _require_bool(label: str, value: Any) -> bool:
    if type(value) is not bool:
        raise ad.CP1Error(f"{label}: expected bool")
    return value


def _require_u64_string(label: str, value: Any, minimum: int = 0) -> str:
    text = _require_str(label, value)
    if not re.fullmatch(r"0|[1-9][0-9]*", text):
        raise ad.CP1Error(f"{label}: expected canonical uint64 string")
    number = int(text)
    if number < minimum or number > (1 << 64) - 1:
        raise ad.CP1Error(f"{label}: uint64 value out of range")
    return text


def _file_identity(path: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as source:
        while True:
            chunk = source.read(4 * 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    if size <= 0:
        raise ad.CP1Error(f"live model is empty: {path}")
    return size, digest.hexdigest()


def _find_fleet_binary(path: str | None) -> str:
    if path:
        binary = Path(path).expanduser()
        if binary.exists():
            return str(binary)
        raise ad.CP1Error(f"--fleet-binary path does not exist: {binary}")

    env = os.environ.get("LLAMA_PHONE_PIM_FLEET")
    if env:
        binary = Path(env).expanduser()
        if binary.exists():
            return str(binary)
        raise ad.CP1Error(f"LLAMA_PHONE_PIM_FLEET path does not exist: {binary}")

    candidates = [
        REPO_ROOT / "build-phone-pim/bin/llama-phone-pim-fleet",
        REPO_ROOT / "build-phone-pim/llama-phone-pim-fleet",
        REPO_ROOT / "build/bin/llama-phone-pim-fleet",
        REPO_ROOT / "llama-phone-pim-fleet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    fallback = shutil.which("llama-phone-pim-fleet")
    if fallback is not None:
        return fallback
    raise ad.CP1Error(
        "phone-pim-fleet binary not found. set --fleet-binary or LLAMA_PHONE_PIM_FLEET "
        "and build the phone-pim target first"
    )


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    if p <= 0.0:
        return data[0]
    if p >= 1.0:
        return data[-1]
    index = int(round((len(data) - 1) * p))
    if index < 0:
        index = 0
    if index >= len(data):
        index = len(data) - 1
    return data[index]


def _parse_fleet_record(raw: dict[str, Any]) -> dict[str, Any]:
    schema_version = _require_int("fleet.record_schema_version", raw.get("record_schema_version"), 1)
    if schema_version != LIVE_FLEET_SCHEME:
        raise ad.CP1Error(
            f"fleet.record_schema_version {schema_version} != {LIVE_FLEET_SCHEME}"
        )

    verdict = _require_str("fleet.verdict", raw.get("verdict"), allowed=LIVE_FLEET_VERDICTS)
    scheduler = _require_str(
        "fleet.scheduler", raw.get("scheduler"), allowed={"persistent_session_work_stealing_v1"}
    )
    protocol_version = _require_int("fleet.protocol_version", raw.get("protocol_version"), 1)
    if protocol_version < 3:
        raise ad.CP1Error(f"fleet.protocol_version {protocol_version} is unsupported")

    device_count = _require_int("fleet.device_count", raw.get("device_count"), 1)
    jobs = _require_int("fleet.jobs", raw.get("jobs"), 1)
    m = _require_int("fleet.M", raw.get("M"), 1)
    route_epoch = _require_u64_string("fleet.route_epoch", raw.get("route_epoch"), 1)
    island_id = _require_u64_string("fleet.island_id", raw.get("island_id"), 1)
    model_source = _require_str(
        "fleet.model_source", raw.get("model_source"), allowed={"prestaged", "published_store"}
    )
    prefix = _require_str("fleet.prefix", raw.get("prefix"))
    model_bytes = _require_int("fleet.model_bytes", raw.get("model_bytes"), 1)
    model_sha256 = _require_str("fleet.model_sha256", raw.get("model_sha256"))
    if not re.fullmatch(r"[0-9a-f]{64}", model_sha256):
        raise ad.CP1Error("fleet.model_sha256 must be lowercase SHA-256")
    _require_str("fleet.oracle_backend", raw.get("oracle_backend"), allowed={"CPU"})
    rel_l2_limit = _require_float("fleet.rel_l2_limit", raw.get("rel_l2_limit"), 1e-15)
    rel_l2_max = _require_float("fleet.rel_l2_max", raw.get("rel_l2_max"), 0.0)
    if rel_l2_max > rel_l2_limit:
        raise ad.CP1Error("fleet.rel_l2_max exceeds fleet.rel_l2_limit")
    validated_dispatch_makespan_ms = _require_float(
        "fleet.validated_dispatch_makespan_ms", raw.get("validated_dispatch_makespan_ms"), 0.0
    )
    last_rpc_complete_ms = _require_float(
        "fleet.last_rpc_complete_ms", raw.get("last_rpc_complete_ms"), 0.0
    )
    if last_rpc_complete_ms > validated_dispatch_makespan_ms:
        raise ad.CP1Error("fleet.last_rpc_complete_ms exceeds dispatch makespan")

    devices = raw.get("devices")
    if type(devices) is not list or len(devices) != device_count:
        raise ad.CP1Error("fleet.devices must contain exactly device_count records")
    device_ids: set[str] = set()
    physical_ids: set[str] = set()
    endpoints: set[tuple[str, int]] = set()
    device_job_counts: dict[str, int] = {}
    device_rel_l2: dict[str, float] = {}
    for item in devices:
        if type(item) is not dict:
            raise ad.CP1Error("fleet.devices item must be object")
        device_id = _require_str("fleet.devices[].id", item.get("id"))
        physical_id = _require_str("fleet.devices[].physical_id", item.get("physical_id"))
        _require_str(
            "fleet.devices[].physical_id_source",
            item.get("physical_id_source"),
            allowed={"caller_asserted"},
        )
        host = _require_str("fleet.devices[].host", item.get("host"))
        port = _require_int("fleet.devices[].port", item.get("port"), 1)
        if port > 65535:
            raise ad.CP1Error("fleet.devices[].port exceeds 65535")
        _require_u64_string("fleet.devices[].session_epoch", item.get("session_epoch"), 1)
        _require_str("fleet.devices[].backend", item.get("backend"))
        _require_str("fleet.devices[].capability", item.get("capability"))
        _require_u64_string("fleet.devices[].initial_generation", item.get("initial_generation"), 1)
        _require_bool("fleet.devices[].initial_ready", item.get("initial_ready"))
        _require_u64_string("fleet.devices[].final_generation", item.get("final_generation"), 1)
        count = _require_int("fleet.devices[].jobs", item.get("jobs"), 1)
        client_p50 = _require_float(
            "fleet.devices[].client_operation_p50_ms", item.get("client_operation_p50_ms"), 0.0
        )
        client_p95 = _require_float(
            "fleet.devices[].client_operation_p95_ms", item.get("client_operation_p95_ms"), 0.0
        )
        _require_float(
            "fleet.devices[].worker_compute_p50_ms", item.get("worker_compute_p50_ms"), 0.0
        )
        if client_p50 > client_p95:
            raise ad.CP1Error("fleet device p50 exceeds p95")
        item_rel_l2 = _require_float("fleet.devices[].rel_l2_max", item.get("rel_l2_max"), 0.0)
        if item_rel_l2 > rel_l2_limit:
            raise ad.CP1Error("fleet device rel_l2_max exceeds limit")
        if device_id in device_ids or physical_id in physical_ids or (host, port) in endpoints:
            raise ad.CP1Error("fleet.devices contains a duplicate identity or endpoint")
        device_ids.add(device_id)
        physical_ids.add(physical_id)
        endpoints.add((host, port))
        device_job_counts[device_id] = count
        device_rel_l2[device_id] = item_rel_l2
    if sum(device_job_counts.values()) != jobs:
        raise ad.CP1Error("fleet device job counts do not sum to fleet.jobs")

    assignments = raw.get("assignments")
    if type(assignments) is not list:
        raise ad.CP1Error("fleet.assignments must be an array")
    if len(assignments) != jobs:
        raise ad.CP1Error(
            f"fleet.assignments length {len(assignments)} != fleet.jobs {jobs}"
        )
    worker_times: list[float] = []
    client_times: list[float] = []
    assignment_ids: set[int] = set()
    assignment_counts = {device_id: 0 for device_id in device_ids}
    assignment_rel_l2 = {device_id: 0.0 for device_id in device_ids}
    for item in assignments:
        if type(item) is not dict:
            raise ad.CP1Error("fleet.assignments item must be object")
        job_id = _require_int("fleet.assignments[].job", item.get("job"), 0)
        device_id = _require_str("fleet.assignments[].device", item.get("device"))
        if device_id not in device_ids:
            raise ad.CP1Error("fleet assignment references an unknown device")
        if job_id in assignment_ids:
            raise ad.CP1Error("fleet assignments contain a duplicate job")
        assignment_ids.add(job_id)
        worker_ms = _require_float(
            "fleet.assignments[].worker_compute_ms", item.get("worker_compute_ms"), 0.0
        )
        client_ms = _require_float(
            "fleet.assignments[].client_operation_ms", item.get("client_operation_ms"), 0.0
        )
        if worker_ms > client_ms:
            raise ad.CP1Error("fleet worker compute exceeds client operation wall")
        rel_l2 = _require_float("fleet.assignments[].rel_l2", item.get("rel_l2"), 0.0)
        if rel_l2 > rel_l2_limit:
            raise ad.CP1Error("fleet assignment rel_l2 exceeds limit")
        worker_times.append(worker_ms)
        client_times.append(client_ms)
        assignment_counts[device_id] += 1
        assignment_rel_l2[device_id] = max(assignment_rel_l2[device_id], rel_l2)

    if assignment_ids != set(range(jobs)):
        raise ad.CP1Error("fleet assignments are not exactly jobs 0..N-1")
    if assignment_counts != device_job_counts:
        raise ad.CP1Error("fleet assignment counts disagree with device job counts")
    tolerance = max(1e-12, rel_l2_limit * 1e-9)
    if abs(max(assignment_rel_l2.values()) - rel_l2_max) > tolerance:
        raise ad.CP1Error("fleet rel_l2_max disagrees with assignments")
    for device_id in device_ids:
        if abs(assignment_rel_l2[device_id] - device_rel_l2[device_id]) > tolerance:
            raise ad.CP1Error("fleet device rel_l2_max disagrees with assignments")

    return {
        "record_schema_version": schema_version,
        "verdict": verdict,
        "scheduler": scheduler,
        "protocol_version": protocol_version,
        "device_count": device_count,
        "jobs": jobs,
        "route_epoch": route_epoch,
        "island_id": island_id,
        "M": m,
        "prefix": prefix,
        "model_source": model_source,
        "model_bytes": model_bytes,
        "model_sha256": model_sha256,
        "rel_l2_limit": rel_l2_limit,
        "rel_l2_max": rel_l2_max,
        "validated_dispatch_makespan_ms": validated_dispatch_makespan_ms,
        "last_rpc_complete_ms": last_rpc_complete_ms,
        "assignments": assignments,
        "assignments_client_p50_ms": _percentile(client_times, 0.5),
        "assignments_worker_p50_ms": _percentile(worker_times, 0.5),
        "devices": [dict(item) for item in devices],
    }


def _run_fleet_command(cmd: list[str], timeout_sec: int) -> dict[str, Any]:
    completed = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    if completed.returncode != 0:
        raise ad.CP1Error(
            f"phone-pim-fleet exited with {completed.returncode}: {shlex.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    stdout = completed.stdout.strip()
    if not stdout:
        raise ad.CP1Error("phone-pim-fleet produced no stdout")
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    record = _safe_load_json(lines[-1], "phone-pim-fleet stdout")
    return _parse_fleet_record(record)


def _run_live_fleet(
    control: str,
    phone_jobs: int,
    live_cfg: dict[str, Any],
) -> dict[str, Any]:
    if control not in {C2, C3P}:
        return {
            "requested_jobs": 0,
            "mode": "skipped",
            "reason": "control_not_configured_for_fleet",
        }

    if phone_jobs <= 0:
        return {
            "requested_jobs": 0,
            "mode": "skipped",
            "reason": "no_phone_routes_in_simulation",
        }

    configured_jobs = live_cfg.get("jobs")
    if configured_jobs is None:
        requested_jobs = phone_jobs
    else:
        requested_jobs = _require_int("live_jobs", configured_jobs, 1)
        requested_jobs = min(requested_jobs, phone_jobs)
    if requested_jobs <= 0:
        return {
            "requested_jobs": 0,
            "mode": "skipped",
            "reason": "requested_jobs_not_positive",
        }

    devices: list[FleetDeviceConfig] = live_cfg["devices"]
    if not devices:
        return {
            "requested_jobs": requested_jobs,
            "mode": "skipped",
            "reason": "no_live_devices",
        }

    timeout_ms = _require_int("live_timeout_ms", live_cfg["timeout_ms"], LIVE_FLEET_MIN_TIMEOUT_MS)
    cmd = [_find_fleet_binary(live_cfg["fleet_binary"])]
    for device in devices:
        cmd.extend(["--device", device.to_arg()])
    cmd.extend(
        [
            "--model",
            live_cfg["model"],
            "--prefix",
            live_cfg["prefix"],
            "--M",
            str(live_cfg["M"]),
            "--jobs",
            str(requested_jobs),
            "--route-epoch",
            str(live_cfg["route_epoch"]),
            "--generation-hint",
            str(live_cfg["generation_hint"]),
            "--island-id",
            str(live_cfg["island_id"]),
        ]
    )
    model_source = _require_str("live_model_source", live_cfg.get("model_source"), allowed={"prestaged", "published-store"})
    cmd.extend(["--model-source", model_source])
    if live_cfg.get("release"):
        cmd.append("--release")
    cmd.extend(["--timeout-ms", str(timeout_ms)])

    timeout_sec = max(1, timeout_ms // 1000 + 5)
    record = _run_fleet_command(cmd, timeout_sec)
    if record["jobs"] != requested_jobs:
        raise ad.CP1Error(f"fleet jobs {record['jobs']} != requested_jobs {requested_jobs}")
    if record["device_count"] != len(devices):
        raise ad.CP1Error("fleet device_count differs from configured device count")
    if record["M"] != live_cfg["M"]:
        raise ad.CP1Error("fleet M differs from the requested value")
    if record["route_epoch"] != str(live_cfg["route_epoch"]):
        raise ad.CP1Error("fleet route_epoch differs from the requested value")
    if record["island_id"] != str(live_cfg["island_id"]):
        raise ad.CP1Error("fleet island_id differs from the requested value")
    if record["prefix"] != live_cfg["prefix"]:
        raise ad.CP1Error("fleet prefix differs from the requested value")
    expected_source = "published_store" if model_source == "published-store" else model_source
    if record["model_source"] != expected_source:
        raise ad.CP1Error("fleet model_source differs from the requested value")

    expected_bytes = live_cfg.get("expected_model_bytes")
    expected_sha256 = live_cfg.get("expected_model_sha256")
    if expected_bytes is None or expected_sha256 is None:
        expected_bytes, expected_sha256 = _file_identity(live_cfg["model"])
    if record["model_bytes"] != expected_bytes or record["model_sha256"] != expected_sha256:
        raise ad.CP1Error("fleet model identity differs from the requested file")

    configured = {device.device_id: device for device in devices}
    returned = {item["id"]: item for item in record["devices"]}
    if set(returned) != set(configured):
        raise ad.CP1Error("fleet returned device IDs differ from configured device IDs")
    for device_id, expected in configured.items():
        actual = returned[device_id]
        if (
            actual["physical_id"] != expected.serial
            or actual["host"] != expected.host
            or actual["port"] != expected.port
        ):
            raise ad.CP1Error(f"fleet device identity mismatch for {device_id}")
        if actual["backend"] != live_cfg.get("expected_backend", "HTP0"):
            raise ad.CP1Error(f"fleet backend mismatch for {device_id}")
        expected_capability = live_cfg.get("expected_capability")
        if expected_capability is not None and actual["capability"] != expected_capability:
            raise ad.CP1Error(f"fleet capability mismatch for {device_id}")
        if live_cfg.get("require_initial_ready", True) and not actual["initial_ready"]:
            raise ad.CP1Error(f"fleet device {device_id} was not READY before the run")
        if not live_cfg.get("release") and actual["final_generation"] != actual["initial_generation"]:
            raise ad.CP1Error(f"fleet generation changed unexpectedly for {device_id}")

    return {
        "requested_jobs": requested_jobs,
        "simulated_phone_dispatched": phone_jobs,
        "mode": "executed",
        "fleet_device_count": record["device_count"],
        "fleet_verdict": record["verdict"],
        "fleet_jobs": record["jobs"],
        "fleet_schema_version": record["record_schema_version"],
        "fleet_protocol_version": record["protocol_version"],
        "fleet_model_source": record["model_source"],
        "fleet_model_sha256": record["model_sha256"],
        "fleet_rel_l2_max": record["rel_l2_max"],
        "fleet_m": record["M"],
        "fleet_route_epoch": record["route_epoch"],
        "fleet_island_id": record["island_id"],
        "fleet_last_rpc_complete_ms": record["last_rpc_complete_ms"],
        "fleet_validated_dispatch_makespan_ms": record["validated_dispatch_makespan_ms"],
        "fleet_assignment_count": len(record["assignments"]),
        "fleet_worker_compute_p50_ms": record["assignments_worker_p50_ms"],
        "fleet_client_operation_p50_ms": record["assignments_client_p50_ms"],
        "fleet_command": shlex.join(cmd),
        "fleet_devices": [
            {"id": item.get("id"), "generation": item.get("initial_generation")}
            for item in record.get("devices", [])
        ],
    }


def _overlap_us(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    events = sorted([(s, f) for s, f in a]) + sorted([(s, f) for s, f in b])
    total = 0
    for s1, f1 in a:
        for s2, f2 in b:
            lo, hi = max(s1, s2), min(f1, f2)
            if hi > lo:
                total += hi - lo
    return total


def _load_trace() -> list[dict[str, Any]]:
    data = read_bytes_once(TRACE_PATH)
    if sha256_bytes(data) != MIX_V1_OUTPUT_SHA256:
        raise ad.CP1Error("mix_v1.trace.jsonl digest does not match the frozen mix-v1 output_sha256")
    return parse_jsonl_bytes(data, str(TRACE_PATH))


def _metrics(
    control: str,
    result: dict[str, Any],
    reducer_catalog: dict[str, dict[str, Any]],
    live_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    tc = result["terminal_counts"]
    useful = tc.get("completed_phone", 0) + tc.get("completed_server", 0)
    tardy = tc.get("tardy_phone", 0) + tc.get("tardy_server", 0)
    rejected = tc.get("rejected_queue_full", 0)
    timed_out = tc.get("timed_out", 0)

    phone_routed = sum(1 for o in result["outcomes"] if o["route"] == "phone")
    server_routed = sum(1 for o in result["outcomes"] if o["route"] == "server")

    server_full = sum(1 for si in result["server_intervals"] if si["phase"] == "full")
    server_tail = sum(1 for si in result["server_intervals"] if si["phase"] == "tail")
    server_busy = sum(si["finish_us"] - si["start_us"] for si in result["server_intervals"])

    # CP1 currently uses OP15 only, but C3p/C2 support both devices.
    phone_iv: list[tuple[int, int]] = []
    for dl in result["device_ledgers"]:
        phone_iv.extend((ci["start_us"], ci["finish_us"]) for ci in dl["compute_intervals"])
    server_iv = [(si["start_us"], si["finish_us"]) for si in result["server_intervals"]]
    overlap = _overlap_us(phone_iv, server_iv)

    # Completed-request latency, excluding rejected/timed-out with no finish.
    latencies = sorted(
        o["finish_us"] - o["arrival_us"] for o in result["outcomes"] if o["finish_us"] is not None
    )
    p50 = latencies[len(latencies) // 2] if latencies else None
    mean = (sum(latencies) // len(latencies)) if latencies else None

    island = reducer_catalog["gemma_head_0_2"]
    per_head_save = island["synthetic_server_full_us"] - island["synthetic_server_tail_us"]
    # Relief is one request of head offload per phone-routed request.
    server_compute_relief_us = per_head_save * phone_routed
    # HBM relief only applies when no server_full work remains.
    hbm_relief_bytes = HBM_RELIEF_BYTES if server_full == 0 and phone_routed > 0 else 0

    if result["transfer_bytes_completed"] != 0:
        raise ad.CP1Error(f"{control}: unexpected transfer activity in static residency")

    live_fleet = _run_live_fleet(control, phone_routed, live_cfg) if live_cfg else {
        "requested_jobs": 0,
        "mode": "offline",
    }

    return {
        "control": control,
        "terminal_counts": tc,
        "useful_completions": useful,
        "tardy": tardy,
        "rejected": rejected,
        "timed_out": timed_out,
        "terminal_total": useful + tardy + rejected + timed_out,
        "phone_routed": phone_routed,
        "server_routed": server_routed,
        "server_full_intervals": server_full,
        "server_tail_intervals": server_tail,
        "server_busy_us": server_busy,
        "phone_server_overlap_us": overlap,
        "latency_p50_us": p50,
        "latency_mean_us": mean,
        "server_compute_relief_us": server_compute_relief_us,
        "hbm_relief_bytes": hbm_relief_bytes,
        "result_digest": result["result_digest"],
        "live_fleet": live_fleet,
    }


def _run_point(
    catalog: dict[str, Any],
    reducer_catalog: dict[str, dict[str, Any]],
    mix_records: list[dict[str, Any]],
    den: int,
    deadline_factor: int,
    queue_limit: int,
    live_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    trace = ad.build_reducer_trace(mix_records, reducer_catalog, deadline_factor, 1, den)
    config = ad.build_reducer_config(reducer_catalog, trace, SIM_POLICIES, queue_limit)
    by_sim_policy: dict[str, dict[str, Any]] = {}
    for policy in SIM_POLICIES:
        by_sim_policy[policy] = Simulator(policy, config, reducer_catalog, trace).run()

    per_control: dict[str, dict[str, Any]] = {}
    for control, sim_policy in CONTROL_TO_SIM_POLICY.items():
        per_control[control] = _metrics(control, by_sim_policy[sim_policy], reducer_catalog, live_cfg)

    base = per_control[C0]["useful_completions"]
    window_us = trace[-1]["arrival_us"] - trace[0]["arrival_us"]
    comparison = {}
    for control in (C1, C2, C3P):
        useful = per_control[control]["useful_completions"]
        comparison[control] = {
            "useful_ratio_vs_C0": (useful / base) if base else None,
            "throughput_retained_ge_0_90": (base == 0) or (useful >= 0.90 * base),
            "server_compute_relief_us": per_control[control]["server_compute_relief_us"],
            "hbm_relief_bytes": per_control[control]["hbm_relief_bytes"],
        }

    return {
        "arrival_compression_den": den,
        "synthetic_load": den != 1,
        "window_us": window_us,
        "queue_limit": queue_limit,
        "deadline_factor": deadline_factor,
        "per_control": per_control,
        "comparison": comparison,
        "live_mode": "live" if live_cfg is not None else "offline",
    }


def _default_live_devices() -> list[FleetDeviceConfig]:
    env = os.environ.get("S14_CP1_LIVE_DEVICES", "")
    if not env:
        return []
    devices: list[FleetDeviceConfig] = []
    for item in env.split(";"):
        token = item.strip()
        if not token:
            continue
        devices.append(FleetDeviceConfig(token))
    return devices


def _build_live_config(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.mode != "live":
        return None

    if args.live_devices:
        devices = [FleetDeviceConfig(item) for item in args.live_devices]
    else:
        devices = _default_live_devices()
    if not devices:
        raise ad.CP1Error("--mode live requires at least one --live-device")

    model_path = Path(args.live_model).expanduser()
    if not model_path.exists():
        raise ad.CP1Error(f"live model not found: {model_path}")
    model_bytes, model_sha256 = _file_identity(str(model_path))

    return {
        "fleet_binary": args.fleet_binary,
        "model": str(model_path),
        "prefix": args.live_prefix,
        "M": _require_int("live-M", args.live_m, 1),
        "route_epoch": _require_int("live-route-epoch", args.live_route_epoch, 1),
        "generation_hint": _require_int("live-generation-hint", args.live_generation_hint, 0),
        "island_id": _require_int("live-island-id", args.live_island_id, 0),
        "model_source": _require_str(
            "live-model-source", args.live_model_source, allowed={"prestaged", "published-store"}
        ),
        "release": _require_bool("live-release", args.live_release),
        "timeout_ms": _require_int("live-timeout-ms", args.live_timeout_ms, LIVE_FLEET_MIN_TIMEOUT_MS),
        "jobs": args.live_jobs,
        "expected_model_bytes": model_bytes,
        "expected_model_sha256": model_sha256,
        "expected_backend": "HTP0",
        "expected_capability": "prestaged_gemma4_dense_ffn_v3",
        "require_initial_ready": True,
        "devices": devices,
    }


def run(
    deadline_factor: int = 3,
    queue_limit: int = 64,
    mode: str = "offline",
    live_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if mode == "live" and live_cfg is None:
        raise ad.CP1Error("--mode live requires a live_cfg")
    catalog = ad.load_frozen_catalog()
    mix_records = _load_trace()
    profile = ad.build_measured_profile(catalog)
    reducer_catalog = ad.to_reducer_catalog(profile)

    real = _run_point(catalog, reducer_catalog, mix_records, 1, deadline_factor, queue_limit, live_cfg)
    sweep = [
        _run_point(catalog, reducer_catalog, mix_records, den, deadline_factor, queue_limit, live_cfg)
        for den in SWEEP_DENS
    ]

    c1 = real["comparison"][C1]
    c2 = real["comparison"][C2]
    median_throughput_ge_090 = c2["throughput_retained_ge_0_90"]

    server_busy_c0 = real["per_control"][C0]["server_busy_us"]
    relief_frac = (c2["server_compute_relief_us"] / server_busy_c0) if server_busy_c0 else 0.0
    relief_material = (c2["hbm_relief_bytes"] > 0) or (relief_frac >= 0.01)

    contention_points = [
        pt["comparison"][C2]["useful_ratio_vs_C0"]
        for pt in sweep
        if pt["per_control"][C2]["rejected"] == 0 and pt["per_control"][C0]["rejected"] == 0
    ]
    min_ratio_no_reject = min(r for r in contention_points if r is not None) if contention_points else None
    contention_robust = (min_ratio_no_reject is None) or (min_ratio_no_reject >= 0.90)

    live_fleet_rows_executed = 0
    if mode == "live":
        for point in [real] + sweep:
            for control in [C2, C3P]:
                for key in ("live_fleet",):
                    live_metrics = point["per_control"][control].get(key, {})
                    if live_metrics.get("mode") == "executed":
                        live_fleet_rows_executed += 1

    mechanics_pass = all(
        point["per_control"][p]["terminal_total"] == len(mix_records)
        for point in [real] + sweep
        for p in (C0, C1, C2, C3P)
    )
    sweep_terminal_conservation = all(
        point["per_control"][p]["terminal_total"] == len(mix_records)
        for point in sweep
        for p in (C0, C1, C2, C3P)
    )

    if mechanics_pass and median_throughput_ge_090 and relief_material and contention_robust:
        verdict = "STATIC_MIXED_MECHANICS_PASS"
    elif mechanics_pass:
        verdict = "STATIC_MIXED_MECHANICS_PASS_RELIEF_INSUFFICIENT"
    else:
        verdict = "STATIC_MIXED_MECHANICS_FAIL"

    artifact = {
        "schema": "s14-cp1-runtime-result-v1",
        "scope": "CP1_STATIC_MIXED_RUNTIME_MECHANICS_NO_ENERGY",
        "energy_status": "NOT_RUN",
        "bindings": {
            "island_catalog_hash": catalog["catalog_hash"],
            "mix_v1_output_sha256": MIX_V1_OUTPUT_SHA256,
            "measured_profile": profile,
        },
        "policies": {
            "C0": C0,
            "C1_fixed_phone": C1,
            "C2_static_vq": C2,
            "C3_preview": C3P,
        },
        "verdict": verdict,
        "verdict_basis": {
            "load": "real_median_mix_v1",
            "cp1_gate_complete": False,
            "cp1_gate_blockers": [
                "only one measured phone island is available",
                "BGE has no measured per-encode latency",
                "no live S13 device dispatch was run" if mode == "offline" else "live dispatch enabled",
            ],
            "mechanics_pass_terminal_conservation": mechanics_pass,
            "sweep_terminal_conservation": sweep_terminal_conservation,
            "c1_useful_ratio_vs_C0": c1["useful_ratio_vs_C0"],
            "c2_useful_ratio_vs_C0": c2["useful_ratio_vs_C0"],
            "median_throughput_ge_0_90": median_throughput_ge_090,
            "median_throughput_note": "median load is ~3% server-utilised; retaining throughput here is trivial",
            "c1_hbm_relief_bytes": real["per_control"][C1]["hbm_relief_bytes"],
            "c2_hbm_relief_bytes": c2["hbm_relief_bytes"],
            "c1_server_compute_relief_us": real["per_control"][C1]["server_compute_relief_us"],
            "c2_server_compute_relief_us": c2["server_compute_relief_us"],
            "server_compute_relief_fraction": relief_frac,
            "relief_material_ge_1pct_or_hbm": relief_material,
            "c0_latency_p50_us": real["per_control"][C0]["latency_p50_us"],
            "c1_latency_p50_us": real["per_control"][C1]["latency_p50_us"],
            "c2_latency_p50_us": real["per_control"][C2]["latency_p50_us"],
            "contention_min_useful_ratio_no_reject": min_ratio_no_reject,
            "contention_robust_ge_0_90": contention_robust,
            "live_mode": mode,
            "live_fleet_rows_executed": live_fleet_rows_executed,
        },
        "caveats": [
            "All catalog rows are dispatch-INELIGIBLE (LOWER_BOUND latency, no placement cert): mechanics only.",
            "Single measured island (gemma_head_0_2, 2-layer head): server_tail ~= server_full, so server-compute relief is marginal by construction.",
            "HBM relief (860 MiB) is only counted under exclusive offload (server ran zero fulls).",
            "bge_encoder_0_12 is excluded from the runtime (no measured phone latency).",
            "Phone/server overlap is modeled by the reused S12 reducer; live fleet rows are optional capacity checks only.",
            "Arrival-compression sweep is a SYNTHETIC load stress, not a real-trace claim.",
        ],
        "real_median": real,
        "synthetic_load_sweep": sweep,
    }
    artifact["result_digest"] = cc.sha256_of({k: v for k, v in artifact.items() if k != "result_digest"})
    return artifact


def _fmt(v: Any) -> str:
    return "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def _print_live_summary(record: dict[str, Any]) -> None:
    if not record or record.get("mode") != "executed":
        print("  live-fleet: skipped")
        return
    print(
        f"  live-fleet: jobs={record.get('requested_jobs','n/a')} "
        f"device_count={record.get('fleet_device_count','n/a')} "
        f"dispatch_ms={record.get('fleet_validated_dispatch_makespan_ms','n/a')} "
        f"worker_p50_ms={record.get('fleet_worker_compute_p50_ms','n/a')} "
        f"verdict={record.get('fleet_verdict','n/a')}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S14 CP1 static resident mixed runtime")
    parser.add_argument("--output", default=str(HERE / "cp1_result.json"))
    parser.add_argument("--deadline-factor", type=int, default=3)
    parser.add_argument("--queue-limit", type=int, default=64)
    parser.add_argument("--mode", choices=("offline", "live"), default="offline")

    parser.add_argument("--fleet-binary", default=None, help="path to llama-phone-pim-fleet")
    parser.add_argument(
        "--live-model",
        default=str(REPO_ROOT / "scratchpad" / "phone_pim" / "12b-f16-mid-2-3.gguf"),
    )
    parser.add_argument("--live-prefix", default="blk.2")
    parser.add_argument("--live-m", type=int, default=16)
    parser.add_argument("--live-route-epoch", type=int, default=1)
    parser.add_argument("--live-generation-hint", type=int, default=1)
    parser.add_argument("--live-island-id", type=int, default=1)
    parser.add_argument(
        "--live-model-source",
        choices=("prestaged", "published-store"),
        default="prestaged",
    )
    parser.add_argument("--live-release", action="store_true")
    parser.add_argument("--live-timeout-ms", type=int, default=600000)
    parser.add_argument("--live-jobs", type=int, default=None)
    parser.add_argument("--live-device", action="append", dest="live_devices")
    args = parser.parse_args(argv)

    live_cfg = _build_live_config(args) if args.mode == "live" else None
    artifact = run(args.deadline_factor, args.queue_limit, args.mode, live_cfg)
    Path(args.output).write_bytes(cc.canonical_json(artifact))

    print(f"CP1 verdict: {artifact['verdict']}")
    print(
        f"  bindings: catalog {artifact['bindings']['island_catalog_hash'][:20]}... "
        f"mix-v1 {MIX_V1_OUTPUT_SHA256[:20]}..."
    )
    print(f"  result_digest: {artifact['result_digest']}")
    print()

    print("REAL median mix-v1 (177 requests over ~897 s):")
    rm = artifact["real_median"]["per_control"]
    for pol in (C0, C1, C2, C3P):
        m = rm[pol]
        print(
            f"  {pol:18s} useful={m['useful_completions']:3d} phone={m['phone_routed']:3d} "
            f"server_full={m['server_full_intervals']:3d} tail={m['server_tail_intervals']:3d} "
            f"lat_p50={_fmt(m['latency_p50_us'])} overlap_us={m['phone_server_overlap_us']} "
            f"hbm_relief={m['hbm_relief_bytes']}"
        )
        if artifact["verdict_basis"]["live_mode"] == "live":
            _print_live_summary(m.get("live_fleet", {}))

    vb = artifact["verdict_basis"]
    print(
        f"  C1 vs C0 useful ratio: {_fmt(vb['c1_useful_ratio_vs_C0'])} "
        "(native fixed_static_phone in S12)"
    )
    print(
        f"  C2 vs C0 useful ratio: {_fmt(vb['c2_useful_ratio_vs_C0'])} "
        f"(median ~3% util, trivial)  lat C0->C2: {vb['c0_latency_p50_us']} -> {vb['c2_latency_p50_us']} us"
    )
    print(
        f"  relief: hbm={vb['c2_hbm_relief_bytes']}B compute_frac={_fmt(vb['server_compute_relief_fraction'])} "
        f"material={vb['relief_material_ge_1pct_or_hbm']}  "
        f"contention min ratio (no-reject)={_fmt(vb['contention_min_useful_ratio_no_reject'])} "
        f"robust={vb['contention_robust_ge_0_90']}"
    )
    print(
        f"  mode: {vb['live_mode']} rows={vb['live_fleet_rows_executed']}"
    )
    print(
        "SYNTHETIC arrival-compression sweep (den, C2 useful ratio, C2 phone, C2 server_full, C2 rejected):"
    )
    for pt in artifact["synthetic_load_sweep"]:
        m = pt["per_control"][C2]
        cmp = pt["comparison"][C2]
        print(
            f"  den={pt['arrival_compression_den']:4d} window={pt['window_us']:>11d}us "
            f"ratio={_fmt(cmp['useful_ratio_vs_C0'])} phone={m['phone_routed']:3d} "
            f"server_full={m['server_full_intervals']:3d} rejected={m['rejected']:3d} "
            f"overlap_us={m['phone_server_overlap_us']}"
        )
        if artifact["verdict_basis"]["live_mode"] == "live":
            _print_live_summary(m.get("live_fleet", {}))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
