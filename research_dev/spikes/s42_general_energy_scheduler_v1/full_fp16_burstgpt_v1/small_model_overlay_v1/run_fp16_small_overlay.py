#!/usr/bin/env python3
"""Run and measure the Llama 1B overlay beside the F16 BurstGPT replay."""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import replace
import hashlib
import http.client
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
S42_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[4]
MIXED_ROOT = S42_ROOT / "mixed_model_trace_v1"
sys.path[:0] = [str(REPO_ROOT), str(S42_ROOT), str(MIXED_ROOT), str(HERE)]

import run_six_model_trace as physical  # noqa: E402
from research_dev.scheduler import (  # noqa: E402
    BackgroundRuntimeMonitor,
    BackgroundRuntimeSnapshot,
    Decision,
    DeviceMemoryCapacity,
    HeterogeneousRuntimeSnapshot,
    MarginalSystemCostContext,
    Request,
    RuntimeCapabilityCatalog,
    RuntimePhaseObservation,
    RuntimePlacementSnapshot,
    RuntimeProtectedWorkObservation,
    RuntimeRequestTicket,
    UnifiedScheduler,
    decision_to_json,
)
from research_dev.scheduler.adapters import (  # noqa: E402
    CanonicalPhysicalAdapter,
    CanonicalHttpExecutionBackend,
    CanonicalStaticSplitPrewarmer,
    DeviceRuntimeTelemetry,
    EndpointRuntimeSample,
    HttpEndpointFailure,
    LlamaCppCompletionPayload,
    LlamaCppHttpClient,
    PolledPhonePowerSampler,
    RaplNvmlPhoneEnergyMeter,
    RuntimeActivityTracker,
    RuntimeSnapshotBuilder,
    integrate_milliwatt_samples,
    static_split_launch_contract,
    validate_static_split_ready_log,
)


CONFIRMATION = "RUN_FULL_FP16_LLAMA1B_OVERLAY"
MANIFEST_SCHEMAS = {
    "s42-full-fp16-llama1b-idle-split-calibration-manifest-v1",
    "s42-full-fp16-llama1b-natural-validation-manifest-v1",
    "s42-full-fp16-llama1b-overlay-manifest-v1",
    "s42-full-fp16-llama1b-phase-overlay-manifest-v1",
}
TRACE_SCHEMAS = {
    "s42-full-fp16-llama1b-idle-split-calibration-v1",
    "s42-full-fp16-llama1b-natural-validation-v1",
    "s42-full-fp16-llama1b-overlay-v1",
    "s42-full-fp16-llama1b-phase-overlay-v1",
}
BASE_RESULT_SCHEMA = "s41-hierarchical-burstgpt-result-v1"
RESULT_SCHEMA = "s42-full-fp16-llama1b-combined-result-v2"
EVENT_SCHEMA = "s42-full-fp16-llama1b-event-v1"
LLAMA1 = "llama-3.2-1b-instruct-q4_0"
WORKLOAD_ID = "llama-1b-resident-task"
SPLIT_ROUTE = "cpu-phone-ffn-split"
FFN_MANIFEST_SCHEMA = "s42-llama-dense-ffn-manifest-v1"
FFN_POLICY_SCHEMA = "s42-llama-ffn-vq-compiled-policy-v1"
GIB = 1024**3
LEASE_RENEWAL_GUARD_US = 250_000
LEASE_RENEWAL_QUANTUM_US = 2_000_000
LARGE_PHASE_IDS = {
    "idle": 0,
    "qwen": 1,
    "switching": 2,
    "gemma": 3,
}


class OverlayError(ValueError):
    pass


EndpointTransportError = HttpEndpointFailure


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OverlayError(message)


def sleep_until_ns(target_ns: int) -> None:
    while True:
        remaining_ns = target_ns - time.monotonic_ns()
        if remaining_ns <= 0:
            return
        time.sleep(min(remaining_ns / 1e9, 0.01))


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


def write_new_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    require(
        not path.exists() and not temporary.exists(),
        f"new atomic output path: {path}",
    )
    try:
        temporary.write_bytes(canonical(value))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def verified_record(path: Path, schema: str) -> dict[str, Any]:
    value = load_object(path)
    supplied = value.get("record_sha256")
    unsigned = {key: row for key, row in value.items()
                if key != "record_sha256"}
    require(
        value.get("schema") == schema
        and type(supplied) is str
        and supplied == hashlib.sha256(canonical(unsigned)).hexdigest(),
        f"record identity: {path}",
    )
    return value


def output_token_receipt(
    rows: list[dict[str, Any]],
    index_key: str,
) -> dict[str, object]:
    token_payload = [
        {
            "event_id": row["event_id"],
            "index": row[index_key],
            "tokens": row["tokens"],
        }
        for row in sorted(rows, key=lambda item: item[index_key])
    ]
    shape_payload = [
        {
            "actual_output_tokens": len(row["tokens"]),
            "event_id": row["event_id"],
            "index": row[index_key],
            "input_tokens": row["input_tokens"],
            "requested_output_tokens": row["output_tokens"],
        }
        for row in sorted(rows, key=lambda item: item[index_key])
    ]
    require(
        all(
            row["actual_output_tokens"]
                == row["requested_output_tokens"]
            for row in shape_payload
        ),
        "exact output shape receipt",
    )
    return {
        "actual_output_tokens": sum(len(row["tokens"]) for row in rows),
        "output_tokens": sum(row["output_tokens"] for row in rows),
        "request_count": len(rows),
        "shape_sha256": hashlib.sha256(
            canonical(shape_payload)
        ).hexdigest(),
        "token_sha256": hashlib.sha256(
            canonical(token_payload)
        ).hexdigest(),
    }


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(type(value) is dict, f"object expected: {path}")
    return value


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, raw in enumerate(
        path.read_bytes().splitlines(keepends=True), 1
    ):
        require(raw.endswith(b"\n"), f"line {line_number}: framing")
        row = json.loads(raw)
        require(
            type(row) is dict and canonical(row) == raw,
            f"line {line_number}: canonical JSON",
        )
        rows.append(row)
    return rows


def meminfo(text: str) -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        name, separator, raw = line.partition(":")
        if not separator or name not in {"MemTotal", "MemAvailable"}:
            continue
        fields = raw.strip().split()
        require(len(fields) == 2 and fields[1] == "kB", "meminfo units")
        values[name] = int(fields[0]) * 1024
    require(set(values) == {"MemTotal", "MemAvailable"}, "meminfo fields")
    return values["MemTotal"], values["MemAvailable"]


def cpu_times(text: str) -> tuple[int, int]:
    fields = text.splitlines()[0].split()
    require(fields[0] == "cpu" and len(fields) >= 8, "/proc/stat CPU row")
    values = [int(value) for value in fields[1:]]
    total = sum(values)
    idle = values[3] + values[4]
    return total, idle


def cpu_utilization_pct(
    previous: tuple[int, int],
    current: tuple[int, int],
    last_value: int | None,
) -> int:
    total_delta = current[0] - previous[0]
    idle_delta = current[1] - previous[1]
    if total_delta == 0 and idle_delta == 0:
        return 100 if last_value is None else last_value
    require(
        total_delta > 0 and 0 <= idle_delta <= total_delta,
        "CPU utilization counters",
    )
    return round(100 * (total_delta - idle_delta) / total_delta)


class HostContentionSampler:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.previous = cpu_times(
            Path("/proc/stat").read_text(encoding="ascii")
        )
        self.last_cpu_utilization_pct: int | None = None

    def sample(self) -> dict[str, int]:
        with self.lock:
            current = cpu_times(
                Path("/proc/stat").read_text(encoding="ascii")
            )
            utilization_pct = cpu_utilization_pct(
                self.previous,
                current,
                self.last_cpu_utilization_pct,
            )
            if current != self.previous:
                self.previous = current
            self.last_cpu_utilization_pct = utilization_pct
        load_fields = Path("/proc/loadavg").read_text(
            encoding="ascii"
        ).split()
        require(len(load_fields) >= 3, "/proc/loadavg")
        pressure_line = next(
            line for line in Path("/proc/pressure/memory").read_text(
                encoding="ascii"
            ).splitlines()
            if line.startswith("some ")
        )
        pressure = dict(
            field.split("=", 1) for field in pressure_line.split()[1:]
        )
        return {
            "cpu_load_1m_milli": round(float(load_fields[0]) * 1000),
            "cpu_utilization_pct": utilization_pct,
            "memory_stall_avg10_basis_points": round(
                float(pressure["avg10"]) * 100
            ),
        }


def phase_power_observation(
    sampler: physical.DynamicSampler,
) -> dict[str, int | str | bool]:
    rows = list(sampler.rows[-6:])
    intervals = []
    for left, right in zip(rows, rows[1:]):
        left_rapl = left.get("rapl_package")
        right_rapl = right.get("rapl_package")
        if type(left_rapl) is not dict or type(right_rapl) is not dict:
            continue
        left_t_ns = left_rapl.get("sample_t_ns")
        right_t_ns = right_rapl.get("sample_t_ns")
        left_energy_uj = left_rapl.get("energy_uj")
        right_energy_uj = right_rapl.get("energy_uj")
        maximum_uj = left_rapl.get("max_energy_range_uj")
        if not (
            type(left_t_ns) is int
            and type(right_t_ns) is int
            and right_t_ns > left_t_ns
            and type(left_energy_uj) is int
            and type(right_energy_uj) is int
            and type(maximum_uj) is int
            and maximum_uj > 0
            and right_rapl.get("max_energy_range_uj") == maximum_uj
        ):
            continue
        delta_uj = right_energy_uj - left_energy_uj
        if delta_uj < 0:
            delta_uj += maximum_uj
        if not 0 <= delta_uj < maximum_uj:
            continue
        cpu_package_power_mw = round(
            delta_uj * 1_000_000 / (right_t_ns - left_t_ns)
        )
        left_gpu = left.get("gpu")
        right_gpu = right.get("gpu")
        if type(left_gpu) is not dict or type(right_gpu) is not dict:
            continue
        left_gpu_mw = left_gpu.get("power_mw")
        right_gpu_mw = right_gpu.get("power_mw")
        if type(left_gpu_mw) is not int or type(right_gpu_mw) is not int:
            continue
        gpu_board_power_mw = round((left_gpu_mw + right_gpu_mw) / 2)
        intervals.append((cpu_package_power_mw, gpu_board_power_mw))
    if not intervals:
        return {
            "hardware_counter_available": False,
            "sample_count": 0,
            "status": "UNAVAILABLE",
        }
    cpu_package_power_mw = round(
        sum(row[0] for row in intervals) / len(intervals)
    )
    gpu_board_power_mw = round(
        sum(row[1] for row in intervals) / len(intervals)
    )
    return {
        "cpu_package_power_mw": cpu_package_power_mw,
        "gpu_board_power_mw": gpu_board_power_mw,
        "hardware_counter_available": True,
        "sample_count": len(intervals),
        "source": "RAPL_package_plus_NVML_board",
        "status": "MEASURED",
        "total_server_power_mw": (
            cpu_package_power_mw + gpu_board_power_mw
        ),
    }


def live_large_model_phase(
    base_output: Path,
    now_ns: int,
) -> dict[str, Any]:
    events_path = base_output / "events.jsonl"
    require(events_path.is_file(), "base event stream")
    rows = []
    for raw in events_path.read_bytes().splitlines(keepends=True):
        if not raw.endswith(b"\n"):
            continue
        value = json.loads(raw)
        if type(value) is dict and value.get("t_ns", now_ns) <= now_ns:
            rows.append(value)
    phase_rows = [row for row in rows if row.get("kind") == "large_model_phase"]
    require(phase_rows, "live large-model phase receipt")
    phase = max(phase_rows, key=lambda row: row["t_ns"])
    phase_name = phase.get("phase")
    require(phase_name in LARGE_PHASE_IDS, "large-model phase name")
    if phase_name == "qwen":
        role = "hot"
        arrived = sum(
            row.get("kind") == "request_arrival" and row.get("role") == role
            for row in rows
        )
        completed = sum(
            row.get("kind") == "request_complete" and row.get("role") == role
            for row in rows
        )
        active_large_requests = max(0, arrived - completed)
    elif phase_name == "gemma":
        completed = sum(
            row.get("kind") == "request_complete"
            and row.get("role") == "cold"
            for row in rows
        )
        active_large_requests = max(0, 17 - completed)
    else:
        active_large_requests = 0
    return {
        "active_large_requests": active_large_requests,
        "cuda_owner": phase.get("cuda_owner"),
        "functionfs_owner": phase.get("functionfs_owner"),
        "large_model_arm": phase.get("large_model_arm"),
        "phase": phase_name,
        "phase_id": LARGE_PHASE_IDS[phase_name],
        "phase_start_ns": phase["t_ns"],
    }


def endpoint_json(host: str, port: int, path: str, timeout: float) -> dict[str, Any]:
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        content = response.read()
        require(response.status == 200, f"endpoint status {response.status}")
        value = json.loads(content)
        require(type(value) is dict, "endpoint JSON object")
        return value
    finally:
        connection.close()


def endpoint_ready(host: str, port: int) -> bool:
    try:
        return endpoint_json(host, port, "/health", 1).get("status") == "ok"
    except (OSError, ValueError, json.JSONDecodeError, OverlayError):
        return False


def endpoint_has_idle_slot(host: str, port: int) -> bool:
    connection = http.client.HTTPConnection(host, port, timeout=1)
    try:
        connection.request("GET", "/slots")
        response = connection.getresponse()
        content = response.read()
        if response.status != 200:
            return False
        value = json.loads(content)
        return (
            type(value) is list
            and any(
                type(slot) is dict and slot.get("is_processing") is False
                for slot in value
            )
        )
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        http.client.HTTPException,
    ):
        return False
    finally:
        connection.close()


def endpoint_slots(host: str, port: int, timeout: float) -> list[dict[str, Any]]:
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request("GET", "/slots")
        response = connection.getresponse()
        content = response.read()
        require(response.status == 200, "slots endpoint status")
        value = json.loads(content)
        require(
            type(value) is list
            and all(type(slot) is dict for slot in value),
            "slots endpoint JSON",
        )
        return value
    finally:
        connection.close()


def executor_runtime_snapshot(host: str, port: int) -> dict[str, object]:
    healthy = endpoint_ready(host, port)
    if not healthy:
        return {
            "free_slots": 0,
            "health": "unhealthy",
            "processing_task_ids": [],
            "slots_probe": "not_attempted",
            "total_slots": 0,
        }
    try:
        slots = endpoint_slots(host, port, 1)
    except (OSError, ValueError, json.JSONDecodeError, OverlayError):
        return {
            "free_slots": 0,
            "health": "healthy",
            "processing_task_ids": [],
            "slots_probe": "failed",
            "total_slots": 0,
        }
    return {
        "free_slots": sum(
            slot.get("is_processing") is False for slot in slots
        ),
        "health": "healthy",
        "processing_task_ids": sorted(
            slot["id_task"] for slot in slots
            if type(slot.get("id_task")) is int
            and slot.get("is_processing") is True
        ),
        "slots_probe": "live",
        "total_slots": len(slots),
    }


def start_split_server(
    spec: physical.ModelSpec,
    server_cpu: Path,
    cpu_lib_dir: Path,
    output: Path,
    phone_host: str,
    phone_port: int,
    manifest: dict[str, Any],
    compiled_policy: dict[str, Any],
) -> tuple[physical.run_trace.CapturedProcess, float, dict[str, Any]]:
    command, environment = physical.server_command(spec, server_cpu, server_cpu)
    environment["LD_LIBRARY_PATH"] = (
        f"{cpu_lib_dir}:{server_cpu.parent}:"
        + environment.get("LD_LIBRARY_PATH", "")
    )
    launch_contract = static_split_launch_contract(
        manifest,
        compiled_policy,
        phone_host=phone_host,
        phone_port=phone_port,
    )
    environment.update(launch_contract.environment)
    process = physical.run_trace.CapturedProcess(
        command, environment, output, "llama1-cpu-phone-ffn"
    )
    started_ns = time.monotonic_ns()
    process.start()
    succeeded = False
    try:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            require(
                process.process is not None
                and process.process.poll() is None,
                "split executor exited during load",
            )
            if endpoint_ready("127.0.0.1", spec.port):
                break
            time.sleep(0.1)
        else:
            raise OverlayError("split executor readiness timeout")
        props = endpoint_json("127.0.0.1", spec.port, "/props", 10)
        require(props.get("model_alias") == spec.model_id, "split alias")
        validate_static_split_ready_log(
            process.stderr_lines, launch_contract
        )
        succeeded = True
        return process, (time.monotonic_ns() - started_ns) / 1e6, props
    finally:
        if not succeeded:
            process.terminate()


def cached_runtime_value(
    snapshot: BackgroundRuntimeSnapshot,
) -> object | None:
    if snapshot.stale or snapshot.error is not None:
        return None
    return snapshot.value


def live_phone_snapshot(
    host: str,
    diagnostic_port: int,
    reserve_bytes: int,
) -> dict[str, object] | None:
    try:
        value = endpoint_json(
            host, diagnostic_port, "/snapshot.json", 1
        )
        captured_epoch_s = value.get("captured_epoch_s")
        available_kib = value.get("mem_available_kib")
        total_kib = value.get("mem_total_kib")
        temperature = value.get("temperature_max_millic")
        android_thermal_status = value.get("android_thermal_status")
        require(
            value.get("schema") == "s42-op15-live-snapshot-v1"
            and type(captured_epoch_s) is int
            and abs(time.time() - captured_epoch_s) <= 5
            and type(available_kib) is int
            and type(total_kib) is int
            and 0 < available_kib <= total_kib
            and type(temperature) is int
            and temperature > 0
            and type(android_thermal_status) is int
            and android_thermal_status >= -1
            and value.get("thermal_state")
                in {"nominal", "hot", "unqualified"}
            and value.get("throttling_state") in {
                "not-observed",
                "android-thermal-throttling",
                "unqualified",
            }
            and (
                (
                    android_thermal_status == 0
                    and value["thermal_state"] == "nominal"
                    and value["throttling_state"] == "not-observed"
                )
                or (
                    android_thermal_status > 0
                    and value["thermal_state"] == "hot"
                    and value["throttling_state"]
                        == "android-thermal-throttling"
                )
                or (
                    android_thermal_status == -1
                    and value["thermal_state"] == "unqualified"
                    and value["throttling_state"] == "unqualified"
                )
            )
            and type(value.get("task_server_alive")) is bool,
            "live phone diagnostic",
        )
        available_bytes = available_kib * 1024
        return {
            **value,
            "available_bytes": available_bytes,
            "capacity_bytes": total_kib * 1024,
            "memory_reserve_met": available_bytes >= reserve_bytes,
            "sample_age_s": time.time() - captured_epoch_s,
            "thermal_qualified": android_thermal_status == 0,
        }
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        http.client.HTTPException,
        OverlayError,
    ):
        return None


def live_phone_power_snapshot(
    host: str, diagnostic_port: int
) -> dict[str, int] | None:
    before_ns = time.monotonic_ns()
    try:
        value = endpoint_json(
            host, diagnostic_port, "/power.json", 1
        )
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        http.client.HTTPException,
        OverlayError,
    ):
        return None
    after_ns = time.monotonic_ns()
    uptime_s = value.get("uptime_s")
    fields = {
        name: value.get(name) for name in (
            "battery_charge_counter_uah",
            "battery_current_ma",
            "battery_voltage_uv",
            "usb_current_ua",
            "usb_voltage_uv",
        )
    }
    try:
        require(
            value.get("schema") == "s42-op15-live-power-v1"
            and type(uptime_s) in {int, float}
            and not isinstance(uptime_s, bool)
            and uptime_s > 0
            and all(type(row) is int for row in fields.values())
            and fields["usb_current_ua"] >= 0
            and fields["usb_voltage_uv"] > 0
            and fields["battery_voltage_uv"] > 0,
            "live phone power diagnostic",
        )
    except OverlayError:
        return None
    usb_power_mw = round(
        fields["usb_current_ua"] * fields["usb_voltage_uv"] / 1_000_000_000
    )
    battery_power_mw = round(
        max(0, fields["battery_current_ma"])
        * fields["battery_voltage_uv"] / 1_000_000
    )
    return {
        **fields,
        "battery_discharge_power_mw": battery_power_mw,
        "host_sample_t_ns": (before_ns + after_ns) // 2,
        "phone_uptime_ns": round(float(uptime_s) * 1_000_000_000),
        "usb_input_power_mw": usb_power_mw,
    }


def wait_for_live_phone_snapshot(
    host: str,
    diagnostic_port: int,
    reserve_bytes: int,
    timeout_s: float,
) -> dict[str, object] | None:
    deadline = time.monotonic() + timeout_s
    while True:
        snapshot = live_phone_snapshot(
            host, diagnostic_port, reserve_bytes
        )
        if snapshot is not None:
            return snapshot
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            return None
        time.sleep(min(0.25, remaining_s))


def phone_memory_capacity(
    phone_live: dict[str, object] | None,
    fallback_total_bytes: int,
    reserve_bytes: int,
) -> DeviceMemoryCapacity:
    if phone_live is None:
        capacity_bytes = fallback_total_bytes
        available_bytes = 0
    else:
        capacity_bytes = int(phone_live["capacity_bytes"])
        available_bytes = int(phone_live["available_bytes"])
    effective_reserve = min(reserve_bytes, available_bytes)
    return DeviceMemoryCapacity(
        "op15-ram",
        capacity_bytes,
        capacity_bytes - available_bytes,
        effective_reserve,
    )


def executor_url(host: str, port: int) -> str:
    authority = f"[{host}]" if ":" in host else host
    return f"http://{authority}:{port}"


def endpoint_completion(
    host: str,
    port: int,
    row: dict[str, Any],
    stream_path: Path,
    on_first: Callable[[int], None],
    timeout_s: float = 3600,
    control_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    endpoint = executor_url(host, port)
    return LlamaCppHttpClient(endpoint_slots).complete(
        endpoint,
        LlamaCppCompletionPayload(
            request_id=str(row.get(
                "event_id",
                f"fp16-overlay-{row['overlay_request_index']}",
            )),
            expected_model_alias=LLAMA1,
            input_tokens=int(row.get(
                "input_tokens", len(row["prompt_tokens"])
            )),
            output_tokens=row["output_tokens"],
            prompt_tokens=tuple(row["prompt_tokens"]),
            seed=row["overlay_request_index"],
            stream_path=stream_path,
            on_first_token=on_first,
            timeout_s=timeout_s,
        ),
        (lambda: None) if control_check is None else control_check,
    )


def protected_work_observation(
    profile: dict[str, Any],
    large_model_policy: str,
    phase: dict[str, Any],
    now_us: int,
    power_observation: dict[str, object] | None = None,
) -> tuple[RuntimeProtectedWorkObservation, dict[str, object]]:
    require(
        profile.get("schema")
            == "s42-fp16-overlay-marginal-system-profile-v1",
        "marginal system profile schema",
    )
    arms = profile.get("arms")
    require(type(arms) is dict, "marginal system profile arms")
    arm = arms.get(large_model_policy)
    require(type(arm) is dict, "marginal system profile arm")
    phase_name = phase.get("phase")
    phases = arm.get("phases")
    require(
        phase_name in LARGE_PHASE_IDS
        and type(phases) is dict
        and type(phases.get(phase_name)) is dict,
        "marginal system phase",
    )
    phase_profile = phases[phase_name]
    measured = (
        arm.get("measured") is True
        and phase_profile.get("measured", True) is True
    )
    receipt: dict[str, object] = {
        "arm": large_model_policy,
        "evidence": arm.get("evidence"),
        "measured": measured,
        "phase": phase_name,
        "power_observation": power_observation,
        "profile_id": profile.get("profile_id"),
    }
    trace_duration_upper_us = arm.get("trace_duration_upper_us")
    require(
        type(trace_duration_upper_us) is int
        and trace_duration_upper_us > 0,
        "marginal trace duration upper bound",
    )
    critical_path_end_us = (
        now_us
        if phase_name == "idle"
        else max(now_us, trace_duration_upper_us)
    )
    phone_phase_power_mw = arm.get("phone_phase_power_mw", 0)
    require(
        type(phone_phase_power_mw) is int and phone_phase_power_mw >= 0,
        "marginal phone phase power",
    )
    measured_phase_power_mw = (
        arm["phase_power_mw"] + phone_phase_power_mw
    )
    power_sample_count = int(arm["sample_count"])
    if (
        type(power_observation) is dict
        and power_observation.get("status") == "MEASURED"
        and type(power_observation.get("total_server_power_mw")) is int
        and type(power_observation.get("sample_count")) is int
        and int(power_observation["sample_count"]) > 0
    ):
        measured_phase_power_mw = (
            power_observation["total_server_power_mw"]
            + phone_phase_power_mw
        )
        power_sample_count = min(
            power_sample_count,
            int(power_observation["sample_count"]),
        )
    observation = RuntimeProtectedWorkObservation(
        observation_id=(
            f"{profile['profile_id']}:{large_model_policy}:{phase_name}"
        ),
        critical_path_end_us=critical_path_end_us,
        phase_power_mw=int(measured_phase_power_mw),
        stranded_idle_power_mw=int(arm["gpu_idle_power_mw"]),
        causal_tail_power_mw=int(arm["causal_tail_power_mw"]),
        sample_count=power_sample_count,
        measured=measured,
    )
    receipt.update({
        "context_id": observation.observation_id,
        "critical_path_end_us": critical_path_end_us,
        "phase_power_mw": observation.phase_power_mw,
        "phone_phase_power_mw": phone_phase_power_mw,
        "status": "MEASURED" if measured else "UNAVAILABLE_UNMEASURED",
    })
    return observation, receipt


def marginal_system_context(
    profile: dict[str, Any],
    large_model_policy: str,
    phase: dict[str, Any],
    now_us: int,
    power_observation: dict[str, object] | None = None,
) -> tuple[MarginalSystemCostContext | None, dict[str, object]]:
    observation, receipt = protected_work_observation(
        profile,
        large_model_policy,
        phase,
        now_us,
        power_observation,
    )
    if not observation.measured:
        return None, receipt
    arm = profile["arms"][large_model_policy]
    phase_profile = arm["phases"][phase["phase"]]
    context = MarginalSystemCostContext(
        context_id=(
            observation.observation_id
        ),
        critical_path_end_us=observation.critical_path_end_us,
        phase_power_mw=observation.phase_power_mw,
        gpu_idle_power_mw=observation.stranded_idle_power_mw,
        causal_tail_power_mw=observation.causal_tail_power_mw,
        route_cpu_interference_ppm=dict(
            phase_profile["route_cpu_interference_ppm"]
        ),
        lower_error_ppm=int(arm["lower_error_ppm"]),
        upper_error_ppm=int(arm["upper_error_ppm"]),
        sample_count=observation.sample_count,
        measured=True,
    )
    context.validate()
    return context, receipt


def request(
    row: dict[str, Any],
) -> Request:
    return Request(
        request_id=f"fp16-overlay-{row['overlay_request_index']}",
        workload_id=WORKLOAD_ID,
        arrival_us=row["arrival_us"],
        deadline_us=row["arrival_us"] + row["slo_us"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        quality_requirement="bounded_numeric",
    )


def wait_for_trace_start(base_output: Path, timeout_s: float) -> int:
    events = base_output / "events.jsonl"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if events.exists():
            for line in events.read_text(encoding="ascii").splitlines():
                value = json.loads(line)
                if value.get("kind") == "trace_start":
                    start = value.get("t_ns")
                    require(type(start) is int and start > 0, "trace start")
                    return start
        failure = base_output / "FAILURE.json"
        if failure.exists():
            raise OverlayError("base replay failed before trace start")
        time.sleep(0.05)
    raise OverlayError("base trace start timeout")


def wait_for_phase_start(
    base_output: Path,
    phase: str,
    timeout_s: float,
) -> int:
    require(phase in LARGE_PHASE_IDS, "target large-model phase")
    events = base_output / "events.jsonl"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if events.exists():
            for raw in events.read_bytes().splitlines(keepends=True):
                if not raw.endswith(b"\n"):
                    continue
                value = json.loads(raw)
                if (
                    type(value) is dict
                    and value.get("kind") == "large_model_phase"
                    and value.get("phase") == phase
                ):
                    started_ns = value.get("t_ns")
                    require(
                        type(started_ns) is int and started_ns > 0,
                        "large-model phase timestamp",
                    )
                    return started_ns
        failure = base_output / "FAILURE.json"
        if failure.exists():
            raise OverlayError("base replay failed before target phase")
        time.sleep(0.05)
    raise OverlayError(f"large-model phase timeout: {phase}")


def materialize_arrival(
    row: dict[str, Any],
    base_output: Path,
    paid_start_ns: int,
) -> dict[str, Any]:
    target_phase = row.get("target_large_phase")
    if target_phase is None:
        target_ns = paid_start_ns + row["arrival_us"] * 1000
    else:
        phase_offset_us = row.get("phase_offset_us")
        require(
            target_phase in LARGE_PHASE_IDS
            and type(phase_offset_us) is int
            and phase_offset_us >= 0,
            "phase-anchored arrival",
        )
        target_ns = (
            wait_for_phase_start(base_output, target_phase, 7200)
            + phase_offset_us * 1000
        )
    sleep_until_ns(target_ns)
    effective = dict(row)
    effective["declared_arrival_us"] = row["arrival_us"]
    effective["arrival_us"] = max(
        0, (target_ns - paid_start_ns) // 1000
    )
    return effective


def wait_for_base_result(base_output: Path, timeout_s: float) -> dict[str, Any]:
    result = base_output / "RESULT.json"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if result.exists():
            return load_object(result)
        failure = base_output / "FAILURE.json"
        if failure.exists():
            raise OverlayError("base replay failed")
        time.sleep(0.2)
    raise OverlayError("base result timeout")


def timing_stats(rows: list[dict[str, int]], key: str) -> dict[str, float]:
    values = sorted(row[key] / 1000 for row in rows)
    require(values, f"timing samples for {key}")

    def percentile(fraction: float) -> float:
        return values[round((len(values) - 1) * fraction)]

    return {
        "max_us": values[-1],
        "mean_us": sum(values) / len(values),
        "p50_us": percentile(0.50),
        "p95_us": percentile(0.95),
        "samples": len(values),
        "sum_us": sum(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--resident-release-file", type=Path, required=True)
    parser.add_argument("--server-cuda", type=Path, required=True)
    parser.add_argument("--server-cpu", type=Path, required=True)
    parser.add_argument("--cuda-lib-dir", type=Path, required=True)
    parser.add_argument("--cpu-lib-dir", type=Path, required=True)
    parser.add_argument("--llama1-model", type=Path, required=True)
    parser.add_argument("--llama1-cpus", default="20-23")
    parser.add_argument("--llama1-port", type=int, default=18484)
    parser.add_argument("--llama1-cuda-port", type=int, default=18485)
    parser.add_argument("--llama1-split-port", type=int, default=18486)
    parser.add_argument("--phone-host", required=True)
    parser.add_argument("--phone-port", type=int, default=18382)
    parser.add_argument("--phone-ffn-port", type=int, default=18384)
    parser.add_argument("--phone-diagnostic-port", type=int, required=True)
    parser.add_argument("--phone-memory-total-bytes", type=int, required=True)
    parser.add_argument(
        "--phone-memory-available-bytes", type=int, required=True
    )
    parser.add_argument("--phone-memory-reserve-bytes", type=int, required=True)
    parser.add_argument("--phone-model-sha256", required=True)
    parser.add_argument("--ffn-manifest", type=Path)
    parser.add_argument("--ffn-compiled-policy", type=Path)
    parser.add_argument("--ffn-prewarm-timeout-s", type=float, default=120)
    parser.add_argument(
        "--forced-static-route",
        choices=("desktop-cpu", SPLIT_ROUTE),
        default="desktop-cpu",
    )
    parser.add_argument("--runtime-profile", type=Path, required=True)
    parser.add_argument("--automated-catalog", type=Path)
    parser.add_argument("--phone-battery-ppm", type=int, default=1_000_000)
    parser.add_argument(
        "--marginal-system-profile", type=Path, required=True
    )
    parser.add_argument(
        "--large-model-policy",
        choices=("cpu-overflow", "op15-assistance"),
        required=True,
    )
    parser.add_argument(
        "--requested-large-model-policy",
        choices=("cpu-overflow", "op15-assistance", "runtime-auto"),
    )
    parser.add_argument(
        "--small-model-policy",
        choices=("static-cpu", "runtime-scheduler"),
        required=True,
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args()
    if args.requested_large_model_policy is None:
        args.requested_large_model_policy = args.large_model_policy

    require(args.execute and args.confirm == CONFIRMATION, "confirmation")
    require(
        callable(getattr(physical.DynamicSampler, "latest_gpu_snapshot", None)),
        "physical runner lacks cached GPU snapshot API",
    )
    require(args.ffn_prewarm_timeout_s > 0, "FFN prewarm timeout")
    require(
        (args.ffn_manifest is None) == (args.ffn_compiled_policy is None),
        "paired FFN split inputs",
    )
    require(
        args.forced_static_route == "desktop-cpu",
        "static split execution is only available through scheduler selection",
    )
    require(
        args.output.is_absolute()
        and not args.output.exists()
        and args.ready_file.is_absolute()
        and not args.ready_file.exists()
        and args.resident_release_file.is_absolute()
        and not args.resident_release_file.exists(),
        "new absolute output paths",
    )
    manifest = load_object(args.manifest)
    rows = load_rows(args.requests)
    overlay_manifest = manifest.get("overlay_trace", {})
    overlay_count = overlay_manifest.get("record_count")
    require(
        manifest.get("schema") in MANIFEST_SCHEMAS
        and type(overlay_count) is int
        and overlay_count > 0
        and len(rows) == overlay_count
        and all(row.get("schema") in TRACE_SCHEMAS for row in rows)
        and digest(args.requests)
            == overlay_manifest.get("sha256")
        and [row["overlay_request_index"] for row in rows]
            == list(range(overlay_count)),
        "overlay identity",
    )
    overlay_input_tokens = overlay_manifest.get("input_tokens")
    overlay_output_tokens = overlay_manifest.get("output_tokens")
    combined_work = manifest.get("combined_work", {})
    require(
        type(overlay_input_tokens) is int
        and type(overlay_output_tokens) is int
        and overlay_input_tokens == sum(row["input_tokens"] for row in rows)
        and overlay_output_tokens == sum(row["output_tokens"] for row in rows)
        and combined_work == {
            "input_tokens": 33_843 + overlay_input_tokens,
            "output_tokens": 11_605 + overlay_output_tokens,
            "record_count": 74 + overlay_count,
        },
        "combined work identity",
    )
    inventory = manifest["model_inventory"][LLAMA1]
    require(
        args.llama1_model.is_file()
        and args.llama1_model.stat().st_size == inventory["artifact_bytes"]
        and digest(args.llama1_model) == inventory["artifact_sha256"]
        and args.phone_model_sha256 == inventory["artifact_sha256"],
        "Llama model identity",
    )
    require(
        args.phone_memory_available_bytes >= args.phone_memory_reserve_bytes
        and args.phone_memory_total_bytes
            >= args.phone_memory_available_bytes,
        "phone combined residency reserve",
    )
    for path in (
        args.server_cuda,
        args.server_cpu,
        args.cuda_lib_dir,
        args.cpu_lib_dir,
        args.runtime_profile,
        args.marginal_system_profile,
        *(
            ()
            if args.ffn_manifest is None
            else (args.ffn_manifest, args.ffn_compiled_policy)
        ),
    ):
        require(path.exists(), f"missing runtime dependency: {path}")
    require(
        args.automated_catalog is not None
        and args.automated_catalog.exists(),
        "every execution mode requires an automated capability catalog",
    )
    require(
        0 <= args.phone_battery_ppm <= 1_000_000,
        "phone battery observation",
    )

    ffn_manifest = (
        None
        if args.ffn_manifest is None
        else verified_record(args.ffn_manifest, FFN_MANIFEST_SCHEMA)
    )
    ffn_compiled_policy = (
        None
        if args.ffn_compiled_policy is None
        else verified_record(args.ffn_compiled_policy, FFN_POLICY_SCHEMA)
    )
    if ffn_manifest is not None and ffn_compiled_policy is not None:
        require(
            ffn_manifest["model"]["sha256"] == inventory["artifact_sha256"]
            and ffn_manifest["model"]["size_bytes"]
                == inventory["artifact_bytes"]
            and ffn_compiled_policy["evidence"][
                "manifest_record_sha256"
            ] == ffn_manifest["record_sha256"],
            "FFN split model and policy identity",
        )
    marginal_profile = load_object(args.marginal_system_profile)
    require(
        marginal_profile.get("schema")
            == "s42-fp16-overlay-marginal-system-profile-v1",
        "marginal system profile",
    )
    automated_catalog = RuntimeCapabilityCatalog.from_json(
        load_object(args.automated_catalog)
    )
    split_route_profiled = bool(automated_catalog.composite_executors)
    scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
    scheduler.register_runtime_capabilities(automated_catalog)
    automated_manifest = scheduler.register_gguf_model(
        LLAMA1, args.llama1_model
    )
    require(
        automated_manifest.artifact_sha256
            == "sha256:" + inventory["artifact_sha256"]
        and automated_manifest.artifact_bytes
            == inventory["artifact_bytes"],
        "automated GGUF manifest identity",
    )
    snapshot_builder = RuntimeSnapshotBuilder(
        automated_catalog, automated_manifest
    )
    activity_tracker = RuntimeActivityTracker(automated_catalog)
    automated_split_capability = None
    if automated_catalog.composite_executors:
        require(
            len(automated_catalog.composite_executors) == 1
            and ffn_manifest is not None
            and ffn_compiled_policy is not None,
            "physical composite adapter coverage",
        )
        automated_split_capability = (
            automated_catalog.composite_executors[0]
        )
    split_executor_required = (
        automated_split_capability is not None
    )
    require(
        not split_executor_required
        or (ffn_manifest is not None and ffn_compiled_policy is not None),
        "split executor lacks physical artifacts",
    )
    scheduler_lock = threading.Lock()

    args.output.mkdir(parents=True)
    events = physical.run_trace.EventWriter(args.output / "events.jsonl")
    cpu_spec = physical.ModelSpec(
        LLAMA1,
        args.llama1_model,
        "cpu",
        args.llama1_port,
        4,
        8192,
        2048,
        512,
        threads=4,
        cpus=args.llama1_cpus,
    )
    cuda_spec = physical.ModelSpec(
        LLAMA1,
        args.llama1_model,
        "cuda",
        args.llama1_cuda_port,
        1,
        4096,
        1024,
        256,
    )
    split_spec = physical.ModelSpec(
        LLAMA1,
        args.llama1_model,
        "cpu",
        args.llama1_split_port,
        1,
        4096,
        1024,
        512,
        threads=4,
        cpus=args.llama1_cpus,
    )
    cpu_process: physical.run_trace.CapturedProcess | None = None
    cpu_load_ms: float | None = None
    cpu_props: dict[str, Any] | None = None
    cpu_warm_ms: float | None = None
    split_process: physical.run_trace.CapturedProcess | None = None
    split_load_ms: float | None = None
    split_props: dict[str, Any] | None = None
    split_warm_ms: float | None = None
    split_warm_shape: dict[str, int] | None = None
    cuda_process: physical.run_trace.CapturedProcess | None = None
    cuda_load_ms: float | None = None
    cuda_props: dict[str, Any] | None = None
    cuda_loading = False
    cuda_state_lock = threading.Lock()
    cuda_state_condition = threading.Condition(cuda_state_lock)
    sampler: physical.DynamicSampler | None = None
    phone_power_sampler: PolledPhonePowerSampler | None = None
    runtime_monitor: BackgroundRuntimeMonitor | None = None
    runtime_monitor_final: dict[str, object] = {}
    phase_source_id = "large-model-phone-transport"
    phase_history: list[dict[str, object]] = []
    phase_monitor_stop = threading.Event()
    phase_monitor_thread: threading.Thread | None = None
    phase_monitor_error: list[str] = []
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    result_lock = threading.Lock()
    estimates_by_index: dict[int, dict[str, object]] = {}
    decisions_by_index: dict[int, Decision] = {}
    online_receipts_by_index: dict[int, dict[str, object]] = {}
    decision_attempts_by_index: dict[int, list[dict[str, object]]] = {}
    activations_by_index: dict[int, list[dict[str, int | str]]] = {}
    dispatch_receipts_by_index: dict[int, list[dict[str, object]]] = {}
    renewals_by_index: dict[int, list[dict[str, object]]] = {}
    releases_by_index: dict[int, dict[str, object]] = {}
    recoveries_by_index: dict[int, dict[str, object]] = {}
    runtime_context_by_index: dict[int, dict[str, object]] = {}
    adapter_results_by_index: dict[int, object] = {}
    overhead_rows: list[dict[str, int]] = []
    recovery_overhead_rows: list[dict[str, int]] = []
    resident_release_status = "CONTROLLER_ABORT"

    try:
        warm = dict(min(rows, key=lambda row: row["input_tokens"]))
        warm["output_tokens"] = 2
        require(endpoint_ready(args.phone_host, args.phone_port), "phone health")
        phone_props = endpoint_json(
            args.phone_host, args.phone_port, "/props", 10
        )
        require(phone_props.get("model_alias") == LLAMA1, "phone alias")
        phone_live_initial = wait_for_live_phone_snapshot(
            args.phone_host,
            args.phone_diagnostic_port,
            args.phone_memory_reserve_bytes,
            15,
        )
        require(phone_live_initial is not None, "phone live diagnostic")
        started_ns = time.monotonic_ns()
        endpoint_completion(
            args.phone_host,
            args.phone_port,
            warm,
            args.output / "warm-llama1-phone.raw",
            lambda _: None,
        )
        phone_warm_ms = (time.monotonic_ns() - started_ns) / 1e6
        contention_sampler = HostContentionSampler()

        sampler = physical.DynamicSampler(args.output)
        sampler.start()
        phone_power_sampler = PolledPhonePowerSampler(
            lambda: live_phone_power_snapshot(
                args.phone_host, args.phone_diagnostic_port
            )
        )
        phone_power_sampler.start()
        time.sleep(0.5)
        args.ready_file.write_bytes(canonical({
            "cpu_executor_state": "deferred_until_paid_start",
            "phone_props": phone_props,
            "phone_live_snapshot": phone_live_initial,
            "phone_warm_ms": phone_warm_ms,
            "schema": "s42-full-fp16-llama1b-controller-ready-v1",
            "status": "READY",
        }))

        paid_start_ns = wait_for_trace_start(args.base_output, 900)
        cpu_load_started_ns = time.monotonic_ns()
        cpu_process, cpu_load_ms, cpu_props = physical.start_server(
            cpu_spec,
            args.server_cuda,
            args.server_cpu,
            args.cuda_lib_dir,
            args.cpu_lib_dir,
            args.output,
            "llama1-cpu",
        )
        started_ns = time.monotonic_ns()
        endpoint_completion(
            "127.0.0.1",
            args.llama1_port,
            warm,
            args.output / "warm-llama1-cpu.raw",
            lambda _: None,
        )
        cpu_warm_ms = (time.monotonic_ns() - started_ns) / 1e6
        sampler.set_pid("llama1-cpu", cpu_process.pid)
        if split_executor_required:
            assert ffn_manifest is not None
            assert ffn_compiled_policy is not None
            split_process, split_load_ms, split_props = start_split_server(
                split_spec,
                args.server_cpu,
                args.cpu_lib_dir,
                args.output,
                args.phone_host,
                args.phone_ffn_port,
                ffn_manifest,
                ffn_compiled_policy,
            )
            assert automated_split_capability is not None
            split_warm_receipt = CanonicalStaticSplitPrewarmer(
                LlamaCppHttpClient(endpoint_slots)
            ).prewarm(
                automated_split_capability,
                artifact_sha256=(
                    "sha256:" + inventory["artifact_sha256"]
                ),
                manifest=ffn_manifest,
                policy=ffn_compiled_policy,
                rows=rows,
                event_id="s42:llama1b-split-accelerator-prewarm",
                request_index=2_147_000_001,
                expected_model_alias=LLAMA1,
                stream_path=(
                    args.output / "warm-llama1-cpu-phone-ffn.raw"
                ),
                timeout_s=args.ffn_prewarm_timeout_s,
            )
            split_warm_ms = split_warm_receipt.duration_ms
            sampler.set_pid("llama1-cpu-phone-ffn", split_process.pid)
            events.write({
                "kind": "split_executor_ready",
                "load_ms": split_load_ms,
                "policy_sha256": ffn_compiled_policy["record_sha256"],
                "prewarm": split_warm_receipt.to_json(),
                "schema": EVENT_SCHEMA,
                "shape": {
                    "input_tokens": split_warm_receipt.input_tokens,
                    "phone_columns": split_warm_receipt.phone_columns,
                },
                "t_ns": time.monotonic_ns(),
                "warm_ms": split_warm_ms,
            })
        monitor_probes: dict[str, Callable[[], object]] = {
            "desktop-cpu": lambda: executor_runtime_snapshot(
                "127.0.0.1", args.llama1_port
            ),
            "desktop-cuda": lambda: executor_runtime_snapshot(
                "127.0.0.1", args.llama1_cuda_port
            ),
            "phone-adreno": lambda: executor_runtime_snapshot(
                args.phone_host, args.phone_port
            ),
            "phone-live": lambda: live_phone_snapshot(
                args.phone_host,
                args.phone_diagnostic_port,
                args.phone_memory_reserve_bytes,
            ),
        }
        if split_process is not None:
            monitor_probes[SPLIT_ROUTE] = lambda: executor_runtime_snapshot(
                "127.0.0.1", args.llama1_split_port
            )
        runtime_monitor = BackgroundRuntimeMonitor(
            monitor_probes,
            refresh_interval_s=1.0,
            stale_after_s=4.0,
        )
        runtime_monitor.start()
        require(
            runtime_monitor.wait_until_populated(
                tuple(
                    sorted(monitor_probes)
                ),
                10,
            ),
            "background runtime snapshots",
        )
        events.write({
            "kind": "cpu_executor_ready",
            "load_started_ns": cpu_load_started_ns,
            "load_ms": cpu_load_ms,
            "schema": EVENT_SCHEMA,
            "t_ns": time.monotonic_ns(),
            "warm_ms": cpu_warm_ms,
        })
        events.write({
            "kind": "overlay_trace_start",
            "large_model_policy": args.large_model_policy,
            "small_model_policy": args.small_model_policy,
            "schema": EVENT_SCHEMA,
            "t_ns": paid_start_ns,
        })
        def report_large_phase(
            phase: dict[str, Any], now_us: int
        ) -> dict[str, object]:
            owner = phase.get("functionfs_owner")
            observation = RuntimePhaseObservation(
                source_id=phase_source_id,
                phase_id=phase["phase"],
                owner_id=owner,
                phase_start_us=max(
                    0,
                    (
                        phase["phase_start_ns"] - paid_start_ns
                    ) // 1000,
                ),
                occupied_resource_ids=(
                    ()
                    if owner is None
                    else (
                        "desktop-usb-root",
                        "op15-adreno",
                        "op15-functionfs",
                        "op15-htp",
                    )
                ),
            )
            state = scheduler.observe_runtime_phase(
                observation,
                observed_at_us=now_us,
            )
            return {
                "functionfs_owner": state["owner_id"],
                "phase": state["phase_id"],
                "release_estimate_us": state["release_estimate_us"],
                "resource_ids": state["resource_ids"],
                "status": state["status"],
            }

        def prepare_physical_transition(
            command: object,
            _payload: object,
            control_check: Callable[[], None],
        ) -> bool:
            nonlocal cuda_process, cuda_load_ms, cuda_props, cuda_loading
            participant = getattr(command, "participant", None)
            if (
                participant is None
                or participant.endpoint != executor_url(
                    "127.0.0.1", args.llama1_cuda_port
                )
            ):
                control_check()
                return False
            control_check()
            with cuda_state_condition:
                while cuda_process is None and cuda_loading:
                    control_check()
                    cuda_state_condition.wait(timeout=0.1)
                if cuda_process is not None:
                    return True
                cuda_loading = True
            succeeded = False
            process = None
            try:
                gpu = physical.run_trace.gpu_snapshot()
                require(
                    gpu["memory_free_bytes"]
                    >= automated_manifest.artifact_bytes + 512 * 1024**2,
                    "selected CUDA transition lacks physical VRAM",
                )
                process, load_ms, props = physical.start_server(
                    cuda_spec,
                    args.server_cuda,
                    args.server_cpu,
                    args.cuda_lib_dir,
                    args.cpu_lib_dir,
                    args.output,
                    "llama1-cuda",
                )
                with cuda_state_condition:
                    cuda_process = process
                    cuda_load_ms = load_ms
                    cuda_props = props
                assert sampler is not None
                sampler.set_pid("llama1-cuda", process.pid)
                events.write({
                    "kind": "cuda_executor_ready",
                    "load_ms": load_ms,
                    "schema": EVENT_SCHEMA,
                    "t_ns": time.monotonic_ns(),
                })
                assert runtime_monitor is not None
                runtime_monitor.request_refresh()
                succeeded = True
                return True
            finally:
                with cuda_state_condition:
                    cuda_loading = False
                    cuda_state_condition.notify_all()
                if not succeeded and process is not None:
                    process.terminate()

        def monitor_large_resources() -> None:
            previous_snapshot = None
            try:
                while not phase_monitor_stop.is_set():
                    now_ns = time.monotonic_ns()
                    phase = live_large_model_phase(args.base_output, now_ns)
                    now_us = max(0, (now_ns - paid_start_ns) // 1000)
                    with scheduler_lock:
                        snapshot = report_large_phase(phase, now_us)
                    if snapshot != previous_snapshot:
                        events.write({
                            "kind": "large_resource_ownership",
                            "ownership": snapshot,
                            "schema": EVENT_SCHEMA,
                            "t_ns": time.monotonic_ns(),
                        })
                        previous_snapshot = snapshot
                    phase_monitor_stop.wait(0.5)
            except BaseException as error:
                phase_monitor_error.append(
                    f"{type(error).__name__}: {error}"
                )
                phase_monitor_stop.set()

        phase_monitor_thread = threading.Thread(
            target=monitor_large_resources,
            name="fp16-large-resource-monitor",
            daemon=True,
        )
        phase_monitor_thread.start()

        def capture_runtime_context(
            row: dict[str, Any],
        ) -> dict[str, object]:
            assert sampler is not None
            captured_ns = time.monotonic_ns()
            phase = live_large_model_phase(args.base_output, captured_ns)
            captured_us = max(
                0, (captured_ns - paid_start_ns) // 1000
            )
            expected_arm = (
                "control"
                if args.large_model_policy == "cpu-overflow"
                else "op15"
            )
            require(
                phase["large_model_arm"] == expected_arm,
                "live large-model arm",
            )
            host = contention_sampler.sample()
            activity = activity_tracker.snapshot()
            active_small_cpu = activity.active_by_device_kind.get("cpu", 0)
            active_phone = activity.active_by_device_kind.get("phone", 0)
            active_cuda = activity.active_by_device_kind.get("gpu", 0)
            with scheduler_lock:
                transport_ownership = report_large_phase(
                    phase, captured_us
                )
            active_large = phase["active_large_requests"]
            large_cpu_capacity = {
                "gemma": 8,
                "idle": 0,
                "qwen": 4,
                "switching": 8,
            }[phase["phase"]]
            active_large_cpu_slots = min(
                active_large, large_cpu_capacity
            )
            active_small_cpu_slots = min(
                active_small_cpu, cpu_spec.parallel + split_spec.parallel
            )
            actual_batch_size = min(
                row["input_tokens"], cpu_spec.ubatch_size
            )
            memory_bandwidth_pressure = host[
                "memory_stall_avg10_basis_points"
            ]
            features = {
                "active_cpu_requests": (
                    active_large_cpu_slots + active_small_cpu_slots
                ),
                "active_cpu_slots": (
                    active_large_cpu_slots + active_small_cpu_slots
                ),
                "actual_batch_size": actual_batch_size,
                "cpu_utilization_pct": host["cpu_utilization_pct"],
                "large_model_op15": int(
                    args.large_model_policy == "op15-assistance"
                ),
                "large_phase_id": phase["phase_id"],
                "memory_stall_avg10_basis_points": memory_bandwidth_pressure,
                "memory_bandwidth_pressure_basis_points": (
                    memory_bandwidth_pressure
                ),
                "prompt_ubatch_count": (
                    row["input_tokens"] + cpu_spec.ubatch_size - 1
                ) // cpu_spec.ubatch_size,
            }
            features["contention_class_id"] = (
                (
                    4
                    if features["large_phase_id"] == 0
                    else features["large_phase_id"]
                )
                + 4 * features["large_model_op15"]
            )
            return {
                "active_phone_requests": active_phone,
                "active_cuda_requests": active_cuda,
                "captured_ns": captured_ns,
                "cost_features": features,
                "host_contention_observation": {
                    **host,
                    "active_large_cpu_slots": active_large_cpu_slots,
                    "active_small_cpu_slots": active_small_cpu_slots,
                },
                "large_model": phase,
                "memory_bandwidth_observation": {
                    "hardware_counter_available": False,
                    "proxy_feature": (
                        "memory_bandwidth_pressure_basis_points"
                    ),
                    "proxy_kind": "Linux_memory_PSI_some_avg10",
                    "reason": (
                        "host perf_event_paranoid blocks uncore IMC "
                        "bandwidth counters"
                    ),
                },
                "overlay_request_index": row["overlay_request_index"],
                "phase_power_observation": phase_power_observation(sampler),
                "transport_ownership": transport_ownership,
            }

        def schedule_runtime_attempt(
            row: dict[str, Any],
            attempt_kind: str,
            observation_only: bool = False,
        ) -> dict[str, object]:
            total_started_ns = time.perf_counter_ns()
            context_started_ns = total_started_ns
            runtime_context = capture_runtime_context(row)
            context_ended_ns = time.perf_counter_ns()
            probe_started_ns = context_ended_ns
            assert runtime_monitor is not None
            cached_names = [
                "desktop-cpu",
                "desktop-cuda",
                "phone-adreno",
                "phone-live",
            ]
            if split_process is not None:
                cached_names.append(SPLIT_ROUTE)
            cached = {
                name: runtime_monitor.snapshot(name)
                for name in cached_names
            }
            cpu_value = cached_runtime_value(cached["desktop-cpu"])
            cpu_executor = (
                cpu_value
                if type(cpu_value) is dict
                else {
                    "free_slots": 0,
                    "health": "stale_or_unavailable",
                    "processing_task_ids": [],
                    "slots_probe": "background_cache_unavailable",
                    "total_slots": 0,
                }
            )
            with cuda_state_lock:
                cuda_process_ready = cuda_process is not None
            cuda_value = cached_runtime_value(cached["desktop-cuda"])
            cuda_executor = (
                cuda_value
                if cuda_process_ready and type(cuda_value) is dict
                else {
                    "free_slots": 0,
                    "health": (
                        "stale_or_unavailable"
                        if cuda_process_ready
                        else "not_started"
                    ),
                    "processing_task_ids": [],
                    "slots_probe": "background_cache_unavailable",
                    "total_slots": 0,
                    "transition_available": not cuda_process_ready,
                }
            )
            phone_value = cached_runtime_value(cached["phone-adreno"])
            phone_executor = (
                phone_value
                if type(phone_value) is dict
                else {
                    "free_slots": 0,
                    "health": "stale_or_unavailable",
                    "processing_task_ids": [],
                    "slots_probe": "background_cache_unavailable",
                    "total_slots": 0,
                }
            )
            split_value = (
                None
                if split_process is None
                else cached_runtime_value(cached[SPLIT_ROUTE])
            )
            split_executor = (
                split_value
                if type(split_value) is dict
                else {
                    "free_slots": 0,
                    "health": "not_started",
                    "processing_task_ids": [],
                    "slots_probe": "background_cache_unavailable",
                    "total_slots": 0,
                }
            )
            phone_live_value = cached_runtime_value(cached["phone-live"])
            phone_live = (
                dict(phone_live_value)
                if type(phone_live_value) is dict
                else None
            )
            if phone_live is not None:
                cache_age_s = max(
                    0.0,
                    (
                        time.monotonic_ns()
                        - cached["phone-live"].captured_at_ns
                    ) / 1e9,
                )
                phone_live["controller_cache_age_s"] = cache_age_s
                phone_live["effective_sample_age_s"] = (
                    float(phone_live["sample_age_s"]) + cache_age_s
                )
            probe_ended_ns = time.perf_counter_ns()
            snapshot_started_ns = probe_ended_ns
            host_total, host_available = meminfo(
                Path("/proc/meminfo").read_text(encoding="ascii")
            )
            gpu = sampler.latest_gpu_snapshot(2_000_000_000)
            now_us = max(
                row["arrival_us"],
                (time.monotonic_ns() - paid_start_ns) // 1000,
            )
            gpu_captured_at_us = max(
                0, (gpu["sample_t_ns"] - paid_start_ns) // 1000
            )
            gpu_sample_age_us = now_us - gpu_captured_at_us
            require(
                0 <= gpu_sample_age_us <= 2_000_000,
                "GPU sampler snapshot age",
            )
            runtime_context["gpu_capacity_sample"] = {
                "age_us": gpu_sample_age_us,
                "captured_at_us": gpu_captured_at_us,
                "sample_t_ns": gpu["sample_t_ns"],
            }
            snapshot = RuntimePlacementSnapshot(
                snapshot_id=(
                    f"fp16-overlay-{row['overlay_request_index']}-"
                    f"{attempt_kind}-{gpu_captured_at_us}"
                ),
                captured_at_us=gpu_captured_at_us,
                valid_until_us=gpu_captured_at_us + 2_500_000,
                capacities={
                    "cuda0-vram": DeviceMemoryCapacity(
                        "cuda0-vram",
                        gpu["memory_total_bytes"],
                        gpu["memory_used_bytes"],
                        min(512 * 1024**2, gpu["memory_free_bytes"]),
                    ),
                    "host-ram": DeviceMemoryCapacity(
                        "host-ram",
                        host_total,
                        host_total - host_available,
                        min(2 * GIB, host_available),
                    ),
                    "op15-ram": phone_memory_capacity(
                        phone_live,
                        args.phone_memory_total_bytes,
                        args.phone_memory_reserve_bytes,
                    ),
                },
            )
            cpu_endpoint = executor_url("127.0.0.1", args.llama1_port)
            cuda_endpoint = executor_url(
                "127.0.0.1", args.llama1_cuda_port
            )
            phone_endpoint = executor_url(args.phone_host, args.phone_port)
            raw_by_endpoint = {
                cpu_endpoint: cpu_executor,
                cuda_endpoint: cuda_executor,
                phone_endpoint: phone_executor,
            }
            if automated_split_capability is not None:
                raw_by_endpoint[
                    automated_split_capability.endpoint
                ] = split_executor
            endpoint_samples = {
                endpoint: EndpointRuntimeSample.from_mapping(raw)
                for endpoint, raw in raw_by_endpoint.items()
            }
            phone_telemetry = {
                device_id: DeviceRuntimeTelemetry(
                    temperature_millic=(
                        1_000_000
                        if phone_live is None
                        else int(phone_live["temperature_max_millic"])
                    ),
                    battery_ppm=args.phone_battery_ppm,
                )
                for device_id, device in (
                    automated_catalog.placement_profile.devices.items()
                )
                if device.kind == "phone"
            }
            resident_endpoints = {cpu_endpoint, phone_endpoint}
            if cuda_process_ready:
                resident_endpoints.add(cuda_endpoint)
            automated_snapshot = snapshot_builder.build(
                snapshot_id=(
                    f"fp16-overlay-system-"
                    f"{row['overlay_request_index']}-{attempt_kind}-"
                    f"{now_us}"
                ),
                captured_at_us=now_us,
                valid_until_us=snapshot.valid_until_us,
                memory=snapshot,
                endpoint_samples=endpoint_samples,
                device_telemetry=phone_telemetry,
                resident_endpoints=tuple(sorted(resident_endpoints)),
                cost_features=runtime_context["cost_features"],
            )
            snapshot_ended_ns = time.perf_counter_ns()
            cost_features = runtime_context["cost_features"]
            require(type(cost_features) is dict, "runtime cost features")
            runtime_request = request(row)
            protected_work, _ = protected_work_observation(
                marginal_profile,
                args.large_model_policy,
                runtime_context["large_model"],
                now_us,
                runtime_context["phase_power_observation"],
            )
            if automated_snapshot is not None:
                automated_snapshot = replace(
                    automated_snapshot,
                    protected_work=protected_work,
                )
            _, marginal_receipt = marginal_system_context(
                marginal_profile,
                args.large_model_policy,
                runtime_context["large_model"],
                now_us,
                runtime_context["phase_power_observation"],
            )
            if observation_only:
                require(
                    automated_snapshot is not None,
                    "automated runtime observation",
                )
                return {
                    "runtime_context": runtime_context,
                    "snapshot": automated_snapshot,
                }
            require(attempt_kind == "arrival", "runner schedules arrivals only")
            with scheduler_lock:
                core_started_ns = time.perf_counter_ns()
                ticket = scheduler.submit_automated_request(
                    runtime_request,
                    LLAMA1,
                    automated_snapshot,
                    observed_at_us=now_us,
                    selection_mode=(
                        "desktop-baseline"
                        if args.small_model_policy == "static-cpu"
                        else "energy-aware"
                    ),
                )
                core_ended_ns = time.perf_counter_ns()
            estimate_set = ticket.cost_estimates
            decision = ticket.decision
            online_receipt = (
                None
                if ticket.online_placement_receipt is None
                else ticket.online_placement_receipt.to_json()
            )
            activation = [
                dict(receipt) for receipt in ticket.activation_receipts()
            ]
            cancelled = ()
            total_ended_ns = time.perf_counter_ns()
            overhead = {
                "context_ns": context_ended_ns - context_started_ns,
                "estimate_ns": 0,
                "policy_ns": core_ended_ns - core_started_ns,
                "probe_ns": probe_ended_ns - probe_started_ns,
                "snapshot_ns": snapshot_ended_ns - snapshot_started_ns,
                "total_controller_ns": total_ended_ns - total_started_ns,
                "total_core_ns": core_ended_ns - core_started_ns,
            }
            index = row["overlay_request_index"]
            attempt = {
                "activation": activation,
                "attempt_kind": attempt_kind,
                "cancelled_tokens": list(cancelled),
                "cost_estimates": estimate_set.to_json(),
                "decision": decision_to_json(decision),
                "dispatchable": ticket.dispatch_state == "ACQUIRED",
                "executor_availability": {
                    "background_snapshots": {
                        name: snapshot.to_json()
                        for name, snapshot in sorted(cached.items())
                    },
                    "desktop_cpu": cpu_executor,
                    "desktop_cuda": cuda_executor,
                    "phone_adreno": phone_executor,
                    "cpu_phone_ffn_split": split_executor,
                    "phone_live_snapshot": phone_live,
                    "system_snapshot": automated_snapshot.to_json(),
                },
                "overhead_ns": overhead,
                "marginal_system_context": marginal_receipt,
                "online_placement_receipt": online_receipt,
                "runtime_context": runtime_context,
                "status": ticket.dispatch_state,
                "ticket": ticket.to_json(),
            }
            with result_lock:
                decision_attempts_by_index.setdefault(index, []).append(
                    attempt
                )
                overhead_rows.append(overhead)
                runtime_context_by_index.setdefault(index, runtime_context)
                estimates_by_index[index] = estimate_set.to_json()
                decisions_by_index[index] = decision
                activations_by_index[index] = activation
                if online_receipt is not None:
                    online_receipts_by_index[index] = online_receipt
            events.write({
                "attempt": attempt,
                "kind": "runtime_scheduler_decision",
                "overlay_request_index": index,
                "schema": EVENT_SCHEMA,
                "t_ns": time.monotonic_ns(),
            })
            return {
                "decision": decision,
                "ticket": ticket,
            }

        def execute_scheduler_ticket(
            ticket: RuntimeRequestTicket,
            row: dict[str, Any],
            stream_path: Path,
            on_first: Callable[[int], None],
        ) -> object:
            index = row["overlay_request_index"]

            def on_renewal(receipt: Any) -> None:
                receipt_json = receipt.to_json()
                with result_lock:
                    renewals_by_index.setdefault(index, []).append(
                        receipt_json
                    )
                events.write({
                    "kind": "runtime_lease_renewal",
                    "overlay_request_index": index,
                    "renewal": receipt_json,
                    "schema": EVENT_SCHEMA,
                    "t_ns": time.monotonic_ns(),
                })

            def current_snapshot(
                _ticket: RuntimeRequestTicket, _at_us: int
            ) -> HeterogeneousRuntimeSnapshot:
                observed = schedule_runtime_attempt(
                    row,
                    "physical-adapter-refresh",
                    observation_only=True,
                )
                value = observed["snapshot"]
                require(
                    isinstance(value, HeterogeneousRuntimeSnapshot),
                    "automated runtime snapshot",
                )
                return value

            def server_rows() -> tuple[dict[str, Any], ...]:
                assert sampler is not None
                with sampler.lock:
                    return tuple(sampler.rows)

            def execution_finished(command: object) -> None:
                activity_tracker.finish(command)
                assert runtime_monitor is not None
                runtime_monitor.request_refresh()

            assert phone_power_sampler is not None
            energy_meter = RaplNvmlPhoneEnergyMeter(
                server_rows,
                physical.run_trace.server_energy_summary,
                phone_power_sampler,
                energy_boundary_id=(
                    automated_catalog.placement_profile.energy_boundary_id
                ),
            )
            backend = CanonicalHttpExecutionBackend(
                LlamaCppHttpClient(),
                energy_meter,
                epoch_ns=paid_start_ns,
                prepare_transition=prepare_physical_transition,
                on_execution_start=activity_tracker.start,
                on_execution_finish=execution_finished,
            )
            adapter = CanonicalPhysicalAdapter(
                scheduler,
                backend,
                epoch_ns=paid_start_ns,
                snapshot_provider=current_snapshot,
                lease_guard_us=LEASE_RENEWAL_GUARD_US,
                lease_quantum_us=LEASE_RENEWAL_QUANTUM_US,
                on_renewal=on_renewal,
            )
            payload = LlamaCppCompletionPayload(
                request_id=row["event_id"],
                expected_model_alias=LLAMA1,
                input_tokens=row["input_tokens"],
                output_tokens=row["output_tokens"],
                prompt_tokens=tuple(row["prompt_tokens"]),
                seed=row["overlay_request_index"],
                stream_path=stream_path,
                on_first_token=on_first,
            )
            result = adapter.execute(ticket, payload)
            with result_lock:
                adapter_results_by_index[index] = result
                releases_by_index[index] = result.completion.to_json()
                if result.recoveries:
                    recoveries_by_index[index] = (
                        result.recoveries[-1].to_json()
                    )
                dispatch_receipts_by_index.setdefault(index, []).extend(
                    receipt.to_json()
                    for receipt in result.dispatch_receipts
                )
            for receipt in result.dispatch_receipts:
                events.write({
                    "dispatch": receipt.to_json(),
                    "kind": "runtime_dispatch_queue_wake",
                    "overlay_request_index": index,
                    "schema": EVENT_SCHEMA,
                    "t_ns": time.monotonic_ns(),
                })
            return result

        def execute(
            row: dict[str, Any],
            ticket: RuntimeRequestTicket,
        ) -> None:
            index = row["overlay_request_index"]
            first: list[int] = []
            initial_ticket = ticket
            active_ticket = ticket
            recovery = None
            try:
                adapter_result = execute_scheduler_ticket(
                    active_ticket,
                    row,
                    args.output / f"stream-scheduler-{index:02d}.raw",
                    first.append,
                )
                active_ticket = adapter_result.ticket
                active_route = adapter_result.command.route_id
                value = adapter_result.observation.payload
                release = adapter_result.completion.to_json()
                dispatch_ns = (
                    paid_start_ns
                    + adapter_result.observation.started_us * 1000
                )
                recovery = (
                    None
                    if not adapter_result.recoveries
                    else adapter_result.recoveries[-1].to_json()
                )
                require(len(first) == 1, "first-token accounting")
                completion_ns = time.monotonic_ns()
                scheduled_ns = paid_start_ns + row["arrival_us"] * 1000
                final_estimate = (
                    None
                    if active_ticket is None
                    else next(
                        estimate
                        for estimate in active_ticket.cost_estimates.estimates
                        if estimate.route_id
                            == active_ticket.decision.route_id
                    )
                )
                final_ticket_json = active_ticket.to_json()
                record = {
                    "calibration_split": row.get("calibration_split"),
                    "combined_request_index": row["combined_request_index"],
                    "completion_ns": completion_ns,
                    "dispatch_ns": dispatch_ns,
                    "event_id": row["event_id"],
                    "endpoint_slot_id": value["endpoint_slot_id"],
                    "endpoint_task_id": value["endpoint_task_id"],
                    "endpoint_model_alias": value[
                        "endpoint_model_alias"
                    ],
                    "endpoint_slot_probe_errors": value[
                        "endpoint_slot_probe_errors"
                    ],
                    "execution_model_id": LLAMA1,
                    "first_token_ns": first[0],
                    "input_tokens": row["input_tokens"],
                    "output_tokens": row["output_tokens"],
                    "overlay_request_index": index,
                    "predicted_ms": value["predicted_ms"],
                    "prompt_ms": value["prompt_ms"],
                    "route": active_route,
                    "runtime_prompt_tokens": value["runtime_prompt_tokens"],
                    "runtime_context": runtime_context_by_index[index],
                    "scheduled_arrival_ns": scheduled_ns,
                    "scheduler_decision": (
                        decision_to_json(initial_ticket.decision)
                    ),
                    "scheduler_final_decision": (
                        decision_to_json(active_ticket.decision)
                    ),
                    "scheduler_final_binding": (
                        active_ticket.binding.to_json()
                    ),
                    "scheduler_final_candidate_baseline": (
                        None
                        if final_estimate is None
                        else final_estimate.baseline
                    ),
                    "scheduler_final_execution_plan": (
                        final_ticket_json.get("execution_plan")
                    ),
                    "scheduler_final_ticket": final_ticket_json,
                    "scheduler_release": release,
                    "scheduler_recovery": recovery,
                    "slo_met": completion_ns <= scheduled_ns + row["slo_us"] * 1000,
                    "slo_us": row["slo_us"],
                    "source_overlay_request_index": row.get(
                        "source_overlay_request_index"
                    ),
                    "target_large_phase": row.get("target_large_phase"),
                    "stream_sha256": value["stream_sha256"],
                    "tokens": value["tokens"],
                }
                with result_lock:
                    results.append(record)
                events.write({"kind": "overlay_request_complete", **record})
            except BaseException as error:
                with result_lock:
                    errors.append(f"{index}: {type(error).__name__}: {error}")

        def execute_runtime_attempt(
            row: dict[str, Any],
            attempt: dict[str, object],
        ) -> None:
            selected = attempt["ticket"]
            require(
                isinstance(selected, RuntimeRequestTicket),
                "runtime ticket",
            )
            execute(row, selected)

        futures: list[concurrent.futures.Future[None]] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(32, overlay_count),
            thread_name_prefix="fp16-llama1b",
        ) as pool:
            for source_row in sorted(
                rows,
                key=lambda item: item.get(
                    "arrival_order", item["arrival_us"]
                ),
            ):
                row = materialize_arrival(
                    source_row, args.base_output, paid_start_ns
                )
                events.write({
                    "effective_arrival_us": row["arrival_us"],
                    "kind": "overlay_request_arrival",
                    "overlay_request_index": row["overlay_request_index"],
                    "schema": EVENT_SCHEMA,
                    "t_ns": time.monotonic_ns(),
                })
                attempt = schedule_runtime_attempt(row, "arrival")
                futures.append(pool.submit(
                    execute_runtime_attempt,
                    row,
                    attempt,
                ))
            for future in futures:
                future.result(timeout=3600)

        require(not errors, "overlay request errors: " + "; ".join(errors))
        require(
            len(results) == overlay_count
            and sorted(row["overlay_request_index"] for row in results)
                == list(range(overlay_count))
            and set(runtime_context_by_index) == set(range(overlay_count)),
            "overlay request conservation",
        )
        require(
            cpu_load_ms is not None
            and cpu_props is not None
            and cpu_warm_ms is not None,
            "CPU executor receipt",
        )
        if split_executor_required:
            require(
                split_process is not None
                and split_load_ms is not None
                and split_props is not None
                and split_warm_ms is not None,
                "split executor receipt",
            )
        require(
            set(estimates_by_index) == set(range(overlay_count))
            and set(decisions_by_index) == set(range(overlay_count))
            and set(decision_attempts_by_index) == set(range(overlay_count))
            and set(releases_by_index) == set(range(overlay_count))
            and set(activations_by_index) == set(range(overlay_count))
            and set(dispatch_receipts_by_index) == set(range(overlay_count))
            and len(overhead_rows) >= overlay_count,
            "unified scheduler coverage",
        )

        phone_live_final = wait_for_live_phone_snapshot(
            args.phone_host,
            args.phone_diagnostic_port,
            args.phone_memory_reserve_bytes,
            15,
        )
        require(phone_live_final is not None, "final phone live diagnostic")
        write_new_atomic(args.resident_release_file, {
            "overlay_completed": overlay_count,
            "overlay_end_ns": max(row["completion_ns"] for row in results),
            "released_at_ns": time.monotonic_ns(),
            "schema": "s42-resident-release-v1",
            "status": "OVERLAY_COMPLETE",
        })
        resident_release_status = "OVERLAY_COMPLETE"
        resident_release_sha256 = digest(args.resident_release_file)
        base = wait_for_base_result(args.base_output, 7200)
        base_scheduler = base.get("fp16_resident_scheduler", {})
        expected_arm = (
            "control"
            if args.large_model_policy == "cpu-overflow"
            else "op15"
        )
        require(
            base.get("schema") == BASE_RESULT_SCHEMA
            and base.get("status") == "PASS"
            and base.get("mode") == "fp16-switch"
            and base.get("metrics", {}).get("completed") == 74
            and base.get("metrics", {}).get("output_tokens") == 11_605
            and base.get("paid_start_ns") == paid_start_ns
            and base_scheduler.get("arm") == expected_arm
            and base_scheduler.get("arm_source")
                in {
                    "experimental_override",
                    "model_device_runtime_bootstrap",
                    "runtime_placement",
                }
            and base_scheduler.get("planned_arm") in {"control", "op15"}
            and (
                base_scheduler.get("arm_source")
                    not in {
                        "model_device_runtime_bootstrap",
                        "runtime_placement",
                    }
                or base_scheduler.get("planned_arm") == expected_arm
            ),
            "base F16 replay identity",
        )
        if args.requested_large_model_policy == "runtime-auto":
            require(
                base_scheduler.get("arm_source")
                    == "model_device_runtime_bootstrap"
                and base_scheduler.get("scheduler_scope")
                    == "causal_request_level"
                and len(base_scheduler.get(
                    "request_level_online_receipts", {}
                )) == 74,
                "runtime-auto large-model request scheduler",
            )
        paid_end_ns = max(
            base["paid_end_ns"], max(row["completion_ns"] for row in results)
        )
        phase_monitor_stop.set()
        if phase_monitor_thread is not None:
            phase_monitor_thread.join(timeout=30)
            require(
                not phase_monitor_thread.is_alive(),
                "large resource monitor stop",
            )
        require(
            not phase_monitor_error,
            "large resource monitor: " + "; ".join(phase_monitor_error),
        )
        with scheduler_lock:
            scheduler.finish_runtime_phase(
                phase_source_id,
                (paid_end_ns - paid_start_ns) // 1000,
            )
            phase_history = [
                {
                    **dict(row),
                    "functionfs_owner": row["owner_id"],
                    "phase": row["phase_id"],
                }
                for row in scheduler.runtime_phase_history(phase_source_id)
            ]
        assert runtime_monitor is not None
        runtime_monitor_final = {
            name: runtime_monitor.snapshot(name).to_json()
            for name in sorted(monitor_probes)
        }
        runtime_monitor.stop()
        runtime_monitor = None
        time.sleep(0.5)
        sampler.stop()
        samples = list(sampler.rows)
        sampler = None
        assert phone_power_sampler is not None
        phone_power_sampler.stop()
        phone_power_rows = list(phone_power_sampler.rows)
        phone_power_sampler = None
        phone_power_samples_path = args.output / "phone-power-samples.json"
        write_new_atomic(phone_power_samples_path, {
            "rows": phone_power_rows,
            "schema": "s42-op15-live-power-samples-v1",
        })
        cpu_process.terminate()
        cpu_process = None
        if split_process is not None:
            split_process.terminate()
            split_process = None
        with cuda_state_lock:
            if cuda_process is not None:
                cuda_process.terminate()
                cuda_process = None

        paid_samples = [
            row for row in samples
            if paid_start_ns <= row["t_ns"] <= paid_end_ns
        ]
        require(paid_samples, "paid resource samples")
        base_rows = base["request_results"]
        base_slo_met = sum(
            row["completion_ns"]
                <= row["scheduled_arrival_ns"] + row["slo_us"] * 1000
            for row in base_rows
        )
        duration_s = (paid_end_ns - paid_start_ns) / 1e9
        route_counts: dict[str, int] = {}
        for row in results:
            route_counts[row["route"]] = route_counts.get(row["route"], 0) + 1
        overhead_summary = (
            None
            if not overhead_rows
            else {
                key: timing_stats(overhead_rows, key)
                for key in sorted(overhead_rows[0])
            }
        )
        recovery_overhead_summary = (
            None
            if not recovery_overhead_rows
            else {
                key: timing_stats(recovery_overhead_rows, key)
                for key in sorted(recovery_overhead_rows[0])
            }
        )
        with scheduler_lock:
            runtime_scheduler_state = scheduler.runtime_controller_snapshot()
        result = {
            "base": {
                "arm": expected_arm,
                "arm_source": base_scheduler["arm_source"],
                "duration_s": base["metrics"]["duration_s"],
                "output_tokens": base["metrics"]["output_tokens"],
                "planned_arm": base_scheduler["planned_arm"],
                "request_count": base["metrics"]["completed"],
                "result_path": str(args.base_output / "RESULT.json"),
                "result_sha256": digest(args.base_output / "RESULT.json"),
            },
            "input_sha256": {
                "base_trace": manifest["base_trace"]["sha256"],
                "manifest": digest(args.manifest),
                "overlay_trace": digest(args.requests),
                "runtime_profile": digest(args.runtime_profile),
                "automated_catalog": digest(args.automated_catalog),
                **(
                    {}
                    if args.ffn_manifest is None
                    else {
                        "ffn_compiled_policy": digest(
                            args.ffn_compiled_policy
                        ),
                        "ffn_manifest": digest(args.ffn_manifest),
                    }
                ),
            },
            "execution_sha256": {
                "phone_power_samples": digest(phone_power_samples_path),
                "resident_release": resident_release_sha256,
            },
            "metrics": {
                "completed": 74 + overlay_count,
                "duration_s": duration_s,
                "input_tokens": 33_843 + overlay_input_tokens,
                "output_throughput_tokens_s": (
                    (11_605 + overlay_output_tokens) / duration_s
                ),
                "output_tokens": 11_605 + overlay_output_tokens,
                "small_model": {
                    "completed": overlay_count,
                    "completion_s": physical.run_trace.stats([
                        (row["completion_ns"] - row["scheduled_arrival_ns"])
                            / 1e9
                        for row in results
                    ]),
                    "input_tokens": overlay_input_tokens,
                    "output_tokens": overlay_output_tokens,
                    "slo_met": sum(row["slo_met"] for row in results),
                },
                "slo_met": base_slo_met + sum(
                    row["slo_met"] for row in results
                ),
            },
            "model_identities": manifest["model_inventory"],
            "paid_end_ns": paid_end_ns,
            "paid_start_ns": paid_start_ns,
            "policy": {
                "large_model_arm": expected_arm,
                "large_model_policy": args.large_model_policy,
                "requested_large_model_policy": (
                    args.requested_large_model_policy
                ),
                "large_model_gpu_sequence": [
                    "qwen3-14b-q4km-dequant-f16",
                    "gemma-4-12b-q40-dequant-f16",
                ],
                "policy_id": (
                    f"{args.large_model_policy}+{args.small_model_policy}"
                ),
                "small_model_policy": args.small_model_policy,
                "static_route": args.forced_static_route,
                "small_model_cpu_load_boundary": (
                    "inside_paid_interval_before_overlay_dispatch"
                ),
                "small_model_cpu_resident": True,
                "small_model_phone_resident": True,
            },
            "request_results": sorted(
                results, key=lambda row: row["overlay_request_index"]
            ),
            "resources": {
                "executors": {
                    "desktop-cpu": {
                        "load_ms": cpu_load_ms,
                        "model_alias": cpu_props.get("model_alias"),
                    },
                    "desktop-cuda": {
                        "load_ms": cuda_load_ms,
                        "model_alias": (
                            None
                            if cuda_props is None
                            else cuda_props.get("model_alias")
                        ),
                        "started": cuda_load_ms is not None,
                    },
                    SPLIT_ROUTE: {
                        "compiled_policy": (
                            None
                            if ffn_compiled_policy is None
                            else ffn_compiled_policy["policy_text"]
                        ),
                        "load_ms": split_load_ms,
                        "model_alias": (
                            None
                            if split_props is None
                            else split_props.get("model_alias")
                        ),
                        "prewarm_shape": split_warm_shape,
                        "started": split_load_ms is not None,
                        "warm_ms": split_warm_ms,
                    },
                    "phone-adreno": {
                        "model_alias": phone_props.get("model_alias"),
                        "warm_ms": phone_warm_ms,
                    },
                },
                "gpu_memory_used_max_bytes": max(
                    row["gpu"]["memory_used_bytes"] for row in paid_samples
                ),
                "gpu_power_w": physical.run_trace.stats([
                    row["gpu"]["power_mw"] / 1000 for row in paid_samples
                ]),
                "gpu_utilization_pct": physical.run_trace.stats([
                    row["gpu"]["utilization_pct"] for row in paid_samples
                ]),
                "llama1_cpu_swap_max_bytes": max(
                    row["pids"].get("llama1-cpu", {}).get("swap_bytes", 0)
                    for row in paid_samples
                ),
                "llama1_split_swap_max_bytes": max(
                    row["pids"].get(
                        "llama1-cpu-phone-ffn", {}
                    ).get("swap_bytes", 0)
                    for row in paid_samples
                ),
                "system_available_min_bytes": min(
                    row["system"]["available_bytes"] for row in paid_samples
                ),
                "system_swap_free_min_bytes": min(
                    row["system"]["swap_free_bytes"] for row in paid_samples
                ),
            },
            "scheduler_runtime": {
                "active_execution_lease_policy": (
                    "predicted_bound_then_event_driven_renewal_if_needed"
                ),
                "activation_receipts": {
                    str(index): activations_by_index[index]
                    for index in sorted(activations_by_index)
                },
                "cost_estimates": {
                    str(index): estimates_by_index[index]
                    for index in sorted(estimates_by_index)
                },
                "decisions": [
                    {
                        **decision_to_json(decisions_by_index[index]),
                        "overlay_request_index": index,
                    }
                    for index in sorted(decisions_by_index)
                ],
                "decision_attempts": {
                    str(index): decision_attempts_by_index[index]
                    for index in sorted(decision_attempts_by_index)
                },
                "dispatch_queue": runtime_scheduler_state[
                    "dispatch_queue"
                ],
                "dispatch_receipts": {
                    str(index): dispatch_receipts_by_index[index]
                    for index in sorted(dispatch_receipts_by_index)
                },
                "decision_log": scheduler.runtime_decision_log(),
                "decision_mode": "automated_gguf_capability_routes",
                "enabled": True,
                "external_reservations": phase_history,
                "health_snapshot_policy": (
                    "background_cached_with_completion_refresh"
                ),
                "health_snapshots_final": runtime_monitor_final,
                "online_placement_receipts": {
                    str(index): online_receipts_by_index[index]
                    for index in sorted(online_receipts_by_index)
                },
                "online_placement_required": False,
                "physical_execution_receipt_required": True,
                "selection_mode": (
                    "desktop-baseline"
                    if args.small_model_policy == "static-cpu"
                    else "energy-aware"
                ),
                "marginal_system_profile": {
                    "path": str(args.marginal_system_profile),
                    "profile_id": marginal_profile["profile_id"],
                    "sha256": digest(args.marginal_system_profile),
                },
                "operator_split": (
                    None
                    if ffn_compiled_policy is None
                    else {
                        "compiler": ffn_compiled_policy["compiler"],
                        "manifest_record_sha256": ffn_manifest[
                            "record_sha256"
                        ],
                        "physical_route_profiled": split_route_profiled,
                        "policy_record_sha256": ffn_compiled_policy[
                            "record_sha256"
                        ],
                        "policy_text": ffn_compiled_policy["policy_text"],
                        "qualification": ffn_compiled_policy[
                            "qualification"
                        ],
                    }
                ),
                "phone_probe_policy": (
                    "health_admits_busy_executor_calendar_queues_occupancy"
                ),
                "overhead": overhead_summary,
                "recoveries": {
                    str(index): recoveries_by_index[index]
                    for index in sorted(recoveries_by_index)
                },
                "recovery_overhead": recovery_overhead_summary,
                "renewal_receipts": {
                    str(index): renewals_by_index[index]
                    for index in sorted(renewals_by_index)
                },
                "release_receipts": {
                    str(index): releases_by_index[index]
                    for index in sorted(releases_by_index)
                },
                "route_counts": dict(sorted(route_counts.items())),
                "route_uncertainty": runtime_scheduler_state[
                    "route_uncertainty"
                ],
                "runtime_controller": dict(runtime_scheduler_state),
                "transport_ownership_policy": (
                    "large-phase-exclusive-route-gate-with-renewable-leases"
                ),
                "runtime_context_snapshots": {
                    str(index): runtime_context_by_index[index]
                    for index in sorted(runtime_context_by_index)
                },
            },
            "server_energy": physical.run_trace.server_energy_summary(
                samples, paid_start_ns, paid_end_ns
            ),
            "phone_residency_snapshot": {
                "final": phone_live_final,
                "final_observed_before_resident_release": True,
                "reserve_bytes": args.phone_memory_reserve_bytes,
                "startup": phone_live_initial,
                "task_artifact_sha256": args.phone_model_sha256,
                "transport": "functionfs-plus-ncm",
            },
            "work_receipts": {
                "large_model_outputs": output_token_receipt(
                    base_rows, "request_index"
                ),
                "small_model_outputs": output_token_receipt(
                    results, "overlay_request_index"
                ),
            },
            "schema": RESULT_SCHEMA,
            "status": "PASS",
        }
        physical.run_trace.write_json(args.output / "RESULT.json", result)
        print(json.dumps({
            "duration_s": duration_s,
            "route_counts": result["scheduler_runtime"]["route_counts"],
            "scheduling_core_mean_us": (
                None
                if overhead_summary is None
                else overhead_summary["total_core_ns"]["mean_us"]
            ),
            "server_energy_j": result["server_energy"][
                "server_compute_device_energy_j"
            ],
            "status": "PASS",
        }, sort_keys=True))
        return 0
    except BaseException as error:
        physical.run_trace.write_json(args.output / "FAILURE.json", {
            "error": f"{type(error).__name__}: {error}",
            "schema": "s42-full-fp16-llama1b-combined-failure-v1",
            "status": "FAIL",
        })
        raise
    finally:
        if not args.resident_release_file.exists():
            try:
                write_new_atomic(args.resident_release_file, {
                    "released_at_ns": time.monotonic_ns(),
                    "schema": "s42-resident-release-v1",
                    "status": resident_release_status,
                })
            except OSError:
                pass
        events.close()
        if sampler is not None:
            sampler.stop()
        if phone_power_sampler is not None:
            phone_power_sampler.stop()
        if runtime_monitor is not None:
            runtime_monitor.stop()
        phase_monitor_stop.set()
        if phase_monitor_thread is not None:
            phase_monitor_thread.join(timeout=5)
        with cuda_state_lock:
            if cuda_process is not None:
                cuda_process.terminate()
        if split_process is not None:
            split_process.terminate()
        if cpu_process is not None:
            cpu_process.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
