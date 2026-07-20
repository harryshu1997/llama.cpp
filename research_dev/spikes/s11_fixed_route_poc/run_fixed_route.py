#!/usr/bin/env python3
"""Fixed-route Q-PIM proof-of-concept runner.

This runner compares the same deterministic greedy workload on:

  SERVER_ONLY: full model on one selected CUDA GPU
  A0_OP15:     OP15 owns [0, k2), the selected CUDA GPU owns [k2, n)
  A0_OP15_OP12: OP15 owns [0, k2), OP12 owns [k2, k3), CUDA owns [k3, n)

The optional measurement is GPU_BOARD only. It never claims server-wall, phone,
USB, or total-system energy. A result is invalid unless every paired route
returns exactly the same token ids.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import shlex
import statistics
import subprocess
import sys
import threading
import time
from decimal import Decimal, ROUND_HALF_UP


ROUTE_CONTROL = "SERVER_ONLY"
ROUTE_OP15 = "A0_OP15"
ROUTE_TWO_PHONE = "A0_OP15_OP12"
PLAN_SCHEMA = "s11-fixed-route-poc-v4"
RESULT_SCHEMA = "s11-fixed-route-poc-result-v4"
FAILURE_SCHEMA = "s11-fixed-route-poc-failure-v4"
RUN_SCHEMA = "s11-fixed-route-poc-run-v4"
ROUTE_RECORD_SCHEMA = "layersplit-route-v2"
MIN_MEASUREMENT_PAIRS = 8
MIN_INDEPENDENT_UPDATES = 100
MIN_MEASURED_GENERATED_TOKENS = 32
MAX_MEASURED_REQUESTS = 4096
MAX_SAMPLE_GAP_US = 250_000
NVML_UNCERTAINTY_MW = 5_000
NVML_AVERAGING_WINDOW_US = 1_000_000
S11_E0_GPU_UUID = "GPU-431a4567-fa90-7a73-625a-2ee6e7b5eaaf"
S11_E0_BATCH_SIZE = 8
S11_E0_GENERATED_TOKENS = 32
S11_E0_DRIVER_CONTEXT = 96
S11_E0_DRIVER_MAX_PREFILL = 64
S11_E0_OP15_END = 2
S11_E0_N_LAYER = 48
PSTATE_PATTERN = re.compile(r"P(?:[0-9]|1[0-5])")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
BOOT_UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
RUNNER_PATH = pathlib.Path(__file__).resolve()
REPO_ROOT = RUNNER_PATH.parents[3]
LAYERSPLIT_SOURCE_PATH = REPO_ROOT / "examples/layersplit/layersplit.cpp"
LLAMA_MODEL_SOURCE_PATH = REPO_ROOT / "src/llama-model.cpp"


class ExperimentError(RuntimeError):
    pass


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ExperimentError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ExperimentError(f"invalid JSON constant: {value}")


def strict_json_loads(text, label):
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise ExperimentError(f"invalid {label}: {exc}") from exc
    if type(value) is not dict:
        raise ExperimentError(f"{label} must be an object")
    return value


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_bytes(value)
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def run_checked(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        raise ExperimentError(
            f"command failed ({result.returncode}): {shlex.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result


def adb(serial, *arguments, check=True):
    command = ["adb", "-s", serial, *arguments]
    if check:
        return run_checked(command)
    return subprocess.run(command, capture_output=True, text=True)


def connected_serials():
    result = run_checked(["adb", "devices"])
    serials = set()
    for line in result.stdout.splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 2 and fields[1] == "device":
            serials.add(fields[0])
    return serials


def query_phone_state(serial):
    battery_text = adb(serial, "shell", "dumpsys", "battery").stdout
    battery = {}
    battery_keys = {
        "USB powered": "usb_powered",
        "status": "status",
        "level": "level",
        "temperature": "temperature_tenths_c",
        "voltage": "voltage_mv",
        "Charge counter": "charge_counter_uah",
    }
    for line in battery_text.splitlines():
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        output_key = battery_keys.get(key)
        if output_key is None:
            continue
        if value in ("true", "false"):
            battery[output_key] = value == "true"
        else:
            try:
                battery[output_key] = int(value)
            except ValueError:
                battery[output_key] = value

    thermal_text = adb(serial, "shell", "dumpsys", "thermalservice").stdout
    thermal_status = None
    current_temperatures = []
    in_current = False
    pattern = re.compile(
        r"Temperature\{mValue=([^,]+), mType=(\d+), "
        r"mName=([^,]+), mStatus=(\d+)\}")
    for line in thermal_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Thermal Status:"):
            try:
                thermal_status = int(stripped.split(":", 1)[1].strip())
            except ValueError:
                thermal_status = None
        if stripped == "Current temperatures from HAL:":
            in_current = True
            continue
        if stripped == "Current cooling devices from HAL:":
            in_current = False
        if not in_current:
            continue
        match = pattern.fullmatch(stripped)
        if match is None:
            continue
        try:
            value_millic = int(
                (Decimal(match.group(1)) * Decimal(1000)).to_integral_value(
                    rounding=ROUND_HALF_UP))
        except ArithmeticError:
            continue
        current_temperatures.append({
            "name": match.group(3),
            "type": int(match.group(2)),
            "status": int(match.group(4)),
            "value_millic": value_millic,
        })
    current_temperatures.sort(key=lambda item: (item["type"], item["name"]))
    return {
        "serial": serial,
        "timestamp_monotonic_us": time.monotonic_ns() // 1000,
        "battery": battery,
        "thermal_status": thermal_status,
        "current_temperatures": current_temperatures,
    }


def phone_uptime_us(serial):
    """One boundary read of the phone's monotonic uptime, in microseconds.

    Used only at the window edges (before GO, after DRIVER_DONE) to bracket the
    on-device thermal log against the paid window; it is not called during the
    window, so it adds no periodic ADB traffic to the measured interval.
    """
    raw = adb(serial, "shell", "cut -d' ' -f1 /proc/uptime").stdout.strip()
    seconds = Decimal(raw)
    return int((seconds * Decimal(1_000_000)).to_integral_value(
        rounding=ROUND_HALF_UP))


def phone_boundary_quality(serials, before, after):
    reasons = []
    for serial in sorted(serials):
        for boundary, states in (("BEFORE", before), ("AFTER", after)):
            state = states.get(serial)
            if type(state) is not dict:
                reasons.append(f"E_PHONE_STATE_{boundary}:{serial}")
                continue
            thermal_status = state.get("thermal_status")
            if type(thermal_status) is not int or thermal_status != 0:
                reasons.append(f"E_PHONE_THERMAL_STATUS_{boundary}:{serial}")
            temperatures = state.get("current_temperatures")
            if type(temperatures) is not list or not temperatures:
                reasons.append(f"E_PHONE_TEMPERATURES_{boundary}:{serial}")
    return {
        "valid": not reasons,
        "coverage": "BOUNDARY_ONLY",
        "reasons": reasons,
    }


def query_gpu(gpu_uuid):
    result = run_checked([
        "nvidia-smi", "-i", gpu_uuid,
        "--query-gpu=uuid,pstate,power.draw,power.limit,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ])
    fields = [field.strip() for field in result.stdout.strip().split(",")]
    if len(fields) != 6 or fields[0] != gpu_uuid:
        raise ExperimentError(f"unexpected nvidia-smi output: {result.stdout!r}")
    return {
        "uuid": fields[0],
        "pstate": fields[1],
        "power_mw": watts_to_mw(fields[2]),
        "power_limit_mw": watts_to_mw(fields[3]),
        "utilization_pct": int(fields[4]),
        "memory_used_mib": int(fields[5]),
    }


def wait_for_gpu_idle(gpu_uuid, max_utilization_pct, max_memory_mib,
                      timeout_s=60, stable_samples=4, interval_s=0.25):
    deadline = time.monotonic() + timeout_s
    stable = 0
    last = None
    while time.monotonic() < deadline:
        last = query_gpu(gpu_uuid)
        if (last["utilization_pct"] <= max_utilization_pct and
                last["memory_used_mib"] <= max_memory_mib):
            stable += 1
            if stable >= stable_samples:
                return last
        else:
            stable = 0
        time.sleep(interval_s)
    raise ExperimentError(
        f"selected GPU did not become idle within {timeout_s}s: {last}")


def query_gpu_processes(gpu_uuid, timeout_s=None):
    result = run_checked([
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ], timeout=timeout_s)
    processes = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4 or fields[0] != gpu_uuid:
            continue
        processes.append({
            "pid": int(fields[1]),
            "name": fields[2],
            "memory_used_mib": int(fields[3]),
        })
    return processes


def watts_to_mw(text):
    value = Decimal(text.strip()) * Decimal(1000)
    return int(value.to_integral_value(rounding=ROUND_HALF_UP))


def parse_power_sample(raw, gpu_uuid, timestamp_us):
    """Parse one nvidia-smi power line, binding power.limit (CP1.4).

    Raises ExperimentError on any malformed field. power_limit_mw is carried on
    every sample so the paid window can prove the limit never moved; it is an
    extra key that integration and quality ignore, so it does not change the ZOH
    energy or the sample-quality gates.
    """
    fields = [field.strip() for field in raw.strip().split(",")]
    if len(fields) != 6 or fields[0] != gpu_uuid:
        raise ExperimentError(f"bad power sample: {raw.rstrip()!r}")
    return {
        "kind": "sample",
        "board_uuid": fields[0],
        "timestamp_us": timestamp_us,
        "power_mw": watts_to_mw(fields[1]),
        "pstate": fields[2],
        "utilization_pct": int(fields[3]),
        "memory_used_mib": int(fields[4]),
        "power_limit_mw": watts_to_mw(fields[5]),
    }


class PowerSampler:
    def __init__(self, gpu_uuid, output_path, interval_ms=100):
        self.gpu_uuid = gpu_uuid
        self.output_path = pathlib.Path(output_path)
        self.interval_ms = interval_ms
        self.samples = []
        self.errors = []
        self.records = []
        self._lock = threading.Lock()
        self._process = None
        self._thread = None

    def start(self):
        command = [
            "nvidia-smi", "-i", self.gpu_uuid,
            "--query-gpu=uuid,power.draw,pstate,utilization.gpu,memory.used,"
            "power.limit",
            "--format=csv,noheader,nounits",
            f"--loop-ms={self.interval_ms}",
        ]
        self._process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1)
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self):
        assert self._process is not None
        assert self._process.stdout is not None
        for raw in self._process.stdout:
            timestamp_us = time.monotonic_ns() // 1000
            try:
                sample = parse_power_sample(raw, self.gpu_uuid, timestamp_us)
            except (ExperimentError, ValueError, ArithmeticError) as exc:
                with self._lock:
                    message = f"bad sample: {raw.rstrip()!r}: {exc}"
                    self.errors.append(message)
                    self.records.append({
                        "kind": "error", "board_uuid": self.gpu_uuid,
                        "timestamp_us": timestamp_us, "message": message})
                continue
            with self._lock:
                self.samples.append(sample)
                self.records.append(sample)

    def observed_power_limits(self):
        with self._lock:
            return sorted({sample["power_limit_mw"] for sample in self.samples})

    def count(self):
        with self._lock:
            return len(self.samples)

    def wait_for_samples(self, count, timeout_s):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.count() >= count:
                return
            if self._process is not None and self._process.poll() is not None:
                break
            time.sleep(0.02)
        raise ExperimentError(f"NVML sampler did not produce {count} samples")

    def stop(self):
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)
        if self._thread is not None:
            self._thread.join(timeout=3)
            if self._thread.is_alive():
                raise ExperimentError("NVML sampler reader did not terminate")
        with self._lock:
            samples = list(self.samples)
            errors = list(self.errors)
            records = list(self.records)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="ascii") as handle:
            for record in records:
                handle.write(json.dumps(
                    record, sort_keys=True, separators=(",", ":")) + "\n")
        return samples, errors


PROCESS_SAMPLE_FIELDS = frozenset({
    "kind", "board_uuid", "probe_start_us", "probe_end_us", "timestamp_us",
    "processes"})
PROCESS_ERROR_FIELDS = frozenset({
    "kind", "board_uuid", "probe_start_us", "probe_end_us", "message"})
PROCESS_BOUNDARY_FIELDS = frozenset({
    "kind", "board_uuid", "phase", "probe_start_us", "probe_end_us",
    "timestamp_us", "processes"})
PROCESS_ENTRY_FIELDS = frozenset({"pid", "name", "memory_used_mib"})


class GpuProcessMonitor:
    """Continuously samples selected-GPU compute apps through the paid window.

    The boundary-only check the runner shipped with cannot see a process that
    starts and finishes between the ready and done boundaries. This monitor polls
    at a fixed cadence for the whole paid window; any compute app whose PID is not
    the exact driver PID contaminates the slot.

    Every probe runs in a bounded worker thread so a hung nvidia-smi cannot stall
    the loop past `probe_timeout_s`; a probe that overruns or raises is persisted
    as an error record rather than dropped. Each record carries the monotonic
    probe-start and probe-completion timestamps, and an observation is stamped at
    completion (never before the probe returned), so a stale reading cannot be
    back-dated into the window. Both sample and error records are written to the
    hashed artifact, so quality is derived entirely from persisted bytes. stop()
    fails closed if the worker thread will not join.
    """

    def __init__(self, gpu_uuid, output_path, interval_ms=100, probe=None,
                 probe_timeout_s=2.0):
        self.gpu_uuid = gpu_uuid
        self.output_path = pathlib.Path(output_path)
        self.interval_ms = interval_ms
        self.probe_timeout_s = probe_timeout_s
        self._probe = probe
        self.records = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _probe_once(self):
        """Run a killable nvidia-smi probe. Returns (entries, error)."""
        try:
            value = (self._probe() if self._probe is not None else
                     query_gpu_processes(
                         self.gpu_uuid, timeout_s=self.probe_timeout_s))
            entries = [
                {"pid": int(entry["pid"]), "name": str(entry["name"]),
                 "memory_used_mib": int(entry["memory_used_mib"])}
                for entry in value
            ]
        except subprocess.TimeoutExpired:
            return None, f"probe exceeded {self.probe_timeout_s}s bound"
        except Exception as exc:  # noqa: BLE001 - malformed probe output
            return None, f"probe error: {exc!r}"
        return entries, None

    def _record_probe(self, kind, phase=None):
        probe_start_us = time.monotonic_ns() // 1000
        entries, error = self._probe_once()
        probe_end_us = time.monotonic_ns() // 1000
        if error is not None:
            record = {
                "kind": "error", "board_uuid": self.gpu_uuid,
                "probe_start_us": probe_start_us,
                "probe_end_us": probe_end_us, "message": error}
        else:
            record = {
                "kind": kind, "board_uuid": self.gpu_uuid,
                "probe_start_us": probe_start_us,
                "probe_end_us": probe_end_us,
                "timestamp_us": probe_end_us, "processes": entries}
            if phase is not None:
                record["phase"] = phase
        with self._lock:
            self.records.append(record)
        return entries or [], error

    def record_boundary(self, phase):
        if phase not in ("READY", "DONE"):
            raise ExperimentError(f"invalid GPU process boundary: {phase}")
        return self._record_probe("boundary", phase)

    def _run(self):
        while not self._stop.is_set():
            self._record_probe("sample")
            self._stop.wait(self.interval_ms / 1000.0)

    def stop(self):
        self.stop_sampling()
        with self._lock:
            records = list(self.records)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="ascii") as handle:
            for record in records:
                handle.write(json.dumps(
                    record, sort_keys=True, separators=(",", ":")) + "\n")
        samples = [r for r in records if r["kind"] == "sample"]
        boundaries = [r for r in records if r["kind"] == "boundary"]
        errors = [r["message"] for r in records if r["kind"] == "error"]
        return {
            "path": str(self.output_path.resolve()),
            "record_count": len(records),
            "sha256": sha256_file(self.output_path),
            "errors": errors,
            "samples": samples,
            "boundaries": boundaries,
        }

    def stop_sampling(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise ExperimentError(
                    "GPU process monitor thread did not terminate")
            self._thread = None


def evaluate_process_telemetry(samples, driver_pid, errors=None, boundaries=None,
                               window_start_us=None, window_end_us=None,
                               min_updates=2):
    """Validity of continuous GPU-process telemetry. Pure and fail-closed.

    Invalid if the probe reported any error, if fewer than `min_updates` ticks
    were taken, if -- when a window is supplied -- successful samples do not
    bracket both window edges, if any inter-tick gap exceeds the frozen gap
    bound (a coverage hole across the paid window), or if any tick observed a
    compute app whose PID is not the exact driver PID (contamination).
    """
    reasons = []
    if errors:
        reasons.append(f"E_PROCESS_MONITOR_ERROR:{len(errors)}")
    count = len(samples) if type(samples) is list else 0
    if count < min_updates:
        reasons.append(f"E_PROCESS_MONITOR_UPDATES:{count}")
        return {"valid": False, "reasons": reasons, "foreign_pids": [],
                "max_gap_us": None}
    ordered = samples
    boundary_records = boundaries or []
    if boundaries is not None:
        phases = [record["phase"] for record in boundary_records]
        if phases != ["READY", "DONE"]:
            reasons.append(f"E_PROCESS_BOUNDARIES:{phases}")
    if window_start_us is not None and window_end_us is not None:
        if ordered[0]["timestamp_us"] > window_start_us:
            reasons.append("E_PROCESS_NO_LEADING_SAMPLE")
        if ordered[-1]["timestamp_us"] < window_end_us:
            reasons.append("E_PROCESS_NO_TRAILING_SAMPLE")
        if boundaries is not None and len(boundary_records) == 2:
            if boundary_records[0]["timestamp_us"] > window_start_us:
                reasons.append("E_PROCESS_READY_AFTER_WINDOW")
            if boundary_records[1]["timestamp_us"] < window_end_us:
                reasons.append("E_PROCESS_DONE_BEFORE_WINDOW")
    previous = None
    max_gap_us = 0
    for sample in ordered:
        timestamp_us = sample["timestamp_us"]
        if previous is not None:
            max_gap_us = max(max_gap_us, timestamp_us - previous)
        previous = timestamp_us
    if max_gap_us > MAX_SAMPLE_GAP_US:
        reasons.append(f"E_PROCESS_GAP:{max_gap_us}")
    foreign = sorted({
        entry["pid"]
        for sample in ordered + boundary_records
        for entry in sample["processes"]
        if entry["pid"] != driver_pid
    })
    if foreign:
        reasons.append(f"E_GPU_CONTAMINATION:{foreign}")
    return {"valid": not reasons, "reasons": reasons,
            "foreign_pids": foreign, "max_gap_us": max_gap_us}


def _require_process_artifact(value, field):
    if type(value) is not dict or set(value) != {
            "board_uuid", "path", "record_count", "sha256",
            "window_start_us", "window_end_us", "driver_pid"}:
        raise ExperimentError(f"{field} is not a complete process artifact")
    if type(value["board_uuid"]) is not str or not value["board_uuid"]:
        raise ExperimentError(f"{field}.board_uuid is invalid")
    if type(value["path"]) is not str or not value["path"]:
        raise ExperimentError(f"{field}.path is invalid")
    _require_exact_int(value["record_count"], f"{field}.record_count", minimum=2)
    if type(value["sha256"]) is not str or SHA256_PATTERN.fullmatch(
            value["sha256"]) is None:
        raise ExperimentError(f"{field}.sha256 is invalid")
    _require_exact_int(
        value["window_start_us"], f"{field}.window_start_us", minimum=0)
    _require_exact_int(
        value["window_end_us"], f"{field}.window_end_us", minimum=1)
    _require_exact_int(value["driver_pid"], f"{field}.driver_pid", minimum=1)
    if value["window_end_us"] <= value["window_start_us"]:
        raise ExperimentError(f"{field} window is empty or inverted")


def _validate_process_record(record, field):
    if type(record) is not dict or "kind" not in record:
        raise ExperimentError(f"{field} is not a process record")
    kind = record["kind"]
    if kind in ("sample", "boundary"):
        expected_fields = (
            PROCESS_SAMPLE_FIELDS if kind == "sample" else PROCESS_BOUNDARY_FIELDS)
        if set(record) != expected_fields:
            raise ExperimentError(f"{field} {kind} fields differ")
        for name in ("probe_start_us", "probe_end_us", "timestamp_us"):
            _require_exact_int(record[name], f"{field}.{name}", minimum=0)
        if record["probe_end_us"] < record["probe_start_us"]:
            raise ExperimentError(f"{field} probe chronology is inverted")
        if record["timestamp_us"] != record["probe_end_us"]:
            raise ExperimentError(f"{field}.timestamp_us is not probe completion")
        if kind == "boundary" and record["phase"] not in ("READY", "DONE"):
            raise ExperimentError(f"{field}.phase is invalid")
        if type(record["processes"]) is not list:
            raise ExperimentError(f"{field}.processes is not an array")
        for entry in record["processes"]:
            if type(entry) is not dict or set(entry) != PROCESS_ENTRY_FIELDS:
                raise ExperimentError(f"{field} process entry fields differ")
            _require_exact_int(entry["pid"], f"{field}.pid", minimum=0)
            _require_exact_int(
                entry["memory_used_mib"], f"{field}.memory_used_mib", minimum=0)
            if type(entry["name"]) is not str:
                raise ExperimentError(f"{field}.name is invalid")
    elif kind == "error":
        if set(record) != PROCESS_ERROR_FIELDS:
            raise ExperimentError(f"{field} error fields differ")
        for name in ("probe_start_us", "probe_end_us"):
            _require_exact_int(record[name], f"{field}.{name}", minimum=0)
        if record["probe_end_us"] < record["probe_start_us"]:
            raise ExperimentError(f"{field} probe chronology is inverted")
        if type(record["message"]) is not str or not record["message"]:
            raise ExperimentError(f"{field}.message is invalid")
    else:
        raise ExperimentError(f"{field} has an unknown record kind: {kind!r}")
    if type(record["board_uuid"]) is not str or not record["board_uuid"]:
        raise ExperimentError(f"{field}.board_uuid is invalid")
    return kind


def reintegrate_process_artifact(artifact, field):
    """Reopen the process telemetry ONCE as immutable bytes and recompute.

    Mirrors reintegrate_power_artifact: the stored process quality is never
    trusted. The bytes on disk are hashed, counted, strictly parsed by record
    kind, and re-evaluated for coverage, bracketing, gap, and contamination
    against the driver PID bound in the artifact. A byte-read failure, hash
    mismatch, record-count mismatch, or unparseable/duplicate-keyed line fails
    closed by raising; an under-covered or contaminated stream is a soft invalid
    outcome that is reported, not raised.
    """
    _require_process_artifact(artifact, field)
    path = pathlib.Path(artifact["path"])
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ExperimentError(f"{field} cannot be reopened: {exc}") from exc
    actual_sha = sha256_bytes(data)
    if actual_sha != artifact["sha256"]:
        raise ExperimentError(
            f"{field} sha256 mismatch: the bytes on disk hash to {actual_sha} "
            f"but the record binds {artifact['sha256']}")
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ExperimentError(f"{field} is not ASCII: {exc}") from exc
    records = [
        strict_json_loads(line, f"{field} record")
        for line in text.split("\n") if line
    ]
    if len(records) != artifact["record_count"]:
        raise ExperimentError(
            f"{field} record-count mismatch: the bytes on disk carry "
            f"{len(records)} records but the record binds "
            f"{artifact['record_count']}")
    samples = []
    boundaries = []
    errors = []
    board_uuids = set()
    previous_probe_end = None
    for index, record in enumerate(records):
        kind = _validate_process_record(record, f"{field} record {index}")
        if (previous_probe_end is not None and
                record["probe_start_us"] < previous_probe_end):
            raise ExperimentError(
                f"{field} record {index} overlaps or reorders prior probe")
        previous_probe_end = record["probe_end_us"]
        board_uuids.add(record["board_uuid"])
        if kind == "sample":
            samples.append(record)
        elif kind == "boundary":
            boundaries.append(record)
        else:
            errors.append(record["message"])
    if board_uuids != {artifact["board_uuid"]}:
        raise ExperimentError(
            f"{field} board UUID mismatch: bytes={sorted(board_uuids)} "
            f"artifact={artifact['board_uuid']}")
    quality = evaluate_process_telemetry(
        samples, artifact["driver_pid"], errors=errors, boundaries=boundaries,
        window_start_us=artifact["window_start_us"],
        window_end_us=artifact["window_end_us"])
    return {
        "reopened_bytes": len(data),
        "recomputed_sha256": actual_sha,
        "recomputed_record_count": len(records),
        "sample_count": len(samples),
        "boundary_count": len(boundaries),
        "error_count": len(errors),
        "board_uuid": next(iter(board_uuids)),
        "quality_valid": quality["valid"],
        "quality_reasons": quality["reasons"],
        "foreign_pids": quality["foreign_pids"],
        "max_gap_us": quality["max_gap_us"],
    }


THERMAL_MAX_GAP_US = 2_000_000
THERMAL_HEADER_FIELDS = frozenset({"kind", "boot_id", "serial"})
THERMAL_FOOTER_FIELDS = frozenset({"kind", "boot_id", "serial", "timestamp_us"})
THERMAL_SAMPLE_FIELDS = frozenset({
    "kind", "timestamp_us", "thermal_status", "temperatures"})
THERMAL_ERROR_FIELDS = frozenset({"kind", "timestamp_us", "message"})
THERMAL_TEMP_FIELDS = frozenset({"name", "temp_millic"})


def _thermal_logger_script(remote_log, remote_stop, serial):
    """A self-contained on-device thermal logger (CP1.3).

    Runs entirely inside one phone shell, so the host issues no periodic ADB
    calls during the paid window. It writes a header carrying boot identity, then
    one JSON record per tick with a monotonic /proc/uptime microsecond stamp, the
    Android thermal status, and every sysfs thermal-zone temperature. It stops
    when the host touches the stop flag (a single boundary ADB call).
    """
    log = shlex.quote(remote_log)
    stop = shlex.quote(remote_stop)
    serial_q = serial.replace('"', '')
    return (
        f'LOG={log}; STOP={stop}; rm -f "$LOG" "$STOP"; '
        f'boot=$(cat /proc/sys/kernel/random/boot_id) || exit 1; '
        f'printf \'{{"kind":"header","boot_id":"%s","serial":"%s"}}\\n\' '
        f'"$boot" "{serial_q}" > "$LOG"; '
        f'while [ ! -f "$STOP" ]; do '
        f'us=$(awk \'{{printf "%.0f", $1*1000000}}\' /proc/uptime); '
        f'st=$(dumpsys thermalservice 2>/dev/null | '
        f'awk \'/Thermal Status/{{gsub(/[^0-9]/, "", $0); print; exit}}\'); '
        f'temps=""; for z in /sys/class/thermal/thermal_zone*; do '
        f'IFS= read -r t < "$z/temp" 2>/dev/null || continue; '
        f'IFS= read -r n < "$z/type" 2>/dev/null || continue; '
        f'temps="$temps{{\\"name\\":\\"$n\\",\\"temp_millic\\":$t}},"; done; '
        f'temps=${{temps%,}}; '
        f'if [ -z "$st" ]; then '
        f'printf \'{{"kind":"error","message":"no thermal status",'
        f'"timestamp_us":%s}}\\n\' "$us" >> "$LOG"; else '
        f'printf \'{{"kind":"sample","temperatures":[%s],'
        f'"thermal_status":%s,"timestamp_us":%s}}\\n\' '
        f'"$temps" "$st" "$us" >> "$LOG"; fi; '
        f'sleep 0.5; done; '
        f'endboot=$(cat /proc/sys/kernel/random/boot_id) || exit 1; '
        f'us=$(awk \'{{printf "%.0f", $1*1000000}}\' /proc/uptime); '
        f'printf \'{{"kind":"footer","boot_id":"%s","serial":"%s",'
        f'"timestamp_us":%s}}\\n\' "$endboot" "{serial_q}" "$us" >> "$LOG"'
    )


class PhoneThermalLogger:
    """Launches and retrieves the on-device thermal logger for one timeline."""

    def __init__(self, serial, remote_dir, port, host_log_path):
        self.serial = serial
        self.remote_log = f"{remote_dir}/thermal_{port}.jsonl"
        self.remote_stop = f"{remote_dir}/thermal_{port}.stop"
        self.remote_script = f"{remote_dir}/thermal_{port}.sh"
        self.host_log_path = pathlib.Path(host_log_path)
        self.process = None
        self._log_handle = None

    def start(self):
        script = _thermal_logger_script(
            self.remote_log, self.remote_stop, self.serial)
        local_script = self.host_log_path.with_suffix(".sh")
        local_script.parent.mkdir(parents=True, exist_ok=True)
        local_script.write_text(script, encoding="ascii")
        adb(self.serial, "push", str(local_script), self.remote_script)
        self._log_handle = open(
            self.host_log_path.with_suffix(".launch.log"), "w",
            encoding="utf-8", errors="replace")
        self.process = subprocess.Popen(
            ["adb", "-s", self.serial, "shell",
             f"sh {shlex.quote(self.remote_script)}"],
            stdout=self._log_handle, stderr=subprocess.STDOUT, text=True)

    def stop_and_pull(self):
        if self.process is None:
            raise ExperimentError("thermal logger was not started")
        adb(self.serial, "shell", f"touch {shlex.quote(self.remote_stop)}",
            check=False)
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        self.host_log_path.parent.mkdir(parents=True, exist_ok=True)
        adb(self.serial, "pull", self.remote_log, str(self.host_log_path))
        adb(self.serial, "shell",
            f"rm -f {shlex.quote(self.remote_log)} "
            f"{shlex.quote(self.remote_stop)} "
            f"{shlex.quote(self.remote_script)}", check=False)
        if self._log_handle is not None:
            self._log_handle.close()
        line_count = sum(
            1 for _ in self.host_log_path.read_text(
                encoding="ascii", errors="replace").split("\n") if _)
        return {
            "serial": self.serial,
            "path": str(self.host_log_path.resolve()),
            "record_count": line_count,
            "sha256": sha256_file(self.host_log_path),
        }


def _validate_thermal_record(record, field):
    if type(record) is not dict or "kind" not in record:
        raise ExperimentError(f"{field} is not a thermal record")
    kind = record["kind"]
    if kind == "header":
        if set(record) != THERMAL_HEADER_FIELDS:
            raise ExperimentError(f"{field} header fields differ")
        if (type(record["boot_id"]) is not str or
                BOOT_UUID_PATTERN.fullmatch(record["boot_id"]) is None):
            raise ExperimentError(f"{field}.boot_id is invalid")
        if type(record["serial"]) is not str or not record["serial"]:
            raise ExperimentError(f"{field}.serial is invalid")
    elif kind == "footer":
        if set(record) != THERMAL_FOOTER_FIELDS:
            raise ExperimentError(f"{field} footer fields differ")
        if (type(record["boot_id"]) is not str or
                BOOT_UUID_PATTERN.fullmatch(record["boot_id"]) is None):
            raise ExperimentError(f"{field}.boot_id is invalid")
        if type(record["serial"]) is not str or not record["serial"]:
            raise ExperimentError(f"{field}.serial is invalid")
        _require_exact_int(record["timestamp_us"], f"{field}.timestamp_us", 0)
    elif kind == "sample":
        if set(record) != THERMAL_SAMPLE_FIELDS:
            raise ExperimentError(f"{field} sample fields differ")
        _require_exact_int(record["timestamp_us"], f"{field}.timestamp_us", 0)
        _require_exact_int(record["thermal_status"], f"{field}.thermal_status", 0)
        if type(record["temperatures"]) is not list:
            raise ExperimentError(f"{field}.temperatures is not an array")
        for entry in record["temperatures"]:
            if type(entry) is not dict or set(entry) != THERMAL_TEMP_FIELDS:
                raise ExperimentError(f"{field} temperature fields differ")
            if type(entry["name"]) is not str or not entry["name"]:
                raise ExperimentError(f"{field} temperature name is invalid")
            _require_exact_int(entry["temp_millic"], f"{field}.temp_millic")
    elif kind == "error":
        if set(record) != THERMAL_ERROR_FIELDS:
            raise ExperimentError(f"{field} error fields differ")
        _require_exact_int(record["timestamp_us"], f"{field}.timestamp_us", 0)
        if type(record["message"]) is not str or not record["message"]:
            raise ExperimentError(f"{field}.message is invalid")
    else:
        raise ExperimentError(f"{field} has an unknown record kind: {kind!r}")
    return kind


def evaluate_thermal_telemetry(headers, footers, samples, errors, serial,
                               window_start_us, window_end_us, min_updates=2,
                               sequence_reasons=None):
    """Validity of the on-device thermal log. Pure and fail-closed.

    Invalid if the logger reported any error, if the boot-identity header is
    missing or names another phone, if fewer than `min_updates` samples were
    taken, if successful samples do not bracket both window edges, if any gap
    across the window exceeds the frozen thermal gap bound, if any sample shows a
    non-zero Android thermal status, or if any sample carries no sensors.
    """
    reasons = list(sequence_reasons or [])
    if errors:
        reasons.append(f"E_THERMAL_LOGGER_ERROR:{len(errors)}")
    if len(headers) != 1:
        reasons.append(f"E_THERMAL_HEADER:{len(headers)}")
    elif headers[0]["serial"] != serial:
        reasons.append(
            f"E_THERMAL_SERIAL:{headers[0]['serial']}!={serial}")
    if len(footers) != 1:
        reasons.append(f"E_THERMAL_FOOTER:{len(footers)}")
    elif footers[0]["serial"] != serial:
        reasons.append(
            f"E_THERMAL_FOOTER_SERIAL:{footers[0]['serial']}!={serial}")
    if len(headers) == 1 and len(footers) == 1:
        if headers[0]["boot_id"] != footers[0]["boot_id"]:
            reasons.append("E_THERMAL_BOOT_CHANGED")
    count = len(samples)
    if count < min_updates:
        reasons.append(f"E_THERMAL_UPDATES:{count}")
        return {"valid": False, "reasons": reasons, "max_gap_us": None,
                "boot_id": headers[0]["boot_id"] if len(headers) == 1 else None}
    ordered = samples
    if ordered[0]["timestamp_us"] > window_start_us:
        reasons.append("E_THERMAL_NO_LEADING_SAMPLE")
    if ordered[-1]["timestamp_us"] < window_end_us:
        reasons.append("E_THERMAL_NO_TRAILING_SAMPLE")
    previous = None
    max_gap_us = 0
    for sample in ordered:
        if previous is not None:
            max_gap_us = max(max_gap_us, sample["timestamp_us"] - previous)
        previous = sample["timestamp_us"]
    if max_gap_us > THERMAL_MAX_GAP_US:
        reasons.append(f"E_THERMAL_GAP:{max_gap_us}")
    if any(sample["thermal_status"] != 0 for sample in ordered):
        reasons.append("E_THERMAL_STATUS_NONZERO")
    if any(not sample["temperatures"] for sample in ordered):
        reasons.append("E_THERMAL_SENSORS_EMPTY")
    return {"valid": not reasons, "reasons": reasons, "max_gap_us": max_gap_us,
            "boot_id": headers[0]["boot_id"] if len(headers) == 1 else None}


def _require_thermal_artifact(value, field):
    if type(value) is not dict or set(value) != {
            "serial", "path", "record_count", "sha256",
            "window_start_us", "window_end_us"}:
        raise ExperimentError(f"{field} is not a complete thermal artifact")
    if type(value["serial"]) is not str or not value["serial"]:
        raise ExperimentError(f"{field}.serial is invalid")
    if type(value["path"]) is not str or not value["path"]:
        raise ExperimentError(f"{field}.path is invalid")
    _require_exact_int(value["record_count"], f"{field}.record_count", minimum=4)
    if type(value["sha256"]) is not str or SHA256_PATTERN.fullmatch(
            value["sha256"]) is None:
        raise ExperimentError(f"{field}.sha256 is invalid")
    _require_exact_int(
        value["window_start_us"], f"{field}.window_start_us", minimum=0)
    _require_exact_int(
        value["window_end_us"], f"{field}.window_end_us", minimum=1)
    if value["window_end_us"] <= value["window_start_us"]:
        raise ExperimentError(f"{field} window is empty or inverted")


def reintegrate_thermal_artifact(artifact, field):
    """Reopen the on-device thermal log ONCE as immutable bytes and recompute.

    Same contract as the power and process reintegrations: a byte-read failure,
    hash mismatch, record-count mismatch, or unparseable/duplicate-keyed line
    fails closed by raising; an uncovered, throttled, or sensorless stream is a
    soft invalid outcome that is reported, not raised. The boot-identity header
    must name the phone bound in the artifact.
    """
    _require_thermal_artifact(artifact, field)
    path = pathlib.Path(artifact["path"])
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ExperimentError(f"{field} cannot be reopened: {exc}") from exc
    actual_sha = sha256_bytes(data)
    if actual_sha != artifact["sha256"]:
        raise ExperimentError(
            f"{field} sha256 mismatch: the bytes on disk hash to {actual_sha} "
            f"but the record binds {artifact['sha256']}")
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ExperimentError(f"{field} is not ASCII: {exc}") from exc
    records = [
        strict_json_loads(line, f"{field} record")
        for line in text.split("\n") if line
    ]
    if len(records) != artifact["record_count"]:
        raise ExperimentError(
            f"{field} record-count mismatch: the bytes on disk carry "
            f"{len(records)} records but the record binds "
            f"{artifact['record_count']}")
    headers = []
    footers = []
    samples = []
    errors = []
    sequence_reasons = []
    previous_timestamp = None
    for index, record in enumerate(records):
        kind = _validate_thermal_record(record, f"{field} record {index}")
        if kind == "header":
            headers.append(record)
            if index != 0:
                sequence_reasons.append("E_THERMAL_HEADER_ORDER")
        elif kind == "footer":
            footers.append(record)
            if index != len(records) - 1:
                sequence_reasons.append("E_THERMAL_FOOTER_ORDER")
        elif kind == "sample":
            samples.append(record)
        else:
            errors.append(record)
        if kind in ("sample", "error", "footer"):
            timestamp = record["timestamp_us"]
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                sequence_reasons.append(
                    f"E_THERMAL_TIMESTAMP_ORDER:{index}")
            previous_timestamp = timestamp
    quality = evaluate_thermal_telemetry(
        headers, footers, samples, errors, artifact["serial"],
        artifact["window_start_us"], artifact["window_end_us"],
        sequence_reasons=sequence_reasons)
    return {
        "reopened_bytes": len(data),
        "recomputed_sha256": actual_sha,
        "recomputed_record_count": len(records),
        "sample_count": len(samples),
        "error_count": len(errors),
        "quality_valid": quality["valid"],
        "quality_reasons": quality["reasons"],
        "max_gap_us": quality["max_gap_us"],
        "boot_id": quality["boot_id"],
    }


PLACEMENT_CERT_SCHEMA = "layersplit-scheduled-placement-v2"
PLACEMENT_ROLES = frozenset({"phone_stage", "host_tail", "monodriver"})
PLACEMENT_CERT_FIELDS = frozenset({
    "schema", "role", "mode", "layer_start", "layer_end", "n_layer",
    "pid", "run_rc", "compute_nodes", "copy_nodes", "metadata_nodes",
    "missing_buffer_compute_nodes", "compute_by_buffer_type", "compute_by_op",
    "compute_by_op_and_buffer", "copy_by_buffer_type", "status"})
PLACEMENT_RECORD_FIELDS = frozenset({"source", "certificate"})
PLACEMENT_DESCRIPTOR_FIELDS = frozenset({
    "source", "role", "mode", "layer_start", "layer_end", "n_layer",
    "default_compute_buffer_type", "compute_buffer_overrides", "pid"})


def _validate_placement_cert(cert, field):
    if type(cert) is not dict or set(cert) != PLACEMENT_CERT_FIELDS:
        raise ExperimentError(f"{field} fields differ")
    if cert["schema"] != PLACEMENT_CERT_SCHEMA:
        raise ExperimentError(f"{field}.schema is invalid")
    if cert["role"] not in PLACEMENT_ROLES:
        raise ExperimentError(f"{field}.role is invalid")
    if type(cert["mode"]) is not str or not cert["mode"]:
        raise ExperimentError(f"{field}.mode is invalid")
    for name in ("layer_start", "layer_end", "n_layer", "compute_nodes",
                 "copy_nodes", "metadata_nodes", "missing_buffer_compute_nodes",
                 "pid"):
        _require_exact_int(cert[name], f"{field}.{name}", minimum=0)
    _require_exact_int(cert["run_rc"], f"{field}.run_rc")
    if cert["layer_end"] <= cert["layer_start"]:
        raise ExperimentError(f"{field} layer range is empty or inverted")
    for name in ("compute_by_buffer_type", "compute_by_op", "copy_by_buffer_type"):
        counts = cert[name]
        if (type(counts) is not dict or
                any(type(key) is not str or not key or type(value) is not int or
                    value < 0 for key, value in counts.items())):
            raise ExperimentError(f"{field}.{name} is invalid")
    nested = cert["compute_by_op_and_buffer"]
    if type(nested) is not dict:
        raise ExperimentError(f"{field}.compute_by_op_and_buffer is invalid")
    nested_total = 0
    for op_name, buffer_counts in nested.items():
        if type(op_name) is not str or not op_name or type(buffer_counts) is not dict:
            raise ExperimentError(f"{field}.compute_by_op_and_buffer is invalid")
        if any(type(name) is not str or not name or type(count) is not int or
               count < 0 for name, count in buffer_counts.items()):
            raise ExperimentError(f"{field}.compute_by_op_and_buffer is invalid")
        op_total = sum(buffer_counts.values())
        if cert["compute_by_op"].get(op_name) != op_total:
            raise ExperimentError(
                f"{field}.compute_by_op_and_buffer count differs for {op_name}")
        nested_total += op_total
    if set(nested) != set(cert["compute_by_op"]):
        raise ExperimentError(f"{field}.compute_by_op_and_buffer ops differ")
    if sum(cert["compute_by_buffer_type"].values()) != cert["compute_nodes"]:
        raise ExperimentError(f"{field}.compute_by_buffer_type count differs")
    if sum(cert["compute_by_op"].values()) != cert["compute_nodes"]:
        raise ExperimentError(f"{field}.compute_by_op count differs")
    if nested_total != cert["compute_nodes"]:
        raise ExperimentError(f"{field}.compute_by_op_and_buffer count differs")
    if sum(cert["copy_by_buffer_type"].values()) != cert["copy_nodes"]:
        raise ExperimentError(f"{field}.copy_by_buffer_type count differs")
    if type(cert["status"]) is not str or not cert["status"]:
        raise ExperimentError(f"{field}.status is invalid")
    return cert["role"]


def parse_placement_certs(text, source):
    """Strictly parse every PLACEMENTCERT line emitted on a stream."""
    certs = []
    for line in text.splitlines():
        if line.startswith("PLACEMENTCERT "):
            cert = strict_json_loads(
                line[len("PLACEMENTCERT "):], f"{source} placement cert")
            _validate_placement_cert(cert, f"{source} placement cert")
            certs.append(cert)
    return certs


def _validate_placement_descriptor(descriptor, field):
    if type(descriptor) is not dict or set(descriptor) != PLACEMENT_DESCRIPTOR_FIELDS:
        raise ExperimentError(f"{field} fields differ")
    for name in ("source", "mode", "default_compute_buffer_type"):
        if type(descriptor[name]) is not str or not descriptor[name]:
            raise ExperimentError(f"{field}.{name} is invalid")
    if descriptor["role"] not in PLACEMENT_ROLES:
        raise ExperimentError(f"{field}.role is invalid")
    for name in ("layer_start", "layer_end", "n_layer"):
        _require_exact_int(descriptor[name], f"{field}.{name}", minimum=0)
    if descriptor["layer_end"] <= descriptor["layer_start"]:
        raise ExperimentError(f"{field} layer range is empty or inverted")
    if descriptor["pid"] is not None:
        _require_exact_int(descriptor["pid"], f"{field}.pid", minimum=1)
    overrides = descriptor["compute_buffer_overrides"]
    if type(overrides) is not dict:
        raise ExperimentError(f"{field}.compute_buffer_overrides is invalid")
    for op_name, buffers in overrides.items():
        if (type(op_name) is not str or not op_name or type(buffers) is not list or
                not buffers or any(type(name) is not str or not name for name in buffers) or
                len(buffers) != len(set(buffers))):
            raise ExperimentError(f"{field}.compute_buffer_overrides is invalid")


def evaluate_placement_cert(cert, descriptor=None):
    """Validate one scheduled-buffer observation against an independent route."""
    _validate_placement_cert(cert, "placement certificate")
    reasons = []
    if cert["status"] != "SCHEDULED_PLACEMENT_OK":
        reasons.append(f"E_PLACEMENT_STATUS:{cert['status']}")
    if cert["run_rc"] != 0:
        reasons.append(f"E_PLACEMENT_RUN_RC:{cert['run_rc']}")
    if cert["compute_nodes"] <= 0:
        reasons.append("E_PLACEMENT_NO_COMPUTE")
    if cert["missing_buffer_compute_nodes"] != 0:
        reasons.append(
            f"E_PLACEMENT_MISSING_BUFFER:{cert['missing_buffer_compute_nodes']}")
    if descriptor is None:
        reasons.append("E_PLACEMENT_EXPECTATION_MISSING")
    else:
        _validate_placement_descriptor(descriptor, "placement descriptor")
        for name in ("role", "mode", "layer_start", "layer_end", "n_layer"):
            if cert[name] != descriptor[name]:
                reasons.append(
                    f"E_PLACEMENT_{name.upper()}:{cert[name]}!={descriptor[name]}")
        if descriptor["pid"] is not None and cert["pid"] != descriptor["pid"]:
            reasons.append(f"E_PLACEMENT_PID:{cert['pid']}!={descriptor['pid']}")
        default_buffer = descriptor["default_compute_buffer_type"]
        overrides = descriptor["compute_buffer_overrides"]
        for op_name, observed_counts in cert["compute_by_op_and_buffer"].items():
            allowed = set(overrides.get(op_name, [default_buffer]))
            unexpected = set(observed_counts) - allowed
            if unexpected:
                reasons.append(
                    f"E_PLACEMENT_BUFFER:{op_name}:{sorted(unexpected)}")
    return {"valid": not reasons, "reasons": reasons}


def _require_placement_artifact(value, field):
    if type(value) is not dict or set(value) != {
            "path", "record_count", "sha256"}:
        raise ExperimentError(f"{field} is not a complete placement artifact")
    if type(value["path"]) is not str or not value["path"]:
        raise ExperimentError(f"{field}.path is invalid")
    _require_exact_int(value["record_count"], f"{field}.record_count", minimum=1)
    if type(value["sha256"]) is not str or SHA256_PATTERN.fullmatch(
            value["sha256"]) is None:
        raise ExperimentError(f"{field}.sha256 is invalid")


def reintegrate_placement_artifact(artifact, expected_descriptors, field):
    """Reopen the placement certificates ONCE as immutable bytes and recompute.

    Same contract as the other reintegrations. A byte-read failure, hash
    mismatch, record-count mismatch, or unparseable/duplicate-keyed line fails
    closed by raising. A missing expected role, a duplicated role, an unexpected
    role, a zero-compute, CPU-fallback, or wrong-backend certificate is a soft
    invalid outcome reported in the reasons, not raised.
    """
    _require_placement_artifact(artifact, field)
    if type(expected_descriptors) is not list or not expected_descriptors:
        raise ExperimentError(f"{field} expected descriptors are missing")
    expected_by_source = {}
    for index, descriptor in enumerate(expected_descriptors):
        _validate_placement_descriptor(
            descriptor, f"{field} descriptor {index}")
        source = descriptor["source"]
        if source in expected_by_source:
            raise ExperimentError(f"{field} duplicates descriptor source {source}")
        expected_by_source[source] = descriptor
    path = pathlib.Path(artifact["path"])
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ExperimentError(f"{field} cannot be reopened: {exc}") from exc
    actual_sha = sha256_bytes(data)
    if actual_sha != artifact["sha256"]:
        raise ExperimentError(
            f"{field} sha256 mismatch: the bytes on disk hash to {actual_sha} "
            f"but the record binds {artifact['sha256']}")
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ExperimentError(f"{field} is not ASCII: {exc}") from exc
    records = [
        strict_json_loads(line, f"{field} record")
        for line in text.split("\n") if line
    ]
    if len(records) != artifact["record_count"]:
        raise ExperimentError(
            f"{field} record-count mismatch: the bytes on disk carry "
            f"{len(records)} certificates but the record binds "
            f"{artifact['record_count']}")
    reasons = []
    by_source = {}
    for index, record in enumerate(records):
        if type(record) is not dict or set(record) != PLACEMENT_RECORD_FIELDS:
            raise ExperimentError(f"{field} record {index} fields differ")
        source = record["source"]
        if type(source) is not str or not source:
            raise ExperimentError(f"{field} record {index}.source is invalid")
        cert = record["certificate"]
        _validate_placement_cert(cert, f"{field} record {index}.certificate")
        if source in by_source:
            reasons.append(f"E_PLACEMENT_DUPLICATE:{source}")
        by_source[source] = cert
    for source, descriptor in expected_by_source.items():
        cert = by_source.get(source)
        if cert is None:
            reasons.append(f"E_PLACEMENT_MISSING:{source}")
            continue
        outcome = evaluate_placement_cert(cert, descriptor)
        if not outcome["valid"]:
            reasons.extend(f"{source}:{reason}" for reason in outcome["reasons"])
    for source in by_source:
        if source not in expected_by_source:
            reasons.append(f"E_PLACEMENT_UNEXPECTED:{source}")
    return {
        "reopened_bytes": len(data),
        "recomputed_sha256": actual_sha,
        "recomputed_record_count": len(records),
        "quality_valid": not reasons,
        "quality_reasons": reasons,
        "sources": sorted(by_source),
    }


def power_limit_invariant(observed_limits, declared_limit_mw):
    """Require exactly one power limit through a slot, matching the declaration.

    CP1.4: the GPU power limit must not move within a slot. `observed_limits` is
    the set of power.limit values seen across the continuously sampled window.
    """
    reasons = []
    limits = sorted(set(observed_limits))
    if not limits:
        reasons.append("E_POWER_LIMIT_UNOBSERVED")
    elif len(limits) != 1:
        reasons.append(f"E_POWER_LIMIT_CHANGED:{limits}")
    elif limits[0] != declared_limit_mw:
        reasons.append(
            f"E_POWER_LIMIT_MISMATCH:{limits[0]}!={declared_limit_mw}")
    return {"valid": not reasons, "reasons": reasons,
            "power_limit_mw": limits[0] if limits else None}


POWER_SAMPLE_FIELDS = frozenset({
    "kind", "board_uuid", "timestamp_us", "power_mw", "pstate",
    "utilization_pct", "memory_used_mib", "power_limit_mw",
})
POWER_ERROR_FIELDS = frozenset({
    "kind", "board_uuid", "timestamp_us", "message"})


def _validate_power_record(record, field):
    if type(record) is not dict or "kind" not in record:
        raise ExperimentError(f"{field} is not a power record")
    if record["kind"] == "sample":
        validate_power_samples([record])
    elif record["kind"] == "error":
        if set(record) != POWER_ERROR_FIELDS:
            raise ExperimentError(f"{field} error fields differ")
        if type(record["board_uuid"]) is not str or not record["board_uuid"]:
            raise ExperimentError(f"{field}.board_uuid is invalid")
        _require_exact_int(record["timestamp_us"], f"{field}.timestamp_us", 0)
        if type(record["message"]) is not str or not record["message"]:
            raise ExperimentError(f"{field}.message is invalid")
    else:
        raise ExperimentError(f"{field} has an unknown record kind")


def validate_power_samples(samples):
    """Strict schema for a raw NVML power row (CP1.3 / power-limit evidence).

    Every raw sample must carry the exact six fields nvidia-smi emits and each
    must have its exact integer type, including power_limit_mw, utilization_pct,
    and memory_used_mib. Anything short of that -- a missing field, an extra
    field, a float where an int is required, or a bool masquerading as an int
    (`type() is int` excludes bool) -- fails closed, so a stream that forged or
    dropped the power limit cannot be integrated.
    """
    if not isinstance(samples, list):
        raise ExperimentError("measurement samples are not an array")
    previous_timestamp = None
    for index, sample in enumerate(samples):
        if type(sample) is not dict:
            raise ExperimentError(f"measurement sample {index} is not an object")
        if set(sample) != POWER_SAMPLE_FIELDS:
            raise ExperimentError(
                f"measurement sample {index} fields differ: "
                f"missing={sorted(POWER_SAMPLE_FIELDS - set(sample))}, "
                f"extra={sorted(set(sample) - POWER_SAMPLE_FIELDS)}")
        timestamp_us = sample["timestamp_us"]
        power_mw = sample["power_mw"]
        pstate = sample["pstate"]
        utilization_pct = sample["utilization_pct"]
        memory_used_mib = sample["memory_used_mib"]
        power_limit_mw = sample["power_limit_mw"]
        if sample["kind"] != "sample":
            raise ExperimentError(
                f"measurement sample {index} kind is invalid")
        if type(sample["board_uuid"]) is not str or not sample["board_uuid"]:
            raise ExperimentError(
                f"measurement sample {index} board_uuid is invalid")
        if type(timestamp_us) is not int or timestamp_us < 0:
            raise ExperimentError(
                f"measurement sample {index} timestamp_us is invalid")
        if previous_timestamp is not None and timestamp_us <= previous_timestamp:
            raise ExperimentError(
                f"measurement sample {index} timestamp_us is not increasing")
        if type(power_mw) is not int or power_mw < 0:
            raise ExperimentError(
                f"measurement sample {index} power_mw is invalid")
        if type(pstate) is not str or PSTATE_PATTERN.fullmatch(pstate) is None:
            raise ExperimentError(
                f"measurement sample {index} pstate is invalid")
        if type(utilization_pct) is not int or not 0 <= utilization_pct <= 100:
            raise ExperimentError(
                f"measurement sample {index} utilization_pct is invalid")
        if type(memory_used_mib) is not int or memory_used_mib < 0:
            raise ExperimentError(
                f"measurement sample {index} memory_used_mib is invalid")
        if type(power_limit_mw) is not int or power_limit_mw < 1:
            raise ExperimentError(
                f"measurement sample {index} power_limit_mw is invalid")
        previous_timestamp = timestamp_us


def bracketed_samples(samples, window_start_us, window_end_us):
    validate_power_samples(samples)
    if type(window_start_us) is not int or type(window_end_us) is not int:
        raise ExperimentError("measurement window timestamps are invalid")
    if window_end_us <= window_start_us:
        raise ExperimentError("measurement window is empty")
    if len(samples) < 2:
        raise ExperimentError("measurement has fewer than two samples")
    if samples[0]["timestamp_us"] > window_start_us:
        raise ExperimentError("measurement window lacks a leading sample")
    if samples[-1]["timestamp_us"] < window_end_us:
        raise ExperimentError("measurement window lacks a trailing sample")
    first = max(i for i, sample in enumerate(samples)
                if sample["timestamp_us"] <= window_start_us)
    last = min(i for i, sample in enumerate(samples)
               if sample["timestamp_us"] >= window_end_us)
    return samples[first:last + 1]


def integrate_zoh(samples, window_start_us, window_end_us):
    selected = bracketed_samples(samples, window_start_us, window_end_us)
    energy_nj = 0
    for left, right in zip(selected, selected[1:]):
        start_us = max(left["timestamp_us"], window_start_us)
        end_us = min(right["timestamp_us"], window_end_us)
        if end_us > start_us:
            energy_nj += left["power_mw"] * (end_us - start_us)
    return energy_nj


def nvml_uncertainty(window_us, power_limit_mw):
    if type(window_us) is not int or window_us <= 0:
        raise ExperimentError("NVML uncertainty window is invalid")
    if type(power_limit_mw) is not int or power_limit_mw <= 0:
        raise ExperimentError("NVML power limit is invalid")
    accuracy_nj = NVML_UNCERTAINTY_MW * window_us
    boundary_nj = 2 * power_limit_mw * NVML_AVERAGING_WINDOW_US
    return {
        "nvml_accuracy": accuracy_nj,
        "one_second_average_boundaries": boundary_nj,
        "total": accuracy_nj + boundary_nj,
    }


def measurement_quality(samples, window_start_us, window_end_us):
    try:
        selected = bracketed_samples(samples, window_start_us, window_end_us)
    except ExperimentError as exc:
        return {"valid": False, "reasons": [str(exc)]}
    reasons = []
    gaps = [
        right["timestamp_us"] - left["timestamp_us"]
        for left, right in zip(selected, selected[1:])
    ]
    max_gap_us = max(gaps) if gaps else 0
    if max_gap_us > MAX_SAMPLE_GAP_US:
        reasons.append(f"E_GAP:{max_gap_us}")

    updates = 0
    previous = selected[0]["power_mw"]
    for sample in selected[1:]:
        if sample["timestamp_us"] >= window_end_us:
            break
        if sample["power_mw"] != previous:
            updates += 1
        previous = sample["power_mw"]
    if updates < MIN_INDEPENDENT_UPDATES:
        reasons.append(f"E_UPDATES:{updates}")

    pstates = sorted({
        left["pstate"]
        for left, right in zip(selected, selected[1:])
        if min(right["timestamp_us"], window_end_us)
        > max(left["timestamp_us"], window_start_us)
    })
    pstate_transitions = sum(
        left["pstate"] != right["pstate"]
        and window_start_us < right["timestamp_us"] < window_end_us
        for left, right in zip(selected, selected[1:])
    )
    return {
        "valid": not reasons,
        "reasons": reasons,
        "independent_updates": updates,
        "max_gap_us": max_gap_us,
        "pstate_policy": "OBSERVED_OUTCOME_TRANSITIONS_ALLOWED",
        "pstate_transitions": pstate_transitions,
        "pstates": pstates,
    }


def parse_route_records(stderr_text):
    records = []
    done = None
    for line in stderr_text.splitlines():
        if line.startswith("ROUTEJSON "):
            record = strict_json_loads(
                line[len("ROUTEJSON "):], "ROUTEJSON")
            required = {
                "status", "route", "request_index", "prompt_tokens",
                "batch_index", "batch_size", "stream_index",
                "requested_tokens", "generated_tokens", "eog", "prefill_us",
                "decode_us", "request_wall_us", "stage_a_us", "stage_b_us",
                "host_us", "token_ids",
            }
            if set(record) != required:
                raise ExperimentError(
                    f"ROUTEJSON fields differ: missing={sorted(required - set(record))}, "
                    f"extra={sorted(set(record) - required)}")
            if record["status"] != "ok":
                raise ExperimentError("ROUTEJSON status is not ok")
            if not isinstance(record["route"], str) or not record["route"]:
                raise ExperimentError("ROUTEJSON route is invalid")
            if type(record["request_index"]) is not int or record["request_index"] < 0:
                raise ExperimentError("ROUTEJSON request_index is invalid")
            if type(record["batch_index"]) is not int or record["batch_index"] < 0:
                raise ExperimentError("ROUTEJSON batch_index is invalid")
            if type(record["batch_size"]) is not int or record["batch_size"] <= 0:
                raise ExperimentError("ROUTEJSON batch_size is invalid")
            if (type(record["stream_index"]) is not int or
                    record["stream_index"] < 0 or
                    record["stream_index"] >= record["batch_size"]):
                raise ExperimentError("ROUTEJSON stream_index is invalid")
            expected_index = (
                record["batch_index"] * record["batch_size"] +
                record["stream_index"])
            if record["request_index"] != expected_index:
                raise ExperimentError("ROUTEJSON request/batch/stream identity mismatch")
            positive_fields = ("prompt_tokens", "requested_tokens", "request_wall_us")
            for field in positive_fields:
                if type(record[field]) is not int or record[field] <= 0:
                    raise ExperimentError(f"ROUTEJSON {field} is invalid")
            nonnegative_fields = (
                "generated_tokens", "prefill_us", "decode_us",
                "stage_a_us", "stage_b_us", "host_us")
            for field in nonnegative_fields:
                if type(record[field]) is not int or record[field] < 0:
                    raise ExperimentError(f"ROUTEJSON {field} is invalid")
            if type(record["eog"]) is not int or record["eog"] not in (0, 1):
                raise ExperimentError("ROUTEJSON eog is invalid")
            if (record["generated_tokens"] <= 0 or
                    record["generated_tokens"] > record["requested_tokens"]):
                raise ExperimentError("ROUTEJSON generated token count is invalid")
            compute_us = record["prefill_us"] + record["decode_us"]
            route_us = (
                record["stage_a_us"] + record["stage_b_us"] + record["host_us"])
            if compute_us != route_us:
                raise ExperimentError("ROUTEJSON route timing does not close")
            if record["request_wall_us"] < compute_us:
                raise ExperimentError("ROUTEJSON request wall is shorter than compute")
            if not isinstance(record["token_ids"], list):
                raise ExperimentError("ROUTEJSON token_ids is not an array")
            if record["generated_tokens"] != len(record["token_ids"]):
                raise ExperimentError("ROUTEJSON generated_tokens mismatch")
            if any(type(token) is not int for token in record["token_ids"]):
                raise ExperimentError("ROUTEJSON token_ids contains a non-integer")
            records.append(record)
        elif line.startswith("DRIVER_DONE "):
            if done is not None:
                raise ExperimentError("multiple DRIVER_DONE records")
            done = strict_json_loads(
                line[len("DRIVER_DONE "):], "DRIVER_DONE")
            required_done = {"status", "route", "requests"}
            if set(done) != required_done:
                raise ExperimentError(
                    "DRIVER_DONE fields differ: "
                    f"missing={sorted(required_done - set(done))}, "
                    f"extra={sorted(set(done) - required_done)}")
            if done["status"] != "ok":
                raise ExperimentError("DRIVER_DONE status is not ok")
            if type(done["route"]) is not str or not done["route"]:
                raise ExperimentError("DRIVER_DONE route is invalid")
            if type(done["requests"]) is not int or done["requests"] <= 0:
                raise ExperimentError("DRIVER_DONE requests is invalid")
    if done is None:
        raise ExperimentError("missing DRIVER_DONE record")
    records.sort(key=lambda record: record["request_index"])
    if [record["request_index"] for record in records] != list(range(len(records))):
        raise ExperimentError("ROUTEJSON request indexes are not contiguous")
    if done["requests"] != len(records):
        raise ExperimentError("DRIVER_DONE does not match ROUTEJSON records")
    if any(record["route"] != done["route"] for record in records):
        raise ExperimentError("route differs between ROUTEJSON and DRIVER_DONE")
    batch_sizes = {record["batch_size"] for record in records}
    if len(batch_sizes) != 1:
        raise ExperimentError("ROUTEJSON records mix batch sizes")
    return records


def same_work(control_records, treatment_records):
    if len(control_records) != len(treatment_records):
        return False, "request_count"
    fields = (
        "batch_index", "batch_size", "stream_index", "prompt_tokens",
        "requested_tokens", "generated_tokens", "eog", "token_ids")
    for index, (control, treatment) in enumerate(zip(control_records, treatment_records)):
        for field in fields:
            if control[field] != treatment[field]:
                return False, f"request_{index}:{field}"
    return True, "exact"


def batch_metrics(records):
    if not records:
        raise ExperimentError("cannot summarize empty records")
    batch_size = records[0]["batch_size"]
    groups = {}
    for record in records:
        if record["batch_size"] != batch_size:
            raise ExperimentError("records mix batch sizes")
        groups.setdefault(record["batch_index"], []).append(record)
    if sorted(groups) != list(range(len(groups))):
        raise ExperimentError("batch indexes are not contiguous")

    group_records = []
    timing_fields = (
        "prefill_us", "decode_us", "request_wall_us",
        "stage_a_us", "stage_b_us", "host_us")
    for batch_index in sorted(groups):
        group = sorted(groups[batch_index], key=lambda record: record["stream_index"])
        if len(group) != batch_size:
            raise ExperimentError(
                f"batch {batch_index} has {len(group)} records, expected {batch_size}")
        if [record["stream_index"] for record in group] != list(range(batch_size)):
            raise ExperimentError(f"batch {batch_index} stream indexes are incomplete")
        for field in timing_fields:
            if len({record[field] for record in group}) != 1:
                raise ExperimentError(
                    f"batch {batch_index} does not share group timing field {field}")
        group_records.append({
            "batch_index": batch_index,
            **{field: group[0][field] for field in timing_fields},
        })

    group_wall_us = [record["request_wall_us"] for record in group_records]
    median_group_wall_us = int(statistics.median(group_wall_us))
    sorted_group_wall_us = sorted(group_wall_us)
    p95_index = (95 * len(sorted_group_wall_us) + 99) // 100 - 1
    return {
        "batch_size": batch_size,
        "group_count": len(group_records),
        "group_records": group_records,
        "median_group_wall_us": median_group_wall_us,
        "p95_group_wall_us": sorted_group_wall_us[p95_index],
        "median_useful_requests_per_s": (
            batch_size * 1_000_000.0 / median_group_wall_us),
    }


def paired_order(pair_count, treatment_route):
    order = []
    for pair_index in range(pair_count):
        routes = (
            [ROUTE_CONTROL, treatment_route]
            if pair_index % 2 == 0
            else [treatment_route, ROUTE_CONTROL]
        )
        for position, route in enumerate(routes):
            order.append({
                "pair_index": pair_index,
                "pair_position": position,
                "route": route,
            })
    return order


class StreamCapture:
    def __init__(self, process, run_dir):
        self.process = process
        self.run_dir = pathlib.Path(run_dir)
        self.ready = threading.Event()
        self.done = threading.Event()
        self.done_timestamp_us = None
        self.lines = {"stdout": [], "stderr": []}
        self.threads = []

    def start(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for name, stream in (("stdout", self.process.stdout), ("stderr", self.process.stderr)):
            thread = threading.Thread(
                target=self._drain, args=(name, stream), daemon=True)
            thread.start()
            self.threads.append(thread)

    def _drain(self, name, stream):
        assert stream is not None
        path = self.run_dir / f"{name}.log"
        with open(path, "w", encoding="utf-8", errors="replace") as handle:
            for line in stream:
                self.lines[name].append(line)
                handle.write(line)
                handle.flush()
                if name == "stderr" and line.startswith("DRIVER_READY "):
                    self.ready.set()
                if name == "stderr" and line.startswith("DRIVER_DONE "):
                    self.done_timestamp_us = time.monotonic_ns() // 1000
                    self.done.set()

    def wait_ready(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.ready.wait(timeout=0.1):
                return
            if self.process.poll() is not None:
                break
        raise ExperimentError(
            "driver exited or timed out before DRIVER_READY\n" +
            "".join(self.lines["stderr"][-20:]))

    def wait_done(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.done.wait(timeout=0.1):
                return self.done_timestamp_us
            if self.process.poll() is not None:
                break
        raise ExperimentError(
            "driver exited or timed out before DRIVER_DONE\n" +
            "".join(self.lines["stderr"][-20:]))

    def join(self):
        for thread in self.threads:
            thread.join(timeout=3)

    def text(self, name):
        return "".join(self.lines[name])


class PhoneStage:
    def __init__(self, serial, remote_dir, remote_bin, model, layer_start,
                 layer_end, host_port, backend, ngl, hexagon_mbuf_mib,
                 batch_size, driver_context, driver_max_prefill, n_gen, log_path,
                 placement_cert=False, decode_no_fa=False):
        self.serial = serial
        self.placement_cert = placement_cert
        self.decode_no_fa = decode_no_fa
        self.remote_dir = remote_dir
        self.remote_bin = remote_bin
        self.model = model
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.host_port = host_port
        self.backend = backend
        self.ngl = ngl
        self.hexagon_mbuf_mib = hexagon_mbuf_mib
        self.batch_size = batch_size
        self.driver_context = driver_context
        self.driver_max_prefill = driver_max_prefill
        self.n_gen = n_gen
        self.log_path = pathlib.Path(log_path)
        self.process = None
        self._log_handle = None
        self.returncode = None

    def start(self, timeout_s):
        adb(self.serial, "forward", "--remove", f"tcp:{self.host_port}", check=False)
        adb(self.serial, "forward", f"tcp:{self.host_port}", f"tcp:{self.host_port}")
        environment = []
        if self.layer_start > 0:
            environment.append(f"LLAMA_LAYER_START={self.layer_start}")
        environment.append(f"LLAMA_LAYER_END={self.layer_end}")
        if self.placement_cert:
            environment.append("LAYERSPLIT_PLACEMENT_CERT=1")
        if self.decode_no_fa:
            environment.append("GGML_DECODE_NO_FA=1")
        command = (
            f"cd {shlex.quote(self.remote_dir)} && "
            f"env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. "
            f"GGML_HEXAGON_MBUF={self.hexagon_mbuf_mib} "
            f"{' '.join(environment)} "
            f"./{shlex.quote(self.remote_bin)} "
            f"-m {shlex.quote(self.model)} --devices {shlex.quote(self.backend)} "
            f"-ngl {self.ngl} --mode stagenet --port {self.host_port} "
            f"-n {self.n_gen} --driver-batch {self.batch_size} "
            f"--driver-context {self.driver_context} "
            f"--driver-max-prefill {self.driver_max_prefill}"
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self.process = subprocess.Popen(
            ["adb", "-s", self.serial, "shell", command],
            stdout=self._log_handle, stderr=subprocess.STDOUT, text=True)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self._log_handle.flush()
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
            if "[stagenet] listening" in text:
                return
            if self.process.poll() is not None:
                raise ExperimentError(
                    f"phone stage exited before listening:\n{text[-4000:]}")
            time.sleep(0.2)
        raise ExperimentError(
            f"phone stage did not listen within {timeout_s}s:\n" +
            self.log_path.read_text(encoding="utf-8", errors="replace")[-4000:])

    def finish(self, timeout_s=60):
        if self.process is None:
            raise ExperimentError(f"phone stage {self.serial} was not started")
        try:
            self.returncode = self.process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise ExperimentError(
                f"phone stage {self.serial} did not exit within {timeout_s}s") from exc
        if self._log_handle is not None:
            self._log_handle.flush()
            self._log_handle.close()
            self._log_handle = None
        if self.returncode != 0:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
            raise ExperimentError(
                f"phone stage {self.serial} failed with rc={self.returncode}:\n"
                f"{text[-4000:]}")
        return self.returncode

    def stop(self):
        if self.process is not None:
            try:
                self.returncode = self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                adb(self.serial, "shell", f"pkill -9 -f {shlex.quote(self.remote_bin)}",
                    check=False)
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
        adb(self.serial, "forward", "--remove", f"tcp:{self.host_port}", check=False)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


def deploy_binary(serial, local_path, remote_dir, remote_name):
    local_path = pathlib.Path(local_path)
    if not local_path.is_file():
        raise ExperimentError(f"phone binary not found: {local_path}")
    temp = f"{remote_dir}/{remote_name}.tmp"
    final = f"{remote_dir}/{remote_name}"
    adb(serial, "shell", f"mkdir -p {shlex.quote(remote_dir)}")
    adb(serial, "push", str(local_path), temp)
    adb(serial, "shell",
        f"chmod 755 {shlex.quote(temp)} && mv {shlex.quote(temp)} {shlex.quote(final)}")
    remote_hash = adb(serial, "shell", f"sha256sum {shlex.quote(final)}").stdout.split()[0]
    local_hash = sha256_file(local_path)
    if remote_hash != local_hash:
        raise ExperimentError(
            f"deployed binary hash mismatch: local={local_hash} remote={remote_hash}")
    return local_hash


def driver_command(args, route):
    command = [
        str(pathlib.Path(args.host_bin).resolve()),
        "-m", str(pathlib.Path(args.host_model).resolve()),
        "-ngl", str(args.host_ngl),
        "--mode", "monodriver" if route == ROUTE_CONTROL else "pipedriver",
        "-p", args.prompt,
        "-n", str(args.n_gen),
        "--driver-requests", str(args.requests),
        "--driver-warmup", str(args.warmups),
        "--driver-batch", str(args.batch_size),
        "--driver-context", str(args.driver_context),
        "--driver-max-prefill", str(args.driver_max_prefill),
        "--wait-for-go",
    ]
    if args.chat:
        command.append("--chat")
    if route != ROUTE_CONTROL:
        command += ["--host", "127.0.0.1", "--port", str(args.op15_port)]
        if route == ROUTE_TWO_PHONE:
            command += ["--port2", str(args.op12_port)]
    return command


def driver_environment(args, route):
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    bin_dir = str(pathlib.Path(args.host_bin).resolve().parent)
    old_ld = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = bin_dir + (":" + old_ld if old_ld else "")
    environment.pop("LLAMA_LAYER_START", None)
    environment.pop("LLAMA_LAYER_END", None)
    if route == ROUTE_OP15:
        environment["LLAMA_LAYER_START"] = str(args.op15_end)
    elif route == ROUTE_TWO_PHONE:
        environment["LLAMA_LAYER_START"] = str(args.op12_end)
    if args.placement_cert:
        environment["LAYERSPLIT_PLACEMENT_CERT"] = "1"
    return environment


def placement_descriptors(args, route, host_pid):
    """Independent scheduled-placement expectations for the frozen S11 route."""
    _require_exact_int(host_pid, "placement host pid", minimum=1)
    if route == ROUTE_CONTROL:
        return [{
            "source": "host", "role": "monodriver", "mode": "monodriver",
            "layer_start": 0, "layer_end": S11_E0_N_LAYER,
            "n_layer": S11_E0_N_LAYER,
            "default_compute_buffer_type": "CUDA0",
            "compute_buffer_overrides": {
                "GET_ROWS": ["CUDA0", "CUDA_Host"]},
            "pid": host_pid,
        }]
    if route != ROUTE_OP15:
        raise ExperimentError(
            "scheduled-placement evidence is currently defined only for S11-E0 op15")
    return [{
        "source": "host", "role": "host_tail", "mode": "pipedriver",
        "layer_start": args.op15_end, "layer_end": S11_E0_N_LAYER,
        "n_layer": S11_E0_N_LAYER,
        "default_compute_buffer_type": "CUDA0",
        "compute_buffer_overrides": {
            "GET_ROWS": ["CUDA0", "CUDA_Host"]},
        "pid": host_pid,
    }, {
        "source": f"phone:{args.op15_serial}", "role": "phone_stage",
        "mode": "stagenet", "layer_start": 0, "layer_end": args.op15_end,
        "n_layer": S11_E0_N_LAYER,
        "default_compute_buffer_type": args.phone_backend,
        "compute_buffer_overrides": {"GET_ROWS": ["CPU"]}, "pid": None,
    }]


def launch_stages(args, route, run_dir):
    stages = []
    if route == ROUTE_CONTROL:
        return stages
    try:
        op15 = PhoneStage(
            args.op15_serial, args.op15_dir, args.phone_remote_bin,
            args.op15_model, 0, args.op15_end, args.op15_port,
            args.phone_backend, args.phone_ngl, args.hexagon_mbuf_mib,
            args.batch_size, args.driver_context, args.driver_max_prefill,
            args.n_gen,
            pathlib.Path(run_dir) / "op15.log",
            placement_cert=args.placement_cert,
            decode_no_fa=args.phone_decode_no_fa)
        stages.append(op15)
        op15.start(args.stage_timeout)
        if route == ROUTE_TWO_PHONE:
            op12 = PhoneStage(
                args.op12_serial, args.op12_dir, args.phone_remote_bin,
                args.op12_model, args.op15_end, args.op12_end, args.op12_port,
                args.phone_backend, args.phone_ngl, args.hexagon_mbuf_mib,
                args.batch_size, args.driver_context, args.driver_max_prefill,
                args.n_gen,
                pathlib.Path(run_dir) / "op12.log",
                placement_cert=args.placement_cert,
                decode_no_fa=args.phone_decode_no_fa)
            stages.append(op12)
            op12.start(args.stage_timeout)
        return stages
    except Exception:
        for stage in reversed(stages):
            stage.stop()
        raise


def run_route(args, route, run_dir):
    run_dir = pathlib.Path(run_dir)
    stages = []
    sampler = None
    monitor = None
    thermal_logger = None
    process = None
    capture = None
    try:
        if args.measure or args.readiness:
            wait_for_gpu_idle(
                args.gpu_uuid, args.max_idle_gpu_util,
                args.max_idle_gpu_memory_mib)

        stages = launch_stages(args, route, run_dir)
        command = driver_command(args, route)
        process = subprocess.Popen(
            command, env=driver_environment(args, route),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1)
        capture = StreamCapture(process, run_dir)
        capture.start()
        capture.wait_ready(args.ready_timeout)
        phone_state_before = {
            stage.serial: query_phone_state(stage.serial) for stage in stages
        }
        # CP1.3: launch the on-device thermal logger for treatment timelines and
        # let it take at least one sample before the paid window opens, so it
        # brackets the window with no periodic ADB traffic during it.
        phone_window_start_us = None
        phone_window_end_us = None
        if stages and args.phone_thermal_logger:
            thermal_logger = PhoneThermalLogger(
                stages[0].serial, args.op15_dir, args.op15_port,
                run_dir / "phone_thermal.jsonl")
            thermal_logger.start()
            time.sleep(args.thermal_preroll_ms / 1000.0)
        gpu_ready = query_gpu(args.gpu_uuid)
        if args.measure:
            monitor = GpuProcessMonitor(
                args.gpu_uuid, run_dir / "gpu_processes.jsonl", args.sample_ms)
            gpu_processes_ready, _ = monitor.record_boundary("READY")
        else:
            gpu_processes_ready = query_gpu_processes(args.gpu_uuid, timeout_s=2)
        unexpected_ready = [
            entry for entry in gpu_processes_ready if entry["pid"] != process.pid
        ]
        if (args.measure or args.readiness) and unexpected_ready:
            raise ExperimentError(
                f"selected GPU has competing processes at ready: {unexpected_ready}")

        samples = []
        sampler_errors = []
        if args.measure:
            sampler = PowerSampler(
                args.gpu_uuid, run_dir / "power.jsonl", args.sample_ms)
            sampler.start()
            sampler.wait_for_samples(2, 5)
            monitor.start()
            time.sleep(args.preroll_ms / 1000.0)

        if thermal_logger is not None:
            phone_window_start_us = phone_uptime_us(stages[0].serial)
        window_start_us = time.monotonic_ns() // 1000
        assert process.stdin is not None
        process.stdin.write("GO\n")
        process.stdin.flush()
        process.stdin.close()
        window_end_us = capture.wait_done(args.run_timeout)
        if thermal_logger is not None:
            phone_window_end_us = phone_uptime_us(stages[0].serial)
        phone_state_after = {
            stage.serial: query_phone_state(stage.serial) for stage in stages
        }
        if monitor is None:
            gpu_processes_done = query_gpu_processes(args.gpu_uuid, timeout_s=2)
        else:
            gpu_processes_done = []
        try:
            returncode = process.wait(timeout=60)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait(timeout=5)
            raise ExperimentError(
                "driver emitted DRIVER_DONE but did not exit within 60s") from exc
        capture.join()
        if returncode != 0:
            raise ExperimentError(
                f"driver failed with rc={returncode}\n{capture.text('stderr')[-6000:]}")
        for stage in stages:
            stage.finish(timeout_s=60)

        observed_power_limits = []
        process_telemetry = None
        if sampler is not None:
            time.sleep(args.postroll_ms / 1000.0)
            observed_power_limits = sampler.observed_power_limits()
            samples, sampler_errors = sampler.stop()
            sampler = None
        if monitor is not None:
            monitor.stop_sampling()
            gpu_processes_done, _ = monitor.record_boundary("DONE")
            process_telemetry = monitor.stop()
            monitor = None
        unexpected_done = [
            entry for entry in gpu_processes_done if entry["pid"] != process.pid
        ]
        thermal_artifact = None
        thermal_quality = None
        if thermal_logger is not None:
            thermal_telemetry = thermal_logger.stop_and_pull()
            thermal_logger = None
            thermal_artifact = {
                "serial": thermal_telemetry["serial"],
                "path": thermal_telemetry["path"],
                "record_count": thermal_telemetry["record_count"],
                "sha256": thermal_telemetry["sha256"],
                "window_start_us": phone_window_start_us,
                "window_end_us": phone_window_end_us,
            }
            try:
                recomputed = reintegrate_thermal_artifact(
                    thermal_artifact, "phone_thermal_artifact")
                thermal_quality = {
                    "valid": recomputed["quality_valid"],
                    "reasons": recomputed["quality_reasons"],
                    "coverage": "CONTINUOUS_ON_DEVICE",
                    "boot_id": recomputed["boot_id"],
                }
            except ExperimentError as exc:
                thermal_quality = {
                    "valid": False, "reasons": [str(exc)],
                    "coverage": "CONTINUOUS_ON_DEVICE", "boot_id": None}
        records = parse_route_records(capture.text("stderr"))
        expected_route = route
        if any(record["route"] != expected_route for record in records):
            raise ExperimentError(
                f"driver emitted route other than {expected_route}")
        if len(records) != args.requests:
            raise ExperimentError(
                f"driver emitted {len(records)} requests, expected {args.requests}")
        if (args.measure or args.readiness) and any(
                record["generated_tokens"] < MIN_MEASURED_GENERATED_TOKENS
                for record in records):
            raise ExperimentError(
                "measured request terminated before the minimum generated-token floor")
        metrics = batch_metrics(records)
        if metrics["batch_size"] != args.batch_size:
            raise ExperimentError(
                f"driver emitted batch {metrics['batch_size']}, expected {args.batch_size}")

        # CP1.5: harvest scheduled output-buffer placement observations. The host
        # driver writes its cert to stderr; a treatment run also collects the
        # phone stage's cert from the op15 stage log. They are written to a hashed
        # file so reverify can recompute the no-fallback proof from the bytes.
        placement_artifact = None
        placement_quality = None
        placement_expected = None
        if args.placement_cert:
            host_certs = parse_placement_certs(capture.text("stderr"), "host")
            placement_records = [
                {"source": "host", "certificate": cert} for cert in host_certs]
            if route == ROUTE_CONTROL:
                pass
            else:
                for stage in stages:
                    phone_text = stage.log_path.read_text(
                        encoding="utf-8", errors="replace")
                    phone_certs = parse_placement_certs(
                        phone_text, f"phone:{stage.serial}")
                    placement_records.extend({
                        "source": f"phone:{stage.serial}", "certificate": cert,
                    } for cert in phone_certs)
            placement_expected = placement_descriptors(args, route, process.pid)
            placement_path = run_dir / "placement_certs.jsonl"
            records_sorted = sorted(
                placement_records, key=lambda item: item["source"])
            with open(placement_path, "w", encoding="ascii") as handle:
                for item in records_sorted:
                    handle.write(json.dumps(
                        item, sort_keys=True, separators=(",", ":")) + "\n")
            placement_artifact = {
                "path": str(placement_path.resolve()),
                "record_count": len(records_sorted),
                "sha256": sha256_file(placement_path),
            }
            try:
                recomputed = reintegrate_placement_artifact(
                    placement_artifact, placement_expected,
                    "placement_artifact")
                placement_quality = {
                    "valid": recomputed["quality_valid"],
                    "reasons": recomputed["quality_reasons"],
                    "sources": recomputed["sources"],
                }
            except ExperimentError as exc:
                placement_quality = {"valid": False, "reasons": [str(exc)],
                                     "sources": []}

        measurement = {
            "enabled": args.measure,
            "scope": "GPU_BOARD" if args.measure else "NONE",
            "instrument": "NVML_BOARD" if args.measure else "NONE",
            "window_start_us": window_start_us,
            "window_end_us": window_end_us,
            "window_us": window_end_us - window_start_us,
            "sampler_errors": sampler_errors,
            # CP1.3: thermal evidence binds only to treatment timelines; the
            # control has no phone, so it is explicitly NOT_APPLICABLE.
            "phone_thermal_artifact": thermal_artifact,
            "phone_thermal_quality": thermal_quality,
            "phone_thermal_binding": (
                "TREATMENT" if thermal_artifact is not None else "NOT_APPLICABLE"),
            # CP1.5: scheduled output-buffer placement for this timeline.
            "placement_artifact": placement_artifact,
            "placement_expectations": placement_expected,
            "placement_quality": placement_quality,
        }
        if args.measure:
            power_path = run_dir / "power.jsonl"
            measurement["power_artifact"] = {
                "board_uuid": args.gpu_uuid,
                "path": str(power_path.resolve()),
                "record_count": sum(
                    1 for line in power_path.read_text(
                        encoding="ascii").splitlines() if line),
                "sha256": sha256_file(power_path),
                "window_start_us": window_start_us,
                "window_end_us": window_end_us,
                "power_limit_mw": gpu_ready["power_limit_mw"],
            }
            phone_quality = phone_boundary_quality(
                [stage.serial for stage in stages],
                phone_state_before,
                phone_state_after)
            measurement["phone_boundary_quality"] = phone_quality

            # CP1.2: continuous process telemetry, hashed and evaluated. The
            # artifact is a full reintegration record (board, window, driver pid)
            # so reverify_pairs can recompute contamination from the bytes.
            process_quality = evaluate_process_telemetry(
                process_telemetry["samples"] if process_telemetry else [],
                process.pid,
                errors=process_telemetry["errors"] if process_telemetry else None,
                boundaries=(
                    process_telemetry["boundaries"] if process_telemetry else None),
                window_start_us=window_start_us,
                window_end_us=window_end_us)
            measurement["process_monitor_artifact"] = None if process_telemetry \
                is None else {
                    "board_uuid": args.gpu_uuid,
                    "path": process_telemetry["path"],
                    "record_count": process_telemetry["record_count"],
                    "sha256": process_telemetry["sha256"],
                    "window_start_us": window_start_us,
                    "window_end_us": window_end_us,
                    "driver_pid": process.pid,
                }
            measurement["process_monitor_quality"] = process_quality

            # CP1.4: the power limit must not move within the slot.
            limit_quality = power_limit_invariant(
                observed_power_limits, gpu_ready["power_limit_mw"])
            measurement["power_limit_quality"] = limit_quality
            measurement["power_limit_observed_mw"] = limit_quality["power_limit_mw"]

            quality = measurement_quality(samples, window_start_us, window_end_us)
            if sampler_errors:
                quality["valid"] = False
                quality["reasons"].append("E_SAMPLER")
            # READY/DONE GPU process boundaries are part of the hashed process
            # artifact. Phone boundary snapshots remain diagnostic; continuous
            # on-device thermal evidence supersedes them as the validity gate.
            if not process_quality["valid"]:
                quality["valid"] = False
                quality["reasons"].extend(process_quality["reasons"])
            if not limit_quality["valid"]:
                quality["valid"] = False
                quality["reasons"].extend(limit_quality["reasons"])
            # CP1.3: a treatment timeline must carry valid continuous thermal
            # evidence; the control has none and imposes no thermal requirement.
            if thermal_quality is not None and not thermal_quality["valid"]:
                quality["valid"] = False
                quality["reasons"].extend(thermal_quality["reasons"])
            # CP1.5: every timeline must carry a valid no-fallback placement cert.
            if placement_quality is not None and not placement_quality["valid"]:
                quality["valid"] = False
                quality["reasons"].extend(placement_quality["reasons"])
            measurement["quality"] = quality
            if quality["valid"]:
                measurement["energy_nj"] = integrate_zoh(
                    samples, window_start_us, window_end_us)
                uncertainty = nvml_uncertainty(
                    measurement["window_us"], gpu_ready["power_limit_mw"])
                measurement["uncertainty_nj"] = uncertainty["total"]
                measurement["uncertainty_breakdown_nj"] = {
                    key: value for key, value in uncertainty.items() if key != "total"
                }
            else:
                measurement["energy_nj"] = None
                measurement["uncertainty_nj"] = None
                measurement["uncertainty_breakdown_nj"] = None
        else:
            measurement["power_artifact"] = None
            measurement["phone_boundary_quality"] = {
                "valid": False,
                "coverage": "BOUNDARY_ONLY",
                "reasons": ["MEASUREMENT_NOT_REQUESTED"],
            }
            measurement["quality"] = {
                "valid": False,
                "reasons": ["MEASUREMENT_NOT_REQUESTED"],
            }
            measurement["energy_nj"] = None
            measurement["uncertainty_nj"] = None
            measurement["uncertainty_breakdown_nj"] = None

        readiness_reasons = []
        if args.readiness:
            if unexpected_ready:
                readiness_reasons.append("E_GPU_CONTAMINATION_READY")
            if unexpected_done:
                readiness_reasons.append("E_GPU_CONTAMINATION_DONE")
            if placement_quality is None or not placement_quality["valid"]:
                readiness_reasons.extend(
                    ["E_PLACEMENT_MISSING"] if placement_quality is None else
                    placement_quality["reasons"])
            if route != ROUTE_CONTROL:
                if thermal_quality is None or not thermal_quality["valid"]:
                    readiness_reasons.extend(
                        ["E_THERMAL_MISSING"] if thermal_quality is None else
                        thermal_quality["reasons"])
        readiness = {
            "enabled": args.readiness,
            "valid": args.readiness and not readiness_reasons,
            "reasons": readiness_reasons if args.readiness else ["READINESS_NOT_REQUESTED"],
        }

        token_payload = [record["token_ids"] for record in records]
        result = {
            "schema": RUN_SCHEMA,
            "route_record_schema": ROUTE_RECORD_SCHEMA,
            "route": route,
            "driver_pid": process.pid,
            "command": command,
            "returncode": returncode,
            "records": records,
            "batch_metrics": metrics,
            "token_sha256": sha256_bytes(canonical_bytes(token_payload)),
            "gpu_ready": gpu_ready,
            "gpu_processes_ready": gpu_processes_ready,
            "gpu_processes_done": gpu_processes_done,
            "phone_state_before": phone_state_before,
            "phone_state_after": phone_state_after,
            "measurement": measurement,
            "readiness": readiness,
        }
        write_json(run_dir / "run.json", result)
        return result
    finally:
        if sampler is not None:
            sampler.stop()
        if monitor is not None:
            try:
                monitor.stop()
            except ExperimentError:
                pass
        if thermal_logger is not None:
            try:
                thermal_logger.stop_and_pull()
            except (ExperimentError, OSError, subprocess.SubprocessError):
                pass
        if process is not None and process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        if capture is not None:
            capture.join()
        for stage in reversed(stages):
            stage.stop()


def _require_exact_int(value, field, minimum=None, allow_none=False):
    if value is None and allow_none:
        return
    if type(value) is not int:
        raise ExperimentError(f"{field} is not an exact integer")
    if minimum is not None and value < minimum:
        raise ExperimentError(f"{field} is below {minimum}")


def _require_positive_number(value, field):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ExperimentError(f"{field} is not a positive finite number")


def _require_power_artifact(value, field):
    if type(value) is not dict or set(value) != {
            "board_uuid", "path", "record_count", "sha256",
            "window_start_us", "window_end_us", "power_limit_mw"}:
        raise ExperimentError(f"{field} is not a complete power artifact")
    if type(value["board_uuid"]) is not str or not value["board_uuid"]:
        raise ExperimentError(f"{field}.board_uuid is invalid")
    if type(value["path"]) is not str or not value["path"]:
        raise ExperimentError(f"{field}.path is invalid")
    _require_exact_int(value["record_count"], f"{field}.record_count", minimum=2)
    if type(value["sha256"]) is not str or SHA256_PATTERN.fullmatch(value["sha256"]) is None:
        raise ExperimentError(f"{field}.sha256 is invalid")
    _require_exact_int(
        value["window_start_us"], f"{field}.window_start_us", minimum=0)
    _require_exact_int(
        value["window_end_us"], f"{field}.window_end_us", minimum=1)
    _require_exact_int(
        value["power_limit_mw"], f"{field}.power_limit_mw", minimum=1)
    if value["window_end_us"] <= value["window_start_us"]:
        raise ExperimentError(f"{field} window is empty or inverted")


def reintegrate_power_artifact(artifact, field):
    """Reopen the raw power stream ONCE as immutable bytes and recompute.

    The stored energy is never trusted. These bytes on disk are hashed, counted,
    strictly parsed, validated, and re-integrated here, and only the recomputed
    values reach aggregation. A byte read failure, a hash mismatch, a sample-count
    mismatch, an unparseable or duplicate-keyed line, or a bracket/gap structural
    failure all fail closed by raising.

    Returns the recomputed quality, gross ZOH energy, and conservative
    uncertainty. Energy and uncertainty are None when the recomputed quality is
    invalid (for example too few in-window updates); an invalid stream is a soft
    timeline outcome, not a structural corruption, so it is reported rather than
    raised.
    """
    _require_power_artifact(artifact, field)
    path = pathlib.Path(artifact["path"])
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ExperimentError(f"{field} cannot be reopened: {exc}") from exc
    actual_sha = sha256_bytes(data)
    if actual_sha != artifact["sha256"]:
        raise ExperimentError(
            f"{field} sha256 mismatch: the bytes on disk hash to {actual_sha} "
            f"but the record binds {artifact['sha256']}")
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ExperimentError(f"{field} is not ASCII: {exc}") from exc
    records = [
        strict_json_loads(line, f"{field} record")
        for line in text.split("\n") if line
    ]
    if len(records) != artifact["record_count"]:
        raise ExperimentError(
            f"{field} record-count mismatch: the bytes on disk carry "
            f"{len(records)} records but the record binds "
            f"{artifact['record_count']}")
    samples = []
    errors = []
    board_uuids = set()
    previous_timestamp = None
    for index, record in enumerate(records):
        _validate_power_record(record, f"{field} record {index}")
        timestamp = record["timestamp_us"]
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ExperimentError(
                f"{field} record {index} timestamp_us is not increasing")
        previous_timestamp = timestamp
        board_uuids.add(record["board_uuid"])
        if record["kind"] == "sample":
            samples.append(record)
        else:
            errors.append(record["message"])
    if board_uuids != {artifact["board_uuid"]}:
        raise ExperimentError(
            f"{field} board UUID mismatch: bytes={sorted(board_uuids)} "
            f"artifact={artifact['board_uuid']}")
    validate_power_samples(samples)
    start = artifact["window_start_us"]
    end = artifact["window_end_us"]
    quality = measurement_quality(samples, start, end)
    # Recompute the observed power-limit set from the hashed bytes, never from
    # the record metadata. A stream whose rows disagree with the declared limit
    # (a metadata-only forgery) fails the invariant here and invalidates the
    # timeline, and the uncertainty below is derived from the recomputed limit.
    observed_limits = sorted({sample["power_limit_mw"] for sample in samples})
    limit_check = power_limit_invariant(observed_limits, artifact["power_limit_mw"])
    valid = quality["valid"] and limit_check["valid"] and not errors
    reasons = list(quality["reasons"]) + list(limit_check["reasons"])
    if errors:
        reasons.append(f"E_SAMPLER:{len(errors)}")
    observed_limit = limit_check["power_limit_mw"]
    recomputed = {
        "reopened_bytes": len(data),
        "recomputed_sha256": actual_sha,
        "recomputed_record_count": len(records),
        "recomputed_sample_count": len(samples),
        "sample_count": len(samples),
        "error_count": len(errors),
        "board_uuid": next(iter(board_uuids)),
        "quality_valid": valid,
        "quality_reasons": reasons,
        "independent_updates": quality.get("independent_updates"),
        "max_gap_us": quality.get("max_gap_us"),
        "pstates": quality.get("pstates"),
        "observed_power_limit_mw": observed_limit,
        "energy_nj": None,
        "uncertainty_nj": None,
    }
    if valid:
        recomputed["energy_nj"] = integrate_zoh(samples, start, end)
        recomputed["uncertainty_nj"] = nvml_uncertainty(
            end - start, observed_limit)["total"]
    return recomputed


def reverify_pairs(pair_results):
    """Reintegrate every measured pair from its hashed bytes before aggregation.

    Both the power stream and the process telemetry of each side are reopened and
    recomputed. Timeline validity is derived here from the recomputed evidence --
    exact work, valid recomputed power quality (including the power-limit
    invariant), and valid recomputed process telemetry -- and the stored
    `measurement_valid` boolean is OVERWRITTEN with it, so aggregation consumes
    recomputed validity rather than a booleaen a slot wrote about itself.

    For a pair that stays valid the recomputed energy and uncertainty MUST
    reproduce the stored numbers from the same bytes; a disagreement is fatal,
    because the number entering the sum would otherwise be unverified. Invalid
    pairs have their energy nulled so no unverified number survives. Each pair is
    marked `reintegrated` so aggregation can refuse any pair that skipped this.
    """
    for pair in pair_results:
        pair_index = pair.get("pair_index")
        evidence = {}
        side_valid = {}
        for side in ("control", "treatment"):
            power_field = f"pair {pair_index} {side}_power_artifact"
            process_field = f"pair {pair_index} {side}_process_artifact"
            placement_field = f"pair {pair_index} {side}_placement_artifact"
            power = reintegrate_power_artifact(
                pair.get(f"{side}_power_artifact"), power_field)
            process = reintegrate_process_artifact(
                pair.get(f"{side}_process_artifact"), process_field)
            placement = reintegrate_placement_artifact(
                pair.get(f"{side}_placement_artifact"),
                pair.get(f"{side}_placement_expectations"), placement_field)
            evidence[side] = {
                "power": power, "process": process, "placement": placement}
            side_valid[side] = (
                power["quality_valid"] and process["quality_valid"]
                and placement["quality_valid"])
            pair[f"{side}_power_limit_observed_mw"] = power["observed_power_limit_mw"]
        # CP1.3: the treatment timeline must additionally carry valid recomputed
        # thermal evidence; the control is NOT_APPLICABLE and imposes none.
        thermal = reintegrate_thermal_artifact(
            pair.get("treatment_thermal_artifact"),
            f"pair {pair_index} treatment_thermal_artifact")
        pair["treatment_thermal_boot_id"] = thermal["boot_id"]
        evidence["treatment_thermal"] = thermal
        exact_work = bool(pair.get("exact_work"))
        # Conjoin, do not replace, the runner's stored validity. Some invalidation
        # reasons exist only at runtime and are not in the reintegrated bytes:
        # E_SAMPLER (a dropped malformed power line never reaches power.jsonl),
        # E_GPU_CONTAMINATION_BOUNDARY (the done-snapshot), and phone boundary
        # quality. Dropping them would let a byte-clean-but-runtime-vetoed timeline
        # be resurrected to valid, and would also turn a benign single dropped
        # sample (stored energy None) into a whole-experiment disagreement raise.
        # ANDing with stored_valid keeps every runtime veto AND still lets the
        # byte recompute veto a slot -- strictly more fail-closed than either alone.
        stored_valid = pair.get("measurement_valid")
        if type(stored_valid) is not bool:
            raise ExperimentError(
                f"pair {pair_index} stored measurement validity is not boolean")
        recomputed_valid = (
            exact_work
            and side_valid["control"] and side_valid["treatment"]
            and thermal["quality_valid"])
        pair["stored_measurement_valid"] = stored_valid
        pair["stored_validity_disagrees"] = stored_valid != recomputed_valid
        if recomputed_valid:
            for side in ("control", "treatment"):
                power = evidence[side]["power"]
                for name in ("energy_nj", "uncertainty_nj"):
                    stored = pair.get(f"{side}_{name}")
                    if power[name] != stored:
                        raise ExperimentError(
                            f"pair {pair_index} {side}_{name} disagreement: "
                            f"stored {stored} but reintegration of the hashed "
                            f"bytes yields {power[name]}")
                    pair[f"{side}_{name}"] = power[name]
        else:
            for side in ("control", "treatment"):
                pair[f"{side}_energy_nj"] = None
                pair[f"{side}_uncertainty_nj"] = None
        pair["measurement_valid"] = recomputed_valid
        pair["reintegration"] = evidence
        pair["reintegrated"] = True
    return pair_results


def summarize_pairs(order, runs):
    if len(order) != len(runs):
        raise ExperimentError(
            f"pair order/run length mismatch: {len(order)} != {len(runs)}")
    pairs = {}
    for slot_index, (slot, run) in enumerate(zip(order, runs)):
        if type(slot) is not dict or type(run) is not dict:
            raise ExperimentError(f"slot {slot_index} is not an object")
        pair_index = slot.get("pair_index")
        pair_position = slot.get("pair_position")
        slot_route = slot.get("route")
        _require_exact_int(pair_index, f"slot {slot_index} pair_index", 0)
        _require_exact_int(pair_position, f"slot {slot_index} pair_position", 0)
        if pair_position not in (0, 1):
            raise ExperimentError(
                f"slot {slot_index} pair_position is not 0 or 1")
        if slot_route not in (ROUTE_CONTROL, ROUTE_OP15, ROUTE_TWO_PHONE):
            raise ExperimentError(f"slot {slot_index} route is invalid")
        if run.get("route") != slot_route:
            raise ExperimentError(
                f"slot {slot_index} route/run mismatch: "
                f"{slot_route} != {run.get('route')}")
        pair = pairs.setdefault(pair_index, {
            "positions": set(),
            "routes": {},
        })
        if pair_position in pair["positions"]:
            raise ExperimentError(
                f"pair {pair_index} duplicates position {pair_position}")
        if slot_route in pair["routes"]:
            raise ExperimentError(
                f"pair {pair_index} duplicates route {slot_route}")
        pair["positions"].add(pair_position)
        pair["routes"][slot_route] = run
    pair_indexes = sorted(pairs)
    if pair_indexes != list(range(len(pairs))):
        raise ExperimentError("pair indexes are not contiguous")

    pair_results = []
    treatment_routes = set()
    for pair_index in sorted(pairs):
        pair = pairs[pair_index]
        if pair["positions"] != {0, 1}:
            raise ExperimentError(f"pair {pair_index} positions are incomplete")
        routes = pair["routes"]
        control = routes.get(ROUTE_CONTROL)
        treatment_keys = [key for key in routes if key != ROUTE_CONTROL]
        if control is None or len(treatment_keys) != 1:
            raise ExperimentError(f"pair {pair_index} is incomplete")
        treatment = routes[treatment_keys[0]]
        treatment_routes.add(treatment["route"])
        exact, reason = same_work(control["records"], treatment["records"])
        control_quality = control["measurement"]["quality"]["valid"]
        treatment_quality = treatment["measurement"]["quality"]["valid"]
        if type(control_quality) is not bool or type(treatment_quality) is not bool:
            raise ExperimentError(
                f"pair {pair_index} measurement validity is not boolean")
        control_memory_mib = control["gpu_ready"]["memory_used_mib"]
        treatment_memory_mib = treatment["gpu_ready"]["memory_used_mib"]
        _require_exact_int(
            control_memory_mib,
            f"pair {pair_index} control GPU memory", minimum=0)
        _require_exact_int(
            treatment_memory_mib,
            f"pair {pair_index} treatment GPU memory", minimum=0)
        valid_measurement = (
            exact and
            control_quality and
            treatment_quality
        )
        completed_requests = len(control["records"])
        generated_tokens = sum(record["generated_tokens"] for record in control["records"])
        if completed_requests <= 0 or generated_tokens <= 0:
            raise ExperimentError(f"pair {pair_index} has no completed work")
        pair_results.append({
            "pair_index": pair_index,
            "treatment_route": treatment["route"],
            "exact_work": exact,
            "exact_reason": reason,
            "completed_requests": completed_requests,
            "generated_tokens": generated_tokens,
            "measurement_valid": valid_measurement,
            "control_gpu_ready_memory_mib": control_memory_mib,
            "treatment_gpu_ready_memory_mib": treatment_memory_mib,
            "gpu_ready_memory_relief_mib": (
                control_memory_mib - treatment_memory_mib),
            "control_energy_nj": control["measurement"]["energy_nj"],
            "control_uncertainty_nj": control["measurement"]["uncertainty_nj"],
            "treatment_energy_nj": treatment["measurement"]["energy_nj"],
            "treatment_uncertainty_nj": treatment["measurement"]["uncertainty_nj"],
            "control_power_artifact": control["measurement"].get("power_artifact"),
            "treatment_power_artifact": treatment["measurement"].get("power_artifact"),
            "control_process_artifact": control["measurement"].get(
                "process_monitor_artifact"),
            "treatment_process_artifact": treatment["measurement"].get(
                "process_monitor_artifact"),
            "control_thermal_binding": control["measurement"].get(
                "phone_thermal_binding"),
            "treatment_thermal_binding": treatment["measurement"].get(
                "phone_thermal_binding"),
            "treatment_thermal_artifact": treatment["measurement"].get(
                "phone_thermal_artifact"),
            "control_placement_artifact": control["measurement"].get(
                "placement_artifact"),
            "treatment_placement_artifact": treatment["measurement"].get(
                "placement_artifact"),
            "control_placement_expectations": control["measurement"].get(
                "placement_expectations"),
            "treatment_placement_expectations": treatment["measurement"].get(
                "placement_expectations"),
            "control_readiness_valid": control.get("readiness", {}).get("valid"),
            "treatment_readiness_valid": treatment.get("readiness", {}).get("valid"),
            "control_driver_pid": control.get("driver_pid"),
            "treatment_driver_pid": treatment.get("driver_pid"),
            "control_gpu_processes_ready": control.get("gpu_processes_ready"),
            "control_gpu_processes_done": control.get("gpu_processes_done"),
            "treatment_gpu_processes_ready": treatment.get("gpu_processes_ready"),
            "treatment_gpu_processes_done": treatment.get("gpu_processes_done"),
            "control_batch_metrics": control["batch_metrics"],
            "treatment_batch_metrics": treatment["batch_metrics"],
        })
    if len(treatment_routes) != 1:
        raise ExperimentError("pairs mix treatment routes")
    return pair_results


def _readiness_gpu_boundary_valid(entries, driver_pid, field):
    _require_exact_int(driver_pid, f"{field}.driver_pid", minimum=1)
    if type(entries) is not list:
        raise ExperimentError(f"{field} is not an array")
    foreign = []
    for index, entry in enumerate(entries):
        if type(entry) is not dict or set(entry) != PROCESS_ENTRY_FIELDS:
            raise ExperimentError(f"{field} entry {index} fields differ")
        _require_exact_int(entry["pid"], f"{field} entry {index}.pid", minimum=1)
        _require_exact_int(
            entry["memory_used_mib"],
            f"{field} entry {index}.memory_used_mib", minimum=0)
        if type(entry["name"]) is not str:
            raise ExperimentError(f"{field} entry {index}.name is invalid")
        if entry["pid"] != driver_pid:
            foreign.append(entry["pid"])
    return not foreign, sorted(set(foreign))


def reverify_readiness_pairs(pair_results):
    """Reopen readiness evidence without constructing any energy evidence."""
    for pair in pair_results:
        pair_index = pair.get("pair_index")
        evidence = {}
        valid = bool(pair.get("exact_work"))
        for side in ("control", "treatment"):
            placement = reintegrate_placement_artifact(
                pair.get(f"{side}_placement_artifact"),
                pair.get(f"{side}_placement_expectations"),
                f"pair {pair_index} {side}_placement_artifact")
            evidence[f"{side}_placement"] = placement
            valid = valid and placement["quality_valid"]
            driver_pid = pair.get(f"{side}_driver_pid")
            for boundary in ("ready", "done"):
                boundary_valid, foreign = _readiness_gpu_boundary_valid(
                    pair.get(f"{side}_gpu_processes_{boundary}"), driver_pid,
                    f"pair {pair_index} {side}_{boundary}")
                evidence[f"{side}_{boundary}_foreign_pids"] = foreign
                valid = valid and boundary_valid
        thermal = reintegrate_thermal_artifact(
            pair.get("treatment_thermal_artifact"),
            f"pair {pair_index} treatment_thermal_artifact")
        evidence["treatment_thermal"] = thermal
        valid = valid and thermal["quality_valid"]
        for side in ("control", "treatment"):
            stored = pair.get(f"{side}_readiness_valid")
            if type(stored) is not bool:
                raise ExperimentError(
                    f"pair {pair_index} {side}_readiness_valid is not boolean")
        pair["readiness_valid"] = bool(valid)
        pair["readiness_reintegration"] = evidence
        pair["readiness_reintegrated"] = True
    return pair_results


def aggregate_readiness_result(pair_results):
    if not pair_results:
        raise ExperimentError("cannot aggregate an empty readiness pair set")
    placement_hashes = []
    thermal_hashes = []
    for position, pair in enumerate(pair_results):
        if pair.get("pair_index") != position:
            raise ExperimentError("readiness pair indexes are not contiguous")
        if pair.get("exact_work") is not True:
            raise ExperimentError(f"readiness pair {position} work is not exact")
        if pair.get("readiness_reintegrated") is not True:
            raise ExperimentError(
                f"readiness pair {position} skipped evidence reintegration")
        if type(pair.get("readiness_valid")) is not bool:
            raise ExperimentError(
                f"readiness pair {position} validity is not boolean")
        for side in ("control", "treatment"):
            artifact = pair.get(f"{side}_placement_artifact")
            _require_placement_artifact(
                artifact, f"readiness pair {position} {side}_placement")
            placement_hashes.append(artifact["sha256"])
        thermal = pair.get("treatment_thermal_artifact")
        _require_thermal_artifact(
            thermal, f"readiness pair {position} treatment_thermal")
        thermal_hashes.append(thermal["sha256"])
    if len(placement_hashes) != len(set(placement_hashes)):
        raise ExperimentError("readiness reuses a placement artifact")
    if len(thermal_hashes) != len(set(thermal_hashes)):
        raise ExperimentError("readiness reuses a thermal artifact")
    valid = all(pair["readiness_valid"] for pair in pair_results)
    relief = [pair["gpu_ready_memory_relief_mib"] for pair in pair_results]
    return {
        "scope": "MECHANICS_ONLY",
        "formal_claim": "NONE",
        "status": "FUNCTIONAL_EXACT_PASS",
        "exact_work_all_pairs": True,
        "readiness_status": "PASS" if valid else "FAIL",
        "valid_readiness_pairs": sum(
            pair["readiness_valid"] for pair in pair_results),
        "required_readiness_pairs": len(pair_results),
        "gpu_ready_memory_relief_mib_min": min(relief),
        "gpu_ready_memory_relief_mib_max": max(relief),
        "energy_status": "NOT_MEASURED",
        "phone_energy_status": "UNKNOWN",
        "total_system_energy_status": "UNKNOWN",
    }


def aggregate_result(pair_results, measurement_requested, slo_p95_us=None):
    if type(measurement_requested) is not bool:
        raise ExperimentError("measurement_requested is not boolean")
    if measurement_requested:
        _require_exact_int(slo_p95_us, "slo_p95_us", minimum=1)
    elif slo_p95_us is not None:
        raise ExperimentError("slo_p95_us requires measurement mode")
    if not pair_results:
        raise ExperimentError("cannot aggregate an empty pair set")
    pair_indexes = []
    power_artifact_hashes = []
    power_artifact_board_uuids = []
    process_artifact_hashes = []
    thermal_artifact_hashes = []
    placement_artifact_hashes = []
    board_uuids = []
    observed_power_limits = []
    for position, pair in enumerate(pair_results):
        if type(pair) is not dict:
            raise ExperimentError(f"pair result {position} is not an object")
        pair_index = pair.get("pair_index")
        _require_exact_int(pair_index, f"pair result {position} pair_index", 0)
        pair_indexes.append(pair_index)

        exact_work = pair.get("exact_work")
        if type(exact_work) is not bool:
            raise ExperimentError(f"pair {pair_index} exact_work is not boolean")
        if not exact_work:
            raise ExperimentError(f"pair {pair_index} work is not exact")
        _require_exact_int(
            pair.get("completed_requests"),
            f"pair {pair_index} completed_requests", minimum=1)
        _require_exact_int(
            pair.get("generated_tokens"),
            f"pair {pair_index} generated_tokens", minimum=1)
        measurement_valid = pair.get("measurement_valid")
        if type(measurement_valid) is not bool:
            raise ExperimentError(
                f"pair {pair_index} measurement_valid is not boolean")
        if not measurement_requested and measurement_valid:
            raise ExperimentError(
                f"pair {pair_index} has measurement validity without measurement")

        _require_exact_int(
            pair.get("gpu_ready_memory_relief_mib"),
            f"pair {pair_index} gpu_ready_memory_relief_mib", minimum=0)
        energy_fields = (
            "control_energy_nj", "control_uncertainty_nj",
            "treatment_energy_nj", "treatment_uncertainty_nj")
        for field in energy_fields:
            _require_exact_int(
                pair.get(field), f"pair {pair_index} {field}",
                minimum=0, allow_none=True)
        if measurement_valid and any(pair.get(field) is None for field in energy_fields):
            raise ExperimentError(
                f"pair {pair_index} valid measurement lacks energy fields")
        if not measurement_requested and any(
                pair.get(field) is not None for field in energy_fields):
            raise ExperimentError(
                f"pair {pair_index} has energy without measurement")
        if measurement_requested and pair.get("reintegrated") is not True:
            raise ExperimentError(
                f"pair {pair_index} reached aggregation without reintegration; "
                f"aggregation must consume values recomputed from the hashed "
                f"power bytes")
        for side in ("control", "treatment"):
            artifact = pair.get(f"{side}_power_artifact")
            process_artifact = pair.get(f"{side}_process_artifact")
            observed_limit = pair.get(f"{side}_power_limit_observed_mw")
            if measurement_requested:
                _require_power_artifact(
                    artifact, f"pair {pair_index} {side}_power_artifact")
                power_artifact_hashes.append(artifact["sha256"])
                power_artifact_board_uuids.append(artifact["board_uuid"])
                board_uuids.append(artifact["board_uuid"])
                _require_process_artifact(
                    process_artifact,
                    f"pair {pair_index} {side}_process_artifact")
                process_artifact_hashes.append(process_artifact["sha256"])
                board_uuids.append(process_artifact["board_uuid"])
                _require_exact_int(
                    observed_limit,
                    f"pair {pair_index} {side}_power_limit_observed_mw",
                    minimum=1)
                observed_power_limits.append(observed_limit)
                placement_artifact = pair.get(f"{side}_placement_artifact")
                _require_placement_artifact(
                    placement_artifact,
                    f"pair {pair_index} {side}_placement_artifact")
                placement_artifact_hashes.append(placement_artifact["sha256"])
            else:
                if artifact is not None:
                    raise ExperimentError(
                        f"pair {pair_index} has a power artifact without measurement")
                if process_artifact is not None:
                    raise ExperimentError(
                        f"pair {pair_index} has a process artifact without measurement")
                if pair.get(f"{side}_placement_artifact") is not None:
                    raise ExperimentError(
                        f"pair {pair_index} has a placement artifact without measurement")

        # CP1.3: thermal evidence binds to the treatment timeline only.
        if measurement_requested:
            if pair.get("control_thermal_binding") != "NOT_APPLICABLE":
                raise ExperimentError(
                    f"pair {pair_index} control thermal binding is not "
                    f"NOT_APPLICABLE")
            if pair.get("treatment_thermal_binding") != "TREATMENT":
                raise ExperimentError(
                    f"pair {pair_index} treatment thermal binding is not TREATMENT")
            thermal_artifact = pair.get("treatment_thermal_artifact")
            _require_thermal_artifact(
                thermal_artifact,
                f"pair {pair_index} treatment_thermal_artifact")
            thermal_artifact_hashes.append(thermal_artifact["sha256"])
        elif pair.get("treatment_thermal_artifact") is not None:
            raise ExperimentError(
                f"pair {pair_index} has a thermal artifact without measurement")

        for side in ("control", "treatment"):
            metrics = pair.get(f"{side}_batch_metrics")
            if type(metrics) is not dict:
                raise ExperimentError(
                    f"pair {pair_index} {side}_batch_metrics is not an object")
            _require_positive_number(
                metrics.get("median_useful_requests_per_s"),
                f"pair {pair_index} {side} throughput")
            _require_exact_int(
                metrics.get("p95_group_wall_us"),
                f"pair {pair_index} {side} p95_group_wall_us", minimum=1)

    if len(set(pair_indexes)) != len(pair_indexes):
        raise ExperimentError("pair result indexes are duplicated")
    if sorted(pair_indexes) != list(range(len(pair_results))):
        raise ExperimentError("pair result indexes are not contiguous")
    if len(power_artifact_hashes) != len(set(power_artifact_hashes)):
        raise ExperimentError("power artifact is reused across measurement slots")
    if len(process_artifact_hashes) != len(set(process_artifact_hashes)):
        raise ExperimentError("process artifact is reused across measurement slots")
    if len(thermal_artifact_hashes) != len(set(thermal_artifact_hashes)):
        raise ExperimentError("thermal artifact is reused across measurement slots")
    if len(placement_artifact_hashes) != len(set(placement_artifact_hashes)):
        raise ExperimentError(
            "placement artifact is reused across measurement slots")
    if measurement_requested and len(set(power_artifact_board_uuids)) != 1:
        raise ExperimentError("measurement slots mix GPU board UUIDs")
    if measurement_requested and len(set(board_uuids)) != 1:
        raise ExperimentError(
            "measurement slots mix GPU board UUIDs across power and process")
    if measurement_requested:
        # CP1.4: one GPU power limit across all 16 slots, taken from the limit
        # recomputed from each stream's hashed bytes -- not from record metadata.
        power_limits = set(observed_power_limits)
        if len(power_limits) != 1:
            raise ExperimentError(
                f"measurement slots do not share one GPU power limit: "
                f"{sorted(power_limits)}")
    work_shapes = {
        (pair["completed_requests"], pair["generated_tokens"])
        for pair in pair_results
    }
    if len(work_shapes) != 1:
        raise ExperimentError("pair results do not carry identical closed work")

    exact = True
    valid_pairs = [pair for pair in pair_results if pair["measurement_valid"]]
    result = {
        "scope": "GPU_BOARD" if measurement_requested else "NONE",
        "formal_claim": "NONE",
        "exact_work_all_pairs": exact,
        "valid_measurement_pairs": len(valid_pairs),
        "required_measurement_pairs": MIN_MEASUREMENT_PAIRS,
        "status": "FUNCTIONAL_EXACT_PASS" if exact else "FUNCTIONAL_MISMATCH",
    }
    if measurement_requested:
        result["gpu_board_uuid"] = power_artifact_board_uuids[0]
        result["label"] = "MEASUREMENT_INVALID"
        result["phone_energy_status"] = "UNKNOWN"
        result["total_system_energy_status"] = "UNKNOWN"
    if pair_results:
        relief = [pair["gpu_ready_memory_relief_mib"] for pair in pair_results]
        result["gpu_ready_memory_relief_mib_min"] = min(relief)
        result["gpu_ready_memory_relief_mib_max"] = max(relief)
        throughput_ratios = [
            pair["treatment_batch_metrics"]["median_useful_requests_per_s"] /
            pair["control_batch_metrics"]["median_useful_requests_per_s"]
            for pair in pair_results
        ]
        result["treatment_vs_control_throughput_ratio_median"] = (
            statistics.median(throughput_ratios))
        result["treatment_vs_control_throughput_ratio_min"] = min(throughput_ratios)
        result["treatment_vs_control_throughput_ratio_max"] = max(throughput_ratios)
    if not measurement_requested:
        result["measurement_status"] = "NOT_RUN"
        return result
    if len(valid_pairs) != len(pair_results):
        result["measurement_status"] = "INVALID_TIMELINE"
        return result
    if len(valid_pairs) < MIN_MEASUREMENT_PAIRS:
        result["measurement_status"] = "INSUFFICIENT_PAIRS"
        return result

    control_energy = sum(pair["control_energy_nj"] for pair in valid_pairs)
    control_uncertainty = sum(pair["control_uncertainty_nj"] for pair in valid_pairs)
    treatment_energy = sum(pair["treatment_energy_nj"] for pair in valid_pairs)
    treatment_uncertainty = sum(pair["treatment_uncertainty_nj"] for pair in valid_pairs)
    control_lower_nj = control_energy - control_uncertainty
    treatment_upper_nj = treatment_energy + treatment_uncertainty
    relief_lower_nj = control_lower_nj - treatment_upper_nj
    relief_positive = treatment_upper_nj < control_lower_nj
    relief_ge_10pct = (
        control_lower_nj > 0 and treatment_upper_nj * 10 <= control_lower_nj * 9
    )
    requests_per_pair, tokens_per_pair = next(iter(work_shapes))
    completed_requests = requests_per_pair * len(valid_pairs)
    generated_tokens = tokens_per_pair * len(valid_pairs)
    control_p95_us = max(
        pair["control_batch_metrics"]["p95_group_wall_us"] for pair in valid_pairs)
    treatment_p95_us = max(
        pair["treatment_batch_metrics"]["p95_group_wall_us"] for pair in valid_pairs)
    slo_met = control_p95_us <= slo_p95_us and treatment_p95_us <= slo_p95_us
    if not slo_met:
        label = "GPU_BOARD_DIAGNOSTIC_SLO_FAIL"
    elif relief_ge_10pct:
        label = "GPU_BOARD_DIAGNOSTIC_10PCT_OBSERVED_TOTAL_ENERGY_UNKNOWN"
    elif relief_positive:
        label = "GPU_BOARD_DIAGNOSTIC_BELOW_10PCT_TOTAL_ENERGY_UNKNOWN"
    else:
        label = "GPU_BOARD_DIAGNOSTIC_RELIEF_FAIL"
    result.update({
        "measurement_status": "EXPLORATORY_COMPLETE",
        "control_energy_sum_nj": control_energy,
        "control_uncertainty_sum_nj": control_uncertainty,
        "treatment_energy_sum_nj": treatment_energy,
        "treatment_uncertainty_sum_nj": treatment_uncertainty,
        "gross_delta_nj": control_energy - treatment_energy,
        "control_conservative_lower_nj": control_lower_nj,
        "treatment_conservative_upper_nj": treatment_upper_nj,
        "conservative_relief_lower_nj": relief_lower_nj,
        "board_relief_observed": relief_positive,
        "board_relief_ge_10pct": relief_ge_10pct,
        "completed_requests_sum": completed_requests,
        "generated_tokens_sum": generated_tokens,
        "slo_p95_us": slo_p95_us,
        "control_p95_group_wall_us_max": control_p95_us,
        "treatment_p95_group_wall_us_max": treatment_p95_us,
        "slo_met_all_pairs": slo_met,
        "control_energy_per_request": {
            "denominator_requests": completed_requests,
            "numerator_nj": control_energy,
        },
        "treatment_energy_per_request": {
            "denominator_requests": completed_requests,
            "numerator_nj": treatment_energy,
        },
        "control_energy_per_generated_token": {
            "denominator_tokens": generated_tokens,
            "numerator_nj": control_energy,
        },
        "treatment_energy_per_generated_token": {
            "denominator_tokens": generated_tokens,
            "numerator_nj": treatment_energy,
        },
        "label": label,
        "phone_energy_status": "UNKNOWN",
        "total_system_energy_status": "UNKNOWN",
        "limitations": [
            "NVML measures one GPU board, not server wall power",
            "phone, USB, charger, host CPU, DRAM, and PSU energy are excluded",
            "the run plan has no independent enumerable pre-registration anchor",
        ],
    })
    return result


def remote_file_info(serial, path, include_hash):
    quoted = shlex.quote(path)
    result = adb(
        serial, "shell",
        f"stat -c '%s' {quoted} && " +
        (f"sha256sum {quoted}" if include_hash else "true"))
    lines = result.stdout.splitlines()
    if not lines:
        raise ExperimentError(f"cannot stat remote file {serial}:{path}")
    record = {"path": path, "bytes": int(lines[0])}
    record["sha256"] = lines[1].split()[0] if include_hash else None
    return record


def build_effective_config(args, treatment_route):
    config = {
        key: value
        for key, value in vars(args).items()
        if key != "output"
    }
    for key in ("host_bin", "host_model", "deploy_phone_bin"):
        if config.get(key):
            config[key] = str(pathlib.Path(config[key]).resolve())
    config["resolved_treatment_route"] = treatment_route
    config["measurement_gates"] = {
        "minimum_pairs": MIN_MEASUREMENT_PAIRS,
        "minimum_independent_updates": MIN_INDEPENDENT_UPDATES,
        "minimum_generated_tokens_per_request": MIN_MEASURED_GENERATED_TOKENS,
        "maximum_requests_per_timeline": MAX_MEASURED_REQUESTS,
        "maximum_sample_gap_us": MAX_SAMPLE_GAP_US,
        "nvml_uncertainty_mw": NVML_UNCERTAINTY_MW,
        "nvml_averaging_window_us": NVML_AVERAGING_WINDOW_US,
        "pstate_format": "P0..P15",
        "pstate_policy": "OBSERVED_OUTCOME_TRANSITIONS_ALLOWED",
        "integration": "LEFT_EDGE_ZOH",
        "phone_thermal_coverage": "TREATMENT_CONTINUOUS_ON_DEVICE",
        "phone_thermal_max_gap_us": THERMAL_MAX_GAP_US,
        "phone_thermal_status_required": 0,
        "phone_temperature_list_required_nonempty": True,
        "s11_e0_route": "op15",
        "s11_e0_gpu_uuid": S11_E0_GPU_UUID,
        "s11_e0_batch_size": S11_E0_BATCH_SIZE,
        "s11_e0_generated_tokens": S11_E0_GENERATED_TOKENS,
        "s11_e0_driver_context": S11_E0_DRIVER_CONTEXT,
        "s11_e0_driver_max_prefill": S11_E0_DRIVER_MAX_PREFILL,
        "s11_e0_op15_end": S11_E0_OP15_END,
    }
    return config


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-bin", required=True)
    parser.add_argument("--host-model", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--route", choices=("auto", "op15", "two-phone"), default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="Explain why batching improves accelerator utilization.")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--n-gen", type=int, default=8)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--driver-context", type=int, default=4096)
    parser.add_argument("--driver-max-prefill", type=int, default=512)
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--readiness", action="store_true")
    parser.add_argument("--slo-p95-us", type=int)
    parser.add_argument("--sample-ms", type=int, default=100)
    parser.add_argument("--preroll-ms", type=int, default=200)
    parser.add_argument("--postroll-ms", type=int, default=1000)
    parser.add_argument("--phone-thermal-logger", action="store_true")
    parser.add_argument("--thermal-preroll-ms", type=int, default=800)
    parser.add_argument("--placement-cert", action="store_true")
    parser.add_argument("--host-ngl", type=int, default=99)
    parser.add_argument("--phone-ngl", type=int, default=99)
    parser.add_argument("--phone-backend", default="HTP0")
    parser.add_argument("--phone-decode-no-fa", action="store_true")
    parser.add_argument("--hexagon-mbuf-mib", type=int, default=3072)
    parser.add_argument("--op15-serial", default="3C15AU002CL00000")
    parser.add_argument("--op15-dir", default="/data/local/tmp/ls-npu")
    parser.add_argument("--op15-model", default="/data/local/tmp/ls-npu/12b-f16-head-0-2.gguf")
    parser.add_argument("--op15-end", type=int, default=2)
    parser.add_argument("--op15-port", type=int, default=15555)
    parser.add_argument("--op12-serial", default="5ae7a43d")
    parser.add_argument("--op12-dir", default="/data/local/tmp/ls-npu")
    parser.add_argument("--op12-model", default="/data/local/tmp/ls-npu/12b-f16-mid-2-3.gguf")
    parser.add_argument("--op12-end", type=int, default=3)
    parser.add_argument("--op12-port", type=int, default=15556)
    parser.add_argument("--phone-remote-bin", default="llama-layersplit-s11")
    parser.add_argument("--deploy-phone-bin")
    parser.add_argument("--ready-timeout", type=float, default=300)
    parser.add_argument("--stage-timeout", type=float, default=300)
    parser.add_argument("--run-timeout", type=float, default=900)
    parser.add_argument("--max-idle-gpu-util", type=int, default=5)
    parser.add_argument("--max-idle-gpu-memory-mib", type=int, default=512)
    args = parser.parse_args(argv)
    if args.measure and args.readiness:
        parser.error("--measure and --readiness are mutually exclusive")
    if args.n_gen <= 0 or args.requests <= 0 or args.warmups < 0 or args.pairs <= 0:
        parser.error("n-gen, requests, and pairs must be positive; warmups must be nonnegative")
    if args.batch_size <= 0 or args.batch_size > 64:
        parser.error("batch-size must be in 1..64")
    if args.requests % args.batch_size != 0:
        parser.error("requests must be divisible by batch-size")
    if args.driver_max_prefill <= 0 or args.driver_max_prefill > 512:
        parser.error("driver-max-prefill must be in 1..512")
    if args.driver_context < args.driver_max_prefill + args.n_gen:
        parser.error("driver-context must cover driver-max-prefill + n-gen")
    if args.measure and args.pairs < MIN_MEASUREMENT_PAIRS:
        parser.error(f"--measure requires --pairs >= {MIN_MEASUREMENT_PAIRS}")
    evidence_mode = args.measure or args.readiness
    mode_name = "--measure" if args.measure else "--readiness"
    if evidence_mode and args.route != "op15":
        parser.error(f"{mode_name} requires the frozen --route op15")
    if evidence_mode and args.gpu_uuid != S11_E0_GPU_UUID:
        parser.error(f"{mode_name} requires --gpu-uuid {S11_E0_GPU_UUID}")
    if evidence_mode and args.batch_size != S11_E0_BATCH_SIZE:
        parser.error(f"{mode_name} requires --batch-size {S11_E0_BATCH_SIZE}")
    if evidence_mode and args.n_gen != S11_E0_GENERATED_TOKENS:
        parser.error(f"{mode_name} requires --n-gen {S11_E0_GENERATED_TOKENS}")
    if evidence_mode and args.driver_context != S11_E0_DRIVER_CONTEXT:
        parser.error(
            f"{mode_name} requires --driver-context {S11_E0_DRIVER_CONTEXT}")
    if evidence_mode and args.driver_max_prefill != S11_E0_DRIVER_MAX_PREFILL:
        parser.error(
            f"{mode_name} requires --driver-max-prefill "
            f"{S11_E0_DRIVER_MAX_PREFILL}")
    if evidence_mode and args.op15_end != S11_E0_OP15_END:
        parser.error(f"{mode_name} requires --op15-end {S11_E0_OP15_END}")
    if evidence_mode and not args.chat:
        parser.error(f"{mode_name} requires --chat")
    if evidence_mode and args.warmups < 2:
        parser.error(f"{mode_name} requires --warmups >= 2")
    if evidence_mode and (args.host_ngl != 99 or args.phone_ngl != 99):
        parser.error(f"{mode_name} requires --host-ngl 99 and --phone-ngl 99")
    if evidence_mode and args.phone_backend != "HTP0":
        parser.error(f"{mode_name} requires --phone-backend HTP0")
    if evidence_mode and not args.phone_decode_no_fa:
        parser.error(f"{mode_name} requires --phone-decode-no-fa")
    if evidence_mode and not args.phone_thermal_logger:
        parser.error(f"{mode_name} requires --phone-thermal-logger")
    if evidence_mode and not args.placement_cert:
        parser.error(f"{mode_name} requires --placement-cert")
    if args.readiness and args.pairs != 1:
        parser.error("--readiness requires --pairs 1")
    if args.thermal_preroll_ms < 0:
        parser.error("thermal-preroll-ms must be nonnegative")
    if args.measure and args.requests > MAX_MEASURED_REQUESTS:
        parser.error(
            f"--measure requires --requests <= {MAX_MEASURED_REQUESTS}")
    if args.measure and (args.slo_p95_us is None or args.slo_p95_us <= 0):
        parser.error("--measure requires positive --slo-p95-us")
    if not args.measure and args.slo_p95_us is not None:
        parser.error("--slo-p95-us requires --measure")
    if args.op15_end <= 0 or args.op12_end <= args.op15_end:
        parser.error("layer boundaries require 0 < op15-end < op12-end")
    if pathlib.PurePosixPath(args.phone_remote_bin).name != args.phone_remote_bin:
        parser.error("phone-remote-bin must be a basename")
    if args.op15_port == args.op12_port:
        parser.error("op15-port and op12-port must differ")
    if args.hexagon_mbuf_mib <= 0:
        parser.error("hexagon-mbuf-mib must be positive")
    if args.sample_ms <= 0 or args.preroll_ms < 0 or args.postroll_ms < 0:
        parser.error("sample-ms must be positive; preroll/postroll must be nonnegative")
    if args.ready_timeout <= 0 or args.stage_timeout <= 0 or args.run_timeout <= 0:
        parser.error("timeouts must be positive")
    if args.max_idle_gpu_util < 0 or args.max_idle_gpu_memory_mib < 0:
        parser.error("idle GPU thresholds must be nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    output = pathlib.Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ExperimentError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if not pathlib.Path(args.host_bin).is_file():
        raise ExperimentError(f"host binary not found: {args.host_bin}")
    if not pathlib.Path(args.host_model).is_file():
        raise ExperimentError(f"host model not found: {args.host_model}")
    if not LAYERSPLIT_SOURCE_PATH.is_file():
        raise ExperimentError(
            f"layersplit source not found: {LAYERSPLIT_SOURCE_PATH}")
    if not LLAMA_MODEL_SOURCE_PATH.is_file():
        raise ExperimentError(
            f"llama model source not found: {LLAMA_MODEL_SOURCE_PATH}")

    serials = connected_serials()
    if args.op15_serial not in serials:
        raise ExperimentError(f"OP15 is not connected: {args.op15_serial}")
    if args.route == "two-phone" and args.op12_serial not in serials:
        raise ExperimentError(f"OP12 is not connected: {args.op12_serial}")
    if args.route == "auto":
        treatment_route = (
            ROUTE_TWO_PHONE if args.op12_serial in serials else ROUTE_OP15)
    else:
        treatment_route = (
            ROUTE_TWO_PHONE if args.route == "two-phone" else ROUTE_OP15)

    gpu = query_gpu(args.gpu_uuid)
    if (args.measure or args.readiness) and (
            gpu["utilization_pct"] > args.max_idle_gpu_util or
            gpu["memory_used_mib"] > args.max_idle_gpu_memory_mib):
        raise ExperimentError(f"selected GPU is not idle: {gpu}")

    deployed = {}
    if args.deploy_phone_bin:
        deployed[args.op15_serial] = deploy_binary(
            args.op15_serial, args.deploy_phone_bin,
            args.op15_dir, args.phone_remote_bin)
        if treatment_route == ROUTE_TWO_PHONE:
            deployed[args.op12_serial] = deploy_binary(
                args.op12_serial, args.deploy_phone_bin,
                args.op12_dir, args.phone_remote_bin)

    effective_config = build_effective_config(args, treatment_route)
    measurement_config = {
        "requested": args.measure,
        "instrument": "NVML_BOARD" if args.measure else "NONE",
        "scope": "GPU_BOARD" if args.measure else "NONE",
        "sample_interval_ms": args.sample_ms,
        "preroll_ms": args.preroll_ms,
        "postroll_ms": args.postroll_ms,
        "minimum_pairs": MIN_MEASUREMENT_PAIRS,
        "minimum_independent_updates": MIN_INDEPENDENT_UPDATES,
        "minimum_generated_tokens_per_request": MIN_MEASURED_GENERATED_TOKENS,
        "maximum_requests_per_timeline": MAX_MEASURED_REQUESTS,
        "maximum_sample_gap_us": MAX_SAMPLE_GAP_US,
        "nvml_uncertainty_mw": NVML_UNCERTAINTY_MW,
        "nvml_averaging_window_us": NVML_AVERAGING_WINDOW_US,
        "max_idle_gpu_util_pct": args.max_idle_gpu_util,
        "max_idle_gpu_memory_mib": args.max_idle_gpu_memory_mib,
        "phone_thermal_coverage": "TREATMENT_CONTINUOUS_ON_DEVICE",
        "phone_thermal_max_gap_us": THERMAL_MAX_GAP_US,
        "phone_thermal_status_required": 0,
        "phone_temperature_list_required_nonempty": True,
        "pstate_policy": "OBSERVED_OUTCOME_TRANSITIONS_ALLOWED",
        "slo_p95_us": args.slo_p95_us,
    }
    plan = {
        "schema": PLAN_SCHEMA,
        "route_record_schema": ROUTE_RECORD_SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "GPU_BOARD" if args.measure else "MECHANICS_ONLY",
        "formal_claim_authorized": False,
        "treatment_route": treatment_route,
        "pair_order": paired_order(args.pairs, treatment_route),
        "implementation": {
            "runner_path": str(RUNNER_PATH.relative_to(REPO_ROOT)),
            "runner_sha256": sha256_file(RUNNER_PATH),
            "layersplit_source_path": str(
                LAYERSPLIT_SOURCE_PATH.relative_to(REPO_ROOT)),
            "layersplit_source_sha256": sha256_file(LAYERSPLIT_SOURCE_PATH),
            "llama_model_source_path": str(
                LLAMA_MODEL_SOURCE_PATH.relative_to(REPO_ROOT)),
            "llama_model_source_sha256": sha256_file(
                LLAMA_MODEL_SOURCE_PATH),
            "effective_config_sha256": sha256_bytes(
                canonical_bytes(effective_config)),
        },
        "effective_config": effective_config,
        "measurement_config": measurement_config,
        "workload": {
            "prompt": args.prompt,
            "chat": args.chat,
            "n_gen": args.n_gen,
            "requests": args.requests,
            "batch_size": args.batch_size,
            "measured_groups": args.requests // args.batch_size,
            "warmup_groups": args.warmups,
            "driver_context_per_sequence": args.driver_context,
            "driver_max_prefill_tokens": args.driver_max_prefill,
            "sampling": "greedy_argmax",
        },
        "host": {
            "binary": str(pathlib.Path(args.host_bin).resolve()),
            "binary_sha256": sha256_file(args.host_bin),
            "model": str(pathlib.Path(args.host_model).resolve()),
            "model_bytes": pathlib.Path(args.host_model).stat().st_size,
            "model_sha256": sha256_file(args.host_model),
            "gpu": gpu,
        },
        "phones": {
            "op15": {
                "serial": args.op15_serial,
                "layer_range": [0, args.op15_end],
                "model": remote_file_info(
                    args.op15_serial, args.op15_model, True),
                "binary": remote_file_info(
                    args.op15_serial,
                    f"{args.op15_dir}/{args.phone_remote_bin}", True),
            },
            "op12": (
                {
                    "serial": args.op12_serial,
                    "layer_range": [args.op15_end, args.op12_end],
                    "model": remote_file_info(
                        args.op12_serial, args.op12_model, True),
                    "binary": remote_file_info(
                        args.op12_serial,
                        f"{args.op12_dir}/{args.phone_remote_bin}", True),
                }
                if treatment_route == ROUTE_TWO_PHONE else None
            ),
            "deployed_binary_sha256": deployed,
            "backend": args.phone_backend,
        },
    }
    plan["plan_sha256"] = sha256_bytes(canonical_bytes(plan))
    write_json(output / "plan.json", plan)

    order = plan["pair_order"]
    runs = []
    for slot_index, slot in enumerate(order):
        run_dir = output / "runs" / f"slot{slot_index:02d}_{slot['route'].lower()}"
        print(
            f"[{slot_index + 1}/{len(order)}] pair={slot['pair_index']} "
            f"route={slot['route']}", flush=True)
        run = run_route(args, slot["route"], run_dir)
        run["slot_index"] = slot_index
        run["pair_index"] = slot["pair_index"]
        write_json(run_dir / "run.json", run)
        runs.append(run)
        if slot_index % 2 == 1:
            left, right = runs[-2], runs[-1]
            control = left if left["route"] == ROUTE_CONTROL else right
            treatment = right if left["route"] == ROUTE_CONTROL else left
            exact, reason = same_work(control["records"], treatment["records"])
            if not exact:
                failure = {
                    "schema": FAILURE_SCHEMA,
                    "route_record_schema": ROUTE_RECORD_SCHEMA,
                    "plan_sha256": plan["plan_sha256"],
                    "pair_index": slot["pair_index"],
                    "reason": reason,
                    "control_token_sha256": control["token_sha256"],
                    "treatment_token_sha256": treatment["token_sha256"],
                }
                write_json(output / "failure.json", failure)
                raise ExperimentError(
                    f"pair {slot['pair_index']} exact-output mismatch: {reason}")

    pair_results = summarize_pairs(order, runs)
    if args.measure:
        # CP1.1: reopen and reintegrate every measured power stream from its
        # hashed bytes before aggregation. Fails closed on any structural defect.
        pair_results = reverify_pairs(pair_results)
    elif args.readiness:
        pair_results = reverify_readiness_pairs(pair_results)
    aggregate = (
        aggregate_readiness_result(pair_results) if args.readiness else
        aggregate_result(pair_results, args.measure, args.slo_p95_us))
    summary = {
        "schema": RESULT_SCHEMA,
        "route_record_schema": ROUTE_RECORD_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "treatment_route": treatment_route,
        "pairs": pair_results,
        "aggregate": aggregate,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary["aggregate"], sort_keys=True))
    if not aggregate["exact_work_all_pairs"]:
        return 2
    if args.measure and aggregate["measurement_status"] != "EXPLORATORY_COMPLETE":
        return 3
    if args.readiness and aggregate["readiness_status"] != "PASS":
        return 3
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExperimentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
