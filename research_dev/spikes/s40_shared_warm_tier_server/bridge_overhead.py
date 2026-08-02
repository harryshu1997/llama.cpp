#!/usr/bin/env python3
"""Measure the fresh-process gateway bridge against direct Unix sockets."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable

from evidence_common import (
    EvidenceError,
    canonical_bytes,
    digest_file,
    is_int,
    parse_json,
    percentile,
    require,
    require_int,
    require_string,
    validate_digest,
)


FANOUTS = (1, 8)
MODES = ("DIRECT_SOCKET", "FRESH_PYTHON_BRIDGE")
MINIMUM_SAMPLES_PER_CELL = 50
ROW_KEYS = {
    "batch_index",
    "batch_makespan_ns",
    "command_bytes",
    "fanout",
    "item_index",
    "latency_ns",
    "mode",
    "result_bytes",
}
NATIVE_ROW_KEYS = {
    "batch_index",
    "batch_makespan_ns",
    "command_bytes",
    "command_id",
    "completed_ns",
    "item_index",
    "latency_ns",
    "result_bytes",
    "started_ns",
}
NATIVE_KEYS = {
    "fanout",
    "rows",
    "sample_count",
    "schema",
    "socket_path",
    "transport",
}
NATIVE_BINARY_KEYS = {
    "bytes",
    "path",
    "sha256",
}
NATIVE_INVOCATION_KEYS = {
    "argv",
    "completed_ns",
    "exit_code",
    "fanout",
    "peer_pid",
    "peer_start_time_ticks",
    "started_ns",
    "stderr_base64",
    "stdout_base64",
}
COMBINED_KEYS = {
    "executor_bundle_manifest_sha256",
    "host_boot_id",
    "native_bench_binary",
    "native_invocations",
    "python_executable_sha256",
    "rows",
    "schema",
    "summary",
}


def recv_line(connection: socket.socket, limit: int = 4 * 1024 * 1024) -> bytes:
    data = bytearray()
    while len(data) <= limit:
        block = connection.recv(min(65536, limit + 1 - len(data)))
        if not block:
            break
        data.extend(block)
        if data.endswith(b"\n"):
            break
    require(data.endswith(b"\n") and len(data) <= limit,
            "bridge diagnostic: invalid frame")
    return bytes(data)


def command(command_id: int) -> bytes:
    return canonical_bytes({
        "command_id": command_id,
        "controller_epoch": 1,
        "executor_id": "noop",
        "executor_instance_id": "noop-instance",
        "kind": 0,
        "max_output_tokens": 1,
        "model_id": "model",
        "request": {
            "committed_output_tokens": [],
            "model_id": "model",
            "owner_id": "noop",
            "ownership_epoch": 1,
            "position": 128,
            "prompt_tokens": list(range(128)),
            "publication_index": 0,
            "request_id": f"request-{command_id}",
            "state": 1,
        },
        "request_id": f"request-{command_id}",
        "schema": "llama-server-warm-tier-command-v3",
        "total_output_tokens": 8,
    })


def current_process_start_time_ticks() -> int:
    raw = Path(f"/proc/{os.getpid()}/stat").read_text(encoding="ascii")
    marker = raw.rfind(") ")
    require(marker > 0, "bridge diagnostic: invalid process stat")
    fields = raw[marker + 2:].split()
    require(len(fields) > 19, "bridge diagnostic: truncated process stat")
    try:
        result = int(fields[19])
    except ValueError as error:
        raise EvidenceError(
            "bridge diagnostic: invalid process start time") from error
    require(result > 0, "bridge diagnostic: invalid process start time")
    return result


def result(raw: bytes) -> bytes:
    value = parse_json(raw, "bridge diagnostic command")
    request = value["request"]
    return canonical_bytes({
        "command_id": value["command_id"],
        "controller_epoch": value["controller_epoch"],
        "detail": "",
        "executor_id": value["executor_id"],
        "executor_instance_id": value["executor_instance_id"],
        "has_replay_snapshot": False,
        "kind": value["kind"],
        "model_id": value["model_id"],
        "publications": [{
            "owner_id": value["executor_id"],
            "ownership_epoch": request["ownership_epoch"],
            "position": request["position"],
            "publication_index": request["publication_index"],
            "token": 1,
        }],
        "replay_snapshot": None,
        "request_complete": False,
        "request_id": value["request_id"],
        "schema": "llama-server-warm-tier-result-v2",
        "success": True,
    })


class NoopGateway:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(str(path))
        self.listener.listen(64)
        self.listener.settimeout(0.1)
        self.stop = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run_connection(self, connection: socket.socket) -> None:
        with connection:
            raw = recv_line(connection)
            connection.sendall(result(raw))

    def run(self) -> None:
        workers: list[threading.Thread] = []
        try:
            while not self.stop.is_set():
                try:
                    connection, _ = self.listener.accept()
                except socket.timeout:
                    continue
                worker = threading.Thread(
                    target=self.run_connection,
                    args=(connection,),
                    daemon=True,
                )
                worker.start()
                workers.append(worker)
        except BaseException as error:
            self.error = error
        finally:
            for worker in workers:
                worker.join()

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=5)
        self.listener.close()
        require(not self.thread.is_alive(), "no-op gateway did not stop")
        if self.error is not None:
            raise EvidenceError(f"no-op gateway failed: {self.error}")


def direct_call(path: Path, raw: bytes) -> bytes:
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        connection.connect(str(path))
        connection.sendall(raw)
        connection.shutdown(socket.SHUT_WR)
        return recv_line(connection)
    finally:
        connection.close()


def bridge_environment(bundle: Path) -> dict[str, str]:
    manifest = bundle / "MANIFEST.json"
    require(manifest.is_file(), "executor bundle manifest is missing")
    return {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(bundle),
        "S40_EXECUTOR_BUNDLE": "1",
        "S40_EXECUTOR_BUNDLE_MANIFEST": str(manifest),
        "S40_EXECUTOR_BUNDLE_SHA256": digest_file(manifest),
    }


def bridge_call(
        bundle: Path,
        path: Path,
        raw: bytes,
) -> bytes:
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-s",
            "-P",
            str(bundle / "gateway_bridge.py"),
            "--socket",
            str(path),
            "--timeout",
            "10",
        ],
        check=False,
        input=raw,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=bridge_environment(bundle),
        timeout=15,
    )
    require(
        completed.returncode == 0 and not completed.stderr,
        "fresh Python bridge failed",
    )
    return completed.stdout


def run_batch(
        operation: Callable[[bytes], bytes],
        fanout: int,
        batch_index: int,
        command_id: int,
) -> list[dict[str, Any]]:
    inputs = [command(command_id + index) for index in range(fanout)]

    def one(index: int) -> tuple[int, int, int]:
        started = time.monotonic_ns()
        output = operation(inputs[index])
        elapsed = time.monotonic_ns() - started
        require(output == result(inputs[index]),
                "bridge diagnostic result mismatch")
        return elapsed, len(inputs[index]), len(output)

    batch_started = time.monotonic_ns()
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=fanout) as executor:
        measured = list(executor.map(one, range(fanout)))
    makespan = time.monotonic_ns() - batch_started
    return [{
        "batch_index": batch_index,
        "batch_makespan_ns": makespan,
        "command_bytes": command_bytes,
        "fanout": fanout,
        "item_index": index,
        "latency_ns": latency_ns,
        "mode": "",
        "result_bytes": result_bytes,
    } for index, (latency_ns, command_bytes, result_bytes)
        in enumerate(measured)]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for mode in MODES:
        for fanout in FANOUTS:
            cell = [
                row for row in rows
                if row["mode"] == mode and row["fanout"] == fanout
            ]
            latencies = [row["latency_ns"] for row in cell]
            makespans = sorted({
                (row["batch_index"], row["batch_makespan_ns"])
                for row in cell
            })
            output.append({
                "batch_makespan_p50_ns": percentile(
                    [item[1] for item in makespans], 1, 2),
                "batch_makespan_p95_ns": percentile(
                    [item[1] for item in makespans], 19, 20),
                "fanout": fanout,
                "latency_p50_ns": percentile(latencies, 1, 2),
                "latency_p95_ns": percentile(latencies, 19, 20),
                "mode": mode,
                "sample_count": len(cell),
            })
    return output


def validate_measurement(
        value: Any,
        minimum_samples: int = MINIMUM_SAMPLES_PER_CELL,
) -> dict[str, Any]:
    require(isinstance(value, dict), "bridge measurement: expected object")
    require(
        set(value) == {
            "executor_bundle_manifest_sha256",
            "host_boot_id",
            "python_executable_sha256",
            "rows",
            "schema",
            "summary",
        }
        and value["schema"] == "s40-bridge-overhead-v1",
        "bridge measurement: schema mismatch",
    )
    validate_digest(
        value["executor_bundle_manifest_sha256"],
        "bridge measurement.executor bundle",
    )
    validate_digest(
        value["python_executable_sha256"],
        "bridge measurement.python executable",
    )
    require_string(value["host_boot_id"], "bridge measurement.host boot")
    rows = value["rows"]
    require(isinstance(rows, list), "bridge measurement.rows: expected array")
    seen = set()
    for index, row in enumerate(rows):
        field = f"bridge measurement.rows[{index}]"
        require(
            isinstance(row, dict) and set(row) == ROW_KEYS,
            f"{field}: key set mismatch",
        )
        require(row["mode"] in MODES, f"{field}: invalid mode")
        fanout = require_int(row["fanout"], f"{field}.fanout", 1)
        require(fanout in FANOUTS, f"{field}: invalid fanout")
        batch = require_int(row["batch_index"], f"{field}.batch_index")
        item = require_int(row["item_index"], f"{field}.item_index")
        require(item < fanout, f"{field}: item outside fanout")
        key = row["mode"], fanout, batch, item
        require(key not in seen, f"{field}: duplicate sample")
        seen.add(key)
        for name in (
                "batch_makespan_ns",
                "command_bytes",
                "latency_ns",
                "result_bytes"):
            require_int(row[name], f"{field}.{name}", 1)
    derived = summarize(rows)
    require(value["summary"] == derived,
            "bridge measurement: summary mismatch")
    for cell in derived:
        require(
            cell["sample_count"] >= minimum_samples,
            "bridge measurement: insufficient samples",
        )
    direct = {
        row["fanout"]: row for row in derived
        if row["mode"] == "DIRECT_SOCKET"
    }
    bridge = {
        row["fanout"]: row for row in derived
        if row["mode"] == "FRESH_PYTHON_BRIDGE"
    }
    return {
        "incremental_p95_ns": {
            str(fanout): max(
                0,
                bridge[fanout]["latency_p95_ns"]
                - direct[fanout]["latency_p95_ns"],
            )
            for fanout in FANOUTS
        },
        "status": "MEASURED",
        "summary": derived,
    }


def validate_native_measurement(
        value: Any,
        minimum_samples: int = MINIMUM_SAMPLES_PER_CELL,
) -> dict[str, Any]:
    require(isinstance(value, dict) and set(value) == NATIVE_KEYS,
            "native executor measurement: schema mismatch")
    require(
        value["schema"] == "s40-native-unix-executor-bench-v1"
        and value["transport"] == "UNIX_SOCKET",
        "native executor measurement: identity mismatch",
    )
    socket_path = Path(require_string(
        value["socket_path"], "native executor measurement.socket_path"))
    require(
        socket_path.is_absolute()
        and len(str(socket_path).encode("ascii", errors="ignore")) <= 103
        and str(socket_path).isascii(),
        "native executor measurement: invalid socket path",
    )
    fanout = require_int(
        value["fanout"], "native executor measurement.fanout", 1)
    require(fanout in FANOUTS,
            "native executor measurement: unsupported fanout")
    sample_count = require_int(
        value["sample_count"],
        "native executor measurement.sample_count",
        minimum_samples,
    )
    rows = value["rows"]
    require(
        isinstance(rows, list)
        and len(rows) == sample_count
        and sample_count % fanout == 0,
        "native executor measurement: row count mismatch",
    )
    command_ids = set()
    batch_rows: dict[int, list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        field = f"native executor measurement.rows[{index}]"
        require(isinstance(row, dict) and set(row) == NATIVE_ROW_KEYS,
                f"{field}: key set mismatch")
        batch = require_int(row["batch_index"], f"{field}.batch_index")
        item = require_int(row["item_index"], f"{field}.item_index")
        require(item < fanout, f"{field}: item outside fanout")
        command_id = require_int(
            row["command_id"], f"{field}.command_id", 1)
        require(command_id not in command_ids,
                f"{field}: duplicate command ID")
        command_ids.add(command_id)
        started = require_int(row["started_ns"], f"{field}.started_ns", 1)
        completed = require_int(
            row["completed_ns"], f"{field}.completed_ns", started)
        latency = require_int(row["latency_ns"], f"{field}.latency_ns", 1)
        require(latency == completed - started,
                f"{field}: latency mismatch")
        makespan = require_int(
            row["batch_makespan_ns"],
            f"{field}.batch_makespan_ns",
            latency,
        )
        for name in ("command_bytes", "result_bytes"):
            require_int(row[name], f"{field}.{name}", 1)
        batch_rows.setdefault(batch, []).append(row)
    expected_batches = list(range(sample_count // fanout))
    require(sorted(batch_rows) == expected_batches,
            "native executor measurement: batch sequence mismatch")
    makespans = []
    for batch in expected_batches:
        current = batch_rows[batch]
        require(
            len(current) == fanout
            and sorted(row["item_index"] for row in current)
            == list(range(fanout))
            and len({row["batch_makespan_ns"] for row in current}) == 1,
            "native executor measurement: incomplete batch",
        )
        makespans.append(current[0]["batch_makespan_ns"])
    latencies = [row["latency_ns"] for row in rows]
    return {
        "batch_makespan_p50_ns": percentile(makespans, 1, 2),
        "batch_makespan_p95_ns": percentile(makespans, 19, 20),
        "fanout": fanout,
        "latency_p50_ns": percentile(latencies, 1, 2),
        "latency_p95_ns": percentile(latencies, 19, 20),
        "sample_count": sample_count,
        "status": "MEASURED",
    }


def validate_native_transport_gate(
        native_measurements: list[Any],
        direct_measurement: Any,
        fastest_desktop_execute_ns: int,
        minimum_samples: int = MINIMUM_SAMPLES_PER_CELL,
) -> dict[str, Any]:
    require(
        is_int(fastest_desktop_execute_ns)
        and fastest_desktop_execute_ns > 0,
        "native executor gate: invalid desktop execute duration",
    )
    require(
        isinstance(native_measurements, list)
        and len(native_measurements) == len(FANOUTS),
        "native executor gate: measurement count mismatch",
    )
    native = [
        validate_native_measurement(value, minimum_samples)
        for value in native_measurements
    ]
    require(
        sorted(row["fanout"] for row in native) == list(FANOUTS),
        "native executor gate: fanout set mismatch",
    )
    direct = validate_measurement(
        direct_measurement, minimum_samples)["summary"]
    direct_by_fanout = {
        row["fanout"]: row for row in direct
        if row["mode"] == "DIRECT_SOCKET"
    }
    threshold = min(1_000_000, fastest_desktop_execute_ns // 20)
    require(threshold > 0,
            "native executor gate: threshold rounds to zero")
    cells = []
    for row in sorted(native, key=lambda item: item["fanout"]):
        fanout = row["fanout"]
        incremental = max(
            0,
            row["latency_p95_ns"]
            - direct_by_fanout[fanout]["latency_p95_ns"],
        )
        cells.append({
            "direct_p95_ns":
                direct_by_fanout[fanout]["latency_p95_ns"],
            "fanout": fanout,
            "incremental_p95_ns": incremental,
            "native_p95_ns": row["latency_p95_ns"],
            "pass": incremental <= threshold,
            "threshold_ns": threshold,
        })
    return {
        "cells": cells,
        "performance_claim_authorized":
            all(cell["pass"] for cell in cells),
        "status": (
            "PASS" if all(cell["pass"] for cell in cells)
            else "FAIL_MATERIAL_TRANSPORT_OVERHEAD"
        ),
        "threshold_ns": threshold,
    }


def _decode_base64(value: Any, field: str) -> bytes:
    require(isinstance(value, str) and value.isascii(),
            f"{field}: expected ASCII string")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise EvidenceError(f"{field}: invalid base64") from error
    require(base64.b64encode(raw).decode("ascii") == value,
            f"{field}: noncanonical base64")
    return raw


def validate_combined_measurement(
        value: Any,
        minimum_samples: int = MINIMUM_SAMPLES_PER_CELL,
) -> dict[str, Any]:
    require(isinstance(value, dict) and set(value) == COMBINED_KEYS,
            "transport measurement: schema mismatch")
    require(value["schema"] == "s40-transport-overhead-v2",
            "transport measurement: unsupported schema")
    legacy = {
        "executor_bundle_manifest_sha256":
            value["executor_bundle_manifest_sha256"],
        "host_boot_id": value["host_boot_id"],
        "python_executable_sha256": value["python_executable_sha256"],
        "rows": value["rows"],
        "schema": "s40-bridge-overhead-v1",
        "summary": value["summary"],
    }
    legacy_result = validate_measurement(legacy, minimum_samples)
    binary = value["native_bench_binary"]
    require(
        isinstance(binary, dict) and set(binary) == NATIVE_BINARY_KEYS,
        "transport measurement: native binary schema mismatch",
    )
    binary_path = Path(require_string(
        binary["path"], "transport measurement.native binary path"))
    require(binary_path.is_absolute(),
            "transport measurement: native binary path is not absolute")
    binary_bytes = require_int(
        binary["bytes"], "transport measurement.native binary bytes", 1)
    binary_sha256 = validate_digest(
        binary["sha256"], "transport measurement.native binary SHA-256")
    invocations = value["native_invocations"]
    require(
        isinstance(invocations, list)
        and len(invocations) == len(FANOUTS),
        "transport measurement: native invocation count mismatch",
    )
    native_values = []
    seen_fanouts = set()
    socket_paths = set()
    for index, invocation in enumerate(invocations):
        field = f"transport measurement.native_invocations[{index}]"
        require(
            isinstance(invocation, dict)
            and set(invocation) == NATIVE_INVOCATION_KEYS,
            f"{field}: key set mismatch",
        )
        fanout = require_int(invocation["fanout"], f"{field}.fanout", 1)
        require(fanout in FANOUTS and fanout not in seen_fanouts,
                f"{field}: duplicate or unsupported fanout")
        seen_fanouts.add(fanout)
        started = require_int(
            invocation["started_ns"], f"{field}.started_ns", 1)
        require_int(
            invocation["completed_ns"], f"{field}.completed_ns", started)
        require(invocation["exit_code"] == 0,
                f"{field}: native benchmark failed")
        peer_pid = require_int(
            invocation["peer_pid"], f"{field}.peer_pid", 2)
        peer_start_time_ticks = require_int(
            invocation["peer_start_time_ticks"],
            f"{field}.peer_start_time_ticks",
            1,
        )
        stderr = _decode_base64(
            invocation["stderr_base64"], f"{field}.stderr")
        stdout = _decode_base64(
            invocation["stdout_base64"], f"{field}.stdout")
        require(not stderr and stdout.endswith(b"\n"),
                f"{field}: invalid native output")
        native = parse_json(stdout, f"{field}.stdout")
        require(
            isinstance(native, dict)
            and stdout == canonical_bytes(native),
            f"{field}: native output is not canonical JSON",
        )
        validate_native_measurement(native, minimum_samples)
        require(native["fanout"] == fanout,
                f"{field}: output fanout mismatch")
        socket_paths.add(native["socket_path"])
        argv = invocation["argv"]
        require(
            isinstance(argv, list)
            and argv == [
                str(binary_path),
                "--unix-bench",
                native["socket_path"],
                str(native["sample_count"]),
                str(fanout),
                str(peer_pid),
                str(peer_start_time_ticks),
            ],
            f"{field}: argv mismatch",
        )
        native_values.append(native)
    require(seen_fanouts == set(FANOUTS) and len(socket_paths) == 1,
            "transport measurement: gateway socket mismatch")
    return {
        "direct": legacy_result,
        "native_bench_binary_bytes": binary_bytes,
        "native_bench_binary_sha256": binary_sha256,
        "native_measurements": native_values,
        "socket_path": next(iter(socket_paths)),
        "status": "MEASURED",
    }


def validate_combined_transport_gate(
        value: Any,
        fastest_desktop_execute_ns: int,
        minimum_samples: int = MINIMUM_SAMPLES_PER_CELL,
) -> dict[str, Any]:
    validated = validate_combined_measurement(value, minimum_samples)
    legacy = {
        "executor_bundle_manifest_sha256":
            value["executor_bundle_manifest_sha256"],
        "host_boot_id": value["host_boot_id"],
        "python_executable_sha256": value["python_executable_sha256"],
        "rows": value["rows"],
        "schema": "s40-bridge-overhead-v1",
        "summary": value["summary"],
    }
    result = validate_native_transport_gate(
        validated["native_measurements"],
        legacy,
        fastest_desktop_execute_ns,
        minimum_samples,
    )
    return {
        **result,
        "native_bench_binary_sha256":
            validated["native_bench_binary_sha256"],
        "socket_path": validated["socket_path"],
    }


def measure(
        executor_bundle: Path,
        samples_per_cell: int = MINIMUM_SAMPLES_PER_CELL,
        warmup_batches: int = 3,
        native_bench: Path | None = None,
) -> dict[str, Any]:
    require(
        executor_bundle.is_absolute()
        and (executor_bundle / "gateway_bridge.py").is_file(),
        "executor bundle path is invalid",
    )
    require(samples_per_cell >= 1 and warmup_batches >= 0,
            "bridge measurement counts are invalid")
    rows = []
    native_invocations = []
    with tempfile.TemporaryDirectory(prefix="s40_bridge_") as directory:
        socket_path = Path(directory) / "noop.sock"
        gateway = NoopGateway(socket_path)
        try:
            operations = {
                "DIRECT_SOCKET":
                    lambda raw: direct_call(socket_path, raw),
                "FRESH_PYTHON_BRIDGE":
                    lambda raw: bridge_call(
                        executor_bundle, socket_path, raw),
            }
            command_id = 1
            for mode in MODES:
                for fanout in FANOUTS:
                    for batch in range(warmup_batches):
                        run_batch(
                            operations[mode],
                            fanout,
                            -(batch + 1),
                            command_id,
                        )
                        command_id += fanout
                    batches = (
                        samples_per_cell + fanout - 1) // fanout
                    for batch in range(batches):
                        batch_rows = run_batch(
                            operations[mode],
                            fanout,
                            batch,
                            command_id,
                        )
                        command_id += fanout
                        for row in batch_rows:
                            if len([
                                    existing for existing in rows
                                    if existing["mode"] == mode
                                    and existing["fanout"] == fanout
                            ]) >= samples_per_cell:
                                break
                            row["mode"] = mode
                            rows.append(row)
            if native_bench is not None:
                require(
                    native_bench.is_absolute()
                    and native_bench.is_file()
                    and os.access(native_bench, os.X_OK),
                    "native benchmark binary is invalid",
                )
                peer_pid = os.getpid()
                peer_start_time_ticks = current_process_start_time_ticks()
                for fanout in FANOUTS:
                    sample_count = (
                        (samples_per_cell + fanout - 1) // fanout
                    ) * fanout
                    argv = [
                        str(native_bench),
                        "--unix-bench",
                        str(socket_path),
                        str(sample_count),
                        str(fanout),
                        str(peer_pid),
                        str(peer_start_time_ticks),
                    ]
                    started_ns = time.monotonic_ns()
                    completed = subprocess.run(
                        argv,
                        check=False,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=120,
                    )
                    completed_ns = time.monotonic_ns()
                    require(
                        completed.returncode == 0
                        and not completed.stderr
                        and 0 < len(completed.stdout) <= 64 * 1024 * 1024,
                        "native Unix executor benchmark failed",
                    )
                    native_invocations.append({
                        "argv": argv,
                        "completed_ns": completed_ns,
                        "exit_code": completed.returncode,
                        "fanout": fanout,
                        "peer_pid": peer_pid,
                        "peer_start_time_ticks":
                            peer_start_time_ticks,
                        "started_ns": started_ns,
                        "stderr_base64": base64.b64encode(
                            completed.stderr).decode("ascii"),
                        "stdout_base64": base64.b64encode(
                            completed.stdout).decode("ascii"),
                    })
        finally:
            gateway.close()
    value = {
        "executor_bundle_manifest_sha256": digest_file(
            executor_bundle / "MANIFEST.json"),
        "host_boot_id": Path(
            "/proc/sys/kernel/random/boot_id").read_text(
                encoding="ascii").strip(),
        "python_executable_sha256": digest_file(Path(sys.executable)),
        "rows": rows,
        "schema": "s40-bridge-overhead-v1",
        "summary": summarize(rows),
    }
    validate_measurement(value, samples_per_cell)
    if native_bench is not None:
        value = {
            "executor_bundle_manifest_sha256":
                value["executor_bundle_manifest_sha256"],
            "host_boot_id": value["host_boot_id"],
            "native_bench_binary": {
                "bytes": native_bench.stat().st_size,
                "path": str(native_bench),
                "sha256": digest_file(native_bench),
            },
            "native_invocations": native_invocations,
            "python_executable_sha256":
                value["python_executable_sha256"],
            "rows": value["rows"],
            "schema": "s40-transport-overhead-v2",
            "summary": value["summary"],
        }
        validate_combined_measurement(value, samples_per_cell)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executor-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--samples-per-cell",
        type=int,
        default=MINIMUM_SAMPLES_PER_CELL,
    )
    parser.add_argument("--native-bench", type=Path)
    args = parser.parse_args()
    try:
        value = measure(
            args.executor_bundle,
            args.samples_per_cell,
            native_bench=args.native_bench,
        )
        require(args.output.is_absolute() and not args.output.exists(),
                "output path is invalid")
        with args.output.open("xb", buffering=0) as output:
            output.write(canonical_bytes(value))
            output.flush()
            os.fsync(output.fileno())
    except (EvidenceError, OSError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}")
        return 2
    validated = (
        validate_combined_measurement(value, args.samples_per_cell)
        if args.native_bench is not None
        else validate_measurement(value, args.samples_per_cell)
    )
    print(canonical_bytes(validated).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
