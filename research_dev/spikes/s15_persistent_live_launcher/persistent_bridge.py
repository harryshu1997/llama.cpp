#!/usr/bin/env python3
"""Translate the typed S15 JSONL contract to the LayerSplit child contract."""

from __future__ import annotations

import argparse
from collections import deque
import subprocess
import sys
import threading
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
PERSISTENT = HERE.parent / "s15_persistent_runtime"
S15 = HERE.parent / "s15_runtime_dispatch"
S14 = HERE.parent / "s14_mixed_streaming_scheduler"
sys.path[:0] = [str(HERE), str(PERSISTENT), str(S15), str(S14)]

from live_contract import (  # noqa: E402
    CHILD_RESULT_SCHEMA,
    EXECUTE_SCHEMA,
    FrozenRoute,
    HOST_PLACEMENT_PREFIX,
    LiveContractError,
    MAX_PROMPT_BYTES,
    RESULT_SCHEMA,
    canonical,
    child_command,
    digest,
    integer,
    parse_host_placement,
    sha256_text,
    strict_line,
    text,
)
from session_adapter import parse_session_cert  # noqa: E402


CONFIG_SCHEMA = "s15-persistent-bridge-config-v1"
READY_PREFIX = b"PERSISTENT_DRIVER_READY "
CHILD_END_PREFIX = b"PERSISTENT_DRIVER_EXCHANGE_END "
MAX_CHILD_LINE = 4 * 1024 * 1024
MAX_CPP_COMMAND_BYTES = 64 * 1024


class BridgeError(LiveContractError):
    pass


def route_from(value: dict) -> FrozenRoute:
    required = {
        "route_id", "profile_id", "evidence_sha256", "device_id",
        "device_boot_id", "worker_binary_sha256", "layer_range",
        "host_tail_range",
        "route_epoch", "residency_epoch", "device_boot_epoch",
        "registry_generation", "batch_size", "max_n_gen",
    }
    if type(value) is not dict or set(value) != required:
        raise BridgeError("bridge route has missing or unknown fields")
    layer_range = value["layer_range"]
    host_tail_range = value["host_tail_range"]
    if type(layer_range) is not list or len(layer_range) != 2 \
            or type(host_tail_range) is not list or len(host_tail_range) != 2:
        raise BridgeError("bridge layer ranges must be two-item lists")
    route = FrozenRoute(
        value["route_id"], value["profile_id"], value["evidence_sha256"],
        value["device_id"], value["device_boot_id"], value["worker_binary_sha256"],
        tuple(layer_range), tuple(host_tail_range), value["route_epoch"], value["residency_epoch"],
        value["device_boot_epoch"], value["registry_generation"],
        value["batch_size"], value["max_n_gen"],
    )
    route.validate()
    return route


def load_config(path: Path) -> tuple[FrozenRoute, tuple[str, ...], Path, int]:
    value = strict_line(path.read_bytes(), "bridge config")
    if set(value) != {
        "schema", "route", "child_command", "artifact_root", "child_ready_timeout_s",
    } \
            or value["schema"] != CONFIG_SCHEMA:
        raise BridgeError("bridge config has missing, unknown, or invalid fields")
    command = value["child_command"]
    if type(command) is not list or not command \
            or any(type(item) is not str or not item for item in command):
        raise BridgeError("child_command must be a non-empty string list")
    artifact_root = Path(text("artifact_root", value["artifact_root"]))
    if not artifact_root.is_absolute() or artifact_root.exists():
        raise BridgeError("artifact_root must be an absent absolute path")
    ready_timeout_s = integer(
        "child_ready_timeout_s", value["child_ready_timeout_s"], 1, 600,
    )
    return route_from(value["route"]), tuple(command), artifact_root, ready_timeout_s


class ChildMonitor:
    def __init__(self, command: tuple[str, ...]) -> None:
        self.cv = threading.Condition()
        self.stdout_lines: deque[bytes] = deque()
        self.stderr_raw = bytearray()
        self.stderr_partial = bytearray()
        self.ready = False
        self.ready_metadata: dict | None = None
        self.active = False
        self.cert_line: bytes | None = None
        self.host_placement_line: bytes | None = None
        self.cert_end: int | None = None
        self.exchange_end: int | None = None
        self.active_launch_id: int | None = None
        self.session_stderr_start = 0
        self.poison: str | None = None
        self.stdout_eof = False
        self.stderr_eof = False
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0,
        )
        self.threads = (
            threading.Thread(target=self._stdout, daemon=True),
            threading.Thread(target=self._stderr, daemon=True),
        )
        for thread in self.threads:
            thread.start()

    def _fail_locked(self, reason: str) -> None:
        if self.poison is None:
            self.poison = reason
        self.cv.notify_all()

    def _stdout(self) -> None:
        assert self.process.stdout is not None
        partial = bytearray()
        while True:
            chunk = self.process.stdout.read(4096)
            with self.cv:
                if not chunk:
                    self.stdout_eof = True
                    if partial:
                        self._fail_locked("child stdout ended with a partial line")
                    elif self.active and not self.stdout_lines:
                        self._fail_locked("child exited without a result")
                    self.cv.notify_all()
                    return
                partial.extend(chunk)
                while b"\n" in partial:
                    end = partial.index(b"\n") + 1
                    line = bytes(partial[:end])
                    del partial[:end]
                    if len(line) > MAX_CHILD_LINE:
                        self._fail_locked("child result exceeded its bound")
                        return
                    if not self.active:
                        self._fail_locked("unsolicited child stdout")
                        return
                    self.stdout_lines.append(line)
                    self.cv.notify_all()
                if len(partial) > MAX_CHILD_LINE:
                    self._fail_locked("partial child result exceeded its bound")
                    return

    def _stderr(self) -> None:
        assert self.process.stderr is not None
        line_offset = 0
        while True:
            chunk = self.process.stderr.read(4096)
            with self.cv:
                if not chunk:
                    self.stderr_eof = True
                    if self.active and self.exchange_end is None:
                        self._fail_locked(
                            "child stderr ended without SESSIONCERT or exchange-end marker"
                        )
                    self.cv.notify_all()
                    return
                self.stderr_raw.extend(chunk)
                self.stderr_partial.extend(chunk)
                while b"\n" in self.stderr_partial:
                    end = self.stderr_partial.index(b"\n") + 1
                    line = bytes(self.stderr_partial[:end])
                    del self.stderr_partial[:end]
                    start = line_offset
                    line_offset += len(line)
                    if line.startswith(READY_PREFIX) and not self.ready and not self.active:
                        try:
                            ready = strict_line(
                                line[len(READY_PREFIX):], "child readiness record",
                            )
                        except BridgeError as exc:
                            self._fail_locked(str(exc))
                            return
                        required = {"schema", "host_pid", "batch_size", "max_n_gen"}
                        if set(ready) != required \
                                or ready.get("schema") != "layersplit-persistent-driver-v1":
                            self._fail_locked("child readiness identity failed")
                            return
                        try:
                            integer("ready host_pid", ready["host_pid"], 1)
                            integer("ready batch_size", ready["batch_size"], 1)
                            integer("ready max_n_gen", ready["max_n_gen"], 1)
                        except LiveContractError as exc:
                            self._fail_locked(str(exc))
                            return
                        self.ready_metadata = ready
                        self.ready = True
                        self.cv.notify_all()
                    elif line.startswith(b"SESSIONCERT "):
                        if not self.active or self.cert_line is not None:
                            self._fail_locked("unsolicited or duplicate SESSIONCERT")
                            return
                        self.cert_line = line
                        self.cert_end = line_offset
                        self.cv.notify_all()
                    elif line.startswith(HOST_PLACEMENT_PREFIX):
                        if not self.active or self.host_placement_line is not None:
                            self._fail_locked("unsolicited or duplicate host PLACEMENTCERT")
                            return
                        self.host_placement_line = line
                        self.cv.notify_all()
                    elif line.startswith(CHILD_END_PREFIX):
                        try:
                            marker = strict_line(
                                line[len(CHILD_END_PREFIX):], "child exchange-end marker",
                            )
                        except LiveContractError as exc:
                            self._fail_locked(str(exc))
                            return
                        if set(marker) != {"launch_id"} \
                                or marker["launch_id"] != self.active_launch_id \
                                or self.cert_line is None \
                                or self.host_placement_line is None:
                            self._fail_locked("child exchange-end marker identity failed")
                            return
                        if self.exchange_end is not None:
                            self._fail_locked("duplicate child exchange-end marker")
                            return
                        self.exchange_end = start
                        self.cv.notify_all()
                    elif self.ready and not self.active:
                        self._fail_locked("child stderr outside a session")
                        return
                    elif self.active and self.exchange_end is not None:
                        self._fail_locked("child stderr followed its exchange-end marker")
                        return

    def wait_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        with self.cv:
            while not self.ready and self.poison is None and time.monotonic() < deadline:
                self.cv.wait(0.05)
            if not self.ready:
                raise BridgeError(self.poison or "child readiness timed out")

    def validate_ready(self, route: FrozenRoute) -> None:
        with self.cv:
            if self.ready_metadata is None \
                    or self.ready_metadata["batch_size"] != route.batch_size \
                    or self.ready_metadata["max_n_gen"] != route.max_n_gen:
                raise BridgeError("child readiness differs from the frozen route")

    def begin(self, launch_id: int) -> int:
        with self.cv:
            if self.poison is not None or not self.ready or self.active \
                    or self.stdout_lines or self.process.poll() is not None:
                raise BridgeError(self.poison or "child is not reusable")
            self.active = True
            self.active_launch_id = launch_id
            self.cert_line = None
            self.host_placement_line = None
            self.cert_end = None
            self.exchange_end = None
            self.session_stderr_start = len(self.stderr_raw)
            return self.session_stderr_start

    def exchange(self, payload: bytes, timeout_us: int,
                 launch_id: int) -> tuple[bytes, bytes, bytes, bytes, int]:
        self.begin(launch_id)
        start_ns = time.monotonic_ns()
        try:
            assert self.process.stdin is not None
            self.process.stdin.write(payload)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            with self.cv:
                self._fail_locked(f"child command write failed: {exc}")
        deadline = time.monotonic() + timeout_us / 1_000_000
        with self.cv:
            while self.poison is None \
                    and (not self.stdout_lines or self.cert_line is None
                         or self.host_placement_line is None
                         or self.exchange_end is None) \
                    and time.monotonic() < deadline:
                self.cv.wait(0.05)
            elapsed_us = max(0, (time.monotonic_ns() - start_ns) // 1000)
            if self.poison is not None:
                raise BridgeError(self.poison)
            if not self.stdout_lines or self.cert_line is None \
                    or self.host_placement_line is None or self.exchange_end is None:
                self._fail_locked(
                    "child result, SESSIONCERT, PLACEMENTCERT, or exchange-end marker timed out"
                )
                raise BridgeError(self.poison)
            result = self.stdout_lines.popleft()
            if self.stdout_lines:
                self._fail_locked("duplicate child result")
                raise BridgeError(self.poison)
            assert self.exchange_end is not None
            stderr = bytes(self.stderr_raw[self.session_stderr_start:self.exchange_end])
            cert = self.cert_line
            host_placement = self.host_placement_line
            self.active = False
            self.active_launch_id = None
            self.cv.notify_all()
            return result, stderr, cert, host_placement, elapsed_us

    def require_running(self) -> None:
        with self.cv:
            if self.poison is not None or self.process.poll() is not None:
                raise BridgeError(self.poison or "child terminated after DETACH")

    def require_stopped(self, timeout_s: float) -> None:
        try:
            returncode = self.process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise BridgeError("child did not terminate after STOP") from exc
        if returncode != 0:
            raise BridgeError("child exited nonzero after STOP")

    def terminate(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        for thread in self.threads:
            thread.join(timeout=2)


def validate_execute(value: dict, route: FrozenRoute) -> None:
    expected_keys = {
        "schema", "command", "protocol_version", "launch_id", "prompt", "n_gen",
        "batch_size", "request_ids", "session_end", "deadline_us",
        "expected_route_id", "expected_profile_id", "expected_evidence_sha256",
        "expected_device_id", "expected_device_boot_id",
        "expected_worker_binary_sha256", "expected_layer_range", "route_epoch",
        "expected_host_tail_range",
        "residency_epoch", "lease_epoch", "device_boot_epoch",
        "registry_generation", "cohort_sha256", "input_manifest_sha256",
    }
    if set(value) != expected_keys:
        raise BridgeError("EXECUTE has missing or unknown fields")
    expected = {
        "schema": EXECUTE_SCHEMA,
        "command": "EXECUTE",
        "protocol_version": 1,
        "batch_size": route.batch_size,
        "expected_route_id": route.route_id,
        "expected_profile_id": route.profile_id,
        "expected_evidence_sha256": route.evidence_sha256,
        "expected_device_id": route.device_id,
        "expected_device_boot_id": route.device_boot_id,
        "expected_worker_binary_sha256": route.worker_binary_sha256,
        "expected_layer_range": list(route.layer_range),
        "expected_host_tail_range": list(route.host_tail_range),
        "route_epoch": route.route_epoch,
        "residency_epoch": route.residency_epoch,
        "device_boot_epoch": route.device_boot_epoch,
        "registry_generation": route.registry_generation,
    }
    for key, wanted in expected.items():
        if type(value.get(key)) is not type(wanted) or value.get(key) != wanted:
            raise BridgeError(f"EXECUTE identity mismatch: {key}")
    integer("launch_id", value["launch_id"], 1)
    text("prompt", value["prompt"], MAX_PROMPT_BYTES)
    integer("n_gen", value["n_gen"], 1, route.max_n_gen)
    integer("deadline_us", value["deadline_us"], 1)
    integer("lease_epoch", value["lease_epoch"], 1)
    sha256_text("cohort_sha256", value["cohort_sha256"])
    sha256_text("input_manifest_sha256", value["input_manifest_sha256"])
    if value["session_end"] not in ("DETACH", "STOP"):
        raise BridgeError("EXECUTE session_end is invalid")
    request_ids = value["request_ids"]
    if type(request_ids) is not list or len(request_ids) != route.batch_size \
            or len(set(request_ids)) != len(request_ids):
        raise BridgeError("EXECUTE request_ids do not match its batch")
    for request_id in request_ids:
        text("request_id", request_id)


def validate_child_result(value: dict, execute: dict, ready: dict) -> None:
    required = {
        "schema", "launch_id", "outcome", "host_pid", "request_count",
        "batch_size", "n_gen", "session_end", "elapsed_us", "route_wall_us",
        "token_ids",
    }
    if set(value) != required:
        raise BridgeError("child RESULT has missing or unknown fields")
    expected = {
        "schema": CHILD_RESULT_SCHEMA,
        "launch_id": execute["launch_id"],
        "outcome": "completed",
        "request_count": execute["batch_size"],
        "batch_size": execute["batch_size"],
        "n_gen": execute["n_gen"],
        "session_end": execute["session_end"],
        "host_pid": ready["host_pid"],
    }
    for key, wanted in expected.items():
        if type(value.get(key)) is not type(wanted) or value.get(key) != wanted:
            raise BridgeError(f"child RESULT identity mismatch: {key}")
    integer("host_pid", value["host_pid"], 1)
    integer("elapsed_us", value["elapsed_us"], 1)
    integer("route_wall_us", value["route_wall_us"], 1)
    if value["route_wall_us"] > value["elapsed_us"]:
        raise BridgeError("child route wall time exceeds its elapsed time")
    rows = value["token_ids"]
    if type(rows) is not list or len(rows) != execute["batch_size"]:
        raise BridgeError("child RESULT token row count is wrong")
    for row in rows:
        if type(row) is not list or len(row) != execute["n_gen"]:
            raise BridgeError("child RESULT token count is wrong")
        for token in row:
            integer("token id", token)


def identity_from(execute: dict) -> dict:
    keys = (
        "expected_route_id", "expected_profile_id", "expected_evidence_sha256",
        "expected_device_id", "expected_device_boot_id",
        "expected_worker_binary_sha256", "expected_layer_range", "route_epoch",
        "expected_host_tail_range",
        "residency_epoch", "lease_epoch", "device_boot_epoch",
        "registry_generation", "cohort_sha256", "input_manifest_sha256",
    )
    return {key: execute[key] for key in keys}


def bridge_once(monitor: ChildMonitor, execute: dict, route: FrozenRoute,
                artifact_root: Path) -> dict:
    validate_execute(execute, route)
    command = canonical(child_command(execute))
    if len(command) - 1 > MAX_CPP_COMMAND_BYTES:
        raise BridgeError("encoded child command exceeds the C++ line bound")
    raw_result, raw_stderr, cert_line, host_placement_line, bridge_elapsed_us = monitor.exchange(
        command, max(1, execute["deadline_us"]), execute["launch_id"],
    )
    child = strict_line(raw_result, "child RESULT")
    if monitor.ready_metadata is None:
        raise BridgeError("child readiness metadata is unavailable")
    validate_child_result(child, execute, monitor.ready_metadata)
    parse_host_placement(
        host_placement_line, route.host_tail_range, monitor.ready_metadata["host_pid"],
    )
    cert = parse_session_cert(cert_line)
    if cert.get("session_end") != execute["session_end"]:
        raise BridgeError("SESSIONCERT end does not match EXECUTE")
    if execute["session_end"] == "DETACH":
        monitor.require_running()
        terminal = "DETACHED"
        detach_ack = 0
        worker_terminated = False
    else:
        monitor.require_stopped(5)
        terminal = "STOPPED"
        detach_ack = None
        worker_terminated = True

    launch_dir = artifact_root / f"launch-{execute['launch_id']:06d}"
    launch_dir.mkdir()
    artifacts = {
        "child_command": command,
        "child_result": raw_result,
        "child_stdout": raw_result,
        "child_stderr": raw_stderr,
        "session_cert": cert_line,
        "host_placement": host_placement_line,
        "tokens": canonical({"token_ids": child["token_ids"]}),
        "placement": canonical({
            "compute_by_op_and_buffer": cert.get("compute_by_op_and_buffer"),
            "missing_buffer_compute_nodes": cert.get("missing_buffer_compute_nodes"),
            "placement_status": cert.get("placement_status"),
        }),
    }
    bindings = {}
    for name, payload in artifacts.items():
        relative = f"launch-{execute['launch_id']:06d}/{name}.bin"
        (artifact_root / relative).write_bytes(payload)
        bindings[name] = {"path": relative, "sha256": digest(payload)}
    token_sha256 = digest(artifacts["tokens"])
    return {
        "schema": RESULT_SCHEMA,
        "protocol_version": 1,
        "launch_id": execute["launch_id"],
        "terminal": terminal,
        "request_count": child["request_count"],
        "request_ids": execute["request_ids"],
        "session_end": execute["session_end"],
        "elapsed_us": child["elapsed_us"],
        "route_wall_us": child["route_wall_us"],
        "bridge_elapsed_us": bridge_elapsed_us,
        "child_host_pid": child["host_pid"],
        "token_sha256": token_sha256,
        "detach_ack": detach_ack,
        "worker_terminated": worker_terminated,
        "identity": identity_from(execute),
        "artifact_hashes": bindings,
        "error": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    route, child_command_line, artifact_root, child_ready_timeout_s = load_config(args.config)
    artifact_root.mkdir(parents=True)
    monitor = ChildMonitor(child_command_line)
    try:
        monitor.wait_ready(child_ready_timeout_s)
        monitor.validate_ready(route)
        ready = canonical({
            "contract": EXECUTE_SCHEMA,
            "route_id": route.route_id,
            "worker_binary_sha256": route.worker_binary_sha256,
        })
        sys.stderr.buffer.write(b"LAUNCHER_READY " + ready)
        sys.stderr.buffer.flush()
        for line in sys.stdin.buffer:
            execute = strict_line(line, "outer EXECUTE")
            result = bridge_once(monitor, execute, route, artifact_root)
            marker = canonical({"launch_id": execute["launch_id"]})
            sys.stderr.buffer.write(b"LAUNCHER_EXCHANGE_END " + marker)
            sys.stderr.buffer.flush()
            sys.stdout.buffer.write(canonical(result))
            sys.stdout.buffer.flush()
            if execute["session_end"] == "STOP":
                break
        return 0
    finally:
        monitor.terminate()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BridgeError, LiveContractError) as exc:
        print(f"BRIDGE_ERROR {exc}", file=sys.stderr)
        raise SystemExit(2)
