#!/usr/bin/env python3
"""Multiplex one persistent host tail and one phone worker into one child stream."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


CONFIG_SCHEMA = "s15-physical-mux-config-v1"
HOST_READY_PREFIX = b"PERSISTENT_DRIVER_READY "
HOST_END_PREFIX = b"PERSISTENT_DRIVER_EXCHANGE_END "
PHONE_READY_PREFIX = b"[stagenet] listening"
SESSION_PREFIX = b"SESSIONCERT "
MAX_LINE = 4 * 1024 * 1024
MAX_STREAM = 64 * 1024 * 1024


class MuxError(RuntimeError):
    pass


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def parse_object(payload: bytes, label: str) -> dict:
    if type(payload) is not bytes or not payload or len(payload) > MAX_LINE \
            or not payload.endswith(b"\n") or payload.count(b"\n") != 1:
        raise MuxError(f"{label} is not one bounded JSON line")

    def no_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise MuxError(f"duplicate key in {label}: {key}")
            value[key] = item
        return value

    try:
        text = payload.decode("ascii")
        value = json.loads(text, object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MuxError(f"invalid {label}: {exc}") from exc
    if type(value) is not dict:
        raise MuxError(f"{label} is not an object")
    return value


def strict_line(payload: bytes, label: str) -> dict:
    value = parse_object(payload, label)
    if canonical(value) != payload:
        raise MuxError(f"{label} is not canonical")
    return value


def command(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not list or not value \
            or any(type(item) is not str or not item for item in value):
        raise MuxError(f"{label} must be a non-empty string list")
    return tuple(value)


def load_config(path: Path) -> tuple[
        tuple[str, ...], tuple[str, ...], dict[str, str], Path, int, int, str]:
    value = strict_line(path.read_bytes(), "mux config")
    if set(value) != {"schema", "host_command", "phone_command", "host_env", "artifact_root",
                     "host_layer_start", "host_layer_end", "host_backend"} \
            or value["schema"] != CONFIG_SCHEMA:
        raise MuxError("mux config has missing, unknown, or invalid fields")
    environment = value["host_env"]
    if type(environment) is not dict or not environment \
            or any(type(key) is not str or not key or type(item) is not str
                   for key, item in environment.items()):
        raise MuxError("host_env must be a non-empty string map")
    if type(value["artifact_root"]) is not str or not value["artifact_root"]:
        raise MuxError("artifact_root must be a string")
    artifact_root = Path(value["artifact_root"])
    if not artifact_root.is_absolute() or artifact_root.exists():
        raise MuxError("artifact_root must be an absent absolute path")
    layer_start = value["host_layer_start"]
    layer_end = value["host_layer_end"]
    backend = value["host_backend"]
    if type(layer_start) is not int or type(layer_end) is not int \
            or layer_start < 0 or layer_end <= layer_start:
        raise MuxError("host layer range is invalid")
    if type(backend) is not str or not backend:
        raise MuxError("host_backend must be a non-empty string")
    return (
        command(value["host_command"], "host_command"),
        command(value["phone_command"], "phone_command"),
        dict(environment), artifact_root, layer_start, layer_end, backend,
    )


class MonitoredProcess:
    def __init__(self, argv: tuple[str, ...], env: dict[str, str] | None) -> None:
        self.cv = threading.Condition()
        self.stdout_lines: list[bytes] = []
        self.stderr_lines: list[bytes] = []
        self.stdout_raw = bytearray()
        self.stderr_raw = bytearray()
        self.error: str | None = None
        self.process = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env, bufsize=0,
        )
        self.threads = (
            threading.Thread(target=self._read, args=("stdout",), daemon=True),
            threading.Thread(target=self._read, args=("stderr",), daemon=True),
        )
        for thread in self.threads:
            thread.start()

    def _read(self, stream_name: str) -> None:
        source = getattr(self.process, stream_name)
        raw = getattr(self, f"{stream_name}_raw")
        lines = getattr(self, f"{stream_name}_lines")
        partial = bytearray()
        assert source is not None
        while True:
            try:
                chunk = source.read(4096)
            except OSError:
                chunk = b""
            with self.cv:
                if not chunk:
                    if partial and self.error is None:
                        self.error = f"{stream_name} ended with a partial line"
                    self.cv.notify_all()
                    return
                raw.extend(chunk)
                partial.extend(chunk)
                if len(raw) > MAX_STREAM and self.error is None:
                    self.error = f"{stream_name} exceeded its byte bound"
                    self.cv.notify_all()
                    return
                while b"\n" in partial:
                    end = partial.index(b"\n") + 1
                    line = bytes(partial[:end])
                    del partial[:end]
                    if len(line) > MAX_LINE and self.error is None:
                        self.error = f"{stream_name} line exceeded its byte bound"
                        self.cv.notify_all()
                        return
                    lines.append(line)
                    self.cv.notify_all()

    def snapshot(self) -> tuple[int, int]:
        with self.cv:
            return len(self.stdout_lines), len(self.stderr_lines)

    def wait_line(self, stream_name: str, prefix: bytes, timeout_s: float) -> bytes:
        deadline = time.monotonic() + timeout_s
        lines = getattr(self, f"{stream_name}_lines")
        with self.cv:
            while time.monotonic() < deadline:
                if self.error is not None:
                    raise MuxError(self.error)
                matches = [line for line in lines if line.startswith(prefix)]
                if matches:
                    return matches[-1]
                if self.process.poll() is not None:
                    raise MuxError(f"process exited before {prefix!r}")
                self.cv.wait(0.05)
        raise MuxError(f"timed out waiting for {prefix!r}")

    def lines_since(self, stream_name: str, start: int) -> list[bytes]:
        with self.cv:
            return list(getattr(self, f"{stream_name}_lines")[start:])

    def send(self, payload: bytes) -> None:
        if self.process.stdin is None or self.process.poll() is not None:
            raise MuxError("process input is unavailable")
        try:
            self.process.stdin.write(payload)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise MuxError(f"process input failed: {exc}") from exc

    def wait(self, timeout_s: float) -> int:
        try:
            returncode = self.process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise MuxError("process did not terminate") from exc
        for thread in self.threads:
            thread.join(timeout=2)
        if any(thread.is_alive() for thread in self.threads):
            raise MuxError("process reader did not terminate")
        return returncode

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

    def persist(self, root: Path, name: str) -> None:
        (root / f"{name}.stdout.bin").write_bytes(bytes(self.stdout_raw))
        (root / f"{name}.stderr.bin").write_bytes(bytes(self.stderr_raw))


def matching(lines: list[bytes], prefix: bytes) -> list[bytes]:
    return [line for line in lines if line.startswith(prefix)]


def validate_host_placement(payload: bytes, host_pid: int, layer_start: int,
                            layer_end: int, backend: str) -> dict:
    value = parse_object(payload, "host placement certificate")
    required = {
        "schema", "role", "mode", "layer_start", "layer_end", "n_layer", "pid",
        "run_rc", "compute_nodes", "missing_buffer_compute_nodes",
        "compute_by_buffer_type", "compute_by_op_and_buffer", "status",
    }
    if not required.issubset(value):
        raise MuxError("host placement certificate is incomplete")
    if value["schema"] != "layersplit-scheduled-placement-v2" \
            or value["role"] != "host_tail" or value["mode"] != "pipedriver" \
            or value["layer_start"] != layer_start or value["layer_end"] != layer_end \
            or type(value["n_layer"]) is not int or value["n_layer"] != layer_end \
            or value["pid"] != host_pid or value["run_rc"] != 0 \
            or value["status"] != "SCHEDULED_PLACEMENT_OK":
        raise MuxError("host placement identity or status failed")
    if type(value["compute_nodes"]) is not int or value["compute_nodes"] <= 0 \
            or value["missing_buffer_compute_nodes"] != 0:
        raise MuxError("host placement compute coverage failed")
    by_buffer = value["compute_by_buffer_type"]
    if type(by_buffer) is not dict or set(by_buffer) != {backend} \
            or type(by_buffer[backend]) is not int or by_buffer[backend] != value["compute_nodes"]:
        raise MuxError("host placement used an unexpected backend")
    by_op = value["compute_by_op_and_buffer"]
    if type(by_op) is not dict or not by_op:
        raise MuxError("host placement has no operation tally")
    for op_name, buffers in by_op.items():
        if type(op_name) is not str or not op_name or type(buffers) is not dict \
                or set(buffers) != {backend} or type(buffers[backend]) is not int \
                or buffers[backend] <= 0:
            raise MuxError("host operation placement used an unexpected backend")
    return value


def wait_exchange(host: MonitoredProcess, phone: MonitoredProcess,
                  host_start: tuple[int, int], phone_start: tuple[int, int],
                  launch_id: int, timeout_us: int, host_pid: int,
                  host_layer_start: int, host_layer_end: int,
                  host_backend: str) -> tuple[bytes, list[bytes], list[bytes], bytes]:
    deadline = time.monotonic() + timeout_us / 1_000_000
    while time.monotonic() < deadline:
        host_stdout = host.lines_since("stdout", host_start[0])
        host_stderr = host.lines_since("stderr", host_start[1])
        phone_stderr = phone.lines_since("stderr", phone_start[1])
        results = [line for line in host_stdout if line.strip()]
        markers = matching(host_stderr, HOST_END_PREFIX)
        placements = matching(host_stderr, b"PLACEMENTCERT ")
        certs = matching(phone_stderr, SESSION_PREFIX)
        if len(results) == 1 and len(markers) == 1 and len(placements) == 1 and len(certs) == 1:
            result = strict_line(results[0], "host result")
            marker = strict_line(markers[0][len(HOST_END_PREFIX):], "host marker")
            cert = parse_object(certs[0][len(SESSION_PREFIX):], "phone SESSIONCERT")
            if result.get("launch_id") != launch_id or marker != {"launch_id": launch_id} \
                    or cert.get("session_id") != launch_id:
                raise MuxError("host/phone exchange identity mismatch")
            validate_host_placement(
                placements[0][len(b"PLACEMENTCERT "):], host_pid,
                host_layer_start, host_layer_end, host_backend,
            )
            return results[0], host_stderr, phone_stderr, markers[0]
        if len(results) > 1 or len(markers) > 1 or len(placements) > 1 or len(certs) > 1:
            raise MuxError("duplicate host result, marker, placement, or phone certificate")
        if host.error is not None or phone.error is not None:
            raise MuxError(host.error or phone.error)
        if host.process.poll() is not None and not results:
            raise MuxError("host exited without an exchange result")
        time.sleep(0.01)
    raise MuxError("physical exchange timed out")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    host_argv, phone_argv, host_overrides, artifact_root, host_layer_start, \
        host_layer_end, host_backend = load_config(args.config)
    artifact_root.mkdir(parents=True)
    host_env = dict(os.environ)
    host_env.update(host_overrides)
    host = None
    phone = None
    final_stop = False
    try:
        phone = MonitoredProcess(phone_argv, None)
        phone.wait_line("stderr", PHONE_READY_PREFIX, 240)
        host = MonitoredProcess(host_argv, host_env)
        ready = host.wait_line("stderr", HOST_READY_PREFIX, 240)
        ready_record = strict_line(ready[len(HOST_READY_PREFIX):], "host readiness")
        if set(ready_record) != {"schema", "host_pid", "batch_size", "max_n_gen"} \
                or ready_record.get("schema") != "layersplit-persistent-driver-v1" \
                or type(ready_record.get("host_pid")) is not int:
            raise MuxError("host readiness identity failed")
        sys.stderr.buffer.write(ready)
        sys.stderr.buffer.flush()

        previous_launch_id = 0
        for payload in sys.stdin.buffer:
            command_value = strict_line(payload, "mux command")
            launch_id = command_value.get("launch_id")
            if type(launch_id) is not int or launch_id != previous_launch_id + 1:
                raise MuxError("mux launch_id is not contiguous")
            if command_value.get("session_end") not in ("DETACH", "STOP"):
                raise MuxError("mux session_end is invalid")
            timeout_us = 10_000_000
            host_start = host.snapshot()
            phone_start = phone.snapshot()
            host.send(payload)
            result, host_stderr, phone_stderr, marker = wait_exchange(
                host, phone, host_start, phone_start, launch_id, timeout_us,
                ready_record["host_pid"], host_layer_start, host_layer_end, host_backend,
            )
            marker_index = host_stderr.index(marker)
            certs = matching(phone_stderr, SESSION_PREFIX)
            for line in host_stderr[:marker_index]:
                sys.stderr.buffer.write(line)
            for line in phone_stderr:
                if line == certs[0] or not line.startswith(b"PLACEMENTCERT "):
                    sys.stderr.buffer.write(line)
            if command_value["session_end"] == "STOP":
                if host.wait(10) != 0 or phone.wait(10) != 0:
                    raise MuxError("host or phone failed to stop cleanly")
                final_stop = True
            else:
                if host.process.poll() is not None or phone.process.poll() is not None:
                    raise MuxError("DETACH did not preserve both resident processes")
            sys.stderr.buffer.write(marker)
            sys.stderr.buffer.flush()
            sys.stdout.buffer.write(result)
            sys.stdout.buffer.flush()
            previous_launch_id = launch_id
            if final_stop:
                break
        if not final_stop:
            raise MuxError("mux input ended before STOP")
        return 0
    finally:
        if host is not None:
            host.terminate()
            host.persist(artifact_root, "host")
        if phone is not None:
            phone.terminate()
            phone.persist(artifact_root, "phone")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, MuxError) as exc:
        print(f"MUX_ERROR {exc}", file=sys.stderr)
        raise SystemExit(2)
