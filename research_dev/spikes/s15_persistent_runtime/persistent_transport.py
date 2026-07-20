#!/usr/bin/env python3
"""Persistent, framed subprocess transport for the S15 executor boundary."""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

from executor_contract import ExecutionRequest
from physical_executor import (
    MAX_REPLY_BYTES,
    PhysicalExecutorError,
    SessionTransport,
    TransportReply,
)


MAX_REQUEST_BYTES = 4 * 1024 * 1024
MAX_STREAM_BYTES = 64 * 1024 * 1024
EXCHANGE_END_PREFIX = b"LAUNCHER_EXCHANGE_END "
STATES = (
    "STARTING", "READY", "ACTIVE", "DRAINING", "STOPPED", "POISONED", "FINALIZED",
)


class PersistentTransportError(PhysicalExecutorError):
    pass


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _strict_object(payload: bytes, label: str, maximum: int) -> dict:
    if type(payload) is not bytes or not payload or len(payload) > maximum \
            or not payload.endswith(b"\n") or payload.count(b"\n") != 1:
        raise PersistentTransportError(f"{label} is not one bounded JSON line")

    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise PersistentTransportError(f"duplicate JSON key {key!r} in {label}")
            result[key] = value
        return result

    def no_constants(value):
        raise PersistentTransportError(f"invalid JSON constant {value!r} in {label}")

    try:
        value = json.loads(
            payload, object_pairs_hook=no_duplicates, parse_constant=no_constants,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PersistentTransportError(f"invalid {label}: {exc}") from exc
    if type(value) is not dict:
        raise PersistentTransportError(f"{label} must be a JSON object")
    canonical = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("ascii")
    if payload != canonical:
        raise PersistentTransportError(f"{label} is not canonical JSON")
    return value


class PersistentPreparedTransport(SessionTransport):
    """Keep one prepared launcher alive across strict JSON-line exchanges.

    Stdout is a framed protocol channel: exactly one canonical JSON line is
    accepted for each canonical request line. Stderr is diagnostic and carries
    the out-of-band readiness record. Any unsolicited, duplicate, partial, or
    oversized stdout poisons the transport and prevents further exchanges.
    """

    def __init__(self, command: tuple[str, ...], artifact_dir: Path,
                 ready_timeout_s: float = 600.0, first_launch_id: int = 1) -> None:
        if type(command) is not tuple or not command \
                or any(type(value) is not str or not value for value in command):
            raise PersistentTransportError("command must be a non-empty string tuple")
        if not isinstance(artifact_dir, Path) or artifact_dir.exists():
            raise PersistentTransportError("artifact directory must not exist")
        if type(ready_timeout_s) not in (int, float) or type(ready_timeout_s) is bool \
                or ready_timeout_s <= 0:
            raise PersistentTransportError("ready_timeout_s must be positive")
        if type(first_launch_id) is not int or first_launch_id < 1:
            raise PersistentTransportError("first_launch_id must be a positive integer")
        artifact_dir.mkdir(parents=True)
        self._artifact_dir = artifact_dir
        self._cv = threading.Condition()
        self._state = "STARTING"
        self._poison_reason: str | None = None
        self._ready_metadata: dict | None = None
        self._stdout_raw = bytearray()
        self._stderr_raw = bytearray()
        self._stdout_partial = bytearray()
        self._stderr_partial = bytearray()
        self._frames: deque[bytes] = deque()
        self._stdout_eof = False
        self._stderr_eof = False
        self._stderr_line_offset = 0
        self._exchange_stderr_end: int | None = None
        self._exchange_number = 0
        self._last_launch_id = first_launch_id - 1
        self._active_launch_id: int | None = None
        self._terminated_by_host = False
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._threads = (
            threading.Thread(target=self._stdout_loop, name="s15-persistent-stdout", daemon=True),
            threading.Thread(target=self._stderr_loop, name="s15-persistent-stderr", daemon=True),
        )
        for thread in self._threads:
            thread.start()

        deadline = time.monotonic() + float(ready_timeout_s)
        with self._cv:
            while self._state == "STARTING" and time.monotonic() < deadline:
                self._cv.wait(min(0.1, max(0.0, deadline - time.monotonic())))
            if self._state == "READY":
                return
            reason = self._poison_reason or "launcher readiness timed out"
        self.terminate()
        raise PersistentTransportError(reason)

    @property
    def state(self) -> str:
        with self._cv:
            return self._state

    @property
    def poison_reason(self) -> str | None:
        with self._cv:
            return self._poison_reason

    @property
    def ready_metadata(self) -> dict:
        with self._cv:
            if type(self._ready_metadata) is not dict:
                raise PersistentTransportError("launcher readiness metadata is unavailable")
            return dict(self._ready_metadata)

    def _poison_locked(self, reason: str) -> None:
        if self._poison_reason is None:
            self._poison_reason = reason
        if self._state not in ("STOPPED", "FINALIZED"):
            self._state = "POISONED"
        self._cv.notify_all()

    def _stdout_loop(self) -> None:
        assert self._process.stdout is not None
        while True:
            try:
                chunk = self._process.stdout.read(4096)
            except OSError:
                chunk = b""
            with self._cv:
                if not chunk:
                    self._stdout_eof = True
                    if self._stdout_partial:
                        self._poison_locked("stdout ended with a partial reply")
                    elif self._state == "ACTIVE" and not self._frames:
                        self._poison_locked("launcher exited without a reply")
                    elif self._state in ("STARTING", "READY"):
                        self._state = "STOPPED"
                    self._cv.notify_all()
                    return
                self._stdout_raw.extend(chunk)
                self._stdout_partial.extend(chunk)
                if len(self._stdout_raw) > MAX_STREAM_BYTES:
                    self._poison_locked("stdout stream exceeded its bound")
                    return
                while b"\n" in self._stdout_partial:
                    end = self._stdout_partial.index(b"\n") + 1
                    frame = bytes(self._stdout_partial[:end])
                    del self._stdout_partial[:end]
                    if len(frame) > MAX_REPLY_BYTES:
                        self._poison_locked("reply frame exceeded its bound")
                        return
                    if self._state != "ACTIVE":
                        self._poison_locked("unsolicited reply outside an active exchange")
                        return
                    self._frames.append(frame)
                    if len(self._frames) != 1 or self._stdout_partial:
                        self._poison_locked("multiple or cross-talk replies for one request")
                        return
                    self._cv.notify_all()
                if len(self._stdout_partial) > MAX_REPLY_BYTES:
                    self._poison_locked("partial reply exceeded its bound")
                    return

    def _stderr_loop(self) -> None:
        assert self._process.stderr is not None
        prefix = b"LAUNCHER_READY "
        while True:
            try:
                chunk = self._process.stderr.read(4096)
            except OSError:
                chunk = b""
            with self._cv:
                if not chunk:
                    self._stderr_eof = True
                    if self._state == "STARTING":
                        self._poison_locked("launcher exited before readiness")
                    elif self._state == "ACTIVE" and self._exchange_stderr_end is None:
                        self._poison_locked("stderr ended without an exchange-end marker")
                    self._cv.notify_all()
                    return
                self._stderr_raw.extend(chunk)
                self._stderr_partial.extend(chunk)
                if len(self._stderr_raw) > MAX_STREAM_BYTES:
                    self._poison_locked("stderr stream exceeded its bound")
                    return
                while b"\n" in self._stderr_partial:
                    end = self._stderr_partial.index(b"\n") + 1
                    line = bytes(self._stderr_partial[:end])
                    del self._stderr_partial[:end]
                    line_start = self._stderr_line_offset
                    self._stderr_line_offset += len(line)
                    if line.startswith(prefix) and self._state == "STARTING":
                        try:
                            metadata = _strict_object(
                                line[len(prefix):], "readiness record", MAX_REPLY_BYTES,
                            )
                        except PersistentTransportError as exc:
                            self._poison_locked(str(exc))
                            return
                        self._ready_metadata = metadata
                        self._state = "READY"
                        self._cv.notify_all()
                    elif line.startswith(EXCHANGE_END_PREFIX):
                        try:
                            marker = _strict_object(
                                line[len(EXCHANGE_END_PREFIX):],
                                "exchange-end marker",
                                MAX_REPLY_BYTES,
                            )
                        except PersistentTransportError as exc:
                            self._poison_locked(str(exc))
                            return
                        if set(marker) != {"launch_id"} \
                                or type(marker["launch_id"]) is not int:
                            self._poison_locked("exchange-end marker has invalid fields")
                            return
                        if self._state != "ACTIVE" \
                                or marker["launch_id"] != self._active_launch_id:
                            self._poison_locked("unsolicited or cross-talk exchange-end marker")
                            return
                        if self._exchange_stderr_end is not None:
                            self._poison_locked("duplicate exchange-end marker")
                            return
                        self._exchange_stderr_end = line_start
                        self._cv.notify_all()
                    elif self._state == "READY":
                        self._poison_locked("unsolicited stderr outside an active exchange")
                        return
                    elif self._state == "ACTIVE" and self._exchange_stderr_end is not None:
                        self._poison_locked("trailing stderr after the exchange-end marker")
                        return

    def exchange(self, request: ExecutionRequest, payload: bytes) -> TransportReply:
        if type(request) is not ExecutionRequest:
            raise PersistentTransportError("request must be an ExecutionRequest")
        request.validate()
        request_record = _strict_object(payload, "request", MAX_REQUEST_BYTES)
        if request_record.get("launch_id") != request.launch_id:
            raise PersistentTransportError("request frame launch_id does not match its typed request")
        with self._cv:
            if self._state != "READY" or self._poison_reason is not None:
                raise PersistentTransportError(
                    f"transport cannot exchange in state {self._state}"
                )
            if self._frames or self._stdout_partial:
                self._poison_locked("reply bytes were present before the request")
                raise PersistentTransportError(self._poison_reason)
            if self._process.poll() is not None or self._process.stdin is None:
                self._state = "STOPPED"
                raise PersistentTransportError("launcher is not running")
            if request.launch_id != self._last_launch_id + 1:
                raise PersistentTransportError("launch_id is not contiguous")
            self._exchange_number += 1
            number = self._exchange_number
            run_dir = self._artifact_dir / f"exchange-{number:06d}-launch-{request.launch_id}"
            run_dir.mkdir()
            stdout_start = len(self._stdout_raw)
            stderr_start = len(self._stderr_raw)
            self._active_launch_id = request.launch_id
            self._exchange_stderr_end = None
            self._state = "ACTIVE"
        (run_dir / "request.json").write_bytes(payload)

        start_ns = time.monotonic_ns()
        try:
            assert self._process.stdin is not None
            self._process.stdin.write(payload)
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            with self._cv:
                self._poison_locked(f"request write failed: {exc}")

        deadline = time.monotonic() + request.timeout_us / 1_000_000
        with self._cv:
            while self._state == "ACTIVE" \
                    and (not self._frames or self._exchange_stderr_end is None) \
                    and time.monotonic() < deadline:
                self._cv.wait(min(0.05, max(0.0, deadline - time.monotonic())))
            elapsed_us = max(0, (time.monotonic_ns() - start_ns) // 1000)
            if self._state == "ACTIVE" and self._frames \
                    and self._exchange_stderr_end is not None:
                reply = self._frames.popleft()
                try:
                    reply_record = _strict_object(reply, "reply", MAX_REPLY_BYTES)
                    if reply_record.get("launch_id") != request.launch_id:
                        raise PersistentTransportError(
                            "reply launch_id does not match the active request"
                        )
                except PersistentTransportError as exc:
                    self._poison_locked(str(exc))
                if self._state == "ACTIVE":
                    self._state = "STOPPED" if self._stdout_eof else "READY"
                state = "reply" if self._poison_reason is None else "error"
            elif self._state == "ACTIVE":
                self._poison_locked("reply timed out")
                reply = b""
                state = "timed_out"
            else:
                reply = b""
                state = "error"
            stdout = bytes(self._stdout_raw[stdout_start:])
            stderr_end = len(self._stderr_raw) if self._exchange_stderr_end is None \
                else self._exchange_stderr_end
            stderr = bytes(self._stderr_raw[stderr_start:stderr_end])
            poison = self._poison_reason
            self._active_launch_id = None
            if state == "reply":
                self._last_launch_id = request.launch_id

        self._write_exchange_artifacts(
            run_dir, payload, reply, stdout, stderr, state, elapsed_us, poison,
        )
        if state != "reply":
            self._stop_process(preserve_poison=True)
        return TransportReply(state, elapsed_us, reply if state == "reply" else b"")

    def drain(self) -> None:
        with self._cv:
            if self._state == "ACTIVE":
                raise PersistentTransportError("cannot drain with an active exchange")
            if self._state != "READY" or self._poison_reason is not None:
                raise PersistentTransportError(f"cannot drain in state {self._state}")
            self._state = "DRAINING"
            stream = self._process.stdin
        if stream is not None and not stream.closed:
            stream.close()

    def terminate(self) -> None:
        with self._cv:
            if self._state == "FINALIZED":
                return
        self._stop_process(preserve_poison=False)

    def _stop_process(self, preserve_poison: bool) -> None:
        if self._process.poll() is None:
            self._terminated_by_host = True
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        with self._cv:
            if preserve_poison and self._poison_reason is not None:
                self._state = "POISONED"
            else:
                self._state = "STOPPED"
            self._active_launch_id = None
            self._cv.notify_all()

    def finalize(self, timeout_s: float = 30.0) -> None:
        with self._cv:
            if self._state == "ACTIVE":
                raise PersistentTransportError("cannot finalize with an active exchange")
            if self._state == "FINALIZED":
                return
            if self._state == "READY":
                raise PersistentTransportError("drain or terminate before finalize")
            state = self._state
        if state == "DRAINING":
            try:
                self._process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired as exc:
                with self._cv:
                    self._poison_locked("launcher drain timed out")
                self._stop_process(preserve_poison=True)
                raise PersistentTransportError("launcher drain timed out") from exc
        elif self._process.poll() is None:
            self._stop_process(preserve_poison=True)
        for thread in self._threads:
            thread.join(timeout=5)
        if any(thread.is_alive() for thread in self._threads):
            raise PersistentTransportError("launcher stream thread did not stop")
        exit_error = self._process.poll() != 0 and not self._terminated_by_host \
            and self._poison_reason is None
        if exit_error:
            with self._cv:
                self._poison_locked("launcher exited nonzero")
        with self._cv:
            stdout = bytes(self._stdout_raw)
            stderr = bytes(self._stderr_raw)
            prior_state = self._state
            poison = self._poison_reason
            returncode = self._process.poll()
            self._state = "FINALIZED"
        (self._artifact_dir / "process.stdout.bin").write_bytes(stdout)
        (self._artifact_dir / "process.stderr.bin").write_bytes(stderr)
        metadata = {
            "exchanges": self._exchange_number,
            "poison_reason": poison,
            "returncode": returncode,
            "state_before_finalize": prior_state,
            "stderr_sha256": _digest(stderr),
            "stdout_sha256": _digest(stdout),
        }
        (self._artifact_dir / "process.json").write_bytes(
            (json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
        )
        if exit_error:
            raise PersistentTransportError("launcher exited nonzero")

    @staticmethod
    def _write_exchange_artifacts(run_dir: Path, request: bytes, reply: bytes,
                                  stdout: bytes, stderr: bytes, state: str,
                                  elapsed_us: int, poison: str | None) -> None:
        (run_dir / "reply.json").write_bytes(reply)
        (run_dir / "stdout.bin").write_bytes(stdout)
        (run_dir / "stderr.bin").write_bytes(stderr)
        metadata = {
            "elapsed_us": elapsed_us,
            "poison_reason": poison,
            "reply_sha256": _digest(reply),
            "request_sha256": _digest(request),
            "state": state,
            "stderr_sha256": _digest(stderr),
            "stdout_sha256": _digest(stdout),
        }
        (run_dir / "transport.json").write_bytes(
            (json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
        )
