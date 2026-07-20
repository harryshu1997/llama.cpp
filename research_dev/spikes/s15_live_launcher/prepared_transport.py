#!/usr/bin/env python3
"""Prepared subprocess transport whose setup completes before exchange timing."""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path

from executor_contract import ExecutionRequest
from physical_executor import MAX_REPLY_BYTES, SessionTransport, TransportReply


class PreparedTransportError(RuntimeError):
    pass


class PreparedSubprocessTransport(SessionTransport):
    def __init__(self, command: tuple[str, ...], artifact_dir: Path,
                 ready_timeout_s: float = 600) -> None:
        if type(command) is not tuple or not command \
                or any(type(value) is not str or not value for value in command):
            raise PreparedTransportError("command must be a non-empty string tuple")
        if not isinstance(artifact_dir, Path) or artifact_dir.exists():
            raise PreparedTransportError("artifact directory must not exist")
        artifact_dir.mkdir(parents=True)
        self._artifact_dir = artifact_dir
        self._stdout = bytearray()
        self._stderr = bytearray()
        self._ready = threading.Event()
        self._reply = threading.Event()
        self._ready_metadata = None
        self._completion_elapsed_us = None
        self._process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._threads = (
            threading.Thread(target=self._read_stdout, daemon=True),
            threading.Thread(target=self._read_stderr, daemon=True),
        )
        for thread in self._threads:
            thread.start()
        deadline = time.monotonic() + ready_timeout_s
        while time.monotonic() < deadline:
            if self._ready.wait(0.1):
                return
            if self._process.poll() is not None:
                self._finish_artifacts("preflight_error", 0)
                raise PreparedTransportError("launcher exited before readiness")
        self.terminate()
        self._finish_artifacts("preflight_timeout", 0)
        raise PreparedTransportError("launcher preflight timed out")

    @property
    def ready_metadata(self) -> dict:
        if type(self._ready_metadata) is not dict:
            raise PreparedTransportError("launcher readiness metadata is unavailable")
        return dict(self._ready_metadata)

    def _read_stdout(self) -> None:
        assert self._process.stdout is not None
        for line in iter(self._process.stdout.readline, b""):
            self._stdout.extend(line)
            if b"\n" in self._stdout:
                self._reply.set()

    def _read_stderr(self) -> None:
        assert self._process.stderr is not None
        for line in iter(self._process.stderr.readline, b""):
            self._stderr.extend(line)
            if line.startswith(b"LAUNCHER_READY "):
                try:
                    metadata = json.loads(line[len(b"LAUNCHER_READY "):])
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if type(metadata) is dict:
                    self._ready_metadata = metadata
                    self._ready.set()

    def exchange(self, request: ExecutionRequest, payload: bytes) -> TransportReply:
        if self._process.poll() is not None or not self._ready.is_set() \
                or self._process.stdin is None:
            return TransportReply("error", 0)
        (self._artifact_dir / "request.json").write_bytes(payload)
        start_ns = time.monotonic_ns()
        try:
            self._process.stdin.write(payload)
            self._process.stdin.close()
            if not self._reply.wait(timeout=request.timeout_us / 1_000_000):
                raise subprocess.TimeoutExpired(self._process.args, request.timeout_us / 1_000_000)
            elapsed_us = (time.monotonic_ns() - start_ns) // 1000
            self._completion_elapsed_us = elapsed_us
            stdout = bytes(self._stdout)
            state = "reply" if 0 < len(stdout) <= MAX_REPLY_BYTES \
                and stdout.count(b"\n") == 1 and stdout.endswith(b"\n") else "error"
        except subprocess.TimeoutExpired:
            elapsed_us = (time.monotonic_ns() - start_ns) // 1000
            self.terminate()
            state = "timed_out"
            self._finish_artifacts(state, elapsed_us, None)
            return TransportReply(state, elapsed_us, b"")
        if state == "error":
            self.terminate()
            self._finish_artifacts(state, elapsed_us, self._process.poll())
        return TransportReply(state, elapsed_us, stdout if state == "reply" else b"")

    def finalize(self, timeout_s: float = 30) -> None:
        try:
            returncode = self._process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            self.terminate()
            self._finish_artifacts("finalize_timeout", self._completion_elapsed_us or 0, None)
            raise PreparedTransportError("launcher teardown timed out") from exc
        for thread in self._threads:
            thread.join(timeout=5)
        if returncode != 0 or any(thread.is_alive() for thread in self._threads):
            self._finish_artifacts("finalize_error", self._completion_elapsed_us or 0, returncode)
            raise PreparedTransportError("launcher failed after emitting its session record")
        self._finish_artifacts("reply", self._completion_elapsed_us or 0, returncode)

    def terminate(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)

    def _finish_artifacts(self, state: str, elapsed_us: int,
                          returncode: int | None = None) -> None:
        for thread in getattr(self, "_threads", ()):
            thread.join(timeout=2)
        stdout = bytes(self._stdout)
        stderr = bytes(self._stderr)
        (self._artifact_dir / "stdout.bin").write_bytes(stdout)
        (self._artifact_dir / "stderr.bin").write_bytes(stderr)
        metadata = {
            "state": state,
            "elapsed_us": elapsed_us,
            "returncode": returncode,
            "stdout_sha256": "sha256:" + hashlib.sha256(stdout).hexdigest(),
            "stderr_sha256": "sha256:" + hashlib.sha256(stderr).hexdigest(),
        }
        (self._artifact_dir / "transport.json").write_text(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="ascii",
        )
