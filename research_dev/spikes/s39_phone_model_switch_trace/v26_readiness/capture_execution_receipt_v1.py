#!/usr/bin/env python3
"""Execute one V2.6 capture producer from a verified source descriptor."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from typing import Any


SCHEMA = "s39-cp0-r1-capture-execution-receipt-v2.6"
MODEL_ID = "qwen3-14b-q4_k_m"
PHASE = "A_ONLY"
MAX_BYTES = 512 * 1024 * 1024


class ReceiptError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiptError(message)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=True, sort_keys=True,
                           separators=(",", ":")) + "\n").encode("ascii")
    except (TypeError, UnicodeEncodeError) as error:
        raise ReceiptError("E_CANONICAL") from error


def _stat(value: os.stat_result) -> dict[str, int]:
    return {
        "ctime_ns": value.st_ctime_ns,
        "device_id": value.st_dev,
        "inode": value.st_ino,
        "mode": value.st_mode,
        "mtime_ns": value.st_mtime_ns,
        "size": value.st_size,
    }


def _same_stat(left: dict[str, int], right: dict[str, int]) -> bool:
    return left == right


def read_stable(path: Path, field: str) -> tuple[bytes, dict[str, int], dict[str, int]]:
    require(path.is_absolute(), f"E_PATH: {field}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        before_path = os.lstat(path)
        require(stat.S_ISREG(before_path.st_mode), f"E_REGULAR: {field}")
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ReceiptError(f"E_OPEN: {field}") from error
    try:
        before = _stat(before_path)
        opened = os.fstat(descriptor)
        require(_same_stat(before, _stat(opened)), f"E_OPEN_RACE: {field}")
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            total += len(block)
            require(total <= MAX_BYTES, f"E_SIZE: {field}")
            chunks.append(block)
        after_fd = _stat(os.fstat(descriptor))
    finally:
        os.close(descriptor)
    after_path = _stat(os.lstat(path))
    require(_same_stat(before, after_fd), f"E_MUTATION: {field}")
    require(_same_stat(after_fd, after_path), f"E_MUTATION: {field}")
    raw = b"".join(chunks)
    require(len(raw) == before["size"], f"E_SIZE: {field}")
    return raw, before, after_path


def _now() -> tuple[int, int]:
    return (
        time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
        time.time_ns(),
    )


def _blob(raw: bytes) -> dict[str, Any]:
    return {
        "bytes": len(raw),
        "content_b64": base64.b64encode(raw).decode("ascii"),
        "sha256": digest(raw),
    }


def _input(role: str, path: Path) -> dict[str, Any]:
    raw, before, after = read_stable(path, role)
    return {
        "bytes": len(raw),
        "path": str(path),
        "role": role,
        "sha256": digest(raw),
        "stat_after": after,
        "stat_before": before,
    }


def _source(path: Path) -> tuple[dict[str, Any], bytes]:
    raw, before, after = read_stable(path, "source")
    return ({
        "bytes": len(raw),
        "execution_mode": "VERIFIED_OPEN_FD",
        "path": str(path),
        "sha256": digest(raw),
        "stat_after": after,
        "stat_before": before,
    }, raw)


def _fd_exec_argv(
    python: str,
    source_path: Path,
    source_fd: int,
    producer_argv: list[str],
) -> list[str]:
    wrapper = (
        "import os,sys; fd=int(sys.argv[1]); "
        "src=os.read(fd,524288000); "
        "ns={'__name__':'__main__','__file__':sys.argv[2],"
        "'__package__':None,'__cached__':None}; "
        "sys.argv=sys.argv[2:]; exec(compile(src,sys.argv[0],'exec'),ns)"
    )
    return [python, "-c", wrapper, str(source_fd), str(source_path), *producer_argv[1:]]


def execute(
    *,
    capture_kind: str,
    producer_role: str,
    phase_id: str,
    contract_sha256: str,
    execution_plan_sha256: str,
    runtime_bundle_plan_sha256: str,
    source_path: Path,
    input_paths: list[tuple[str, Path]],
    producer_argv: list[str],
    cwd: Path,
    environment: dict[str, str],
    timeout_seconds: int,
    result_path: Path,
) -> dict[str, Any]:
    require(capture_kind in {"artifact_root", "fast_fresh_readiness"}, "E_KIND")
    require(producer_role in {"capture.artifact_root", "capture.fast_fresh_readiness"}, "E_ROLE")
    require(phase_id.startswith("cp0-r1-v26-a-only-"), "E_PHASE_ID")
    for value, field in (
        (contract_sha256, "contract"),
        (execution_plan_sha256, "execution_plan"),
        (runtime_bundle_plan_sha256, "runtime_plan"),
    ):
        require(type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value), f"E_DIGEST: {field}")
    require(type(producer_argv) is list and len(producer_argv) >= 2, "E_ARGV")
    require(cwd.is_absolute() and result_path.is_absolute(), "E_PATH")
    require(1 <= timeout_seconds <= 7200, "E_TIMEOUT")
    source, source_raw = _source(source_path)
    inputs = [_input(role, path) for role, path in input_paths]
    start_mono, start_utc = _now()
    fd = os.open(source_path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        before_fd = os.fstat(fd)
        require(_stat(before_fd) == source["stat_before"], "E_SOURCE_FD")
        argv = _fd_exec_argv(str(Path(sys.executable).resolve()), source_path, fd, producer_argv)
        completed = subprocess.run(
            argv,
            cwd=str(cwd),
            env={str(key): str(value) for key, value in environment.items()},
            pass_fds=(fd,),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise ReceiptError("E_TIMEOUT") from error
    finally:
        os.close(fd)
    try:
        source_after_run = _stat(os.lstat(source_path))
    except OSError as error:
        raise ReceiptError("E_SOURCE_AFTER") from error
    require(source_after_run == source["stat_after"], "E_SOURCE_MUTATION")
    end_mono, end_utc = _now()
    require(completed.returncode == 0, f"E_RETURNCODE: {completed.returncode}")
    result_raw, result_before, result_after = read_stable(result_path, "result")
    return {
        "capture_kind": capture_kind,
        "clock_id": "HOST_MONOTONIC_RAW",
        "completed_monotonic_ns": end_mono,
        "completed_utc_ns": end_utc,
        "contract_sha256": contract_sha256,
        "execution_plan_sha256": execution_plan_sha256,
        "inputs": inputs,
        "invocation": {
            "argv": argv,
            "cwd": str(cwd),
            "env": dict(environment),
            "timeout_seconds": timeout_seconds,
        },
        "model_id": MODEL_ID,
        "phase": PHASE,
        "phase_id": phase_id,
        "process": {
            "returncode": 0,
            "stderr_b64": base64.b64encode(completed.stderr).decode("ascii"),
            "stderr_bytes": len(completed.stderr),
            "stderr_sha256": digest(completed.stderr),
            "stdout_b64": base64.b64encode(completed.stdout).decode("ascii"),
            "stdout_bytes": len(completed.stdout),
            "stdout_sha256": digest(completed.stdout),
        },
        "producer_role": producer_role,
        "result": {
            "bytes": len(result_raw),
            "content_b64": base64.b64encode(result_raw).decode("ascii"),
            "path": str(result_path),
            "sha256": digest(result_raw),
            "stat": result_before,
        },
        "runtime_bundle_plan_sha256": runtime_bundle_plan_sha256,
        "schema": SCHEMA,
        "source": source,
        "started_monotonic_ns": start_mono,
        "started_utc_ns": start_utc,
    }


def validate_receipt(value: dict[str, Any], *, source_path: Path, result_path: Path) -> None:
    require(value.get("schema") == SCHEMA, "E_SCHEMA")
    require(value.get("phase") == PHASE, "E_PHASE")
    require(value.get("model_id") == MODEL_ID, "E_MODEL")
    require(value.get("started_monotonic_ns", 0) < value.get("completed_monotonic_ns", 0), "E_INTERVAL")
    require(value.get("started_utc_ns", 0) < value.get("completed_utc_ns", 0), "E_INTERVAL")
    source_raw, source_before, source_after = read_stable(source_path, "source")
    source = value.get("source", {})
    require(source.get("path") == str(source_path), "E_SOURCE_PATH")
    require(source.get("sha256") == digest(source_raw), "E_SOURCE_DIGEST")
    require(source.get("stat_before") == source_before and source.get("stat_after") == source_after, "E_SOURCE_STAT")
    inputs = value.get("inputs")
    require(type(inputs) is list and len(inputs) >= 5, "E_INPUTS")
    seen_roles: set[str] = set()
    for row in inputs:
        require(type(row) is dict, "E_INPUT_ROW")
        role = row.get("role")
        path = Path(row.get("path", ""))
        require(type(role) is str and role not in seen_roles, "E_INPUT_ROLE")
        seen_roles.add(role)
        raw, before, after = read_stable(path, role)
        require(row.get("bytes") == len(raw), f"E_INPUT_BYTES: {role}")
        require(row.get("sha256") == digest(raw), f"E_INPUT_DIGEST: {role}")
        require(row.get("stat_before") == before and row.get("stat_after") == after, f"E_INPUT_STAT: {role}")
    python_rows = [row for row in inputs if row.get("role") == "python"]
    require(len(python_rows) == 1, "E_PYTHON_INPUT")
    invocation = value.get("invocation", {})
    require(invocation.get("argv", [None])[0] == python_rows[0]["path"], "E_PYTHON_ARGV")
    result_raw, result_before, _ = read_stable(result_path, "result")
    result = value.get("result", {})
    require(result.get("path") == str(result_path), "E_RESULT_PATH")
    require(result.get("sha256") == digest(result_raw), "E_RESULT_DIGEST")
    require(result.get("stat") == result_before, "E_RESULT_STAT")
    process = value.get("process", {})
    for stream in ("stdout", "stderr"):
        raw = base64.b64decode(process.get(stream + "_b64", ""), validate=True)
        require(len(raw) == process.get(stream + "_bytes"), f"E_PROCESS_{stream}")
        require(digest(raw) == process.get(stream + "_sha256"), f"E_PROCESS_{stream}")
