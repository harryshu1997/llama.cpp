#!/usr/bin/env python3
"""Capture and validate exclusive selected-GPU use for an S40 run."""

from __future__ import annotations

import argparse
import base64
import csv
import fcntl
import io
import os
from pathlib import Path
import re
import stat
import subprocess
import time
from typing import Any, Callable

from evidence_common import (
    EvidenceError,
    canonical_bytes,
    digest_file,
    is_int,
    read_jsonl,
    require,
    require_int,
    require_string,
    validate_digest,
)


IDENTITY_QUERY = "uuid,name,pci.bus_id,index"
PROCESS_QUERY = "pid,gpu_uuid,process_name,used_gpu_memory"
HEADER_KEYS = {
    "gpu_name",
    "gpu_uuid",
    "host_boot_id",
    "identity_argv",
    "nvidia_smi_bytes",
    "nvidia_smi_path",
    "nvidia_smi_sha256",
    "process_argv",
    "run_id",
    "schema",
    "started_ns",
    "type",
}
SAMPLE_KEYS = {
    "completed_ns",
    "identity_stderr_base64",
    "identity_stdout_base64",
    "process_observations",
    "process_stderr_base64",
    "process_stdout_base64",
    "sequence",
    "started_ns",
    "type",
}
PROCESS_OBSERVATION_KEYS = {
    "cmdline_base64",
    "pid",
    "stat_base64",
}
FOOTER_KEYS = {
    "completed_ns",
    "sample_count",
    "type",
}
LOCK_KEYS = {
    "acquired_ns",
    "device",
    "gpu_uuid",
    "host_boot_id",
    "inode",
    "lock_path",
    "owner_pid",
    "owner_start_ticks",
    "released_ns",
    "run_id",
    "schema",
}
GPU_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def _ascii_b64(raw: bytes, field: str) -> str:
    try:
        raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise EvidenceError(f"{field}: expected ASCII") from error
    return base64.b64encode(raw).decode("ascii")


def _decode_b64(value: Any, field: str) -> bytes:
    require(isinstance(value, str) and value.isascii(),
            f"{field}: expected ASCII string")
    text = value
    try:
        raw = base64.b64decode(text, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise EvidenceError(f"{field}: invalid base64") from error
    require(base64.b64encode(raw).decode("ascii") == text,
            f"{field}: noncanonical base64")
    return raw


def _csv_rows(raw: bytes, field: str) -> list[list[str]]:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise EvidenceError(f"{field}: expected ASCII") from error
    try:
        return [
            [item.strip() for item in row]
            for row in csv.reader(io.StringIO(text))
            if row and any(item.strip() for item in row)
        ]
    except csv.Error as error:
        raise EvidenceError(f"{field}: invalid CSV") from error


def parse_gpu_identity(
        raw: bytes,
        expected_uuid: str,
        expected_name: str,
) -> dict[str, Any]:
    rows = _csv_rows(raw, "GPU identity")
    require(len(rows) == 1 and len(rows[0]) == 4,
            "GPU identity: expected one four-field row")
    uuid, name, pci_bus_id, index = rows[0]
    require(uuid == expected_uuid, "GPU identity: UUID mismatch")
    require(name == expected_name, "GPU identity: name mismatch")
    require(bool(pci_bus_id) and pci_bus_id.isascii(),
            "GPU identity: invalid PCI bus ID")
    require(index.isascii() and index.isdigit(),
            "GPU identity: invalid index")
    return {
        "gpu_index": int(index),
        "gpu_name": name,
        "gpu_pci_bus_id": pci_bus_id,
        "gpu_uuid": uuid,
    }


def parse_gpu_processes(
        raw: bytes,
        expected_uuid: str,
) -> list[dict[str, Any]]:
    result = []
    pids: set[int] = set()
    for index, row in enumerate(_csv_rows(raw, "GPU processes")):
        require(len(row) == 4,
                f"GPU processes[{index}]: expected four fields")
        pid_text, uuid, process_name, used_text = row
        require(pid_text.isascii() and pid_text.isdigit(),
                f"GPU processes[{index}]: invalid PID")
        pid = int(pid_text)
        require(pid > 0 and pid not in pids,
                f"GPU processes[{index}]: duplicate or invalid PID")
        pids.add(pid)
        require(uuid == expected_uuid,
                f"GPU processes[{index}]: GPU UUID mismatch")
        require(bool(process_name) and process_name.isascii(),
                f"GPU processes[{index}]: invalid process name")
        require(used_text.isascii() and used_text.isdigit(),
                f"GPU processes[{index}]: invalid memory")
        result.append({
            "gpu_uuid": uuid,
            "pid": pid,
            "process_name": process_name,
            "used_gpu_memory_mib": int(used_text),
        })
    return result


def parse_process_stat(raw: bytes) -> tuple[int, int]:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise EvidenceError("process stat: expected ASCII") from error
    opening = text.find("(")
    closing = text.rfind(")")
    require(
        opening > 0
        and closing > opening
        and text[:opening].strip().isdigit(),
        "process stat: invalid framing",
    )
    fields = text[closing + 1:].split()
    require(len(fields) > 19, "process stat: missing fields")
    require(fields[19].isascii() and fields[19].isdigit(),
            "process stat: invalid start ticks")
    return int(text[:opening].strip()), int(fields[19])


def parse_cmdline(raw: bytes) -> list[str]:
    require(raw.endswith(b"\0"), "process cmdline: incomplete")
    parts = raw[:-1].split(b"\0")
    require(parts and all(parts), "process cmdline: empty argument")
    result = []
    for part in parts:
        try:
            value = part.decode("utf-8")
        except UnicodeDecodeError as error:
            raise EvidenceError("process cmdline: invalid UTF-8") from error
        require(bool(value), "process cmdline: empty argument")
        result.append(value)
    return result


def _process_start_ticks(pid: int) -> int:
    raw = Path(f"/proc/{pid}/stat").read_bytes()
    parsed_pid, start_ticks = parse_process_stat(raw)
    require(parsed_pid == pid, "lock owner: PID mismatch")
    require(start_ticks > 0, "lock owner: invalid start ticks")
    return start_ticks


def _write_all(descriptor: int, raw: bytes) -> None:
    offset = 0
    while offset < len(raw):
        written = os.write(descriptor, raw[offset:])
        require(written > 0, "GPU lock: short write")
        offset += written


def canonical_lock_path(gpu_uuid: str) -> Path:
    require(
        isinstance(gpu_uuid, str)
        and GPU_ID.fullmatch(gpu_uuid) is not None,
        "GPU lock: invalid GPU identity",
    )
    return Path("/tmp") / f"llama-s40-{gpu_uuid}.lock"


class ExclusiveGpuLock:
    def __init__(
        self,
        path: Path,
        gpu_uuid: str,
        run_id: str,
        host_boot_id: str,
    ):
        require(path.is_absolute(), "GPU lock: path must be absolute")
        require(path.parent.is_dir(), "GPU lock: parent is missing")
        require_string(gpu_uuid, "GPU lock UUID")
        require_string(run_id, "GPU lock run ID")
        require_string(host_boot_id, "GPU lock boot ID")
        self.path = path
        self.gpu_uuid = gpu_uuid
        self.run_id = run_id
        self.host_boot_id = host_boot_id
        self.descriptor: int | None = None
        self.record: dict[str, Any] | None = None

    def acquire(self) -> dict[str, Any]:
        require(self.descriptor is None, "GPU lock: already acquired")
        flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        try:
            info = os.fstat(descriptor)
            require(stat.S_ISREG(info.st_mode), "GPU lock: not a regular file")
            require(info.st_uid == os.geteuid(),
                    "GPU lock: owner mismatch")
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired_ns = time.monotonic_ns()
            record = {
                "acquired_ns": acquired_ns,
                "device": info.st_dev,
                "gpu_uuid": self.gpu_uuid,
                "host_boot_id": self.host_boot_id,
                "inode": info.st_ino,
                "lock_path": str(self.path),
                "owner_pid": os.getpid(),
                "owner_start_ticks": _process_start_ticks(os.getpid()),
                "released_ns": None,
                "run_id": self.run_id,
                "schema": "s40-selected-gpu-lock-v1",
            }
            raw = canonical_bytes(record)
            os.ftruncate(descriptor, 0)
            _write_all(descriptor, raw)
            os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            require(
                os.read(descriptor, len(raw) + 1) == raw,
                "GPU lock: active record write mismatch",
            )
            self.descriptor = descriptor
            self.record = record
            return dict(record)
        except BaseException:
            os.close(descriptor)
            raise

    def release(self) -> dict[str, Any]:
        require(
            self.descriptor is not None and self.record is not None,
            "GPU lock: not acquired",
        )
        record = dict(self.record)
        record["released_ns"] = time.monotonic_ns()
        require(record["released_ns"] >= record["acquired_ns"],
                "GPU lock: invalid interval")
        raw = canonical_bytes(record)
        os.ftruncate(self.descriptor, 0)
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        _write_all(self.descriptor, raw)
        os.fsync(self.descriptor)
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        require(
            os.read(self.descriptor, len(raw) + 1) == raw,
            "GPU lock: released record write mismatch",
        )
        fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        os.close(self.descriptor)
        self.descriptor = None
        self.record = record
        return dict(record)

    def __enter__(self) -> "ExclusiveGpuLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.descriptor is not None:
            self.release()


def validate_lock_record(
        value: Any,
        expected_uuid: str,
        expected_run_id: str,
        expected_boot_id: str,
) -> dict[str, Any]:
    require(isinstance(value, dict) and set(value) == LOCK_KEYS,
            "GPU lock evidence: key set mismatch")
    require(
        value["schema"] == "s40-selected-gpu-lock-v1"
        and value["gpu_uuid"] == expected_uuid
        and value["run_id"] == expected_run_id
        and value["host_boot_id"] == expected_boot_id,
        "GPU lock evidence: identity mismatch",
    )
    acquired = require_int(value["acquired_ns"], "GPU lock acquired", 1)
    released = require_int(value["released_ns"], "GPU lock released", acquired)
    require_int(value["device"], "GPU lock device")
    require_int(value["inode"], "GPU lock inode", 1)
    require_int(value["owner_pid"], "GPU lock PID", 1)
    require_int(value["owner_start_ticks"], "GPU lock start ticks", 1)
    path = Path(require_string(value["lock_path"], "GPU lock path"))
    require(path.is_absolute(), "GPU lock evidence: path is not absolute")
    return {
        "acquired_ns": acquired,
        "released_ns": released,
    }


def _query(
        argv: list[str],
        runner: Callable[..., subprocess.CompletedProcess[bytes]],
) -> tuple[bytes, bytes]:
    result = runner(
        argv,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )
    require(result.returncode == 0, "nvidia-smi query failed")
    require(not result.stderr, "nvidia-smi query wrote stderr")
    return result.stdout, result.stderr


def capture_sample(
        sequence: int,
        identity_argv: list[str],
        process_argv: list[str],
        expected_uuid: str,
        expected_name: str,
        nvidia_smi: Path,
        nvidia_smi_sha256: str,
        runner: Callable[..., subprocess.CompletedProcess[bytes]]
        = subprocess.run,
        proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    require_int(sequence, "GPU sample sequence")
    require(
        nvidia_smi.is_absolute()
        and nvidia_smi.is_file()
        and digest_file(nvidia_smi) == nvidia_smi_sha256,
        "GPU sample: nvidia-smi changed before query",
    )
    started_ns = time.monotonic_ns()
    identity_stdout, identity_stderr = _query(identity_argv, runner)
    parse_gpu_identity(identity_stdout, expected_uuid, expected_name)
    process_stdout = b""
    process_stderr = b""
    observations = []
    last_error: BaseException | None = None
    for attempt in range(3):
        try:
            process_stdout, process_stderr = _query(process_argv, runner)
            processes = parse_gpu_processes(process_stdout, expected_uuid)
            observations = []
            for process in processes:
                pid = process["pid"]
                stat_raw = (proc_root / str(pid) / "stat").read_bytes()
                cmdline_raw = (proc_root / str(pid) / "cmdline").read_bytes()
                parsed_pid, start_ticks = parse_process_stat(stat_raw)
                require(parsed_pid == pid and start_ticks > 0,
                        "GPU process identity changed")
                parse_cmdline(cmdline_raw)
                observations.append({
                    "cmdline_base64": _ascii_b64(
                        cmdline_raw, "GPU process cmdline"),
                    "pid": pid,
                    "stat_base64": _ascii_b64(
                        stat_raw, "GPU process stat"),
                })
            last_error = None
            break
        except (EvidenceError, OSError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(0.01)
    if last_error is not None:
        raise EvidenceError(
            f"GPU process snapshot did not stabilize: {last_error}"
        ) from last_error
    require(
        nvidia_smi.is_file()
        and digest_file(nvidia_smi) == nvidia_smi_sha256,
        "GPU sample: nvidia-smi changed after query",
    )
    return {
        "completed_ns": time.monotonic_ns(),
        "identity_stderr_base64": _ascii_b64(
            identity_stderr, "GPU identity stderr"),
        "identity_stdout_base64": _ascii_b64(
            identity_stdout, "GPU identity stdout"),
        "process_observations": observations,
        "process_stderr_base64": _ascii_b64(
            process_stderr, "GPU process stderr"),
        "process_stdout_base64": _ascii_b64(
            process_stdout, "GPU process stdout"),
        "sequence": sequence,
        "started_ns": started_ns,
        "type": "SAMPLE",
    }


def observation_commands(
        nvidia_smi: Path,
        gpu_uuid: str,
) -> tuple[list[str], list[str]]:
    executable = str(nvidia_smi)
    return (
        [
            executable,
            "--id", gpu_uuid,
            f"--query-gpu={IDENTITY_QUERY}",
            "--format=csv,noheader,nounits",
        ],
        [
            executable,
            "--id", gpu_uuid,
            f"--query-compute-apps={PROCESS_QUERY}",
            "--format=csv,noheader,nounits",
        ],
    )


def observe(
        *,
        output: Path,
        stop_file: Path,
        run_id: str,
        gpu_uuid: str,
        gpu_name: str,
        nvidia_smi: Path,
        nvidia_smi_sha256: str,
        interval_ms: int,
) -> int:
    require(output.is_absolute() and stop_file.is_absolute(),
            "GPU observer: paths must be absolute")
    require(not output.exists(), "GPU observer: output already exists")
    require(not stop_file.exists(), "GPU observer: stop file already exists")
    require(50 <= interval_ms <= 1000,
            "GPU observer: interval must be 50..1000 ms")
    require(nvidia_smi.is_absolute() and nvidia_smi.is_file(),
            "GPU observer: missing nvidia-smi")
    validate_digest(nvidia_smi_sha256, "nvidia-smi SHA-256")
    require(
        digest_file(nvidia_smi) == nvidia_smi_sha256,
        "GPU observer: nvidia-smi digest mismatch",
    )
    host_boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="ascii").strip()
    identity_argv, process_argv = observation_commands(nvidia_smi, gpu_uuid)
    header = {
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
        "host_boot_id": host_boot_id,
        "identity_argv": identity_argv,
        "nvidia_smi_bytes": nvidia_smi.stat().st_size,
        "nvidia_smi_path": str(nvidia_smi),
        "nvidia_smi_sha256": nvidia_smi_sha256,
        "process_argv": process_argv,
        "run_id": run_id,
        "schema": "s40-selected-gpu-observer-v1",
        "started_ns": time.monotonic_ns(),
        "type": "START",
    }
    count = 0
    with output.open("xb", buffering=0) as sink:
        sink.write(canonical_bytes(header))
        sink.flush()
        os.fsync(sink.fileno())
        while True:
            row = capture_sample(
                count,
                identity_argv,
                process_argv,
                gpu_uuid,
                gpu_name,
                nvidia_smi,
                nvidia_smi_sha256,
            )
            sink.write(canonical_bytes(row))
            sink.flush()
            os.fsync(sink.fileno())
            count += 1
            if stop_file.exists():
                break
            time.sleep(interval_ms / 1000)
        footer = {
            "completed_ns": time.monotonic_ns(),
            "sample_count": count,
            "type": "STOP",
        }
        sink.write(canonical_bytes(footer))
        sink.flush()
        os.fsync(sink.fileno())
    require(
        digest_file(nvidia_smi) == nvidia_smi_sha256,
        "GPU observer: nvidia-smi changed after acquisition",
    )
    return 0


def _validate_header(
        row: Any,
        expected_run_id: str,
        expected_uuid: str,
        expected_name: str,
) -> dict[str, Any]:
    require(isinstance(row, dict) and set(row) == HEADER_KEYS,
            "GPU observer header: key set mismatch")
    require(
        row["type"] == "START"
        and row["schema"] == "s40-selected-gpu-observer-v1"
        and row["run_id"] == expected_run_id
        and row["gpu_uuid"] == expected_uuid
        and row["gpu_name"] == expected_name,
        "GPU observer header: identity mismatch",
    )
    path = Path(require_string(
        row["nvidia_smi_path"], "GPU observer nvidia-smi path"))
    require(path.is_absolute(), "GPU observer: nvidia-smi path is not absolute")
    validate_digest(row["nvidia_smi_sha256"], "GPU observer binary digest")
    require_int(row["nvidia_smi_bytes"], "GPU observer binary bytes", 1)
    identity_argv, process_argv = observation_commands(path, expected_uuid)
    require(
        row["identity_argv"] == identity_argv
        and row["process_argv"] == process_argv,
        "GPU observer header: command mismatch",
    )
    require_int(row["started_ns"], "GPU observer start", 1)
    require_string(row["host_boot_id"], "GPU observer boot ID")
    return row


def _validate_sample(
        row: Any,
        expected_sequence: int,
        expected_uuid: str,
        expected_name: str,
) -> tuple[dict[str, Any], list[tuple[int, int, list[str]]]]:
    field = f"GPU observer sample[{expected_sequence}]"
    require(isinstance(row, dict) and set(row) == SAMPLE_KEYS,
            f"{field}: key set mismatch")
    require(
        row["type"] == "SAMPLE"
        and row["sequence"] == expected_sequence,
        f"{field}: sequence mismatch",
    )
    started = require_int(row["started_ns"], f"{field}.started_ns", 1)
    completed = require_int(
        row["completed_ns"], f"{field}.completed_ns", started)
    identity_stdout = _decode_b64(
        row["identity_stdout_base64"], f"{field}.identity_stdout")
    identity_stderr = _decode_b64(
        row["identity_stderr_base64"], f"{field}.identity_stderr")
    process_stdout = _decode_b64(
        row["process_stdout_base64"], f"{field}.process_stdout")
    process_stderr = _decode_b64(
        row["process_stderr_base64"], f"{field}.process_stderr")
    require(not identity_stderr and not process_stderr,
            f"{field}: nvidia-smi wrote stderr")
    identity = parse_gpu_identity(
        identity_stdout, expected_uuid, expected_name)
    processes = parse_gpu_processes(process_stdout, expected_uuid)
    observations = row["process_observations"]
    require(isinstance(observations, list)
            and len(observations) == len(processes),
            f"{field}: process observation count mismatch")
    by_pid = {process["pid"]: process for process in processes}
    require(len(by_pid) == len(processes),
            f"{field}: duplicate process PID")
    identities = []
    seen: set[int] = set()
    for index, observation in enumerate(observations):
        item = f"{field}.process_observations[{index}]"
        require(
            isinstance(observation, dict)
            and set(observation) == PROCESS_OBSERVATION_KEYS,
            f"{item}: key set mismatch",
        )
        pid = require_int(observation["pid"], f"{item}.pid", 1)
        require(pid in by_pid and pid not in seen,
                f"{item}: process query mismatch")
        seen.add(pid)
        stat_raw = _decode_b64(
            observation["stat_base64"], f"{item}.stat")
        cmdline_raw = _decode_b64(
            observation["cmdline_base64"], f"{item}.cmdline")
        parsed_pid, start_ticks = parse_process_stat(stat_raw)
        require(parsed_pid == pid and start_ticks > 0,
                f"{item}: process identity mismatch")
        identities.append((pid, start_ticks, parse_cmdline(cmdline_raw)))
    require(seen == set(by_pid), f"{field}: missing process observation")
    return {
        "completed_ns": completed,
        "identity": identity,
        "processes": processes,
        "started_ns": started,
    }, identities


def validate_observation(
        path: Path,
        *,
        expected_run_id: str,
        expected_uuid: str,
        expected_name: str,
        expected_boot_id: str,
        trace_start_ns: int,
        trace_end_ns: int,
        allowed_processes: dict[tuple[int, int], list[str]],
        max_gap_ns: int = 1_000_000_000,
        max_probe_duration_ns: int = 1_000_000_000,
) -> dict[str, Any]:
    rows = read_jsonl(path, "GPU observer")
    require(len(rows) >= 4, "GPU observer: insufficient rows")
    header = _validate_header(
        rows[0], expected_run_id, expected_uuid, expected_name)
    require(header["host_boot_id"] == expected_boot_id,
            "GPU observer: boot identity mismatch")
    require(is_int(trace_start_ns) and is_int(trace_end_ns)
            and 0 < trace_start_ns <= trace_end_ns,
            "GPU observer: invalid trace interval")
    require(is_int(max_gap_ns) and max_gap_ns > 0,
            "GPU observer: invalid maximum gap")
    require(
        is_int(max_probe_duration_ns) and max_probe_duration_ns > 0,
        "GPU observer: invalid maximum probe duration",
    )
    require(
        all(
            type(key) is tuple
            and len(key) == 2
            and is_int(key[0])
            and is_int(key[1])
            and key[0] > 0
            and key[1] > 0
            and type(argv) is list
            and argv
            and all(type(item) is str and item for item in argv)
            for key, argv in allowed_processes.items()
        ),
        "GPU observer: invalid allowed process set",
    )
    footer = rows[-1]
    require(isinstance(footer, dict) and set(footer) == FOOTER_KEYS,
            "GPU observer footer: key set mismatch")
    require(footer["type"] == "STOP", "GPU observer footer: type mismatch")
    sample_rows = rows[1:-1]
    require(
        require_int(
            footer["sample_count"], "GPU observer sample count", 1)
        == len(sample_rows),
        "GPU observer footer: sample count mismatch",
    )
    validated = []
    observed_allowed: set[tuple[int, int]] = set()
    previous_completed = None
    for sequence, row in enumerate(sample_rows):
        sample, identities = _validate_sample(
            row, sequence, expected_uuid, expected_name)
        require(
            sample["completed_ns"] - sample["started_ns"]
            <= max_probe_duration_ns,
            "GPU observer: probe duration exceeded",
        )
        if previous_completed is not None:
            require(
                sample["started_ns"] >= previous_completed
                and sample["completed_ns"] - previous_completed
                <= max_gap_ns,
                "GPU observer: sample gap exceeded",
            )
        previous_completed = sample["completed_ns"]
        for pid, start_ticks, argv in identities:
            identity = (pid, start_ticks)
            require(identity in allowed_processes,
                    "GPU observer: foreign compute process")
            require(argv == allowed_processes[identity],
                    "GPU observer: compute process command mismatch")
            observed_allowed.add(identity)
        validated.append(sample)
    require(not validated[0]["processes"],
            "GPU observer: selected GPU was busy before launch")
    require(not validated[-1]["processes"],
            "GPU observer: selected GPU was busy after cleanup")
    require(
        validated[0]["completed_ns"] <= trace_start_ns
        and validated[-1]["started_ns"] >= trace_end_ns,
        "GPU observer: trace interval is not bracketed",
    )
    require(
        observed_allowed == set(allowed_processes),
        "GPU observer: expected compute process was never observed",
    )
    completed = require_int(
        footer["completed_ns"], "GPU observer completion",
        validated[-1]["completed_ns"])
    require(completed >= trace_end_ns,
            "GPU observer: stopped before trace completion")
    return {
        "allowed_process_count": len(allowed_processes),
        "completed_ns": completed,
        "gpu_index": validated[0]["identity"]["gpu_index"],
        "gpu_pci_bus_id": validated[0]["identity"]["gpu_pci_bus_id"],
        "host_boot_id": header["host_boot_id"],
        "nvidia_smi_bytes": header["nvidia_smi_bytes"],
        "nvidia_smi_path": header["nvidia_smi_path"],
        "nvidia_smi_sha256": header["nvidia_smi_sha256"],
        "run_id": header["run_id"],
        "sample_count": len(validated),
        "started_ns": header["started_ns"],
        "status": "PASS",
    }


def run_locked_observer(
        *,
        output: Path,
        stop_file: Path,
        lock_path: Path,
        lock_output: Path,
        run_id: str,
        gpu_uuid: str,
        gpu_name: str,
        nvidia_smi: Path,
        nvidia_smi_sha256: str,
        interval_ms: int,
) -> int:
    require(
        lock_path.is_absolute() and lock_output.is_absolute(),
        "GPU lock: paths must be absolute",
    )
    require(not lock_output.exists(), "GPU lock: output already exists")
    host_boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="ascii").strip()
    lock = ExclusiveGpuLock(lock_path, gpu_uuid, run_id, host_boot_id)
    lock.acquire()
    try:
        return observe(
            output=output,
            stop_file=stop_file,
            run_id=run_id,
            gpu_uuid=gpu_uuid,
            gpu_name=gpu_name,
            nvidia_smi=nvidia_smi,
            nvidia_smi_sha256=nvidia_smi_sha256,
            interval_ms=interval_ms,
        )
    finally:
        record = lock.release()
        raw = canonical_bytes(record)
        require(
            lock_path.read_bytes() == raw,
            "GPU lock: released on-disk record mismatch",
        )
        with lock_output.open("xb", buffering=0) as sink:
            sink.write(raw)
            sink.flush()
            os.fsync(sink.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--gpu-name", required=True)
    parser.add_argument("--nvidia-smi", type=Path, required=True)
    parser.add_argument("--nvidia-smi-sha256", required=True)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument("--lock-output", type=Path, required=True)
    parser.add_argument("--interval-ms", type=int, default=200)
    args = parser.parse_args()
    try:
        return run_locked_observer(
            output=args.output,
            stop_file=args.stop_file,
            lock_path=args.lock_path,
            lock_output=args.lock_output,
            run_id=args.run_id,
            gpu_uuid=args.gpu_uuid,
            gpu_name=args.gpu_name,
            nvidia_smi=args.nvidia_smi,
            nvidia_smi_sha256=args.nvidia_smi_sha256,
            interval_ms=args.interval_ms,
        )
    except (EvidenceError, OSError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
