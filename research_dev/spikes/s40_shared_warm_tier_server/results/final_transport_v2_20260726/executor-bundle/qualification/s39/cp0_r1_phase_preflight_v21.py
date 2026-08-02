#!/usr/bin/env python3

import argparse
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import cp0_r1_evidence_v2 as v2
import cp0_r1_evidence_v21 as v21


CLOCK_ID = time.CLOCK_MONOTONIC_RAW


def _read_lock(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    rows = raw.splitlines(keepends=True)
    v2.require(len(rows) == 1, f"E_ROWS: {path}")
    row = v2.parse_json(rows[0], str(path))
    v2.require(v2.canonical_line(row) == rows[0], f"E_CANONICAL: {path}")
    return row


def _write_exclusive(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    fd = os.open(path, flags, 0o644)
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(fd, raw[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


def _probe(
    phase: str,
    phase_id: str,
    label: str,
    argv: list[str],
    timeout_s: int,
) -> dict[str, Any]:
    started_ns = time.clock_gettime_ns(CLOCK_ID)
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_s,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="backslashreplace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="backslashreplace")
        timed_out = True
    return {
        "acquisition_id": phase_id,
        "argv": argv,
        "event_ns": time.clock_gettime_ns(CLOCK_ID),
        "kind": "probe",
        "label": label,
        "phase": phase,
        "phase_id": phase_id,
        "returncode": returncode,
        "role": "phase.preflight",
        "started_ns": started_ns,
        "stderr": stderr,
        "stdout": stdout,
        "timed_out": timed_out,
    }


def collect(
    contract: dict[str, Any],
    candidate: dict[str, Any],
    phase: str,
    phase_id: str,
    locks: dict[str, dict[str, Any]],
    timeout_s: int,
) -> bytes:
    expected_slots = {
        "A_ONLY": ["A"],
        "B_ONLY": ["B"],
        "PAIR": ["A", "B"],
    }[phase]
    v2.exact(sorted(locks), sorted(expected_slots), "preflight.lock_slots")
    model_locks = []
    for slot in expected_slots:
        model = next(model for model in candidate["models"] if model["slot"] == slot)
        lock = locks[slot]
        v2.exact(lock["model_id"], model["model_id"], f"preflight.{slot}.model")
        v2.exact(
            lock["model_sha256"],
            model["artifact"]["sha256"],
            f"preflight.{slot}.sha256",
        )
        model_locks.append((model, lock))
    commands = v21.expected_preflight_commands(contract, model_locks)
    rows = [
        _probe(phase, phase_id, label, commands[label], timeout_s)
        for label in sorted(commands)
    ]
    completed_ns = time.clock_gettime_ns(CLOCK_ID)
    rows.append(
        {
            "acquisition_id": phase_id,
            "completed_ns": completed_ns,
            "event_ns": completed_ns,
            "forbidden_work_executed": False,
            "kind": "meta",
            "phase": phase,
            "phase_id": phase_id,
            "probe_labels": sorted(commands),
            "role": "phase.preflight",
        }
    )
    return b"".join(v2.canonical_line(row) for row in rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture exact CP0-R1 V2.1 readiness immediately before acquisition"
    )
    parser.add_argument("--contract", type=Path, default=v21.DEFAULT_CONTRACT)
    parser.add_argument("--candidate", type=Path, default=v21.DEFAULT_CANDIDATE)
    parser.add_argument("--phase", choices=("A_ONLY", "B_ONLY", "PAIR"), required=True)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument(
        "--model-lock",
        action="append",
        default=[],
        metavar="SLOT=JSONL",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        contract, _, candidate, _, _ = v21.validate_inputs(
            args.contract,
            args.candidate,
        )
        locks = {}
        for value in args.model_lock:
            slot, separator, path = value.partition("=")
            v2.require(separator == "=" and slot in ("A", "B"), "E_LOCK_ARGUMENT")
            v2.require(slot not in locks, f"E_LOCK_REUSE: {slot}")
            locks[slot] = _read_lock(Path(path))
        v2.require(args.timeout_seconds > 0, "E_TIMEOUT")
        raw = collect(
            contract,
            candidate,
            args.phase,
            args.phase_id,
            locks,
            args.timeout_seconds,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _write_exclusive(args.output, raw)
        print(f"{v2.sha256_bytes(raw)}  {args.output}")
        return 0
    except (v2.EvidenceError, OSError, KeyError, ValueError) as exc:
        print(f"CP0_R1_V2_1_PREFLIGHT_REFUSED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
