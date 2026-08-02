#!/usr/bin/env python3
"""Fail-closed correctness and B32 probe for one resident phone range."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import struct
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence


HERE = Path(__file__).resolve().parent
S22 = HERE.parent / "s22_slo_overlap_pipeline"
for dependency in (S22,):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from activation_compare import compare_vectors
from async_pipeline import parse_endpoint, parse_tokens
from stage_v3_client import BatchResult, BatchRow, Hello, ProtocolError, StageV3Client


SCHEMA = "s32-quantized-residency-probe-v1"
SERIAL_RE = re.compile(r"^[A-Za-z0-9._:-]+$")


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def nearest_rank(values: Sequence[int], numerator: int, denominator: int) -> int:
    if not values or numerator <= 0 or denominator <= 0 or numerator > denominator:
        raise ValueError("invalid nearest-rank input")
    ordered = sorted(values)
    rank = (numerator * len(ordered) + denominator - 1) // denominator
    return ordered[rank - 1]


def summarize(values: Sequence[int]) -> dict[str, int | float]:
    if not values or any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("latency samples must be positive integers")
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "p95": nearest_rank(values, 95, 100),
        "max": max(values),
    }


def validate_hello(hello: Hello, start: int, end: int, batch: int, reference: bool) -> None:
    if (hello.layer_start, hello.layer_end) != (start, end):
        raise ProtocolError("worker layer range differs from requested range")
    if hello.n_layer != 48 or hello.n_embd != 3840:
        raise ProtocolError("worker model shape differs from Gemma-4-12B")
    if not reference and hello.max_streams < batch:
        raise ProtocolError("phone worker has fewer sequence slots than the requested batch")
    if min(hello.n_batch, hello.n_ubatch) < (1 if reference else batch):
        raise ProtocolError("worker token batch capacity is too small")


def capture_sequence(
    client: StageV3Client,
    tokens: Sequence[int],
    request_id: int,
    route_epoch: int,
) -> tuple[tuple[float, ...], ...]:
    hidden: list[tuple[float, ...]] = []
    for position, token in enumerate(tokens):
        results = client.batch([BatchRow(request_id, route_epoch, 0, position, token)])
        if len(results) != 1 or results[0].hidden is None or results[0].token is not None:
            raise ProtocolError("head correctness trial returned an invalid row")
        hidden.append(results[0].hidden)
    status = client.remove(0, request_id, route_epoch)
    if status.active_sequences != 0:
        raise ProtocolError("correctness trial retained sequence state")
    return tuple(hidden)


def compare_sequences(
    reference: Sequence[Sequence[float]], candidate: Sequence[Sequence[float]],
) -> list[dict[str, object]]:
    if not reference or len(reference) != len(candidate):
        raise ValueError("correctness sequences have different lengths")
    return [
        {"position": position, **compare_vectors(ref, value)}
        for position, (ref, value) in enumerate(zip(reference, candidate))
    ]


def numeric_pass(rows: Sequence[dict[str, object]], max_rel_l2: float, min_cosine: float) -> bool:
    return bool(rows) and all(
        type(row.get("rel_l2")) is float
        and type(row.get("cosine")) is float
        and row["rel_l2"] <= max_rel_l2
        and row["cosine"] >= min_cosine
        for row in rows
    )


def hidden_digest(results: Sequence[BatchResult]) -> str:
    digest = hashlib.sha256()
    for result in results:
        if result.hidden is None or result.token is not None:
            raise ProtocolError("phone stage returned a terminal result")
        for value in result.hidden:
            if not math.isfinite(value):
                raise ProtocolError("phone stage returned a non-finite activation")
        digest.update(struct.pack(f"<{len(result.hidden)}f", *result.hidden))
    return digest.hexdigest()


def run_cohort(
    client: StageV3Client,
    hello: Hello,
    batch: int,
    tokens: Sequence[int],
    identity: int,
) -> tuple[list[int], list[str], int]:
    request_ids = tuple(identity + offset for offset in range(batch))
    epochs = request_ids
    latencies: list[int] = []
    digests: list[str] = []
    hidden = None
    if hello.layer_start > 0:
        hidden = tuple(((index % 31) - 15) / 16.0 for index in range(hello.n_embd))
    for position, token in enumerate(tokens):
        rows = [
            BatchRow(
                request_ids[seq_id], epochs[seq_id], seq_id, position, token,
                hidden,
            )
            for seq_id in range(batch)
        ]
        started_ns = time.monotonic_ns()
        results = client.batch(rows)
        elapsed_us = (time.monotonic_ns() - started_ns) // 1000
        if len(results) != batch:
            raise ProtocolError("B32 result count differs from request count")
        latencies.append(elapsed_us)
        digests.append(hidden_digest(results))
    for seq_id, request_id in enumerate(request_ids):
        status = client.remove(seq_id, request_id, request_id)
    if status.active_sequences != 0:
        raise ProtocolError("B32 cohort retained sequence state")
    return latencies, digests, identity + batch


def parse_kib_fields(text: str) -> dict[str, int]:
    fields: dict[str, int] = {}
    for line in text.splitlines():
        name, separator, rest = line.partition(":")
        if not separator or name not in {"VmPeak", "VmSize", "VmHWM", "VmRSS", "VmSwap", "MemAvailable"}:
            continue
        parts = rest.split()
        if not parts or not parts[0].isdigit() or (len(parts) > 1 and parts[1] != "kB"):
            raise ValueError(f"invalid memory field: {line}")
        fields[name] = int(parts[0])
    return fields


def adb_memory_snapshot(serial: str, pid: int) -> dict[str, object]:
    if not SERIAL_RE.fullmatch(serial) or type(pid) is not int or pid <= 0:
        raise ValueError("invalid ADB identity")
    command = (
        f"cat /proc/{pid}/status; "
        "echo S32_MEMINFO; cat /proc/meminfo"
    )
    completed = subprocess.run(
        ["adb", "-s", serial, "shell", command],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )
    if completed.returncode != 0:
        raise RuntimeError("ADB memory snapshot failed")
    raw = completed.stdout.replace(b"\r", b"")
    if b"S32_MEMINFO\n" not in raw:
        raise RuntimeError("ADB memory snapshot is incomplete")
    process_raw, system_raw = raw.split(b"S32_MEMINFO\n", 1)
    try:
        process_text = process_raw.decode("ascii")
        system_text = system_raw.decode("ascii")
    except UnicodeError as exc:
        raise RuntimeError("ADB memory snapshot is not ASCII") from exc
    process = parse_kib_fields(process_text)
    system = parse_kib_fields(system_text)
    required = {"VmHWM", "VmRSS", "VmSwap"}
    if not required.issubset(process) or "MemAvailable" not in system:
        raise RuntimeError("ADB memory snapshot lacks required fields")
    return {
        "process_kib": process,
        "system_kib": system,
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
    }


def resource_failures(memory_after: dict[str, object]) -> list[str]:
    process = memory_after.get("process_kib")
    if not isinstance(process, dict) or type(process.get("VmSwap")) is not int:
        raise ValueError("memory snapshot lacks an integer VmSwap")
    failures: list[str] = []
    if process["VmSwap"] != 0:
        failures.append("PHONE_SWAP_NONZERO")
    return failures


def finish_client(client: StageV3Client, session_end: str) -> None:
    status = client.status()
    if status.active_sequences != 0:
        raise ProtocolError("worker has live sequences at shutdown")
    drained = client.drain()
    if drained.active_sequences != 0 or not drained.draining:
        raise ProtocolError("worker drain failed")
    if session_end == "stop":
        client.stop()
    else:
        client.detach()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phone", type=parse_endpoint, required=True)
    parser.add_argument("--reference", type=parse_endpoint)
    parser.add_argument("--expected-start", type=int, required=True)
    parser.add_argument("--expected-end", type=int, required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--steps", type=parse_tokens, default=(2, 532, 236772, 564))
    parser.add_argument("--correctness-tokens", type=parse_tokens, default=(2, 532, 236772, 564))
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-rel-l2", type=float, default=5e-3)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--quantization", choices=("Q8_0", "Q4_0"), required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--adb-serial", required=True)
    parser.add_argument("--phone-pid", type=int, required=True)
    parser.add_argument("--continue-on-numeric-fail", action="store_true")
    parser.add_argument("--session-end", choices=("stop", "detach"), default="stop")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not 0 <= args.expected_start < args.expected_end < 48
        or not 1 <= args.batch <= 64
        or args.warmups < 1
        or args.reps < 2
        or args.timeout <= 0
        or not re.fullmatch(r"[0-9a-f]{64}", args.model_sha256)
    ):
        parser.error("invalid probe configuration")

    phone = None
    reference = None
    diagnostics: dict[str, object] = {}
    report: dict[str, object]
    try:
        phone = StageV3Client.connect(*args.phone, args.timeout)
        phone_hello = phone.hello()
        validate_hello(phone_hello, args.expected_start, args.expected_end, args.batch, False)
        memory_before = adb_memory_snapshot(args.adb_serial, args.phone_pid)
        diagnostics = {
            "phone_hello": asdict(phone_hello),
            "memory_before": memory_before,
        }
        repeat: list[dict[str, object]] = []
        cross_backend: list[dict[str, object]] = []
        correctness_status = "NOT_RUN"
        correctness_pass = False
        reference_hello = None
        if args.reference is not None:
            if args.expected_start != 0:
                raise ValueError("CPU correctness comparison currently requires a head range")
            reference = StageV3Client.connect(*args.reference, args.timeout)
            reference_hello = reference.hello()
            validate_hello(reference_hello, args.expected_start, args.expected_end, 1, True)
            phone_a = capture_sequence(phone, args.correctness_tokens, 1001, 1)
            phone_b = capture_sequence(phone, args.correctness_tokens, 1002, 2)
            cpu = capture_sequence(reference, args.correctness_tokens, 1003, 3)
            repeat = compare_sequences(phone_a, phone_b)
            cross_backend = compare_sequences(cpu, phone_a)
            diagnostics.update({
                "reference_hello": asdict(reference_hello),
                "phone_repeat": repeat,
                "cpu_vs_phone": cross_backend,
            })
            if not all(row["byte_equal"] for row in repeat):
                raise ValueError("phone correctness trial is nondeterministic")
            correctness_pass = numeric_pass(
                cross_backend, args.max_rel_l2, args.min_cosine,
            )
            correctness_status = "PASS" if correctness_pass else "FAIL"
            if not correctness_pass and not args.continue_on_numeric_fail:
                raise ValueError("phone boundary differs from the CPU reference")

        identity = 10000
        for _ in range(args.warmups):
            _, _, identity = run_cohort(
                phone, phone_hello, args.batch, args.steps, identity,
            )
        measured: list[dict[str, object]] = []
        all_latencies: list[int] = []
        for rep in range(args.reps):
            latencies, digests, identity = run_cohort(
                phone, phone_hello, args.batch, args.steps, identity,
            )
            all_latencies.extend(latencies)
            measured.append({
                "rep": rep,
                "step_us": latencies,
                "activation_sha256": digests,
            })
        memory_after = adb_memory_snapshot(args.adb_serial, args.phone_pid)
        failed_resources = resource_failures(memory_after)
        if phone.status().active_sequences != 0:
            raise ProtocolError("worker retained sequence state after the probe")
        if reference is not None and reference.status().active_sequences != 0:
            raise ProtocolError("reference retained sequence state after the probe")

        report = {
            "schema": SCHEMA,
            "status": (
                "PROBE_RESOURCE_FAIL"
                if failed_resources
                else ("PROBE_PASS" if correctness_pass else "PROBE_PERF_ONLY")
            ),
            "scheduler_eligible": False,
            "resource_failures": failed_resources,
            "correctness_status": correctness_status,
            "quantization": args.quantization,
            "model_sha256": args.model_sha256,
            "layer_range": [args.expected_start, args.expected_end],
            "batch": args.batch,
            "steps": list(args.steps),
            "warmups_discarded": args.warmups,
            "reps": args.reps,
            "phone_hello": asdict(phone_hello),
            "reference_hello": None if reference_hello is None else asdict(reference_hello),
            "thresholds": {
                "max_rel_l2": args.max_rel_l2,
                "min_cosine": args.min_cosine,
            },
            "phone_repeat": repeat,
            "cpu_vs_phone": cross_backend,
            "b32_step_us": summarize(all_latencies),
            "b32_per_layer_median_us": statistics.median(all_latencies) / (
                args.expected_end - args.expected_start
            ),
            "measured_cohorts": measured,
            "memory_before": memory_before,
            "memory_after": memory_after,
        }
        finish_client(phone, args.session_end)
        if reference is not None:
            finish_client(reference, "stop")
        phone.close()
        if reference is not None:
            reference.close()
        phone = None
        reference = None
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_bytes(report))
        print(canonical_bytes(report).decode("ascii"), end="")
        if failed_resources:
            return 2
        return 0 if correctness_pass else 3
    except BaseException as exc:
        report = {
            "schema": SCHEMA,
            "status": "PROBE_FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "diagnostics": diagnostics,
        }
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(canonical_bytes(report))
        except OSError:
            pass
        print(canonical_bytes(report).decode("ascii"), end="", file=sys.stderr)
        return 2
    finally:
        for client in (phone, reference):
            if client is not None:
                try:
                    client.close()
                except OSError:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
