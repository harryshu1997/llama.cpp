#!/usr/bin/env python3
"""Run the frozen S33 natural-prompt quantized-route quality gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence


HERE = Path(__file__).resolve().parent
S22 = HERE.parent / "s22_slo_overlap_pipeline"
if str(S22) not in sys.path:
    sys.path.insert(0, str(S22))

from async_pipeline import parse_endpoint
from stage_v3_client import (
    BatchResult,
    BatchRow,
    Hello,
    ProtocolError,
    STAGE_V3_CAP_TERMINAL,
    StageV3Client,
    require_same_model,
)


SCHEMA = "s33-quantized-quality-probe-v1"
CORPUS_SCHEMA = "s33-wikitext-prompt-v1"
MODEL_CONFIGS = {
    "Q4_0": {
        "sha256": "494518c2262a26e2a607af0e40bca11c4de5a0b108e7c21308906dbbfb1c6f8c",
        "head_end": 4,
        "middle_end": 24,
    },
    "Q8_0": {
        "sha256": "7b56cbd0e0d96d5c8d7df9c21b39d67264e58cbabf7a96d4681eff8b3d492848",
        "head_end": 2,
        "middle_end": 16,
    },
}
BATCH = 32
PROMPTS = 128
PROMPT_TOKENS = 8
OUTPUT_TOKENS = 8
SERIAL_RE = re.compile(r"^[A-Za-z0-9._:-]+$")


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def parse_kib_fields(text: str) -> dict[str, int]:
    fields: dict[str, int] = {}
    for line in text.splitlines():
        name, separator, rest = line.partition(":")
        if not separator or name not in {
            "Pid", "VmPeak", "VmSize", "VmHWM", "VmRSS", "VmSwap", "MemAvailable",
        }:
            continue
        parts = rest.split()
        valid_units = len(parts) == 1 if name == "Pid" else len(parts) == 2 and parts[1] == "kB"
        if not parts or not parts[0].isdigit() or not valid_units:
            raise ValueError(f"invalid memory field: {line}")
        fields[name] = int(parts[0])
    return fields


def adb_memory_snapshot(serial: str, pid: int) -> dict[str, object]:
    if not SERIAL_RE.fullmatch(serial) or type(pid) is not int or pid <= 0:
        raise ValueError("invalid ADB identity")
    command = f"cat /proc/{pid}/status; echo S33_MEMINFO; cat /proc/meminfo"
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
    if b"S33_MEMINFO\n" not in raw:
        raise RuntimeError("ADB memory snapshot is incomplete")
    process_raw, system_raw = raw.split(b"S33_MEMINFO\n", 1)
    try:
        process = parse_kib_fields(process_raw.decode("ascii"))
        system = parse_kib_fields(system_raw.decode("ascii"))
    except UnicodeError as exc:
        raise RuntimeError("ADB memory snapshot is not ASCII") from exc
    if (
        not {"Pid", "VmHWM", "VmRSS", "VmSwap"}.issubset(process)
        or process["Pid"] != pid
        or "MemAvailable" not in system
    ):
        raise RuntimeError("ADB memory snapshot lacks required fields")
    return {
        "adb_serial": serial,
        "pid": pid,
        "process_kib": process,
        "system_kib": system,
        "raw_sha256": sha256(raw),
    }


def memory_failures(memory: dict[str, dict[str, object]]) -> list[str]:
    failures: list[str] = []
    for device in ("head", "middle"):
        identities: set[tuple[object, object]] = set()
        for phase in ("before", "after"):
            snapshot = memory.get(f"{device}_{phase}")
            process = snapshot.get("process_kib") if isinstance(snapshot, dict) else None
            swap = process.get("VmSwap") if isinstance(process, dict) else None
            if type(swap) is not int:
                raise ValueError("memory snapshot lacks an integer VmSwap")
            identities.add((snapshot.get("adb_serial"), snapshot.get("pid")))
            if swap != 0:
                failures.append(f"{device.upper()}_{phase.upper()}_SWAP_NONZERO")
        if len(identities) != 1:
            raise ValueError("memory snapshot identity changed during the run")
    return failures


def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_corpus(path: Path) -> tuple[list[dict[str, object]], str]:
    raw = path.read_bytes()
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        try:
            value = json.loads(line, object_pairs_hook=strict_object)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid corpus row {line_number}") from exc
        if type(value) is not dict or canonical(value).rstrip(b"\n") != line:
            raise ValueError(f"noncanonical corpus row {line_number}")
        records.append(value)
    if len(records) != PROMPTS:
        raise ValueError("corpus must contain exactly 128 rows")
    for index, record in enumerate(records):
        if record.get("schema") != CORPUS_SCHEMA or record.get("prompt_id") != index:
            raise ValueError("corpus identity mismatch")
        tokens = record.get("tokens")
        if type(tokens) is not list or len(tokens) != PROMPT_TOKENS:
            raise ValueError("corpus prompt length mismatch")
        if any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("corpus contains an invalid token")
        if re.fullmatch(r"[0-9a-f]{64}", str(record.get("text_sha256"))) is None:
            raise ValueError("corpus text digest is invalid")
    return records, sha256(raw)


def validate_topology(
    head: Hello, middle: Hello, tail: Hello, reference: Hello,
    head_end: int, middle_end: int,
) -> None:
    expected = ((head, 0, head_end, False), (middle, head_end, middle_end, False),
                (tail, middle_end, 48, True), (reference, 0, middle_end, False))
    for hello, start, end, terminal in expected:
        if (hello.layer_start, hello.layer_end) != (start, end):
            raise ProtocolError("worker range differs from the frozen Q4 route")
        if hello.n_layer != 48 or hello.n_embd != 3840:
            raise ProtocolError("worker model shape mismatch")
        if hello.max_streams < BATCH or min(hello.n_batch, hello.n_ubatch) < BATCH:
            raise ProtocolError("worker cannot execute B32")
        if bool(hello.capabilities & STAGE_V3_CAP_TERMINAL) != terminal:
            raise ProtocolError("worker terminal capability mismatch")


def token_rows(tokens: Sequence[int], position: int, identity_base: int) -> list[BatchRow]:
    if len(tokens) != BATCH:
        raise ProtocolError("token row count mismatch")
    return [
        BatchRow(identity_base + seq_id, identity_base + seq_id, seq_id, position, token)
        for seq_id, token in enumerate(tokens)
    ]


def hidden_rows(
    results: Sequence[BatchResult], tokens: Sequence[int], position: int,
    identity_base: int,
) -> list[BatchRow]:
    if len(results) != BATCH or len(tokens) != BATCH:
        raise ProtocolError("hidden row count mismatch")
    rows: list[BatchRow] = []
    for index, result in enumerate(results):
        if (
            result.request_id != identity_base + index
            or result.route_epoch != identity_base + index
            or result.seq_id != index
            or result.position != position
            or result.hidden is None
            or result.token is not None
            or len(result.hidden) != 3840
        ):
            raise ProtocolError("nonterminal result lineage or payload mismatch")
        rows.append(BatchRow(
            result.request_id, result.route_epoch, result.seq_id,
            position, tokens[index], result.hidden,
        ))
    return rows


def output_tokens(results: Sequence[BatchResult], identity_base: int) -> list[int]:
    if len(results) != BATCH:
        raise ProtocolError("terminal result count mismatch")
    tokens: list[int] = []
    for seq_id, result in enumerate(results):
        if (
            result.request_id != identity_base + seq_id
            or result.route_epoch != identity_base + seq_id
            or result.seq_id != seq_id
            or result.hidden is not None
            or type(result.token) is not int
            or result.token < 0
        ):
            raise ProtocolError("terminal result lineage or payload mismatch")
        tokens.append(result.token)
    return tokens


def timed_batch(client: StageV3Client, rows: Sequence[BatchRow]) -> tuple[tuple[BatchResult, ...], int]:
    started_ns = time.monotonic_ns()
    results = client.batch(rows)
    return results, (time.monotonic_ns() - started_ns) // 1000


def clear(client: StageV3Client, identity_base: int) -> None:
    status = None
    for seq_id in range(BATCH):
        status = client.remove(seq_id, identity_base + seq_id, identity_base + seq_id)
    if status is None or status.active_sequences != 0:
        raise ProtocolError("worker retained sequence state")


def run_physical(
    head: StageV3Client,
    middle: StageV3Client,
    tail: StageV3Client,
    prompts: Sequence[Sequence[int]],
    identity_base: int,
) -> tuple[list[list[int]], dict[str, int]]:
    outputs = [[] for _ in range(BATCH)]
    timing = {"head_us": 0, "middle_us": 0, "tail_us": 0}
    prediction: list[int] = []
    for position in range(PROMPT_TOKENS + OUTPUT_TOKENS - 1):
        current = [row[position] for row in prompts] if position < PROMPT_TOKENS else prediction
        head_result, elapsed = timed_batch(head, token_rows(current, position, identity_base))
        timing["head_us"] += elapsed
        middle_result, elapsed = timed_batch(
            middle, hidden_rows(head_result, current, position, identity_base),
        )
        timing["middle_us"] += elapsed
        tail_result, elapsed = timed_batch(
            tail, hidden_rows(middle_result, current, position, identity_base),
        )
        timing["tail_us"] += elapsed
        prediction = output_tokens(tail_result, identity_base)
        if position >= PROMPT_TOKENS - 1:
            for index, token in enumerate(prediction):
                outputs[index].append(token)
    if any(len(row) != OUTPUT_TOKENS for row in outputs):
        raise ProtocolError("physical output length mismatch")
    for client in (head, middle, tail):
        clear(client, identity_base)
    return outputs, timing


def run_reference(
    reference: StageV3Client,
    tail: StageV3Client,
    prompts: Sequence[Sequence[int]],
    identity_base: int,
) -> tuple[list[list[int]], dict[str, int]]:
    outputs = [[] for _ in range(BATCH)]
    timing = {"head_us": 0, "tail_us": 0}
    prediction: list[int] = []
    for position in range(PROMPT_TOKENS + OUTPUT_TOKENS - 1):
        current = [row[position] for row in prompts] if position < PROMPT_TOKENS else prediction
        hidden, elapsed = timed_batch(reference, token_rows(current, position, identity_base))
        timing["head_us"] += elapsed
        results, elapsed = timed_batch(
            tail, hidden_rows(hidden, current, position, identity_base),
        )
        timing["tail_us"] += elapsed
        prediction = output_tokens(results, identity_base)
        if position >= PROMPT_TOKENS - 1:
            for index, token in enumerate(prediction):
                outputs[index].append(token)
    if any(len(row) != OUTPUT_TOKENS for row in outputs):
        raise ProtocolError("reference output length mismatch")
    clear(reference, identity_base)
    clear(tail, identity_base)
    return outputs, timing


def quality_summary(
    physical: Sequence[Sequence[int]], reference: Sequence[Sequence[int]],
) -> dict[str, object]:
    if len(physical) != PROMPTS or len(reference) != PROMPTS:
        raise ValueError("quality output count mismatch")
    first = sum(left[0] == right[0] for left, right in zip(physical, reference))
    exact = sum(left == right for left, right in zip(physical, reference))
    decisions = sum(
        left == right
        for left_row, right_row in zip(physical, reference)
        for left, right in zip(left_row, right_row)
    )
    return {
        "first_token_matches": first,
        "first_token_agreement": first / PROMPTS,
        "token_decision_matches": decisions,
        "token_decision_agreement": decisions / (PROMPTS * OUTPUT_TOKENS),
        "exact_sequence_matches": exact,
        "exact_sequence_agreement": exact / PROMPTS,
    }


def finish(client: StageV3Client) -> None:
    status = client.status()
    if status.active_sequences != 0:
        raise ProtocolError("worker has live state before shutdown")
    drained = client.drain()
    if not drained.draining or drained.active_sequences != 0:
        raise ProtocolError("worker drain failed")
    client.stop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", type=parse_endpoint, required=True)
    parser.add_argument("--middle", type=parse_endpoint, required=True)
    parser.add_argument("--tail", type=parse_endpoint, required=True)
    parser.add_argument("--reference", type=parse_endpoint, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--quantization", choices=tuple(MODEL_CONFIGS), required=True)
    parser.add_argument("--head-adb-serial", required=True)
    parser.add_argument("--head-pid", type=int, required=True)
    parser.add_argument("--middle-adb-serial", required=True)
    parser.add_argument("--middle-pid", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.timeout <= 0
        or not SERIAL_RE.fullmatch(args.head_adb_serial)
        or not SERIAL_RE.fullmatch(args.middle_adb_serial)
        or args.head_pid <= 0
        or args.middle_pid <= 0
    ):
        parser.error("invalid quality-probe configuration")
    clients: dict[str, StageV3Client] = {}
    try:
        model_config = MODEL_CONFIGS[args.quantization]
        corpus, corpus_digest = load_corpus(args.corpus)
        manifest_raw = args.corpus_manifest.read_bytes()
        manifest = json.loads(manifest_raw, object_pairs_hook=strict_object)
        if (
            type(manifest) is not dict
            or canonical(manifest) != manifest_raw
            or manifest.get("schema") != "s33-wikitext-corpus-manifest-v1"
            or manifest.get("source_revision") != "b08601e04326c79dfdd32d625aee71d232d685c3"
            or manifest.get("source_sha256") != "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"
            or manifest.get("output_sha256") != corpus_digest
            or manifest.get("output_records") != PROMPTS
            or manifest.get("model_sha256") != model_config["sha256"]
        ):
            raise ValueError("corpus manifest binding mismatch")
        for name in ("head", "middle", "tail", "reference"):
            clients[name] = StageV3Client.connect(*getattr(args, name), args.timeout)
        hellos = {name: client.hello() for name, client in clients.items()}
        require_same_model(
            hellos,
            expected_model_sha256=model_config["sha256"],
            expected_file_type={"Q4_0": 2, "Q8_0": 7}[args.quantization],
        )
        validate_topology(
            hellos["head"], hellos["middle"], hellos["tail"], hellos["reference"],
            model_config["head_end"], model_config["middle_end"],
        )
        memory = {
            "head_before": adb_memory_snapshot(args.head_adb_serial, args.head_pid),
            "middle_before": adb_memory_snapshot(args.middle_adb_serial, args.middle_pid),
        }

        physical: list[list[int]] = []
        reference: list[list[int]] = []
        timings: list[dict[str, object]] = []
        for cohort in range(PROMPTS // BATCH):
            rows = corpus[cohort * BATCH:(cohort + 1) * BATCH]
            prompts = [row["tokens"] for row in rows]
            physical_tokens, physical_us = run_physical(
                clients["head"], clients["middle"], clients["tail"],
                prompts, 10000 + cohort * 100,
            )
            reference_tokens, reference_us = run_reference(
                clients["reference"], clients["tail"],
                prompts, 20000 + cohort * 100,
            )
            physical.extend(physical_tokens)
            reference.extend(reference_tokens)
            timings.append({
                "cohort": cohort,
                "physical_us": physical_us,
                "reference_us": reference_us,
            })

        quality = quality_summary(physical, reference)
        memory.update({
            "head_after": adb_memory_snapshot(args.head_adb_serial, args.head_pid),
            "middle_after": adb_memory_snapshot(args.middle_adb_serial, args.middle_pid),
        })
        resource_problems = memory_failures(memory)
        thresholds = {
            "min_first_token_agreement": 0.95,
            "min_token_decision_agreement": 0.95,
            "min_exact_sequence_agreement": 0.80,
        }
        quality_passed = (
            quality["first_token_agreement"] >= thresholds["min_first_token_agreement"]
            and quality["token_decision_agreement"] >= thresholds["min_token_decision_agreement"]
            and quality["exact_sequence_agreement"] >= thresholds["min_exact_sequence_agreement"]
        )
        passed = quality_passed and not resource_problems
        report = {
            "schema": SCHEMA,
            "status": (
                "QUALITY_PASS" if passed
                else "QUALITY_FAIL" if not quality_passed
                else "RESOURCE_FAIL"
            ),
            "scheduler_eligible": False,
            "scope": f"WIKITEXT_128_PROMPTS_{args.quantization}_GREEDY_AGREEMENT",
            "quantization": args.quantization,
            "model_sha256": model_config["sha256"],
            "corpus_sha256": corpus_digest,
            "corpus_manifest_sha256": sha256(manifest_raw),
            "route": [
                ["OP12", 0, model_config["head_end"]],
                ["OP15", model_config["head_end"], model_config["middle_end"]],
                ["CUDA", model_config["middle_end"], 48],
            ],
            "reference_route": [
                ["CUDA", 0, model_config["middle_end"]],
                ["CUDA", model_config["middle_end"], 48],
            ],
            "batch": BATCH,
            "cohorts": PROMPTS // BATCH,
            "prompt_tokens": PROMPT_TOKENS,
            "output_tokens": OUTPUT_TOKENS,
            "quality": quality,
            "quality_gate_pass": quality_passed,
            "resource_gate_pass": not resource_problems,
            "resource_failures": resource_problems,
            "memory": memory,
            "thresholds": thresholds,
            "physical_tokens": physical,
            "reference_tokens": reference,
            "physical_tokens_sha256": sha256(canonical(physical)),
            "reference_tokens_sha256": sha256(canonical(reference)),
            "timings": timings,
            "physical_cohort_median_us": statistics.median(
                sum(row["physical_us"].values()) for row in timings
            ),
            "reference_cohort_median_us": statistics.median(
                sum(row["reference_us"].values()) for row in timings
            ),
            "hellos": {name: asdict(hello) for name, hello in hellos.items()},
        }
        for client in clients.values():
            finish(client)
        for client in clients.values():
            client.close()
        clients.clear()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical(report))
        print(canonical(report).decode("ascii"), end="")
        return 0 if passed else 3
    except BaseException as exc:
        report = {
            "schema": SCHEMA,
            "status": "PROBE_FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(canonical(report))
        except OSError:
            pass
        print(canonical(report).decode("ascii"), end="", file=sys.stderr)
        return 2
    finally:
        for client in clients.values():
            try:
                client.close()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
