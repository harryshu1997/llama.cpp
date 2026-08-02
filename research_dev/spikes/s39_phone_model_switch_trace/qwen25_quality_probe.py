#!/usr/bin/env python3
"""Compare the direct two-phone Qwen2.5 route with a CUDA reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Sequence, TypeVar


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


SCHEMA = "s39-qwen25-route-quality-v1"
CORPUS_SCHEMA = "s33-wikitext-prompt-v1"
MANIFEST_SCHEMA = "s33-wikitext-corpus-manifest-v1"
MODEL_SHA256 = "924a4c39ef9fc6c139875ab6771c2e8172a3b40ffec5c720eca69ad7a0edfae7"
SOURCE_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
SOURCE_SHA256 = "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"
FILE_TYPE = 2
FILE_TYPE_LABELS = {
    2: "Q4_0",
    7: "Q8_0",
}
N_LAYER = 48
N_EMBD = 5120
PROMPTS = 128
BATCH = 32
PROMPT_TOKENS = 8
OUTPUT_TOKENS = 8
PREFILL_CHUNK = 2
THRESHOLDS = {
    "min_exact_sequence_agreement": 0.80,
    "min_first_token_agreement": 0.95,
    "min_token_decision_agreement": 0.95,
}

T = TypeVar("T")


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def phase_call(phase: str, operation: Callable[[], T]) -> T:
    try:
        return operation()
    except (OSError, ProtocolError) as exc:
        raise ProtocolError(f"{phase}: {exc}") from exc


def decode_json(raw: bytes, name: str) -> dict[str, object]:
    try:
        value = json.loads(raw, object_pairs_hook=strict_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid JSON") from exc
    if type(value) is not dict or canonical(value) != raw:
        raise ValueError(f"{name} is not canonical")
    return value


def load_corpus(
    corpus_path: Path,
    manifest_path: Path,
    expected_model_sha256: str = MODEL_SHA256,
) -> tuple[list[dict[str, object]], str, str]:
    corpus_raw = corpus_path.read_bytes()
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(corpus_raw.splitlines(), 1):
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
        tokens = record.get("tokens")
        if (
            record.get("schema") != CORPUS_SCHEMA
            or record.get("prompt_id") != index
            or type(tokens) is not list
            or len(tokens) != PROMPT_TOKENS
            or any(type(token) is not int or token < 0 for token in tokens)
            or re.fullmatch(r"[0-9a-f]{64}", str(record.get("text_sha256")))
            is None
        ):
            raise ValueError(f"invalid corpus record {index}")

    manifest_raw = manifest_path.read_bytes()
    manifest = decode_json(manifest_raw, "corpus manifest")
    corpus_digest = sha256(corpus_raw)
    selection = manifest.get("selection")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("source_revision") != SOURCE_REVISION
        or manifest.get("source_sha256") != SOURCE_SHA256
        or manifest.get("model_sha256") != expected_model_sha256
        or manifest.get("output_sha256") != corpus_digest
        or manifest.get("output_records") != PROMPTS
        or type(selection) is not dict
        or selection.get("prompt_tokens") != PROMPT_TOKENS
    ):
        raise ValueError("corpus manifest binding mismatch")
    return records, corpus_digest, sha256(manifest_raw)


def validate_hello(name: str, hello: Hello) -> None:
    if (
        hello.layer_start != 0
        or hello.layer_end != N_LAYER
        or hello.n_layer != N_LAYER
        or hello.n_embd != N_EMBD
        or not hello.capabilities & STAGE_V3_CAP_TERMINAL
        or hello.max_streams < BATCH
        or hello.n_ctx_seq < PROMPT_TOKENS + OUTPUT_TOKENS - 1
        or min(hello.n_batch, hello.n_ubatch) < BATCH * PREFILL_CHUNK
    ):
        raise ProtocolError(f"{name} topology or capacity mismatch")


def prefill_rows(
    prompts: Sequence[Sequence[int]],
    identity_base: int,
    position_start: int,
    position_end: int,
) -> list[BatchRow]:
    if len(prompts) != BATCH:
        raise ProtocolError("cohort must contain 32 prompts")
    if (
        position_start < 0
        or position_start >= position_end
        or position_end > PROMPT_TOKENS
        or position_end - position_start > PREFILL_CHUNK
    ):
        raise ProtocolError("prefill chunk is outside the frozen prompt")
    rows: list[BatchRow] = []
    for seq_id, tokens in enumerate(prompts):
        if len(tokens) != PROMPT_TOKENS:
            raise ProtocolError("prompt token count mismatch")
        for position in range(position_start, position_end):
            rows.append(BatchRow(
                identity_base + seq_id,
                identity_base + seq_id,
                seq_id,
                position,
                tokens[position],
            ))
    return rows


def output_tokens(
    results: Sequence[BatchResult],
    identity_base: int,
    expected_position: int,
) -> list[int]:
    if len(results) != BATCH:
        raise ProtocolError("terminal result count mismatch")
    tokens: list[int] = []
    for seq_id, result in enumerate(results):
        if (
            result.request_id != identity_base + seq_id
            or result.route_epoch != identity_base + seq_id
            or result.seq_id != seq_id
            or result.position != expected_position
            or result.hidden is not None
            or type(result.token) is not int
            or result.token < 0
        ):
            raise ProtocolError("terminal result lineage mismatch")
        tokens.append(result.token)
    return tokens


def prefill_tokens(
    results: Sequence[BatchResult],
    identity_base: int,
    chunk_tokens: int,
    expected_position: int,
) -> list[int]:
    if (
        chunk_tokens <= 0
        or chunk_tokens > PREFILL_CHUNK
        or len(results) != BATCH * chunk_tokens
    ):
        raise ProtocolError("prefill result count mismatch")
    selected = [
        results[(seq_id + 1) * chunk_tokens - 1]
        for seq_id in range(BATCH)
    ]
    return output_tokens(selected, identity_base, expected_position)


def timed_batch(
    client: StageV3Client,
    rows: Sequence[BatchRow],
) -> tuple[tuple[BatchResult, ...], int]:
    started_ns = time.monotonic_ns()
    results = client.batch(rows)
    return results, (time.monotonic_ns() - started_ns) // 1000


def run_cohort(
    client: StageV3Client,
    prompts: Sequence[Sequence[int]],
    identity_base: int,
) -> tuple[list[list[int]], int]:
    outputs = [[] for _ in range(BATCH)]
    elapsed_us = 0
    prediction: list[int] = []
    for position_start in range(0, PROMPT_TOKENS, PREFILL_CHUNK):
        position_end = min(position_start + PREFILL_CHUNK, PROMPT_TOKENS)
        results, batch_us = timed_batch(
            client,
            prefill_rows(
                prompts,
                identity_base,
                position_start,
                position_end,
            ),
        )
        elapsed_us += batch_us
        prediction = prefill_tokens(
            results,
            identity_base,
            position_end - position_start,
            position_end - 1,
        )
    for seq_id, token in enumerate(prediction):
        outputs[seq_id].append(token)

    for output_index in range(1, OUTPUT_TOKENS):
        position = PROMPT_TOKENS + output_index - 1
        rows = [
            BatchRow(
                identity_base + seq_id,
                identity_base + seq_id,
                seq_id,
                position,
                token,
            )
            for seq_id, token in enumerate(prediction)
        ]
        results, batch_us = timed_batch(client, rows)
        elapsed_us += batch_us
        prediction = output_tokens(results, identity_base, position)
        for seq_id, token in enumerate(prediction):
            outputs[seq_id].append(token)

    for seq_id in range(BATCH):
        status = client.remove(
            seq_id,
            identity_base + seq_id,
            identity_base + seq_id,
        )
    if status.active_sequences != 0:
        raise ProtocolError("cohort cleanup left live sequences")
    return outputs, elapsed_us


def quality_summary(
    physical: Sequence[Sequence[int]],
    reference: Sequence[Sequence[int]],
) -> dict[str, object]:
    prompt_count = len(physical)
    if prompt_count == 0 or len(reference) != prompt_count:
        raise ValueError("quality output count mismatch")
    if any(
        len(row) != OUTPUT_TOKENS
        for collection in (physical, reference)
        for row in collection
    ):
        raise ValueError("quality output width mismatch")
    first = sum(left[0] == right[0] for left, right in zip(physical, reference))
    exact = sum(left == right for left, right in zip(physical, reference))
    decisions = sum(
        left == right
        for left_row, right_row in zip(physical, reference)
        for left, right in zip(left_row, right_row)
    )
    return {
        "exact_sequence_agreement": exact / prompt_count,
        "exact_sequence_matches": exact,
        "first_token_agreement": first / prompt_count,
        "first_token_matches": first,
        "token_decision_agreement": decisions / (prompt_count * OUTPUT_TOKENS),
        "token_decision_matches": decisions,
    }


def finish(client: StageV3Client, session_end: str) -> None:
    status = client.status()
    if status.active_sequences != 0:
        raise ProtocolError("route has live state before shutdown")
    drained = client.drain()
    if not drained.draining or drained.active_sequences != 0:
        raise ProtocolError("route drain failed")
    if session_end == "detach":
        client.detach()
    else:
        client.stop()


def write_atomic(path: Path, value: dict[str, object]) -> None:
    raw = canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phone-route", type=parse_endpoint, required=True)
    parser.add_argument("--cuda-reference", type=parse_endpoint, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--model-sha256", default=MODEL_SHA256)
    parser.add_argument(
        "--file-type",
        type=int,
        choices=tuple(FILE_TYPE_LABELS),
        default=FILE_TYPE,
    )
    parser.add_argument(
        "--phone-session-end",
        choices=("detach", "stop"),
        default="stop",
    )
    parser.add_argument(
        "--phone-attention",
        choices=("explicit", "fused"),
        default="fused",
    )
    parser.add_argument(
        "--cohorts",
        type=int,
        choices=range(1, PROMPTS // BATCH + 1),
        default=PROMPTS // BATCH,
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.timeout <= 0
        or args.output.exists()
        or re.fullmatch(r"[0-9a-f]{64}", args.model_sha256) is None
    ):
        parser.error("invalid output or timeout")

    clients: dict[str, StageV3Client] = {}
    stopped: set[str] = set()
    try:
        corpus, corpus_digest, manifest_digest = load_corpus(
            args.corpus,
            args.corpus_manifest,
            args.model_sha256,
        )
        clients["phone"] = phase_call(
            "CONNECT_PHONE",
            lambda: StageV3Client.connect(
                *args.phone_route,
                args.timeout,
            ),
        )
        clients["cuda"] = phase_call(
            "CONNECT_CUDA",
            lambda: StageV3Client.connect(
                *args.cuda_reference,
                args.timeout,
            ),
        )
        hellos = {
            name: phase_call(
                f"HELLO_{name.upper()}",
                client.hello,
            )
            for name, client in clients.items()
        }
        for name, hello in hellos.items():
            validate_hello(name, hello)
        require_same_model(
            hellos,
            expected_model_sha256=args.model_sha256,
            expected_file_type=args.file_type,
        )

        physical: list[list[int]] = []
        reference: list[list[int]] = []
        timings: list[dict[str, int]] = []
        for cohort in range(args.cohorts):
            records = corpus[cohort * BATCH:(cohort + 1) * BATCH]
            prompts = [record["tokens"] for record in records]
            physical_tokens, physical_us = phase_call(
                f"PHONE_COHORT_{cohort}",
                lambda: run_cohort(
                    clients["phone"],
                    prompts,
                    10000 + cohort * 100,
                ),
            )
            reference_tokens, reference_us = phase_call(
                f"CUDA_COHORT_{cohort}",
                lambda: run_cohort(
                    clients["cuda"],
                    prompts,
                    20000 + cohort * 100,
                ),
            )
            physical.extend(physical_tokens)
            reference.extend(reference_tokens)
            timings.append({
                "cohort": cohort,
                "cuda_us": reference_us,
                "phone_us": physical_us,
            })

        quality = quality_summary(physical, reference)
        quality_pass = (
            quality["first_token_agreement"]
            >= THRESHOLDS["min_first_token_agreement"]
            and quality["token_decision_agreement"]
            >= THRESHOLDS["min_token_decision_agreement"]
            and quality["exact_sequence_agreement"]
            >= THRESHOLDS["min_exact_sequence_agreement"]
        )
        phase_call(
            "FINISH_PHONE",
            lambda: finish(clients["phone"], args.phone_session_end),
        )
        stopped.add("phone")
        phase_call("FINISH_CUDA", lambda: finish(clients["cuda"], "stop"))
        stopped.add("cuda")
        report: dict[str, object] = {
            "schema": SCHEMA,
            "status": "QUALITY_PASS" if quality_pass else "QUALITY_FAIL",
            "scheduler_eligible": False,
            "scope": (
                f"WIKITEXT_{args.cohorts * BATCH}_PROMPTS_"
                f"QWEN25_{FILE_TYPE_LABELS[args.file_type]}_GREEDY_AGREEMENT"
            ),
            "model_sha256": args.model_sha256,
            "file_type": args.file_type,
            "corpus_sha256": corpus_digest,
            "corpus_manifest_sha256": manifest_digest,
            "batch": BATCH,
            "cohorts": args.cohorts,
            "prompts": args.cohorts * BATCH,
            "prefill_chunk": PREFILL_CHUNK,
            "prompt_tokens": PROMPT_TOKENS,
            "output_tokens": OUTPUT_TOKENS,
            "quality": quality,
            "quality_gate_pass": quality_pass,
            "thresholds": THRESHOLDS,
            "physical_tokens": physical,
            "reference_tokens": reference,
            "physical_tokens_sha256": sha256(canonical(physical)),
            "reference_tokens_sha256": sha256(canonical(reference)),
            "timings": timings,
            "hellos": {name: asdict(hello) for name, hello in hellos.items()},
            "phone_session_end": args.phone_session_end.upper(),
            "phone_attention": args.phone_attention,
        }
        write_atomic(args.output, report)
        print(canonical(report).decode("ascii"), end="")
        return 0 if quality_pass else 3
    finally:
        for name, client in clients.items():
            if name not in stopped:
                try:
                    client.stop()
                except (OSError, ProtocolError):
                    pass
            try:
                client.close()
            except OSError:
                pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ProtocolError, ValueError) as exc:
        print(canonical({
            "error": str(exc),
            "status": "QUALITY_ERROR",
        }).decode("ascii"), end="")
        raise SystemExit(2)
