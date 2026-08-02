#!/usr/bin/python3 -I
"""Capture one independent B8 monolithic CUDA execution for V2.4."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import struct
import subprocess
import sys
import time
from typing import Any


sys.dont_write_bytecode = True

SCHEMA = "s39-cp0-r1-v24-cuda-monolithic-raw-v1"
CONFIRMATION = "RUN_V24_CUDA_MONOLITHIC_A_ONLY"
LAUNCH_SCHEMA = "s39-cp0-r1-v24-cuda-monolithic-launch-v1"
HISTORY_SCHEMA = "s39-cp0-r1-token-history-v2.4"
MODEL_ID = "qwen3-14b-q4_k_m"
PHASE = "A_ONLY"
DATASET = "cais/mmlu"
DATASET_REVISION = "bc5d09e5f0d160a95bcd36354bb5e16e50afe270"
CORPUS_SHA256 = "3ffafee1615ae2de690a2726b880823e167a3d9c210c5faed86d8f0e93ecff4f"
CANDIDATE_SHA256 = "ee3196ca660fa7eb7ea293260dc98dd6fdbf14571a4d4c5aece5343fb29b28d8"
REQUEST_IDS = list(range(1, 9))
CORPUS_ITEM_INDICES = list(range(8))
N_CTX_SEQ = 512
MAX_PROMPT_TOKENS = N_CTX_SEQ - 8
VOCAB_SIZE = 151936
PROMPT_FORMAT = (
    "Question: {question}\n"
    "A. {choice0}\n"
    "B. {choice1}\n"
    "C. {choice2}\n"
    "D. {choice3}\n"
    "Answer with exactly one uppercase letter: A, B, C, or D.\n"
    "Answer:"
)

STAGE_STOP = -1
STAGE_V3_HELLO = -8
STAGE_V3_BATCH = -9
STAGE_V3_SEQ_REMOVE = -10
STAGE_V3_STATUS = -11
STAGE_V3_IDENTITY = -13
STAGE_V3_MAGIC = 0x4C535633
STAGE_V3_VERSION = 3
STAGE_IDENTITY_MAGIC = 0x4C534944
STAGE_IDENTITY_VERSION = 1
STAGE_V3_REQUIRED_CAPABILITIES = 0x3F

MAX_INPUT_BYTES = 4 * 1024 * 1024
MAX_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_LOG_BYTES = 16 * 1024 * 1024
MAX_COMMAND_ITEM_BYTES = 128 * 1024
MAX_PREFILL_ROWS = 64
ALLOWED_SYSTEM_ROOTS = [
    "/mnt/storage/s21_deps/cuda-13.2.1/lib/",
    "/usr/lib/x86_64-linux-gnu/",
]
EXPECTED_MODEL_MAP_OFFSETS = (
    0x26645000,
    0x2188BD000,
)
EXPECTED_MODEL_MAP_PERMISSIONS = "r--s"


class CaptureError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CaptureError(message)


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        require(key not in result, f"E_DUPLICATE_KEY: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise CaptureError(f"E_JSON_NUMBER: {value}")


def canonical_bytes(value: Any) -> bytes:
    try:
        raw = (
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, UnicodeEncodeError) as error:
        raise CaptureError("E_CANONICAL") from error
    require(len(raw) <= MAX_OUTPUT_BYTES, "E_OUTPUT_SIZE")
    return raw


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def launch_bundle_digest(
    components: list[dict[str, Any]],
    launcher_component_id: str,
) -> str:
    identity = {
        "bundle_id": "cuda_monolithic",
        "components": components,
        "endpoint": "cuda",
        "launcher_component_id": launcher_component_id,
        "process_role": "cuda_monolithic",
        "schema": "s39-cp0-r1-runtime-bundle-root-identity-v2.4",
    }
    return sha256(canonical_bytes(identity))


def exact_keys(value: Any, keys: set[str], field: str) -> dict[str, Any]:
    require(type(value) is dict, f"E_TYPE: {field}")
    actual = set(value)
    require(actual == keys, f"E_KEYS: {field}")
    return value


def integer(value: Any, field: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f"E_INTEGER: {field}")
    return value


def text(value: Any, field: str, maximum: int = 4096) -> str:
    require(type(value) is str and 0 < len(value) <= maximum, f"E_TEXT: {field}")
    require(
        all(0x20 <= ord(character) <= 0x7E for character in value),
        f"E_ASCII: {field}",
    )
    return value


def digest(value: Any, field: str) -> str:
    value = text(value, field, 64)
    require(
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"E_DIGEST: {field}",
    )
    return value


def read_regular(path: Path, maximum: int = MAX_INPUT_BYTES) -> bytes:
    require(path.is_absolute(), f"E_ABSOLUTE_PATH: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CaptureError(f"E_OPEN: {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"E_FILE_TYPE: {path}")
        require(0 < before.st_size <= maximum, f"E_FILE_SIZE: {path}")
        raw = bytearray()
        while block := os.read(descriptor, min(1024 * 1024, maximum + 1)):
            raw.extend(block)
            require(len(raw) <= maximum, f"E_FILE_SIZE: {path}")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_mode,
    )
    require(identity(before) == identity(after), f"E_FILE_CHANGED: {path}")
    require(len(raw) == before.st_size, f"E_FILE_CHANGED: {path}")
    return bytes(raw)


def parse_canonical(raw: bytes, field: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CaptureError(f"E_JSON: {field}") from error
    require(type(value) is dict, f"E_TYPE: {field}")
    require(canonical_bytes(value) == raw, f"E_CANONICAL: {field}")
    return value


def read_canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = read_regular(path)
    value = parse_canonical(raw, str(path))
    return value, raw


def read_proc(path: Path, maximum: int) -> bytes:
    require(path.is_absolute(), f"E_PROC_PATH: {path}")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    except OSError as error:
        raise CaptureError(f"E_PROC_OPEN: {path}: {error}") from error
    try:
        result = bytearray()
        while block := os.read(descriptor, min(1024 * 1024, maximum + 1)):
            result.extend(block)
            require(len(result) <= maximum, f"E_PROC_SIZE: {path}")
    finally:
        os.close(descriptor)
    require(bool(result), f"E_PROC_EMPTY: {path}")
    return bytes(result)


def durable_write_new(path: Path, raw: bytes) -> None:
    require(path.is_absolute() and not path.exists(), "E_OUTPUT_PATH")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o644,
    )
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            require(written > 0, "E_OUTPUT_WRITE")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def monotonic_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def load_phase(pre_dir: Path, phase_id: str) -> tuple[dict[str, Any], bytes]:
    require(pre_dir.is_absolute() and pre_dir.is_dir(), "E_PRE_DIR")
    raw = read_regular(pre_dir / "phase_lock.jsonl")
    lines = raw.splitlines()
    require(len(lines) == 1, "E_PHASE_LOCK_ROWS")
    try:
        row = json.loads(
            lines[0].decode("ascii"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CaptureError("E_PHASE_LOCK_JSON") from error
    require(type(row) is dict, "E_PHASE_LOCK_TYPE")
    require(canonical_bytes(row).rstrip(b"\n") == lines[0], "E_PHASE_LOCK_CANONICAL")
    exact_keys(
        row,
        {
            "artifact_root_sha256",
            "candidate_sha256",
            "contract_sha256",
            "device_boot_ids",
            "event_ns",
            "model_id",
            "phase",
            "phase_id",
            "preparation_sha256",
            "quality_corpus_sha256",
            "runtime_bundle_plan_sha256",
            "schema",
        },
        "phase_lock",
    )
    require(
        row["schema"] == "s39-cp0-r1-phase-lock-v2.4",
        "E_PHASE_LOCK_SCHEMA",
    )
    require(row.get("phase") == PHASE, "E_PHASE_LOCK_PHASE")
    require(row.get("phase_id") == phase_id, "E_PHASE_LOCK_ID")
    require(row.get("model_id") == MODEL_ID, "E_PHASE_LOCK_MODEL")
    digest(row.get("quality_corpus_sha256"), "phase_lock.quality_corpus_sha256")
    return row, raw


def load_corpus(
    pre_dir: Path,
    phase_id: str,
    phase_lock: dict[str, Any],
) -> tuple[bytes, list[str], list[bytes]]:
    raw = read_regular(pre_dir / "quality_corpus.jsonl")
    require(sha256(raw) == phase_lock["quality_corpus_sha256"], "E_CORPUS_LOCK")
    lines = raw.splitlines()
    require(len(lines) == 64, "E_CORPUS_ROWS")
    prompt_sha256s = []
    prompts = []
    content_rows = []
    dynamic_keys = {
        "acquisition_id",
        "event_ns",
        "kind",
        "phase",
        "phase_id",
        "role",
    }
    for index, line in enumerate(lines):
        try:
            row = json.loads(
                line.decode("ascii"),
                object_pairs_hook=strict_object,
                parse_constant=reject_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CaptureError(f"E_CORPUS_JSON: {index}") from error
        require(type(row) is dict, f"E_CORPUS_TYPE: {index}")
        require(
            canonical_bytes(row).rstrip(b"\n") == line,
            f"E_CORPUS_CANONICAL: {index}",
        )
        require(row.get("phase") == PHASE, f"E_CORPUS_PHASE: {index}")
        require(row.get("phase_id") == phase_id, f"E_CORPUS_PHASE_ID: {index}")
        require(row.get("role") == "quality.corpus", f"E_CORPUS_ROLE: {index}")
        require(row.get("dataset") == DATASET, f"E_CORPUS_DATASET: {index}")
        require(
            row.get("dataset_revision") == DATASET_REVISION,
            f"E_CORPUS_REVISION: {index}",
        )
        require(row.get("item_index") == index, f"E_CORPUS_INDEX: {index}")
        choices = row.get("choices")
        require(
            type(choices) is list
            and len(choices) == 4
            and all(type(choice) is str for choice in choices),
            f"E_CORPUS_CHOICES: {index}",
        )
        question = row.get("question")
        require(type(question) is str and bool(question), f"E_CORPUS_QUESTION: {index}")
        prompt = PROMPT_FORMAT.format(
            question=question,
            choice0=choices[0],
            choice1=choices[1],
            choice2=choices[2],
            choice3=choices[3],
        )
        prompt_raw = prompt.encode("utf-8")
        prompt_sha256s.append(sha256(prompt_raw))
        prompts.append(prompt_raw)
        content_rows.append({
            key: value
            for key, value in row.items()
            if key not in dynamic_keys
        })
    content_raw = b"".join(canonical_bytes(row) for row in content_rows)
    require(sha256(content_raw) == CORPUS_SHA256, "E_CORPUS_CONTENT_SHA256")
    return raw, prompt_sha256s, prompts


def validate_history_group(
    group: Any,
    group_index: int,
    all_histories: list[list[int]],
) -> tuple[list[list[int]], dict[str, Any]]:
    field = f"histories.quality_groups[{group_index}]"
    exact_keys(
        group,
        {"decode_calls", "group_index", "item_indices", "prefill_partitions"},
        field,
    )
    item_indices = list(range(group_index * 8, group_index * 8 + 8))
    require(
        group["group_index"] == group_index
        and group["item_indices"] == item_indices,
        f"E_HISTORY_GROUP_IDENTITY: {group_index}",
    )
    histories = [all_histories[item_index] for item_index in item_indices]
    partitions = group["prefill_partitions"]
    require(
        type(partitions) is list and bool(partitions),
        f"E_PREFILL_PARTITIONS: {group_index}",
    )
    seen_rows = []
    position_partitions: dict[int, int] = {}
    for call_index, partition in enumerate(partitions):
        partition_field = f"{field}.prefill_partitions[{call_index}]"
        exact_keys(partition, {"call_index", "rows"}, partition_field)
        require(
            partition["call_index"] == call_index,
            f"E_PREFILL_CALL_INDEX: {group_index}:{call_index}",
        )
        rows = partition["rows"]
        require(
            type(rows) is list and 0 < len(rows) <= MAX_PREFILL_ROWS,
            f"E_PREFILL_ROWS: {group_index}:{call_index}",
        )
        for row_index, row in enumerate(rows):
            row_field = f"{partition_field}.rows[{row_index}]"
            exact_keys(
                row,
                {"item_index", "position", "request_id", "seq_id", "token_id"},
                row_field,
            )
            sequence = integer(row["seq_id"], f"{row_field}.seq_id")
            require(sequence < 8, f"E_PREFILL_SEQ: {row_field}")
            item_index = item_indices[sequence]
            require(
                row["item_index"] == item_index
                and row["request_id"] == REQUEST_IDS[sequence],
                f"E_PREFILL_IDENTITY: {row_field}",
            )
            position = integer(row["position"], f"{row_field}.position")
            require(
                position < len(histories[sequence])
                and row["token_id"] == histories[sequence][position],
                f"E_PREFILL_TOKEN: {row_field}",
            )
            previous_partition = position_partitions.setdefault(position, call_index)
            require(
                previous_partition == call_index,
                f"E_PREFILL_SPLIT_WAVE: {group_index}:{position}",
            )
            seen_rows.append(row)
    expected_rows = [
        {
            "item_index": item_indices[sequence],
            "position": position,
            "request_id": REQUEST_IDS[sequence],
            "seq_id": sequence,
            "token_id": histories[sequence][position],
        }
        for position in range(max(len(history) for history in histories))
        for sequence in range(8)
        if position < len(histories[sequence])
    ]
    require(
        seen_rows == expected_rows,
        f"E_PREFILL_ROW_ORDER: {group_index}",
    )

    decode_calls = group["decode_calls"]
    require(
        type(decode_calls) is list and len(decode_calls) == 7,
        f"E_DECODE_CALLS: {group_index}",
    )
    for decode_index, call in enumerate(decode_calls):
        call_field = f"{field}.decode_calls[{decode_index}]"
        exact_keys(
            call,
            {
                "call_index",
                "continuation_input_ordinal",
                "continuation_output_ordinal",
                "rows",
            },
            call_field,
        )
        require(
            call["call_index"] == len(partitions) + decode_index
            and call["continuation_input_ordinal"] == decode_index
            and call["continuation_output_ordinal"] == decode_index + 1,
            f"E_DECODE_CALL_INDEX: {group_index}:{decode_index}",
        )
        expected_decode_rows = [
            {
                "item_index": item_indices[sequence],
                "position": len(histories[sequence]) + decode_index,
                "request_id": REQUEST_IDS[sequence],
                "seq_id": sequence,
            }
            for sequence in range(8)
        ]
        require(
            call["rows"] == expected_decode_rows,
            f"E_DECODE_ROWS: {group_index}:{decode_index}",
        )
    return histories, group


def load_histories(
    path: Path,
    model_sha256: str,
    corpus_sha256: str,
    prompt_sha256s: list[str],
    prompts: list[bytes],
    expected_sha256: str | None = None,
) -> tuple[list[list[int]], dict[str, Any], bytes]:
    raw = read_regular(path)
    if expected_sha256 is not None:
        require(
            sha256(raw) == digest(expected_sha256, "histories_sha256"),
            "E_HISTORIES_SHA256",
        )
    value = parse_canonical(raw, "histories")
    exact_keys(
        value,
        {
            "batch",
            "candidate_sha256",
            "continuation_tokens_per_request",
            "corpus_sha256",
            "mechanics_b8",
            "model_id",
            "model_sha256",
            "n_batch",
            "n_ctx_seq",
            "n_ubatch",
            "prefill_chunking",
            "prefill_row_order",
            "quality_groups",
            "requests",
            "schema",
            "tokenizer",
        },
        "histories",
    )
    require(value["schema"] == HISTORY_SCHEMA, "E_HISTORY_SCHEMA")
    require(value["model_id"] == MODEL_ID, "E_HISTORY_MODEL")
    require(value["model_sha256"] == model_sha256, "E_HISTORY_MODEL_SHA256")
    require(value["batch"] == 8, "E_HISTORY_BATCH")
    require(value["continuation_tokens_per_request"] == 8, "E_HISTORY_CONTINUATIONS")
    require(value["candidate_sha256"] == CANDIDATE_SHA256, "E_HISTORY_CANDIDATE")
    require(value["corpus_sha256"] == corpus_sha256, "E_HISTORY_CORPUS_SHA256")
    require(
        value["n_batch"] == 64
        and value["n_ubatch"] == 64
        and value["n_ctx_seq"] == N_CTX_SEQ,
        "E_HISTORY_BATCH_LIMIT",
    )
    require(
        value["prefill_chunking"] == "WHOLE_POSITION_WAVES_MAX_64_ROWS"
        and value["prefill_row_order"] == "POSITION_MAJOR_THEN_ITEM_INDEX",
        "E_HISTORY_PREFILL_POLICY",
    )
    tokenizer = exact_keys(
        value["tokenizer"],
        {"component_id", "path", "plan_sha256", "sha256"},
        "histories.tokenizer",
    )
    text(tokenizer["component_id"], "histories.tokenizer.component_id", 256)
    tokenizer_path = text(tokenizer["path"], "histories.tokenizer.path")
    require(Path(tokenizer_path).is_absolute(), "E_HISTORY_TOKENIZER_PATH")
    digest(tokenizer["plan_sha256"], "histories.tokenizer.plan_sha256")
    digest(tokenizer["sha256"], "histories.tokenizer.sha256")

    requests = value["requests"]
    require(type(requests) is list and len(requests) == 64, "E_HISTORY_REQUESTS")
    all_histories = []
    for index, request in enumerate(requests):
        field = f"histories.requests[{index}]"
        exact_keys(
            request,
            {
                "item_index",
                "prompt_sha256",
                "prompt_utf8_base64",
                "prompt_utf8_bytes",
                "request_id",
                "seq_id",
                "token_ids",
            },
            field,
        )
        sequence = index % 8
        require(
            request["item_index"] == index
            and request["request_id"] == REQUEST_IDS[sequence]
            and request["seq_id"] == sequence,
            f"E_HISTORY_REQUEST_IDENTITY: {index}",
        )
        require(
            request["prompt_sha256"] == prompt_sha256s[index],
            f"E_HISTORY_PROMPT_SHA256: {index}",
        )
        try:
            prompt_raw = base64.b64decode(
                request["prompt_utf8_base64"].encode("ascii"),
                validate=True,
            )
        except (UnicodeEncodeError, ValueError) as error:
            raise CaptureError(f"E_HISTORY_PROMPT_BASE64: {index}") from error
        require(
            prompt_raw == prompts[index]
            and request["prompt_utf8_bytes"] == len(prompt_raw),
            f"E_HISTORY_PROMPT_BYTES: {index}",
        )
        history = request["token_ids"]
        require(
            type(history) is list
            and bool(history)
            and len(history) <= MAX_PROMPT_TOKENS
            and all(
                type(token) is int and 0 <= token < VOCAB_SIZE
                for token in history
            ),
            f"E_HISTORY_ROW: {index}",
        )
        all_histories.append(history)

    quality_groups = value["quality_groups"]
    require(
        type(quality_groups) is list and len(quality_groups) == 8,
        "E_HISTORY_QUALITY_GROUPS",
    )
    validated = [
        validate_history_group(group, group_index, all_histories)
        for group_index, group in enumerate(quality_groups)
    ]
    require(
        value["mechanics_b8"] == quality_groups[0],
        "E_HISTORY_MECHANICS_BINDING",
    )
    mechanics_histories, mechanics_plan = validated[0]
    return mechanics_histories, mechanics_plan, raw


def stat_record(path: Path) -> dict[str, int]:
    metadata = path.stat(follow_symlinks=False)
    require(stat.S_ISREG(metadata.st_mode), f"E_RUNTIME_FILE_TYPE: {path}")
    return {
        "ctime_ns": metadata.st_ctime_ns,
        "device_id": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
        "mtime_ns": metadata.st_mtime_ns,
        "size": metadata.st_size,
    }


def producer_identity(
    launch: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    del launch
    source_path = Path(__file__).resolve(strict=True)
    require(bool(sys.argv), "E_PRODUCER_ARGV")
    argv_source = Path(sys.argv[0]).resolve(strict=True)
    require(argv_source == source_path, "E_PRODUCER_ARGV_SOURCE")
    source_raw = read_regular(source_path)
    source_stat = stat_record(source_path)
    require(source_stat["size"] == len(source_raw), "E_PRODUCER_SIZE")
    source_sha256 = sha256(source_raw)
    boot_id = read_proc(
        Path("/proc/sys/kernel/random/boot_id"),
        256,
    ).decode("ascii").strip()
    text(boot_id, "producer.boot_id", 128)
    artifact = {
        "bytes": len(source_raw),
        "path": str(source_path),
        "sha256": source_sha256,
        "stat": source_stat,
    }
    receipt = {
        "argv": list(sys.argv),
        "boot_id": boot_id,
        "cwd": str(Path.cwd().resolve(strict=True)),
        "pid": os.getpid(),
        "schema": "s39-cp0-r1-v24-producer-process-receipt-v1",
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "start_ticks": process_start_ticks(os.getpid()),
    }
    validate_producer_identity(artifact, receipt)
    return artifact, receipt, source_raw


def validate_producer_identity(
    artifact: Any,
    receipt: Any,
) -> None:
    artifact = exact_keys(
        artifact,
        {"bytes", "path", "sha256", "stat"},
        "producer_artifact",
    )
    receipt = exact_keys(
        receipt,
        {
            "argv",
            "boot_id",
            "cwd",
            "pid",
            "schema",
            "source_path",
            "source_sha256",
            "start_ticks",
        },
        "producer_process_receipt",
    )
    require(
        receipt["schema"] == "s39-cp0-r1-v24-producer-process-receipt-v1",
        "E_PRODUCER_RECEIPT_SCHEMA",
    )
    source_path = Path(text(receipt["source_path"], "producer.source_path"))
    require(source_path.is_absolute(), "E_PRODUCER_SOURCE_PATH")
    require(artifact["path"] == str(source_path), "E_PRODUCER_ARTIFACT_PATH")
    source_raw = read_regular(source_path)
    require(
        integer(artifact["bytes"], "producer.bytes", 1) == len(source_raw),
        "E_PRODUCER_ARTIFACT_BYTES",
    )
    source_sha256 = digest(artifact["sha256"], "producer.sha256")
    require(source_sha256 == sha256(source_raw), "E_PRODUCER_SOURCE_MUTATED")
    require(
        receipt["source_sha256"] == source_sha256,
        "E_PRODUCER_RECEIPT_DIGEST",
    )
    require(
        stat_record(source_path) == artifact["stat"],
        "E_PRODUCER_SOURCE_STAT",
    )
    argv = receipt["argv"]
    require(
        type(argv) is list
        and bool(argv)
        and all(type(item) is str for item in argv),
        "E_PRODUCER_RECEIPT_ARGV",
    )
    require(
        str(Path(argv[0]).resolve(strict=True)) == str(source_path),
        "E_PRODUCER_RECEIPT_SOURCE",
    )
    cwd = Path(text(receipt["cwd"], "producer.cwd"))
    require(cwd.is_absolute() and cwd.is_dir(), "E_PRODUCER_RECEIPT_CWD")
    integer(receipt["pid"], "producer.pid", 1)
    integer(receipt["start_ticks"], "producer.start_ticks", 1)
    text(receipt["boot_id"], "producer.boot_id", 128)


def load_launch(
    path: Path,
    model_sha256: str,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    raw = read_regular(path)
    if expected_sha256 is not None:
        require(
            sha256(raw) == digest(expected_sha256, "launch_plan_sha256"),
            "E_LAUNCH_PLAN_SHA256",
        )
    value = parse_canonical(raw, "launch")
    exact_keys(
        value,
        {
            "allowed_system_roots",
            "bundle_id",
            "bundle_root",
            "bundle_sha256",
            "command",
            "cwd",
            "endpoint",
            "env",
            "expected_capabilities",
            "expected_file_type",
            "expected_max_streams",
            "expected_n_batch",
            "expected_n_ctx_seq",
            "expected_n_embd",
            "expected_n_layer",
            "expected_n_ubatch",
            "host",
            "io_timeout_ms",
            "launcher_component_id",
            "model_id",
            "model_artifact",
            "model_sha256",
            "port",
            "required_components",
            "route_epoch",
            "schema",
            "shutdown_timeout_ms",
            "startup_timeout_ms",
        },
        "launch",
    )
    require(value["schema"] == LAUNCH_SCHEMA, "E_LAUNCH_SCHEMA")
    require(value["model_id"] == MODEL_ID, "E_LAUNCH_MODEL")
    require(value["model_sha256"] == model_sha256, "E_LAUNCH_MODEL_SHA256")
    command = value["command"]
    require(
        type(command) is list
        and bool(command)
        and all(
            type(item) is str
            and 0 < len(item.encode("utf-8")) <= MAX_COMMAND_ITEM_BYTES
            for item in command
        ),
        "E_LAUNCH_COMMAND",
    )
    executable = Path(command[0])
    require(executable.is_absolute(), "E_LAUNCH_EXECUTABLE")
    cwd = Path(text(value["cwd"], "launch.cwd"))
    require(cwd.is_absolute() and cwd.is_dir(), "E_LAUNCH_CWD")
    bundle_root = Path(text(value["bundle_root"], "launch.bundle_root"))
    require(
        bundle_root.is_absolute()
        and bundle_root.is_dir()
        and not bundle_root.is_symlink(),
        "E_LAUNCH_BUNDLE_ROOT",
    )
    environment = value["env"]
    require(type(environment) is dict, "E_LAUNCH_ENV")
    for key, item in environment.items():
        text(key, "launch.env.key", 128)
        text(item, f"launch.env.{key}", 4096)
        require("=" not in key, f"E_LAUNCH_ENV_KEY: {key}")
    require(
        environment.get("LAYERSPLIT_PLACEMENT_CERT") == "1",
        "E_LAUNCH_PLACEMENT_CERT",
    )
    require(
        environment.get("LAYERSPLIT_MODEL_SHA256") == model_sha256,
        "E_LAUNCH_MODEL_SHA256_ENV",
    )
    require(
        environment
        == {
            "CUDA_VISIBLE_DEVICES": "0",
            "HOME": "/home/zhihao",
            "LAYERSPLIT_MEMORY_CERT": "1",
            "LAYERSPLIT_MODEL_SHA256": model_sha256,
            "LAYERSPLIT_PLACEMENT_CERT": "1",
            "LC_ALL": "C",
            "LD_LIBRARY_PATH": str(bundle_root),
        },
        "E_LAUNCH_ENV",
    )
    host = text(value["host"], "launch.host", 255)
    require(host in ("127.0.0.1", "::1"), "E_LAUNCH_HOST")
    port = integer(value["port"], "launch.port", 1)
    require(port <= 65535, "E_LAUNCH_PORT")
    expected = {
        "expected_capabilities": STAGE_V3_REQUIRED_CAPABILITIES,
        "expected_max_streams": 8,
        "expected_n_batch": 64,
        "expected_n_ctx_seq": N_CTX_SEQ,
        "expected_n_layer": 40,
        "expected_n_ubatch": 64,
    }
    for key, expected_value in expected.items():
        require(value[key] == expected_value, f"E_LAUNCH_{key.upper()}")
    integer(value["expected_file_type"], "launch.expected_file_type")
    integer(value["expected_n_embd"], "launch.expected_n_embd", 1)
    for name in ("io_timeout_ms", "shutdown_timeout_ms", "startup_timeout_ms"):
        duration = integer(value[name], f"launch.{name}", 1)
        require(duration <= 600_000, f"E_LAUNCH_TIMEOUT: {name}")
    require(value["bundle_id"] == "cuda_monolithic", "E_LAUNCH_BUNDLE")
    require(value["endpoint"] == "cuda", "E_LAUNCH_ENDPOINT")
    claimed_bundle_sha256 = digest(
        value["bundle_sha256"],
        "launch.bundle_sha256",
    )
    launcher_id = text(
        value["launcher_component_id"],
        "launch.launcher_component_id",
        256,
    )
    components = value["required_components"]
    require(type(components) is list and bool(components), "E_LAUNCH_COMPONENTS")
    component_ids = []
    launcher_path = None
    for index, component in enumerate(components):
        field = f"launch.required_components[{index}]"
        exact_keys(component, {"component_id", "path", "sha256", "stat"}, field)
        component_id = text(component["component_id"], f"{field}.component_id", 256)
        require(
            not component_ids or component_ids[-1] < component_id,
            "E_LAUNCH_COMPONENT_ORDER",
        )
        component_ids.append(component_id)
        component_path = Path(text(component["path"], f"{field}.path"))
        require(component_path.is_absolute(), f"E_LAUNCH_COMPONENT_PATH: {index}")
        try:
            component_path.relative_to(bundle_root)
        except ValueError as error:
            raise CaptureError(f"E_LAUNCH_COMPONENT_ROOT: {index}") from error
        digest(component["sha256"], f"{field}.sha256")
        require(
            component["stat"] == stat_record(component_path),
            f"E_LAUNCH_COMPONENT_STAT: {component_id}",
        )
        if component_id == launcher_id:
            launcher_path = str(component_path)
    require(launcher_path is not None, "E_LAUNCHER_COMPONENT")
    require(str(executable) == launcher_path, "E_LAUNCHER_COMMAND")
    require(
        claimed_bundle_sha256
        == launch_bundle_digest(components, launcher_id),
        "E_LAUNCH_BUNDLE_SHA256",
    )
    roots = value["allowed_system_roots"]
    require(
        type(roots) is list
        and roots == ALLOWED_SYSTEM_ROOTS,
        "E_LAUNCH_SYSTEM_ROOTS",
    )
    model_artifact = exact_keys(
        value["model_artifact"],
        {"path", "sha256", "stat"},
        "launch.model_artifact",
    )
    model_path = text(model_artifact["path"], "launch.model_artifact.path")
    require(Path(model_path).is_absolute(), "E_LAUNCH_MODEL_PATH")
    require(
        model_artifact["sha256"] == model_sha256,
        "E_LAUNCH_MODEL_ARTIFACT_SHA256",
    )
    require(
        model_artifact["stat"] == stat_record(Path(model_path)),
        "E_LAUNCH_MODEL_ARTIFACT_STAT",
    )
    integer(value["route_epoch"], "launch.route_epoch", 1)
    exact_options = {
        "--devices": "CUDA0",
        "--driver-batch": "8",
        "--driver-context": str(N_CTX_SEQ),
        "--driver-max-prefill": "8",
        "--mode": "monov3",
        "--port": str(port),
    }
    for option, expected_value in exact_options.items():
        require(command.count(option) == 1, f"E_LAUNCH_OPTION: {option}")
        index = command.index(option)
        require(
            index + 1 < len(command) and command[index + 1] == expected_value,
            f"E_LAUNCH_OPTION_VALUE: {option}",
        )
    require(command.count("-m") == 1, "E_LAUNCH_OPTION: -m")
    model_index = command.index("-m")
    require(
        model_index + 1 < len(command) and command[model_index + 1] == model_path,
        "E_LAUNCH_OPTION_VALUE: -m",
    )
    return value, raw


def process_start_ticks(pid: int) -> int:
    raw = read_proc(Path(f"/proc/{pid}/stat"), 64 * 1024)
    right = raw.rfind(b")")
    require(right > 0, "E_RUNTIME_PROC_STAT")
    fields = raw[right + 2:].split()
    require(len(fields) > 19, "E_RUNTIME_PROC_STAT")
    try:
        return integer(int(fields[19]), "runtime.start_ticks", 1)
    except ValueError as error:
        raise CaptureError("E_RUNTIME_PROC_STAT") from error


def system_dependency(path: Path) -> dict[str, Any]:
    return {
        "build_id": None,
        "path": str(path),
        **stat_record(path),
    }


def validate_model_mapping_rows(
    rows: list[dict[str, Any]],
    model_path: Path,
    model_stat: dict[str, int],
) -> None:
    require(len(rows) == len(EXPECTED_MODEL_MAP_OFFSETS), "E_RUNTIME_MODEL_MAP_COUNT")
    addresses = set()
    offsets = []
    for index, row in enumerate(rows):
        field = f"runtime.model_mapping_rows[{index}]"
        exact_keys(
            row,
            {
                "address_range",
                "device_major",
                "device_minor",
                "inode",
                "offset_bytes",
                "path",
                "permissions",
            },
            field,
        )
        address = text(row["address_range"], f"{field}.address_range", 64)
        try:
            start_text, end_text = address.split("-", 1)
            start = int(start_text, 16)
            end = int(end_text, 16)
        except ValueError as error:
            raise CaptureError(f"E_RUNTIME_MODEL_MAP_ADDRESS: {index}") from error
        require(
            0 < start < end and address not in addresses,
            f"E_RUNTIME_MODEL_MAP_ADDRESS: {index}",
        )
        addresses.add(address)
        offset = integer(row["offset_bytes"], f"{field}.offset_bytes")
        require(offset not in offsets, f"E_RUNTIME_MODEL_MAP_DUPLICATE: {offset}")
        offsets.append(offset)
        require(
            row["path"] == str(model_path)
            and row["permissions"] == EXPECTED_MODEL_MAP_PERMISSIONS
            and row["device_major"] == os.major(model_stat["device_id"])
            and row["device_minor"] == os.minor(model_stat["device_id"])
            and row["inode"] == model_stat["inode"],
            f"E_RUNTIME_MODEL_MAP_IDENTITY: {index}",
        )
    require(
        tuple(offsets) == EXPECTED_MODEL_MAP_OFFSETS,
        "E_RUNTIME_MODEL_MAP_OFFSETS",
    )


def capture_runtime_process(
    process: subprocess.Popen[bytes],
    launch: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(process.poll() is None, "E_RUNTIME_PROCESS_DEAD")
    pid = integer(process.pid, "runtime.pid", 1)
    start_ticks = process_start_ticks(pid)
    boot_id = read_proc(
        Path("/proc/sys/kernel/random/boot_id"),
        256,
    ).decode("ascii").strip()
    text(boot_id, "runtime.boot_id", 128)
    try:
        executable = str(Path(os.readlink(f"/proc/{pid}/exe")).resolve(strict=True))
    except OSError as error:
        raise CaptureError("E_RUNTIME_EXE") from error
    components = {
        str(Path(component["path"]).resolve(strict=True)): component
        for component in launch["required_components"]
    }
    require(
        executable
        == str(Path(launch["command"][0]).resolve(strict=True)),
        "E_RUNTIME_LAUNCHER",
    )
    command_raw = read_proc(Path(f"/proc/{pid}/cmdline"), MAX_INPUT_BYTES)
    try:
        command = [
            item.decode("utf-8")
            for item in command_raw.rstrip(b"\0").split(b"\0")
        ]
    except UnicodeDecodeError as error:
        raise CaptureError("E_RUNTIME_COMMAND") from error
    require(command == launch["command"], "E_RUNTIME_COMMAND")
    bundle_root = Path(launch["bundle_root"]).resolve(strict=True)
    model_path = Path(launch["model_artifact"]["path"]).resolve(strict=True)
    model_stat = stat_record(model_path)
    require(
        model_stat == launch["model_artifact"]["stat"],
        "E_RUNTIME_MODEL_STAT",
    )
    mapped = {executable}
    model_mappings = []
    other_gguf_mapping_paths = []
    maps_raw = read_proc(Path(f"/proc/{pid}/maps"), 16 * 1024 * 1024)
    for index, line in enumerate(maps_raw.decode("utf-8").splitlines()):
        fields = line.split(maxsplit=5)
        require(len(fields) >= 5, f"E_RUNTIME_MAPS: {index}")
        if len(fields) < 6:
            continue
        raw_path = fields[5]
        if raw_path.startswith("["):
            continue
        require(not raw_path.endswith(" (deleted)"), "E_RUNTIME_DELETED_MAP")
        if raw_path.startswith("/memfd:"):
            continue
        path = Path(raw_path)
        if not path.is_absolute() or not path.exists():
            continue
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode):
            continue
        resolved = path.resolve(strict=True)
        if resolved == model_path:
            try:
                device_major, device_minor = (
                    int(item, 16)
                    for item in fields[3].split(":", 1)
                )
                mapping_inode = int(fields[4])
                offset_bytes = int(fields[2], 16)
            except ValueError as error:
                raise CaptureError("E_RUNTIME_MODEL_MAP") from error
            require(
                device_major == os.major(metadata.st_dev)
                and device_minor == os.minor(metadata.st_dev)
                and mapping_inode == metadata.st_ino,
                "E_RUNTIME_MODEL_MAP_IDENTITY",
            )
            model_mappings.append({
                "address_range": fields[0],
                "device_major": device_major,
                "device_minor": device_minor,
                "inode": mapping_inode,
                "offset_bytes": offset_bytes,
                "path": str(model_path),
                "permissions": fields[1],
            })
        elif ".gguf" in resolved.name.lower():
            other_gguf_mapping_paths.append(str(resolved))
        if "x" in fields[1]:
            mapped.add(str(resolved))
    validate_model_mapping_rows(model_mappings, model_path, model_stat)
    require(
        not other_gguf_mapping_paths,
        f"E_RUNTIME_OTHER_MODEL_MAP: {other_gguf_mapping_paths}",
    )
    loaded_ids = []
    dependencies = {}
    for mapped_path in sorted(mapped):
        path = Path(mapped_path)
        try:
            path.relative_to(bundle_root)
            inside = True
        except ValueError:
            inside = False
        if inside:
            require(mapped_path in components, f"E_RUNTIME_UNPLANNED_COMPONENT: {mapped_path}")
            component = components[mapped_path]
            require(
                component["stat"] == stat_record(path),
                f"E_RUNTIME_COMPONENT_CHANGED: {component['component_id']}",
            )
            loaded_ids.append(component["component_id"])
            continue
        require(
            any(mapped_path.startswith(root) for root in launch["allowed_system_roots"]),
            f"E_RUNTIME_SYSTEM_PATH: {mapped_path}",
        )
        dependencies[mapped_path] = system_dependency(path)
    loaded_ids.sort()
    require(
        loaded_ids
        == [component["component_id"] for component in launch["required_components"]],
        "E_RUNTIME_COMPONENT_SET",
    )
    require(bool(dependencies), "E_RUNTIME_SYSTEM_DEPENDENCIES")
    runtime_process = {
        "boot_id": boot_id,
        "bundle_id": launch["bundle_id"],
        "bundle_sha256": launch["bundle_sha256"],
        "endpoint": launch["endpoint"],
        "launcher_component_id": launch["launcher_component_id"],
        "launcher_path": executable,
        "loaded_repo_component_ids": loaded_ids,
        "observed_ns": monotonic_ns(),
        "pid": pid,
        "start_ticks": start_ticks,
        "system_dependencies": [
            dependencies[path]
            for path in sorted(dependencies)
        ],
    }
    model_binding = {
        "argv": command,
        "model_mapping_rows": model_mappings,
        "model_path": str(model_path),
        "model_sha256": launch["model_artifact"]["sha256"],
        "model_stat": model_stat,
        "observed_ns": runtime_process["observed_ns"],
        "other_gguf_mapping_paths": other_gguf_mapping_paths,
        "pid": pid,
        "start_ticks": start_ticks,
    }
    return runtime_process, model_binding


def require_same_process(
    before: dict[str, Any],
    after: dict[str, Any],
) -> None:
    stable_keys = set(before) - {"observed_ns"}
    require(set(after) - {"observed_ns"} == stable_keys, "E_RUNTIME_RECHECK_KEYS")
    for key in sorted(stable_keys):
        require(before[key] == after[key], f"E_RUNTIME_RECHECK: {key}")
    require(before["observed_ns"] <= after["observed_ns"], "E_RUNTIME_RECHECK_TIME")


def require_same_model_binding(
    before: dict[str, Any],
    after: dict[str, Any],
) -> None:
    stable_keys = set(before) - {"observed_ns"}
    require(set(after) - {"observed_ns"} == stable_keys, "E_MODEL_RECHECK_KEYS")
    for key in sorted(stable_keys):
        require(before[key] == after[key], f"E_MODEL_RECHECK: {key}")
    require(before["observed_ns"] <= after["observed_ns"], "E_MODEL_RECHECK_TIME")


def parse_placement_certificate(
    log_raw: bytes,
    runtime_process: dict[str, Any],
) -> dict[str, Any]:
    prefix = b"PLACEMENTCERT "
    lines = [line[len(prefix):] for line in log_raw.splitlines() if line.startswith(prefix)]
    require(len(lines) == 1, "E_PLACEMENT_CERT_COUNT")
    try:
        value = json.loads(
            lines[0].decode("ascii"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CaptureError("E_PLACEMENT_CERT_JSON") from error
    exact_keys(
        value,
        {
            "compute_by_buffer_type",
            "compute_by_op",
            "compute_by_op_and_buffer",
            "compute_nodes",
            "copy_by_buffer_type",
            "copy_nodes",
            "layer_end",
            "layer_start",
            "metadata_nodes",
            "missing_buffer_compute_nodes",
            "mode",
            "n_layer",
            "pid",
            "role",
            "run_rc",
            "schema",
            "status",
        },
        "placement_certificate",
    )
    require(
        value.get("schema") == "layersplit-scheduled-placement-v2"
        and value.get("role") == "monov3"
        and value.get("mode") == "monov3"
        and value.get("layer_start") == 0
        and value.get("layer_end") == 40
        and value.get("n_layer") == 40
        and value.get("pid") == runtime_process["pid"]
        and value.get("run_rc") == 0
        and value.get("status") == "SCHEDULED_PLACEMENT_OK",
        "E_PLACEMENT_CERT_IDENTITY",
    )
    compute_nodes = integer(
        value["compute_nodes"],
        "placement_certificate.compute_nodes",
        1,
    )
    require(
        integer(
            value["missing_buffer_compute_nodes"],
            "placement_certificate.missing_buffer_compute_nodes",
        ) == 0,
        "E_PLACEMENT_CERT_COMPUTE",
    )
    integer(value["metadata_nodes"], "placement_certificate.metadata_nodes")
    copy_nodes = integer(
        value["copy_nodes"],
        "placement_certificate.copy_nodes",
    )
    copy_buffers = value["copy_by_buffer_type"]
    require(
        type(copy_buffers) is dict
        and all(
            type(key) is str
            and type(count) is int
            and count > 0
            for key, count in copy_buffers.items()
        )
        and sum(copy_buffers.values()) == copy_nodes
        and copy_nodes == 0,
        "E_PLACEMENT_CERT_COPY",
    )
    buffers = value.get("compute_by_buffer_type")
    require(
        type(buffers) is dict
        and bool(buffers)
        and set(buffers) <= {"CUDA0", "CUDA_Host"}
        and all(type(count) is int and count > 0 for count in buffers.values())
        and type(buffers.get("CUDA0")) is int
        and buffers["CUDA0"] > 0
        and sum(buffers.values()) == compute_nodes,
        "E_PLACEMENT_CERT_BACKEND",
    )
    by_op = value["compute_by_op"]
    nested = value["compute_by_op_and_buffer"]
    require(
        type(by_op) is dict
        and bool(by_op)
        and type(nested) is dict
        and set(nested) == set(by_op)
        and all(
            type(operation) is str
            and bool(operation)
            and operation.isascii()
            and type(count) is int
            and count > 0
            for operation, count in by_op.items()
        )
        and sum(by_op.values()) == compute_nodes,
        "E_PLACEMENT_CERT_OPS",
    )
    derived_buffers: dict[str, int] = {}
    for operation, count in by_op.items():
        operation_buffers = nested[operation]
        require(
            type(operation_buffers) is dict
            and bool(operation_buffers)
            and all(
                backend in {"CUDA0", "CUDA_Host"}
                and type(backend_count) is int
                and backend_count > 0
                for backend, backend_count in operation_buffers.items()
            )
            and sum(operation_buffers.values()) == count,
            f"E_PLACEMENT_CERT_OP_TOTAL: {operation}",
        )
        if operation == "GET_ROWS":
            require(
                set(operation_buffers) <= {"CUDA0", "CUDA_Host"},
                "E_PLACEMENT_CERT_GET_ROWS",
            )
        else:
            require(
                set(operation_buffers) == {"CUDA0"},
                f"E_PLACEMENT_CERT_HOST_OP: {operation}",
            )
        for backend, backend_count in operation_buffers.items():
            derived_buffers[backend] = (
                derived_buffers.get(backend, 0) + backend_count
            )
    require(derived_buffers == buffers, "E_PLACEMENT_CERT_BUFFER_TOTAL")
    return value


def parse_memory_certificate(
    log_raw: bytes,
    runtime_process: dict[str, Any],
) -> dict[str, Any]:
    prefix = b"MEMORYCERT "
    lines = [
        line[len(prefix):]
        for line in log_raw.splitlines()
        if line.startswith(prefix)
    ]
    require(len(lines) == 1, "E_MEMORY_CERT_COUNT")
    try:
        value = json.loads(
            lines[0].decode("ascii"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CaptureError("E_MEMORY_CERT_JSON") from error
    exact_keys(
        value,
        {
            "compute_buffer_bytes",
            "host_compute_buffer_bytes",
            "host_context_buffer_bytes",
            "host_model_buffer_bytes",
            "kv_buffer_bytes",
            "model_buffer_bytes",
            "pid",
            "role",
            "schema",
        },
        "memory_certificate",
    )
    require(
        value["schema"] == "layersplit-memory-breakdown-v1"
        and value["role"] == "monov3"
        and value["pid"] == runtime_process["pid"],
        "E_MEMORY_CERT_IDENTITY",
    )
    for key in (
        "compute_buffer_bytes",
        "host_compute_buffer_bytes",
        "host_context_buffer_bytes",
        "host_model_buffer_bytes",
        "kv_buffer_bytes",
        "model_buffer_bytes",
    ):
        integer(value[key], f"memory_certificate.{key}")
    require(
        value["model_buffer_bytes"] > 0
        and value["kv_buffer_bytes"] > 0,
        "E_MEMORY_CERT_DEVICE_BYTES",
    )
    return value


def pack_i32(values) -> bytes:
    values = tuple(values)
    return struct.pack(f"<{len(values)}i", *values)


def pack_i64(values) -> bytes:
    values = tuple(values)
    return struct.pack(f"<{len(values)}q", *values)


class StageClient:
    def __init__(self, connection: socket.socket):
        self.connection = connection
        self.n_batch = 0
        self.n_ubatch = 0

    def recv_exact(self, size: int) -> bytes:
        require(size >= 0, "E_RECV_SIZE")
        result = bytearray()
        while len(result) < size:
            block = self.connection.recv(size - len(result))
            require(bool(block), "E_UNEXPECTED_EOF")
            result.extend(block)
        return bytes(result)

    def recv_i32(self, count: int) -> tuple[int, ...]:
        return struct.unpack(f"<{count}i", self.recv_exact(count * 4))

    def hello(
        self,
        launch: dict[str, Any],
        model_sha256: str,
    ) -> dict[str, Any]:
        self.connection.sendall(pack_i32([STAGE_V3_HELLO]))
        words = self.recv_i32(11)
        require(words[0] == STAGE_V3_MAGIC, "E_HELLO_MAGIC")
        require(words[1] == STAGE_V3_VERSION, "E_HELLO_VERSION")
        expected = (
            0,
            launch["expected_n_layer"],
            launch["expected_n_layer"],
            launch["expected_n_embd"],
            launch["expected_max_streams"],
            launch["expected_n_ctx_seq"],
            launch["expected_n_batch"],
            launch["expected_n_ubatch"],
            launch["expected_capabilities"],
        )
        require(words[2:] == expected, "E_HELLO_GEOMETRY")
        self.n_batch = words[8]
        self.n_ubatch = words[9]
        self.connection.sendall(pack_i32([STAGE_V3_IDENTITY]))
        magic, version, file_type = self.recv_i32(3)
        model_digest = self.recv_exact(32).hex()
        require(
            magic == STAGE_IDENTITY_MAGIC
            and version == STAGE_IDENTITY_VERSION
            and file_type == launch["expected_file_type"]
            and model_digest == model_sha256,
            "E_MODEL_IDENTITY",
        )
        return {
            "capabilities": words[10],
            "file_type": file_type,
            "layer_end": words[3],
            "layer_start": words[2],
            "max_streams": words[6],
            "model_sha256": model_digest,
            "n_batch": words[8],
            "n_ctx_seq": words[7],
            "n_embd": words[5],
            "n_layer": words[4],
            "n_ubatch": words[9],
            "schema": "layersplit-stage-v3-identity-v1",
            "stage_identity_version": version,
            "stage_protocol_version": words[1],
        }

    def status(self) -> tuple[int, int, bool]:
        self.connection.sendall(pack_i32([STAGE_V3_STATUS, STAGE_V3_VERSION]))
        code, version, active, maximum, draining = self.recv_i32(5)
        require(code == 0 and version == STAGE_V3_VERSION, "E_STATUS")
        require(
            0 <= active <= maximum and maximum >= 8 and draining in (0, 1),
            "E_STATUS_VALUE",
        )
        return active, maximum, bool(draining)

    def batch(
        self,
        rows: list[tuple[int, int, int, int, int]],
    ) -> list[int]:
        require(
            bool(rows)
            and len(rows) <= MAX_PREFILL_ROWS
            and len(rows) <= min(self.n_batch, self.n_ubatch),
            "E_BATCH_SIZE",
        )
        require(all(row[0] > 0 and row[1] > 0 for row in rows), "E_BATCH_IDENTITY")
        payload = pack_i32([STAGE_V3_BATCH, STAGE_V3_VERSION, len(rows), 0])
        payload += pack_i64(row[0] for row in rows)
        payload += pack_i64(row[1] for row in rows)
        payload += pack_i32(row[2] for row in rows)
        payload += pack_i32(row[3] for row in rows)
        payload += pack_i32(row[4] for row in rows)
        self.connection.sendall(payload)
        require(self.recv_i32(1)[0] == 0, "E_BATCH_STATUS")
        count, width = self.recv_i32(2)
        require(count == len(rows) and width == 0, "E_BATCH_RESPONSE_SHAPE")
        request_ids = struct.unpack(
            f"<{count}q",
            self.recv_exact(8 * count),
        )
        route_epochs = struct.unpack(
            f"<{count}q",
            self.recv_exact(8 * count),
        )
        seq_ids = struct.unpack(
            f"<{count}i",
            self.recv_exact(4 * count),
        )
        positions = struct.unpack(
            f"<{count}i",
            self.recv_exact(4 * count),
        )
        tokens = self.recv_i32(count)
        require(
            request_ids == tuple(row[0] for row in rows)
            and route_epochs == tuple(row[1] for row in rows)
            and seq_ids == tuple(row[2] for row in rows)
            and positions == tuple(row[3] for row in rows),
            "E_BATCH_LINEAGE",
        )
        require(all(token >= 0 for token in tokens), "E_BATCH_TOKEN")
        return list(tokens)

    def remove(
        self,
        request_id: int,
        route_epoch: int,
        seq_id: int,
    ) -> None:
        payload = pack_i32([STAGE_V3_SEQ_REMOVE, STAGE_V3_VERSION, seq_id])
        payload += pack_i64([request_id, route_epoch])
        self.connection.sendall(payload)
        code, version, active, maximum, draining = self.recv_i32(5)
        require(
            code == 0
            and version == STAGE_V3_VERSION
            and 0 <= active <= maximum
            and draining in (0, 1),
            "E_REMOVE",
        )

    def stop(self) -> None:
        self.connection.sendall(pack_i32([STAGE_STOP]))


def connect(
    host: str,
    port: int,
    startup_timeout_ms: int,
    io_timeout_ms: int,
    process: subprocess.Popen[bytes],
) -> StageClient:
    deadline = time.monotonic() + startup_timeout_ms / 1000
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        require(process.poll() is None, "E_WORKER_EARLY_EXIT")
        try:
            connection = socket.create_connection((host, port), timeout=0.2)
            connection.settimeout(io_timeout_ms / 1000)
            return StageClient(connection)
        except OSError as error:
            last_error = error
            time.sleep(0.02)
    raise CaptureError(f"E_WORKER_CONNECT: {last_error}")


def call_shape(
    call_index: int,
    phase: str,
    rows: list[tuple[int, int, int, int, int]],
) -> dict[str, Any]:
    return {
        "call_index": call_index,
        "n_seqs": len({row[2] for row in rows}),
        "n_tokens": len(rows),
        "phase": phase,
        "positions": [row[3] for row in rows],
        "request_ids": [row[0] for row in rows],
        "seq_ids": [row[2] for row in rows],
    }


def execute(
    launch: dict[str, Any],
    histories: list[list[int]],
    history_plan: dict[str, Any],
    model_sha256: str,
    log_path: Path,
) -> tuple[
    list[list[int]],
    list[dict[str, Any]],
    int,
    int,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    log_file = None
    process: subprocess.Popen[bytes] | None = None
    connection: socket.socket | None = None
    try:
        require(not log_path.exists(), "E_LOG_EXISTS")
        log_file = log_path.open("xb")
        process = subprocess.Popen(
            launch["command"],
            cwd=launch["cwd"],
            env=launch["env"],
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        client = connect(
            launch["host"],
            launch["port"],
            launch["startup_timeout_ms"],
            launch["io_timeout_ms"],
            process,
        )
        connection = client.connection
        protocol_identity = client.hello(launch, model_sha256)
        state_before = client.status()[0]
        require(state_before == 0, "E_STATE_BEFORE")
        runtime_process, runtime_model_binding = capture_runtime_process(
            process,
            launch,
        )

        predictions: list[int | None] = [None] * 8
        calls = []
        route_epoch = launch["route_epoch"]
        for partition in history_plan["prefill_partitions"]:
            chunk = [
                (
                    row["request_id"],
                    route_epoch,
                    row["seq_id"],
                    row["position"],
                    row["token_id"],
                )
                for row in partition["rows"]
            ]
            tokens = client.batch(chunk)
            require(partition["call_index"] == len(calls), "E_PREFILL_RUNTIME_CALL")
            calls.append(call_shape(partition["call_index"], "prefill", chunk))
            for row, token in zip(chunk, tokens):
                sequence = row[2]
                if row[3] == len(histories[sequence]) - 1:
                    require(predictions[sequence] is None, "E_PREFILL_FINAL_REUSE")
                    predictions[sequence] = token
        require(all(token is not None for token in predictions), "E_PREFILL_FINAL_MISSING")
        current = [int(token) for token in predictions]
        outputs = [[token] for token in current]
        for decode_index, decode_call in enumerate(history_plan["decode_calls"]):
            decode_rows = [
                (
                    row["request_id"],
                    route_epoch,
                    row["seq_id"],
                    row["position"],
                    current[row["seq_id"]],
                )
                for row in decode_call["rows"]
            ]
            require(
                decode_call["call_index"] == len(calls)
                and decode_call["continuation_input_ordinal"] == decode_index,
                "E_DECODE_RUNTIME_CALL",
            )
            response = client.batch(decode_rows)
            current = [0] * 8
            for row, token in zip(decode_rows, response):
                sequence = row[2]
                current[sequence] = token
                outputs[sequence].append(token)
            calls.append(call_shape(decode_call["call_index"], "decode", decode_rows))
        require(all(len(row) == 8 for row in outputs), "E_CONTINUATION_COUNT")
        for sequence, request_id in enumerate(REQUEST_IDS):
            client.remove(request_id, route_epoch, sequence)
        state_after = client.status()[0]
        require(state_after == 0, "E_STATE_AFTER")
        runtime_recheck, model_recheck = capture_runtime_process(process, launch)
        require_same_process(runtime_process, runtime_recheck)
        require_same_model_binding(runtime_model_binding, model_recheck)
        client.stop()
        connection.close()
        connection = None
        try:
            return_code = process.wait(
                timeout=launch["shutdown_timeout_ms"] / 1000
            )
        except subprocess.TimeoutExpired as error:
            raise CaptureError("E_WORKER_STOP_TIMEOUT") from error
        require(return_code == 0, f"E_WORKER_EXIT: {return_code}")
        process = None
        return (
            outputs,
            calls,
            state_before,
            state_after,
            runtime_process,
            runtime_model_binding,
            protocol_identity,
        )
    finally:
        if connection is not None:
            connection.close()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if log_file is not None:
            log_file.flush()
            os.fsync(log_file.fileno())
            log_file.close()
        if log_path.exists():
            metadata = log_path.stat(follow_symlinks=False)
            require(
                stat.S_ISREG(metadata.st_mode)
                and metadata.st_size <= MAX_LOG_BYTES,
                "E_LOG_FILE",
            )


def make_rows(
    histories: list[list[int]],
    continuations: list[list[int]],
    calls: list[dict[str, Any]],
    model_sha256: str,
    program_sha256: str,
    state_before: int,
    state_after: int,
    event_ns: int,
) -> list[dict[str, Any]]:
    rows = [{
        "backend": "CUDA0",
        "call_shapes": calls,
        "event_ns": event_ns,
        "kind": "meta",
        "model_id": MODEL_ID,
        "model_sha256": model_sha256,
        "program_sha256": program_sha256,
        "state_count_after": state_after,
        "state_count_before": state_before,
    }]
    for sequence, request_id in enumerate(REQUEST_IDS):
        rows.append({
            "continuation_tokens": continuations[sequence],
            "event_ns": event_ns,
            "input_tokens": histories[sequence],
            "kind": "request",
            "model_id": MODEL_ID,
            "model_sha256": model_sha256,
            "owner_after": "RELEASED",
            "owner_before": "CUDA",
            "ownership_epoch_after": 2,
            "ownership_epoch_before": 1,
            "positions": list(range(len(histories[sequence]))),
            "request_id": request_id,
        })
    return rows


def capture(args: argparse.Namespace) -> dict[str, Any]:
    started_ns = monotonic_ns()
    acquisition_started_ns = integer(args.started, "started", 1)
    require(acquisition_started_ns <= started_ns, "E_ACQUISITION_ORDER")
    phase_id = text(args.phase_id, "phase_id", 128)
    require(
        phase_id.startswith("cp0-r1-v24-a-only-")
        and all(character.isalnum() or character in ".-_" for character in phase_id),
        "E_PHASE_ID",
    )
    command_plan_sha256 = digest(args.plan, "plan")
    mechanism_sha256 = digest(
        args.mechanism_commands_sha256,
        "mechanism_commands_sha256",
    )
    model_sha256 = digest(args.model_sha256, "model_sha256")
    output = Path(args.output)
    require(output.is_absolute() and not output.exists(), "E_OUTPUT")
    pre_dir = Path(args.pre_dir)
    phase_lock, phase_lock_raw = load_phase(pre_dir, phase_id)
    corpus_raw, prompt_sha256s, prompts = load_corpus(
        pre_dir,
        phase_id,
        phase_lock,
    )
    histories, history_plan, history_raw = load_histories(
        Path(args.histories),
        model_sha256,
        CORPUS_SHA256,
        prompt_sha256s,
        prompts,
        args.histories_sha256,
    )
    launch, launch_raw = load_launch(
        Path(args.launch_plan),
        model_sha256,
        args.launch_plan_sha256,
    )
    (
        producer_artifact,
        producer_process_receipt,
        source_raw,
    ) = producer_identity(launch)
    program_sha256 = sha256(
        b"s39:v24:cuda-monolithic-program:v1\0"
        + bytes.fromhex(command_plan_sha256)
        + bytes.fromhex(sha256(source_raw))
        + bytes.fromhex(sha256(launch_raw))
        + bytes.fromhex(sha256(history_raw))
        + bytes.fromhex(sha256(phase_lock_raw))
        + bytes.fromhex(sha256(corpus_raw))
    )
    log_path = output.with_suffix(output.suffix + ".worker.log")
    (
        continuations,
        calls,
        state_before,
        state_after,
        runtime_process,
        runtime_model_binding,
        protocol_identity,
    ) = execute(
        launch,
        histories,
        history_plan,
        model_sha256,
        log_path,
    )
    event_ns = monotonic_ns()
    rows = make_rows(
        histories,
        continuations,
        calls,
        model_sha256,
        program_sha256,
        state_before,
        state_after,
        event_ns,
    )
    log_raw = read_regular(log_path, MAX_LOG_BYTES)
    placement_certificate = parse_placement_certificate(
        log_raw,
        runtime_process,
    )
    memory_certificate = parse_memory_certificate(
        log_raw,
        runtime_process,
    )
    completed_ns = monotonic_ns()
    require(started_ns < event_ns <= completed_ns, "E_CAPTURE_INTERVAL")
    require(
        started_ns <= runtime_process["observed_ns"] <= completed_ns,
        "E_RUNTIME_INTERVAL",
    )
    require(
        started_ns <= runtime_model_binding["observed_ns"] <= completed_ns,
        "E_RUNTIME_MODEL_INTERVAL",
    )
    validate_producer_identity(
        producer_artifact,
        producer_process_receipt,
    )
    result = {
        "completed_ns": completed_ns,
        "history_binding": {
            "corpus_item_indices": CORPUS_ITEM_INDICES,
            "histories_sha256": sha256(history_raw),
            "prompt_sha256s": [
                prompt_sha256s[index]
                for index in CORPUS_ITEM_INDICES
            ],
            "quality_corpus_sha256": sha256(corpus_raw),
            "token_history_corpus_sha256": CORPUS_SHA256,
        },
        "launch_binding": {
            "launch": launch,
            "launch_plan_sha256": sha256(launch_raw),
            "runtime_boot_id": runtime_process["boot_id"],
            "runtime_pid": runtime_process["pid"],
            "runtime_start_ticks": runtime_process["start_ticks"],
        },
        "mechanism_commands_sha256": mechanism_sha256,
        "memory_certificate": memory_certificate,
        "model_id": MODEL_ID,
        "model_sha256": model_sha256,
        "oracle_cuda_monolithic_rows": rows,
        "phase_id": phase_id,
        "placement_certificate": placement_certificate,
        "producer_artifact": producer_artifact,
        "producer_process_receipt": producer_process_receipt,
        "producer_sha256": producer_artifact["sha256"],
        "protocol_identity": protocol_identity,
        "runtime_process": runtime_process,
        "runtime_model_binding": runtime_model_binding,
        "schema": SCHEMA,
        "started_ns": started_ns,
        "worker_log_bytes": len(log_raw),
        "worker_log_path": str(log_path),
        "worker_log_sha256": sha256(log_raw),
    }
    durable_write_new(output, canonical_bytes(result))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--pre-dir", required=True)
    parser.add_argument("--started", type=int, required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--mechanism-commands-sha256", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--histories", required=True)
    parser.add_argument("--histories-sha256", required=True)
    parser.add_argument("--launch-plan", required=True)
    parser.add_argument("--launch-plan-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args(argv)
    require(args.execute, "E_EXECUTE_CONFIRMATION")
    require(args.confirm == CONFIRMATION, "E_EXECUTE_CONFIRMATION")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        capture(parse_args(argv))
        return 0
    except (
        CaptureError,
        OSError,
        struct.error,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        print(
            f"A_ONLY_CUDA_MONOLITHIC_REFUSED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
