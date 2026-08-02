#!/usr/bin/env python3

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from typing import Any


MODEL_ID = "qwen3-14b-q4_k_m"
MODEL_SHA256 = "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0"
MODEL_BYTES = 9001752960
VOCAB_SIZE = 151936
CANDIDATE_SHA256 = "ee3196ca660fa7eb7ea293260dc98dd6fdbf14571a4d4c5aece5343fb29b28d8"
CANDIDATE_BYTES = 2173
CORPUS_SHA256 = "3ffafee1615ae2de690a2726b880823e167a3d9c210c5faed86d8f0e93ecff4f"
CORPUS_BYTES = 33885
CORPUS_ITEMS = 64
CORPUS_REVISION = "bc5d09e5f0d160a95bcd36354bb5e16e50afe270"
MECHANICS_ITEM_IDS = list(range(8))
ALL_ITEM_IDS = list(range(64))
N_CTX_SEQ = 512
CONTINUATION_TOKENS = 8
MAX_PROMPT_TOKENS = N_CTX_SEQ - CONTINUATION_TOKENS
MAX_PREFILL_ROWS = 64
PLAN_SCHEMA = "s39-cp0-r1-a-only-tokenizer-plan-v2"
HISTORY_SCHEMA = "s39-cp0-r1-token-history-v2.4"
MAX_SMALL_FILE = 4 * 1024 * 1024
MAX_CODEC_OUTPUT = 4 * 1024 * 1024
MAX_CODEC_STDERR = 4 * 1024 * 1024


class HistoryError(RuntimeError):
    pass


@dataclass(frozen=True)
class Locks:
    model_sha256: str = MODEL_SHA256
    model_bytes: int = MODEL_BYTES
    vocab_size: int = VOCAB_SIZE
    candidate_sha256: str = CANDIDATE_SHA256
    candidate_bytes: int = CANDIDATE_BYTES
    corpus_sha256: str = CORPUS_SHA256
    corpus_bytes: int = CORPUS_BYTES
    corpus_items: int = CORPUS_ITEMS
    corpus_revision: str = CORPUS_REVISION


PRODUCTION_LOCKS = Locks()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HistoryError(message)


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"E_DUPLICATE_KEY: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise HistoryError(f"E_JSON_NUMBER: {value}")


def canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, UnicodeEncodeError) as error:
        raise HistoryError("E_CANONICAL") from error


def parse_json(raw: bytes, field: str) -> Any:
    try:
        return json.loads(
            raw.decode("ascii"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HistoryError(f"E_JSON: {field}") from error


def exact_keys(value: Any, keys: set[str], field: str) -> dict[str, Any]:
    require(type(value) is dict, f"E_TYPE: {field}")
    actual = set(value)
    require(
        actual == keys,
        f"E_KEYS: {field}: missing={sorted(keys - actual)} "
        f"unknown={sorted(actual - keys)}",
    )
    return value


def integer(value: Any, field: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f"E_INTEGER: {field}")
    return value


def text(value: Any, field: str, maximum: int = 4096) -> str:
    require(type(value) is str and 0 < len(value) <= maximum, f"E_TEXT: {field}")
    return value


def digest(value: Any, field: str) -> str:
    value = text(value, field, 64)
    require(
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"E_DIGEST: {field}",
    )
    return value


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_mode,
    )


def snapshot_file(path: Path, maximum: int | None = None) -> dict[str, Any]:
    require(path.is_absolute(), f"E_ABSOLUTE_PATH: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise HistoryError(f"E_OPEN: {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"E_FILE_TYPE: {path}")
        require(before.st_size > 0, f"E_FILE_SIZE: {path}")
        if maximum is not None:
            require(before.st_size <= maximum, f"E_FILE_SIZE: {path}")
        value = hashlib.sha256()
        consumed = 0
        while block := os.read(descriptor, 1024 * 1024):
            value.update(block)
            consumed += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    require(_identity(before) == _identity(after), f"E_SOURCE_MUTATED: {path}")
    require(consumed == before.st_size, f"E_SOURCE_MUTATED: {path}")
    return {
        "bytes": before.st_size,
        "path": str(path),
        "sha256": value.hexdigest(),
    }


def read_small_canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    snapshot = snapshot_file(path, MAX_SMALL_FILE)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise HistoryError(f"E_READ: {path}: {error}") from error
    require(len(raw) == snapshot["bytes"], f"E_SOURCE_MUTATED: {path}")
    require(sha256(raw) == snapshot["sha256"], f"E_SOURCE_MUTATED: {path}")
    value = parse_json(raw, str(path))
    require(type(value) is dict, f"E_TYPE: {path}")
    require(canonical_bytes(value) == raw, f"E_CANONICAL: {path}")
    return value, raw


def read_corpus(path: Path, locks: Locks) -> tuple[list[dict[str, Any]], bytes]:
    snapshot = snapshot_file(path, MAX_SMALL_FILE)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise HistoryError(f"E_READ: {path}: {error}") from error
    require(
        len(raw) == locks.corpus_bytes
        and sha256(raw) == locks.corpus_sha256
        and snapshot["bytes"] == locks.corpus_bytes
        and snapshot["sha256"] == locks.corpus_sha256,
        "E_CORPUS_IDENTITY",
    )
    lines = raw.splitlines(keepends=True)
    require(len(lines) == locks.corpus_items, "E_CORPUS_COUNT")
    rows = []
    seen = set()
    expected_keys = {
        "choices",
        "dataset",
        "dataset_revision",
        "expected_answer",
        "item_index",
        "question",
        "source_row",
        "subject",
    }
    for index, line in enumerate(lines):
        row = parse_json(line, f"corpus[{index}]")
        exact_keys(row, expected_keys, f"corpus[{index}]")
        require(canonical_bytes(row) == line, f"E_CANONICAL: corpus[{index}]")
        item_id = integer(row["item_index"], f"corpus[{index}].item_index")
        require(item_id not in seen, f"E_CORPUS_DUPLICATE_ID: {item_id}")
        require(item_id == index, f"E_CORPUS_MISSING_OR_REORDERED_ID: {index}")
        seen.add(item_id)
        require(
            row["dataset"] == "cais/mmlu"
            and row["dataset_revision"] == locks.corpus_revision,
            f"E_CORPUS_SOURCE: {index}",
        )
        require(
            type(row["choices"]) is list
            and len(row["choices"]) == 4
            and all(type(choice) is str for choice in row["choices"]),
            f"E_CORPUS_CHOICES: {index}",
        )
        require(
            type(row["question"]) is str
            and type(row["subject"]) is str
            and row["expected_answer"] in "ABCD",
            f"E_CORPUS_ROW: {index}",
        )
        rows.append(row)
    require(seen == set(range(locks.corpus_items)), "E_CORPUS_MISSING_ID")
    return rows, raw


def load_candidate(path: Path, locks: Locks) -> tuple[dict[str, Any], bytes]:
    candidate, raw = read_small_canonical(path)
    require(
        len(raw) == locks.candidate_bytes
        and sha256(raw) == locks.candidate_sha256,
        "E_CANDIDATE_IDENTITY",
    )
    exact_keys(
        candidate,
        {
            "candidate_attempt",
            "candidate_attempt_limit",
            "contract_sha256",
            "historical_routes",
            "models",
            "schema",
            "status",
            "task_suite",
        },
        "candidate",
    )
    task = candidate["task_suite"]
    require(type(task) is dict, "E_CANDIDATE_TASK")
    require(
        task.get("dataset") == "cais/mmlu"
        and task.get("revision") == locks.corpus_revision
        and task.get("items") == locks.corpus_items
        and task.get("maximum_output_tokens") == CONTINUATION_TOKENS
        and task.get("chat_template") == "NONE_RAW_COMPLETION",
        "E_CANDIDATE_TASK",
    )
    require(type(task.get("prompt_format")) is str, "E_PROMPT_FORMAT")
    return candidate, raw


def load_plan(path: Path, locks: Locks) -> tuple[dict[str, Any], bytes]:
    plan, raw = read_small_canonical(path)
    exact_keys(
        plan,
        {
            "command_template",
            "component_id",
            "cwd",
            "environment",
            "executable",
            "model",
            "protocol",
            "schema",
            "timeout_seconds",
        },
        "plan",
    )
    require(plan["schema"] == PLAN_SCHEMA, "E_PLAN_SCHEMA")
    executable = exact_keys(
        plan["executable"], {"bytes", "path", "sha256"}, "plan.executable"
    )
    model = exact_keys(
        plan["model"],
        {"bytes", "model_id", "path", "sha256", "vocab_size"},
        "plan.model",
    )
    executable_path = Path(text(executable["path"], "plan.executable.path"))
    model_path = Path(text(model["path"], "plan.model.path"))
    require(executable_path.is_absolute(), "E_EXECUTABLE_PATH")
    require(model_path.is_absolute(), "E_MODEL_PATH")
    digest(executable["sha256"], "plan.executable.sha256")
    require(integer(executable["bytes"], "plan.executable.bytes", 1) > 0, "E_EXEC")
    require(model["model_id"] == MODEL_ID, "E_MODEL_ID")
    require(model["sha256"] == locks.model_sha256, "E_MODEL_SHA256")
    require(model["bytes"] == locks.model_bytes, "E_MODEL_BYTES")
    require(model["vocab_size"] == locks.vocab_size, "E_VOCAB_SIZE")
    component_id = text(plan["component_id"], "plan.component_id", 256)
    require(
        all(
            character.isalnum() or character in "._-"
            for character in component_id
        ),
        "E_COMPONENT_ID",
    )
    command = plan["command_template"]
    require(
        command
        == [
            str(executable_path),
            "-m",
            str(model_path),
            "--ids",
            "-f",
            "{PROMPT_FILE}",
            "--log-disable",
        ],
        "E_COMMAND",
    )
    require(plan["cwd"] == str(executable_path.parent), "E_CWD")
    require(
        plan["protocol"]
        == {
            "add_bos": "MODEL_DEFAULT",
            "escape": True,
            "output_format": "BRACKETED_DECIMAL_IDS",
            "parse_special": True,
            "prompt_file_placeholder": "{PROMPT_FILE}",
        },
        "E_PROTOCOL",
    )
    environment = plan["environment"]
    require(
        type(environment) is dict
        and set(environment).issubset({"LC_ALL", "LD_LIBRARY_PATH"}),
        "E_ENVIRONMENT",
    )
    for key, value in environment.items():
        require(
            type(key) is str
            and type(value) is str
            and "\x00" not in key
            and "\x00" not in value,
            "E_ENVIRONMENT",
        )
    timeout = integer(plan["timeout_seconds"], "plan.timeout_seconds", 1)
    require(timeout <= 300, "E_TIMEOUT")
    return plan, raw


def verify_plan_files(plan: dict[str, Any]) -> None:
    for name in ("executable", "model"):
        actual = snapshot_file(Path(plan[name]["path"]))
        require(actual == {
            "bytes": plan[name]["bytes"],
            "path": plan[name]["path"],
            "sha256": plan[name]["sha256"],
        }, f"E_{name.upper()}_MUTATED")
    mode = os.stat(plan["executable"]["path"], follow_symlinks=False).st_mode
    require(mode & 0o111 != 0, "E_EXECUTABLE_MODE")


def make_prompts(
    corpus: list[dict[str, Any]],
    candidate: dict[str, Any],
    item_ids: list[int],
) -> list[dict[str, Any]]:
    prompt_format = candidate["task_suite"]["prompt_format"]
    result = []
    for item_id in item_ids:
        item = corpus[item_id]
        try:
            prompt = prompt_format.format(
                question=item["question"],
                choice0=item["choices"][0],
                choice1=item["choices"][1],
                choice2=item["choices"][2],
                choice3=item["choices"][3],
            )
        except (IndexError, KeyError, ValueError) as error:
            raise HistoryError(f"E_PROMPT_FORMAT: {item_id}") from error
        raw = prompt.encode("utf-8")
        result.append({
            "item_index": item_id,
            "corpus_item_sha256": sha256(canonical_bytes(item)),
            "prompt_utf8_base64": base64.b64encode(raw).decode("ascii"),
            "prompt_utf8_bytes": len(raw),
            "prompt_utf8_sha256": sha256(raw),
        })
    return result


def invoke_tokenizer(
    plan: dict[str, Any],
    prompts: list[dict[str, Any]],
) -> list[list[int]]:
    histories = []
    with tempfile.TemporaryDirectory(prefix="s39-token-history-") as temporary:
        root = Path(temporary)
        for index, prompt in enumerate(prompts):
            prompt_raw = base64.b64decode(
                prompt["prompt_utf8_base64"].encode("ascii"), validate=True
            )
            require(
                len(prompt_raw) == prompt["prompt_utf8_bytes"]
                and sha256(prompt_raw) == prompt["prompt_utf8_sha256"],
                f"E_PROMPT_BYTES: {index}",
            )
            prompt_path = root / f"prompt-{index:02d}.txt"
            descriptor = os.open(
                prompt_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                0o600,
            )
            try:
                offset = 0
                while offset < len(prompt_raw):
                    offset += os.write(descriptor, prompt_raw[offset:])
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            require(
                snapshot_file(prompt_path, MAX_SMALL_FILE)["sha256"]
                == prompt["prompt_utf8_sha256"],
                f"E_PROMPT_FILE: {index}",
            )
            command = [
                str(prompt_path) if value == "{PROMPT_FILE}" else value
                for value in plan["command_template"]
            ]
            with (
                tempfile.TemporaryFile() as stdout_file,
                tempfile.TemporaryFile() as stderr_file,
            ):
                process = subprocess.Popen(
                    command,
                    cwd=plan["cwd"],
                    env=plan["environment"],
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_file,
                    stderr=stderr_file,
                )
                try:
                    process.wait(timeout=plan["timeout_seconds"])
                except subprocess.TimeoutExpired as error:
                    process.kill()
                    process.wait()
                    raise HistoryError("E_TOKENIZER_TIMEOUT") from error
                require(
                    process.returncode == 0,
                    f"E_TOKENIZER_EXIT: {index}:{process.returncode}",
                )
                stdout_size = stdout_file.tell()
                stderr_size = stderr_file.tell()
                require(stdout_size <= MAX_CODEC_OUTPUT, "E_TOKENIZER_STDOUT_SIZE")
                require(stderr_size <= MAX_CODEC_STDERR, "E_TOKENIZER_STDERR_SIZE")
                stdout_file.seek(0)
                stderr_file.seek(0)
                stdout_raw = stdout_file.read()
                stderr_raw = stderr_file.read()
            require(stderr_raw == b"", f"E_TOKENIZER_STDERR: {index}")
            try:
                tokens = json.loads(stdout_raw.decode("ascii"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise HistoryError(f"E_TOKENIZER_OUTPUT: {index}") from error
            require(
                type(tokens) is list
                and 0 < len(tokens) <= MAX_PROMPT_TOKENS,
                f"E_TOKEN_COUNT: {index}",
            )
            for token_index, token in enumerate(tokens):
                require(
                    type(token) is int
                    and 0 <= token < plan["model"]["vocab_size"],
                    f"E_TOKEN_RANGE: {index}:{token_index}",
                )
            expected_stdout = (
                "[" + ", ".join(str(token) for token in tokens) + "]\n"
            ).encode("ascii")
            require(
                stdout_raw == expected_stdout,
                f"E_TOKENIZER_OUTPUT_CANONICAL: {index}",
            )
            histories.append(tokens)
    return histories


def make_prefill(
    histories: list[list[int]],
    item_ids: list[int],
) -> list[dict[str, Any]]:
    require(len(histories) == len(item_ids) == 8, "E_GROUP_SIZE")
    groups = []
    for position in range(max(len(history) for history in histories)):
        group = []
        for request_id, history in enumerate(histories):
            if position < len(history):
                group.append({
                    "item_index": item_ids[request_id],
                    "position": position,
                    "request_id": request_id + 1,
                    "seq_id": request_id,
                    "token_id": history[position],
                })
        require(0 < len(group) <= 8, "E_PREFILL_POSITION_GROUP")
        groups.append(group)
    rows = [row for group in groups for row in group]
    partition_rows: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for group in groups:
        if current and len(current) + len(group) > MAX_PREFILL_ROWS:
            partition_rows.append(current)
            current = []
        current.extend(group)
    if current:
        partition_rows.append(current)
    partitions = []
    for call_index, call_rows in enumerate(partition_rows):
        partitions.append({
            "call_index": call_index,
            "rows": call_rows,
        })
    require(bool(partitions), "E_PREFILL_EMPTY")
    require(
        all(0 < len(item["rows"]) <= MAX_PREFILL_ROWS for item in partitions),
        "E_PREFILL_PARTITION",
    )
    require(
        [row for item in partitions for row in item["rows"]] == rows,
        "E_PREFILL_ROW_ORDER",
    )
    return partitions


def make_decode_calls(
    histories: list[list[int]],
    item_ids: list[int],
    first_call_index: int,
) -> list[dict[str, Any]]:
    require(len(histories) == len(item_ids) == 8, "E_GROUP_SIZE")
    result = []
    for decode_index in range(7):
        result.append({
            "call_index": first_call_index + decode_index,
            "continuation_input_ordinal": decode_index,
            "continuation_output_ordinal": decode_index + 1,
            "rows": [
                {
                    "item_index": item_ids[request_id],
                    "position": len(histories[request_id]) + decode_index,
                    "request_id": request_id + 1,
                    "seq_id": request_id,
                }
                for request_id in range(8)
            ],
        })
    return result


def make_group(
    all_histories: list[list[int]],
    item_ids: list[int],
    group_index: int,
) -> dict[str, Any]:
    histories = [all_histories[item_id] for item_id in item_ids]
    prefill = make_prefill(histories, item_ids)
    return {
        "decode_calls": make_decode_calls(histories, item_ids, len(prefill)),
        "group_index": group_index,
        "item_indices": item_ids,
        "prefill_partitions": prefill,
    }


def build_history(
    corpus_path: Path,
    candidate_path: Path,
    plan_path: Path,
    locks: Locks = PRODUCTION_LOCKS,
) -> dict[str, Any]:
    plan, plan_raw = load_plan(plan_path, locks)
    verify_plan_files(plan)
    corpus, corpus_raw = read_corpus(corpus_path, locks)
    candidate, candidate_raw = load_candidate(candidate_path, locks)
    prompts = make_prompts(corpus, candidate, ALL_ITEM_IDS)
    histories = invoke_tokenizer(plan, prompts)
    verify_plan_files(plan)
    corpus_after = snapshot_file(corpus_path, MAX_SMALL_FILE)
    candidate_after = snapshot_file(candidate_path, MAX_SMALL_FILE)
    require(
        corpus_after["sha256"] == locks.corpus_sha256
        and corpus_after["bytes"] == locks.corpus_bytes,
        "E_CORPUS_MUTATED",
    )
    require(
        candidate_after["sha256"] == locks.candidate_sha256
        and candidate_after["bytes"] == locks.candidate_bytes,
        "E_CANDIDATE_MUTATED",
    )
    requests = [
        {
            "item_index": prompt["item_index"],
            "prompt_utf8_base64": prompt["prompt_utf8_base64"],
            "prompt_utf8_bytes": prompt["prompt_utf8_bytes"],
            "prompt_sha256": prompt["prompt_utf8_sha256"],
            "request_id": prompt["item_index"] % 8 + 1,
            "seq_id": prompt["item_index"] % 8,
            "token_ids": tokens,
        }
        for prompt, tokens in zip(prompts, histories)
    ]
    groups = [
        make_group(
            histories,
            list(range(group_index * 8, group_index * 8 + 8)),
            group_index,
        )
        for group_index in range(8)
    ]
    return {
        "batch": 8,
        "candidate_sha256": sha256(candidate_raw),
        "continuation_tokens_per_request": CONTINUATION_TOKENS,
        "corpus_sha256": sha256(corpus_raw),
        "mechanics_b8": groups[0],
        "model_id": MODEL_ID,
        "model_sha256": plan["model"]["sha256"],
        "n_batch": 64,
        "n_ctx_seq": N_CTX_SEQ,
        "n_ubatch": 64,
        "prefill_chunking": "WHOLE_POSITION_WAVES_MAX_64_ROWS",
        "prefill_row_order": "POSITION_MAJOR_THEN_ITEM_INDEX",
        "quality_groups": groups,
        "requests": requests,
        "schema": HISTORY_SCHEMA,
        "tokenizer": {
            "component_id": plan["component_id"],
            "path": plan["executable"]["path"],
            "plan_sha256": sha256(plan_raw),
            "sha256": plan["executable"]["sha256"],
        },
    }


def validate_history(
    history: dict[str, Any],
    corpus_path: Path,
    candidate_path: Path,
    plan_path: Path,
    locks: Locks = PRODUCTION_LOCKS,
) -> None:
    expected = build_history(corpus_path, candidate_path, plan_path, locks)
    require(history == expected, "E_HISTORY_MISMATCH")


def write_exclusive(path: Path, value: Any) -> None:
    require(path.is_absolute() and not path.exists(), "E_OUTPUT_PATH")
    raw = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o644,
    )
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
