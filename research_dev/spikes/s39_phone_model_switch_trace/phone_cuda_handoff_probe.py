#!/usr/bin/env python3
"""Exercise exact token-history handoff from a phone route to a CUDA route."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar


HERE = Path(__file__).resolve().parent
S22 = HERE.parent / "s22_slo_overlap_pipeline"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
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

from qwen25_quality_probe import load_corpus


SCHEMA = "s39-phone-cuda-handoff-report-v1"
CONTRACT_SCHEMA = "s39-phone-cuda-handoff-contract-v1"
DEFAULT_CONTRACT = HERE / "W5_HANDOFF_CONTRACT.json"
HEX64 = re.compile(r"[0-9a-f]{64}")
T = TypeVar("T")


class HandoffError(RuntimeError):
    pass


@dataclass(frozen=True)
class Contract:
    raw_sha256: str
    model_sha256: str
    file_type: int
    n_layer: int
    n_embd: int
    corpus_path: Path
    corpus_sha256: str
    manifest_path: Path
    manifest_sha256: str
    prompt_ids: tuple[int, ...]
    prompt_tokens: int
    batch: int
    phone_committed_tokens: int
    cuda_continuation_tokens: int
    phone_prefill_chunk: int
    cuda_control_chunk: int
    cuda_catchup_chunk: int
    max_rows_per_batch: int


@dataclass(frozen=True)
class ReplayMetrics:
    history_batches: int
    continuation_batches: int
    rows: int
    elapsed_us: int


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
            raise HandoffError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HandoffError(message)


def is_int(value: object) -> bool:
    return type(value) is int


def exact_keys(value: object, keys: set[str], field: str) -> dict[str, Any]:
    require(type(value) is dict, f"{field}: expected object")
    result = value
    require(set(result) == keys, f"{field}: key mismatch")
    return result


def safe_relative(path: object, field: str) -> Path:
    require(type(path) is str and path != "", f"{field}: invalid path")
    candidate = Path(path)
    require(
        not candidate.is_absolute() and ".." not in candidate.parts,
        f"{field}: path must stay below the spike directory",
    )
    resolved = (HERE / candidate).resolve()
    require(
        resolved == HERE or HERE in resolved.parents,
        f"{field}: path escapes the spike directory",
    )
    return resolved


def checked_digest(value: object, field: str) -> str:
    require(
        type(value) is str and HEX64.fullmatch(value) is not None,
        f"{field}: invalid SHA-256",
    )
    return value


def checked_positive(value: object, field: str) -> int:
    require(is_int(value) and value > 0, f"{field}: expected positive integer")
    return value


def load_contract(path: Path) -> Contract:
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=strict_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise HandoffError("contract: invalid JSON") from exc
    require(type(value) is dict and canonical(value) == raw, "contract: not canonical")
    root = exact_keys(
        value,
        {
            "corpus",
            "execution",
            "model",
            "requirements",
            "scheduler_eligible_on_pass",
            "schema",
            "scope",
            "status",
        },
        "contract",
    )
    require(root["schema"] == CONTRACT_SCHEMA, "contract: schema mismatch")
    require(
        root["status"] == "FROZEN_BEFORE_ACQUISITION",
        "contract: status mismatch",
    )
    require(root["scope"] == "MECHANICS_ONLY", "contract: scope mismatch")
    require(
        root["scheduler_eligible_on_pass"] is False,
        "contract: scheduler eligibility must remain false",
    )
    expected_requirements = [
        "PHONE_HISTORY_IS_AUTHORITATIVE",
        "CUDA_CONTROL_AND_CATCHUP_CONTINUATIONS_EXACT",
        "CUDA_CATCHUP_USES_FEWER_HISTORY_BATCHES",
        "NO_PUBLISHED_TOKEN_GAP_OR_DUPLICATE",
        "ONE_PUBLISHING_OWNER_PER_EPOCH",
        "ALL_SEQUENCE_STATE_REMOVED",
    ]
    require(
        root["requirements"] == expected_requirements,
        "contract: requirement set mismatch",
    )

    model = exact_keys(
        root["model"],
        {"file_type", "n_embd", "n_layer", "sha256"},
        "contract.model",
    )
    model_sha256 = checked_digest(model["sha256"], "contract.model.sha256")
    file_type = checked_positive(model["file_type"], "contract.model.file_type")
    n_layer = checked_positive(model["n_layer"], "contract.model.n_layer")
    n_embd = checked_positive(model["n_embd"], "contract.model.n_embd")

    corpus = exact_keys(
        root["corpus"],
        {
            "manifest_path",
            "manifest_sha256",
            "path",
            "prompt_ids",
            "prompt_tokens",
            "sha256",
        },
        "contract.corpus",
    )
    corpus_path = safe_relative(corpus["path"], "contract.corpus.path")
    manifest_path = safe_relative(
        corpus["manifest_path"],
        "contract.corpus.manifest_path",
    )
    corpus_sha256 = checked_digest(corpus["sha256"], "contract.corpus.sha256")
    manifest_sha256 = checked_digest(
        corpus["manifest_sha256"],
        "contract.corpus.manifest_sha256",
    )
    require(corpus_path.is_file(), "contract: corpus does not exist")
    require(manifest_path.is_file(), "contract: corpus manifest does not exist")
    require(
        sha256(corpus_path.read_bytes()) == corpus_sha256,
        "contract: corpus digest mismatch",
    )
    require(
        sha256(manifest_path.read_bytes()) == manifest_sha256,
        "contract: corpus manifest digest mismatch",
    )
    prompt_ids_value = corpus["prompt_ids"]
    require(type(prompt_ids_value) is list, "contract: invalid prompt IDs")
    require(
        all(is_int(item) and item >= 0 for item in prompt_ids_value),
        "contract: invalid prompt ID",
    )
    prompt_ids = tuple(prompt_ids_value)
    require(
        len(prompt_ids) == len(set(prompt_ids)),
        "contract: duplicate prompt ID",
    )
    prompt_tokens = checked_positive(
        corpus["prompt_tokens"],
        "contract.corpus.prompt_tokens",
    )

    execution = exact_keys(
        root["execution"],
        {
            "batch",
            "cuda_catchup_chunk",
            "cuda_continuation_tokens",
            "cuda_control_chunk",
            "max_rows_per_batch",
            "phone_committed_tokens",
            "phone_prefill_chunk",
        },
        "contract.execution",
    )
    values = {
        key: checked_positive(value, f"contract.execution.{key}")
        for key, value in execution.items()
    }
    batch = values["batch"]
    require(len(prompt_ids) == batch, "contract: prompt count and batch differ")
    for chunk_field in (
        "phone_prefill_chunk",
        "cuda_control_chunk",
        "cuda_catchup_chunk",
    ):
        require(
            batch * values[chunk_field] <= values["max_rows_per_batch"],
            f"contract: {chunk_field} exceeds row cap",
        )
    require(
        values["cuda_catchup_chunk"] > values["cuda_control_chunk"],
        "contract: catch-up must use a larger history chunk",
    )
    return Contract(
        sha256(raw),
        model_sha256,
        file_type,
        n_layer,
        n_embd,
        corpus_path,
        corpus_sha256,
        manifest_path,
        manifest_sha256,
        prompt_ids,
        prompt_tokens,
        batch,
        values["phone_committed_tokens"],
        values["cuda_continuation_tokens"],
        values["phone_prefill_chunk"],
        values["cuda_control_chunk"],
        values["cuda_catchup_chunk"],
        values["max_rows_per_batch"],
    )


def validate_hello(name: str, hello: Hello, contract: Contract) -> None:
    required_context = (
        contract.prompt_tokens
        + contract.phone_committed_tokens
        + contract.cuda_continuation_tokens
        - 1
    )
    required_rows = contract.batch * max(
        contract.phone_prefill_chunk,
        contract.cuda_control_chunk,
        contract.cuda_catchup_chunk,
    )
    require(
        hello.layer_start == 0
        and hello.layer_end == contract.n_layer
        and hello.n_layer == contract.n_layer
        and hello.n_embd == contract.n_embd
        and bool(hello.capabilities & STAGE_V3_CAP_TERMINAL)
        and hello.max_streams >= contract.batch
        and hello.n_ctx_seq >= required_context
        and min(hello.n_batch, hello.n_ubatch) >= required_rows,
        f"{name}: topology or capacity mismatch",
    )


def phase_call(phase: str, operation: Callable[[], T]) -> T:
    try:
        return operation()
    except (OSError, ProtocolError, HandoffError) as exc:
        raise HandoffError(f"{phase}: {exc}") from exc


def build_rows(
    histories: Sequence[Sequence[int]],
    identity_base: int,
    position_start: int,
    position_end: int,
) -> list[BatchRow]:
    require(bool(histories), "history batch is empty")
    width = len(histories[0])
    require(
        0 <= position_start < position_end <= width,
        "history position range is invalid",
    )
    require(
        all(
            len(history) == width
            and all(is_int(token) and token >= 0 for token in history)
            for history in histories
        ),
        "histories are not rectangular nonnegative token IDs",
    )
    return [
        BatchRow(
            identity_base + seq_id,
            identity_base + seq_id,
            seq_id,
            position,
            history[position],
        )
        for seq_id, history in enumerate(histories)
        for position in range(position_start, position_end)
    ]


def select_predictions(
    results: Sequence[BatchResult],
    batch: int,
    chunk: int,
    identity_base: int,
    expected_position: int,
) -> list[int]:
    require(len(results) == batch * chunk, "terminal result count mismatch")
    predictions: list[int] = []
    for seq_id in range(batch):
        result = results[(seq_id + 1) * chunk - 1]
        require(
            result.request_id == identity_base + seq_id
            and result.route_epoch == identity_base + seq_id
            and result.seq_id == seq_id
            and result.position == expected_position
            and result.hidden is None
            and is_int(result.token)
            and result.token >= 0,
            "terminal result lineage mismatch",
        )
        predictions.append(result.token)
    return predictions


def timed_batch(
    client: StageV3Client,
    rows: Sequence[BatchRow],
) -> tuple[tuple[BatchResult, ...], int]:
    started_ns = time.monotonic_ns()
    results = client.batch(rows)
    return results, (time.monotonic_ns() - started_ns) // 1000


def run_replay(
    client: StageV3Client,
    histories: Sequence[Sequence[int]],
    identity_base: int,
    history_chunk: int,
    continuation_tokens: int,
) -> tuple[list[list[int]], ReplayMetrics]:
    require(bool(histories), "replay has no histories")
    require(history_chunk > 0, "history chunk must be positive")
    require(continuation_tokens > 0, "continuation count must be positive")
    batch = len(histories)
    history_width = len(histories[0])
    require(history_width > 0, "replay history is empty")
    outputs = [[] for _ in histories]
    elapsed_us = 0
    rows_total = 0
    history_batches = 0
    predictions: list[int] = []
    for position_start in range(0, history_width, history_chunk):
        position_end = min(position_start + history_chunk, history_width)
        rows = build_rows(
            histories,
            identity_base,
            position_start,
            position_end,
        )
        results, batch_us = timed_batch(client, rows)
        elapsed_us += batch_us
        rows_total += len(rows)
        history_batches += 1
        predictions = select_predictions(
            results,
            batch,
            position_end - position_start,
            identity_base,
            position_end - 1,
        )
    for seq_id, token in enumerate(predictions):
        outputs[seq_id].append(token)

    continuation_batches = 0
    for output_index in range(1, continuation_tokens):
        position = history_width + output_index - 1
        rows = [
            BatchRow(
                identity_base + seq_id,
                identity_base + seq_id,
                seq_id,
                position,
                token,
            )
            for seq_id, token in enumerate(predictions)
        ]
        results, batch_us = timed_batch(client, rows)
        elapsed_us += batch_us
        rows_total += len(rows)
        continuation_batches += 1
        predictions = select_predictions(
            results,
            batch,
            1,
            identity_base,
            position,
        )
        for seq_id, token in enumerate(predictions):
            outputs[seq_id].append(token)
    return outputs, ReplayMetrics(
        history_batches,
        continuation_batches,
        rows_total,
        elapsed_us,
    )


def remove_group(
    client: StageV3Client,
    batch: int,
    identity_base: int,
) -> None:
    status = None
    for seq_id in range(batch):
        status = client.remove(
            seq_id,
            identity_base + seq_id,
            identity_base + seq_id,
        )
        require(
            status.active_sequences == batch - seq_id - 1,
            "sequence removal count mismatch",
        )
    require(status is not None and status.active_sequences == 0, "state leak")


def finish(client: StageV3Client, session_end: str) -> None:
    status = client.status()
    require(status.active_sequences == 0, "route has live state before finish")
    drained = client.drain()
    require(
        drained.draining and drained.active_sequences == 0,
        "route drain failed",
    )
    if session_end == "detach":
        client.detach()
    else:
        client.stop()


def histories_digest(histories: Sequence[Sequence[int]]) -> str:
    return sha256(canonical([list(history) for history in histories]))


def expected_events(
    phone_committed: int,
    cuda_continuation: int,
) -> list[dict[str, object]]:
    return [
        {
            "event_index": 0,
            "kind": "PHONE_AUTHORITATIVE",
            "owner": "PHONE",
            "owner_epoch": 1,
            "published_tokens_per_request": 0,
        },
        {
            "event_index": 1,
            "kind": "PHONE_FRONTIER_FROZEN",
            "owner": "PHONE",
            "owner_epoch": 1,
            "published_tokens_per_request": phone_committed,
        },
        {
            "event_index": 2,
            "kind": "CUDA_PREPARED_NOT_PUBLISHED",
            "owner": "PHONE",
            "owner_epoch": 1,
            "published_tokens_per_request": phone_committed,
        },
        {
            "event_index": 3,
            "kind": "OWNERSHIP_COMMIT",
            "owner": "CUDA",
            "owner_epoch": 2,
            "published_tokens_per_request": (
                phone_committed + cuda_continuation
            ),
        },
        {
            "event_index": 4,
            "kind": "REQUESTS_COMPLETE",
            "owner": "NONE",
            "owner_epoch": 3,
            "published_tokens_per_request": (
                phone_committed + cuda_continuation
            ),
        },
    ]


def validate_report(report: object, contract: Contract) -> None:
    require(type(report) is dict, "report: expected object")
    required = {
        "batch",
        "contract_sha256",
        "corpus_manifest_sha256",
        "corpus_sha256",
        "cuda_catchup",
        "cuda_continuation_tokens",
        "cuda_control",
        "events",
        "hellos",
        "model_sha256",
        "phone_committed_tokens",
        "prompts",
        "scheduler_eligible",
        "schema",
        "scope",
        "sequences",
        "state_counts",
        "status",
    }
    require(set(report) == required, "report: key mismatch")
    require(report["schema"] == SCHEMA, "report: schema mismatch")
    require(report["scope"] == "MECHANICS_ONLY", "report: scope mismatch")
    require(report["scheduler_eligible"] is False, "report: eligibility mismatch")
    require(report["contract_sha256"] == contract.raw_sha256, "report: contract mismatch")
    require(report["model_sha256"] == contract.model_sha256, "report: model mismatch")
    require(report["corpus_sha256"] == contract.corpus_sha256, "report: corpus mismatch")
    require(
        report["corpus_manifest_sha256"] == contract.manifest_sha256,
        "report: corpus manifest mismatch",
    )
    require(report["batch"] == contract.batch, "report: batch mismatch")
    require(
        report["prompts"] == list(contract.prompt_ids),
        "report: prompt selection mismatch",
    )
    require(
        report["phone_committed_tokens"] == contract.phone_committed_tokens,
        "report: phone frontier mismatch",
    )
    require(
        report["cuda_continuation_tokens"] == contract.cuda_continuation_tokens,
        "report: CUDA continuation mismatch",
    )
    require(
        report["events"] == expected_events(
            contract.phone_committed_tokens,
            contract.cuda_continuation_tokens,
        ),
        "report: ownership event mismatch",
    )
    hellos = report["hellos"]
    require(
        type(hellos) is dict and set(hellos) == {"cuda", "phone"},
        "report: route hello set mismatch",
    )
    parsed_hellos: dict[str, Hello] = {}
    hello_keys = {
        "capabilities",
        "file_type",
        "layer_end",
        "layer_start",
        "max_streams",
        "model_sha256",
        "n_batch",
        "n_ctx_seq",
        "n_embd",
        "n_layer",
        "n_ubatch",
    }
    for name, value in hellos.items():
        require(
            type(value) is dict and set(value) == hello_keys,
            f"report: invalid {name} hello",
        )
        for field in hello_keys - {"model_sha256"}:
            require(
                is_int(value[field]),
                f"report: invalid {name} hello {field}",
            )
        require(
            type(value["model_sha256"]) is str,
            f"report: invalid {name} hello model digest",
        )
        hello = Hello(**value)
        validate_hello(name, hello, contract)
        parsed_hellos[name] = hello
    try:
        require_same_model(
            parsed_hellos,
            expected_model_sha256=contract.model_sha256,
            expected_file_type=contract.file_type,
        )
    except ProtocolError as exc:
        raise HandoffError(f"report: {exc}") from exc

    state_counts = report["state_counts"]
    require(
        state_counts == {
            "cuda_after_completion": 0,
            "cuda_control_released": 0,
            "cuda_prepared": contract.batch,
            "phone_frontier": contract.batch,
            "phone_released": 0,
        },
        "report: sequence-state counts mismatch",
    )
    sequences = report["sequences"]
    require(
        type(sequences) is list and len(sequences) == contract.batch,
        "report: sequence count mismatch",
    )
    all_exact = True
    for index, sequence in enumerate(sequences):
        require(type(sequence) is dict, "report: invalid sequence")
        require(
            set(sequence) == {
                "catchup_continuation",
                "committed_history",
                "control_continuation",
                "phone_committed",
                "prompt_id",
                "prompt_tokens",
                "published_tokens",
                "sequence_index",
            },
            "report: sequence key mismatch",
        )
        require(sequence["sequence_index"] == index, "report: sequence order mismatch")
        require(
            sequence["prompt_id"] == contract.prompt_ids[index],
            "report: prompt ID mismatch",
        )
        prompt = sequence["prompt_tokens"]
        phone = sequence["phone_committed"]
        history = sequence["committed_history"]
        catchup = sequence["catchup_continuation"]
        control = sequence["control_continuation"]
        published = sequence["published_tokens"]
        for value, width, field in (
            (prompt, contract.prompt_tokens, "prompt"),
            (phone, contract.phone_committed_tokens, "phone"),
            (
                catchup,
                contract.cuda_continuation_tokens,
                "catchup continuation",
            ),
            (
                control,
                contract.cuda_continuation_tokens,
                "control continuation",
            ),
        ):
            require(
                type(value) is list
                and len(value) == width
                and all(is_int(token) and token >= 0 for token in value),
                f"report: invalid {field} tokens",
            )
        require(history == prompt + phone, "report: committed history has a gap")
        require(
            published == phone + catchup,
            "report: published token history has a gap or duplicate",
        )
        all_exact = all_exact and catchup == control

    control = report["cuda_control"]
    catchup = report["cuda_catchup"]
    require(type(control) is dict and type(catchup) is dict, "report: invalid metrics")
    for name, metrics in (("control", control), ("catchup", catchup)):
        require(
            set(metrics) == {
                "continuation_batches",
                "elapsed_us",
                "history_batches",
                "history_sha256",
                "rows",
            },
            f"report: {name} metric keys",
        )
        for field in ("continuation_batches", "elapsed_us", "history_batches", "rows"):
            require(
                is_int(metrics[field]) and metrics[field] >= 0,
                f"report: invalid {name} {field}",
            )
    histories = [sequence["committed_history"] for sequence in sequences]
    expected_history_sha = histories_digest(histories)
    require(
        control["history_sha256"] == expected_history_sha
        and catchup["history_sha256"] == expected_history_sha,
        "report: replay history digest mismatch",
    )
    history_width = (
        contract.prompt_tokens + contract.phone_committed_tokens
    )
    expected_continuation_batches = contract.cuda_continuation_tokens - 1
    expected_rows = contract.batch * (
        history_width + expected_continuation_batches
    )
    expected_control_batches = (
        history_width + contract.cuda_control_chunk - 1
    ) // contract.cuda_control_chunk
    expected_catchup_batches = (
        history_width + contract.cuda_catchup_chunk - 1
    ) // contract.cuda_catchup_chunk
    require(
        control["history_batches"] == expected_control_batches
        and catchup["history_batches"] == expected_catchup_batches
        and control["continuation_batches"] == expected_continuation_batches
        and catchup["continuation_batches"] == expected_continuation_batches
        and control["rows"] == expected_rows
        and catchup["rows"] == expected_rows,
        "report: replay accounting mismatch",
    )
    fewer_batches = catchup["history_batches"] < control["history_batches"]
    expected_status = (
        "HANDOFF_MECHANICS_PASS"
        if all_exact and fewer_batches
        else "HANDOFF_MECHANICS_FAIL"
    )
    require(report["status"] == expected_status, "report: status mismatch")


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
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--phone-route", type=parse_endpoint, required=True)
    parser.add_argument("--cuda-route", type=parse_endpoint, required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--phone-session-end",
        choices=("detach", "stop"),
        default="stop",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.timeout <= 0 or args.output.exists():
        parser.error("invalid timeout or existing output")

    clients: dict[str, StageV3Client] = {}
    stopped: set[str] = set()
    active: dict[str, tuple[int, int] | None] = {
        "phone": None,
        "cuda": None,
    }
    try:
        contract = load_contract(args.contract)
        corpus, _, _ = load_corpus(
            contract.corpus_path,
            contract.manifest_path,
            contract.model_sha256,
        )
        selected = [corpus[prompt_id] for prompt_id in contract.prompt_ids]
        prompts = [record["tokens"] for record in selected]

        clients["phone"] = phase_call(
            "CONNECT_PHONE",
            lambda: StageV3Client.connect(*args.phone_route, args.timeout),
        )
        clients["cuda"] = phase_call(
            "CONNECT_CUDA",
            lambda: StageV3Client.connect(*args.cuda_route, args.timeout),
        )
        hellos = {
            name: phase_call(f"HELLO_{name.upper()}", client.hello)
            for name, client in clients.items()
        }
        for name, hello in hellos.items():
            validate_hello(name, hello, contract)
        require_same_model(
            hellos,
            expected_model_sha256=contract.model_sha256,
            expected_file_type=contract.file_type,
        )

        phone_base = 10000
        control_base = 20000
        catchup_base = 30000
        phone_tokens, phone_metrics = phase_call(
            "PHONE_AUTHORITATIVE",
            lambda: run_replay(
                clients["phone"],
                prompts,
                phone_base,
                contract.phone_prefill_chunk,
                contract.phone_committed_tokens,
            ),
        )
        active["phone"] = (contract.batch, phone_base)
        phone_frontier = phase_call(
            "PHONE_FRONTIER_STATUS",
            clients["phone"].status,
        )
        require(
            phone_frontier.active_sequences == contract.batch,
            "phone frontier state count mismatch",
        )
        histories = [
            list(prompt) + list(committed)
            for prompt, committed in zip(prompts, phone_tokens)
        ]

        control_tokens, control_metrics = phase_call(
            "CUDA_CONTROL",
            lambda: run_replay(
                clients["cuda"],
                histories,
                control_base,
                contract.cuda_control_chunk,
                contract.cuda_continuation_tokens,
            ),
        )
        active["cuda"] = (contract.batch, control_base)
        phase_call(
            "CUDA_CONTROL_REMOVE",
            lambda: remove_group(
                clients["cuda"],
                contract.batch,
                control_base,
            ),
        )
        active["cuda"] = None
        cuda_control_released = phase_call(
            "CUDA_CONTROL_RELEASED_STATUS",
            clients["cuda"].status,
        )
        require(
            cuda_control_released.active_sequences == 0,
            "CUDA control state leak",
        )

        catchup_tokens, catchup_metrics = phase_call(
            "CUDA_CATCHUP",
            lambda: run_replay(
                clients["cuda"],
                histories,
                catchup_base,
                contract.cuda_catchup_chunk,
                contract.cuda_continuation_tokens,
            ),
        )
        active["cuda"] = (contract.batch, catchup_base)
        cuda_ready = phase_call("CUDA_READY_STATUS", clients["cuda"].status)
        require(
            cuda_ready.active_sequences == contract.batch,
            "CUDA catch-up state count mismatch",
        )

        phase_call(
            "PHONE_RELEASE",
            lambda: remove_group(
                clients["phone"],
                contract.batch,
                phone_base,
            ),
        )
        active["phone"] = None
        phone_released = phase_call(
            "PHONE_RELEASED_STATUS",
            clients["phone"].status,
        )
        require(
            phone_released.active_sequences == 0,
            "phone state leak after ownership commit",
        )

        sequences = [
            {
                "catchup_continuation": list(catchup_tokens[index]),
                "committed_history": list(histories[index]),
                "control_continuation": list(control_tokens[index]),
                "phone_committed": list(phone_tokens[index]),
                "prompt_id": contract.prompt_ids[index],
                "prompt_tokens": list(prompts[index]),
                "published_tokens": (
                    list(phone_tokens[index]) + list(catchup_tokens[index])
                ),
                "sequence_index": index,
            }
            for index in range(contract.batch)
        ]
        exact = catchup_tokens == control_tokens
        fewer_batches = (
            catchup_metrics.history_batches
            < control_metrics.history_batches
        )

        phase_call(
            "CUDA_HANDOFF_REMOVE",
            lambda: remove_group(
                clients["cuda"],
                contract.batch,
                catchup_base,
            ),
        )
        active["cuda"] = None
        cuda_after_completion = phase_call(
            "CUDA_COMPLETION_STATUS",
            clients["cuda"].status,
        )
        require(
            cuda_after_completion.active_sequences == 0,
            "CUDA state leak after completion",
        )

        phase_call(
            "FINISH_PHONE",
            lambda: finish(clients["phone"], args.phone_session_end),
        )
        stopped.add("phone")
        phase_call("FINISH_CUDA", lambda: finish(clients["cuda"], "stop"))
        stopped.add("cuda")

        history_sha = histories_digest(histories)
        report: dict[str, object] = {
            "batch": contract.batch,
            "contract_sha256": contract.raw_sha256,
            "corpus_manifest_sha256": contract.manifest_sha256,
            "corpus_sha256": contract.corpus_sha256,
            "cuda_catchup": {
                **asdict(catchup_metrics),
                "history_sha256": history_sha,
            },
            "cuda_continuation_tokens": contract.cuda_continuation_tokens,
            "cuda_control": {
                **asdict(control_metrics),
                "history_sha256": history_sha,
            },
            "events": expected_events(
                contract.phone_committed_tokens,
                contract.cuda_continuation_tokens,
            ),
            "hellos": {name: asdict(hello) for name, hello in hellos.items()},
            "model_sha256": contract.model_sha256,
            "phone_committed_tokens": contract.phone_committed_tokens,
            "prompts": list(contract.prompt_ids),
            "scheduler_eligible": False,
            "schema": SCHEMA,
            "scope": "MECHANICS_ONLY",
            "sequences": sequences,
            "state_counts": {
                "cuda_after_completion": (
                    cuda_after_completion.active_sequences
                ),
                "cuda_control_released": (
                    cuda_control_released.active_sequences
                ),
                "cuda_prepared": cuda_ready.active_sequences,
                "phone_frontier": phone_frontier.active_sequences,
                "phone_released": phone_released.active_sequences,
            },
            "status": (
                "HANDOFF_MECHANICS_PASS"
                if exact and fewer_batches
                else "HANDOFF_MECHANICS_FAIL"
            ),
        }
        validate_report(report, contract)
        write_atomic(args.output, report)
        print(canonical(report).decode("ascii"), end="")
        return 0 if report["status"] == "HANDOFF_MECHANICS_PASS" else 3
    finally:
        for name, client in clients.items():
            state = active.get(name)
            if state is not None:
                batch, identity_base = state
                try:
                    remove_group(client, batch, identity_base)
                except (OSError, ProtocolError, HandoffError):
                    pass
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
    except (OSError, ProtocolError, HandoffError, ValueError) as exc:
        print(canonical({
            "error": str(exc),
            "status": "HANDOFF_MECHANICS_ERROR",
        }).decode("ascii"), end="")
        raise SystemExit(2)
