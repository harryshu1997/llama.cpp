#!/usr/bin/env python3
"""Run concurrent phone progress and CUDA token-delta catch-up."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar


HERE = Path(__file__).resolve().parent
S22 = HERE.parent / "s22_slo_overlap_pipeline"
for module_path in (HERE, S22):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import phone_cuda_handoff_probe as w5
from async_pipeline import parse_endpoint
from qwen25_quality_probe import load_corpus
from stage_v3_client import (
    BatchResult,
    BatchRow,
    Hello,
    ProtocolError,
    StageV3Client,
    require_same_model,
)


SCHEMA = "s39-phone-cuda-delta-report-v1"
CONTRACT_SCHEMA = "s39-phone-cuda-delta-contract-v1"
JOURNAL_SCHEMA = "s39-ownership-record-v2"
DEFAULT_CONTRACT = HERE / "W6_DELTA_CONTRACT.json"
DEFAULT_BASE_CONTRACT = HERE / "W5_HANDOFF_CONTRACT.json"
HEX64 = re.compile(r"[0-9a-f]{64}")
ZERO_SHA256 = "0" * 64
T = TypeVar("T")
JOURNAL_STATES = (
    ("PHONE_FRONTIER", "PHONE", 1, True, True),
    ("CUDA_PREPARED", "PHONE", 1, True, True),
    ("CUDA_COMMITTED", "CUDA", 2, True, True),
    ("PHONE_RELEASED", "CUDA", 2, False, True),
    ("CUDA_CONTINUATION", "CUDA", 2, False, True),
    ("COMPLETE", "NONE", 3, False, False),
)


class DeltaError(RuntimeError):
    pass


@dataclass(frozen=True)
class DeltaContract:
    raw_sha256: str
    base_contract_sha256: str
    batch: int
    phone_snapshot_tokens: int
    phone_delta_tokens: int
    cuda_continuation_tokens: int
    cuda_snapshot_chunk: int
    cuda_delta_chunk: int
    cuda_control_chunk: int
    max_rows_per_batch: int
    min_overlap_shorter_ppm: int
    journal_phases: tuple[str, ...]


@dataclass(frozen=True)
class BatchMetrics:
    batches: int
    rows: int
    elapsed_us: int


@dataclass(frozen=True)
class ConcurrentLeg:
    started_ns: int
    ended_ns: int


@dataclass(frozen=True)
class JournalEntry:
    name: str
    sha256: str
    value: dict[str, object]


def canonical(value: object) -> bytes:
    return w5.canonical(value)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DeltaError(message)


def is_int(value: object) -> bool:
    return type(value) is int


def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DeltaError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def exact_keys(value: object, keys: set[str], field: str) -> dict[str, Any]:
    require(type(value) is dict, f"{field}: expected object")
    result = value
    require(set(result) == keys, f"{field}: key mismatch")
    return result


def checked_positive(value: object, field: str) -> int:
    require(is_int(value) and value > 0, f"{field}: expected positive integer")
    return value


def checked_digest(value: object, field: str) -> str:
    require(
        type(value) is str and HEX64.fullmatch(value) is not None,
        f"{field}: invalid SHA-256",
    )
    return value


def read_canonical(path: Path, field: str) -> tuple[dict[str, object], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=strict_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise DeltaError(f"{field}: invalid JSON") from exc
    require(type(value) is dict and canonical(value) == raw, f"{field}: not canonical")
    return value, raw


def load_contract(
    path: Path,
    base_contract: w5.Contract,
) -> DeltaContract:
    value, raw = read_canonical(path, "delta_contract")
    root = exact_keys(
        value,
        {
            "base_contract_sha256",
            "execution",
            "journal",
            "requirements",
            "scheduler_eligible_on_pass",
            "schema",
            "scope",
            "status",
        },
        "delta_contract",
    )
    require(root["schema"] == CONTRACT_SCHEMA, "delta_contract: schema mismatch")
    require(
        root["status"] == "FROZEN_BEFORE_ACQUISITION",
        "delta_contract: status mismatch",
    )
    require(root["scope"] == "MECHANICS_ONLY", "delta_contract: scope mismatch")
    require(
        root["scheduler_eligible_on_pass"] is False,
        "delta_contract: scheduler eligibility must remain false",
    )
    base_digest = checked_digest(
        root["base_contract_sha256"],
        "delta_contract.base_contract_sha256",
    )
    require(
        base_digest == base_contract.raw_sha256,
        "delta_contract: base contract mismatch",
    )
    expected_requirements = [
        "PHONE_ADVANCES_DURING_CUDA_SNAPSHOT_REPLAY",
        "CUDA_INGESTS_EXACT_PHONE_DELTA",
        "CUDA_CONTROL_AND_DELTA_CATCHUP_CONTINUATIONS_EXACT",
        "DURABLE_SINGLE_PUBLICATION_OWNER",
        "RUN_UNIQUE_OWNERSHIP_TRANSACTION",
        "FAILED_COMMIT_RETAINS_PHONE_STATE",
        "NO_PUBLISHED_TOKEN_GAP_OR_DUPLICATE",
        "ALL_SEQUENCE_STATE_REMOVED",
    ]
    require(
        root["requirements"] == expected_requirements,
        "delta_contract: requirements mismatch",
    )
    execution = exact_keys(
        root["execution"],
        {
            "batch",
            "cuda_continuation_tokens",
            "cuda_control_chunk",
            "cuda_delta_chunk",
            "cuda_snapshot_chunk",
            "max_rows_per_batch",
            "min_overlap_shorter_ppm",
            "phone_delta_tokens",
            "phone_snapshot_tokens",
        },
        "delta_contract.execution",
    )
    values = {
        key: checked_positive(value, f"delta_contract.execution.{key}")
        for key, value in execution.items()
    }
    require(values["batch"] == base_contract.batch, "delta_contract: batch mismatch")
    require(
        values["phone_snapshot_tokens"] == base_contract.phone_committed_tokens,
        "delta_contract: snapshot frontier mismatch",
    )
    require(
        values["cuda_continuation_tokens"]
        == base_contract.cuda_continuation_tokens,
        "delta_contract: CUDA continuation mismatch",
    )
    require(
        values["min_overlap_shorter_ppm"] <= 1_000_000,
        "delta_contract: overlap threshold exceeds one",
    )
    for field in (
        "cuda_control_chunk",
        "cuda_delta_chunk",
        "cuda_snapshot_chunk",
    ):
        require(
            values["batch"] * values[field] <= values["max_rows_per_batch"],
            f"delta_contract: {field} exceeds row cap",
        )
    require(
        values["cuda_snapshot_chunk"] > values["cuda_control_chunk"],
        "delta_contract: snapshot catch-up is not batched more deeply",
    )
    journal = exact_keys(
        root["journal"],
        {"commit_order", "phases", "schema"},
        "delta_contract.journal",
    )
    require(journal["schema"] == JOURNAL_SCHEMA, "delta_contract: journal schema")
    require(
        journal["commit_order"]
        == "DURABLE_CUDA_COMMIT_BEFORE_PHONE_STATE_RELEASE",
        "delta_contract: unsafe commit order",
    )
    phases = tuple(state[0] for state in JOURNAL_STATES)
    require(journal["phases"] == list(phases), "delta_contract: phase mismatch")
    return DeltaContract(
        sha256(raw),
        base_digest,
        values["batch"],
        values["phone_snapshot_tokens"],
        values["phone_delta_tokens"],
        values["cuda_continuation_tokens"],
        values["cuda_snapshot_chunk"],
        values["cuda_delta_chunk"],
        values["cuda_control_chunk"],
        values["max_rows_per_batch"],
        values["min_overlap_shorter_ppm"],
        phases,
    )


class OwnershipJournal:
    def __init__(
        self,
        path: Path,
        transaction_id: str,
        model_sha256: str,
        request_ids: Sequence[int],
        run_id: str,
    ) -> None:
        require(
            HEX64.fullmatch(transaction_id) is not None,
            "journal: invalid transaction ID",
        )
        require(
            HEX64.fullmatch(model_sha256) is not None,
            "journal: invalid model digest",
        )
        require(HEX64.fullmatch(run_id) is not None, "journal: invalid run ID")
        require(
            bool(request_ids)
            and all(is_int(value) and value >= 0 for value in request_ids)
            and len(set(request_ids)) == len(request_ids),
            "journal: invalid request IDs",
        )
        path.mkdir(mode=0o700, parents=False, exist_ok=False)
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        self.path = path
        self.transaction_id = transaction_id
        self.model_sha256 = model_sha256
        self.request_ids = tuple(request_ids)
        self.run_id = run_id
        self.entries: list[JournalEntry] = []

    def append(
        self,
        *,
        phase: str,
        owner: str,
        owner_epoch: int,
        published_tokens_per_request: int,
        token_history_sha256: str,
        phone_active: bool,
        cuda_active: bool,
    ) -> JournalEntry:
        require(type(phase) is str and phase != "", "journal: invalid phase")
        require(owner in {"PHONE", "CUDA", "NONE"}, "journal: invalid owner")
        require(is_int(owner_epoch) and owner_epoch > 0, "journal: invalid epoch")
        require(
            is_int(published_tokens_per_request)
            and published_tokens_per_request >= 0,
            "journal: invalid published count",
        )
        checked_digest(token_history_sha256, "journal.token_history_sha256")
        require(type(phone_active) is bool, "journal: invalid phone state")
        require(type(cuda_active) is bool, "journal: invalid CUDA state")
        index = len(self.entries)
        require(index < len(JOURNAL_STATES), "journal: transaction is complete")
        expected = JOURNAL_STATES[index]
        require(
            (phase, owner, owner_epoch, phone_active, cuda_active) == expected,
            "journal: invalid state transition",
        )
        previous = self.entries[-1].sha256 if self.entries else ZERO_SHA256
        value: dict[str, object] = {
            "cuda_active": cuda_active,
            "model_sha256": self.model_sha256,
            "owner": owner,
            "owner_epoch": owner_epoch,
            "phase": phase,
            "phone_active": phone_active,
            "previous_record_sha256": previous,
            "published_tokens_per_request": published_tokens_per_request,
            "record_index": index,
            "request_ids": list(self.request_ids),
            "run_id": self.run_id,
            "schema": JOURNAL_SCHEMA,
            "token_history_sha256": token_history_sha256,
            "transaction_id": self.transaction_id,
        }
        raw = canonical(value)
        name = f"{index:06d}.json"
        target = self.path / name
        with target.open("xb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        directory_fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        entry = JournalEntry(name, sha256(raw), value)
        self.entries.append(entry)
        return entry


def load_journal(
    path: Path,
    contract: DeltaContract,
    base_contract: w5.Contract,
    transaction_id: str,
    run_id: str,
) -> list[JournalEntry]:
    require(path.is_dir() and not path.is_symlink(), "journal: invalid directory")
    children = sorted(path.iterdir(), key=lambda item: item.name)
    expected_names = [
        f"{index:06d}.json"
        for index in range(len(contract.journal_phases))
    ]
    require(
        [child.name for child in children] == expected_names,
        "journal: file sequence mismatch",
    )
    entries: list[JournalEntry] = []
    previous = ZERO_SHA256
    record_keys = {
        "cuda_active",
        "model_sha256",
        "owner",
        "owner_epoch",
        "phase",
        "phone_active",
        "previous_record_sha256",
        "published_tokens_per_request",
        "record_index",
        "request_ids",
        "run_id",
        "schema",
        "token_history_sha256",
        "transaction_id",
    }
    expected_owner = ("PHONE", "PHONE", "CUDA", "CUDA", "CUDA", "NONE")
    expected_epoch = (1, 1, 2, 2, 2, 3)
    expected_phone = (True, True, True, False, False, False)
    expected_cuda = (True, True, True, True, True, False)
    for index, child in enumerate(children):
        require(child.is_file() and not child.is_symlink(), "journal: unsafe file")
        value, raw = read_canonical(child, f"journal.{child.name}")
        exact_keys(value, record_keys, f"journal.{child.name}")
        require(value["schema"] == JOURNAL_SCHEMA, "journal: schema mismatch")
        require(value["record_index"] == index, "journal: index mismatch")
        require(
            value["previous_record_sha256"] == previous,
            "journal: hash chain mismatch",
        )
        require(
            value["transaction_id"] == transaction_id,
            "journal: transaction mismatch",
        )
        require(value["run_id"] == run_id, "journal: run ID mismatch")
        require(
            value["model_sha256"] == base_contract.model_sha256,
            "journal: model mismatch",
        )
        require(
            value["request_ids"] == list(base_contract.prompt_ids),
            "journal: request set mismatch",
        )
        require(
            value["phase"] == contract.journal_phases[index],
            "journal: phase mismatch",
        )
        require(value["owner"] == expected_owner[index], "journal: owner mismatch")
        require(
            value["owner_epoch"] == expected_epoch[index],
            "journal: owner epoch mismatch",
        )
        require(
            value["phone_active"] is expected_phone[index]
            and value["cuda_active"] is expected_cuda[index],
            "journal: active-state mismatch",
        )
        checked_digest(
            value["token_history_sha256"],
            f"journal.{child.name}.token_history_sha256",
        )
        require(
            is_int(value["published_tokens_per_request"])
            and value["published_tokens_per_request"] >= 0,
            "journal: invalid published count",
        )
        digest = sha256(raw)
        entries.append(JournalEntry(child.name, digest, value))
        previous = digest
    return entries


def transaction_id(
    delta_contract: DeltaContract,
    base_contract: w5.Contract,
    run_id: str,
) -> str:
    checked_digest(run_id, "run_id")
    return sha256(canonical({
        "base_contract_sha256": base_contract.raw_sha256,
        "delta_contract_sha256": delta_contract.raw_sha256,
        "model_sha256": base_contract.model_sha256,
        "prompt_ids": list(base_contract.prompt_ids),
        "run_id": run_id,
    }))


def replay_history_only(
    client: StageV3Client,
    histories: Sequence[Sequence[int]],
    identity_base: int,
    chunk: int,
) -> tuple[list[int], BatchMetrics]:
    require(bool(histories), "snapshot replay has no histories")
    require(chunk > 0, "snapshot chunk must be positive")
    width = len(histories[0])
    require(width > 0, "snapshot history is empty")
    batches = 0
    rows_total = 0
    elapsed_us = 0
    predictions: list[int] = []
    for start in range(0, width, chunk):
        end = min(start + chunk, width)
        rows = w5.build_rows(histories, identity_base, start, end)
        results, batch_us = w5.timed_batch(client, rows)
        batches += 1
        rows_total += len(rows)
        elapsed_us += batch_us
        predictions = w5.select_predictions(
            results,
            len(histories),
            end - start,
            identity_base,
            end - 1,
        )
    return predictions, BatchMetrics(batches, rows_total, elapsed_us)


def advance_active(
    client: StageV3Client,
    predictions: Sequence[int],
    identity_base: int,
    position_start: int,
    token_count: int,
) -> tuple[list[list[int]], list[int], BatchMetrics]:
    require(bool(predictions), "active advance has no predictions")
    require(position_start >= 0, "active advance has invalid position")
    require(token_count > 0, "active advance token count must be positive")
    batch = len(predictions)
    current = list(predictions)
    outputs = [[] for _ in predictions]
    elapsed_us = 0
    rows_total = 0
    for offset in range(token_count):
        position = position_start + offset
        rows = [
            BatchRow(
                identity_base + seq_id,
                identity_base + seq_id,
                seq_id,
                position,
                token,
            )
            for seq_id, token in enumerate(current)
        ]
        results, batch_us = w5.timed_batch(client, rows)
        elapsed_us += batch_us
        rows_total += len(rows)
        current = w5.select_predictions(
            results,
            batch,
            1,
            identity_base,
            position,
        )
        for seq_id, token in enumerate(current):
            outputs[seq_id].append(token)
    return outputs, current, BatchMetrics(token_count, rows_total, elapsed_us)


def feed_known_tokens(
    client: StageV3Client,
    tokens: Sequence[Sequence[int]],
    identity_base: int,
    position_start: int,
    chunk: int,
) -> tuple[list[int], BatchMetrics]:
    require(bool(tokens), "delta feed has no sequences")
    width = len(tokens[0])
    require(
        width > 0
        and all(
            len(row) == width
            and all(is_int(token) and token >= 0 for token in row)
            for row in tokens
        ),
        "delta tokens are invalid",
    )
    require(position_start >= 0 and chunk > 0, "delta feed range is invalid")
    batch = len(tokens)
    predictions: list[int] = []
    batches = 0
    rows_total = 0
    elapsed_us = 0
    for offset_start in range(0, width, chunk):
        offset_end = min(offset_start + chunk, width)
        rows = [
            BatchRow(
                identity_base + seq_id,
                identity_base + seq_id,
                seq_id,
                position_start + offset,
                row[offset],
            )
            for seq_id, row in enumerate(tokens)
            for offset in range(offset_start, offset_end)
        ]
        results, batch_us = w5.timed_batch(client, rows)
        batches += 1
        rows_total += len(rows)
        elapsed_us += batch_us
        predictions = w5.select_predictions(
            results,
            batch,
            offset_end - offset_start,
            identity_base,
            position_start + offset_end - 1,
        )
    return predictions, BatchMetrics(batches, rows_total, elapsed_us)


def continue_from_prediction(
    client: StageV3Client,
    prediction: Sequence[int],
    identity_base: int,
    position_start: int,
    token_count: int,
) -> tuple[list[list[int]], BatchMetrics]:
    require(bool(prediction), "CUDA continuation has no prediction")
    require(token_count > 0, "CUDA continuation count must be positive")
    outputs = [[token] for token in prediction]
    current = list(prediction)
    elapsed_us = 0
    rows_total = 0
    for offset in range(1, token_count):
        position = position_start + offset - 1
        rows = [
            BatchRow(
                identity_base + seq_id,
                identity_base + seq_id,
                seq_id,
                position,
                token,
            )
            for seq_id, token in enumerate(current)
        ]
        results, batch_us = w5.timed_batch(client, rows)
        elapsed_us += batch_us
        rows_total += len(rows)
        current = w5.select_predictions(
            results,
            len(prediction),
            1,
            identity_base,
            position,
        )
        for seq_id, token in enumerate(current):
            outputs[seq_id].append(token)
    return outputs, BatchMetrics(token_count - 1, rows_total, elapsed_us)


def run_concurrently(
    phone_operation: Callable[[], T],
    cuda_operation: Callable[[], Any],
    timeout_s: float,
) -> tuple[T, Any, dict[str, int]]:
    require(timeout_s > 0, "concurrent timeout must be positive")
    barrier = threading.Barrier(3)
    values: dict[str, Any] = {}
    legs: dict[str, ConcurrentLeg] = {}
    errors: dict[str, BaseException] = {}

    def worker(name: str, operation: Callable[[], Any]) -> None:
        try:
            barrier.wait(timeout=timeout_s)
            started_ns = time.monotonic_ns()
            values[name] = operation()
            ended_ns = time.monotonic_ns()
            legs[name] = ConcurrentLeg(started_ns, ended_ns)
        except BaseException as exc:
            errors[name] = exc

    threads = [
        threading.Thread(
            target=worker,
            args=("phone", phone_operation),
            daemon=True,
        ),
        threading.Thread(
            target=worker,
            args=("cuda", cuda_operation),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()
    try:
        barrier.wait(timeout=timeout_s)
    except threading.BrokenBarrierError as exc:
        raise DeltaError("concurrent start barrier failed") from exc
    for thread in threads:
        thread.join(timeout=timeout_s)
    require(not any(thread.is_alive() for thread in threads), "concurrent leg timed out")
    if errors:
        name = sorted(errors)[0]
        raise DeltaError(f"concurrent {name} leg failed: {errors[name]}") from errors[name]
    require(set(values) == {"phone", "cuda"}, "concurrent result is incomplete")
    phone = legs["phone"]
    cuda = legs["cuda"]
    overlap_ns = max(
        0,
        min(phone.ended_ns, cuda.ended_ns)
        - max(phone.started_ns, cuda.started_ns),
    )
    phone_ns = phone.ended_ns - phone.started_ns
    cuda_ns = cuda.ended_ns - cuda.started_ns
    shorter_ns = min(phone_ns, cuda_ns)
    require(shorter_ns > 0, "concurrent leg duration is zero")
    overlap_ppm = overlap_ns * 1_000_000 // shorter_ns
    timing = {
        "cuda_ended_ns": cuda.ended_ns,
        "cuda_started_ns": cuda.started_ns,
        "cuda_wall_ns": cuda_ns,
        "overlap_ns": overlap_ns,
        "overlap_shorter_ppm": overlap_ppm,
        "phone_ended_ns": phone.ended_ns,
        "phone_started_ns": phone.started_ns,
        "phone_wall_ns": phone_ns,
        "shorter_ns": shorter_ns,
    }
    return values["phone"], values["cuda"], timing


def commit_cutover(
    journal: OwnershipJournal,
    *,
    published_tokens: int,
    history_sha256: str,
    remove_phone: Callable[[], None],
) -> None:
    journal.append(
        phase="CUDA_COMMITTED",
        owner="CUDA",
        owner_epoch=2,
        published_tokens_per_request=published_tokens,
        token_history_sha256=history_sha256,
        phone_active=True,
        cuda_active=True,
    )
    remove_phone()
    journal.append(
        phase="PHONE_RELEASED",
        owner="CUDA",
        owner_epoch=2,
        published_tokens_per_request=published_tokens,
        token_history_sha256=history_sha256,
        phone_active=False,
        cuda_active=True,
    )


def journal_summary(entries: Sequence[JournalEntry]) -> dict[str, object]:
    return {
        "record_count": len(entries),
        "records": [
            {"name": entry.name, "sha256": entry.sha256}
            for entry in entries
        ],
        "terminal_owner": entries[-1].value["owner"] if entries else None,
        "transaction_id": (
            entries[-1].value["transaction_id"] if entries else None
        ),
    }


def validate_report(
    report: object,
    contract: DeltaContract,
    base_contract: w5.Contract,
    journal_dir: Path,
    expected_run_id: str | None = None,
) -> None:
    required = {
        "base_contract_sha256",
        "batch",
        "concurrency",
        "contract_sha256",
        "corpus_manifest_sha256",
        "corpus_sha256",
        "hellos",
        "journal",
        "metrics",
        "model_sha256",
        "phone_delta_tokens",
        "phone_snapshot_tokens",
        "prompts",
        "run_id",
        "scheduler_eligible",
        "schema",
        "scope",
        "sequences",
        "state_counts",
        "status",
    }
    root = exact_keys(report, required, "report")
    require(root["schema"] == SCHEMA, "report: schema mismatch")
    require(root["scope"] == "MECHANICS_ONLY", "report: scope mismatch")
    require(root["scheduler_eligible"] is False, "report: eligibility mismatch")
    require(root["contract_sha256"] == contract.raw_sha256, "report: contract mismatch")
    require(
        root["base_contract_sha256"] == base_contract.raw_sha256,
        "report: base contract mismatch",
    )
    require(root["model_sha256"] == base_contract.model_sha256, "report: model mismatch")
    require(root["corpus_sha256"] == base_contract.corpus_sha256, "report: corpus mismatch")
    require(
        root["corpus_manifest_sha256"] == base_contract.manifest_sha256,
        "report: corpus manifest mismatch",
    )
    require(root["batch"] == contract.batch, "report: batch mismatch")
    require(root["prompts"] == list(base_contract.prompt_ids), "report: prompt mismatch")
    report_run_id = checked_digest(root["run_id"], "report.run_id")
    if expected_run_id is not None:
        require(report_run_id == expected_run_id, "report: run ID mismatch")
    require(
        root["phone_snapshot_tokens"] == contract.phone_snapshot_tokens,
        "report: snapshot count mismatch",
    )
    require(
        root["phone_delta_tokens"] == contract.phone_delta_tokens,
        "report: delta count mismatch",
    )

    hellos = exact_keys(root["hellos"], {"cuda", "phone"}, "report.hellos")
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
        hello_value = exact_keys(value, hello_keys, f"report.hellos.{name}")
        for field in hello_keys - {"model_sha256"}:
            require(is_int(hello_value[field]), f"report: invalid {name} hello")
        require(type(hello_value["model_sha256"]) is str, "report: invalid model ID")
        hello = Hello(**hello_value)
        w5.validate_hello(name, hello, base_contract)
        parsed_hellos[name] = hello
    try:
        require_same_model(
            parsed_hellos,
            expected_model_sha256=base_contract.model_sha256,
            expected_file_type=base_contract.file_type,
        )
    except ProtocolError as exc:
        raise DeltaError(f"report: {exc}") from exc

    concurrency = exact_keys(
        root["concurrency"],
        {
            "cuda_ended_ns",
            "cuda_started_ns",
            "cuda_wall_ns",
            "overlap_ns",
            "overlap_shorter_ppm",
            "phone_ended_ns",
            "phone_started_ns",
            "phone_wall_ns",
            "shorter_ns",
        },
        "report.concurrency",
    )
    require(
        all(is_int(value) and value >= 0 for value in concurrency.values()),
        "report: invalid concurrency timing",
    )
    phone_ns = concurrency["phone_ended_ns"] - concurrency["phone_started_ns"]
    cuda_ns = concurrency["cuda_ended_ns"] - concurrency["cuda_started_ns"]
    overlap_ns = max(
        0,
        min(concurrency["phone_ended_ns"], concurrency["cuda_ended_ns"])
        - max(concurrency["phone_started_ns"], concurrency["cuda_started_ns"]),
    )
    shorter_ns = min(phone_ns, cuda_ns)
    require(
        phone_ns > 0
        and cuda_ns > 0
        and concurrency["phone_wall_ns"] == phone_ns
        and concurrency["cuda_wall_ns"] == cuda_ns
        and concurrency["overlap_ns"] == overlap_ns
        and concurrency["shorter_ns"] == shorter_ns
        and concurrency["overlap_shorter_ppm"]
        == overlap_ns * 1_000_000 // shorter_ns,
        "report: concurrency accounting mismatch",
    )
    overlap_pass = (
        concurrency["overlap_shorter_ppm"]
        >= contract.min_overlap_shorter_ppm
    )

    sequences = root["sequences"]
    require(
        type(sequences) is list and len(sequences) == contract.batch,
        "report: sequence count mismatch",
    )
    frontier_histories: list[list[int]] = []
    final_histories: list[list[int]] = []
    continuation_exact = True
    for index, value in enumerate(sequences):
        sequence = exact_keys(
            value,
            {
                "control_continuation",
                "cuda_continuation",
                "final_published_tokens",
                "phone_delta",
                "phone_published_tokens",
                "phone_snapshot",
                "prompt_id",
                "prompt_tokens",
                "sequence_index",
            },
            f"report.sequences.{index}",
        )
        require(sequence["sequence_index"] == index, "report: sequence order mismatch")
        require(
            sequence["prompt_id"] == base_contract.prompt_ids[index],
            "report: prompt ID mismatch",
        )
        widths = {
            "prompt_tokens": base_contract.prompt_tokens,
            "phone_snapshot": contract.phone_snapshot_tokens,
            "phone_delta": contract.phone_delta_tokens,
            "cuda_continuation": contract.cuda_continuation_tokens,
            "control_continuation": contract.cuda_continuation_tokens,
        }
        for field, width in widths.items():
            tokens = sequence[field]
            require(
                type(tokens) is list
                and len(tokens) == width
                and all(is_int(token) and token >= 0 for token in tokens),
                f"report: invalid {field}",
            )
        phone_tokens = sequence["phone_snapshot"] + sequence["phone_delta"]
        require(
            sequence["phone_published_tokens"] == phone_tokens,
            "report: phone publication gap or duplicate",
        )
        require(
            sequence["final_published_tokens"]
            == phone_tokens + sequence["cuda_continuation"],
            "report: final publication gap or duplicate",
        )
        continuation_exact = (
            continuation_exact
            and sequence["cuda_continuation"]
            == sequence["control_continuation"]
        )
        frontier_histories.append(sequence["prompt_tokens"] + phone_tokens)
        final_histories.append(
            sequence["prompt_tokens"]
            + phone_tokens
            + sequence["cuda_continuation"]
        )

    metrics = exact_keys(
        root["metrics"],
        {
            "cuda_continuation",
            "cuda_control",
            "cuda_delta",
            "cuda_snapshot",
            "phone_delta",
            "phone_snapshot",
        },
        "report.metrics",
    )
    replay_keys = {
        "continuation_batches",
        "elapsed_us",
        "history_batches",
        "rows",
    }
    batch_keys = {"batches", "elapsed_us", "rows"}
    for name in ("cuda_control", "phone_snapshot"):
        exact_keys(metrics[name], replay_keys, f"report.metrics.{name}")
    for name in (
        "cuda_continuation",
        "cuda_delta",
        "cuda_snapshot",
        "phone_delta",
    ):
        exact_keys(metrics[name], batch_keys, f"report.metrics.{name}")
    for value in metrics.values():
        require(
            all(is_int(item) and item >= 0 for item in value.values()),
            "report: invalid metric",
        )
    prompt_width = base_contract.prompt_tokens
    snapshot_width = prompt_width + contract.phone_snapshot_tokens
    frontier_width = snapshot_width + contract.phone_delta_tokens
    continuation_batches = contract.cuda_continuation_tokens - 1
    expected = {
        "phone_snapshot": {
            "history_batches": (
                prompt_width + base_contract.phone_prefill_chunk - 1
            ) // base_contract.phone_prefill_chunk,
            "continuation_batches": contract.phone_snapshot_tokens - 1,
            "rows": contract.batch
            * (prompt_width + contract.phone_snapshot_tokens - 1),
        },
        "phone_delta": {
            "batches": contract.phone_delta_tokens,
            "rows": contract.batch * contract.phone_delta_tokens,
        },
        "cuda_snapshot": {
            "batches": (
                snapshot_width + contract.cuda_snapshot_chunk - 1
            ) // contract.cuda_snapshot_chunk,
            "rows": contract.batch * snapshot_width,
        },
        "cuda_delta": {
            "batches": (
                contract.phone_delta_tokens + contract.cuda_delta_chunk - 1
            ) // contract.cuda_delta_chunk,
            "rows": contract.batch * contract.phone_delta_tokens,
        },
        "cuda_continuation": {
            "batches": continuation_batches,
            "rows": contract.batch * continuation_batches,
        },
        "cuda_control": {
            "history_batches": (
                frontier_width + contract.cuda_control_chunk - 1
            ) // contract.cuda_control_chunk,
            "continuation_batches": continuation_batches,
            "rows": contract.batch
            * (frontier_width + continuation_batches),
        },
    }
    for name, fields in expected.items():
        require(
            all(metrics[name][field] == value for field, value in fields.items()),
            f"report: {name} accounting mismatch",
        )
    fewer_catchup_batches = (
        metrics["cuda_snapshot"]["batches"] + metrics["cuda_delta"]["batches"]
        < metrics["cuda_control"]["history_batches"]
    )

    state_counts = root["state_counts"]
    require(
        state_counts == {
            "cuda_after_completion": 0,
            "cuda_after_snapshot": contract.batch,
            "cuda_control_released": 0,
            "cuda_prepared": contract.batch,
            "phone_after_commit": 0,
            "phone_at_frontier": contract.batch,
        },
        "report: state count mismatch",
    )

    tx_id = transaction_id(contract, base_contract, report_run_id)
    entries = load_journal(
        journal_dir,
        contract,
        base_contract,
        tx_id,
        report_run_id,
    )
    frontier_sha = w5.histories_digest(frontier_histories)
    final_sha = w5.histories_digest(final_histories)
    expected_published = (
        contract.phone_snapshot_tokens + contract.phone_delta_tokens
    )
    for index, entry in enumerate(entries):
        expected_history = final_sha if index >= 4 else frontier_sha
        expected_count = (
            expected_published + contract.cuda_continuation_tokens
            if index >= 4
            else expected_published
        )
        require(
            entry.value["token_history_sha256"] == expected_history
            and entry.value["published_tokens_per_request"] == expected_count,
            "report: journal frontier mismatch",
        )
    require(
        root["journal"] == journal_summary(entries),
        "report: journal summary mismatch",
    )

    pass_status = (
        continuation_exact
        and overlap_pass
        and fewer_catchup_batches
        and entries[-1].value["owner"] == "NONE"
    )
    expected_status = (
        "CONCURRENT_DELTA_MECHANICS_PASS"
        if pass_status
        else "CONCURRENT_DELTA_MECHANICS_FAIL"
    )
    require(root["status"] == expected_status, "report: status mismatch")


def write_atomic(path: Path, value: dict[str, object]) -> None:
    w5.write_atomic(path, value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--base-contract",
        type=Path,
        default=DEFAULT_BASE_CONTRACT,
    )
    parser.add_argument("--phone-route", type=parse_endpoint, required=True)
    parser.add_argument("--cuda-route", type=parse_endpoint, required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--journal-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.timeout <= 0
        or args.output.exists()
        or args.journal_dir.exists()
        or HEX64.fullmatch(args.run_id) is None
    ):
        parser.error("invalid timeout or existing output")

    clients: dict[str, StageV3Client] = {}
    stopped: set[str] = set()
    active: dict[str, tuple[int, int] | None] = {
        "phone": None,
        "cuda": None,
    }
    try:
        base_contract = w5.load_contract(args.base_contract)
        contract = load_contract(args.contract, base_contract)
        corpus, _, _ = load_corpus(
            base_contract.corpus_path,
            base_contract.manifest_path,
            base_contract.model_sha256,
        )
        selected = [corpus[prompt_id] for prompt_id in base_contract.prompt_ids]
        prompts = [record["tokens"] for record in selected]
        tx_id = transaction_id(contract, base_contract, args.run_id)
        journal = OwnershipJournal(
            args.journal_dir,
            tx_id,
            base_contract.model_sha256,
            base_contract.prompt_ids,
            args.run_id,
        )

        clients["phone"] = StageV3Client.connect(*args.phone_route, args.timeout)
        clients["cuda"] = StageV3Client.connect(*args.cuda_route, args.timeout)
        hellos = {name: client.hello() for name, client in clients.items()}
        for name, hello in hellos.items():
            w5.validate_hello(name, hello, base_contract)
        require_same_model(
            hellos,
            expected_model_sha256=base_contract.model_sha256,
            expected_file_type=base_contract.file_type,
        )

        phone_base = 10000
        catchup_base = 30000
        control_base = 40000
        snapshot_tokens, snapshot_metrics = w5.run_replay(
            clients["phone"],
            prompts,
            phone_base,
            base_contract.phone_prefill_chunk,
            contract.phone_snapshot_tokens,
        )
        active["phone"] = (contract.batch, phone_base)
        phone_snapshot_status = clients["phone"].status()
        require(
            phone_snapshot_status.active_sequences == contract.batch,
            "phone snapshot state count mismatch",
        )
        snapshot_histories = [
            list(prompt) + list(snapshot)
            for prompt, snapshot in zip(prompts, snapshot_tokens)
        ]

        active["cuda"] = (contract.batch, catchup_base)
        phone_result, cuda_result, concurrency = run_concurrently(
            lambda: advance_active(
                clients["phone"],
                [tokens[-1] for tokens in snapshot_tokens],
                phone_base,
                base_contract.prompt_tokens
                + contract.phone_snapshot_tokens
                - 1,
                contract.phone_delta_tokens,
            ),
            lambda: replay_history_only(
                clients["cuda"],
                snapshot_histories,
                catchup_base,
                contract.cuda_snapshot_chunk,
            ),
            args.timeout,
        )
        delta_tokens, _, phone_delta_metrics = phone_result
        _, cuda_snapshot_metrics = cuda_result
        require(
            concurrency["overlap_shorter_ppm"]
            >= contract.min_overlap_shorter_ppm,
            "concurrent overlap gate failed",
        )
        phone_frontier_status = clients["phone"].status()
        cuda_snapshot_status = clients["cuda"].status()
        require(
            phone_frontier_status.active_sequences == contract.batch,
            "phone frontier state count mismatch",
        )
        require(
            cuda_snapshot_status.active_sequences == contract.batch,
            "CUDA snapshot state count mismatch",
        )
        frontier_histories = [
            list(prompt) + list(snapshot) + list(delta)
            for prompt, snapshot, delta in zip(
                prompts,
                snapshot_tokens,
                delta_tokens,
            )
        ]
        frontier_sha = w5.histories_digest(frontier_histories)
        published_phone_tokens = (
            contract.phone_snapshot_tokens + contract.phone_delta_tokens
        )
        journal.append(
            phase="PHONE_FRONTIER",
            owner="PHONE",
            owner_epoch=1,
            published_tokens_per_request=published_phone_tokens,
            token_history_sha256=frontier_sha,
            phone_active=True,
            cuda_active=True,
        )

        cuda_prediction, cuda_delta_metrics = feed_known_tokens(
            clients["cuda"],
            delta_tokens,
            catchup_base,
            base_contract.prompt_tokens + contract.phone_snapshot_tokens,
            contract.cuda_delta_chunk,
        )
        cuda_prepared_status = clients["cuda"].status()
        require(
            cuda_prepared_status.active_sequences == contract.batch,
            "CUDA prepared state count mismatch",
        )
        journal.append(
            phase="CUDA_PREPARED",
            owner="PHONE",
            owner_epoch=1,
            published_tokens_per_request=published_phone_tokens,
            token_history_sha256=frontier_sha,
            phone_active=True,
            cuda_active=True,
        )

        def remove_phone() -> None:
            w5.remove_group(clients["phone"], contract.batch, phone_base)

        commit_cutover(
            journal,
            published_tokens=published_phone_tokens,
            history_sha256=frontier_sha,
            remove_phone=remove_phone,
        )
        active["phone"] = None
        phone_after_commit = clients["phone"].status()
        require(
            phone_after_commit.active_sequences == 0,
            "phone state leak after durable commit",
        )

        cuda_tokens, cuda_continuation_metrics = continue_from_prediction(
            clients["cuda"],
            cuda_prediction,
            catchup_base,
            base_contract.prompt_tokens
            + contract.phone_snapshot_tokens
            + contract.phone_delta_tokens,
            contract.cuda_continuation_tokens,
        )
        final_histories = [
            frontier + list(continuation)
            for frontier, continuation in zip(
                frontier_histories,
                cuda_tokens,
            )
        ]
        final_sha = w5.histories_digest(final_histories)
        final_published = (
            published_phone_tokens + contract.cuda_continuation_tokens
        )
        journal.append(
            phase="CUDA_CONTINUATION",
            owner="CUDA",
            owner_epoch=2,
            published_tokens_per_request=final_published,
            token_history_sha256=final_sha,
            phone_active=False,
            cuda_active=True,
        )
        w5.remove_group(clients["cuda"], contract.batch, catchup_base)
        active["cuda"] = None
        cuda_after_completion = clients["cuda"].status()
        require(
            cuda_after_completion.active_sequences == 0,
            "CUDA state leak after completion",
        )
        journal.append(
            phase="COMPLETE",
            owner="NONE",
            owner_epoch=3,
            published_tokens_per_request=final_published,
            token_history_sha256=final_sha,
            phone_active=False,
            cuda_active=False,
        )

        control_tokens, control_metrics = w5.run_replay(
            clients["cuda"],
            frontier_histories,
            control_base,
            contract.cuda_control_chunk,
            contract.cuda_continuation_tokens,
        )
        active["cuda"] = (contract.batch, control_base)
        w5.remove_group(clients["cuda"], contract.batch, control_base)
        active["cuda"] = None
        cuda_control_released = clients["cuda"].status()
        require(
            cuda_control_released.active_sequences == 0,
            "CUDA control state leak",
        )

        w5.finish(clients["phone"], "stop")
        stopped.add("phone")
        w5.finish(clients["cuda"], "stop")
        stopped.add("cuda")

        entries = load_journal(
            args.journal_dir,
            contract,
            base_contract,
            tx_id,
            args.run_id,
        )
        sequences = [
            {
                "control_continuation": list(control_tokens[index]),
                "cuda_continuation": list(cuda_tokens[index]),
                "final_published_tokens": (
                    list(snapshot_tokens[index])
                    + list(delta_tokens[index])
                    + list(cuda_tokens[index])
                ),
                "phone_delta": list(delta_tokens[index]),
                "phone_published_tokens": (
                    list(snapshot_tokens[index])
                    + list(delta_tokens[index])
                ),
                "phone_snapshot": list(snapshot_tokens[index]),
                "prompt_id": base_contract.prompt_ids[index],
                "prompt_tokens": list(prompts[index]),
                "sequence_index": index,
            }
            for index in range(contract.batch)
        ]
        exact = cuda_tokens == control_tokens
        fewer_batches = (
            cuda_snapshot_metrics.batches + cuda_delta_metrics.batches
            < control_metrics.history_batches
        )
        report: dict[str, object] = {
            "base_contract_sha256": base_contract.raw_sha256,
            "batch": contract.batch,
            "concurrency": concurrency,
            "contract_sha256": contract.raw_sha256,
            "corpus_manifest_sha256": base_contract.manifest_sha256,
            "corpus_sha256": base_contract.corpus_sha256,
            "hellos": {name: asdict(hello) for name, hello in hellos.items()},
            "journal": journal_summary(entries),
            "metrics": {
                "cuda_continuation": asdict(cuda_continuation_metrics),
                "cuda_control": asdict(control_metrics),
                "cuda_delta": asdict(cuda_delta_metrics),
                "cuda_snapshot": asdict(cuda_snapshot_metrics),
                "phone_delta": asdict(phone_delta_metrics),
                "phone_snapshot": asdict(snapshot_metrics),
            },
            "model_sha256": base_contract.model_sha256,
            "phone_delta_tokens": contract.phone_delta_tokens,
            "phone_snapshot_tokens": contract.phone_snapshot_tokens,
            "prompts": list(base_contract.prompt_ids),
            "run_id": args.run_id,
            "scheduler_eligible": False,
            "schema": SCHEMA,
            "scope": "MECHANICS_ONLY",
            "sequences": sequences,
            "state_counts": {
                "cuda_after_completion": (
                    cuda_after_completion.active_sequences
                ),
                "cuda_after_snapshot": (
                    cuda_snapshot_status.active_sequences
                ),
                "cuda_control_released": (
                    cuda_control_released.active_sequences
                ),
                "cuda_prepared": cuda_prepared_status.active_sequences,
                "phone_after_commit": phone_after_commit.active_sequences,
                "phone_at_frontier": (
                    phone_frontier_status.active_sequences
                ),
            },
            "status": (
                "CONCURRENT_DELTA_MECHANICS_PASS"
                if exact
                and fewer_batches
                and concurrency["overlap_shorter_ppm"]
                >= contract.min_overlap_shorter_ppm
                else "CONCURRENT_DELTA_MECHANICS_FAIL"
            ),
        }
        validate_report(
            report,
            contract,
            base_contract,
            args.journal_dir,
            args.run_id,
        )
        write_atomic(args.output, report)
        print(canonical(report).decode("ascii"), end="")
        return 0 if report["status"] == "CONCURRENT_DELTA_MECHANICS_PASS" else 3
    finally:
        for name, client in clients.items():
            state = active.get(name)
            if state is not None:
                batch, identity_base = state
                try:
                    w5.remove_group(client, batch, identity_base)
                except (OSError, ProtocolError, w5.HandoffError):
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
    except (
        OSError,
        ProtocolError,
        DeltaError,
        w5.HandoffError,
        ValueError,
    ) as exc:
        print(canonical({
            "error": str(exc),
            "status": "CONCURRENT_DELTA_MECHANICS_ERROR",
        }).decode("ascii"), end="")
        raise SystemExit(2)
