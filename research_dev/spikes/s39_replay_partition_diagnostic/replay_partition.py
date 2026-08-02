#!/usr/bin/env python3
"""Shared mechanics for the frozen CUDA replay-partition diagnostic."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
S39 = HERE.parent / "s39_phone_model_switch_trace"
S22 = HERE.parent / "s22_slo_overlap_pipeline"
for module_path in (S39, S22):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import phone_cuda_delta_probe as w6
import phone_cuda_handoff_probe as w5
from stage_v3_client import BatchResult, BatchRow, Hello, StageV3Client


CONTRACT_SCHEMA = "s39-replay-partition-diagnostic-contract-v1"
INPUT_SCHEMA = "s39-replay-partition-input-v1"
RAW_SCHEMA = "s39-replay-partition-raw-v1"
RUN_RECORD_SCHEMA = "s39-replay-partition-run-record-v1"
ANALYSIS_SCHEMA = "s39-replay-partition-analysis-v1"
HEX64 = re.compile(r"[0-9a-f]{64}")


class DiagnosticError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


def is_int(value: object) -> bool:
    return type(value) is int


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_canonical(path: Path, field: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw, object_pairs_hook=strict_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"{field}: invalid JSON") from exc
    require(type(value) is dict, f"{field}: expected object")
    require(canonical(value) == raw, f"{field}: not canonical JSON")
    return value, raw


def write_atomic(path: Path, value: dict[str, object]) -> None:
    raw = canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def checked_digest(value: object, field: str) -> str:
    require(
        type(value) is str and HEX64.fullmatch(value) is not None,
        f"{field}: invalid SHA-256",
    )
    return value


def repo_path(value: object, field: str) -> Path:
    require(type(value) is str and value != "", f"{field}: invalid path")
    relative = Path(value)
    require(
        not relative.is_absolute() and ".." not in relative.parts,
        f"{field}: path must be repository-relative",
    )
    resolved = (ROOT / relative).resolve()
    require(ROOT == resolved or ROOT in resolved.parents, f"{field}: path escape")
    return resolved


def host_path(value: object, field: str) -> Path:
    require(type(value) is str and value != "", f"{field}: invalid path")
    path = Path(value)
    return path.resolve() if path.is_absolute() else repo_path(value, field)


def verify_sha256_manifest(path: Path, field: str) -> None:
    base = path.parent.resolve()
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise DiagnosticError(f"{field}: unreadable manifest") from exc
    require(bool(lines), f"{field}: empty manifest")
    for line in lines:
        digest, separator, name = line.partition("  ")
        require(
            separator == "  "
            and HEX64.fullmatch(digest) is not None
            and name not in seen,
            f"{field}: invalid manifest line",
        )
        seen.add(name)
        target = (base / name).resolve()
        require(
            base in target.parents and target.is_file(),
            f"{field}: invalid manifest target: {name}",
        )
        require(
            sha256(target.read_bytes()) == digest,
            f"{field}: changed manifest target: {name}",
        )


def _check_run_spec(value: object, index: int) -> dict[str, Any]:
    field = f"contract.execution.runs.{index}"
    require(type(value) is dict, f"{field}: expected object")
    required = {"fresh_process", "name", "paths"}
    require(set(value) == required, f"{field}: key mismatch")
    require(
        type(value["name"]) is str and value["name"] != "",
        f"{field}: invalid name",
    )
    require(type(value["fresh_process"]) is bool, f"{field}: invalid fresh flag")
    paths = value["paths"]
    require(type(paths) is list and len(paths) in {1, 2}, f"{field}: invalid paths")
    for path_index, path in enumerate(paths):
        path_field = f"{field}.paths.{path_index}"
        require(type(path) is dict, f"{path_field}: expected object")
        require(
            set(path)
            == {
                "delta_chunk",
                "history",
                "identity_base",
                "name",
                "replay_chunk",
            },
            f"{path_field}: key mismatch",
        )
        require(
            path["name"] in {"incremental_8_3_plus_1", "full_f1_chunk2", "full_f1_8_4"}
            and path["history"] in {"F0_PLUS_DELTA", "F1"}
            and is_int(path["identity_base"])
            and path["identity_base"] >= 1000
            and is_int(path["replay_chunk"])
            and path["replay_chunk"] > 0
            and is_int(path["delta_chunk"])
            and path["delta_chunk"] >= 0,
            f"{path_field}: invalid path geometry",
        )
        if path["history"] == "F0_PLUS_DELTA":
            require(
                path["replay_chunk"] == 8 and path["delta_chunk"] == 1,
                f"{path_field}: incremental geometry changed",
            )
        elif path["name"] == "full_f1_chunk2":
            require(
                path["replay_chunk"] == 2 and path["delta_chunk"] == 0,
                f"{path_field}: chunk2 geometry changed",
            )
        else:
            require(
                path["replay_chunk"] == 8 and path["delta_chunk"] == 0,
                f"{path_field}: 8+4 geometry changed",
            )
    return value


def load_contract(path: Path) -> tuple[dict[str, Any], str]:
    value, raw = read_canonical(path, "contract")
    required = {
        "artifacts",
        "claim_boundary",
        "execution",
        "inputs",
        "requirements",
        "schema",
        "scope",
        "source_evidence",
        "source_sha256",
        "status",
    }
    require(set(value) == required, "contract: key mismatch")
    require(value["schema"] == CONTRACT_SCHEMA, "contract: schema mismatch")
    require(
        value["status"] == "FROZEN_BEFORE_CUDA_ACQUISITION",
        "contract: status mismatch",
    )
    require(value["scope"] == "CUDA_ONLY_DIAGNOSTIC", "contract: scope mismatch")
    require(
        value["claim_boundary"]
        == {
            "controller_integration": False,
            "energy": False,
            "performance": False,
            "scheduler_eligible": False,
            "task_quality": False,
        },
        "contract: claim boundary changed",
    )
    require(
        value["requirements"]
        == [
            "EXACT_W9_F0_F1_HISTORY_DIGESTS",
            "TWO_FRESH_INCREMENTAL_REPETITIONS",
            "TWO_FRESH_FULL_F1_CHUNK2_REPETITIONS",
            "ONE_INCREMENTAL_REMOVE_FULL_F1_SAME_PROCESS_SEQUENCE",
            "RAW_ROWS_AND_VECTORS_PERSIST_BEFORE_COMPARISON",
            "ZERO_SEQUENCE_STATE_AFTER_EACH_PATH",
            "CUDA0_ONLY_REALIZED_PLACEMENT",
            "NO_W9_MUTATION_OR_REPLACEMENT",
        ],
        "contract: requirements changed",
    )

    inputs = value["inputs"]
    require(type(inputs) is dict and set(inputs) == {"path", "sha256"}, "contract: inputs")
    inputs_path = repo_path(inputs["path"], "contract.inputs.path")
    require(inputs_path.is_file(), "contract: input file missing")
    require(
        sha256(inputs_path.read_bytes())
        == checked_digest(inputs["sha256"], "contract.inputs.sha256"),
        "contract: input digest mismatch",
    )

    artifacts = value["artifacts"]
    require(
        type(artifacts) is dict
        and set(artifacts)
        == {"cuda_uuid", "host_relay", "host_worker", "model"},
        "contract: artifacts",
    )
    require(
        artifacts["cuda_uuid"] == "GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f",
        "contract: CUDA UUID changed",
    )
    for key in ("host_relay", "host_worker", "model"):
        item = artifacts[key]
        require(
            type(item) is dict and set(item) == {"path", "sha256"},
            f"contract.artifacts.{key}: key mismatch",
        )
        checked_digest(item["sha256"], f"contract.artifacts.{key}.sha256")
        host_path(item["path"], f"contract.artifacts.{key}.path")

    evidence = value["source_evidence"]
    require(
        type(evidence) is dict
        and set(evidence)
        == {
            "f0_history_sha256",
            "f1_history_sha256",
            "w8_treatment_report",
            "w9_contract",
            "w9_final_ledger_record",
            "w9_root_manifest",
            "w9_run_id",
            "w9_transaction_id",
            "w9_continuation_sha256",
        },
        "contract: source evidence keys",
    )
    for field in (
        "f0_history_sha256",
        "f1_history_sha256",
        "w9_run_id",
        "w9_transaction_id",
        "w9_continuation_sha256",
    ):
        checked_digest(evidence[field], f"contract.source_evidence.{field}")
    for field in (
        "w8_treatment_report",
        "w9_contract",
        "w9_final_ledger_record",
        "w9_root_manifest",
    ):
        item = evidence[field]
        require(
            type(item) is dict and set(item) == {"path", "sha256"},
            f"contract.source_evidence.{field}: key mismatch",
        )
        source_path = repo_path(
            item["path"],
            f"contract.source_evidence.{field}.path",
        )
        require(source_path.is_file(), f"contract: evidence missing: {field}")
        require(
            sha256(source_path.read_bytes())
            == checked_digest(
                item["sha256"],
                f"contract.source_evidence.{field}.sha256",
            ),
            f"contract: evidence digest mismatch: {field}",
        )
        if field == "w9_root_manifest":
            verify_sha256_manifest(source_path, "contract.source_evidence.w9_root_manifest")

    sources = value["source_sha256"]
    require(type(sources) is dict and bool(sources), "contract: empty source map")
    for source, expected in sources.items():
        source_path = repo_path(source, f"contract.source_sha256.{source}")
        require(source_path.is_file(), f"contract: source missing: {source}")
        require(
            sha256(source_path.read_bytes())
            == checked_digest(expected, f"contract.source_sha256.{source}"),
            f"contract: source digest mismatch: {source}",
        )

    execution = value["execution"]
    require(
        type(execution) is dict
        and set(execution)
        == {
            "batch",
            "continuation_tokens",
            "include_optional_w5_geometry",
            "layer_split",
            "n_ctx_seq",
            "runs",
        },
        "contract: execution keys",
    )
    require(
        execution["batch"] == 8
        and execution["continuation_tokens"] == 11
        and execution["include_optional_w5_geometry"] is True
        and execution["layer_split"] == [30, 48]
        and execution["n_ctx_seq"] == 64,
        "contract: execution geometry changed",
    )
    runs = execution["runs"]
    require(type(runs) is list and len(runs) == 7, "contract: run count")
    checked_runs = [_check_run_spec(run, index) for index, run in enumerate(runs)]
    names = [run["name"] for run in checked_runs]
    require(
        names
        == [
            "fresh_incremental_r1",
            "fresh_incremental_r2",
            "fresh_full_chunk2_r1",
            "fresh_full_chunk2_r2",
            "same_process_incremental_then_full",
            "fresh_full_8_4_r1",
            "fresh_full_8_4_r2",
        ],
        "contract: run order changed",
    )
    require(
        all(len(run["paths"]) == 1 and run["fresh_process"] for run in checked_runs[:4])
        and len(checked_runs[4]["paths"]) == 2
        and checked_runs[4]["fresh_process"] is True
        and all(len(run["paths"]) == 1 and run["fresh_process"] for run in checked_runs[5:]),
        "contract: process isolation changed",
    )
    return value, sha256(raw)


def load_inputs(path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    value, raw = read_canonical(path, "inputs")
    required = {
        "batch",
        "f0_histories",
        "f0_history_sha256",
        "f1_histories",
        "f1_history_sha256",
        "schema",
        "sequences",
        "source",
        "w9_continuation",
        "w9_continuation_sha256",
    }
    require(set(value) == required, "inputs: key mismatch")
    require(value["schema"] == INPUT_SCHEMA, "inputs: schema mismatch")
    require(value["batch"] == contract["execution"]["batch"], "inputs: batch mismatch")
    require(
        sha256(raw) == contract["inputs"]["sha256"],
        "inputs: contract digest mismatch",
    )
    f0 = value["f0_histories"]
    f1 = value["f1_histories"]
    continuation = value["w9_continuation"]
    batch = contract["execution"]["batch"]
    count = contract["execution"]["continuation_tokens"]
    require(
        type(f0) is list
        and type(f1) is list
        and type(continuation) is list
        and len(f0) == len(f1) == len(continuation) == batch,
        "inputs: nonrectangular batch",
    )
    require(
        all(
            type(row) is list
            and len(row) == 11
            and all(is_int(token) and token >= 0 for token in row)
            for row in f0
        ),
        "inputs: invalid F0",
    )
    require(
        all(
            type(row) is list
            and len(row) == 12
            and row[:11] == f0[index]
            and all(is_int(token) and token >= 0 for token in row)
            for index, row in enumerate(f1)
        ),
        "inputs: invalid F1",
    )
    require(
        all(
            type(row) is list
            and len(row) == count
            and all(is_int(token) and token >= 0 for token in row)
            for row in continuation
        ),
        "inputs: invalid W9 continuation",
    )
    require(
        w5.histories_digest(f0) == value["f0_history_sha256"]
        == contract["source_evidence"]["f0_history_sha256"],
        "inputs: F0 digest mismatch",
    )
    require(
        w5.histories_digest(f1) == value["f1_history_sha256"]
        == contract["source_evidence"]["f1_history_sha256"],
        "inputs: F1 digest mismatch",
    )
    require(
        sha256(canonical(continuation)) == value["w9_continuation_sha256"]
        == contract["source_evidence"]["w9_continuation_sha256"],
        "inputs: W9 continuation digest mismatch",
    )
    return value


def validate_hello(hello: Hello, contract: dict[str, Any]) -> None:
    batch = contract["execution"]["batch"]
    split = contract["execution"]["layer_split"]
    require(
        hello.layer_start == 0
        and hello.layer_end == split[1]
        and hello.n_layer == split[1]
        and hello.max_streams >= batch
        and hello.n_ctx_seq >= contract["execution"]["n_ctx_seq"]
        and min(hello.n_batch, hello.n_ubatch) >= batch * 8
        and hello.model_sha256 == contract["artifacts"]["model"]["sha256"],
        "CUDA route identity or capacity mismatch",
    )


def row_record(row: BatchRow) -> dict[str, object]:
    return {
        "hidden": None if row.hidden is None else list(row.hidden),
        "position": row.position,
        "request_id": row.request_id,
        "route_epoch": row.route_epoch,
        "seq_id": row.seq_id,
        "token": row.token,
    }


def result_record(result: BatchResult) -> dict[str, object]:
    return {
        "hidden": None if result.hidden is None else list(result.hidden),
        "position": result.position,
        "request_id": result.request_id,
        "route_epoch": result.route_epoch,
        "seq_id": result.seq_id,
        "token": result.token,
    }


class RecordingClient:
    """Record complete protocol rows while delegating to an unchanged client."""

    def __init__(self, client: StageV3Client) -> None:
        self.client = client
        self.calls: list[dict[str, object]] = []
        self.phase = "UNSET"

    def set_phase(self, phase: str) -> None:
        require(type(phase) is str and phase != "", "invalid recording phase")
        self.phase = phase

    def batch(self, rows: Sequence[BatchRow]) -> tuple[BatchResult, ...]:
        require(self.phase != "UNSET", "recording phase not set")
        persisted_rows = [row_record(row) for row in rows]
        started_ns = time.monotonic_ns()
        results = self.client.batch(rows)
        ended_ns = time.monotonic_ns()
        counts: dict[int, int] = {}
        for row in rows:
            counts[row.seq_id] = counts.get(row.seq_id, 0) + 1
        self.calls.append({
            "call_index": len(self.calls),
            "ended_ns": ended_ns,
            "input_rows": persisted_rows,
            "output_rows": [result_record(result) for result in results],
            "phase": self.phase,
            "shape": {
                "positions": sorted({row.position for row in rows}),
                "row_count": len(rows),
                "rows_per_sequence": [
                    counts[seq_id] for seq_id in sorted(counts)
                ],
                "sequence_count": len(counts),
            },
            "started_ns": started_ns,
        })
        return results

    def status(self):
        return self.client.status()

    def remove(self, seq_id: int, request_id: int, route_epoch: int):
        return self.client.remove(seq_id, request_id, route_epoch)

    def drain(self):
        return self.client.drain()

    def stop(self) -> None:
        self.client.stop()

    def close(self) -> None:
        self.client.close()


def execute_path(
    client: RecordingClient,
    spec: dict[str, Any],
    inputs: dict[str, Any],
    continuation_tokens: int,
) -> dict[str, object]:
    before = client.status()
    require(before.active_sequences == 0, "path starts with live CUDA state")
    call_start = len(client.calls)
    identity_base = spec["identity_base"]
    if spec["history"] == "F0_PLUS_DELTA":
        client.set_phase("REPLAY_F0")
        prediction, replay_metrics = w6.replay_history_only(
            client,
            inputs["f0_histories"],
            identity_base,
            spec["replay_chunk"],
        )
        delta = [
            [f1[-1]]
            for f1 in inputs["f1_histories"]
        ]
        client.set_phase("INGEST_F1_MINUS_F0")
        prediction, delta_metrics = w6.feed_known_tokens(
            client,
            delta,
            identity_base,
            len(inputs["f0_histories"][0]),
            spec["delta_chunk"],
        )
    else:
        client.set_phase("REPLAY_F1")
        prediction, replay_metrics = w6.replay_history_only(
            client,
            inputs["f1_histories"],
            identity_base,
            spec["replay_chunk"],
        )
        delta_metrics = w6.BatchMetrics(0, 0, 0)

    initial_prediction = list(prediction)
    client.set_phase("AUTONOMOUS_CONTINUATION")
    continuation, continuation_metrics = w6.continue_from_prediction(
        client,
        prediction,
        identity_base,
        len(inputs["f1_histories"][0]),
        continuation_tokens,
    )
    before_remove = client.status()
    require(
        before_remove.active_sequences == len(inputs["f1_histories"]),
        "path state count before removal mismatch",
    )
    w5.remove_group(
        client,
        len(inputs["f1_histories"]),
        identity_base,
    )
    after = client.status()
    require(after.active_sequences == 0, "path leaked CUDA state")
    return {
        "call_end": len(client.calls),
        "call_start": call_start,
        "continuation": continuation,
        "continuation_metrics": asdict(continuation_metrics),
        "delta_metrics": asdict(delta_metrics),
        "history": spec["history"],
        "history_sha256": (
            inputs["f0_history_sha256"]
            if spec["history"] == "F0_PLUS_DELTA"
            else inputs["f1_history_sha256"]
        ),
        "identity_base": identity_base,
        "initial_prediction": initial_prediction,
        "name": spec["name"],
        "replay_metrics": asdict(replay_metrics),
        "state_counts": {
            "after_remove": after.active_sequences,
            "before": before.active_sequences,
            "before_remove": before_remove.active_sequences,
        },
    }


def first_mismatch(
    left: Sequence[Sequence[int]],
    right: Sequence[Sequence[int]],
    *,
    first_position: int = 12,
) -> dict[str, int] | None:
    require(len(left) == len(right), "mismatch: batch widths differ")
    for sequence_index, (left_row, right_row) in enumerate(zip(left, right)):
        require(len(left_row) == len(right_row), "mismatch: token widths differ")
        for offset, (left_token, right_token) in enumerate(zip(left_row, right_row)):
            if left_token != right_token:
                return {
                    "absolute_position": first_position + offset,
                    "left_token": left_token,
                    "right_token": right_token,
                    "sequence_index": sequence_index,
                    "token_offset": offset,
                }
    return None
