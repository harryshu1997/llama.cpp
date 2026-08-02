#!/usr/bin/env python3
"""Continue a live phone session while a cold CUDA route becomes ready."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import phone_cuda_cold_promotion_probe as cold
import phone_cuda_delta_probe as w6
import phone_cuda_handoff_probe as w5
from async_pipeline import parse_endpoint
from qwen25_quality_probe import load_corpus
from stage_v3_client import BatchRow, Hello, ProtocolError, StageV3Client


SCHEMA = "s39-live-session-promotion-treatment-v1"
CONTRACT_SCHEMA = "s39-live-session-promotion-contract-v1"
CONTRACT_R1_SCHEMA = "s39-live-session-promotion-contract-r1-v1"
START_SCHEMA = "s39-live-promotion-start-v1"
DEFAULT_CONTRACT = Path(__file__).resolve().parent / "W8_LIVE_SESSION_CONTRACT.json"


class LivePromotionError(RuntimeError):
    pass


@dataclass(frozen=True)
class LiveContract:
    raw_sha256: str
    base_contract_sha256: str
    delta_contract_sha256: str
    physical_gate_sha256: str
    batch: int
    connector_poll_us: int
    cuda_continuation_tokens: int
    cuda_control_chunk: int
    cuda_delta_chunk: int
    cuda_replay_chunk: int
    max_cuda_ready_us: int
    max_inter_batch_gap_us: int
    max_launch_delay_us: int
    max_phone_service_tokens: int
    max_rows_per_batch: int
    min_phone_service_tokens: int
    min_useful_phone_tokens_before_cuda_ready: int
    phone_delta_tokens: int
    phone_prefill_chunk: int
    preexisting_committed_tokens: int


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LivePromotionError(message)


def load_contract(
    path: Path,
    base_contract: w5.Contract,
    delta_contract: w6.DeltaContract,
    physical_gate_sha256: str,
) -> LiveContract:
    value, raw = w6.read_canonical(path, "live_contract")
    root = w6.exact_keys(
        value,
        {
            "base_contract_sha256",
            "control",
            "delta_contract_sha256",
            "execution",
            "physical_gate_sha256",
            "predecessor",
            "requirements",
            "scheduler_eligible_on_pass",
            "schema",
            "scope",
            "status",
        },
        "live_contract",
    )
    require(
        root["schema"] in (CONTRACT_SCHEMA, CONTRACT_R1_SCHEMA)
        and root["status"] == "FROZEN_BEFORE_ACQUISITION"
        and root["scope"] == "MECHANICS_ONLY"
        and root["scheduler_eligible_on_pass"] is False,
        "live_contract: labels",
    )
    require(
        root["base_contract_sha256"] == base_contract.raw_sha256
        and root["delta_contract_sha256"] == delta_contract.raw_sha256
        and root["physical_gate_sha256"] == physical_gate_sha256,
        "live_contract: dependency",
    )
    v1_predecessor = {
        "contract_sha256": (
            "8d7af69e91a6a336d4c7c0dcc8a95995"
            "2f7a26faf6fff6dc56dc607fa93e5e84"
        ),
        "failure_manifest_sha256": (
            "abc5fc99ae33a828e414a9348f3aeb27"
            "b2e32725c9441d508b9f9c420ae02a67"
        ),
        "failure_status": "FAIL_READINESS_ORDER",
    }
    r1_predecessor = {
        "contract_sha256": (
            "d6166038410db479e8e27036ff6d6912"
            "2d719b4bec81cd883a1ac0d00efeb788"
        ),
        "failure_manifest_sha256": (
            "02f772a49b24948f2a84c54e071041f"
            "2e1c88faa8d9054b556cc4a126839a5bd"
        ),
        "failure_status": "FAIL_MATCHED_GREEDY_CONTROL",
    }
    v1_control = {
        "fresh_cuda_processes": True,
        "post_start_tokens_match_treatment": True,
        "preexisting_history_matches_treatment": True,
        "request_start_precedes_cuda_launch": True,
        "same_artifacts": True,
        "same_prompt_ids": True,
    }
    r1_control = {
        "fresh_cuda_processes": True,
        "greedy_agreement_is_diagnostic": True,
        "preexisting_history_matches_treatment": True,
        "request_start_precedes_cuda_launch": True,
        "same_artifacts": True,
        "same_prompt_ids": True,
        "teacher_forced_post_start_tokens": True,
    }
    v1_requirements = [
            "ACTIVE_PHONE_KV_PRECEDES_PROMOTION",
            "CUDA_MODEL_PROCESSES_ABSENT_AT_PROMOTION_START",
            "PHONE_DECODE_OVERLAPS_CUDA_PROCESS_LOAD",
            "PHONE_TOKEN_READY_BEFORE_CUDA",
            "CUDA_REPLAYS_DYNAMIC_PHONE_FRONTIER",
            "CUDA_INGESTS_EXACT_POST_READY_DELTA",
            "DURABLE_SINGLE_PUBLICATION_OWNER",
            "MATCHED_COLD_SERVER_QUEUE_CONTROL",
            "REALIZED_BACKEND_PLACEMENT",
            "ZERO_TERMINAL_SEQUENCE_STATE",
    ]
    r1_requirements = [
        "ACTIVE_PHONE_KV_PRECEDES_PROMOTION",
        "CUDA_MODEL_PROCESSES_ABSENT_AT_PROMOTION_START",
        "PHONE_DECODE_OVERLAPS_CUDA_PROCESS_LOAD",
        "PHONE_TOKEN_READY_BEFORE_CUDA",
        "CUDA_REPLAYS_DYNAMIC_PHONE_FRONTIER",
        "CUDA_INGESTS_EXACT_POST_READY_DELTA",
        "DURABLE_SINGLE_PUBLICATION_OWNER",
        "MATCHED_TEACHER_FORCED_COLD_SERVER_CONTROL",
        "GREEDY_AGREEMENT_REPORTED_NOT_GATED",
        "REALIZED_BACKEND_PLACEMENT",
        "ZERO_TERMINAL_SEQUENCE_STATE",
    ]
    expected = (
        (v1_predecessor, v1_control, v1_requirements)
        if root["schema"] == CONTRACT_SCHEMA
        else (r1_predecessor, r1_control, r1_requirements)
    )
    require(
        (
            root["predecessor"],
            root["control"],
            root["requirements"],
        )
        == expected,
        "live_contract: versioned policy",
    )
    names = {
        "batch",
        "connector_poll_us",
        "cuda_continuation_tokens",
        "cuda_control_chunk",
        "cuda_delta_chunk",
        "cuda_replay_chunk",
        "max_cuda_ready_us",
        "max_inter_batch_gap_us",
        "max_launch_delay_us",
        "max_phone_service_tokens",
        "max_rows_per_batch",
        "min_phone_service_tokens",
        "min_useful_phone_tokens_before_cuda_ready",
        "phone_delta_tokens",
        "phone_prefill_chunk",
        "preexisting_committed_tokens",
    }
    execution = w6.exact_keys(root["execution"], names, "live_contract.execution")
    values = {
        name: w6.checked_positive(
            execution[name],
            f"live_contract.execution.{name}",
        )
        for name in names
    }
    require(values["batch"] == base_contract.batch, "live_contract: batch")
    require(
        values["phone_delta_tokens"] == delta_contract.phone_delta_tokens
        and values["cuda_continuation_tokens"]
        == delta_contract.cuda_continuation_tokens,
        "live_contract: delta geometry",
    )
    require(
        values["min_phone_service_tokens"]
        <= values["max_phone_service_tokens"],
        "live_contract: service bounds",
    )
    for name in (
        "cuda_control_chunk",
        "cuda_delta_chunk",
        "cuda_replay_chunk",
        "phone_prefill_chunk",
    ):
        require(
            values["batch"] * values[name] <= values["max_rows_per_batch"],
            f"live_contract: {name} exceeds row cap",
        )
    return LiveContract(
        w6.sha256(raw),
        base_contract.raw_sha256,
        delta_contract.raw_sha256,
        physical_gate_sha256,
        **values,
    )


def continue_active_until_ready(
    client: StageV3Client,
    prediction: Sequence[int],
    identity_base: int,
    position_start: int,
    min_tokens: int,
    max_tokens: int,
    connector: cold.ReadyConnector,
) -> tuple[list[list[int]], cold.ServiceMetrics]:
    require(
        bool(prediction)
        and all(w6.is_int(token) and token >= 0 for token in prediction)
        and position_start >= 0
        and 0 < min_tokens <= max_tokens,
        "live service: invalid input",
    )
    current = list(prediction)
    outputs = [[] for _ in current]
    intervals: list[cold.BatchInterval] = []
    token_ready_ns: list[int] = []
    while True:
        ready = connector.is_ready()
        ready_ns = connector.ready_ns
        if (
            len(outputs[0]) >= min_tokens
            and ready
            and ready_ns is not None
            and ready_ns <= token_ready_ns[-1]
        ):
            break
        require(
            len(outputs[0]) < max_tokens,
            "live service: CUDA missed the bounded decode window",
        )
        position = position_start + len(outputs[0])
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
        results, interval = cold.timed_rows(
            client,
            rows,
            "DECODE",
            position,
            position + 1,
        )
        intervals.append(interval)
        current = w5.select_predictions(
            results,
            len(prediction),
            1,
            identity_base,
            position,
        )
        for seq_id, token in enumerate(current):
            outputs[seq_id].append(token)
        token_ready_ns.append(interval.ended_ns)
        connector.raise_if_failed()
    return outputs, cold.ServiceMetrics(
        tuple(intervals),
        len(outputs[0]),
        sum(
            (interval.ended_ns - interval.started_ns) // 1000
            for interval in intervals
        ),
        0,
        sum(interval.rows for interval in intervals),
        tuple(token_ready_ns),
    )


def transaction_id(
    contract: LiveContract,
    base_contract: w5.Contract,
    run_id: str,
) -> str:
    w6.checked_digest(run_id, "run_id")
    return w6.sha256(w6.canonical({
        "base_contract_sha256": base_contract.raw_sha256,
        "live_contract_sha256": contract.raw_sha256,
        "model_sha256": base_contract.model_sha256,
        "prompt_ids": list(base_contract.prompt_ids),
        "run_id": run_id,
    }))


def validate_batch_metrics(
    value: object,
    field: str,
    expected_batches: int,
    expected_rows: int,
) -> None:
    metrics = w6.exact_keys(value, {"batches", "elapsed_us", "rows"}, field)
    require(
        all(w6.is_int(item) and item >= 0 for item in metrics.values())
        and metrics["batches"] == expected_batches
        and metrics["rows"] == expected_rows,
        f"{field}: accounting",
    )


def validate_report(
    report: object,
    contract: LiveContract,
    base_contract: w5.Contract,
    delta_contract: w6.DeltaContract,
    journal_dir: Path,
    expected_run_id: str | None = None,
) -> None:
    root = w6.exact_keys(
        report,
        {
            "base_contract_sha256",
            "batch",
            "concurrency",
            "contract_sha256",
            "corpus_manifest_sha256",
            "corpus_sha256",
            "cuda_ready",
            "delta_contract_sha256",
            "hellos",
            "journal",
            "metrics",
            "model_sha256",
            "physical_gate_sha256",
            "preexisting",
            "prompts",
            "run_id",
            "scheduler_eligible",
            "schema",
            "scope",
            "sequences",
            "state_counts",
            "status",
        },
        "live_treatment",
    )
    require(
        root["schema"] == SCHEMA
        and root["scope"] == "MECHANICS_ONLY"
        and root["scheduler_eligible"] is False,
        "live_treatment: labels",
    )
    run_id = w6.checked_digest(root["run_id"], "live_treatment.run_id")
    if expected_run_id is not None:
        require(run_id == expected_run_id, "live_treatment: run ID")
    require(
        root["contract_sha256"] == contract.raw_sha256
        and root["base_contract_sha256"] == base_contract.raw_sha256
        and root["delta_contract_sha256"] == delta_contract.raw_sha256
        and root["physical_gate_sha256"] == contract.physical_gate_sha256
        and root["model_sha256"] == base_contract.model_sha256
        and root["corpus_sha256"] == base_contract.corpus_sha256
        and root["corpus_manifest_sha256"] == base_contract.manifest_sha256,
        "live_treatment: identity",
    )
    require(
        root["batch"] == contract.batch
        and root["prompts"] == list(base_contract.prompt_ids),
        "live_treatment: workload",
    )

    hellos = w6.exact_keys(
        root["hellos"],
        {"cuda", "phone"},
        "live_treatment.hellos",
    )
    parsed: dict[str, Hello] = {}
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
        hello_value = w6.exact_keys(
            value,
            hello_keys,
            f"live_treatment.hellos.{name}",
        )
        require(
            type(hello_value["model_sha256"]) is str
            and all(
                w6.is_int(item)
                for key, item in hello_value.items()
                if key != "model_sha256"
            ),
            f"live_treatment: invalid {name} hello",
        )
        parsed[name] = Hello(**hello_value)
        w5.validate_hello(name, parsed[name], base_contract)
    try:
        cold.require_same_model(
            parsed,
            expected_model_sha256=base_contract.model_sha256,
            expected_file_type=base_contract.file_type,
        )
    except ProtocolError as exc:
        raise LivePromotionError(f"live_treatment: {exc}") from exc

    preexisting = w6.exact_keys(
        root["preexisting"],
        {
            "ended_ns",
            "metrics",
            "preexisting_committed_tokens",
            "state_count",
            "started_ns",
        },
        "live_treatment.preexisting",
    )
    require(
        w6.is_int(preexisting["started_ns"])
        and w6.is_int(preexisting["ended_ns"])
        and 0 < preexisting["started_ns"] < preexisting["ended_ns"]
        and preexisting["preexisting_committed_tokens"]
        == contract.preexisting_committed_tokens
        and preexisting["state_count"] == contract.batch,
        "live_treatment: preexisting state",
    )
    pre_metrics = w6.exact_keys(
        preexisting["metrics"],
        {
            "batch_timeline",
            "continuation_batches",
            "elapsed_us",
            "history_batches",
            "rows",
            "token_ready_ns",
        },
        "live_treatment.preexisting.metrics",
    )
    require(
        pre_metrics["rows"]
        == contract.batch
        * (
            base_contract.prompt_tokens
            + contract.preexisting_committed_tokens
            - 1
        )
        and pre_metrics["continuation_batches"]
        == contract.preexisting_committed_tokens - 1
        and pre_metrics["history_batches"]
        == (
            base_contract.prompt_tokens + contract.phone_prefill_chunk - 1
        )
        // contract.phone_prefill_chunk,
        "live_treatment: preexisting accounting",
    )

    ready = w6.exact_keys(
        root["cuda_ready"],
        {
            "attempts",
            "cuda_ready_ns",
            "max_inter_batch_gap_us",
            "ownership_commit_ns",
            "phone_first_token_ns",
            "phone_frontier_ns",
            "phone_service_tokens",
            "request_complete_ns",
            "request_start_ns",
            "useful_phone_tokens_before_ready",
        },
        "live_treatment.cuda_ready",
    )
    require(
        all(w6.is_int(item) and item >= 0 for item in ready.values()),
        "live_treatment: readiness types",
    )
    service_tokens = ready["phone_service_tokens"]
    require(
        contract.min_phone_service_tokens
        <= service_tokens
        <= contract.max_phone_service_tokens
        and preexisting["ended_ns"] <= ready["request_start_ns"]
        < ready["phone_first_token_ns"]
        < ready["cuda_ready_ns"]
        <= ready["phone_frontier_ns"]
        < ready["ownership_commit_ns"]
        < ready["request_complete_ns"]
        and ready["useful_phone_tokens_before_ready"]
        >= contract.min_useful_phone_tokens_before_cuda_ready
        and ready["max_inter_batch_gap_us"]
        <= contract.max_inter_batch_gap_us,
        "live_treatment: readiness gate",
    )

    metrics = w6.exact_keys(
        root["metrics"],
        {
            "cuda_continuation",
            "cuda_delta",
            "cuda_replay",
            "cuda_warm_control",
            "phone_delta",
            "phone_service",
        },
        "live_treatment.metrics",
    )
    service = w6.exact_keys(
        metrics["phone_service"],
        {
            "batch_timeline",
            "continuation_batches",
            "elapsed_us",
            "history_batches",
            "rows",
            "token_ready_ns",
        },
        "live_treatment.metrics.phone_service",
    )
    timeline = service["batch_timeline"]
    token_ready = service["token_ready_ns"]
    require(
        type(timeline) is list
        and len(timeline) == service_tokens
        and type(token_ready) is list
        and len(token_ready) == service_tokens
        and service["history_batches"] == 0
        and service["continuation_batches"] == service_tokens
        and service["rows"] == contract.batch * service_tokens,
        "live_treatment: service accounting",
    )
    previous_end = None
    recomputed_gap = 0
    elapsed_us = 0
    for index, item in enumerate(timeline):
        interval = w6.exact_keys(
            item,
            {
                "ended_ns",
                "kind",
                "position_end",
                "position_start",
                "rows",
                "started_ns",
            },
            f"live_treatment.metrics.phone_service.timeline.{index}",
        )
        require(
            interval["kind"] == "DECODE"
            and all(
                w6.is_int(value)
                for key, value in interval.items()
                if key != "kind"
            )
            and interval["rows"] == contract.batch
            and interval["started_ns"] < interval["ended_ns"]
            and interval["position_end"] == interval["position_start"] + 1
            and (
                previous_end is None
                or interval["started_ns"] >= previous_end
            ),
            "live_treatment: service timeline",
        )
        if previous_end is not None:
            recomputed_gap = max(
                recomputed_gap,
                (interval["started_ns"] - previous_end) // 1000,
            )
        previous_end = interval["ended_ns"]
        elapsed_us += (
            interval["ended_ns"] - interval["started_ns"]
        ) // 1000
    require(
        token_ready == [item["ended_ns"] for item in timeline]
        and service["elapsed_us"] == elapsed_us
        and ready["phone_first_token_ns"] == token_ready[0]
        and ready["phone_frontier_ns"] == token_ready[-1]
        and ready["useful_phone_tokens_before_ready"]
        == sum(item < ready["cuda_ready_ns"] for item in token_ready)
        and ready["max_inter_batch_gap_us"] == recomputed_gap,
        "live_treatment: readiness accounting",
    )

    replay_width = (
        base_contract.prompt_tokens
        + contract.preexisting_committed_tokens
        + service_tokens
    )
    frontier_width = replay_width + contract.phone_delta_tokens
    continuation_batches = contract.cuda_continuation_tokens - 1
    validate_batch_metrics(
        metrics["phone_delta"],
        "live_treatment.metrics.phone_delta",
        contract.phone_delta_tokens,
        contract.batch * contract.phone_delta_tokens,
    )
    validate_batch_metrics(
        metrics["cuda_replay"],
        "live_treatment.metrics.cuda_replay",
        (replay_width + contract.cuda_replay_chunk - 1)
        // contract.cuda_replay_chunk,
        contract.batch * replay_width,
    )
    validate_batch_metrics(
        metrics["cuda_delta"],
        "live_treatment.metrics.cuda_delta",
        (contract.phone_delta_tokens + contract.cuda_delta_chunk - 1)
        // contract.cuda_delta_chunk,
        contract.batch * contract.phone_delta_tokens,
    )
    validate_batch_metrics(
        metrics["cuda_continuation"],
        "live_treatment.metrics.cuda_continuation",
        continuation_batches,
        contract.batch * continuation_batches,
    )
    warm = w6.exact_keys(
        metrics["cuda_warm_control"],
        {
            "continuation_batches",
            "elapsed_us",
            "history_batches",
            "rows",
        },
        "live_treatment.metrics.cuda_warm_control",
    )
    require(
        all(w6.is_int(item) and item >= 0 for item in warm.values())
        and warm["history_batches"]
        == (frontier_width + contract.cuda_control_chunk - 1)
        // contract.cuda_control_chunk
        and warm["continuation_batches"] == continuation_batches
        and warm["rows"]
        == contract.batch * (frontier_width + continuation_batches),
        "live_treatment: warm control accounting",
    )

    concurrency = w6.exact_keys(
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
        "live_treatment.concurrency",
    )
    require(
        all(w6.is_int(item) and item >= 0 for item in concurrency.values()),
        "live_treatment: concurrency types",
    )
    phone_wall = (
        concurrency["phone_ended_ns"] - concurrency["phone_started_ns"]
    )
    cuda_wall = concurrency["cuda_ended_ns"] - concurrency["cuda_started_ns"]
    overlap = max(
        0,
        min(concurrency["phone_ended_ns"], concurrency["cuda_ended_ns"])
        - max(concurrency["phone_started_ns"], concurrency["cuda_started_ns"]),
    )
    shorter = min(phone_wall, cuda_wall)
    require(
        phone_wall > 0
        and cuda_wall > 0
        and concurrency["phone_wall_ns"] == phone_wall
        and concurrency["cuda_wall_ns"] == cuda_wall
        and concurrency["overlap_ns"] == overlap
        and concurrency["shorter_ns"] == shorter
        and concurrency["overlap_shorter_ppm"]
        == overlap * 1_000_000 // shorter
        and concurrency["overlap_shorter_ppm"]
        >= delta_contract.min_overlap_shorter_ppm,
        "live_treatment: concurrency accounting",
    )

    sequences = root["sequences"]
    require(
        type(sequences) is list and len(sequences) == contract.batch,
        "live_treatment: sequence count",
    )
    frontier_histories = []
    final_histories = []
    exact = True
    for index, item in enumerate(sequences):
        sequence = w6.exact_keys(
            item,
            {
                "control_continuation",
                "cuda_continuation",
                "final_published_tokens",
                "phone_delta",
                "phone_service",
                "preexisting_tokens",
                "prompt_id",
                "prompt_tokens",
                "sequence_index",
            },
            f"live_treatment.sequences.{index}",
        )
        require(
            sequence["sequence_index"] == index
            and sequence["prompt_id"] == base_contract.prompt_ids[index],
            "live_treatment: sequence identity",
        )
        widths = {
            "prompt_tokens": base_contract.prompt_tokens,
            "preexisting_tokens": contract.preexisting_committed_tokens,
            "phone_service": service_tokens,
            "phone_delta": contract.phone_delta_tokens,
            "cuda_continuation": contract.cuda_continuation_tokens,
            "control_continuation": contract.cuda_continuation_tokens,
        }
        for field, width in widths.items():
            require(
                type(sequence[field]) is list
                and len(sequence[field]) == width
                and all(
                    w6.is_int(token) and token >= 0
                    for token in sequence[field]
                ),
                f"live_treatment: invalid {field}",
            )
        published = (
            sequence["preexisting_tokens"]
            + sequence["phone_service"]
            + sequence["phone_delta"]
            + sequence["cuda_continuation"]
        )
        require(
            sequence["final_published_tokens"] == published,
            "live_treatment: publication gap or duplicate",
        )
        exact = exact and (
            sequence["cuda_continuation"]
            == sequence["control_continuation"]
        )
        frontier = (
            sequence["prompt_tokens"]
            + sequence["preexisting_tokens"]
            + sequence["phone_service"]
            + sequence["phone_delta"]
        )
        frontier_histories.append(frontier)
        final_histories.append(frontier + sequence["cuda_continuation"])

    require(
        root["state_counts"] == {
            "cuda_after_completion": 0,
            "cuda_control_released": 0,
            "cuda_prepared": contract.batch,
            "phone_after_commit": 0,
            "phone_at_frontier": contract.batch,
            "phone_before_promotion": contract.batch,
        },
        "live_treatment: state counts",
    )
    tx_id = transaction_id(contract, base_contract, run_id)
    entries = w6.load_journal(
        journal_dir,
        delta_contract,
        base_contract,
        tx_id,
        run_id,
    )
    frontier_sha = w5.histories_digest(frontier_histories)
    final_sha = w5.histories_digest(final_histories)
    frontier_tokens = (
        contract.preexisting_committed_tokens
        + service_tokens
        + contract.phone_delta_tokens
    )
    for index, entry in enumerate(entries):
        require(
            entry.value["token_history_sha256"]
            == (final_sha if index >= 4 else frontier_sha)
            and entry.value["published_tokens_per_request"]
            == (
                frontier_tokens + contract.cuda_continuation_tokens
                if index >= 4
                else frontier_tokens
            ),
            "live_treatment: journal frontier",
        )
    require(
        root["journal"] == w6.journal_summary(entries),
        "live_treatment: journal summary",
    )
    require(
        root["status"]
        == (
            "LIVE_SESSION_PROMOTION_PASS"
            if exact
            else "LIVE_SESSION_PROMOTION_FAIL"
        ),
        "live_treatment: status",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--base-contract",
        type=Path,
        default=cold.DEFAULT_BASE_CONTRACT,
    )
    parser.add_argument(
        "--delta-contract",
        type=Path,
        default=cold.DEFAULT_DELTA_CONTRACT,
    )
    parser.add_argument(
        "--physical-gate",
        type=Path,
        default=cold.DEFAULT_PHYSICAL_GATE,
    )
    parser.add_argument("--phone-route", type=parse_endpoint, required=True)
    parser.add_argument("--cuda-route", type=parse_endpoint, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--start-marker", type=Path, required=True)
    parser.add_argument("--journal-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    if (
        args.timeout <= 0
        or w6.HEX64.fullmatch(args.run_id) is None
        or args.start_marker.exists()
        or args.journal_dir.exists()
        or args.output.exists()
    ):
        parser.error("invalid arguments or existing output")

    clients: dict[str, StageV3Client] = {}
    active: dict[str, tuple[int, int] | None] = {
        "cuda": None,
        "phone": None,
    }
    stopped: set[str] = set()
    connector: cold.ReadyConnector | None = None
    try:
        base_contract = w5.load_contract(args.base_contract)
        delta_contract = w6.load_contract(args.delta_contract, base_contract)
        contract = load_contract(
            args.contract,
            base_contract,
            delta_contract,
            w6.sha256(args.physical_gate.read_bytes()),
        )
        corpus, _, _ = load_corpus(
            base_contract.corpus_path,
            base_contract.manifest_path,
            base_contract.model_sha256,
        )
        prompts = [
            corpus[prompt_id]["tokens"]
            for prompt_id in base_contract.prompt_ids
        ]
        tx_id = transaction_id(contract, base_contract, args.run_id)
        journal = w6.OwnershipJournal(
            args.journal_dir,
            tx_id,
            base_contract.model_sha256,
            base_contract.prompt_ids,
            args.run_id,
        )

        clients["phone"] = StageV3Client.connect(*args.phone_route, args.timeout)
        phone_hello = clients["phone"].hello()
        w5.validate_hello("phone", phone_hello, base_contract)
        phone_base = 10000
        catchup_base = 30000
        control_base = 40000
        active["phone"] = (contract.batch, phone_base)
        preexisting_started_ns = time.monotonic_ns()
        preexisting_tokens, preexisting_metrics = cold.run_fixed_generation(
            clients["phone"],
            prompts,
            phone_base,
            contract.phone_prefill_chunk,
            contract.preexisting_committed_tokens,
        )
        preexisting_ended_ns = time.monotonic_ns()
        phone_before_promotion = clients["phone"].status()
        require(
            phone_before_promotion.active_sequences == contract.batch,
            "live treatment: preexisting phone state",
        )

        request_start_ns = time.monotonic_ns()
        w5.write_atomic(args.start_marker, {
            "phone_state_count": phone_before_promotion.active_sequences,
            "preexisting_ended_ns": preexisting_ended_ns,
            "process_pid": os.getpid(),
            "request_start_ns": request_start_ns,
            "run_id": args.run_id,
            "schema": START_SCHEMA,
        })
        connector = cold.ReadyConnector(
            args.cuda_route,
            contract.max_cuda_ready_us / 1_000_000,
            contract.connector_poll_us,
        )
        connector.start()
        service_tokens_by_seq, service_metrics = continue_active_until_ready(
            clients["phone"],
            [tokens[-1] for tokens in preexisting_tokens],
            phone_base,
            base_contract.prompt_tokens
            + contract.preexisting_committed_tokens
            - 1,
            contract.min_phone_service_tokens,
            contract.max_phone_service_tokens,
            connector,
        )
        connection = connector.take()
        clients["cuda"] = connection.client
        cuda_hello = connection.hello
        w5.validate_hello("cuda", cuda_hello, base_contract)
        cold.require_same_model(
            {"cuda": cuda_hello, "phone": phone_hello},
            expected_model_sha256=base_contract.model_sha256,
            expected_file_type=base_contract.file_type,
        )
        service_tokens = len(service_tokens_by_seq[0])
        require(
            all(
                len(tokens) == service_tokens
                for tokens in service_tokens_by_seq
            ),
            "live treatment: nonrectangular service",
        )
        service_histories = [
            list(prompt) + list(preexisting) + list(service)
            for prompt, preexisting, service in zip(
                prompts,
                preexisting_tokens,
                service_tokens_by_seq,
            )
        ]

        active["cuda"] = (contract.batch, catchup_base)
        phone_result, cuda_result, concurrency = w6.run_concurrently(
            lambda: w6.advance_active(
                clients["phone"],
                [tokens[-1] for tokens in service_tokens_by_seq],
                phone_base,
                base_contract.prompt_tokens
                + contract.preexisting_committed_tokens
                + service_tokens
                - 1,
                contract.phone_delta_tokens,
            ),
            lambda: w6.replay_history_only(
                clients["cuda"],
                service_histories,
                catchup_base,
                contract.cuda_replay_chunk,
            ),
            args.timeout,
        )
        delta_tokens, _, phone_delta_metrics = phone_result
        _, cuda_replay_metrics = cuda_result
        require(
            concurrency["overlap_shorter_ppm"]
            >= delta_contract.min_overlap_shorter_ppm,
            "live treatment: concurrent replay overlap",
        )
        phone_frontier_status = clients["phone"].status()
        require(
            phone_frontier_status.active_sequences == contract.batch,
            "live treatment: phone frontier",
        )
        frontier_histories = [
            list(history) + list(tokens)
            for history, tokens in zip(service_histories, delta_tokens)
        ]
        frontier_sha = w5.histories_digest(frontier_histories)
        published_phone_tokens = (
            contract.preexisting_committed_tokens
            + service_tokens
            + contract.phone_delta_tokens
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
        cuda_prediction, cuda_delta_metrics = w6.feed_known_tokens(
            clients["cuda"],
            delta_tokens,
            catchup_base,
            base_contract.prompt_tokens
            + contract.preexisting_committed_tokens
            + service_tokens,
            contract.cuda_delta_chunk,
        )
        cuda_prepared = clients["cuda"].status()
        require(
            cuda_prepared.active_sequences == contract.batch,
            "live treatment: CUDA prepared state",
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
        w6.commit_cutover(
            journal,
            published_tokens=published_phone_tokens,
            history_sha256=frontier_sha,
            remove_phone=lambda: w5.remove_group(
                clients["phone"],
                contract.batch,
                phone_base,
            ),
        )
        ownership_commit_ns = time.monotonic_ns()
        active["phone"] = None
        phone_after_commit = clients["phone"].status()
        require(
            phone_after_commit.active_sequences == 0,
            "live treatment: phone state leak",
        )

        cuda_tokens, cuda_continuation_metrics = w6.continue_from_prediction(
            clients["cuda"],
            cuda_prediction,
            catchup_base,
            base_contract.prompt_tokens
            + published_phone_tokens,
            contract.cuda_continuation_tokens,
        )
        request_complete_ns = time.monotonic_ns()
        final_histories = [
            frontier + list(tokens)
            for frontier, tokens in zip(frontier_histories, cuda_tokens)
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
            "live treatment: CUDA state leak",
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

        active["cuda"] = (contract.batch, control_base)
        control_tokens, control_metrics = w5.run_replay(
            clients["cuda"],
            frontier_histories,
            control_base,
            contract.cuda_control_chunk,
            contract.cuda_continuation_tokens,
        )
        w5.remove_group(clients["cuda"], contract.batch, control_base)
        active["cuda"] = None
        cuda_control_released = clients["cuda"].status()
        require(
            cuda_control_released.active_sequences == 0,
            "live treatment: CUDA control state leak",
        )
        w5.finish(clients["phone"], "stop")
        stopped.add("phone")
        w5.finish(clients["cuda"], "stop")
        stopped.add("cuda")

        entries = w6.load_journal(
            args.journal_dir,
            delta_contract,
            base_contract,
            tx_id,
            args.run_id,
        )
        token_ready_ns = service_metrics.token_ready_ns
        report: dict[str, object] = {
            "base_contract_sha256": base_contract.raw_sha256,
            "batch": contract.batch,
            "concurrency": concurrency,
            "contract_sha256": contract.raw_sha256,
            "corpus_manifest_sha256": base_contract.manifest_sha256,
            "corpus_sha256": base_contract.corpus_sha256,
            "cuda_ready": {
                "attempts": connection.attempts,
                "cuda_ready_ns": connection.ready_ns,
                "max_inter_batch_gap_us": (
                    cold.max_inter_batch_gap_us(service_metrics)
                ),
                "ownership_commit_ns": ownership_commit_ns,
                "phone_first_token_ns": token_ready_ns[0],
                "phone_frontier_ns": token_ready_ns[-1],
                "phone_service_tokens": service_tokens,
                "request_complete_ns": request_complete_ns,
                "request_start_ns": request_start_ns,
                "useful_phone_tokens_before_ready": sum(
                    item < connection.ready_ns for item in token_ready_ns
                ),
            },
            "delta_contract_sha256": delta_contract.raw_sha256,
            "hellos": {
                "cuda": asdict(cuda_hello),
                "phone": asdict(phone_hello),
            },
            "journal": w6.journal_summary(entries),
            "metrics": {
                "cuda_continuation": asdict(cuda_continuation_metrics),
                "cuda_delta": asdict(cuda_delta_metrics),
                "cuda_replay": asdict(cuda_replay_metrics),
                "cuda_warm_control": asdict(control_metrics),
                "phone_delta": asdict(phone_delta_metrics),
                "phone_service": cold.service_metrics_value(service_metrics),
            },
            "model_sha256": base_contract.model_sha256,
            "physical_gate_sha256": contract.physical_gate_sha256,
            "preexisting": {
                "ended_ns": preexisting_ended_ns,
                "metrics": cold.service_metrics_value(preexisting_metrics),
                "preexisting_committed_tokens": (
                    contract.preexisting_committed_tokens
                ),
                "started_ns": preexisting_started_ns,
                "state_count": phone_before_promotion.active_sequences,
            },
            "prompts": list(base_contract.prompt_ids),
            "run_id": args.run_id,
            "scheduler_eligible": False,
            "schema": SCHEMA,
            "scope": "MECHANICS_ONLY",
            "sequences": [
                {
                    "control_continuation": list(control_tokens[index]),
                    "cuda_continuation": list(cuda_tokens[index]),
                    "final_published_tokens": (
                        list(preexisting_tokens[index])
                        + list(service_tokens_by_seq[index])
                        + list(delta_tokens[index])
                        + list(cuda_tokens[index])
                    ),
                    "phone_delta": list(delta_tokens[index]),
                    "phone_service": list(service_tokens_by_seq[index]),
                    "preexisting_tokens": list(preexisting_tokens[index]),
                    "prompt_id": base_contract.prompt_ids[index],
                    "prompt_tokens": list(prompts[index]),
                    "sequence_index": index,
                }
                for index in range(contract.batch)
            ],
            "state_counts": {
                "cuda_after_completion": (
                    cuda_after_completion.active_sequences
                ),
                "cuda_control_released": (
                    cuda_control_released.active_sequences
                ),
                "cuda_prepared": cuda_prepared.active_sequences,
                "phone_after_commit": phone_after_commit.active_sequences,
                "phone_at_frontier": (
                    phone_frontier_status.active_sequences
                ),
                "phone_before_promotion": (
                    phone_before_promotion.active_sequences
                ),
            },
            "status": (
                "LIVE_SESSION_PROMOTION_PASS"
                if cuda_tokens == control_tokens
                else "LIVE_SESSION_PROMOTION_FAIL"
            ),
        }
        validate_report(
            report,
            contract,
            base_contract,
            delta_contract,
            args.journal_dir,
            args.run_id,
        )
        w5.write_atomic(args.output, report)
        print(w6.canonical(report).decode("ascii"), end="")
        return 0 if cuda_tokens == control_tokens else 3
    finally:
        if connector is not None:
            connector.close()
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
        KeyError,
        OSError,
        ProtocolError,
        TypeError,
        ValueError,
        LivePromotionError,
        w5.HandoffError,
        w6.DeltaError,
    ) as exc:
        print(w6.canonical({
            "error": str(exc),
            "status": "LIVE_SESSION_PROMOTION_ERROR",
        }).decode("ascii"), end="")
        raise SystemExit(2)
