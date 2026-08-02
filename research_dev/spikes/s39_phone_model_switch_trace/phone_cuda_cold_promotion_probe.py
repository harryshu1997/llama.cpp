#!/usr/bin/env python3
"""Keep phone service live while a cold CUDA route becomes ready."""

from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import phone_cuda_delta_probe as w6
import phone_cuda_handoff_probe as w5
from async_pipeline import parse_endpoint
from qwen25_quality_probe import load_corpus
from stage_v3_client import (
    BatchRow,
    Hello,
    ProtocolError,
    StageV3Client,
    require_same_model,
)


SCHEMA = "s39-cold-promotion-treatment-v1"
CONTRACT_SCHEMA = "s39-cold-promotion-contract-v2"
START_SCHEMA = "s39-promotion-request-start-v1"
DEFAULT_CONTRACT = (
    Path(__file__).resolve().parent / "W7_COLD_PROMOTION_CONTRACT_R1.json"
)
DEFAULT_BASE_CONTRACT = Path(__file__).resolve().parent / "W5_HANDOFF_CONTRACT.json"
DEFAULT_DELTA_CONTRACT = Path(__file__).resolve().parent / "W6_DELTA_CONTRACT.json"
DEFAULT_PHYSICAL_GATE = Path(__file__).resolve().parent / "W6_PHYSICAL_GATE.json"


class PromotionError(RuntimeError):
    pass


@dataclass(frozen=True)
class PromotionContract:
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
    warmup_generated_tokens: int


@dataclass(frozen=True)
class BatchInterval:
    ended_ns: int
    kind: str
    position_end: int
    position_start: int
    rows: int
    started_ns: int


@dataclass(frozen=True)
class ServiceMetrics:
    batch_timeline: tuple[BatchInterval, ...]
    continuation_batches: int
    elapsed_us: int
    history_batches: int
    rows: int
    token_ready_ns: tuple[int, ...]


@dataclass(frozen=True)
class ConnectionResult:
    attempts: int
    client: StageV3Client
    hello: Hello
    ready_ns: int


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PromotionError(message)


def load_contract(
    path: Path,
    base_contract: w5.Contract,
    delta_contract: w6.DeltaContract,
    physical_gate_sha256: str,
) -> PromotionContract:
    value, raw = w6.read_canonical(path, "promotion_contract")
    root = w6.exact_keys(
        value,
        {
            "base_contract_sha256",
            "control",
            "delta_contract_sha256",
            "execution",
            "physical_gate_sha256",
            "predecessor",
            "preparation",
            "requirements",
            "scheduler_eligible_on_pass",
            "schema",
            "scope",
            "status",
        },
        "promotion_contract",
    )
    require(root["schema"] == CONTRACT_SCHEMA, "promotion_contract: schema")
    require(
        root["status"] == "FROZEN_BEFORE_R1_ACQUISITION",
        "promotion_contract: status",
    )
    require(root["scope"] == "MECHANICS_ONLY", "promotion_contract: scope")
    require(
        root["scheduler_eligible_on_pass"] is False,
        "promotion_contract: scheduler eligibility",
    )
    require(
        root["base_contract_sha256"] == base_contract.raw_sha256
        and root["delta_contract_sha256"] == delta_contract.raw_sha256
        and root["physical_gate_sha256"] == physical_gate_sha256,
        "promotion_contract: dependency mismatch",
    )
    require(
        root["control"] == {
            "fresh_cuda_processes": True,
            "generated_tokens_match_treatment": True,
            "request_start_precedes_cuda_launch": True,
            "same_artifacts": True,
            "same_prompt_ids": True,
        },
        "promotion_contract: control mismatch",
    )
    require(
        root["predecessor"] == {
            "contract_sha256": (
                "613aa0807275e768d308584fd99469951"
                "b8d0d812e6bad5a586d980760db37d4"
            ),
            "failure_manifest_sha256": (
                "cca31839d45f6e18306cbdf009f4be04"
                "d2642ea6248e8a97816b8ef7d9174bab"
            ),
            "failure_status": "FAIL_READINESS_ORDER",
        },
        "promotion_contract: predecessor mismatch",
    )
    preparation = w6.exact_keys(
        root["preparation"],
        {
            "batch",
            "generated_tokens",
            "phone_prefill_chunk",
            "same_resident_workers_required",
            "session_end",
            "state_reset_required",
        },
        "promotion_contract.preparation",
    )
    require(
        preparation["batch"] == base_contract.batch
        and preparation["phone_prefill_chunk"]
        == base_contract.phone_prefill_chunk
        and w6.is_int(preparation["generated_tokens"])
        and preparation["generated_tokens"] >= 2
        and preparation["session_end"] == "DETACH"
        and preparation["same_resident_workers_required"] is True
        and preparation["state_reset_required"] is True,
        "promotion_contract: preparation mismatch",
    )
    require(
        root["requirements"] == [
            "PHONE_ROUTE_PREPARED_BEFORE_REQUEST",
            "SAME_RESIDENT_PHONE_WORKERS_AFTER_PREPARATION",
            "CUDA_MODEL_PROCESSES_ABSENT_AT_REQUEST_START",
            "PHONE_SERVICE_OVERLAPS_CUDA_PROCESS_LOAD",
            "CUDA_READY_BEFORE_PHONE_SERVICE_BOUND",
            "CUDA_REPLAYS_DYNAMIC_PHONE_FRONTIER",
            "CUDA_INGESTS_EXACT_POST_READY_DELTA",
            "DURABLE_SINGLE_PUBLICATION_OWNER",
            "MATCHED_COLD_SERVER_QUEUE_CONTROL",
            "REALIZED_BACKEND_PLACEMENT",
            "ZERO_TERMINAL_SEQUENCE_STATE",
        ],
        "promotion_contract: requirements mismatch",
    )
    execution = w6.exact_keys(
        root["execution"],
        {
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
        },
        "promotion_contract.execution",
    )
    values = {
        key: w6.checked_positive(
            item,
            f"promotion_contract.execution.{key}",
        )
        for key, item in execution.items()
    }
    require(values["batch"] == base_contract.batch, "promotion_contract: batch")
    require(
        values["phone_delta_tokens"] == delta_contract.phone_delta_tokens
        and values["cuda_continuation_tokens"]
        == delta_contract.cuda_continuation_tokens,
        "promotion_contract: delta geometry",
    )
    require(
        values["min_phone_service_tokens"]
        <= values["max_phone_service_tokens"],
        "promotion_contract: phone bounds",
    )
    for name in (
        "cuda_control_chunk",
        "cuda_delta_chunk",
        "cuda_replay_chunk",
        "phone_prefill_chunk",
    ):
        require(
            values["batch"] * values[name] <= values["max_rows_per_batch"],
            f"promotion_contract: {name} exceeds row cap",
        )
    return PromotionContract(
        w6.sha256(raw),
        base_contract.raw_sha256,
        delta_contract.raw_sha256,
        physical_gate_sha256,
        **values,
        warmup_generated_tokens=preparation["generated_tokens"],
    )


class ReadyConnector:
    def __init__(
        self,
        endpoint: tuple[str, int],
        timeout_s: float,
        poll_us: int,
    ) -> None:
        require(timeout_s > 0 and poll_us > 0, "connector: invalid timing")
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.poll_us = poll_us
        self.ready = threading.Event()
        self.stop_requested = threading.Event()
        self.attempts = 0
        self.client: StageV3Client | None = None
        self.hello: Hello | None = None
        self.ready_ns: int | None = None
        self.error: BaseException | None = None
        self.taken = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        deadline = time.monotonic() + self.timeout_s
        while not self.stop_requested.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.error = PromotionError("connector: CUDA readiness timed out")
                self.ready.set()
                return
            self.attempts += 1
            client: StageV3Client | None = None
            try:
                client = StageV3Client.connect(
                    *self.endpoint,
                    min(0.25, remaining),
                )
                hello = client.hello()
            except (ConnectionRefusedError, TimeoutError, socket.timeout, OSError):
                if client is not None:
                    client.close()
                self.stop_requested.wait(self.poll_us / 1_000_000)
                continue
            except ProtocolError as exc:
                if client is not None:
                    client.close()
                self.error = exc
                self.ready.set()
                return
            self.client = client
            self.hello = hello
            self.ready_ns = time.monotonic_ns()
            self.ready.set()
            return

    def raise_if_failed(self) -> None:
        if self.ready.is_set() and self.error is not None:
            raise PromotionError(f"connector: {self.error}") from self.error

    def is_ready(self) -> bool:
        self.raise_if_failed()
        return (
            self.ready.is_set()
            and self.client is not None
            and self.hello is not None
            and self.ready_ns is not None
        )

    def take(self) -> ConnectionResult:
        self.thread.join(timeout=self.timeout_s + 1.0)
        self.raise_if_failed()
        require(not self.thread.is_alive(), "connector: thread did not finish")
        require(self.is_ready(), "connector: CUDA route is not ready")
        self.taken = True
        return ConnectionResult(
            self.attempts,
            self.client,
            self.hello,
            self.ready_ns,
        )

    def close(self) -> None:
        self.stop_requested.set()
        self.thread.join(timeout=1.0)
        if self.client is not None and not self.taken:
            try:
                self.client.close()
            except OSError:
                pass


def timed_rows(
    client: StageV3Client,
    rows: Sequence[BatchRow],
    kind: str,
    position_start: int,
    position_end: int,
) -> tuple[tuple[Any, ...], BatchInterval]:
    started_ns = time.monotonic_ns()
    results = client.batch(rows)
    ended_ns = time.monotonic_ns()
    return results, BatchInterval(
        ended_ns,
        kind,
        position_end,
        position_start,
        len(rows),
        started_ns,
    )


def serve_until_ready(
    client: StageV3Client,
    prompts: Sequence[Sequence[int]],
    identity_base: int,
    history_chunk: int,
    min_tokens: int,
    max_tokens: int,
    connector: ReadyConnector,
) -> tuple[list[list[int]], ServiceMetrics]:
    require(bool(prompts), "phone service: no prompts")
    batch = len(prompts)
    width = len(prompts[0])
    require(
        width > 0
        and history_chunk > 0
        and 0 < min_tokens <= max_tokens
        and all(len(prompt) == width for prompt in prompts),
        "phone service: invalid geometry",
    )
    intervals: list[BatchInterval] = []
    predictions: list[int] = []
    history_batches = 0
    for start in range(0, width, history_chunk):
        end = min(start + history_chunk, width)
        rows = w5.build_rows(prompts, identity_base, start, end)
        results, interval = timed_rows(
            client,
            rows,
            "PREFILL",
            start,
            end,
        )
        intervals.append(interval)
        history_batches += 1
        predictions = w5.select_predictions(
            results,
            batch,
            end - start,
            identity_base,
            end - 1,
        )
        connector.raise_if_failed()

    outputs = [[token] for token in predictions]
    token_ready_ns = [intervals[-1].ended_ns]
    while True:
        ready = connector.is_ready()
        ready_ns = getattr(connector, "ready_ns", None)
        if (
            len(outputs[0]) >= min_tokens
            and ready
            and (ready_ns is None or ready_ns <= token_ready_ns[-1])
        ):
            break
        require(
            len(outputs[0]) < max_tokens,
            "phone service: CUDA missed the bounded service window",
        )
        position = width + len(outputs[0]) - 1
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
        results, interval = timed_rows(
            client,
            rows,
            "DECODE",
            position,
            position + 1,
        )
        intervals.append(interval)
        predictions = w5.select_predictions(
            results,
            batch,
            1,
            identity_base,
            position,
        )
        for seq_id, token in enumerate(predictions):
            outputs[seq_id].append(token)
        token_ready_ns.append(interval.ended_ns)
        connector.raise_if_failed()

    return outputs, ServiceMetrics(
        tuple(intervals),
        len(outputs[0]) - 1,
        sum(
            (interval.ended_ns - interval.started_ns) // 1000
            for interval in intervals
        ),
        history_batches,
        sum(interval.rows for interval in intervals),
        tuple(token_ready_ns),
    )


def run_fixed_generation(
    client: StageV3Client,
    prompts: Sequence[Sequence[int]],
    identity_base: int,
    history_chunk: int,
    token_count: int,
) -> tuple[list[list[int]], ServiceMetrics]:
    class AlwaysReady:
        @staticmethod
        def raise_if_failed() -> None:
            return None

        @staticmethod
        def is_ready() -> bool:
            return True

    return serve_until_ready(
        client,
        prompts,
        identity_base,
        history_chunk,
        token_count,
        token_count,
        AlwaysReady(),
    )


def service_metrics_value(metrics: ServiceMetrics) -> dict[str, object]:
    return {
        "batch_timeline": [asdict(item) for item in metrics.batch_timeline],
        "continuation_batches": metrics.continuation_batches,
        "elapsed_us": metrics.elapsed_us,
        "history_batches": metrics.history_batches,
        "rows": metrics.rows,
        "token_ready_ns": list(metrics.token_ready_ns),
    }


def max_inter_batch_gap_us(metrics: ServiceMetrics) -> int:
    if len(metrics.batch_timeline) < 2:
        return 0
    return max(
        (
            current.started_ns - previous.ended_ns
        ) // 1000
        for previous, current in zip(
            metrics.batch_timeline,
            metrics.batch_timeline[1:],
        )
    )


def transaction_id(
    contract: PromotionContract,
    base_contract: w5.Contract,
    run_id: str,
) -> str:
    w6.checked_digest(run_id, "run_id")
    return w6.sha256(w6.canonical({
        "base_contract_sha256": base_contract.raw_sha256,
        "model_sha256": base_contract.model_sha256,
        "promotion_contract_sha256": contract.raw_sha256,
        "prompt_ids": list(base_contract.prompt_ids),
        "run_id": run_id,
    }))


def validate_report(
    report: object,
    contract: PromotionContract,
    base_contract: w5.Contract,
    delta_contract: w6.DeltaContract,
    journal_dir: Path,
    expected_run_id: str | None = None,
) -> None:
    keys = {
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
        "prompts",
        "run_id",
        "scheduler_eligible",
        "schema",
        "scope",
        "sequences",
        "state_counts",
        "status",
    }
    root = w6.exact_keys(report, keys, "treatment")
    require(root["schema"] == SCHEMA, "treatment: schema")
    require(root["scope"] == "MECHANICS_ONLY", "treatment: scope")
    require(root["scheduler_eligible"] is False, "treatment: eligibility")
    require(
        root["contract_sha256"] == contract.raw_sha256
        and root["base_contract_sha256"] == base_contract.raw_sha256
        and root["delta_contract_sha256"] == delta_contract.raw_sha256
        and root["physical_gate_sha256"] == contract.physical_gate_sha256,
        "treatment: contract mismatch",
    )
    run_id = w6.checked_digest(root["run_id"], "treatment.run_id")
    if expected_run_id is not None:
        require(run_id == expected_run_id, "treatment: run ID mismatch")
    require(
        root["model_sha256"] == base_contract.model_sha256
        and root["corpus_sha256"] == base_contract.corpus_sha256
        and root["corpus_manifest_sha256"] == base_contract.manifest_sha256,
        "treatment: model or corpus mismatch",
    )
    require(
        root["batch"] == contract.batch
        and root["prompts"] == list(base_contract.prompt_ids),
        "treatment: request set mismatch",
    )

    hellos = w6.exact_keys(root["hellos"], {"cuda", "phone"}, "treatment.hellos")
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
            f"treatment.hellos.{name}",
        )
        for field in hello_keys - {"model_sha256"}:
            require(
                w6.is_int(hello_value[field]),
                f"treatment: invalid {name} hello",
            )
        require(
            type(hello_value["model_sha256"]) is str,
            f"treatment: invalid {name} identity",
        )
        hello = Hello(**hello_value)
        w5.validate_hello(name, hello, base_contract)
        parsed[name] = hello
    try:
        require_same_model(
            parsed,
            expected_model_sha256=base_contract.model_sha256,
            expected_file_type=base_contract.file_type,
        )
    except ProtocolError as exc:
        raise PromotionError(f"treatment: {exc}") from exc

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
        "treatment.cuda_ready",
    )
    require(
        all(w6.is_int(item) and item >= 0 for item in ready.values()),
        "treatment: invalid readiness timing",
    )
    service_tokens = ready["phone_service_tokens"]
    require(
        contract.min_phone_service_tokens
        <= service_tokens
        <= contract.max_phone_service_tokens,
        "treatment: phone service bound",
    )
    require(
        ready["request_start_ns"] < ready["phone_first_token_ns"]
        and ready["request_start_ns"] < ready["cuda_ready_ns"]
        and ready["phone_first_token_ns"] < ready["cuda_ready_ns"]
        and ready["cuda_ready_ns"] <= ready["phone_frontier_ns"],
        "treatment: readiness order",
    )
    require(
        ready["phone_frontier_ns"]
        < ready["ownership_commit_ns"]
        < ready["request_complete_ns"],
        "treatment: cutover timing order",
    )
    require(
        ready["useful_phone_tokens_before_ready"]
        >= contract.min_useful_phone_tokens_before_cuda_ready,
        "treatment: no useful phone service before CUDA readiness",
    )
    require(
        ready["max_inter_batch_gap_us"] <= contract.max_inter_batch_gap_us,
        "treatment: phone service gap",
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
        "treatment.metrics",
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
        "treatment.metrics.phone_service",
    )
    timeline = service["batch_timeline"]
    token_ready = service["token_ready_ns"]
    require(
        type(timeline) is list
        and type(token_ready) is list
        and len(token_ready) == service_tokens,
        "treatment: service timeline shape",
    )
    interval_keys = {
        "ended_ns",
        "kind",
        "position_end",
        "position_start",
        "rows",
        "started_ns",
    }
    previous_end = None
    recomputed_gap = 0
    for index, item in enumerate(timeline):
        interval = w6.exact_keys(
            item,
            interval_keys,
            f"treatment.metrics.phone_service.batch_timeline.{index}",
        )
        for field in interval_keys - {"kind"}:
            require(w6.is_int(interval[field]), "treatment: invalid interval")
        require(
            interval["kind"] in {"PREFILL", "DECODE"}
            and interval["started_ns"] < interval["ended_ns"]
            and interval["position_start"] < interval["position_end"]
            and interval["rows"] > 0,
            "treatment: invalid interval",
        )
        if previous_end is not None:
            require(
                interval["started_ns"] >= previous_end,
                "treatment: overlapping phone calls",
            )
            recomputed_gap = max(
                recomputed_gap,
                (interval["started_ns"] - previous_end) // 1000,
            )
        previous_end = interval["ended_ns"]
    require(
        token_ready
        == [
            item["ended_ns"]
            for item in timeline
            if item["position_end"] == base_contract.prompt_tokens
            or item["kind"] == "DECODE"
        ],
        "treatment: token readiness mismatch",
    )
    require(
        ready["phone_first_token_ns"] == token_ready[0]
        and ready["phone_frontier_ns"] == token_ready[-1]
        and ready["useful_phone_tokens_before_ready"]
        == sum(item < ready["cuda_ready_ns"] for item in token_ready)
        and ready["max_inter_batch_gap_us"] == recomputed_gap,
        "treatment: readiness accounting",
    )
    expected_service = {
        "continuation_batches": service_tokens - 1,
        "history_batches": (
            base_contract.prompt_tokens + contract.phone_prefill_chunk - 1
        ) // contract.phone_prefill_chunk,
        "rows": contract.batch
        * (base_contract.prompt_tokens + service_tokens - 1),
    }
    require(
        all(service[name] == value for name, value in expected_service.items()),
        "treatment: phone service accounting",
    )
    require(
        service["elapsed_us"]
        == sum(
            (item["ended_ns"] - item["started_ns"]) // 1000
            for item in timeline
        ),
        "treatment: phone elapsed accounting",
    )

    batch_keys = {"batches", "elapsed_us", "rows"}
    replay_keys = {
        "continuation_batches",
        "elapsed_us",
        "history_batches",
        "rows",
    }
    for name in (
        "cuda_continuation",
        "cuda_delta",
        "cuda_replay",
        "phone_delta",
    ):
        value = w6.exact_keys(metrics[name], batch_keys, f"treatment.metrics.{name}")
        require(
            all(w6.is_int(item) and item >= 0 for item in value.values()),
            f"treatment: invalid {name} metrics",
        )
    warm_control = w6.exact_keys(
        metrics["cuda_warm_control"],
        replay_keys,
        "treatment.metrics.cuda_warm_control",
    )
    require(
        all(w6.is_int(item) and item >= 0 for item in warm_control.values()),
        "treatment: invalid warm control metrics",
    )
    replay_width = base_contract.prompt_tokens + service_tokens
    frontier_width = replay_width + contract.phone_delta_tokens
    continuation_batches = contract.cuda_continuation_tokens - 1
    expected_metrics = {
        "phone_delta": {
            "batches": contract.phone_delta_tokens,
            "rows": contract.batch * contract.phone_delta_tokens,
        },
        "cuda_replay": {
            "batches": (
                replay_width + contract.cuda_replay_chunk - 1
            ) // contract.cuda_replay_chunk,
            "rows": contract.batch * replay_width,
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
        "cuda_warm_control": {
            "history_batches": (
                frontier_width + contract.cuda_control_chunk - 1
            ) // contract.cuda_control_chunk,
            "continuation_batches": continuation_batches,
            "rows": contract.batch * (frontier_width + continuation_batches),
        },
    }
    for name, fields in expected_metrics.items():
        require(
            all(metrics[name][field] == value for field, value in fields.items()),
            f"treatment: {name} accounting",
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
        "treatment.concurrency",
    )
    require(
        all(w6.is_int(value) and value >= 0 for value in concurrency.values()),
        "treatment: invalid concurrency",
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
        "treatment: concurrency accounting",
    )

    sequences = root["sequences"]
    require(
        type(sequences) is list and len(sequences) == contract.batch,
        "treatment: sequence count",
    )
    frontier_histories: list[list[int]] = []
    final_histories: list[list[int]] = []
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
                "prompt_id",
                "prompt_tokens",
                "sequence_index",
            },
            f"treatment.sequences.{index}",
        )
        require(
            sequence["sequence_index"] == index
            and sequence["prompt_id"] == base_contract.prompt_ids[index],
            "treatment: sequence identity",
        )
        widths = {
            "prompt_tokens": base_contract.prompt_tokens,
            "phone_service": service_tokens,
            "phone_delta": contract.phone_delta_tokens,
            "cuda_continuation": contract.cuda_continuation_tokens,
            "control_continuation": contract.cuda_continuation_tokens,
        }
        for field, width in widths.items():
            require(
                type(sequence[field]) is list
                and len(sequence[field]) == width
                and all(w6.is_int(token) and token >= 0 for token in sequence[field]),
                f"treatment: invalid {field}",
            )
        published = (
            sequence["phone_service"]
            + sequence["phone_delta"]
            + sequence["cuda_continuation"]
        )
        require(
            sequence["final_published_tokens"] == published,
            "treatment: publication gap or duplicate",
        )
        exact = exact and (
            sequence["cuda_continuation"]
            == sequence["control_continuation"]
        )
        frontier = (
            sequence["prompt_tokens"]
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
        },
        "treatment: state count",
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
    frontier_tokens = service_tokens + contract.phone_delta_tokens
    for index, entry in enumerate(entries):
        expected_sha = final_sha if index >= 4 else frontier_sha
        expected_count = (
            frontier_tokens + contract.cuda_continuation_tokens
            if index >= 4
            else frontier_tokens
        )
        require(
            entry.value["token_history_sha256"] == expected_sha
            and entry.value["published_tokens_per_request"] == expected_count,
            "treatment: journal frontier",
        )
    require(
        root["journal"] == w6.journal_summary(entries),
        "treatment: journal summary",
    )
    expected_status = (
        "COLD_PROMOTION_TREATMENT_PASS"
        if exact
        else "COLD_PROMOTION_TREATMENT_FAIL"
    )
    require(root["status"] == expected_status, "treatment: status mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--base-contract",
        type=Path,
        default=DEFAULT_BASE_CONTRACT,
    )
    parser.add_argument(
        "--delta-contract",
        type=Path,
        default=DEFAULT_DELTA_CONTRACT,
    )
    parser.add_argument(
        "--physical-gate",
        type=Path,
        default=DEFAULT_PHYSICAL_GATE,
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
    connector: ReadyConnector | None = None
    try:
        base_contract = w5.load_contract(args.base_contract)
        delta_contract = w6.load_contract(
            args.delta_contract,
            base_contract,
        )
        physical_gate_sha256 = w6.sha256(args.physical_gate.read_bytes())
        contract = load_contract(
            args.contract,
            base_contract,
            delta_contract,
            physical_gate_sha256,
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

        request_start_ns = time.monotonic_ns()
        w5.write_atomic(args.start_marker, {
            "process_pid": os.getpid(),
            "request_start_ns": request_start_ns,
            "run_id": args.run_id,
            "schema": START_SCHEMA,
        })
        connector = ReadyConnector(
            args.cuda_route,
            contract.max_cuda_ready_us / 1_000_000,
            contract.connector_poll_us,
        )
        connector.start()

        phone_base = 10000
        catchup_base = 30000
        control_base = 40000
        active["phone"] = (contract.batch, phone_base)
        phone_tokens, service_metrics = serve_until_ready(
            clients["phone"],
            prompts,
            phone_base,
            contract.phone_prefill_chunk,
            contract.min_phone_service_tokens,
            contract.max_phone_service_tokens,
            connector,
        )
        connection = connector.take()
        clients["cuda"] = connection.client
        cuda_hello = connection.hello
        w5.validate_hello("cuda", cuda_hello, base_contract)
        require_same_model(
            {"cuda": cuda_hello, "phone": phone_hello},
            expected_model_sha256=base_contract.model_sha256,
            expected_file_type=base_contract.file_type,
        )
        service_tokens = len(phone_tokens[0])
        require(
            all(len(tokens) == service_tokens for tokens in phone_tokens),
            "phone service: nonrectangular output",
        )
        phone_status = clients["phone"].status()
        require(
            phone_status.active_sequences == contract.batch,
            "phone service: state count",
        )
        service_histories = [
            list(prompt) + list(tokens)
            for prompt, tokens in zip(prompts, phone_tokens)
        ]

        active["cuda"] = (contract.batch, catchup_base)
        phone_result, cuda_result, concurrency = w6.run_concurrently(
            lambda: w6.advance_active(
                clients["phone"],
                [tokens[-1] for tokens in phone_tokens],
                phone_base,
                base_contract.prompt_tokens + service_tokens - 1,
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
            "treatment: concurrent replay overlap",
        )
        phone_frontier_status = clients["phone"].status()
        require(
            phone_frontier_status.active_sequences == contract.batch,
            "treatment: phone frontier state",
        )
        frontier_histories = [
            list(history) + list(tokens)
            for history, tokens in zip(service_histories, delta_tokens)
        ]
        frontier_sha = w5.histories_digest(frontier_histories)
        published_phone_tokens = service_tokens + contract.phone_delta_tokens
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
            base_contract.prompt_tokens + service_tokens,
            contract.cuda_delta_chunk,
        )
        cuda_prepared = clients["cuda"].status()
        require(
            cuda_prepared.active_sequences == contract.batch,
            "treatment: CUDA prepared state",
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
            "treatment: phone state leak",
        )

        cuda_tokens, cuda_continuation_metrics = w6.continue_from_prediction(
            clients["cuda"],
            cuda_prediction,
            catchup_base,
            base_contract.prompt_tokens
            + service_tokens
            + contract.phone_delta_tokens,
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
            "treatment: CUDA state leak",
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
            "treatment: CUDA control state leak",
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
        ready_value = {
            "attempts": connection.attempts,
            "cuda_ready_ns": connection.ready_ns,
            "max_inter_batch_gap_us": max_inter_batch_gap_us(service_metrics),
            "ownership_commit_ns": ownership_commit_ns,
            "phone_first_token_ns": token_ready_ns[0],
            "phone_frontier_ns": token_ready_ns[-1],
            "phone_service_tokens": service_tokens,
            "request_complete_ns": request_complete_ns,
            "request_start_ns": request_start_ns,
            "useful_phone_tokens_before_ready": sum(
                item < connection.ready_ns for item in token_ready_ns
            ),
        }
        sequences = [
            {
                "control_continuation": list(control_tokens[index]),
                "cuda_continuation": list(cuda_tokens[index]),
                "final_published_tokens": (
                    list(phone_tokens[index])
                    + list(delta_tokens[index])
                    + list(cuda_tokens[index])
                ),
                "phone_delta": list(delta_tokens[index]),
                "phone_service": list(phone_tokens[index]),
                "prompt_id": base_contract.prompt_ids[index],
                "prompt_tokens": list(prompts[index]),
                "sequence_index": index,
            }
            for index in range(contract.batch)
        ]
        exact = cuda_tokens == control_tokens
        report: dict[str, object] = {
            "base_contract_sha256": base_contract.raw_sha256,
            "batch": contract.batch,
            "concurrency": concurrency,
            "contract_sha256": contract.raw_sha256,
            "corpus_manifest_sha256": base_contract.manifest_sha256,
            "corpus_sha256": base_contract.corpus_sha256,
            "cuda_ready": ready_value,
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
                "phone_service": service_metrics_value(service_metrics),
            },
            "model_sha256": base_contract.model_sha256,
            "physical_gate_sha256": contract.physical_gate_sha256,
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
                "cuda_control_released": (
                    cuda_control_released.active_sequences
                ),
                "cuda_prepared": cuda_prepared.active_sequences,
                "phone_after_commit": phone_after_commit.active_sequences,
                "phone_at_frontier": (
                    phone_frontier_status.active_sequences
                ),
            },
            "status": (
                "COLD_PROMOTION_TREATMENT_PASS"
                if exact
                else "COLD_PROMOTION_TREATMENT_FAIL"
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
        return 0 if exact else 3
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
        OSError,
        KeyError,
        ProtocolError,
        PromotionError,
        TypeError,
        w5.HandoffError,
        w6.DeltaError,
        ValueError,
    ) as exc:
        print(w6.canonical({
            "error": str(exc),
            "status": "COLD_PROMOTION_TREATMENT_ERROR",
        }).decode("ascii"), end="")
        raise SystemExit(2)
