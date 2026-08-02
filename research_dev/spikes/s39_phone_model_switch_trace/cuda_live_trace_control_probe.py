#!/usr/bin/env python3
"""Replay the treatment token trace on a fresh CUDA route."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import cuda_cold_control_probe as cold_control
import cuda_live_control_probe as greedy_control
import phone_cuda_cold_promotion_probe as cold
import phone_cuda_delta_probe as w6
import phone_cuda_handoff_probe as w5
import phone_cuda_live_promotion_probe as live
from async_pipeline import parse_endpoint
from stage_v3_client import BatchRow, ProtocolError, StageV3Client


SCHEMA = "s39-live-session-trace-control-v1"
MODE = "TEACHER_FORCED_TRACE_REPLAY"


def load_treatment_trace(
    path: Path,
    contract: live.LiveContract,
    base_contract: w5.Contract,
    run_id: str,
) -> tuple[list[list[int]], list[list[int]]]:
    value, _ = w6.read_canonical(path, "live_treatment")
    live.require(
        value.get("schema") == live.SCHEMA
        and value.get("run_id") == run_id
        and value.get("contract_sha256") == contract.raw_sha256
        and value.get("base_contract_sha256") == base_contract.raw_sha256
        and value.get("batch") == contract.batch
        and value.get("prompts") == list(base_contract.prompt_ids)
        and value.get("status") == "LIVE_SESSION_PROMOTION_PASS",
        "trace_control: treatment identity",
    )
    sequences = value.get("sequences")
    live.require(
        type(sequences) is list and len(sequences) == contract.batch,
        "trace_control: treatment shape",
    )
    histories = []
    trace = []
    for index, sequence in enumerate(sequences):
        live.require(
            type(sequence) is dict
            and sequence.get("sequence_index") == index
            and sequence.get("prompt_id") == base_contract.prompt_ids[index],
            "trace_control: sequence identity",
        )
        fields = {
            "prompt_tokens": base_contract.prompt_tokens,
            "preexisting_tokens": contract.preexisting_committed_tokens,
            "phone_delta": contract.phone_delta_tokens,
            "cuda_continuation": contract.cuda_continuation_tokens,
        }
        ready = value.get("cuda_ready")
        live.require(type(ready) is dict, "trace_control: missing readiness")
        service_tokens = ready.get("phone_service_tokens")
        live.require(
            w6.is_int(service_tokens)
            and contract.min_phone_service_tokens
            <= service_tokens
            <= contract.max_phone_service_tokens,
            "trace_control: service count",
        )
        fields["phone_service"] = service_tokens
        for name, width in fields.items():
            tokens = sequence.get(name)
            live.require(
                type(tokens) is list
                and len(tokens) == width
                and all(w6.is_int(token) and token >= 0 for token in tokens),
                f"trace_control: invalid {name}",
            )
        histories.append(
            list(sequence["prompt_tokens"])
            + list(sequence["preexisting_tokens"])
        )
        trace.append(
            list(sequence["phone_service"])
            + list(sequence["phone_delta"])
            + list(sequence["cuda_continuation"])
        )
    live.require(
        len({len(tokens) for tokens in trace}) == 1,
        "trace_control: nonrectangular trace",
    )
    return histories, trace


def run_teacher_forced(
    client: StageV3Client,
    histories: Sequence[Sequence[int]],
    trace: Sequence[Sequence[int]],
    identity_base: int,
    history_chunk: int,
) -> tuple[list[list[int]], cold.ServiceMetrics]:
    live.require(
        bool(histories)
        and len(histories) == len(trace)
        and history_chunk > 0,
        "trace_control: invalid replay input",
    )
    history_width = len(histories[0])
    trace_width = len(trace[0])
    live.require(
        history_width > 0
        and trace_width > 0
        and all(len(tokens) == history_width for tokens in histories)
        and all(
            len(tokens) == trace_width
            and all(w6.is_int(token) and token >= 0 for token in tokens)
            for tokens in trace
        ),
        "trace_control: invalid replay geometry",
    )
    batch = len(histories)
    intervals: list[cold.BatchInterval] = []
    predictions: list[list[int]] = [[] for _ in histories]
    history_batches = 0
    current: list[int] = []
    for start in range(0, history_width, history_chunk):
        end = min(start + history_chunk, history_width)
        rows = w5.build_rows(histories, identity_base, start, end)
        results, interval = cold.timed_rows(
            client,
            rows,
            "PREFILL",
            start,
            end,
        )
        intervals.append(interval)
        history_batches += 1
        current = w5.select_predictions(
            results,
            batch,
            end - start,
            identity_base,
            end - 1,
        )
    for seq_id, token in enumerate(current):
        predictions[seq_id].append(token)
    token_ready_ns = [intervals[-1].ended_ns]

    for offset in range(trace_width - 1):
        position = history_width + offset
        rows = [
            BatchRow(
                identity_base + seq_id,
                identity_base + seq_id,
                seq_id,
                position,
                trace[seq_id][offset],
            )
            for seq_id in range(batch)
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
            batch,
            1,
            identity_base,
            position,
        )
        for seq_id, token in enumerate(current):
            predictions[seq_id].append(token)
        token_ready_ns.append(interval.ended_ns)

    return predictions, cold.ServiceMetrics(
        tuple(intervals),
        trace_width - 1,
        sum(
            (interval.ended_ns - interval.started_ns) // 1000
            for interval in intervals
        ),
        history_batches,
        sum(interval.rows for interval in intervals),
        tuple(token_ready_ns),
    )


def validate_report(
    report: object,
    contract: live.LiveContract,
    base_contract: w5.Contract,
    run_id: str,
    trace: Sequence[Sequence[int]],
) -> None:
    root = w6.exact_keys(
        report,
        {
            "base_contract_sha256",
            "batch",
            "connector_attempts",
            "contract_sha256",
            "control_mode",
            "cuda_launch_ns",
            "cuda_ready_ns",
            "first_token_ns",
            "greedy_agreement",
            "hello",
            "metrics",
            "model_sha256",
            "post_start_tokens",
            "preexisting_committed_tokens",
            "prompts",
            "request_complete_ns",
            "request_start_ns",
            "run_id",
            "scheduler_eligible",
            "schema",
            "scope",
            "sequences",
            "state_count",
            "status",
        },
        "trace_control",
    )
    live.require(
        root["schema"] == SCHEMA
        and root["control_mode"] == MODE
        and root["scope"] == "MECHANICS_ONLY"
        and root["scheduler_eligible"] is False
        and root["status"] == "LIVE_SESSION_SERVER_TRACE_CONTROL_PASS",
        "trace_control: labels",
    )
    live.require(
        bool(trace)
        and len(trace) == contract.batch
        and len({len(tokens) for tokens in trace}) == 1,
        "trace_control: expected trace",
    )
    post_start_tokens = len(trace[0])
    sequences = root["sequences"]
    live.require(
        type(sequences) is list and len(sequences) == contract.batch,
        "trace_control: sequence count",
    )
    predicted = []
    replayed = []
    adapter_sequences = []
    for index, sequence in enumerate(sequences):
        value = w6.exact_keys(
            sequence,
            {
                "predicted_tokens",
                "preexisting_tokens",
                "prompt_id",
                "prompt_tokens",
                "replayed_tokens",
                "sequence_index",
            },
            f"trace_control.sequences.{index}",
        )
        live.require(
            value["sequence_index"] == index
            and value["prompt_id"] == base_contract.prompt_ids[index]
            and type(value["prompt_tokens"]) is list
            and len(value["prompt_tokens"]) == base_contract.prompt_tokens
            and type(value["preexisting_tokens"]) is list
            and len(value["preexisting_tokens"])
            == contract.preexisting_committed_tokens
            and type(value["predicted_tokens"]) is list
            and len(value["predicted_tokens"]) == post_start_tokens
            and type(value["replayed_tokens"]) is list
            and value["replayed_tokens"] == list(trace[index])
            and all(
                w6.is_int(token) and token >= 0
                for token in (
                    value["prompt_tokens"]
                    + value["preexisting_tokens"]
                    + value["predicted_tokens"]
                    + value["replayed_tokens"]
                )
            ),
            "trace_control: sequence",
        )
        predicted.append(value["predicted_tokens"])
        replayed.append(value["replayed_tokens"])
        adapter_sequences.append({
            "generated_tokens": value["predicted_tokens"],
            "preexisting_tokens": value["preexisting_tokens"],
            "prompt_id": value["prompt_id"],
            "prompt_tokens": value["prompt_tokens"],
            "sequence_index": value["sequence_index"],
        })

    matches = sum(
        predicted_token == replayed_token
        for predicted_tokens, replayed_tokens in zip(predicted, replayed)
        for predicted_token, replayed_token in zip(
            predicted_tokens,
            replayed_tokens,
        )
    )
    total = contract.batch * post_start_tokens
    agreement = w6.exact_keys(
        root["greedy_agreement"],
        {"all_match", "matching_tokens", "total_tokens"},
        "trace_control.greedy_agreement",
    )
    live.require(
        agreement == {
            "all_match": matches == total,
            "matching_tokens": matches,
            "total_tokens": total,
        },
        "trace_control: greedy agreement",
    )

    adapter = {
        key: value
        for key, value in root.items()
        if key not in {"control_mode", "greedy_agreement"}
    }
    adapter["schema"] = greedy_control.SCHEMA
    adapter["status"] = "LIVE_SESSION_SERVER_CONTROL_PASS"
    adapter["sequences"] = adapter_sequences
    greedy_control.validate_report(
        adapter,
        contract,
        base_contract,
        run_id,
        post_start_tokens,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
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
    parser.add_argument("--treatment-report", type=Path, required=True)
    parser.add_argument("--launch-record", type=Path, required=True)
    parser.add_argument("--cuda-route", type=parse_endpoint, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    if (
        args.timeout <= 0
        or w6.HEX64.fullmatch(args.run_id) is None
        or args.output.exists()
    ):
        parser.error("invalid arguments or existing output")

    client: StageV3Client | None = None
    active = False
    stopped = False
    connector: cold.ReadyConnector | None = None
    try:
        base_contract = w5.load_contract(args.base_contract)
        delta_contract = w6.load_contract(args.delta_contract, base_contract)
        contract = live.load_contract(
            args.contract,
            base_contract,
            delta_contract,
            w6.sha256(args.physical_gate.read_bytes()),
        )
        launch = cold_control.load_launch_record(
            args.launch_record,
            args.run_id,
            "CONTROL",
        )
        live.require(
            launch["cuda_launch_ns"] - launch["request_start_ns"]
            <= contract.max_launch_delay_us * 1000,
            "trace_control: CUDA launch delay",
        )
        histories, trace = load_treatment_trace(
            args.treatment_report,
            contract,
            base_contract,
            args.run_id,
        )
        connector = cold.ReadyConnector(
            args.cuda_route,
            contract.max_cuda_ready_us / 1_000_000,
            contract.connector_poll_us,
        )
        connector.start()
        connection = connector.take()
        client = connection.client
        hello = connection.hello
        w5.validate_hello("cuda", hello, base_contract)
        active = True
        predictions, metrics = run_teacher_forced(
            client,
            histories,
            trace,
            50000,
            contract.cuda_control_chunk,
        )
        w5.remove_group(client, contract.batch, 50000)
        active = False
        status = client.status()
        live.require(status.active_sequences == 0, "trace_control: state leak")
        w5.finish(client, "stop")
        stopped = True
        matches = sum(
            predicted == expected
            for predicted_tokens, expected_tokens in zip(predictions, trace)
            for predicted, expected in zip(predicted_tokens, expected_tokens)
        )
        total = contract.batch * len(trace[0])
        report: dict[str, object] = {
            "base_contract_sha256": base_contract.raw_sha256,
            "batch": contract.batch,
            "connector_attempts": connection.attempts,
            "contract_sha256": contract.raw_sha256,
            "control_mode": MODE,
            "cuda_launch_ns": launch["cuda_launch_ns"],
            "cuda_ready_ns": connection.ready_ns,
            "first_token_ns": metrics.token_ready_ns[0],
            "greedy_agreement": {
                "all_match": matches == total,
                "matching_tokens": matches,
                "total_tokens": total,
            },
            "hello": asdict(hello),
            "metrics": cold.service_metrics_value(metrics),
            "model_sha256": base_contract.model_sha256,
            "post_start_tokens": len(trace[0]),
            "preexisting_committed_tokens": (
                contract.preexisting_committed_tokens
            ),
            "prompts": list(base_contract.prompt_ids),
            "request_complete_ns": metrics.token_ready_ns[-1],
            "request_start_ns": launch["request_start_ns"],
            "run_id": args.run_id,
            "scheduler_eligible": False,
            "schema": SCHEMA,
            "scope": "MECHANICS_ONLY",
            "sequences": [
                {
                    "predicted_tokens": list(predictions[index]),
                    "preexisting_tokens": list(
                        histories[index][base_contract.prompt_tokens:]
                    ),
                    "prompt_id": base_contract.prompt_ids[index],
                    "prompt_tokens": list(
                        histories[index][:base_contract.prompt_tokens]
                    ),
                    "replayed_tokens": list(trace[index]),
                    "sequence_index": index,
                }
                for index in range(contract.batch)
            ],
            "state_count": status.active_sequences,
            "status": "LIVE_SESSION_SERVER_TRACE_CONTROL_PASS",
        }
        validate_report(
            report,
            contract,
            base_contract,
            args.run_id,
            trace,
        )
        w5.write_atomic(args.output, report)
        print(w6.canonical(report).decode("ascii"), end="")
        return 0
    finally:
        if connector is not None:
            connector.close()
        if client is not None:
            if active:
                try:
                    w5.remove_group(client, contract.batch, 50000)
                except (OSError, ProtocolError, w5.HandoffError):
                    pass
            if not stopped:
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
        live.LivePromotionError,
        w5.HandoffError,
        w6.DeltaError,
    ) as exc:
        print(w6.canonical({
            "error": str(exc),
            "status": "LIVE_SESSION_TRACE_CONTROL_ERROR",
        }).decode("ascii"), end="")
        raise SystemExit(2)
