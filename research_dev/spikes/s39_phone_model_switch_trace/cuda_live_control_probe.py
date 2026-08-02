#!/usr/bin/env python3
"""Run the fresh-CUDA control for an already-live token history."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import cuda_cold_control_probe as cold_control
import phone_cuda_cold_promotion_probe as cold
import phone_cuda_delta_probe as w6
import phone_cuda_handoff_probe as w5
import phone_cuda_live_promotion_probe as live
from async_pipeline import parse_endpoint
from stage_v3_client import ProtocolError, StageV3Client


SCHEMA = "s39-live-session-control-v1"


def load_treatment_histories(
    path: Path,
    contract: live.LiveContract,
    base_contract: w5.Contract,
    run_id: str,
) -> tuple[list[list[int]], int]:
    value, _ = w6.read_canonical(path, "live_treatment")
    live.require(
        value.get("schema") == live.SCHEMA
        and value.get("run_id") == run_id
        and value.get("contract_sha256") == contract.raw_sha256
        and value.get("base_contract_sha256") == base_contract.raw_sha256
        and value.get("batch") == contract.batch
        and value.get("prompts") == list(base_contract.prompt_ids),
        "live_control: treatment identity",
    )
    sequences = value.get("sequences")
    ready = value.get("cuda_ready")
    live.require(
        type(sequences) is list
        and len(sequences) == contract.batch
        and type(ready) is dict,
        "live_control: treatment shape",
    )
    service_tokens = ready.get("phone_service_tokens")
    live.require(
        w6.is_int(service_tokens)
        and contract.min_phone_service_tokens
        <= service_tokens
        <= contract.max_phone_service_tokens,
        "live_control: service count",
    )
    histories = []
    for index, sequence in enumerate(sequences):
        live.require(
            type(sequence) is dict
            and sequence.get("sequence_index") == index
            and sequence.get("prompt_id") == base_contract.prompt_ids[index]
            and type(sequence.get("prompt_tokens")) is list
            and len(sequence["prompt_tokens"]) == base_contract.prompt_tokens
            and type(sequence.get("preexisting_tokens")) is list
            and len(sequence["preexisting_tokens"])
            == contract.preexisting_committed_tokens,
            "live_control: preexisting history",
        )
        histories.append(
            list(sequence["prompt_tokens"])
            + list(sequence["preexisting_tokens"])
        )
    return (
        histories,
        service_tokens
        + contract.phone_delta_tokens
        + contract.cuda_continuation_tokens,
    )


def validate_report(
    report: object,
    contract: live.LiveContract,
    base_contract: w5.Contract,
    run_id: str,
    post_start_tokens: int,
) -> None:
    root = w6.exact_keys(
        report,
        {
            "base_contract_sha256",
            "batch",
            "connector_attempts",
            "contract_sha256",
            "cuda_launch_ns",
            "cuda_ready_ns",
            "first_token_ns",
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
        "live_control",
    )
    live.require(
        root["schema"] == SCHEMA
        and root["scope"] == "MECHANICS_ONLY"
        and root["scheduler_eligible"] is False
        and root["status"] == "LIVE_SESSION_SERVER_CONTROL_PASS",
        "live_control: labels",
    )
    live.require(
        root["run_id"] == run_id
        and root["contract_sha256"] == contract.raw_sha256
        and root["base_contract_sha256"] == base_contract.raw_sha256
        and root["model_sha256"] == base_contract.model_sha256,
        "live_control: identity",
    )
    live.require(
        root["batch"] == contract.batch
        and root["prompts"] == list(base_contract.prompt_ids)
        and root["preexisting_committed_tokens"]
        == contract.preexisting_committed_tokens
        and root["post_start_tokens"] == post_start_tokens,
        "live_control: workload",
    )
    for field in (
        "connector_attempts",
        "cuda_launch_ns",
        "cuda_ready_ns",
        "first_token_ns",
        "post_start_tokens",
        "preexisting_committed_tokens",
        "request_complete_ns",
        "request_start_ns",
        "state_count",
    ):
        live.require(
            w6.is_int(root[field]) and root[field] >= 0,
            f"live_control: invalid {field}",
        )
    live.require(
        root["request_start_ns"]
        <= root["cuda_launch_ns"]
        < root["cuda_ready_ns"]
        < root["first_token_ns"]
        <= root["request_complete_ns"],
        "live_control: timing order",
    )
    hello = w6.exact_keys(
        root["hello"],
        {
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
        },
        "live_control.hello",
    )
    live.require(
        type(hello["model_sha256"]) is str
        and all(
            w6.is_int(value)
            for name, value in hello.items()
            if name != "model_sha256"
        ),
        "live_control: hello types",
    )
    w5.validate_hello("cuda", live.Hello(**hello), base_contract)
    metrics = w6.exact_keys(
        root["metrics"],
        {
            "batch_timeline",
            "continuation_batches",
            "elapsed_us",
            "history_batches",
            "rows",
            "token_ready_ns",
        },
        "live_control.metrics",
    )
    history_width = (
        base_contract.prompt_tokens + contract.preexisting_committed_tokens
    )
    history_batches = (
        history_width + contract.cuda_control_chunk - 1
    ) // contract.cuda_control_chunk
    continuation_batches = post_start_tokens - 1
    live.require(
        all(
            w6.is_int(metrics[field]) and metrics[field] >= 0
            for field in (
                "continuation_batches",
                "elapsed_us",
                "history_batches",
                "rows",
            )
        )
        and type(metrics["batch_timeline"]) is list
        and type(metrics["token_ready_ns"]) is list,
        "live_control: metric types",
    )
    live.require(
        metrics["history_batches"] == history_batches
        and metrics["continuation_batches"] == continuation_batches
        and metrics["rows"]
        == contract.batch * (history_width + post_start_tokens - 1)
        and len(metrics["token_ready_ns"]) == post_start_tokens
        and root["first_token_ns"] == metrics["token_ready_ns"][0]
        and root["request_complete_ns"] == metrics["token_ready_ns"][-1],
        "live_control: metric accounting",
    )
    timeline = metrics["batch_timeline"]
    live.require(
        len(timeline) == history_batches + continuation_batches,
        "live_control: timeline count",
    )
    expected_ranges = [
        (
            "PREFILL",
            start,
            min(start + contract.cuda_control_chunk, history_width),
        )
        for start in range(0, history_width, contract.cuda_control_chunk)
    ]
    expected_ranges.extend(
        ("DECODE", history_width + offset, history_width + offset + 1)
        for offset in range(continuation_batches)
    )
    elapsed_us = 0
    previous_end = None
    token_ready_ns = []
    for index, (item, expected) in enumerate(zip(timeline, expected_ranges)):
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
            f"live_control.metrics.batch_timeline.{index}",
        )
        kind, position_start, position_end = expected
        expected_rows = contract.batch * (
            position_end - position_start
        )
        live.require(
            interval["kind"] == kind
            and all(
                w6.is_int(value)
                for name, value in interval.items()
                if name != "kind"
            )
            and interval["position_start"] == position_start
            and interval["position_end"] == position_end
            and interval["rows"] == expected_rows
            and interval["started_ns"] < interval["ended_ns"]
            and (
                previous_end is None
                or interval["started_ns"] >= previous_end
            ),
            "live_control: invalid timeline",
        )
        elapsed_us += (
            interval["ended_ns"] - interval["started_ns"]
        ) // 1000
        previous_end = interval["ended_ns"]
        if index == history_batches - 1 or index >= history_batches:
            token_ready_ns.append(interval["ended_ns"])
    live.require(
        metrics["elapsed_us"] == elapsed_us
        and metrics["token_ready_ns"] == token_ready_ns,
        "live_control: timeline accounting",
    )
    sequences = root["sequences"]
    live.require(
        type(sequences) is list and len(sequences) == contract.batch,
        "live_control: sequence count",
    )
    for index, sequence in enumerate(sequences):
        value = w6.exact_keys(
            sequence,
            {
                "generated_tokens",
                "preexisting_tokens",
                "prompt_id",
                "prompt_tokens",
                "sequence_index",
            },
            f"live_control.sequences.{index}",
        )
        live.require(
            value["sequence_index"] == index
            and value["prompt_id"] == base_contract.prompt_ids[index]
            and type(value["prompt_tokens"]) is list
            and len(value["prompt_tokens"]) == base_contract.prompt_tokens
            and type(value["preexisting_tokens"]) is list
            and len(value["preexisting_tokens"])
            == contract.preexisting_committed_tokens
            and type(value["generated_tokens"]) is list
            and len(value["generated_tokens"]) == post_start_tokens
            and all(
                w6.is_int(token) and token >= 0
                for token in (
                    value["prompt_tokens"]
                    + value["preexisting_tokens"]
                    + value["generated_tokens"]
                )
            ),
            "live_control: sequence",
        )
    live.require(root["state_count"] == 0, "live_control: state leak")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=live.DEFAULT_CONTRACT)
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
            "live_control: CUDA launch delay",
        )
        histories, post_start_tokens = load_treatment_histories(
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
        tokens, metrics = cold.run_fixed_generation(
            client,
            histories,
            50000,
            contract.cuda_control_chunk,
            post_start_tokens,
        )
        w5.remove_group(client, contract.batch, 50000)
        active = False
        status = client.status()
        live.require(status.active_sequences == 0, "live_control: state leak")
        w5.finish(client, "stop")
        stopped = True
        report: dict[str, object] = {
            "base_contract_sha256": base_contract.raw_sha256,
            "batch": contract.batch,
            "connector_attempts": connection.attempts,
            "contract_sha256": contract.raw_sha256,
            "cuda_launch_ns": launch["cuda_launch_ns"],
            "cuda_ready_ns": connection.ready_ns,
            "first_token_ns": metrics.token_ready_ns[0],
            "hello": asdict(hello),
            "metrics": cold.service_metrics_value(metrics),
            "model_sha256": base_contract.model_sha256,
            "post_start_tokens": post_start_tokens,
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
                    "generated_tokens": list(tokens[index]),
                    "preexisting_tokens": list(
                        histories[index][base_contract.prompt_tokens:]
                    ),
                    "prompt_id": base_contract.prompt_ids[index],
                    "prompt_tokens": list(
                        histories[index][:base_contract.prompt_tokens]
                    ),
                    "sequence_index": index,
                }
                for index in range(contract.batch)
            ],
            "state_count": status.active_sequences,
            "status": "LIVE_SESSION_SERVER_CONTROL_PASS",
        }
        validate_report(
            report,
            contract,
            base_contract,
            args.run_id,
            post_start_tokens,
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
            "status": "LIVE_SESSION_CONTROL_ERROR",
        }).decode("ascii"), end="")
        raise SystemExit(2)
