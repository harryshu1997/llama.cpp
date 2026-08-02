#!/usr/bin/env python3
"""Run the matched server-queue control with fresh CUDA workers."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import phone_cuda_cold_promotion_probe as promotion
import phone_cuda_delta_probe as w6
import phone_cuda_handoff_probe as w5
from async_pipeline import parse_endpoint
from qwen25_quality_probe import load_corpus
from stage_v3_client import ProtocolError, StageV3Client


SCHEMA = "s39-cold-promotion-control-v1"
LAUNCH_SCHEMA = "s39-cuda-launch-v1"


def load_launch_record(
    path: Path,
    run_id: str,
    phase: str,
) -> dict[str, object]:
    value, _ = w6.read_canonical(path, "launch_record")
    root = w6.exact_keys(
        value,
        {
            "cuda_launch_ns",
            "phase",
            "preexisting_route_pids",
            "request_start_ns",
            "run_id",
            "schema",
        },
        "launch_record",
    )
    promotion.require(root["schema"] == LAUNCH_SCHEMA, "launch_record: schema")
    promotion.require(root["phase"] == phase, "launch_record: phase")
    promotion.require(root["run_id"] == run_id, "launch_record: run ID")
    promotion.require(
        w6.is_int(root["request_start_ns"])
        and w6.is_int(root["cuda_launch_ns"])
        and 0 < root["request_start_ns"] <= root["cuda_launch_ns"],
        "launch_record: invalid timing",
    )
    promotion.require(
        root["preexisting_route_pids"] == [],
        "launch_record: CUDA route was already resident",
    )
    return root


def load_treatment_token_count(
    path: Path,
    contract: promotion.PromotionContract,
    base_contract: w5.Contract,
    run_id: str,
) -> int:
    value, _ = w6.read_canonical(path, "treatment")
    promotion.require(
        value.get("schema") == promotion.SCHEMA
        and value.get("run_id") == run_id
        and value.get("contract_sha256") == contract.raw_sha256
        and value.get("base_contract_sha256") == base_contract.raw_sha256
        and value.get("batch") == contract.batch
        and value.get("prompts") == list(base_contract.prompt_ids),
        "treatment: identity mismatch",
    )
    ready = value.get("cuda_ready")
    promotion.require(type(ready) is dict, "treatment: missing readiness")
    service_tokens = ready.get("phone_service_tokens")
    promotion.require(
        w6.is_int(service_tokens)
        and contract.min_phone_service_tokens
        <= service_tokens
        <= contract.max_phone_service_tokens,
        "treatment: invalid service-token count",
    )
    return (
        service_tokens
        + contract.phone_delta_tokens
        + contract.cuda_continuation_tokens
    )


def validate_report(
    report: object,
    contract: promotion.PromotionContract,
    base_contract: w5.Contract,
    run_id: str,
    total_tokens: int,
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
            "total_generated_tokens",
        },
        "control",
    )
    promotion.require(root["schema"] == SCHEMA, "control: schema")
    promotion.require(root["scope"] == "MECHANICS_ONLY", "control: scope")
    promotion.require(root["scheduler_eligible"] is False, "control: eligibility")
    promotion.require(
        root["run_id"] == run_id
        and root["contract_sha256"] == contract.raw_sha256
        and root["base_contract_sha256"] == base_contract.raw_sha256
        and root["model_sha256"] == base_contract.model_sha256,
        "control: identity mismatch",
    )
    promotion.require(
        root["batch"] == contract.batch
        and root["prompts"] == list(base_contract.prompt_ids)
        and root["total_generated_tokens"] == total_tokens,
        "control: workload mismatch",
    )
    for field in (
        "connector_attempts",
        "cuda_launch_ns",
        "cuda_ready_ns",
        "first_token_ns",
        "request_complete_ns",
        "request_start_ns",
        "state_count",
        "total_generated_tokens",
    ):
        promotion.require(
            w6.is_int(root[field]) and root[field] >= 0,
            f"control: invalid {field}",
        )
    promotion.require(
        root["request_start_ns"]
        <= root["cuda_launch_ns"]
        < root["cuda_ready_ns"]
        < root["first_token_ns"]
        <= root["request_complete_ns"],
        "control: timing order",
    )
    hello_value = w6.exact_keys(
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
        "control.hello",
    )
    promotion.require(
        type(hello_value["model_sha256"]) is str
        and all(
            w6.is_int(value)
            for name, value in hello_value.items()
            if name != "model_sha256"
        ),
        "control: invalid hello",
    )
    hello = promotion.Hello(**hello_value)
    w5.validate_hello("cuda", hello, base_contract)
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
        "control.metrics",
    )
    token_ready = metrics["token_ready_ns"]
    timeline = metrics["batch_timeline"]
    promotion.require(
        type(token_ready) is list
        and len(token_ready) == total_tokens
        and all(w6.is_int(value) and value > 0 for value in token_ready)
        and type(timeline) is list
        and bool(timeline),
        "control: metric shape",
    )
    previous_end = None
    elapsed_us = 0
    interval_keys = {
        "ended_ns",
        "kind",
        "position_end",
        "position_start",
        "rows",
        "started_ns",
    }
    for index, value in enumerate(timeline):
        interval = w6.exact_keys(
            value,
            interval_keys,
            f"control.metrics.batch_timeline.{index}",
        )
        promotion.require(
            interval["kind"] in {"PREFILL", "DECODE"}
            and all(
                w6.is_int(interval[field])
                for field in interval_keys - {"kind"}
            )
            and interval["started_ns"] < interval["ended_ns"]
            and interval["position_start"] < interval["position_end"]
            and interval["rows"] > 0
            and (
                previous_end is None
                or interval["started_ns"] >= previous_end
            ),
            "control: invalid batch timeline",
        )
        previous_end = interval["ended_ns"]
        elapsed_us += (
            interval["ended_ns"] - interval["started_ns"]
        ) // 1000
    derived_ready = [
        value["ended_ns"]
        for value in timeline
        if value["position_end"] == base_contract.prompt_tokens
        or value["kind"] == "DECODE"
    ]
    promotion.require(
        token_ready == derived_ready
        and root["first_token_ns"] == token_ready[0]
        and root["request_complete_ns"] == token_ready[-1],
        "control: token timing mismatch",
    )
    expected_history_batches = (
        base_contract.prompt_tokens + contract.cuda_control_chunk - 1
    ) // contract.cuda_control_chunk
    promotion.require(
        metrics["history_batches"] == expected_history_batches
        and metrics["continuation_batches"] == total_tokens - 1
        and metrics["rows"]
        == contract.batch * (base_contract.prompt_tokens + total_tokens - 1),
        "control: row accounting",
    )
    promotion.require(
        w6.is_int(metrics["elapsed_us"])
        and metrics["elapsed_us"] == elapsed_us,
        "control: elapsed accounting",
    )
    sequences = root["sequences"]
    promotion.require(
        type(sequences) is list and len(sequences) == contract.batch,
        "control: sequence count",
    )
    for index, sequence in enumerate(sequences):
        promotion.require(
            type(sequence) is dict
            and sequence.get("sequence_index") == index
            and sequence.get("prompt_id") == base_contract.prompt_ids[index]
            and sequence.get("prompt_tokens") is not None
            and len(sequence["prompt_tokens"]) == base_contract.prompt_tokens
            and sequence.get("generated_tokens") is not None
            and len(sequence["generated_tokens"]) == total_tokens
            and all(
                w6.is_int(token) and token >= 0
                for token in sequence["prompt_tokens"]
                + sequence["generated_tokens"]
            ),
            "control: invalid sequence",
        )
    promotion.require(root["state_count"] == 0, "control: state leak")
    promotion.require(root["status"] == "COLD_SERVER_QUEUE_CONTROL_PASS", "control: status")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=promotion.DEFAULT_CONTRACT)
    parser.add_argument(
        "--base-contract",
        type=Path,
        default=promotion.DEFAULT_BASE_CONTRACT,
    )
    parser.add_argument(
        "--delta-contract",
        type=Path,
        default=promotion.DEFAULT_DELTA_CONTRACT,
    )
    parser.add_argument(
        "--physical-gate",
        type=Path,
        default=promotion.DEFAULT_PHYSICAL_GATE,
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
    connector: promotion.ReadyConnector | None = None
    try:
        base_contract = w5.load_contract(args.base_contract)
        delta_contract = w6.load_contract(args.delta_contract, base_contract)
        contract = promotion.load_contract(
            args.contract,
            base_contract,
            delta_contract,
            w6.sha256(args.physical_gate.read_bytes()),
        )
        launch = load_launch_record(args.launch_record, args.run_id, "CONTROL")
        promotion.require(
            launch["cuda_launch_ns"] - launch["request_start_ns"]
            <= contract.max_launch_delay_us * 1000,
            "control: CUDA launch delay exceeded",
        )
        total_tokens = load_treatment_token_count(
            args.treatment_report,
            contract,
            base_contract,
            args.run_id,
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
        connector = promotion.ReadyConnector(
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
        tokens, metrics = promotion.run_fixed_generation(
            client,
            prompts,
            50000,
            contract.cuda_control_chunk,
            total_tokens,
        )
        w5.remove_group(client, contract.batch, 50000)
        active = False
        status = client.status()
        promotion.require(status.active_sequences == 0, "control: state leak")
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
            "metrics": promotion.service_metrics_value(metrics),
            "model_sha256": base_contract.model_sha256,
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
                    "prompt_id": base_contract.prompt_ids[index],
                    "prompt_tokens": list(prompts[index]),
                    "sequence_index": index,
                }
                for index in range(contract.batch)
            ],
            "state_count": status.active_sequences,
            "status": "COLD_SERVER_QUEUE_CONTROL_PASS",
            "total_generated_tokens": total_tokens,
        }
        validate_report(
            report,
            contract,
            base_contract,
            args.run_id,
            total_tokens,
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
        OSError,
        KeyError,
        ProtocolError,
        promotion.PromotionError,
        TypeError,
        w5.HandoffError,
        w6.DeltaError,
        ValueError,
    ) as exc:
        print(w6.canonical({
            "error": str(exc),
            "status": "COLD_SERVER_QUEUE_CONTROL_ERROR",
        }).decode("ascii"), end="")
        raise SystemExit(2)
