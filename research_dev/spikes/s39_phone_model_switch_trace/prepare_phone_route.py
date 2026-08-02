#!/usr/bin/env python3
"""Prepare a resident phone route and detach with request state reset."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import phone_cuda_cold_promotion_probe as promotion
import phone_cuda_delta_probe as w6
import phone_cuda_handoff_probe as w5
from async_pipeline import parse_endpoint
from qwen25_quality_probe import load_corpus
from stage_v3_client import ProtocolError, StageV3Client


SCHEMA = "s39-phone-route-preparation-v1"


def validate_report(
    report: object,
    contract: promotion.PromotionContract,
    base_contract: w5.Contract,
    run_id: str,
) -> None:
    root = w6.exact_keys(
        report,
        {
            "base_contract_sha256",
            "batch",
            "contract_sha256",
            "ended_ns",
            "hello",
            "metrics",
            "model_sha256",
            "prompts",
            "request_state_reset",
            "run_id",
            "scheduler_eligible",
            "schema",
            "scope",
            "sequences",
            "session_end",
            "started_ns",
            "state_count",
            "status",
            "warmup_generated_tokens",
        },
        "preparation",
    )
    promotion.require(
        root["schema"] == SCHEMA
        and root["scope"] == "MECHANICS_ONLY"
        and root["scheduler_eligible"] is False
        and root["status"] == "PHONE_ROUTE_PREPARATION_PASS",
        "preparation: labels",
    )
    promotion.require(
        root["contract_sha256"] == contract.raw_sha256
        and root["base_contract_sha256"] == base_contract.raw_sha256
        and root["model_sha256"] == base_contract.model_sha256
        and root["run_id"] == run_id,
        "preparation: identity",
    )
    promotion.require(
        root["batch"] == contract.batch
        and root["prompts"] == list(base_contract.prompt_ids)
        and root["warmup_generated_tokens"] == contract.warmup_generated_tokens
        and root["session_end"] == "DETACH"
        and root["request_state_reset"] is True
        and root["state_count"] == 0,
        "preparation: execution contract",
    )
    promotion.require(
        w6.is_int(root["started_ns"])
        and w6.is_int(root["ended_ns"])
        and 0 < root["started_ns"] < root["ended_ns"],
        "preparation: timing",
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
        "preparation.hello",
    )
    promotion.require(
        type(hello["model_sha256"]) is str
        and all(
            w6.is_int(value)
            for name, value in hello.items()
            if name != "model_sha256"
        ),
        "preparation: hello types",
    )
    w5.validate_hello("phone", promotion.Hello(**hello), base_contract)

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
        "preparation.metrics",
    )
    expected_rows = contract.batch * (
        base_contract.prompt_tokens + contract.warmup_generated_tokens - 1
    )
    expected_history_batches = (
        base_contract.prompt_tokens + contract.phone_prefill_chunk - 1
    ) // contract.phone_prefill_chunk
    promotion.require(
        metrics["rows"] == expected_rows
        and metrics["history_batches"] == expected_history_batches
        and metrics["continuation_batches"]
        == contract.warmup_generated_tokens - 1
        and type(metrics["batch_timeline"]) is list
        and type(metrics["token_ready_ns"]) is list
        and len(metrics["token_ready_ns"]) == contract.warmup_generated_tokens,
        "preparation: metric accounting",
    )

    sequences = root["sequences"]
    promotion.require(
        type(sequences) is list and len(sequences) == contract.batch,
        "preparation: sequence count",
    )
    for index, sequence in enumerate(sequences):
        value = w6.exact_keys(
            sequence,
            {
                "generated_tokens",
                "prompt_id",
                "prompt_tokens",
                "sequence_index",
            },
            f"preparation.sequences.{index}",
        )
        promotion.require(
            value["sequence_index"] == index
            and value["prompt_id"] == base_contract.prompt_ids[index]
            and type(value["prompt_tokens"]) is list
            and len(value["prompt_tokens"]) == base_contract.prompt_tokens
            and type(value["generated_tokens"]) is list
            and len(value["generated_tokens"])
            == contract.warmup_generated_tokens
            and all(
                w6.is_int(token) and token >= 0
                for token in value["prompt_tokens"] + value["generated_tokens"]
            ),
            "preparation: sequence",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=promotion.DEFAULT_CONTRACT,
    )
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
    parser.add_argument("--phone-route", type=parse_endpoint, required=True)
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
    detached = False
    try:
        base_contract = w5.load_contract(args.base_contract)
        delta_contract = w6.load_contract(args.delta_contract, base_contract)
        contract = promotion.load_contract(
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
        client = StageV3Client.connect(*args.phone_route, args.timeout)
        hello = client.hello()
        w5.validate_hello("phone", hello, base_contract)
        started_ns = time.monotonic_ns()
        active = True
        tokens, metrics = promotion.run_fixed_generation(
            client,
            prompts,
            70000,
            contract.phone_prefill_chunk,
            contract.warmup_generated_tokens,
        )
        w5.remove_group(client, contract.batch, 70000)
        active = False
        status = client.status()
        promotion.require(
            status.active_sequences == 0,
            "preparation: state leak",
        )
        ended_ns = time.monotonic_ns()
        w5.finish(client, "detach")
        detached = True
        report: dict[str, object] = {
            "base_contract_sha256": base_contract.raw_sha256,
            "batch": contract.batch,
            "contract_sha256": contract.raw_sha256,
            "ended_ns": ended_ns,
            "hello": asdict(hello),
            "metrics": promotion.service_metrics_value(metrics),
            "model_sha256": base_contract.model_sha256,
            "prompts": list(base_contract.prompt_ids),
            "request_state_reset": True,
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
            "session_end": "DETACH",
            "started_ns": started_ns,
            "state_count": status.active_sequences,
            "status": "PHONE_ROUTE_PREPARATION_PASS",
            "warmup_generated_tokens": contract.warmup_generated_tokens,
        }
        validate_report(report, contract, base_contract, args.run_id)
        w5.write_atomic(args.output, report)
        print(w6.canonical(report).decode("ascii"), end="")
        return 0
    finally:
        if client is not None:
            if active:
                try:
                    w5.remove_group(client, contract.batch, 70000)
                except (OSError, ProtocolError, w5.HandoffError):
                    pass
            if not detached:
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
        promotion.PromotionError,
        w5.HandoffError,
        w6.DeltaError,
    ) as exc:
        print(w6.canonical({
            "error": str(exc),
            "status": "PHONE_ROUTE_PREPARATION_ERROR",
        }).decode("ascii"), end="")
        raise SystemExit(2)
