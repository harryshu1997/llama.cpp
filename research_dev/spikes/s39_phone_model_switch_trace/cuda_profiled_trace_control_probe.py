#!/usr/bin/env python3
"""Run the fresh-CUDA teacher-forced control for one W9 treatment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import cuda_live_trace_control_probe as trace_control
import phone_cuda_cold_promotion_probe as cold
import phone_cuda_delta_probe as w6
import phone_cuda_handoff_probe as w5
import phone_cuda_live_promotion_probe as live
import w9_profiled_cutover as w9
from async_pipeline import parse_endpoint
from stage_v3_client import ProtocolError, StageV3Client


SCHEMA = "s39-profiled-cutover-trace-control-v1"
READY_SCHEMA = "s39-profiled-cutover-control-ready-v1"


def load_launch(path: Path, run_id: str, pair_ordinal: str) -> dict[str, object]:
    value, _ = w6.read_canonical(path, "control_launch")
    root = w9.exact_keys(
        value,
        {
            "cuda_launch_ns",
            "launch_deadline_ns",
            "pair_ordinal",
            "phase",
            "preexisting_route_pids",
            "request_start_ns",
            "run_id",
            "schema",
        },
        "control_launch",
    )
    w9.require(
        root["schema"] == "s39-profiled-cutover-launch-v1"
        and root["phase"] == "CONTROL"
        and root["run_id"] == run_id
        and root["pair_ordinal"] == pair_ordinal
        and root["preexisting_route_pids"] == []
        and all(
            w9.is_int(root[name]) and root[name] > 0
            for name in (
                "cuda_launch_ns",
                "launch_deadline_ns",
                "request_start_ns",
            )
        )
        and root["request_start_ns"]
        < root["launch_deadline_ns"]
        <= root["cuda_launch_ns"],
        "control: invalid launch record",
    )
    return root


def load_trace(
    path: Path,
    contract: w9.CutoverContract,
    run_id: str,
    pair_ordinal: str,
) -> tuple[list[list[int]], list[list[int]], dict[str, object]]:
    value, _ = w6.read_canonical(path, "w9_treatment")
    w9.require(
        value.get("schema")
        == "s39-profiled-cutover-treatment-v1"
        and value.get("status") == "PROFILED_ZERO_EXTRA_TREATMENT_PASS"
        and value.get("contract_sha256") == contract.raw_sha256
        and value.get("run_id") == run_id
        and value.get("pair_ordinal") == pair_ordinal
        and value.get("batch") == contract.batch,
        "control: treatment identity",
    )
    sequences = value.get("sequences")
    w9.require(
        type(sequences) is list and len(sequences) == contract.batch,
        "control: treatment sequence count",
    )
    histories = []
    trace = []
    for index, sequence in enumerate(sequences):
        w9.require(
            type(sequence) is dict
            and sequence.get("sequence_index") == index,
            f"control: treatment sequence {index}",
        )
        history = (
            sequence.get("prompt_tokens", [])
            + sequence.get("preexisting_tokens", [])
        )
        tokens = (
            sequence.get("phone_service", [])
            + sequence.get("inflight_phone_tokens", [])
            + sequence.get("cuda_continuation", [])
        )
        w9.require(
            len(tokens) == contract.output_tokens
            and all(w9.is_int(token) and token >= 0 for token in history + tokens),
            f"control: treatment tokens {index}",
        )
        histories.append(list(history))
        trace.append(list(tokens))
    w9.require(
        len({len(row) for row in histories}) == 1
        and len({len(row) for row in trace}) == 1,
        "control: nonrectangular trace",
    )
    return histories, trace, value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--w8-contract", type=Path, required=True)
    parser.add_argument("--base-contract", type=Path, required=True)
    parser.add_argument("--delta-contract", type=Path, required=True)
    parser.add_argument("--physical-gate", type=Path, required=True)
    parser.add_argument("--treatment-report", type=Path, required=True)
    parser.add_argument("--launch-record", type=Path, required=True)
    parser.add_argument("--cuda-route", type=parse_endpoint, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pair-ordinal", required=True)
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    if (
        args.timeout <= 0
        or w6.HEX64.fullmatch(args.run_id) is None
        or args.output.exists()
        or args.ready_marker.exists()
    ):
        parser.error("invalid arguments or existing output")

    client: StageV3Client | None = None
    active = False
    stopped = False
    connector: cold.ReadyConnector | None = None
    try:
        base_contract = w5.load_contract(args.base_contract)
        delta_contract = w6.load_contract(args.delta_contract, base_contract)
        w8_contract = live.load_contract(
            args.w8_contract,
            base_contract,
            delta_contract,
            w6.sha256(args.physical_gate.read_bytes()),
        )
        contract = w9.load_contract(args.contract)
        w9.require(
            contract.w8_contract_sha256 == w8_contract.raw_sha256
            and args.pair_ordinal in contract.pair_ordinals,
            "control: contract dependency",
        )
        launch = load_launch(args.launch_record, args.run_id, args.pair_ordinal)
        w9.require(
            launch["cuda_launch_ns"] - launch["request_start_ns"]
            <= contract.max_launch_delay_us * 1000,
            "control: launch delay",
        )
        histories, trace, treatment = load_trace(
            args.treatment_report,
            contract,
            args.run_id,
            args.pair_ordinal,
        )
        connector = cold.ReadyConnector(
            args.cuda_route,
            contract.max_cuda_ready_us / 1_000_000,
            w8_contract.connector_poll_us,
        )
        connector.start()
        connection = connector.take()
        client = connection.client
        hello = connection.hello
        w5.validate_hello("cuda", hello, base_contract)
        w9.write_atomic(args.ready_marker, {
            "cuda_ready_ns": connection.ready_ns,
            "pair_ordinal": args.pair_ordinal,
            "process_pid": __import__("os").getpid(),
            "run_id": args.run_id,
            "schema": READY_SCHEMA,
        })
        active = True
        predictions, metrics = trace_control.run_teacher_forced(
            client,
            histories,
            trace,
            50000,
            w8_contract.cuda_control_chunk,
        )
        w5.remove_group(client, contract.batch, 50000)
        active = False
        status = client.status()
        w9.require(status.active_sequences == 0, "control: state leak")
        w5.finish(client, "stop")
        stopped = True
        matches = sum(
            prediction == expected
            for predicted_row, expected_row in zip(predictions, trace)
            for prediction, expected in zip(predicted_row, expected_row)
        )
        total = contract.batch * contract.output_tokens
        report: dict[str, object] = {
            "base_contract_sha256": base_contract.raw_sha256,
            "batch": contract.batch,
            "connector_attempts": connection.attempts,
            "contract_sha256": contract.raw_sha256,
            "control_mode": "TEACHER_FORCED_TRACE_REPLAY",
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
            "pair_ordinal": args.pair_ordinal,
            "post_start_tokens": contract.output_tokens,
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
            "status": "PROFILED_ZERO_EXTRA_TRACE_CONTROL_PASS",
            "timing": {
                "completion_us": (
                    metrics.token_ready_ns[-1] - launch["request_start_ns"]
                )
                // 1000,
                "cuda_ready_us": (
                    connection.ready_ns - launch["request_start_ns"]
                )
                // 1000,
                "new_request_ttft_us": None,
                "promotion_next_token_us": (
                    metrics.token_ready_ns[0] - launch["request_start_ns"]
                )
                // 1000,
            },
            "trace_treatment_sha256": w6.sha256(
                args.treatment_report.read_bytes()
            ),
            "w8_contract_sha256": w8_contract.raw_sha256,
        }
        w9.require(
            metrics.history_batches
            == (
                base_contract.prompt_tokens
                + contract.preexisting_committed_tokens
                + w8_contract.cuda_control_chunk
                - 1
            )
            // w8_contract.cuda_control_chunk
            and metrics.continuation_batches == contract.output_tokens - 1
            and metrics.rows
            == contract.batch
            * (
                base_contract.prompt_tokens
                + contract.preexisting_committed_tokens
                + contract.output_tokens
                - 1
            )
            and metrics.batch_timeline[0].started_ns >= connection.ready_ns
            and treatment["timing"]["request_start_ns"]
            == treatment["cuda_ready"]["request_start_ns"],
            "control: accounting or causal timing",
        )
        w9.write_atomic(args.output, report)
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
        w9.W9Error,
        w5.HandoffError,
        w6.DeltaError,
        live.LivePromotionError,
    ) as exc:
        print(w6.canonical({
            "error": str(exc),
            "status": "PROFILED_ZERO_EXTRA_CONTROL_ERROR",
        }).decode("ascii"), end="")
        raise SystemExit(2)
