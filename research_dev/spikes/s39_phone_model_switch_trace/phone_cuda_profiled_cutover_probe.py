#!/usr/bin/env python3
"""Run one W9 phone-to-CUDA profiled cutover treatment."""

from __future__ import annotations

import argparse
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import phone_cuda_cold_promotion_probe as cold
import phone_cuda_delta_probe as w6
import phone_cuda_handoff_probe as w5
import phone_cuda_live_promotion_probe as live
import w9_profiled_cutover as w9
from async_pipeline import parse_endpoint
from qwen25_quality_probe import load_corpus
from stage_v3_client import BatchRow, ProtocolError, StageV3Client


SCHEMA = "s39-profiled-cutover-treatment-v1"
START_SCHEMA = "s39-profiled-cutover-start-v1"
READY_SCHEMA = "s39-profiled-cutover-cuda-ready-v1"


def publish_round(
    ledger: w9.PublicationLedger,
    *,
    owner: str,
    owner_epoch: int,
    request_ids: Sequence[int],
    tokens: Sequence[int],
    positions: Sequence[int],
    classification: str,
) -> list[int]:
    w9.require(
        owner in {"PHONE", "CUDA"}
        and owner_epoch in {1, 2}
        and len(request_ids) == len(tokens) == len(positions)
        and all(w9.is_int(token) and token >= 0 for token in tokens)
        and all(w9.is_int(position) and position >= 0 for position in positions),
        "publication: invalid token round",
    )
    published = []
    for request_id, token, position in zip(request_ids, tokens, positions):
        record, _ = ledger.append(
            "TOKEN_PUBLISHED",
            {
                "classification": classification,
                "owner": owner,
                "owner_epoch": owner_epoch,
                "position": position,
                "request_id": request_id,
                "token": token,
            },
        )
        published.append(record.value["event_ns"])
    return published


def run_phone_batch(
    client: StageV3Client,
    prediction: Sequence[int],
    identity_base: int,
    position: int,
    start_state: dict[str, object],
) -> dict[str, object]:
    started_ns = time.monotonic_ns()
    start_state["started_ns"] = started_ns
    event = start_state["event"]
    w9.require(type(event) is threading.Event, "phone batch: start event")
    event.set()
    outputs, current, metrics = w6.advance_active(
        client,
        prediction,
        identity_base,
        position,
        1,
    )
    ended_ns = time.monotonic_ns()
    return {
        "current": current,
        "ended_ns": ended_ns,
        "metrics": asdict(metrics),
        "outputs": outputs,
        "position": position,
        "started_ns": started_ns,
    }


def run_cuda_replay(
    client: StageV3Client,
    histories: Sequence[Sequence[int]],
    identity_base: int,
    chunk: int,
    start_state: dict[str, object],
) -> dict[str, object]:
    started_ns = time.monotonic_ns()
    start_state["started_ns"] = started_ns
    event = start_state["event"]
    w9.require(type(event) is threading.Event, "CUDA replay: start event")
    event.set()
    prediction, metrics = w6.replay_history_only(
        client,
        histories,
        identity_base,
        chunk,
    )
    ended_ns = time.monotonic_ns()
    return {
        "ended_ns": ended_ns,
        "metrics": asdict(metrics),
        "prediction": prediction,
        "started_ns": started_ns,
    }


def continue_cuda(
    client: StageV3Client,
    prediction: Sequence[int],
    *,
    identity_base: int,
    history_width: int,
    token_count: int,
    ledger: w9.PublicationLedger,
    request_ids: Sequence[int],
    ownership_commit_ns: int,
) -> tuple[list[list[int]], dict[str, object], list[list[int]]]:
    w9.require(token_count > 0, "CUDA continuation must be positive")
    current = list(prediction)
    outputs = [[] for _ in current]
    publication_ns = [[] for _ in current]
    intervals = []
    for offset in range(token_count):
        if offset:
            position = history_width + offset - 1
            rows = [
                BatchRow(
                    identity_base + sequence,
                    identity_base + sequence,
                    sequence,
                    position,
                    token,
                )
                for sequence, token in enumerate(current)
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
                len(current),
                1,
                identity_base,
                position,
            )
        positions = [history_width + offset] * len(current)
        times = publish_round(
            ledger,
            owner="CUDA",
            owner_epoch=2,
            request_ids=request_ids,
            tokens=current,
            positions=positions,
            classification="CUDA_CONTINUATION",
        )
        w9.require(
            min(times) > ownership_commit_ns,
            "CUDA publication precedes durable ownership commit",
        )
        for sequence, token in enumerate(current):
            outputs[sequence].append(token)
            publication_ns[sequence].append(times[sequence])
    elapsed_us = sum(
        (interval.ended_ns - interval.started_ns) // 1000
        for interval in intervals
    )
    return outputs, {
        "batches": len(intervals),
        "elapsed_us": elapsed_us,
        "rows": sum(interval.rows for interval in intervals),
        "timeline": [asdict(interval) for interval in intervals],
    }, publication_ns


def flatten(histories: Sequence[Sequence[int]]) -> list[list[int]]:
    return [list(row) for row in histories]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--w8-contract", type=Path, required=True)
    parser.add_argument("--base-contract", type=Path, required=True)
    parser.add_argument("--delta-contract", type=Path, required=True)
    parser.add_argument("--physical-gate", type=Path, required=True)
    parser.add_argument("--phone-route", type=parse_endpoint, required=True)
    parser.add_argument("--cuda-route", type=parse_endpoint, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pair-ordinal", required=True)
    parser.add_argument("--prepaid-ready-marker", type=Path, required=True)
    parser.add_argument("--paid-permit", type=Path, required=True)
    parser.add_argument("--start-marker", type=Path, required=True)
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    if (
        args.timeout <= 0
        or w6.HEX64.fullmatch(args.run_id) is None
        or any(
            path.exists()
            for path in (
                args.start_marker,
                args.ready_marker,
                args.ledger_dir,
                args.output,
                args.prepaid_ready_marker,
                args.paid_permit,
            )
        )
    ):
        parser.error("invalid arguments or existing output")

    clients: dict[str, StageV3Client] = {}
    active: dict[str, tuple[int, int] | None] = {
        "cuda": None,
        "phone": None,
    }
    stopped: set[str] = set()
    connector: cold.ReadyConnector | None = None
    pool: ThreadPoolExecutor | None = None
    committed = False
    contract: w9.CutoverContract | None = None
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
            args.pair_ordinal in contract.pair_ordinals
            and contract.w8_contract_sha256 == w8_contract.raw_sha256
            and contract.base_contract_sha256 == base_contract.raw_sha256
            and contract.delta_contract_sha256 == delta_contract.raw_sha256
            and contract.physical_gate_sha256
            == w6.sha256(args.physical_gate.read_bytes())
            and contract.batch == base_contract.batch
            and contract.preexisting_committed_tokens
            == w8_contract.preexisting_committed_tokens,
            "treatment: contract dependency",
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
        tx_id = w9.transaction_id(contract, args.run_id, base_contract.prompt_ids)
        ledger = w9.PublicationLedger(
            args.ledger_dir,
            transaction_id=tx_id,
            run_id=args.run_id,
            request_ids=base_contract.prompt_ids,
        )

        clients["phone"] = StageV3Client.connect(*args.phone_route, args.timeout)
        phone_hello = clients["phone"].hello()
        w5.validate_hello("phone", phone_hello, base_contract)
        phone_base = 10000
        cuda_base = 30000
        control_base = 40000
        active["phone"] = (contract.batch, phone_base)
        preexisting_started_ns = time.monotonic_ns()
        preexisting_tokens, preexisting_metrics = cold.run_fixed_generation(
            clients["phone"],
            prompts,
            phone_base,
            w8_contract.phone_prefill_chunk,
            contract.preexisting_committed_tokens,
        )
        preexisting_ended_ns = time.monotonic_ns()
        phone_before = clients["phone"].status()
        w9.require(
            phone_before.active_sequences == contract.batch,
            "treatment: preexisting phone state",
        )
        w9.write_atomic(args.prepaid_ready_marker, {
            "pair_ordinal": args.pair_ordinal,
            "phone_state_count": phone_before.active_sequences,
            "preexisting_ended_ns": preexisting_ended_ns,
            "process_pid": os.getpid(),
            "run_id": args.run_id,
            "schema": "s39-profiled-cutover-prepaid-ready-v1",
        })
        permit_deadline = time.monotonic() + args.timeout
        while not args.paid_permit.is_file():
            w9.require(
                time.monotonic() < permit_deadline,
                "treatment: paid permit timed out",
            )
            time.sleep(0.01)
        permit, _ = w6.read_canonical(args.paid_permit, "paid_permit")
        w9.require(
            permit
            == {
                "pair_ordinal": args.pair_ordinal,
                "run_id": args.run_id,
                "schema": "s39-profiled-cutover-paid-permit-v1",
            },
            "treatment: invalid paid permit",
        )

        request_start_ns = time.monotonic_ns()
        launch_deadline_ns = request_start_ns + 100_000_000
        w9.write_atomic(args.start_marker, {
            "launch_deadline_ns": launch_deadline_ns,
            "pair_ordinal": args.pair_ordinal,
            "phone_state_count": phone_before.active_sequences,
            "preexisting_ended_ns": preexisting_ended_ns,
            "process_pid": os.getpid(),
            "request_start_ns": request_start_ns,
            "run_id": args.run_id,
            "schema": START_SCHEMA,
        })
        ledger.append(
            "PAID_START",
            {
                "owner": "PHONE",
                "owner_epoch": 1,
                "pair_ordinal": args.pair_ordinal,
            },
            event_ns=request_start_ns,
        )
        connector = cold.ReadyConnector(
            args.cuda_route,
            contract.max_cuda_ready_us / 1_000_000,
            w8_contract.connector_poll_us,
        )
        connector.start()
        pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="w9-cutover")

        current = [tokens[-1] for tokens in preexisting_tokens]
        service_tokens = [[] for _ in range(contract.batch)]
        service_publication_ns = [[] for _ in range(contract.batch)]
        phone_batches = []
        next_position = (
            base_contract.prompt_tokens
            + contract.preexisting_committed_tokens
            - 1
        )
        phone_start_state: dict[str, object] = {
            "event": threading.Event(),
        }
        phone_future: Future[dict[str, object]] | None = pool.submit(
            run_phone_batch,
            clients["phone"],
            current,
            phone_base,
            next_position,
            phone_start_state,
        )
        w9.require(
            phone_start_state["event"].wait(timeout=1.0),
            "treatment: phone batch did not start",
        )
        coordinator_lock = threading.Lock()
        f0_snapshot_ns = 0
        while True:
            connector.raise_if_failed()
            if connector.is_ready():
                with coordinator_lock:
                    f0_snapshot_ns = time.monotonic_ns()
                break
            if phone_future.done():
                batch_result = phone_future.result()
                with coordinator_lock:
                    if connector.is_ready():
                        f0_snapshot_ns = time.monotonic_ns()
                        break
                    outputs = batch_result["outputs"]
                    current = list(batch_result["current"])
                    round_tokens = [row[0] for row in outputs]
                    positions = [
                        base_contract.prompt_tokens
                        + contract.preexisting_committed_tokens
                        + len(service_tokens[index])
                        for index in range(contract.batch)
                    ]
                    times = publish_round(
                        ledger,
                        owner="PHONE",
                        owner_epoch=1,
                        request_ids=base_contract.prompt_ids,
                        tokens=round_tokens,
                        positions=positions,
                        classification="PHONE_F0",
                    )
                    for index, token in enumerate(round_tokens):
                        service_tokens[index].append(token)
                        service_publication_ns[index].append(times[index])
                    phone_batches.append({
                        **batch_result,
                        "classification": "F0",
                        "publication_ns": times,
                    })
                    w9.require(
                        len(service_tokens[0])
                        < w8_contract.max_phone_service_tokens,
                        "treatment: CUDA missed bounded service window",
                    )
                    next_position += 1
                    if connector.is_ready():
                        phone_future = None
                        f0_snapshot_ns = time.monotonic_ns()
                        break
                    phone_start_state = {"event": threading.Event()}
                    phone_future = pool.submit(
                        run_phone_batch,
                        clients["phone"],
                        current,
                        phone_base,
                        next_position,
                        phone_start_state,
                    )
                    w9.require(
                        phone_start_state["event"].wait(timeout=1.0),
                        "treatment: phone batch did not start",
                    )
            else:
                time.sleep(0.001)

        connection = connector.take()
        clients["cuda"] = connection.client
        cuda_hello = connection.hello
        w5.validate_hello("cuda", cuda_hello, base_contract)
        cold.require_same_model(
            {"cuda": cuda_hello, "phone": phone_hello},
            expected_model_sha256=base_contract.model_sha256,
            expected_file_type=base_contract.file_type,
        )
        w9.require(
            len(service_tokens[0]) >= 1
            and all(len(row) == len(service_tokens[0]) for row in service_tokens),
            "treatment: no rectangular pre-ready phone service",
        )
        w9.write_atomic(args.ready_marker, {
            "cuda_ready_ns": connection.ready_ns,
            "f0_snapshot_ns": f0_snapshot_ns,
            "pair_ordinal": args.pair_ordinal,
            "process_pid": os.getpid(),
            "run_id": args.run_id,
            "schema": READY_SCHEMA,
        })
        ledger.append(
            "CUDA_READY",
            {"cuda_ready_ns": connection.ready_ns},
            event_ns=f0_snapshot_ns,
        )
        f0_histories = [
            list(prompt) + list(preexisting) + list(service)
            for prompt, preexisting, service in zip(
                prompts,
                preexisting_tokens,
                service_tokens,
            )
        ]
        f0_positions = [len(history) - 1 for history in f0_histories]
        f0_sha = w5.histories_digest(f0_histories)
        ledger.append(
            "F0_SNAPSHOT",
            {
                "history_sha256": f0_sha,
                "positions": f0_positions,
                "snapshot_ns": f0_snapshot_ns,
            },
            event_ns=f0_snapshot_ns,
        )

        inflight_present = phone_future is not None
        inflight_started_ns = (
            int(phone_start_state["started_ns"]) if inflight_present else 0
        )
        inflight_elapsed_us = (
            max(0, (f0_snapshot_ns - inflight_started_ns) // 1000)
            if inflight_present
            else 0
        )
        decision = w9.select_cutover(
            contract,
            phone_tokens_at_f0=len(service_tokens[0]),
            inflight_present=inflight_present,
            inflight_elapsed_us=inflight_elapsed_us,
        )
        w9.require(
            decision.k_extra == 0,
            "treatment: real profile selected unexpected extra batch",
        )

        active["cuda"] = (contract.batch, cuda_base)
        replay_start_state: dict[str, object] = {
            "event": threading.Event(),
        }
        replay_future: Future[dict[str, object]] = pool.submit(
            run_cuda_replay,
            clients["cuda"],
            f0_histories,
            cuda_base,
            w8_contract.cuda_replay_chunk,
            replay_start_state,
        )
        w9.require(
            replay_start_state["event"].wait(timeout=1.0),
            "treatment: CUDA replay did not start",
        )
        replay_started_ns = int(replay_start_state["started_ns"])
        ledger.append(
            "CUDA_REPLAY_STARTED",
            {
                "f0_history_sha256": f0_sha,
                "replay_started_ns": replay_started_ns,
            },
            event_ns=replay_started_ns,
        )

        inflight_tokens = [[] for _ in range(contract.batch)]
        inflight_publication_ns = [[] for _ in range(contract.batch)]
        inflight_result = None
        realized_inflight_remaining_us = 0
        if phone_future is not None:
            inflight_result = phone_future.result(timeout=args.timeout)
            realized_inflight_remaining_us = max(
                0,
                (int(inflight_result["ended_ns"]) - f0_snapshot_ns) // 1000,
            )
            outputs = inflight_result["outputs"]
            current = list(inflight_result["current"])
            round_tokens = [row[0] for row in outputs]
            positions = [position + 1 for position in f0_positions]
            times = publish_round(
                ledger,
                owner="PHONE",
                owner_epoch=1,
                request_ids=base_contract.prompt_ids,
                tokens=round_tokens,
                positions=positions,
                classification="PHONE_INFLIGHT",
            )
            for index, token in enumerate(round_tokens):
                inflight_tokens[index].append(token)
                inflight_publication_ns[index].append(times[index])
            phone_batches.append({
                **inflight_result,
                "classification": "INFLIGHT",
                "publication_ns": times,
            })
        d_inflight = 1 if inflight_present else 0
        d_actual = d_inflight
        w9.require(
            d_actual in {0, 1}
            and all(len(row) == d_actual for row in inflight_tokens),
            "treatment: invalid realized delta",
        )
        f1_histories = [
            history + delta
            for history, delta in zip(f0_histories, inflight_tokens)
        ]
        f1_positions = [len(history) - 1 for history in f1_histories]
        f1_sha = w5.histories_digest(f1_histories)
        f1_ack_ns = time.monotonic_ns()
        ledger.append(
            "F1_ACK",
            {
                "d_actual": d_actual,
                "disposition": "PUBLISHED" if d_inflight else "ABSENT",
                "history_sha256": f1_sha,
                "owner_epoch": 1,
                "positions": f1_positions,
            },
            event_ns=f1_ack_ns,
        )

        replay_result = replay_future.result(timeout=args.timeout)
        replay_prediction = list(replay_result["prediction"])
        if d_actual:
            cuda_prediction, delta_metrics = w6.feed_known_tokens(
                clients["cuda"],
                inflight_tokens,
                cuda_base,
                len(f0_histories[0]),
                w8_contract.cuda_delta_chunk,
            )
            delta_mode = "POSITIVE"
        else:
            cuda_prediction = replay_prediction
            delta_metrics = w6.BatchMetrics(0, 0, 0)
            delta_mode = "NOOP"
        catchup_ns = time.monotonic_ns()
        ledger.append(
            "CUDA_CAUGHT_UP",
            {
                "delta_mode": delta_mode,
                "f1_history_sha256": f1_sha,
                "positions": f1_positions,
            },
            event_ns=catchup_ns,
        )
        commit_record, ownership_commit_ns = ledger.append(
            "CUDA_COMMITTED",
            {
                "history_sha256": f1_sha,
                "new_owner": "CUDA",
                "new_owner_epoch": 2,
                "old_owner_epoch": 1,
                "positions": f1_positions,
            },
        )
        ledger.append(
            "CUDA_COMMIT_DURABLE",
            {
                "committed_record_sha256": commit_record.sha256,
                "durable_ns": ownership_commit_ns,
            },
            event_ns=ownership_commit_ns,
        )
        committed = True
        w5.remove_group(clients["phone"], contract.batch, phone_base)
        active["phone"] = None
        phone_after = clients["phone"].status()
        w9.require(
            phone_after.active_sequences == 0,
            "treatment: phone state leak",
        )
        ledger.append(
            "PHONE_RELEASED",
            {"old_owner_epoch": 1, "positions": f1_positions},
        )

        cuda_continuation_tokens = (
            contract.output_tokens - len(service_tokens[0]) - d_actual
        )
        w9.require(
            cuda_continuation_tokens >= contract.min_cuda_continuation_tokens,
            "treatment: exhausted CUDA continuation",
        )
        cuda_tokens, cuda_metrics, cuda_publication_ns = continue_cuda(
            clients["cuda"],
            cuda_prediction,
            identity_base=cuda_base,
            history_width=len(f1_histories[0]),
            token_count=cuda_continuation_tokens,
            ledger=ledger,
            request_ids=base_contract.prompt_ids,
            ownership_commit_ns=ownership_commit_ns,
        )
        request_complete_ns = max(row[-1] for row in cuda_publication_ns)
        final_histories = [
            history + continuation
            for history, continuation in zip(f1_histories, cuda_tokens)
        ]
        final_sha = w5.histories_digest(final_histories)
        w5.remove_group(clients["cuda"], contract.batch, cuda_base)
        active["cuda"] = None
        cuda_after = clients["cuda"].status()
        w9.require(
            cuda_after.active_sequences == 0,
            "treatment: CUDA state leak",
        )
        ledger.append(
            "COMPLETE",
            {
                "final_history_sha256": final_sha,
                "owner": "NONE",
                "post_start_tokens": contract.output_tokens,
            },
        )

        active["cuda"] = (contract.batch, control_base)
        control_tokens, control_metrics = w5.run_replay(
            clients["cuda"],
            f1_histories,
            control_base,
            w8_contract.cuda_control_chunk,
            cuda_continuation_tokens,
        )
        w5.remove_group(clients["cuda"], contract.batch, control_base)
        active["cuda"] = None
        control_state = clients["cuda"].status()
        w9.require(
            control_state.active_sequences == 0,
            "treatment: CUDA oracle state leak",
        )
        w9.require(
            cuda_tokens == control_tokens,
            "treatment: CUDA continuation mismatch",
        )
        w5.finish(clients["phone"], "stop")
        stopped.add("phone")
        w5.finish(clients["cuda"], "stop")
        stopped.add("cuda")

        records = w9.load_ledger(
            args.ledger_dir,
            transaction_id=tx_id,
            run_id=args.run_id,
            request_ids=base_contract.prompt_ids,
        )
        all_publication_ns = [
            service_publication_ns[index]
            + inflight_publication_ns[index]
            + cuda_publication_ns[index]
            for index in range(contract.batch)
        ]
        maximum_gaps = []
        for times in all_publication_ns:
            w9.require(
                len(times) == contract.output_tokens,
                "treatment: publication count",
            )
            boundaries = [request_start_ns] + times
            maximum_gaps.append(
                max(
                    (right - left) // 1000
                    for left, right in zip(boundaries, boundaries[1:])
                )
            )
        phone_first_ns = min(row[0] for row in service_publication_ns)
        phone_last_ns = max(
            row[-1]
            for row in (
                inflight_publication_ns
                if d_actual
                else service_publication_ns
            )
        )
        cuda_first_ns = min(row[0] for row in cuda_publication_ns)
        overlap_ns: int | None = None
        if inflight_result is not None:
            overlap_ns = max(
                0,
                min(
                    inflight_result["ended_ns"],
                    replay_result["ended_ns"],
                )
                - max(
                    inflight_result["started_ns"],
                    replay_result["started_ns"],
                ),
            )
            w9.require(overlap_ns > 0, "treatment: no in-flight replay overlap")
        report: dict[str, object] = {
            "base_contract_sha256": base_contract.raw_sha256,
            "batch": contract.batch,
            "contract_sha256": contract.raw_sha256,
            "cuda_ready": {
                "attempts": connection.attempts,
                "cuda_ready_ns": connection.ready_ns,
                "f0_snapshot_ns": f0_snapshot_ns,
                "request_start_ns": request_start_ns,
            },
            "decision": decision.as_dict(),
            "delta_contract_sha256": delta_contract.raw_sha256,
            "frontiers": {
                "d_actual": d_actual,
                "d_inflight": d_inflight,
                "f0_history_sha256": f0_sha,
                "f0_positions": f0_positions,
                "f1_ack_ns": f1_ack_ns,
                "f1_history_sha256": f1_sha,
                "f1_positions": f1_positions,
                "inflight_disposition": (
                    "PUBLISHED" if inflight_present else "ABSENT"
                ),
                "k_extra": decision.k_extra,
                "phone_tokens_at_f0": len(service_tokens[0]),
                "realized_inflight_remaining_us": (
                    realized_inflight_remaining_us
                ),
            },
            "hellos": {
                "cuda": asdict(cuda_hello),
                "phone": asdict(phone_hello),
            },
            "ledger": w9.ledger_summary(records),
            "metrics": {
                "cuda_continuation": cuda_metrics,
                "cuda_delta": asdict(delta_metrics),
                "cuda_oracle": asdict(control_metrics),
                "cuda_replay": replay_result,
                "phone_batches": phone_batches,
                "preexisting": cold.service_metrics_value(preexisting_metrics),
            },
            "model_sha256": base_contract.model_sha256,
            "ownership": {
                "cuda_catchup_ns": catchup_ns,
                "cuda_first_publication_ns": cuda_first_ns,
                "cuda_replay_start_ns": replay_result["started_ns"],
                "handoff_gap_us": (cuda_first_ns - phone_last_ns) // 1000,
                "inflight_overlap_ns": overlap_ns,
                "ownership_commit_ns": ownership_commit_ns,
                "phone_last_publication_ns": phone_last_ns,
            },
            "pair_ordinal": args.pair_ordinal,
            "physical_gate_sha256": contract.physical_gate_sha256,
            "preexisting": {
                "ended_ns": preexisting_ended_ns,
                "preexisting_committed_tokens": (
                    contract.preexisting_committed_tokens
                ),
                "started_ns": preexisting_started_ns,
            },
            "prompts": list(base_contract.prompt_ids),
            "run_id": args.run_id,
            "scheduler_eligible": False,
            "schema": SCHEMA,
            "scope": "MECHANICS_ONLY",
            "sequences": [
                {
                    "cuda_continuation": list(cuda_tokens[index]),
                    "cuda_continuation_control": list(control_tokens[index]),
                    "cuda_publication_ns": cuda_publication_ns[index],
                    "extra_phone_tokens": [],
                    "final_published_tokens": (
                        list(preexisting_tokens[index])
                        + list(service_tokens[index])
                        + list(inflight_tokens[index])
                        + list(cuda_tokens[index])
                    ),
                    "inflight_phone_tokens": list(inflight_tokens[index]),
                    "inflight_publication_ns": inflight_publication_ns[index],
                    "max_inter_token_gap_us": maximum_gaps[index],
                    "phone_service": list(service_tokens[index]),
                    "phone_service_publication_ns": (
                        service_publication_ns[index]
                    ),
                    "preexisting_tokens": list(preexisting_tokens[index]),
                    "prompt_id": base_contract.prompt_ids[index],
                    "prompt_tokens": list(prompts[index]),
                    "sequence_index": index,
                }
                for index in range(contract.batch)
            ],
            "state_counts": {
                "cuda_after_completion": cuda_after.active_sequences,
                "cuda_oracle_released": control_state.active_sequences,
                "phone_after_commit": phone_after.active_sequences,
                "phone_before_promotion": phone_before.active_sequences,
            },
            "status": "PROFILED_ZERO_EXTRA_TREATMENT_PASS",
            "timing": {
                "completion_us": (
                    request_complete_ns - request_start_ns
                )
                // 1000,
                "cuda_ready_us": (
                    connection.ready_ns - request_start_ns
                )
                // 1000,
                "maximum_inter_token_gap_us": max(maximum_gaps),
                "new_request_ttft_us": None,
                "promotion_next_token_us": (
                    phone_first_ns - request_start_ns
                )
                // 1000,
                "request_complete_ns": request_complete_ns,
                "request_start_ns": request_start_ns,
                "useful_pre_ready_phone_tokens": sum(
                    timestamp < connection.ready_ns
                    for timestamp in service_publication_ns[0]
                ),
            },
            "transaction_id": tx_id,
            "w8_contract_sha256": w8_contract.raw_sha256,
        }
        w9.require(
            all(
                len(sequence["phone_service"])
                + len(sequence["inflight_phone_tokens"])
                + len(sequence["cuda_continuation"])
                == contract.output_tokens
                for sequence in report["sequences"]
            ),
            "treatment: output budget",
        )
        w9.write_atomic(args.output, report)
        print(w6.canonical(report).decode("ascii"), end="")
        return 0
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        if connector is not None:
            connector.close()
        for name, client in clients.items():
            state = active.get(name)
            if state is not None and (committed or name == "cuda"):
                try:
                    w5.remove_group(client, state[0], state[1])
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
        w9.W9Error,
        w5.HandoffError,
        w6.DeltaError,
        live.LivePromotionError,
    ) as exc:
        print(w6.canonical({
            "error": str(exc),
            "status": "PROFILED_ZERO_EXTRA_TREATMENT_ERROR",
        }).decode("ascii"), end="")
        raise SystemExit(2)
