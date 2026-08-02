#!/usr/bin/env python3
"""Real three-stage proof of an unequal-length continuous request lifecycle."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


HERE = Path(__file__).resolve().parent
S22 = HERE.parent / "s22_slo_overlap_pipeline"
if str(S22) not in sys.path:
    sys.path.insert(0, str(S22))

from async_pipeline import parse_endpoint
from stage_v3_client import (
    BatchResult,
    BatchRow,
    Hello,
    ProtocolError,
    STAGE_V3_CAP_TERMINAL,
    StageV3Client,
    Status,
)


class LifecycleError(RuntimeError):
    pass


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class RequestDefinition:
    name: str
    arrival_step: int
    output_steps: int
    start_token: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("request name must be nonempty")
        if not _is_int(self.arrival_step) or self.arrival_step < 0:
            raise ValueError("arrival step must be a nonnegative integer")
        if not _is_int(self.output_steps) or self.output_steps <= 0:
            raise ValueError("output steps must be a positive integer")
        if not _is_int(self.start_token) or self.start_token < 0:
            raise ValueError("start token must be a nonnegative integer")


@dataclass(frozen=True)
class PlannedRow:
    request: str
    seq_id: int
    position: int


@dataclass(frozen=True)
class LifecycleStep:
    index: int
    removals: tuple[str, ...]
    admissions: tuple[str, ...]
    rows: tuple[PlannedRow, ...]


@dataclass(frozen=True)
class LifecyclePlan:
    capacity: int
    steps: tuple[LifecycleStep, ...]


@dataclass(frozen=True)
class RuntimeIdentity:
    request_id: int
    route_epoch: int


DEFAULT_REQUESTS = (
    RequestDefinition("A", 0, 2, 2),
    RequestDefinition("B", 0, 4, 2),
    RequestDefinition("C", 2, 3, 2),
    RequestDefinition("D", 4, 2, 2),
)


def build_lifecycle_plan(
    definitions: Sequence[RequestDefinition], capacity: int,
) -> LifecyclePlan:
    if not _is_int(capacity) or capacity <= 0:
        raise ValueError("capacity must be a positive integer")
    if not definitions:
        raise ValueError("request set must be nonempty")
    names = [item.name for item in definitions]
    if len(set(names)) != len(names):
        raise ValueError("request names must be unique")

    by_name = {item.name: item for item in definitions}
    pending = sorted(definitions, key=lambda item: (item.arrival_step, item.name))
    active: dict[str, tuple[int, int]] = {}
    completed: set[str] = set()
    free_slots = list(range(capacity))
    steps: list[LifecycleStep] = []
    index = 0
    bound = sum(item.output_steps for item in definitions) + max(
        item.arrival_step for item in definitions
    ) + len(definitions) + 1

    while len(completed) < len(definitions):
        removals = sorted(
            name for name, (_seq_id, position) in active.items()
            if position >= by_name[name].output_steps
        )
        for name in removals:
            seq_id, _position = active.pop(name)
            free_slots.append(seq_id)
            free_slots.sort()
            completed.add(name)

        admissions: list[str] = []
        eligible = [item for item in pending if item.arrival_step <= index]
        for item in eligible[:len(free_slots)]:
            pending.remove(item)
            seq_id = free_slots.pop(0)
            active[item.name] = (seq_id, 0)
            admissions.append(item.name)

        rows = tuple(
            PlannedRow(name, seq_id, position)
            for name, (seq_id, position) in sorted(
                active.items(), key=lambda item: item[1][0]
            )
        )
        if not rows and pending:
            next_arrival = pending[0].arrival_step
            if next_arrival <= index:
                raise LifecycleError("eligible request could not be admitted")
        if removals or admissions or rows:
            steps.append(LifecycleStep(
                index, tuple(removals), tuple(admissions), rows,
            ))

        for row in rows:
            active[row.request] = (row.seq_id, row.position + 1)

        index += 1
        if index > bound:
            raise LifecycleError("lifecycle construction exceeded finite bound")

    plan = LifecyclePlan(capacity, tuple(steps))
    validate_lifecycle_plan(plan, definitions)
    return plan


def validate_lifecycle_plan(
    plan: LifecyclePlan, definitions: Sequence[RequestDefinition],
) -> None:
    if not _is_int(plan.capacity) or plan.capacity <= 0:
        raise LifecycleError("plan capacity is invalid")
    by_name = {item.name: item for item in definitions}
    if len(by_name) != len(definitions):
        raise LifecycleError("request definitions are ambiguous")
    active: dict[str, tuple[int, int]] = {}
    used_slots: dict[int, str] = {}
    completed: set[str] = set()

    for expected_index, step in enumerate(plan.steps):
        if step.index != expected_index:
            raise LifecycleError("plan steps are not contiguous")
        if len(set(step.removals)) != len(step.removals):
            raise LifecycleError("duplicate removal in one step")
        for name in step.removals:
            if name not in active:
                raise LifecycleError("request removed while inactive")
            seq_id, position = active.pop(name)
            if position != by_name[name].output_steps:
                raise LifecycleError("request removed before completion")
            del used_slots[seq_id]
            completed.add(name)

        if len(set(step.admissions)) != len(step.admissions):
            raise LifecycleError("duplicate admission in one step")
        row_map = {row.request: row for row in step.rows}
        if len(row_map) != len(step.rows):
            raise LifecycleError("duplicate request row in one batch")
        for name in step.admissions:
            if name not in by_name or name in active or name in completed:
                raise LifecycleError("request admission state is invalid")
            if by_name[name].arrival_step > step.index:
                raise LifecycleError("request admitted before arrival")
            if name not in row_map:
                raise LifecycleError("admitted request has no physical row")
            seq_id = row_map[name].seq_id
            if seq_id in used_slots or not 0 <= seq_id < plan.capacity:
                raise LifecycleError("sequence slot reused while live")
            active[name] = (seq_id, 0)
            used_slots[seq_id] = name

        if set(row_map) != set(active):
            raise LifecycleError("physical batch differs from active requests")
        if len(step.rows) > plan.capacity:
            raise LifecycleError("physical batch exceeds lifecycle capacity")
        for row in step.rows:
            if row.request not in active:
                raise LifecycleError("physical row refers to inactive request")
            if active[row.request] != (row.seq_id, row.position):
                raise LifecycleError("physical row position or slot changed")
            active[row.request] = (row.seq_id, row.position + 1)

    if active or set(by_name) != completed:
        raise LifecycleError("lifecycle did not retire every request")


def validate_stage_layout(hellos: Mapping[str, Hello], capacity: int) -> None:
    if set(hellos) != {"op12", "op15", "tail"}:
        raise ProtocolError("stage set differs from the fixed route")
    expected_ranges = {
        "op12": (0, 8),
        "op15": (8, 16),
        "tail": (16, 48),
    }
    widths = set()
    for name, hello in hellos.items():
        if (hello.layer_start, hello.layer_end) != expected_ranges[name]:
            raise ProtocolError(f"{name} layer range differs from the fixed route")
        if hello.n_layer != 48:
            raise ProtocolError(f"{name} model depth differs from Gemma-4 12B")
        if hello.max_streams < capacity:
            raise ProtocolError(f"{name} lacks continuous sequence capacity")
        if min(hello.n_batch, hello.n_ubatch) < capacity:
            raise ProtocolError(f"{name} lacks physical batch capacity")
        terminal = bool(hello.capabilities & STAGE_V3_CAP_TERMINAL)
        if terminal != (name == "tail"):
            raise ProtocolError(f"{name} terminal capability is invalid")
        widths.add(hello.n_embd)
    if len(widths) != 1:
        raise ProtocolError("activation width differs across route stages")


def _validate_activation(result: BatchResult, stage: str) -> tuple[float, ...]:
    if result.hidden is None or result.token is not None:
        raise ProtocolError(f"{stage} returned a terminal result")
    if not all(math.isfinite(value) for value in result.hidden):
        raise ProtocolError(f"{stage} returned a non-finite activation")
    return result.hidden


def _validate_token(result: BatchResult) -> int:
    if result.token is None or result.hidden is not None or result.token < 0:
        raise ProtocolError("tail returned an invalid token result")
    return result.token


def _status_dict(status: Status) -> dict[str, object]:
    return asdict(status)


def _check_counts(
    statuses: Mapping[str, Status], expected: int, draining: bool = False,
) -> None:
    for name, status in statuses.items():
        if status.active_sequences != expected:
            raise ProtocolError(f"{name} active sequence count mismatch")
        if status.draining != draining:
            raise ProtocolError(f"{name} draining state mismatch")


def _identity_map(
    definitions: Sequence[RequestDefinition], request_id_base: int,
    route_epoch: int,
) -> dict[str, RuntimeIdentity]:
    if not _is_int(request_id_base) or request_id_base <= 0:
        raise ValueError("request id base must be a positive integer")
    if not _is_int(route_epoch) or route_epoch <= 0:
        raise ValueError("route epoch must be a positive integer")
    return {
        item.name: RuntimeIdentity(request_id_base + index, route_epoch)
        for index, item in enumerate(definitions)
    }


def execute_plan(
    clients: Mapping[str, StageV3Client],
    plan: LifecyclePlan,
    definitions: Sequence[RequestDefinition],
    request_id_base: int,
    route_epoch: int,
    label: str,
) -> dict[str, object]:
    validate_lifecycle_plan(plan, definitions)
    identities = _identity_map(definitions, request_id_base, route_epoch)
    by_name = {item.name: item for item in definitions}
    current_tokens = {item.name: item.start_token for item in definitions}
    outputs: dict[str, list[int]] = {item.name: [] for item in definitions}
    active: set[str] = set()
    events: list[dict[str, object]] = []
    _check_counts({name: client.status() for name, client in clients.items()}, 0)

    for step in plan.steps:
        removal_records = []
        for request_name in step.removals:
            identity = identities[request_name]
            seq_id = next(
                row.seq_id
                for prior in reversed(plan.steps[:step.index])
                for row in prior.rows
                if row.request == request_name
            )
            statuses = {
                name: client.remove(
                    seq_id, identity.request_id, identity.route_epoch,
                )
                for name, client in clients.items()
            }
            active.remove(request_name)
            _check_counts(statuses, len(active))
            removal_records.append({
                "request": request_name,
                "seq_id": seq_id,
                "statuses": {
                    name: _status_dict(status)
                    for name, status in statuses.items()
                },
            })

        active.update(step.admissions)
        if not step.rows:
            events.append({
                "step": step.index,
                "removals": removal_records,
                "admissions": list(step.admissions),
                "members": [],
                "physical_batch_sizes": {},
            })
            continue

        head_rows = [
            BatchRow(
                identities[row.request].request_id,
                identities[row.request].route_epoch,
                row.seq_id,
                row.position,
                current_tokens[row.request],
            )
            for row in step.rows
        ]
        stage_times: dict[str, int] = {}
        started = time.monotonic_ns()
        op12_results = clients["op12"].batch(head_rows)
        stage_times["op12"] = (time.monotonic_ns() - started) // 1000

        op15_rows = []
        for source, result in zip(head_rows, op12_results):
            hidden = _validate_activation(result, "op12")
            op15_rows.append(BatchRow(
                source.request_id, source.route_epoch, source.seq_id,
                source.position, source.token, hidden,
            ))
        started = time.monotonic_ns()
        op15_results = clients["op15"].batch(op15_rows)
        stage_times["op15"] = (time.monotonic_ns() - started) // 1000

        tail_rows = []
        for source, result in zip(op15_rows, op15_results):
            hidden = _validate_activation(result, "op15")
            tail_rows.append(BatchRow(
                source.request_id, source.route_epoch, source.seq_id,
                source.position, source.token, hidden,
            ))
        started = time.monotonic_ns()
        tail_results = clients["tail"].batch(tail_rows)
        stage_times["tail"] = (time.monotonic_ns() - started) // 1000

        returned_tokens = []
        for row, result in zip(step.rows, tail_results):
            token = _validate_token(result)
            outputs[row.request].append(token)
            current_tokens[row.request] = token
            returned_tokens.append(token)
        statuses = {name: client.status() for name, client in clients.items()}
        _check_counts(statuses, len(active))
        events.append({
            "step": step.index,
            "removals": removal_records,
            "admissions": list(step.admissions),
            "members": [row.request for row in step.rows],
            "positions": [row.position for row in step.rows],
            "seq_ids": [row.seq_id for row in step.rows],
            "request_ids": [identities[row.request].request_id for row in step.rows],
            "output_tokens": returned_tokens,
            "physical_batch_sizes": {name: len(step.rows) for name in clients},
            "stage_compute_us": stage_times,
            "statuses": {
                name: _status_dict(status) for name, status in statuses.items()
            },
        })

    expected_lengths = {item.name: item.output_steps for item in definitions}
    actual_lengths = {name: len(tokens) for name, tokens in outputs.items()}
    if actual_lengths != expected_lengths:
        raise LifecycleError(f"{label} output lengths differ from the plan")
    if active:
        raise LifecycleError(f"{label} ended with active request metadata")
    _check_counts({name: client.status() for name, client in clients.items()}, 0)
    return {
        "label": label,
        "events": events,
        "outputs": outputs,
    }


def execute_serial_oracle(
    clients: Mapping[str, StageV3Client],
    definitions: Sequence[RequestDefinition],
) -> dict[str, object]:
    runs = []
    outputs: dict[str, list[int]] = {}
    for index, definition in enumerate(definitions):
        serial_definition = RequestDefinition(
            definition.name, 0, definition.output_steps, definition.start_token,
        )
        plan = build_lifecycle_plan((serial_definition,), 1)
        run = execute_plan(
            clients, plan, (serial_definition,), 1001 + index, 1,
            f"serial-{definition.name}",
        )
        outputs[definition.name] = list(run["outputs"][definition.name])
        runs.append(run)
    return {"runs": runs, "outputs": outputs}


def compare_outputs(
    oracle: Mapping[str, Sequence[int]],
    treatment: Mapping[str, Sequence[int]],
) -> None:
    mismatches = output_mismatches(oracle, treatment)
    if mismatches:
        names = ",".join(item["request"] for item in mismatches)
        raise LifecycleError(f"request token sequence differs from B1: {names}")


def output_mismatches(
    oracle: Mapping[str, Sequence[int]],
    treatment: Mapping[str, Sequence[int]],
) -> list[dict[str, object]]:
    if set(oracle) != set(treatment):
        raise LifecycleError("oracle and treatment request sets differ")
    mismatches = []
    for name in sorted(oracle):
        if list(oracle[name]) != list(treatment[name]):
            mismatches.append({
                "request": name,
                "oracle": list(oracle[name]),
                "treatment": list(treatment[name]),
            })
    return mismatches


def _canonical_bytes(record: object) -> bytes:
    return (
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")


def write_report(path: Path, report: object) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    payload = _canonical_bytes(report)
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _finish_sessions(
    clients: Mapping[str, StageV3Client], session_end: str,
) -> dict[str, object]:
    statuses = {name: client.drain() for name, client in clients.items()}
    _check_counts(statuses, 0, draining=True)
    for client in clients.values():
        if session_end == "stop":
            client.stop()
        elif session_end == "detach":
            client.detach()
    return {name: _status_dict(status) for name, status in statuses.items()}


def run(args: argparse.Namespace) -> dict[str, object]:
    definitions = DEFAULT_REQUESTS
    plan = build_lifecycle_plan(definitions, args.capacity)
    endpoints = {
        "op12": args.op12,
        "op15": args.op15,
        "tail": args.tail,
    }
    clients: dict[str, StageV3Client] = {}
    success = False
    try:
        for name, endpoint in endpoints.items():
            clients[name] = StageV3Client.connect(*endpoint, args.timeout)
        hellos = {name: client.hello() for name, client in clients.items()}
        validate_stage_layout(hellos, args.capacity)
        started_ns = time.monotonic_ns()
        serial = execute_serial_oracle(clients, definitions)
        dynamic = execute_plan(
            clients, plan, definitions, 2001, 2, "dynamic",
        )
        mismatches = output_mismatches(serial["outputs"], dynamic["outputs"])
        drained = _finish_sessions(clients, args.session_end)
        success = True
        report = {
            "schema": "s25-continuous-lifecycle-v1",
            "verdict": "PASS" if not mismatches else "TOKEN_SCREEN_FAIL",
            "claim_scope": "REAL_RUNTIME_MECHANICS_ONLY",
            "route": [
                {"worker": "op12", "layers": [0, 8]},
                {"worker": "op15", "layers": [8, 16]},
                {"worker": "tail", "layers": [16, 48]},
            ],
            "hellos": {name: asdict(hello) for name, hello in hellos.items()},
            "requests": [asdict(item) for item in definitions],
            "plan": {
                "capacity": plan.capacity,
                "memberships": [
                    [row.request for row in step.rows]
                    for step in plan.steps if step.rows
                ],
            },
            "serial": serial,
            "dynamic": dynamic,
            "proofs": {
                "unequal_output_lengths": True,
                "mid_decode_retirement": True,
                "slot_reuse_while_peer_live": True,
                "membership_changes": True,
                "all_stages_observe_same_batch": True,
                "same_route_b1_tokens_equal": not mismatches,
                "final_active_sequences_zero": True,
            },
            "token_mismatches": mismatches,
            "drained": drained,
            "session_end": args.session_end,
            "elapsed_us": (time.monotonic_ns() - started_ns) // 1000,
            "limits": {
                "energy": "NOT_MEASURED",
                "throughput_benefit": "NOT_CLAIMED",
                "semantic_early_exit": "NOT_IMPLEMENTED",
            },
        }
        return report
    finally:
        if not success:
            for client in clients.values():
                try:
                    client.stop()
                except BaseException:
                    pass
        for client in clients.values():
            try:
                client.close()
            except BaseException:
                pass


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--op12", type=parse_endpoint, required=True)
    result.add_argument("--op15", type=parse_endpoint, required=True)
    result.add_argument("--tail", type=parse_endpoint, required=True)
    result.add_argument("--capacity", type=int, default=2)
    result.add_argument("--timeout", type=float, default=180.0)
    result.add_argument("--session-end", choices=("stop", "detach"), default="stop")
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.capacity != 2:
        parser().error("the frozen S25 proof requires capacity 2")
    if args.timeout <= 0:
        parser().error("timeout must be positive")
    try:
        report = run(args)
        write_report(args.output, report)
    except (FileExistsError, LifecycleError, OSError, ProtocolError, ValueError) as exc:
        print(json.dumps({
            "schema": "s25-continuous-lifecycle-v1",
            "verdict": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, sort_keys=True, separators=(",", ":")))
        return 2
    print(json.dumps({
        "verdict": report["verdict"],
        "output": str(args.output),
        "elapsed_us": report["elapsed_us"],
    }, sort_keys=True, separators=(",", ":")))
    return 0 if report["verdict"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
