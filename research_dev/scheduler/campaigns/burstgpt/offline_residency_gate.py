#!/usr/bin/env python3
"""Physically prove offline residency and progressive session publication."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from statistics import median
import sys
import threading
import time
import traceback
from typing import Any, Callable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    AdaptiveDecodeConfig,
    Request,
)
from research_dev.scheduler.adapters import (  # noqa: E402
    CanonicalArrivalCoordinator,
    CanonicalOfflinePhoneResidencyPreloader,
    CanonicalRuntimeSubmission,
    LlamaCppCompletionPayload,
    PhysicalAdapterError,
)
from research_dev.scheduler.campaigns.burstgpt import runner  # noqa: E402
from research_dev.scheduler.adapters.llama_server import llama_server_runtime_timing  # noqa: E402


RESULT_SCHEMA = "s42-offline-phone-residency-gate-v1"
INTERRUPTION_METRIC_SCHEMA = "s42-retained-session-call-gap-v2"
REFERENCE_INTERVAL_COUNT = 30
DECISION_LOG_SCHEMA = "s42-offline-phone-residency-decision-log-v1"
REQUIRED_FRACTIONS = frozenset({0, 250_000, 500_000, 750_000, 1_000_000})
FIRST_TOKEN_TIMEOUT_S = 300


class OfflineResidencyGateError(RuntimeError):
    pass


class _SessionCallDeltaTimeout(OfflineResidencyGateError):
    def __init__(
        self,
        message: str,
        call_diagnostics: Mapping[str, object],
    ) -> None:
        super().__init__(message)
        self.call_diagnostics = dict(call_diagnostics)


class _SessionCallGapError(OfflineResidencyGateError):
    def __init__(self, measurement: Mapping[str, object]) -> None:
        super().__init__(str(measurement.get("reason", (
            "retained session call gap exceeds 2x median: "
            + str(measurement["session_id"])
        ))))
        self.call_gap_measurement = dict(measurement)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OfflineResidencyGateError(message)


def _runtime_timing_files(output: Path) -> dict[str, object]:
    return {
        path.name: {
            "stderr_sha256": "sha256:" + runner.digest(path),
            **llama_server_runtime_timing(path.read_text().splitlines()),
        }
        for path in sorted(output.glob("*.stderr"))
    }


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _now_us(epoch_ns: int) -> int:
    return max(0, (time.monotonic_ns() - epoch_ns) // 1000)


def _write_new(path: Path, value: object) -> None:
    require(not path.exists(), "gate artifact already exists: " + str(path))
    path.write_bytes(_canonical(value))


def _energy_json(value: object) -> dict[str, object] | None:
    return runner.measured_energy(value)


def _request_from_item(
    item: Mapping[str, Any],
    *,
    request_id: str | None = None,
    arrival_us: int | None = None,
    deadline_window_us: int = 3_600_000_000,
) -> Request:
    row = item["row"]
    arrival = int(row["arrival_us"] if arrival_us is None else arrival_us)
    return Request(
        request_id=(str(row["event_id"]) if request_id is None else request_id),
        workload_id="offline-gate:" + str(item["model_id"]),
        arrival_us=arrival,
        deadline_us=arrival + deadline_window_us,
        input_tokens=int(row["input_tokens"]),
        output_tokens=int(row["output_tokens"]),
        quality_requirement="semantic",
    )


def _forecast_requests(
    items: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[Request, ...]]:
    by_model: dict[str, list[Request]] = {}
    for item in items:
        by_model.setdefault(str(item["model_id"]), []).append(
            _request_from_item(item)
        )
    return {
        model_id: tuple(requests)
        for model_id, requests in sorted(by_model.items())
    }


@dataclass
class _RequestProbe:
    request: Request
    payload: LlamaCppCompletionPayload
    first_token: threading.Event
    first_token_ns: list[int]


def _request_probe(
    item: Mapping[str, Any],
    aliases: Mapping[str, str],
    streams: Path,
    *,
    request_id: str,
    arrival_us: int,
    seed: int,
) -> _RequestProbe:
    row = item["row"]
    request = _request_from_item(
        item, request_id=request_id, arrival_us=arrival_us
    )
    event = threading.Event()
    first_token_ns: list[int] = []

    def on_first_token(value_ns: int) -> None:
        if not first_token_ns:
            first_token_ns.append(value_ns)
        event.set()

    payload = LlamaCppCompletionPayload(
        request_id=request.request_id,
        expected_model_alias=aliases[str(item["model_id"])],
        input_tokens=request.input_tokens,
        output_tokens=request.output_tokens,
        prompt_tokens=tuple(int(value) for value in row["prompt_tokens"]),
        seed=seed,
        stream_path=streams / (request_id + ".raw"),
        on_first_token=on_first_token,
        quality_mode="semantic",
        timeout_s=3_600,
    )
    return _RequestProbe(request, payload, event, first_token_ns)


class _SnapshotStore:
    def __init__(self, output: Path, rig: object) -> None:
        self._output = output
        self._rig = rig
        self._lock = threading.Lock()
        self._sequence = 0
        self.fail_verification_session_id: str | None = None

    def _persist(self, name: str, snapshot: object) -> None:
        with self._lock:
            sequence = self._sequence
            self._sequence += 1
        path = self._output / (
            f"{sequence:04d}-" + name.replace(":", "-") + ".json"
        )
        path.write_bytes(_canonical(snapshot.to_json()))

    def submission(self, ticket: object, scheduler: object) -> None:
        _write_new(
            self._output.parent / ("SUBMISSION-" + ticket.request.request_id + ".json"),
            {
                "ticket": ticket.to_json(),
                "scheduler_decision_log": scheduler.runtime_decision_log(),
            },
        )

    def capture(
        self,
        request: Request,
        model_id: str,
        observed_at_us: int,
        label: str,
    ):
        snapshot = self._rig.snapshot(
            request, model_id, observed_at_us
        )
        self._persist(label + "-physical", snapshot)
        return snapshot

    def runtime(self, ticket: object, observed_at_us: int):
        return self.capture(
            ticket.request,
            ticket.model.model_id,
            observed_at_us,
            "runtime-" + ticket.request.request_id,
        )

    def plan(self, scheduler, requests_by_model, snapshot, epoch_ns):
        model_id, requests = next(iter(requests_by_model.items()))
        return CanonicalOfflinePhoneResidencyPreloader.plan_with_observation_refresh(
            scheduler, requests_by_model, snapshot=snapshot,
            snapshot_provider=lambda at_us: self.capture(
                requests[0], model_id, at_us, "residency-telemetry-refresh"
            ),
            refresh_observation=self._rig.request_runtime_observation_refresh,
            epoch_ns=epoch_ns,
        )

    def offline(self, stage: object, observed_at_us: int):
        snapshot = self.capture(
            stage.request,
            stage.model_id,
            observed_at_us,
            "offline-" + stage.stage_id,
        )
        session_id = self.fail_verification_session_id
        if session_id is None or stage.state != "LOADING":
            return snapshot
        rows = tuple(snapshot.phone_session_residency)
        target = next((
            row for row in rows if row.session_id == session_id
        ), None)
        require(target is not None, "fault session is absent physically")
        self.fail_verification_session_id = None
        injected = replace(target, resident_bytes=target.resident_bytes + 1)
        result = replace(
            snapshot,
            snapshot_id=snapshot.snapshot_id + "-fault-injected",
            phone_session_residency=tuple(
                injected if row.session_id == session_id else row
                for row in rows
            ),
        )
        self._persist("fault-injected-" + stage.stage_id, result)
        return result


def _wait_for(
    description: str,
    probe: Callable[[], object | None],
    *,
    timeout_s: float,
) -> object:
    deadline = time.monotonic() + timeout_s
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            value = probe()
            if value is not None:
                return value
        except (OSError, PhysicalAdapterError) as error:
            last_error = error
        time.sleep(0.05)
    detail = "" if last_error is None else ": " + str(last_error)
    raise OfflineResidencyGateError(
        "timed out waiting for " + description + detail
    )


def _helper_events(scheduler: object, request_id: str) -> list[dict[str, Any]]:
    return [
        dict(row) for row in scheduler.request_helper_events()
        if row.get("request_id") == request_id
    ]


def _request_completed(scheduler: object, request_id: str) -> bool:
    return scheduler.runtime_ticket(request_id).dispatch_state == "COMPLETED"


def _persist_cold_start_diagnostic(
    output: Path,
    scheduler: object,
    request_id: str,
) -> None:
    ticket = scheduler.runtime_ticket(request_id)
    _write_new(output / "COLD_START_DIAGNOSTIC.json", {
        "request_helper_events": _helper_events(scheduler, request_id),
        "request_id": request_id,
        "scheduler_decision_log": scheduler.runtime_decision_log(),
        "schema": "s42-offline-phone-cold-start-diagnostic-v1",
        "ticket": ticket.to_json(),
    })


def _wait_helper_event(
    scheduler: object,
    request_id: str,
    kind: str,
    *,
    timeout_s: float = 300,
) -> dict[str, Any]:
    return _wait_for(
        kind + " for " + request_id,
        lambda: next((
            row for row in _helper_events(scheduler, request_id)
            if row.get("kind") == kind
        ), None),
        timeout_s=timeout_s,
    )


def _call_totals(
    rig: object,
    artifact_sha256: str,
) -> dict[tuple[str, int], int]:
    totals: dict[tuple[str, int], int] = {}
    for row in rig.phone_residency_call_events:
        if row["artifact_sha256"] != artifact_sha256:
            continue
        key = (str(row["session_id"]), int(row["session_generation"]))
        totals[key] = max(totals.get(key, 0), int(row["calls"]))
    return totals


def _wait_call_advance(
    rig: object,
    artifact_sha256: str,
    baseline: Mapping[tuple[str, int], int],
    *,
    allowed_session_ids: Sequence[str] | None = None,
    timeout_s: float = 300,
) -> dict[tuple[str, int], int]:
    allowed = None if allowed_session_ids is None else set(allowed_session_ids)

    def probe() -> object | None:
        totals = _call_totals(rig, artifact_sha256)
        advanced = any(
            (allowed is None or session_id in allowed)
            and count > baseline.get((session_id, generation), 0)
            for (session_id, generation), count in totals.items()
        )
        return totals if advanced else None

    return _wait_for(
        "physical phone call",
        probe,
        timeout_s=timeout_s,
    )


def _wait_session_call_deltas(
    rig: object,
    artifact_sha256: str,
    baseline: Mapping[tuple[str, int], int],
    session_generations: Mapping[str, int],
    *,
    minimum_delta: int,
    request_completed: Callable[[], bool],
    timeout_s: float = 600,
) -> dict[tuple[str, int], int]:
    require(minimum_delta > 0, "call delta must be positive")
    require(callable(request_completed), "request completion probe is invalid")
    deadline = time.monotonic() + timeout_s
    last_error: BaseException | None = None
    totals = _call_totals(rig, artifact_sha256)
    while time.monotonic() < deadline:
        try:
            totals = _call_totals(rig, artifact_sha256)
        except (OSError, PhysicalAdapterError) as error:
            last_error = error
            time.sleep(0.05)
            continue
        if all(
            totals.get((session_id, generation), 0)
                >= baseline.get((session_id, generation), 0) + minimum_delta
            for session_id, generation in session_generations.items()
        ):
            return totals
        if request_completed():
            return totals
        time.sleep(0.05)
    try:
        totals = _call_totals(rig, artifact_sha256)
    except (OSError, PhysicalAdapterError) as error:
        last_error = error
    keys = sorted(set(baseline) | set(totals))
    diagnostics = {
        "artifact_sha256": artifact_sha256,
        "baselines": [
            {
                "calls": int(baseline.get(key, 0)),
                "session_generation": key[1],
                "session_id": key[0],
            }
            for key in keys
        ],
        "minimum_delta": minimum_delta,
        "observed_totals": [
            {
                "calls": int(totals.get(key, 0)),
                "session_generation": key[1],
                "session_id": key[0],
            }
            for key in keys
        ],
        "target_session_generations": {
            key: int(value)
            for key, value in sorted(session_generations.items())
        },
    }
    detail = "" if last_error is None else ": " + str(last_error)
    raise _SessionCallDeltaTimeout(
        "timed out waiting for per-session physical phone call history"
        + detail,
        diagnostics,
    )


def _stage_target_shard(stage: object):
    return next(
        row for row in stage.layout.layout.shards
        if row.session_id == stage.selected_session_id
    )


def _wait_interruption_reference_phase(scheduler, request_id):
    def observed():
        if _request_completed(scheduler, request_id):
            raise RuntimeError(
                "INSUFFICIENT interruption reference: request completed before exploitation"
            )
        if not any(row.get("kind") == "FRACTION_APPLIED"
                   for row in _helper_events(scheduler, request_id)):
            return None
        snapshot = scheduler.adaptive_decode_snapshot(request_id)
        return snapshot if snapshot.get("state") == "EXPLOITING" else None

    return _wait_for("scheduler exploitation before interruption reference",
                     observed, timeout_s=600)


def _stage_target_generation(stage: object) -> int:
    return stage.layout.layout.session_generation_by_id[stage.selected_session_id]


def _stage_phase_interval(
    phase_events: Sequence[Mapping[str, object]],
    stage: object,
) -> dict[str, object]:
    shard = _stage_target_shard(stage)
    generation = _stage_target_generation(stage)
    rows = [
        row for row in phase_events
        if row.get("component") == "resident-manager"
        and row.get("session_id") == stage.selected_session_id
        and row.get("artifact_sha256") == shard.artifact_sha256
        and row.get("session_generation") == generation
    ]
    loads = [row for row in rows if row.get("phase") == "LOAD_AUTHORIZED"]
    verified = [row for row in rows if row.get("phase") == "VERIFIED"]
    ready = [row for row in rows if row.get("phase") == "READY"]
    require(loads and verified and ready, "physical stage phases are incomplete")
    load = loads[-1]
    finish = next((
        row for row in ready
        if int(row["epoch_us"]) >= int(load["epoch_us"])
    ), None)
    proof = next((
        row for row in verified
        if int(load["epoch_us"]) <= int(row["epoch_us"])
        and finish is not None
        and int(row["epoch_us"]) <= int(finish["epoch_us"])
    ), None)
    require(finish is not None and proof is not None, "stage phase order differs")
    return {
        "artifact_sha256": shard.artifact_sha256,
        "load_authorized_epoch_us": int(load["epoch_us"]),
        "load_authorized_monotonic_us": int(load["monotonic_us"]),
        "ready_epoch_us": int(finish["epoch_us"]),
        "ready_monotonic_us": int(finish["monotonic_us"]),
        "selected_session_id": stage.selected_session_id,
        "session_generation": generation,
        "verified_epoch_us": int(proof["epoch_us"]),
        "verified_monotonic_us": int(proof["monotonic_us"]),
    }


def _calls_during(
    call_events: Sequence[Mapping[str, object]],
    interval: Mapping[str, object],
    *,
    artifact_sha256: str,
    allowed_session_ids: Sequence[str],
) -> list[dict[str, object]]:
    start = int(interval["load_authorized_epoch_us"])
    finish = int(interval["ready_epoch_us"])
    allowed = set(allowed_session_ids)
    return [
        dict(row) for row in call_events
        if row.get("artifact_sha256") == artifact_sha256
        and row.get("session_id") in allowed
        and start <= int(row.get("epoch_us", -1)) <= finish
    ]


def _submit_request(
    scheduler: object,
    rig: object,
    snapshots: _SnapshotStore,
    epoch_ns: int,
    probe: _RequestProbe,
    selection_mode: str,
) -> tuple[CanonicalArrivalCoordinator, object]:
    coordinator = CanonicalArrivalCoordinator(
        scheduler,
        None,
        epoch_ns=epoch_ns,
        snapshot_provider=snapshots.runtime,
        max_workers=4,
        backend_factory=lambda _ticket, _payload: rig.backend(),
    )
    observed_at_us = max(
        probe.request.arrival_us,
        _now_us(epoch_ns),
    )
    snapshot = snapshots.capture(
        probe.request,
        str(probe.request.workload_id).removeprefix("offline-gate:"),
        observed_at_us,
        "submit-" + probe.request.request_id,
    )
    ticket = coordinator.submit(
        CanonicalRuntimeSubmission(
            request=probe.request,
            model_id=str(probe.request.workload_id).removeprefix(
                "offline-gate:"
            ),
            snapshot=snapshot,
            payload=probe.payload,
            selection_mode=selection_mode,
        ),
        observed_at_us=max(observed_at_us, snapshot.captured_at_us),
    )
    snapshots.submission(ticket, scheduler)
    return coordinator, ticket


def _execution_json(result: object) -> dict[str, object]:
    payload = result.observation.payload
    proof = (
        payload.get("physical_execution_proof")
        if isinstance(payload, Mapping) else None
    )
    return {
        "attempt_ticket_ids": list(result.attempt_ticket_ids),
        "command": result.command.to_json(),
        "completion": result.completion.to_json(),
        "dispatch_receipts": [
            row.to_json() for row in result.dispatch_receipts
        ],
        "energy": _energy_json(result.observation.energy),
        "finished_us": result.observation.finished_us,
        "output_sha256": result.observation.output_sha256,
        "physical_execution_proof": proof,
        "recoveries": [row.to_json() for row in result.recoveries],
        "started_us": result.observation.started_us,
    }


def _physical_phone_calls(result: object) -> int:
    payload = result.observation.payload
    proof = (
        payload.get("physical_execution_proof")
        if isinstance(payload, Mapping) else None
    )
    require(isinstance(proof, Mapping), "physical execution proof is absent")
    return int(proof.get("phone_call_count") or 0)


def _assert_no_fallback(scheduler: object, result: object) -> None:
    require(not result.recoveries, "physical execution used a fallback")
    records = scheduler.runtime_decision_log().get("records", ())
    require(
        not any(row.get("event_kind") == "FALLBACK" for row in records),
        "scheduler recorded a fallback",
    )


def _logical_physical_map(
    scheduler: object, rig: object
) -> dict[str, dict[str, object]]:
    logical = {
        str(row["session_id"]): dict(row)
        for row in scheduler.phone_residency_session_states()
    }
    physical_rows = rig.direct_phone_residency_state["phone_shards"]
    physical = {str(row["session_id"]): dict(row) for row in physical_rows}
    require(set(logical) == set(physical), "logical and physical sessions differ")
    for session_id, shard in physical.items():
        state = logical[session_id]
        require(state["state"] == "READY", "logical session is not READY")
        for logical_key, physical_key in (
            ("resident_artifact_sha256", "artifact_sha256"),
            ("shard_geometry_sha256", "resident_geometry_sha256"),
            ("operator_plan_sha256", "operator_plan_sha256"),
            ("session_generation", "session_generation"),
            ("resident_bytes", "resident_bytes"),
            ("endpoint", "endpoint"),
        ):
            require(
                state[logical_key] == shard[physical_key],
                "logical and physical session identity differs: " + session_id,
            )
    return {session_id: physical[session_id] for session_id in sorted(physical)}


def _adaptive_group(
    scheduler: object,
    request_id: str,
    ticket_ids: Sequence[str],
) -> dict[str, Any]:
    groups = scheduler.adaptive_decode_observation_snapshot().get(
        "groups", ()
    )
    expected = set(ticket_ids)
    exact = [
        dict(group) for group in groups
        if group.get("request_id") == request_id
        and group.get("ticket_id") in expected
    ]
    require(exact, "adaptive observation group is absent: " + request_id)
    return exact[-1]


def _fraction_weighted_coverage(
    scheduler: object,
    request: Request,
    ticket_ids: Sequence[str],
) -> dict[str, object]:
    attached = [
        row for row in _helper_events(scheduler, request.request_id)
        if row.get("kind") == "ATTACHED"
        and isinstance(row.get("helper_attachment"), Mapping)
    ]
    require(attached, "eligible request was never attached: " + request.request_id)
    eligible_start = max(
        1,
        min(
            int(row["helper_attachment"]["start_token_index"])
            for row in attached
        ),
    )
    eligible_tokens = max(0, request.output_tokens - eligible_start)
    minimum = _gate_adaptive_config().minimum_remaining_tokens
    require(
        eligible_tokens >= minimum,
        "attached request has no configured helper opportunity",
    )
    group = _adaptive_group(scheduler, request.request_id, ticket_ids)
    windows = sorted(
        (dict(row) for row in group.get("windows", ())),
        key=lambda row: (int(row["token_start"]), int(row["token_end"])),
    )
    weighted_ppm_tokens = 0
    last_end = eligible_start
    for row in windows:
        start = max(eligible_start, int(row["token_start"]))
        finish = min(request.output_tokens, int(row["token_end"]))
        if finish <= start:
            continue
        require(start >= last_end, "adaptive coverage windows overlap")
        fraction = int(row["policy"]["split_fraction_ppm"])
        weighted_ppm_tokens += (finish - start) * fraction
        last_end = finish
    tail_tokens = max(0, request.output_tokens - last_end)
    final_policy = group.get("final_policy")
    tail_fraction = (
        0 if not isinstance(final_policy, Mapping)
        else int(final_policy["split_fraction_ppm"])
    )
    weighted_ppm_tokens += tail_tokens * tail_fraction
    coverage_ppm = weighted_ppm_tokens // eligible_tokens
    return {
        "coverage_ppm": coverage_ppm,
        "eligible_start_token_index": eligible_start,
        "eligible_token_count": eligible_tokens,
        "final_fraction_ppm": tail_fraction,
        "fraction_weighted_assisted_tokens": (
            weighted_ppm_tokens / 1_000_000
        ),
        "request_id": request.request_id,
        "window_count": len(windows),
    }


def _require_online_coverage(online: Mapping[str, object]) -> None:
    for model in ("qwen", "gemma"):
        require(
            int(online[model + "_weighted_assisted_coverage"]["coverage_ppm"])
                >= 700_000,
            ("Qwen" if model == "qwen" else "Gemma")
                + " fraction-weighted coverage is below 70 percent",
        )


def _desktop_session_call_events(
    rig: object,
    *,
    artifact_sha256: str,
    request_id: str,
    physical_sessions: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    """Map timestamped server FFN calls to exact resident session masks."""

    counters: dict[tuple[str, int], int] = {}
    result = []
    for event in rig.desktop_ffn_call_events:
        if event.get("artifact_sha256") != artifact_sha256:
            continue
        contexts = tuple(event.get("contexts", ()))
        if not any(
            row.get("scheduler_request_id") == request_id
            for row in contexts
            if isinstance(row, Mapping)
        ):
            continue
        layer = int(event["layer"])
        owners = [
            (session_id, row)
            for session_id, row in physical_sessions.items()
            if row.get("artifact_sha256") == artifact_sha256
            and int(row["layer_mask"]) & (1 << layer)
        ]
        require(
            len(owners) == 1,
            "desktop FFN call has no exact physical session owner",
        )
        session_id, owner = owners[0]
        generation = int(owner["session_generation"])
        key = (session_id, generation)
        counters[key] = counters.get(key, 0) + 1
        result.append({
            "artifact_sha256": artifact_sha256,
            "calls": counters[key],
            "epoch_us": int(event["observed_epoch_us"]),
            "layer": layer,
            "session_generation": generation,
            "session_id": session_id,
        })
    return result


def _retained_session_call_gap(
    call_events: Sequence[Mapping[str, object]],
    interval: Mapping[str, object],
    *,
    artifact_sha256: str,
    session_id: str,
    session_generation: int,
) -> dict[str, object]:
    rows_by_call = {}
    for row in call_events:
        if (
            row.get("artifact_sha256") == artifact_sha256
            and row.get("session_id") == session_id
            and row.get("session_generation") == session_generation
        ):
            rows_by_call[int(row["calls"])] = dict(row)
    rows = sorted(rows_by_call.values(), key=lambda row: int(row["monotonic_us"]))
    started_us = int(interval["load_authorized_monotonic_us"])
    finished_us = int(interval["ready_monotonic_us"])
    before = [row for row in rows if int(row["monotonic_us"]) < started_us]
    require(
        len(before) >= 31,
        "retained session lacks 30 pre-transition call intervals: "
        + session_id,
    )
    reference = before[-31:]
    reference_calls = [int(row["calls"]) for row in reference]
    require(
        all(right == left + 1 for left, right in zip(
            reference_calls, reference_calls[1:]
        )),
        "retained session call log is not per-call: " + session_id,
    )
    reference_gaps = [
        int(right["monotonic_us"]) - int(left["monotonic_us"])
        for left, right in zip(reference, reference[1:])
    ]
    after_start = [
        row for row in rows if int(row["monotonic_us"]) >= started_us
    ]
    first_after = next((
        index for index, row in enumerate(after_start)
        if int(row["monotonic_us"]) > finished_us
    ), None)
    require(
        first_after is not None,
        "retained session lacks a post-transition call: " + session_id,
    )
    measured = [reference[-1], *after_start[:first_after + 1]]
    measured_calls = [int(row["calls"]) for row in measured]
    require(
        all(right == left + 1 for left, right in zip(
            measured_calls, measured_calls[1:]
        )),
        "retained session lost a call event during transition: " + session_id,
    )
    transition_gaps = [
        int(right["monotonic_us"]) - int(left["monotonic_us"])
        for left, right in zip(measured, measured[1:])
        if int(left["monotonic_us"]) <= finished_us
        and int(right["monotonic_us"]) >= started_us
    ]
    require(transition_gaps, "retained session has no transition call gaps")
    baseline_median_us = median(reference_gaps)
    maximum_gap_us = max(transition_gaps)
    bound_us = baseline_median_us * 2
    measurement = {
        "artifact_sha256": artifact_sha256,
        "baseline_interval_count": len(reference_gaps),
        "baseline_median_inter_call_us": baseline_median_us,
        "baseline_maximum_inter_call_us": max(reference_gaps),
        "bound_us": bound_us,
        "clock": "phone_monotonic_us",
        "maximum_transition_inter_call_us": maximum_gap_us,
        "measured_transition_interval_count": len(transition_gaps),
        "reference_calls": reference,
        "session_generation": session_generation,
        "session_id": session_id,
        "transition_calls": measured,
    }
    if maximum_gap_us > bound_us:
        raise _SessionCallGapError(measurement)
    return measurement


def _retained_session_call_gaps(
    call_events: Sequence[Mapping[str, object]],
    interval: Mapping[str, object],
    *,
    artifact_sha256: str,
    session_generations: Mapping[str, int],
) -> list[dict[str, object]]:
    return [
        _retained_session_call_gap(
            call_events,
            interval,
            artifact_sha256=artifact_sha256,
            session_id=session_id,
            session_generation=generation,
        )
        for session_id, generation in sorted(session_generations.items())
    ]


def _matched_retained_call_rows(
    native_events: Sequence[Mapping[str, object]],
    phone_events: Sequence[Mapping[str, object]],
    group: Mapping[str, object],
    physical_sessions: Mapping[str, Mapping[str, object]],
    baselines: Mapping[tuple[str, int], int],
    request_id: str,
    artifact_sha256: str,
) -> list[dict[str, object]]:
    policies = {}
    for window in group["windows"]:
        acknowledgement = window.get("applied_ack")
        if acknowledgement is None:
            continue
        policy = window["policy"]
        generation = int(acknowledgement["plan_generation"])
        identity = tuple(policy[name] for name in (
            "layer_mask", "columns", "split_fraction_ppm",
        ))
        require(generation not in policies or policies[generation] == identity,
                "control generation has inconsistent physical masks")
        policies[generation] = identity
    phone = {
        (row["session_id"], row["session_generation"], row["calls"]): row
        for row in phone_events if row["artifact_sha256"] == artifact_sha256
    }
    counters = dict(baselines)
    result = []
    previous_layer = 1 << 64
    token_ordinal = 0
    for event in native_events:
        contexts = [row for row in event.get("contexts", ())
                    if row.get("scheduler_request_id") == request_id]
        if not contexts or event.get("artifact_sha256") != artifact_sha256:
            continue
        require(len(contexts) == 1 and event["tokens"] == 1,
                "interruption measurement requires exact single-token calls")
        control_generation = int(contexts[0]["plan_generation"])
        require(control_generation in policies,
                "native call lacks an acknowledged physical mask")
        mask, columns, fraction = policies[control_generation]
        layer = int(event["layer"])
        require(mask & (1 << layer) != 0 and columns == event["columns"],
                "native call differs from its acknowledged physical mask")
        if layer <= previous_layer:
            token_ordinal += 1
        previous_layer = layer
        owners = [(session_id, row) for session_id, row in physical_sessions.items()
                  if row["artifact_sha256"] == artifact_sha256
                  and int(row["layer_mask"]) & (1 << layer)]
        if not owners:
            continue
        require(len(owners) == 1, "native call has ambiguous retained ownership")
        session_id, shard = owners[0]
        generation = int(shard["session_generation"])
        key = (session_id, generation)
        counters[key] = counters.get(key, 0) + 1
        physical = phone.get((*key, counters[key]))
        require(physical is not None, "native call lacks its exact phone counter")
        result.append({
            **physical,
            "active_layer_mask": mask,
            "columns": columns,
            "fraction_ppm": fraction,
            "layer": layer,
            "plan_generation": control_generation,
            "request_id": request_id,
            "token_ordinal": token_ordinal,
            "tokens": int(event["tokens"]),
        })
    return result


def _equivalent_call_intervals(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[object, ...], list[dict[str, object]]]:
    classes: dict[tuple[object, ...], list[dict[str, object]]] = {}
    previous = None
    previous_by_layer = {}
    for row in rows:
        candidates = []
        if previous is not None and previous["token_ordinal"] == row["token_ordinal"]:
            candidates.append(("within_token", previous))
        same_layer = previous_by_layer.get(row["layer"])
        if same_layer is not None and row["token_ordinal"] == same_layer["token_ordinal"] + 1:
            candidates.append(("same_layer_next_token", same_layer))
        for kind, left in candidates:
            fields = ("active_layer_mask", "columns", "fraction_ppm", "tokens")
            if (any(left[field] != row[field] for field in fields)
                    or left["plan_generation"] != row["plan_generation"]):
                continue
            start = int(left["monotonic_us"])
            end = int(row["monotonic_us"])
            require(end > start, "matched phone call clock moved backward")
            key = (kind, left["layer"], row["layer"], *(row[name] for name in fields))
            classes.setdefault(key, []).append({
                "from_call": left["calls"], "to_call": row["calls"],
                "started_monotonic_us": start, "finished_monotonic_us": end,
                "gap_us": end - start,
            })
        previous = row
        previous_by_layer[row["layer"]] = row
    return classes


def _matched_session_call_gap(
    rows: Sequence[Mapping[str, object]],
    interval: Mapping[str, object],
    session_id: str,
    generation: int,
) -> dict[str, object]:
    rows = [row for row in rows if row["session_id"] == session_id
            and row["session_generation"] == generation]
    require(all(right["calls"] == left["calls"] + 1
                for left, right in zip(rows, rows[1:])),
            "matched retained call counters are not consecutive")
    started = int(interval["load_authorized_monotonic_us"])
    finished = int(interval["ready_monotonic_us"])
    measured_classes = []
    for key, pairs in sorted(_equivalent_call_intervals(rows).items()):
        during = [row for row in pairs if row["started_monotonic_us"] <= finished
                  and row["finished_monotonic_us"] >= started]
        if not during:
            continue
        before = [row for row in pairs if row["finished_monotonic_us"] < started]
        after = [row for row in pairs if row["started_monotonic_us"] > finished]
        reference = before[-REFERENCE_INTERVAL_COUNT:]
        reference += after[:REFERENCE_INTERVAL_COUNT - len(reference)]
        baseline = median(row["gap_us"] for row in reference) if reference else None
        maximum = max(row["gap_us"] for row in during)
        status = (
            "INSUFFICIENT" if len(reference) < REFERENCE_INTERVAL_COUNT else
            "PASS" if maximum <= 2 * baseline else "FAIL"
        )
        measured_classes.append({
            "kind": key[0], "from_layer": key[1], "to_layer": key[2],
            "active_layer_mask": key[3], "columns": key[4],
            "fraction_ppm": key[5], "tokens": key[6],
            "baseline_median_us": baseline,
            "baseline_before_count": min(len(before), REFERENCE_INTERVAL_COUNT),
            "baseline_after_count": max(0, len(reference) - len(before)),
            "bound_us": None if baseline is None else 2 * baseline,
            "maximum_transition_gap_us": maximum,
            "reference_intervals": reference, "transition_intervals": during,
            "status": status,
        })
    return {
        "schema": INTERRUPTION_METRIC_SCHEMA,
        "session_id": session_id, "session_generation": generation,
        "clock": "phone_monotonic_us",
        "reference_interval_count_per_class": REFERENCE_INTERVAL_COUNT,
        "classes": measured_classes,
        "reason": None if measured_classes else
                  "retained session has no equivalent transition intervals",
        "status": "INSUFFICIENT" if not measured_classes else
                  "PASS" if all(row["status"] == "PASS" for row in measured_classes)
                  else "FAIL",
    }


def _matched_retained_session_call_gaps(
    rig: object, scheduler: object, execution: object,
    interval: Mapping[str, object], *, request_id: str,
    artifact_sha256: str,
    physical_sessions: Mapping[str, Mapping[str, object]],
    baselines: Mapping[tuple[str, int], int],
    collect_incomplete_reference: bool = False,
) -> list[dict[str, object]]:
    phone_events = list(rig.phone_residency_call_events)
    native_events = list(rig.desktop_ffn_call_events)
    rows = _matched_retained_call_rows(
        native_events, phone_events,
        _adaptive_group(scheduler, request_id, execution.attempt_ticket_ids),
        physical_sessions, baselines, request_id, artifact_sha256,
    )
    measurements = []
    for session_id, shard in sorted(physical_sessions.items()):
        generation = int(shard["session_generation"])
        measured = _matched_session_call_gap(rows, interval, session_id, generation)
        try:
            legacy = _retained_session_call_gap(
                phone_events, interval, artifact_sha256=artifact_sha256,
                session_id=session_id, session_generation=generation,
            )
            legacy["status"] = "PASS"
        except _SessionCallGapError as error:
            legacy = {**error.call_gap_measurement, "status": "FAIL"}
        except OfflineResidencyGateError as error:
            legacy = {"status": "INSUFFICIENT", "reason": str(error)}
        measured["legacy_pooled_v1_diagnostic"] = legacy
        measured["matched_calls"] = [row for row in rows if row["session_id"] == session_id]
        measurements.append(measured)
    _require_interruption_evidence(
        measurements, collect_incomplete_reference=collect_incomplete_reference,
    )
    return measurements


def _require_interruption_evidence(
    measurements: Sequence[Mapping[str, object]], *,
    collect_incomplete_reference: bool = False,
) -> None:
    require(bool(measurements), "retained session interruption evidence is absent")
    if all(row["status"] == "PASS" for row in measurements):
        return
    incomplete_only = all(
        row["status"] in ("PASS", "INSUFFICIENT")
        and (row["baseline_median_us"] is None
             or row["maximum_transition_gap_us"] <= 2 * row["baseline_median_us"])
        for measurement in measurements for row in measurement["classes"]
    )
    if not (collect_incomplete_reference and incomplete_only):
        raise _SessionCallGapError({
            "schema": INTERRUPTION_METRIC_SCHEMA,
            "session_id": ",".join(str(row["session_id"]) for row in measurements),
            "reason": "equivalent-call interruption gate failed or lacks reference evidence",
            "status": "FAIL", "sessions": measurements,
        })


def _drain_timeline(
    helper_events: Sequence[Mapping[str, object]],
    timing_events: Sequence[Mapping[str, object]],
    group: Mapping[str, object],
    stage: object,
) -> dict[str, object]:
    bound = next(row for row in helper_events
                 if row["kind"] == "REBIND_DRAIN_POLICY_BOUND")
    request_id = bound["request_id"]
    policy_hash = bound["drain_policy_sha256"]
    requested_us = int(bound["observed_at_us"])
    boundary = next(row for row in timing_events
                    if row["request_id"] == request_id
                    and row["kind"] == "DECODE_BOUNDARY_OBSERVED"
                    and row["observed_at_us"] >= requested_us
                    and not row["terminal"])
    issued = next(row for row in timing_events
                  if row["request_id"] == request_id
                  and row["kind"] == "CONTROL_ISSUED"
                  and row["control"]["policy_hash"] == policy_hash)
    quiesced = next(row for row in helper_events
                    if row["kind"] == "REBIND_QUIESCED"
                    and row["drain_policy_sha256"] == policy_hash)
    receipt = stage.transition_receipts[0]
    windows = [row for row in group["windows"]
               if row["started_at_us"] <= requested_us <= row["finished_at_us"]]
    require(len(windows) == 1, "drain request does not identify its old-policy window")
    window = windows[0]
    return {
        "schema": "s42-session-drain-timeline-v1",
        "clock": "desktop_measurement_epoch_us",
        "request_id": request_id,
        "selected_session_id": stage.selected_session_id,
        "drain_policy_sha256": policy_hash,
        "drain_requested_at_us": requested_us,
        "next_safe_boundary_observed_at_us": boundary["observed_at_us"],
        "next_safe_boundary_token_index": boundary["token_index"],
        "control_issued_at_us": issued["observed_at_us"],
        "applied_ack_and_quiesced_at_us": quiesced["observed_at_us"],
        "replacement_loading_started_at_us": receipt.started_us,
        "physical_ready_ack_at_us": receipt.finished_us,
        "ready_publication_at_us": stage.verified_at_us,
        "drain_to_quiesced_us": quiesced["observed_at_us"] - requested_us,
        "old_policy_window": window,
        "configured_minimum_window_tokens": _gate_adaptive_config().minimum_window_tokens,
        "shortened_window_tokens": window["token_end"] - window["token_start"],
    }


def _gate_adaptive_config() -> AdaptiveDecodeConfig:
    return AdaptiveDecodeConfig(
        minimum_remaining_tokens=16,
        minimum_window_tokens=32,
        maximum_window_tokens=128,
        maximum_probe_tokens=1_024,
        maximum_probe_candidates=4,
        measurement_resolution_us=5_000_000,
        transition_cost_us=2_000,
        transition_energy_uj=20_000,
        minimum_energy_saving_ppm=0,
        maximum_latency_ppm=10_000_000,
        uncertainty_ppm=0,
        exploration_latency_budget_ppm=1_000_000,
        exploration_energy_budget_ppm=1_000_000,
        warmup_windows_per_policy=0,
        allow_assumed_phone_power_for_operational_selection=True,
        coarse_probe_fractions_ppm=(
            1_000_000, 750_000, 500_000, 250_000
        ),
    )


def _new_scheduler(args: argparse.Namespace, models: object):
    return runner._build_scheduler(
        args,
        models,
        adaptive_decode_config=_gate_adaptive_config(),
    )[0:2]


def _advance_stage(
    scheduler: object,
    preloader: CanonicalOfflinePhoneResidencyPreloader,
    snapshots: _SnapshotStore,
    plan: object,
    payload: LlamaCppCompletionPayload,
    epoch_ns: int,
) -> object:
    stage = plan.current_stage
    require(stage is not None, "offline stage is absent")
    observed_at_us = max(_now_us(epoch_ns), stage.request.arrival_us)
    snapshot = snapshots.offline(stage, observed_at_us)
    return preloader.execute_next_stage(
        plan.plan_id,
        payload,
        snapshot=snapshot,
        observed_at_us=max(observed_at_us, snapshot.captured_at_us),
    )


def _propose_next_stage(
    scheduler: object,
    snapshots: _SnapshotStore,
    plan: object,
    epoch_ns: int,
) -> object:
    stage = plan.current_stage
    require(stage is not None, "completed offline stage is absent")
    observed_at_us = max(_now_us(epoch_ns), stage.request.arrival_us)
    snapshot = snapshots.offline(stage, observed_at_us)
    return CanonicalOfflinePhoneResidencyPreloader.next_with_observation_refresh(
        scheduler, plan.plan_id, snapshot=snapshot,
        snapshot_provider=lambda at_us: snapshots.offline(stage, at_us),
        refresh_observation=snapshots._rig.request_runtime_observation_refresh,
        epoch_ns=epoch_ns,
    )


def _select_gate_items(
    models: object,
    selected: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], Any, Any]:
    large = [dict(row) for row in selected if row["source"] == "large"]
    qwen = [
        row for row in large
        if row["model_id"] == models.expected_qwen.model_id
    ]
    gemma = [
        row for row in large
        if row["model_id"] == models.expected_gemma.model_id
    ]
    require(qwen and gemma, "gate model requests are absent")
    qwen_probe = max(
        qwen,
        key=lambda row: (
            row["row"]["output_tokens"],
            row["row"]["input_tokens"],
        ),
    )
    gemma_probe = max(
        gemma,
        key=lambda row: (
            row["row"]["output_tokens"],
            -row["row"]["input_tokens"],
        ),
    )
    return large, qwen_probe, gemma_probe


def _cold_preload(
    args: argparse.Namespace,
    models: object,
    scheduler: object,
    manifests: Mapping[str, object],
    rig: object,
    aliases: Mapping[str, str],
    forecasts: Mapping[str, tuple[Request, ...]],
    qwen_item: Mapping[str, Any],
    streams: Path,
    snapshots: _SnapshotStore,
) -> tuple[object, object, dict[str, object]]:
    qwen_id = models.expected_qwen.model_id
    qwen_artifact = manifests[qwen_id].artifact_sha256
    epoch_ns = time.monotonic_ns()
    rig.begin_offline_preload(epoch_ns)
    representative = forecasts[qwen_id][0]
    initial = snapshots.capture(
        representative, qwen_id, _now_us(epoch_ns), "cold-plan"
    )
    plan = snapshots.plan(
        scheduler, {qwen_id: forecasts[qwen_id]}, initial, epoch_ns,
    )
    require(
        len(plan.target_layout.shards) == 3
        and {row.artifact_sha256 for row in plan.target_layout.shards}
            == {qwen_artifact},
        "cold preload target is not three Qwen shards",
    )
    require(
        plan.target_layout.resident_bytes <= plan.phone_wide_limit_bytes,
        "cold preload target exceeds phone memory",
    )
    request_probe = _request_probe(
        qwen_item,
        aliases,
        streams,
        request_id="offline-gate-cold-qwen",
        arrival_us=_now_us(epoch_ns),
        seed=42,
    )
    coordinator, _ticket = _submit_request(
        scheduler,
        rig,
        snapshots,
        epoch_ns,
        request_probe,
        args.selection_mode,
    )
    first_token_ready = request_probe.first_token.wait(FIRST_TOKEN_TIMEOUT_S)
    if not first_token_ready:
        _persist_cold_start_diagnostic(
            args.output,
            scheduler,
            request_probe.request.request_id,
        )
    require(
        first_token_ready,
        "cold Qwen desktop execution produced no first token",
    )
    first_token_us = (
        request_probe.first_token_ns[0] - epoch_ns
    ) // 1000
    preloader = CanonicalOfflinePhoneResidencyPreloader(
        scheduler,
        rig.backend(),
        epoch_ns=epoch_ns,
        snapshot_provider=snapshots.offline,
    )
    stages = []
    first_call_baseline: dict[tuple[str, int], int] | None = None
    while plan.state != "READY":
        result = _advance_stage(
            scheduler,
            preloader,
            snapshots,
            plan,
            request_probe.payload,
            epoch_ns,
        )
        plan = result.plan
        stage = plan.current_stage
        require(stage is not None and stage.state == "READY", "stage not READY")
        stages.append(stage)
        if len(stages) == 1:
            first_call_baseline = _call_totals(rig, qwen_artifact)
            _wait_helper_event(
                scheduler, request_probe.request.request_id, "ATTACHED"
            )
            _wait_call_advance(
                rig,
                qwen_artifact,
                first_call_baseline,
                allowed_session_ids=(stage.selected_session_id,),
                timeout_s=600,
            )
        if plan.state != "READY":
            plan = _propose_next_stage(
                scheduler, snapshots, plan, epoch_ns
            )
    completed = coordinator.drain(timeout_s=3_600)
    coordinator.close()
    execution = completed.executions[request_probe.request.request_id]
    _assert_no_fallback(scheduler, execution)
    require(_physical_phone_calls(execution) > 0, "cold Qwen used no phone calls")
    require(
        execution.observation.started_us < stages[0].started_at_us
        and first_token_us < stages[0].started_at_us,
        "desktop execution did not start before phone loading",
    )
    state = rig.direct_phone_residency_state
    require(
        state["load_count_by_session"]
            == {row.session_id: 1 for row in plan.target_layout.shards},
        "cold preload loaded a session more than once",
    )
    require(
        {
            row["session_id"]: row["session_generation"]
            for row in state["phone_shards"]
        } == {row.session_id: 1 for row in plan.target_layout.shards},
        "cold Qwen session generations are not 1/1/1",
    )
    phase_events = list(rig.phone_residency_phase_events)
    call_events = list(rig.phone_residency_call_events)
    intervals = [_stage_phase_interval(phase_events, stage) for stage in stages]
    overlap_calls = []
    if len(stages) > 1:
        first_session = stages[0].selected_session_id
        overlap_calls = _calls_during(
            call_events,
            intervals[1],
            artifact_sha256=qwen_artifact,
            allowed_session_ids=(first_session,),
        )
        require(
            overlap_calls,
            "first READY Qwen session did not serve during the next load",
        )
    finished_ns = time.monotonic_ns()
    energy = rig.trace_energy(epoch_ns, finished_ns)
    return plan, execution, {
        "concurrent_validation_request_ids": [request_probe.request.request_id],
        "energy": _energy_json(energy),
        "finished_epoch_ns": finished_ns,
        "first_ready_session_calls_during_next_load": overlap_calls,
        "first_token_us": first_token_us,
        "layout_phase_intervals": intervals,
        "physical_state": dict(state),
        "plan": plan.to_json(),
        "request": _execution_json(execution),
        "request_helper_events": _helper_events(
            scheduler, request_probe.request.request_id
        ),
        "started_epoch_ns": epoch_ns,
        "wall_time_us": (finished_ns - epoch_ns) // 1000,
    }


def _online_gate(
    args: argparse.Namespace,
    models: object,
    scheduler: object,
    manifests: Mapping[str, object],
    rig: object,
    aliases: Mapping[str, str],
    forecasts: Mapping[str, tuple[Request, ...]],
    persisted_plan: object,
    qwen_item: Mapping[str, Any],
    gemma_item: Mapping[str, Any],
    streams: Path,
    snapshots: _SnapshotStore,
) -> dict[str, object]:
    qwen_id = models.expected_qwen.model_id
    gemma_id = models.expected_gemma.model_id
    qwen_artifact = manifests[qwen_id].artifact_sha256
    gemma_artifact = manifests[gemma_id].artifact_sha256
    epoch_ns = time.monotonic_ns()
    rig.begin_trace(epoch_ns)
    initial = snapshots.capture(
        forecasts[qwen_id][0], qwen_id, _now_us(epoch_ns), "reuse-adopt"
    )
    adoption_started_ns = time.monotonic_ns()
    adopted = scheduler.adopt_offline_phone_residency(
        persisted_plan,
        {qwen_id: forecasts[qwen_id]},
        snapshot=initial,
        observed_at_us=initial.captured_at_us,
    )
    adoption_finished_ns = time.monotonic_ns()
    load_counts_at_adoption = dict(
        rig.direct_phone_residency_state["load_count_by_session"]
    )
    require(adopted.state == "READY", "resident reuse was not adopted")

    gemma_demand = {gemma_id: (forecasts[gemma_id][0],)}
    gemma_snapshot = snapshots.capture(
        gemma_demand[gemma_id][0],
        gemma_id,
        _now_us(epoch_ns),
        "gemma-plan",
    )
    plan = snapshots.plan(
        scheduler, gemma_demand, gemma_snapshot, epoch_ns,
    )
    selected = plan.current_stage.selected_session_id
    target_artifacts = [
        row.artifact_sha256 for row in plan.target_layout.shards
    ]
    require(
        plan.current_stage.layout.layout.changed_session_ids == (selected,),
        "Gemma replacement is not one scheduler-selected session",
    )
    require(
        target_artifacts.count(gemma_artifact) == 1
        and target_artifacts.count(qwen_artifact) == 2
        and not plan.pending_layouts,
        "online target is not one-stage mixed Qwen/Gemma residency",
    )
    adopted_by_session = {
        row["session_id"]: dict(row)
        for row in rig.direct_phone_residency_state["phone_shards"]
    }
    source_generation = int(
        adopted_by_session[selected]["session_generation"]
    )
    retained = tuple(sorted(set(adopted_by_session) - {selected}))
    retained_generations = {
        session_id: int(
            adopted_by_session[session_id]["session_generation"]
        )
        for session_id in retained
    }
    qwen_probe = _request_probe(
        qwen_item,
        aliases,
        streams,
        request_id="offline-gate-online-qwen",
        arrival_us=_now_us(epoch_ns),
        seed=43,
    )
    qwen_counter_baseline = _call_totals(rig, qwen_artifact)
    coordinator, _ticket = _submit_request(
        scheduler,
        rig,
        snapshots,
        epoch_ns,
        qwen_probe,
        args.selection_mode,
    )
    _write_new(
        args.output / "ONLINE_QWEN_SUBMISSION.json",
        _ticket.to_json(),
    )
    require(
        type(_ticket.execution_plan.adapter_parameters.get(
            "dormant_phone_ffn_runtime_v1"
        )) is str,
        "online Qwen desktop parent lacks dormant phone runtime",
    )
    require(
        qwen_probe.first_token.wait(900),
        "online Qwen produced no first token",
    )
    qwen_baseline = _call_totals(rig, qwen_artifact)
    qwen_attached = _wait_helper_event(
        scheduler, qwen_probe.request.request_id, "ATTACHED"
    )
    _wait_session_call_deltas(
        rig,
        qwen_artifact,
        qwen_baseline,
        retained_generations,
        minimum_delta=31,  # Thirty preceding per-call intervals.
        request_completed=lambda: _request_completed(
            scheduler, qwen_probe.request.request_id
        ),
    )
    preloader = CanonicalOfflinePhoneResidencyPreloader(
        scheduler,
        rig.backend(),
        epoch_ns=epoch_ns,
        snapshot_provider=snapshots.offline,
    )
    replacement = _advance_stage(
        scheduler,
        preloader,
        snapshots,
        plan,
        _request_probe(
            gemma_item,
            aliases,
            streams,
            request_id="offline-gate-gemma-transition-payload",
            arrival_us=_now_us(epoch_ns),
            seed=44,
        ).payload,
        epoch_ns,
    )
    plan = replacement.plan
    replacement_stage = plan.current_stage
    require(
        replacement_stage.selected_session_id == selected
        and replacement_stage.state == "READY",
        "Gemma replacement changed its authorized session",
    )
    require(plan.state == "READY", "mixed replacement did not finish in one stage")
    post_forward_baseline = _call_totals(rig, qwen_artifact)
    _wait_session_call_deltas(
        rig,
        qwen_artifact,
        post_forward_baseline,
        retained_generations,
        minimum_delta=1,
        request_completed=lambda: _request_completed(
            scheduler, qwen_probe.request.request_id
        ),
    )
    phase_events = list(rig.phone_residency_phase_events)
    call_events = list(rig.phone_residency_call_events)
    replacement_interval = _stage_phase_interval(
        phase_events, replacement_stage
    )
    retained_calls = _calls_during(
        call_events,
        replacement_interval,
        artifact_sha256=qwen_artifact,
        allowed_session_ids=retained,
    )
    require(retained_calls, "Qwen did not serve on retained sessions during load")
    after_replacement = rig.direct_phone_residency_state
    after_replacement_loads = dict(
        after_replacement["load_count_by_session"]
    )
    require(
        {
            session_id: after_replacement["load_count_by_session"][session_id]
            - load_counts_at_adoption[session_id]
            for session_id in load_counts_at_adoption
        } == {
            session_id: int(session_id == selected)
            for session_id in load_counts_at_adoption
        },
        "replacement reloaded more than the selected session",
    )
    replacement_by_session = {
        row["session_id"]: row
        for row in after_replacement["phone_shards"]
    }
    require(
        replacement_by_session[selected]["artifact_sha256"] == gemma_artifact
        and replacement_by_session[selected]["session_generation"]
            == source_generation + 1
        and all(
            replacement_by_session[session_id]["artifact_sha256"]
                == qwen_artifact
            and replacement_by_session[session_id]["session_generation"] == 1
            for session_id in retained
        ),
        "mixed layout identity differs after replacement",
    )
    qwen_completed = coordinator.drain(timeout_s=3_600)
    coordinator.close()
    qwen_execution = qwen_completed.executions[qwen_probe.request.request_id]
    _assert_no_fallback(scheduler, qwen_execution)
    require(_physical_phone_calls(qwen_execution) > 0, "online Qwen used no phone")
    qwen_group = _adaptive_group(
        scheduler,
        qwen_probe.request.request_id,
        qwen_execution.attempt_ticket_ids,
    )
    _write_new(args.output / "ONLINE_QWEN_RESULT.json", {
        "adaptive_group": qwen_group,
        "physical_state": dict(rig.direct_phone_residency_state),
        "replacement_stage": replacement_stage.to_json(),
        "request": _execution_json(qwen_execution),
        "request_helper_events": _helper_events(
            scheduler, qwen_probe.request.request_id
        ),
        "adaptive_timing_events": list(rig.backend().adaptive_timing_events),
    })
    qwen_windows = [dict(row) for row in qwen_group.get("windows", ())]
    fractions = {
        int(row["policy"]["split_fraction_ppm"])
        for row in qwen_windows
    }
    require(
        REQUIRED_FRACTIONS.issubset(fractions),
        "runtime fraction sweep is incomplete: "
        + ",".join(str(value) for value in sorted(fractions)),
    )
    require(
        rig.direct_phone_residency_state["load_count_by_session"]
            == after_replacement_loads,
        "runtime fraction changes reloaded phone weights",
    )
    qwen_events = _helper_events(scheduler, qwen_probe.request.request_id)
    event_kinds = {row["kind"] for row in qwen_events}
    require(
        {"REBIND_DRAIN_POLICY_BOUND", "REBIND_QUIESCED"}.issubset(
            event_kinds
        ),
        "replacement lacks an acknowledged session mask",
    )
    drain_timeline = _drain_timeline(
        qwen_events, rig.backend().adaptive_timing_events, qwen_group,
        replacement_stage,
    )
    _write_new(args.output / "DRAIN_TIMELINE.json", drain_timeline)
    forward_gaps = _matched_retained_session_call_gaps(
        rig, scheduler, qwen_execution,
        replacement_interval,
        request_id=qwen_probe.request.request_id,
        artifact_sha256=qwen_artifact,
        physical_sessions={key: adopted_by_session[key] for key in retained},
        baselines=qwen_counter_baseline,
    )
    qwen_coverage = _fraction_weighted_coverage(
        scheduler,
        qwen_probe.request,
        qwen_execution.attempt_ticket_ids,
    )
    gemma_probe = _request_probe(
        gemma_item,
        aliases,
        streams,
        request_id="offline-gate-online-gemma",
        arrival_us=_now_us(epoch_ns),
        seed=45,
    )
    gemma_coordinator, _ticket = _submit_request(
        scheduler,
        rig,
        snapshots,
        epoch_ns,
        gemma_probe,
        args.selection_mode,
    )
    _write_new(args.output / "ONLINE_GEMMA_SUBMISSION.json", _ticket.to_json())
    gemma_loads_before = dict(
        rig.direct_phone_residency_state["load_count_by_session"]
    )
    require(
        gemma_probe.first_token.wait(900),
        "Gemma produced no first token",
    )
    gemma_baseline = _call_totals(rig, gemma_artifact)
    _wait_helper_event(scheduler, gemma_probe.request.request_id, "ATTACHED")
    _wait_call_advance(
        rig,
        gemma_artifact,
        gemma_baseline,
        allowed_session_ids=(selected,),
        timeout_s=600,
    )
    gemma_completed = gemma_coordinator.drain(timeout_s=3_600)
    gemma_coordinator.close()
    gemma_execution = gemma_completed.executions[
        gemma_probe.request.request_id
    ]
    _assert_no_fallback(scheduler, gemma_execution)
    require(_physical_phone_calls(gemma_execution) > 0, "Gemma used no phone")
    gemma_loads_after = dict(
        rig.direct_phone_residency_state["load_count_by_session"]
    )
    require(gemma_loads_after == gemma_loads_before,
            "Gemma attachment reloaded a resident shard")
    _write_new(args.output / "ONLINE_GEMMA_RESULT.json", {
        "load_count_before_attachment": gemma_loads_before,
        "load_count_after_execution": gemma_loads_after,
        "scheduler_decision_log": scheduler.runtime_decision_log(),
        "adaptive_group": _adaptive_group(
            scheduler,
            gemma_probe.request.request_id,
            gemma_execution.attempt_ticket_ids,
        ),
        "physical_state": dict(rig.direct_phone_residency_state),
        "request": _execution_json(gemma_execution),
        "request_helper_events": _helper_events(
            scheduler, gemma_probe.request.request_id
        ),
    })
    gemma_coverage = _fraction_weighted_coverage(
        scheduler,
        gemma_probe.request,
        gemma_execution.attempt_ticket_ids,
    )
    reverse_snapshot = snapshots.capture(
        forecasts[qwen_id][0],
        qwen_id,
        _now_us(epoch_ns),
        "qwen-reverse-plan",
    )
    reverse_plan = snapshots.plan(
        scheduler, {qwen_id: (forecasts[qwen_id][0],)}, reverse_snapshot, epoch_ns,
    )
    reverse_stage = reverse_plan.current_stage
    _write_new(args.output / "REVERSE_PLAN.json", reverse_plan.to_json())
    require(
        reverse_stage.selected_session_id == selected
        and reverse_stage.layout.layout.changed_session_ids == (selected,)
        and not reverse_plan.pending_layouts,
        "reverse replacement did not preserve the selected session",
    )
    reverse_target = _stage_target_shard(reverse_stage)
    require(
        reverse_target.artifact_sha256 == qwen_artifact
        and _stage_target_generation(reverse_stage) == source_generation + 2,
        "reverse Qwen target identity differs",
    )
    reverse_fault_probe = _request_probe(
        qwen_item,
        aliases,
        streams,
        request_id="offline-gate-reverse-fault-qwen",
        arrival_us=_now_us(epoch_ns),
        seed=46,
    )
    fault_counter_baseline = _call_totals(rig, qwen_artifact)
    fault_coordinator, _ticket = _submit_request(
        scheduler,
        rig,
        snapshots,
        epoch_ns,
        reverse_fault_probe,
        args.selection_mode,
    )
    require(
        reverse_fault_probe.first_token.wait(900),
        "reverse-fault Qwen produced no first token",
    )
    _wait_helper_event(
        scheduler, reverse_fault_probe.request.request_id, "ATTACHED"
    )
    _write_new(args.output / "REVERSE_FAULT_REFERENCE_PHASE.json",
               dict(_wait_interruption_reference_phase(
                   scheduler, reverse_fault_probe.request.request_id)))
    fault_call_baseline = _call_totals(rig, qwen_artifact)
    _wait_session_call_deltas(
        rig,
        qwen_artifact,
        fault_call_baseline,
        retained_generations,
        minimum_delta=31 * max(int(adopted_by_session[key]["layer_mask"]).bit_count()
                               for key in retained),
        request_completed=lambda: _request_completed(
            scheduler, reverse_fault_probe.request.request_id
        ),
    )
    source_map = {
        row["session_id"]: dict(row)
        for row in rig.direct_phone_residency_state["phone_shards"]
    }
    snapshots.fail_verification_session_id = selected
    fault_error = None
    fault_traceback = None
    try:
        _advance_stage(
            scheduler,
            preloader,
            snapshots,
            reverse_plan,
            reverse_fault_probe.payload,
            epoch_ns,
        )
    except PhysicalAdapterError as error:
        fault_error = str(error)
        fault_traceback = traceback.format_exc()
    fault_stage = scheduler.offline_phone_residency_stage(
        reverse_plan.plan_id
    )
    _write_new(args.output / "REVERSE_FAULT_RESULT.json", {
        "error": fault_error,
        "traceback": fault_traceback,
        "stage": None if fault_stage is None else fault_stage.to_json(),
        "injection_consumed": snapshots.fail_verification_session_id is None,
        "physical_state": dict(rig.direct_phone_residency_state),
        "phone_residency_phase_events": list(rig.phone_residency_phase_events),
    })
    require(fault_error is not None, "reverse post-load fault did not fail")
    require(
        snapshots.fail_verification_session_id is None,
        "reverse preparation failed before the injected verification: "
        + str(None if fault_stage is None else fault_stage.failure_reason),
    )
    require(
        fault_stage is not None and fault_stage.state == "FAILED",
        "reverse fault did not fail its one stage",
    )
    fault_interval = _stage_phase_interval(
        list(rig.phone_residency_phase_events), fault_stage
    )
    post_fault_baseline = _call_totals(rig, qwen_artifact)
    _wait_session_call_deltas(
        rig,
        qwen_artifact,
        post_fault_baseline,
        retained_generations,
        minimum_delta=1,
        request_completed=lambda: _request_completed(
            scheduler, reverse_fault_probe.request.request_id
        ),
    )
    logical_physical = _logical_physical_map(scheduler, rig)
    require(
        logical_physical[selected]["artifact_sha256"]
            == source_map[selected]["artifact_sha256"]
        and logical_physical[selected]["session_generation"]
            == source_generation + 3
        and all(
            logical_physical[session_id]
                == source_map[session_id]
            for session_id in retained
        ),
        "reverse rollback disturbed the physical session map",
    )
    failed_loads = dict(
        rig.direct_phone_residency_state["load_count_by_session"]
    )
    require(
        failed_loads[selected]
            == after_replacement_loads[selected] + 2
        and all(
            failed_loads[session_id]
                == after_replacement_loads[session_id]
            for session_id in retained
        ),
        "reverse rollback reloaded an unrelated session",
    )
    fault_completed = fault_coordinator.drain(timeout_s=3_600)
    fault_coordinator.close()
    fault_execution = fault_completed.executions[
        reverse_fault_probe.request.request_id
    ]
    _assert_no_fallback(scheduler, fault_execution)
    require(
        _physical_phone_calls(fault_execution) > 0,
        "reverse-fault Qwen used no retained session",
    )
    restored = [row for row in rig.phone_residency_phase_events
                if row["session_id"] == selected
                and row["artifact_sha256"] == source_map[selected]["artifact_sha256"]
                and row["session_generation"] == source_generation + 3
                and row["phase"] == "READY"]
    require(restored, "physical rollback lacks a READY timestamp")
    fault_interval = {
        **fault_interval,
        "loaded_target_ready_monotonic_us": fault_interval["ready_monotonic_us"],
        "ready_monotonic_us": restored[-1]["monotonic_us"],
        "ready_epoch_us": restored[-1]["epoch_us"],
        "measurement_scope": "replacement_and_physical_rollback",
    }
    fault_gaps = _matched_retained_session_call_gaps(
        rig, scheduler, fault_execution, fault_interval,
        request_id=reverse_fault_probe.request.request_id,
        artifact_sha256=qwen_artifact,
        physical_sessions={key: source_map[key] for key in retained},
        baselines=fault_counter_baseline,
        collect_incomplete_reference=True,
    )

    retry_snapshot = snapshots.capture(
        forecasts[qwen_id][0],
        qwen_id,
        _now_us(epoch_ns),
        "qwen-reverse-retry-plan",
    )
    retry_plan = snapshots.plan(
        scheduler, {qwen_id: (forecasts[qwen_id][0],)}, retry_snapshot, epoch_ns,
    )
    retry_stage = retry_plan.current_stage
    retry_target = _stage_target_shard(retry_stage)
    require(
        retry_stage.selected_session_id == selected
        and retry_stage.layout.layout.changed_session_ids == (selected,)
        and retry_target.artifact_sha256 == qwen_artifact
        and _stage_target_generation(retry_stage) == source_generation + 4
        and not retry_plan.pending_layouts,
        "reverse retry changed its authorized assignment",
    )
    reverse_retry_probe = _request_probe(
        qwen_item,
        aliases,
        streams,
        request_id="offline-gate-reverse-retry-qwen",
        arrival_us=_now_us(epoch_ns),
        seed=47,
    )
    retry_counter_baseline = _call_totals(rig, qwen_artifact)
    retry_coordinator, _ticket = _submit_request(
        scheduler,
        rig,
        snapshots,
        epoch_ns,
        reverse_retry_probe,
        args.selection_mode,
    )
    require(
        reverse_retry_probe.first_token.wait(900),
        "reverse-retry Qwen produced no first token",
    )
    _wait_helper_event(
        scheduler, reverse_retry_probe.request.request_id, "ATTACHED"
    )
    _write_new(args.output / "REVERSE_RETRY_REFERENCE_PHASE.json",
               dict(_wait_interruption_reference_phase(
                   scheduler, reverse_retry_probe.request.request_id)))
    retry_call_baseline = _call_totals(rig, qwen_artifact)
    _wait_session_call_deltas(
        rig,
        qwen_artifact,
        retry_call_baseline,
        retained_generations,
        minimum_delta=31 * max(int(source_map[key]["layer_mask"]).bit_count()
                               for key in retained),
        request_completed=lambda: _request_completed(
            scheduler, reverse_retry_probe.request.request_id
        ),
    )
    retry_result = _advance_stage(
        scheduler,
        preloader,
        snapshots,
        retry_plan,
        reverse_retry_probe.payload,
        epoch_ns,
    )
    require(
        retry_result.plan.state == "READY"
        and retry_result.plan.current_stage.selected_session_id == selected,
        "reverse retry did not publish Qwen READY",
    )
    retry_interval = _stage_phase_interval(
        list(rig.phone_residency_phase_events),
        retry_result.plan.current_stage,
    )
    post_retry_baseline = _call_totals(rig, qwen_artifact)
    final_generations = {
        **retained_generations,
        selected: source_generation + 4,
    }
    _wait_session_call_deltas(
        rig,
        qwen_artifact,
        post_retry_baseline,
        final_generations,
        minimum_delta=1,
        request_completed=lambda: _request_completed(
            scheduler, reverse_retry_probe.request.request_id
        ),
    )
    retry_completed = retry_coordinator.drain(timeout_s=3_600)
    retry_coordinator.close()
    retry_execution = retry_completed.executions[
        reverse_retry_probe.request.request_id
    ]
    _assert_no_fallback(scheduler, retry_execution)
    require(
        _physical_phone_calls(retry_execution) > 0,
        "reverse-retry Qwen used no phone",
    )
    retry_gaps = _matched_retained_session_call_gaps(
        rig, scheduler, retry_execution, retry_interval,
        request_id=reverse_retry_probe.request.request_id,
        artifact_sha256=qwen_artifact,
        physical_sessions={key: source_map[key] for key in retained},
        baselines=retry_counter_baseline,
        collect_incomplete_reference=True,
    )
    final_map = _logical_physical_map(scheduler, rig)
    require(
        final_map[selected]["artifact_sha256"] == qwen_artifact
        and final_map[selected]["session_generation"]
            == source_generation + 4
        and all(
            final_map[session_id]["session_generation"]
                == retained_generations[session_id]
            for session_id in retained
        ),
        "reverse retry final session map differs",
    )
    final_loads = dict(
        rig.direct_phone_residency_state["load_count_by_session"]
    )
    require(
        final_loads[selected] == load_counts_at_adoption[selected] + 4
        and all(
            final_loads[session_id] == load_counts_at_adoption[session_id]
            for session_id in retained
        ),
        "reverse sequence reloaded an unrelated session",
    )
    finished_ns = time.monotonic_ns()
    energy = rig.trace_energy(epoch_ns, finished_ns)
    applied_at_us = min(
        int(row["observed_at_us"]) for row in qwen_events
        if row["kind"] == "FRACTION_APPLIED"
        and row.get("accepted") is True
        and int(row.get("selected_fraction_ppm", 0)) > 0
    )
    helper_attachment_time_us = applied_at_us - qwen_execution.observation.started_us
    require(helper_attachment_time_us >= 0, "helper attachment precedes execution")
    return {
        "adoption": {
            "finished_epoch_ns": adoption_finished_ns,
            "load_count_after": load_counts_at_adoption,
            "load_count_before": load_counts_at_adoption,
            "plan": adopted.to_json(),
            "started_epoch_ns": adoption_started_ns,
            "verification_time_us": (
                adoption_finished_ns - adoption_started_ns
            ) // 1000,
        },
        "energy": _energy_json(energy),
        "failed_reverse_replacement": {
            "error": fault_error,
            "interference": fault_gaps,
            "layout_phase_interval": fault_interval,
            "logical_physical_map": logical_physical,
            "qwen_request": _execution_json(fault_execution),
            "qwen_request_helper_events": _helper_events(
                scheduler, reverse_fault_probe.request.request_id
            ),
            "selected_session_id": selected,
        },
        "finished_epoch_ns": finished_ns,
        "gemma_request": _execution_json(gemma_execution),
        "gemma_request_helper_events": _helper_events(
            scheduler, gemma_probe.request.request_id
        ),
        "gemma_weighted_assisted_coverage": gemma_coverage,
        "helper_attachment_time_us": helper_attachment_time_us,
        "logical_attachment_at_us": int(qwen_attached["observed_at_us"]),
        "first_nonzero_fraction_applied_at_us": applied_at_us,
        "qwen_request": _execution_json(qwen_execution),
        "qwen_request_helper_events": qwen_events,
        "qwen_weighted_assisted_coverage": qwen_coverage,
        "replacement": {
            "drain_timeline": drain_timeline,
            "interference": forward_gaps,
            "layout_phase_interval": replacement_interval,
            "retained_qwen_calls_during_load": retained_calls,
            "retained_session_ids": list(retained),
            "selected_session_id": selected,
            "stage": replacement_stage.to_json(),
        },
        "reverse_retry": {
            "final_logical_physical_map": final_map,
            "interference": retry_gaps,
            "layout_phase_interval": retry_interval,
            "qwen_request": _execution_json(retry_execution),
            "qwen_request_helper_events": _helper_events(
                scheduler, reverse_retry_probe.request.request_id
            ),
            "selected_session_id": selected,
            "stage": retry_result.plan.current_stage.to_json(),
        },
        "runtime_fractions_ppm": sorted(fractions),
        "started_epoch_ns": epoch_ns,
        "wall_time_us": (finished_ns - epoch_ns) // 1000,
    }


def _terminal_checks(rig: object) -> dict[str, object]:
    receipts = list(rig.direct_phone_receipts)
    terminal_rows = [row for row in receipts if "terminal" in row]
    require(len(terminal_rows) == 1, "phone endpoint restarted during the gate")
    terminal = terminal_rows[0]["terminal"]
    require(terminal["reset_recoveries"] == 0, "phone USB reset was observed")
    require(terminal["status"] == 0, "phone terminal status failed")
    require(
        any(
            row.get("artifact_sha256") is not None
            and row.get("session_generation") == 1
            and row.get("calls", 0) > 0
            for row in terminal.get("session_proofs", ())
        ),
        "historical generation-1 execution proof is absent",
    )
    return {
        "direct_phone_receipts": receipts,
        "terminal": terminal,
        "transport_qualifications": list(rig.transport_qualifications),
        "usb_restore_receipts": list(rig.usb_restore_receipts),
    }


def _combined_journal(cold_scheduler: object, online_scheduler: object):
    body = {
        "cold_preload": cold_scheduler.runtime_decision_log(),
        "online_reuse_and_replacement": online_scheduler.runtime_decision_log(),
        "schema": DECISION_LOG_SCHEMA,
    }
    return {**body, "journal_sha256": _sha256_bytes(_canonical(body))}


def run_gate(args: argparse.Namespace) -> dict[str, object]:
    host_dependencies = runner._validate_arguments(args)
    models = runner._load_trace_models(args)
    cold_scheduler, manifests = _new_scheduler(args, models)
    aliases, selected, _replay, _isolated = runner._select_replay(
        args, models, manifests
    )
    large, qwen_item, gemma_item = _select_gate_items(models, selected)
    all_forecasts = _forecast_requests(large)
    forecasts = {
        models.expected_qwen.model_id: all_forecasts[
            models.expected_qwen.model_id
        ],
        models.expected_gemma.model_id: all_forecasts[
            models.expected_gemma.model_id
        ],
    }
    args.output.mkdir(parents=True)
    streams = args.output / "streams"
    snapshots_path = args.output / "snapshots"
    streams.mkdir()
    snapshots_path.mkdir()
    rig = runner._build_rig(args, models, manifests, host_dependencies)
    preflight = rig.direct_phone_preflight()
    _write_new(args.output / "DIRECT_PHONE_PREFLIGHT.json", preflight.to_json())
    snapshots = _SnapshotStore(snapshots_path, rig)
    active_scheduler = cold_scheduler
    primary_error: BaseException | None = None
    terminal = None
    try:
        rig.start(runner._warm_payload(models, aliases, streams))
        persisted_plan, _cold_execution, cold = _cold_preload(
            args,
            models,
            cold_scheduler,
            manifests,
            rig,
            aliases,
            forecasts,
            qwen_item,
            streams,
            snapshots,
        )
        _write_new(args.output / "COLD_PRELOAD_RESULT.json", cold)
        before_restart = dict(rig.direct_phone_residency_state)
        restart = dict(rig.restart_desktop_campaign())
        after_restart = dict(rig.direct_phone_residency_state)
        require(before_restart == after_restart, "desktop restart changed residency")
        online_scheduler, online_manifests = _new_scheduler(args, models)
        active_scheduler = online_scheduler
        require(
            {
                key: value.artifact_sha256
                for key, value in manifests.items()
            } == {
                key: value.artifact_sha256
                for key, value in online_manifests.items()
            },
            "desktop restart changed model identities",
        )
        online = _online_gate(
            args,
            models,
            online_scheduler,
            online_manifests,
            rig,
            aliases,
            forecasts,
            persisted_plan,
            qwen_item,
            gemma_item,
            streams,
            snapshots,
        )
        _write_new(args.output / "ONLINE_LIFECYCLE_RESULT.json", online)
        phase_events = list(rig.phone_residency_phase_events)
        call_events = list(rig.phone_residency_call_events)
        rig.end_trace()
        terminal = _terminal_checks(rig)
        _write_new(args.output / "TERMINAL_PROOF.json", terminal)
        journal = _combined_journal(cold_scheduler, online_scheduler)
        _write_new(args.output / "SCHEDULER_DECISION_LOG.json", journal)
        for phase in ("replacement", "failed_reverse_replacement", "reverse_retry"):
            _require_interruption_evidence(online[phase]["interference"])
        _require_online_coverage(online)
        result = {
            "interruption_metric_schema": INTERRUPTION_METRIC_SCHEMA,
            "cuda_graph_mode_by_artifact": {
                row.artifact_sha256: row.cuda_graph_mode
                for row in models.catalog.desktop_control_profiles
            },
            "absolute_runtime_timing": _runtime_timing_files(args.output),
            "adaptive_timing_events": list(rig.backend().adaptive_timing_events),
            "calls": call_events,
            "catalog_sha256": "sha256:" + runner.digest(
                args.capability_catalog
            ),
            "checks": {
                "automatic_ready_helper_attachment": True,
                "cold_qwen_generation_1_1_1": True,
                "desktop_restart_reused_residency": True,
                "desktop_waited_for_phone_preparation": False,
                "failed_session_isolated": True,
                "fraction_weighted_coverage_at_least_70_percent": True,
                "global_phone_endpoint_restarts": 0,
                "logical_physical_maps_equal": True,
                "progressive_session_publication": True,
                "equivalent_call_class_gap_at_most_2x_matched_median_v2": True,
                "reverse_replacement_used_selected_session": True,
                "runtime_fraction_reload_count": 0,
                "runtime_fractions_complete": True,
                "scheduler_fallbacks": 0,
                "stale_generation_failures": 0,
                "usb_resets": 0,
            },
            "cold_offline_preload": cold,
            "desktop_restart": restart,
            "model_artifacts": {
                model_id: {
                    "artifact_bytes": manifest.artifact_bytes,
                    "artifact_sha256": manifest.artifact_sha256,
                }
                for model_id, manifest in sorted(manifests.items())
            },
            "online_reuse_and_replacement": online,
            "phases": phase_events,
            "schema": RESULT_SCHEMA,
            "source_manifest_sha256": (
                None
                if args.source_manifest is None else
                "sha256:" + runner.digest(args.source_manifest)
            ),
            "status": "PASS",
            "terminal_proof": terminal,
            "trace_executed": False,
        }
        _write_new(args.output / "RESULT.json", result)
        _write_new(
            args.output / "AUTOMATED_OBSERVATIONS.json",
            dict(online_scheduler.automated_observation_snapshot()),
        )
        _write_new(
            args.output / "ADAPTIVE_DECODE_OBSERVATIONS.json",
            dict(online_scheduler.adaptive_decode_observation_snapshot()),
        )
        return result
    except BaseException as error:
        primary_error = error
        try:
            failure = {
                "cuda_graph_mode_by_artifact": {
                    row.artifact_sha256: row.cuda_graph_mode
                    for row in models.catalog.desktop_control_profiles
                },
                "absolute_runtime_timing": _runtime_timing_files(args.output),
                "scheduler_decision_log": active_scheduler.runtime_decision_log(),
                "interruption_metric_schema": INTERRUPTION_METRIC_SCHEMA,
                "adaptive_timing_events": list(rig.backend().adaptive_timing_events),
                "desktop_ffn_call_events": list(rig.desktop_ffn_call_events),
                "direct_phone_receipts": list(rig.direct_phone_receipts),
                "error": type(error).__name__ + ": " + str(error),
                "physical_state": dict(rig.direct_phone_residency_state),
                "phone_residency_phase_events": list(rig.phone_residency_phase_events),
                "phone_residency_call_events": list(rig.phone_residency_call_events),
                "request_helper_events": [
                    dict(row) for row in active_scheduler.request_helper_events()
                ],
                "schema": "s42-offline-phone-residency-gate-failure-v1",
                "status": "FAIL",
                "trace_executed": False,
                "traceback": traceback.format_exc(),
            }
            call_diagnostics = getattr(error, "call_diagnostics", None)
            if isinstance(call_diagnostics, Mapping):
                failure["session_call_diagnostics"] = dict(
                    call_diagnostics
                )
            call_gap = getattr(error, "call_gap_measurement", None)
            if isinstance(call_gap, Mapping):
                failure["session_call_gap_measurement"] = dict(call_gap)
            _write_new(args.output / "FAILURE.json", failure)
        except BaseException:
            pass
        raise
    finally:
        if terminal is None:
            try:
                rig.close(require_phone_execution=primary_error is None)
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                primary_error.add_note(
                    "physical cleanup failed: " + str(cleanup_error)
                )
        else:
            rig.close(require_phone_execution=False)
        for name, rows in (
            ("resource-samples.jsonl", rig.host_samples),
        ):
            path = args.output / name
            if not path.exists():
                with path.open("xb") as stream:
                    for row in rows:
                        stream.write(_canonical(row))
        for name, value in (
            ("host-power-diagnostics.json", rig.host_power_diagnostics),
            ("phone-power-diagnostics.json", rig.phone_power_diagnostics),
            ("ADAPTIVE_DECODE_OBSERVATIONS.json", dict(
                active_scheduler.adaptive_decode_observation_snapshot()
            )),
            ("PHONE_TERMINAL_RECEIPTS.json", {
                "direct_phone_receipts": list(rig.direct_phone_receipts),
                "usb_restore_receipts": list(rig.usb_restore_receipts),
            }),
        ):
            path = args.output / name
            if not path.exists():
                path.write_bytes(_canonical(value))


def parse_args() -> argparse.Namespace:
    return runner._build_parser().parse_args()


def main() -> int:
    args = parse_args()
    result = run_gate(args)
    print(json.dumps({
        "schema": result["schema"],
        "status": result["status"],
        "trace_executed": result["trace_executed"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
