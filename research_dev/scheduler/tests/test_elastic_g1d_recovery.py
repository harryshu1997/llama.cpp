#!/usr/bin/env python3
"""Elastic phones, G1d: a helper loss on a LIVE co-tenant desktop server.

Hardware run G1d (two-phone eval_v2 arm, ``elastic_phones {drop_recovery, join}``): the Pixel worker
was SIGTERMed while three requests decoded on the Qwen desktop parent ``physical:hot:desktop``. The
classifier named the Pixel (helper_lost) for all three, but the scheduler found no fallback and the
arrival coordinator aborted the run. Root cause: llama-server does NOT exit on a helper loss --
``update_slots`` catches the aborted decode, errors every processing slot and keeps serving
(tools/server/server-context.cpp), and ``S41SERVERFFNERROR`` is printed only at shutdown
(tools/server/server.cpp ``finish``) -- while the Pixel's FFN client stays latched failed
(examples/layersplit/ffn-split-client.cpp ``set_error``). The rig therefore never reaped the server,
the desktop route stayed hot, and the recovery target of a request that ran ON that route was the
failed route itself (``_select_*_recovery``): no candidate. The FAILED rows' ``candidates: []`` and
original ``decision_reason`` are how every FAILED row is logged, not the recovery's view.

Recorded tests, no hardware: the synthetic two-phone catalog of two_phone_harness (primary "op15",
co-helper "pixel") with the rig's PCIe links at the rig's capacity, so three requests share the
desktop server as on hardware.
"""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from research_dev.scheduler import (
    RuntimeCapabilityCatalog,
    UnifiedScheduleError,
    UnifiedScheduler,
    elastic_phones_configuration,
)
from research_dev.scheduler._internal.runtime_execution import RuntimeExecutionFailure
from research_dev.scheduler.adapters import (
    CanonicalArrivalCoordinator,
    CanonicalRuntimeSubmission,
    PhysicalAdapterError,
)
from research_dev.scheduler.adapters import heterogeneous_rig as rig_module
from research_dev.scheduler.adapters.activity import RuntimeActivityTracker
from research_dev.scheduler.adapters.contracts import (
    CompletionStreamError,
    LlamaServerExitedError,
    PhysicalHelperLostError,
    RawEnergyMeasurement,
    RawExecutionObservation,
    RawTransitionObservation,
)
from research_dev.scheduler.adapters.heterogeneous_rig import (
    HeterogeneousPhysicalRig,
    _LiveExecutorResidency,
)
from research_dev.scheduler.adapters.http_backend import (
    CanonicalHttpExecutionBackend,
    LlamaCppCompletionPayload,
    LlamaCppHttpClient,
)
from research_dev.scheduler.adapters.llama_server import ManagedLlamaServer
from research_dev.scheduler.adapters.probes import PhoneRuntimeProbe
from research_dev.scheduler.adapters.snapshot import (
    EndpointRuntimeSample,
    UnifiedRuntimeSnapshotBuilder,
)

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
import two_phone_harness as h  # noqa: E402

ELASTIC = elastic_phones_configuration({"drop_recovery": True, "join": True})
DESKTOP = "physical:two:desktop"
# The launch environment of the desktop parent: its dormant FFN runtime connects both phones.
TWO_HELPERS = {
    "S41_SERVER_FFN_HELPERS": "2",
    "S41_SERVER_FFN_HELPER0_LABEL": "op15",
    "S41_SERVER_FFN_HELPER0_TRANSPORT": "functionfs-usb",
    "S41_SERVER_FFN_HELPER1_LABEL": "pixel",
    "S41_SERVER_FFN_HELPER1_TRANSPORT": "tcp",
}
CO_TENANTS = 3
STREAMED_BEFORE_LOSS = 3


def _sse(tokens, predicted, *, stop=False):
    body = {"content": "x" * len(tokens), "id_slot": 0, "stop": stop, "tokens": tokens,
            "tokens_predicted": predicted}
    return ("data: " + json.dumps(body, sort_keys=True) + "\n\n").encode("ascii")


class LiveCoTenantServer:
    """A managed desktop llama-server with two FFN helpers, as the C++ server behaves: a helper
    loss aborts the decode, the server logs ``decode() failed`` and errors every processing slot,
    and KEEPS RUNNING (no exit, no shutdown line). ``stop`` (SIGTERM) ends it; its shutdown prints
    the failed helper's ``S41SERVERFFNERROR`` line (``finish``)."""

    def __init__(self, generation: int, *, stop_s: float = 0.0) -> None:
        self.process = SimpleNamespace(pid=5000 + generation, poll=self.exit_code)
        self.stop_s = stop_s
        self.on_stopped = None
        self.environment = dict(TWO_HELPERS)
        self.generation = generation
        self.returncode = None
        self.lines = ["srv  load_model: loaded"]
        self.stop_calls = 0
        self.evidence_calls = []
        self._condition = threading.Condition()

    def exit_code(self):
        with self._condition:
            return self.returncode

    def lose_helper(self) -> None:
        with self._condition:
            self.lines += [
                "srv          decode: Compute aborted. off = 0, n_batch = 3, ret = 2",
                "srv    update_slots: decode() failed: Compute aborted.",
            ]
            self._condition.notify_all()

    def failure_evidence(self, stderr_index, *, timeout_s, decisive=None):
        started = time.monotonic()
        deadline = started + timeout_s
        with self._condition:
            while True:
                lines = tuple(self.lines[stderr_index:])
                if (
                    self.returncode is not None
                    or (decisive is not None and decisive(lines))
                    or time.monotonic() >= deadline
                ):
                    self.evidence_calls.append(
                        (timeout_s, decisive is not None, time.monotonic() - started)
                    )
                    return lines, self.returncode
                self._condition.wait(min(0.05, max(0.0, deadline - time.monotonic())))

    def stop(self) -> None:
        if self.stop_s:
            time.sleep(self.stop_s)  # the shutdown handler, then CUDA teardown
        with self._condition:
            self.stop_calls += 1
            if self.returncode is None:
                self.lines.append(
                    "S41SERVERFFNERROR helper=pixel detail=FFN split EXECUTE header exchange failed"
                )
                self.returncode = 0
            self._condition.notify_all()
        if self.on_stopped is not None:
            self.on_stopped()


def co_tenant_catalog(model):
    """The two-phone catalog with the rig's PCIe links (``campaigns/burstgpt/configs/v7/rig.json``
    gives them capacity 8): three requests run on the desktop parent at once, as on hardware."""
    value = h.runtime_catalog(model, declaration=h.co_helpers())
    resources = dict(value.resources)
    for resource_id in ("link:pcie-out", "link:pcie-in"):
        resources[resource_id] = replace(resources[resource_id], capacity=8)
    return RuntimeCapabilityCatalog.from_json(replace(value, resources=resources).to_json())


def hot_snapshot(model, catalog):
    return h.with_residency_executor(h.snapshot(model, catalog, desktop_hot=True), DESKTOP)


def reaped_snapshot(hot, *, gpu_free_bytes=None):
    """The rig's view once the desktop server left the residency map: cold and not ready."""
    executors = dict(hot.executors)
    executors[DESKTOP] = replace(executors[DESKTOP], ready=False, free_slots=0)
    value = replace(hot, executors=executors, residency=tuple(
        row for row in hot.residency if row.executor_id != DESKTOP
    ))
    if gpu_free_bytes is None:
        return value
    capacities = dict(value.memory.capacities)
    vram = capacities["cuda-vram"]
    capacities["cuda-vram"] = replace(
        vram, occupied_bytes=vram.capacity_bytes - vram.reserve_bytes - gpu_free_bytes
    )
    return replace(value, memory=replace(value.memory, capacities=capacities))


def partial_rig(root: Path, model, epoch_ns: int) -> HeterogeneousPhysicalRig:
    rig = HeterogeneousPhysicalRig.__new__(HeterogeneousPhysicalRig)
    rig.configuration = SimpleNamespace(
        output_directory=root, phone_device_id=h.OP15,
        resident_executor_id="physical:resident", elastic_phones=ELASTIC,
    )
    rig.epoch_ns = epoch_ns
    rig._lock, rig._transition_active, rig._scheduler = threading.RLock(), False, None
    rig._live_executors, rig._execution_markers, rig._active_large = {}, {}, {}
    rig._dormant_forget_server = mock.Mock()
    rig._co_helper_lifecycles = {model.artifact_sha256: SimpleNamespace(declaration=h.co_helpers())}
    # the membership probe of the Pixel: its worker is gone
    rig._co_helper_sessions = {h.PIXEL: SimpleNamespace(alive=lambda: False)}
    return rig


def publish(rig, model, server, plan=None) -> None:
    """The residency map entry of the desktop server (``plan``: the execution plan it serves)."""
    rig._live_executors[DESKTOP] = _LiveExecutorResidency(
        executor_id=DESKTOP, endpoint="http://127.0.0.1:18571", server=server, manifest=model,
        parameters={"parallel": 4} if plan is None else dict(plan["adapter_parameters"]),
        operator_plan={} if plan is None else plan, generation=server.generation,
        participant_device_ids=(h.CPU, h.GPU), replacement_resource_ids=(), session_resource_ids=(),
    )


def desktop_plan(model, catalog) -> dict[str, object]:
    scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
    scheduler.register_runtime_capabilities(catalog)
    scheduler.register_model_manifest(model)
    ticket = scheduler.submit_automated_request(
        h.request("plan"), model.model_id, hot_snapshot(model, catalog))
    assert ticket.decision.route_id == "auto:coordinated:" + DESKTOP + ":residency:hot"
    return ticket.execution_plan.to_json()


def with_dormant_parent(model, catalog, hot):
    """``hot`` whose desktop parent residency carries the dormant FFN runtime of an energy-aware
    launch (the rig reports the live server's launch parameters)."""
    probe = UnifiedScheduler.for_runtime_discovery("enforce")
    probe.register_runtime_capabilities(catalog)
    probe.register_model_manifest(model)
    parent = probe.submit_automated_request(
        h.request("parent"), model.model_id, h.snapshot(model, catalog),
        selection_mode="energy-aware").execution_plan
    return replace(hot, residency=tuple(
        replace(row, resident_adapter_parameters=dict(parent.adapter_parameters))
        if row.device_id in {h.CPU, h.GPU} else row for row in hot.residency))


def stamped(value, captured_at_us: int, validity_us: int):
    """A snapshot captured at ``captured_at_us`` and valid for ``validity_us`` (the rig's clock)."""
    until = captured_at_us + validity_us
    return replace(
        value, snapshot_id=value.snapshot_id + "-" + str(captured_at_us),
        captured_at_us=captured_at_us, valid_until_us=until,
        memory=replace(value.memory, captured_at_us=captured_at_us, valid_until_us=until),
    )


class G1dScenario:
    """Three co-tenant requests stream on the live desktop server; the Pixel dies mid-stream.

    ``validity_us`` makes the snapshot provider behave like the rig's: every snapshot is captured
    when it is taken and valid only ``validity_us`` from then; ``stop_s`` is how long the retired
    server takes to exit after SIGTERM."""

    def __init__(self, case: unittest.TestCase, *, gpu_free_bytes=None, validity_us=None,
                 stop_s: float = 0.0, late: int = 0, racing_start: bool = False,
                 adaptive: bool = False, replans: int = 0) -> None:
        self.late = late
        # adaptive: the desktop parent was launched with its dormant FFN runtime, so every
        # execution on it opens an adaptive session (as the HTTP backend does); replans: each
        # recovery is woken to replan that often before it may be acquired (G1f)
        self.adaptive, self.replans = adaptive, {}
        self.replans_per_recovery = replans
        self.adaptive_starts = {}
        # racing_start: g1d-2 was acquired on the hot route with the others but starts only after
        # the retire (and before the reload), the race of a co-tenant dispatched at the loss
        self.racing_start = racing_start
        self.retired, self.late_started = threading.Event(), threading.Event()
        scope = h.TemporaryModel()
        self.model = scope.__enter__()
        case.addCleanup(scope.__exit__)
        directory = tempfile.TemporaryDirectory()
        case.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.catalog = co_tenant_catalog(self.model)
        self.scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        self.scheduler.register_runtime_capabilities(self.catalog)
        self.scheduler.register_model_manifest(self.model)
        self.hot = hot_snapshot(self.model, self.catalog)
        if adaptive:
            self.hot = with_dormant_parent(self.model, self.catalog, self.hot)
        self.cold = reaped_snapshot(self.hot, gpu_free_bytes=gpu_free_bytes)
        self.epoch_ns = time.monotonic_ns()
        self.rig = partial_rig(self.root, self.model, self.epoch_ns)
        self.rig._failure_evidence_timeout_s = 30.0
        self.server = LiveCoTenantServer(1, stop_s=stop_s)
        self.server.on_stopped = lambda: threading.Timer(0.05, self.retired.set).start()
        publish(self.rig, self.model, self.server)
        self.validity_us = validity_us
        self.snapshots = []
        self.launches, self.transitions, self.recovered_on = [], [], []
        self.in_flight = threading.Barrier(
            CO_TENANTS - int(racing_start), action=self.server.lose_helper, timeout=20)
        self.http = CanonicalHttpExecutionBackend(
            self._client(), SimpleNamespace(measure=lambda *_args: None), epoch_ns=self.epoch_ns,
            on_execution_start=self._begin, on_execution_finish=self._finish,
            failure_classifier=self.rig._classify_execution_failure,
        )

    def now_us(self) -> int:
        return (time.monotonic_ns() - self.epoch_ns) // 1_000

    def provider(self, _ticket, _at_us):
        with self.rig._lock:
            live = DESKTOP in self.rig._live_executors
        value = self.hot if live else self.cold
        if self.validity_us is not None:
            value = stamped(value, max(_at_us, self.now_us()), self.validity_us)
        self.snapshots.append(value)
        return value

    def _client(self):
        scenario = self

        class StreamingClient(LlamaCppHttpClient):
            """The attempt on the first server streams, waits until all co-tenants stream, then
            gets the server's error chunk: the Pixel's FFN client failed and every slot errored."""

            def complete(self, endpoint, payload, control_check, **_options):
                with payload.stream_path.open("xb") as stream:
                    for index in range(STREAMED_BEFORE_LOSS):
                        stream.write(_sse([100 + index], index + 1))
                payload.on_first_token(time.monotonic_ns())
                if scenario.adaptive:
                    scenario.adaptive_start(payload.request_id)
                scenario.in_flight.wait()
                raise CompletionStreamError(
                    "completion stream chunk is invalid", server_error_message="Compute aborted."
                )

        return StreamingClient()

    def _begin(self, command) -> None:
        """The rig's ``_execution_start``: refuse a retired endpoint, then mark the execution."""
        with self.rig._lock:
            state = self.rig._live_executors.get(command.executor_id)
        if command.request_id == "g1d-2":
            self.late_started.set()
        self.rig._require_unretired_endpoint(command.executor_id, state)
        if state is None:
            raise PhysicalAdapterError("physical execution endpoint differs from the ticket")
        with self.rig._lock:
            server = state.server
            self.rig._execution_markers[command.ticket_id] = (
                server, None, SimpleNamespace(stderr_index=len(server.lines)), (8, 30),
            )

    def _finish(self, command) -> None:
        with self.rig._lock:
            self.rig._execution_markers.pop(command.ticket_id)

    def energy(self, links=()):
        domains = sorted("energy:" + device for device in self.catalog.placement_profile.devices)
        return RawEnergyMeasurement(
            energy_boundary_id=self.catalog.placement_profile.energy_boundary_id,
            fleet_energy_uj_by_domain={domain: 100 for domain in domains},
            transfer_energy_uj_by_link={link: 1 for link in links},
            measurement_evidence_ids=("recorded",), attribution_kind="matched_abba")

    def backend(self):
        scenario = self

        class Backend:
            def bind_scheduler(self, bound):
                scenario.http.bind_scheduler(bound)

            def apply_transition(self, transition, payload, control_check):
                """The rig's ``_begin_transition_execution``: a live matching server is reused,
                otherwise the endpoint is launched and published (generation + 1)."""
                control_check()
                if scenario.racing_start:
                    scenario.late_started.wait(10)
                started = scenario.now_us()
                with scenario.rig._lock:
                    live = scenario.rig._live_executors.get(DESKTOP)
                    if live is None:
                        scenario.launches.append(transition.transition.transition_id)
                        publish(scenario.rig, scenario.model, LiveCoTenantServer(2))
                scenario.transitions.append(transition.transition.transition_id)
                return RawTransitionObservation(
                    started_us=started, finished_us=scenario.now_us() + 1, status="COMPLETED",
                    evicted_artifact_sha256s=())

            def execute(self, command, payload, control_check):
                with scenario.rig._lock:
                    state = scenario.rig._live_executors.get(command.executor_id)
                if state is None or state.server is scenario.server:
                    # the first server, live or already retired: the HTTP backend and the rig
                    if scenario.racing_start and command.request_id == "g1d-2":
                        scenario.retired.wait(10)
                    return scenario.http.execute(command, payload, control_check)
                server = state.server
                control_check()
                scenario.recovered_on.append((command.request_id, server.generation))
                adaptive = scenario.adaptive and CanonicalHttpExecutionBackend._adaptive_enabled(command)
                if adaptive:
                    directive = scenario.adaptive_start(command.request_id)
                    while directive is not None and directive.target_token_index is not None:
                        directive = scenario.adaptive_window(command.request_id, directive)
                with payload.stream_path.open("xb") as stream:
                    for index in range(payload.output_tokens):
                        stream.write(_sse([200 + index], index + 1))
                    stream.write(_sse([], payload.output_tokens, stop=True))
                if adaptive:
                    scenario.scheduler.complete_adaptive_decode(command.request_id)
                links = (row.removeprefix("link:") for row in command.operator_plan["resource_ids"]
                         if row.startswith("link:"))
                return RawExecutionObservation(
                    started_us=command.planned_start_us, finished_us=command.planned_finish_us,
                    output_sha256="f" * 64, payload={"tokens": [200]}, energy=scenario.energy(links))

        return Backend()

    def adaptive_start(self, request_id):
        from research_dev.scheduler import AdaptiveDecodeConfig

        directive = self.scheduler.start_adaptive_decode(
            request_id, slot_id=0, first_token_index=1, at_us=self.now_us(),
            config=AdaptiveDecodeConfig(**G1fRecoveredAdaptiveAttempt.WINDOW))
        with self.rig._lock:
            self.adaptive_starts.setdefault(request_id, []).append(
                self.scheduler.runtime_ticket(request_id).ticket_id)
        return directive

    def adaptive_window(self, request_id, directive):
        from research_dev.scheduler._internal.adaptive_decode_contracts import (
            AdaptiveDecodeRawWindowObservation,
        )
        boundary = self.scheduler.adaptive_decode_boundary(
            request_id, slot_id=0, token_index=directive.target_token_index, at_us=self.now_us()).boundary
        return self.scheduler.record_adaptive_decode_window(request_id, boundary, AdaptiveDecodeRawWindowObservation(
            fleet_energy_uj_by_domain={"fleet": 200}, phone_compute_us=0, usb_transfer_us=0, rpc_us=0,
            exposed_tail_us=0, output_valid=True, evidence_ids=("recorded-window",),
            energy_boundary_id=self.catalog.placement_profile.energy_boundary_id,
            energy_attribution_kind="isolated"))

    def payload(self, index: int):
        return LlamaCppCompletionPayload(
            request_id=f"g1d-{index}", expected_model_alias="two-phone", input_tokens=8,
            output_tokens=30, prompt_tokens=tuple(range(8)), seed=0,
            stream_path=self.root / f"request-00{index}.raw", on_first_token=lambda _ns: None)

    def run(self):
        if self.replans_per_recovery:
            return self._run_with_replans()
        return self._run()

    def _run_with_replans(self):
        """A recovery that waits for the reload (its plan loads nothing itself) is woken to replan
        ``replans_per_recovery`` times while it is queued (another model's reload defers it on
        hardware), then it may be acquired."""
        scheduler, original_wait = self.scheduler, self.scheduler.wait_runtime_request

        def wait(request_id, epoch):
            current = scheduler.runtime_ticket(request_id)
            remaining = self.replans.setdefault(request_id, self.replans_per_recovery)
            if (remaining and current.previous_ticket_id is not None
                    and current.dispatch_state == "QUEUED"
                    and not current.execution_plan.transitions):
                self.replans[request_id] = remaining - 1
                with scheduler._runtime_lock:
                    scheduler._runtime_controller.replan_queued_now(
                        (request_id,), "residency_observation_changed", self.now_us(),
                        cancel_owner=scheduler.cancel,
                        release_memory=scheduler._runtime_memory.release_owner)
            return original_wait(request_id, epoch)

        with mock.patch.object(scheduler, "wait_runtime_request", wait):
            return self._run()

    def _run(self):
        coordinator = CanonicalArrivalCoordinator(
            self.scheduler, self.backend(), epoch_ns=self.epoch_ns,
            snapshot_provider=self.provider, max_workers=CO_TENANTS,
            lease_guard_us=50_000, lease_quantum_us=100_000,
        )
        self.payloads = [self.payload(index) for index in range(CO_TENANTS + self.late)]
        try:
            # the co-tenants arrive together (the scheduler's estimate of a synthetic request is a
            # few milliseconds, far shorter than the recorded stream, so staggered submissions would
            # queue behind each other instead of sharing the server as on hardware); a ``late``
            # request queues behind them on the same hot route (the server's context holds three)
            arrival_us = h.request("g1d-0").arrival_us
            coordinator.wait_for_arrival(arrival_us)
            for index in range(CO_TENANTS + self.late):
                coordinator.submit(CanonicalRuntimeSubmission(
                    h.request(f"g1d-{index}"), self.model.model_id, self.hot, self.payloads[index],
                ), observed_at_us=arrival_us)
            return coordinator.drain(timeout_s=60)
        finally:
            coordinator.close(wait=False)

    def events(self, request_id: str):
        return [row for row in self.scheduler.runtime_decision_log()["records"]
                if row["request_ids"] == [request_id]]


class CoTenantHelperLossTests(unittest.TestCase):
    def test_three_co_tenants_recover_through_one_reload(self) -> None:
        """G1d recorded: the Pixel dies while three requests decode on the live desktop parent.
        Expect: the server is retired once (SERVER_EXITED, cause HELPER_LOST), every request falls
        back to the paired desktop route through ONE reload (the first recovery plans it, the
        others find the relaunched server), REQUEST_RECOVERED x3, DEVICE_QUARANTINED for the Pixel
        once, no coordinator abort."""
        scenario = G1dScenario(self)
        started = time.monotonic()
        completed = scenario.run()
        elapsed = time.monotonic() - started
        request_ids = tuple(f"g1d-{index}" for index in range(CO_TENANTS))

        self.assertEqual(completed.request_ids, request_ids)
        for request_id in request_ids:
            self.assertEqual(scenario.scheduler.runtime_ticket(request_id).dispatch_state, "COMPLETED")
        # the live server was retired exactly once, although three requests classified the loss
        self.assertEqual(scenario.server.stop_calls, 1)
        (exited,) = scenario.rig.server_exit_events
        self.assertEqual(
            {key: exited[key] for key in ("kind", "executor_id", "cause", "failed_device_ids",
                                          "returncode", "generation")},
            {"kind": "SERVER_EXITED", "executor_id": DESKTOP, "cause": "HELPER_LOST",
             "failed_device_ids": [h.PIXEL], "returncode": 0, "generation": 1})
        # the classifier stopped waiting once the server logged that it errored its slots
        self.assertTrue(all(decisive for _timeout, decisive, _waited in scenario.server.evidence_calls))
        self.assertLess(elapsed, scenario.rig._failure_evidence_timeout_s)
        # one reload: the first recovery to run loads the endpoint, the others were replanned onto
        # the relaunched server once its load was published; every recovered attempt ran there
        self.assertEqual(len(scenario.launches), 1)
        self.assertTrue(scenario.launches[0].startswith("load:"))
        self.assertEqual(scenario.transitions, scenario.launches)
        self.assertEqual(sorted(
            len(completed.executions[rid].ticket.execution_plan.transitions) for rid in request_ids
        ), [0, 0, 1])
        self.assertEqual(sorted(scenario.recovered_on), [(rid, 2) for rid in request_ids])
        for index, request_id in enumerate(request_ids):
            result = completed.executions[request_id]
            (event,) = result.recovery_events
            self.assertEqual(
                {key: event[key] for key in ("kind", "failure_kind", "device_id", "executor_id",
                                             "returncode", "tokens_discarded", "attempt_stream")},
                {"kind": "REQUEST_RECOVERED", "failure_kind": "helper_lost", "device_id": h.PIXEL,
                 "executor_id": DESKTOP, "returncode": 0, "tokens_discarded": STREAMED_BEFORE_LOSS,
                 "attempt_stream": f"request-00{index}.raw.attempt1"})
            kinds = [row["event_kind"] for row in scenario.events(request_id)]
            self.assertEqual(kinds[:3], ["DECISION", "ACQUIRED", "FALLBACK"])
            self.assertEqual(kinds[-1], "COMPLETED")
            fallback = next(row for row in scenario.events(request_id) if row["event_kind"] == "FALLBACK")
            self.assertEqual(fallback["decision_reason"], "PAIRED_DESKTOP_RECOVERY")
            self.assertTrue(fallback["candidates"])
            self.assertNotIn(h.PIXEL, result.ticket.execution_plan.device_ids)
            self.assertIn(b"\"stop\": true", scenario.payloads[index].stream_path.read_bytes())
        self.assertEqual(
            [(row["kind"], row["device_id"]) for row in scenario.scheduler.device_membership_events()],
            [("DEVICE_QUARANTINED", h.PIXEL)])
        self.assertIn(h.PIXEL, dict(scenario.scheduler.quarantined_devices()))

    def test_g1e_recovery_uses_a_snapshot_taken_after_a_retire_longer_than_its_validity(self) -> None:
        """G1e: the rig's snapshots are valid 2.5 s from their capture; the retire (server stop and
        NVML wait) outlasts that. The recovery must be planned from a snapshot captured after the
        retire, at that snapshot's capture time, and succeed; before the fix the scheduler judged
        the fresh snapshot at the (earlier) failure time: ``system snapshot is stale``."""
        scenario = G1dScenario(self, validity_us=150_000, stop_s=0.4)
        completed = scenario.run()
        request_ids = tuple(f"g1d-{index}" for index in range(CO_TENANTS))
        self.assertEqual(completed.request_ids, request_ids)
        self.assertEqual(scenario.server.stop_calls, 1)
        self.assertEqual(len(scenario.launches), 1)
        (exited,) = scenario.rig.server_exit_events
        for request_id in request_ids:
            result = completed.executions[request_id]
            self.assertEqual(result.recovery_events[0]["failure_kind"], "helper_lost")
            self.assertEqual(scenario.scheduler.runtime_ticket(request_id).dispatch_state, "COMPLETED")
            fallback = next(row for row in scenario.events(request_id)
                            if row["event_kind"] == "FALLBACK")
            # planned after the retire, from a snapshot that reflects it (cold desktop)
            self.assertGreaterEqual(fallback["event_time_us"], exited["at_us"])
            self.assertEqual(fallback["decision_reason"], "PAIRED_DESKTOP_RECOVERY")
        self.assertIsNone(scenario.scheduler.automated_recovery_unavailable_reason("g1d-0"))

    def test_an_expired_recovery_snapshot_is_replaced_not_used(self) -> None:
        """The stale-snapshot error cannot end an elastic recovery: a snapshot whose validity ended
        before the failure is refused before the ticket changes, and the adapter plans the
        recovery from a fresh one (bounded); a provider that only returns expired snapshots ends
        the request with that reason (fail closed)."""
        for always_stale in (False, True):
            with self.subTest(always_stale=always_stale):
                scenario = G1dScenario(self, validity_us=2_500_000)
                original = scenario.provider
                served = []

                def provider(ticket, at_us, original=original, served=served,
                             always_stale=always_stale, scenario=scenario):
                    fresh = original(ticket, at_us)
                    if DESKTOP in scenario.rig._live_executors or (served and not always_stale):
                        return fresh
                    served.append(fresh)
                    return stamped(fresh, 1_000, 100)  # captured before the failure, expired

                scenario.provider = provider
                if not always_stale:
                    completed = scenario.run()
                    self.assertEqual(len(served), 1)
                    self.assertEqual(len(completed.request_ids), CO_TENANTS)
                    continue
                with self.assertRaises(PhysicalAdapterError) as caught:
                    scenario.run()
                message = " ".join(str(row) for row in _chain(caught.exception))
                self.assertIn("physical_backend_failed:helper_lost:" + h.PIXEL, message)
                self.assertIn("recovery snapshot", message)
                self.assertIn("SYSTEM_SNAPSHOT_STALE", message)
                self.assertGreater(len(served), 1)

    def test_co_tenants_acquired_before_the_retire_are_recovered_or_replanned(self) -> None:
        """The other consumers of the retire: g1d-2 was acquired on the hot route with the others
        but starts only after the server was retired (before the reload): its start finds the
        endpoint gone, which is a ``server_exited`` before the start (recovered through the
        reload, never a ticket mismatch that aborts the run); a request queued behind the three is
        replanned onto the reloaded server."""
        scenario = G1dScenario(self, validity_us=2_500_000, racing_start=True)
        completed = scenario.run()
        self.assertEqual(len(completed.request_ids), CO_TENANTS)
        kinds = {rid: [row["failure_kind"] for row in completed.executions[rid].recovery_events]
                 for rid in completed.request_ids}
        self.assertEqual(kinds, {"g1d-0": ["helper_lost"], "g1d-1": ["helper_lost"],
                                 "g1d-2": ["server_exited"]})
        (event,) = completed.executions["g1d-2"].recovery_events
        self.assertEqual((event["executor_id"], event["returncode"], event["tokens_discarded"]),
                         (DESKTOP, 0, 0))
        self.assertEqual(len(scenario.rig.server_exit_events), 1)
        self.assertEqual(len(scenario.launches), 1)

        queued = G1dScenario(self, validity_us=2_500_000, late=1)
        completed = queued.run()
        self.assertEqual(len(completed.request_ids), CO_TENANTS + 1)
        self.assertEqual(completed.executions["g1d-3"].recovery_events, ())
        kinds = [row["event_kind"] for row in queued.events("g1d-3")]
        self.assertIn("REPLAN", kinds)
        self.assertEqual(kinds[-1], "COMPLETED")
        self.assertIn(("g1d-3", 2), queued.recovered_on)
        self.assertEqual(len(queued.launches), 1)

    def test_a_live_server_leaves_the_failed_route_as_the_only_recovery(self) -> None:
        """What G1d logged, at the scheduler: the co-tenants fail with helper_lost while their
        executor stays hot (nothing reaped a live server). The desktop recovery target IS the
        failed route, the CPU recovery fallback is not admitted, so no fallback exists; each FAILED
        row carries ``candidates: []`` and the ticket's original decision reason (every FAILED row
        is logged that way). The same failures against the reaped snapshot recover."""
        scenario = G1dScenario(self)
        tickets = [scenario.scheduler.submit_automated_request(
            h.request(f"hot-{index}"), scenario.model.model_id, scenario.hot) for index in range(CO_TENANTS)]
        self.assertEqual({row.decision.start_us for row in tickets}, {tickets[0].decision.start_us})
        epoch_ns = time.monotonic_ns() - tickets[0].decision.start_us * 1_000
        active = [scenario.scheduler.wait_runtime_request(row.request.request_id, epoch_ns)
                  for row in tickets]
        self.assertEqual({row.dispatch_state for row in active}, {"ACQUIRED"})
        for ticket in active:
            request_id = ticket.request.request_id
            recovery = scenario.scheduler.fail_automated_request(
                request_id, failed_at_us=ticket.decision.start_us + 100,
                reason="physical_backend_failed:helper_lost:" + h.PIXEL, snapshot=scenario.hot,
                physical_failure=RuntimeExecutionFailure(
                    "helper_lost", True, True, failed_device_ids=(h.PIXEL,)))
            self.assertIsNone(recovery.fallback)
            (failed,) = [row for row in scenario.events(request_id) if row["event_kind"] == "FAILED"]
            (decision,) = [row for row in scenario.events(request_id) if row["event_kind"] == "DECISION"]
            self.assertEqual(failed["candidates"], [])
            self.assertEqual(failed["decision_reason"], decision["decision_reason"])
            self.assertTrue(decision["candidates"])
            reason = getattr(scenario.scheduler, "automated_recovery_unavailable_reason", None)
            if callable(reason):
                self.assertIn("failed adaptive route is the paired desktop baseline", reason(request_id))

        reaped = G1dScenario(self)
        tickets = [reaped.scheduler.submit_automated_request(
            h.request(f"cold-{index}"), reaped.model.model_id, reaped.hot) for index in range(CO_TENANTS)]
        epoch_ns = time.monotonic_ns() - tickets[0].decision.start_us * 1_000
        for ticket in tickets:
            active = reaped.scheduler.wait_runtime_request(ticket.request.request_id, epoch_ns)
            recovery = reaped.scheduler.fail_automated_request(
                ticket.request.request_id, failed_at_us=active.decision.start_us + 100,
                reason="physical_backend_failed:helper_lost:" + h.PIXEL, snapshot=reaped.cold,
                physical_failure=RuntimeExecutionFailure(
                    "helper_lost", True, True, failed_device_ids=(h.PIXEL,)))
            self.assertIsNotNone(recovery.fallback)
            self.assertEqual(recovery.fallback.decision.reason, "PAIRED_DESKTOP_RECOVERY")
            self.assertEqual(recovery.fallback.binding.executor_id, DESKTOP)
            self.assertEqual(recovery.fallback.execution_plan.residency_variant, "cold")

    def test_a_recovery_after_the_reload_runs_on_the_relaunched_server(self) -> None:
        """A co-tenant whose recovery is planned after another co-tenant's reload was published
        sees the desktop route hot again: that is a new server (the failed attempt's server
        exited), so it recovers there without a load. Without the exit fact the failed route
        stays excluded (a live server that failed is never the recovery)."""
        scenario = G1dScenario(self)
        tickets = [scenario.scheduler.submit_automated_request(
            h.request(f"late-{index}"), scenario.model.model_id, scenario.hot) for index in range(2)]
        epoch_ns = time.monotonic_ns() - tickets[0].decision.start_us * 1_000
        active = [scenario.scheduler.wait_runtime_request(row.request.request_id, epoch_ns)
                  for row in tickets]
        lost = dict(failed_device_ids=(h.PIXEL,))
        first = scenario.scheduler.fail_automated_request(
            "late-0", failed_at_us=active[0].decision.start_us + 100,
            reason="physical_backend_failed:helper_lost:" + h.PIXEL, snapshot=scenario.cold,
            physical_failure=RuntimeExecutionFailure(
                "helper_lost", True, True, exited_executor_id=DESKTOP, **lost))
        self.assertEqual(len(first.fallback.execution_plan.transitions), 1)
        # the reload of late-0 was published: the desktop route is hot again (generation 2)
        for exited, expected in ((None, None), (DESKTOP, "hot")):
            with self.subTest(exited_executor_id=exited):
                probe = G1dScenario(self) if exited is None else scenario
                if exited is None:
                    ticket = probe.scheduler.submit_automated_request(
                        h.request("late-1"), probe.model.model_id, probe.hot)
                    epoch = time.monotonic_ns() - ticket.decision.start_us * 1_000
                    current = probe.scheduler.wait_runtime_request("late-1", epoch)
                else:
                    current = active[1]
                recovery = probe.scheduler.fail_automated_request(
                    "late-1", failed_at_us=current.decision.start_us + 200,
                    reason="physical_backend_failed:helper_lost:" + h.PIXEL, snapshot=probe.hot,
                    physical_failure=RuntimeExecutionFailure(
                        "helper_lost", True, True, exited_executor_id=exited, **lost))
                if expected is None:
                    self.assertIsNone(recovery.fallback)
                    continue
                self.assertEqual(recovery.fallback.decision.reason, "PAIRED_DESKTOP_RECOVERY")
                self.assertEqual(recovery.fallback.decision.route_id, current.decision.route_id)
                self.assertEqual(recovery.fallback.execution_plan.transitions, ())
        with self.assertRaises(Exception):
            RuntimeExecutionFailure("execution_control", False, True, exited_executor_id=DESKTOP)

    def test_a_reload_that_cannot_fit_fails_with_its_reason_and_stays_fail_fast(self) -> None:
        """No memory for the reload (e.g. another model's load holds the VRAM): the request fails
        with the scheduler's own reason, not a bare helper_lost, and the run aborts (fail-fast:
        a request without an output would leave the RESULT incomplete)."""
        scenario = G1dScenario(self, gpu_free_bytes=0)
        with self.assertRaises(PhysicalAdapterError) as caught:
            scenario.run()
        message = "".join(str(row) for row in _chain(caught.exception))
        self.assertIn("physical_backend_failed:helper_lost:" + h.PIXEL, message)
        self.assertIn("recovery unavailable", message)
        self.assertIn("paired desktop recovery is not memory safe", message)
        self.assertEqual(scenario.server.stop_calls, 1)
        self.assertEqual(scenario.launches, [])


class RigRetireTests(unittest.TestCase):
    """The rig's side: a live server whose own helper was lost is retired exactly once."""

    def setUp(self) -> None:
        scope = h.TemporaryModel()
        self.model = scope.__enter__()
        self.addCleanup(scope.__exit__)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.rig = partial_rig(Path(directory.name), self.model, time.monotonic_ns())
        self.rig._failure_evidence_timeout_s = 5.0
        self.rig.configuration.catalog = SimpleNamespace(
            executors=(), transitions=(SimpleNamespace(executor_id=DESKTOP),),
            composite_executors=(SimpleNamespace(executor_id=DESKTOP),),
        )
        self.server = LiveCoTenantServer(1)
        publish(self.rig, self.model, self.server)

    def mark(self, ticket_id: str) -> SimpleNamespace:
        self.rig._execution_markers[ticket_id] = (
            self.server, None, SimpleNamespace(stderr_index=len(self.server.lines)), (8, 30),
        )
        return SimpleNamespace(ticket_id=ticket_id, executor_id=DESKTOP)

    @staticmethod
    def cut() -> CompletionStreamError:
        return CompletionStreamError(
            "completion stream chunk is invalid", server_error_message="Compute aborted."
        )

    def test_co_tenants_of_a_live_server_retire_it_once(self) -> None:
        commands = [self.mark(f"t{index}:attempt:0") for index in range(CO_TENANTS)]
        self.server.lose_helper()
        results = [None] * CO_TENANTS
        threads = [threading.Thread(target=lambda index=index: results.__setitem__(
            index, self.rig._classify_execution_failure(commands[index], self.cut())))
            for index in range(CO_TENANTS)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        self.assertEqual(
            {(row.phase, row.failed_device_ids, row.executor_id, row.returncode) for row in results},
            {("helper_lost", (h.PIXEL,), DESKTOP, 0)})
        self.assertEqual(self.server.stop_calls, 1)
        self.assertEqual(self.rig._live_executors, {})
        self.rig._dormant_forget_server.assert_called_once_with("http://127.0.0.1:18571")
        (event,) = self.rig.server_exit_events
        self.assertEqual((event["cause"], event["failed_device_ids"], event["returncode"]),
                         ("HELPER_LOST", [h.PIXEL], 0))
        self.assertEqual(getattr(self.rig, "_retiring_executors", {}), {})

    def test_a_retiring_server_serves_nothing_and_no_reaper_takes_it(self) -> None:
        command = self.mark("t0:attempt:0")
        co_tenant_command = self.mark("t1:attempt:0")
        self.server.lose_helper()
        release, stopping = threading.Event(), threading.Event()
        stop = self.server.stop

        def slow_stop():
            stopping.set()
            stop()  # the process has exited, the retire has not finished yet
            release.wait(10)

        self.server.stop = slow_stop
        classified = []
        thread = threading.Thread(target=lambda: classified.append(
            self.rig._classify_execution_failure(command, self.cut())))
        thread.start()
        self.assertTrue(stopping.wait(10))
        sample = self.rig._executor_samples()[DESKTOP]
        self.assertEqual((sample.health, sample.ready, sample.transition_available),
                         ("unavailable", False, False))
        self.assertIsNone(self.rig.reap_exited_executor(DESKTOP, expected_server=self.server))
        self.assertIn(DESKTOP, self.rig._live_executors)
        # a co-tenant of the same server waits for the retire instead of stopping it again
        waiter = []
        co_tenant = threading.Thread(target=lambda: waiter.append(self.rig.retire_helper_lost_executor(
            DESKTOP, expected_server=self.server, device_ids=(h.PIXEL,))))
        co_tenant.start()
        # a co-tenant whose failure is classified while the server stops (its exit is already
        # visible): the classification returns only once the retire forgot the server
        seen = []
        classifier = threading.Thread(target=lambda: seen.append((
            self.rig._classify_execution_failure(co_tenant_command, self.cut()),
            DESKTOP in self.rig._live_executors)))
        classifier.start()
        co_tenant.join(0.3)
        classifier.join(0.1)
        self.assertTrue(co_tenant.is_alive())
        self.assertTrue(classifier.is_alive())
        release.set()
        thread.join(10)
        co_tenant.join(10)
        classifier.join(10)
        self.assertEqual(waiter, [0])
        ((late, still_live),) = seen
        self.assertEqual((late.phase, late.failed_device_ids, late.returncode, still_live),
                         ("helper_lost", (h.PIXEL,), 0, False))
        self.assertEqual(classified[0].executor_id, DESKTOP)
        self.assertEqual(self.server.stop_calls, 1)
        self.assertEqual(len(self.rig.server_exit_events), 1)
        self.assertNotIn(DESKTOP, self.rig._live_executors)
        sample = self.rig._executor_samples()[DESKTOP]
        self.assertEqual((sample.ready, sample.transition_available), (False, True))

    def test_a_later_failure_on_a_retired_server_names_the_lost_helper(self) -> None:
        command = self.mark("t0:attempt:0")
        self.server.lose_helper()
        self.rig._classify_execution_failure(command, self.cut())
        # a request that started after the loss: its window holds only the shutdown of the
        # retired server (no helper line of its own), and its stream was cut by the stop
        self.rig._execution_markers["late:attempt:0"] = (
            self.server, None, SimpleNamespace(stderr_index=len(self.server.lines)), (8, 30),
        )
        late = self.rig._classify_execution_failure(
            SimpleNamespace(ticket_id="late:attempt:0", executor_id=DESKTOP),
            ConnectionResetError("connection reset by peer"),
        )
        self.assertEqual((late.phase, late.failed_device_ids, late.returncode),
                         ("helper_lost", (h.PIXEL,), 0))
        self.assertIn("retired", late.evidence)
        self.assertEqual(len(self.rig.server_exit_events), 1)

    def test_a_start_on_the_retired_endpoint_is_a_server_exit_before_the_start(self) -> None:
        command = self.mark("t0:attempt:0")
        self.server.lose_helper()
        self.rig._classify_execution_failure(command, self.cut())
        self.rig._request_shapes = {"late": (8, 30)}
        self.rig.configuration.manifests = {self.model.model_id: self.model}
        late = SimpleNamespace(executor_id=DESKTOP, request_id="late", ticket_id="late:attempt:0",
                               artifact_sha256=self.model.artifact_sha256)
        with self.assertRaises(LlamaServerExitedError) as caught:
            self.rig._execution_start(late)
        self.assertEqual((caught.exception.executor_id, caught.exception.returncode), (DESKTOP, 0))
        classification = self.rig._classify_execution_failure(late, caught.exception)
        self.assertEqual((classification.phase, classification.executor_id, classification.returncode),
                         ("server_exited", DESKTOP, 0))
        self.assertEqual(len(self.rig.server_exit_events), 1)
        # a relaunched endpoint is served as usual; without drop recovery nothing changes
        publish(self.rig, self.model, LiveCoTenantServer(2))
        self.rig._require_unretired_endpoint(DESKTOP, self.rig._live_executors[DESKTOP])
        self.rig.configuration.elastic_phones = None
        self.rig._require_unretired_endpoint(DESKTOP, None)

    def test_a_server_without_the_lost_device_as_helper_is_kept(self) -> None:
        self.server.environment = {"S41_SERVER_FFN_TRANSPORT": "functionfs-usb"}  # op15 only
        command = self.mark("t0:attempt:0")
        try:
            raise PhysicalHelperLostError("worker gone", device_id=h.PIXEL)
        except PhysicalHelperLostError as error:
            classification = self.rig._classify_execution_failure(command, error)
        self.assertEqual((classification.failed_device_ids, classification.executor_id),
                         ((h.PIXEL,), None))
        self.assertEqual(self.server.stop_calls, 0)
        self.assertIn(DESKTOP, self.rig._live_executors)

    def test_a_stop_that_leaves_the_server_running_keeps_todays_failure(self) -> None:
        command = self.mark("t0:attempt:0")
        self.server.lose_helper()
        self.server.stop = mock.Mock()  # SIGTERM ignored: the process never exits
        with self.assertRaisesRegex(PhysicalAdapterError, "retired llama-server did not exit"):
            self.rig._classify_execution_failure(command, self.cut())
        self.assertIn(DESKTOP, self.rig._live_executors)
        self.assertEqual(getattr(self.rig, "_retiring_executors", {}), {})
        self.assertEqual(self.rig.server_exit_events, ())


class EvidenceWaitTests(unittest.TestCase):
    def test_a_live_server_that_errored_its_slots_ends_the_wait(self) -> None:
        """llama-server logs ``decode() failed`` and keeps running: the classifier's wait for its
        exit ends there instead of at the evidence timeout (10 s on hardware)."""
        from research_dev.scheduler.adapters.heterogeneous_rig_ops.observations import (
            _server_aborted_slots,
        )

        with tempfile.TemporaryDirectory() as directory:
            server = object.__new__(ManagedLlamaServer)
            server.command = (
                sys.executable, "-c",
                "import sys, time; print('srv  update_slots: decode() failed: Compute aborted.',"
                " file=sys.stderr, flush=True); time.sleep(60)",
            )
            server.environment = dict(os.environ)
            server.output_directory = Path(directory)
            server.label = "live-llama-server"
            server.process = None
            server.stderr_lines, server.stderr_observed_epoch_us = [], []
            server._stderr_lock = threading.Condition()
            server._stderr_thread = server._stderr_file = server._stdout_file = None
            server.launch_contract = None
            server.start()
            try:
                started = time.monotonic()
                lines, returncode = server.failure_evidence(
                    0, timeout_s=20, decisive=_server_aborted_slots
                )
                self.assertLess(time.monotonic() - started, 10)
                self.assertIsNone(returncode)
                self.assertEqual(lines, ("srv  update_slots: decode() failed: Compute aborted.",))
            finally:
                server.stop()
                server.process.stderr.close()
        self.assertFalse(_server_aborted_slots(("S41SERVERFFNERROR helper=pixel detail=x",)))

    def test_an_unlabeled_abort_rechecks_the_helpers_until_the_deadline(self) -> None:
        """Two helpers, an unlabeled ``Compute aborted``: the Pixel's session may report the loss a
        moment after the server logged it, so liveness is re-checked until the evidence deadline."""
        with h.TemporaryModel() as model, tempfile.TemporaryDirectory() as directory:
            rig = partial_rig(Path(directory), model, time.monotonic_ns())
            rig._failure_evidence_timeout_s = 5.0
            server = LiveCoTenantServer(1)
            publish(rig, model, server)
            checks = []
            rig._co_helper_sessions = {h.PIXEL: SimpleNamespace(
                alive=lambda: checks.append(time.monotonic()) or len(checks) < 3)}
            rig._execution_markers["t0:attempt:0"] = (
                server, None, SimpleNamespace(stderr_index=len(server.lines)), (8, 30),
            )
            server.lose_helper()
            classification = rig._classify_execution_failure(
                SimpleNamespace(ticket_id="t0:attempt:0", executor_id=DESKTOP),
                CompletionStreamError("completion stream chunk is invalid",
                                      server_error_message="Compute aborted."),
            )
        self.assertEqual(len(checks), 3)
        self.assertEqual((classification.phase, classification.failed_device_ids),
                         ("helper_lost", (h.PIXEL,)))
        self.assertEqual(classification.evidence, "compute aborted; helper session not alive")


class _Sampler:
    """The host sampler's latest GPU row (NVML through nvidia-smi, sampled periodically)."""

    def __init__(self, used: int) -> None:
        self.rows = [self.row(used, time.monotonic_ns())]

    @staticmethod
    def row(used: int, sample_t_ns: int) -> dict[str, object]:
        return {"memory_total_bytes": 2_000_000_000, "memory_used_bytes": used,
                "memory_free_bytes": 2_000_000_000 - used, "power_mw": 1000,
                "sample_t_ns": sample_t_ns}

    def latest_gpu(self):
        return dict(self.rows[-1])


class SnapshotAfterReapTests(unittest.TestCase):
    """The rig snapshot the recovery is planned from: cold residency, released VRAM."""

    SERVER_BYTES = 700_000_000

    def setUp(self) -> None:
        scope = h.TemporaryModel()
        self.model = scope.__enter__()
        self.addCleanup(scope.__exit__)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.catalog = co_tenant_catalog(self.model)
        rig = partial_rig(Path(directory.name), self.model, time.monotonic_ns() - 1_000_000_000)
        rig.configuration = SimpleNamespace(
            catalog=self.catalog, manifests={self.model.model_id: self.model},
            phone_device_id=h.OP15, phone_memory_resource_id="op15-ram",
            host_memory_resource_id="host-ram", gpu_memory_resource_id="cuda-vram",
            large_phase_id_by_model={self.model.model_id: 1}, active_device_cost_features={},
            preloaded_model_by_executor={}, transition_phase_id=10,
            resident_executor_id="physical:resident", elastic_phones=ELASTIC,
        )
        rig._co_helper_sessions = {}
        rig._active_large_since_ns, rig._request_shapes, rig._link_bandwidth_samples = {}, {}, {}
        rig._phone_residency = None
        rig._host_probe = SimpleNamespace(sample=lambda: SimpleNamespace(
            memory_total_bytes=4_000_000_000, memory_available_bytes=3_000_000_000,
            cpu_utilization_pct=0, memory_stall_avg10_basis_points=0))
        self.sampler = _Sampler(1_500_000_000)
        rig._sampler = self.sampler
        rig._activity = RuntimeActivityTracker(self.catalog)
        rig._snapshot_builder = UnifiedRuntimeSnapshotBuilder(self.catalog)
        rig._runtime_monitor = SimpleNamespace(
            snapshot=lambda _name: SimpleNamespace(
                value=EndpointRuntimeSample("healthy", "live", 1), error=None, stale=False),
            request_refresh=mock.Mock())
        rig.phone_runtime_observation = lambda: (
            PhoneRuntimeProbe(2_000_000_000, 1_000_000_000, 40_000, 900_000, True, True),
            {"validity": "VALID", "valid": True, "source": "test-phone-runtime",
             "sample_timestamp_ns": time.monotonic_ns(), "age_us": 0, "maximum_age_us": 5_000_000})
        self.rig = rig
        self.server = LiveCoTenantServer(1)
        publish(rig, self.model, self.server, desktop_plan(self.model, self.catalog))
        probe = mock.patch.object(rig_module, "probe_nvidia_process_memory_bytes",
                                  return_value=self.SERVER_BYTES)
        probe.start()
        self.addCleanup(probe.stop)

    def snapshot(self):
        observed_at_us = (time.monotonic_ns() - self.rig.epoch_ns) // 1000
        return self.rig.snapshot(h.request("snap"), self.model.model_id, observed_at_us)

    def desktop_residency(self, snapshot):
        return [row for row in snapshot.residency if row.executor_id == DESKTOP]

    def vram(self, snapshot):
        return snapshot.memory.capacities["cuda-vram"].occupied_bytes

    def retire(self) -> None:
        self.rig.retire_helper_lost_executor(DESKTOP, expected_server=self.server,
                                             device_ids=(h.PIXEL,))

    def test_an_exited_server_is_cold_in_the_snapshot_that_sees_the_exit(self) -> None:
        self.assertTrue(self.desktop_residency(self.snapshot()))
        self.server.returncode = -6  # died while idle, between two snapshots
        self.sampler.rows.append(_Sampler.row(800_000_000, time.monotonic_ns() + 10**9))
        snapshot = self.snapshot()
        self.assertEqual(self.desktop_residency(snapshot), [])
        self.assertFalse(snapshot.executors[DESKTOP].ready)
        self.assertEqual([row["executor_id"] for row in self.rig.server_exit_events], [DESKTOP])

    def test_snapshots_after_a_retire_wait_for_a_gpu_sample_after_the_exit(self) -> None:
        """Co-tenant recoveries take their snapshots concurrently right after the retire; each
        waits for the first NVML row sampled after the exit instead of seeing the VRAM held."""
        self.snapshot()  # the server is live: its NVML bytes are recorded
        self.retire()
        exited_ns = time.monotonic_ns()
        # a row sampled during the shutdown still counts the server
        self.sampler.rows.append(_Sampler.row(1_500_000_000, exited_ns - 1_000_000))

        def sample_later():
            time.sleep(0.4)
            self.sampler.rows.append(
                _Sampler.row(1_500_000_000 - self.SERVER_BYTES, time.monotonic_ns()))

        snapshots = []
        threads = [threading.Thread(target=lambda: snapshots.append(self.snapshot()))
                   for _ in range(CO_TENANTS)]
        for thread in threads:
            thread.start()
        threading.Thread(target=sample_later, daemon=True).start()
        for thread in threads:
            thread.join(10)
        self.assertEqual([self.vram(row) for row in snapshots],
                         [1_500_000_000 - self.SERVER_BYTES] * CO_TENANTS)
        self.assertGreaterEqual(time.monotonic_ns() - exited_ns, 300_000_000)
        self.assertTrue(all(self.desktop_residency(row) == [] for row in snapshots))
        # the sample caught up with the exit: later snapshots neither wait nor credit
        self.assertEqual(getattr(self.rig, "_reaped_gpu_memory", []), [])

    def test_a_lagging_gpu_sample_is_credited_with_the_retired_servers_bytes(self) -> None:
        self.sampler.rows[-1]["sample_t_ns"] = time.monotonic_ns() - 2 * 10**9
        self.snapshot()
        self.rig._reaped_gpu_sample_timeout_s = 0.2
        self.retire()
        # the sampler is stuck on a row taken while the server still held its allocation
        self.assertEqual(self.vram(self.snapshot()), 1_500_000_000 - self.SERVER_BYTES)
        # a row sampled during the shutdown may or may not count the server: used as it is
        self.sampler.rows.append(_Sampler.row(1_500_000_000, time.monotonic_ns()))
        self.assertEqual(self.vram(self.snapshot()), 1_500_000_000)

    def test_without_drop_recovery_the_snapshot_is_unchanged(self) -> None:
        self.rig.configuration.elastic_phones = None
        self.snapshot()
        self.assertFalse(hasattr(self.rig, "_executor_gpu_bytes"))
        self.server.returncode = -6
        # the dead server keeps its hot residency until a transition replaces it (today)
        self.assertTrue(self.desktop_residency(self.snapshot()))
        self.assertEqual(self.rig.server_exit_events, ())


class G1fRecoveredAdaptiveAttempt:
    """G1f recorded: a request streams on a phone-assisted route with an adaptive session
    (``adaptive-split``, or the dormant FFN runtime of the desktop parent), the Pixel is lost
    (helper_lost), the FALLBACK (PAIRED_DESKTOP_RECOVERY) is queued behind another model's reload
    and replanned twice (DESKTOP_BASELINE_CONTROL), then acquired on the desktop parent that the
    other request reloaded WITH its dormant runtime, so the recovered execution opens an adaptive
    session again (as the HTTP backend does for such a command) and runs to completion."""

    WINDOW = dict(minimum_remaining_tokens=4, minimum_window_tokens=2, maximum_window_tokens=16,
                  maximum_probe_tokens=20, maximum_probe_candidates=2, measurement_resolution_us=1,
                  transition_cost_us=1, transition_energy_uj=1, warmup_windows_per_policy=0)

    SPLIT = "physical:two:phone-assisted:operator_split"

    def __init__(self, case: unittest.TestCase, requests, *, replans: int = 2,
                 split_hot: bool = False) -> None:
        from research_dev.scheduler import AdaptiveDecodeConfig
        from research_dev.scheduler.adapters import interpret_runtime_ticket
        from research_dev.scheduler.adapters.llama_server import llama_server_launch_contract
        import test_elastic_join as join_fixture

        fixture = type("Fixture", (join_fixture.HelperLossEndToEndTests,), {"runTest": lambda self: None})()
        fixture.setUp()
        case.addCleanup(fixture.doCleanups)
        self.fixture, self.config = fixture, AdaptiveDecodeConfig(**self.WINDOW)
        self.scheduler, self.catalog, self.model = fixture.scheduler, fixture.catalog, fixture.model
        # split_hot: the two-phone split server is already resident (co-tenants need no load)
        self.cold = (fixture.snapshot if not split_hot else h.with_residency_executor(
            h.snapshot(self.model, self.catalog, desktop_hot=True), self.SPLIT))
        self.tickets = {
            request_id: self.scheduler.submit_automated_request(
                h.request(request_id, output_tokens=30), self.model.model_id, self.cold,
                selection_mode=mode)
            for request_id, mode in requests
        }
        first = next(iter(self.tickets.values()))
        environment = llama_server_launch_contract(interpret_runtime_ticket(first), self.model).ffn_environment
        self.rig, _lifecycle, _worker = fixture._rig(environment)
        self.rig.begin_trace(self.rig.epoch_ns)
        self.epoch_ns = time.monotonic_ns() - first.decision.start_us * 1_000
        # the other request's reload: the desktop parent, hot, launched with its dormant runtime
        probe = UnifiedScheduler.for_runtime_discovery("enforce")
        probe.register_runtime_capabilities(self.catalog)
        probe.register_model_manifest(self.model)
        parent = probe.submit_automated_request(
            h.request("parent"), self.model.model_id, self.cold, selection_mode="energy-aware").execution_plan
        hot = h.with_residency_executor(h.snapshot(self.model, self.catalog, desktop_hot=True), DESKTOP)
        self.hot = replace(hot, residency=tuple(
            replace(row, resident_adapter_parameters=dict(parent.adapter_parameters))
            if row.device_id in {h.CPU, h.GPU} else row for row in hot.residency))
        self.view = self.cold
        self.replans = {request_id: replans for request_id in self.tickets}
        self.log, self.starts = [], {}
        self.lost = threading.Barrier(len(self.tickets), timeout=20)

    def now_us(self) -> int:
        return (time.monotonic_ns() - self.epoch_ns) // 1_000

    def window(self, request_id, directive):
        from research_dev.scheduler._internal.adaptive_decode_contracts import (
            AdaptiveDecodeRawWindowObservation,
        )
        boundary = self.scheduler.adaptive_decode_boundary(
            request_id, slot_id=0, token_index=directive.target_token_index, at_us=self.now_us()).boundary
        return self.scheduler.record_adaptive_decode_window(request_id, boundary, AdaptiveDecodeRawWindowObservation(
            fleet_energy_uj_by_domain={"fleet": 200}, phone_compute_us=0, usb_transfer_us=0, rpc_us=0,
            exposed_tail_us=0, output_valid=True, evidence_ids=("recorded-window",),
            energy_boundary_id=self.catalog.placement_profile.energy_boundary_id,
            energy_attribution_kind="isolated"))

    def start(self, request_id):
        directive = self.scheduler.start_adaptive_decode(
            request_id, slot_id=0, first_token_index=1, at_us=self.now_us(), config=self.config)
        self.starts.setdefault(request_id, []).append(
            self.scheduler.runtime_ticket(request_id).ticket_id)
        return directive

    def http(self):
        scenario = self

        class StreamingClient(LlamaCppHttpClient):
            """The first attempt: tokens, an adaptive session (one window when it can probe),
            then the server's error chunk of the Pixel's failed FFN client."""

            def complete(self, endpoint, payload, control_check, **_options):
                request_id = payload.request_id
                with payload.stream_path.open("xb") as stream:
                    for index in range(3):
                        stream.write(_sse([100 + index], index + 1))
                payload.on_first_token(time.monotonic_ns())
                directive = scenario.start(request_id)
                if directive.target_token_index is not None and scenario.tickets[
                        request_id].execution_plan.execution_contract.execution_mode == "adaptive-split":
                    scenario.window(request_id, directive)
                scenario.scheduler.record_runtime_decode_progress(request_id, token_index=3,
                                                                  at_us=scenario.now_us())
                scenario.lost.wait()
                raise CompletionStreamError("completion stream chunk is invalid",
                                            server_error_message="Compute aborted.")

        def begin(command):
            self.rig._execution_markers[command.ticket_id] = (
                self.rig._server, None, SimpleNamespace(stderr_index=0), (8, 30))

        return CanonicalHttpExecutionBackend(
            StreamingClient(), SimpleNamespace(measure=lambda *_args: None), epoch_ns=self.epoch_ns,
            on_execution_start=begin,
            on_execution_finish=lambda command: self.rig._execution_markers.pop(command.ticket_id, None),
            failure_classifier=self.rig._classify_execution_failure)

    def backend(self):
        scenario, http = self, self.http()
        first_attempts = set()

        class Backend:
            def bind_scheduler(self, bound):
                http.bind_scheduler(bound)

            def apply_transition(self, command, payload, control_check):
                control_check()
                scenario.log.append(("transition", command.transition.transition_id))
                return RawTransitionObservation(
                    started_us=scenario.now_us(), finished_us=scenario.now_us() + 1,
                    status="COMPLETED", evicted_artifact_sha256s=())

            def execute(self, command, payload, control_check):
                if command.request_id not in first_attempts:
                    first_attempts.add(command.request_id)
                    return http.execute(command, payload, control_check)
                control_check()
                adaptive = CanonicalHttpExecutionBackend._adaptive_enabled(command)
                scenario.log.append(("recovered", command.ticket_id, adaptive))
                if adaptive:
                    directive = scenario.start(command.request_id)
                    while directive is not None and directive.target_token_index is not None:
                        directive = scenario.window(command.request_id, directive)
                with payload.stream_path.open("xb") as stream:
                    for index in range(payload.output_tokens):
                        stream.write(_sse([200 + index], index + 1))
                    stream.write(_sse([], payload.output_tokens, stop=True))
                if adaptive:
                    scenario.log.append((
                        "adaptive", command.request_id,
                        scenario.scheduler.complete_adaptive_decode(command.request_id).terminal_status))
                domains = sorted("energy:" + device for device in scenario.catalog.placement_profile.devices)
                return RawExecutionObservation(
                    started_us=command.planned_start_us, finished_us=command.planned_finish_us,
                    output_sha256="f" * 64, payload={"tokens": [200]},
                    energy=RawEnergyMeasurement(
                        energy_boundary_id=scenario.catalog.placement_profile.energy_boundary_id,
                        fleet_energy_uj_by_domain={domain: 100 for domain in domains},
                        transfer_energy_uj_by_link={
                            row.removeprefix("link:"): 1 for row in command.operator_plan["resource_ids"]
                            if row.startswith("link:")},
                        measurement_evidence_ids=("recorded",), attribution_kind="matched_abba"))

        return Backend()

    def run(self):
        from research_dev.scheduler.adapters import CanonicalPhysicalAdapter

        scheduler, original_wait = self.scheduler, self.scheduler.wait_runtime_request

        def wait(request_id, epoch):
            # the reload of the recovery is deferred behind another model: the queued recovery is
            # woken to replan (residency changed) until the other request's reload is published
            current = scheduler.runtime_ticket(request_id)
            if (self.replans[request_id] and current.previous_ticket_id is not None
                    and current.dispatch_state == "QUEUED"):
                self.replans[request_id] -= 1
                if not any(self.replans.values()):
                    self.view = self.hot
                with scheduler._runtime_lock:
                    scheduler._runtime_controller.replan_queued_now(
                        (request_id,), "residency_observation_changed", self.now_us(),
                        cancel_owner=scheduler.cancel,
                        release_memory=scheduler._runtime_memory.release_owner)
            return original_wait(request_id, epoch)

        backend, results, errors = self.backend(), {}, {}

        def lifecycle(request_id, ticket):
            payload = LlamaCppCompletionPayload(
                request_id=request_id, expected_model_alias="two-phone", input_tokens=8,
                output_tokens=30, prompt_tokens=tuple(range(8)), seed=0,
                stream_path=self.fixture.root / (request_id + ".raw"), on_first_token=lambda _ns: None)
            try:
                results[request_id] = CanonicalPhysicalAdapter(
                    scheduler, backend, epoch_ns=self.epoch_ns,
                    snapshot_provider=lambda _ticket, _at_us: self.view,
                    lease_guard_us=50_000, lease_quantum_us=100_000,
                ).execute(ticket, payload)
            except BaseException as error:  # noqa: BLE001 - reported by the test
                errors[request_id] = error

        with mock.patch.object(scheduler, "wait_runtime_request", wait):
            threads = [threading.Thread(target=lifecycle, args=row) for row in self.tickets.items()]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(60)
        return results, errors

    def events(self, request_id):
        return [(row["event_kind"], row["decision_reason"])
                for row in self.scheduler.runtime_decision_log()["records"]
                if row["request_ids"] == [request_id]]


class RecoveredAdaptiveAttemptTests(unittest.TestCase):
    """G1f: a recovered request's new attempt must not find its failed attempt's adaptive
    registration (``adaptive request already exists`` -> physical_execution_control_failed)."""

    G1F_EVENTS = [
        ("FALLBACK", "PAIRED_DESKTOP_RECOVERY"),
        ("REPLAN", "DESKTOP_BASELINE_CONTROL"), ("REPLAN", "DESKTOP_BASELINE_CONTROL"),
        ("ACQUIRED", "DESKTOP_BASELINE_CONTROL"), ("COMPLETED", "DESKTOP_BASELINE_CONTROL"),
    ]

    def assert_recovered(self, scenario, results, errors, request_id, registration):
        self.assertNotIn(request_id, errors, _chain(errors.get(request_id)))
        result = results[request_id]
        self.assertEqual([row["failure_kind"] for row in result.recovery_events], ["helper_lost"])
        self.assertEqual(scenario.events(request_id)[2:], self.G1F_EVENTS)
        # the recovered attempt ran on the reloaded desktop parent and opened its own session
        attempts = scenario.starts[request_id]
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[1], result.ticket.ticket_id)
        self.assertIn(("adaptive", request_id, "COMPLETED"), scenario.log)
        grouped = scenario.scheduler.adaptive_decode_grouped_observation(request_id)
        self.assertEqual((grouped.ticket_id, grouped.terminal_status), (attempts[1], "COMPLETED"))
        rows = [row for row in scenario.scheduler.adaptive_decode_restarted_attempts()
                if row["request_id"] == request_id]
        self.assertEqual([(row["ticket_id"], row["registration"], row["current_ticket_id"])
                          for row in rows], [(attempts[0], registration, attempts[1])])
        return rows[0]

    def test_g1f_adaptive_split_attempt_recovers_through_replans_on_the_reloaded_parent(self):
        scenario = G1fRecoveredAdaptiveAttempt(self, (("t1", "adaptive-decode"),))
        self.assertEqual(
            scenario.tickets["t1"].execution_plan.execution_contract.execution_mode, "adaptive-split")
        results, errors = scenario.run()
        row = self.assert_recovered(scenario, results, errors, "t1", "COMPLETED")
        # the failed attempt's observation is kept (history), only its registration was released
        self.assertEqual(row["terminal_status"], "FAILED")
        history = scenario.scheduler._adaptive_decode._history
        self.assertEqual(history[row["grouped_observation_sha256"]].ticket_id, row["ticket_id"])
        self.assertEqual(scenario.scheduler.quarantined_devices(), {h.PIXEL: "HELPER_LOST"})

    def test_g1f_co_tenants_on_the_dormant_parent_recover_through_replans(self):
        """G1f co-tenants: three requests decode on the desktop parent whose dormant FFN runtime
        opened an adaptive session for each (execution mode ``desktop``); the Pixel is lost, the
        server retired, every recovery is replanned twice before the reload is acquired, and each
        recovered attempt opens a new session on the reloaded parent and completes."""
        scenario = G1dScenario(self, adaptive=True, replans=2)
        completed = scenario.run()
        request_ids = tuple(f"g1d-{index}" for index in range(CO_TENANTS))
        self.assertEqual(completed.request_ids, request_ids)
        reopened = []
        for request_id in request_ids:
            result = completed.executions[request_id]
            self.assertEqual([row["failure_kind"] for row in result.recovery_events], ["helper_lost"])
            kinds = [row["event_kind"] for row in scenario.events(request_id)]
            self.assertEqual(kinds[:3], ["DECISION", "ACQUIRED", "FALLBACK"])
            self.assertEqual(kinds[-2:], ["ACQUIRED", "COMPLETED"])
            attempts = scenario.adaptive_starts[request_id]
            if not result.ticket.execution_plan.transitions:
                # waited for the reload, replanned twice, ran on the reloaded dormant parent
                self.assertGreaterEqual(kinds.count("REPLAN"), 2)
                self.assertEqual(len(attempts), 2)
                self.assertEqual(attempts[1], result.ticket.ticket_id)
                grouped = scenario.scheduler.adaptive_decode_grouped_observation(request_id)
                self.assertEqual((grouped.ticket_id, grouped.terminal_status),
                                 (attempts[1], "COMPLETED"))
                reopened.append(request_id)
            else:
                # the recovery that loads the parent launches it without the dormant runtime
                self.assertEqual(len(attempts), 1)
        self.assertEqual(len(reopened), CO_TENANTS - 1)
        self.assertEqual(scenario.server.stop_calls, 1)
        self.assertEqual(len(scenario.launches), 1)
        # the failures closed the sessions (they held no window): nothing was left to replace
        self.assertEqual(scenario.scheduler.adaptive_decode_restarted_attempts(), ())

    def test_only_a_failed_earlier_attempt_of_the_same_request_is_replaced(self):
        """A duplicate start of the current attempt and a start of a request without an elastic
        failure still fail; another request's registration is never touched."""
        scenario = G1fRecoveredAdaptiveAttempt(self, (("t1", "adaptive-decode"),), split_hot=True)
        scheduler = scenario.scheduler
        epoch_ns = time.monotonic_ns() - scenario.tickets["t1"].decision.start_us * 1_000
        scheduler.wait_runtime_request("t1", epoch_ns)
        scenario.start("t1")
        with self.assertRaisesRegex(UnifiedScheduleError, "adaptive request already exists"):
            scenario.start("t1")
        adaptive = scheduler._adaptive_decode
        with self.assertRaisesRegex(Exception, "adaptive restarted attempts are invalid"):
            adaptive.release_restarted_attempt("t1", (), "x")
        # the current attempt's registration is not a stale one
        ticket_id = scheduler.runtime_ticket("t1").ticket_id
        self.assertIsNone(adaptive.release_restarted_attempt("t1", ("t1:attempt:9",), "x"))
        self.assertIsNone(adaptive.release_restarted_attempt("other", (ticket_id,), "x"))
        self.assertIn("t1", adaptive._sessions)
        self.assertEqual(scheduler.adaptive_decode_restarted_attempts(), ())


def _chain(error):
    rows, current = [], error
    while current is not None and len(rows) < 16:
        rows.append(current)
        current = current.__cause__ or current.__context__
    return rows


if __name__ == "__main__":
    unittest.main()
