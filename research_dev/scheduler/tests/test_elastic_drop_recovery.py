#!/usr/bin/env python3
"""Elastic phones slice 1 (DROP): recover the request, reload a dead server.

Recorded tests only (no phone, no desktop rig): research_dev/scheduler/campaigns/burstgpt/
reports/20260925-elastic-phones/SPEC.md section 3.
"""

from __future__ import annotations

from dataclasses import replace
import http.server
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
    RuntimeTransitionCapability,
    UnifiedScheduleError,
    UnifiedScheduler,
)
from research_dev.scheduler._internal.adaptive_decode_contracts import AdaptiveDecodeError
from research_dev.scheduler._internal.runtime_controller_ops.completion import (
    _retained_route_kind,
)
from research_dev.scheduler._internal.runtime_execution import (
    RuntimeExecutionCoordinatorError,
    RuntimeExecutionFailure,
)
from research_dev.scheduler._unified.automated_selection_ops.objectives import (
    quarantined_device_ids,
)
from research_dev.scheduler.adapters import (
    CanonicalArrivalCoordinator,
    CanonicalPhysicalAdapter,
    CanonicalRuntimeSubmission,
    HeterogeneousPhysicalRig,
    PhysicalAdapterError,
    PhysicalBackendFailure,
    interpret_runtime_ticket,
)
from research_dev.scheduler.adapters.contracts import (
    CompletionStreamError,
    LlamaServerExitedError,
    PhysicalFailureClassification,
    PhysicalHelperLostError,
    StalePhysicalSlotError,
    elastic_drop_recovery_enabled,
)
from research_dev.scheduler.adapters.heterogeneous_rig import _LiveExecutorResidency
from research_dev.scheduler.adapters.heterogeneous_rig_ops.observations import (
    classify_helper_failure,
    device_id_for_helper_label,
    server_helper_rows,
)
from research_dev.scheduler.adapters.http_backend import (
    CanonicalHttpExecutionBackend,
    LlamaCppCompletionPayload,
    LlamaCppHttpClient,
)
from research_dev.scheduler.adapters.llama_server import ManagedLlamaServer
from research_dev.scheduler.adapters.runtime import move_partial_stream_aside

try:
    from .test_automated_runtime import (
        catalog,
        executor_state,
        request,
        runtime_snapshot,
    )
    from .test_gguf_cost import write_synthetic_gguf
    from .test_physical_adapter import FakeMeasuredBackend, composite_catalog
    # a module, not the TestCase: its tests must not be collected here
    from . import test_adaptive_runtime as adaptive_fixture
except ImportError:
    from test_automated_runtime import (
        catalog,
        executor_state,
        request,
        runtime_snapshot,
    )
    from test_gguf_cost import write_synthetic_gguf
    from test_physical_adapter import FakeMeasuredBackend, composite_catalog
    import test_adaptive_runtime as adaptive_fixture


ARTIFACT = "sha256:" + "a" * 64
TOKENS_BEFORE_LOSS = 3


def sse_chunk(tokens: list[int], predicted: int, *, stop: bool = False) -> bytes:
    body = {
        "content": "x" * len(tokens),
        "id_slot": 0,
        "stop": stop,
        "tokens": tokens,
        "tokens_predicted": predicted,
    }
    return ("data: " + json.dumps(body, sort_keys=True) + "\n\n").encode("ascii")


def helper_lost(command) -> PhysicalBackendFailure:
    return PhysicalBackendFailure(
        "helper_lost: S41SERVERFFNERROR helper=pixel detail=recv failed",
        phase="helper_lost",
        retry_safe=True,
        execution_started=True,
        started_us=command.planned_start_us,
        finished_us=command.planned_start_us + 5_000,
        failed_device_ids=("helper-c",),
    )


def server_exited(command) -> PhysicalBackendFailure:
    return PhysicalBackendFailure(
        "server_exited: llama-server exited before the request started",
        phase="server_exited",
        retry_safe=True,
        execution_started=False,
        started_us=command.planned_start_us,
        finished_us=command.planned_start_us,
        executor_id=command.executor_id,
        returncode=-6,
    )


def mid_stream_control_failure(command) -> BaseException:
    return PhysicalAdapterError("completion final chunk is absent")


class StreamingBackend(FakeMeasuredBackend):
    """Writes each attempt's llama.cpp SSE stream like the HTTP client does ("xb").

    ``failures`` maps a zero-based attempt index to a failure factory; a failing
    attempt that had started streams ``TOKENS_BEFORE_LOSS`` tokens first.
    """

    def __init__(self, failures, *, on_failure=None) -> None:
        super().__init__(5_000)
        self.failures = dict(failures)
        self.on_failure = on_failure
        self.attempts = 0

    def execute(self, command, payload, control_check):
        attempt = self.attempts
        self.attempts += 1
        factory = self.failures.get(attempt)
        if factory is None:
            with payload.stream_path.open("xb") as stream:
                for index in range(payload.output_tokens):
                    stream.write(sse_chunk([200 + index], index + 1))
                stream.write(sse_chunk([], payload.output_tokens, stop=True))
            return super().execute(command, payload, control_check)
        self.execution_commands.append(command)
        control_check()
        failure = factory(command)
        if getattr(failure, "execution_started", True):
            with payload.stream_path.open("xb") as stream:
                for index in range(TOKENS_BEFORE_LOSS):
                    stream.write(sse_chunk([100 + index], index + 1))
        if self.on_failure is not None:
            self.on_failure(failure)
        raise failure


class DropRecoverySchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.path = self.root / "drop-model.gguf"
        write_synthetic_gguf(self.path)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def make_scheduler(self):
        probe = UnifiedScheduler.for_runtime_discovery("enforce")
        probe.register_runtime_capabilities(catalog())
        manifest = probe.register_gguf_model("drop-model", self.path)
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(
            composite_catalog(manifest.artifact_sha256)
        )
        scheduler.register_model_manifest(manifest)
        return scheduler, manifest

    @staticmethod
    def snapshot(manifest):
        value = runtime_snapshot(
            manifest, include_phone=True, phone_bandwidth=8_000_000_000
        )
        return replace(value, executors={
            **value.executors,
            "coordinator:cpu-phone": executor_state("coordinator:cpu-phone"),
        })

    @staticmethod
    def reaped_snapshot(snapshot, executor_id: str, device_id: str):
        """What the rig reports after reaping: cold residency, stopped endpoint."""
        executors = dict(snapshot.executors)
        executors[executor_id] = replace(
            executors[executor_id], ready=False, free_slots=0
        )
        return replace(
            snapshot,
            executors=executors,
            residency=tuple(
                row for row in snapshot.residency if row.device_id != device_id
            ),
        )

    def payload(self, name: str, output_tokens: int = 4):
        return SimpleNamespace(
            stream_path=self.root / (name + ".raw"),
            output_tokens=output_tokens,
            prompt_tokens=(1, 2),
        )

    @staticmethod
    def adapter(scheduler, ticket, backend, provider):
        return CanonicalPhysicalAdapter(
            scheduler,
            backend,
            epoch_ns=time.monotonic_ns() - ticket.decision.start_us * 1_000,
            snapshot_provider=provider,
            lease_guard_us=50_000,
            lease_quantum_us=100_000,
        )

    @staticmethod
    def events(scheduler, request_id: str) -> list[str]:
        return [
            row["event_kind"]
            for row in scheduler.runtime_decision_log()["records"]
            if row["request_ids"] == [request_id]
        ]

    def test_helper_lost_mid_stream_recovers_on_a_desktop_route(self) -> None:
        scheduler, manifest = self.make_scheduler()
        snapshot = self.snapshot(manifest)
        ticket = scheduler.submit_automated_request(
            request("drop-helper"), manifest.model_id, snapshot
        )
        self.assertIn("helper-c", ticket.execution_plan.device_ids)
        failed_route = ticket.decision.route_id
        payload = self.payload("request-000")
        backend = StreamingBackend({0: helper_lost})
        with mock.patch.object(
            scheduler, "quarantine_device", create=True
        ) as quarantine:
            result = self.adapter(
                scheduler, ticket, backend, lambda _t, _at: snapshot
            ).execute(ticket, payload)

        self.assertEqual(backend.execution_commands[0].route_id, failed_route)
        self.assertNotEqual(result.command.route_id, failed_route)
        self.assertNotIn("helper-c", result.ticket.execution_plan.device_ids)
        self.assertEqual(result.ticket.previous_ticket_id, ticket.ticket_id)
        failed_at_us = backend.execution_commands[0].planned_start_us + 5_000
        quarantine.assert_called_once_with(
            "helper-c", reason="HELPER_LOST", at_us=failed_at_us
        )
        # The route is healthy; the device is what failed.
        self.assertFalse(scheduler.runtime_route_is_quarantined(failed_route))
        self.assertEqual(
            scheduler.runtime_controller_snapshot()["quarantined_resources"], []
        )
        (recovery,) = result.recoveries
        self.assertEqual(
            recovery.quarantine_action, "route_retained_helper_device_lost"
        )
        self.assertEqual(
            recovery.reason, "physical_backend_failed:helper_lost:helper-c"
        )
        self.assertEqual(
            self.events(scheduler, "drop-helper"),
            ["DECISION", "ACQUIRED", "FALLBACK", "ACQUIRED", "COMPLETED"],
        )
        # Canonical stream = the recovered attempt; the partial is kept aside.
        canonical = payload.stream_path.read_bytes()
        self.assertIn(b"\"stop\": true", canonical)
        self.assertEqual(canonical.count(b"data: "), 5)
        partial = payload.stream_path.with_name("request-000.raw.attempt1")
        self.assertEqual(partial.read_bytes().count(b"data: "), TOKENS_BEFORE_LOSS)
        (event,) = result.recovery_events
        self.assertEqual(dict(event), {
            "attempt_started_us": backend.execution_commands[0].planned_start_us,
            "attempt_stream": "request-000.raw.attempt1",
            "device_id": "helper-c",
            "executor_id": None,
            "failed_at_us": failed_at_us,
            "failed_device_ids": ["helper-c"],
            "failed_ticket": ticket.ticket_id,
            "failure_kind": "helper_lost",
            "kind": "REQUEST_RECOVERED",
            "new_ticket": result.ticket.ticket_id,
            "penalty_us": 5_000,
            "quarantine_action": "route_retained_helper_device_lost",
            "quarantined_device_ids": ["helper-c"],
            "request": "drop-helper",
            "returncode": None,
            "tokens_discarded": TOKENS_BEFORE_LOSS,
        })

    @unittest.skipUnless(
        callable(getattr(UnifiedScheduler, "quarantine_device", None)),
        "elastic-phones slice 2 quarantine API is not merged",
    )
    def test_helper_lost_quarantines_the_device_through_the_facade(self) -> None:
        scheduler, manifest = self.make_scheduler()
        snapshot = self.snapshot(manifest)
        ticket = scheduler.submit_automated_request(
            request("drop-facade"), manifest.model_id, snapshot
        )
        result = self.adapter(
            scheduler, ticket, StreamingBackend({0: helper_lost}),
            lambda _t, _at: snapshot,
        ).execute(ticket, self.payload("request-020"))
        self.assertNotIn("helper-c", result.ticket.execution_plan.device_ids)
        self.assertIn("helper-c", dict(scheduler.quarantined_devices()))
        self.assertEqual(
            [
                (row["kind"], row["device_id"], row["reason"])
                for row in scheduler.device_membership_events()
            ],
            [("DEVICE_QUARANTINED", "helper-c", "HELPER_LOST")],
        )
        self.assertEqual(
            result.recovery_events[0]["quarantined_device_ids"], ["helper-c"]
        )
        later = scheduler.submit_automated_request(
            request("drop-facade-later", arrival_us=500_000),
            manifest.model_id,
            snapshot,
        )
        self.assertNotIn("helper-c", later.execution_plan.device_ids)

    def test_helper_lost_recovery_does_not_need_the_quarantine_api(self) -> None:
        scheduler, manifest = self.make_scheduler()
        snapshot = self.snapshot(manifest)
        ticket = scheduler.submit_automated_request(
            request("drop-no-api"), manifest.model_id, snapshot
        )
        with mock.patch.object(
            scheduler, "quarantine_device", None, create=True
        ):
            result = self.adapter(
                scheduler, ticket, StreamingBackend({0: helper_lost}),
                lambda _t, _at: snapshot,
            ).execute(ticket, self.payload("request-001"))
        self.assertNotIn("helper-c", result.ticket.execution_plan.device_ids)
        self.assertEqual(result.recovery_events[0]["quarantined_device_ids"], [])

    def test_recovery_never_runs_on_the_lost_device(self) -> None:
        scheduler, manifest = self.make_scheduler()
        snapshot = self.snapshot(manifest)
        ticket = scheduler.submit_automated_request(
            request("drop-lost-host"), manifest.model_id, snapshot
        )

        def host_lost(command):
            return replace_devices(helper_lost(command), ("host-a",))

        with mock.patch.object(
            scheduler, "quarantine_device", create=True
        ), self.assertRaisesRegex(
            PhysicalAdapterError, "physical_backend_failed:helper_lost:host-a"
        ):
            self.adapter(
                scheduler, ticket, StreamingBackend({0: host_lost}),
                lambda _t, _at: snapshot,
            ).execute(ticket, self.payload("request-002"))
        self.assertEqual(
            scheduler.runtime_ticket("drop-lost-host").dispatch_state, "FAILED"
        )

    def test_m2_a_stale_loss_is_recovered_without_a_new_quarantine(self) -> None:
        scheduler, manifest = self.make_scheduler()
        snapshot = self.snapshot(manifest)
        ticket = scheduler.submit_automated_request(
            request("drop-stale"), manifest.model_id, snapshot
        )

        def stale_loss(command):
            failure = helper_lost(command)
            return PhysicalBackendFailure(
                str(failure), phase="helper_lost", retry_safe=True, execution_started=True,
                started_us=failure.started_us, finished_us=failure.finished_us,
                failed_device_ids=("helper-c",), stale_device_ids=("helper-c",),
            )

        with mock.patch.object(
            scheduler, "quarantine_device", create=True
        ) as quarantine:
            result = self.adapter(
                scheduler, ticket, StreamingBackend({0: stale_loss}),
                lambda _t, _at: snapshot,
            ).execute(ticket, self.payload("request-030"))
        quarantine.assert_not_called()
        self.assertNotIn("helper-c", result.ticket.execution_plan.device_ids)
        (event,) = result.recovery_events
        self.assertEqual(
            (event["quarantined_device_ids"], event["stale_device_ids"]), ([], ["helper-c"])
        )
        with self.assertRaisesRegex(PhysicalAdapterError, "stale lost devices"):
            PhysicalBackendFailure(
                "x", phase="helper_lost", retry_safe=True, execution_started=True,
                started_us=0, finished_us=1, failed_device_ids=("helper-c",),
                stale_device_ids=("host-a",),
            )

    def test_quarantined_devices_are_excluded_from_recovery(self) -> None:
        scheduler, manifest = self.make_scheduler()
        snapshot = self.snapshot(manifest)
        ticket = scheduler.submit_automated_request(
            request("drop-quarantined"), manifest.model_id, snapshot
        )
        with mock.patch.object(
            scheduler, "quarantined_devices",
            lambda: frozenset({"host-a"}), create=True,
        ), mock.patch.object(
            scheduler, "quarantine_device", create=True
        ), self.assertRaisesRegex(PhysicalAdapterError, "helper_lost"):
            self.adapter(
                scheduler, ticket, StreamingBackend({0: helper_lost}),
                lambda _t, _at: snapshot,
            ).execute(ticket, self.payload("request-003"))

    def test_server_exit_before_start_reloads_the_reaped_executor(self) -> None:
        scheduler, manifest = self.make_scheduler()
        snapshot = self.snapshot(manifest)
        ticket = scheduler.submit_automated_request(
            request("drop-server"), manifest.model_id, snapshot,
            selection_mode="desktop-baseline",
        )
        self.assertEqual(ticket.binding.executor_id, "executor:host-a")
        self.assertEqual(ticket.execution_plan.transitions, ())
        failed_route = ticket.decision.route_id
        reaped = {"value": False}
        cold = self.reaped_snapshot(snapshot, "executor:host-a", "host-a")

        def provider(_ticket, _at_us):
            return cold if reaped["value"] else snapshot

        def on_failure(_failure):
            # The rig reaps the exited server while classifying the failure.
            reaped["value"] = True

        payload = self.payload("request-004")
        backend = StreamingBackend({0: server_exited}, on_failure=on_failure)
        result = self.adapter(scheduler, ticket, backend, provider).execute(
            ticket, payload
        )

        self.assertEqual(result.command.executor_id, "executor:host-a")
        self.assertNotEqual(result.command.route_id, failed_route)
        self.assertEqual(result.ticket.execution_plan.residency_variant, "cold")
        self.assertEqual(
            [row.transition.transition_id for row in backend.transition_commands],
            ["load:host-a"],
        )
        self.assertFalse(scheduler.runtime_route_is_quarantined(failed_route))
        self.assertEqual(
            result.recoveries[0].quarantine_action,
            "route_retained_executor_reload",
        )
        self.assertEqual(
            self.events(scheduler, "drop-server"),
            ["DECISION", "ACQUIRED", "FALLBACK", "ACQUIRED", "COMPLETED"],
        )
        (event,) = result.recovery_events
        self.assertEqual(event["failure_kind"], "server_exited")
        self.assertEqual(event["executor_id"], "executor:host-a")
        self.assertEqual(event["returncode"], -6)
        self.assertIsNone(event["device_id"])
        self.assertIsNone(event["attempt_stream"])
        self.assertEqual(event["tokens_discarded"], 0)
        self.assertEqual(payload.stream_path.read_bytes().count(b"data: "), 5)

    def test_non_helper_mid_stream_failure_still_fails_fast(self) -> None:
        scheduler, manifest = self.make_scheduler()
        snapshot = self.snapshot(manifest)
        ticket = scheduler.submit_automated_request(
            request("drop-control"), manifest.model_id, snapshot
        )
        payload = self.payload("request-005")
        with self.assertRaisesRegex(
            PhysicalAdapterError, "physical_execution_control_failed"
        ):
            self.adapter(
                scheduler, ticket,
                StreamingBackend({0: mid_stream_control_failure}),
                lambda _t, _at: snapshot,
            ).execute(ticket, payload)
        self.assertEqual(
            scheduler.runtime_ticket("drop-control").dispatch_state, "FAILED"
        )
        # Nothing is renamed without a recovered elastic failure.
        self.assertTrue(payload.stream_path.exists())
        self.assertFalse(
            payload.stream_path.with_name("request-005.raw.attempt1").exists()
        )

    def test_started_server_exit_without_helper_evidence_is_not_restarted(
        self,
    ) -> None:
        scheduler, manifest = self.make_scheduler()
        snapshot = self.snapshot(manifest)
        ticket = scheduler.submit_automated_request(
            request("drop-crash"), manifest.model_id, snapshot
        )

        def crashed(command):
            failure = server_exited(command)
            return PhysicalBackendFailure(
                str(failure),
                phase="server_exited",
                retry_safe=True,
                execution_started=True,
                started_us=failure.started_us,
                finished_us=failure.finished_us + 1,
                executor_id=failure.executor_id,
                returncode=-11,
            )

        with self.assertRaisesRegex(
            PhysicalAdapterError, "physical_backend_failed:server_exited"
        ):
            self.adapter(
                scheduler, ticket, StreamingBackend({0: crashed}),
                lambda _t, _at: snapshot,
            ).execute(ticket, self.payload("request-006"))
        # Not restarted, but the route still is not blamed for the crash.
        self.assertFalse(
            scheduler.runtime_route_is_quarantined(ticket.decision.route_id)
        )

    def test_arrival_coordinator_keeps_running_after_a_helper_loss(self) -> None:
        scheduler, manifest = self.make_scheduler()
        snapshot = self.snapshot(manifest)
        backend = StreamingBackend({0: helper_lost})
        coordinator = CanonicalArrivalCoordinator(
            scheduler,
            backend,
            epoch_ns=time.monotonic_ns(),
            snapshot_provider=lambda _ticket, _at_us: snapshot,
            max_workers=2,
            lease_guard_us=50_000,
            lease_quantum_us=100_000,
        )
        payloads = {}
        try:
            with mock.patch.object(scheduler, "quarantine_device", create=True):
                for index, arrival_us in enumerate((0, 1_000)):
                    request_id = f"elastic-arrival-{index}"
                    payloads[request_id] = self.payload(f"request-01{index}")
                    coordinator.submit_at_arrival(CanonicalRuntimeSubmission(
                        request(request_id, arrival_us=arrival_us),
                        manifest.model_id,
                        snapshot,
                        payloads[request_id],
                    ))
                    if index == 0:
                        coordinator._futures[request_id].result(timeout=30)
                completed = coordinator.drain(timeout_s=30)
        finally:
            coordinator.close(wait=False)
        self.assertEqual(completed.request_ids, (
            "elastic-arrival-0", "elastic-arrival-1"
        ))
        first = completed.executions["elastic-arrival-0"]
        self.assertEqual(first.recovery_events[0]["failure_kind"], "helper_lost")
        self.assertEqual(completed.executions["elastic-arrival-1"].recovery_events, ())
        for request_id in completed.request_ids:
            self.assertEqual(
                scheduler.runtime_ticket(request_id).dispatch_state, "COMPLETED"
            )


def replace_devices(failure: PhysicalBackendFailure, devices) -> PhysicalBackendFailure:
    return PhysicalBackendFailure(
        str(failure),
        phase=failure.phase,
        retry_safe=failure.retry_safe,
        execution_started=failure.execution_started,
        started_us=failure.started_us,
        finished_us=failure.finished_us,
        failed_device_ids=tuple(devices),
    )


class AdaptiveReloadRecoveryTests(unittest.TestCase):
    """helper_lost on an adaptive split whose paired desktop server died with it."""

    def setUp(self) -> None:
        # Composition, not inheritance: the fixture's own tests must not rerun here.
        self.fixture = adaptive_fixture.AdaptiveRuntimeIntegrationTests(
            "test_adaptive_ticket_leases_envelope_and_recovers_to_parent"
        )
        self.fixture.setUp()
        desktop = self.fixture.desktop
        # The physical catalog can relaunch a composite desktop endpoint
        # (heterogeneous_rig transition_executors); the fixture adds that load.
        self.fixture.profile = RuntimeCapabilityCatalog.from_json(replace(
            self.fixture.profile,
            transitions=(
                *self.fixture.profile.transitions,
                RuntimeTransitionCapability(
                    transition_id="load:" + desktop.executor_id,
                    device_id="host-a",
                    source_state="cold",
                    target_state="hot",
                    fixed_latency_us=100,
                    bandwidth_bytes_per_s=1_000_000_000,
                    fixed_energy_uj=100,
                    dynamic_pj_per_byte=100,
                    resource_ids=desktop.resource_ids,
                    maturity="QUALIFIED",
                    evidence_ids=("synthetic-desktop-reload",),
                    executor_id=desktop.executor_id,
                    prepares_device_ids=desktop.participant_device_ids,
                ),
            ),
        ).to_json())
        self.fixture.scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        self.fixture.scheduler.register_runtime_capabilities(self.fixture.profile)
        self.fixture.scheduler.register_model_manifest(self.fixture.manifest)

    def tearDown(self) -> None:
        self.fixture.tearDown()

    def recover(self, request_id: str, snapshot):
        ticket = self.fixture._active_ticket(request_id)
        self.assertEqual(
            ticket.execution_plan.execution_contract.execution_mode,
            "adaptive-split",
        )
        return ticket, self.fixture.scheduler.fail_automated_request(
            request_id,
            failed_at_us=ticket.decision.start_us + 100,
            reason="physical_backend_failed:helper_lost:helper-c",
            snapshot=snapshot,
            physical_failure=RuntimeExecutionFailure(
                "helper_lost", True, True, failed_device_ids=("helper-c",)
            ),
        )

    def test_dead_paired_desktop_is_reloaded_for_the_recovery(self) -> None:
        desktop = self.fixture.desktop
        snapshot = self.fixture.snapshot
        executors = dict(snapshot.executors)
        for executor_id in (desktop.executor_id, self.fixture.phone.executor_id):
            # a reaped endpoint: transition_available keeps it healthy, not ready
            executors[executor_id] = replace(
                executors[executor_id], ready=False, free_slots=0
            )
        reaped = replace(
            snapshot,
            executors=executors,
            residency=tuple(
                row for row in snapshot.residency
                if row.device_id not in desktop.participant_device_ids
            ),
        )
        ticket, recovery = self.recover("adaptive-reload", reaped)
        fallback = recovery.fallback
        self.assertIsNotNone(fallback)
        self.assertEqual(fallback.binding.executor_id, desktop.executor_id)
        self.assertEqual(fallback.decision.reason, "PAIRED_DESKTOP_RECOVERY")
        self.assertEqual(fallback.execution_plan.residency_variant, "cold")
        self.assertEqual(
            [row.transition_id for row in fallback.execution_plan.transitions],
            ["load:" + desktop.executor_id],
        )
        self.assertNotIn("helper-c", fallback.execution_plan.device_ids)
        self.assertEqual(
            recovery.quarantine_action, "route_retained_helper_device_lost"
        )
        self.assertFalse(self.fixture.scheduler.runtime_route_is_quarantined(
            ticket.decision.route_id
        ))

    def test_live_paired_desktop_recovers_without_a_load(self) -> None:
        _ticket, recovery = self.recover("adaptive-hot", self.fixture.snapshot)
        self.assertEqual(recovery.fallback.execution_plan.transitions, ())
        self.assertEqual(
            recovery.fallback.decision.reason, "PAIRED_DESKTOP_RECOVERY"
        )


class FailureContractTests(unittest.TestCase):
    def test_helper_lost_allows_a_restart_after_the_start(self) -> None:
        lost = RuntimeExecutionFailure(
            "helper_lost", True, True, failed_device_ids=("pixel10pro-phone",)
        )
        self.assertTrue(lost.fallback_allowed)
        self.assertEqual(lost.failed_device_id, "pixel10pro-phone")
        self.assertFalse(RuntimeExecutionFailure("request", False, True).fallback_allowed)
        self.assertFalse(RuntimeExecutionFailure("execution_control", True, True).fallback_allowed)
        self.assertFalse(RuntimeExecutionFailure("server_exited", True, True).fallback_allowed)
        self.assertTrue(RuntimeExecutionFailure("server_exited", True, False).fallback_allowed)
        self.assertFalse(
            RuntimeExecutionFailure("helper_lost", False, True, failed_device_ids=("a",)).fallback_allowed
        )
        with self.assertRaises(RuntimeExecutionCoordinatorError):
            RuntimeExecutionFailure("helper_lost", True, True)
        with self.assertRaises(RuntimeExecutionCoordinatorError):
            RuntimeExecutionFailure("connect", True, False, failed_device_ids=("a",))

    def test_backend_failure_requires_its_structured_fields(self) -> None:
        with self.assertRaises(PhysicalAdapterError):
            PhysicalBackendFailure(
                "x", phase="helper_lost", retry_safe=True,
                execution_started=True, started_us=0, finished_us=1,
            )
        with self.assertRaises(PhysicalAdapterError):
            PhysicalBackendFailure(
                "x", phase="server_exited", retry_safe=True,
                execution_started=False, started_us=0, finished_us=1,
            )
        failure = PhysicalBackendFailure(
            "x", phase="helper_lost", retry_safe=True, execution_started=True,
            started_us=0, finished_us=1, failed_device_ids=("b", "a"),
        )
        self.assertEqual(failure.failed_device_ids, ("a", "b"))
        self.assertEqual(failure.failed_device_id, "a")

    def test_only_elastic_reasons_keep_the_route(self) -> None:
        self.assertEqual(
            _retained_route_kind("physical_backend_failed:helper_lost:pixel10pro-phone"),
            "helper_lost",
        )
        self.assertEqual(
            _retained_route_kind("physical_backend_failed:server_exited:physical:desktop"),
            "server_exited",
        )
        for reason in (
            "physical_backend_failed:connect",
            "physical_execution_control_failed",
            "physical_transition_failed:load:helper_lost",
            "synthetic GPU transport failure",
        ):
            self.assertEqual(_retained_route_kind(reason), "physical")

    def test_elastic_configuration_is_opt_in(self) -> None:
        self.assertFalse(elastic_drop_recovery_enabled(None))
        self.assertFalse(elastic_drop_recovery_enabled({"join": True}))
        self.assertTrue(elastic_drop_recovery_enabled({"drop_recovery": True}))
        with self.assertRaises(PhysicalAdapterError):
            elastic_drop_recovery_enabled(True)

    def test_quarantined_devices_reader(self) -> None:
        self.assertEqual(quarantined_device_ids(SimpleNamespace()), frozenset())
        self.assertEqual(
            quarantined_device_ids(SimpleNamespace(
                _runtime_controller=SimpleNamespace(
                    quarantined_devices=frozenset({"pixel10pro-phone"})
                )
            )),
            frozenset({"pixel10pro-phone"}),
        )
        self.assertEqual(
            quarantined_device_ids(SimpleNamespace(
                quarantined_devices=lambda: {"op15-phone": "HELPER_LOST"}
            )),
            frozenset({"op15-phone"}),
        )
        with self.assertRaises(Exception):
            quarantined_device_ids(SimpleNamespace(quarantined_devices="op15"))

    def test_partial_streams_are_numbered_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stream = Path(directory) / "request-007.raw"
            payload = SimpleNamespace(stream_path=stream)
            self.assertEqual(move_partial_stream_aside(payload), (None, 0))
            self.assertEqual(move_partial_stream_aside({}), (None, 0))
            stream.write_bytes(sse_chunk([1, 2], 2) + b"data: {\"tok")
            self.assertEqual(
                move_partial_stream_aside(payload), ("request-007.raw.attempt1", 2)
            )
            stream.write_bytes(b"data: [\"not an object\"]\n\n")
            self.assertEqual(
                move_partial_stream_aside(payload), ("request-007.raw.attempt2", None)
            )
            self.assertFalse(stream.exists())


class HelperClassificationTests(unittest.TestCase):
    @staticmethod
    def rig(*, primary: str = "op15-phone"):
        declaration = SimpleNamespace(
            primary_label="op15",
            helpers=(SimpleNamespace(label="pixel", device_id="pixel10pro-phone"),),
        )
        return SimpleNamespace(
            configuration=SimpleNamespace(phone_device_id=primary),
            _co_helper_lifecycles={ARTIFACT: SimpleNamespace(declaration=declaration)},
        )

    def test_labels_map_to_rig_devices(self) -> None:
        rig = self.rig()
        self.assertEqual(device_id_for_helper_label(rig, None), "op15-phone")
        self.assertEqual(device_id_for_helper_label(rig, "op15"), "op15-phone")
        self.assertEqual(device_id_for_helper_label(rig, "pixel"), "pixel10pro-phone")
        self.assertIsNone(device_id_for_helper_label(rig, "unknown"))
        self.assertEqual(server_helper_rows(rig, {}), ())
        self.assertEqual(
            server_helper_rows(rig, {"S41_SERVER_FFN_TRANSPORT": "functionfs-usb"}),
            ((None, "op15-phone", "functionfs-usb"),),
        )
        self.assertEqual(server_helper_rows(rig, {
            "S41_SERVER_FFN_HELPERS": "2",
            "S41_SERVER_FFN_HELPER0_LABEL": "op15",
            "S41_SERVER_FFN_HELPER0_TRANSPORT": "functionfs-usb",
            "S41_SERVER_FFN_HELPER1_LABEL": "pixel",
            "S41_SERVER_FFN_HELPER1_TRANSPORT": "tcp",
        }), (
            ("op15", "op15-phone", "functionfs-usb"),
            ("pixel", "pixel10pro-phone", "tcp"),
        ))
        with self.assertRaises(PhysicalAdapterError):
            server_helper_rows(rig, {"S41_SERVER_FFN_HELPERS": "2"})

    def test_stderr_and_stream_evidence(self) -> None:
        two = (
            ("op15", "op15-phone", "functionfs-usb"),
            ("pixel", "pixel10pro-phone", "tcp"),
        )
        single = ((None, "op15-phone", "functionfs-usb"),)
        devices, evidence = classify_helper_failure(
            ["srv  update_slots: Compute aborted. off = 0, n_batch = 1, ret = 2",
             "S41SERVERFFNERROR helper=pixel detail=recv failed"],
            "Compute aborted.", two,
        )
        self.assertEqual(devices, ("pixel10pro-phone",))
        self.assertIn("helper=pixel", evidence)
        self.assertEqual(classify_helper_failure(
            [], "helper pixel: FFN runtime policy send failed", two,
        )[0], ("pixel10pro-phone",))
        self.assertEqual(classify_helper_failure(
            ["ffn-split usb: bulk transfer: LIBUSB_ERROR_NO_DEVICE"], None, two,
        )[0], ("op15-phone",))
        # An unlabeled abort names a helper only for a single-helper server.
        self.assertEqual(classify_helper_failure([], "Compute aborted.", two)[0], ())
        self.assertEqual(
            classify_helper_failure([], "Compute aborted.", single)[0], ("op15-phone",)
        )
        self.assertEqual(classify_helper_failure(
            ["S41SERVERFFNERROR detail=usb read failed"], None, single,
        )[0], ("op15-phone",))
        # Unknown labels and servers without helpers never claim a device.
        self.assertEqual(classify_helper_failure(
            ["S41SERVERFFNERROR helper=ghost detail=x"], None, two,
        )[0], ())
        self.assertEqual(classify_helper_failure(
            ["S41SERVERFFNERROR detail=x"], "Compute aborted.", (),
        ), ((), ""))
        # Summary JSON naming a helper is not failure evidence.
        self.assertEqual(classify_helper_failure(
            ['S41SERVERFFNSUMMARY {"helper":"pixel","status":"ok"}'], None, two,
        )[0], ())


class FakeServer:
    """A managed desktop llama-server stand-in with scripted stderr; ``stop`` ends a live one
    like SIGTERM does (llama-server exits 0 after its shutdown handler)."""

    def __init__(self, returncode, lines=(), environment=None) -> None:
        self.process = SimpleNamespace(pid=4242)
        self.returncode = returncode
        self.lines = tuple(lines)
        self.environment = dict(environment or {})
        self.stop = mock.Mock(side_effect=self._terminate)
        self.evidence_calls = []

    def _terminate(self) -> None:
        if self.returncode is None:
            self.returncode = 0

    def exit_code(self):
        return self.returncode

    def failure_evidence(self, stderr_index, *, timeout_s, decisive=None):
        self.evidence_calls.append((stderr_index, timeout_s))
        return self.lines[stderr_index:], self.returncode


class ExitingServer(FakeServer):
    """A server still shutting down when its request fails (G1b): it has exited only for a
    classifier that waits for it."""

    def __init__(self, lines=(), environment=None) -> None:
        super().__init__(None, lines, environment)

    def failure_evidence(self, stderr_index, *, timeout_s, decisive=None):
        if timeout_s > 0:
            self.returncode = 1
        return super().failure_evidence(
            stderr_index, timeout_s=timeout_s, decisive=decisive
        )


class _ControlHandler(http.server.BaseHTTPRequestHandler):
    """The llama-server control endpoint answering every call with ``reply`` (status, JSON)."""

    reply = (200, {"success": False, "message": "request and active slot differ"})

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        status, value = self.reply
        body = json.dumps(value).encode("ascii")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:
        return


def control_failure(reply, call):
    """The error one real ``LlamaCppHttpClient`` control call raises for ``reply``."""
    handler = type("Handler", (_ControlHandler,), {"reply": reply})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        call(f"http://127.0.0.1:{server.server_address[1]}")
    except PhysicalAdapterError as error:
        return error
    finally:
        server.shutdown()
        server.server_close()
    raise AssertionError("the control call succeeded")


def boundary_stats_failure():
    """G1b: the decode boundary's stats call meets the slot the server already released."""
    return control_failure(
        (200, {"success": False, "message": "request and active slot differ"}),
        lambda endpoint: LlamaCppHttpClient.read_ffn_stats(endpoint, "g1b", 0),
    )


def discarded_stale_window_failure():
    """The same stale slot mid-stream: the window discard refuses it and the scheduler's error
    (``adaptive stale window transaction differs``) carries the stale slot as its context."""
    try:
        try:
            raise boundary_stats_failure()
        except StalePhysicalSlotError:
            try:
                raise AdaptiveDecodeError("adaptive stale window transaction differs")
            except AdaptiveDecodeError as exc:
                raise UnifiedScheduleError(str(exc)) from exc
    except UnifiedScheduleError as error:
        return error


class RigDropRecoveryTests(unittest.TestCase):
    TWO_HELPERS = {
        "S41_SERVER_FFN_HELPERS": "2",
        "S41_SERVER_FFN_HELPER0_LABEL": "op15",
        "S41_SERVER_FFN_HELPER0_TRANSPORT": "functionfs-usb",
        "S41_SERVER_FFN_HELPER1_LABEL": "pixel",
        "S41_SERVER_FFN_HELPER1_TRANSPORT": "tcp",
    }

    def rig(self, server, *, elastic=None):
        rig = object.__new__(HeterogeneousPhysicalRig)
        rig._lock = threading.RLock()
        rig.epoch_ns = time.monotonic_ns()
        rig._transition_active = False
        rig._active_large = {}
        rig._execution_markers = {}
        rig._dormant_forget_server = mock.Mock()
        rig._elastic_phones = elastic
        rig._co_helper_lifecycles = {ARTIFACT: SimpleNamespace(declaration=SimpleNamespace(
            primary_label="op15",
            helpers=(SimpleNamespace(label="pixel", device_id="pixel10pro-phone"),),
        ))}
        rig.configuration = SimpleNamespace(
            resident_executor_id="executor:resident",
            phone_device_id="op15-phone",
            catalog=SimpleNamespace(executors=(), transitions=(), composite_executors=()),
        )
        rig._live_executors = {"physical:desktop": _LiveExecutorResidency(
            executor_id="physical:desktop",
            endpoint="http://127.0.0.1:29381",
            server=server,
            manifest=SimpleNamespace(artifact_sha256=ARTIFACT),
            parameters={},
            operator_plan={},
            generation=4,
            participant_device_ids=("desktop-cuda",),
            replacement_resource_ids=(),
            session_resource_ids=(),
            owns_phone_session=False,
        )}
        return rig

    @staticmethod
    def command(ticket_id="drop:attempt:0"):
        return SimpleNamespace(ticket_id=ticket_id, executor_id="physical:desktop")

    def test_exit_before_start_is_reaped_once(self) -> None:
        server = FakeServer(-6)
        rig = self.rig(server, elastic={"drop_recovery": True})
        error = LlamaServerExitedError(
            "llama-server execution endpoint is not active",
            executor_id="physical:desktop", returncode=-6,
        )
        classification = rig._classify_execution_failure(self.command(), error)
        self.assertEqual(classification, PhysicalFailureClassification(
            "server_exited", executor_id="physical:desktop", returncode=-6,
            evidence="llama-server exited before the request started",
        ))
        self.assertEqual(rig._live_executors, {})
        server.stop.assert_called_once_with()
        rig._dormant_forget_server.assert_called_once_with("http://127.0.0.1:29381")
        (event,) = rig.server_exit_events
        self.assertEqual(
            {key: event[key] for key in ("kind", "executor_id", "returncode", "generation")},
            {"kind": "SERVER_EXITED", "executor_id": "physical:desktop",
             "returncode": -6, "generation": 4},
        )
        # A second request on the same dead server yields no second exit row.
        again = rig._classify_execution_failure(self.command("drop-b:attempt:0"), error)
        self.assertEqual(again.phase, "server_exited")
        self.assertEqual(len(rig.server_exit_events), 1)

    def test_co_tenant_requests_share_one_server_exit(self) -> None:
        server = FakeServer(-6, (
            "srv  update_slots: Compute aborted. off = 0, n_batch = 2, ret = 2",
            "S41SERVERFFNERROR helper=pixel detail=recv failed",
        ), self.TWO_HELPERS)
        rig = self.rig(server, elastic={"drop_recovery": True})
        for ticket_id in ("a:attempt:0", "b:attempt:0"):
            rig._execution_markers[ticket_id] = (
                server, None, SimpleNamespace(stderr_index=0), (8, 4)
            )
        error = CompletionStreamError(
            "completion stream chunk is invalid",
            server_error_message="Compute aborted.",
        )
        results = []
        threads = [
            threading.Thread(target=lambda ticket_id=ticket_id: results.append(
                rig._classify_execution_failure(self.command(ticket_id), error)
            ))
            for ticket_id in ("a:attempt:0", "b:attempt:0")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        self.assertEqual(
            sorted((row.phase, row.failed_device_ids) for row in results),
            [("helper_lost", ("pixel10pro-phone",))] * 2,
        )
        self.assertEqual(len(rig.server_exit_events), 1)
        server.stop.assert_called_once_with()

    def test_resident_executor_is_never_reaped(self) -> None:
        rig = self.rig(FakeServer(-6), elastic={"drop_recovery": True})
        error = LlamaServerExitedError(
            "llama-server execution endpoint is not active",
            executor_id="executor:resident", returncode=1,
        )
        self.assertIsNone(rig._classify_execution_failure(self.command(), error))
        # Nor does a resident request that died mid-stream become a reloadable exit.
        server = FakeServer(-6, ("Compute error.",))
        rig._execution_markers["resident:attempt:0"] = (
            server, None, SimpleNamespace(stderr_index=0), (8, 4)
        )
        self.assertIsNone(rig._classify_execution_failure(
            SimpleNamespace(ticket_id="resident:attempt:0", executor_id="executor:resident"),
            PhysicalAdapterError("completion final chunk is absent"),
        ))
        self.assertIn("physical:desktop", rig._live_executors)

    def test_labeled_helper_error_is_helper_lost_and_reaps_the_server(self) -> None:
        server = FakeServer(-6, (
            "earlier request line",
            "srv  update_slots: Compute aborted. off = 0, n_batch = 1, ret = 2",
            "S41SERVERFFNERROR helper=pixel detail=recv failed",
        ), self.TWO_HELPERS)
        rig = self.rig(server, elastic={"drop_recovery": True})
        rig._failure_evidence_timeout_s = 0.25
        rig._execution_markers["drop:attempt:0"] = (
            server, None, SimpleNamespace(stderr_index=1), (8, 4)
        )
        error = CompletionStreamError(
            "completion stream chunk is invalid",
            server_error_message="Compute aborted.",
        )
        classification = rig._classify_execution_failure(self.command(), error)
        self.assertEqual(classification.phase, "helper_lost")
        self.assertEqual(classification.failed_device_ids, ("pixel10pro-phone",))
        self.assertEqual(classification.executor_id, "physical:desktop")
        self.assertEqual(server.evidence_calls, [(1, 0.25)])
        self.assertEqual(rig._live_executors, {})
        self.assertEqual(len(rig.server_exit_events), 1)

    def test_unlabeled_abort_uses_helper_liveness_or_stays_server_exited(self) -> None:
        lines = ("srv  update_slots: Compute aborted. off = 0, n_batch = 1, ret = 2",)
        for alive, expected in ((False, ("helper_lost", ("pixel10pro-phone",))),
                                (None, ("server_exited", ()))):
            with self.subTest(alive=alive):
                server = FakeServer(-6, lines, self.TWO_HELPERS)
                rig = self.rig(server, elastic={"drop_recovery": True})
                if alive is not None:
                    rig._co_helper_sessions = {
                        "pixel10pro-phone": SimpleNamespace(alive=lambda: alive)
                    }
                rig._execution_markers["drop:attempt:0"] = (
                    server, None, SimpleNamespace(stderr_index=0), (8, 4)
                )
                classification = rig._classify_execution_failure(
                    self.command(), PhysicalAdapterError("completion final chunk is absent")
                )
                self.assertEqual(
                    (classification.phase, classification.failed_device_ids), expected
                )

    def test_live_server_non_helper_error_keeps_todays_failure(self) -> None:
        server = FakeServer(None, ("slot released",), self.TWO_HELPERS)
        rig = self.rig(server, elastic={"drop_recovery": True})
        rig._execution_markers["drop:attempt:0"] = (
            server, None, SimpleNamespace(stderr_index=0), (8, 4)
        )
        error = PhysicalAdapterError("completion semantic quality failed: x")
        self.assertIsNone(rig._classify_execution_failure(self.command(), error))
        # Not a server-side symptom: no wait for the server to exit.
        self.assertEqual(server.evidence_calls, [(0, 0.0)])
        self.assertIn("physical:desktop", rig._live_executors)
        # A stream cut inside its last line is a server-side symptom: wait for the exit.
        rig._failure_evidence_timeout_s = 0.5
        truncated = json.JSONDecodeError("Unterminated string", "{\"tok", 1)
        self.assertIsNone(rig._classify_execution_failure(self.command(), truncated))
        self.assertEqual(server.evidence_calls[-1], (0, 0.5))

    def test_transport_error_names_its_device_without_a_marker(self) -> None:
        rig = self.rig(FakeServer(None), elastic={"drop_recovery": True})
        try:
            try:
                raise PhysicalHelperLostError("worker gone", device_id="op15-phone")
            except PhysicalHelperLostError as cause:
                raise PhysicalAdapterError("phone session failed") from cause
        except PhysicalAdapterError as error:
            classification = rig._classify_execution_failure(self.command(), error)
        self.assertEqual(classification.failed_device_ids, ("op15-phone",))

    def test_idle_dead_server_is_reaped_only_with_drop_recovery(self) -> None:
        for elastic, reaped in ((None, False), ({"join": True}, False),
                                ({"drop_recovery": True}, True)):
            with self.subTest(elastic=elastic):
                server = FakeServer(1)
                rig = self.rig(server, elastic=elastic)
                rig._executor_samples()
                self.assertEqual("physical:desktop" not in rig._live_executors, reaped)
                self.assertEqual(len(rig.server_exit_events), int(reaped))
        server = FakeServer(None)
        rig = self.rig(server, elastic={"drop_recovery": True})
        rig._executor_samples()
        self.assertIn("physical:desktop", rig._live_executors)
        # During a transition the transition owns the endpoint set.
        rig = self.rig(FakeServer(1), elastic={"drop_recovery": True})
        rig._transition_active = True
        rig._executor_samples()
        self.assertIn("physical:desktop", rig._live_executors)

    def test_reap_requires_the_same_server(self) -> None:
        server = FakeServer(1)
        rig = self.rig(server, elastic={"drop_recovery": True})
        self.assertIsNone(rig.reap_exited_executor(
            "physical:desktop", expected_server=FakeServer(1)
        ))
        self.assertIsNone(rig.reap_exited_executor("physical:absent"))
        self.assertIsNotNone(rig.reap_exited_executor(
            "physical:desktop", expected_server=server
        ))
        self.assertEqual(rig._live_executors, {})

    def test_m1_a_local_failure_of_a_live_server_reads_no_co_tenant_evidence(self) -> None:
        # the stderr since this request's marker also holds another slot's helper failure
        server = FakeServer(None, (
            "S41SERVERFFNERROR helper=pixel detail=recv failed",
        ), self.TWO_HELPERS)
        rig = self.rig(server, elastic={"drop_recovery": True})
        rig._co_helper_sessions = {
            "pixel10pro-phone": SimpleNamespace(alive=lambda: False)
        }
        rig._execution_markers["drop:attempt:0"] = (
            server, None, SimpleNamespace(stderr_index=0), (8, 4)
        )
        for error in (
            PhysicalAdapterError("runtime lease expired before the next boundary"),
            RuntimeError("control check: request was cancelled"),
        ):
            with self.subTest(error=str(error)):
                self.assertIsNone(rig._classify_execution_failure(self.command(), error))
        self.assertIn("physical:desktop", rig._live_executors)
        # a server-side symptom of this request: the same line names its lost helper
        cut = CompletionStreamError(
            "completion stream chunk is invalid", server_error_message="Compute aborted."
        )
        classification = rig._classify_execution_failure(self.command(), cut)
        self.assertEqual(
            (classification.phase, classification.failed_device_ids),
            ("helper_lost", ("pixel10pro-phone",)),
        )
        # an exited server is read after any failure (its shutdown lines are its own)
        server.returncode = -6
        classification = rig._classify_execution_failure(
            self.command(), PhysicalAdapterError("runtime lease expired before the next boundary")
        )
        self.assertEqual(classification.phase, "helper_lost")
        self.assertEqual(len(rig.server_exit_events), 1)

    def test_m2_a_loss_from_an_attempt_older_than_the_readmission_is_stale(self) -> None:
        from research_dev.scheduler.adapters.heterogeneous_rig import _PhoneMembership

        server = FakeServer(-6, (
            "S41SERVERFFNERROR helper=pixel detail=recv failed",
        ), self.TWO_HELPERS)
        rig = self.rig(server, elastic={"drop_recovery": True, "join": True})
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        rig.configuration.output_directory = Path(directory.name)
        rig._co_helper_sessions = {}
        rig._init_phone_membership(rig._elastic_phones)
        joined_ns = time.monotonic_ns()
        rig._membership["pixel10pro-phone"] = _PhoneMembership(
            "pixel10pro-phone", "MEMBER", "READMITTED", joined_ns,
            readmissions=1, readmitted_ns=joined_ns,
        )
        for ticket_id in ("old:attempt:0", "new:attempt:0"):
            rig._execution_markers[ticket_id] = (
                server, None, SimpleNamespace(stderr_index=0), (8, 4)
            )
        cut = CompletionStreamError(
            "completion stream chunk is invalid", server_error_message="Compute aborted."
        )
        old = rig._classify_execution_failure(
            self.command("old:attempt:0"), cut, started_ns=joined_ns - 10**9
        )
        # still a helper_lost failure (the request is recovered), but it quarantines nothing
        self.assertEqual(old.failed_device_ids, ("pixel10pro-phone",))
        self.assertEqual(old.stale_device_ids, ("pixel10pro-phone",))
        (stale,) = [row for row in rig.helper_membership_events if row["kind"] == "HELPER_LOST_STALE"]
        self.assertEqual(stale["ticket_id"], "old:attempt:0")
        new = rig._classify_execution_failure(
            self.command("new:attempt:0"), cut, started_ns=joined_ns + 1
        )
        self.assertEqual(new.stale_device_ids, ())
        # without the attempt start nothing is stale (fail closed: the loss quarantines)
        self.assertEqual(rig._classify_execution_failure(
            self.command("new:attempt:0"), cut
        ).stale_device_ids, ())

    def test_configuration_field_enables_drop_recovery(self) -> None:
        rig = self.rig(FakeServer(None))
        self.assertFalse(rig._elastic_drop_recovery())
        rig.configuration.elastic_phones = {"drop_recovery": True}
        self.assertTrue(rig._elastic_drop_recovery())

    def test_g1b_decode_boundary_stale_slot_is_a_server_side_helper_loss(self) -> None:
        """G1b: the Pixel's worker died while the request decoded on the two-phone route. The
        adaptive boundary's stats call met the slot the server had released (before the SSE
        error chunk), then the server exited. The stale slot, bare or behind the refused window
        discard, is a server-side symptom: the classifier waits for the exit, reads the helper
        line since the marker and reaps the server."""
        lines = (
            "earlier request line",
            "S41SERVERFFNERROR helper=pixel detail=FFN split EXECUTE header exchange failed",
            "srv  update_slots: decode() failed: Compute aborted.",
            "srv    send_error: task id = 7, error: Compute aborted.",
        )
        for error in (boundary_stats_failure(), discarded_stale_window_failure()):
            with self.subTest(error=type(error).__name__):
                server = ExitingServer(lines, self.TWO_HELPERS)
                rig = self.rig(server, elastic={"drop_recovery": True})
                rig._failure_evidence_timeout_s = 0.25
                rig._execution_markers["drop:attempt:0"] = (
                    server, None, SimpleNamespace(stderr_index=1), (8, 30)
                )
                classification = rig._classify_execution_failure(self.command(), error)
                self.assertEqual(
                    (classification.phase, classification.failed_device_ids,
                     classification.executor_id, classification.returncode),
                    ("helper_lost", ("pixel10pro-phone",), "physical:desktop", 1),
                )
                self.assertIn("helper=pixel", classification.evidence)
                self.assertEqual(server.evidence_calls, [(1, 0.25)])
                self.assertEqual(rig._live_executors, {})
                self.assertEqual(len(rig.server_exit_events), 1)

    def test_a_failed_policy_apply_names_its_helper_from_the_server_text(self) -> None:
        """A runtime policy / context apply the server refused with ``helper <label>: ...`` is
        this request's own server-side evidence, read even before the server exits."""
        error = control_failure(
            (200, {"success": False,
                   "message": "helper pixel: FFN runtime policy send failed"}),
            lambda endpoint: LlamaCppHttpClient.read_ffn_stats(endpoint, "g1b", 0),
        )
        self.assertNotIsInstance(error, StalePhysicalSlotError)
        server = FakeServer(None, ("slot released",), self.TWO_HELPERS)
        rig = self.rig(server, elastic={"drop_recovery": True})
        rig._failure_evidence_timeout_s = 0.05
        rig._execution_markers["drop:attempt:0"] = (
            server, None, SimpleNamespace(stderr_index=0), (8, 30)
        )
        classification = rig._classify_execution_failure(self.command(), error)
        # G1d: the server keeps running after the loss, so it is retired (stopped and reaped)
        self.assertEqual(
            (classification.phase, classification.failed_device_ids,
             classification.executor_id, classification.returncode),
            ("helper_lost", ("pixel10pro-phone",), "physical:desktop", 0),
        )
        self.assertIn("helper pixel:", classification.evidence)
        self.assertEqual(server.evidence_calls, [(0, 0.05)])
        self.assertNotIn("physical:desktop", rig._live_executors)
        server.stop.assert_called_once_with()
        (event,) = rig.server_exit_events
        self.assertEqual((event["cause"], event["failed_device_ids"]),
                         ("HELPER_LOST", ["pixel10pro-phone"]))

    def test_a_genuinely_stale_slot_without_helper_evidence_keeps_todays_failure(self) -> None:
        for error in (boundary_stats_failure(), discarded_stale_window_failure()):
            with self.subTest(error=type(error).__name__):
                # a live server, no helper line since the marker: not a helper loss
                server = FakeServer(None, (
                    "S41SERVERFFNERROR helper=pixel detail=before this request",
                    "slot released",
                ), self.TWO_HELPERS)
                rig = self.rig(server, elastic={"drop_recovery": True})
                rig._failure_evidence_timeout_s = 0.05
                rig._co_helper_sessions = {
                    "pixel10pro-phone": SimpleNamespace(alive=lambda: True)
                }
                rig._execution_markers["drop:attempt:0"] = (
                    server, None, SimpleNamespace(stderr_index=1), (8, 30)
                )
                self.assertIsNone(rig._classify_execution_failure(self.command(), error))
                self.assertEqual(server.evidence_calls, [(1, 0.05)])
                self.assertIn("physical:desktop", rig._live_executors)
                self.assertEqual(rig.server_exit_events, ())
        # a desktop-only server that stays up: no helper to lose
        server = FakeServer(None, ("S41SERVERFFNERROR detail=x",), {})
        rig = self.rig(server, elastic={"drop_recovery": True})
        rig._failure_evidence_timeout_s = 0.05
        rig._execution_markers["drop:attempt:0"] = (
            server, None, SimpleNamespace(stderr_index=0), (8, 30)
        )
        self.assertIsNone(rig._classify_execution_failure(
            self.command(), boundary_stats_failure()
        ))


class RunnerRecoveryRowsTests(unittest.TestCase):
    def test_result_keys_appear_only_with_events(self) -> None:
        from research_dev.scheduler.campaigns.burstgpt import runner

        self.assertEqual(
            runner._elastic_drop_result(SimpleNamespace(), [{"request_id": "r"}]),
            {},
        )
        exited = {"kind": "SERVER_EXITED", "executor_id": "physical:desktop",
                  "returncode": -6, "at_us": 5}
        recovered = {"kind": "REQUEST_RECOVERED", "request": "r"}
        self.assertEqual(
            runner._elastic_drop_result(
                SimpleNamespace(server_exit_events=(exited,)),
                [{"request_recovered": [recovered]}, {"request_id": "s"}],
            ),
            {
                "request_recovered_events": [recovered],
                "server_exited_events": [exited],
            },
        )
        with self.assertRaises(Exception):
            runner._elastic_drop_result(
                SimpleNamespace(server_exit_events=({"kind": "OTHER"},)), []
            )


# The request row keys of runner._request_result before elastic phones (a recovered row adds
# request_recovered only).
BASELINE_REQUEST_ROW_KEYS = frozenset({
    "actual_endpoint", "actual_executor_id", "actual_latency_us", "attempt_ticket_ids",
    "combined_request_index", "completion", "dispatch_receipts", "execution_command", "first_token_ns",
    "fraction_history", "initial_ticket", "input_tokens", "measured_energy", "measured_energy_scope",
    "model_id", "output_quality", "output_sha256", "output_tokens", "physical_execution_proof",
    "prompt_sha256", "recoveries", "replay_arrival_us", "request_id", "scheduling_overhead", "seed",
    "source", "source_arrival_us", "source_slo_us", "terminal_ticket", "trace_arrival_us",
})


class RecoveredRequestTimingTests(unittest.TestCase):
    """M4: a recovered request reports its latency from the first attempt's start to the terminal
    finish and the terminal attempt's first token; the per-attempt numbers stay in the recovery row."""

    EPOCH_NS = 7_000_000_000

    def result(self, *, recovery_events=(), history=()):
        from research_dev.scheduler.campaigns.burstgpt import runner

        request_id = "r"
        observation = SimpleNamespace(started_us=50_000, finished_us=80_000, payload={}, energy=None,
                                      output_sha256="o" * 64)
        binding = SimpleNamespace(endpoint="http://127.0.0.1:1", executor_id="physical:desktop")
        ticket = SimpleNamespace(binding=binding, ticket_id="r:attempt:1", decode_cohort=None,
                                 execution_plan=SimpleNamespace(plan_sha256="p", execution_contract=None))
        command = SimpleNamespace(endpoint=binding.endpoint, executor_id=binding.executor_id,
                                  operator_plan_sha256="p", to_json=lambda: {"command": 1})
        execution = SimpleNamespace(
            command=command, ticket=ticket, observation=observation,
            attempt_ticket_ids=("r:attempt:0", "r:attempt:1"), completion=SimpleNamespace(to_json=lambda: {}),
            dispatch_receipts=(), recoveries=(), recovery_events=tuple(recovery_events))
        terminal = SimpleNamespace(execution_receipt=SimpleNamespace(endpoint=binding.endpoint),
                                   request=SimpleNamespace(input_tokens=8, output_tokens=30), to_json=lambda: {})
        scheduler = SimpleNamespace(runtime_ticket=lambda _request_id: terminal)
        completed = SimpleNamespace(executions={request_id: execution},
                                    tickets={request_id: SimpleNamespace(to_json=lambda: {})})
        item = {"model_id": "m", "source": "trace", "row": {
            "prompt_tokens": [1, 2], "source_arrival_us": 0, "slo_us": 1, "replay_arrival_us": 0}}
        state = runner._ArrivalState(epoch_ns=self.EPOCH_NS)
        state.overheads[request_id] = {}
        for value in history:
            state.first_token_ns.setdefault(request_id, value)
            state.first_token_history.setdefault(request_id, []).append(value)
        with mock.patch.object(runner, "execution_fraction_history", return_value={}), \
                mock.patch.object(runner, "measured_energy", return_value=None):
            return runner._request_result(scheduler, completed, request_id, 0, item, [], state)

    def at_us(self, value_us):
        return self.EPOCH_NS + value_us * 1_000

    def test_a_request_that_was_not_recovered_is_unchanged(self):
        row = self.result(history=(self.at_us(52_000),))
        self.assertEqual(set(row), BASELINE_REQUEST_ROW_KEYS)
        self.assertEqual((row["actual_latency_us"], row["first_token_ns"]), (30_000, self.at_us(52_000)))

    def test_a_recovered_request_counts_from_its_first_attempt(self):
        event = {"kind": "REQUEST_RECOVERED", "attempt_started_us": 10_000, "failed_at_us": 14_000}
        row = self.result(recovery_events=(event,), history=(self.at_us(12_000), self.at_us(55_000)))
        self.assertEqual(set(row), BASELINE_REQUEST_ROW_KEYS | {"request_recovered"})
        # first attempt start -> terminal finish, not the terminal attempt alone
        self.assertEqual(row["actual_latency_us"], 70_000)
        # the terminal attempt's first token, not the discarded attempt's
        self.assertEqual(row["first_token_ns"], self.at_us(55_000))
        (recovered,) = row["request_recovered"]
        self.assertEqual((recovered["attempt_latency_us"], recovered["first_attempt_ttft_us"]), (30_000, 2_000))
        self.assertEqual(recovered["failed_at_us"], 14_000)

    def test_attempts_without_tokens_and_several_failures(self):
        first = {"kind": "REQUEST_RECOVERED", "attempt_started_us": 20_000}
        second = {"kind": "REQUEST_RECOVERED", "attempt_started_us": 30_000}
        # only the second failed attempt streamed a token before failing
        row = self.result(recovery_events=(first, second),
                          history=(self.at_us(31_000), self.at_us(60_000)))
        self.assertEqual((row["actual_latency_us"], row["first_token_ns"]), (60_000, self.at_us(60_000)))
        self.assertEqual({value["first_attempt_ttft_us"] for value in row["request_recovered"]}, {None})
        # a terminal attempt that streamed no token has no first token
        row = self.result(recovery_events=(first,), history=(self.at_us(21_000),))
        self.assertIsNone(row["first_token_ns"])
        self.assertEqual(row["request_recovered"][0]["first_attempt_ttft_us"], 1_000)

    def test_recovery_rows_without_attempt_timing_fail_closed(self):
        with self.assertRaisesRegex(Exception, "recovered request attempt timing"):
            self.result(recovery_events=({"kind": "REQUEST_RECOVERED"},))


class ManagedServerLivenessTests(unittest.TestCase):
    def test_exit_code_and_failure_evidence_of_a_dead_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            server = object.__new__(ManagedLlamaServer)
            server.command = (
                sys.executable, "-c",
                "import sys; print('before', file=sys.stderr);"
                " print('S41SERVERFFNERROR detail=usb gone', file=sys.stderr);"
                " sys.exit(3)",
            )
            server.environment = dict(os.environ)
            server.output_directory = Path(directory)
            server.label = "fake-llama-server"
            server.process = None
            server.stderr_lines = []
            server.stderr_observed_epoch_us = []
            server._stderr_lock = threading.Condition()
            server._stderr_thread = None
            server._stderr_file = None
            server._stdout_file = None
            server.launch_contract = None
            self.assertIsNone(server.exit_code())
            server.start()
            try:
                lines, returncode = server.failure_evidence(1, timeout_s=30)
                self.assertEqual(returncode, 3)
                self.assertEqual(server.exit_code(), 3)
                self.assertEqual(lines, ("S41SERVERFFNERROR detail=usb gone",))
                with mock.patch.multiple(
                    "research_dev.scheduler.adapters.llama_server",
                    validate_physical_execution_command=mock.Mock(),
                    llama_server_launch_contract=mock.Mock(),
                    _launch_contract_supports_execution=mock.Mock(return_value=True),
                ):
                    from research_dev.scheduler.adapters import ticket as ticket_module
                    command = object.__new__(ticket_module.PhysicalExecutionCommand)
                    object.__setattr__(command, "executor_id", "physical:desktop")
                    with self.assertRaisesRegex(
                        PhysicalAdapterError,
                        "^llama-server execution endpoint is not active$",
                    ) as caught:
                        server.begin_execution(command, None)
                self.assertIsInstance(caught.exception, LlamaServerExitedError)
                self.assertEqual(caught.exception.returncode, 3)
                self.assertEqual(caught.exception.executor_id, "physical:desktop")
                with self.assertRaises(PhysicalAdapterError):
                    server.failure_evidence(-1, timeout_s=0)
            finally:
                server.stop()
                server.process.stderr.close()


class _ErrorStreamHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(sse_chunk([7], 1))
        self.wfile.write(
            b'data: {"error": {"code": 500, "message": "Compute aborted.",'
            b' "type": "server_error"}}\n\n'
        )
        self.wfile.flush()

    def log_message(self, *_args) -> None:
        return


class HttpBackendDropRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        path = self.root / "http-model.gguf"
        write_synthetic_gguf(path)
        probe = UnifiedScheduler.for_runtime_discovery("enforce")
        probe.register_runtime_capabilities(catalog())
        manifest = probe.register_gguf_model("http-model", path)
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(
            composite_catalog(manifest.artifact_sha256)
        )
        scheduler.register_model_manifest(manifest)
        snapshot = DropRecoverySchedulerTests.snapshot(manifest)
        ticket = scheduler.submit_automated_request(
            request("http-drop"), manifest.model_id, snapshot
        )
        self.command = interpret_runtime_ticket(ticket)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def payload(self, name: str) -> LlamaCppCompletionPayload:
        return LlamaCppCompletionPayload(
            request_id="http-drop",
            expected_model_alias="http-model",
            input_tokens=2,
            output_tokens=4,
            prompt_tokens=(1, 2),
            seed=0,
            stream_path=self.root / (name + ".raw"),
            on_first_token=lambda _ns: None,
        )

    def backend(self, error, *, classifier=None, start_error=None, calls=None):
        calls = [] if calls is None else calls

        class Client(LlamaCppHttpClient):
            def complete(self, *_args, **_kwargs):
                calls.append("complete")
                raise error

        def on_start(_command):
            calls.append("start")
            if start_error is not None:
                raise start_error

        def classify(command, raised, *, started_ns):
            calls.append("classify")
            self.assertIsInstance(started_ns, int)
            return classifier(command, raised)

        return CanonicalHttpExecutionBackend(
            Client(),
            SimpleNamespace(measure=lambda *_args: None),
            epoch_ns=time.monotonic_ns(),
            on_execution_start=on_start,
            on_execution_finish=lambda _command: calls.append("finish"),
            **({} if classifier is None else {"failure_classifier": classify}),
        )

    def test_stream_error_becomes_helper_lost_before_the_marker_is_dropped(self) -> None:
        error = CompletionStreamError(
            "completion stream chunk is invalid", server_error_message="Compute aborted."
        )
        calls = []
        backend = self.backend(error, calls=calls, classifier=lambda _c, _e: (
            PhysicalFailureClassification(
                "helper_lost", failed_device_ids=("helper-c",),
                evidence="S41SERVERFFNERROR helper=pixel detail=recv failed",
            )
        ))
        with self.assertRaises(PhysicalBackendFailure) as caught:
            backend.execute(self.command, self.payload("h1"), lambda: None)
        failure = caught.exception
        self.assertEqual(failure.phase, "helper_lost")
        self.assertTrue(failure.retry_safe)
        self.assertTrue(failure.execution_started)
        self.assertEqual(failure.failed_device_ids, ("helper-c",))
        self.assertIs(failure.__cause__, error)
        self.assertLessEqual(failure.started_us, failure.finished_us)
        self.assertEqual(calls, ["start", "complete", "classify", "finish"])

    def test_exit_at_begin_execution_becomes_server_exited(self) -> None:
        start_error = LlamaServerExitedError(
            "llama-server execution endpoint is not active",
            executor_id=self.command.executor_id, returncode=-6,
        )
        calls = []
        backend = self.backend(
            AssertionError("not reached"), calls=calls, start_error=start_error,
            classifier=lambda command, _e: PhysicalFailureClassification(
                "server_exited", executor_id=command.executor_id, returncode=-6,
            ),
        )
        with self.assertRaises(PhysicalBackendFailure) as caught:
            backend.execute(self.command, self.payload("h2"), lambda: None)
        self.assertEqual(caught.exception.phase, "server_exited")
        self.assertFalse(caught.exception.execution_started)
        self.assertEqual(caught.exception.executor_id, self.command.executor_id)
        self.assertEqual(calls, ["start", "classify"])

    def test_without_a_classifier_failures_are_unchanged(self) -> None:
        error = CompletionStreamError(
            "completion stream chunk is invalid", server_error_message="Compute aborted."
        )
        with self.assertRaises(CompletionStreamError) as caught:
            self.backend(error).execute(self.command, self.payload("h3"), lambda: None)
        self.assertIs(caught.exception, error)
        start_error = LlamaServerExitedError(
            "llama-server execution endpoint is not active",
            executor_id="x", returncode=1,
        )
        with self.assertRaises(LlamaServerExitedError):
            self.backend(
                AssertionError("not reached"), start_error=start_error
            ).execute(self.command, self.payload("h4"), lambda: None)

    def test_classifier_errors_keep_the_original_failure(self) -> None:
        error = PhysicalAdapterError("completion final chunk is absent")

        def broken(_command, _error, **_options):
            raise ValueError("evidence unreadable")

        with self.assertRaises(PhysicalAdapterError) as caught:
            self.backend(error, classifier=broken).execute(
                self.command, self.payload("h5"), lambda: None
            )
        self.assertIs(caught.exception, error)
        self.assertIn(
            "execution failure classification failed: ValueError: evidence unreadable",
            caught.exception.__notes__,
        )

    def test_http_client_keeps_the_server_stream_error(self) -> None:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ErrorStreamHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = LlamaCppHttpClient(slots_probe=lambda *_args: [])
            payload = self.payload("h6")
            with self.assertRaises(CompletionStreamError) as caught:
                client.complete(
                    f"http://127.0.0.1:{server.server_address[1]}",
                    payload,
                    lambda: None,
                )
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(str(caught.exception), "completion stream chunk is invalid")
        self.assertEqual(caught.exception.server_error_message, "Compute aborted.")
        # The partial stream is on disk for the adapter to move aside.
        self.assertIn(b"Compute aborted.", payload.stream_path.read_bytes())

    def test_failed_control_calls_keep_type_and_message_and_mark_the_server_text(self) -> None:
        """Every FFN control / stats call the server refused keeps its historical type and
        message (flag-absent logs are unchanged) and carries ``server_control_message``."""
        from research_dev.scheduler._internal.adaptive_decode_contracts import (
            AdaptiveDecodeControl,
        )
        from research_dev.scheduler.adapters.heterogeneous_rig_ops.observations import (
            _server_side_failure,
        )

        control = mock.Mock(spec=AdaptiveDecodeControl)
        control.to_server_json.return_value = {"action": "ffn_split"}
        control.policy = SimpleNamespace(policy_hash="sha256:" + "4" * 64)
        control.plan_generation = 1
        control.__class__ = AdaptiveDecodeControl
        members = (("a", 0), ("b", 1))
        stale = {"success": False, "message": "request and active slot differ"}
        refused = {"success": False, "message": "helper pixel: connect failed"}
        cases = (
            ((200, stale), lambda e: LlamaCppHttpClient.read_ffn_stats(e, "g1b", 0),
             StalePhysicalSlotError, "FFN stats failed: request and active slot differ",
             "request and active slot differ"),
            ((200, stale), lambda e: LlamaCppHttpClient.apply_ffn_control(e, control),
             StalePhysicalSlotError, "FFN control failed: request and active slot differ",
             "request and active slot differ"),
            ((200, refused), lambda e: LlamaCppHttpClient.apply_ffn_control(e, control),
             PhysicalAdapterError, "FFN control failed: helper pixel: connect failed",
             "helper pixel: connect failed"),
            ((500, {}), lambda e: LlamaCppHttpClient.read_ffn_stats(e, "g1b", 0),
             PhysicalAdapterError, "FFN stats status is 500", ""),
            ((200, refused),
             lambda e: LlamaCppHttpClient.apply_ffn_cohort_control(e, control, members),
             PhysicalAdapterError, "FFN cohort control failed: helper pixel: connect failed",
             "helper pixel: connect failed"),
            ((503, {}), lambda e: LlamaCppHttpClient.read_ffn_cohort_stats(e, members),
             PhysicalAdapterError, "FFN cohort stats status is 503", ""),
        )
        for reply, call, kind, message, server_text in cases:
            with self.subTest(message=message):
                error = control_failure(reply, call)
                self.assertIs(type(error), kind)
                self.assertEqual(str(error), message)
                self.assertEqual(error.server_control_message, server_text)
                self.assertTrue(_server_side_failure((error,)))
        # a malformed acknowledgement is not a server refusal: still a local failure
        local = PhysicalAdapterError("FFN runtime stats are invalid")
        self.assertFalse(_server_side_failure((local,)))


if __name__ == "__main__":
    unittest.main()
