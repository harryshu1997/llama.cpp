#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from research_dev.scheduler import (
    ResourceProfile,
    RuntimeCapabilityCatalog,
    RuntimeCompositeExecutorCapability,
    RuntimeResidencyEviction,
    RuntimeTransitionCapability,
    UnifiedScheduler,
)
from research_dev.scheduler.adapters import (
    CanonicalPhysicalAdapter,
    PhysicalAdapterError,
    PhysicalBackendFailure,
    RawEnergyMeasurement,
    RawExecutionObservation,
    RawTransitionObservation,
    RuntimeActivityTracker,
    interpret_runtime_ticket,
    transition_receipt_from_observation,
)
from research_dev.scheduler._internal.route_generation import (
    DesktopControlUnavailableError,
)

try:
    from .test_automated_runtime import (
        catalog,
        executor_state,
        request,
        runtime_snapshot,
    )
    from .test_gguf_cost import write_synthetic_gguf
except ImportError:
    from test_automated_runtime import (
        catalog,
        executor_state,
        request,
        runtime_snapshot,
    )
    from test_gguf_cost import write_synthetic_gguf


class FakeMeasuredBackend:
    def __init__(self, fleet_energy_uj: int) -> None:
        self.fleet_energy_uj = fleet_energy_uj
        self.execution_commands = []
        self.transition_commands = []

    def apply_transition(self, command, payload, control_check):
        self.transition_commands.append(command)
        control_check()
        return RawTransitionObservation(
            started_us=1_000,
            finished_us=1_001,
            status="COMPLETED",
            evicted_artifact_sha256s=tuple(sorted({
                row.artifact_sha256
                for row in command.transition.evictions
            })),
        )

    def execute(self, command, payload, control_check):
        self.execution_commands.append(command)
        control_check()
        domains = (
            "energy:accelerator-b",
            "energy:helper-c",
            "energy:host-a",
        )
        base = self.fleet_energy_uj // len(domains)
        energy = {domain: base for domain in domains}
        energy[domains[-1]] += self.fleet_energy_uj - sum(energy.values())
        transfers = {
            resource_id.removeprefix("link:"): 1
            for resource_id in command.operator_plan["resource_ids"]
            if resource_id.startswith("link:")
        }
        output = hashlib.sha256(
            (command.ticket_id + command.operator_plan_sha256).encode("ascii")
        ).hexdigest()
        return RawExecutionObservation(
            started_us=command.planned_start_us,
            finished_us=command.planned_finish_us,
            output_sha256=output,
            payload={"tokens": [1, 2, 3]},
            energy=RawEnergyMeasurement(
                energy_boundary_id="synthetic-whole-fleet",
                fleet_energy_uj_by_domain=energy,
                transfer_energy_uj_by_link=transfers,
                measurement_evidence_ids=("synthetic-matched-meter",),
                attribution_kind="matched_abba",
            ),
        )


class FailFirstMeasuredBackend(FakeMeasuredBackend):
    def __init__(self, fleet_energy_uj: int) -> None:
        super().__init__(fleet_energy_uj)
        self.failed = False

    def execute(self, command, payload, control_check):
        self.execution_commands.append(command)
        control_check()
        if not self.failed:
            self.failed = True
            raise PhysicalBackendFailure(
                "synthetic connect failure",
                phase="connect",
                retry_safe=True,
                execution_started=False,
                started_us=command.planned_start_us,
                finished_us=command.planned_start_us,
                failed_resource_ids=("compute:helper-c",),
            )
        self.execution_commands.pop()
        return super().execute(command, payload, control_check)


class SlowTerminalMeasuredBackend(FakeMeasuredBackend):
    def __init__(self, fleet_energy_uj: int) -> None:
        super().__init__(fleet_energy_uj)
        self.capacity_released = threading.Event()
        self.finish_terminal_receipt = threading.Event()

    def execute_with_capacity_release(
        self, command, payload, control_check, capacity_release
    ):
        observation = super().execute(command, payload, control_check)
        capacity_release(observation.finished_us)
        self.capacity_released.set()
        if not self.finish_terminal_receipt.wait(1):
            raise RuntimeError("synthetic terminal measurement timeout")
        return observation


def composite_catalog(manifest_sha256: str) -> RuntimeCapabilityCatalog:
    source = catalog(
        phone_ops_per_s=8_000_000_000,
        phone_power_mw=500,
        phone_bandwidth=8_000_000_000,
        phone_whole_model=False,
        gpu_whole_model=False,
    )
    resources = dict(source.resources)
    resources["coordinator:cpu-phone"] = ResourceProfile(
        resource_id="coordinator:cpu-phone",
        kind="coordinator",
        capacity=1,
        ready=True,
        identity="coordinator:cpu-phone",
    )
    physical = RuntimeCompositeExecutorCapability(
        executor_id="coordinator:cpu-phone",
        endpoint="synthetic://cpu-phone",
        backend="backend:composite",
        coordinator_device_id="host-a",
        participant_device_ids=("host-a", "helper-c"),
        participant_resource_ids={
            "host-a": (
                "compute:host-a", "coordinator:cpu-phone"
            ),
            "helper-c": ("compute:helper-c",),
        },
        route_family="operator_split",
        assisted_operator_kind="ffn",
        split_axis="column",
        split_fractions_ppm=(500_000,),
        layer_fractions_ppm=(),
        residency_states=("hot",),
        resource_ids=(
            "compute:host-a",
            "compute:helper-c",
            "coordinator:cpu-phone",
            "link:usb-out",
            "link:usb-in",
        ),
        operator_plan_protocol="synthetic-composite-v1",
        maturity="QUALIFIED",
        evidence_ids=("physical-composite-evidence",),
        artifact_sha256=manifest_sha256,
        adapter_parameters={
            "launch_contract": "synthetic-static-split-v1",
            "split_columns": 128,
        },
    )
    return RuntimeCapabilityCatalog.from_json(replace(
        source,
        resources=resources,
        executors=tuple(
            replace(
                row,
                coordinated_route_families=(),
                operator_plan_protocol=None,
            )
            for row in source.executors
        ),
        composite_executors=(physical,),
        transitions=(
            next(
                row for row in source.transitions
                if row.device_id == "host-a"
            ),
            RuntimeTransitionCapability(
                transition_id="load:coordinator:cpu-phone",
                device_id="host-a",
                source_state="cold",
                target_state="hot",
                fixed_latency_us=100,
                bandwidth_bytes_per_s=1_000_000_000,
                fixed_energy_uj=100,
                dynamic_pj_per_byte=100,
                resource_ids=physical.resource_ids,
                maturity="QUALIFIED",
                evidence_ids=("physical-composite-transition",),
                executor_id=physical.executor_id,
                prepares_device_ids=physical.participant_device_ids,
            ),
        ),
    ).to_json())


class PhysicalAdapterTests(unittest.TestCase):
    def test_phone_layout_publication_uses_physical_epoch(self) -> None:
        scheduler, manifest = self.make_scheduler()
        snapshot = self.physical_snapshot(manifest, include_phone=False)
        with mock.patch(
            "research_dev.scheduler.adapters.runtime.time.monotonic_ns",
            return_value=9_000_000_000,
        ):
            CanonicalPhysicalAdapter(
                scheduler,
                FakeMeasuredBackend(300),
                epoch_ns=8_000_000_000,
                snapshot_provider=lambda _ticket, _at_us: snapshot,
            )
            scheduler._model_placement_controller.record_phone_layout_evaluation(
                100, {"reason": "physical-clock"}
            )
        event = scheduler._model_placement_controller.phone_layout_events()[-1]
        self.assertEqual(event["observed_at_us"], 100)
        self.assertEqual(event["published_at_us"], 1_000_000)

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "adapter-model.gguf"
        write_synthetic_gguf(self.path)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def make_scheduler(self):
        probe = UnifiedScheduler.for_runtime_discovery("enforce")
        probe.register_runtime_capabilities(catalog())
        manifest = probe.register_gguf_model("adapter-model", self.path)
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(
            composite_catalog(manifest.artifact_sha256)
        )
        scheduler.register_model_manifest(manifest)
        return scheduler, manifest

    @staticmethod
    def physical_snapshot(manifest, *, include_phone: bool):
        snapshot = runtime_snapshot(
            manifest,
            include_phone=include_phone,
            phone_bandwidth=8_000_000_000,
        )
        if include_phone:
            snapshot = replace(
                snapshot,
                executors={
                    **snapshot.executors,
                    "coordinator:cpu-phone": executor_state(
                        "coordinator:cpu-phone"
                    ),
                },
            )
        return snapshot

    def execute(self, scheduler, ticket, snapshot, backend):
        adapter = CanonicalPhysicalAdapter(
            scheduler,
            backend,
            epoch_ns=time.monotonic_ns()
                - ticket.decision.start_us * 1_000,
            snapshot_provider=lambda _ticket, _at_us: snapshot,
            lease_guard_us=50_000,
            lease_quantum_us=100_000,
        )
        return adapter.execute(ticket, {"prompt_tokens": [1, 2]})

    def test_request_completion_does_not_stop_model_helper_preparation(
        self,
    ) -> None:
        scheduler, manifest = self.make_scheduler()
        snapshot = self.physical_snapshot(manifest, include_phone=False)
        ticket = scheduler.submit_automated_request(
            request("adapter-background-helper"),
            manifest.model_id,
            snapshot,
            selection_mode="desktop-baseline",
        )
        adapter = CanonicalPhysicalAdapter(
            scheduler,
            FakeMeasuredBackend(5_000),
            epoch_ns=(
                time.monotonic_ns()
                - ticket.decision.start_us * 1_000
            ),
            snapshot_provider=lambda _ticket, _at_us: snapshot,
            lease_guard_us=50_000,
            lease_quantum_us=100_000,
        )
        release = threading.Event()
        stopped = threading.Event()
        finished = threading.Event()
        stop = threading.Event()

        def prepare() -> None:
            while not release.is_set():
                if stop.wait(0.01):
                    stopped.set()
                    return
            finished.set()

        thread = threading.Thread(target=prepare)
        thread.start()
        handle = SimpleNamespace(thread=thread, stop=stop)
        with mock.patch.object(
            adapter,
            "_start_helper_preparation",
            return_value=handle,
        ):
            result = adapter.execute(ticket, {"prompt_tokens": [1, 2]})
        self.assertEqual(result.completion.status, "released")
        self.assertFalse(stop.is_set())
        self.assertFalse(stopped.is_set())
        self.assertTrue(thread.is_alive())
        release.set()
        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())
        self.assertTrue(finished.is_set())

    def test_dynamic_helper_reader_rejects_retired_ticket_generation(
        self,
    ) -> None:
        request_value = SimpleNamespace(
            request_id="retired-helper-generation",
            arrival_us=0,
        )
        ticket = SimpleNamespace(
            request=request_value,
            dispatch_state="ACQUIRED",
            ticket_id="retired-helper-generation:attempt:0",
        )
        command = SimpleNamespace(
            adapter_parameters={},
            helper_envelope=None,
        )

        class Scheduler:
            def __init__(self) -> None:
                self.background_allowed = threading.Event()
                self.dynamic_reads = 0
                self.preparation_calls = 0

            def runtime_ticket(self, _request_id):
                return ticket

            def runtime_background_helper_preparation_allowed(
                self, *_args, **_kwargs
            ):
                return self.background_allowed.is_set()

            def runtime_request_helper_preparation_envelope(
                self, *_args, **_kwargs
            ):
                self.dynamic_reads += 1
                return None

            def begin_request_helper_preparation(self, *_args, **_kwargs):
                self.preparation_calls += 1
                return {"status": "INCOMPATIBLE"}

            def check_request_helper_preparation(self, *_args, **_kwargs):
                return None

            def complete_request_helper_preparation(
                self, *_args, **_kwargs
            ):
                return None

            def fail_request_helper_preparation(self, *_args, **_kwargs):
                return None

        scheduler = Scheduler()
        adapter = object.__new__(CanonicalPhysicalAdapter)
        adapter._scheduler = scheduler
        adapter._epoch_ns = time.monotonic_ns()
        adapter._helper_preparation_lock = threading.Lock()
        adapter._helper_preparation_watchers = set()
        adapter._started_helper_preparations = set()
        adapter._helper_preparations = []
        adapter._snapshot = lambda _ticket, _at_us: SimpleNamespace(
            captured_at_us=_at_us
        )
        with mock.patch(
            "research_dev.scheduler.adapters.runtime.interpret_runtime_ticket",
            return_value=command,
        ):
            handle = adapter._start_helper_preparation(
                ticket, command, object()
            )
            self.assertIsNotNone(handle)
            time.sleep(0.1)
            self.assertEqual(scheduler.dynamic_reads, 0)
            scheduler.background_allowed.set()
            deadline = time.monotonic() + 1.0
            while scheduler.dynamic_reads == 0 and time.monotonic() < deadline:
                time.sleep(0.01)
            handle.stop.set()
            handle.thread.join(timeout=1.0)
        self.assertGreater(scheduler.dynamic_reads, 0)
        self.assertEqual(scheduler.preparation_calls, 0)
        self.assertFalse(handle.thread.is_alive())

    def test_queued_watcher_replans_on_state_change_not_every_50ms(self) -> None:
        """Physical v14: each request-helper watcher burned 1.3-2.0 CPU-s per
        5 s while its request was merely QUEUED, because every 50 ms iteration
        captured a full runtime snapshot and re-projected the helper envelope.
        The watcher now re-plans only when the scheduler's helper state
        generation moves or a bounded fallback interval elapses."""
        request_value = SimpleNamespace(request_id="queued-watch", arrival_us=0)
        ticket = SimpleNamespace(
            request=request_value,
            dispatch_state="QUEUED",
            ticket_id="queued-watch:attempt:0",
        )
        command = SimpleNamespace(adapter_parameters={}, helper_envelope=None)

        class Scheduler:
            def __init__(self) -> None:
                self.generation = 7
                self.dynamic_reads = 0

            def runtime_ticket(self, _request_id):
                return ticket

            def runtime_helper_state_generation(self):
                return self.generation

            def runtime_background_helper_preparation_allowed(self, *_a, **_k):
                return True

            def runtime_request_helper_preparation_envelope(self, *_a, **_k):
                self.dynamic_reads += 1
                return None

            def begin_request_helper_preparation(self, *_a, **_k):
                return {"status": "INCOMPATIBLE"}

            def check_request_helper_preparation(self, *_a, **_k):
                return None

            def complete_request_helper_preparation(self, *_a, **_k):
                return None

            def fail_request_helper_preparation(self, *_a, **_k):
                return None

        scheduler = Scheduler()
        snapshots = []
        adapter = object.__new__(CanonicalPhysicalAdapter)
        adapter._scheduler = scheduler
        adapter._epoch_ns = time.monotonic_ns()
        adapter._helper_preparation_lock = threading.Lock()
        adapter._helper_preparation_watchers = set()
        adapter._started_helper_preparations = set()
        adapter._helper_preparations = []
        adapter._snapshot = lambda _ticket, at_us: (
            snapshots.append(at_us) or SimpleNamespace(captured_at_us=at_us)
        )
        with mock.patch(
            "research_dev.scheduler.adapters.runtime.interpret_runtime_ticket",
            return_value=command,
        ):
            handle = adapter._start_helper_preparation(ticket, command, object())
            self.assertIsNotNone(handle)
            try:
                time.sleep(0.6)
                # One planning pass at start, none of the ~12 that a 50 ms
                # poll would have made while nothing changed.
                self.assertEqual(len(snapshots), 1)
                self.assertEqual(scheduler.dynamic_reads, 1)
                # A state change wakes it within the 50 ms idle check.
                scheduler.generation += 1
                deadline = time.monotonic() + 0.5
                while len(snapshots) < 2 and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(len(snapshots), 2)
                # Bounded fallback: with a 100 ms fallback and no state
                # change, it still re-plans, but at that pace only.
                adapter._helper_watch_fallback_us = 100_000
                before = len(snapshots)
                time.sleep(0.55)
                self.assertGreaterEqual(len(snapshots) - before, 3)
                self.assertLessEqual(len(snapshots) - before, 7)
            finally:
                handle.stop.set()
                handle.thread.join(timeout=1.0)
        self.assertFalse(handle.thread.is_alive())

    def test_ready_helper_refresh_precedes_new_preparation(self) -> None:
        request_value = SimpleNamespace(
            request_id="ready-helper-first",
            arrival_us=0,
        )
        ticket = SimpleNamespace(
            request=request_value,
            dispatch_state="ACQUIRED",
            ticket_id="ready-helper-first:attempt:0",
        )
        command = SimpleNamespace(
            adapter_parameters={},
            helper_envelope=object(),
        )

        class Scheduler:
            def __init__(self) -> None:
                self.dynamic_reads = 0
                self.refreshes = 0
                self.preparation_calls = 0

            def runtime_ticket(self, _request_id):
                return ticket

            def runtime_background_helper_preparation_allowed(
                self, *_args, **_kwargs
            ):
                return True

            def runtime_ready_helper_refresh_needed(
                self, *_args, **_kwargs
            ):
                return True

            def refresh_ready_request_helper(self, *_args, **_kwargs):
                self.refreshes += 1
                return True

            def runtime_request_helper_preparation_envelope(
                self, *_args, **_kwargs
            ):
                self.dynamic_reads += 1
                return None

            def begin_request_helper_preparation(self, *_args, **_kwargs):
                self.preparation_calls += 1
                return {"status": "INCOMPATIBLE"}

            def check_request_helper_preparation(self, *_args, **_kwargs):
                return None

            def complete_request_helper_preparation(
                self, *_args, **_kwargs
            ):
                return None

            def fail_request_helper_preparation(self, *_args, **_kwargs):
                return None

        scheduler = Scheduler()
        adapter = object.__new__(CanonicalPhysicalAdapter)
        adapter._scheduler = scheduler
        adapter._epoch_ns = time.monotonic_ns()
        adapter._helper_preparation_lock = threading.Lock()
        adapter._helper_preparation_watchers = set()
        adapter._started_helper_preparations = set()
        adapter._helper_preparations = []
        adapter._snapshot = lambda _ticket, _at_us: SimpleNamespace(
            captured_at_us=_at_us
        )
        with mock.patch(
            "research_dev.scheduler.adapters.runtime."
            "interpret_runtime_ticket",
            return_value=command,
        ):
            handle = adapter._start_helper_preparation(
                ticket, command, object()
            )
            self.assertIsNotNone(handle)
            handle.thread.join(timeout=1.0)
        self.assertFalse(handle.thread.is_alive())
        self.assertEqual(scheduler.refreshes, 1)
        self.assertEqual(scheduler.dynamic_reads, 0)
        self.assertEqual(scheduler.preparation_calls, 0)

    def test_ticket_interpretation_and_receipt_are_scheduler_bound(self) -> None:
        scheduler, manifest = self.make_scheduler()
        snapshot = self.physical_snapshot(manifest, include_phone=True)
        ticket = scheduler.submit_automated_request(
            request("adapter-bound-ticket"), manifest.model_id, snapshot
        )
        self.assertEqual(
            set(ticket.execution_plan.device_ids), {"host-a", "helper-c"}
        )
        backend = FakeMeasuredBackend(5_000)
        result = self.execute(scheduler, ticket, snapshot, backend)

        self.assertEqual(len(backend.execution_commands), 1)
        command = backend.execution_commands[0]
        self.assertEqual(command.route_id, ticket.decision.route_id)
        self.assertEqual(command.executor_id, ticket.binding.executor_id)
        self.assertEqual(
            command.operator_plan_sha256,
            ticket.execution_plan.plan_sha256,
        )
        self.assertEqual(
            command.operator_plan["split_fraction_ppm"],
            ticket.execution_plan.split_fraction_ppm,
        )
        self.assertEqual(
            dict(command.adapter_parameters),
            {
                "launch_contract": "synthetic-static-split-v1",
                "split_columns": 128,
            },
        )
        self.assertEqual(
            command.operator_plan["adapter_parameters"],
            dict(command.adapter_parameters),
        )
        self.assertEqual(
            {row["resource_id"] for row in command.leases},
            {row.resource_id for row in ticket.decision.leases},
        )
        self.assertEqual(
            result.completion.execution_receipt.executor_id,
            ticket.binding.executor_id,
        )
        self.assertEqual(
            result.observation.payload, {"tokens": [1, 2, 3]}
        )

    def test_capacity_is_released_before_terminal_measurement(self) -> None:
        scheduler, manifest = self.make_scheduler()
        snapshot = self.physical_snapshot(manifest, include_phone=False)
        ticket = scheduler.submit_automated_request(
            request("adapter-staged-capacity"),
            manifest.model_id,
            snapshot,
            selection_mode="desktop-baseline",
        )
        backend = SlowTerminalMeasuredBackend(5_000)
        results = []
        failures = []

        def run() -> None:
            try:
                results.append(self.execute(
                    scheduler, ticket, snapshot, backend
                ))
            except BaseException as error:
                failures.append(error)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        self.assertTrue(backend.capacity_released.wait(1))
        staged = scheduler.runtime_ticket(ticket.request.request_id)
        self.assertEqual(staged.dispatch_state, "ACQUIRED")
        self.assertEqual(
            staged.lease_status, "RELEASED_PENDING_RECEIPT"
        )
        self.assertEqual(
            scheduler.runtime_controller_snapshot()["dispatch_queue"]
            ["entry_states"][ticket.request.request_id]["state"],
            "FINISHING",
        )
        self.assertTrue(worker.is_alive())

        backend.finish_terminal_receipt.set()
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(len(results), 1)
        self.assertEqual(
            scheduler.runtime_ticket(ticket.request.request_id)
            .dispatch_state,
            "COMPLETED",
        )

    def test_terminal_proof_failure_can_cancel_already_released_capacity(self) -> None:
        staged_tickets = []

        class FailedTerminalBackend(FakeMeasuredBackend):
            def execute_with_capacity_release(
                self, command, payload, control_check, capacity_release,
            ):
                observation = super().execute(command, payload, control_check)
                capacity_release(observation.finished_us)
                staged_tickets.append(scheduler.runtime_ticket(command.request_id))
                raise PhysicalAdapterError("synthetic invalid terminal proof")

        scheduler, manifest = self.make_scheduler()
        snapshot = self.physical_snapshot(manifest, include_phone=False)
        ticket = scheduler.submit_automated_request(
            request("adapter-failed-terminal"), manifest.model_id, snapshot,
            selection_mode="desktop-baseline",
        )
        self.assertTrue(ticket.memory_reservations)
        with self.assertRaisesRegex(PhysicalAdapterError, "after capacity release"):
            self.execute(scheduler, ticket, snapshot, FailedTerminalBackend(5_000))
        self.assertEqual(len(staged_tickets), 1)
        staged = staged_tickets[0]
        self.assertEqual(staged.memory_reservation_status, "RELEASED")
        self.assertEqual(staged.lease_status, "RELEASED_PENDING_RECEIPT")
        terminal = scheduler.runtime_ticket(ticket.request.request_id)
        self.assertEqual(terminal.dispatch_state, "CANCELLED")
        self.assertEqual(terminal.memory_reservation_status, "CANCELLED")
        self.assertIsNone(terminal.execution_receipt)
        self.assertEqual(scheduler.cancel_runtime_request_if_pending(
            ticket.request.request_id, staged.actual_end_us, "cleanup_retry",
        ), ())
        state = scheduler.runtime_controller_snapshot()
        self.assertEqual(state["capacity_releases"], {})
        self.assertEqual(state["dispatch_queue"]["active"], {})
        self.assertEqual(state["dispatch_queue"]["queued"], {})

    def test_stale_replan_snapshot_is_retried_without_terminal_record(
        self,
    ) -> None:
        scheduler, manifest = self.make_scheduler()
        snapshot = self.physical_snapshot(manifest, include_phone=False)
        ticket = scheduler.submit_automated_request(
            request("adapter-stale-replan"),
            manifest.model_id,
            snapshot,
            selection_mode="desktop-baseline",
        )
        self.assertTrue(
            scheduler._runtime_controller.require_queued_replan(
                ticket.request.request_id, "synthetic_snapshot_refresh"
            )
        )
        stale = replace(
            snapshot,
            snapshot_id="adapter-stale-runtime",
            captured_at_us=0,
            valid_until_us=1,
            memory=replace(
                snapshot.memory,
                snapshot_id="adapter-stale-memory",
                captured_at_us=0,
                valid_until_us=1,
            ),
        )
        fresh = replace(
            snapshot,
            snapshot_id="adapter-fresh-runtime",
            captured_at_us=ticket.request.arrival_us,
            valid_until_us=snapshot.valid_until_us,
        )
        snapshots = [stale, fresh]

        def snapshot_provider(_ticket, _wake_observed_at_us):
            return snapshots.pop(0)

        adapter = CanonicalPhysicalAdapter(
            scheduler,
            FakeMeasuredBackend(5_000),
            epoch_ns=(
                time.monotonic_ns()
                - ticket.decision.start_us * 1_000
            ),
            snapshot_provider=snapshot_provider,
            lease_guard_us=50_000,
            lease_quantum_us=100_000,
        )
        result = adapter.execute(ticket, {"prompt_tokens": [1, 2]})

        self.assertEqual(snapshots, [])
        self.assertEqual(result.completion.request_id, ticket.request.request_id)
        self.assertEqual(result.ticket.attempt_index, 1)
        self.assertNotIn(
            "FAILED",
            tuple(
                row["event_kind"]
                for row in scheduler.runtime_decision_log()["records"]
            ),
        )

    def test_transient_desktop_control_gap_is_retried_without_terminal_record(
        self,
    ) -> None:
        scheduler, manifest = self.make_scheduler()
        snapshot = self.physical_snapshot(manifest, include_phone=False)
        ticket = scheduler.submit_automated_request(
            request("adapter-transient-desktop-control"),
            manifest.model_id,
            snapshot,
            selection_mode="desktop-baseline",
        )
        self.assertTrue(
            scheduler._runtime_controller.require_queued_replan(
                ticket.request.request_id, "synthetic_live_state_gap"
            )
        )
        original = scheduler._select_automated_candidate
        selection_calls = 0

        def select_with_one_gap(*args, **kwargs):
            nonlocal selection_calls
            selection_calls += 1
            if selection_calls == 1:
                raise DesktopControlUnavailableError(
                    "qualified desktop control was not generated"
                )
            return original(*args, **kwargs)

        snapshots = []

        def snapshot_provider(_ticket, _wake_observed_at_us):
            snapshots.append(snapshot.snapshot_id)
            return snapshot

        adapter = CanonicalPhysicalAdapter(
            scheduler,
            FakeMeasuredBackend(5_000),
            epoch_ns=(
                time.monotonic_ns()
                - ticket.decision.start_us * 1_000
            ),
            snapshot_provider=snapshot_provider,
            lease_guard_us=50_000,
            lease_quantum_us=100_000,
        )
        with mock.patch.object(
            scheduler,
            "_select_automated_candidate",
            side_effect=select_with_one_gap,
        ):
            result = adapter.execute(ticket, {"prompt_tokens": [1, 2]})

        self.assertEqual(selection_calls, 2)
        self.assertEqual(len(snapshots), 2)
        self.assertEqual(result.completion.request_id, ticket.request.request_id)
        self.assertEqual(result.ticket.attempt_index, 1)
        self.assertNotIn(
            "FAILED",
            tuple(
                row["event_kind"]
                for row in scheduler.runtime_decision_log()["records"]
            ),
        )

    def test_cold_residency_transitions_are_executed_by_the_adapter(self) -> None:
        scheduler, manifest = self.make_scheduler()
        snapshot = self.physical_snapshot(manifest, include_phone=True)
        snapshot = replace(snapshot, residency=())
        ticket = scheduler.submit_automated_request(
            request("adapter-cold-transition"), manifest.model_id, snapshot
        )
        self.assertTrue(ticket.execution_plan.transitions)
        backend = FakeMeasuredBackend(5_000)
        result = self.execute(scheduler, ticket, snapshot, backend)

        self.assertEqual(
            [row.transition.transition_id for row in backend.transition_commands],
            [row.transition_id for row in ticket.execution_plan.transitions],
        )
        self.assertTrue(all(
            row.participant.executor_id == ticket.binding.executor_id
            and row.participant.endpoint == ticket.binding.endpoint
            and set(row.transition.prepares_device_ids)
                == set(ticket.execution_plan.device_ids)
            for row in backend.transition_commands
        ))
        terminal = scheduler.runtime_ticket(ticket.request.request_id)
        self.assertEqual(terminal.transition_status, "COMPLETED")
        self.assertEqual(
            [row.transition_id for row in terminal.transition_receipts],
            [row.transition_id for row in ticket.execution_plan.transitions],
        )
        self.assertEqual(result.completion.route_id, ticket.decision.route_id)

    def test_failed_transition_receipt_does_not_claim_eviction(self) -> None:
        scheduler, manifest = self.make_scheduler()
        snapshot = replace(
            self.physical_snapshot(manifest, include_phone=True),
            residency=(),
        )
        ticket = scheduler.submit_automated_request(
            request("adapter-failed-transition"), manifest.model_id, snapshot
        )
        command = interpret_runtime_ticket(ticket).transitions[0]
        command = replace(
            command,
            transition=replace(
                command.transition,
                evictions=(RuntimeResidencyEviction(
                    model_id="old-model",
                    artifact_sha256="sha256:" + "a" * 64,
                    device_id=command.transition.device_id,
                    resident_bytes=100,
                    generation=3,
                ),),
            ),
        )
        receipt = transition_receipt_from_observation(
            command,
            RawTransitionObservation(
                started_us=10,
                finished_us=20,
                status="FAILED",
            ),
        )
        self.assertEqual(receipt.status, "FAILED")
        self.assertEqual(receipt.evicted_artifact_sha256s, ())

    def test_multi_resource_eviction_has_one_artifact_receipt(self) -> None:
        scheduler, manifest = self.make_scheduler()
        snapshot = replace(
            self.physical_snapshot(manifest, include_phone=True),
            residency=(),
        )
        ticket = scheduler.submit_automated_request(
            request("adapter-multi-resource-eviction"),
            manifest.model_id,
            snapshot,
        )
        command = interpret_runtime_ticket(ticket).transitions[0]
        source_hash = "sha256:" + "a" * 64
        command = replace(
            command,
            transition=replace(
                command.transition,
                evictions=tuple(
                    RuntimeResidencyEviction(
                        model_id="old-model",
                        artifact_sha256=source_hash,
                        device_id=device_id,
                        resident_bytes=100,
                        generation=3,
                    )
                    for device_id in command.transition.prepares_device_ids
                ),
            ),
        )

        receipt = transition_receipt_from_observation(
            command,
            RawTransitionObservation(
                started_us=10,
                finished_us=20,
                status="COMPLETED",
                evicted_artifact_sha256s=(source_hash,),
            ),
        )

        self.assertEqual(
            receipt.evicted_artifact_sha256s, (source_hash,)
        )

    def test_transition_energy_has_its_own_receipt_boundary(self) -> None:
        scheduler, manifest = self.make_scheduler()
        snapshot = replace(
            self.physical_snapshot(manifest, include_phone=True),
            residency=(),
        )
        ticket = scheduler.submit_automated_request(
            request("adapter-transition-energy"),
            manifest.model_id,
            snapshot,
        )
        command = interpret_runtime_ticket(ticket).transitions[0]
        receipt = transition_receipt_from_observation(
            command,
            RawTransitionObservation(
                started_us=10,
                finished_us=20,
                status="COMPLETED",
                energy=RawEnergyMeasurement(
                    energy_boundary_id="synthetic-whole-fleet",
                    fleet_energy_uj_by_domain={
                        "energy:accelerator-b": 10,
                        "energy:helper-c": 20,
                        "energy:host-a": 30,
                    },
                    measurement_evidence_ids=(
                        "synthetic-transition-meter",
                    ),
                    attribution_kind="isolated",
                ),
            ),
        )

        self.assertEqual(receipt.actual_latency_us, 10)
        self.assertEqual(receipt.whole_fleet_energy_uj, 60)
        self.assertEqual(receipt.energy_attribution_kind, "isolated")

    def test_desktop_then_cpu_phone_feedback_changes_later_route(self) -> None:
        scheduler, manifest = self.make_scheduler()
        desktop_snapshot = self.physical_snapshot(
            manifest, include_phone=False
        )
        baseline = scheduler.submit_automated_request(
            request("adapter-baseline"),
            manifest.model_id,
            desktop_snapshot,
        )
        self.assertNotIn("helper-c", baseline.execution_plan.device_ids)
        self.execute(
            scheduler,
            baseline,
            desktop_snapshot,
            FakeMeasuredBackend(10_000),
        )

        composite_snapshot = self.physical_snapshot(
            manifest, include_phone=True
        )
        composite_route = None
        for index in range(4):
            ticket = scheduler.submit_automated_request(
                request(
                    "adapter-refine-" + str(index),
                    arrival_us=100_000 + index * 100_000,
                ),
                manifest.model_id,
                composite_snapshot,
            )
            self.assertEqual(
                set(ticket.execution_plan.device_ids),
                {"host-a", "helper-c"},
            )
            composite_route = ticket.decision.route_id
            self.execute(
                scheduler,
                ticket,
                composite_snapshot,
                FakeMeasuredBackend(1_000_000_000),
            )

        adapted = scheduler.submit_automated_request(
            request("adapter-adapted", arrival_us=600_000),
            manifest.model_id,
            composite_snapshot,
        )
        self.assertNotEqual(adapted.decision.route_id, composite_route)
        self.assertNotEqual(
            set(adapted.execution_plan.device_ids),
            {"host-a", "helper-c"},
        )
        records = scheduler.runtime_decision_log()["records"]
        self.assertEqual(
            len([row for row in records if row["event_kind"] == "COMPLETED"]),
            5,
        )
        self.assertTrue(all(
            row["selected"]["execution_receipt"] is not None
            for row in records if row["event_kind"] == "COMPLETED"
        ))

    def test_desktop_baseline_is_a_scheduler_selected_ticket(self) -> None:
        scheduler, manifest = self.make_scheduler()
        snapshot = self.physical_snapshot(manifest, include_phone=True)
        ticket = scheduler.submit_automated_request(
            request("adapter-desktop-control"),
            manifest.model_id,
            snapshot,
            selection_mode="desktop-baseline",
        )

        selected = next(
            row for row in ticket.cost_estimates.estimates
            if row.route_id == ticket.decision.route_id
        )
        self.assertTrue(selected.baseline)
        self.assertEqual(ticket.selection_mode, "desktop-baseline")
        self.assertEqual(ticket.decision.reason, "DESKTOP_BASELINE_CONTROL")
        self.assertGreater(len(ticket.cost_estimates.estimates), 1)
        self.assertTrue(all(
            reason == "BASELINE_CONTROL_NOT_SELECTED"
            for route_id, reason in ticket.decision.rejected
            if route_id != ticket.decision.route_id
        ))

        backend = FakeMeasuredBackend(10_000)
        result = self.execute(scheduler, ticket, snapshot, backend)
        self.assertEqual(
            result.command.executor_id, ticket.binding.executor_id
        )
        terminal = scheduler.runtime_ticket(ticket.request.request_id)
        self.assertEqual(terminal.dispatch_state, "COMPLETED")
        self.assertEqual(terminal.selection_mode, "desktop-baseline")

    def test_activity_tracker_reports_composite_participants(self) -> None:
        scheduler, manifest = self.make_scheduler()
        snapshot = self.physical_snapshot(manifest, include_phone=True)
        ticket = scheduler.submit_automated_request(
            request("adapter-activity"), manifest.model_id, snapshot
        )
        command = interpret_runtime_ticket(ticket)
        tracker = RuntimeActivityTracker(
            composite_catalog(manifest.artifact_sha256)
        )
        tracker.start(command, input_tokens=11, output_tokens=7)
        active = tracker.snapshot()
        self.assertEqual(active.active_by_device_kind["cpu"], 1)
        self.assertEqual(active.active_by_device_kind["phone"], 1)
        self.assertEqual(
            active.active_input_tokens_by_model[command.model_id], 11
        )
        self.assertEqual(
            active.active_output_tokens_by_model[command.model_id], 7
        )
        self.assertEqual(
            active.active_requests_by_model[command.model_id], 1
        )
        tracker.finish(command)
        self.assertEqual(dict(tracker.snapshot().active_by_device_id), {})

    def test_adapter_owns_failure_replan_and_logs_the_fallback(self) -> None:
        scheduler, manifest = self.make_scheduler()
        snapshot = self.physical_snapshot(manifest, include_phone=True)
        ticket = scheduler.submit_automated_request(
            request("adapter-fallback"), manifest.model_id, snapshot
        )
        failed_route = ticket.decision.route_id
        backend = FailFirstMeasuredBackend(5_000)
        result = self.execute(scheduler, ticket, snapshot, backend)

        self.assertEqual(len(result.recoveries), 1)
        self.assertEqual(len(backend.execution_commands), 2)
        self.assertEqual(
            backend.execution_commands[0].route_id, failed_route
        )
        self.assertNotEqual(result.command.route_id, failed_route)
        self.assertEqual(
            result.command.executor_id, result.ticket.binding.executor_id
        )
        self.assertEqual(
            scheduler.runtime_controller_snapshot()[
                "quarantined_resources"
            ],
            ["compute:helper-c"],
        )
        records = [
            row for row in scheduler.runtime_decision_log()["records"]
            if row["request_ids"] == ["adapter-fallback"]
        ]
        self.assertEqual(
            [row["event_kind"] for row in records],
            ["DECISION", "ACQUIRED", "FALLBACK", "ACQUIRED", "COMPLETED"],
        )
        self.assertEqual(
            records[2]["attempt_index"], records[0]["attempt_index"] + 1
        )
        self.assertEqual(
            records[2]["previous_ticket_id"],
            ticket.ticket_id,
        )
        self.assertTrue(all(
            row["candidates"]
            for row in records
            if row["event_kind"] in {"DECISION", "FALLBACK"}
        ))

    def test_acquire_reports_scheduler_queue_replan_attempts(self) -> None:
        replan_receipt = SimpleNamespace(
            status="REPLAN_REQUIRED",
            observed_at_us=10,
            wake_reason="predecessor_replan",
        )
        acquired_receipt = SimpleNamespace(
            status="ACQUIRED",
            observed_at_us=20,
            wake_reason="calendar_elapsed",
        )
        initial = SimpleNamespace(
            ticket_id="synthetic-request:attempt:0",
            request=SimpleNamespace(request_id="synthetic-request"),
            decision=SimpleNamespace(reason="ENERGY_AWARE"),
        )
        replanned = SimpleNamespace(
            ticket_id="synthetic-request:attempt:1",
            request=initial.request,
        )

        class Scheduler:
            def __init__(self) -> None:
                self.waits = 0

            def wait_runtime_request(self, _request_id, _epoch_ns):
                self.waits += 1
                if self.waits == 1:
                    return SimpleNamespace(
                        **vars(initial), dispatch_receipt=replan_receipt
                    )
                return SimpleNamespace(
                    **vars(replanned), dispatch_receipt=acquired_receipt
                )

            def replan_automated_request(self, *_args, **_kwargs):
                return replanned

        adapter = object.__new__(CanonicalPhysicalAdapter)
        adapter._scheduler = Scheduler()
        adapter._epoch_ns = 0
        adapter._snapshot = lambda _ticket, _at_us: object()

        current, receipts, attempt_ids = adapter._acquire(initial)

        self.assertEqual(current.ticket_id, replanned.ticket_id)
        self.assertEqual(receipts, (replan_receipt, acquired_receipt))
        self.assertEqual(
            attempt_ids,
            (initial.ticket_id, replanned.ticket_id),
        )
        initial.decision.reason = "STARTUP_PARENT_PRELOAD"
        adapter._scheduler = Scheduler()
        adapter._snapshot = mock.Mock(side_effect=AssertionError("startup must not replan"))
        with self.assertRaisesRegex(PhysicalAdapterError, "fresh preload plan"):
            adapter._acquire(initial)
        adapter._snapshot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
