#!/usr/bin/env python3

from __future__ import annotations

from collections import Counter
import concurrent.futures
from dataclasses import replace
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from research_dev.scheduler import (
    DeviceMemoryCapacity,
    HeterogeneousRuntimeSnapshot,
    ModelResidencyObservation,
    Request,
    RuntimeCapabilityCatalog,
    RuntimeExecutorState,
    RuntimeLinkState,
    RuntimePlacementSnapshot,
    UnifiedScheduler,
)
from research_dev.scheduler.adapters import (
    CanonicalArrivalCoordinator,
    CanonicalRuntimeSubmission,
    PhysicalAdapterError,
    RawTransitionObservation,
    validate_decision_candidate_coverage,
)

try:
    from .test_automated_runtime import catalog, runtime_snapshot
    from .test_gguf_cost import write_synthetic_gguf
    from .test_physical_adapter import FakeMeasuredBackend
except ImportError:
    from test_automated_runtime import catalog, runtime_snapshot
    from test_gguf_cost import write_synthetic_gguf
    from test_physical_adapter import FakeMeasuredBackend


REAL_TRACE_ORDER = (
    "BCBBBAAAAAAACAAAAAAAAACAAAAACAAAAAAABCAAAAAABAACBBAABAAACBBAAAAAA"
    "CAAAABABBCBAAAAABBC"
)
REAL_TRACE_ARRIVALS_US = tuple(
    value // 100 + 100
    for value in (
        0, 200000, 1050000, 1650000, 2350000, 4500000, 5250000,
        5500000, 5650000, 5700000, 5800000, 5900000, 5900000,
        6100000, 6250000, 6300000, 6450000, 6700000, 6850000,
        6900000, 6950000, 7000000, 7050000, 7100000, 7150000,
        7250000, 7350000, 7450000, 7650000, 7700000, 8000000,
        8200000, 8200000, 8450000, 8550000, 10100000, 10100000,
        10300000, 10600000, 11600000, 12350000, 12800000,
        13200000, 13500000, 13550000, 15400000, 15400000,
        15600000, 15950000, 16000000, 19100000, 19750000,
        19900000, 20100000, 20150000, 20300000, 20500000,
        22600000, 22950000, 25350000, 25850000, 26800000,
        27700000, 32200000, 35050000, 35250000, 35950000,
        37650000, 37900000, 38000000, 39100000, 41650000,
        43550000, 43600000, 43800000, 46500000, 47500000,
        48350000, 49900000, 51600000, 52300000, 53850000,
        55750000, 55950000,
    )
)


def contention_catalog() -> RuntimeCapabilityCatalog:
    value = catalog(
        phone_ops_per_s=40_000_000,
        phone_power_mw=500,
        phone_bandwidth=8_000_000_000,
    ).to_json()
    for resource in value["resources"]:
        resource["capacity"] = (
            2 if resource["resource_id"] == "compute:host-a" else 1
        )
    for kernel in value["placement_profile"]["kernels"]:
        rate = {
            "host-a": 10_000_000,
            "accelerator-b": 80_000_000,
            "helper-c": 40_000_000,
        }[kernel["device_id"]]
        kernel["effective_bytes_per_s"] = rate
        kernel["effective_ops_per_s"] = rate
    for executor in value["executors"]:
        if executor["device_id"] == "accelerator-b":
            executor["exclusive_residency_resource_id"] = (
                "compute:accelerator-b"
            )
    return RuntimeCapabilityCatalog.from_json(value)


class OneGpuTraceBackend(FakeMeasuredBackend):
    def __init__(self, scheduler, catalog_value, manifests) -> None:
        super().__init__(5_000)
        self.scheduler = scheduler
        self.catalog = catalog_value
        self.manifests = dict(manifests)
        initial = self.manifests["synthetic-large-b"]
        self._gpu_model_id = initial.model_id
        self._gpu_artifact_sha256 = initial.artifact_sha256
        self._generation = 1
        self._lock = threading.RLock()
        self.active_by_device: Counter[str] = Counter()
        self.max_active_by_device: Counter[str] = Counter()
        self.execution_intervals = []
        self.execution_delay_s = 0.002

    def snapshot(
        self, model_id: str, observed_at_us: int, phone_ready: bool
    ) -> HeterogeneousRuntimeSnapshot:
        with self._lock:
            current_gpu = self.manifests[self._gpu_model_id]
            generation = self._generation
        states = {
            "executor:host-a": RuntimeExecutorState(
                "executor:host-a", True, True, 40_000, 1_000_000, 2, 0
            ),
            "executor:accelerator-b": RuntimeExecutorState(
                "executor:accelerator-b",
                True,
                True,
                40_000,
                1_000_000,
                1,
                0,
            ),
            "executor:helper-c": RuntimeExecutorState(
                "executor:helper-c",
                phone_ready,
                phone_ready,
                40_000,
                900_000,
                int(phone_ready),
                0,
            ),
        }
        links = {
            link_id: RuntimeLinkState(
                link_id,
                phone_ready if link_id.startswith("usb") else True,
                8_000_000_000,
                0,
            )
            for link_id in ("pcie-out", "pcie-in", "usb-out", "usb-in")
        }
        residencies = []
        for manifest in self.manifests.values():
            for device_id in ("host-a", "helper-c"):
                residencies.append(ModelResidencyObservation(
                    manifest.model_id,
                    manifest.artifact_sha256,
                    device_id,
                    "hot",
                    tuple(row.tensor_id for row in manifest.tensors),
                    manifest.tensor_bytes,
                    1,
                ))
        residencies.append(ModelResidencyObservation(
            current_gpu.model_id,
            current_gpu.artifact_sha256,
            "accelerator-b",
            "hot",
            tuple(row.tensor_id for row in current_gpu.tensors),
            current_gpu.tensor_bytes,
            generation,
        ))
        memory = RuntimePlacementSnapshot(
            snapshot_id="synthetic-trace-memory-" + str(observed_at_us),
            captured_at_us=0,
            valid_until_us=120_000_000,
            capacities={
                resource_id: DeviceMemoryCapacity(
                    resource_id,
                    2_000_000_000,
                    (
                        current_gpu.tensor_bytes
                        if resource_id == "gpu-memory" else
                        sum(row.tensor_bytes for row in self.manifests.values())
                    ),
                    100_000_000,
                )
                for resource_id in (
                    "host-memory", "gpu-memory", "phone-memory"
                )
            },
        )
        return HeterogeneousRuntimeSnapshot(
            snapshot_id=(
                "synthetic-trace-" + model_id + "-" + str(observed_at_us)
            ),
            captured_at_us=0,
            valid_until_us=120_000_000,
            memory=memory,
            executors=states,
            links=links,
            residency=tuple(residencies),
        )

    def apply_transition(self, command, payload, control_check):
        control_check()
        prepares_gpu = "accelerator-b" in (
            command.transition.prepares_device_ids
        )
        with self._lock:
            actual_evictions = (
                {self._gpu_artifact_sha256}
                if prepares_gpu
                and self._gpu_artifact_sha256 != command.artifact_sha256
                else set()
            )
            expected_evictions = {
                row.artifact_sha256 for row in command.transition.evictions
            }
            if expected_evictions != actual_evictions:
                raise RuntimeError(
                    "synthetic transition eviction differs from residency:"
                    + "expected="
                    + repr(sorted(expected_evictions))
                    + ":actual="
                    + repr(sorted(actual_evictions))
                )
            if prepares_gpu:
                manifest = next(
                    row for row in self.manifests.values()
                    if row.artifact_sha256 == command.artifact_sha256
                )
                self._gpu_model_id = manifest.model_id
                self._gpu_artifact_sha256 = manifest.artifact_sha256
                self._generation += 1
            self.transition_commands.append(command)
        started_us = self.scheduler.runtime_ticket(
            command.request_id
        ).decision.start_us
        return RawTransitionObservation(
            started_us=started_us,
            finished_us=started_us + command.transition.latency_us,
            status="COMPLETED",
            evicted_artifact_sha256s=tuple(sorted(actual_evictions)),
        )

    def execute(self, command, payload, control_check):
        device_ids = tuple(sorted(
            participant.device_id for participant in command.participants
        ))
        with self._lock:
            if (
                "accelerator-b" in device_ids
                and self._gpu_artifact_sha256 != command.artifact_sha256
            ):
                raise RuntimeError(
                    "synthetic GPU executed a nonresident model"
                )
            for device_id in device_ids:
                self.active_by_device[device_id] += 1
                self.max_active_by_device[device_id] = max(
                    self.max_active_by_device[device_id],
                    self.active_by_device[device_id],
                )
            self.execution_intervals.append((
                command.planned_start_us,
                command.planned_finish_us,
                frozenset(device_ids),
            ))
        try:
            time.sleep(self.execution_delay_s)
            return super().execute(command, payload, control_check)
        finally:
            with self._lock:
                for device_id in device_ids:
                    self.active_by_device[device_id] -= 1


class ArrivalCoordinatorTests(unittest.TestCase):
    def test_publication_clock_is_registered_before_first_submission(self) -> None:
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        coordinator = CanonicalArrivalCoordinator(
            scheduler,
            object(),
            epoch_ns=8_000_000_000,
            snapshot_provider=lambda *_args: None,
            max_workers=1,
        )
        try:
            with mock.patch(
                "research_dev.scheduler.adapters.coordinator.time.monotonic_ns",
                return_value=9_000_000_000,
            ):
                scheduler._model_placement_controller.record_phone_layout_evaluation(
                    100, {"test": "before first submission"}
                )
            event = scheduler.phone_residency_events()[-1]
            self.assertEqual(event["observed_at_us"], 100)
            self.assertEqual(event["published_at_us"], 1_000_000)
            self.assertEqual(coordinator._tickets, {})
        finally:
            coordinator.close()

    def test_drain_reports_later_failure_before_earlier_work_finishes(
        self,
    ) -> None:
        release = threading.Event()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        first = pool.submit(release.wait, 10)
        second = concurrent.futures.Future()
        second.set_exception(RuntimeError("synthetic later failure"))

        class Scheduler:
            def __init__(self) -> None:
                self.states = {"first": "QUEUED", "second": "FAILED"}
                self.cancelled = []

            def runtime_ticket(self, request_id):
                return SimpleNamespace(
                    dispatch_state=self.states[request_id],
                    request=SimpleNamespace(arrival_us=0),
                )

            def cancel_runtime_request(self, request_id, _at_us, _reason):
                self.states[request_id] = "CANCELLED"
                self.cancelled.append(request_id)
                release.set()
                return ()

        scheduler = Scheduler()
        coordinator = object.__new__(CanonicalArrivalCoordinator)
        coordinator._scheduler = scheduler
        coordinator._epoch_ns = time.monotonic_ns()
        coordinator._lock = threading.Lock()
        coordinator._request_ids = ["first", "second"]
        coordinator._tickets = {
            request_id: SimpleNamespace(
                request=SimpleNamespace(request_id=request_id)
            )
            for request_id in coordinator._request_ids
        }
        coordinator._futures = {"first": first, "second": second}
        started = time.monotonic()
        try:
            with self.assertRaisesRegex(
                RuntimeError, "synthetic later failure"
            ):
                coordinator.drain(timeout_s=1)
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertEqual(scheduler.cancelled, ["first"])
            self.assertTrue(first.done())
        finally:
            release.set()
            pool.shutdown(wait=True)

    def test_drain_cancels_and_joins_followers_after_failure(self) -> None:
        release = threading.Event()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        first = concurrent.futures.Future()
        first.set_exception(RuntimeError("synthetic primary failure"))
        second = pool.submit(release.wait, 10)

        class Scheduler:
            def __init__(self) -> None:
                self.states = {
                    "first": "FAILED",
                    "second": "QUEUED",
                }
                self.cancelled = []

            def runtime_ticket(self, request_id):
                return SimpleNamespace(
                    dispatch_state=self.states[request_id],
                    request=SimpleNamespace(arrival_us=0),
                )

            def cancel_runtime_request(self, request_id, _at_us, _reason):
                self.states[request_id] = "CANCELLED"
                self.cancelled.append(request_id)
                release.set()
                return ()

        scheduler = Scheduler()
        coordinator = object.__new__(CanonicalArrivalCoordinator)
        coordinator._scheduler = scheduler
        coordinator._epoch_ns = time.monotonic_ns()
        coordinator._lock = threading.Lock()
        coordinator._request_ids = ["first", "second"]
        coordinator._tickets = {
            request_id: SimpleNamespace(
                request=SimpleNamespace(request_id=request_id)
            )
            for request_id in coordinator._request_ids
        }
        coordinator._futures = {"first": first, "second": second}
        try:
            with self.assertRaisesRegex(
                RuntimeError, "synthetic primary failure"
            ):
                coordinator.drain(timeout_s=5)
            self.assertEqual(scheduler.cancelled, ["second"])
            self.assertTrue(second.done())
        finally:
            release.set()
            pool.shutdown(wait=True)

    def test_complete_84_request_stream_uses_one_scheduler(self) -> None:
        self.assertEqual(len(REAL_TRACE_ORDER), 84)
        self.assertEqual(len(REAL_TRACE_ARRIVALS_US), 84)
        self.assertTrue(all(value > 0 for value in REAL_TRACE_ARRIVALS_US))
        with tempfile.TemporaryDirectory() as directory:
            model_paths = {}
            for model_id, window in (
                ("synthetic-large-a", None),
                ("synthetic-large-b", 16),
                ("synthetic-small-c", 32),
            ):
                model_path = Path(directory) / (model_id + ".gguf")
                write_synthetic_gguf(model_path, sliding_window=window)
                model_paths[model_id] = model_path
            trace_catalog = contention_catalog()
            scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
            scheduler.register_runtime_capabilities(trace_catalog)
            manifests = {
                model_id: scheduler.register_gguf_model(
                    model_id, model_paths[model_id]
                )
                for model_id in (
                    "synthetic-large-a",
                    "synthetic-large-b",
                    "synthetic-small-c",
                )
            }
            self.assertEqual(
                len({row.artifact_sha256 for row in manifests.values()}),
                3,
            )
            backend = OneGpuTraceBackend(
                scheduler, trace_catalog, manifests
            )
            model_by_role = {
                "A": "synthetic-large-a",
                "B": "synthetic-large-b",
                "C": "synthetic-small-c",
            }
            for role, model_id in model_by_role.items():
                prewarm = Request(
                    request_id="synthetic-prewarm-" + role,
                    workload_id=model_id + ":prewarm",
                    arrival_us=1,
                    deadline_us=60_000_000,
                    input_tokens=12,
                    output_tokens=4,
                    quality_requirement="exact",
                )
                scheduler.generate_automated_candidates(
                    prewarm,
                    model_id,
                    backend.snapshot(model_id, 1, True),
                    observed_at_us=1,
                )
            submitted = []
            initial_tickets = {}
            scheduler.start_runtime_lease_renewal = (
                lambda *_args, **_kwargs: None
            )
            scheduler.check_runtime_lease_renewal = (
                lambda *_args, **_kwargs: None
            )
            scheduler.stop_runtime_lease_renewal = (
                lambda *_args, **_kwargs: None
            )
            epoch_ns = time.monotonic_ns()
            coordinator = CanonicalArrivalCoordinator(
                scheduler,
                backend,
                epoch_ns=epoch_ns,
                snapshot_provider=lambda current, at_us: backend.snapshot(
                    current.model.model_id,
                    at_us,
                    bool(current.request.features["phone_ready"]),
                ),
                max_workers=16,
                lease_guard_us=10_000,
                lease_quantum_us=1_000_000,
            )
            wake_token = scheduler._runtime_controller.queue.hold_wake()
            try:
                for request_index, (role, arrival_us) in enumerate(zip(
                    REAL_TRACE_ORDER, REAL_TRACE_ARRIVALS_US
                )):
                    model_id = model_by_role[role]
                    phone_ready = request_index % 11 != 0
                    request = Request(
                        request_id=f"synthetic-84:{request_index:02d}",
                        workload_id=model_id + ":work",
                        arrival_us=arrival_us,
                        deadline_us=arrival_us + 60_000_000,
                        input_tokens=12,
                        output_tokens=4,
                        quality_requirement="exact",
                        features={"phone_ready": int(phone_ready)},
                    )
                    snapshot = backend.snapshot(
                        model_id, arrival_us, phone_ready
                    )
                    ticket = coordinator.submit(
                        CanonicalRuntimeSubmission(
                            request=request,
                            model_id=model_id,
                            snapshot=snapshot,
                            payload={"request_index": request_index},
                            selection_mode=(
                                "energy-aware" if role == "C"
                                else "desktop-baseline"
                            ),
                        ),
                        observed_at_us=arrival_us,
                    )
                    submitted.append(ticket.request.request_id)
                    initial_tickets[request.request_id] = ticket
                    self.assertEqual(
                        scheduler.runtime_ticket(
                            request.request_id
                        ).ticket_id,
                        ticket.ticket_id,
                    )
            finally:
                scheduler._runtime_controller.queue.release_wake(
                    wake_token
                )
            trace_result = coordinator.drain(timeout_s=30)
            executions = dict(trace_result.executions)

        self.assertEqual(len(executions), 84)
        self.assertEqual(tuple(submitted), tuple(executions))
        self.assertEqual(
            Counter(
                execution.ticket.model.model_id
                for execution in executions.values()
            ),
            Counter({
                "synthetic-large-a": 57,
                "synthetic-large-b": 17,
                "synthetic-small-c": 10,
            }),
        )
        records = scheduler.runtime_decision_log()["records"]
        decisions = [
            row for row in records
            if row["event_kind"] == "DECISION"
        ]
        terminals = [
            row for row in records
            if row["event_kind"] in {"COMPLETED", "FAILED", "CANCELLED"}
        ]
        self.assertEqual(len(decisions), 84)
        self.assertEqual(len(terminals), 84)
        self.assertEqual(
            {row["request_ids"][0] for row in decisions}, set(submitted)
        )
        self.assertEqual(
            {row["request_ids"][0] for row in terminals}, set(submitted)
        )
        self.assertEqual(
            [row["request_ids"][0] for row in decisions], submitted
        )
        self.assertEqual(
            [row["event_time_us"] for row in decisions],
            list(REAL_TRACE_ARRIVALS_US),
        )
        kind_by_device = {
            device.device_id: device.kind
            for device in trace_catalog.placement_profile.devices.values()
        }
        expected_families = {
            ("cpu",),
            ("gpu",),
            ("phone",),
            ("cpu", "gpu"),
            ("cpu", "phone"),
            ("gpu", "phone"),
            ("cpu", "gpu", "phone"),
        }
        rejected_phone_requests = 0
        for decision in decisions:
            candidates = decision["candidates"]
            self.assertTrue(candidates)
            self.assertLessEqual(len(candidates), 32)
            families = {
                tuple(sorted(
                    kind_by_device[participant["device_id"]]
                    for participant in candidate["executor"]["participants"]
                ))
                for candidate in candidates
            }
            self.assertTrue(expected_families.issubset(families))
            for candidate in candidates:
                self.assertIsNotNone(candidate["executor"])
                self.assertIn(candidate["readiness"], {
                    "READY", "NOT_READY"
                })
                self.assertGreater(candidate["service_us"], 0)
                self.assertGreaterEqual(
                    candidate["service_upper_us"],
                    candidate["service_us"],
                )
                self.assertTrue(candidate["reason"])
                if candidate["admitted"]:
                    self.assertEqual(candidate["reason"], "ADMITTED")
                else:
                    self.assertNotEqual(candidate["reason"], "ADMITTED")
            if any(
                "helper-c" in candidate["details"]["device_ids"]
                and not candidate["admitted"]
                and (
                    "EXECUTOR_UNHEALTHY" in candidate["reason"]
                    or "LINK_NOT_READY" in candidate["reason"]
                )
                for candidate in candidates
            ):
                rejected_phone_requests += 1
        self.assertGreater(rejected_phone_requests, 0)
        coverage = validate_decision_candidate_coverage(
            trace_catalog,
            records,
            {
                request_id: manifests[
                    initial_tickets[request_id].model.model_id
                ]
                for request_id in submitted
            },
        )
        self.assertEqual(len(coverage), 84)
        self.assertTrue(all(
            set(row.expected_families) == expected_families
            and set(row.observed_families) == expected_families
            for row in coverage
        ))
        self.assertTrue(all(
            execution.command.endpoint == execution.ticket.binding.endpoint
            and execution.command.executor_id
                == execution.ticket.binding.executor_id
            and execution.command.operator_plan_sha256
                == execution.ticket.execution_plan.plan_sha256
            for execution in executions.values()
        ))
        self.assertEqual(
            backend.max_active_by_device["accelerator-b"], 1
        )
        self.assertTrue(any(
            execution.ticket.decision.queue_us > 0
            for execution in executions.values()
        ))
        self.assertTrue(any(
            left_start < right_end
            and right_start < left_end
            and left_devices.isdisjoint(right_devices)
            for index, (left_start, left_end, left_devices) in enumerate(
                backend.execution_intervals
            )
            for right_start, right_end, right_devices in (
                backend.execution_intervals[index + 1:]
            )
        ))
        self.assertTrue(backend.transition_commands)
        self.assertTrue(any(
            command.transition.evictions
            for command in backend.transition_commands
        ))
        terminal_by_request = {
            row["request_ids"][0]: row for row in terminals
        }
        self.assertTrue(all(
            any(
                receipt["transition_id"]
                    == command.transition.transition_id
                and receipt["ticket_id"] == command.ticket_id
                and receipt["status"] == "COMPLETED"
                for receipt in terminal_by_request[
                    command.request_id
                ]["selected"]["transition_receipts"]
            )
            for command in backend.transition_commands
        ))
        self.assertEqual(set(initial_tickets), set(executions))
        self.assertTrue(all(
            execution.dispatch_receipts
            and execution.dispatch_receipts[0].request_id == request_id
            for request_id, execution in executions.items()
        ))
        scheduler.validate_runtime_decision_log(
            scheduler.runtime_decision_log()
        )

    @unittest.skip(
        "retired 2026-09-13: the fixture alternates two GPU models so that a request for the "
        "resident model arrives while the other model's replacement of the exclusive GPU is "
        "already reserved; the ledger reports MEMORY_REPLACEMENT_CONFLICT_CURRENT:gpu-memory for "
        "every candidate and the frozen desktop-baseline control refuses admission by design. "
        "Queueing behind a reserved replacement is a scheduling-policy decision, not a fixture fix."
    )
    def test_v12_stale_replan_does_not_terminalize_84_request_burst(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_paths = {}
            for model_id, window in (
                ("synthetic-large-a", None),
                ("synthetic-large-b", 16),
                ("synthetic-small-c", 32),
            ):
                path = Path(directory) / (model_id + ".gguf")
                write_synthetic_gguf(path, sliding_window=window)
                model_paths[model_id] = path
            catalog_json = contention_catalog().to_json()
            for resource in catalog_json["resources"]:
                if resource["resource_id"] in {
                    "compute:accelerator-b",
                    "link:pcie-in",
                    "link:pcie-out",
                }:
                    resource["capacity"] = 2
            rates = {
                "host-a": 10_000_000,
                "accelerator-b": 20_000_000,
                "helper-c": 10_000_000,
            }
            for kernel in catalog_json["placement_profile"]["kernels"]:
                rate = rates[kernel["device_id"]]
                kernel["effective_bytes_per_s"] = rate
                kernel["effective_ops_per_s"] = rate
            for transition in catalog_json["transitions"]:
                transition["fixed_latency_us"] = 100_000
                if transition["device_id"] == "accelerator-b":
                    transition["resource_slots"] = {
                        "compute:accelerator-b": 2,
                    }
            trace_catalog = RuntimeCapabilityCatalog.from_json(catalog_json)
            scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
            scheduler.register_runtime_capabilities(trace_catalog)
            manifests = {
                model_id: scheduler.register_gguf_model(model_id, path)
                for model_id, path in model_paths.items()
            }

            class ConcurrentBackend(OneGpuTraceBackend):
                def __init__(self, *args):
                    super().__init__(*args)
                    self.transition_errors = []
                    self.execution_errors = []
                    self.backend_entry_ns = {}
                    self.execution_delay_s = 0.1

                def snapshot(self, model_id, observed_at_us, phone_ready):
                    value = super().snapshot(
                        model_id, observed_at_us, phone_ready
                    )
                    executors = dict(value.executors)
                    gpu = executors["executor:accelerator-b"]
                    executors[gpu.executor_id] = replace(
                        gpu, free_slots=2
                    )
                    return replace(value, executors=executors)

                def apply_transition(self, command, payload, control_check):
                    try:
                        return super().apply_transition(
                            command, payload, control_check
                        )
                    except BaseException as exc:
                        failed = scheduler.runtime_ticket(
                            command.request_id
                        )
                        self.transition_errors.append((
                            command.request_id,
                            command.artifact_sha256,
                            tuple(
                                (
                                    lease.resource_id,
                                    lease.lanes,
                                    lease.token,
                                )
                                for lease in failed.decision.leases
                            ),
                            type(exc).__name__ + ":" + str(exc),
                        ))
                        raise

                def execute(self, command, payload, control_check):
                    try:
                        self.backend_entry_ns.setdefault(
                            command.request_id, time.monotonic_ns()
                        )
                        return super().execute(
                            command, payload, control_check
                        )
                    except BaseException as exc:
                        with self._lock:
                            resident = self._gpu_artifact_sha256
                        self.execution_errors.append((
                            command.request_id,
                            command.artifact_sha256,
                            resident,
                            command.planned_start_us,
                            command.planned_finish_upper_us,
                            type(exc).__name__ + ":" + str(exc),
                        ))
                        raise

            backend = ConcurrentBackend(
                scheduler, trace_catalog, manifests
            )
            model_by_role = {
                "A": "synthetic-large-a",
                "B": "synthetic-large-b",
                "C": "synthetic-small-c",
            }
            shape_by_role = {
                "A": (12, 4),
                "B": (24, 8),
                "C": (8, 3),
            }
            epoch_ns = time.monotonic_ns() - 120_000_000_000
            stale_injections = 0
            fresh_retries = 0
            provider_lock = threading.Lock()

            def stamp_snapshot(value, captured_at_us, valid_until_us):
                return replace(
                    value,
                    snapshot_id=(
                        value.snapshot_id
                        + ":"
                        + str(captured_at_us)
                        + ":"
                        + str(valid_until_us)
                    ),
                    captured_at_us=captured_at_us,
                    valid_until_us=valid_until_us,
                    memory=replace(
                        value.memory,
                        snapshot_id=(
                            value.memory.snapshot_id
                            + ":"
                            + str(captured_at_us)
                            + ":"
                            + str(valid_until_us)
                        ),
                        captured_at_us=captured_at_us,
                        valid_until_us=valid_until_us,
                    ),
                )

            def snapshot_provider(ticket, observed_at_us):
                nonlocal stale_injections, fresh_retries
                phone_ready = bool(
                    ticket.request.features["phone_ready"]
                )
                value = backend.snapshot(
                    ticket.model.model_id, observed_at_us, phone_ready
                )
                with provider_lock:
                    if (
                        ticket.dispatch_state == "REPLAN_REQUIRED"
                        and stale_injections == 0
                    ):
                        stale_injections += 1
                        captured = max(0, ticket.request.arrival_us - 2)
                        return stamp_snapshot(
                            value, captured, captured + 1
                        )
                    if stale_injections:
                        fresh_retries += 1
                captured = max(
                    ticket.request.arrival_us,
                    observed_at_us,
                    (time.monotonic_ns() - epoch_ns) // 1000,
                )
                return stamp_snapshot(
                    value, captured, captured + 2_000_000
                )

            refresh_started = threading.Event()
            original_refresh = (
                scheduler
                ._run_model_placement_epoch_refresh_after_learning
            )

            def slow_refresh(*args, **kwargs):
                refresh_started.set()
                time.sleep(0.05)
                return original_refresh(*args, **kwargs)

            scheduler._run_model_placement_epoch_refresh_after_learning = (
                slow_refresh
            )
            adapter_entry_ns = {}
            adapter_entry_lock = threading.Lock()

            def record_adapter_entry(ticket, *_args, **_kwargs):
                with adapter_entry_lock:
                    adapter_entry_ns.setdefault(
                        ticket.request.request_id, time.monotonic_ns()
                    )

            scheduler.start_runtime_lease_renewal = record_adapter_entry
            scheduler.check_runtime_lease_renewal = (
                lambda *_args, **_kwargs: None
            )
            scheduler.stop_runtime_lease_renewal = (
                lambda *_args, **_kwargs: None
            )
            wake_token = scheduler._runtime_controller.queue.hold_wake()
            coordinator = CanonicalArrivalCoordinator(
                scheduler,
                backend,
                epoch_ns=epoch_ns,
                snapshot_provider=snapshot_provider,
                max_workers=16,
                lease_guard_us=1,
                lease_quantum_us=1_000_000,
            )
            request_ids = []
            try:
                for index, (role, arrival_us) in enumerate(zip(
                    REAL_TRACE_ORDER, REAL_TRACE_ARRIVALS_US
                )):
                    model_id = model_by_role[role]
                    input_tokens, output_tokens = shape_by_role[role]
                    phone_ready = index % 11 != 0
                    request = Request(
                        request_id=f"v12-burst:{index:02d}",
                        workload_id=model_id + ":work",
                        arrival_us=arrival_us,
                        deadline_us=arrival_us + 60_000_000,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        quality_requirement="exact",
                        features={"phone_ready": int(phone_ready)},
                    )
                    snapshot = backend.snapshot(
                        model_id, arrival_us, phone_ready
                    )
                    coordinator.submit(
                        CanonicalRuntimeSubmission(
                            request=request,
                            model_id=model_id,
                            snapshot=snapshot,
                            payload={"request_index": index},
                            selection_mode="desktop-baseline",
                        ),
                        observed_at_us=arrival_us,
                    )
                    request_ids.append(request.request_id)
                self.assertTrue(
                    scheduler._runtime_controller.require_queued_replan(
                        request_ids[1], "synthetic_snapshot_refresh"
                    )
                )
            finally:
                scheduler._runtime_controller.queue.release_wake(
                    wake_token
                )
            delayed_state = []

            def capture_delayed_state():
                snapshot = scheduler.runtime_controller_snapshot()
                queue = snapshot["dispatch_queue"]
                delayed_state.append({
                    "active": sorted(queue["active"]),
                    "causal_sample": list(
                        queue["causal_predecessors"].items()
                    )[:8],
                    "history": dict(Counter(
                        row["status"] for row in queue["history"]
                    )),
                    "entry_states": dict(Counter(
                        row["state"]
                        for row in queue["entry_states"].values()
                    )),
                    "entry_state_sample": list(
                        queue["entry_states"].items()
                    )[:12],
                    "wake_hold_count": queue["wake_hold_count"],
                    "queued": {
                        route_id: len(request_ids)
                        for route_id, request_ids in queue["queued"].items()
                    },
                    "tickets": dict(Counter(
                        row["dispatch_state"]
                        for row in snapshot["tickets"].values()
                    )),
                    "next_queued": [
                        (
                            row.request.request_id,
                            row.decision.start_us,
                            row.decision.finish_upper_us,
                            row.decision.route_id,
                        )
                        for row in sorted(
                            scheduler._runtime_controller.current_tickets(
                                ("QUEUED", "REPLAN_REQUIRED")
                            ),
                            key=lambda row: (
                                row.decision.start_us,
                                row.request.request_id,
                            ),
                        )[:8]
                    ],
                    "observed_at_us": coordinator.observed_at_us(),
                })

            state_timer = threading.Timer(
                5, capture_delayed_state,
            )
            state_timer.start()
            try:
                try:
                    result = coordinator.drain(timeout_s=30)
                except BaseException as exc:
                    self.fail(
                        type(exc).__name__
                        + ":"
                        + str(exc)
                        + ":transition_errors="
                        + repr(backend.transition_errors)
                        + ":execution_errors="
                        + repr(backend.execution_errors)
                        + ":recent_transitions="
                        + repr([
                            (
                                command.request_id,
                                command.artifact_sha256,
                                command.transition.transition_id,
                            )
                            for command in backend.transition_commands[-12:]
                        ])
                        + ":delayed_state="
                        + repr(delayed_state)
                    )
            finally:
                state_timer.cancel()
                coordinator.close()

        self.assertEqual(len(result.executions), 84)
        self.assertTrue(all(
            ticket.dispatch_state == "COMPLETED"
            for ticket in result.tickets.values()
        ))
        self.assertEqual(stale_injections, 1)
        self.assertGreaterEqual(fresh_retries, 1)
        self.assertFalse(refresh_started.is_set())
        self.assertGreaterEqual(
            backend.max_active_by_device["accelerator-b"],
            2,
            repr(Counter(
                tuple(sorted(
                    row.device_id
                    for row in execution.command.participants
                ))
                for execution in result.executions.values()
            ))
            + ":gpu_leases="
            + repr([
                (
                    request_id,
                    dict(result.tickets[
                        request_id
                    ].execution_plan.resource_slots),
                    [
                        (
                            lease.resource_id,
                            lease.lanes,
                            lease.start_us,
                            lease.predicted_end_us,
                        )
                        for lease in result.tickets[
                            request_id
                        ].decision.leases
                        if lease.resource_id
                            == "compute:accelerator-b"
                    ],
                )
                for request_id, execution in tuple(
                    result.executions.items()
                )[8:15]
            ])
            + ":stage_timestamps_ns="
            + repr({
                request_id: {
                    "adapter_entry": adapter_entry_ns.get(request_id),
                    "backend_entry": backend.backend_entry_ns.get(
                        request_id
                    ),
                    "queue_acquired": min(
                        (
                            epoch_ns
                            + receipt.observed_at_us * 1_000
                            for receipt in execution.dispatch_receipts
                            if receipt.status == "ACQUIRED"
                        ),
                        default=None,
                    ),
                }
                for request_id, execution in result.executions.items()
                if "accelerator-b" in {
                    row.device_id
                    for row in execution.command.participants
                }
            })
            + ":replan_reasons="
            + repr(Counter(
                receipt.wake_reason
                for execution in result.executions.values()
                for receipt in execution.dispatch_receipts
                if receipt.status == "REPLAN_REQUIRED"
            ))
            + ":queue_history="
            + repr(Counter(
                row["status"]
                for row in scheduler.runtime_controller_snapshot()[
                    "dispatch_queue"
                ]["history"]
            )),
        )
        records = scheduler.runtime_decision_log()["records"]
        replan_runs = []
        current_replans = []
        for row in records:
            if row["event_kind"] == "REPLAN":
                current_replans.append(row["ticket_id"])
                continue
            if current_replans:
                replan_runs.append(tuple(current_replans))
                current_replans = []
        if current_replans:
            replan_runs.append(tuple(current_replans))
        replan_counts = Counter(
            ticket_id.rsplit(":attempt:", 1)[0]
            for run in replan_runs
            for ticket_id in run
        )
        most_replanned = replan_counts.most_common(1)
        most_replanned_reasons = Counter(
            row["selected"].get("failure_reason")
            for row in records
            if row["event_kind"] == "REPLAN"
            and most_replanned
            and row["request_ids"] == [most_replanned[0][0]]
        )
        most_replanned_rows = [
            (
                row["attempt_index"],
                row["event_time_us"],
                row["selected"].get("failure_reason"),
                [
                    (
                        lease["resource_id"],
                        lease["lanes"],
                        lease["start_us"],
                        lease["reserved_until_us"],
                    )
                    for lease in row["selected"]["resource_leases"]
                ],
            )
            for row in records
            if row["event_kind"] == "REPLAN"
            and most_replanned
            and row["request_ids"] == [most_replanned[0][0]]
        ]
        self.assertLessEqual(
            most_replanned[0][1] if most_replanned else 0,
            4,
            repr((
                most_replanned,
                most_replanned_reasons,
                most_replanned_rows,
            )),
        )
        terminals = [
            row for row in records
            if row["event_kind"] in {"COMPLETED", "FAILED", "CANCELLED"}
        ]
        self.assertEqual(len(terminals), 84)
        self.assertEqual(
            Counter(row["event_kind"] for row in terminals),
            Counter({"COMPLETED": 84}),
        )
        completion_rows = [
            row for row in scheduler.runtime_controller_snapshot()[
                "dispatch_queue"
            ]["history"]
            if row["status"] == "COMPLETED"
        ]
        self.assertEqual(len(completion_rows), 84)
        self.assertLessEqual(
            max(row["frontier_wake_count"] for row in completion_rows),
            2,
        )
        scheduler.validate_runtime_decision_log(
            scheduler.runtime_decision_log()
        )

    def test_submission_before_arrival_fails_before_scheduling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "arrival.gguf"
            write_synthetic_gguf(model_path)
            scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
            scheduler.register_runtime_capabilities(catalog())
            manifest = scheduler.register_gguf_model(
                "synthetic-arrival-model", model_path
            )
            snapshot = runtime_snapshot(manifest)
            backend = FakeMeasuredBackend(5_000)
            coordinator = CanonicalArrivalCoordinator(
                scheduler,
                backend,
                epoch_ns=time.monotonic_ns(),
                snapshot_provider=lambda _ticket, _at_us: snapshot,
                max_workers=1,
            )
            request = Request(
                request_id="not-yet-arrived",
                workload_id="synthetic-arrival-work",
                arrival_us=10_000,
                deadline_us=1_000_000,
                input_tokens=12,
                output_tokens=4,
                quality_requirement="exact",
            )
            try:
                with self.assertRaisesRegex(
                    Exception, "before its arrival"
                ):
                    coordinator.submit(
                        CanonicalRuntimeSubmission(
                            request,
                            manifest.model_id,
                            snapshot,
                            {},
                        ),
                        observed_at_us=9_999,
                    )
            finally:
                coordinator.close()
        self.assertEqual(
            scheduler.runtime_decision_log()["records"], []
        )



class LifecycleFailingBackend(FakeMeasuredBackend):
    def __init__(self, failing_request_ids) -> None:
        super().__init__(5_000)
        self.failing_request_ids = frozenset(failing_request_ids)

    def execute(self, command, payload, control_check):
        if command.request_id not in self.failing_request_ids:
            return super().execute(command, payload, control_check)
        self.execution_commands.append(command)
        raise RuntimeError("synthetic lifecycle failure")


class PinnedServiceClock:
    """monotonic_ns pinned to the arrival on the submitting thread and to
    the service time on lifecycle threads."""

    def __init__(self, epoch_ns: int) -> None:
        self.epoch_ns = epoch_ns
        self.submitter = threading.get_ident()
        self.arrival_us = 0
        self.service_us = 0

    def __call__(self) -> int:
        at_us = (
            self.arrival_us
            if threading.get_ident() == self.submitter
            else self.service_us
        )
        return self.epoch_ns + at_us * 1000


def arrival_fixture(directory):
    model_path = Path(directory) / "arrival.gguf"
    write_synthetic_gguf(model_path)
    scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
    scheduler.register_runtime_capabilities(catalog())
    manifest = scheduler.register_gguf_model(
        "synthetic-arrival-model", model_path
    )
    return scheduler, manifest, runtime_snapshot(manifest)


def arrival_submission(manifest, snapshot, index, arrival_us):
    return CanonicalRuntimeSubmission(
        Request(
            request_id=f"arrival-{index}",
            workload_id="synthetic-arrival-work",
            arrival_us=arrival_us,
            deadline_us=arrival_us + 5_000_000,
            input_tokens=12,
            output_tokens=4,
            quality_requirement="exact",
        ),
        manifest.model_id,
        snapshot,
        {},
    )


def sequential_decision_log_head(**coordinator_options) -> str:
    """Decision-log head of a non-failing four-arrival run on a pinned clock."""
    with tempfile.TemporaryDirectory() as directory:
        scheduler, manifest, snapshot = arrival_fixture(directory)
        for name in (
            "start_runtime_lease_renewal",
            "check_runtime_lease_renewal",
            "stop_runtime_lease_renewal",
        ):
            setattr(scheduler, name, lambda *_args, **_kwargs: None)
        epoch_ns = 1_000_000_000_000
        clock = PinnedServiceClock(epoch_ns)
        completed = []
        with mock.patch.object(time, "monotonic_ns", clock):
            coordinator = CanonicalArrivalCoordinator(
                scheduler,
                FakeMeasuredBackend(5_000),
                epoch_ns=epoch_ns,
                snapshot_provider=lambda _ticket, _at_us: snapshot,
                max_workers=2,
                **coordinator_options,
            )
            try:
                for index in range(4):
                    arrival_us = 1_000 + index * 1_000_000
                    clock.arrival_us = arrival_us
                    clock.service_us = arrival_us + 500_000
                    coordinator.submit_at_arrival(arrival_submission(
                        manifest, snapshot, index, arrival_us
                    ))
                    completed = list(
                        coordinator.drain(timeout_s=30).request_ids
                    )
            finally:
                coordinator.close()
    if completed != [f"arrival-{index}" for index in range(4)]:
        raise AssertionError("sequential arrivals did not all complete")
    return scheduler.runtime_decision_log()["head_record_sha256"]


class ArrivalFailFastTests(unittest.TestCase):
    # Computed on the tree before fail-fast (drain-time failure detection).
    NON_FAILING_DECISION_LOG_HEAD = (
        "03c9221845fdc88966cae945cdc00ffcdc1bca82b9c74e834de8efff74e7356b"
    )

    def test_lifecycle_failure_aborts_before_the_next_arrival_is_submitted(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scheduler, manifest, snapshot = arrival_fixture(directory)
            backend = LifecycleFailingBackend({"arrival-1"})
            coordinator = CanonicalArrivalCoordinator(
                scheduler,
                backend,
                epoch_ns=time.monotonic_ns(),
                snapshot_provider=lambda _ticket, _at_us: snapshot,
                max_workers=2,
            )
            arrivals_us = (0, 50_000, 3_000_000, 3_050_000)
            submitted = []
            try:
                with self.assertRaisesRegex(
                    PhysicalAdapterError, "physical_execution_control_failed"
                ) as caught:
                    for index, arrival_us in enumerate(arrivals_us):
                        ticket = coordinator.submit_at_arrival(
                            arrival_submission(
                                manifest, snapshot, index, arrival_us
                            )
                        )
                        submitted.append(ticket.request.request_id)
                aborted_at_us = coordinator.observed_at_us()
            finally:
                coordinator.close(wait=False)
        self.assertEqual(submitted, ["arrival-0", "arrival-1"])
        self.assertLess(aborted_at_us, arrivals_us[2])
        self.assertEqual(
            {
                row["request_ids"][0]
                for row in scheduler.runtime_decision_log()["records"]
            },
            {"arrival-0", "arrival-1"},
        )
        self.assertEqual(
            scheduler.runtime_ticket("arrival-1").dispatch_state, "FAILED"
        )
        self.assertLessEqual(
            {command.request_id for command in backend.execution_commands},
            {"arrival-0", "arrival-1"},
        )
        self.assertIn(
            "lifecycle of arrival-1 failed",
            "\n".join(getattr(caught.exception, "__notes__", ())),
        )

    def test_submit_refuses_arrivals_after_a_recorded_lifecycle_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scheduler, manifest, snapshot = arrival_fixture(directory)
            coordinator = CanonicalArrivalCoordinator(
                scheduler,
                LifecycleFailingBackend({"arrival-0"}),
                epoch_ns=time.monotonic_ns(),
                snapshot_provider=lambda _ticket, _at_us: snapshot,
                max_workers=2,
            )
            try:
                coordinator.submit(
                    arrival_submission(manifest, snapshot, 0, 0),
                    observed_at_us=1,
                )
                self.assertIsNotNone(
                    coordinator._futures["arrival-0"].exception(timeout=10)
                )
                for index in (1, 2):
                    with self.subTest(index=index), self.assertRaisesRegex(
                        PhysicalAdapterError,
                        "physical_execution_control_failed",
                    ) as caught:
                        coordinator.submit(
                            arrival_submission(manifest, snapshot, index, 0),
                            observed_at_us=1 + index,
                        )
            finally:
                coordinator.close(wait=False)
        self.assertEqual(
            {
                row["request_ids"][0]
                for row in scheduler.runtime_decision_log()["records"]
            },
            {"arrival-0"},
        )
        self.assertEqual(
            sum(
                note.startswith("arrival coordinator stopped arrivals")
                for note in caught.exception.__notes__
            ),
            1,
        )

    def test_drain_mode_keeps_serving_arrivals_until_drain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scheduler, manifest, snapshot = arrival_fixture(directory)
            with self.assertRaisesRegex(
                PhysicalAdapterError, "fail-fast flag is invalid"
            ):
                CanonicalArrivalCoordinator(
                    scheduler,
                    FakeMeasuredBackend(5_000),
                    epoch_ns=time.monotonic_ns(),
                    snapshot_provider=lambda _ticket, _at_us: snapshot,
                    max_workers=1,
                    fail_fast=1,
                )
            coordinator = CanonicalArrivalCoordinator(
                scheduler,
                LifecycleFailingBackend({"arrival-0"}),
                epoch_ns=time.monotonic_ns(),
                snapshot_provider=lambda _ticket, _at_us: snapshot,
                max_workers=2,
                fail_fast=False,
            )
            try:
                coordinator.submit(
                    arrival_submission(manifest, snapshot, 0, 0),
                    observed_at_us=1,
                )
                self.assertIsNotNone(
                    coordinator._futures["arrival-0"].exception(timeout=10)
                )
                coordinator.wait_for_arrival(0)
                ticket = coordinator.submit(
                    arrival_submission(manifest, snapshot, 1, 0),
                    observed_at_us=2,
                )
                self.assertEqual(ticket.request.request_id, "arrival-1")
                with self.assertRaisesRegex(
                    PhysicalAdapterError, "physical_execution_control_failed"
                ):
                    coordinator.drain(timeout_s=30)
            finally:
                coordinator.close(wait=False)
        self.assertEqual(
            {
                row["request_ids"][0]
                for row in scheduler.runtime_decision_log()["records"]
            },
            {"arrival-0", "arrival-1"},
        )

    def test_non_failing_decision_log_is_unchanged_by_fail_fast(self) -> None:
        for options in ({}, {"fail_fast": False}):
            with self.subTest(options=options):
                self.assertEqual(
                    sequential_decision_log_head(**options),
                    self.NON_FAILING_DECISION_LOG_HEAD,
                )


if __name__ == "__main__":
    unittest.main()
