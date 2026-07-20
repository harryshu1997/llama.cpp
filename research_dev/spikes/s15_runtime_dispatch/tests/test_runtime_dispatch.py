#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
S15 = HERE.parent
S14 = S15.parent / "s14_mixed_streaming_scheduler"
sys.path.insert(0, str(S15))
sys.path.insert(0, str(S14))

import route_fixtures as F  # noqa: E402
from power_frontier_policy import (  # noqa: E402
    BatchDecision,
    BoundaryCertificate,
    CertifiedBatchPoint,
    WorkItem,
)
from priority_batch_runtime import Launch, RouteConfig  # noqa: E402

from executor_contract import (  # noqa: E402
    Executor,
    ExecutorError,
    ExecutionResult,
    RecordedExecutor,
    RecordedOutcome,
    recorded_key,
)
from route_registry import ReadyRouteRegistry, RouteSnapshot, route_content_digest  # noqa: E402
from runtime_dispatch import (  # noqa: E402
    DispatchError,
    LaneBinding,
    MixedDispatchCoordinator,
)

OP15_ROUTE = "op15-gemma-head-0-8"
OP12_ROUTE = "op12-gemma-head-0-6"
GEN = "generation"
MODEL = "gemma-4-12b-it-f16"
MANIFEST = "sha256:" + "ef" * 32
COHORT = "sha256:" + "de" * 32


def phone_lanes() -> tuple[LaneBinding, ...]:
    return (
        LaneBinding("op15_gemma_head", OP15_ROUTE, "phone", timeout_us=4_000_000, cohort_sha256=COHORT, input_manifest_sha256=MANIFEST),
        LaneBinding("op12_gemma_head", OP12_ROUTE, "phone", timeout_us=4_000_000, cohort_sha256=COHORT, input_manifest_sha256=MANIFEST),
    )


def gemma_item(request_id, island, *, arrival=0, deadline=10_000_000, priority=1, key=None):
    return WorkItem(request_id, GEN, MODEL, island,
                    key or f"{MODEL}|decode|{island}", arrival, deadline, priority)


def completed_script(registry, entries):
    script = {}
    for route_id, request_ids in entries:
        snapshot = registry.get(route_id)
        duration = min(point.duration_us for point in snapshot.config.points)
        script[recorded_key(route_id, tuple(request_ids))] = RecordedOutcome(
            outcome="completed", finish_delay_us=duration,
            profile_id=snapshot.profile_id, route_epoch=snapshot.route_epoch,
            residency_epoch=snapshot.residency_epoch,
            device_boot_epoch=snapshot.device_boot_epoch,
            cohort_sha256=COHORT, input_manifest_sha256=MANIFEST,
        )
    return RecordedExecutor(script)


def syn_memory_snapshot(route_id="syn-mem", island="gemma-head-syn",
                        high_priority_max=0, credits=4) -> RouteSnapshot:
    digest = "sha256:" + "5a" * 32
    points = tuple(
        CertifiedBatchPoint(batch, duration, f"{digest}#correct", f"{digest}#placement")
        for batch, duration in ((1, 100), (2, 120), (4, 150))
    )
    config = RouteConfig(route_id, GEN, MODEL, island, digest, 4,
                         "memory_bound", points, high_priority_max)
    return RouteSnapshot(
        config=config, content_digest=route_content_digest(config),
        device_id=F.OP15_DEVICE, layer_range=(0, 8), state="READY",
        residency_epoch=1, lease_epoch=1, device_boot_epoch=1,
        thermal_ceiling_millic=95_000, thermal_observed_millic=60_000,
        execution_credits=credits, compound_kind="SINGLE", correctness_certified=True,
    )


class IndependentLaneTests(unittest.TestCase):
    def _coordinator(self):
        registry = ReadyRouteRegistry()
        registry.install([F.op15_head_snapshot(), F.op12_head_snapshot()])
        executor = completed_script(registry, [(OP15_ROUTE, ("r0",)), (OP12_ROUTE, ("r1",))])
        return MixedDispatchCoordinator(registry, executor, phone_lanes(), queue_capacity=8)

    def test_both_phone_routes_reach_completed_phone(self) -> None:
        coordinator = self._coordinator()
        coordinator.admit(gemma_item("r0", "gemma-head-0-8"), 0)
        coordinator.admit(gemma_item("r1", "gemma-head-0-6"), 0)
        self.assertEqual(coordinator.pending_count(OP15_ROUTE), 1)
        self.assertEqual(coordinator.pending_count(OP12_ROUTE), 1)
        launch15 = coordinator.dispatch(OP15_ROUTE, 0)
        # dispatching OP15 must not touch the OP12 lane (no serial chain)
        self.assertIsInstance(launch15, Launch)
        self.assertEqual(coordinator.pending_count(OP12_ROUTE), 1)
        coordinator.dispatch(OP12_ROUTE, 0)
        self.assertEqual(coordinator.terminal_of("r0"), "completed_phone")
        self.assertEqual(coordinator.terminal_of("r1"), "completed_phone")
        coordinator.assert_conservation(["r0", "r1"])


class MeasuredB32PolicyTests(unittest.TestCase):
    def test_op15_launches_b32_within_five_second_slo(self) -> None:
        snapshot = F.op15_b32_snapshot()
        registry = ReadyRouteRegistry()
        registry.install([snapshot])
        request_ids = tuple(f"r{index:02d}" for index in range(32))
        b32 = next(point for point in snapshot.config.points if point.batch_size == 32)
        script = {
            recorded_key(OP15_ROUTE, request_ids): RecordedOutcome(
                "completed", b32.duration_us, snapshot.profile_id,
                snapshot.route_epoch, snapshot.residency_epoch,
                snapshot.device_boot_epoch, COHORT, MANIFEST,
            )
        }
        coordinator = MixedDispatchCoordinator(
            registry,
            RecordedExecutor(script),
            (LaneBinding("op15_gemma_head", OP15_ROUTE, "phone", timeout_us=4_000_000, cohort_sha256=COHORT, input_manifest_sha256=MANIFEST),),
            queue_capacity=32,
        )
        for request_id in request_ids:
            coordinator.admit(
                gemma_item(request_id, "gemma-head-0-8", deadline=5_000_000, key="K"), 0
            )
        launch = coordinator.dispatch(OP15_ROUTE, 0)
        self.assertIsInstance(launch, Launch)
        self.assertEqual(launch.batch_size, 32)
        self.assertEqual(launch.reason, "target_batch_ready")
        self.assertTrue(all(coordinator.terminal_of(value) == "completed_phone"
                            for value in request_ids))

    def test_op12_does_not_launch_b32_past_five_second_slo(self) -> None:
        snapshot = F.op12_b32_snapshot()
        registry = ReadyRouteRegistry()
        registry.install([snapshot])
        coordinator = MixedDispatchCoordinator(
            registry,
            RecordedExecutor({}),
            (LaneBinding("op12_gemma_head", OP12_ROUTE, "phone", timeout_us=12_000_000, cohort_sha256=COHORT, input_manifest_sha256=MANIFEST),),
            queue_capacity=32,
        )
        for index in range(32):
            coordinator.admit(
                gemma_item(f"r{index:02d}", "gemma-head-0-6", deadline=5_000_000, key="K"), 0
            )
        launch = coordinator.dispatch(OP12_ROUTE, 0)
        self.assertIsInstance(launch, Launch)
        self.assertEqual(launch.batch_size, 1)
        self.assertEqual(launch.reason, "latest_start_reached")


class CompoundRouteTests(unittest.TestCase):
    def test_shared_tail_route_never_dispatches(self) -> None:
        registry = ReadyRouteRegistry()
        registry.install([F.shared_tail_snapshot(state="UNAVAILABLE")])
        route_id = "twophone-sharedtail-0-12"
        lanes = (LaneBinding("compound", route_id, "phone", timeout_us=4_000_000, cohort_sha256=COHORT, input_manifest_sha256=MANIFEST),)
        coordinator = MixedDispatchCoordinator(registry, RecordedExecutor({}), lanes, queue_capacity=8)
        result = coordinator.admit(gemma_item("r0", "gemma-head-0-12"), 0)
        self.assertEqual(result.disposition, "server_fallback")
        self.assertEqual(result.reason, "route_not_ready")
        self.assertEqual(coordinator.terminal_of("r0"), "completed_server")
        decision = coordinator.dispatch(route_id, 0)
        self.assertIsInstance(decision, BatchDecision)
        self.assertEqual(decision.action, "NO_WORK")


class PriorityIsolationTests(unittest.TestCase):
    def _coordinator(self, executor=None):
        registry = ReadyRouteRegistry()
        registry.install([syn_memory_snapshot()])
        executor = executor or RecordedExecutor({})
        lanes = (LaneBinding("syn", "syn-mem", "phone", timeout_us=1_000_000, cohort_sha256=COHORT, input_manifest_sha256=MANIFEST),)
        return registry, MixedDispatchCoordinator(registry, executor, lanes, queue_capacity=16)

    def test_higher_priority_key_is_served_first_without_mixing(self) -> None:
        _, coordinator = self._coordinator()
        for index in range(2):
            coordinator.admit(gemma_item(f"a{index}", "gemma-head-syn", priority=0, key="A"), 0)
            coordinator.admit(gemma_item(f"b{index}", "gemma-head-syn", priority=1, key="B"), 0)
        decision = coordinator.dispatch("syn-mem", 0)
        self.assertIsInstance(decision, Launch)
        self.assertEqual(decision.compatibility_key, "A")
        self.assertEqual(set(decision.request_ids), {"a0", "a1"})

    def test_bounded_slo_wait_then_launch(self) -> None:
        _, coordinator = self._coordinator()
        for index in range(2):
            coordinator.admit(gemma_item(f"r{index}", "gemma-head-syn", priority=1,
                                         key="K", deadline=10_000), 0)
        wait = coordinator.dispatch("syn-mem", 0)
        self.assertIsInstance(wait, BatchDecision)
        self.assertEqual(wait.action, "WAIT")
        self.assertGreater(wait.next_wake_us, 0)
        launch = coordinator.dispatch("syn-mem", wait.next_wake_us)
        self.assertIsInstance(launch, Launch)
        self.assertEqual(launch.batch_size, 2)

    def test_exact_per_request_boundary_ownership(self) -> None:
        registry = ReadyRouteRegistry()
        registry.install([syn_memory_snapshot()])
        snapshot = registry.get("syn-mem")
        script = {recorded_key("syn-mem", ("r0", "r1")): RecordedOutcome(
            "completed", 120, snapshot.profile_id, snapshot.route_epoch,
            snapshot.residency_epoch, snapshot.device_boot_epoch, COHORT, MANIFEST,
            admit={"r1": False},
        )}
        lanes = (LaneBinding("syn", "syn-mem", "phone", timeout_us=1_000_000, cohort_sha256=COHORT, input_manifest_sha256=MANIFEST),)
        coordinator = MixedDispatchCoordinator(registry, RecordedExecutor(script), lanes, queue_capacity=16)
        for index in range(2):
            coordinator.admit(gemma_item(f"r{index}", "gemma-head-syn", priority=1,
                                         key="K", deadline=10_000), 0)
        wait = coordinator.dispatch("syn-mem", 0)
        coordinator.dispatch("syn-mem", wait.next_wake_us)
        self.assertEqual(coordinator.terminal_of("r0"), "completed_phone")
        self.assertEqual(coordinator.terminal_of("r1"), "fallback_required")


class FallbackAndBackpressureTests(unittest.TestCase):
    def _coordinator(self):
        registry = ReadyRouteRegistry()
        registry.install([F.op15_head_snapshot()])
        lanes = (LaneBinding("op15_gemma_head", OP15_ROUTE, "phone", timeout_us=4_000_000, cohort_sha256=COHORT, input_manifest_sha256=MANIFEST),)
        return registry, MixedDispatchCoordinator(
            registry, RecordedExecutor({}), lanes, queue_capacity=2)

    def test_infeasible_deadline_takes_immediate_server_fallback(self) -> None:
        _, coordinator = self._coordinator()
        result = coordinator.admit(gemma_item("r0", "gemma-head-0-8", deadline=1000), 0)
        self.assertEqual(result.disposition, "server_fallback")
        self.assertEqual(result.reason, "latest_start_infeasible")
        self.assertEqual(coordinator.terminal_of("r0"), "completed_server")
        self.assertEqual(coordinator.pending_count(OP15_ROUTE), 0)

    def test_no_matching_route_takes_server_fallback(self) -> None:
        _, coordinator = self._coordinator()
        result = coordinator.admit(gemma_item("r0", "gemma-head-0-99"), 0)
        self.assertEqual(result.reason, "no_matching_route")
        self.assertEqual(coordinator.terminal_of("r0"), "completed_server")

    def test_queue_capacity_rejects_backpressure(self) -> None:
        _, coordinator = self._coordinator()
        coordinator.admit(gemma_item("r0", "gemma-head-0-8"), 0)
        coordinator.admit(gemma_item("r1", "gemma-head-0-8"), 0)
        result = coordinator.admit(gemma_item("r2", "gemma-head-0-8"), 0)
        self.assertEqual(result.disposition, "rejected_backpressure")
        self.assertEqual(coordinator.terminal_of("r2"), "rejected_backpressure")

    def test_duplicate_admission_is_rejected(self) -> None:
        _, coordinator = self._coordinator()
        coordinator.admit(gemma_item("r0", "gemma-head-0-8"), 0)
        with self.assertRaisesRegex(DispatchError, "already admitted"):
            coordinator.admit(gemma_item("r0", "gemma-head-0-8"), 0)


class ExecutorOutcomeTests(unittest.TestCase):
    def _coordinator(self, executor):
        registry = ReadyRouteRegistry()
        registry.install([F.op15_head_snapshot()])
        lanes = (LaneBinding("op15_gemma_head", OP15_ROUTE, "phone", timeout_us=4_000_000, cohort_sha256=COHORT, input_manifest_sha256=MANIFEST),)
        return registry, MixedDispatchCoordinator(registry, executor, lanes, queue_capacity=8)

    def test_unknown_launch_times_out(self) -> None:
        _, coordinator = self._coordinator(RecordedExecutor({}))
        coordinator.admit(gemma_item("r0", "gemma-head-0-8"), 0)
        coordinator.dispatch(OP15_ROUTE, 0)
        self.assertEqual(coordinator.terminal_of("r0"), "timed_out")

    def test_recorded_error_requires_fallback(self) -> None:
        registry = ReadyRouteRegistry()
        registry.install([F.op15_head_snapshot()])
        snapshot = registry.get(OP15_ROUTE)
        script = {recorded_key(OP15_ROUTE, ("r0",)): RecordedOutcome(
            "error", 10, snapshot.profile_id, snapshot.route_epoch,
            snapshot.residency_epoch, snapshot.device_boot_epoch, COHORT, MANIFEST,
        )}
        lanes = (LaneBinding("op15_gemma_head", OP15_ROUTE, "phone", timeout_us=4_000_000, cohort_sha256=COHORT, input_manifest_sha256=MANIFEST),)
        coordinator = MixedDispatchCoordinator(registry, RecordedExecutor(script), lanes, queue_capacity=8)
        coordinator.admit(gemma_item("r0", "gemma-head-0-8"), 0)
        coordinator.dispatch(OP15_ROUTE, 0)
        self.assertEqual(coordinator.terminal_of("r0"), "fallback_required")

    def test_executor_contract_failure_releases_credit_and_requires_fallback(self) -> None:
        class BrokenExecutor(Executor):
            def launch(self, request, now_us):
                raise ExecutorError("invalid physical session record")

        registry, coordinator = self._coordinator(BrokenExecutor())
        coordinator.admit(gemma_item("r0", "gemma-head-0-8"), 0)
        coordinator.dispatch(OP15_ROUTE, 0)
        self.assertEqual(coordinator.terminal_of("r0"), "fallback_required")
        self.assertEqual(registry.outstanding(OP15_ROUTE), 0)

    def test_invalid_boundary_value_requires_fallback(self) -> None:
        class InvalidBoundaryExecutor(Executor):
            def launch(self, request, now_us):
                certificate = BoundaryCertificate(request.request_ids[0], 1, True, True, True)
                return ExecutionResult(
                    request.launch_id, "completed", now_us + 1,
                    request.expected_boundary_schema, (certificate,),
                )

        registry, coordinator = self._coordinator(InvalidBoundaryExecutor())
        coordinator.admit(gemma_item("r0", "gemma-head-0-8"), 0)
        coordinator.dispatch(OP15_ROUTE, 0)
        self.assertEqual(coordinator.terminal_of("r0"), "fallback_required")
        self.assertEqual(registry.outstanding(OP15_ROUTE), 0)

    def test_stale_lease_midflight_requires_fallback(self) -> None:
        registry = ReadyRouteRegistry()
        registry.install([F.op15_head_snapshot()])
        snapshot = registry.get(OP15_ROUTE)
        inner = completed_script(registry, [(OP15_ROUTE, ("r0",))])

        class MidflightMutator(Executor):
            def launch(self, request, now_us):
                registry.install([F.op15_head_snapshot(residency_epoch=2)])
                return inner.launch(request, now_us)

        lanes = (LaneBinding("op15_gemma_head", OP15_ROUTE, "phone", timeout_us=4_000_000, cohort_sha256=COHORT, input_manifest_sha256=MANIFEST),)
        coordinator = MixedDispatchCoordinator(registry, MidflightMutator(), lanes, queue_capacity=8)
        coordinator.admit(gemma_item("r0", "gemma-head-0-8"), 0)
        coordinator.dispatch(OP15_ROUTE, 0)
        self.assertEqual(coordinator.terminal_of("r0"), "fallback_required")


class FourLaneTests(unittest.TestCase):
    def test_server_bge_lane_is_isolated_from_phone_gemma(self) -> None:
        registry = ReadyRouteRegistry()
        registry.install([
            F.op15_head_snapshot(),
            F.synthetic_bge_snapshot(),
        ])
        bge_snapshot = registry.get("server-bge-encoder")
        script = {
            recorded_key(OP15_ROUTE, ("g0",)): RecordedOutcome(
                "completed", 100, registry.get(OP15_ROUTE).profile_id,
                registry.get(OP15_ROUTE).route_epoch, 1, 1, COHORT, MANIFEST),
            recorded_key("server-bge-encoder", ("e0",)): RecordedOutcome(
                "completed", 100, bge_snapshot.profile_id, bge_snapshot.route_epoch,
                1, 1, COHORT, MANIFEST),
        }
        lanes = (
            LaneBinding("op15_gemma_head", OP15_ROUTE, "phone", timeout_us=4_000_000, cohort_sha256=COHORT, input_manifest_sha256=MANIFEST),
            LaneBinding("server_bge", "server-bge-encoder", "server", timeout_us=1_000_000, cohort_sha256=COHORT, input_manifest_sha256=MANIFEST),
        )
        coordinator = MixedDispatchCoordinator(registry, RecordedExecutor(script), lanes, queue_capacity=8)
        coordinator.admit(gemma_item("g0", "gemma-head-0-8"), 0)
        coordinator.admit(WorkItem("e0", "embedding", "bge-small-en-v1.5-f16",
                                   "bge-encoder-0-12", "bge|encode|l32", 0, 1_000_000, 0), 0)
        coordinator.dispatch(OP15_ROUTE, 0)
        coordinator.dispatch("server-bge-encoder", 0)
        self.assertEqual(coordinator.terminal_of("g0"), "completed_phone")
        self.assertEqual(coordinator.terminal_of("e0"), "completed_server")


if __name__ == "__main__":
    unittest.main()
