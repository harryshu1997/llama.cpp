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
from power_frontier_policy import CertifiedBatchPoint, WorkItem  # noqa: E402
from priority_batch_runtime import RouteConfig  # noqa: E402

from executor_contract import RecordedExecutor, RecordedOutcome, recorded_key  # noqa: E402
from route_registry import ReadyRouteRegistry, RouteSnapshot, route_content_digest  # noqa: E402
from runtime_dispatch import (  # noqa: E402
    TERMINAL_STATES,
    LaneBinding,
    MixedDispatchCoordinator,
)

OP15_ROUTE = "op15-gemma-head-0-8"
GEN = "generation"
MODEL = "gemma-4-12b-it-f16"
MANIFEST = "sha256:" + "ef" * 32
COHORT = "sha256:" + "de" * 32


def gemma_item(request_id, island="gemma-head-0-8", *, deadline=10_000_000, priority=1):
    return WorkItem(request_id, GEN, MODEL, island,
                    f"{MODEL}|decode|{island}", 0, deadline, priority)


def op15_coordinator(executor, capacity=8):
    registry = ReadyRouteRegistry()
    registry.install([F.op15_head_snapshot()])
    lanes = (LaneBinding("op15_gemma_head", OP15_ROUTE, "phone", timeout_us=4_000_000, cohort_sha256=COHORT, input_manifest_sha256=MANIFEST),)
    return registry, MixedDispatchCoordinator(registry, executor, lanes, queue_capacity=capacity)


def completed_executor(registry, request_ids=("r0",), finish_delay=None):
    snapshot = registry.get(OP15_ROUTE)
    duration = finish_delay if finish_delay is not None else min(
        point.duration_us for point in snapshot.config.points)
    return RecordedExecutor({recorded_key(OP15_ROUTE, request_ids): RecordedOutcome(
        "completed", duration, snapshot.profile_id, snapshot.route_epoch,
        snapshot.residency_epoch, snapshot.device_boot_epoch, COHORT, MANIFEST)})


def syn_snapshot(route_id="syn-mem", island="gemma-head-syn"):
    digest = "sha256:" + "7c" * 32
    points = (CertifiedBatchPoint(1, 100, f"{digest}#c", f"{digest}#p"),)
    config = RouteConfig(route_id, GEN, MODEL, island, digest, 4, "memory_bound", points, 0)
    return RouteSnapshot(
        config=config, content_digest=route_content_digest(config), device_id=F.OP15_DEVICE,
        layer_range=(0, 8), state="READY", residency_epoch=1, lease_epoch=1, device_boot_epoch=1,
        thermal_ceiling_millic=95_000, thermal_observed_millic=60_000, execution_credits=1,
        compound_kind="SINGLE", correctness_certified=True)


class TerminalStateTests(unittest.TestCase):
    def _completed_phone(self) -> str:
        registry = ReadyRouteRegistry()
        registry.install([F.op15_head_snapshot()])
        coordinator = MixedDispatchCoordinator(
            registry, completed_executor(registry),
            (LaneBinding("op15_gemma_head", OP15_ROUTE, "phone", timeout_us=4_000_000, cohort_sha256=COHORT, input_manifest_sha256=MANIFEST),),
            queue_capacity=8)
        coordinator.admit(gemma_item("r0"), 0)
        coordinator.dispatch(OP15_ROUTE, 0)
        return coordinator.terminal_of("r0")

    def _server(self) -> str:
        _, coordinator = op15_coordinator(RecordedExecutor({}))
        coordinator.admit(gemma_item("r0", deadline=1000), 0)  # infeasible on the phone
        return coordinator.terminal_of("r0")

    def _tardy(self) -> str:
        registry = ReadyRouteRegistry()
        registry.install([syn_snapshot()])
        snapshot = registry.get("syn-mem")
        script = {recorded_key("syn-mem", ("r0",)): RecordedOutcome(
            "completed", 120, snapshot.profile_id, snapshot.route_epoch,
            snapshot.residency_epoch, snapshot.device_boot_epoch, COHORT, MANIFEST)}
        lanes = (LaneBinding("syn", "syn-mem", "phone", timeout_us=1_000_000, cohort_sha256=COHORT, input_manifest_sha256=MANIFEST),)
        coordinator = MixedDispatchCoordinator(registry, RecordedExecutor(script), lanes, queue_capacity=8)
        # feasible at admit (min duration 100 <= 110), but the 120us finish is late
        coordinator.admit(WorkItem("r0", GEN, MODEL, "gemma-head-syn",
                                   "k", 0, 110, 1), 0)
        coordinator.dispatch("syn-mem", 0)
        return coordinator.terminal_of("r0")

    def _fallback(self) -> str:
        registry = ReadyRouteRegistry()
        registry.install([F.op15_head_snapshot()])
        snapshot = registry.get(OP15_ROUTE)
        script = {recorded_key(OP15_ROUTE, ("r0",)): RecordedOutcome(
            "error", 10, snapshot.profile_id, snapshot.route_epoch,
            snapshot.residency_epoch, snapshot.device_boot_epoch, COHORT, MANIFEST)}
        lanes = (LaneBinding("op15_gemma_head", OP15_ROUTE, "phone", timeout_us=4_000_000, cohort_sha256=COHORT, input_manifest_sha256=MANIFEST),)
        coordinator = MixedDispatchCoordinator(registry, RecordedExecutor(script), lanes, queue_capacity=8)
        coordinator.admit(gemma_item("r0"), 0)
        coordinator.dispatch(OP15_ROUTE, 0)
        return coordinator.terminal_of("r0")

    def _timed_out(self) -> str:
        _, coordinator = op15_coordinator(RecordedExecutor({}))
        coordinator.admit(gemma_item("r0"), 0)
        coordinator.dispatch(OP15_ROUTE, 0)
        return coordinator.terminal_of("r0")

    def _backpressure(self) -> str:
        _, coordinator = op15_coordinator(RecordedExecutor({}), capacity=1)
        coordinator.admit(gemma_item("r0"), 0)
        result = coordinator.admit(gemma_item("r1"), 0)
        return result.terminal

    def test_tardy_result_is_reachable(self) -> None:
        self.assertEqual(self._tardy(), "tardy_result")

    def test_all_six_terminal_states_are_reachable_and_distinct(self) -> None:
        observed = {
            self._completed_phone(),
            self._server(),
            self._tardy(),
            self._fallback(),
            self._timed_out(),
            self._backpressure(),
        }
        self.assertEqual(observed, set(TERMINAL_STATES))

    def test_conservation_holds_and_states_are_mutually_exclusive(self) -> None:
        registry = ReadyRouteRegistry()
        registry.install([F.op15_head_snapshot()])
        coordinator = MixedDispatchCoordinator(
            registry, completed_executor(registry, ("keep",)),
            (LaneBinding("op15_gemma_head", OP15_ROUTE, "phone", timeout_us=4_000_000, cohort_sha256=COHORT, input_manifest_sha256=MANIFEST),),
            queue_capacity=8)
        coordinator.admit(gemma_item("keep"), 0)
        coordinator.dispatch(OP15_ROUTE, 0)
        coordinator.admit(gemma_item("srv", deadline=1000), 0)
        coordinator.assert_conservation(["keep", "srv"])
        terminals = coordinator.terminals()
        self.assertEqual(terminals["keep"], "completed_phone")
        self.assertEqual(terminals["srv"], "completed_server")

    def test_conservation_detects_a_missing_terminal(self) -> None:
        registry = ReadyRouteRegistry()
        registry.install([F.op15_head_snapshot()])
        coordinator = MixedDispatchCoordinator(
            registry, RecordedExecutor({}),
            (LaneBinding("op15_gemma_head", OP15_ROUTE, "phone", timeout_us=4_000_000, cohort_sha256=COHORT, input_manifest_sha256=MANIFEST),),
            queue_capacity=8)
        coordinator.admit(gemma_item("queued"), 0)  # queued, never dispatched
        with self.assertRaisesRegex(Exception, "conservation failed"):
            coordinator.assert_conservation(["queued"])


if __name__ == "__main__":
    unittest.main()
