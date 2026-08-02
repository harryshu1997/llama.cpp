#!/usr/bin/env python3

from __future__ import annotations

import threading
import unittest
from dataclasses import dataclass

from route_runtime import (
    FiniteRoute,
    ResidentStage,
    RouteCleanupError,
    RouteDeviceBatcher,
    RouteError,
    RouteRequest,
    RouteRunner,
    StageEndpoint,
    validate_fixed_routes,
    validate_shared_treatment,
)
from runtime_support import SequenceSlotPool, SerializedStageClient
from stage_v3_client import BatchResult, BatchRow, ProtocolError


@dataclass(frozen=True)
class FakeStatus:
    active_sequences: int
    max_streams: int
    draining: bool = False


class FakeRawClient:
    def __init__(
        self,
        name: str,
        layer_start: int,
        layer_end: int,
        terminal: bool,
        remove_order: list[str],
        fail_batch: bool = False,
        fail_remove: bool = False,
        mutate_lineage: bool = False,
    ):
        self.name = name
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.terminal = terminal
        self.remove_order = remove_order
        self.fail_batch = fail_batch
        self.fail_remove = fail_remove
        self.mutate_lineage = mutate_lineage
        self.active: dict[int, tuple[int, int, int]] = {}
        self.rows: list[BatchRow] = []
        self.batch_sizes: list[int] = []

    def batch(self, rows):
        self.batch_sizes.append(len(rows))
        results = []
        for row in rows:
            if self.layer_start == 0 and row.hidden is not None:
                raise AssertionError("prefix received hidden state")
            if self.layer_start > 0 and row.hidden is None:
                raise AssertionError("non-prefix omitted hidden state")
            current = self.active.get(row.seq_id)
            if current is None:
                if row.position != 0:
                    raise AssertionError("fake sequence did not start at zero")
                current = (row.request_id, row.route_epoch, 0)
            request_id, route_epoch, next_position = current
            if (
                request_id != row.request_id
                or route_epoch != row.route_epoch
                or next_position != row.position
            ):
                raise AssertionError("fake sequence lineage mismatch")
            self.active[row.seq_id] = (
                request_id, route_epoch, next_position + 1,
            )
            self.rows.append(row)
            if self.terminal:
                results.append(BatchResult(
                    row.request_id,
                    row.route_epoch,
                    row.seq_id,
                    row.position,
                    None,
                    row.token + self.layer_end,
                ))
            else:
                base = float(row.token + self.layer_end)
                if row.hidden is not None:
                    base += row.hidden[0]
                results.append(BatchResult(
                    row.request_id,
                    row.route_epoch,
                    row.seq_id,
                    row.position,
                    (base, float(row.position + 1)),
                    None,
                ))
        if self.fail_batch:
            raise RuntimeError(f"{self.name} injected batch failure")
        if self.mutate_lineage and results:
            first = results[0]
            results[0] = BatchResult(
                first.request_id + 1,
                first.route_epoch,
                first.seq_id,
                first.position,
                first.hidden,
                first.token,
            )
        return tuple(results)

    def remove(self, seq_id, request_id, route_epoch):
        self.remove_order.append(self.name)
        if self.fail_remove:
            raise RuntimeError(f"{self.name} injected remove failure")
        current = self.active.get(seq_id)
        if current is None or current[:2] != (request_id, route_epoch):
            raise RuntimeError(f"{self.name} remove identity mismatch")
        del self.active[seq_id]
        return FakeStatus(len(self.active), 8)


class RuntimeFixture:
    def __init__(self, case: unittest.TestCase):
        self.case = case
        self.remove_order: list[str] = []
        self.batchers: list[RouteDeviceBatcher] = []
        self.raw: dict[str, FakeRawClient] = {}

    def stage(
        self,
        name: str,
        start: int,
        end: int,
        terminal: bool,
        knee: int = 1,
        gather_us: int = 0,
        slots: SequenceSlotPool | None = None,
        client: SerializedStageClient | None = None,
        fail_batch: bool = False,
        fail_remove: bool = False,
        mutate_lineage: bool = False,
        batcher_name: str | None = None,
    ) -> ResidentStage:
        if client is None:
            raw = FakeRawClient(
                name,
                start,
                end,
                terminal,
                self.remove_order,
                fail_batch,
                fail_remove,
                mutate_lineage,
            )
            self.raw[name] = raw
            client = SerializedStageClient(raw)
        if slots is None:
            slots = SequenceSlotPool(8)
        batcher = RouteDeviceBatcher(
            batcher_name or name, client, knee, gather_us, 32,
        )
        self.batchers.append(batcher)
        return ResidentStage(
            name,
            start,
            end,
            StageEndpoint("127.0.0.1", 20000 + len(self.batchers)),
            client,
            slots,
            batcher,
            terminal,
        )

    def close(self) -> None:
        for batcher in self.batchers:
            try:
                batcher.stop(2.0)
            except BaseException:
                pass


class RouteRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = RuntimeFixture(self)
        self.addCleanup(self.fixture.close)

    def route(self, fail_mid: bool = False, fail_tail_remove: bool = False):
        prefix = self.fixture.stage("cuda-prefix", 0, 8, False)
        middle = self.fixture.stage(
            "cuda-mid", 8, 16, False, fail_batch=fail_mid,
        )
        tail = self.fixture.stage(
            "cuda-tail", 16, 48, True,
            fail_remove=fail_tail_remove,
        )
        return FiniteRoute("R0", (prefix, middle, tail))

    def request(self) -> RouteRequest:
        return RouteRequest(
            101, 7, "R0", (2, 3), 3, 1_000_000, 0, 0, 2,
        )

    def test_ordered_route_preserves_lineage_and_cleans_all_workers(self) -> None:
        route = self.route()
        runner = RouteRunner([route])
        outcome = runner.run(self.request(), 2.0)
        self.assertEqual(outcome.route_id, "R0")
        self.assertEqual(len(outcome.output_tokens), 3)
        self.assertEqual(
            [row.position for row in self.fixture.raw["cuda-prefix"].rows],
            [0, 1, 2, 3],
        )
        self.assertEqual(
            self.fixture.remove_order,
            ["cuda-tail", "cuda-mid", "cuda-prefix"],
        )
        self.assertFalse(runner.pinned())
        for stage in route.stages:
            self.assertEqual(stage.slots.available(), 8)
            self.assertFalse(stage.slots.leased())
        self.assertEqual(
            {(item.worker_name, item.layer_end) for item in outcome.boundaries},
            {("cuda-prefix", 8), ("cuda-mid", 16)},
        )

    def test_route_remains_pinned_for_prefill_and_decode(self) -> None:
        route = self.route()
        runner = RouteRunner([route])
        runner.run(self.request(), 2.0)
        for stage in route.stages:
            self.assertEqual(
                {event_route for event in stage.batcher.events
                 for event_route in event["routes"]},
                {"R0"},
            )
            positions = [
                position for event in stage.batcher.events
                for position in event["positions"]
            ]
            self.assertEqual(positions, [0, 1, 2, 3])

    def test_group_route_keeps_every_decode_step_at_b4(self) -> None:
        prefix = self.fixture.stage(
            "cuda-prefix", 0, 8, False, knee=4, gather_us=5000,
        )
        middle = self.fixture.stage(
            "cuda-mid", 8, 16, False, knee=4, gather_us=5000,
        )
        tail = self.fixture.stage(
            "cuda-tail", 16, 48, True, knee=4, gather_us=5000,
        )
        route = FiniteRoute("R0", (prefix, middle, tail))
        runner = RouteRunner([route])
        requests = tuple(
            RouteRequest(
                request_id,
                request_id + 100,
                "R0",
                (2,),
                4,
                1_000_000,
                2,
                5000,
                1,
            )
            for request_id in range(101, 105)
        )
        outcomes = runner.run_group(requests, 2.0)
        self.assertEqual(len(outcomes), 4)
        self.assertEqual(len({outcome.output_tokens for outcome in outcomes}), 1)
        for worker in ("cuda-prefix", "cuda-mid", "cuda-tail"):
            self.assertEqual(self.fixture.raw[worker].batch_sizes, [4, 4, 4, 4])
        self.assertFalse(runner.pinned())
        for stage in route.stages:
            self.assertFalse(stage.slots.leased())

    def test_group_route_rejects_mixed_shapes_before_pinning(self) -> None:
        route = self.route()
        runner = RouteRunner([route])
        requests = (
            self.request(),
            RouteRequest(102, 8, "R0", (2,), 3, 1_000_000, 0, 0, 1),
        )
        with self.assertRaisesRegex(RouteError, "homogeneous"):
            runner.run_group(requests, 2.0)
        self.assertFalse(runner.pinned())

    def test_batch_failure_removes_touched_kv_and_releases_slots(self) -> None:
        route = self.route(fail_mid=True)
        runner = RouteRunner([route])
        with self.assertRaisesRegex(RuntimeError, "injected batch failure"):
            runner.run(self.request(), 2.0)
        self.assertEqual(
            self.fixture.remove_order,
            ["cuda-mid", "cuda-prefix"],
        )
        self.assertFalse(runner.pinned())
        for stage in route.stages:
            self.assertFalse(stage.slots.leased())

    def test_remove_failure_retains_failed_lease_and_route_pin(self) -> None:
        route = self.route(fail_tail_remove=True)
        runner = RouteRunner([route])
        with self.assertRaises(RouteCleanupError) as raised:
            runner.run(self.request(), 2.0)
        self.assertIsNone(raised.exception.primary)
        self.assertEqual(runner.pinned(), {101: (7, "R0")})
        self.assertEqual(
            route.stages[-1].slots.leased(),
            {0: 101},
        )
        self.assertFalse(route.stages[0].slots.leased())
        self.assertFalse(route.stages[1].slots.leased())

    def test_result_lineage_failure_is_recorded_before_cleanup(self) -> None:
        prefix = self.fixture.stage(
            "cuda-prefix", 0, 8, False, mutate_lineage=True,
        )
        middle = self.fixture.stage("cuda-mid", 8, 16, False)
        tail = self.fixture.stage("cuda-tail", 16, 48, True)
        route = FiniteRoute("R0", (prefix, middle, tail))
        runner = RouteRunner([route])
        with self.assertRaisesRegex(ProtocolError, "lineage mismatch"):
            runner.run(self.request(), 2.0)
        self.assertEqual(prefix.batcher.events[-1]["status"], "ERROR")
        self.assertEqual(
            prefix.batcher.events[-1]["error_type"], "ProtocolError",
        )
        self.assertFalse(runner.pinned())

    def test_unknown_route_fails_closed(self) -> None:
        runner = RouteRunner([self.route()])
        request = RouteRequest(
            1, 1, "R9", (2,), 1, 1000, 0,
        )
        with self.assertRaisesRegex(RouteError, "unknown route"):
            runner.run(request, 1.0)


class ConvergenceBatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = RuntimeFixture(self)
        self.addCleanup(self.fixture.close)

    def middle_batcher(
        self, knee: int, gather_us: int,
    ) -> tuple[RouteDeviceBatcher, FakeRawClient]:
        stage = self.fixture.stage(
            "op15-mid", 8, 16, False, knee=knee,
            gather_us=gather_us,
        )
        return stage.batcher, self.fixture.raw["op15-mid"]

    def submit(
        self,
        batcher: RouteDeviceBatcher,
        request_id: int,
        seq_id: int,
        route_id: str,
        upstream: str,
        wait_us: int | None,
        priority: int = 0,
    ):
        return batcher.submit(
            BatchRow(
                request_id, 1, seq_id, 0, 2,
                (float(request_id), 1.0),
            ),
            route_id,
            upstream,
            2.0,
            wait_us,
            priority,
        )

    def test_op15_batch_converges_r1_and_r2_at_knee(self) -> None:
        batcher, raw = self.middle_batcher(2, 100_000)
        first = self.submit(
            batcher, 1, 0, "R1", "cuda-prefix", None,
        )
        second = self.submit(
            batcher, 2, 1, "R2", "op12-prefix", None,
        )
        first.result(timeout=2)
        second.result(timeout=2)
        batcher.stop(2)
        self.fixture.batchers.remove(batcher)
        self.assertEqual(raw.batch_sizes, [2])
        self.assertEqual(
            batcher.events[0]["contributing_routes"], ["R1", "R2"],
        )
        self.assertEqual(
            batcher.events[0]["contributing_upstreams"],
            ["cuda-prefix", "op12-prefix"],
        )
        self.assertEqual(
            batcher.events[0]["dispatch_reason"], "BATCH_KNEE",
        )

    def test_latest_safe_start_dispatches_without_knee(self) -> None:
        batcher, _raw = self.middle_batcher(4, 100_000)
        self.submit(
            batcher, 1, 0, "R1", "cuda-prefix", 0,
        ).result(timeout=2)
        batcher.stop(2)
        self.fixture.batchers.remove(batcher)
        self.assertEqual(
            batcher.events[0]["dispatch_reason"], "LATEST_SAFE_START",
        )

    def test_gather_timer_does_not_wait_for_an_upstream_cohort(self) -> None:
        batcher, raw = self.middle_batcher(4, 1_000)
        self.submit(
            batcher, 1, 0, "R1", "cuda-prefix", None,
        ).result(timeout=2)
        batcher.stop(2)
        self.fixture.batchers.remove(batcher)
        self.assertEqual(raw.batch_sizes, [1])
        self.assertEqual(
            batcher.events[0]["dispatch_reason"], "GATHER_TIMER",
        )
        self.assertEqual(
            batcher.events[0]["contributing_routes"], ["R1"],
        )

    def test_shared_tail_converges_all_three_routes(self) -> None:
        stage = self.fixture.stage(
            "cuda-tail", 16, 48, True, knee=3, gather_us=100_000,
        )
        futures = [
            stage.batcher.submit(
                BatchRow(
                    request_id, 1, seq_id, 0, 2,
                    (float(request_id), 1.0),
                ),
                route_id,
                upstream,
                2.0,
                None,
            )
            for request_id, seq_id, route_id, upstream in (
                (1, 0, "R0", "cuda-mid"),
                (2, 1, "R1", "op15-mid"),
                (3, 2, "R2", "op15-mid"),
            )
        ]
        for future in futures:
            future.result(timeout=2)
        stage.batcher.stop(2)
        self.fixture.batchers.remove(stage.batcher)
        self.assertEqual(
            stage.batcher.events[0]["contributing_routes"],
            ["R0", "R1", "R2"],
        )

    def test_shared_tail_isolates_urgent_from_background(self) -> None:
        batcher, raw = self.middle_batcher(3, 100_000)
        background = self.submit(
            batcher, 1, 0, "R2", "op15-mid", None, 2,
        )
        urgent = self.submit(
            batcher, 2, 1, "R0", "cuda-mid", None, 0,
        )
        background_peer = self.submit(
            batcher, 3, 2, "R1", "op15-mid", None, 1,
        )
        urgent.result(timeout=2)
        background.result(timeout=2)
        background_peer.result(timeout=2)
        batcher.stop(2)
        self.fixture.batchers.remove(batcher)
        self.assertEqual(raw.batch_sizes, [1, 2])
        self.assertEqual(batcher.events[0]["request_ids"], [2])
        self.assertEqual(batcher.events[0]["priorities"], [0])
        self.assertEqual(
            sorted(batcher.events[1]["request_ids"]), [1, 3],
        )
        self.assertEqual(
            sorted(batcher.events[1]["priorities"]), [1, 2],
        )


class FixedTopologyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = RuntimeFixture(self)
        self.addCleanup(self.fixture.close)

    def treatment(self):
        cuda_prefix = self.fixture.stage("cuda-prefix", 0, 8, False)
        cuda_mid = self.fixture.stage("cuda-mid", 8, 16, False)
        op12_prefix = self.fixture.stage("op12-prefix", 0, 8, False)
        op15_mid = self.fixture.stage("op15-mid", 8, 16, False)
        cuda_tail = self.fixture.stage("cuda-tail", 16, 48, True)
        return (
            FiniteRoute("R0", (cuda_prefix, cuda_mid, cuda_tail)),
            FiniteRoute("R1", (cuda_prefix, op15_mid, cuda_tail)),
            FiniteRoute("R2", (op12_prefix, op15_mid, cuda_tail)),
        )

    def test_fixed_treatment_requires_exact_routes_and_shared_queues(self) -> None:
        routes = self.treatment()
        validate_fixed_routes(routes)
        validate_shared_treatment(routes)

    def test_route_isolated_op15_is_rejected_as_treatment(self) -> None:
        routes = list(self.treatment())
        shared = routes[1].stages[1]
        isolated = self.fixture.stage(
            "op15-mid",
            8,
            16,
            False,
            slots=shared.slots,
            client=shared.client,
            batcher_name="op15-mid-r2",
        )
        routes[2] = FiniteRoute(
            "R2", (routes[2].stages[0], isolated, routes[2].stages[2]),
        )
        validate_fixed_routes(routes)
        with self.assertRaisesRegex(RouteError, "share one OP15"):
            validate_shared_treatment(routes)

    def test_layer_gap_is_rejected(self) -> None:
        prefix = self.fixture.stage("prefix", 0, 8, False)
        tail = self.fixture.stage("tail", 16, 48, True)
        with self.assertRaisesRegex(ValueError, "gap or overlap"):
            FiniteRoute("R", (prefix, tail))


if __name__ == "__main__":
    unittest.main()
