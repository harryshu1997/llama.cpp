#!/usr/bin/env python3
"""Continuous join at publication: joiners that arrived during their server's load (hardware s1c).

s1c: Gemma 000 decoded; the Qwen switch 001 waited behind it (000 overran, so 001 was
deferred) and Gemma 002 was reserved hot behind 001 (affinity/bypass refused: no earlier
start). 000 finished, 001 loaded Qwen; Qwen 003/004 arrived during that load and queued
behind 002. When 001's server went live, 002 was still awaiting its replan with its stale
transition-free plan, so no rule saw a residency change to displace; 002 then replanned into
the Gemma switch and nothing re-evaluated 003/004 until the switch had run (737 s wait).

The fixtures below replay that sequence on the synthetic GPU server (model A = Qwen, the
loading model; model B = Gemma) and on the multi-slot server whose leader holds the
capacity-one phone lanes.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest

from research_dev.scheduler import RuntimeDispatchPolicy
from research_dev.scheduler._internal.lifecycle import UnifiedScheduleError

try:
    from . import test_dispatch_policy as dispatch_tests
    from .test_automated_runtime import (
        AutomatedRuntimeTests,
        FakeAutomatedPhysicalAdapter,
        runtime_snapshot,
        write_synthetic_gguf,
    )
except ImportError:  # run from the tests directory
    import test_dispatch_policy as dispatch_tests
    from test_automated_runtime import (
        AutomatedRuntimeTests,
        FakeAutomatedPhysicalAdapter,
        runtime_snapshot,
        write_synthetic_gguf,
    )


# Held in a namespace: a module-level TestCase alias would be collected (and run) here again.
FIXTURES = SimpleNamespace(
    gpu=dispatch_tests.DispatchPolicySchedulerTests,
    phone=dispatch_tests.ContinuousJoinSchedulerTests,
)
AFFINITY_JOIN = RuntimeDispatchPolicy(
    work_conserving_admission=True, model_affinity=True, continuous_join=True,
    max_barrier_extension_s=120,
)
JOIN_ONLY = RuntimeDispatchPolicy(
    work_conserving_admission=True, continuous_join=True, max_barrier_extension_s=120,
)
AFFINITY_ONLY = RuntimeDispatchPolicy(work_conserving_admission=True, model_affinity=True)
SERVER_LIVE = "continuous_join_server_live"
PUBLICATION_STAT = "continuous_join_publication_replans"
BYPASS_KIND = "CONTINUOUS_JOIN_BARRIER_BYPASS"
AFFINITY_KIND = "MODEL_AFFINITY_DISPLACEMENT"
JOIN_REASON = "CONTINUOUS_JOIN_DESKTOP_PARENT"
PHONE_LANES_HELD = "PHONE_LANES_HELD_BY_RUNNING_REQUEST"
SERVER = "compute:accelerator-b"


def rows(model, device_id: str, executor_id: str):
    """Hot residency of ``model`` on one device, served by ``executor_id``."""
    return tuple(
        replace(row, executor_id=executor_id)
        for row in runtime_snapshot(model, resident_devices=(device_id,)).residency
    )


class S1cPattern:
    """Replays s1c up to the replan of the switch that follows the leader's publication."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.paths = (
            Path(self.directory.name) / "model-a.gguf",
            Path(self.directory.name) / "model-b.gguf",
        )
        write_synthetic_gguf(self.paths[0], block_count=2, sliding_window=32)
        write_synthetic_gguf(self.paths[1])

    def build(self, policy, *, phone: bool = False):
        if phone:
            scheduler, model_a, model_b, hot = FIXTURES.phone.scheduler(self, policy, parallel=4)
        else:
            scheduler, model_a, model_b, hot = FIXTURES.gpu.scheduler(self, policy)
        self.phone = phone
        self.hot = hot
        return scheduler, model_a, model_b

    def at(self, at_us: int, gpu_model, *, free: int | None = None):
        """The snapshot at ``at_us``: ``gpu_model`` (or nothing) on the GPU server.

        With the phone fixture model A stays resident on helper-c (the early
        re-provision already put its shards there).
        """
        snapshot = self.hot if free is None else FIXTURES.phone.free_slots(self.hot, free)
        residency = ()
        if self.phone:
            residency += rows(self.model_a, "helper-c", "executor:helper-c")
        if gpu_model is not None:
            residency += rows(gpu_model, "accelerator-b", "executor:accelerator-b")
        return FIXTURES.gpu.published(replace(snapshot, residency=residency), at_us)

    def submit(self, scheduler, model, snapshot, request_id, arrival_us, tokens, mode):
        return FIXTURES.phone.submit(scheduler, model, snapshot, request_id, arrival_us, tokens, mode=mode)

    @staticmethod
    def view(scheduler):
        return scheduler._runtime_controller.dispatch_order_view()

    @staticmethod
    def wake(scheduler, request_id: str, at_us: int):
        return scheduler.wait_runtime_request(request_id, time.monotonic_ns() - at_us * 1_000)

    def replan(self, scheduler, request_id: str, at_us: int, snapshot):
        """Replan one REPLAN_REQUIRED attempt as its dispatcher would; return (ticket, wake reason)."""
        self.assertEqual(self.view(scheduler)[request_id]["state"], "REPLAN_REQUIRED")
        wake = self.wake(scheduler, request_id, at_us)
        reason = wake.dispatch_receipt.wake_reason
        ticket = scheduler.replan_automated_request(
            request_id, observed_at_us=at_us, reason=reason, snapshot=snapshot,
            expected_ticket_id=wake.ticket_id,
            expected_queue_generation=wake.dispatch_receipt.queue_generation,
        )
        return ticket, reason

    def drain(self, scheduler, at_us: int, snapshot) -> list[tuple[str, str]]:
        """Replan every waiting attempt in queue order, as their dispatchers would."""
        wakes = []
        for _ in range(12):
            view = self.view(scheduler)
            pending = sorted(
                (key for key, row in view.items() if row["state"] == "REPLAN_REQUIRED"),
                key=lambda key: view[key]["sequence"],
            )
            if not pending:
                return wakes
            _, reason = self.replan(scheduler, pending[0], at_us, snapshot)
            wakes.append((pending[0], reason))
        self.fail("replans did not settle")

    def s1c(self, policy, *, phone: bool = False, joiner_tokens: int = 64,
            switch_first: bool = False):
        """b0 (B) overruns; a1 (A switch) defers; b1 (B) reserved hot behind a1; b0 ends; a1
        loads A; a2, a3 (A) arrive during the load; a1 publishes; b1 replans into the switch.

        ``switch_first``: b1 replans on a snapshot sampled before the publication, and the
        publication is observed afterwards (the other order of the two dispatcher threads).
        """
        scheduler, model_a, model_b = self.build(policy, phone=phone)
        self.model_a = model_a
        leader_mode = "energy-aware" if phone else "desktop-baseline"
        b0 = self.submit(scheduler, model_b, self.at(0, model_b), "b0", 1_000, 640, "desktop-baseline")
        self.assertEqual(self.wake(scheduler, "b0", 1_000).dispatch_state, "ACQUIRED")
        self.submit(scheduler, model_a, self.at(1_100, model_b, free=3), "a1", 1_100, 640, leader_mode)
        b0_end = max(lease.reserved_until_us for lease in b0.decision.leases)
        scheduler.extend_runtime_request("b0", at_us=b0_end - 10, reserved_until_us=b0_end + 3_000)
        self.assertEqual(self.view(scheduler)["a1"]["state"], "DEFERRED_REPLAN")
        b1 = self.submit(scheduler, model_b, self.at(1_200, model_b, free=3), "b1", 1_200, 640,
                         "desktop-baseline")
        # 002 at 312: reserved hot at its arrival, yet ordered behind the deferred switch.
        self.assertEqual(b1.execution_plan.transitions, ())
        self.assertEqual(b1.decision.start_us, 1_200)
        self.assertIn("a1", self.view(scheduler)["b1"]["predecessor_request_ids"])
        released_us = b0_end + 2_000
        scheduler.complete_automated_request("b0", AutomatedRuntimeTests.execution_receipt(
            scheduler.runtime_ticket("b0"), finished_us=released_us,
        ))
        leader, _ = self.replan(scheduler, "a1", released_us, self.at(released_us, model_b))
        self.assertTrue(leader.execution_plan.transitions)
        active = self.wake(scheduler, "a1", leader.decision.start_us)
        self.assertEqual(active.dispatch_state, "ACQUIRED")
        loading = self.at(released_us + 100, None, free=3)
        for request_id, offset in (("a2", 100), ("a3", 200)):
            self.submit(scheduler, model_a, replace(loading, captured_at_us=released_us + offset),
                        request_id, released_us + offset, joiner_tokens, leader_mode)
        self.assertIn("b1", self.view(scheduler)["a2"]["predecessor_request_ids"])
        loaded_us = released_us + 300
        scheduler.record_automated_transition_receipts("a1", tuple(
            replace(row, finished_us=loaded_us)
            for row in FakeAutomatedPhysicalAdapter._transition_receipts(active)
        ))
        published = self.at(loaded_us, model_a, free=3)
        self.assertEqual(self.view(scheduler)["b1"]["state"], "REPLAN_REQUIRED")
        self.assertEqual(scheduler.runtime_ticket("b1").execution_plan.transitions, ())
        run = SimpleNamespace(
            scheduler=scheduler, model_a=model_a, published=published, loaded_us=loaded_us,
            leader=scheduler.runtime_ticket("a1"), observed=None,
        )
        if switch_first:
            switch, _ = self.replan(scheduler, "b1", loaded_us, loading)
        else:
            # The switch still awaits its replan with its stale hot plan: nothing to displace.
            self.assertEqual(
                scheduler.observe_automated_runtime_snapshot(published, observed_at_us=loaded_us),
                (),
            )
            switch, _ = self.replan(scheduler, "b1", loaded_us, published)
        self.assertTrue(switch.execution_plan.transitions)
        if switch_first:
            run.observed = scheduler.observe_automated_runtime_snapshot(
                published, observed_at_us=loaded_us
            )
        return run

    @staticmethod
    def last_record(scheduler, request_id: str):
        return [
            row for row in scheduler.runtime_decision_log()["records"]
            if row["request_ids"] == [request_id] and row["event_kind"] in {"DECISION", "REPLAN"}
        ][-1]

    @staticmethod
    def statistics(scheduler):
        return scheduler.runtime_dispatch_policy_state()["statistics"]

    def server_end_us(self, scheduler, *request_ids: str) -> int:
        return max(
            lease.reserved_until_us
            for request_id in request_ids
            for lease in scheduler.runtime_ticket(request_id).decision.leases
            if lease.resource_id == SERVER
        )

    def assert_joined_ahead_of_the_switch(self, run, kind: str) -> None:
        scheduler = run.scheduler
        for request_id in ("a2", "a3"):
            with self.subTest(joiner=request_id):
                ticket = scheduler.runtime_ticket(request_id)
                self.assertEqual(ticket.execution_plan.transitions, ())
                self.assertEqual(ticket.failure_reason, SERVER_LIVE)
                self.assertNotIn("b1", self.view(scheduler)[request_id]["predecessor_request_ids"])
                self.assertIn(request_id, self.view(scheduler)["b1"]["predecessor_request_ids"])
                note = self.last_record(scheduler, request_id)["selected"]["dispatch_policy"]
                self.assertEqual(note["kind"], kind)
                self.assertEqual(note["replan_reason"], SERVER_LIVE)
                self.assertIn("b1", note["displaced_request_ids"])
        self.assertEqual(scheduler.runtime_ticket("a2").decision.start_us, run.loaded_us)
        self.assertTrue(FIXTURES.gpu.ready_now(scheduler, "a2", run.loaded_us))
        switch = scheduler.runtime_ticket("b1")
        self.assertTrue(switch.execution_plan.transitions)
        self.assertGreaterEqual(switch.decision.start_us, self.server_end_us(scheduler, "a2", "a3"))


class ContinuousJoinPublicationTests(S1cPattern, unittest.TestCase):
    def test_s1c_joiners_are_replanned_once_the_switch_is_reserved_after_publication(self) -> None:
        run = self.s1c(AFFINITY_JOIN)
        view = self.view(run.scheduler)
        self.assertEqual(view["b1"]["state"], "QUEUED")
        for request_id in ("a2", "a3"):
            self.assertEqual(view[request_id]["state"], "REPLAN_REQUIRED")
        self.assertEqual(self.statistics(run.scheduler)[PUBLICATION_STAT], 2)
        wakes = self.drain(run.scheduler, run.loaded_us, run.published)
        self.assertEqual(
            [reason for request_id, reason in wakes if request_id in {"a2", "a3"}],
            [SERVER_LIVE, SERVER_LIVE],
        )
        self.assert_joined_ahead_of_the_switch(run, AFFINITY_KIND)
        statistics = self.statistics(run.scheduler)
        self.assertEqual(statistics["affinity_displacements"], 2)
        self.assertEqual(statistics[PUBLICATION_STAT], 2)
        self.assertEqual(run.scheduler.runtime_dispatch_policy_state()["bypass_counts"], {"b1": 2})

    def test_s1c_joiners_take_the_bounded_bypass_without_model_affinity(self) -> None:
        run = self.s1c(JOIN_ONLY)
        wakes = self.drain(run.scheduler, run.loaded_us, run.published)
        self.assertIn(("b1", "continuous_join_displaced"), wakes)
        self.assert_joined_ahead_of_the_switch(run, BYPASS_KIND)
        for request_id in ("a2", "a3"):
            note = self.last_record(run.scheduler, request_id)["selected"]["dispatch_policy"]
            self.assertLessEqual(note["extension_us"], note["max_barrier_extension_us"])
            self.assertEqual(note["max_barrier_extension_us"], 120_000_000)
            self.assertLess(note["reserved_start_us"], note["barrier_start_us"])
            self.assertEqual(note["barrier_request_ids"], ["b1"])
            self.assertGreaterEqual(note["committed_end_us"], run.leader.decision.finish_upper_us)
            self.assertEqual(
                note["extension_us"], max(0, note["finish_upper_us"] - note["committed_end_us"])
            )
        statistics = self.statistics(run.scheduler)
        self.assertEqual(statistics["continuous_join_bypasses"], 2)
        self.assertEqual(statistics[PUBLICATION_STAT], 2)
        self.assertEqual(statistics["affinity_displacements"], 0)

    def test_publication_observed_after_the_switch_replan_wakes_the_joiners(self) -> None:
        # The switch replanned on a snapshot from before the publication, so its commit found
        # no live server; the later observation of the publication wakes the joiners.
        for policy, kind in ((AFFINITY_JOIN, AFFINITY_KIND), (JOIN_ONLY, BYPASS_KIND)):
            with self.subTest(kind=kind):
                run = self.s1c(policy, switch_first=True)
                self.assertIn("a2", run.observed)
                self.assertEqual(self.view(run.scheduler)["a2"]["state"], "REPLAN_REQUIRED")
                self.assertGreaterEqual(self.statistics(run.scheduler)[PUBLICATION_STAT], 1)
                self.drain(run.scheduler, run.loaded_us, run.published)
                self.assert_joined_ahead_of_the_switch(run, kind)

    def test_policy_off_twin_keeps_the_joiners_behind_the_switch(self) -> None:
        run = self.s1c(AFFINITY_ONLY)
        view = self.view(run.scheduler)
        for request_id in ("a2", "a3"):
            self.assertNotEqual(view[request_id]["state"], "REPLAN_REQUIRED")
            self.assertIn("b1", view[request_id]["predecessor_request_ids"])
        statistics = self.statistics(run.scheduler)
        self.assertNotIn(PUBLICATION_STAT, statistics)
        self.assertNotIn("continuous_join_bypasses", statistics)
        self.assertEqual(statistics["affinity_displacements"], 0)

    def test_joiner_past_the_extension_bound_stays_behind_the_switch_and_is_woken_once(self) -> None:
        for policy in (
            replace(JOIN_ONLY, max_barrier_extension_s=0),
            replace(AFFINITY_JOIN, max_barrier_extension_s=0, affinity_maximum_bypasses=0),
        ):
            with self.subTest(model_affinity=policy.model_affinity):
                run = self.s1c(policy, joiner_tokens=2_048)
                scheduler = run.scheduler
                woken = self.statistics(scheduler)[PUBLICATION_STAT]
                self.assertGreaterEqual(woken, 1)
                self.assertEqual(self.view(scheduler)["a2"]["state"], "REPLAN_REQUIRED")
                self.drain(scheduler, run.loaded_us, run.published)
                refusals = [
                    row for row in scheduler.runtime_dispatch_policy_state()["refusals"]
                    if row["request_id"] in {"a2", "a3"}
                ]
                self.assertEqual(
                    sorted(row["request_id"] for row in refusals if row["reason"]
                           == "joiner would extend the committed busy window"),
                    ["a2", "a3"],
                )
                view = self.view(scheduler)
                for request_id in ("a2", "a3"):
                    self.assertIn("b1", view[request_id]["predecessor_request_ids"])
                    self.assertNotIn(
                        "dispatch_policy", self.last_record(scheduler, request_id)["selected"]
                    )
                self.assertEqual(scheduler.runtime_dispatch_policy_state()["bypass_counts"], {})
                later = FIXTURES.gpu.published(run.published, run.loaded_us + 50)
                # Replanned after the publication: judged, never woken again.
                self.assertEqual(
                    scheduler.observe_automated_runtime_snapshot(
                        later, observed_at_us=run.loaded_us + 50
                    ),
                    (),
                )
                self.assertEqual(self.statistics(scheduler)[PUBLICATION_STAT], woken)

    def test_phone_lane_leader_admits_the_joiners_as_desktop_parents(self) -> None:
        for policy, kind in ((AFFINITY_JOIN, AFFINITY_KIND), (JOIN_ONLY, BYPASS_KIND)):
            with self.subTest(kind=kind):
                run = self.s1c(policy, phone=True)
                self.assertIn("compute:helper-c", {
                    lease.resource_id for lease in run.leader.decision.leases
                })
                self.assertEqual(self.view(run.scheduler)["a2"]["state"], "REPLAN_REQUIRED")
                self.assertEqual(self.statistics(run.scheduler)[PUBLICATION_STAT], 1)
                wakes = self.drain(run.scheduler, run.loaded_us, run.published)
                self.assertIn(("a2", SERVER_LIVE), wakes)
                for request_id in ("a2", "a3"):
                    ticket = run.scheduler.runtime_ticket(request_id)
                    self.assertEqual(ticket.decision.reason, JOIN_REASON)
                    self.assertEqual(ticket.decision.start_us, run.loaded_us)
                    self.assertEqual(ticket.binding.endpoint, run.leader.binding.endpoint)
                    self.assertEqual(ticket.execution_plan.execution_contract.execution_mode, "desktop")
                    self.assertEqual({row.resource_id for row in ticket.decision.leases}, {SERVER})
                    self.assertIn(PHONE_LANES_HELD, dict(ticket.decision.rejected).values())
                    self.assertTrue(FIXTURES.gpu.ready_now(run.scheduler, request_id, run.loaded_us))
                note = self.last_record(run.scheduler, "a2")["selected"]["dispatch_policy"]
                self.assertEqual(note["kind"], kind)
                self.assertEqual(note["replan_reason"], SERVER_LIVE)

    def test_publication_replans_are_deterministic(self) -> None:
        outcomes = []
        for _ in range(2):
            run = self.s1c(JOIN_ONLY)
            self.drain(run.scheduler, run.loaded_us, run.published)
            outcomes.append((
                [
                    (ticket.decision.route_id, ticket.decision.start_us, ticket.decision.reason)
                    for ticket in (run.scheduler.runtime_ticket(key) for key in ("a2", "a3", "b1"))
                ],
                [self.last_record(run.scheduler, key)["selected"].get("dispatch_policy")
                 for key in ("a2", "a3")],
                dict(run.scheduler.runtime_dispatch_policy_state()),
                run.scheduler.runtime_controller_snapshot()["dispatch_queue"]["causal_predecessors"],
            ))
        self.assertEqual(outcomes[0], outcomes[1])


class ReplanBypassAtPublicationTests(unittest.TestCase):
    """A joiner woken at publication (switch already queued) replans through the bypass."""

    # The GPU fixture's scenario helpers, bound to this case (its tests are not collected).
    scheduler = FIXTURES.gpu.scheduler
    arrival_during_load = FIXTURES.gpu.arrival_during_load
    replan_woken = FIXTURES.gpu.replan_woken
    submit = staticmethod(FIXTURES.gpu.submit)
    queue_view = staticmethod(FIXTURES.gpu.queue_view)
    published = staticmethod(FIXTURES.gpu.published)

    def setUp(self) -> None:
        S1cPattern.setUp(self)

    def test_woken_joiner_takes_the_bypass_on_its_replan(self) -> None:
        for policy, bypassed in ((JOIN_ONLY, True), (dispatch_tests.WORK_CONSERVING, False)):
            with self.subTest(continuous_join=policy.continuous_join):
                scheduler, hot, loaded_us = self.arrival_during_load(policy)
                published = FIXTURES.gpu.published(hot, loaded_us)
                self.assertEqual(
                    scheduler.observe_automated_runtime_snapshot(published, observed_at_us=loaded_us),
                    ("b1", "a2"),
                )
                for request_id in ("b1", "a2"):
                    self.replan_woken(scheduler, request_id, loaded_us, published)
                a2 = scheduler.runtime_ticket("a2")
                view = scheduler._runtime_controller.dispatch_order_view()
                if not bypassed:
                    self.assertTrue(a2.execution_plan.transitions)
                    self.assertIn("b1", view["a2"]["predecessor_request_ids"])
                    continue
                self.assertEqual(a2.execution_plan.transitions, ())
                self.assertEqual(a2.decision.start_us, loaded_us)
                self.assertEqual(view["a2"]["predecessor_request_ids"], ())
                self.assertEqual(view["b1"]["state"], "REPLAN_REQUIRED")
                record = [
                    row for row in scheduler.runtime_decision_log()["records"]
                    if row["request_ids"] == ["a2"] and row["event_kind"] == "REPLAN"
                ][-1]
                note = record["selected"]["dispatch_policy"]
                self.assertEqual(note["kind"], BYPASS_KIND)
                self.assertEqual(note["replan_reason"], "residency_observation_changed")
                self.assertEqual(
                    scheduler.runtime_dispatch_policy_state()["statistics"][
                        "continuous_join_bypasses"
                    ],
                    1,
                )


class JoinPublicationUnitTests(unittest.TestCase):
    @staticmethod
    def ticket(request_id, *, state="ACQUIRED", artifact="sha256:a", executor="gpu",
               status="COMPLETED", finished=(), observed_at_us=0):
        return SimpleNamespace(
            request=SimpleNamespace(request_id=request_id),
            model=SimpleNamespace(artifact_sha256=artifact),
            binding=SimpleNamespace(executor_id=executor),
            dispatch_state=state,
            transition_status=status,
            transition_receipts=tuple(SimpleNamespace(finished_us=value) for value in finished),
            runtime_observation=SimpleNamespace(captured_at_us=observed_at_us),
        )

    def test_publication_time_is_the_latest_completed_preparation_of_a_co_tenant(self) -> None:
        from research_dev.scheduler._unified.automated_requests_ops import join_publication

        joiner = self.ticket("a2", state="QUEUED", status="NOT_REQUIRED", observed_at_us=150)
        tickets = (
            joiner,
            self.ticket("a1", finished=(100, 200)),
            self.ticket("a0", finished=(900,), state="COMPLETED"),
            self.ticket("b1", finished=(950,), artifact="sha256:b"),
            self.ticket("c1", finished=(960,), executor="phone"),
            self.ticket("a4", finished=(970,), status="PENDING"),
        )
        self.assertEqual(join_publication.server_published_at_us(joiner, tickets), 200)
        self.assertTrue(join_publication.planned_before_publication(joiner, tickets))
        for observed_at_us in (200, 201):
            late = self.ticket("a2", state="QUEUED", observed_at_us=observed_at_us)
            self.assertFalse(join_publication.planned_before_publication(late, tickets))
        self.assertIsNone(join_publication.server_published_at_us(joiner, (joiner,)))
        self.assertFalse(join_publication.planned_before_publication(joiner, (joiner,)))

    def test_replan_bypass_refusal_names_start_residency_then_extension(self) -> None:
        from research_dev.scheduler._unified.automated_requests_ops import continuous_join

        note = {"barrier_start_us": 1_000, "committed_end_us": 5_000}
        exclusive = {"gpu": "compute:gpu"}
        load = SimpleNamespace(
            transition_id="load:a", source_state="cold", target_state="hot",
            prepares_device_ids=("gpu",),
        )
        plan = lambda *transitions: SimpleNamespace(plan=SimpleNamespace(transitions=transitions))
        preview = lambda start, upper: SimpleNamespace(start_us=start, finish_upper_us=upper)
        refuse = continuous_join.replan_bypass_refusal
        self.assertEqual(refuse(plan(), preview(900, 5_400), note, exclusive, 400), (None, 400))
        self.assertEqual(
            refuse(plan(), preview(1_000, 4_000), note, exclusive, 0),
            (f"{continuous_join.REPLAN_BYPASS_PREFIX}: re-resolved start 1000 >= original 1000", 0),
        )
        self.assertEqual(
            refuse(plan(load), preview(900, 4_000), note, exclusive, 0)[0],
            f"{continuous_join.REPLAN_BYPASS_PREFIX}: re-resolved plan changes residency:"
            " load:a (cold->hot)",
        )
        self.assertEqual(
            refuse(plan(), preview(900, 5_401), note, exclusive, 400),
            ("joiner would extend the committed busy window", 401),
        )

    def test_publication_scan_is_inert_off_policy_and_fail_closed_on_it(self) -> None:
        from research_dev.scheduler._unified.automated_requests_ops import join_publication

        test = S1cPattern()
        test.addCleanup = self.addCleanup
        S1cPattern.setUp(test)
        for policy in (None, AFFINITY_ONLY, JOIN_ONLY):
            scheduler, _, _ = test.build(policy)
            with self.subTest(policy=policy):
                if policy is None or not policy.continuous_join:
                    self.assertEqual(
                        join_publication.server_live_joiners(scheduler, object(), 1.5), ()
                    )
                    continue
                with self.assertRaises(UnifiedScheduleError):
                    join_publication.server_live_joiners(scheduler, object(), 1)
                for observed_at_us in (1.5, -1, True):
                    with self.assertRaises(UnifiedScheduleError):
                        join_publication.server_live_joiners(
                            scheduler, test.hot, observed_at_us
                        )
                self.assertEqual(join_publication.server_live_joiners(scheduler, test.hot, 1), ())


if __name__ == "__main__":
    unittest.main()
