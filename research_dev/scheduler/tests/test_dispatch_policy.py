#!/usr/bin/env python3
"""Opt-in dispatch policy: work-conserving admission (#2), model affinity (#5), continuous join,
residency hysteresis."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re
import tempfile
import threading
import time
import unittest
from unittest import mock

from research_dev.scheduler import (
    Decision,
    LeaseRecord,
    RuntimeCapabilityCatalog,
    RuntimeDispatchPolicy,
    RuntimeDispatchPolicyError,
    RuntimeDispatchQueue,
    RuntimeQueueError,
    UnifiedScheduler,
)
from research_dev.scheduler.config import (
    CAMPAIGN_MANIFEST_SCHEMA,
    CampaignManifest,
    SchedulerConfigurationError,
)
from research_dev.scheduler.campaigns.burstgpt import arguments, runner
from research_dev.scheduler._internal.adaptive_decode_contracts import AdaptiveDecodeConfig
from research_dev.scheduler._internal.lifecycle import UnifiedScheduleError
from research_dev.scheduler._internal.runtime_controller import (
    RuntimeController,
    RuntimeControllerError,
)
from research_dev.scheduler._unified.automated_requests_ops import selection as selection_ops
from research_dev.scheduler._unified.automated_requests_ops import (
    affinity as affinity_ops,
    continuous_join as join_request_ops,
    observations as observation_ops,
)
from research_dev.scheduler._unified.automated_selection_ops import (
    continuous_join as join_selection_ops,
)

try:
    from .test_automated_runtime import (
        AutomatedRuntimeTests,
        FakeAutomatedPhysicalAdapter,
        catalog,
        executor_state,
        request,
        runtime_snapshot,
        write_synthetic_gguf,
    )
except ImportError:  # run from the tests directory
    from test_automated_runtime import (
        AutomatedRuntimeTests,
        FakeAutomatedPhysicalAdapter,
        catalog,
        executor_state,
        request,
        runtime_snapshot,
        write_synthetic_gguf,
    )


WORK_CONSERVING = RuntimeDispatchPolicy(work_conserving_admission=True)
CONTINUOUS_JOIN = RuntimeDispatchPolicy(work_conserving_admission=True, continuous_join=True)
HYSTERESIS_S = 20
HYSTERESIS_US = HYSTERESIS_S * 1_000_000
HYSTERESIS = RuntimeDispatchPolicy(
    work_conserving_admission=True, model_affinity=True, residency_hysteresis_s=HYSTERESIS_S,
)
# Threshold 0 keeps the unconditional (speculative) hold of the first design.
SPECULATIVE = replace(HYSTERESIS, residency_hysteresis_min_probability_ppm=0)
AFFINITY = RuntimeDispatchPolicy(work_conserving_admission=True, model_affinity=True)
HELD_KIND = "RESIDENCY_HYSTERESIS_HELD"
SKIPPED_KIND = "RESIDENCY_HYSTERESIS_SKIPPED"
JOIN_REASON = "CONTINUOUS_JOIN_DESKTOP_PARENT"
PHONE_LANES_HELD = "PHONE_LANES_HELD_BY_RUNNING_REQUEST"
RESIDENCY_TRANSITION = "CONTINUOUS_JOIN_RESIDENCY_TRANSITION"
BYPASS_KIND = "CONTINUOUS_JOIN_BARRIER_BYPASS"
# A start-half refusal ends with what holds the re-resolved start.
BOUND_SUFFIX = r" \((bounded by [^)]+|no barrier recorded)\)$"


def leased(request_id: str, *windows: tuple[str, tuple[int, ...], int, int]) -> Decision:
    """A decision holding one lease per (resource, lanes, start, end) window."""
    leases = tuple(
        LeaseRecord(
            token=f"{request_id}:{index}",
            owner_id=request_id,
            lease_id=f"lease-{index}",
            resource_id=resource_id,
            lanes=lanes,
            start_us=start_us,
            predicted_end_us=end_us,
            reserved_until_us=end_us,
        )
        for index, (resource_id, lanes, start_us, end_us) in enumerate(windows)
    )
    start_us = min(row.start_us for row in leases)
    end_us = max(row.reserved_until_us for row in leases)
    return Decision(
        request_id=request_id,
        workload_id="dispatch-policy",
        mode="enforce",
        route_id="route:" + request_id,
        granularity="task",
        start_us=start_us,
        finish_us=end_us,
        finish_upper_us=end_us,
        service_us=end_us - start_us,
        queue_us=0,
        queue_by_resource_us={},
        blocking_resources=(),
        leases=leases,
        runtime_gate=None,
        energy_uj=1,
        energy_upper_uj=1,
        energy_breakdown=None,
        server_busy_us=end_us - start_us,
        reason="TEST",
        rejected=(),
    )


def predecessors(queue: RuntimeDispatchQueue, request_id: str) -> tuple[str, ...]:
    return queue.dispatch_order_view()[request_id]["predecessor_request_ids"]


def state(queue: RuntimeDispatchQueue, request_id: str) -> str:
    return queue.dispatch_order_view()[request_id]["state"]


def acquire(queue: RuntimeDispatchQueue, request_id: str, at_us: int) -> None:
    receipt = queue.wait(request_id, time.monotonic_ns() - at_us * 1_000)
    if receipt.status != "ACQUIRED":
        raise AssertionError(receipt)


def dispatchable(queue: RuntimeDispatchQueue, request_id: str, at_us: int) -> bool:
    """Whether wait_ready would acquire now, evaluated without blocking."""
    with queue._condition:
        entry = queue._entries[request_id]
        return (
            entry.state == "QUEUED"
            and queue._causal_ready(entry)
            and not queue._active_conflict(entry)
            and not queue._earlier_queued_conflict(entry)
            and entry.decision.start_us <= at_us
        )


class RuntimeDispatchPolicyTests(unittest.TestCase):
    def test_policy_fields_are_validated(self) -> None:
        self.assertFalse(RuntimeDispatchPolicy().enabled)
        with self.assertRaises(RuntimeDispatchPolicyError):
            RuntimeDispatchPolicy(model_affinity=True)
        for changes in (
            {"work_conserving_admission": 1},
            {"affinity_maximum_bypasses": -1},
            {"affinity_maximum_wait_us": True},
        ):
            with self.subTest(changes=changes), self.assertRaises(
                RuntimeDispatchPolicyError
            ):
                RuntimeDispatchPolicy(**changes)
        policy = RuntimeDispatchPolicy(
            work_conserving_admission=True,
            model_affinity=True,
            affinity_maximum_bypasses=2,
            affinity_maximum_wait_us=5,
        )
        self.assertEqual(RuntimeDispatchPolicy.from_json(policy.to_json()), policy)
        with self.assertRaises(RuntimeDispatchPolicyError):
            RuntimeDispatchPolicy.from_json({"work_conserving": True})

    def test_queue_policy_changes_only_while_empty(self) -> None:
        queue = RuntimeDispatchQueue()
        queue.admit(leased("a", ("gpu", (0,), 0, 10)), 0)
        with self.assertRaises(RuntimeQueueError):
            queue.set_policy(WORK_CONSERVING)
        self.assertNotIn("dispatch_policy", queue.snapshot())

    def test_continuous_join_fields_are_validated_and_omitted_when_default(self) -> None:
        with self.assertRaises(RuntimeDispatchPolicyError):
            RuntimeDispatchPolicy(continuous_join=True)
        for changes in (
            {"work_conserving_admission": True, "continuous_join": 1},
            {"max_barrier_extension_s": -1},
            {"max_barrier_extension_s": True},
            {"max_barrier_extension_s": 2.0},
        ):
            with self.subTest(changes=changes), self.assertRaises(
                RuntimeDispatchPolicyError
            ):
                RuntimeDispatchPolicy(**changes)
        legacy = WORK_CONSERVING.to_json()
        self.assertNotIn("continuous_join", legacy)
        self.assertNotIn("max_barrier_extension_s", legacy)
        self.assertEqual(RuntimeDispatchPolicy.from_json(legacy), WORK_CONSERVING)
        self.assertFalse(WORK_CONSERVING.precedence_enabled)
        policy = replace(CONTINUOUS_JOIN, max_barrier_extension_s=3)
        self.assertEqual(policy.to_json()["continuous_join"], True)
        self.assertEqual(policy.to_json()["max_barrier_extension_s"], 3)
        self.assertNotIn("max_barrier_extension_s", CONTINUOUS_JOIN.to_json())
        self.assertEqual(RuntimeDispatchPolicy.from_json(policy.to_json()), policy)
        self.assertTrue(policy.enabled)
        self.assertTrue(policy.precedence_enabled)

    def test_continuous_join_enables_queue_precedence(self) -> None:
        queue = RuntimeDispatchQueue(CONTINUOUS_JOIN)
        queue.admit(leased("a1", ("gpu", (0,), 0, 10)), 0)
        acquire(queue, "a1", 0)
        queue.admit(leased("b1", ("gpu", (0, 1), 20, 30)), 1,
                    residency_transition_barrier=True)
        queue.admit(leased("a2", ("gpu", (1,), 25, 40)), 2,
                    precede_request_ids=("b1",))
        self.assertIn("a2", predecessors(queue, "b1"))
        self.assertEqual(predecessors(queue, "a2"), ())


def wait_for(predicate, timeout_s: float = 5.0) -> None:
    """Poll ``predicate`` (another thread's progress) until true."""
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.005)


class ResidencyHysteresisTests(unittest.TestCase):
    """A queued residency change of another model waits H after the resident model's last release."""

    def test_policy_field_is_validated_and_omitted_when_default(self) -> None:
        with self.assertRaises(RuntimeDispatchPolicyError):
            RuntimeDispatchPolicy(residency_hysteresis_s=5)
        for changes in (
            {"work_conserving_admission": True, "residency_hysteresis_s": -1},
            {"work_conserving_admission": True, "residency_hysteresis_s": True},
            {"work_conserving_admission": True, "residency_hysteresis_s": 2.0},
        ):
            with self.subTest(changes=changes), self.assertRaises(RuntimeDispatchPolicyError):
                RuntimeDispatchPolicy(**changes)
        legacy = AFFINITY.to_json()
        self.assertNotIn("residency_hysteresis_s", legacy)
        self.assertEqual(RuntimeDispatchPolicy.from_json(legacy), AFFINITY)
        self.assertEqual(HYSTERESIS.to_json()["residency_hysteresis_s"], HYSTERESIS_S)
        self.assertEqual(RuntimeDispatchPolicy.from_json(HYSTERESIS.to_json()), HYSTERESIS)
        self.assertEqual(HYSTERESIS.residency_hysteresis_us, HYSTERESIS_US)
        # Without affinity the window still holds; the hold alone needs work conservation.
        RuntimeDispatchPolicy(work_conserving_admission=True, residency_hysteresis_s=1)

    @staticmethod
    def released_owner_and_switch(policy, *, switch_start_us=1_000, keyed=True):
        """a1 (model A) ran alone and released at 1,000; b1 (model B) switches after it."""
        queue = RuntimeDispatchQueue(policy)
        key = {"residency_hysteresis_key": "A"} if keyed else {}
        queue.admit(leased("a1", ("gpu", (0,), 0, 1_000)), 0, **key)
        acquire(queue, "a1", 0)
        queue.admit(
            leased("b1", ("gpu", (0, 1), switch_start_us, switch_start_us + 200)), 10,
            residency_transition_barrier=True,
            **({"residency_hysteresis_key": "B"} if keyed else {}),
        )
        queue.complete("a1", 1_000)
        return queue

    def test_admit_validates_the_key(self) -> None:
        queue = RuntimeDispatchQueue(HYSTERESIS)
        for key in ("", 1, True):
            with self.subTest(key=key), self.assertRaises(RuntimeQueueError):
                queue.admit(leased("a1", ("gpu", (0,), 0, 10)), 0, residency_hysteresis_key=key)

    def test_switch_of_another_model_is_held_h_after_the_release(self) -> None:
        queue = self.released_owner_and_switch(SPECULATIVE)
        self.assertEqual(queue.residency_hysteresis_until("b1"), 1_000 + HYSTERESIS_US)
        # The hold never exceeds H beyond the change's own start.
        late = self.released_owner_and_switch(SPECULATIVE, switch_start_us=1_000 + 2 * HYSTERESIS_US)
        self.assertEqual(late.residency_hysteresis_until("b1"), 1_000 + HYSTERESIS_US)
        self.assertLess(
            late.residency_hysteresis_until("b1"),
            late.dispatch_order_view()["b1"]["scheduled_start_us"]
            if "scheduled_start_us" in late.dispatch_order_view()["b1"]
            else 1_000 + 2 * HYSTERESIS_US,
        )

    def test_no_hold_without_the_key_the_policy_or_for_the_same_model(self) -> None:
        self.assertIsNone(self.released_owner_and_switch(AFFINITY).residency_hysteresis_until("b1"))
        self.assertIsNone(
            self.released_owner_and_switch(HYSTERESIS, keyed=False).residency_hysteresis_until("b1")
        )
        self.assertNotIn("residency_hysteresis_holds", self.released_owner_and_switch(AFFINITY).policy_events())
        queue = RuntimeDispatchQueue(HYSTERESIS)
        queue.admit(leased("a1", ("gpu", (0,), 0, 1_000)), 0, residency_hysteresis_key="A")
        acquire(queue, "a1", 0)
        # A reload of the resident model (cold plan) is a barrier of the same model: not held.
        queue.admit(leased("a2", ("gpu", (0, 1), 1_000, 1_200)), 10,
                    residency_transition_barrier=True, residency_hysteresis_key="A")
        queue.complete("a1", 1_000)
        self.assertIsNone(queue.residency_hysteresis_until("a2"))
        self.assertTrue(dispatchable(queue, "a2", 1_000))

    def test_short_same_model_arrival_inside_the_window_runs_before_the_held_switch(self) -> None:
        queue = self.released_owner_and_switch(SPECULATIVE)
        # Frees its lane before the hold ends: work-conserving order puts it first.
        queue.admit(leased("a2", ("gpu", (0,), 2_000, 5_000)), 2_000, residency_hysteresis_key="A")
        self.assertEqual(predecessors(queue, "a2"), ())
        self.assertIn("a2", predecessors(queue, "b1"))
        self.assertTrue(dispatchable(queue, "a2", 2_000))
        acquire(queue, "a2", 2_000)
        self.assertEqual(queue.policy_events()["residency_hysteresis_admissions"], 1)
        # A long arrival needs affinity's precedence (the window ends before it would).
        queue.admit(leased("a3", ("gpu", (1,), 3_000, 1_000 + 2 * HYSTERESIS_US)), 3_000,
                    residency_hysteresis_key="A", precede_request_ids=("b1",))
        self.assertIn("a3", predecessors(queue, "b1"))
        self.assertEqual(queue.policy_events()["residency_hysteresis_admissions"], 2)
        checkpoint = queue.checkpoint()
        queue.restore(checkpoint)
        self.assertEqual(queue.checkpoint(), checkpoint)

    def test_waiter_observes_the_hold_and_the_switch_starts_when_it_ends(self) -> None:
        queue = self.released_owner_and_switch(SPECULATIVE)
        receipts = []
        waiter = threading.Thread(
            target=lambda: receipts.append(
                queue.wait_ready("b1", time.monotonic_ns() - 1_500 * 1_000)
            ),
        )
        waiter.start()
        wait_for(lambda: queue.residency_hysteresis_hold("b1") is not None)
        self.assertEqual(
            dict(queue.residency_hysteresis_hold("b1")),
            {"held_until_us": 1_000 + HYSTERESIS_US, "released_at_us": 1_000},
        )
        self.assertEqual(queue.policy_events()["residency_hysteresis_holds"], 1)
        # A same-model arrival displaces the held change (affinity); the waiter wakes to replan.
        self.assertTrue(queue.require_replan("b1", "model_affinity_displaced"))
        waiter.join(timeout=5)
        self.assertFalse(waiter.is_alive())
        self.assertEqual(receipts[0].status, "REPLAN_REQUIRED")
        queue.retire_replan("b1")
        queue.admit(leased("b1", ("gpu", (0, 1), 1_000, 1_200)), 20,
                    residency_transition_barrier=True, residency_hysteresis_key="B")
        # After the window the change acquires, naming the hysteresis wake.
        receipt = queue.wait("b1", time.monotonic_ns() - (1_000 + HYSTERESIS_US) * 1_000)
        self.assertEqual(receipt.status, "ACQUIRED")
        self.assertEqual(receipt.wake_reason, "residency_hysteresis_released")
        self.assertEqual(queue.policy_events()["residency_hysteresis_holds"], 1)

    def test_commit_refuses_a_receipt_observed_inside_the_window(self) -> None:
        queue = self.released_owner_and_switch(SPECULATIVE)
        early = queue.wait_ready("b1", time.monotonic_ns() - (1_000 + HYSTERESIS_US) * 1_000)
        self.assertEqual(early.status, "ACQUIRED")
        inside = replace(early, observed_at_us=1_500)
        self.assertIsNone(queue.commit_ready(inside))
        self.assertIsNotNone(queue.commit_ready(early))

    def test_checkpoint_restores_the_release_and_the_holds(self) -> None:
        queue = self.released_owner_and_switch(SPECULATIVE)
        # The decision is taken at the first evaluation and checkpointed with the release.
        self.assertEqual(queue.residency_hysteresis_until("b1"), 1_000 + HYSTERESIS_US)
        checkpoint = queue.checkpoint()
        self.assertEqual(checkpoint.residency_release, (1_000, "A", ("gpu",)))
        other = RuntimeDispatchQueue(SPECULATIVE)
        other.restore(checkpoint)
        self.assertEqual(other.residency_hysteresis_until("b1"), 1_000 + HYSTERESIS_US)
        self.assertEqual(other.checkpoint(), checkpoint)
        for broken in (
            replace(checkpoint, residency_release=(1_000, "")),
            replace(checkpoint, residency_release=(1_000, "", ("gpu",))),
            replace(checkpoint, residency_release=(True, "A", ("gpu",))),
            replace(checkpoint, residency_release=(1_000, "A", ["gpu"])),
            replace(checkpoint, residency_hysteresis_holds=(("b1", 5, 10),)),
        ):
            with self.subTest(broken=broken), self.assertRaises(RuntimeQueueError):
                RuntimeDispatchQueue(SPECULATIVE).restore(broken)


class ResidencyHysteresisGateTests(unittest.TestCase):
    """s1a: the hold is applied only when it can pay (queued or likely same-model work)."""

    @staticmethod
    def history(queue, key, *admitted_at_us, lanes=("gpu", (0,))):
        """Admit, run and complete one request of ``key`` per admission time (its arrivals)."""
        for index, at_us in enumerate(admitted_at_us):
            request_id = f"{key.lower()}-history-{index}"
            queue.admit(leased(request_id, (lanes[0], lanes[1], at_us, at_us + 10)), at_us,
                        residency_hysteresis_key=key)
            acquire(queue, request_id, at_us)
            queue.complete(request_id, at_us + 10)

    def released_switch(self, policy=HYSTERESIS, *, history_us=(), switch_admitted_us=None,
                        switch_lanes=("gpu", (0, 1)), release_us=None):
        """Model A's history, then a1 (A) runs and releases; b1 (model B) is the queued switch."""
        queue = RuntimeDispatchQueue(policy)
        self.history(queue, "A", *history_us)
        start = (history_us[-1] + 100) if history_us else 0
        release_us = start + 1_000 if release_us is None else release_us
        queue.admit(leased("a1", ("gpu", (0,), start, release_us)), start,
                    residency_hysteresis_key="A")
        acquire(queue, "a1", start)
        admitted_us = start + 10 if switch_admitted_us is None else switch_admitted_us
        queue.admit(
            leased("b1", (switch_lanes[0], switch_lanes[1], release_us, release_us + 200)),
            admitted_us, residency_transition_barrier=True, residency_hysteresis_key="B",
        )
        queue.complete("a1", release_us)
        return queue, release_us

    def test_s1a_switch_that_waited_longer_than_the_window_is_not_held(self) -> None:
        # Qwen 001 was admitted 204 s before Gemma 000 released: holding it is unfair.
        queue, release_us = self.released_switch(
            SPECULATIVE, switch_admitted_us=5, release_us=5 + HYSTERESIS_US + 1,
        )
        self.assertIsNone(queue.residency_hysteresis_until("b1"))
        self.assertTrue(dispatchable(queue, "b1", release_us))
        decision = queue.residency_hysteresis_decision("b1")
        self.assertEqual(decision["reason"], "OWN_MODEL_WAITED_LONGER_THAN_HYSTERESIS")
        self.assertFalse(decision["held"])
        self.assertEqual(queue.policy_events()["residency_hysteresis_skips"], 1)
        receipt = queue.wait("b1", time.monotonic_ns() - (release_us + 1) * 1_000)
        self.assertEqual(receipt.status, "ACQUIRED")
        self.assertNotEqual(receipt.wake_reason, "residency_hysteresis_released")
        self.assertNotIn("residency_hysteresis_holds", queue.policy_events())

    def test_switch_without_arrival_history_is_not_held(self) -> None:
        queue, release_us = self.released_switch()
        self.assertIsNone(queue.residency_hysteresis_until("b1"))
        self.assertEqual(
            dict(queue.residency_hysteresis_decision("b1")),
            {"barrier_request_id": "b1", "held": False,
             "reason": "ARRIVAL_HISTORY_INSUFFICIENT", "released_at_us": release_us},
        )
        # The decision is taken once per release window.
        queue.residency_hysteresis_until("b1")
        self.assertEqual(queue.policy_events()["residency_hysteresis_skips"], 1)
        self.assertEqual(len(queue.residency_hysteresis_decisions()), 1)

    def test_s1a_unlikely_same_model_arrival_is_not_held(self) -> None:
        # Gemma/Qwen arrived every ~250-500 s: P(arrival within 20 s) is about 7-10 %
        # (a1 itself is one more arrival right after the history).
        gap_us = 250_000_000
        queue, _ = self.released_switch(history_us=(0, gap_us, 2 * gap_us))
        self.assertIsNone(queue.residency_hysteresis_until("b1"))
        decision = queue.residency_hysteresis_decision("b1")
        self.assertEqual(decision["reason"], "ARRIVAL_UNLIKELY")
        self.assertLess(decision["arrival_probability_ppm"], 150_000)
        self.assertGreater(decision["arrival_probability_ppm"], 50_000)

    def test_likely_same_model_arrival_holds_the_switch(self) -> None:
        gap_us = 5_000_000
        queue, release_us = self.released_switch(history_us=(0, gap_us, 2 * gap_us))
        self.assertEqual(queue.residency_hysteresis_until("b1"), release_us + HYSTERESIS_US)
        decision = queue.residency_hysteresis_decision("b1")
        self.assertEqual(decision["reason"], "SAME_MODEL_ARRIVAL_LIKELY")
        self.assertEqual(decision["held_until_us"], release_us + HYSTERESIS_US)
        self.assertGreater(decision["arrival_probability_ppm"], 950_000)
        self.assertNotIn("residency_hysteresis_skips", queue.policy_events())
        # A stricter threshold than the prediction skips the same window.
        strict = replace(HYSTERESIS, residency_hysteresis_min_probability_ppm=999_999)
        queue, _ = self.released_switch(strict, history_us=(0, gap_us, 2 * gap_us))
        self.assertIsNone(queue.residency_hysteresis_until("b1"))

    def test_queued_same_model_request_holds_until_it_can_no_longer_run_first(self) -> None:
        queue, release_us = self.released_switch()
        queue.admit(leased("a2", ("gpu", (1,), release_us + 50, release_us + 90)), release_us,
                    residency_hysteresis_key="A")
        self.assertTrue(queue.require_replan("a2", "capacity_released_early"))
        # a2 is not yet admissible (replanning) and does not wait on the switch: hold.
        self.assertEqual(queue.residency_hysteresis_until("b1"), release_us + HYSTERESIS_US)
        self.assertEqual(queue.residency_hysteresis_decision("b1")["reason"], "SAME_MODEL_QUEUED")
        queue.wait("a2", time.monotonic_ns() - (release_us + 50) * 1_000)
        queue.retire_replan("a2")
        queue.admit(leased("a2", ("gpu", (1,), release_us + 50, release_us + 90)),
                    release_us + 50, residency_hysteresis_key="A")
        acquire(queue, "a2", release_us + 50)
        # a2 is dispatched: the hold it justified ends (b1 still waits on a2's lanes).
        self.assertEqual(state(queue, "b1"), "QUEUED")
        self.assertIsNone(queue.residency_hysteresis_until("b1"))
        # Its admission and its replanned admission both happened inside the hold.
        self.assertEqual(queue.policy_events()["residency_hysteresis_admissions"], 2)

    def test_same_model_request_queued_behind_the_switch_does_not_hold_it(self) -> None:
        # s1a Gemma 002 was queued behind the Qwen 001 switch before Gemma 000 released.
        queue = RuntimeDispatchQueue(HYSTERESIS)
        queue.admit(leased("a1", ("gpu", (0,), 0, 1_000)), 0, residency_hysteresis_key="A")
        acquire(queue, "a1", 0)
        queue.admit(leased("b1", ("gpu", (0, 1), 1_000, 1_200)), 10,
                    residency_transition_barrier=True, residency_hysteresis_key="B")
        queue.admit(leased("a2", ("gpu", (0,), 1_200, 1_900)), 20, residency_hysteresis_key="A")
        self.assertIn("b1", predecessors(queue, "a2"))
        queue.complete("a1", 1_000)
        self.assertIsNone(queue.residency_hysteresis_until("b1"))
        self.assertEqual(
            queue.residency_hysteresis_decision("b1")["reason"], "ARRIVAL_HISTORY_INSUFFICIENT"
        )

    def test_change_on_another_residency_resource_is_not_held(self) -> None:
        queue, _ = self.released_switch(SPECULATIVE, switch_lanes=("npu", (0,)))
        self.assertIsNone(queue.residency_hysteresis_until("b1"))
        self.assertEqual(
            queue.residency_hysteresis_decision("b1")["reason"], "OTHER_RESIDENCY_RESOURCE"
        )

    def test_arrival_history_learns_first_admissions_only(self) -> None:
        queue = RuntimeDispatchQueue(HYSTERESIS)
        self.history(queue, "A", 0, 1_000, 3_000)
        with queue._condition:
            history = queue._residency_arrivals["A"]
        self.assertEqual((history.last_admitted_at_us, history.gap_count), (3_000, 2))
        self.assertEqual(history.mean_gap_us, (3 * 1_000 + 2_000) // 4)
        queue.admit(leased("a9", ("gpu", (0,), 4_000, 5_000)), 4_000, residency_hysteresis_key="A")
        self.assertTrue(queue.require_replan("a9", "capacity_released_early"))
        queue.wait("a9", time.monotonic_ns() - 4_000 * 1_000)
        queue.retire_replan("a9")
        queue.admit(leased("a9", ("gpu", (0,), 4_500, 5_000)), 4_500, residency_hysteresis_key="A")
        with queue._condition:
            self.assertEqual(queue._residency_arrivals["A"].gap_count, 3)
            self.assertEqual(queue._entries["a9"].queued_since_us, 4_000)
        # Without the window nothing is learned.
        plain = RuntimeDispatchQueue(AFFINITY)
        plain.admit(leased("a1", ("gpu", (0,), 0, 10)), 0, residency_hysteresis_key="A")
        self.assertEqual(plain.checkpoint().residency_arrivals, ())

    def test_checkpoint_carries_decisions_and_history_and_rejects_bad_rows(self) -> None:
        queue, _ = self.released_switch(history_us=(0, 5_000_000, 10_000_000))
        queue.residency_hysteresis_until("b1")
        checkpoint = queue.checkpoint()
        other = RuntimeDispatchQueue(HYSTERESIS)
        other.restore(checkpoint)
        self.assertEqual(other.checkpoint(), checkpoint)
        self.assertEqual(other.residency_hysteresis_decisions(), queue.residency_hysteresis_decisions())
        decision = checkpoint.residency_hysteresis_decisions[0]
        for broken in (
            replace(checkpoint, residency_arrivals=(("A", None),)),
            replace(checkpoint, residency_arrivals=[]),
            replace(checkpoint, residency_hysteresis_decisions=(decision, decision)),
            replace(checkpoint, residency_hysteresis_log=("decision",)),
        ):
            with self.subTest(broken=broken), self.assertRaises(RuntimeQueueError):
                RuntimeDispatchQueue(HYSTERESIS).restore(broken)

    def test_probability_threshold_is_validated_and_serialized_only_when_set(self) -> None:
        self.assertEqual(HYSTERESIS.residency_hysteresis_min_probability_ppm, 500_000)
        self.assertNotIn("residency_hysteresis_min_probability_ppm", HYSTERESIS.to_json())
        self.assertEqual(SPECULATIVE.to_json()["residency_hysteresis_min_probability_ppm"], 0)
        self.assertEqual(RuntimeDispatchPolicy.from_json(SPECULATIVE.to_json()), SPECULATIVE)
        for value in (-1, 1_000_001, True, 0.5, "500000", None):
            with self.subTest(value=value), self.assertRaises(RuntimeDispatchPolicyError):
                replace(HYSTERESIS, residency_hysteresis_min_probability_ppm=value)
        with self.assertRaises(RuntimeDispatchPolicyError):
            RuntimeDispatchPolicy(work_conserving_admission=True,
                                  residency_hysteresis_min_probability_ppm=0)


class WorkConservingQueueTests(unittest.TestCase):
    """Queue order for work that keeps the residency versus a residency change."""

    def running_owner_and_switch(self, policy: RuntimeDispatchPolicy):
        queue = RuntimeDispatchQueue(policy)
        work_conserving = policy.work_conserving_admission
        queue.admit(
            leased("a1", ("gpu", (0,), 0, 1_000)), 0,
            residency_transition_barrier=not work_conserving,
        )
        acquire(queue, "a1", 0)
        queue.admit(
            leased("b1", ("gpu", (0, 1), 1_100, 1_200), ("gpu", (0,), 1_200, 1_300)),
            10,
            residency_transition_barrier=True,
        )
        return queue

    def test_same_residency_work_runs_before_a_later_residency_change(self) -> None:
        queue = self.running_owner_and_switch(WORK_CONSERVING)
        queue.admit(leased("a2", ("gpu", (1,), 100, 400)), 100)
        self.assertEqual(predecessors(queue, "a2"), ())
        self.assertIn("a2", predecessors(queue, "b1"))
        self.assertTrue(dispatchable(queue, "a2", 100))
        acquire(queue, "a2", 100)

    def test_policy_off_keeps_arrival_order_behind_the_residency_change(self) -> None:
        queue = self.running_owner_and_switch(RuntimeDispatchPolicy())
        queue.admit(
            leased("a2", ("gpu", (1,), 100, 400)), 100,
            residency_transition_barrier=True,
        )
        self.assertIn("b1", predecessors(queue, "a2"))

    def test_overlapping_same_residency_work_keeps_arrival_order(self) -> None:
        queue = self.running_owner_and_switch(WORK_CONSERVING)
        queue.admit(leased("a2", ("gpu", (1,), 100, 1_150)), 100)
        self.assertIn("b1", predecessors(queue, "a2"))

    def test_residency_changes_keep_arrival_order(self) -> None:
        queue = self.running_owner_and_switch(WORK_CONSERVING)
        queue.admit(
            leased("c1", ("gpu", (1,), 100, 200)), 100,
            residency_transition_barrier=True,
        )
        self.assertIn("b1", predecessors(queue, "c1"))

    def test_cancelled_residency_change_is_bounded_by_its_running_predecessors(self) -> None:
        queue = self.running_owner_and_switch(WORK_CONSERVING)
        self.assertTrue(queue.require_replan(
            "b1", "residency_observation_changed", defer_behind_predecessors=True,
        ))
        self.assertEqual(state(queue, "b1"), "DEFERRED_REPLAN")
        queue.admit(leased("a2", ("gpu", (1,), 100, 900)), 100)
        self.assertIn("a2", predecessors(queue, "b1"))
        queue.admit(leased("a3", ("gpu", (1,), 900, 1_500)), 110)
        self.assertIn("b1", predecessors(queue, "a3"))

    def test_published_owner_releases_its_deferred_followers(self) -> None:
        for policy, expected in (
            (WORK_CONSERVING, "REPLAN_REQUIRED"),
            (RuntimeDispatchPolicy(), "DEFERRED_REPLAN"),
        ):
            with self.subTest(policy=policy):
                queue = RuntimeDispatchQueue(policy)
                queue.admit(
                    leased("a1", ("gpu", (0, 1), 0, 100), ("gpu", (0,), 100, 1_000)),
                    0, residency_transition_barrier=True,
                )
                acquire(queue, "a1", 0)
                queue.admit(
                    leased("a2", ("gpu", (1,), 90, 400)), 10,
                    residency_transition_barrier=not policy.work_conserving_admission,
                )
                queue.admit(
                    leased("a3", ("gpu", (1,), 400, 600)), 20,
                    residency_transition_barrier=not policy.work_conserving_admission,
                )
                self.assertIn("a1", predecessors(queue, "a2"))
                queue.require_replan(
                    "a2", "preparation_phase_completed",
                    defer_behind_predecessors=True,
                )
                queue.require_replan(
                    "a3", "residency_projection_invalid",
                    defer_behind_predecessors=True,
                )
                queue.release_prepare_leases("a1", ("a1:0",), preparation_complete=True)
                self.assertEqual(state(queue, "a2"), expected)
                self.assertEqual(state(queue, "a3"), "DEFERRED_REPLAN")

    def test_early_completion_offers_freed_lanes_to_same_residency_work(self) -> None:
        for policy, expected in (
            (WORK_CONSERVING, ("a3",)),
            (RuntimeDispatchPolicy(), ()),
        ):
            with self.subTest(policy=policy):
                queue = RuntimeDispatchQueue(policy)
                barrier = not policy.work_conserving_admission
                queue.admit(leased("a1", ("gpu", (0,), 0, 1_000)), 0,
                            residency_transition_barrier=barrier)
                queue.admit(leased("a2", ("gpu", (1,), 0, 2_000)), 1,
                            residency_transition_barrier=barrier)
                acquire(queue, "a1", 0)
                acquire(queue, "a2", 0)
                queue.admit(leased("a3", ("gpu", (1,), 2_000, 2_500)), 2,
                            residency_transition_barrier=barrier)
                self.assertEqual(predecessors(queue, "a3"), ("a2",))
                frontier = queue.early_completion_frontier(
                    "a1", 300, {"a1:0": 1_000}, {"gpu": 2}
                )
                self.assertEqual(frontier, expected)
                if frontier:
                    queue.require_replan(
                        "a3", "capacity_released_early",
                        defer_behind_predecessors=True,
                    )
                    queue.release_capacity(
                        "a1", 300, early_replan_request_ids=frontier
                    )
                    self.assertEqual(state(queue, "a3"), "REPLAN_REQUIRED")
                    self.assertEqual(
                        queue.policy_events()["early_capacity_promotions"], 1
                    )

    def test_precedence_requires_affinity_and_queued_targets(self) -> None:
        with self.assertRaises(RuntimeQueueError):
            RuntimeDispatchQueue(WORK_CONSERVING).admit(
                leased("x", ("gpu", (0,), 0, 10)), 0, precede_request_ids=("y",)
            )
        policy = RuntimeDispatchPolicy(
            work_conserving_admission=True, model_affinity=True
        )
        queue = RuntimeDispatchQueue(policy)
        queue.admit(leased("a1", ("gpu", (0,), 0, 10)), 0)
        acquire(queue, "a1", 0)
        queue.admit(leased("b1", ("gpu", (0, 1), 20, 30)), 1,
                    residency_transition_barrier=True)
        for targets in (("absent",), ("a1",), ("a2",)):
            with self.subTest(targets=targets), self.assertRaises(RuntimeQueueError):
                queue.admit(leased("a2", ("gpu", (1,), 25, 40)), 2,
                            precede_request_ids=targets)
        queue.admit(leased("a2", ("gpu", (1,), 25, 40)), 2,
                    precede_request_ids=("b1",))
        self.assertIn("a2", predecessors(queue, "b1"))
        checkpoint = queue.checkpoint()
        queue.restore(checkpoint)
        self.assertEqual(queue.checkpoint(), checkpoint)

    def test_replanned_admission_precedes_the_displaced_switch(self) -> None:
        policy = RuntimeDispatchPolicy(
            work_conserving_admission=True, model_affinity=True
        )
        queue = self.running_owner_and_switch(policy)
        queue.admit(leased("b2", ("gpu", (0, 1), 1_300, 1_350)), 15,
                    residency_transition_barrier=True)
        queue.admit(
            leased("a2", ("gpu", (0, 1), 1_350, 1_400), ("gpu", (1,), 1_400, 1_600)),
            20, residency_transition_barrier=True,
        )
        self.assertIn("b1", predecessors(queue, "b2"))
        queue.require_replan("a2", "residency_observation_changed")
        queue.retire_replan("a2")
        with self.assertRaises(RuntimeQueueError):
            # a2 still waits on b1 through b2, so b1 cannot follow it.
            queue.admit(leased("a2", ("gpu", (1,), 150, 600)), 150,
                        precede_request_ids=("b1",))
        queue = self.running_owner_and_switch(policy)
        queue.admit(
            leased("a2", ("gpu", (0, 1), 1_300, 1_400), ("gpu", (1,), 1_400, 1_600)),
            20, residency_transition_barrier=True,
        )
        queue.require_replan("b1", "model_affinity_displaced")
        queue.require_replan("a2", "residency_observation_changed")
        queue.retire_replan("a2")
        queue.admit(leased("a2", ("gpu", (1,), 150, 600)), 150,
                    precede_request_ids=("b1",))
        self.assertEqual(predecessors(queue, "a2"), ())
        self.assertEqual(set(predecessors(queue, "b1")), {"a1", "a2"})
        self.assertTrue(dispatchable(queue, "a2", 150))
        checkpoint = queue.checkpoint()
        queue.restore(checkpoint)
        self.assertEqual(queue.checkpoint(), checkpoint)


class DispatchPolicySchedulerTests(unittest.TestCase):
    """Two models share one exclusive two-lane accelerator; model A is resident."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.paths = (
            Path(self.directory.name) / "model-a.gguf",
            Path(self.directory.name) / "model-b.gguf",
        )
        write_synthetic_gguf(self.paths[0], block_count=2, sliding_window=32)
        write_synthetic_gguf(self.paths[1])

    def scheduler(self, policy: RuntimeDispatchPolicy | None = None):
        source = catalog()
        gpu = replace(
            source.executor_by_device["accelerator-b"],
            qualified_fallback=True,
            exclusive_residency_resource_id="compute:accelerator-b",
        )
        transition = replace(
            next(row for row in source.transitions if row.device_id == "accelerator-b"),
            executor_id=gpu.executor_id,
            prepares_device_ids=("accelerator-b",),
            resource_slots={"compute:accelerator-b": 2},
        )
        resources = dict(source.resources)
        for resource_id in ("compute:accelerator-b", "link:pcie-in", "link:pcie-out"):
            resources[resource_id] = replace(resources[resource_id], capacity=2)
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        if policy is not None:
            scheduler.configure_runtime_dispatch_policy(policy)
        scheduler.register_runtime_capabilities(RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=(gpu,),
            composite_executors=(),
            transitions=(transition,),
            resources=resources,
        ).to_json()))
        model_a = scheduler.register_gguf_model("model-a", self.paths[0])
        model_b = scheduler.register_gguf_model("model-b", self.paths[1])
        hot = runtime_snapshot(
            model_a, include_phone=False, resident_devices=("accelerator-b",),
            gpu_free_slots=2,
        )
        hot = replace(
            hot,
            executors={gpu.executor_id: executor_state(gpu.executor_id, free_slots=2)},
            residency=tuple(replace(row, executor_id=gpu.executor_id) for row in hot.residency),
        )
        return scheduler, model_a, model_b, hot

    @staticmethod
    def submit(scheduler, model, snapshot, request_id, arrival_us, output_tokens):
        return scheduler.submit_automated_request(
            request(request_id, arrival_us=arrival_us, output_tokens=output_tokens),
            model.model_id, snapshot, selection_mode="desktop-baseline",
        )

    @staticmethod
    def queue_view(scheduler):
        return scheduler._runtime_controller.dispatch_order_view()

    @staticmethod
    def ready_now(scheduler, request_id: str, at_us: int) -> bool:
        return dispatchable(scheduler._runtime_controller.queue, request_id, at_us)

    def same_model_behind_far_switch(self, policy):
        scheduler, model_a, model_b, hot = self.scheduler(policy)
        a1 = self.submit(scheduler, model_a, hot, "a1", 1_000, 640)
        b1 = self.submit(scheduler, model_b, hot, "b1", 1_100, 8)
        a2 = self.submit(scheduler, model_a, hot, "a2", 1_200, 4)
        return scheduler, a1, b1, a2

    def test_policy_off_keeps_same_model_arrival_behind_the_switch(self) -> None:
        scheduler, a1, b1, a2 = self.same_model_behind_far_switch(None)
        view = self.queue_view(scheduler)
        self.assertEqual(a2.execution_plan.transitions, ())
        self.assertIn("b1", view["a2"]["predecessor_request_ids"])
        self.assertTrue(all(row["residency_transition_barrier"] for row in view.values()))
        self.assertFalse(self.ready_now(scheduler, "a2", a2.decision.start_us))

    def test_same_model_arrival_joins_the_running_server(self) -> None:
        scheduler, a1, b1, a2 = self.same_model_behind_far_switch(WORK_CONSERVING)
        view = self.queue_view(scheduler)
        self.assertFalse(view["a2"]["residency_transition_barrier"])
        self.assertTrue(view["b1"]["residency_transition_barrier"])
        self.assertEqual(view["a2"]["predecessor_request_ids"], ())
        self.assertIn("a2", view["b1"]["predecessor_request_ids"])
        scheduler.wait_runtime_request("a1", time.monotonic_ns() - a1.decision.start_us * 1_000)
        self.assertEqual(a2.decision.start_us, 1_200)
        self.assertTrue(self.ready_now(scheduler, "a2", 1_200))
        active = scheduler.wait_runtime_request(
            "a2", time.monotonic_ns() - a2.decision.start_us * 1_000
        )
        self.assertEqual(active.dispatch_state, "ACQUIRED")

    def test_residency_change_still_waits_for_every_exclusive_lease(self) -> None:
        scheduler, a1, b1, a2 = self.same_model_behind_far_switch(WORK_CONSERVING)
        self.assertTrue(b1.execution_plan.transitions)
        exclusive_end = max(
            lease.reserved_until_us
            for ticket in (a1, a2)
            for lease in ticket.decision.leases
            if lease.resource_id == "compute:accelerator-b"
        )
        self.assertGreaterEqual(b1.decision.start_us, exclusive_end)
        self.assertEqual(
            set(self.queue_view(scheduler)["b1"]["predecessor_request_ids"]), {"a1", "a2"}
        )

    def test_long_replan_after_load_precedes_later_model_switch(self) -> None:
        scheduler, model_a, model_b, hot = self.scheduler(WORK_CONSERVING)
        cold = replace(hot, residency=())
        first = self.submit(scheduler, model_a, cold, "a1", 1_000, 640)
        for request_id, model, arrival_us, output_tokens in (
            ("a2", model_a, 1_100, 640),
            ("a3", model_a, 1_200, 8),
            ("b1", model_b, 1_300, 8),
            ("a4", model_a, 1_400, 8),
            ("b2", model_b, 1_500, 8),
        ):
            self.submit(scheduler, model, cold, request_id, arrival_us, output_tokens)
        active = scheduler.wait_runtime_request(
            "a1", time.monotonic_ns() - first.decision.start_us * 1_000
        )
        loaded_at_us = max(first.decision.start_us + 150, 1_600)
        scheduler.record_automated_transition_receipts("a1", tuple(
            replace(row, finished_us=loaded_at_us)
            for row in FakeAutomatedPhysicalAdapter._transition_receipts(active)
        ))
        wake = scheduler.wait_runtime_request(
            "a2", time.monotonic_ns() - loaded_at_us * 1_000
        )
        self.assertEqual(wake.dispatch_receipt.wake_reason, "preparation_phase_completed")
        snapshot = replace(
            hot, snapshot_id="published", captured_at_us=loaded_at_us,
            valid_until_us=loaded_at_us + 10_000_000,
            memory=replace(hot.memory, snapshot_id="published-memory",
                           captured_at_us=loaded_at_us,
                           valid_until_us=loaded_at_us + 10_000_000),
        )
        replacement = scheduler.replan_automated_request(
            "a2", observed_at_us=loaded_at_us,
            reason=wake.dispatch_receipt.wake_reason, snapshot=snapshot,
            expected_ticket_id=wake.ticket_id,
            expected_queue_generation=wake.dispatch_receipt.queue_generation,
        )
        self.assertEqual(replacement.execution_plan.transitions, ())
        self.assertEqual(replacement.decision.start_us, loaded_at_us)
        self.assertTrue(self.ready_now(scheduler, "a2", loaded_at_us))
        self.assertEqual(scheduler.runtime_ticket("a1").dispatch_state, "ACQUIRED")
        switch = scheduler.runtime_ticket("b1")
        self.assertGreaterEqual(switch.decision.start_us, replacement.decision.finish_us)
        self.assertIn("a2", self.queue_view(scheduler)["b1"]["predecessor_request_ids"])

    def test_queued_request_joins_the_server_when_its_load_is_published(self) -> None:
        for policy, replanned in ((WORK_CONSERVING, True), (None, False)):
            with self.subTest(policy=policy):
                scheduler, model_a, _, hot = self.scheduler(policy)
                cold = replace(hot, residency=())
                a1 = self.submit(scheduler, model_a, cold, "a1", 1_000, 640)
                self.submit(scheduler, model_a, cold, "a2", 1_100, 8)
                self.assertTrue(a1.execution_plan.transitions)
                active = scheduler.wait_runtime_request(
                    "a1", time.monotonic_ns() - a1.decision.start_us * 1_000
                )
                loaded_at_us = a1.decision.start_us + 150
                scheduler.record_automated_transition_receipts("a1", tuple(
                    replace(row, finished_us=loaded_at_us)
                    for row in FakeAutomatedPhysicalAdapter._transition_receipts(active)
                ))
                view = self.queue_view(scheduler)
                if not replanned:
                    self.assertEqual(view["a2"]["state"], "DEFERRED_REPLAN")
                    continue
                self.assertEqual(view["a2"]["state"], "REPLAN_REQUIRED")
                wake = scheduler.wait_runtime_request(
                    "a2", time.monotonic_ns() - loaded_at_us * 1_000
                )
                snapshot = replace(
                    hot, snapshot_id="published", captured_at_us=loaded_at_us,
                    valid_until_us=loaded_at_us + 10_000_000,
                    memory=replace(hot.memory, snapshot_id="published-memory",
                                   captured_at_us=loaded_at_us,
                                   valid_until_us=loaded_at_us + 10_000_000),
                )
                replacement = scheduler.replan_automated_request(
                    "a2", observed_at_us=loaded_at_us,
                    reason=wake.dispatch_receipt.wake_reason, snapshot=snapshot,
                    expected_ticket_id=wake.ticket_id,
                    expected_queue_generation=wake.dispatch_receipt.queue_generation,
                )
                self.assertEqual(replacement.execution_plan.transitions, ())
                self.assertEqual(replacement.decision.start_us, loaded_at_us)
                self.assertTrue(self.ready_now(scheduler, "a2", loaded_at_us))
                self.assertEqual(
                    scheduler.runtime_dispatch_policy_state()["statistics"][
                        "published_work_promotions"
                    ],
                    1,
                )

    def affinity_scenario(self, **bounds):
        policy = RuntimeDispatchPolicy(
            work_conserving_admission=True, model_affinity=True, **bounds
        )
        scheduler, model_a, model_b, hot = self.scheduler(policy)
        tickets = {
            "a1": self.submit(scheduler, model_a, hot, "a1", 1_000, 64),
            "b1": self.submit(scheduler, model_b, hot, "b1", 1_100, 8),
            "a2": self.submit(scheduler, model_a, hot, "a2", 1_200, 4),
            "a3": self.submit(scheduler, model_a, hot, "a3", 1_300, 64),
            "a4": self.submit(scheduler, model_a, hot, "a4", 1_400, 64),
        }
        return scheduler, tickets

    def test_model_affinity_admits_the_resident_model_before_a_queued_switch(self) -> None:
        scheduler, tickets = self.affinity_scenario(affinity_maximum_bypasses=1)
        a2 = tickets["a2"]
        self.assertEqual(a2.execution_plan.transitions, ())
        self.assertEqual(a2.decision.start_us, 1_200)
        view = self.queue_view(scheduler)
        self.assertEqual(view["b1"]["state"], "REPLAN_REQUIRED")
        self.assertIn("a2", view["b1"]["predecessor_request_ids"])
        decision = next(
            row for row in scheduler.runtime_decision_log()["records"]
            if row["event_kind"] == "DECISION" and row["request_ids"] == ["a2"]
        )
        note = decision["selected"]["dispatch_policy"]
        self.assertEqual(note["kind"], "MODEL_AFFINITY_DISPLACEMENT")
        self.assertEqual(note["bypassed_request_ids"], ["b1"])
        self.assertEqual(note["displaced_request_ids"], ["b1"])
        self.assertEqual(note["bypass_counts"], {"b1": 1})
        self.assertLess(note["reserved_start_us"], note["reserved_start_without_displacement_us"])

    def test_other_model_is_protected_after_the_bypass_bound(self) -> None:
        scheduler, tickets = self.affinity_scenario(affinity_maximum_bypasses=1)
        view = self.queue_view(scheduler)
        self.assertIn("b1", view["a3"]["predecessor_request_ids"])
        self.assertIn("b1", view["a4"]["predecessor_request_ids"])
        state = scheduler.runtime_dispatch_policy_state()
        self.assertEqual(state["bypass_counts"], {"b1": 1})
        self.assertEqual(state["statistics"]["affinity_displacements"], 1)
        self.assertEqual(state["statistics"]["affinity_refusals"], 2)

    def test_other_model_is_protected_after_the_wait_bound(self) -> None:
        scheduler, tickets = self.affinity_scenario(
            affinity_maximum_bypasses=10, affinity_maximum_wait_us=150,
        )
        view = self.queue_view(scheduler)
        self.assertIn("a2", view["b1"]["predecessor_request_ids"])
        self.assertIn("b1", view["a3"]["predecessor_request_ids"])
        self.assertEqual(scheduler.runtime_dispatch_policy_state()["bypass_counts"], {"b1": 1})

    def test_displaced_switch_replans_after_the_resident_work(self) -> None:
        scheduler, tickets = self.affinity_scenario(affinity_maximum_bypasses=1)
        self.assertEqual(self.queue_view(scheduler)["b1"]["state"], "REPLAN_REQUIRED")
        wake = scheduler.wait_runtime_request("b1", time.monotonic_ns() - 1_400 * 1_000)
        self.assertEqual(wake.dispatch_state, "REPLAN_REQUIRED")
        self.assertEqual(wake.dispatch_receipt.wake_reason, "model_affinity_displaced")
        _, _, _, hot = self.scheduler()
        replacement = scheduler.replan_automated_request(
            "b1", observed_at_us=1_400, reason=wake.dispatch_receipt.wake_reason,
            snapshot=replace(hot, captured_at_us=1_400),
            expected_ticket_id=wake.ticket_id,
            expected_queue_generation=wake.dispatch_receipt.queue_generation,
        )
        self.assertTrue(replacement.execution_plan.transitions)
        resident_end = max(
            lease.reserved_until_us
            for request_id in ("a1", "a2")
            for lease in scheduler.runtime_ticket(request_id).decision.leases
            if lease.resource_id == "compute:accelerator-b"
        )
        self.assertGreaterEqual(replacement.decision.start_us, resident_end)
        self.assertIn("a2", self.queue_view(scheduler)["b1"]["predecessor_request_ids"])

    def test_affinity_keeps_the_server_slot_capacity(self) -> None:
        scheduler, tickets = self.affinity_scenario(affinity_maximum_bypasses=1)
        windows = [
            (lease.start_us, lease.reserved_until_us)
            for ticket in (scheduler.runtime_ticket(name) for name in ("a1", "a2", "a3", "a4"))
            for lease in ticket.decision.leases
            if lease.resource_id == "compute:accelerator-b"
        ]
        for instant in sorted({start for start, _ in windows}):
            concurrent = sum(start <= instant < end for start, end in windows)
            self.assertLessEqual(concurrent, 2)

    def test_affinity_decisions_are_deterministic(self) -> None:
        logs = []
        for _ in range(2):
            scheduler, _ = self.affinity_scenario(affinity_maximum_bypasses=1)
            logs.append((
                scheduler.runtime_decision_log()["records"],
                dict(scheduler.runtime_dispatch_policy_state()),
                scheduler.runtime_controller_snapshot()["dispatch_queue"]["causal_predecessors"],
            ))
        self.assertEqual(logs[0], logs[1])

    @staticmethod
    def published(hot, at_us: int):
        return replace(
            hot, snapshot_id=f"published-{at_us}", captured_at_us=at_us,
            valid_until_us=at_us + 10_000_000,
            memory=replace(hot.memory, snapshot_id=f"published-memory-{at_us}",
                           captured_at_us=at_us,
                           valid_until_us=at_us + 10_000_000),
        )

    def arrival_during_load(self, policy):
        """a1 loads model A; b1 (model B) and a2 (model A) arrive meanwhile."""
        scheduler, model_a, model_b, hot = self.scheduler(policy)
        cold = replace(hot, residency=())
        a1 = self.submit(scheduler, model_a, cold, "a1", 1_000, 640)
        self.submit(scheduler, model_b, cold, "b1", 1_100, 8)
        a2 = self.submit(scheduler, model_a, cold, "a2", 1_200, 8)
        self.assertTrue(a2.execution_plan.transitions)
        self.assertIn("b1", self.queue_view(scheduler)["a2"]["predecessor_request_ids"])
        active = scheduler.wait_runtime_request(
            "a1", time.monotonic_ns() - a1.decision.start_us * 1_000
        )
        loaded_at_us = 1_300
        scheduler.record_automated_transition_receipts("a1", tuple(
            replace(row, finished_us=loaded_at_us)
            for row in FakeAutomatedPhysicalAdapter._transition_receipts(active)
        ))
        return scheduler, hot, loaded_at_us

    def replan_woken(self, scheduler, request_id, at_us, snapshot):
        self.assertEqual(
            self.queue_view(scheduler)[request_id]["state"], "REPLAN_REQUIRED"
        )
        wake = scheduler.wait_runtime_request(
            request_id, time.monotonic_ns() - at_us * 1_000
        )
        replacement = scheduler.replan_automated_request(
            request_id, observed_at_us=at_us,
            reason=wake.dispatch_receipt.wake_reason, snapshot=snapshot,
            expected_ticket_id=wake.ticket_id,
            expected_queue_generation=wake.dispatch_receipt.queue_generation,
        )
        return replacement, wake.dispatch_receipt.wake_reason

    def replan_at_publication(self, policy, woken=("b1", "a2")):
        scheduler, hot, loaded_at_us = self.arrival_during_load(policy)
        snapshot = self.published(hot, loaded_at_us)
        self.assertEqual(
            scheduler.observe_automated_runtime_snapshot(
                snapshot, observed_at_us=loaded_at_us
            ),
            woken,
        )
        for request_id in woken:
            replacement, _ = self.replan_woken(
                scheduler, request_id, loaded_at_us, snapshot
            )
        return scheduler, replacement, snapshot, loaded_at_us

    def assert_displaced_by_the_replan(self, scheduler, a2, loaded_at_us) -> None:
        self.assertEqual(a2.execution_plan.transitions, ())
        self.assertEqual(a2.decision.start_us, loaded_at_us)
        self.assertTrue(self.ready_now(scheduler, "a2", loaded_at_us))
        view = self.queue_view(scheduler)
        self.assertEqual(view["b1"]["state"], "REPLAN_REQUIRED")
        self.assertIn("a2", view["b1"]["predecessor_request_ids"])
        record = [
            row for row in scheduler.runtime_decision_log()["records"]
            if row["event_kind"] == "REPLAN" and row["request_ids"] == ["a2"]
        ][-1]
        note = record["selected"]["dispatch_policy"]
        self.assertEqual(note["kind"], "MODEL_AFFINITY_DISPLACEMENT")
        self.assertEqual(note["replan_reason"], "residency_observation_changed")
        self.assertEqual(note["displaced_request_ids"], ["b1"])
        self.assertEqual(note["bypass_counts"], {"b1": 1})
        statistics = scheduler.runtime_dispatch_policy_state()["statistics"]
        self.assertEqual(statistics["affinity_displacements"], 1)
        self.assertEqual(statistics["affinity_refusals"], 0)

    def test_publication_replans_the_resident_model_ahead_of_the_queued_switch(self) -> None:
        scheduler, a2, _, loaded_at_us = self.replan_at_publication(
            RuntimeDispatchPolicy(work_conserving_admission=True, model_affinity=True),
            woken=("a2",),
        )
        self.assert_displaced_by_the_replan(scheduler, a2, loaded_at_us)
        self.assertEqual(
            scheduler.runtime_dispatch_policy_state()["statistics"][
                "publication_replans"
            ],
            1,
        )

    def test_any_replan_of_the_resident_model_displaces_the_queued_switch(self) -> None:
        # Woken by the obsolete-transition rule instead of the affinity wake.
        with mock.patch.object(
            observation_ops, "model_affinity_replan_displacement",
            return_value=None, create=True,
        ):
            scheduler, a2, _, loaded_at_us = self.replan_at_publication(
                RuntimeDispatchPolicy(
                    work_conserving_admission=True, model_affinity=True
                )
            )
        self.assert_displaced_by_the_replan(scheduler, a2, loaded_at_us)

    def test_displaced_switch_replans_behind_without_deferring_the_resident_work(self) -> None:
        scheduler, a2, snapshot, loaded_at_us = self.replan_at_publication(
            RuntimeDispatchPolicy(work_conserving_admission=True, model_affinity=True),
            woken=("a2",),
        )
        self.assertEqual(a2.execution_plan.transitions, ())
        b1, reason = self.replan_woken(scheduler, "b1", loaded_at_us, snapshot)
        self.assertEqual(reason, "model_affinity_displaced")
        self.assertTrue(b1.execution_plan.transitions)
        resident_end = max(
            lease.reserved_until_us
            for request_id in ("a1", "a2")
            for lease in scheduler.runtime_ticket(request_id).decision.leases
            if lease.resource_id == "compute:accelerator-b"
        )
        self.assertGreaterEqual(b1.decision.start_us, resident_end)
        self.assertEqual(self.queue_view(scheduler)["a2"]["state"], "QUEUED")
        self.assertTrue(self.ready_now(scheduler, "a2", loaded_at_us))

    def test_replan_keeps_the_switch_first_without_affinity_or_past_the_bound(self) -> None:
        for policy, refusals in (
            (WORK_CONSERVING, 0),
            (RuntimeDispatchPolicy(
                work_conserving_admission=True, model_affinity=True,
                affinity_maximum_bypasses=0,
            ), 1),
        ):
            with self.subTest(policy=policy):
                scheduler, a2, _, loaded_at_us = self.replan_at_publication(policy)
                self.assertTrue(a2.execution_plan.transitions)
                self.assertIn(
                    "b1", self.queue_view(scheduler)["a2"]["predecessor_request_ids"]
                )
                self.assertFalse(self.ready_now(scheduler, "a2", loaded_at_us))
                state = scheduler.runtime_dispatch_policy_state()
                self.assertEqual(state["bypass_counts"], {})
                self.assertEqual(state["statistics"]["affinity_refusals"], refusals)

    def test_publication_wakes_the_resident_model_waiting_behind_a_switch(self) -> None:
        policy = RuntimeDispatchPolicy(
            work_conserving_admission=True, model_affinity=True
        )
        # Replanned while its executor was not ready: queued behind the switch.
        with mock.patch.object(affinity_ops, "_model_is_resident", return_value=False):
            scheduler, a2, snapshot, loaded_at_us = self.replan_at_publication(policy)
        self.assertTrue(a2.execution_plan.transitions)
        self.assertEqual(self.queue_view(scheduler)["b1"]["state"], "QUEUED")
        later = self.published(snapshot, loaded_at_us + 1)
        self.assertEqual(
            scheduler.observe_automated_runtime_snapshot(
                later, observed_at_us=loaded_at_us + 1
            ),
            ("a2",),
        )
        a2, reason = self.replan_woken(scheduler, "a2", loaded_at_us + 1, later)
        self.assertEqual(reason, "residency_observation_changed")
        self.assertEqual(a2.execution_plan.transitions, ())
        self.assertTrue(self.ready_now(scheduler, "a2", loaded_at_us + 1))
        self.assertIn("a2", self.queue_view(scheduler)["b1"]["predecessor_request_ids"])

    def test_replan_affinity_is_deterministic(self) -> None:
        policy = RuntimeDispatchPolicy(
            work_conserving_admission=True, model_affinity=True
        )
        logs = []
        for _ in range(2):
            scheduler, _, _, _ = self.replan_at_publication(policy, woken=("a2",))
            logs.append((
                scheduler.runtime_decision_log()["records"],
                dict(scheduler.runtime_dispatch_policy_state()),
                scheduler.runtime_controller_snapshot()["dispatch_queue"]["causal_predecessors"],
            ))
        self.assertEqual(logs[0], logs[1])

    def switch_replanned_at_the_release(self, policy):
        """cj5 F-B: a1 (model A) decodes alone, b1 (model B) switches after it; a1 finishes at t.

        The switch is replanned at t (capacity_released_early) and is due at once."""
        scheduler, model_a, model_b, hot = self.scheduler(policy)
        a1 = self.submit(scheduler, model_a, hot, "a1", 1_000, 64)
        b1 = self.submit(scheduler, model_b, hot, "b1", 1_100, 64)
        self.assertTrue(b1.execution_plan.transitions)
        active = scheduler.wait_runtime_request(
            "a1", time.monotonic_ns() - a1.decision.start_us * 1_000
        )
        released_at_us = a1.decision.finish_us
        scheduler.complete_automated_request(
            "a1", AutomatedRuntimeTests.execution_receipt(active, finished_us=released_at_us)
        )
        switch, _ = self.replan_woken(scheduler, "b1", released_at_us, self.published(hot, released_at_us))
        self.assertEqual(switch.decision.start_us, released_at_us)
        self.assertTrue(switch.execution_plan.transitions)
        return scheduler, model_a, hot, released_at_us

    def test_without_hysteresis_the_switch_starts_and_the_arrival_reloads_behind_it(self) -> None:
        scheduler, model_a, hot, released_at_us = self.switch_replanned_at_the_release(AFFINITY)
        queue = scheduler._runtime_controller.queue
        self.assertIsNone(queue.residency_hysteresis_until("b1"))
        started = scheduler.wait_runtime_request(
            "b1", time.monotonic_ns() - (released_at_us + 1) * 1_000
        )
        self.assertEqual(started.dispatch_state, "ACQUIRED")
        arrival_us = released_at_us + 100
        a2 = self.submit(scheduler, model_a, self.published(hot, arrival_us), "a2", arrival_us, 64)
        # Two units of timing decide a reload: the switch runs, a2 reloads model A behind it.
        self.assertTrue(a2.execution_plan.transitions)
        self.assertGreaterEqual(
            a2.decision.start_us,
            max(row.reserved_until_us for row in started.decision.leases),
        )
        statistics = scheduler.runtime_dispatch_policy_state()["statistics"]
        self.assertEqual(statistics["affinity_displacements"], 0)
        self.assertNotIn("residency_hysteresis_holds", statistics)

    def test_hysteresis_holds_the_switch_so_the_arrival_is_admitted_first(self) -> None:
        scheduler, model_a, hot, released_at_us = self.switch_replanned_at_the_release(SPECULATIVE)
        controller = scheduler._runtime_controller
        held_until_us = released_at_us + HYSTERESIS_US
        self.assertEqual(controller.queue.residency_hysteresis_until("b1"), held_until_us)
        switch_ticket_id = scheduler.runtime_ticket("b1").ticket_id
        outcomes = []
        waiter = threading.Thread(target=lambda: outcomes.append(
            scheduler.wait_runtime_request(
                "b1", time.monotonic_ns() - (released_at_us + 1) * 1_000
            )
        ))
        waiter.start()
        wait_for(lambda: controller.queue.residency_hysteresis_hold("b1") is not None)
        self.assertEqual(scheduler.runtime_ticket("b1").dispatch_state, "QUEUED")
        arrival_us = released_at_us + 100
        a2 = self.submit(scheduler, model_a, self.published(hot, arrival_us), "a2", arrival_us, 64)
        self.assertEqual(a2.execution_plan.transitions, ())
        self.assertEqual(a2.decision.start_us, arrival_us)
        self.assertTrue(self.ready_now(scheduler, "a2", arrival_us))
        note = next(
            row for row in scheduler.runtime_decision_log()["records"]
            if row["event_kind"] == "DECISION" and row["request_ids"] == ["a2"]
        )["selected"]["dispatch_policy"]
        self.assertEqual(note["kind"], "MODEL_AFFINITY_DISPLACEMENT")
        self.assertEqual(note["displaced_request_ids"], ["b1"])
        waiter.join(timeout=5)
        self.assertFalse(waiter.is_alive())
        self.assertEqual(outcomes[0].dispatch_state, "REPLAN_REQUIRED")
        view = self.queue_view(scheduler)
        self.assertEqual(view["b1"]["state"], "REPLAN_REQUIRED")
        self.assertIn("a2", view["b1"]["predecessor_request_ids"])
        held = controller.dispatch_policy_note(switch_ticket_id)
        self.assertEqual(held["kind"], HELD_KIND)
        self.assertEqual(held["barrier_request_id"], "b1")
        self.assertEqual(held["held_until_us"], held_until_us)
        self.assertEqual(held["released_at_us"], released_at_us)
        state = scheduler.runtime_dispatch_policy_state()
        self.assertEqual(state["policy"]["residency_hysteresis_s"], HYSTERESIS_S)
        self.assertEqual(state["statistics"]["residency_hysteresis_holds"], 1)
        self.assertEqual(state["statistics"]["residency_hysteresis_admissions"], 1)
        self.assertEqual(state["statistics"]["affinity_displacements"], 1)
        # The displaced switch replans behind a2 and is held at most H beyond that start.
        _, _, _, snapshot = self.scheduler()
        switch, reason = self.replan_woken(
            scheduler, "b1", arrival_us, self.published(snapshot, arrival_us)
        )
        self.assertEqual(reason, "model_affinity_displaced")
        self.assertGreaterEqual(
            switch.decision.start_us,
            max(row.reserved_until_us for row in a2.decision.leases
                if row.resource_id == "compute:accelerator-b"),
        )
        self.assertEqual(controller.queue.residency_hysteresis_until("b1"), held_until_us)

    def test_s1a_switch_without_same_model_demand_starts_at_the_release(self) -> None:
        # s1a: five switches were held 20 s each and nothing arrived; now the hold is skipped.
        scheduler, _, _, released_at_us = self.switch_replanned_at_the_release(HYSTERESIS)
        controller = scheduler._runtime_controller
        self.assertIsNone(controller.queue.residency_hysteresis_until("b1"))
        switch_ticket_id = scheduler.runtime_ticket("b1").ticket_id
        started = scheduler.wait_runtime_request(
            "b1", time.monotonic_ns() - (released_at_us + 1) * 1_000
        )
        self.assertEqual(started.dispatch_state, "ACQUIRED")
        self.assertNotEqual(started.dispatch_receipt.wake_reason, "residency_hysteresis_released")
        note = controller.dispatch_policy_note(switch_ticket_id)
        self.assertEqual(note["kind"], SKIPPED_KIND)
        self.assertEqual(note["reason"], "ARRIVAL_HISTORY_INSUFFICIENT")
        self.assertEqual(note["released_at_us"], released_at_us)
        state = scheduler.runtime_dispatch_policy_state()
        self.assertEqual(state["statistics"]["residency_hysteresis_skips"], 1)
        self.assertEqual(state["statistics"]["residency_hysteresis_holds"], 0)
        self.assertEqual(
            [row["reason"] for row in state["residency_hysteresis_decisions"]],
            ["ARRIVAL_HISTORY_INSUFFICIENT"],
        )
        # Without the window the state carries neither the statistic nor the decisions.
        plain, _, _, _ = self.switch_replanned_at_the_release(AFFINITY)
        plain_state = plain.runtime_dispatch_policy_state()
        self.assertNotIn("residency_hysteresis_decisions", plain_state)
        self.assertNotIn("residency_hysteresis_skips", plain_state["statistics"])

    def test_early_capacity_replan_defers_the_displaced_switch_in_its_window(self) -> None:
        scheduler, model_a, model_b, hot = self.scheduler(RuntimeDispatchPolicy(
            work_conserving_admission=True, model_affinity=True,
        ))
        for model, request_id, arrival_us in (
            (model_a, "a1", 1_000), (model_b, "b1", 1_100), (model_a, "a2", 1_200),
        ):
            scheduler.submit_automated_request(
                request(request_id, arrival_us=arrival_us, output_tokens=64 if model is model_a else 8),
                model.model_id, hot, selection_mode="energy-aware",
            )
        view = self.queue_view(scheduler)
        self.assertIn("a2", view["b1"]["predecessor_request_ids"])
        self.assertLess(view["b1"]["sequence"], view["a2"]["sequence"])
        controller = scheduler._runtime_controller
        self.assertTrue(controller.require_queued_replan("a2", "capacity_released_early"))
        b1, _ = self.replan_woken(scheduler, "b1", 1_250, hot)
        a2_end = max(row.reserved_until_us for row in scheduler.runtime_ticket("a2").decision.leases)
        self.assertEqual(b1.decision.start_us, a2_end)

        # Replanned later, a2's slot runs into the switch it displaced; the
        # switch is its dependent and not a compaction follower.
        with mock.patch.object(time, "monotonic_ns", return_value=1_260_000):
            wake = scheduler.wait_runtime_request("a2", 0)
        self.assertEqual(wake.dispatch_receipt.wake_reason, "capacity_released_early")
        self.assertEqual(
            controller.priority_compaction_followers("a2", wake.dispatch_receipt.queue_generation), (),
        )
        a2 = scheduler.replan_automated_request(
            "a2", observed_at_us=1_260, reason="capacity_released_early",
            snapshot=replace(hot, captured_at_us=1_260), expected_ticket_id=wake.ticket_id,
            expected_queue_generation=wake.dispatch_receipt.queue_generation,
        )

        self.assertEqual(a2.execution_plan.transitions, ())
        self.assertEqual(a2.decision.start_us, 1_260)
        view = self.queue_view(scheduler)
        self.assertEqual(view["b1"]["state"], "DEFERRED_REPLAN")
        self.assertIn("a2", view["b1"]["predecessor_request_ids"])
        self.assertEqual(scheduler.runtime_ticket("b1").lease_status, "CANCELLED")
        self.assertEqual(scheduler._runtime_memory.owner_tokens("b1"), ())


class ContinuousJoinSchedulerTests(unittest.TestCase):
    """Model A decodes on a multi-slot GPU server; the phone lanes it uses have capacity one.

    The phone coordinates no route, so every assisted route is served by the GPU
    server (one endpoint); the GPU coordinator declares ``parallel`` slots and a
    two-deep USB batch, so the decode cohort of an assisted plan can hold two rows.
    """

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.paths = (
            Path(self.directory.name) / "model-a.gguf",
            Path(self.directory.name) / "model-b.gguf",
        )
        write_synthetic_gguf(self.paths[0], block_count=2, sliding_window=32)
        write_synthetic_gguf(self.paths[1])

    def scheduler(self, policy: RuntimeDispatchPolicy | None, parallel: int,
                  phones: tuple[str, ...] = ("helper-c",), exclusive_phones: bool = False):
        """``phones`` beyond helper-c are cold; ``exclusive_phones`` declares each phone's
        compute an exclusive residency resource, as the rig does for the HTP."""
        source = catalog(
            phone_ops_per_s=6_000_000_000, phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000, phone_whole_model=False, extra_phones=phones[1:],
        )
        gpu = replace(
            source.executor_by_device["accelerator-b"],
            qualified_fallback=True,
            exclusive_residency_resource_id="compute:accelerator-b",
            adapter_parameters={
                "parallel": parallel, "usb_concurrent_streams": 2, "usb_queue_depth": 2,
            },
        )
        phone_rows = tuple(
            replace(
                source.executor_by_device[device_id],
                coordinated_route_families=(), operator_plan_protocol=None,
                exclusive_residency_resource_id=(
                    "compute:" + device_id if exclusive_phones else None
                ),
            )
            for device_id in phones
        )
        transition = replace(
            next(row for row in source.transitions if row.device_id == "accelerator-b"),
            executor_id=gpu.executor_id,
            prepares_device_ids=("accelerator-b",),
            resource_slots={"compute:accelerator-b": parallel},
        )
        resources = dict(source.resources)
        for resource_id in ("compute:accelerator-b", "link:pcie-in", "link:pcie-out"):
            resources[resource_id] = replace(resources[resource_id], capacity=parallel)
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        if policy is not None:
            scheduler.configure_runtime_dispatch_policy(policy)
        scheduler.register_runtime_capabilities(RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=(gpu, *phone_rows),
            composite_executors=(),
            transitions=(transition, *(
                row for row in source.transitions if row.device_id in phones
            )),
            resources=resources,
        ).to_json()))
        model_a = scheduler.register_gguf_model("model-a", self.paths[0])
        model_b = scheduler.register_gguf_model("model-b", self.paths[1])
        hot = runtime_snapshot(
            model_a, phone_bandwidth=8_000_000_000,
            resident_devices=("accelerator-b", "helper-c"), gpu_free_slots=parallel,
            extra_phones=phones[1:],
        )
        hot = replace(
            hot,
            executors={
                **hot.executors,
                gpu.executor_id: executor_state(gpu.executor_id, free_slots=parallel),
                "executor:helper-c": executor_state("executor:helper-c"),
            },
            residency=tuple(
                replace(row, executor_id=(
                    gpu.executor_id if row.device_id == "accelerator-b"
                    else "executor:helper-c"
                ))
                for row in hot.residency
            ),
        )
        return scheduler, model_a, model_b, hot

    @staticmethod
    def free_slots(snapshot, count: int):
        return replace(snapshot, executors={
            **snapshot.executors,
            "executor:accelerator-b": executor_state("executor:accelerator-b", free_slots=count),
        })

    @staticmethod
    def submit(scheduler, model, snapshot, request_id, arrival_us, output_tokens,
               mode="energy-aware"):
        return scheduler.submit_automated_request(
            request(request_id, arrival_us=arrival_us, output_tokens=output_tokens),
            model.model_id, snapshot, selection_mode=mode,
        )

    @staticmethod
    def acquire(scheduler, request_id: str, arrival_us: int):
        return scheduler.wait_runtime_request(
            request_id, time.monotonic_ns() - arrival_us * 1_000
        )

    @staticmethod
    def queue_view(scheduler):
        return scheduler._runtime_controller.dispatch_order_view()

    @staticmethod
    def ready_now(scheduler, request_id: str, at_us: int) -> bool:
        return dispatchable(scheduler._runtime_controller.queue, request_id, at_us)

    @staticmethod
    def decision_record(scheduler, request_id: str):
        return [
            row for row in scheduler.runtime_decision_log()["records"]
            if row["event_kind"] == "DECISION" and row["request_ids"] == [request_id]
        ][-1]

    @staticmethod
    def lease_end_us(ticket, resource_id: str) -> int:
        return max(
            row.reserved_until_us for row in ticket.decision.leases
            if row.resource_id == resource_id
        )

    def assisted_holder_and_arrival(self, policy):
        """a1 (assisted, holding the phone lanes) decodes; a2 of the same model finds a free slot."""
        scheduler, model_a, _, hot = self.scheduler(policy, parallel=2)
        a1 = self.submit(scheduler, model_a, hot, "a1", 1_000, 64)
        self.assertIn("compute:helper-c", {row.resource_id for row in a1.decision.leases})
        self.assertEqual(self.acquire(scheduler, "a1", 1_000).dispatch_state, "ACQUIRED")
        a2 = self.submit(scheduler, model_a, self.free_slots(hot, 1), "a2", 1_500, 8)
        return scheduler, model_a, hot, a1, a2

    def test_policy_off_reserves_the_arrival_behind_the_phone_lane_holder(self) -> None:
        scheduler, _, _, a1, a2 = self.assisted_holder_and_arrival(WORK_CONSERVING)
        self.assertNotEqual(a2.decision.reason, JOIN_REASON)
        self.assertIn("compute:helper-c", {row.resource_id for row in a2.decision.leases})
        self.assertEqual(a2.decision.start_us, self.lease_end_us(a1, "compute:helper-c"))
        self.assertIn("a1", self.queue_view(scheduler)["a2"]["predecessor_request_ids"])
        self.assertNotIn(PHONE_LANES_HELD, dict(a2.decision.rejected).values())
        self.assertFalse(self.ready_now(scheduler, "a2", 1_500))
        statistics = scheduler.runtime_dispatch_policy_state()["statistics"]
        self.assertNotIn("continuous_join_bypasses", statistics)
        self.assertNotIn("continuous_join_refusals", statistics)

    def test_same_model_arrival_joins_as_the_desktop_parent(self) -> None:
        scheduler, _, _, a1, a2 = self.assisted_holder_and_arrival(CONTINUOUS_JOIN)
        self.assertEqual(a2.decision.reason, JOIN_REASON)
        self.assertEqual(a2.binding.endpoint, a1.binding.endpoint)
        self.assertEqual(a2.execution_plan.execution_contract.execution_mode, "desktop")
        self.assertEqual(a2.execution_plan.transitions, ())
        self.assertEqual(
            {row.resource_id for row in a2.decision.leases}, {"compute:accelerator-b"}
        )
        self.assertEqual(a2.decision.start_us, 1_500)
        self.assertEqual(self.queue_view(scheduler)["a2"]["predecessor_request_ids"], ())
        rejected = dict(a2.decision.rejected)
        self.assertEqual(rejected[a1.decision.route_id], PHONE_LANES_HELD)
        self.assertTrue(all(
            "helper-c" in route_id
            for route_id, reason in rejected.items() if reason == PHONE_LANES_HELD
        ))
        self.assertTrue(self.ready_now(scheduler, "a2", 1_500))
        self.assertEqual(self.acquire(scheduler, "a2", 1_500).dispatch_state, "ACQUIRED")
        lanes = [
            row.lanes for ticket in (a1, a2) for row in ticket.decision.leases
            if row.resource_id == "compute:accelerator-b"
        ]
        self.assertEqual(sorted(lane for lanes_ in lanes for lane in lanes_), [0, 1])

    def test_join_stops_at_the_server_slot_bound(self) -> None:
        scheduler, model_a, hot, a1, a2 = self.assisted_holder_and_arrival(CONTINUOUS_JOIN)
        self.assertEqual(self.acquire(scheduler, "a2", 1_500).dispatch_state, "ACQUIRED")
        # Two of two slots are ACQUIRED: a stale free-slot sample cannot admit a third co-tenant.
        a3 = self.submit(scheduler, model_a, self.free_slots(hot, 1), "a3", 1_600, 8)
        self.assertNotEqual(a3.decision.reason, JOIN_REASON)
        self.assertNotIn(PHONE_LANES_HELD, dict(a3.decision.rejected).values())
        self.assertGreater(a3.decision.start_us, 1_600)
        self.assertFalse(self.ready_now(scheduler, "a3", 1_600))

    def test_join_decisions_are_deterministic(self) -> None:
        # The holder's dispatch receipt carries wall-clock time; compare the decision itself.
        outcomes = []
        for policy in (CONTINUOUS_JOIN, CONTINUOUS_JOIN, WORK_CONSERVING, WORK_CONSERVING):
            scheduler, _, _, _, a2 = self.assisted_holder_and_arrival(policy)
            outcomes.append((
                a2.decision.route_id, a2.decision.reason, a2.decision.start_us,
                a2.decision.rejected,
                scheduler.runtime_controller_snapshot()["dispatch_queue"]["causal_predecessors"],
                dict(scheduler.runtime_dispatch_policy_state()),
            ))
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[2], outcomes[3])
        self.assertNotEqual(outcomes[0][:2], outcomes[2][:2])

    def test_joiner_is_batch_composition_not_external_activity(self) -> None:
        for policy, external in ((WORK_CONSERVING, True), (CONTINUOUS_JOIN, False)):
            with self.subTest(policy=policy):
                scheduler, model_a, _, hot = self.scheduler(policy, parallel=2)
                a1 = self.submit(scheduler, model_a, hot, "a1", 1_000, 64, mode="desktop-baseline")
                self.assertEqual(self.acquire(scheduler, "a1", 1_000).dispatch_state, "ACQUIRED")
                before = scheduler.runtime_external_desktop_activity(a1.ticket_id, 100_000, 1_000_000)
                a2 = self.submit(scheduler, model_a, self.free_slots(hot, 1), "a2", 1_500, 8,
                                 mode="desktop-baseline")
                self.assertEqual(self.acquire(scheduler, "a2", 1_500).dispatch_state, "ACQUIRED")
                after = scheduler.runtime_external_desktop_activity(a1.ticket_id, 100_000, 1_000_000)
                self.assertEqual(before["ticket_ids"], ())
                self.assertNotIn("batch_member_ticket_ids", before)
                if external:
                    self.assertEqual(after["ticket_ids"], (a2.ticket_id,))
                    self.assertNotEqual(after["sha256"], before["sha256"])
                    self.assertNotIn("batch_member_ticket_ids", after)
                else:
                    self.assertEqual(after["ticket_ids"], ())
                    self.assertEqual(after["sha256"], before["sha256"])
                    self.assertEqual(after["batch_member_ticket_ids"], (a2.ticket_id,))

    def joiner_behind_a_queued_switch(self, policy):
        """a1, a2 decode model A; b1 (model B) queues a switch after them; a3 (model A) outlasts them."""
        scheduler, model_a, model_b, hot = self.scheduler(policy, parallel=4)
        a1 = self.submit(scheduler, model_a, hot, "a1", 1_000, 64, mode="desktop-baseline")
        a2 = self.submit(scheduler, model_a, self.free_slots(hot, 3), "a2", 1_100, 64,
                         mode="desktop-baseline")
        for request_id, arrival_us in (("a1", 1_000), ("a2", 1_100)):
            self.assertEqual(self.acquire(scheduler, request_id, arrival_us).dispatch_state, "ACQUIRED")
        b1 = self.submit(scheduler, model_b, self.free_slots(hot, 2), "b1", 1_200, 8,
                         mode="desktop-baseline")
        self.assertTrue(b1.execution_plan.transitions)
        # The committed busy window: the later of each holder's prediction and lease horizon.
        committed_end_us = max(
            max(ticket.decision.finish_upper_us, self.lease_end_us(ticket, "compute:accelerator-b"))
            for ticket in (a1, a2)
        )
        self.assertGreaterEqual(
            b1.decision.start_us,
            max(self.lease_end_us(ticket, "compute:accelerator-b") for ticket in (a1, a2)),
        )
        a3 = self.submit(scheduler, model_a, self.free_slots(hot, 2), "a3", 1_300, 64,
                         mode="desktop-baseline")
        return scheduler, b1, a3, committed_end_us

    def test_policy_off_keeps_the_outlasting_joiner_behind_the_switch(self) -> None:
        scheduler, b1, a3, _ = self.joiner_behind_a_queued_switch(WORK_CONSERVING)
        view = self.queue_view(scheduler)
        self.assertIn("b1", view["a3"]["predecessor_request_ids"])
        self.assertEqual(view["b1"]["state"], "QUEUED")
        self.assertGreaterEqual(
            a3.decision.start_us, self.lease_end_us(b1, "compute:accelerator-b")
        )
        # Reserved after the switch, the joiner reloads its own model.
        self.assertTrue(a3.execution_plan.transitions)
        self.assertFalse(self.ready_now(scheduler, "a3", 1_300))
        self.assertNotIn("dispatch_policy", self.decision_record(scheduler, "a3")["selected"])

    def test_bypass_within_the_extension_bound_precedes_the_switch(self) -> None:
        scheduler, b1, a3, committed_end_us = self.joiner_behind_a_queued_switch(
            replace(CONTINUOUS_JOIN, max_barrier_extension_s=1)
        )
        view = self.queue_view(scheduler)
        self.assertEqual(a3.decision.start_us, 1_300)
        self.assertEqual(a3.execution_plan.transitions, ())
        self.assertEqual(view["a3"]["predecessor_request_ids"], ())
        self.assertTrue(self.ready_now(scheduler, "a3", 1_300))
        self.assertEqual(view["b1"]["state"], "REPLAN_REQUIRED")
        self.assertIn("a3", view["b1"]["predecessor_request_ids"])
        self.assertEqual(scheduler.runtime_ticket("b1").lease_status, "CANCELLED")
        note = self.decision_record(scheduler, "a3")["selected"]["dispatch_policy"]
        self.assertEqual(note["kind"], BYPASS_KIND)
        self.assertEqual(note["barrier_request_id"], "b1")
        self.assertEqual(note["displaced_request_ids"], ["b1"])
        self.assertEqual(note["bypassed_request_ids"], ["b1"])
        self.assertEqual(note["committed_end_us"], committed_end_us)
        self.assertGreater(note["extension_us"], 0)
        self.assertLessEqual(note["extension_us"], note["max_barrier_extension_us"])
        self.assertEqual(note["max_barrier_extension_us"], 1_000_000)
        self.assertEqual(note["finish_upper_us"], committed_end_us + note["extension_us"])
        self.assertEqual(note["reserved_start_us"], 1_300)
        self.assertGreater(note["reserved_start_without_bypass_us"], 1_300)
        statistics = scheduler.runtime_dispatch_policy_state()["statistics"]
        self.assertEqual(statistics["continuous_join_bypasses"], 1)
        self.assertEqual(statistics["continuous_join_refusals"], 0)
        self.assertEqual(self.acquire(scheduler, "a3", 1_300).dispatch_state, "ACQUIRED")
        self.assertEqual(scheduler._runtime_controller.queue.active_request_count(), 3)
        wake = scheduler.wait_runtime_request("b1", time.monotonic_ns() - 1_300 * 1_000)
        self.assertEqual(wake.dispatch_state, "REPLAN_REQUIRED")
        self.assertEqual(wake.dispatch_receipt.wake_reason, "continuous_join_displaced")

    def test_zero_extension_never_extends_the_committed_window(self) -> None:
        scheduler, b1, a3, committed_end_us = self.joiner_behind_a_queued_switch(CONTINUOUS_JOIN)
        view = self.queue_view(scheduler)
        self.assertIn("b1", view["a3"]["predecessor_request_ids"])
        self.assertEqual(view["b1"]["state"], "QUEUED")
        self.assertEqual(scheduler.runtime_ticket("b1").lease_status, "RESERVED")
        self.assertGreaterEqual(
            a3.decision.start_us, self.lease_end_us(b1, "compute:accelerator-b")
        )
        self.assertGreater(a3.decision.finish_upper_us, committed_end_us)
        self.assertNotIn("dispatch_policy", self.decision_record(scheduler, "a3")["selected"])
        statistics = scheduler.runtime_dispatch_policy_state()["statistics"]
        self.assertEqual(statistics["continuous_join_bypasses"], 0)
        self.assertEqual(statistics["continuous_join_refusals"], 1)

    def test_refused_bypass_records_its_reason(self) -> None:
        scheduler, _b1, _a3, _ = self.joiner_behind_a_queued_switch(CONTINUOUS_JOIN)
        refusals = scheduler.runtime_dispatch_policy_state()["refusals"]
        self.assertEqual(len(refusals), 1)
        self.assertEqual(refusals[0]["kind"], "CONTINUOUS_JOIN_REFUSED")
        self.assertEqual(refusals[0]["request_id"], "a3")
        self.assertEqual(refusals[0]["reason"], "joiner would extend the committed busy window")
        self.assertIs(type(refusals[0]["observed_at_us"]), int)

    def test_policy_off_reports_no_refusals(self) -> None:
        scheduler, *_ = self.joiner_behind_a_queued_switch(WORK_CONSERVING)
        self.assertNotIn("refusals", scheduler.runtime_dispatch_policy_state())

    def test_join_bypass_runs_after_an_affinity_refusal(self) -> None:
        # Model affinity finds the queued switch but refuses its displacement (rolled back);
        # the bounded barrier bypass must still judge the arrival by its own rule.
        refused_for = []

        def refused(controller, **kwargs):
            refused_for.append(kwargs["request"].request_id)
            controller._runtime_controller.record_dispatch_policy_event("affinity_refusals")
            return None

        with mock.patch.object(selection_ops, "_submit_with_model_affinity", refused):
            scheduler, _b1, a3, _ = self.joiner_behind_a_queued_switch(
                replace(CONTINUOUS_JOIN, model_affinity=True, max_barrier_extension_s=1)
            )
        self.assertIn("a3", refused_for)
        self.assertEqual(a3.decision.start_us, 1_300)
        note = self.decision_record(scheduler, "a3")["selected"]["dispatch_policy"]
        self.assertEqual(note["kind"], BYPASS_KIND)
        statistics = scheduler.runtime_dispatch_policy_state()["statistics"]
        self.assertEqual(statistics["continuous_join_bypasses"], 1)
        self.assertGreaterEqual(statistics["affinity_refusals"], 1)

    def test_bypass_decisions_are_deterministic(self) -> None:
        # The holders' dispatch receipts carry wall-clock time; compare the bypass itself.
        outcomes = []
        for _ in range(2):
            scheduler, _, a3, _ = self.joiner_behind_a_queued_switch(
                replace(CONTINUOUS_JOIN, max_barrier_extension_s=1)
            )
            outcomes.append((
                a3.decision.route_id, a3.decision.start_us, a3.decision.finish_upper_us,
                self.decision_record(scheduler, "a3")["selected"]["dispatch_policy"],
                dict(scheduler.runtime_dispatch_policy_state()),
                scheduler.runtime_controller_snapshot()["dispatch_queue"]["causal_predecessors"],
            ))
        self.assertEqual(outcomes[0], outcomes[1])


    def assisted_holder_and_queued_switch(self, policy, phones=("helper-c",)):
        """a1 holds the phone lanes with an assisted plan, its cohort is sealed (formation
        window elapsed), and b1 (model B) queues a switch after it."""
        scheduler, model_a, model_b, hot = self.scheduler(
            policy, parallel=4, phones=phones, exclusive_phones=len(phones) > 1,
        )
        gpu = scheduler._runtime_capabilities.executor_by_device["accelerator-b"]
        self.assertEqual(gpu.adapter_parameters["parallel"], 4)
        a1 = self.submit(scheduler, model_a, hot, "a1", 1_000, 64)
        self.assertIn("compute:helper-c", {row.resource_id for row in a1.decision.leases})
        self.assertEqual(self.acquire(scheduler, "a1", 1_000).dispatch_state, "ACQUIRED")
        # A sealed one-member cohort dissolves at acquisition: a1 owns its lanes.
        self.assertIsNone(scheduler.runtime_ticket("a1").decode_cohort)
        b1 = self.submit(scheduler, model_b, self.free_slots(hot, 3), "b1", 1_200, 8)
        self.assertTrue(b1.execution_plan.transitions)
        self.assertEqual(self.queue_view(scheduler)["b1"]["predecessor_request_ids"], ("a1",))
        return scheduler, model_a, hot, a1, b1

    def assisted_joiner_behind_a_queued_switch(self, policy, phones=("helper-c",)):
        """The 003 pattern: a1 holds the phone lanes with an assisted plan, its cohort is
        sealed (formation window elapsed), b1 (model B) queues a switch after it, and the
        energy-aware same-model arrival a3 outlasts a1."""
        scheduler, model_a, hot, a1, b1 = self.assisted_holder_and_queued_switch(policy, phones)
        a3 = self.submit(scheduler, model_a, self.free_slots(hot, 3), "a3", 1_300, 64)
        return scheduler, a1, b1, a3

    def assert_joined_at_arrival(self, scheduler, a3, b1) -> None:
        """a3 is the desktop parent of the running server, reserved at its arrival ahead of b1."""
        self.assertEqual(a3.decision.reason, JOIN_REASON)
        self.assertEqual(a3.decision.start_us, 1_300)
        self.assertEqual(a3.execution_plan.transitions, ())
        self.assertEqual(
            self.decision_record(scheduler, "a3")["selected"]["dispatch_policy"]["kind"], BYPASS_KIND
        )
        self.assertEqual(self.queue_view(scheduler)["b1"]["state"], "REPLAN_REQUIRED")
        self.assertNotIn("refusals", scheduler.runtime_dispatch_policy_state())

    def test_bypass_drops_the_causal_barriers_only_the_cancelled_switch_justified(self) -> None:
        # Hardware (cj4, 003 at t=404): both re-resolutions returned the original start although
        # the switch was cancelled. One way the cancel changes nothing: the arrival carries a
        # causal not-before barrier derived from the switch's leases (a stale-projection repair,
        # ``submit_automated_request``), and the bypass re-resolution inherits it unchanged. After
        # the cancel no live ticket reaches that barrier, so it must not bound the joiner.
        scheduler, model_a, hot, _a1, b1 = self.assisted_holder_and_queued_switch(
            replace(CONTINUOUS_JOIN, max_barrier_extension_s=1)
        )
        switch_end_us = self.lease_end_us(b1, "compute:accelerator-b")
        a3 = scheduler._submit_automated_request_once(
            request("a3", arrival_us=1_300, output_tokens=64), model_a.model_id,
            self.free_slots(hot, 3), selection_mode="energy-aware",
            causal_not_before_by_resource={"compute:accelerator-b": switch_end_us},
        )
        self.assert_joined_at_arrival(scheduler, a3, b1)
        note = self.decision_record(scheduler, "a3")["selected"]["dispatch_policy"]
        self.assertGreaterEqual(note["reserved_start_without_bypass_us"], switch_end_us)

    def test_bypass_keeps_a_causal_barrier_a_live_ticket_still_justifies(self) -> None:
        # The holder's own horizon on the server is not the switch's: a barrier it reaches stays,
        # so the joiner still precedes the switch but not the barrier.
        scheduler, model_a, hot, a1, b1 = self.assisted_holder_and_queued_switch(
            replace(CONTINUOUS_JOIN, max_barrier_extension_s=1)
        )
        holder_end_us = self.lease_end_us(a1, "compute:accelerator-b")
        self.assertLess(holder_end_us, b1.decision.start_us)
        a3 = scheduler._submit_automated_request_once(
            request("a3", arrival_us=1_300, output_tokens=64), model_a.model_id,
            self.free_slots(hot, 3), selection_mode="energy-aware",
            causal_not_before_by_resource={"compute:accelerator-b": holder_end_us},
        )
        self.assertEqual(a3.decision.start_us, holder_end_us)
        self.assertEqual(a3.execution_plan.transitions, ())
        note = self.decision_record(scheduler, "a3")["selected"]["dispatch_policy"]
        self.assertEqual(note["kind"], BYPASS_KIND)
        self.assertGreaterEqual(note["reserved_start_without_bypass_us"], b1.decision.start_us)
        self.assertEqual(self.queue_view(scheduler)["b1"]["state"], "REPLAN_REQUIRED")

    def test_bypass_preprojection_previews_the_desktop_parent_template(self) -> None:
        # The kept epoch's selected template is the assisted route whose lanes a1 holds: previewed
        # first, its start (behind a1, or behind the switch) becomes a not-before barrier on every
        # resource of the desktop parent when residency differs there. During the bypass the
        # pre-projection must preview the epoch's desktop-parent template instead.
        previewed = []
        real = UnifiedScheduler._preview_automated_resources

        def recording(controller, candidate, **kwargs):
            previewed.append((
                controller._runtime_controller.continuous_join_resolution_request_id(),
                candidate.candidate_id,
            ))
            return real(controller, candidate, **kwargs)

        with mock.patch.object(UnifiedScheduler, "_preview_automated_resources", recording):
            scheduler, _a1, b1, a3 = self.assisted_joiner_behind_a_queued_switch(
                replace(CONTINUOUS_JOIN, max_barrier_extension_s=1)
            )
        in_bypass = [route_id for marker, route_id in previewed if marker == "a3"]
        self.assertTrue(in_bypass)
        self.assertEqual(in_bypass[0], "auto:whole:accelerator-b:residency:hot")
        self.assert_joined_at_arrival(scheduler, a3, b1)

    def test_assisted_joiner_behind_the_switch_joins_as_the_desktop_parent(self) -> None:
        scheduler, _a1, b1, a3 = self.assisted_joiner_behind_a_queued_switch(
            replace(CONTINUOUS_JOIN, max_barrier_extension_s=1)
        )
        self.assertEqual(a3.decision.reason, JOIN_REASON)
        self.assertEqual(a3.decision.start_us, 1_300)
        self.assertEqual(a3.execution_plan.transitions, ())
        self.assertEqual(self.queue_view(scheduler)["b1"]["state"], "REPLAN_REQUIRED")
        note = self.decision_record(scheduler, "a3")["selected"]["dispatch_policy"]
        self.assertEqual(note["kind"], BYPASS_KIND)
        self.assertGreaterEqual(
            note["reserved_start_without_bypass_us"], self.lease_end_us(b1, "compute:helper-c")
        )
        self.assertNotIn("refusals", scheduler.runtime_dispatch_policy_state())

    def test_joiner_with_a_free_second_phone_is_the_desktop_parent_not_a_phone_load(self) -> None:
        # Two phones as on the rig (both exclusive residency devices): helper-c is held by
        # a1, helper-d is cold for model A. Under the barrier bypass the joiner is planned as
        # a joiner: every row that would load a phone is out (the bypass never admits a plan
        # that prepares an exclusive device), the held rows are out, the desktop parent wins.
        scheduler, a1, b1, a3 = self.assisted_joiner_behind_a_queued_switch(
            replace(CONTINUOUS_JOIN, max_barrier_extension_s=1), phones=("helper-c", "helper-d"),
        )
        self.assertEqual(a3.decision.reason, JOIN_REASON)
        self.assertEqual(a3.decision.start_us, 1_300)
        self.assertEqual({row.resource_id for row in a3.decision.leases}, {"compute:accelerator-b"})
        rejected = dict(a3.decision.rejected)
        self.assertEqual(rejected[a1.decision.route_id], PHONE_LANES_HELD)
        cold_second_phone = "auto:layers:accelerator-b+helper-d:250000:residency:cold"
        self.assertEqual(rejected[cold_second_phone], RESIDENCY_TRANSITION)
        # Only plans preparing an exclusive device carry the reason: the cold second phone
        # and the non-current whole-model variants of the server; never a helper-c hot row.
        self.assertTrue(all(
            "helper-d" in route_id or route_id.startswith("auto:whole:accelerator-b:")
            for route_id, reason in rejected.items() if reason == RESIDENCY_TRANSITION
        ))
        self.assertEqual(
            self.decision_record(scheduler, "a3")["selected"]["dispatch_policy"]["kind"], BYPASS_KIND
        )
        self.assertEqual(self.queue_view(scheduler)["b1"]["state"], "REPLAN_REQUIRED")
        self.assertNotIn("refusals", scheduler.runtime_dispatch_policy_state())
        self.assertIsNone(scheduler._runtime_controller.continuous_join_resolution_request_id())

    def test_transition_rows_are_rejected_only_during_the_bypass_resolution(self) -> None:
        # Outside a bypass the same arrival keeps its full choice: a free phone may be loaded.
        scheduler, model_a, _, hot = self.scheduler(
            CONTINUOUS_JOIN, parallel=4, phones=("helper-c", "helper-d"), exclusive_phones=True,
        )
        a1 = self.submit(scheduler, model_a, hot, "a1", 1_000, 64)
        self.assertIn("compute:helper-c", {row.resource_id for row in a1.decision.leases})
        self.assertEqual(self.acquire(scheduler, "a1", 1_000).dispatch_state, "ACQUIRED")
        a2 = self.submit(scheduler, model_a, self.free_slots(hot, 3), "a2", 1_500, 8)
        self.assertNotIn(RESIDENCY_TRANSITION, dict(a2.decision.rejected).values())
        self.assertEqual(a2.decision.reason, JOIN_REASON)

    def test_refused_bypass_names_the_failing_start(self) -> None:
        for policy, kind, prefix in (
            (replace(CONTINUOUS_JOIN, max_barrier_extension_s=1), "CONTINUOUS_JOIN_REFUSED",
             "bypass does not start the joiner earlier"),
            (replace(CONTINUOUS_JOIN, model_affinity=True, max_barrier_extension_s=1),
             "AFFINITY_REFUSED", "displacement does not help"),
        ):
            with self.subTest(kind=kind):
                seen, resolve = self.cached_reresolution()
                with mock.patch.object(UnifiedScheduler, "_resolve_automated_submit", resolve):
                    scheduler, _b1, a3, _ = self.joiner_behind_a_queued_switch(policy)
                refusals = [
                    row for row in scheduler.runtime_dispatch_policy_state()["refusals"]
                    if row["kind"] == kind
                ]
                self.assertEqual(len(refusals), 1)
                start_us = seen["a3"].preview.start_us
                self.assertRegex(
                    refusals[0]["reason"],
                    re.escape(f"{prefix}: re-resolved start {start_us} >= original {start_us}")
                    + BOUND_SUFFIX,
                )
                self.assertEqual(a3.decision.start_us, start_us)
                self.assertNotIn("dispatch_policy", self.decision_record(scheduler, "a3")["selected"])

    @staticmethod
    def cached_reresolution():
        """A ``_resolve_automated_submit`` double that returns each request's first
        resolution on every later call, so a displaced re-resolution never starts earlier."""
        seen: dict[str, object] = {}
        real = UnifiedScheduler._resolve_automated_submit

        def resolve(controller, **kwargs):
            request_id = kwargs["request"].request_id
            if request_id not in seen:
                seen[request_id] = real(controller, **kwargs)
            return seen[request_id]

        return seen, resolve

    def test_refused_double_displacement_leaves_no_side_effect(self) -> None:
        # Affinity cancels the switch, re-resolves, refuses and rolls back; the join bypass
        # then cancels the same switch again, refuses and rolls back. Neither attempt may
        # leak a cancellation, a memory release or a queue wake past its transaction.
        states = []
        seen, resolve = self.cached_reresolution()

        def observing(name):
            real = getattr(selection_ops, name)

            def wrapper(controller, **kwargs):
                before = (
                    controller.timeline.causal_state(),
                    controller._runtime_controller.dispatch_order_view(),
                    controller.runtime_ticket("b1").lease_status,
                    controller._runtime_memory.checkpoint(),
                )
                ticket = real(controller, **kwargs)
                after = (
                    controller.timeline.causal_state(),
                    controller._runtime_controller.dispatch_order_view(),
                    controller.runtime_ticket("b1").lease_status,
                    controller._runtime_memory.checkpoint(),
                )
                states.append((name, ticket, before, after))
                return ticket

            return wrapper

        with mock.patch.object(
            selection_ops, "_submit_with_model_affinity", observing("_submit_with_model_affinity")
        ), mock.patch.object(
            selection_ops, "_submit_with_barrier_bypass", observing("_submit_with_barrier_bypass")
        ), mock.patch.object(UnifiedScheduler, "_resolve_automated_submit", resolve):
            scheduler, b1, a3, _ = self.joiner_behind_a_queued_switch(
                replace(CONTINUOUS_JOIN, model_affinity=True, max_barrier_extension_s=1)
            )
        self.assertEqual(
            [name for name, *_ in states],
            ["_submit_with_model_affinity", "_submit_with_barrier_bypass"],
        )
        for name, ticket, before, after in states:
            with self.subTest(attempt=name):
                self.assertIsNone(ticket)
                self.assertEqual(before, after)
                self.assertEqual(after[1]["b1"]["state"], "QUEUED")
                self.assertEqual(after[2], "RESERVED")
        refusals = scheduler.runtime_dispatch_policy_state()["refusals"]
        self.assertEqual(
            [row["kind"] for row in refusals], ["AFFINITY_REFUSED", "CONTINUOUS_JOIN_REFUSED"]
        )
        start_us = seen["a3"].preview.start_us
        for row, prefix in zip(refusals, (
            "displacement does not help", "bypass does not start the joiner earlier",
        )):
            self.assertRegex(
                row["reason"],
                re.escape(f"{prefix}: re-resolved start {start_us} >= original {start_us}")
                + BOUND_SUFFIX,
            )
        self.assertEqual(a3.decision.start_us, start_us)
        statistics = scheduler.runtime_dispatch_policy_state()["statistics"]
        self.assertEqual(statistics["affinity_refusals"], 1)
        self.assertEqual(statistics["continuous_join_refusals"], 1)
        self.assertEqual(statistics["affinity_displacements"], 0)
        self.assertEqual(statistics["continuous_join_bypasses"], 0)
        self.assertIn("b1", self.queue_view(scheduler)["a3"]["predecessor_request_ids"])
        self.assertEqual(scheduler.runtime_ticket("b1").lease_status, "RESERVED")
        self.assertIsNone(scheduler._runtime_controller.continuous_join_resolution_request_id())


class ContinuousJoinUnitTests(unittest.TestCase):
    """Cheap checks of the join helpers that need no scheduler."""

    def test_lease_holders_include_the_decode_cohort_owner(self) -> None:
        decision = leased("a1", ("compute:helper-c", (0,), 1_000, 2_000))
        shared = replace(decision, leases=tuple(
            replace(row, owner_id="decode-cohort-1") for row in decision.leases
        ))
        tickets = (
            mock.Mock(request=mock.Mock(request_id="a1"), live_leases=shared.leases),
            mock.Mock(request=mock.Mock(request_id="a2"), live_leases=decision.leases),
        )
        self.assertEqual(
            join_selection_ops.lease_holders(tickets), frozenset({"a1", "a2", "decode-cohort-1"})
        )

    def test_no_gain_reason_names_the_failing_half(self) -> None:
        def resolution(start_us, transitions=()):
            plan = mock.Mock(transitions=transitions)
            return mock.Mock(
                preview=mock.Mock(start_us=start_us), selected=mock.Mock(plan=plan)
            )

        load = mock.Mock(
            transition_id="load:accelerator-b", source_state="cold", target_state="hot",
            prepares_device_ids=("accelerator-b",),
        )
        phone_only = mock.Mock(
            transition_id="load:helper-c", source_state="cold", target_state="hot",
            prepares_device_ids=("helper-c",),
        )
        exclusive = {"accelerator-b": "compute:accelerator-b"}
        self.assertEqual(
            affinity_ops.no_gain_reason(resolution(1_300), resolution(1_250), exclusive, "p"),
            "p: re-resolved start 1300 >= original 1250",
        )
        self.assertEqual(
            affinity_ops.no_gain_reason(
                resolution(1_200, (phone_only, load)), resolution(1_250), exclusive, "p"
            ),
            "p: re-resolved plan changes residency: load:accelerator-b (cold->hot)",
        )
        self.assertIsNone(
            affinity_ops.no_gain_reason(
                resolution(1_200, (phone_only,)), resolution(1_250), exclusive, "p"
            )
        )

    def test_no_gain_reason_appends_the_bound_to_the_start_half_only(self) -> None:
        plan = mock.Mock(transitions=())
        resolution = mock.Mock(preview=mock.Mock(start_us=1_300), selected=mock.Mock(plan=plan))
        original = mock.Mock(preview=mock.Mock(start_us=1_250))
        self.assertEqual(
            affinity_ops.no_gain_reason(
                resolution, original, {}, "p", bound="bounded by not_before[gpu]=1300"
            ),
            "p: re-resolved start 1300 >= original 1250 (bounded by not_before[gpu]=1300)",
        )
        earlier = mock.Mock(preview=mock.Mock(start_us=1_200), selected=mock.Mock(plan=plan))
        self.assertIsNone(affinity_ops.no_gain_reason(earlier, original, {}, "p", bound="x"))

    def test_bound_detail_names_barriers_transitions_and_calendar(self) -> None:
        load = mock.Mock(transition_id="load:accelerator-b")
        plan = mock.Mock(resource_ids=("compute:accelerator-b", "link:pcie-in"), transitions=(load,))
        preview = mock.Mock(start_us=2_554, blocking_resources=("compute:helper-c",))
        resolution = mock.Mock(preview=preview, selected=mock.Mock(plan=plan))
        context = mock.Mock(live_not_before_by_resource={
            "compute:accelerator-b": 2_554, "link:pcie-in": 100, "compute:helper-c": 9_000,
        })
        self.assertEqual(
            join_request_ops.bound_detail(context, resolution),
            "bounded by not_before[compute:accelerator-b]=2554, transition[load:accelerator-b],"
            " calendar[compute:helper-c]",
        )
        bare = mock.Mock(
            preview=mock.Mock(start_us=1_300, blocking_resources=()),
            selected=mock.Mock(plan=mock.Mock(resource_ids=("compute:accelerator-b",), transitions=())),
        )
        self.assertEqual(
            join_request_ops.bound_detail(mock.Mock(live_not_before_by_resource={}), bare),
            "no barrier recorded",
        )

    def test_live_not_before_keeps_only_barriers_a_live_ticket_reaches(self) -> None:
        def ticket(request_id, resource_id, end_us, *, lease_status="RESERVED", state="QUEUED"):
            decision = leased(request_id, (resource_id, (0,), 1_000, end_us))
            return mock.Mock(
                request=mock.Mock(request_id=request_id), decision=decision,
                final_reserved_until_us={}, lease_status=lease_status, dispatch_state=state,
            )

        controller = mock.Mock()
        controller._runtime_controller.current_tickets.return_value = (
            ticket("a1", "compute:accelerator-b", 2_321, state="ACQUIRED"),
            ticket("b1", "compute:accelerator-b", 2_554),
            ticket("b2", "compute:helper-c", 3_000, lease_status="CANCELLED"),
            ticket("c1", "link:usb-in", 4_000, state="COMPLETED"),
        )
        barriers = {
            "compute:accelerator-b": 2_554, "compute:helper-c": 2_600, "link:usb-in": 3_500,
            "link:pcie-in": 10,
        }
        self.assertEqual(
            join_request_ops.live_not_before_after_displacement(controller, barriers, ("b1",)),
            {},
        )
        self.assertEqual(
            join_request_ops.live_not_before_after_displacement(
                controller, {"compute:accelerator-b": 2_321}, ("b1",)
            ),
            {"compute:accelerator-b": 2_321},
        )
        self.assertEqual(
            join_request_ops.live_not_before_after_displacement(controller, barriers, ()),
            {"compute:accelerator-b": 2_554},
        )
        with self.assertRaises(UnifiedScheduleError):
            join_request_ops.live_not_before_after_displacement(
                controller, {"compute:accelerator-b": 2_554.0}, ()
            )

    def test_continuous_join_resolution_marker_is_scoped_and_fail_closed(self) -> None:
        controller = RuntimeController()
        self.assertIsNone(controller.continuous_join_resolution_request_id())
        with controller.continuous_join_resolution("a3"):
            self.assertEqual(controller.continuous_join_resolution_request_id(), "a3")
            with self.assertRaises(RuntimeControllerError):
                with controller.continuous_join_resolution("a4"):
                    pass
            self.assertEqual(controller.continuous_join_resolution_request_id(), "a3")
        self.assertIsNone(controller.continuous_join_resolution_request_id())
        with self.assertRaises(RuntimeControllerError):
            with controller.continuous_join_resolution(3):
                pass
        with self.assertRaises(ValueError):
            with controller.continuous_join_resolution("a5"):
                raise ValueError("body")
        self.assertIsNone(controller.continuous_join_resolution_request_id())


class DispatchPolicyConfigurationTests(unittest.TestCase):
    ROW = {
        "schema": CAMPAIGN_MANIFEST_SCHEMA, "campaign_id": "dispatch-policy-test",
        "rig_manifest_path": "rig.json", "models_manifest_path": "models.json",
        "evidence_manifest_path": "evidence.json", "selection_mode": "energy-aware",
        "maximum_latency_ppm": 1_100_000, "energy_attribution_kind": "diagnostic",
        "trace": {"large_requests_path": "large.json", "overlay_requests_path": "overlay.json",
                  "trace_manifest_path": "trace.json"},
    }

    def test_campaign_dispatch_policy_is_opt_in_and_validated(self) -> None:
        plain = CampaignManifest.from_json(self.ROW, Path("/inputs"))
        self.assertIsNone(plain.dispatch_policy)
        self.assertNotIn("dispatch_policy", plain.to_json())
        value = {"work_conserving_admission": True, "model_affinity": True,
                 "affinity_maximum_bypasses": 3, "affinity_maximum_wait_us": 300_000_000}
        configured = CampaignManifest.from_json({**self.ROW, "dispatch_policy": value}, Path("/inputs"))
        self.assertEqual(dict(configured.dispatch_policy), value)
        self.assertEqual(CampaignManifest.from_json(configured.to_json(), Path("/inputs")), configured)
        for invalid in ({}, {"model_affinity": True}, {"work_conserving_admission": 1},
                        {"affinity_maximum_bypasses": -1}, {"unknown": True}, [True]):
            with self.subTest(invalid=invalid), self.assertRaises(SchedulerConfigurationError):
                CampaignManifest.from_json({**self.ROW, "dispatch_policy": invalid}, Path("/inputs"))

    def test_campaign_continuous_join_requires_work_conserving_and_an_integer_bound(self) -> None:
        value = {"work_conserving_admission": True, "continuous_join": True,
                 "max_barrier_extension_s": 30}
        configured = CampaignManifest.from_json({**self.ROW, "dispatch_policy": value}, Path("/inputs"))
        self.assertEqual(dict(configured.dispatch_policy), value)
        self.assertEqual(CampaignManifest.from_json(configured.to_json(), Path("/inputs")), configured)
        self.assertEqual(
            runner._dispatch_policy_from_json(json.dumps(dict(configured.dispatch_policy))),
            RuntimeDispatchPolicy(work_conserving_admission=True, continuous_join=True,
                                  max_barrier_extension_s=30),
        )
        for invalid in ({"continuous_join": True},
                        {"work_conserving_admission": True, "continuous_join": 1},
                        {"work_conserving_admission": True, "continuous_join": True,
                         "max_barrier_extension_s": -1},
                        {"work_conserving_admission": True, "continuous_join": True,
                         "max_barrier_extension_s": True},
                        {"work_conserving_admission": True, "continuous_join": True,
                         "max_barrier_extension_s": 1.5}):
            with self.subTest(invalid=invalid), self.assertRaises(SchedulerConfigurationError):
                CampaignManifest.from_json({**self.ROW, "dispatch_policy": invalid}, Path("/inputs"))

    def test_campaign_residency_hysteresis_requires_work_conserving_and_an_integer(self) -> None:
        value = {"work_conserving_admission": True, "model_affinity": True,
                 "residency_hysteresis_s": HYSTERESIS_S}
        configured = CampaignManifest.from_json({**self.ROW, "dispatch_policy": value}, Path("/inputs"))
        self.assertEqual(dict(configured.dispatch_policy), value)
        self.assertEqual(CampaignManifest.from_json(configured.to_json(), Path("/inputs")), configured)
        self.assertEqual(
            runner._dispatch_policy_from_json(json.dumps(dict(configured.dispatch_policy))),
            HYSTERESIS,
        )
        zero = {"work_conserving_admission": True, "residency_hysteresis_s": 0}
        self.assertEqual(
            runner._dispatch_policy_from_json(json.dumps(zero)).to_json(), WORK_CONSERVING.to_json(),
        )
        for invalid in ({"residency_hysteresis_s": 20},
                        {"work_conserving_admission": True, "residency_hysteresis_s": -1},
                        {"work_conserving_admission": True, "residency_hysteresis_s": True},
                        {"work_conserving_admission": True, "residency_hysteresis_s": 2.5},
                        {"work_conserving_admission": True, "residency_hysteresis_s": "20"}):
            with self.subTest(invalid=invalid), self.assertRaises(SchedulerConfigurationError):
                CampaignManifest.from_json({**self.ROW, "dispatch_policy": invalid}, Path("/inputs"))

    def test_campaign_hysteresis_probability_is_a_ppm_integer_with_a_window(self) -> None:
        value = {"work_conserving_admission": True, "model_affinity": True,
                 "residency_hysteresis_s": HYSTERESIS_S,
                 "residency_hysteresis_min_probability_ppm": 0}
        configured = CampaignManifest.from_json({**self.ROW, "dispatch_policy": value}, Path("/inputs"))
        self.assertEqual(dict(configured.dispatch_policy), value)
        self.assertEqual(
            runner._dispatch_policy_from_json(json.dumps(dict(configured.dispatch_policy))),
            SPECULATIVE,
        )
        window = {"work_conserving_admission": True, "residency_hysteresis_s": HYSTERESIS_S}
        for invalid in ({"work_conserving_admission": True,
                         "residency_hysteresis_min_probability_ppm": 500_000},
                        {**window, "residency_hysteresis_min_probability_ppm": 1_000_001},
                        {**window, "residency_hysteresis_min_probability_ppm": -1},
                        {**window, "residency_hysteresis_min_probability_ppm": True},
                        {**window, "residency_hysteresis_min_probability_ppm": 0.5}):
            with self.subTest(invalid=invalid), self.assertRaises(SchedulerConfigurationError):
                CampaignManifest.from_json({**self.ROW, "dispatch_policy": invalid}, Path("/inputs"))

    def test_runner_refuses_continuous_join_without_server_policy_coherence(self) -> None:
        policy = runner._dispatch_policy_from_json(
            json.dumps({"work_conserving_admission": True, "continuous_join": True})
        )
        for adaptive_config in (None, AdaptiveDecodeConfig()):
            with self.subTest(adaptive_config=adaptive_config), self.assertRaises(
                runner.UnifiedTraceError
            ):
                runner._require_continuous_join_prerequisites(policy, adaptive_config)
        runner._require_continuous_join_prerequisites(
            policy, AdaptiveDecodeConfig(server_policy_coherence=True)
        )
        runner._require_continuous_join_prerequisites(WORK_CONSERVING, None)

    def test_runner_builds_the_scheduler_policy_from_the_campaign_field(self) -> None:
        action = next(
            row for row in arguments._build_parser()._actions
            if "--dispatch-policy-json" in row.option_strings
        )
        self.assertEqual(action.dest, "dispatch_policy_json")
        self.assertIsNone(action.default)
        policy = runner._dispatch_policy_from_json(
            json.dumps({"work_conserving_admission": True})
        )
        self.assertEqual(policy, WORK_CONSERVING)
        self.assertIsNone(runner._dispatch_policy_from_json(None))
        with self.assertRaises(runner.UnifiedTraceError):
            runner._dispatch_policy_from_json(json.dumps({"model_affinity": True}))
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.configure_runtime_dispatch_policy(policy)
        self.assertEqual(
            scheduler.runtime_dispatch_policy_state()["policy"], policy.to_json()
        )


if __name__ == "__main__":
    unittest.main()
