#!/usr/bin/env python3
"""Elastic phones: a recovered request keeps its queue place (hardware run g9).

g9 (two-phone eval_v2 arm, ``helper_loss_recovery: mask_out``, work-conserving admission + model
affinity): the Pixel helper was lost while Qwen 001 decoded on the Qwen server, the server masked it
out and 001 got a same-server FALLBACK, but 4 s later the dispatcher replanned the waiting Gemma 002
onto the "idle" server and stopped it; 001 (and the queued Qwen 003) ran only after a Qwen reload
575 s later. Cause: the FALLBACK is a new queue admission with the newest sequence, so the arrival
rule ordered it behind the older Gemma switch whose running predecessor (the failed attempt) had
just left the queue, and the switch's replan then deferred it as a causal dependent.

Now an elastic recovery keeps the failed attempt's place (its sequence and the attempts that waited
on it) when another model's attempt waited on it. Recorded tests, no hardware: the two-model
exclusive-GPU fixture of test_dispatch_policy (model A = Qwen resident, model B = Gemma), at the
scheduler and through the arrival coordinator. Also the THERMAL_DEFERRAL rows (one per onset of a
device's thermal gate in route feasibility).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest

from research_dev.scheduler import (
    RuntimeDispatchPolicy,
    RuntimeDispatchQueue,
    RuntimeQueueError,
)
from research_dev.scheduler._internal.runtime_execution import RuntimeExecutionFailure
from research_dev.scheduler.adapters import (
    CanonicalArrivalCoordinator,
    CanonicalRuntimeSubmission,
    RawTransitionObservation,
)
from research_dev.scheduler.adapters.contracts import PhysicalBackendFailure
from research_dev.scheduler.campaigns.burstgpt.runner import _elastic_drop_result

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
import test_automated_runtime as automated  # noqa: E402
import test_dispatch_policy as dispatch  # noqa: E402
from test_physical_adapter import FakeMeasuredBackend  # noqa: E402

AFFINITY = RuntimeDispatchPolicy(work_conserving_admission=True, model_affinity=True)
HELPER = "helper-c"
FAILED_AT_US = 1_050
LOST = "physical_backend_failed:helper_lost:" + HELPER


def compute_end(ticket) -> int:
    return max(row.reserved_until_us for row in ticket.decision.leases
               if row.resource_id == "compute:accelerator-b")


class TwoModelCase(unittest.TestCase):
    """The test_dispatch_policy fixture: models A and B share one exclusive two-lane GPU, A is
    resident. a1 (A, long) runs, b1 (B) waits for the switch, a3 (A) arrives after b1."""

    setUp = dispatch.DispatchPolicySchedulerTests.setUp
    scheduler = dispatch.DispatchPolicySchedulerTests.scheduler
    submit = staticmethod(dispatch.DispatchPolicySchedulerTests.submit)
    queue_view = staticmethod(dispatch.DispatchPolicySchedulerTests.queue_view)
    ready_now = staticmethod(dispatch.DispatchPolicySchedulerTests.ready_now)
    published = staticmethod(dispatch.DispatchPolicySchedulerTests.published)

    def g9_queue(self, policy=AFFINITY):
        scheduler, model_a, model_b, hot = self.scheduler(policy)
        tickets = {
            "a1": self.submit(scheduler, model_a, hot, "a1", 1_000, 640),
            "b1": self.submit(scheduler, model_b, hot, "b1", 1_100, 8),
            "a3": self.submit(scheduler, model_a, hot, "a3", 1_200, 640),
        }
        scheduler.wait_runtime_request("a1", time.monotonic_ns() - 1_000 * 1_000)
        return scheduler, hot, next(iter(hot.executors)), tickets

    def fail_a1(self, scheduler, snapshot, failure: RuntimeExecutionFailure):
        return scheduler.fail_automated_request(
            "a1", failed_at_us=FAILED_AT_US, reason=LOST,
            snapshot=self.published(snapshot, FAILED_AT_US), physical_failure=failure)

    @staticmethod
    def masked(executor_id: str) -> RuntimeExecutionFailure:
        return RuntimeExecutionFailure("helper_lost", True, True, failed_device_ids=(HELPER,),
                                       masked_executor_id=executor_id)

    @staticmethod
    def retired(executor_id: str) -> RuntimeExecutionFailure:
        return RuntimeExecutionFailure("helper_lost", True, True, failed_device_ids=(HELPER,),
                                       exited_executor_id=executor_id)

    def replan(self, scheduler, request_id: str, at_us: int, snapshot):
        wake = scheduler.wait_runtime_request(request_id, time.monotonic_ns() - at_us * 1_000)
        self.assertEqual(wake.dispatch_state, "REPLAN_REQUIRED")
        return scheduler.replan_automated_request(
            request_id, observed_at_us=at_us, reason=wake.dispatch_receipt.wake_reason,
            snapshot=self.published(snapshot, at_us), expected_ticket_id=wake.ticket_id,
            expected_queue_generation=wake.dispatch_receipt.queue_generation)

    @staticmethod
    def records(scheduler, request_id: str, kind: str):
        return [row for row in scheduler.runtime_decision_log()["records"]
                if row["event_kind"] == kind and row["request_ids"] == [request_id]]


class RetainedOrderQueueTests(unittest.TestCase):
    def test_a_retained_order_keeps_the_sequence_and_the_waiting_followers(self):
        queue = RuntimeDispatchQueue(dispatch.WORK_CONSERVING)
        queue.admit(dispatch.leased("a1", ("gpu", (0,), 0, 1_000)), 0)
        dispatch.acquire(queue, "a1", 0)
        queue.admit(dispatch.leased("b1", ("gpu", (0, 1), 1_000, 1_100)), 10,
                    residency_transition_barrier=True)
        queue.admit(dispatch.leased("c1", ("cpu", (0,), 20, 30)), 20)
        view = queue.dispatch_order_view()
        sequence, waiting = view["a1"]["sequence"], ("b1", "c1")
        self.assertEqual(view["b1"]["predecessor_request_ids"], ("a1",))
        queue.complete("a1", 100)  # the failed attempt leaves the queue (RuntimeController.fail)
        queue.admit(dispatch.leased("a1", ("gpu", (0,), 100, 1_100)), 100,
                    retained_order=(sequence, waiting))
        view = queue.dispatch_order_view()
        self.assertEqual(view["a1"]["sequence"], sequence)
        self.assertEqual(view["a1"]["predecessor_request_ids"], ())
        self.assertEqual(view["b1"]["predecessor_request_ids"], ("a1",))
        # a waiter on other lanes is not ordered behind the recovery
        self.assertEqual(view["c1"]["predecessor_request_ids"], ())
        # the next arrival still gets a new, later sequence
        queue.admit(dispatch.leased("a2", ("gpu", (1,), 200, 300)), 200)
        self.assertGreater(queue.dispatch_order_view()["a2"]["sequence"], view["c1"]["sequence"])

    def test_a_retained_order_is_validated(self):
        queue = RuntimeDispatchQueue(dispatch.WORK_CONSERVING)
        queue.admit(dispatch.leased("a1", ("gpu", (0,), 0, 1_000)), 0)
        queue.admit(dispatch.leased("b1", ("gpu", (1,), 0, 1_000)), 0)
        for order in ((1, ()), (3, ()), (0, ()), (True, ()), (1.0, ()), (None, ()),
                      (2, ("x", "x")), (2, ("c1",)), "12", (2,)):
            with self.subTest(order=order), self.assertRaises(RuntimeQueueError):
                queue.admit(dispatch.leased("c1", ("gpu", (0,), 0, 10)), 0, retained_order=order)
        self.assertEqual(set(queue.dispatch_order_view()), {"a1", "b1"})
        # a replan admission never takes another place
        queue.require_replan("b1", "test")
        queue.retire_replan("b1")
        with self.assertRaises(RuntimeQueueError):
            queue.admit(dispatch.leased("b1", ("gpu", (1,), 0, 10)), 0, retained_order=(1, ()))


class RecoveryDispatchSchedulerTests(TwoModelCase):
    def test_g9_masked_server_serves_the_recovery_and_the_resident_work_before_the_switch(self):
        scheduler, hot, gpu, tickets = self.g9_queue()
        # before the loss: a3 (model affinity) runs ahead of the switch, which waits on a1 and a3
        view = self.queue_view(scheduler)
        self.assertEqual(set(view["b1"]["predecessor_request_ids"]), {"a1", "a3"})
        sequence = view["a1"]["sequence"]
        recovery = self.fail_a1(scheduler, hot, self.masked(gpu))
        fallback = recovery.fallback
        # the recovery stays on the live masked server: same route, no transition
        self.assertEqual(fallback.execution_plan.transitions, ())
        self.assertEqual(fallback.decision.route_id, tickets["a1"].decision.route_id)
        self.assertEqual(fallback.binding.executor_id, gpu)
        view = self.queue_view(scheduler)
        self.assertEqual(view["a1"]["sequence"], sequence)
        self.assertEqual(view["a1"]["predecessor_request_ids"], ())
        self.assertEqual(set(view["b1"]["predecessor_request_ids"]), {"a1", "a3"})
        self.assertTrue(self.ready_now(scheduler, "a1", FAILED_AT_US))
        (record,) = self.records(scheduler, "a1", "FALLBACK")
        self.assertEqual(record["decision_reason"], "PAIRED_DESKTOP_RECOVERY")
        self.assertEqual(record["selected"]["dispatch_policy"], {
            "kind": "RECOVERY_RETAINED_QUEUE_PLACE", "sequence": sequence,
            "waiting_request_ids": ["b1"]})
        # 400 us later the switch replans (g9: 435.4 s): behind the recovery and a3, not now
        switch = self.replan(scheduler, "b1", FAILED_AT_US + 400, hot)
        self.assertTrue(switch.execution_plan.transitions)
        resident_end = max(compute_end(scheduler.runtime_ticket(rid)) for rid in ("a1", "a3"))
        self.assertGreaterEqual(switch.decision.start_us, resident_end)
        view = self.queue_view(scheduler)
        self.assertEqual(view["a1"]["state"], "QUEUED")
        self.assertEqual(set(view["b1"]["predecessor_request_ids"]), {"a1", "a3"})
        self.assertTrue(self.ready_now(scheduler, "a1", FAILED_AT_US + 400))
        self.assertTrue(self.ready_now(scheduler, "a3", 1_200))

    def test_a_protected_switch_still_waits_for_the_recovered_request(self):
        """The switch was protected (bypass bound 0) so a3 queued behind it: with a1 running the
        order is a1, b1, a3, and the recovery keeps it."""
        scheduler, hot, gpu, _ = self.g9_queue(RuntimeDispatchPolicy(
            work_conserving_admission=True, model_affinity=True, affinity_maximum_bypasses=0))
        view = self.queue_view(scheduler)
        self.assertEqual(view["b1"]["predecessor_request_ids"], ("a1",))
        self.assertIn("b1", view["a3"]["predecessor_request_ids"])
        self.fail_a1(scheduler, hot, self.masked(gpu))
        self.assertEqual(self.queue_view(scheduler)["b1"]["predecessor_request_ids"], ("a1",))
        switch = self.replan(scheduler, "b1", FAILED_AT_US + 400, hot)
        self.assertGreaterEqual(switch.decision.start_us, compute_end(scheduler.runtime_ticket("a1")))
        view = self.queue_view(scheduler)
        self.assertEqual(view["b1"]["predecessor_request_ids"], ("a1",))
        self.assertIn("b1", view["a3"]["predecessor_request_ids"])
        self.assertTrue(self.ready_now(scheduler, "a1", FAILED_AT_US + 400))

    def test_retire_mode_plans_the_recovered_reload_before_the_later_switch(self):
        """Retire mode: the server is gone and a reload is unavoidable, but the recovered request
        (original arrival first) is reloaded before the later arrival's switch."""
        scheduler, model_a, model_b, hot = self.scheduler(AFFINITY)
        gpu = next(iter(hot.executors))
        self.submit(scheduler, model_a, hot, "a1", 1_000, 640)
        self.submit(scheduler, model_b, hot, "b1", 1_100, 8)
        scheduler.wait_runtime_request("a1", time.monotonic_ns() - 1_000 * 1_000)
        sequence = self.queue_view(scheduler)["a1"]["sequence"]
        cold = replace(hot, residency=())
        fallback = self.fail_a1(scheduler, cold, self.retired(gpu)).fallback
        self.assertTrue(fallback.execution_plan.transitions)
        view = self.queue_view(scheduler)
        self.assertEqual(view["a1"]["sequence"], sequence)
        self.assertEqual(view["a1"]["predecessor_request_ids"], ())
        self.assertEqual(view["b1"]["predecessor_request_ids"], ("a1",))
        switch = self.replan(scheduler, "b1", FAILED_AT_US + 400, cold)
        self.assertGreaterEqual(switch.decision.start_us, compute_end(scheduler.runtime_ticket("a1")))
        self.assertTrue(self.ready_now(scheduler, "a1", FAILED_AT_US + 400))

    def test_same_model_waiters_keep_todays_recovery_order(self):
        """No other model waits on the failed attempt: the recovery is a new admission as before."""
        scheduler, model_a, _, hot = self.scheduler(AFFINITY)
        gpu = next(iter(hot.executors))
        self.submit(scheduler, model_a, hot, "a1", 1_000, 640)
        self.submit(scheduler, model_a, hot, "a2", 1_100, 640)
        self.submit(scheduler, model_a, hot, "a3", 1_200, 640)
        scheduler.wait_runtime_request("a1", time.monotonic_ns() - 1_000 * 1_000)
        newest = max(row["sequence"] for row in self.queue_view(scheduler).values())
        self.fail_a1(scheduler, hot, self.masked(gpu))
        self.assertEqual(self.queue_view(scheduler)["a1"]["sequence"], newest + 1)
        (record,) = self.records(scheduler, "a1", "FALLBACK")
        self.assertNotIn("dispatch_policy", record["selected"])

    def test_a_failure_that_is_not_elastic_keeps_todays_queue(self):
        """Without elastic phones no failure is helper_lost/server_exited: the queue changes only
        by the failed attempt leaving it, as before."""
        for failure in (RuntimeExecutionFailure("unspecified", True, False),
                        RuntimeExecutionFailure("transport", True, True)):
            with self.subTest(phase=failure.phase):
                scheduler, hot, _, _ = self.g9_queue()
                before = {
                    request_id: (row["sequence"], tuple(sorted(
                        set(row["predecessor_request_ids"]) - {"a1"})), row["state"])
                    for request_id, row in self.queue_view(scheduler).items() if request_id != "a1"
                }
                recovery = self.fail_a1(scheduler, hot, failure)
                self.assertIsNone(recovery.fallback)
                self.assertEqual({
                    request_id: (row["sequence"], row["predecessor_request_ids"], row["state"])
                    for request_id, row in self.queue_view(scheduler).items()
                }, before)
                self.assertIsNone(scheduler._runtime_controller._pending_retained_order)


class TwoModelGpu(FakeMeasuredBackend):
    """One exclusive GPU holding one model at a time; a1's first attempt loses its helper once
    b1 and a3 are queued and a3 started beside it (mask_out: the server stays; retire: it exits
    and the GPU is cold until the next load)."""

    def __init__(self, scheduler, models, executor_id, epoch_ns, *, retire: bool) -> None:
        super().__init__(5_000)
        self.scheduler, self.models, self.executor_id = scheduler, models, executor_id
        self.epoch_ns, self.retire = epoch_ns, retire
        self.resident = models[0]
        self.lock = threading.Lock()
        self.log: list[tuple[str, str, str]] = []
        self.queued = threading.Event()
        self.a3_started = threading.Event()

    def snapshot(self, hot, at_us: int):
        with self.lock:
            model = self.resident
        rows = () if model is None else tuple(replace(
            row, model_id=model.model_id, artifact_sha256=model.artifact_sha256,
            resident_tensor_ids=tuple(tensor.tensor_id for tensor in model.tensors),
            resident_bytes=model.tensor_bytes, generation=1 + len(self.log),
        ) for row in hot.residency)
        return replace(TwoModelCase.published(hot, at_us), snapshot_id=f"gpu-{at_us}-{len(self.log)}",
                       residency=rows)

    def apply_transition(self, command, payload, control_check):
        control_check()
        with self.lock:
            self.resident = next(row for row in self.models
                                 if row.artifact_sha256 == command.artifact_sha256)
            self.log.append(("load", command.request_id, self.resident.model_id))
        start_us = self.scheduler.runtime_ticket(command.request_id).decision.start_us
        return RawTransitionObservation(
            started_us=start_us, finished_us=start_us + command.transition.latency_us,
            status="COMPLETED", evicted_artifact_sha256s=tuple(sorted({
                row.artifact_sha256 for row in command.transition.evictions})))

    def execute(self, command, payload, control_check):
        with self.lock:
            resident = self.resident
            self.log.append(("run", command.ticket_id, None if resident is None else resident.model_id))
        if resident is None or command.artifact_sha256 != resident.artifact_sha256:
            raise AssertionError("the GPU ran a model that is not resident")
        if command.ticket_id == "a3:attempt:0":
            self.a3_started.set()
        if command.ticket_id != "a1:attempt:0":
            return super().execute(command, payload, control_check)
        self.queued.wait(10)
        self.a3_started.wait(10)
        with self.lock:
            if self.retire:
                self.resident = None
        now_us = (time.monotonic_ns() - self.epoch_ns) // 1000
        facts = ({"executor_id": self.executor_id, "returncode": 0} if self.retire
                 else {"masked_executor_id": self.executor_id})
        raise PhysicalBackendFailure(
            "helper lost", phase="helper_lost", retry_safe=True, execution_started=True,
            started_us=command.planned_start_us, finished_us=max(command.planned_start_us, now_us),
            failed_device_ids=(HELPER,), **facts)


class RecoveryDispatchCoordinatorTests(TwoModelCase):
    def run_g9(self, *, retire: bool):
        scheduler, model_a, model_b, hot = self.scheduler(AFFINITY)
        for name in ("start_runtime_lease_renewal", "check_runtime_lease_renewal",
                     "stop_runtime_lease_renewal"):
            setattr(scheduler, name, lambda *_args, **_kwargs: None)
        quarantined = []
        # this fixture's catalog has no phone executor; the quarantine itself is tested elsewhere
        scheduler.quarantine_device = lambda device_id, *, reason, at_us: quarantined.append(device_id)
        epoch_ns = time.monotonic_ns()
        gpu = TwoModelGpu(scheduler, (model_a, model_b), next(iter(hot.executors)), epoch_ns,
                          retire=retire)
        coordinator = CanonicalArrivalCoordinator(
            scheduler, gpu, epoch_ns=epoch_ns, snapshot_provider=lambda _t, at_us: gpu.snapshot(hot, at_us),
            max_workers=4, lease_guard_us=10_000, lease_quantum_us=1_000_000)
        token = scheduler._runtime_controller.queue.hold_wake()
        try:
            for request_id, model, arrival_us, output_tokens in (
                ("a1", model_a, 1_000, 640), ("b1", model_b, 1_100, 8), ("a3", model_a, 1_200, 640),
            ):
                coordinator.submit(CanonicalRuntimeSubmission(
                    dispatch.request(request_id, arrival_us=arrival_us, output_tokens=output_tokens),
                    model.model_id, gpu.snapshot(hot, arrival_us), {},
                    selection_mode="desktop-baseline"), observed_at_us=arrival_us)
        finally:
            scheduler._runtime_controller.queue.release_wake(token)
        gpu.queued.set()
        try:
            completed = coordinator.drain(timeout_s=60)
        finally:
            coordinator.close(wait=False)
        self.assertEqual(sorted(completed.request_ids), ["a1", "a3", "b1"])
        self.assertEqual(quarantined, [HELPER])
        (recovered,) = completed.executions["a1"].recovery_events
        self.assertEqual(recovered["kind"], "REQUEST_RECOVERED")
        return scheduler, gpu.log

    @staticmethod
    def recovered_runs(log):
        return [row for row in log if row[0] == "run" and row[1].startswith("a1:")
                and row[1] != "a1:attempt:0"]

    def test_g9_mask_out_runs_the_recovery_and_the_resident_work_before_the_switch(self):
        scheduler, log = self.run_g9(retire=False)
        # one switch only (no reload of model-a), after the recovered a1 ran on the kept server
        self.assertEqual([row for row in log if row[0] == "load"], [("load", "b1", "model-b")])
        switch = log.index(("load", "b1", "model-b"))
        (recovered,) = self.recovered_runs(log)
        self.assertEqual(recovered[2], "model-a")
        self.assertLess(log.index(recovered), switch)
        self.assertLess(log.index(("run", "a3:attempt:0", "model-a")), switch)
        self.assertEqual([row[1].split(":")[0] for row in log[switch:] if row[0] == "run"], ["b1"])
        (fallback,) = self.records(scheduler, "a1", "FALLBACK")
        self.assertEqual(fallback["selected"]["dispatch_policy"]["kind"], "RECOVERY_RETAINED_QUEUE_PLACE")

    def test_retire_mode_reloads_the_recovered_model_before_the_later_switch(self):
        _, log = self.run_g9(retire=True)
        loads = [row for row in log if row[0] == "load"]
        self.assertEqual(loads, [("load", "a1", "model-a"), ("load", "b1", "model-b")])
        (recovered,) = self.recovered_runs(log)
        self.assertEqual(recovered[2], "model-a")
        self.assertLess(log.index(loads[0]), log.index(recovered))
        self.assertLess(log.index(recovered), log.index(loads[1]))


class ThermalDeferralEventTests(unittest.TestCase):
    """The thermal gate of route feasibility (THERMAL_LIMIT excludes every route through a hot
    device; g9 logged it 512 times) is visible as one THERMAL_DEFERRAL row per onset."""

    setUp = automated.AutomatedRuntimeTests.setUp
    tearDown = automated.AutomatedRuntimeTests.tearDown
    scheduler_and_manifest = automated.AutomatedRuntimeTests.scheduler_and_manifest

    def phone_at(self, snapshot, temperature_millic: int, qualified=None):
        helper = snapshot.executors["executor:helper-c"]
        return replace(snapshot, executors={**snapshot.executors, "executor:helper-c": replace(
            helper, temperature_millic=temperature_millic, thermal_qualified=qualified)})

    def generate(self, scheduler, manifest, snapshot, at_us: int, index: int):
        candidates = scheduler.generate_automated_candidates(
            automated.request(f"thermal-{index}", arrival_us=at_us), manifest.model_id, snapshot,
            observed_at_us=at_us)
        return [row for row in candidates.candidates if HELPER in row.device_ids]

    def test_one_event_per_onset_and_one_when_the_gate_clears(self):
        scheduler, manifest = self.scheduler_and_manifest()
        snapshot = automated.runtime_snapshot(manifest)
        hot, cool = self.phone_at(snapshot, 95_000), self.phone_at(snapshot, 60_000)
        timeline = ((cool, 1_000), (hot, 2_000), (hot, 3_000), (hot, 4_000), (cool, 1_500),
                    (cool, 5_000), (cool, 6_000), (self.phone_at(snapshot, 80_000, False), 7_000))
        for index, (observed, at_us) in enumerate(timeline):
            rows = self.generate(scheduler, manifest, observed, at_us, index)
            self.assertTrue(rows)
            self.assertEqual(all("THERMAL_LIMIT" in row.rejection_reasons for row in rows),
                             observed is hot or at_us == 7_000)
        events = [dict(row) for row in scheduler.thermal_deferral_events()]
        # many tickets and candidates, one row per onset; the older cool observation (1.5 ms)
        # does not close the 2 ms onset
        self.assertEqual([(row["kind"], row["at_us"]) for row in events], [
            ("THERMAL_DEFERRAL", 2_000), ("THERMAL_DEFERRAL_CLEARED", 5_000),
            ("THERMAL_DEFERRAL", 7_000)])
        self.assertEqual(events[0], {
            "at_us": 2_000, "device_id": HELPER, "executor_id": "executor:helper-c",
            "kind": "THERMAL_DEFERRAL", "maximum_temperature_millic": 90_000,
            "observed_temperature_millic": 95_000, "thermal_qualified": None})
        self.assertEqual((events[1]["onset_at_us"], events[1]["observed_temperature_millic"]),
                         (2_000, 60_000))
        # the platform verdict (thermal_qualified False) gates below the catalog limit
        self.assertEqual((events[2]["observed_temperature_millic"], events[2]["thermal_qualified"]),
                         (80_000, False))

    def test_no_row_without_a_thermal_limit(self):
        scheduler, manifest = self.scheduler_and_manifest()
        self.generate(scheduler, manifest, automated.runtime_snapshot(manifest), 1_000, 0)
        self.assertEqual(scheduler.thermal_deferral_events(), ())

    def test_result_carries_the_rows_only_under_elastic_phones(self):
        scheduler, manifest = self.scheduler_and_manifest()
        self.generate(scheduler, manifest, self.phone_at(automated.runtime_snapshot(manifest), 95_000),
                      2_000, 0)
        rig = SimpleNamespace(configuration=SimpleNamespace(elastic_phones=None))
        self.assertEqual(_elastic_drop_result(rig, [], scheduler), {})
        rig.configuration.elastic_phones = {"drop_recovery": True}
        exported = _elastic_drop_result(rig, [], scheduler)
        self.assertEqual(list(exported), ["thermal_deferral_events"])
        self.assertEqual([row["kind"] for row in exported["thermal_deferral_events"]],
                         ["THERMAL_DEFERRAL"])


if __name__ == "__main__":
    unittest.main()
