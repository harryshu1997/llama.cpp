#!/usr/bin/env python3
"""dispatch_policy.event_replanning: revisit waiting decisions when their cause changes.

Built on the s2a Gemma->Qwen handover (2026-09-29): the phone re-provisioning was
``DEFERRED_IN_USE`` (1,052.94 and 1,053.22 s) while the finishing Gemma request's helper held
every session; the request completion re-evaluated it (PROPOSED gen 10 at 1,052.93 s), but
the proposal was never prepared because every fresh OP15 route was ``THERMAL_LIMIT`` and the
preparation envelope returns None without any event (005/007 host-only for ~170 s).

The frozen code re-evaluates a deferral only at a request completion (and only while the
last record names blocked sessions or a desktop load is decided). These tests pin the general
trigger (a decode-completion detach, a cancellation), the no-op for a release that frees no
session, the visible thermal block with preparation at its clearing, and the recovery replans.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research_dev.scheduler import PhoneFfnShardStorageMetadata, UnifiedScheduler
from research_dev.scheduler._internal.runtime_dispatch_policy import (
    RuntimeDispatchPolicy,
    RuntimeDispatchPolicyError,
)
from research_dev.scheduler._internal.types import canonical_sha256
from research_dev.scheduler._unified.automated_requests_ops import event_replanning
from research_dev.scheduler.configuration import campaign as campaign_config

try:
    from . import test_phone_reprovision_portfolio as portfolio
    from . import test_offline_phone_residency as offline
except ImportError:
    import test_phone_reprovision_portfolio as portfolio
    import test_offline_phone_residency as offline

A, B = portfolio.A, portfolio.B
MODEL_A, MODEL_B = portfolio.MODEL_A, portfolio.MODEL_B
RELEASE_STAT = event_replanning.RELEASE_REEVALUATION_STAT
KNOB = SimpleNamespace(
    mode="resident-model", load_bytes_per_second=32, minimum_learned_samples=2,
    count_queued_demand=True, early_on_transition=True,
)


def _detached(binding):
    attachment = dict(binding["helper_attachment"])
    attachment.update(fraction_ppm=0, lease_tokens=[], lease_reserved_until_us=None,
                      fallback_outcome="REQUEST_COMPLETED")
    return {**binding, "fraction_ppm": 0, "helper_attachment": attachment}


class DispatchPolicyKeyTests(unittest.TestCase):
    def test_absent_key_serializes_unchanged_and_round_trips_when_set(self):
        legacy = RuntimeDispatchPolicy(work_conserving_admission=True, continuous_join=True)
        self.assertFalse(legacy.event_replanning)
        self.assertNotIn("event_replanning", legacy.to_json())
        enabled = replace(legacy, event_replanning=True)
        self.assertTrue(enabled.to_json()["event_replanning"])
        self.assertEqual(RuntimeDispatchPolicy.from_json(enabled.to_json()), enabled)
        self.assertEqual(RuntimeDispatchPolicy.from_json(legacy.to_json()), legacy)
        # independent of the ordering policies (it only revisits decisions earlier)
        self.assertFalse(RuntimeDispatchPolicy(event_replanning=True).enabled)
        with self.assertRaises(RuntimeDispatchPolicyError):
            RuntimeDispatchPolicy(event_replanning=1)

    def test_campaign_dispatch_policy_accepts_the_flag(self):
        checked = campaign_config._dispatch_policy({"event_replanning": True})
        self.assertEqual(dict(checked), {"event_replanning": True})
        paper = {"continuous_join": True, "max_barrier_extension_s": 120, "model_affinity": True,
                 "residency_hysteresis_s": 20, "work_conserving_admission": True}
        self.assertEqual(dict(campaign_config._dispatch_policy(paper)), paper)
        with self.assertRaises(ValueError):
            campaign_config._dispatch_policy({"event_replanning": "yes"})

    def test_statistics_are_reported_only_under_the_flag(self):
        for policy, expected in ((RuntimeDispatchPolicy(), False),
                                 (RuntimeDispatchPolicy(event_replanning=True), True)):
            scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
            scheduler.configure_runtime_dispatch_policy(policy)
            statistics = scheduler.runtime_dispatch_policy_state()["statistics"]
            self.assertEqual(RELEASE_STAT in statistics, expected)
            self.assertEqual(
                "event_replanning_preparation_blocks" in statistics, expected
            )


class S2aReleaseTests(unittest.TestCase):
    """Phone re-provisioning deferred IN_USE, then the blocking helper is released."""

    # the real-portfolio stand-in of test_phone_reprovision_portfolio (not its tests)
    scheduler = portfolio.PortfolioReprovisionTests.scheduler
    install_ready = portfolio.PortfolioReprovisionTests.install_ready
    load = staticmethod(portfolio.PortfolioReprovisionTests.load)
    evaluate = staticmethod(portfolio.PortfolioReprovisionTests.evaluate)
    target = staticmethod(portfolio.PortfolioReprovisionTests.target)

    t0 = 1_052_900_000  # the Qwen leader is dispatched with its desktop load; swap deferred
    t1 = 1_053_300_000  # the Gemma holder's helper detaches at its decode completion

    def s2a(self, *, event_replanning: bool):
        using = portfolio.ticket("a-002", MODEL_A, state="ACQUIRED", output_tokens=100)
        leader = portfolio.ticket(
            "b-005", MODEL_B, state="ACQUIRED", transition=portfolio.desktop_load(52_000_000),
            dispatched_at=self.t0, output_tokens=200,
        )
        tickets = [using, leader]
        scheduler, compiler = self.scheduler(tickets, knob=KNOB)
        scheduler._runtime_controller.current_tickets = lambda states=None: tuple(
            row for row in tickets if states is None or row.dispatch_state in states
        )
        scheduler._runtime_controller.dispatch_policy = RuntimeDispatchPolicy(
            event_replanning=event_replanning
        )
        scheduler._runtime_controller.record_dispatch_policy_event = (
            lambda name, count=1: self.stats.__setitem__(name, self.stats.get(name, 0) + count)
        )
        self.stats = {}
        self.install_ready(scheduler, portfolio.layout_with_counts(
            (portfolio.demand_row(MODEL_A, 100),), {A: 3}
        ))
        bindings = {"a-002": portfolio.helper_binding("HTP0", "HTP1", "HTP2")}
        scheduler._model_placement_controller.request_binding = bindings.get
        event = self.evaluate(scheduler, compiler, leader, MODEL_B, self.t0)
        self.assertEqual(event["reason"], "PHONE_RESIDENCY_REPROVISION_DEFERRED_IN_USE")
        self.assertEqual(event["desktop_reprovision"]["in_use_session_ids"], ["HTP0", "HTP1", "HTP2"])
        self.assertIsNone(self.target(scheduler))
        return scheduler, compiler, bindings, using

    def complete_decode(self, scheduler, compiler, bindings, using):
        """Drive the real ``complete_adaptive_decode``: the helper detaches at its last window.

        s2a: 002's fraction was already 0 (FRACTION_CHANGED 1,052.1 s), so no lease is left
        to release; the detach alone frees the sessions."""
        attachment = dict(bindings["a-002"]["helper_attachment"])
        attachment.update(fraction_ppm=0, lease_tokens=[])
        bindings["a-002"] = {**bindings["a-002"], "fraction_ppm": 0, "helper_attachment": attachment}
        # every token decoded: the request is still ACQUIRED but has no remaining work
        scheduler._model_placement_controller._request_decode_progress["a-002"] = (100, 100)

        def detach(request_id, *, fallback_outcome, observed_at_us):
            bindings[request_id] = _detached(bindings[request_id])

        scheduler._model_placement_controller.detach_request_helper = detach
        result = SimpleNamespace(windows=(SimpleNamespace(finished_at_us=self.t1),))
        with patch.object(UnifiedScheduler, "_automated_compiler", return_value=compiler), \
                patch.object(scheduler._adaptive_decode, "complete", return_value=result):
            scheduler.complete_adaptive_decode(using.request.request_id)

    def test_decode_completion_release_proposes_the_swap_at_release_time(self):
        scheduler, compiler, bindings, using = self.s2a(event_replanning=True)
        self.complete_decode(scheduler, compiler, bindings, using)
        target = self.target(scheduler)
        self.assertIsNotNone(target)
        self.assertEqual(target.state, "PROPOSED")
        self.assertEqual(target.proposed_at_us, self.t1)
        event = next(row for row in reversed(scheduler.phone_residency_events())
                     if row["kind"] == "EVALUATED")
        self.assertEqual(event["reason"], "PHONE_RESIDENCY_DESKTOP_REPROVISION")
        self.assertEqual(event["observed_at_us"], self.t1)
        self.assertEqual(event["desktop_reprovision"]["in_use_session_ids"], [])
        self.assertEqual(event["desktop_reprovision"]["leader_request_id"], "b-005")
        self.assertEqual(self.stats, {RELEASE_STAT: 1})
        # nothing blocks the proposed transition at the release time: the helper watchers
        # (woken by the layout event) begin the preparation now, not at the next arrival
        self.assertEqual(
            scheduler._model_placement_controller.phone_layout_transition_blockers(target.generation), ()
        )
        self.load(scheduler, target, self.t1, 8_000_000)
        self.assertEqual(scheduler._model_placement_controller.ready_phone_layout().generation,
                         target.generation)

    def test_without_the_flag_the_release_waits_for_another_event(self):
        scheduler, compiler, bindings, using = self.s2a(event_replanning=False)
        self.complete_decode(scheduler, compiler, bindings, using)
        self.assertIsNone(self.target(scheduler))
        self.assertEqual(self.stats, {})
        self.assertNotIn("_event_replanning_pending", scheduler.__dict__)

    def test_cancelled_holder_releases_the_sessions(self):
        # a cancellation/failure closes the helper runtime but never re-evaluated the layout
        scheduler, compiler, bindings, using = self.s2a(event_replanning=True)
        bindings.pop("a-002")
        scheduler._close_request_helper_runtime("a-002", self.t1, "REQUEST_CANCELLED")
        self.assertIsNone(self.target(scheduler))  # handled at the outermost call's return
        with patch.object(UnifiedScheduler, "_automated_compiler", return_value=compiler):
            event_replanning.process_pending_events(scheduler)
        target = self.target(scheduler)
        self.assertEqual((target.state, target.proposed_at_us), ("PROPOSED", self.t1))
        self.assertEqual(self.stats, {RELEASE_STAT: 1})

    def test_a_release_that_frees_nothing_reevaluates_nothing(self):
        # host policy: fraction 0 and leases released, but the helper may come back, so its
        # sessions stay in use (the transition-blocker mirror) and no evaluation is recorded
        scheduler, compiler, bindings, using = self.s2a(event_replanning=True)
        attachment = dict(bindings["a-002"]["helper_attachment"])
        attachment.update(fraction_ppm=0, lease_tokens=[])
        bindings["a-002"] = {**bindings["a-002"], "fraction_ppm": 0, "helper_attachment": attachment}
        events = len(scheduler.phone_residency_events())
        event_replanning.note_resource_release(
            scheduler, "a-002", self.t1, "HELPER_LEASES_RELEASED"
        )
        with patch.object(UnifiedScheduler, "_automated_compiler", return_value=compiler):
            event_replanning.process_pending_events(scheduler)
        self.assertEqual(len(scheduler.phone_residency_events()), events)
        self.assertIsNone(self.target(scheduler))
        self.assertEqual(self.stats, {})


def _qwen_runtime(event_replanning: bool):
    """The replay catalog (desktop + OP15, three HTP sessions) with Qwen shards stored."""
    offline.OfflinePhoneResidencyTests.setUpClass()
    scheduler, requests_by_model, snapshot = offline.OfflinePhoneResidencyTests.replay_runtime(
        load_evidence=False
    )
    if event_replanning:
        scheduler.configure_runtime_dispatch_policy(RuntimeDispatchPolicy(event_replanning=True))
    model_id, requests = next(iter(requests_by_model.items()))
    model = scheduler.runtime_model_manifest(model_id)
    scheduler.register_phone_ffn_shard_storage(tuple(
        PhoneFfnShardStorageMetadata(
            parent_artifact_sha256=model.artifact_sha256,
            shard_sha256=canonical_sha256([model_id, index]),
            path="/phone/HTP" + str(index) + ".gguf",
            layer_mask=63 << (6 * index),
            maximum_columns=model.feed_forward_length,
            session_id="HTP" + str(index),
        )
        for index in range(3)
    ))
    request = replace(requests[0], arrival_us=snapshot.captured_at_us,
                      deadline_us=snapshot.captured_at_us + 600_000_000)
    ticket = offline.OfflinePhoneResidencyTests.acquire_desktop_request(
        scheduler, request, model_id, snapshot, selection_mode="calibration"
    )
    phone = next(row.executor_id for row in scheduler._runtime_capabilities.executors
                 if row.phone_sessions)
    return scheduler, request, ticket, snapshot, phone


def _thermally_limited(snapshot, phone, at_us):
    retimed = offline.replay_fixture.ReplayDeterminismTests._retime_snapshot(
        snapshot, at_us, "thermal-" + str(at_us)
    )
    return replace(retimed, executors={
        **retimed.executors,
        phone: replace(retimed.executors[phone], thermal_qualified=False),
    })


def _helper_events(scheduler, request_id, kinds):
    return [row for row in scheduler._model_placement_controller.request_helper_events(request_id)
            if row["kind"] in kinds]


class BlockedPreparationTests(unittest.TestCase):
    """The proposed layout of a live request cannot be prepared while the phone is hot."""

    def test_thermal_block_is_recorded_and_preparation_starts_when_it_clears(self):
        scheduler, request, ticket, snapshot, phone = _qwen_runtime(True)
        target = scheduler._model_placement_controller.target_phone_layout()
        self.assertEqual(target.state, "PROPOSED")
        hot_at = snapshot.captured_at_us + 1_000
        for offset in (0, 500):  # the watcher polls; the block is recorded once
            helper = scheduler.runtime_request_helper_preparation_envelope(
                request.request_id, expected_ticket_id=ticket.ticket_id,
                observed_at_us=hot_at + offset,
                snapshot=_thermally_limited(snapshot, phone, hot_at + offset),
            )
            self.assertIsNone(helper)
        blocked = _helper_events(scheduler, request.request_id, {"PREPARATION_BLOCKED"})
        self.assertEqual(len(blocked), 1)
        self.assertEqual(blocked[0]["reason"], "NO_HELPER_OPPORTUNITY")
        self.assertIn("THERMAL_LIMIT", blocked[0]["candidate_rejection_reasons"])
        self.assertGreater(blocked[0]["layout_route_count"], 0)
        self.assertEqual(blocked[0]["phone_layout_generation"], target.generation)
        self.assertEqual(blocked[0]["blocked_since_us"], hot_at)
        cool_at = hot_at + 5_000
        cool = offline.replay_fixture.ReplayDeterminismTests._retime_snapshot(
            snapshot, cool_at, "cool"
        )
        helper = scheduler.runtime_request_helper_preparation_envelope(
            request.request_id, expected_ticket_id=ticket.ticket_id,
            observed_at_us=cool_at, snapshot=cool,
        )
        self.assertIsNotNone(helper)
        unblocked = _helper_events(scheduler, request.request_id, {"PREPARATION_UNBLOCKED"})
        self.assertEqual(len(unblocked), 1)
        self.assertEqual(unblocked[0]["blocked_us"], cool_at - hot_at)
        self.assertEqual(unblocked[0]["blocked_reason"], "NO_HELPER_OPPORTUNITY")
        decision = scheduler.begin_request_helper_preparation(
            request.request_id, observed_at_us=cool_at, snapshot=cool,
            expected_phone_layout_generation=helper.phone_layout_generation,
            expected_phone_layout_geometry_sha256=helper.phone_layout_geometry_sha256,
            expected_operator_plan_sha256=helper.operator_plan_sha256,
        )
        self.assertEqual(decision["status"], "OWNER")
        self.assertEqual(decision["started_at_us"], cool_at)
        statistics = scheduler.runtime_dispatch_policy_state()["statistics"]
        self.assertEqual(statistics["event_replanning_preparation_blocks"], 1)
        self.assertEqual(statistics["event_replanning_recovery_reevaluations"], 1)
        # the thermal gate's clearing was observed by the fresh compile and handled once
        clears = [row for row in scheduler.thermal_deferral_events()
                  if row["kind"] == "THERMAL_DEFERRAL_CLEARED"]
        self.assertEqual(len(clears), 1)
        # the running decode adopts the prepared helper once the layout is READY
        backend = offline._OfflinePhoneBackend(cool)
        command = offline.bind_ready_helper_to_physical_command(
            replace(offline.interpret_runtime_ticket(ticket), helper_envelope=None,
                    helper_transitions=()),
            helper,
        )
        receipts = tuple(
            offline.transition_receipt_from_observation(
                row, backend.apply_transition(row, object(), lambda: None)
            )
            for row in command.helper_transitions
        )
        ready_snapshot = backend.snapshot(SimpleNamespace(helper_envelope=helper), backend.clock_us)
        completed = scheduler.complete_request_helper_preparation(
            request.request_id, decision["preparation_ticket_id"], receipts,
            snapshot=ready_snapshot,
        )
        self.assertEqual(completed["state"], "READY")
        self.assertTrue(scheduler.refresh_ready_request_helper(
            request.request_id, expected_ticket_id=ticket.ticket_id,
            observed_at_us=ready_snapshot.captured_at_us, snapshot=ready_snapshot,
        ))
        self.assertIn(request.request_id, scheduler._late_request_helper_contexts)

    def test_without_the_flag_the_block_is_silent(self):
        scheduler, request, ticket, snapshot, phone = _qwen_runtime(False)
        hot_at = snapshot.captured_at_us + 1_000
        before = len(scheduler._model_placement_controller.request_helper_events(request.request_id))
        helper = scheduler.runtime_request_helper_preparation_envelope(
            request.request_id, expected_ticket_id=ticket.ticket_id,
            observed_at_us=hot_at, snapshot=_thermally_limited(snapshot, phone, hot_at),
        )
        self.assertIsNone(helper)
        self.assertEqual(
            len(scheduler._model_placement_controller.request_helper_events(request.request_id)),
            before,
        )
        self.assertNotIn("event_replanning_preparation_blocks",
                         scheduler.runtime_dispatch_policy_state()["statistics"])


class DeviceRecoveryTests(unittest.TestCase):
    """A phone that is admissible again replans the attempts decided while it was out."""

    def recovery_scheduler(self, estimates_device):
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        catalog = SimpleNamespace(
            executors=(SimpleNamespace(device_id="op15-phone", phone_sessions=()),),
            executor_by_id={"physical:op15-phone": SimpleNamespace(device_id="op15-phone")},
            composite_executor_by_id={
                "physical:hot:phone-assisted": SimpleNamespace(
                    participant_device_ids=("desktop-cpu", estimates_device)
                ),
            },
        )
        scheduler._runtime_capabilities = catalog

        def queued(request_id, decided_at_us, uses_phone=False):
            return SimpleNamespace(
                request=SimpleNamespace(request_id=request_id),
                dispatch_state="QUEUED",
                runtime_observation=SimpleNamespace(captured_at_us=decided_at_us),
                execution_plan=SimpleNamespace(
                    device_ids=("desktop-cpu", "op15-phone") if uses_phone else ("desktop-cpu",)
                ),
                cost_estimates=SimpleNamespace(estimates=(
                    SimpleNamespace(executor_id="physical:hot:desktop"),
                    SimpleNamespace(executor_id="physical:hot:phone-assisted"),
                )),
            )

        tickets = (
            queued("q-before", 100),       # decided before the outage
            queued("q-during", 300),       # decided while the phone was out
            queued("q-uses", 350, True),   # its plan already uses the phone
            queued("q-late", 900),         # decided after the recovery time
        )
        calls = []
        self.stats = {}
        scheduler._runtime_controller = SimpleNamespace(
            dispatch_policy=RuntimeDispatchPolicy(event_replanning=True),
            current_tickets=lambda states=None: tickets,
            dispatch_order_view=lambda: {row.request.request_id: {"sequence": index}
                                         for index, row in enumerate(tickets)},
            replan_queued_now=lambda ids, reason, at_us, **_: calls.append((ids, reason, at_us)) or ids,
            record_dispatch_policy_event=lambda name, count=1: self.stats.__setitem__(
                name, self.stats.get(name, 0) + count),
            checkpoint=lambda: None,
            restore=lambda _value: None,
        )
        return scheduler, calls

    def test_attempts_decided_during_the_outage_are_replanned(self):
        scheduler, calls = self.recovery_scheduler("op15-phone")
        event_replanning.note_device_admissible(scheduler, "op15-phone", 500, "DEVICE_READMITTED", 200)
        event_replanning.process_pending_events(scheduler)
        self.assertEqual(calls, [(("q-during",), event_replanning.DEVICE_ADMISSIBLE_REASON, 500)])
        self.assertEqual(self.stats, {event_replanning.RECOVERY_REPLAN_STAT: 1})

    def test_attempts_without_a_route_through_the_device_are_kept(self):
        scheduler, calls = self.recovery_scheduler("pixel-phone")
        event_replanning.note_device_admissible(scheduler, "op15-phone", 500, "DEVICE_READMITTED", 200)
        event_replanning.process_pending_events(scheduler)
        self.assertEqual(calls, [])
        self.assertEqual(self.stats, {})

    def test_readmission_notes_the_recovery_only_under_the_flag(self):
        for enabled in (False, True):
            scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
            scheduler._runtime_controller.dispatch_policy = RuntimeDispatchPolicy(
                event_replanning=enabled
            )
            noted = []
            with patch.object(event_replanning, "_recover_devices",
                              side_effect=lambda _c, rows: noted.extend(rows)), \
                 patch.object(scheduler._adaptive_decode, "readmit_device", return_value=True), \
                 patch.object(scheduler._runtime_controller, "readmit_device", return_value=True), \
                 patch.object(type(scheduler), "_membership_device", lambda _self, value: value):
                scheduler._membership_events().append(
                    {"at_us": 200, "device_id": "op15-phone", "kind": "DEVICE_QUARANTINED"}
                )
                scheduler.readmit_device("op15-phone", at_us=500,
                                         identity_sha256="sha256:" + "a" * 64)
            self.assertEqual(
                [(row["device_id"], row["since_us"], row["source"]) for row in noted],
                [("op15-phone", 200, "DEVICE_READMITTED")] if enabled else [],
            )

    def test_telemetry_recovery_notes_every_phone_only_under_the_flag(self):
        from research_dev.scheduler._unified.phone_residency_ops.fixed import (
            _phone_telemetry_deferral,
        )

        for enabled in (False, True):
            scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
            scheduler._runtime_controller.dispatch_policy = RuntimeDispatchPolicy(
                event_replanning=enabled
            )
            scheduler._runtime_capabilities = SimpleNamespace(executors=(
                SimpleNamespace(device_id="desktop-cpu", phone_sessions=()),
                SimpleNamespace(device_id="op15-phone", phone_sessions=("HTP0",)),
            ))
            scheduler._phone_telemetry_deferrals["r-1"] = 100
            snapshot = SimpleNamespace(
                telemetry_observations={"op15-phone": {}}, snapshot_id="snap",
                telemetry_unavailable_reason=lambda device_id, at_us: None,
            )
            self.assertIsNone(_phone_telemetry_deferral(scheduler, snapshot, 400, "r-1"))
            self.assertEqual(
                [(row["device_id"], row["since_us"], row["source"])
                 for row in scheduler.__dict__.get("_event_replanning_pending", ())],
                [("op15-phone", 100, "PHONE_TELEMETRY_RECOVERED")] if enabled else [],
            )


class ThermalExportTests(unittest.TestCase):
    def test_result_carries_the_thermal_gate_rows_under_the_flag(self):
        from research_dev.scheduler.campaigns.burstgpt.runner import _elastic_drop_result

        rig = SimpleNamespace(configuration=SimpleNamespace(elastic_phones=None))
        row = {"at_us": 1, "device_id": "op15-phone", "kind": "THERMAL_DEFERRAL"}
        for enabled, expected in ((False, {}), (True, {"thermal_deferral_events": [row]})):
            scheduler = SimpleNamespace(
                thermal_deferral_events=lambda: (row,),
                runtime_dispatch_policy_state=lambda enabled=enabled: {
                    "policy": RuntimeDispatchPolicy(event_replanning=enabled).to_json()
                },
            )
            self.assertEqual(_elastic_drop_result(rig, [], scheduler), expected)


if __name__ == "__main__":
    unittest.main()
