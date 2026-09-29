"""Decode-boundary re-evaluation of a re-provisioned phone layout (change #4 follow-up).

Scenario of the coherentRP rig run: model A decodes on an all-A phone while model B is
queued and not on the phone, with online-learning demand for both. The boundary hook's
dedup compares the recorded learning status of B (with a source route) against the
compiler's UNUSABLE status (without one), so on the base tree every decode boundary ran a
full portfolio evaluation and recorded PHONE_RESIDENCY_REPROVISION_RETAINED (1,202 events).

Only base-tree names are imported and the knob is duck-typed, so the base tree runs the
same scenarios and fails the rate-limit assertions."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research_dev.scheduler import UnifiedScheduler
from research_dev.scheduler._unified.phone_residency_ops.common import _OfflineLearningDemand
from research_dev.scheduler.tests import test_phone_reprovision_portfolio as portfolio
from research_dev.scheduler.tests.test_phone_reprovision_portfolio import (
    KNOB, MODEL_A, MODEL_B, SESSIONS, demand_row, desktop_load, layers, layout_with_counts, ticket,
)

A, B = MODEL_A.artifact_sha256, MODEL_B.artifact_sha256
TOKEN_US = 420_000
START_US = 100_000_000
RETAINED = "PHONE_RESIDENCY_REPROVISION_RETAINED"
UNUSABLE = {"reason": "PHONE_RESIDENCY_ROUTE_EVIDENCE_UNUSABLE"}


class BoundaryReevaluationTests(unittest.TestCase):
    def setUp(self):
        self.evaluations = []

    def scenario(self, *, knob=KNOB, a_transition=None):
        case = portfolio.PortfolioReprovisionTests()
        self.a1 = ticket("a-1", MODEL_A, state="ACQUIRED", output_tokens=500, transition=a_transition)
        self.b1 = ticket("b-1", MODEL_B, output_tokens=132)
        self.tickets = [self.a1, self.b1]
        scheduler, compiler = case.scheduler(self.tickets, knob=knob)
        compiler.phone_residency_demand = lambda model, work: None
        self.evidence = {A: UNUSABLE, B: UNUSABLE}
        compiler.phone_residency_evidence_status = lambda artifact: self.evidence.get(artifact, UNUSABLE)
        for model in (MODEL_A, MODEL_B):
            scheduler._online_learning_phone_demand_cache[model.artifact_sha256] = _OfflineLearningDemand(
                demand_row(model, 100), SESSIONS, "phone-a",
                {"reason": "PHONE_RESIDENCY_LEARNING_EXPLORATION_READY",
                 "source_route_id": "route:" + model.model_id},
            )
        case.install_ready(scheduler, layout_with_counts((demand_row(MODEL_A, 100),), {A: 3}))
        self.remaining = {"a-1": 500}
        scheduler._model_placement_controller.remaining_request_decode_tokens = (
            lambda request_id, output_tokens: self.remaining.get(request_id, output_tokens)
        )
        self.case, self.scheduler, self.compiler = case, scheduler, compiler
        return scheduler

    def patched(self):
        original = UnifiedScheduler._update_phone_residency_portfolio

        def counted(scheduler, request, manifest, observed_at_us, snapshot=None):
            self.evaluations.append(observed_at_us)
            return original(scheduler, request, manifest, observed_at_us, snapshot)

        compiler = patch.object(UnifiedScheduler, "_automated_compiler", return_value=self.compiler)
        return compiler, patch.object(UnifiedScheduler, "_update_phone_residency_portfolio", counted)

    def evaluated(self):
        return [row for row in self.scheduler.phone_residency_events() if row["kind"] == "EVALUATED"]

    def boundaries(self, count, start_us=START_US):
        """`count` decode boundaries of a-1 one token period apart; returns the next boundary time."""
        compiler, counted = self.patched()
        with compiler, counted:
            at_us = start_us
            for _ in range(count):
                self.remaining["a-1"] -= 1
                self.scheduler._reevaluate_pending_phone_layout_at_boundary(self.a1, at_us)
                at_us += TOKEN_US
        return at_us

    def test_unchanged_boundaries_are_rate_limited_and_not_recorded(self):
        self.scenario()
        before = len(self.evaluated())
        self.boundaries(60)  # 25.2 s of decode, nothing changes but the remaining count
        # base: 60 evaluations and 60 RETAINED records
        self.assertEqual(self.evaluations, [START_US, START_US + 24 * TOKEN_US, START_US + 48 * TOKEN_US])
        events = self.evaluated()[before:]
        self.assertEqual([row["reason"] for row in events], [RETAINED])
        self.assertEqual(events[0]["desktop_reprovision"]["mode"], "FOLLOW")
        gate = self.scheduler._phone_reprovision_boundary_gate
        self.assertEqual(gate.counts["boundary_evaluations_skipped"], 57)
        self.assertEqual(gate.counts["unchanged_decisions_coalesced"], 2)

    def test_state_change_reevaluates_at_the_next_boundary_with_the_counts(self):
        self.scenario()
        at_us = self.boundaries(10)
        self.assertEqual(len(self.evaluations), 1)
        self.tickets.append(ticket("b-2", MODEL_B, output_tokens=300))  # queue composition changes
        at_us = self.boundaries(5, at_us)
        self.assertEqual(len(self.evaluations), 2)
        latest = self.evaluated()[-1]
        self.assertEqual(latest["reason"], RETAINED)
        self.assertEqual(latest["desktop_reprovision"]["arrived_work_by_artifact"][B], 432)
        self.assertEqual(latest["desktop_reprovision"]["boundary_evaluations_skipped"], 9)
        self.assertEqual(latest["desktop_reprovision"]["boundary_evaluations_skipped_total"], 9)
        self.assertEqual(latest["desktop_reprovision"]["unchanged_decisions_coalesced"], 0)

    def test_each_state_change_triggers_one_reevaluation(self):
        def arrival(test):
            test.tickets.append(ticket("b-2", MODEL_B, output_tokens=40))

        def completion(test):
            test.tickets.pop()

        def dispatch(test):
            test.tickets[1] = SimpleNamespace(**{**vars(test.b1), "dispatch_state": "ACQUIRED"})

        def desktop_load_finished(test):
            test.a1.transition_status = "COMPLETED"

        def session_verified(test):
            placement = test.scheduler._model_placement_controller
            rows = placement.phone_session_states()
            placement.phone_session_states = lambda: (
                replace(rows[0], session_generation=rows[0].session_generation + 1), *rows[1:])

        def route_evidence(test):
            test.evidence[B] = {"reason": "PHONE_RESIDENCY_ROUTE_EVIDENCE_READY",
                                "source_route_id": "route:b", "normalized_benefit_uj": 5}

        for name, change in (("arrival", arrival), ("completion", completion), ("dispatch", dispatch),
                             ("desktop_load_finished", desktop_load_finished),
                             ("session_verified", session_verified), ("route_evidence", route_evidence)):
            with self.subTest(name):
                self.evaluations = []
                self.scenario(a_transition=desktop_load() if name == "desktop_load_finished" else None)
                if name == "completion":
                    self.tickets.append(ticket("b-2", MODEL_B, output_tokens=40))
                at_us = self.boundaries(5)
                self.assertEqual(len(self.evaluations), 1)
                change(self)
                at_us = self.boundaries(1, at_us)
                self.assertEqual(len(self.evaluations), 2)
                self.boundaries(5, at_us)
                self.assertEqual(len(self.evaluations), 2)

    def test_swap_at_dispatch_still_starts_the_first_stage(self):
        self.scenario()
        at_us = self.boundaries(30)
        self.assertEqual(len(self.evaluations), 2)
        # a-1 completes; the desktop dispatches b-1 with a model load
        leader = ticket("b-1", MODEL_B, state="ACQUIRED", transition=desktop_load(47_000_000),
                        dispatched_at=at_us, output_tokens=132)
        self.tickets[:] = [leader]
        compiler, counted = self.patched()
        with compiler, counted:
            self.scheduler._reevaluate_phone_layout_for_desktop_load(leader)
        self.assertEqual(self.evaluations[-1], at_us)
        event = self.evaluated()[-1]
        self.assertEqual(event["reason"], "PHONE_RESIDENCY_DESKTOP_REPROVISION")
        recorded = event["desktop_reprovision"]
        self.assertEqual((recorded["mode"], recorded["desktop_commitment_source"]), ("FOLLOW", "loading"))
        self.assertEqual(recorded["boundary_evaluations_skipped"], 28)
        self.assertEqual(recorded["unchanged_decisions_coalesced"], 1)
        self.assertTrue(event["selection_confirmed"])
        target = self.scheduler._model_placement_controller.target_phone_layout()
        self.assertEqual(target.state, "PROPOSED")
        self.assertEqual(len(target.layout.changed_session_ids), 1)
        self.assertEqual(layers(target.layout).get(B), 2)

    def test_zero_interval_evaluates_every_boundary_but_records_one_decision(self):
        self.scenario(knob=SimpleNamespace(**vars(KNOB), boundary_reevaluation_interval_us=0))
        before = len(self.evaluated())
        self.boundaries(20)
        self.assertEqual(len(self.evaluations), 20)
        self.assertEqual([row["reason"] for row in self.evaluated()[before:]], [RETAINED])

    def test_knob_off_boundaries_are_unchanged(self):
        # the base behaviour, including its per-boundary LEARNING_RETAINED records
        self.scenario(knob=None)
        before = len(self.evaluated())
        self.boundaries(20)
        self.assertEqual(len(self.evaluations), 20)
        events = self.evaluated()[before:]
        self.assertEqual({row["reason"] for row in events}, {"PHONE_RESIDENCY_LEARNING_RETAINED"})
        self.assertEqual(len(events), 20)
        self.assertFalse(any("desktop_reprovision" in row for row in events))
        self.assertIsNone(getattr(self.scheduler, "_phone_reprovision_boundary_gate", None))


if __name__ == "__main__":
    unittest.main()
