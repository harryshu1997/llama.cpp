"""Incumbent continuity without relaxing helper execution authorization."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from research_dev.scheduler._internal.adaptive_decode import AdaptiveDecodeController
from research_dev.scheduler.tests import test_adaptive_decode as fixtures
from research_dev.scheduler._unified.adaptive_decode_control import AdaptiveDecodeControlMixin
from research_dev.scheduler._unified.helper_envelopes_ops.refresh import retained_execution_layer_mask
from research_dev.scheduler._internal.runtime_plan import RuntimeExecutionContract, RuntimePhoneShard


class SustainedAssistanceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.AdaptiveDecodeControllerTests()
        self.fixture.setUp()
        self.baseline = self.fixture.baseline
        self.full = fixtures.policy("phone-full", 1000, (2, 3))
        self.quarter = fixtures.policy("phone-quarter", 250, (2, 3))
        self.config = replace(self.fixture.config, maximum_probe_tokens=80)
        self.controller = AdaptiveDecodeController()
        self.token, self.at_us = 1, 1000

    def start(self, **overrides):
        args = dict(
            request_id="request-a", ticket_id="request-a:attempt:0",
            model_artifact_sha256=fixtures.ARTIFACT, planning_profile_sha256=fixtures.PLAN,
            baseline=self.baseline, candidates=(self.full, self.quarter), output_tokens=200,
            context_length=64, active_batch=1, deadline_us=1_000_000,
            slot_id=7, first_token_index=self.token, first_token_at_us=self.at_us,
            config=self.config, helper_evidence_state="LEARNING",
        )
        args.update(overrides)
        self.directive = self.controller.start(**args)

    def ack(self):
        self.at_us += 1
        self.directive = self.fixture.acknowledge(
            self.controller, self.directive.control, self.token, self.at_us)

    def window(self, energy, duration=2000, **observation):
        if self.directive.control is not None:
            self.ack()
        active = self.controller.active_policy("request-a")
        self.token = self.directive.target_token_index
        self.at_us += duration
        self.directive = self.fixture.record(
            self.controller, self.token, self.at_us, energy,
            completed_phone_calls=0 if active.baseline else 2, **observation)
        return self.directive

    def test_incumbent_challenger_and_acknowledged_policy_are_distinct(self):
        self.start()
        self.window(100)
        self.window(40, 1000)
        state = self.controller.snapshot("request-a")
        self.assertEqual(state["incumbent_policy_hash"], self.full.policy_hash)
        self.assertEqual(state["challenger_policy_hash"], self.quarter.policy_hash)
        self.assertEqual(state["acknowledged_policy_hash"], self.full.policy_hash)
        self.ack()
        state = self.controller.snapshot("request-a")
        self.assertEqual(state["incumbent_policy_hash"], self.full.policy_hash)
        self.assertEqual(state["acknowledged_policy_hash"], self.quarter.policy_hash)
        self.window(120, 1000)
        self.assertEqual(self.directive.control.policy, self.full)
        self.assertEqual(self.controller.snapshot("request-a")["incumbent_policy_hash"],
                         self.full.policy_hash)

    def test_promising_candidate_reserves_four_comparable_windows_not_one(self):
        self.start(candidates=(self.full,), config=replace(self.config, uncertainty_ppm=100_000))
        self.window(100)
        self.window(84, 1800)
        state = self.controller.snapshot("request-a")
        plan = state["qualification_measurement_plan"]
        self.assertEqual(plan["counts_before"], (1, 1))
        self.assertEqual(plan["target_counts"], (4, 4))
        self.assertEqual(plan["normal_window_tokens"], (2, 2))
        self.assertEqual(plan["status"], "RESERVED")
        self.assertGreaterEqual(state["probe_budget"]["required_tokens"], 12)
        for _ in range(6):
            active = self.directive.control.policy if self.directive.control else self.controller.active_policy("request-a")
            self.window(100 if active.baseline else 84, 2000 if active.baseline else 1800)
        self.assertEqual(self.controller.snapshot("request-a")["incumbent_policy_hash"], self.full.policy_hash)

    def test_one_more_window_without_useful_bound_is_inconclusive_not_rejected(self):
        self.start(candidates=(self.full,), config=replace(self.config, uncertainty_ppm=100_000,
                                                          maximum_probe_tokens=12))
        self.window(100); self.window(84, 1800)
        state = self.controller.snapshot("request-a")
        self.assertEqual(state["qualification_measurement_plan"]["target_counts"], (4, 4))
        self.assertEqual(state["verification"]["outcome"], "INCONCLUSIVE")
        self.assertEqual(state["zero_assistance_reason"], "INCONCLUSIVE")
        self.assertNotIn(self.full.policy_hash, state["eliminated_policy_reasons"])
        self.assertTrue(self.directive.control.policy.baseline)

    def test_unaffordable_next_coarse_candidate_still_verifies_promising_leader(self):
        slow = replace(self.quarter, predicted_latency_per_token_us=1_000_000)
        self.start(candidates=(self.full, slow), config=replace(self.config, uncertainty_ppm=100_000))
        self.window(100)
        self.window(84, 1800)
        state = self.controller.snapshot("request-a")
        self.assertEqual(state["qualification_measurement_plan"]["target_counts"], (4, 4))
        self.assertEqual(state["qualification_measurement_plan"]["status"], "RESERVED")
        self.assertNotIn(slow.policy_hash, state["eliminated_policy_reasons"])
        for _ in range(6):
            active = self.directive.control.policy if self.directive.control else self.controller.active_policy("request-a")
            self.window(100 if active.baseline else 84, 2000 if active.baseline else 1800)
        self.assertEqual(self.controller.snapshot("request-a")["incumbent_policy_hash"], self.full.policy_hash)

    def test_slow_ack_defers_incomplete_pair_before_opening_measurement(self):
        self.start(candidates=(self.full,))
        self.window(100)
        budget = self.controller.snapshot("request-a")["probe_budget"]
        self.at_us = budget["deadline_us"] - 501
        self.ack()
        state = self.controller.snapshot("request-a")
        self.assertEqual(self.directive.reason, "PROBE_INCOMPLETE")
        self.assertTrue(self.directive.control.policy.baseline)
        self.assertNotIn(self.full.policy_hash, state["eliminated_policy_reasons"])
        self.assertEqual(sum(state["probe_attempts"].values()), 1)
        session = self.controller.checkpoint()[1]["request-a"]
        self.assertGreater(session.observed_control_cost_us, 1000)
        self.assertIsNone(self.controller._measurement_pair_budget(session, self.full, self.token, self.at_us))

    def test_observed_warmup_cost_is_budgeted_not_qualified_and_checkpoint_is_owned(self):
        self.start(candidates=(self.full,), config=replace(self.config, warmup_windows_per_policy=1))
        self.window(100, 12_000)
        saved = self.controller.checkpoint()
        session = saved[1]["request-a"]
        self.assertFalse(session.records[-1].measurement_eligible)
        self.assertEqual(session.warmup_latency_us_by_policy[self.baseline.policy_hash], 6000)
        session.warmup_windows_seen_by_policy.clear()
        with_cost = self.controller._measurement_pair_budget(session, self.full, self.token, self.at_us)
        session.warmup_latency_us_by_policy.clear()
        without_cost = self.controller._measurement_pair_budget(session, self.full, self.token, self.at_us)
        self.assertGreater(with_cost["required_us"], without_cost["required_us"])
        self.assertEqual(self.controller.snapshot("request-a")["warmup_latency_us_by_policy"],
                         {self.baseline.policy_hash: 6000})

    def test_unknown_membership_holds_desktop_and_preserves_incumbent_and_attempts(self):
        self.start(candidates=(self.full,))
        self.window(100)
        self.window(40, 1000)
        before = self.controller.snapshot("request-a")
        self.window(40, 1000, execution_context_available=False)
        after = self.controller.snapshot("request-a")
        self.assertTrue(self.directive.control.policy.baseline)
        self.assertEqual(after["zero_assistance_reason"], "EXECUTION_CONTEXT_UNAVAILABLE")
        self.assertEqual(after["incumbent_policy_hash"], before["incumbent_policy_hash"])
        self.assertEqual(after["context_identity_sha256"], before["context_identity_sha256"])
        self.assertEqual(after["probe_attempts"], before["probe_attempts"])
        self.assertEqual(after["probe_tokens"], before["probe_tokens"])
        self.window(100, execution_context_available=False)
        self.window(100)
        self.assertEqual(self.directive.control.policy, self.full)
        records = self.controller.checkpoint()[1]["request-a"].records
        self.assertTrue(all(not row.measurement_eligible for row in records[-3:]))
        self.assertTrue(all(row.whole_fleet_energy_uj > 0 for row in records[-3:]))
        for row in records[-3:]:
            self.assertEqual(type(row).from_json(row.to_json()), row)
        self.ack()
        recovered = self.controller.snapshot("request-a")
        self.assertEqual(recovered["state"], "EXPLOITING")
        self.assertEqual(recovered["probe_attempts"], before["probe_attempts"])

    def test_unknown_start_cannot_emit_cached_or_fresh_positive_control(self):
        self.start(candidates=(self.full,), execution_context_available=False)
        self.assertIsNone(self.directive.control)
        self.assertTrue(self.controller.active_policy("request-a").baseline)
        self.assertEqual(self.controller.helper_attachment_opportunity(
            "request-a", token_index=self.token, at_us=self.at_us), "EXECUTION_CONTEXT_UNAVAILABLE")
        self.window(100)
        self.window(100)
        self.assertEqual(self.directive.control.policy, self.full)

    def test_resolution_block_cannot_hide_observed_latency_violation(self):
        self.start(candidates=(self.full,), config=replace(self.config, uncertainty_ppm=100_000))
        self.window(100); self.window(84, 1800)
        for _ in range(3):
            self.window(100)
        self.window(84, 20_000)
        state = self.controller.snapshot("request-a")
        self.assertIn(self.full.policy_hash, state["eliminated_policy_reasons"])
        self.assertTrue(self.directive.control.policy.baseline)

    def test_sunk_challenger_cost_does_not_discard_future_saving(self):
        self.start()
        self.window(100)
        self.window(40, 1000)
        self.window(1_000_000, 1000)
        self.assertEqual(self.directive.control.policy, self.full)
        session = self.controller.checkpoint()[1]["request-a"]
        spent = sum(row.whole_fleet_energy_uj for row in session.records)
        self.assertGreater(spent, 2_000_000)
        self.assertTrue(self.controller._qualifies(session, self.full, self.token, self.at_us))
        self.assertIsNone(self.controller._measurement_pair_budget(
            session, self.quarter, self.token, self.at_us))
        self.ack()
        self.assertEqual(sum(row.whole_fleet_energy_uj for row in
                             self.controller.checkpoint()[1]["request-a"].records), spent)

    def test_learning_uses_configured_latency_allowance(self):
        for limit, accepted in ((1_250_000, True), (1_000_000, False)):
            with self.subTest(limit=limit):
                self.setUp()
                self.start(candidates=(self.full,), config=replace(
                    self.config, maximum_latency_ppm=limit))
                self.window(100)
                self.window(40, 2100)
                state = self.controller.snapshot("request-a")
                self.assertEqual(state["incumbent_policy_hash"] == self.full.policy_hash,
                                 accepted)
                if not accepted:
                    self.assertTrue(self.directive.control.policy.baseline)

    def test_incomplete_challenger_preserves_incumbent(self):
        self.start(config=replace(self.config, warmup_windows_per_policy=1))
        for energy, duration in ((100, 2000), (100, 2000), (40, 1000), (40, 1000)):
            self.window(energy, duration)
        self.assertEqual(self.directive.control.policy, self.quarter)
        self.window(80, 2_000_000)
        self.assertEqual(self.directive.reason, "PROBE_INCOMPLETE")
        self.assertEqual(self.directive.control.policy, self.full)
        self.assertNotIn(self.quarter.policy_hash,
                         self.controller.snapshot("request-a")["eliminated_policy_reasons"])

    def test_context_change_invalidates_incumbent_not_request_budget(self):
        self.start(candidates=(self.full,), active_batch=2)
        self.window(100, active_batch=2)
        self.window(40, 1000, active_batch=2)
        before = self.controller.snapshot("request-a")
        self.assertEqual(before["incumbent_policy_hash"], self.full.policy_hash)
        self.window(40, 1000, active_batch=2, next_active_batch=1, membership_changed=True)
        after = self.controller.snapshot("request-a")
        self.assertIsNone(after["incumbent_policy_hash"])
        self.assertNotEqual(after["context_identity_sha256"], before["context_identity_sha256"])
        self.assertGreaterEqual(after["probe_tokens"], before["probe_tokens"])
        self.assertTrue(self.directive.control.policy.baseline)

    def test_complete_pair_budget_includes_energy_and_all_controls(self):
        self.start()
        session = self.controller.checkpoint()[1]["request-a"]
        budget = self.controller._measurement_pair_budget(session, self.full, 1, 1000)
        self.assertGreaterEqual(budget["required_energy_uj"],
                                3 * self.config.transition_energy_uj)
        self.assertLessEqual(budget["required_energy_uj"], budget["energy_allowance_uj"])
        self.assertGreaterEqual(budget["required_tokens"], 7)

    def test_unaffordable_energy_pair_defers_without_measured_rejection(self):
        self.start(config=replace(self.config, transition_energy_uj=100_000))
        self.assertIsNone(self.directive.control)
        self.assertEqual(self.controller.helper_attachment_opportunity(
            "request-a", token_index=1, at_us=1000), "INSUFFICIENT_OPPORTUNITY")
        self.assertFalse(self.controller.snapshot("request-a")["eliminated_policy_reasons"])

    def test_unknown_probe_energy_is_not_fabricated(self):
        self.start(baseline=replace(self.baseline, predicted_energy_per_token_uj=None))
        self.assertIsNone(self.directive.control)
        session = self.controller.checkpoint()[1]["request-a"]
        self.assertIsNone(self.controller._measurement_pair_budget(session, self.full, 1, 1000))
        self.assertFalse(session.eliminated_policy_reasons)

    def test_unknown_candidate_energy_is_only_a_bounded_admission_prior(self):
        candidate = replace(self.full, predicted_energy_per_token_uj=None)
        self.start(candidates=(candidate,))
        session = self.controller.checkpoint()[1]["request-a"]
        budget = self.controller._measurement_pair_budget(session, candidate, 1, 1000)
        self.assertEqual(budget["candidate_energy_prior"], 1)
        self.assertGreater(budget["required_energy_uj"], 0)
        self.assertFalse(self.controller._qualifies(session, candidate, 1, 1000))
        self.assertIsNone(session.incumbent_policy)

    def test_fresh_incomplete_evidence_retries_are_bounded(self):
        self.start(candidates=(self.full,))
        for _ in range(18):
            self.window(100, attribution_kind="diagnostic")
        state = self.controller.snapshot("request-a")
        self.assertEqual(sum(state["probe_attempts"].values()), 2)
        self.assertFalse(state["eliminated_policy_reasons"])
        self.assertTrue(self.controller.active_policy("request-a").baseline)
        self.assertLessEqual(state["probe_tokens"], self.config.maximum_probe_tokens)

    def test_early_completion_does_not_issue_another_probe(self):
        self.start(output_tokens=12, candidates=(self.full,))
        while self.token < 12:
            self.window(100)
        self.assertIsNone(self.directive.control)
        self.assertEqual(self.directive.reason, "TERMINAL_WINDOW_RECORDED")
        group = self.controller.complete("request-a", "COMPLETED")
        self.assertEqual(group.terminal_status, "COMPLETED")

    def test_physical_loss_cannot_reapply_incumbent(self):
        self.start(candidates=(self.full,))
        self.window(100)
        self.window(40, 1000)
        self.controller.helper_unavailable("request-a")
        self.window(40, 1000)
        self.assertTrue(self.directive.control.policy.baseline)
        self.assertIsNone(self.controller.snapshot("request-a")["incumbent_policy_hash"])
        self.assertEqual(self.controller.snapshot("request-a")["zero_assistance_reason"],
                         "PHONE_HELPER_UNAVAILABLE")

    def test_checkpoint_owns_attempt_counters(self):
        self.start()
        self.window(100)
        saved = self.controller.checkpoint()
        self.ack()
        self.assertFalse(saved[1]["request-a"].probe_attempts)
        self.assertTrue(self.controller.snapshot("request-a")["probe_attempts"])

    def test_context_refresh_cannot_replenish_spent_exploration_energy(self):
        self.start(candidates=(self.full,))
        self.window(100)
        self.window(1000, 1000)
        before = self.controller.snapshot("request-a")["estimated_exploration_overhead_uj"]
        self.window(10000, active_batch=1, next_active_batch=2, membership_changed=True)
        after = self.controller.snapshot("request-a")["estimated_exploration_overhead_uj"]
        self.assertGreater(before, 0)
        self.assertGreaterEqual(after, before)

    def test_zero_fraction_event_contains_context_budget_and_evidence(self):
        self.start(config=replace(self.config, transition_energy_uj=100_000))
        record = Mock()
        scheduler = SimpleNamespace(
            _adaptive_decode=self.controller,
            _model_placement_controller=SimpleNamespace(record_request_helper_event=record))
        AdaptiveDecodeControlMixin._record_assistance_decision(
            scheduler, "request-a", self.directive, self.token, self.at_us)
        self.assertEqual(record.call_args.args[:3], ("request-a", "ASSISTANCE_DECISION", 1000))
        event = record.call_args.args[3]
        self.assertEqual(event["selected_fraction_ppm"], 0)
        self.assertEqual(event["reason"], "INSUFFICIENT_OPPORTUNITY")
        for key in ("incumbent_policy_hash", "challenger_policy_hash", "context_identity_sha256",
                    "remaining_probe_tokens", "remaining_output_tokens", "evidence"):
            self.assertIn(key, event)

    def rebound(self, candidates, mask, *, source_plan=fixtures.PLAN, generation=4):
        self.controller.helper_rebound(
            "request-a", phone_layout_generation=generation,
            phone_layout_geometry_sha256="sha256:" + "4" * 64,
            component_capability_sha256="sha256:" + "5" * 64,
            candidates=candidates, ticket_policy=None,
            helper_evidence_state="LEARNING",
            compatible_layers_by_plan={fixtures.PLAN: mask, source_plan: mask},
        )

    def refreshed(self, policies):
        return tuple(replace(row, route_id=row.route_id + "-refreshed",
                             operator_plan_sha256="sha256:" + "6" * 64,
                             predicted_energy_per_token_uj=65) for row in policies)

    def test_bookkeeping_rebind_keeps_incumbent_rejection_and_request_budget(self):
        self.start()
        self.window(100)
        self.window(40, 1000)
        self.window(120, 1000)
        self.ack()
        before = self.controller.checkpoint()[1]["request-a"]
        new = self.refreshed((self.full, self.quarter))
        self.rebound(new, self.full.layer_mask)
        after = self.controller.checkpoint()[1]["request-a"]
        self.assertEqual(after.incumbent_policy, new[0])
        self.assertEqual(after.records, before.records)
        self.assertEqual(after.probe_tokens, before.probe_tokens)
        self.assertEqual(after.probe_attempts, before.probe_attempts)
        self.assertEqual(after.eliminated_policy_reasons[new[1].policy_hash],
                         before.eliminated_policy_reasons[self.quarter.policy_hash])
        self.assertEqual(self.controller._context_identity(after),
                         self.controller._context_identity(before))
        self.assertEqual(self.controller._probe_attempt_key(after, new[0]),
                         self.controller._probe_attempt_key(before, self.full))
        self.assertEqual(self.controller.active_policy("request-a"), self.full)
        self.window(40, 1000)
        self.assertEqual(self.directive.control.policy, new[0])
        self.ack()
        self.assertEqual(self.controller.active_policy("request-a"), new[0])
        snapshot = self.controller.snapshot("request-a")
        self.rebound(new, self.full.layer_mask)
        self.assertEqual(snapshot, self.controller.snapshot("request-a"))

    def pending_winner_refresh(self, **start):
        self.start(candidates=(self.full,), **start)
        self.window(100)
        self.window(40, 1000)
        new = self.refreshed((self.full,))
        self.rebound(new, self.full.layer_mask)
        return new[0]

    def test_successive_unsent_refreshes_coalesce_only_exact_compatible_policy(self):
        first = self.pending_winner_refresh()
        before = self.controller.checkpoint()[1]["request-a"]
        second = replace(first, route_id="phone-second-refresh",
                         operator_plan_sha256="sha256:" + "7" * 64)
        self.rebound((second,), self.full.layer_mask,
                     source_plan=first.operator_plan_sha256, generation=5)
        after = self.controller.checkpoint()[1]["request-a"]
        self.assertEqual(after.candidates, (second,))
        self.assertEqual(after.pending_helper_refresh_policy, second)
        self.assertEqual(after.incumbent_policy, second)
        self.assertEqual(after.current_policy, before.current_policy)
        self.assertEqual(after.current_ack, before.current_ack)
        self.assertEqual(after.records, before.records)
        self.assertEqual(after.probe_attempts, before.probe_attempts)
        self.assertEqual(after.probe_tokens, before.probe_tokens)
        self.rebound((second,), self.full.layer_mask,
                     source_plan=first.operator_plan_sha256, generation=5)
        self.assertEqual(self.controller.checkpoint()[1]["request-a"], after)
        self.window(40, 1000)
        self.assertEqual(self.directive.control.policy, second)
        self.ack()
        self.assertEqual(self.controller.active_policy("request-a"), second)

    def test_successive_refresh_preserves_sent_control_until_its_exact_ack(self):
        first = self.pending_winner_refresh()
        self.window(40, 1000)
        sent = self.directive.control
        before = self.controller.checkpoint()[1]["request-a"]
        second = replace(first, route_id="phone-second-refresh",
                         operator_plan_sha256="sha256:" + "7" * 64)
        self.rebound((second,), self.full.layer_mask,
                     source_plan=first.operator_plan_sha256, generation=5)
        after = self.controller.checkpoint()[1]["request-a"]
        self.assertEqual(after.awaiting_control, sent)
        self.assertEqual(after.transition_policy, before.transition_policy)
        self.assertEqual(after.transition_ack, before.transition_ack)
        self.assertEqual(after.pending_helper_refresh_policy, second)
        self.ack()
        self.assertEqual(self.directive.control.policy, second)
        self.assertEqual(self.controller.snapshot("request-a")["acknowledged_policy_hash"],
                         first.policy_hash)
        self.ack()
        self.assertEqual(self.controller.active_policy("request-a"), second)

    def test_incompatible_pending_refresh_leaves_complete_checkpoint_unchanged(self):
        first = self.pending_winner_refresh()
        before = self.controller.checkpoint()
        second = replace(first, route_id="phone-unverified-refresh",
                         operator_plan_sha256="sha256:" + "7" * 64)
        with self.assertRaisesRegex(fixtures.AdaptiveDecodeError, "rebound policy changed"):
            self.rebound((second,), 0, generation=5)
        self.assertEqual(self.controller.checkpoint(), before)
        self.window(40, 1000)
        self.assertEqual(self.directive.control.policy, first)
        self.ack()
        self.rebound((second,), 0, generation=5)
        self.assertEqual(self.controller.checkpoint()[1]["request-a"].candidates, (second,))

    def test_refresh_selection_failure_does_not_publish_partial_state(self):
        self.start(candidates=(self.full,))
        self.window(100)
        self.window(40, 1000)
        before = self.controller.checkpoint()
        with patch.object(self.controller, "_best_valid_policy",
                          side_effect=fixtures.AdaptiveDecodeError("injected selection failure")):
            with self.assertRaisesRegex(fixtures.AdaptiveDecodeError, "injected selection failure"):
                self.rebound(self.refreshed((self.full,)), self.full.layer_mask)
        self.assertEqual(self.controller.checkpoint(), before)

    def test_refresh_cannot_replace_an_unacknowledged_maintenance_mask(self):
        self.start(candidates=(self.full,))
        self.window(100)
        self.window(40, 1000)
        drain = self.controller.request_helper_session_drain(
            "request-a", retained_layer_mask=1 << self.full.layer_indices[0])
        before = self.controller.checkpoint()
        with self.assertRaisesRegex(fixtures.AdaptiveDecodeError, "rebound policy changed"):
            self.rebound(self.refreshed((self.full,)), self.full.layer_mask)
        self.assertEqual(self.controller.checkpoint(), before)
        self.window(40, 1000)
        self.assertEqual(self.directive.control.policy.policy_hash, drain["policy_hash"])
        self.ack()
        self.assertEqual(self.controller.active_policy("request-a").layer_mask,
                         drain["retained_layer_mask"])

    def test_batch_change_cancels_refreshed_winner_and_admits_new_probe(self):
        new = self.pending_winner_refresh(active_batch=2)
        before = self.controller.checkpoint()[1]["request-a"]
        self.window(40, 1000, active_batch=2, next_active_batch=1, membership_changed=True)
        self.assertTrue(self.directive.control.policy.baseline)
        changed = self.controller.checkpoint()[1]["request-a"]
        self.assertIsNone(changed.pending_session_drain_policy)
        self.assertIsNone(changed.pending_helper_refresh_policy)
        self.assertIsNone(changed.incumbent_policy)
        self.assertIsNone(changed.probe_budget)
        self.assertFalse(self.controller._current_valid_records(changed, new))
        self.assertFalse(changed.records[-1].measurement_eligible)
        self.assertEqual(changed.probe_attempts, before.probe_attempts)
        self.assertGreaterEqual(changed.probe_tokens, before.probe_tokens)
        self.ack()
        self.assertIsNone(self.directive.control)
        self.assertTrue(self.controller.active_policy("request-a").baseline)
        self.window(100)
        self.assertEqual(self.directive.control.policy, new)
        admitted = self.controller.checkpoint()[1]["request-a"]
        self.assertEqual(admitted.state, "PROBING")
        self.assertIsNotNone(admitted.probe_budget)
        self.assertEqual(admitted.probe_budget["started_at_us"], self.at_us)
        self.assertIsNone(admitted.incumbent_policy)
        self.window(40, 1000)
        self.assertEqual(self.controller.snapshot("request-a")["incumbent_policy_hash"],
                         new.policy_hash)

    def test_batch_change_cannot_resume_queued_winner_after_budget_exhaustion(self):
        self.start(active_batch=2, config=replace(self.config, maximum_probe_tokens=7))
        self.window(100)
        self.window(40, 1000)
        self.window(120, 1000)
        self.ack()
        self.rebound(self.refreshed((self.full, self.quarter)), self.full.layer_mask)
        self.window(40, 1000, active_batch=2, next_active_batch=1, membership_changed=True)
        self.assertTrue(self.directive.control.policy.baseline)
        before = self.controller.checkpoint()[1]["request-a"]
        self.ack()
        for _ in range(3):
            self.assertIsNone(self.directive.control)
            self.assertTrue(self.controller.active_policy("request-a").baseline)
            self.window(100)
        after = self.controller.checkpoint()[1]["request-a"]
        self.assertIsNone(after.probe_budget)
        self.assertIsNone(after.incumbent_policy)
        self.assertEqual(after.probe_attempts, before.probe_attempts)
        self.assertEqual(after.probe_tokens, before.probe_tokens)

    def test_batch_change_discards_deferred_challenger_intent(self):
        self.start(active_batch=2)
        self.defer_quarter()
        self.rebound(self.refreshed((self.full, self.quarter)), self.full.layer_mask)
        self.window(40, 1000, active_batch=2, next_active_batch=1, membership_changed=True)
        after = self.controller.checkpoint()[1]["request-a"]
        self.assertIsNone(after.deferred_policy)
        self.assertIsNone(after.deferred_control_reason)
        self.assertIsNone(after.pending_session_drain_policy)
        self.assertIsNone(after.pending_helper_refresh_policy)
        self.ack()
        self.assertIsNone(self.directive.control)

    def test_helper_loss_cancels_unsent_refresh_before_baseline_ack(self):
        self.pending_winner_refresh()
        before = self.controller.checkpoint()[1]["request-a"]
        self.controller.helper_unavailable("request-a")
        lost = self.controller.checkpoint()[1]["request-a"]
        self.assertIsNone(lost.pending_session_drain_policy)
        self.assertIsNone(lost.pending_helper_refresh_policy)
        self.assertEqual(lost.records, before.records)
        self.assertEqual(lost.current_policy, before.current_policy)
        self.assertEqual(lost.current_ack, before.current_ack)
        self.window(40, 1000)
        self.assertTrue(self.directive.control.policy.baseline)
        self.ack()
        for _ in range(3):
            self.assertIsNone(self.directive.control)
            self.assertTrue(self.controller.active_policy("request-a").baseline)
            self.window(100)
        self.assertFalse(self.controller.snapshot("request-a")["helper_available"])

    def test_helper_loss_preserves_sent_refresh_ack_and_recovers_directly(self):
        new = self.pending_winner_refresh()
        self.window(40, 1000)
        sent = self.directive.control
        before = self.controller.checkpoint()[1]["request-a"]
        self.controller.helper_unavailable("request-a")
        lost = self.controller.checkpoint()[1]["request-a"]
        self.assertEqual(lost.awaiting_control, sent)
        self.assertEqual(lost.transition_policy, before.transition_policy)
        self.assertEqual(lost.records, before.records)
        self.ack()
        self.assertIsNotNone(self.directive.control)
        self.assertTrue(self.directive.control.policy.baseline)
        recovering = self.controller.checkpoint()[1]["request-a"]
        self.assertEqual(recovering.acknowledged_policy, new)
        self.assertEqual(recovering.transition_policy, new)
        self.assertEqual(recovering.transition_ack.policy_hash, sent.policy.policy_hash)
        self.token += 1
        self.at_us += 1000
        control = self.directive.control
        self.directive = self.controller.acknowledge(
            "request-a", fixtures.AdaptiveDecodePolicyAck(
                request_id="request-a", slot_id=7,
                plan_generation=control.plan_generation,
                applied_token_index=self.token, applied_at_us=self.at_us,
                policy_hash=control.policy.policy_hash,
            ), transition_observation=fixtures.AdaptiveDecodeRawWindowObservation(
                fleet_energy_uj_by_domain={"fleet": 40},
                phone_compute_us=10, usb_transfer_us=5, rpc_us=2,
                exposed_tail_us=1, output_valid=True,
                evidence_ids=("synthetic:delayed-recovery",),
                energy_boundary_id="synthetic-fleet", energy_attribution_kind="isolated",
                completed_phone_calls=2, completed_phone_input_rows=8,
            ),
        )
        after = self.controller.checkpoint()[1]["request-a"]
        self.assertEqual(after.records[:-1], before.records)
        receipt = after.records[-1]
        self.assertEqual(receipt.policy, new)
        self.assertEqual(receipt.applied_ack, recovering.transition_ack)
        self.assertEqual(receipt.completed_phone_calls, 2)
        self.assertEqual(receipt.whole_fleet_energy_uj, 40)
        self.assertFalse(receipt.measurement_eligible)
        self.assertIsNone(self.directive.control)
        self.assertTrue(self.controller.active_policy("request-a").baseline)

    def test_helper_loss_cancels_deferred_probe_without_resetting_attempts(self):
        self.start()
        self.defer_quarter()
        before = self.controller.checkpoint()[1]["request-a"]
        self.controller.helper_unavailable("request-a")
        after = self.controller.checkpoint()[1]["request-a"]
        for field in ("deferred_policy", "deferred_control_reason", "challenger_policy",
                      "probe_retry_policy", "verification_policy", "cached_winner"):
            self.assertIsNone(getattr(after, field), field)
        self.assertEqual(after.probe_attempts, before.probe_attempts)
        self.assertEqual(after.probe_tokens, before.probe_tokens)

    def test_baseline_ack_rechecks_availability_for_queued_positive_control(self):
        new = self.pending_winner_refresh()
        self.controller.helper_unavailable("request-a")
        self.window(40, 1000)
        self.assertTrue(self.directive.control.policy.baseline)
        checkpoint = self.controller.checkpoint()
        checkpoint[1]["request-a"].pending_helper_refresh_policy = new
        self.controller.restore(checkpoint)
        self.ack()
        self.assertIsNone(self.directive.control)
        self.assertTrue(self.controller.active_policy("request-a").baseline)
        self.assertIsNone(self.controller.snapshot("request-a")["pending_session_drain_policy_hash"])
        self.assertIsNone(self.controller.snapshot("request-a")["pending_helper_refresh_policy_hash"])

    def test_positive_control_issuance_rechecks_helper_availability(self):
        self.start(candidates=(self.full,))
        self.controller.helper_unavailable("request-a")
        session = self.controller.checkpoint()[1]["request-a"]
        directive = self.controller._control(session, self.full, self.token, self.at_us)
        self.assertTrue(directive.control.policy.baseline)
        self.assertEqual(session.zero_assistance_reason, "PHONE_HELPER_UNAVAILABLE")

    def defer_quarter(self):
        self.window(100)
        self.window(40, 1000)
        self.assertEqual(self.directive.control.policy, self.quarter)
        self.at_us += 1
        self.directive = self.controller.defer_control(
            "request-a", self.directive.control, "HELPER_RESOURCES_BUSY", at_us=self.at_us)
        self.assertEqual(self.controller.active_policy("request-a"), self.full)

    def test_compatible_batch_change_monitors_only_the_current_fraction(self):
        self.start(candidates=(self.full,), active_batch=2)
        self.window(100)
        self.window(40, 1000)
        before = self.controller.checkpoint()[1]["request-a"]
        self.window(40, 1000, active_batch=2, next_active_batch=1,
                    membership_changed=True, compatible_batch_change=True)
        monitored = self.controller.checkpoint()[1]["request-a"]
        self.assertIsNone(self.directive.control)
        self.assertEqual(monitored.current_policy, self.full)
        self.assertIsNone(monitored.incumbent_policy)
        self.assertEqual(monitored.context_monitor_prior["evidence_state"], "PRIOR_ONLY")
        self.assertEqual(monitored.context_monitor_prior["source_active_batch"], 2)
        self.assertEqual(monitored.records[:-1], before.records)
        self.assertFalse(monitored.records[-1].measurement_eligible)
        self.assertFalse(self.controller._current_valid_records(monitored, self.full))
        self.assertEqual(monitored.probe_candidates, [self.full])
        self.assertIsNotNone(monitored.verification_budget)
        self.assertGreater(monitored.verification_budget["token_limit"], self.token)
        self.assertEqual(sum(monitored.probe_attempts.values()), sum(before.probe_attempts.values()) + 1)
        self.window(40, 1000)
        self.assertTrue(self.directive.control.policy.baseline)
        self.window(100)
        self.assertEqual(self.directive.control.policy, self.full)
        self.assertEqual(self.directive.reason, "VERIFICATION_VERIFIED")
        self.assertEqual(self.controller.snapshot("request-a")["incumbent_policy_hash"],
                         self.full.policy_hash)

    def test_context_monitor_rejects_measurement_without_restart_of_sweep(self):
        self.start()
        self.window(100)
        self.window(40, 1000)
        self.window(120, 1000)
        self.ack()
        self.window(40, 1000, next_active_batch=2, membership_changed=True,
                    compatible_batch_change=True)
        self.assertIsNone(self.directive.control)
        self.window(120, 1000)
        self.window(100)
        self.assertEqual(self.directive.reason, "VERIFICATION_REJECTED")
        self.assertTrue(self.controller.active_policy("request-a").baseline)
        self.assertEqual(self.controller.snapshot("request-a")["probe_fractions_ppm"], (1_000_000,))
        for _ in range(3):
            self.window(100)
            self.assertIsNone(self.directive.control)

    def test_context_monitor_refresh_keeps_reservation_and_request_attempt_identity(self):
        self.start(candidates=(self.full,), active_batch=2)
        self.window(100)
        self.window(40, 1000)
        self.window(40, 1000, next_active_batch=1, membership_changed=True,
                    compatible_batch_change=True)
        before = self.controller.checkpoint()[1]["request-a"]
        new = self.refreshed((self.full,))
        self.rebound(new, self.full.layer_mask)
        after = self.controller.checkpoint()[1]["request-a"]
        self.assertTrue(after.operational_verification)
        self.assertEqual(after.verification_policy, new[0])
        self.assertEqual(after.verification_budget, before.verification_budget)
        self.assertEqual(after.probe_attempts, before.probe_attempts)
        self.assertEqual(after.context_monitor_prior, before.context_monitor_prior)
        self.assertEqual(after.pending_helper_refresh_policy, new[0])
        self.assertIsNone(after.incumbent_policy)
        self.window(40, 1000)
        self.assertEqual(self.directive.control.policy, new[0])
        self.ack()
        self.assertEqual(self.controller.snapshot("request-a")["probe_attempts"], before.probe_attempts)

    def test_helper_loss_during_context_monitor_never_promotes_prior(self):
        self.start(candidates=(self.full,))
        self.window(100)
        self.window(40, 1000)
        self.window(40, 1000, next_active_batch=2, membership_changed=True,
                    compatible_batch_change=True)
        self.controller.helper_unavailable("request-a")
        self.window(40, 1000)
        self.assertTrue(self.directive.control.policy.baseline)
        self.ack()
        after = self.controller.checkpoint()[1]["request-a"]
        self.assertIsNone(after.incumbent_policy)
        self.assertIsNone(after.context_monitor_prior)
        self.assertFalse(after.operational_verification)

    def test_context_monitor_does_not_replenish_exhausted_request_budget(self):
        self.start(config=replace(self.config, maximum_probe_tokens=7))
        self.window(100)
        self.window(40, 1000)
        self.window(120, 1000)
        self.ack()
        before = self.controller.snapshot("request-a")
        self.window(40, 1000, next_active_batch=2, membership_changed=True,
                    compatible_batch_change=True)
        self.assertTrue(self.directive.control.policy.baseline)
        self.assertEqual(self.directive.reason, "CONTEXT_MONITOR_UNAFFORDABLE")
        self.assertEqual(self.controller.snapshot("request-a")["probe_tokens"], before["probe_tokens"])
        self.assertEqual(self.controller.snapshot("request-a")["probe_attempts"], before["probe_attempts"])

    def test_repeated_context_changes_cannot_renew_monitor_indefinitely(self):
        self.start(candidates=(self.full,))
        self.window(100)
        self.window(40, 1000)
        for batch in (2, 1):
            self.window(40, 1000, next_active_batch=batch, membership_changed=True,
                        compatible_batch_change=True)
            self.assertIsNone(self.directive.control)
            self.assertEqual(self.controller.active_policy("request-a"), self.full)
        self.window(40, 1000, next_active_batch=2, membership_changed=True,
                    compatible_batch_change=True)
        self.assertTrue(self.directive.control.policy.baseline)
        self.assertEqual(self.controller.snapshot("request-a")["verification"]["attempts"], 2)

    def test_batch_change_preserves_physical_drain_and_its_ack_identity(self):
        self.pending_winner_refresh(active_batch=2)
        drain = self.controller.request_helper_session_drain(
            "request-a", retained_layer_mask=1 << self.full.layer_indices[0])
        before = self.controller.checkpoint()[1]["request-a"]
        self.assertIsNone(before.pending_helper_refresh_policy)
        self.window(40, 1000, next_active_batch=1, membership_changed=True,
                    compatible_batch_change=True)
        self.assertEqual(self.directive.control.policy.policy_hash, drain["policy_hash"])
        self.assertEqual(self.directive.control.policy.split_fraction_ppm, self.full.split_fraction_ppm)
        sent = self.directive.control
        self.assertEqual(self.controller.checkpoint()[1]["request-a"].transition_policy, self.full)
        self.ack()
        self.assertIsNone(self.directive.control)
        self.assertEqual(self.controller.active_policy("request-a"), sent.policy)
        self.window(40, 1000, next_active_batch=2, membership_changed=True,
                    compatible_batch_change=True)
        self.assertIsNone(self.directive.control)
        self.assertEqual(self.controller.active_policy("request-a"), sent.policy)
        self.assertFalse(self.controller.checkpoint()[1]["request-a"].records[-1].measurement_eligible)

    def test_probe_window_role_survives_bookkeeping_refresh(self):
        self.start()
        self.defer_quarter()
        before = self.controller.checkpoint()[1]["request-a"]
        self.assertEqual(before.window_role, "exploration")
        self.rebound(self.refreshed((self.full, self.quarter)), self.full.layer_mask)
        self.assertEqual(self.controller.snapshot("request-a")["state"], "EXPLOITING")
        self.window(40, 1000)
        after = self.controller.checkpoint()[1]["request-a"]
        self.assertEqual(after.records[-1].window_role, "exploration")
        self.assertEqual(after.probe_tokens, before.probe_tokens + after.records[-1].token_count)

    def test_exploitation_window_role_survives_revalidation_state_change(self):
        self.start(candidates=(self.full,))
        self.window(100)
        self.window(40, 1000)
        before = self.controller.checkpoint()[1]["request-a"]
        self.assertEqual(before.window_role, "exploitation")
        self.rebound(self.refreshed((self.full,)), 0)
        self.assertEqual(self.controller.snapshot("request-a")["state"], "PROBING")
        self.window(40, 1000)
        after = self.controller.checkpoint()[1]["request-a"]
        self.assertEqual(after.records[-1].window_role, "exploitation")
        self.assertEqual(after.probe_tokens, before.probe_tokens)

    def test_monitor_requires_ready_helper_with_supported_batch_and_parent(self):
        helper = SimpleNamespace(
            operator_plan_sha256=fixtures.PLAN,
            desktop_placement_sha256=fixtures.PLACEMENT,
            helper_plan=SimpleNamespace(execution_contract=SimpleNamespace(maximum_batch_size=2)))
        owner = SimpleNamespace(
            runtime_execution_ticket=Mock(return_value=object()),
            _request_helper_envelope=Mock(return_value=helper),
            _ready_request_helper=Mock(return_value=object()))
        boundary = SimpleNamespace(policy=self.full)
        check = AdaptiveDecodeControlMixin._compatible_helper_batch_change
        self.assertTrue(check(owner, "request-a", boundary, 1))
        self.assertTrue(check(owner, "request-a", boundary, 2))
        self.assertFalse(check(owner, "request-a", boundary, 3))
        owner._ready_request_helper.return_value = None
        self.assertFalse(check(owner, "request-a", boundary, 1))
        helper.desktop_placement_sha256 = "sha256:" + "0" * 64
        self.assertFalse(check(owner, "request-a", boundary, 1))

    def test_deferred_challenger_rebinds_and_retries_with_a_complete_pair(self):
        self.start()
        self.defer_quarter()
        before = self.controller.checkpoint()[1]["request-a"]
        new = self.refreshed((self.full, self.quarter))
        self.rebound(new, self.full.layer_mask)
        after = self.controller.checkpoint()[1]["request-a"]
        self.assertEqual(after.deferred_policy, new[1])
        self.assertEqual(after.deferred_control_reason, "HELPER_RESOURCES_BUSY")
        self.assertEqual(after.probe_tokens, before.probe_tokens)
        self.assertEqual(after.probe_attempts, before.probe_attempts)
        self.window(40, 1000)
        self.assertEqual(self.directive.control.policy, new[0])
        self.ack()
        self.window(40, 1000)
        self.assertEqual(self.directive.control.policy, new[1])
        retry = self.controller.checkpoint()[1]["request-a"]
        self.assertIn(self.directive.control.policy, retry.candidates)
        self.assertIsNotNone(retry.probe_budget)
        self.assertEqual(retry.probe_budget["started_at_us"], self.at_us)
        self.assertEqual(retry.state, "PROBING")
        self.ack()
        self.assertIsNone(self.controller.snapshot("request-a")["deferred_policy_hash"])
        self.window(120, 1000)
        self.assertEqual(self.directive.control.policy, new[0])

    def test_incompatible_deferred_challenger_is_discarded_not_substituted(self):
        self.start()
        self.defer_quarter()
        new = self.refreshed((self.full, replace(
            self.quarter, columns=200, split_fraction_ppm=200000)))
        self.rebound(new, self.full.layer_mask)
        after = self.controller.checkpoint()[1]["request-a"]
        self.assertIsNone(after.deferred_policy)
        self.assertIsNone(after.deferred_control_reason)
        self.assertEqual(after.incumbent_policy, new[0])
        self.window(40, 1000)
        self.ack()
        self.window(40, 1000)
        self.assertIsNone(self.directive.control)
        self.assertEqual(self.controller.active_policy("request-a"), new[0])

    def test_unverified_refresh_discards_deferred_authorization(self):
        self.start()
        self.defer_quarter()
        before = self.controller.checkpoint()[1]["request-a"]
        self.rebound(self.refreshed((self.full, self.quarter)), 0)
        after = self.controller.checkpoint()[1]["request-a"]
        self.assertIsNone(after.deferred_policy)
        self.assertIsNone(after.deferred_control_reason)
        self.assertFalse(after.policy_evidence_aliases)
        self.assertEqual(after.probe_tokens, before.probe_tokens)
        self.assertEqual(after.probe_attempts, before.probe_attempts)

    def test_deferred_retry_fences_a_policy_outside_current_candidates(self):
        self.start()
        self.defer_quarter()
        new = self.refreshed((self.full, self.quarter))
        self.rebound(new, self.full.layer_mask)
        self.window(40, 1000)
        self.ack()
        checkpoint = self.controller.checkpoint()
        checkpoint[1]["request-a"].deferred_policy = self.quarter
        self.controller.restore(checkpoint)
        self.window(40, 1000)
        self.assertIsNone(self.directive.control)
        self.assertEqual(self.controller.active_policy("request-a"), new[0])
        self.assertIsNone(self.controller.snapshot("request-a")["deferred_policy_hash"])

    def test_refreshed_deferred_probe_cannot_bypass_remaining_budget(self):
        self.start(output_tokens=32, config=replace(self.config, minimum_remaining_tokens=24))
        self.defer_quarter()
        new = self.refreshed((self.full, self.quarter))
        self.rebound(new, self.full.layer_mask)
        self.window(40, 1000)
        self.ack()
        self.window(40, 1000)
        self.assertEqual(self.directive.reason, "INSUFFICIENT_OPPORTUNITY")
        self.assertIsNone(self.directive.control)
        self.assertEqual(self.controller.active_policy("request-a"), new[0])
        self.assertIsNone(self.controller.snapshot("request-a")["deferred_policy_hash"])

    def test_zero_fraction_refresh_preserves_rejection_and_attempt_identity(self):
        self.start(candidates=(self.full,))
        self.window(100)
        self.window(120, 1000)
        self.ack()
        before = self.controller.checkpoint()[1]["request-a"]
        self.assertTrue(before.current_policy.baseline)
        self.assertEqual(before.eliminated_policy_reasons[self.full.policy_hash],
                         "LEARNING_NO_PAIRED_IMPROVEMENT")
        new = self.refreshed((self.full,))
        self.rebound(new, self.full.layer_mask)
        after = self.controller.checkpoint()[1]["request-a"]
        self.assertEqual(after.eliminated_policy_reasons.get(new[0].policy_hash),
                         before.eliminated_policy_reasons[self.full.policy_hash])
        self.assertEqual(self.controller._context_identity(after),
                         self.controller._context_identity(before))
        self.assertEqual(self.controller._probe_attempt_key(after, new[0]),
                         self.controller._probe_attempt_key(before, self.full))
        self.assertEqual(after.probe_attempts, before.probe_attempts)
        self.assertEqual(after.probe_tokens, before.probe_tokens)
        self.assertEqual(after.records, before.records)
        self.assertEqual(self.controller._current_valid_records(after, new[0]),
                         self.controller._current_valid_records(before, self.full))
        for _ in range(4):
            self.window(100)
            self.assertIsNone(self.directive.control)
        self.assertEqual(self.controller.helper_attachment_opportunity(
            "request-a", token_index=self.token, at_us=self.at_us), "MEASURED_REJECTION")
        self.assertEqual(self.controller.snapshot("request-a")["probe_attempts"], before.probe_attempts)

    def test_zero_fraction_refresh_cannot_replenish_incomplete_attempts(self):
        self.start(candidates=(self.full,))
        for _ in range(18):
            self.window(100, attribution_kind="diagnostic")
        before = self.controller.checkpoint()[1]["request-a"]
        self.assertTrue(before.current_policy.baseline)
        self.assertEqual(sum(before.probe_attempts.values()), 2)
        new = self.refreshed((self.full,))
        self.rebound(new, self.full.layer_mask)
        after = self.controller.checkpoint()[1]["request-a"]
        self.assertEqual(self.controller._probe_attempt_key(after, new[0]),
                         self.controller._probe_attempt_key(before, self.full))
        self.assertEqual(after.probe_attempts, before.probe_attempts)
        self.assertIsNone(self.controller._measurement_pair_budget(after, new[0], self.token, self.at_us))
        self.assertFalse(after.eliminated_policy_reasons)
        for _ in range(4):
            self.window(100, attribution_kind="diagnostic")
            self.assertIsNone(self.directive.control)
        self.assertEqual(self.controller.snapshot("request-a")["probe_attempts"], before.probe_attempts)

    def test_zero_fraction_genuine_change_still_requires_bounded_revalidation(self):
        self.start(candidates=(self.full,))
        self.window(100)
        self.window(120, 1000)
        self.ack()
        before = self.controller.checkpoint()[1]["request-a"]
        new = self.refreshed((replace(self.full, columns=900, split_fraction_ppm=900000),))
        self.rebound(new, 0)
        after = self.controller.checkpoint()[1]["request-a"]
        self.assertFalse(after.policy_evidence_aliases)
        self.assertNotIn(new[0].policy_hash, after.eliminated_policy_reasons)
        self.assertNotEqual(self.controller._probe_attempt_key(after, new[0]),
                            self.controller._probe_attempt_key(before, self.full))
        self.assertEqual(after.probe_attempts, before.probe_attempts)
        self.assertEqual(after.probe_tokens, before.probe_tokens)
        self.window(100)
        self.assertEqual(self.directive.control.policy, new[0])
        self.assertIsNotNone(self.controller.snapshot("request-a")["probe_budget"])

    def test_gemma_tail_reuses_measured_retained_mask_not_full_mask_evidence(self):
        self.start(output_tokens=292, candidates=(self.full,))
        self.window(100)
        self.window(40, 1000)
        while self.token < 101:
            self.window(40, 1000)
        expanded = (replace(self.full, layer_indices=(2, 3, 4), layer_mask=28,
                            operator_plan_sha256="sha256:" + "8" * 64),)
        self.rebound(expanded, 0)
        while self.token < 231:
            if self.directive.control is not None:
                self.ack()
            self.window(100 if self.controller.active_policy("request-a").baseline else 40, 1000)
        self.controller.request_helper_session_drain("request-a", retained_layer_mask=self.full.layer_mask)
        self.window(40, 1000)
        self.ack()
        retained = self.controller.active_policy("request-a")
        while self.token < 251:
            self.window(45, 1000)
        before = self.controller.checkpoint()[1]["request-a"]
        candidates = self.refreshed((retained,))
        self.controller.helper_rebound(
            "request-a", phone_layout_generation=5,
            phone_layout_geometry_sha256="sha256:" + "9" * 64,
            component_capability_sha256="sha256:" + "a" * 64,
            candidates=candidates, ticket_policy=None, helper_evidence_state="LEARNING",
            compatible_layers_by_plan={fixtures.PLAN: retained.layer_mask,
                                      expanded[0].operator_plan_sha256: retained.layer_mask},
        )
        after = self.controller.checkpoint()[1]["request-a"]
        self.assertEqual(after.incumbent_policy, candidates[0])
        rows = self.controller._current_valid_records(after, candidates[0], operational=True)
        self.assertTrue(rows)
        self.assertTrue(all(row.policy.layer_mask == retained.layer_mask for row in rows))
        self.assertEqual(after.records, before.records)
        self.assertNotIn(expanded[0].policy_hash, after.policy_evidence_aliases)
        while self.token < 292:
            self.window(45, 1000)
            policy = (self.directive.control.policy if self.directive.control else
                      self.controller.active_policy("request-a"))
            self.assertFalse(policy.baseline)
        self.assertEqual(self.controller.snapshot("request-a")["probe_tokens"], before.probe_tokens)

    def test_changed_shape_or_unverified_sessions_cannot_reuse_winner(self):
        for change in ("generations", "layers", "columns"):
            with self.subTest(change=change):
                self.setUp()
                self.start(candidates=(self.full,))
                self.window(100)
                self.window(40, 1000)
                new = self.refreshed((self.full,))
                if change == "layers":
                    new = (replace(new[0], layer_indices=(2, 3, 4), layer_mask=28),)
                if change == "columns":
                    new = (replace(new[0], columns=900),)
                before = self.controller.snapshot("request-a")
                if change == "columns":
                    with self.assertRaisesRegex(fixtures.AdaptiveDecodeError, "geometry differs"):
                        self.rebound(new, self.full.layer_mask)
                else:
                    self.rebound(new, 0 if change == "generations" else self.full.layer_mask)
                    session = self.controller.checkpoint()[1]["request-a"]
                    self.assertFalse(self.controller._qualifies(session, new[0], self.token, self.at_us))
                    self.assertFalse(session.policy_evidence_aliases)
                self.assertEqual(self.controller.snapshot("request-a")["probe_tokens"],
                                 before["probe_tokens"])

    def test_retained_evidence_proof_checks_physical_session_and_parent(self):
        shard = RuntimePhoneShard("session-a", "session://phone/a", 12, 1000, 4096,
                                  fixtures.PLAN, fixtures.PLAN, fixtures.ARTIFACT, 1)
        contract = RuntimeExecutionContract("adaptive-split", 0, (0, 1000000),
                                            "single", 1, 1, "phone", "phone://ffn", "ffn", (shard,))
        def helper(row=shard, **overrides):
            values = dict(artifact_sha256=fixtures.ARTIFACT,
                          desktop_parent_route_id="desktop-control",
                          desktop_placement_sha256=fixtures.PLACEMENT, activation_dtype="f16",
                          helper_plan=SimpleNamespace(baseline_executor_id="desktop",
                              execution_contract=replace(contract, phone_shards=(row,))),
                          helper_binding=SimpleNamespace(executor_id="desktop-phone", backend="ffn",
                              endpoint="phone://ffn", operator_plan_protocol="v3", participants=()))
            return SimpleNamespace(**(values | overrides))
        previous = helper()
        layout = SimpleNamespace(layout=SimpleNamespace(shards=(shard,),
                         session_generation_by_id={shard.session_id: 1}))
        self.assertEqual(retained_execution_layer_mask(previous, helper(), layout), 12)
        for changed in (replace(shard, session_generation=2),
                        replace(shard, operator_plan_sha256="sha256:" + "a" * 64),
                        replace(shard, resident_geometry_sha256="sha256:" + "b" * 64),
                        replace(shard, endpoint="session://phone/b"),
                        replace(shard, artifact_sha256="sha256:" + "c" * 64)):
            self.assertEqual(retained_execution_layer_mask(previous, helper(changed), layout), 0)
        self.assertEqual(retained_execution_layer_mask(previous, helper(
            desktop_placement_sha256="sha256:" + "d" * 64), layout), 0)


if __name__ == "__main__":
    unittest.main()
