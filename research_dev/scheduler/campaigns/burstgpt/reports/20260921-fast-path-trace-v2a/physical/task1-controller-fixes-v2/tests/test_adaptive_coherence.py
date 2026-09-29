"""Per-server policy coherence: co-tenants of one model on one desktop parent share the phone policy."""
from dataclasses import replace
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from research_dev.scheduler._internal.adaptive_decode import AdaptiveDecodeController
from research_dev.scheduler._internal.adaptive_decode_contracts import (
    AdaptiveDecodeConfig, AdaptiveDecodePolicyAck, AdaptiveDecodeRawWindowObservation,
)
from research_dev.scheduler._internal.adaptive_decode_ops.coherence import COHERENCE_REASON
from research_dev.scheduler.tests.test_adaptive_decode import ARTIFACT, PLAN, policy
from research_dev.scheduler._unified.helper_preparation_ops.attachment import _attach_helper_window_leases
from research_dev.scheduler._unified.helper_preparation_ops.cleanup import _release_request_helper_leases


class AdaptiveCoherenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = policy("desktop-control", 0, (), baseline=True)
        self.full = policy("phone-full", 1000, (2, 3))
        self.config = AdaptiveDecodeConfig(
            minimum_remaining_tokens=4, minimum_window_tokens=2, maximum_window_tokens=2,
            maximum_probe_tokens=20, maximum_probe_candidates=1, measurement_resolution_us=1,
            transition_cost_us=1, transition_energy_uj=1, minimum_energy_saving_ppm=10_000,
            uncertainty_ppm=10_000, warmup_windows_per_policy=0,
        )

    def start(self, controller, request_id, slot_id, output_tokens, **overrides):
        arguments = dict(
            request_id=request_id, ticket_id=request_id + ":attempt:0",
            model_artifact_sha256=ARTIFACT, planning_profile_sha256=PLAN,
            baseline=self.baseline, candidates=(self.full,), output_tokens=output_tokens,
            context_length=64, active_batch=1, deadline_us=1_000_000, slot_id=slot_id,
            first_token_index=1, first_token_at_us=1_000, config=self.config,
            helper_evidence_state="LEARNING",
        )
        arguments.update(overrides)
        return controller.start(**arguments)

    def record_baseline_window(self, controller, request_id, slot_id, at_us, *, energy_per_token=100,
                               phone_calls=0, next_active_batch=None, execution_context_available=True):
        """A LEARNING session measures one baseline window before it probes; drive that window."""
        session = controller.checkpoint()[1][request_id]
        directive = controller.boundary(request_id, slot_id=slot_id, token_index=session.target_token, at_us=at_us)
        boundary = directive.boundary
        return controller.record_window(request_id, boundary, AdaptiveDecodeRawWindowObservation(
            fleet_energy_uj_by_domain={"fleet": energy_per_token * boundary.token_count},
            phone_compute_us=0, usb_transfer_us=0, rpc_us=0, exposed_tail_us=0, output_valid=True,
            evidence_ids=("synthetic:window",), energy_boundary_id="synthetic-fleet",
            energy_attribution_kind="isolated", failure_reason=None, usb_upload_bytes=0,
            usb_download_bytes=0, desktop_compute_us=20, useful_overlap_us=0, request_queue_delay_us=3,
            protected_interference_us=0, active_batch=None, next_active_batch=next_active_batch,
            membership_changed=next_active_batch is not None,
            execution_context_available=execution_context_available,
            completed_phone_calls=phone_calls,
            completed_phone_input_rows=phone_calls, external_activity_sha256=None))

    def lead(self, controller):
        """Start the leader and bring it to its first phone control (the probe)."""
        self.start(controller, "request-a", 1, 60)
        directive = self.record_baseline_window(controller, "request-a", 1, 3_000)
        self.assertIsNotNone(directive.control)
        self.assertFalse(directive.control.policy.baseline)
        return directive

    def test_a_new_co_tenant_starts_on_the_running_phone_policy(self):
        controller = AdaptiveDecodeController()
        self.lead(controller)
        # Too short to probe on its own: alone it would exploit the baseline (INSUFFICIENT_OPPORTUNITY).
        follower = self.start(controller, "request-b", 2, 8)
        self.assertIsNotNone(follower.control)
        self.assertEqual(controller._policy_identity(follower.control.policy),
                         controller._policy_identity(self.full))
        self.assertEqual(controller.checkpoint()[1]["request-b"].incumbent_policy, self.full)

    def test_a_request_alone_stays_on_the_baseline(self):
        controller = AdaptiveDecodeController()
        alone = self.start(controller, "request-b", 2, 8)
        self.assertIsNone(alone.control)
        self.assertEqual(controller.active_policy("request-b"), self.baseline)

    def test_a_co_tenant_of_another_parent_does_not_follow(self):
        controller = AdaptiveDecodeController()
        other_parent = replace(self.baseline, desktop_placement_sha256="sha256:" + "9" * 64)
        other_full = replace(self.full, desktop_placement_sha256="sha256:" + "9" * 64)
        self.start(controller, "request-a", 1, 60, baseline=other_parent, candidates=(other_full,))
        leader = self.record_baseline_window(controller, "request-a", 1, 3_000)
        self.assertIsNotNone(leader.control)
        follower = self.start(controller, "request-b", 2, 8)
        self.assertIsNone(follower.control)

    def test_an_exploiting_baseline_session_follows_at_its_next_boundary(self):
        controller = AdaptiveDecodeController()
        # Too short to probe on its own, long enough to measure two windows once it follows.
        alone = self.start(controller, "request-b", 2, 8)
        self.assertEqual(alone.state, "EXPLOITING")
        self.assertIsNone(alone.control)
        self.lead(controller)
        # Mid-window boundaries never switch policy; the follow lands when the baseline window is recorded.
        self.assertIsNone(controller.boundary("request-b", slot_id=2, token_index=2, at_us=4_000))
        directive = self.record_baseline_window(controller, "request-b", 2, 5_000)
        self.assertEqual(directive.reason, COHERENCE_REASON)
        self.assertEqual(controller._policy_identity(directive.control.policy),
                         controller._policy_identity(self.full))
        records = controller.checkpoint()[1]["request-b"].records
        self.assertEqual([(row.token_start, row.token_end) for row in records], [(1, 3)])

    def test_a_session_too_short_to_measure_does_not_follow(self):
        controller = AdaptiveDecodeController()
        self.start(controller, "request-b", 2, 5)
        self.lead(controller)
        directive = self.record_baseline_window(controller, "request-b", 2, 5_000)
        self.assertIsNone(directive.control)


class AdaptiveServerPolicyTests(unittest.TestCase):
    def setUp(self):
        AdaptiveCoherenceTests.setUp(self)
        self.config = replace(self.config, server_policy_coherence=True)

    def start(self, controller, request_id, slot_id, output_tokens, **overrides):
        return AdaptiveCoherenceTests.start(
            self, controller, request_id, slot_id, output_tokens,
            helper_layout_generation=overrides.pop("helper_layout_generation", 1),
            helper_layout_geometry_sha256="sha256:" + "7" * 64, **overrides)

    record_baseline_window = AdaptiveCoherenceTests.record_baseline_window
    lead = AdaptiveCoherenceTests.lead

    def test_short_follower_joins_at_its_first_decode_boundary(self):
        controller = AdaptiveDecodeController()
        self.lead(controller)
        follower = self.start(controller, "request-b", 2, 3)
        self.assertEqual(follower.reason, COHERENCE_REASON)
        self.assertEqual(follower.control.policy, self.full)
        self.assertEqual(controller.helper_attachment_opportunity(
            "request-b", token_index=1, at_us=3_001), "ELIGIBLE")

    def test_only_one_request_probes_each_server_layout(self):
        controller = AdaptiveDecodeController()
        self.start(controller, "request-a", 1, 60, active_batch=2)
        self.start(controller, "request-b", 2, 60, active_batch=2)
        follower = self.record_baseline_window(controller, "request-b", 2, 3_000)
        self.assertIsNone(follower.control)
        self.assertEqual(controller.active_policy("request-b"), self.baseline)
        leader = self.record_baseline_window(controller, "request-a", 1, 3_001)
        self.assertEqual(leader.control.policy, self.full)

    def test_follower_seals_partial_window_before_following(self):
        controller = AdaptiveDecodeController()
        self.start(controller, "request-a", 1, 60, active_batch=2)
        self.start(controller, "request-b", 2, 60, active_batch=2)
        self.record_baseline_window(controller, "request-a", 1, 3_000)
        directive = controller.boundary("request-b", slot_id=2, token_index=2, at_us=3_001)
        self.assertEqual(directive.boundary.token_end, 2)
        follower = self.record_baseline_window(controller, "request-b", 2, 4_000)
        self.assertEqual(follower.control.policy, self.full)
        self.assertEqual([(row.token_start, row.token_end) for row in controller.checkpoint()[1]["request-b"].records],
                         [(1, 2)])
        self.assertFalse(controller.checkpoint()[1]["request-b"].records[0].measurement_eligible)

    def test_layout_generation_cannot_inherit_a_running_policy(self):
        controller = AdaptiveDecodeController()
        self.lead(controller)
        follower = self.start(controller, "request-b", 2, 3, helper_layout_generation=2)
        self.assertIsNone(follower.control)
        self.assertNotEqual(controller.shared_server_policy_key("request-a"),
                            controller.shared_server_policy_key("request-b"))

    def test_checkpoint_restores_server_policy_and_owner(self):
        controller = AdaptiveDecodeController()
        self.start(controller, "request-a", 1, 60)
        checkpoint = controller.checkpoint()
        self.record_baseline_window(controller, "request-a", 1, 3_000)
        controller.restore(checkpoint)
        follower = self.start(controller, "request-b", 2, 3)
        self.assertIsNone(follower.control)

    def test_measured_server_winner_survives_the_owner_completion(self):
        controller = AdaptiveDecodeController()
        leader = self.lead(controller)
        controller.acknowledge("request-a", AdaptiveDecodePolicyAck(
            "request-a", 1, leader.control.plan_generation, 3, 3_000, self.full.policy_hash))
        measured = self.record_baseline_window(controller, "request-a", 1, 4_000,
                                              energy_per_token=40, phone_calls=2)
        self.assertIsNone(measured.control)
        self.assertEqual(controller.active_policy("request-a"), self.full)
        controller.complete("request-a", "COMPLETED")
        next_request = self.start(controller, "request-b", 2, 3)
        self.assertEqual(next_request.control.policy, self.full)

    def test_unfinished_probe_continues_with_the_next_owner(self):
        controller = AdaptiveDecodeController()
        leader = self.lead(controller)
        controller.acknowledge("request-a", AdaptiveDecodePolicyAck(
            "request-a", 1, leader.control.plan_generation, 3, 3_000, self.full.policy_hash))
        controller.complete("request-a", "COMPLETED")
        next_request = self.start(controller, "request-b", 2, 8, first_token_at_us=4_000)
        self.assertEqual(next_request.control.policy, self.full)
        self.assertTrue(controller.helper_window_bid("request-b")["server_policy_eligible"])
        controller.acknowledge("request-b", AdaptiveDecodePolicyAck(
            "request-b", 2, next_request.control.plan_generation, 1, 4_000, self.full.policy_hash))
        measured = self.record_baseline_window(controller, "request-b", 2, 5_000,
                                              energy_per_token=40, phone_calls=2)
        self.assertIsNone(measured.control)
        group = controller.checkpoint()[6][controller.shared_server_policy_key("request-b")]
        self.assertTrue(group.qualified)
        self.assertEqual(group.policy, self.full)

    def test_short_owner_completion_reprobes_after_a_stable_membership_window(self):
        controller = AdaptiveDecodeController()
        self.start(controller, "request-a", 1, 3, active_batch=2)
        self.start(controller, "request-b", 2, 60, active_batch=2)
        self.record_baseline_window(controller, "request-a", 1, 2_000)
        controller.complete("request-a", "COMPLETED")
        changed = self.record_baseline_window(
            controller, "request-b", 2, 3_000, next_active_batch=1)
        self.assertIsNone(changed.control)
        session = controller.checkpoint()[1]["request-b"]
        self.assertEqual(session.active_batch, 1)
        self.assertFalse(session.records[-1].measurement_eligible)
        stable = self.record_baseline_window(controller, "request-b", 2, 4_000)
        self.assertEqual(stable.control.policy, self.full)
        self.assertTrue(controller.checkpoint()[1]["request-b"].records[-1].measurement_eligible)

    def test_short_owner_can_fund_a_probe_from_its_long_co_tenant(self):
        controller = AdaptiveDecodeController()
        self.start(controller, "request-a", 1, 8, active_batch=2)
        self.start(controller, "request-b", 2, 60, active_batch=2)
        leader = self.record_baseline_window(controller, "request-a", 1, 3_000)
        self.assertIsNotNone(leader.control)
        self.assertEqual(leader.control.policy, self.full)
        follower = self.record_baseline_window(controller, "request-b", 2, 3_001)
        self.assertEqual(follower.control.policy, self.full)

    def test_probe_cannot_borrow_remaining_work_from_another_layout(self):
        controller = AdaptiveDecodeController()
        self.start(controller, "request-a", 1, 8)
        self.start(controller, "request-b", 2, 60, helper_layout_generation=2)
        leader = self.record_baseline_window(controller, "request-a", 1, 3_000)
        self.assertIsNone(leader.control)

    def test_delayed_ack_uses_the_shared_probe_budget(self):
        controller = AdaptiveDecodeController()
        leader = self.lead(controller)
        controller._sessions["request-a"].probe_budget["deadline_us"] = 2_999
        acknowledged = controller.acknowledge("request-a", AdaptiveDecodePolicyAck(
            "request-a", 1, leader.control.plan_generation, 3, 3_000, self.full.policy_hash))
        self.assertIsNone(acknowledged.control)
        self.assertEqual(controller.active_policy("request-a"), self.full)
        self.assertEqual(controller._sessions["request-a"].window_role, "exploration")

    def test_unknown_context_pauses_without_qualifying_the_host_policy(self):
        controller = AdaptiveDecodeController()
        leader = self.lead(controller)
        controller.acknowledge("request-a", AdaptiveDecodePolicyAck(
            "request-a", 1, leader.control.plan_generation, 3, 3_000, self.full.policy_hash))
        paused = self.record_baseline_window(
            controller, "request-a", 1, 4_000, phone_calls=2,
            execution_context_available=False)
        self.assertEqual(paused.control.policy, self.baseline)
        group = controller._server_policies[controller.shared_server_policy_key("request-a")]
        self.assertFalse(group.qualified)
        self.assertEqual(group.proposal, self.full)
        self.assertEqual(group.probe_tokens, 2)
        controller.acknowledge("request-a", AdaptiveDecodePolicyAck(
            "request-a", 1, paused.control.plan_generation, 5, 4_000, self.baseline.policy_hash))
        resumed = self.record_baseline_window(controller, "request-a", 1, 5_000)
        self.assertEqual(resumed.control.policy, self.full)
        self.assertEqual(group.probe_tokens, 4)

    def test_last_window_can_qualify_a_policy_for_the_next_request(self):
        controller = AdaptiveDecodeController()
        leader = self.lead(controller)
        controller.acknowledge("request-a", AdaptiveDecodePolicyAck(
            "request-a", 1, leader.control.plan_generation, 3, 3_000, self.full.policy_hash))
        controller._sessions["request-a"].output_tokens = 5
        self.record_baseline_window(controller, "request-a", 1, 4_000,
                                    energy_per_token=40, phone_calls=2)
        group = controller.checkpoint()[6][controller.shared_server_policy_key("request-a")]
        self.assertTrue(group.qualified)
        self.assertEqual(group.policy, self.full)

    def test_incomparable_windows_consume_the_shared_probe_budget(self):
        controller = AdaptiveDecodeController()
        leader = self.lead(controller)
        controller.acknowledge("request-a", AdaptiveDecodePolicyAck(
            "request-a", 1, leader.control.plan_generation, 3, 3_000, self.full.policy_hash))
        for step in range(10):
            boundary = controller.boundary("request-a", slot_id=1, token_index=5 + step * 2,
                                           at_us=4_000 + step * 1_000).boundary
            result = controller.record_window("request-a", boundary, AdaptiveDecodeRawWindowObservation(
                fleet_energy_uj_by_domain={"fleet": 80}, phone_compute_us=0, usb_transfer_us=0,
                rpc_us=0, exposed_tail_us=0, output_valid=True, evidence_ids=("synthetic:window",),
                energy_boundary_id="synthetic-fleet", energy_attribution_kind="isolated",
                active_batch=1, next_active_batch=1, membership_changed=True,
                completed_phone_calls=2, completed_phone_input_rows=2), compatible_batch_change=True)
            if step < 9:
                self.assertIsNone(result.control)
                self.assertEqual(controller.active_policy("request-a"), self.full)
        self.assertEqual(result.control.policy, self.baseline)
        self.assertEqual(controller.snapshot("request-a")["zero_assistance_reason"],
                         "SERVER_PROBE_BUDGET_EXHAUSTED")

    def test_baseline_at_another_batch_size_cannot_reject_phone(self):
        controller = AdaptiveDecodeController()
        leader = self.lead(controller)
        controller.acknowledge("request-a", AdaptiveDecodePolicyAck(
            "request-a", 1, leader.control.plan_generation, 3, 3_000, self.full.policy_hash))
        session = controller._sessions["request-a"]
        baseline = session.records[0]
        session.records.clear()
        session.historical_records[controller._policy_identity(self.baseline)] = (
            replace(baseline, active_batch=2),)
        self.assertIsNone(controller._bounds(session, self.baseline, operational=True))
        controller._update_elimination(session, self.full)
        self.assertFalse(session.eliminated_policy_reasons)


class SharedServerHelperLeaseTests(unittest.TestCase):
    def setUp(self):
        self.tickets = [SimpleNamespace(
            request=SimpleNamespace(request_id=name), ticket_id=name + ":0",
            model=SimpleNamespace(artifact_sha256=ARTIFACT), binding=SimpleNamespace(endpoint="http://host:8000"),
            execution_plan=SimpleNamespace(resource_ids=("cpu",), adapter_parameters={"parallel": 4}),
        ) for name in ("a", "b")]
        self.helper = SimpleNamespace(
            desktop_placement_sha256=PLAN, phone_layout_generation=1, phone_layout_geometry_sha256=PLAN,
            resident_layer_mask=3, resident_columns=1000,
            helper_plan=SimpleNamespace(resource_ids=("cpu", "phone", "usb"),
                                        execution_contract=SimpleNamespace(maximum_batch_size=4)),
        )
        self.bindings = {name: {
            "base": {"desktop_placement_sha256": PLAN}, "fraction_ppm": 1_000_000,
            "helper_envelope": {"assisted_layer_mask": 3, "maximum_columns": 1000,
                                "resource_ids": ("cpu", "phone", "usb")},
            "helper_attachment": {"lease_tokens": ("phone-lease", "usb-lease"), "lease_reserved_until_us": 100,
                                  "phone_layout_generation": 1, "phone_layout_geometry_sha256": PLAN},
        } for name in ("a", "b")}
        self.bindings["b"]["helper_attachment"] = None
        def clear(request_id, _fraction, **_kwargs):
            self.bindings[request_id]["helper_attachment"]["lease_tokens"] = ()
        def renew(request_id, *, lease_tokens, reserved_until_us, observed_at_us):
            self.bindings[request_id]["helper_attachment"]["lease_reserved_until_us"] = reserved_until_us
        self.controller = SimpleNamespace(
            _adaptive_decode_config=AdaptiveDecodeConfig(server_policy_coherence=True),
            _runtime_controller=SimpleNamespace(current_tickets=lambda _states: self.tickets),
            _model_placement_controller=SimpleNamespace(
                request_binding=self.bindings.get, record_request_helper_event=Mock(), update_request_fraction=clear,
                renew_request_helper_leases=renew),
            _helper_window_lease_horizon=lambda *args, **kwargs: 200,
            _transaction=lambda **kwargs: nullcontext(), extend_lease=Mock(), reserve_external_resources=Mock(),
            _runtime_renewals={}, release=Mock(),
        )

    def test_shared_reservation_lives_until_the_last_member_releases(self):
        leases = _attach_helper_window_leases(
            self.controller, self.tickets[1], self.helper, SimpleNamespace(generation=1),
            current_attachment=None, token_index=1, at_us=10, fraction_ppm=1_000_000)
        self.assertEqual(leases.lease_tokens, ("phone-lease", "usb-lease"))
        self.assertFalse(leases.newly_reserved_tokens)
        self.controller.reserve_external_resources.assert_not_called()
        self.assertEqual(self.controller.extend_lease.call_count, 2)
        self.assertEqual(self.bindings["a"]["helper_attachment"]["lease_reserved_until_us"], 200)
        self.bindings["b"]["helper_attachment"] = dict(self.bindings["a"]["helper_attachment"])
        _release_request_helper_leases(self.controller, "a", 20)
        self.controller.release.assert_not_called()
        _release_request_helper_leases(self.controller, "b", 21)
        self.assertEqual(self.controller.release.call_count, 2)

    def test_different_endpoint_cannot_borrow_the_reservation(self):
        from research_dev.scheduler._unified.helper_preparation_ops.common import shared_helper_attachments
        self.tickets[1].binding.endpoint = "http://host:8001"
        self.assertEqual(shared_helper_attachments(self.controller, self.tickets[1], self.helper), ())


if __name__ == "__main__":
    unittest.main()
