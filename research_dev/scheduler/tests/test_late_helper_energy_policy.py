"""Late attachment binds operational power policy, not qualification."""

from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research_dev.scheduler import UnifiedScheduler
from research_dev.scheduler._internal.adaptive_decode_contracts import (
    AdaptiveDecodeConfig, AdaptiveDecodePolicyAck,
    AdaptiveDecodeRawWindowObservation,
)
from research_dev.scheduler._internal.runtime_capabilities import RuntimePhonePowerProfile
from research_dev.scheduler.tests.test_adaptive_decode import ARTIFACT, PLAN, policy


class LateHelperEnergyPolicyTests(unittest.TestCase):
    def setUp(self):
        self.baseline = policy("desktop-control", 0, (), baseline=True)
        self.phone = policy("phone-full", 1000, (2, 3))
        self.config = AdaptiveDecodeConfig(
            minimum_remaining_tokens=4, minimum_window_tokens=2,
            maximum_window_tokens=2, maximum_probe_candidates=1,
            measurement_resolution_us=1, transition_cost_us=1,
            transition_energy_uj=1, uncertainty_ppm=10_000,
            warmup_windows_per_policy=0,
        )

    def start_late(self, *, permission=True, explicit=None, ready_valid=True,
                   publish_first=False):
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler._adaptive_decode_config = self.config
        scheduler._runtime_capabilities = SimpleNamespace(
            minimum_energy_saving_ppm=50_000, maximum_latency_ppm=1_250_000,
            phone_power_profile_by_device={} if permission is None else {
                "phone": RuntimePhonePowerProfile.assumed_4p5w(
                    device_id="phone", domain_id="phone-system",
                    allow_assumed_for_scheduling=permission,
                ),
            },
        )
        ticket = SimpleNamespace(
            request=SimpleNamespace(request_id="late", input_tokens=64,
                                    output_tokens=100, deadline_us=1_000_000),
            ticket_id="late:attempt:0", planning_profile_sha256=PLAN,
            model=SimpleNamespace(artifact_sha256=ARTIFACT),
            decision=SimpleNamespace(route_id=self.baseline.route_id),
            execution_plan=SimpleNamespace(
                execution_contract=SimpleNamespace(execution_mode="desktop", phone_device_id=None),
                helper_envelope=None,
            ),
            cost_estimates=SimpleNamespace(estimates=(SimpleNamespace(
                route_id=self.baseline.route_id, details={},
            ),)),
        )
        helper = SimpleNamespace(
            phone_layout_generation=4, phone_layout_geometry_sha256=PLAN,
            operator_plan_sha256=PLAN,
            helper_plan=SimpleNamespace(execution_contract=SimpleNamespace(
                phone_device_id="phone", phone_shards=(),
            )),
        )
        available = [False]
        with ExitStack() as stack:
            mocks = {
                "runtime_execution_ticket": dict(return_value=ticket),
                "_has_dormant_phone_ffn_runtime": dict(return_value=True),
                "_request_helper_opportunity_for_ticket": dict(return_value=None),
                "_adaptive_policies_from_ticket": dict(return_value=(self.baseline, (), None)),
                "_request_helper_envelope": dict(side_effect=lambda _ticket: helper if available[0] else None),
                "_adaptive_start_component_capability_sha256": dict(return_value=PLAN),
                "_track_adaptive_directive": dict(side_effect=lambda _request, directive, **_kwargs: directive),
                "_reevaluate_pending_phone_layout_at_boundary": dict(return_value=None),
                "_ready_request_helper": dict(return_value=helper if ready_valid else None),
                "_attach_ready_request_helper": dict(return_value=True),
                "_remember_request_helper_envelope": dict(return_value=None),
                "_record_ready_helper_event_once": dict(return_value=None),
            }
            for name, arguments in mocks.items():
                stack.enter_context(patch.object(scheduler, name, **arguments))
            scheduler.start_adaptive_decode(
                "late", slot_id=0, first_token_index=1, at_us=1_000,
                config=explicit,
            )
            before = scheduler._adaptive_decode.checkpoint()[1]["late"]
            available[0] = True
            scheduler._late_request_helper_contexts["late"] = SimpleNamespace(
                candidates=(self.phone,), ticket_policy=None, evidence_state="LEARNING",
                component=SimpleNamespace(identity_sha256=PLAN),
            )
            if publish_first:
                scheduler._publish_ready_helper_materialization(
                    ticket,
                    SimpleNamespace(generation=4, layout=SimpleNamespace(geometry_sha256=PLAN)),
                    SimpleNamespace(
                        helper=helper, baseline=self.baseline, policies=(self.phone,),
                        ticket_policy=None, opportunity=SimpleNamespace(evidence_state="LEARNING"),
                        component=SimpleNamespace(identity_sha256=PLAN),
                        generated_parent_route_id=self.baseline.route_id,
                    ),
                    False, 1_500,
                )
            directive = scheduler.adaptive_decode_boundary(
                "late", slot_id=0, token_index=2, at_us=2_000,
            )
            if ready_valid:
                self.assertIsNotNone(directive.boundary)
        return scheduler, before, directive

    def test_late_ready_helper_refreshes_explicit_capability_permission(self):
        scheduler, before, _directive = self.start_late()
        controller = scheduler._adaptive_decode
        after = controller.checkpoint()[1]["late"]
        self.assertFalse(before.config.allow_assumed_phone_power_for_operational_selection)
        self.assertTrue(after.config.allow_assumed_phone_power_for_operational_selection)
        self.assertEqual(replace(after.config, allow_assumed_phone_power_for_operational_selection=False), before.config)
        self.assertEqual(after.helper_layout_generation, 4)
        self.assertEqual(after.baseline, before.baseline)

    def test_missing_or_denied_profile_and_explicit_override_stay_conservative(self):
        for permission, explicit, valid in ((None, None, True), (False, None, True),
                                            (True, self.config, True), (True, None, False)):
            with self.subTest(permission=permission, explicit=explicit is not None, ready_valid=valid):
                scheduler, _before, _directive = self.start_late(
                    permission=permission, explicit=explicit, ready_valid=valid,
                )
                session = scheduler._adaptive_decode.checkpoint()[1]["late"]
                self.assertFalse(session.config.allow_assumed_phone_power_for_operational_selection)

    def test_ready_publication_before_boundary_binds_the_same_policy_once(self):
        for permission, explicit in ((True, None), (False, None), (True, self.config)):
            with self.subTest(permission=permission, explicit=explicit is not None):
                scheduler, _before, _directive = self.start_late(
                    permission=permission, explicit=explicit, publish_first=True,
                )
                state = scheduler._adaptive_decode.snapshot("late")
                self.assertEqual(
                    state["allow_assumed_phone_power_for_operational_selection"],
                    permission and explicit is None,
                )

    def test_late_assumed_windows_drive_bids_but_never_become_qualified(self):
        scheduler, _before, directive = self.start_late()
        controller = scheduler._adaptive_decode
        at_us = 2_000
        for _ in range(10):
            boundary = directive.boundary
            phone = not boundary.policy.baseline
            directive = controller.record_window("late", boundary, AdaptiveDecodeRawWindowObservation(
                fleet_energy_uj_by_domain={"fleet": boundary.token_count * (40_000_000 if phone else 100_000_000)},
                phone_compute_us=10 if phone else 0, usb_transfer_us=5 if phone else 0,
                rpc_us=0, exposed_tail_us=0, output_valid=True,
                evidence_ids=("ASSUMED_4P5W", "physical:rapl", "physical:nvml"),
                energy_boundary_id="test-fleet", energy_attribution_kind="diagnostic",
                completed_phone_calls=boundary.token_count if phone else 0,
                completed_phone_input_rows=boundary.token_count if phone else 0,
            ))
            if phone:
                break
            control = directive.control
            self.assertIsNotNone(control)
            at_us += 1
            opened = controller.acknowledge("late", AdaptiveDecodePolicyAck(
                request_id="late", slot_id=0, plan_generation=control.plan_generation,
                applied_token_index=boundary.token_end, applied_at_us=at_us,
                policy_hash=control.policy.policy_hash,
            ))
            at_us += 1_000
            directive = controller.boundary("late", slot_id=0,
                token_index=opened.target_token_index, at_us=at_us)
        self.assertEqual(directive.state, "EXPLOITING")
        self.assertEqual(controller.active_policy("late"), self.phone)
        bid = controller.helper_window_bid("late", requested_fraction_ppm=1_000_000)
        self.assertEqual(bid["evidence"], "CONSERVATIVE_MEASURED")
        self.assertGreater(bid["gain_per_token_uj"], 0)
        checkpoint = controller.checkpoint()
        session = checkpoint[1]["late"]
        self.assertTrue(all(not row.energy_measurement_eligible for row in session.records))
        self.assertFalse(controller._valid_records(session, self.phone))
        for _ in range(3):
            controller.helper_ready("late", phone_layout_generation=4,
                                    phone_layout_geometry_sha256=PLAN)
            self.assertEqual(controller.checkpoint(), checkpoint)


if __name__ == "__main__":
    unittest.main()
