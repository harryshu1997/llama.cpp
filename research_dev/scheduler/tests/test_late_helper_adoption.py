"""Late-helper adoption: a session that started helper-less attaches the helper when its model's
phone shards become READY mid-session (cj5 F-C, opt-in ``late_helper_adoption``)."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

from research_dev.scheduler._internal.adaptive_decode import AdaptiveDecodeController
from research_dev.scheduler._internal.adaptive_decode_contracts import (
    AdaptiveDecodeConfig, AdaptiveDecodeError, AdaptiveDecodeRawWindowObservation,
)
from research_dev.scheduler._internal.adaptive_decode_ops import budgeting
from research_dev.scheduler.configuration.campaign import _adaptive_decode_overrides
from research_dev.scheduler.tests.test_adaptive_decode import ARTIFACT, PLAN, policy
from research_dev.scheduler._unified.adaptive_decode_control import AdaptiveDecodeControlMixin

GEOMETRY = "sha256:" + "7" * 64


class LateHelperAdoptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = policy("desktop-control", 0, (), baseline=True)
        self.full = policy("phone-full", 1000, (2, 3))
        self.config = AdaptiveDecodeConfig(
            minimum_remaining_tokens=4, minimum_window_tokens=2, maximum_window_tokens=2,
            maximum_probe_tokens=20, maximum_probe_candidates=1, measurement_resolution_us=1,
            transition_cost_us=1, transition_energy_uj=1, minimum_energy_saving_ppm=10_000,
            uncertainty_ppm=10_000, warmup_windows_per_policy=0,
        )
        self.adopting = replace(self.config, late_helper_adoption=True)

    def start(self, controller, request_id, slot_id, output_tokens, **overrides):
        arguments = dict(
            request_id=request_id, ticket_id=request_id + ":attempt:0",
            model_artifact_sha256=ARTIFACT, planning_profile_sha256=PLAN,
            baseline=self.baseline, candidates=(), output_tokens=output_tokens,
            context_length=64, active_batch=1, deadline_us=1_000_000, slot_id=slot_id,
            first_token_index=1, first_token_at_us=1_000, config=self.adopting,
            helper_evidence_state="LEARNING", helper_available=False,
        )
        arguments.update(overrides)
        return controller.start(**arguments)

    def baseline_window(self, controller, request_id, slot_id, at_us):
        """Drive one baseline window to its boundary (the LEARNING session measures before it probes)."""
        session = controller._sessions[request_id]
        directive = controller.boundary(request_id, slot_id=slot_id, token_index=session.target_token, at_us=at_us)
        boundary = directive.boundary
        return controller.record_window(request_id, boundary, AdaptiveDecodeRawWindowObservation(
            fleet_energy_uj_by_domain={"fleet": 100 * boundary.token_count},
            phone_compute_us=0, usb_transfer_us=0, rpc_us=0, exposed_tail_us=0, output_valid=True,
            evidence_ids=("synthetic:window",), energy_boundary_id="synthetic-fleet",
            energy_attribution_kind="isolated", failure_reason=None, usb_upload_bytes=0,
            usb_download_bytes=0, desktop_compute_us=20, useful_overlap_us=0, request_queue_delay_us=3,
            protected_interference_us=0, active_batch=None, next_active_batch=None,
            membership_changed=False, execution_context_available=True, completed_phone_calls=0,
            completed_phone_input_rows=0, external_activity_sha256=None))

    def test_config_fields_are_validated_and_default_off(self) -> None:
        default = AdaptiveDecodeConfig()
        self.assertFalse(default.late_helper_adoption)
        self.assertEqual(default.late_helper_adoption_minimum_tokens, 0)
        self.assertEqual(budgeting.late_helper_adoption_minimum_tokens(default), default.minimum_remaining_tokens)
        self.assertEqual(
            budgeting.late_helper_adoption_minimum_tokens(replace(default, late_helper_adoption_minimum_tokens=7)), 7,
        )
        for changes in ({"late_helper_adoption": 1}, {"late_helper_adoption": "yes"},
                        {"late_helper_adoption_minimum_tokens": -1},
                        {"late_helper_adoption_minimum_tokens": True},
                        {"late_helper_adoption_minimum_tokens": 2.0}):
            with self.subTest(changes=changes), self.assertRaises(AdaptiveDecodeError):
                AdaptiveDecodeConfig(**changes)
        overrides = _adaptive_decode_overrides(
            {"late_helper_adoption": True, "late_helper_adoption_minimum_tokens": 24}
        )
        configured = AdaptiveDecodeConfig(**overrides)
        self.assertTrue(configured.late_helper_adoption)
        self.assertEqual(configured.late_helper_adoption_minimum_tokens, 24)

    def test_helper_less_session_becomes_eligible_only_with_the_knob(self) -> None:
        # Too short to fund a probe pair and alone on its server: the knob is the only admission.
        plain = AdaptiveDecodeController()
        self.start(plain, "request-b", 2, 12, config=self.config)
        self.assertNotEqual(plain.helper_attachment_opportunity("request-b", token_index=1, at_us=1_100), "ELIGIBLE")
        self.assertFalse(plain.adopts_late_helper("request-b", token_index=1))
        adopting = AdaptiveDecodeController()
        self.start(adopting, "request-b", 2, 12)
        self.assertEqual(adopting.helper_attachment_opportunity("request-b", token_index=1, at_us=1_100), "ELIGIBLE")
        self.assertTrue(adopting.adopts_late_helper("request-b", token_index=1))
        with self.assertRaises(AdaptiveDecodeError):
            adopting.adopts_late_helper("request-b", token_index=-1)
        self.assertFalse(adopting.adopts_late_helper("absent", token_index=1))
        # Fewer remaining tokens than the configured minimum: not eligible.
        bounded = AdaptiveDecodeController()
        self.start(bounded, "request-b", 2, 12, config=replace(self.adopting, late_helper_adoption_minimum_tokens=20))
        self.assertNotEqual(bounded.helper_attachment_opportunity("request-b", token_index=1, at_us=1_100), "ELIGIBLE")
        self.assertFalse(bounded.adopts_late_helper("request-b", token_index=1))
        self.assertFalse(adopting.adopts_late_helper("request-b", token_index=10))

    def test_helper_available_from_the_start_is_unchanged_by_the_knob(self) -> None:
        results = []
        for config in (self.config, self.adopting):
            controller = AdaptiveDecodeController()
            self.start(controller, "request-a", 1, 60, candidates=(self.full,), helper_available=True,
                       helper_layout_generation=1, helper_layout_geometry_sha256=GEOMETRY, config=config)
            results.append(controller.helper_attachment_opportunity("request-a", token_index=1, at_us=1_100))
            self.assertFalse(controller.adopts_late_helper("request-a", token_index=1))
        self.assertEqual(results[0], results[1])

    def test_late_ready_shards_attach_and_the_session_explores_the_phone_policy(self) -> None:
        """cj5 F-C: the session starts helper-less; its model's shards become READY later; at the next
        boundary the helper is adopted at fraction 0 and the normal probing path explores the phone."""
        controller = AdaptiveDecodeController()
        self.assertIsNone(self.start(controller, "request-b", 2, 60).control)
        self.assertIsNone(self.baseline_window(controller, "request-b", 2, 3_000).control)
        before = controller.snapshot("request-b")
        self.assertFalse(before["helper_available"])
        self.assertEqual(list(before["candidate_policy_hashes"]), [])
        # The shards are READY now: the fraction-0 attachment gate admits the helper-less session.
        self.assertEqual(controller.helper_attachment_opportunity("request-b", token_index=3, at_us=5_000), "ELIGIBLE")
        self.assertTrue(controller.adopts_late_helper("request-b", token_index=3))
        controller.helper_ready(
            "request-b", phone_layout_generation=1, phone_layout_geometry_sha256=GEOMETRY,
            candidates=(self.full,), helper_evidence_state="LEARNING", ready_at_token_index=3,
        )
        self.assertFalse(controller.adopts_late_helper("request-b", token_index=3))
        after = controller.snapshot("request-b")
        self.assertTrue(after["helper_available"])
        self.assertEqual(list(after["candidate_policy_hashes"]), [self.full.policy_hash])
        directive = self.baseline_window(controller, "request-b", 2, 6_000)
        self.assertIsNotNone(directive.control)
        self.assertEqual(directive.control.policy, self.full)
        probing = controller.snapshot("request-b")
        self.assertTrue(probing["helper_available"])
        self.assertEqual(probing["state"], "PROBING")

    def test_boundary_records_helper_adopted_late_once_for_a_late_adoption(self) -> None:
        events = []
        helper = SimpleNamespace(phone_layout_generation=1, phone_layout_geometry_sha256=GEOMETRY)
        record = AdaptiveDecodeControlMixin._record_late_helper_adoption
        controller = AdaptiveDecodeController()
        self.start(controller, "request-b", 2, 60)
        owner = SimpleNamespace(
            _adaptive_decode=controller,
            _model_placement_controller=SimpleNamespace(
                record_request_helper_event=lambda *args: events.append(args),
            ),
        )
        record(owner, "request-b", helper, token_index=5, at_us=9_000)
        self.assertEqual(events, [("request-b", "HELPER_ADOPTED_LATE", 9_000, {
            "phone_layout_generation": 1, "phone_layout_geometry_sha256": GEOMETRY,
            "ready_at_us": 9_000, "request_id": "request-b", "source": "DECODE_BOUNDARY",
            "token_index": 5,
        })])
        controller.helper_ready("request-b", phone_layout_generation=1, phone_layout_geometry_sha256=GEOMETRY,
                                candidates=(self.full,), helper_evidence_state="LEARNING", ready_at_token_index=5)
        record(owner, "request-b", helper, token_index=6, at_us=9_500)
        self.assertEqual(len(events), 1)
        # Knob off, or a helper ready from the start: nothing is recorded.
        for overrides in ({"config": self.config}, {"helper_available": True, "candidates": (self.full,),
                                                    "helper_layout_generation": 1,
                                                    "helper_layout_geometry_sha256": GEOMETRY}):
            other = AdaptiveDecodeController()
            self.start(other, "request-c", 3, 60, **overrides)
            owner._adaptive_decode = other
            record(owner, "request-c", helper, token_index=5, at_us=9_000)
        self.assertEqual(len(events), 1)


if __name__ == "__main__":
    unittest.main()
