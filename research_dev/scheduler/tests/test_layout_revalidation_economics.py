"""Whole-layout opportunity costs without changing physical session authority."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from research_dev.scheduler._internal.model_placement_controller import ModelPlacementController, ModelPlacementPolicy
from research_dev.scheduler._internal.model_placement_contracts.layout import PhoneLayoutRequestImpact
from research_dev.scheduler._unified.phone_residency_ops.demand import _phone_queue_demand
from research_dev.scheduler._unified.phone_residency_ops.economics import _defer_phone_layout_revalidation
from research_dev.scheduler._unified.phone_residency_ops.portfolio import _update_phone_residency_portfolio
from research_dev.scheduler.tests.test_model_placement_controller import mixed_phone_layout
from research_dev.scheduler.tests import test_sustained_assistance as adaptive


class LayoutRevalidationEconomicsTests(unittest.TestCase):
    def setUp(self):
        self.controller = ModelPlacementController(ModelPlacementPolicy(phone_session_hysteresis_uj=0))
        self.current = mixed_phone_layout(("a", "a", "a"), current_assignments=("a", "a", "a"),
                                          benefit_by_session=(1000, 1000, 500))
        self.target = mixed_phone_layout(("a", "a", "b"), current_assignments=("a", "a", "a"),
                                         benefit_by_session=(1000, 1000, 2000), transition_uj=100)

    def impact(self, **kwargs):
        fields = dict(request_id="running", remaining_tokens=100, current_layer_mask=7,
                      retained_layer_mask=3, evidence_reused=False, verification_feasible=True,
                      retained_assistance_loss_uj=1100, verification_overhead_uj=200)
        fields.update(kwargs)
        return PhoneLayoutRequestImpact(**fields)

    def select(self, impacts, **kwargs):
        return self.controller.select_phone_layout_candidate(
            (self.current, self.target), current_layout=self.current, minimum_energy_saving_ppm=0,
            request_impacts_by_geometry={self.target.geometry_sha256: impacts}, **kwargs)

    def test_retained_loss_charged_once_not_full_incumbent_again(self):
        selected, _, gains = self.select((self.impact(),))
        self.assertEqual(selected, self.target)
        self.assertEqual(len(gains), 1)
        self.assertEqual(gains[0].current_warm_energy_saved_uj, 500)
        self.assertEqual(gains[0].retained_revalidation_cost_uj, 1300)
        self.assertEqual(gains[0].gain_over_current_uj, 2000 - 500 - 100 - 1300)

    def test_all_affected_requests_count_even_on_retained_pair(self):
        selected, _, gains = self.select((self.impact(), self.impact(request_id="other", retained_assistance_loss_uj=200)))
        self.assertEqual(selected, self.current)
        self.assertEqual(gains[0].retained_revalidation_cost_uj, 1700)

    def test_no_affordable_complete_pair_retains_ready_layout(self):
        selected, reason, _ = self.select((self.impact(verification_feasible=False),))
        self.assertEqual(selected, self.current)
        self.assertEqual(reason, "PHONE_RESIDENCY_REVALIDATION_UNAFFORDABLE")

    def test_physical_degradation_not_vetoed_by_economics(self):
        selected, _, _ = self.select((self.impact(verification_feasible=False),), force=True)
        self.assertEqual(selected, self.target)

    def test_preload_rechecks_opportunity_before_requesting_drain(self):
        state = SimpleNamespace(layout=self.target, generation=2, selection_reason="gain")
        authority = Mock()
        authority.ready_phone_layout.return_value = SimpleNamespace(layout=self.current)
        authority.request_helper_rebind_state.return_value = None
        authority.request_helper_events.return_value = ()
        authority.select_phone_layout_candidate.side_effect = self.controller.select_phone_layout_candidate
        owner = SimpleNamespace(_model_placement_controller=authority, _runtime_controller=Mock(),
            _phone_layout_request_impacts=Mock(return_value=(self.impact(verification_feasible=False),)),
            _runtime_capabilities=SimpleNamespace(minimum_energy_saving_ppm=0))
        owner._runtime_controller.current_tickets.return_value = (
            SimpleNamespace(request=SimpleNamespace(request_id="running")),)
        result = _defer_phone_layout_revalidation(owner, "new-model", state, 100, 40)
        self.assertEqual(result["status"], "DEFERRED")
        self.assertEqual(result["reason"], "PHONE_RESIDENCY_REVALIDATION_UNAFFORDABLE")
        authority.request_helper_rebind.assert_not_called()
        authority.begin_phone_layout_transition.assert_not_called()
        authority.request_helper_rebind_state.return_value = {"target_generation": 2}
        owner._phone_layout_request_impacts.reset_mock()
        self.assertIsNone(_defer_phone_layout_revalidation(owner, "new-model", state, 200, 40))
        owner._phone_layout_request_impacts.assert_not_called()

    def test_queue_cost_refresh_does_not_cancel_issued_maintenance(self):
        authority = Mock()
        target = SimpleNamespace(generation=2, layout=self.target)
        authority.phone_preload_inflight.return_value = False
        authority.preparing_phone_layout.return_value = None
        authority.target_phone_layout.return_value = target
        authority.request_helper_rebind_state.return_value = {"target_generation": 2}
        owner = SimpleNamespace(_model_placement_controller=authority, _runtime_controller=Mock(),
            _runtime_capabilities=object(), _fixed_phone_residency=None,
            _phone_telemetry_deferral=Mock(return_value=None), _phone_queue_demand=Mock(),
            _record_preparing_phone_layout_evaluation=Mock(), _phone_candidate_choice=Mock())
        request = SimpleNamespace(request_id="running")
        owner._runtime_controller.current_tickets.return_value = (SimpleNamespace(request=request),)
        self.assertFalse(_update_phone_residency_portfolio(owner, request, object(), 100))
        owner._record_preparing_phone_layout_evaluation.assert_called_once()
        owner._phone_candidate_choice.assert_not_called()
        authority.propose_phone_layout.assert_not_called()

    def test_compatible_evidence_does_not_charge_verification(self):
        selected, _, gains = self.select((self.impact(evidence_reused=True,
            retained_assistance_loss_uj=0, verification_overhead_uj=0),))
        self.assertEqual(selected, self.target)
        self.assertEqual(gains[0].gain_over_current_uj, 1400)

    def test_rough_ops_do_not_pay_measured_joule_cost(self):
        self.current = replace(self.current, objective_kind="queue_rough_compute_ops")
        self.target = replace(self.target, objective_kind="queue_rough_compute_ops")
        selected, reason, _ = self.select((self.impact(),))
        self.assertEqual(selected, self.current)
        self.assertEqual(reason, "PHONE_RESIDENCY_REVALIDATION_ENERGY_UNKNOWN")
        selected, _, _ = self.select(())
        self.assertEqual(selected, self.target)

    def test_running_decode_is_demand_even_with_empty_waiting_queue(self):
        request = SimpleNamespace(request_id="running", output_tokens=292)
        model = SimpleNamespace(artifact_sha256="artifact-a")
        current = SimpleNamespace(request=request, model=model, dispatch_state="ACQUIRED")
        owner = SimpleNamespace(_runtime_controller=Mock(), _model_placement_controller=Mock(),
                                _arrived_decode_work_by_artifact=lambda active, queued: {
                                    key: active.get(key, 0) + queued.get(key, 0) for key in active.keys() | queued.keys()})
        owner._runtime_controller.current_tickets.return_value = (current,)
        owner._model_placement_controller.remaining_request_decode_tokens.return_value = 29
        demand = _phone_queue_demand(owner, request, model)
        self.assertEqual(demand.queued_count_by_artifact, {})
        self.assertEqual(demand.queued_work_by_artifact, {"artifact-a": 29})

    def test_preview_preserves_session_and_exact_incumbent(self):
        fixture = adaptive.SustainedAssistanceTests(); fixture.setUp()
        fixture.start(candidates=(fixture.full,))
        fixture.window(100); fixture.window(40, 1000)
        before = fixture.controller.checkpoint()
        impact = fixture.controller.preview_helper_replacement(
            "request-a", retained_layer_mask=fixture.full.layer_mask, token_index=fixture.token,
            at_us=fixture.at_us, transition_latency_us=100_000)
        self.assertTrue(impact.evidence_reused)
        self.assertEqual(impact.incremental_cost_uj, 0)
        self.assertEqual(fixture.controller.checkpoint(), before)

    def test_gemma_tail_cannot_afford_new_mask_verification_after_load(self):
        fixture = adaptive.SustainedAssistanceTests(); fixture.setUp()
        fixture.start(candidates=(fixture.full,), output_tokens=292)
        fixture.window(100); fixture.window(40, 1000)
        before = fixture.controller.checkpoint()
        impact = fixture.controller.preview_helper_replacement(
            "request-a", retained_layer_mask=1 << 2, token_index=263,
            at_us=fixture.at_us, transition_latency_us=17_786_110)
        self.assertEqual(impact.remaining_tokens, 29)
        self.assertFalse(impact.evidence_reused)
        self.assertFalse(impact.verification_feasible)
        self.assertEqual(fixture.controller.checkpoint(), before)

    def test_changed_mask_with_sufficient_opportunity_prices_only_retained_loss(self):
        fixture = adaptive.SustainedAssistanceTests(); fixture.setUp()
        fixture.start(candidates=(fixture.full,))
        fixture.window(100); fixture.window(40, 1000)
        impact = fixture.controller.preview_helper_replacement(
            "request-a", retained_layer_mask=1 << 2, token_index=fixture.token,
            at_us=fixture.at_us, transition_latency_us=0)
        self.assertFalse(impact.evidence_reused)
        self.assertTrue(impact.verification_feasible)
        self.assertGreater(impact.retained_assistance_loss_uj, 0)
        self.assertGreater(impact.verification_overhead_uj, 0)
