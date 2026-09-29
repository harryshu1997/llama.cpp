"""Evidence rules of the adaptive decode controller (2026-09-24 phone-rejection diagnosis).

F1a an inconclusive incumbent re-check measures more windows instead of demoting.
F1b eligible host windows re-qualify a measured, non-eliminated candidate.
F2  one host window cannot permanently eliminate a probe candidate.
F3  a window read in a token-stream catch-up burst is not evidence.
F4  a session load on the helper's phone is a measurement disturbance.
"""

from dataclasses import replace
from types import SimpleNamespace
import unittest

from research_dev.scheduler._internal.adaptive_decode import AdaptiveDecodeController
from research_dev.scheduler._internal.adaptive_decode_contracts import (
    AdaptiveDecodeConfig,
    AdaptiveDecodeError,
    AdaptiveDecodePolicy,
    AdaptiveDecodePolicyAck,
    AdaptiveDecodeRawWindowObservation,
    AdaptiveDecodeWindowReceipt,
)
from research_dev.scheduler._unified.adaptive_decode_control import AdaptiveDecodeControlMixin


ARTIFACT = "sha256:" + "1" * 64
PLAN = "sha256:" + "2" * 64
PLACEMENT = "sha256:" + "3" * 64
RID = "request-a"
SLOT = 7
LOAD = "HELPER_PHONE_SESSION_LOAD"


def policy(route_id, columns, layers, *, baseline=False):
    return AdaptiveDecodePolicy(
        route_id=route_id,
        executor_id="desktop" if baseline else "desktop-phone",
        operator_plan_sha256=PLAN,
        desktop_parent_route_id="desktop-control",
        desktop_placement_sha256=PLACEMENT,
        layer_indices=layers,
        layer_mask=sum(1 << value for value in layers),
        columns=columns,
        split_fraction_ppm=0 if baseline else columns * 1000,
        resource_ids=("cpu", "gpu") if baseline else ("cpu", "gpu", "phone", "usb"),
        baseline=baseline,
        predicted_latency_per_token_us=1_000,
        predicted_energy_per_token_uj=100_000 if baseline else 60_000,
    )


BASELINE = policy("desktop-control", 0, (), baseline=True)
PHONE = policy("phone-full", 1000, (2, 3))
CONFIG = AdaptiveDecodeConfig(
    minimum_remaining_tokens=8,
    minimum_window_tokens=4,
    maximum_window_tokens=4,
    maximum_probe_tokens=80,
    maximum_probe_candidates=4,
    measurement_resolution_us=1,
    transition_cost_us=1,
    transition_energy_uj=1,
    minimum_energy_saving_ppm=10_000,
    maximum_latency_ppm=1_250_000,
    uncertainty_ppm=100_000,
    warmup_windows_per_policy=0,
)


class Driver:
    """Token-level driver: every token of a window is reported to boundary()."""

    def __init__(self, config=CONFIG, *, helper_available=True, state="LEARNING", output_tokens=200):
        self.controller = AdaptiveDecodeController()
        self.at_us = 1_000
        self.opened = self.controller.start(
            request_id=RID, ticket_id=RID + ":attempt:0",
            model_artifact_sha256=ARTIFACT, planning_profile_sha256=PLAN,
            baseline=BASELINE, candidates=(PHONE,), output_tokens=output_tokens,
            context_length=64, active_batch=1, deadline_us=10**12, slot_id=SLOT,
            first_token_index=1, first_token_at_us=self.at_us, config=config,
            helper_available=helper_available, helper_evidence_state=state,
        )

    def session(self):
        return self.controller.checkpoint()[1][RID]

    def snapshot(self):
        return self.controller.snapshot(RID)

    def window(self, energy, latency_us=1_000, *, gaps_us=None, disturbance_after=None):
        """Close the open window: energy per token (uJ), gaps between observed tokens (us)."""
        session = self.session()
        start = session.window_start_token
        tokens = session.target_token - start
        gaps = gaps_us or [latency_us] * tokens
        at_us = session.window_start_us
        directive = None
        for index, gap in enumerate(gaps):
            if disturbance_after is not None and index == disturbance_after:
                self.controller.helper_disturbance(RID, reason=LOAD)
            at_us += gap
            directive = self.controller.boundary(
                RID, slot_id=SLOT, token_index=start + index + 1, at_us=at_us)
        boundary = directive.boundary
        phone = not boundary.policy.baseline
        calls = 2 if phone else 0
        self.at_us = at_us
        self.recorded = self.controller.record_window(
            RID, boundary, AdaptiveDecodeRawWindowObservation(
                fleet_energy_uj_by_domain={"fleet": energy * boundary.token_count},
                phone_compute_us=10 if phone else 0, usb_transfer_us=5 if phone else 0,
                rpc_us=2 if phone else 0, exposed_tail_us=1 if phone else 0,
                output_valid=True, evidence_ids=("synthetic:window",),
                energy_boundary_id="synthetic-fleet", energy_attribution_kind="isolated",
                completed_phone_calls=calls, completed_phone_input_rows=calls * 4,
            ))
        self.opened = self.recorded
        if self.recorded.control is not None:
            self.acknowledge(self.recorded.control)
        return self.recorded

    def acknowledge(self, control):
        token = self.session().transition_start_token
        self.at_us += 1
        self.opened = self.controller.acknowledge(RID, AdaptiveDecodePolicyAck(
            request_id=RID, slot_id=SLOT, plan_generation=control.plan_generation,
            applied_token_index=token, applied_at_us=self.at_us,
            policy_hash=control.policy.policy_hash,
        ))
        return self.opened

    def last(self):
        return self.session().records[-1]


class InconclusiveIncumbentTests(unittest.TestCase):
    """F1a"""

    def test_overlapping_bounds_resolve_with_more_windows_and_keep_the_phone(self):
        run = Driver()
        run.window(100_000)                       # host reference
        run.window(80_000, 900)                   # qualifies: upper 88.0 <= 89.1 (kJ-scaled)
        self.assertEqual(run.snapshot()["incumbent_policy_hash"], PHONE.policy_hash)
        self.assertEqual(run.controller.active_policy(RID), PHONE)
        directive = run.window(84_000, 900)       # mean 82.0, upper 90.2 > 89.1: overlap only
        self.assertNotEqual(directive.reason, "INCUMBENT_NO_LONGER_BENEFICIAL")
        snapshot = run.snapshot()
        self.assertEqual(snapshot["stage"], "resolving_evidence")
        self.assertEqual(snapshot["qualification_measurement_plan"]["target_counts"], (4, 4))
        self.assertEqual(snapshot["incumbent_policy_hash"], PHONE.policy_hash)
        self.assertTrue(directive.control.policy.baseline)
        for _ in range(3):
            run.window(100_000)
        run.window(82_000, 900)
        run.window(82_000, 900)
        self.assertEqual(run.controller.active_policy(RID), PHONE)
        self.assertEqual(run.snapshot()["state"], "EXPLOITING")
        self.assertEqual(run.snapshot()["eliminated_policy_reasons"], {})

    def test_decisive_recheck_still_demotes(self):
        run = Driver()
        run.window(100_000)
        run.window(80_000, 900)
        directive = run.window(130_000, 900)      # mean 105.0 > required mean 99.0
        self.assertEqual(directive.reason, "INCUMBENT_NO_LONGER_BENEFICIAL")
        self.assertTrue(directive.control.policy.baseline)


class HostRequalificationTests(unittest.TestCase):
    """F1b"""

    def test_host_windows_requalify_a_measured_candidate(self):
        run = Driver(config=replace(CONFIG, maximum_probe_attempts_per_context=1))
        run.window(100_000)
        directive = run.window(84_000, 900)      # upper 92.4 > 89.1; resolution unaffordable
        self.assertTrue(directive.control.policy.baseline)
        self.assertEqual(run.snapshot()["stage"], "evidence_inconclusive")
        self.assertIsNone(run.snapshot()["incumbent_policy_hash"])
        for _ in range(2):                        # host n = 2, 3: band still 10 %
            self.assertIsNone(run.window(100_000).control)
        directive = run.window(100_000)          # host n = 4: band 5 %, required 94.05
        self.assertEqual(directive.reason, "CANDIDATE_REQUALIFIED")
        self.assertEqual(directive.control.policy, PHONE)
        self.assertEqual(run.snapshot()["incumbent_policy_hash"], PHONE.policy_hash)

    def test_host_windows_never_requalify_a_worse_candidate(self):
        run = Driver(config=replace(CONFIG, maximum_probe_attempts_per_context=1))
        run.window(100_000)
        run.window(98_000, 900)
        for _ in range(8):
            directive = run.window(100_000)
            self.assertIsNone(directive.control)
        self.assertTrue(run.controller.active_policy(RID).baseline)


class SingleReferenceEliminationTests(unittest.TestCase):
    """F2"""

    def test_one_host_window_cannot_eliminate_a_learning_candidate(self):
        run = Driver()
        run.window(40_000, 316)                   # low host reference (catch-up artifact)
        directive = run.window(47_000, 530)       # worse than 40.0 and slower than 1.25 x 316
        self.assertNotIn(PHONE.policy_hash, run.snapshot()["eliminated_policy_reasons"])
        self.assertEqual(run.snapshot()["stage"], "reference_baseline")
        self.assertTrue(directive.control.policy.baseline)
        directive = run.window(80_000, 608)       # second reference: host mean 60.0
        self.assertEqual(run.snapshot()["eliminated_policy_reasons"], {})
        self.assertEqual(directive.control.policy, PHONE)
        self.assertEqual(run.snapshot()["incumbent_policy_hash"], PHONE.policy_hash)

    def test_confirmed_rejection_is_permanent_after_the_second_reference(self):
        run = Driver()
        run.window(40_000, 316)
        run.window(47_000, 530)
        run.window(42_000, 330)
        self.assertEqual(
            run.snapshot()["eliminated_policy_reasons"][PHONE.policy_hash],
            "LEARNING_NO_PAIRED_IMPROVEMENT")
        self.assertTrue(run.controller.active_policy(RID).baseline)

    def test_bound_resolved_single_pair_is_still_eliminated_at_once(self):
        run = Driver()
        run.window(100_000)
        directive = run.window(130_000, 900)      # lower 117.0 >= host upper 110.0
        self.assertEqual(
            run.snapshot()["eliminated_policy_reasons"][PHONE.policy_hash],
            "LEARNING_NO_PAIRED_IMPROVEMENT")
        self.assertEqual(directive.reason, "PROBE_CANDIDATE_REJECTED")


class CatchUpGuardTests(unittest.TestCase):
    """F3"""

    def test_catch_up_window_and_the_window_it_starts_are_not_evidence(self):
        run = Driver(helper_available=False)
        run.window(75_000, 600_000)
        self.assertTrue(run.last().measurement_eligible)
        # dev request 000, W1: a 927 ms gap, then tokens 8 and 9 within 1 ms.
        run.window(41_000, gaps_us=[927_300, 334_100, 600, 700])
        self.assertFalse(run.last().measurement_eligible)
        self.assertEqual(run.last().measurement_ineligible_reason, "TOKEN_STREAM_CATCH_UP")
        run.window(75_000, 600_000)               # starts at the late token
        self.assertEqual(run.last().measurement_ineligible_reason, "TOKEN_STREAM_CATCH_UP")
        run.window(75_000, 600_000)
        self.assertTrue(run.last().measurement_eligible)
        # A stall absorbed inside the window (normal last token) stays evidence.
        run.window(75_000, gaps_us=[1_101_100, 800, 513_200, 504_000])
        self.assertTrue(run.last().measurement_eligible)
        self.assertIsNone(run.last().measurement_ineligible_reason)
        self.assertEqual(
            run.snapshot()["measurement_guard_reasons"],
            {"1": "TOKEN_STREAM_CATCH_UP", "2": "TOKEN_STREAM_CATCH_UP"})

    def test_catch_up_warmup_still_counts_as_warmup(self):
        config = replace(CONFIG, warmup_windows_per_policy=1)
        run = Driver(config=config, helper_available=False)
        run.window(41_000, gaps_us=[927_300, 334_100, 600, 700])
        self.assertFalse(run.last().measurement_eligible)
        self.assertIsNone(run.last().measurement_ineligible_reason)
        self.assertEqual(run.session().warmup_windows_seen_by_policy[BASELINE.policy_hash], 1)
        run.window(75_000, 600_000)
        self.assertEqual(run.last().measurement_ineligible_reason, "TOKEN_STREAM_CATCH_UP")
        run.window(75_000, 600_000)
        self.assertTrue(run.last().measurement_eligible)

    def test_guard_reason_is_serialized_only_when_set(self):
        run = Driver(helper_available=False)
        run.window(75_000, 600_000)
        run.window(41_000, gaps_us=[927_300, 334_100, 600, 700])
        clean, guarded = run.session().records
        self.assertNotIn("measurement_ineligible_reason", clean.to_json())
        self.assertEqual(AdaptiveDecodeWindowReceipt.from_json(guarded.to_json()), guarded)
        with self.assertRaises(AdaptiveDecodeError):
            replace(clean, measurement_ineligible_reason="TOKEN_STREAM_CATCH_UP")


class HelperPhoneLoadTests(unittest.TestCase):
    """F4"""

    def test_session_load_removes_phone_evidence_and_defers_the_probe(self):
        run = Driver()
        run.window(100_000)
        self.assertEqual(run.controller.active_policy(RID), PHONE)
        directive = run.window(80_000, 900, disturbance_after=2)
        self.assertEqual(run.last().measurement_ineligible_reason, LOAD)
        self.assertEqual(directive.reason, "PROBE_INCOMPLETE")
        self.assertTrue(directive.control.policy.baseline)
        self.assertEqual(run.snapshot()["eliminated_policy_reasons"], {})
        directive = run.window(100_000)          # load still running: host windows stay evidence
        self.assertTrue(run.last().measurement_eligible)
        self.assertEqual(directive.reason, LOAD)
        self.assertIsNone(directive.control)
        run.controller.helper_disturbance(RID, reason=None)
        directive = run.window(100_000)
        self.assertEqual(directive.control.policy, PHONE)

    def test_load_on_the_helper_phone_is_reported_and_other_phones_are_not(self):
        def state(session_id, value, endpoint):
            return SimpleNamespace(session_id=session_id, state=value, endpoint=endpoint)

        calls = []
        fake = SimpleNamespace(
            _model_placement_controller=SimpleNamespace(
                request_binding=lambda request_id: {
                    "helper_envelope": {"phone_session_ids": ["HTP0", "HTP1"]}},
                phone_session_states=lambda: states,
            ),
            _adaptive_decode=SimpleNamespace(
                helper_disturbance=lambda request_id, *, reason: calls.append(reason)),
        )
        states = (
            state("HTP0", "READY", "session://op15-phone/HTP0"),
            state("HTP1", "READY", "session://op15-phone/HTP1"),
            state("HTP2", "LOADING", "session://op15-phone/HTP2"),
        )
        AdaptiveDecodeControlMixin._report_helper_phone_disturbance(fake, RID)
        states = (
            state("HTP0", "READY", "session://op15-phone/HTP0"),
            state("HTP2", "VERIFIED", "session://op15-phone/HTP2"),
            state("OTHER", "LOADING", "session://op12-phone/OTHER"),
        )
        AdaptiveDecodeControlMixin._report_helper_phone_disturbance(fake, RID)
        self.assertEqual(calls, [LOAD, None])


if __name__ == "__main__":
    unittest.main()
