"""Shared server probe: an inconclusive pair keeps measuring, reasons are per batch size, and the
request-local follow rule is labelled as such.

The energies mirror dev_v2 coherentEF request 002 (host 73.0 J/token, phone 61.8 J/token, a 15.4 %
saving): with one window per side the 10 % bands overlap, so a single pair cannot decide the phone."""
from dataclasses import replace
import unittest

from research_dev.scheduler._internal.adaptive_decode import AdaptiveDecodeController
from research_dev.scheduler._internal.adaptive_decode_contracts import AdaptiveDecodePolicyAck
from research_dev.scheduler._internal.adaptive_decode_ops.coherence import COHERENCE_REASON, server_verdict
# The module, not its classes: names bound here would be collected and run a second time.
from research_dev.scheduler.tests import test_adaptive_coherence as coherence_tests

HOST = 73_000
PHONE_INCONCLUSIVE = 61_780


class ServerProbeTestCase(unittest.TestCase):
    def setUp(self):
        coherence_tests.AdaptiveCoherenceTests.setUp(self)
        self.config = replace(self.config, server_policy_coherence=True, uncertainty_ppm=100_000,
                              maximum_latency_ppm=1_250_000, maximum_probe_attempts_per_context=2)

    start = coherence_tests.AdaptiveServerBatchVerdictTests.start
    record_baseline_window = coherence_tests.AdaptiveCoherenceTests.record_baseline_window
    window = coherence_tests.AdaptiveServerBatchVerdictTests.window
    group = coherence_tests.AdaptiveServerBatchVerdictTests.group

    def step(self, controller, request_id, slot_id, at_us, energies, *, next_active_batch=None):
        """Record the open window with the energy of its policy and batch; acknowledge any control
        at the window end (no transition window)."""
        session = controller._sessions[request_id]
        policy = session.current_policy
        directive = self.window(controller, request_id, slot_id, at_us,
                                energy=energies[(policy.baseline, session.active_batch)],
                                phone_calls=0 if policy.baseline else 2, next_active_batch=next_active_batch)
        acknowledged = directive
        while acknowledged.control is not None:
            acknowledged = controller.acknowledge(request_id, AdaptiveDecodePolicyAck(
                request_id, slot_id, acknowledged.control.plan_generation,
                session.transition_start_token, at_us, acknowledged.control.policy.policy_hash))
        return directive

    def lead(self, controller, output_tokens=60):
        """A measures one host window alone, then probes the phone (the proposal)."""
        self.start(controller, "request-a", 1, output_tokens)
        probe = self.record_baseline_window(controller, "request-a", 1, 3_000, energy_per_token=HOST)
        self.assertEqual(probe.control.policy, self.full)
        controller.acknowledge("request-a", AdaptiveDecodePolicyAck(
            "request-a", 1, probe.control.plan_generation, 3, 3_000, self.full.policy_hash))

    def server_reason(self, controller, request_id):
        return controller.snapshot(request_id)["server_policy"]["reason"]


class ServerPairResolutionTests(ServerProbeTestCase):
    def energies(self, phone):
        return {(True, 1): HOST, (False, 1): phone}

    def test_inconclusive_single_pair_keeps_measuring_until_the_phone_qualifies(self):
        controller = AdaptiveDecodeController()
        self.lead(controller)
        energies = self.energies(PHONE_INCONCLUSIVE)
        pair = self.step(controller, "request-a", 1, 4_000, energies)
        group = self.group(controller, "request-a")
        # Means favor the phone but the bands overlap: no verdict, the proposal stays pending and
        # the owner measures the host side up to the next resolving count (4 windows).
        self.assertIsNone(server_verdict(group, 1))
        self.assertEqual(group.proposal, self.full)
        self.assertEqual(pair.reason, COHERENCE_REASON)
        self.assertEqual(pair.control.policy, self.baseline)
        self.assertEqual(self.server_reason(controller, "request-a"), "SERVER_PAIR_INCONCLUSIVE")
        self.assertEqual(controller.snapshot("request-a")["zero_assistance_reason"],
                         "SERVER_PAIR_INCONCLUSIVE")
        for at_us in (5_000, 6_000):
            self.assertIsNone(self.step(controller, "request-a", 1, at_us, energies).control)
            self.assertIsNone(server_verdict(group, 1))
        resolved = self.step(controller, "request-a", 1, 7_000, energies)
        self.assertEqual(resolved.control.policy, self.full)
        self.assertEqual(server_verdict(group, 1), self.full)
        self.assertEqual(group.attempts, ())
        self.assertIsNone(self.server_reason(controller, "request-a"))
        self.assertEqual(controller.active_policy("request-a"), self.full)

    def test_host_windows_of_the_resolution_do_not_consume_the_probe_budget(self):
        controller = AdaptiveDecodeController()
        self.lead(controller)
        energies = self.energies(PHONE_INCONCLUSIVE)
        self.step(controller, "request-a", 1, 4_000, energies)
        group = self.group(controller, "request-a")
        spent = group.probe_tokens
        self.step(controller, "request-a", 1, 5_000, energies)
        self.step(controller, "request-a", 1, 6_000, energies)
        self.assertEqual(group.probe_tokens, spent)

    def test_rejection_resting_on_one_host_window_takes_a_second_reference(self):
        controller = AdaptiveDecodeController()
        self.lead(controller)
        energies = self.energies(80_000)
        pair = self.step(controller, "request-a", 1, 4_000, energies)
        group = self.group(controller, "request-a")
        self.assertEqual(pair.control.policy, self.baseline)
        self.assertIsNone(server_verdict(group, 1))
        self.assertEqual(self.server_reason(controller, "request-a"), "SERVER_REFERENCE_BASELINE")
        self.step(controller, "request-a", 1, 5_000, energies)
        self.assertEqual(server_verdict(group, 1), self.baseline)
        self.assertEqual(self.server_reason(controller, "request-a"), "SERVER_PAIR_NOT_IMPROVED")

    def test_bound_resolved_rejection_is_final_at_once(self):
        controller = AdaptiveDecodeController()
        self.lead(controller)
        pair = self.step(controller, "request-a", 1, 4_000, self.energies(95_000))
        group = self.group(controller, "request-a")
        self.assertEqual(pair.control.policy, self.baseline)
        self.assertEqual(server_verdict(group, 1), self.baseline)
        self.assertEqual(self.server_reason(controller, "request-a"), "SERVER_PAIR_NOT_IMPROVED")

    def test_clear_saving_qualifies_the_phone_at_once(self):
        controller = AdaptiveDecodeController()
        self.lead(controller)
        pair = self.step(controller, "request-a", 1, 4_000, self.energies(40_000))
        self.assertIsNone(pair.control)
        self.assertEqual(server_verdict(self.group(controller, "request-a"), 1), self.full)

    def test_unresolved_pair_becomes_the_host_verdict_at_the_attempt_cap(self):
        """A 2.7 % saving cannot pass the bands within the budget: each owner measures until the
        shared budget is spent, and only the second exhausted attempt (the cap) decides."""
        controller = AdaptiveDecodeController()
        self.lead(controller)
        energies = self.energies(71_000)
        group = self.group(controller, "request-a")
        at_us = 4_000
        self.step(controller, "request-a", 1, at_us, energies)
        self.assertIsNone(server_verdict(group, 1))
        self.assertEqual(group.proposal, self.full)
        for _ in range(24):
            at_us += 1_000
            self.step(controller, "request-a", 1, at_us, energies)
            self.assertIsNone(server_verdict(group, 1))
            if group.attempts:
                break
        self.assertEqual(group.attempts, ((1, 1),))
        self.assertIsNone(group.proposal)
        self.assertEqual(self.server_reason(controller, "request-a"), "SERVER_PROBE_BUDGET_EXHAUSTED")
        self.assertIsNone(server_verdict(group, 1))
        controller.complete("request-a", "COMPLETED")
        at_us += 10_000
        retry = self.start(controller, "request-c", 3, 60, first_token_at_us=at_us)
        while retry.control is not None:
            retry = controller.acknowledge("request-c", AdaptiveDecodePolicyAck(
                "request-c", 3, retry.control.plan_generation, 1, at_us, retry.control.policy.policy_hash))
        for _ in range(28):
            at_us += 1_000
            self.step(controller, "request-c", 3, at_us, energies)
            if server_verdict(group, 1) is not None:
                break
        self.assertEqual(server_verdict(group, 1), self.baseline)
        self.assertEqual(group.attempts, ((1, 2),))
        self.assertEqual(self.server_reason(controller, "request-c"), "SERVER_PROBE_BUDGET_EXHAUSTED")


class ServerReasonPerBatchTests(ServerProbeTestCase):
    def test_host_reason_survives_a_phone_verdict_at_another_batch_size(self):
        """coherentEF: verdict[1] = host, later verdict[2] = phone; back at batch 1 the host decisions
        must still report why batch 1 runs the host."""
        controller = AdaptiveDecodeController()
        energies = {(True, 1): HOST, (False, 1): 95_000, (True, 2): 40_000, (False, 2): 25_000}
        self.lead(controller)
        self.step(controller, "request-a", 1, 4_000, energies)
        group = self.group(controller, "request-a")
        self.assertEqual(server_verdict(group, 1), self.baseline)
        self.step(controller, "request-a", 1, 5_000, energies)
        self.start(controller, "request-b", 2, 40, active_batch=2, first_token_at_us=5_500)
        self.step(controller, "request-a", 1, 6_000, energies, next_active_batch=2)
        at_us = 6_500
        for _ in range(12):
            self.step(controller, "request-b", 2, at_us, energies)
            self.step(controller, "request-a", 1, at_us + 500, energies)
            at_us += 1_000
            if server_verdict(group, 2) is not None:
                break
        self.assertEqual(server_verdict(group, 2), self.full)
        controller.complete("request-b", "COMPLETED")
        self.step(controller, "request-a", 1, at_us, energies, next_active_batch=1)
        self.step(controller, "request-a", 1, at_us + 1_000, energies)
        self.assertEqual(controller.active_policy("request-a"), self.baseline)
        snapshot = controller.snapshot("request-a")
        self.assertEqual(snapshot["server_policy"]["reason"], "SERVER_PAIR_NOT_IMPROVED")
        self.assertEqual(snapshot["server_policy"]["reasons"], {"1": "SERVER_PAIR_NOT_IMPROVED"})
        self.assertEqual(snapshot["zero_assistance_reason"], "SERVER_PAIR_NOT_IMPROVED")

    def test_host_reason_of_another_batch_size_is_not_reported_on_the_phone(self):
        controller = AdaptiveDecodeController()
        suite = coherence_tests.AdaptiveServerBatchVerdictTests()
        suite.baseline, suite.full, suite.config = self.baseline, self.full, replace(
            self.config, uncertainty_ppm=10_000, maximum_latency_ppm=1_000_000)
        suite.qualified_owner_with_joining_follower(controller)
        suite.comparison_at_batch_two(controller, host_energy=40)
        # The batch-2 rejection rests on one host window: a second reference decides it.
        self.window(controller, "request-a", 1, 9_500, energy=40)
        group = self.group(controller, "request-a")
        self.assertEqual(server_verdict(group, 2), self.baseline)
        controller.complete("request-b", "COMPLETED")
        alone = self.window(controller, "request-a", 1, 10_500, energy=40, next_active_batch=1)
        self.assertEqual(alone.control.policy, self.full)
        server = controller.snapshot("request-a")["server_policy"]
        self.assertIsNone(server["reason"])
        self.assertEqual(server["reasons"], {"2": "SERVER_PAIR_NOT_IMPROVED"})


class CoTenantFollowLabelTests(unittest.TestCase):
    def setUp(self):
        coherence_tests.AdaptiveCoherenceTests.setUp(self)

    start = coherence_tests.AdaptiveCoherenceTests.start
    record_baseline_window = coherence_tests.AdaptiveCoherenceTests.record_baseline_window
    lead = coherence_tests.AdaptiveCoherenceTests.lead

    def test_request_local_follow_is_not_labelled_as_server_coherence(self):
        """server_policy_coherence off: the request-local rule still follows a co-tenant's phone
        policy, but its decisions must not claim the server-policy mechanism."""
        controller = AdaptiveDecodeController()
        self.assertFalse(self.config.server_policy_coherence)
        self.start(controller, "request-b", 2, 8)
        self.lead(controller)
        directive = self.record_baseline_window(controller, "request-b", 2, 5_000)
        self.assertEqual(directive.control.policy, self.full)
        self.assertEqual(directive.reason, "CO_TENANT_POLICY_FOLLOW")
        self.assertIsNone(controller.snapshot("request-b")["server_policy"])


if __name__ == "__main__":
    unittest.main()
