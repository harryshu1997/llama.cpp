"""Batch-growth verdict inheritance (adaptive_decode_overrides.batch_growth_verdict_inheritance).

Hardware runs s1b and s1d (2026-09-28): Gemma 002 decoded on the phone under a batch-1 verdict;
Gemma 006 joined the server, which made the batch 2 while 006 was still prefilling. The owner ran
the batch-1 verdict as the batch-2 proposal, but no batch-2 window was comparable (one decoding
slot at batch 2), so the 80-token shared budget ran out without evidence and the exhausted probe
returned the server to the host: both requests decoded at fraction 0 for the whole ~200 s co-decode
(SERVER_PROBE_BUDGET_EXHAUSTED 107x). Decode is bandwidth-bound on every stage (desktop step 611 ms
at batch 1 and 615-632 ms at batch 4; OnePlus 15 FFN call 9.6 ms for 1 row and 12.7 ms for 4), so a
phone win at batch b is a win at b+1; with the option the larger composition keeps the inherited
phone policy, and only measured evidence at that composition returns it to the host.
"""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from research_dev.scheduler._internal.adaptive_decode import AdaptiveDecodeController
from research_dev.scheduler._internal.adaptive_decode_contracts import (
    AdaptiveDecodeConfig, AdaptiveDecodeError,
)
from research_dev.scheduler._internal.adaptive_decode_ops.coherence import (
    COHERENCE_REASON, INHERITED_REASON, _growth_inheritance, inherited_from_batch, server_verdict,
)
from research_dev.scheduler._unified.adaptive_decode_control import AdaptiveDecodeControlMixin
# The module, not its TestCase classes: names bound here would run their tests again.
from research_dev.scheduler.tests import test_adaptive_coherence as fixtures

EXHAUSTED = "SERVER_PROBE_BUDGET_EXHAUSTED"


class BatchGrowthInheritanceCase(unittest.TestCase):
    """The fixtures of AdaptiveServerBatchVerdictTests; `inherit` switches the option."""

    inherit = True

    def setUp(self):
        fixtures.AdaptiveServerBatchVerdictTests.setUp(self)
        self.config = replace(self.config, batch_growth_verdict_inheritance=self.inherit)

    start = fixtures.AdaptiveServerBatchVerdictTests.start
    record_baseline_window = fixtures.AdaptiveCoherenceTests.record_baseline_window
    lead = fixtures.AdaptiveCoherenceTests.lead
    window = fixtures.AdaptiveServerBatchVerdictTests.window
    ack = fixtures.AdaptiveServerBatchVerdictTests.ack
    group = fixtures.AdaptiveServerBatchVerdictTests.group
    qualified_owner_with_joining_follower = (
        fixtures.AdaptiveServerBatchVerdictTests.qualified_owner_with_joining_follower)
    comparison_at_batch_two = fixtures.AdaptiveServerBatchVerdictTests.comparison_at_batch_two

    def owner_with_batch_one_verdict(self, controller):
        """A qualifies the phone alone: the batch-1 verdict of the server."""
        leader = self.lead(controller)
        self.ack(controller, "request-a", 1, leader.control, 3, 3_000)
        self.assertIsNone(self.window(controller, "request-a", 1, 4_000, energy=40,
                                      phone_calls=2).control)
        group = self.group(controller, "request-a")
        self.assertEqual(server_verdict(group, 1), self.full)
        return group

    def join_while_prefilling(self, controller):
        """s1b/s1d: the batch grows to 2 while the joiner prefills, so the owner is the only
        decoding slot of a batch-2 server and none of its windows is comparable; its phone windows
        spend the whole shared budget (20 tokens here, 2 per window). Returns the last directive."""
        group = self.owner_with_batch_one_verdict(controller)
        joined = self.window(controller, "request-a", 1, 5_000, energy=40, phone_calls=2,
                             next_active_batch=2)
        self.assertIsNone(joined.control)
        self.assertEqual(group.proposal, self.full)
        self.assertEqual(controller.active_policy("request-a"), self.full)
        spent = None
        for step in range(10):
            spent = self.window(controller, "request-a", 1, 6_000 + step * 1_000, energy=40,
                                phone_calls=2)
            self.assertFalse(controller._sessions["request-a"].records[-1].measurement_eligible)
            if step < 9:
                self.assertIsNone(spent.control)
                self.assertEqual(controller.active_policy("request-a"), self.full)
        return group, spent


class JoinWhilePrefillingWithoutInheritanceTests(BatchGrowthInheritanceCase):
    """The option absent: today's behaviour, pinned as the red reference of the defect."""

    inherit = False

    def test_exhausted_budget_moves_the_pair_to_the_host(self):
        controller = AdaptiveDecodeController()
        group, exhausted = self.join_while_prefilling(controller)
        self.assertEqual(exhausted.reason, COHERENCE_REASON)
        self.assertEqual(exhausted.control.policy, self.baseline)
        self.assertEqual(controller.snapshot("request-a")["zero_assistance_reason"], EXHAUSTED)
        self.assertIsNone(server_verdict(group, 2))
        self.assertEqual(group.attempts, ((2, 1),))
        self.assertEqual(group.inherited, ())
        self.assertNotIn("inherited_from_batch", controller.snapshot("request-a")["server_policy"])
        self.ack(controller, "request-a", 1, exhausted.control, 27, 15_100)
        joiner = self.start(controller, "request-b", 2, 60, active_batch=2, first_token_at_us=15_500)
        self.assertEqual(joiner.reason, COHERENCE_REASON)
        self.assertIsNone(joiner.control)
        self.assertEqual(controller.active_policy("request-b"), self.baseline)
        self.window(controller, "request-a", 1, 16_000, energy=100)
        self.assertEqual(controller.active_policy("request-a"), self.baseline)
        self.assertEqual(controller.snapshot("request-a")["zero_assistance_reason"], EXHAUSTED)


class JoinWhilePrefillingWithInheritanceTests(BatchGrowthInheritanceCase):
    def test_exhausted_budget_keeps_the_inherited_phone_policy_for_both(self):
        controller = AdaptiveDecodeController()
        group, exhausted = self.join_while_prefilling(controller)
        self.assertIsNone(exhausted.control)
        self.assertEqual(exhausted.reason, INHERITED_REASON)
        self.assertEqual(controller.active_policy("request-a"), self.full)
        self.assertEqual(server_verdict(group, 2), self.full)
        self.assertEqual(group.inherited, ((2, 1),))
        self.assertEqual(group.attempts, ())
        self.assertIsNone(group.proposal)
        snapshot = controller.snapshot("request-a")
        self.assertIsNone(snapshot["zero_assistance_reason"])
        self.assertEqual(snapshot["server_policy"]["inherited_from_batch"], 1)
        self.assertEqual(snapshot["server_policy"]["inherited_batches"], {"2": 1})
        self.assertEqual(snapshot["server_policy"]["verdict_fractions_ppm"],
                         {"1": 1_000_000, "2": 1_000_000})
        joiner = self.start(controller, "request-b", 2, 60, active_batch=2, first_token_at_us=15_500)
        self.assertEqual(joiner.reason, INHERITED_REASON)
        self.assertEqual(joiner.control.policy, self.full)
        self.ack(controller, "request-b", 2, joiner.control, 1, 15_600)
        later = self.window(controller, "request-a", 1, 16_000, energy=40, phone_calls=2)
        self.assertIsNone(later.control)
        self.assertEqual(later.reason, INHERITED_REASON)
        self.assertEqual(controller.active_policy("request-b"), self.full)

    def test_decisions_before_the_exhaustion_carry_the_inherited_label(self):
        controller = AdaptiveDecodeController()
        self.owner_with_batch_one_verdict(controller)
        joined = self.window(controller, "request-a", 1, 5_000, energy=40, phone_calls=2,
                             next_active_batch=2)
        self.assertEqual(joined.reason, INHERITED_REASON)
        group = self.group(controller, "request-a")
        self.assertEqual(group.inherited, ((2, 1),))
        self.assertIsNone(server_verdict(group, 2))
        self.assertEqual(inherited_from_batch(group, 2), 1)
        self.assertEqual(controller.snapshot("request-a")["server_policy"]["inherited_from_batch"], 1)

    def test_measured_loss_at_the_larger_batch_overturns_the_inherited_policy(self):
        """Guard: the like-for-like probe at batch 2 still runs; a measured rejection decides the
        host for batch 2 and both slots leave the phone."""
        controller = AdaptiveDecodeController()
        group = self.qualified_owner_with_joining_follower(controller)
        self.assertEqual(group.inherited, ((2, 1),))
        reference = self.comparison_at_batch_two(controller, host_energy=40)
        self.assertIsNone(reference.control)
        decided = self.window(controller, "request-a", 1, 10_000, energy=40)
        self.assertIsNone(decided.control)
        self.assertEqual(server_verdict(group, 2), self.baseline)
        self.assertEqual(group.inherited, ())
        self.assertEqual(controller.snapshot("request-a")["zero_assistance_reason"],
                         "SERVER_PAIR_NOT_IMPROVED")
        follower = self.window(controller, "request-b", 2, 10_500, energy=40)
        self.assertIsNone(follower.control)
        self.assertEqual(controller.active_policy("request-b"), self.baseline)
        self.assertEqual(server_verdict(group, 1), self.full)

    def test_measured_win_at_the_larger_batch_replaces_the_inherited_decision(self):
        controller = AdaptiveDecodeController()
        group = self.qualified_owner_with_joining_follower(controller)
        decided = self.comparison_at_batch_two(controller, host_energy=100)
        self.assertEqual(decided.reason, COHERENCE_REASON)
        self.assertEqual(decided.control.policy, self.full)
        self.assertEqual(server_verdict(group, 2), self.full)
        self.assertEqual(group.inherited, ())

    def test_monitored_loss_after_the_exhaustion_returns_the_server_to_the_host(self):
        """Guard: the inherited verdict is monitored like a measured one; batch-2 phone windows
        that the batch-2 host evidence dominates eliminate it."""
        controller = AdaptiveDecodeController()
        group, exhausted = self.join_while_prefilling(controller)
        self.assertEqual(server_verdict(group, 2), self.full)
        host = next(row for row in controller._sessions["request-a"].records if row.policy.baseline)
        group.records = (*group.records, replace(host, request_id="request-z", active_batch=2))
        joiner = self.start(controller, "request-b", 2, 60, active_batch=2, first_token_at_us=15_500)
        self.ack(controller, "request-b", 2, joiner.control, 1, 15_600)
        self.window(controller, "request-a", 1, 16_000, energy=400, phone_calls=2)
        self.window(controller, "request-b", 2, 16_500, energy=400, phone_calls=2)
        decided = self.window(controller, "request-a", 1, 17_000, energy=400, phone_calls=2)
        self.assertEqual(decided.control.policy, self.baseline)
        self.assertEqual(server_verdict(group, 2), self.baseline)
        self.assertEqual(group.inherited, ())
        self.assertEqual(controller.snapshot("request-a")["zero_assistance_reason"],
                         "SERVER_ENERGY_DOMINATED")
        self.assertEqual(server_verdict(group, 1), self.full)

    def test_phone_failure_at_the_larger_batch_ends_the_inheritance(self):
        controller = AdaptiveDecodeController()
        group, _ = self.join_while_prefilling(controller)
        joiner = self.start(controller, "request-b", 2, 60, active_batch=2, first_token_at_us=15_500)
        self.ack(controller, "request-b", 2, joiner.control, 1, 15_600)
        failed = self.window(controller, "request-b", 2, 16_000, energy=40, phone_calls=2,
                             failure_reason="phone_window_failed")
        self.assertEqual(failed.control.policy, self.baseline)
        self.assertEqual(server_verdict(group, 2), self.baseline)
        self.assertEqual(group.inherited, ())
        owner = self.window(controller, "request-a", 1, 16_500, energy=40, phone_calls=2)
        self.assertEqual(owner.reason, COHERENCE_REASON)
        self.assertEqual(owner.control.policy, self.baseline)

    def test_shrinking_batch_uses_the_smaller_batch_verdict(self):
        controller = AdaptiveDecodeController()
        group, _ = self.join_while_prefilling(controller)
        joiner = self.start(controller, "request-b", 2, 60, active_batch=2, first_token_at_us=15_500)
        self.ack(controller, "request-b", 2, joiner.control, 1, 15_600)
        self.window(controller, "request-b", 2, 16_000, energy=40, phone_calls=2)
        controller.complete("request-b", "COMPLETED")
        alone = self.window(controller, "request-a", 1, 17_000, energy=40, phone_calls=2,
                            next_active_batch=1)
        self.assertIsNone(alone.control)
        self.assertEqual(alone.reason, COHERENCE_REASON)
        self.assertEqual(controller.active_policy("request-a"), self.full)
        self.assertEqual(server_verdict(group, 2), self.full)
        self.assertEqual(group.inherited, ((2, 1),))
        self.assertIsNone(controller.snapshot("request-a")["server_policy"]["inherited_from_batch"])
        regrown = self.window(controller, "request-a", 1, 18_000, energy=40, phone_calls=2,
                              next_active_batch=2)
        self.assertEqual(regrown.reason, INHERITED_REASON)
        self.assertEqual(controller.active_policy("request-a"), self.full)


class AssistanceDecisionRecordTests(BatchGrowthInheritanceCase):
    """RESULT `request_helper_events`: ASSISTANCE_DECISION carries the label and its source."""

    def recorded(self, controller, request_id, directive, token_index, at_us):
        placement = Mock()
        owner = SimpleNamespace(_adaptive_decode=controller, _model_placement_controller=placement)
        AdaptiveDecodeControlMixin._record_assistance_decision(
            owner, request_id, directive, token_index, at_us)
        (call,) = placement.record_request_helper_event.call_args_list
        self.assertEqual(call.args[:3], (request_id, "ASSISTANCE_DECISION", at_us))
        return call.args[3]

    def test_inherited_decision_records_its_source_batch(self):
        controller = AdaptiveDecodeController()
        _, exhausted = self.join_while_prefilling(controller)
        event = self.recorded(controller, "request-a", exhausted, 27, 15_000)
        self.assertEqual(event["reason"], INHERITED_REASON)
        self.assertEqual(event["selected_fraction_ppm"], 1_000_000)
        self.assertEqual(event["inherited_from_batch"], 1)
        self.assertEqual(event["server_policy"]["inherited_batches"], {"2": 1})

    def test_without_the_option_the_record_is_unchanged(self):
        self.config = replace(self.config, batch_growth_verdict_inheritance=False)
        controller = AdaptiveDecodeController()
        _, exhausted = self.join_while_prefilling(controller)
        event = self.recorded(controller, "request-a", exhausted, 27, 15_000)
        self.assertEqual(event["reason"], COHERENCE_REASON)
        self.assertEqual(event["selected_fraction_ppm"], 0)
        self.assertNotIn("inherited_from_batch", event)
        self.assertNotIn("inherited_batches", event["server_policy"])
        self.assertNotIn("inherited_from_batch", event["server_policy"])


class InheritanceSourceTests(BatchGrowthInheritanceCase):
    """Which verdict a composition inherits: only the nearest smaller phone verdict of its group."""

    def session_at(self, controller, active_batch):
        self.start(controller, "request-a", 1, 60)
        session = controller._sessions["request-a"]
        session.active_batch = active_batch
        return session, self.group(controller, "request-a")

    def test_only_a_smaller_phone_verdict_is_inherited(self):
        cases = (
            ("smaller phone verdict", 2, ((1, "full"),), (1, "full")),
            ("nearest smaller phone verdict", 3, ((1, "full"),), (1, "full")),
            ("larger verdicts are never inherited", 1, ((2, "full"),), None),
            ("host verdict at the nearest smaller batch blocks", 3,
             ((1, "full"), (2, "baseline")), None),
            ("host verdict alone", 2, ((1, "baseline"),), None),
            ("no verdict", 2, (), None),
        )
        for label, batch, verdicts, expected in cases:
            with self.subTest(label):
                controller = AdaptiveDecodeController()
                session, group = self.session_at(controller, batch)
                group.verdicts = tuple((size, getattr(self, name)) for size, name in verdicts)
                result = _growth_inheritance(controller, session, group)
                self.assertEqual(result, None if expected is None
                                 else (expected[0], getattr(self, expected[1])))

    def test_eliminated_or_helper_unavailable_session_inherits_nothing(self):
        controller = AdaptiveDecodeController()
        session, group = self.session_at(controller, 2)
        group.verdicts = ((1, self.full),)
        session.eliminated_policy_reasons[self.full.policy_hash] = "LATENCY_BOUND_EXCEEDED"
        self.assertIsNone(_growth_inheritance(controller, session, group))
        session.eliminated_policy_reasons.clear()
        session.helper_available = False
        self.assertIsNone(_growth_inheritance(controller, session, group))

    def test_option_off_inherits_nothing(self):
        self.config = replace(self.config, batch_growth_verdict_inheritance=False)
        controller = AdaptiveDecodeController()
        session, group = self.session_at(controller, 2)
        group.verdicts = ((1, self.full),)
        self.assertIsNone(_growth_inheritance(controller, session, group))

    def test_verdicts_never_cross_the_server_policy_key(self):
        """A different layout geometry, desktop placement or model artifact is another group."""
        placement = "sha256:" + "9" * 64
        variants = (
            ("geometry", {"helper_layout_geometry_sha256": "sha256:" + "8" * 64}),
            ("desktop placement", {
                "baseline": replace(self.baseline, desktop_placement_sha256=placement),
                "candidates": (replace(self.full, desktop_placement_sha256=placement),)}),
            ("model artifact", {"model_artifact_sha256": "sha256:" + "5" * 64}),
        )
        for label, overrides in variants:
            with self.subTest(label):
                controller = AdaptiveDecodeController()
                owner_group = self.owner_with_batch_one_verdict(controller)
                other = fixtures.AdaptiveCoherenceTests.start(
                    self, controller, "request-c", 3, 60, active_batch=2,
                    first_token_at_us=4_500, helper_layout_generation=1,
                    **{"helper_layout_geometry_sha256": "sha256:" + "7" * 64, **overrides})
                self.assertIsNone(other.control)
                self.assertNotEqual(other.reason, INHERITED_REASON)
                key = controller.shared_server_policy_key("request-c")
                self.assertNotEqual(key, controller.shared_server_policy_key("request-a"))
                group = controller._server_policies[key]
                self.assertEqual(group.verdicts, ())
                self.assertEqual(group.inherited, ())
                self.assertIsNone(group.proposal)
                self.assertEqual(owner_group.inherited, ())


class ConfigurationTests(unittest.TestCase):
    def test_option_is_typed_and_requires_server_policy_coherence(self):
        for value in (1, "true", None):
            with self.subTest(value=value), self.assertRaises(AdaptiveDecodeError):
                AdaptiveDecodeConfig(server_policy_coherence=True,
                                     batch_growth_verdict_inheritance=value)
        with self.assertRaises(AdaptiveDecodeError):
            AdaptiveDecodeConfig(batch_growth_verdict_inheritance=True)
        enabled = AdaptiveDecodeConfig(server_policy_coherence=True,
                                       batch_growth_verdict_inheritance=True)
        self.assertIs(enabled.to_json()["batch_growth_verdict_inheritance"], True)

    def test_option_absent_keeps_the_serialized_configuration(self):
        self.assertNotIn("batch_growth_verdict_inheritance", AdaptiveDecodeConfig().to_json())
        self.assertNotIn("batch_growth_verdict_inheritance",
                         AdaptiveDecodeConfig(server_policy_coherence=True).to_json())


if __name__ == "__main__":
    unittest.main()
