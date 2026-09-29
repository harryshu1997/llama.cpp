"""Coherent server verdicts keyed by the model's own phone shards (change #4 follow-up).

coherentRP rig run: Gemma 005 (layout generation 9) and 007 (generation 15) ran on the same
24-layer Gemma shards, but the server-policy group was keyed by the generation, so 007 found
no verdict and, at 29 tokens, could not afford a probe (all host). On the base tree the
controller has no layout identity, so the identity tests fail there; the tests without an
identity show the unchanged generation scoping."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

from research_dev.scheduler._internal.adaptive_decode import AdaptiveDecodeController
from research_dev.scheduler._internal.adaptive_decode_contracts import AdaptiveDecodeError, AdaptiveDecodePolicyAck
from research_dev.scheduler._internal.adaptive_decode_ops.coherence import COHERENCE_REASON, server_verdict
from research_dev.scheduler._internal.phone_shards import (
    PhoneShardPlacementError, generate_mixed_ffn_residency_layouts,
)
from research_dev.scheduler._unified.adaptive_decode_control import AdaptiveDecodeControlMixin
from research_dev.scheduler.tests import test_phone_reprovision_portfolio as portfolio
from research_dev.scheduler.tests import test_adaptive_coherence as coherence
from research_dev.scheduler.tests.test_adaptive_decode import ARTIFACT, PLAN
from research_dev.scheduler.tests.test_phone_reprovision_portfolio import (
    LIMIT, MODEL_A, MODEL_B, SESSIONS, demand_row, ticket,
)

try:
    from research_dev.scheduler._internal.phone_shards import artifact_layout_identity_sha256
except ImportError:  # base tree
    artifact_layout_identity_sha256 = None

SHARDS = "sha256:" + "a" * 64  # the model's shards on generations 1 and 2
OTHER_SHARDS = "sha256:" + "b" * 64  # a different layer set
A, B = MODEL_A.artifact_sha256, MODEL_B.artifact_sha256


def geometry(generation):
    return "sha256:" + str(generation % 10) * 64


class ServerVerdictLayoutIdentityTests(unittest.TestCase):
    def setUp(self):
        coherence.AdaptiveCoherenceTests.setUp(self)
        self.config = replace(self.config, server_policy_coherence=True)

    record_baseline_window = coherence.AdaptiveCoherenceTests.record_baseline_window

    def start(self, controller, request_id, slot_id, output_tokens, *, generation, identity=None, **overrides):
        if identity is not None:
            overrides["helper_layout_identity_sha256"] = identity
        return coherence.AdaptiveCoherenceTests.start(
            self, controller, request_id, slot_id, output_tokens, helper_layout_generation=generation,
            helper_layout_geometry_sha256=geometry(generation), **overrides)

    def qualify_phone(self, controller, request_id="request-a", **layout):
        """One owner measures the phone at batch 1 and completes (Gemma 005)."""
        self.start(controller, request_id, 1, 60, **layout)
        leader = self.record_baseline_window(controller, request_id, 1, 3_000)
        controller.acknowledge(request_id, AdaptiveDecodePolicyAck(
            request_id, 1, leader.control.plan_generation, 3, 3_000, self.full.policy_hash))
        self.record_baseline_window(controller, request_id, 1, 4_000, energy_per_token=40, phone_calls=2)
        group = controller.checkpoint()[6][controller.shared_server_policy_key(request_id)]
        self.assertEqual(server_verdict(group, 1), self.full)
        controller.complete(request_id, "COMPLETED")

    def test_recurring_layout_keeps_the_server_verdict_across_a_generation_bump(self):
        controller = AdaptiveDecodeController()
        self.qualify_phone(controller, generation=1, identity=SHARDS)
        # Gemma 007: another generation, same shards, too short to probe on its own
        short = self.start(controller, "request-b", 2, 3, generation=2, identity=SHARDS)
        self.assertEqual(short.reason, COHERENCE_REASON)
        self.assertEqual(short.control.policy, self.full)
        self.assertEqual(controller.shared_server_policy_key("request-b"),
                         (ARTIFACT, self.baseline.desktop_placement_sha256, None, SHARDS))
        snapshot = controller.snapshot("request-b")
        self.assertEqual(snapshot["helper_layout_identity_sha256"], SHARDS)
        self.assertEqual(snapshot["server_policy"]["layout_identity_sha256"], SHARDS)
        self.assertEqual(snapshot["server_policy"]["verdicts"], {"1": self.full.policy_hash})

    def test_a_different_layer_set_starts_fresh(self):
        controller = AdaptiveDecodeController()
        self.qualify_phone(controller, generation=1, identity=SHARDS)
        short = self.start(controller, "request-b", 2, 3, generation=2, identity=OTHER_SHARDS)
        self.assertIsNone(short.control)
        self.assertEqual(controller.active_policy("request-b"), self.baseline)
        group = controller.checkpoint()[6][controller.shared_server_policy_key("request-b")]
        self.assertEqual(group.verdicts, ())
        self.assertEqual(group.owner_request_id, "request-b")

    def test_without_an_identity_a_generation_bump_starts_fresh(self):
        controller = AdaptiveDecodeController()
        self.qualify_phone(controller, generation=1)
        short = self.start(controller, "request-b", 2, 3, generation=2)
        self.assertIsNone(short.control)
        same_generation = self.start(controller, "request-c", 3, 3, generation=1)
        self.assertEqual(same_generation.control.policy, self.full)

    def test_checkpoint_restores_identity_keyed_groups(self):
        controller = AdaptiveDecodeController()
        self.qualify_phone(controller, generation=1, identity=SHARDS)
        controller.restore(controller.checkpoint())
        short = self.start(controller, "request-b", 2, 3, generation=2, identity=SHARDS)
        self.assertEqual(short.control.policy, self.full)

    def test_rebind_keeps_the_group_only_for_unchanged_shards(self):
        controller = AdaptiveDecodeController()
        self.start(controller, "request-a", 1, 60, generation=1, identity=SHARDS)
        key = controller.shared_server_policy_key("request-a")
        arguments = dict(candidates=(self.full,), component_capability_sha256=PLAN, ticket_policy=None,
                         helper_evidence_state="LEARNING")
        controller.helper_rebound("request-a", phone_layout_generation=2, phone_layout_geometry_sha256=geometry(2),
                                  phone_layout_identity_sha256=SHARDS, **arguments)
        self.assertEqual(controller.shared_server_policy_key("request-a"), key)
        controller.helper_rebound("request-a", phone_layout_generation=3, phone_layout_geometry_sha256=geometry(3),
                                  phone_layout_identity_sha256=OTHER_SHARDS, **arguments)
        self.assertEqual(controller.shared_server_policy_key("request-a")[3], OTHER_SHARDS)

    def test_identity_needs_a_helper_generation_and_a_digest(self):
        controller = AdaptiveDecodeController()
        with self.assertRaises(AdaptiveDecodeError):
            coherence.AdaptiveCoherenceTests.start(self, controller, "request-a", 1, 60,
                                                   helper_layout_identity_sha256=SHARDS)
        with self.assertRaises(AdaptiveDecodeError):
            self.start(controller, "request-b", 1, 60, generation=1, identity="not-a-digest")


class DecisionRecordTests(unittest.TestCase):
    """ASSISTANCE_DECISION records through the unified recorder."""

    def record(self, config, **start):
        controller = AdaptiveDecodeController()
        case = SimpleNamespace()
        coherence.AdaptiveCoherenceTests.setUp(case)
        directive = coherence.AdaptiveCoherenceTests.start(
            case, controller, "request-a", 1, 60, config=replace(case.config, **config),
            helper_layout_generation=1, helper_layout_geometry_sha256=geometry(1), **start)
        recorded = []
        owner = SimpleNamespace(
            _adaptive_decode=controller,
            _model_placement_controller=SimpleNamespace(
                record_request_helper_event=lambda *args: recorded.append(args)),
        )
        AdaptiveDecodeControlMixin._record_assistance_decision(owner, "request-a", directive, 1, 1_000)
        self.assertEqual(len(recorded), 1)
        return recorded[0][3]

    def test_coherence_off_record_is_unchanged(self):
        payload = self.record({})
        self.assertNotIn("helper_layout_identity_sha256", payload)
        self.assertIsNone(payload["server_policy"])

    def test_coherent_record_carries_the_layout_identity(self):
        payload = self.record({"server_policy_coherence": True}, helper_layout_identity_sha256=SHARDS)
        self.assertEqual(payload["helper_layout_identity_sha256"], SHARDS)
        self.assertEqual(payload["server_policy"]["layout_identity_sha256"], SHARDS)


def _shape(layout):
    return tuple(sorted((row.session_id, row.artifact_sha256, row.layer_mask) for row in layout.shards))


class ModelLayoutIdentityTests(unittest.TestCase):
    def layouts(self):
        demands = (demand_row(MODEL_A, 100), demand_row(MODEL_B, 100))
        rows = generate_mixed_ffn_residency_layouts(demands, SESSIONS, phone_wide_limit_bytes=LIMIT)
        a_only = next(row for row in rows if _shape(row) == (("HTP0", A, 3), ("HTP1", A, 12)))
        a_full = next(row for row in rows if _shape(row) == (("HTP0", A, 3), ("HTP1", A, 12), ("HTP2", A, 48)))
        grown = generate_mixed_ffn_residency_layouts(
            demands, SESSIONS, phone_wide_limit_bytes=LIMIT, current_shards=a_only.shards)
        with_b = next(row for row in grown if _shape(row) == (("HTP0", A, 3), ("HTP1", A, 12), ("HTP2", B, 3)))
        return a_only, with_b, a_full

    def test_identity_is_the_models_own_shards(self):
        a_only, with_b, a_full = self.layouts()
        self.assertNotEqual(a_only.geometry_sha256, with_b.geometry_sha256)
        self.assertEqual(artifact_layout_identity_sha256(a_only, A), artifact_layout_identity_sha256(with_b, A))
        self.assertNotEqual(artifact_layout_identity_sha256(a_only, A), artifact_layout_identity_sha256(a_full, A))
        with self.assertRaises(PhoneShardPlacementError):
            artifact_layout_identity_sha256(a_only, B)

    def test_scheduler_derives_the_identity_per_generation_only_with_coherence(self):
        a_only, with_b, _ = self.layouts()
        case = portfolio.PortfolioReprovisionTests()
        scheduler, _compiler = case.scheduler([ticket("a-1", MODEL_A, state="ACQUIRED")])
        self.assertEqual(case.install_ready(scheduler, a_only).generation, 1)
        proposed = scheduler._model_placement_controller.propose_phone_layout(
            with_b, workspace_bytes=32, shared_compute_resource_id="phone-htp",
            shared_transport_resource_ids=("phone-functionfs", "phone-usb"), observed_at_us=100_000_000,
        )
        self.assertEqual((proposed.generation, proposed.layout.changed_session_ids), (2, ("HTP2",)))
        a1, b1 = ticket("a-1", MODEL_A, state="ACQUIRED"), ticket("b-1", MODEL_B)
        self.assertIsNone(scheduler._adaptive_layout_identity_sha256(a1, 1))  # coherence off
        scheduler._adaptive_decode_config = replace(scheduler._adaptive_decode_config,
                                                    server_policy_coherence=True)
        identity = scheduler._adaptive_layout_identity_sha256(a1, 1)
        self.assertEqual(identity, artifact_layout_identity_sha256(a_only, A))
        self.assertEqual(scheduler._adaptive_layout_identity_sha256(a1, 2), identity)
        self.assertIsNone(scheduler._adaptive_layout_identity_sha256(b1, 1))  # model not on the phone
        self.assertIsNone(scheduler._adaptive_layout_identity_sha256(a1, 9))  # unknown generation


if __name__ == "__main__":
    unittest.main()
