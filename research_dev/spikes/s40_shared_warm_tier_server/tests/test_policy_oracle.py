#!/usr/bin/env python3

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
S40 = HERE.parent
sys.path.insert(0, str(S40))

from evidence_common import EvidenceError  # noqa: E402
from evidence_common import read_jsonl  # noqa: E402
from policy_oracle import QueueAwarePolicy  # noqa: E402


REQUESTS = read_jsonl(
    S40.parent / "s39_desktop_swap_baseline" / "DESKTOP_REQUESTS.jsonl",
    "frozen_requests",
)


class PolicyOracleTests(unittest.TestCase):
    def policy(self, **kwargs):
        return QueueAwarePolicy(
            models=kwargs.pop("models", ["A", "B"]),
            gpu_model=kwargs.pop("gpu_model", "A"),
            gpu_slots=kwargs.pop("gpu_slots", 1),
            executor_order=kwargs.pop(
                "executor_order", ["GPU", "PHONE", "CPU"]),
            **kwargs,
        )

    def test_gpu_dispatch_is_immediate(self):
        policy = self.policy()
        actions = policy.arrive("r0", "A", 0)
        self.assertEqual(actions[-1]["kind"], "DISPATCH")
        self.assertEqual(actions[-1]["executor_id"], "GPU")

    def test_blocked_oldest_does_not_idle_ready_executor(self):
        policy = self.policy(gpu_slots=1)
        policy.arrive("b0", "B", 0)
        actions = policy.arrive("a1", "A", 1)
        dispatches = [row for row in actions if row["kind"] == "DISPATCH"]
        self.assertEqual([row["request_id"] for row in dispatches], ["a1"])
        self.assertEqual(policy.proposed_target, "B")

    def test_c1_has_no_synthetic_warm_executor(self):
        policy = self.policy()
        policy.arrive("b0", "B", 0)
        self.assertEqual(policy.requests["b0"]["state"], "QUEUED")
        self.assertIsNone(policy.requests["b0"]["executor_id"])
        self.assertEqual(set(policy.executors), {"GPU"})
        self.assertEqual(policy.proposed_target, "B")

    def test_fifo_is_preserved_within_model(self):
        policy = self.policy(gpu_slots=1)
        policy.arrive("a0", "A", 0)
        policy.arrive("a1", "A", 1)
        policy.arrive("a2", "A", 2)
        policy.complete("a0")
        policy.complete("a1")
        dispatches = [
            row["request_id"] for row in policy.actions
            if row["kind"] == "DISPATCH"
        ]
        self.assertEqual(dispatches, ["a0", "a1", "a2"])

    def test_t1_warm_executor_publishes_during_gpu_switch(self):
        policy = self.policy()
        policy.set_executor(
            "PHONE", model_id="B", capacity=2, ready=True)
        policy.arrive("b0", "B", 0)
        self.assertEqual(
            policy.requests["b0"]["executor_id"], "PHONE")
        policy.submit_switch_intent(
            intent_id="i0",
            target_model_id="B",
            trigger_request_id="b0",
            intent_order=0,
        )
        self.assertEqual(policy.proposed_target, "B")
        policy.start_switch()
        policy.complete("b0")
        self.assertEqual(policy.transition_target, "B")
        self.assertEqual(policy.requests["b0"]["state"], "COMPLETED")
        policy.finish_switch()
        self.assertEqual(policy.gpu_model, "B")

    def test_c2_cpu_executor_does_not_suppress_promotion(self):
        policy = self.policy()
        policy.set_executor("CPU", model_id="B", capacity=1, ready=True)
        policy.arrive("b0", "B", 0)
        policy.submit_switch_intent(
            intent_id="i0",
            target_model_id="B",
            trigger_request_id="b0",
            intent_order=0,
        )
        self.assertEqual(policy.requests["b0"]["executor_id"], "CPU")
        self.assertEqual(policy.proposed_target, "B")

    def test_t2_explicitly_disables_promotion(self):
        policy = self.policy(promotion_enabled=False)
        policy.set_executor(
            "PHONE", model_id="B", capacity=1, ready=True)
        policy.arrive("b0", "B", 0)
        policy.submit_switch_intent(
            intent_id="i0",
            target_model_id="B",
            trigger_request_id="b0",
            intent_order=0,
        )
        self.assertIsNone(policy.proposed_target)
        self.assertIn(
            "SWITCH_DISABLED",
            [row["kind"] for row in policy.actions],
        )

    def run_frozen_trace(self, warm: bool, promotion_enabled: bool = True):
        models = ["qwen3-8b-q8_0", "qwen3-14b-q4_k_m"]
        policy = self.policy(
            models=models,
            gpu_model="qwen3-8b-q8_0",
            gpu_slots=8,
            promotion_enabled=promotion_enabled,
        )
        if warm:
            policy.set_executor(
                "WARM",
                model_id="qwen3-14b-q4_k_m",
                capacity=8,
                ready=True,
            )
        first_switch_arrival_us = None
        for order, request in enumerate(REQUESTS):
            policy.arrive(
                request["event_id"],
                request["model_id"],
                order,
            )
            while True:
                progressed = False
                if policy.can_start_switch():
                    if first_switch_arrival_us is None:
                        first_switch_arrival_us = request["arrival_us"]
                    old_model = policy.gpu_model
                    policy.start_switch()
                    progressed = True
                inflight = [
                    (
                        record["demand"].arrival_order,
                        request_id,
                    )
                    for request_id, record in policy.requests.items()
                    if record["state"] == "IN_FLIGHT"
                ]
                for _, request_id in sorted(inflight):
                    policy.complete(request_id)
                    progressed = True
                if policy.transition_target is not None \
                        and policy.drain_complete():
                    policy.finish_switch()
                    if warm:
                        policy.set_executor(
                            "WARM",
                            model_id=old_model,
                            capacity=8,
                            ready=True,
                        )
                    progressed = True
                if not progressed:
                    break
        policy.finalize()
        return policy, first_switch_arrival_us

    def test_exact_74_request_c1_trace_is_live(self):
        policy, first_switch_arrival_us = self.run_frozen_trace(False)
        self.assertEqual(first_switch_arrival_us, 1_950_000)
        self.assertEqual(
            policy.terminal_counts(),
            {"COMPLETED": 74, "STRANDED": 0},
        )
        starts = [
            row for row in policy.actions
            if row["kind"] == "SWITCH_STARTED"
        ]
        self.assertEqual(len(starts), 17)

    def test_exact_74_request_warm_trace_promotes_and_is_live(self):
        policy, first_switch_arrival_us = self.run_frozen_trace(True)
        self.assertEqual(first_switch_arrival_us, 1_950_000)
        self.assertEqual(
            policy.terminal_counts(),
            {"COMPLETED": 74, "STRANDED": 0},
        )
        starts = [
            row for row in policy.actions
            if row["kind"] == "SWITCH_STARTED"
        ]
        self.assertEqual(len(starts), 17)
        warm_dispatches = [
            row for row in policy.actions
            if row["kind"] == "DISPATCH"
            and row["executor_id"] == "WARM"
        ]
        self.assertEqual(len(warm_dispatches), 17)

    def test_exact_74_request_t2_trace_never_promotes(self):
        policy, first_switch_arrival_us = self.run_frozen_trace(
            True, promotion_enabled=False)
        self.assertIsNone(first_switch_arrival_us)
        self.assertEqual(
            policy.terminal_counts(),
            {"COMPLETED": 74, "STRANDED": 0},
        )
        self.assertNotIn(
            "SWITCH_STARTED",
            [row["kind"] for row in policy.actions],
        )

    def test_unstarted_obsolete_target_is_coalesced(self):
        policy = self.policy(models=["A", "B", "C"])
        policy.arrive("b0", "B", 0)
        policy.arrive("c0", "C", 1)
        self.assertEqual(policy.proposed_target, "B")
        policy.cancel_queued("b0")
        self.assertEqual(policy.proposed_target, "C")
        coalesced = [
            row for row in policy.actions
            if row["kind"] == "SWITCH_COALESCED"
        ]
        self.assertTrue(coalesced)
        self.assertEqual(coalesced[-1]["old_target_model_id"], "B")
        self.assertEqual(coalesced[-1]["target_model_id"], "C")

    def test_active_switch_is_immutable(self):
        policy = self.policy(models=["A", "B", "C"])
        policy.arrive("b0", "B", 0)
        policy.start_switch()
        policy.arrive("c0", "C", 1)
        self.assertEqual(policy.transition_target, "B")
        with self.assertRaisesRegex(EvidenceError, "already active"):
            policy.start_switch()

    def test_proposal_drains_gpu_under_continuous_source_demand(self):
        policy = self.policy()
        policy.arrive("a0", "A", 0)
        policy.arrive("b0", "B", 1)
        self.assertEqual(policy.proposed_target, "B")
        self.assertTrue(policy.can_start_switch())
        policy.start_switch()
        for order in range(2, 7):
            policy.arrive(f"a{order}", "A", order)
        dispatched = [
            row["request_id"] for row in policy.actions
            if row["kind"] == "DISPATCH"
        ]
        self.assertEqual(dispatched, ["a0"])
        policy.complete("a0")
        self.assertTrue(policy.drain_complete())
        policy.finish_switch()
        self.assertEqual(policy.requests["b0"]["executor_id"], "GPU")
        self.assertEqual(policy.proposed_target, "A")
        policy.complete("b0")
        self.assertTrue(policy.can_start_switch())
        policy.start_switch()
        self.assertTrue(policy.drain_complete())
        policy.finish_switch()
        for request_id in ("a2", "a3", "a4", "a5", "a6"):
            policy.complete(request_id)
        self.assertEqual(
            policy.terminal_counts(),
            {"COMPLETED": 7, "STRANDED": 0},
        )

    def test_duplicate_request_is_rejected(self):
        policy = self.policy()
        policy.arrive("a0", "A", 0)
        with self.assertRaisesRegex(EvidenceError, "duplicate ID"):
            policy.arrive("a0", "A", 1)

    def test_finalize_accounts_every_request(self):
        policy = self.policy()
        policy.arrive("a0", "A", 0)
        policy.arrive("b0", "B", 1)
        policy.complete("a0")
        policy.finalize()
        self.assertEqual(
            policy.terminal_counts(),
            {"COMPLETED": 1, "STRANDED": 1},
        )


if __name__ == "__main__":
    unittest.main()
