"""Learning-demand decay: unselected phone routes stop reserving phone sessions."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

from research_dev.scheduler import UnifiedScheduleError, UnifiedScheduler
from research_dev.scheduler._unified.phone_residency import (
    _OfflineLearningDemand,
    _PhoneDemandDiscovery,
)

ARTIFACT_A = "sha256:" + "a" * 64
ARTIFACT_B = "sha256:" + "b" * 64
UNUSABLE = {"reason": "PHONE_RESIDENCY_ROUTE_EVIDENCE_UNUSABLE"}


def _ticket(artifact_sha256, request_id, phone_device_id=None):
    plan = None if phone_device_id is None else SimpleNamespace(
        execution_contract=SimpleNamespace(phone_device_id=phone_device_id)
    )
    return SimpleNamespace(
        model=SimpleNamespace(artifact_sha256=artifact_sha256),
        request=SimpleNamespace(request_id=request_id),
        execution_plan=plan,
    )


def _learning(demand, session_id, **changes):
    return _OfflineLearningDemand(
        demand=demand, sessions=(SimpleNamespace(session_id=session_id),),
        helper_id="phone", status={"reason": "PHONE_RESIDENCY_LEARNING_EXPLORATION_READY"},
        **changes,
    )


class LearningDemandDecayTests(unittest.TestCase):
    def setUp(self):
        self.scheduler = UnifiedScheduler.for_runtime_discovery(
            "enforce", learning_demand_decision_window=3
        )

    def _complete_unassisted(self, count):
        for index in range(count):
            self.scheduler._record_phone_route_use(_ticket(ARTIFACT_A, f"host-{index}"))

    def test_decision_window_is_a_positive_integer(self):
        for invalid in (0, -1, True, 3.0, "3"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                UnifiedScheduleError, "decision window"
            ):
                UnifiedScheduler.for_runtime_discovery(
                    "enforce", learning_demand_decision_window=invalid
                )
        self.assertEqual(
            UnifiedScheduler.for_runtime_discovery("enforce")._learning_demand_decision_window, 6
        )

    def test_cold_start_learns_until_the_window_holds_no_phone_selection(self):
        self.assertTrue(self.scheduler._learning_demand_active(ARTIFACT_A))
        self._complete_unassisted(2)
        self.assertTrue(self.scheduler._learning_demand_active(ARTIFACT_A))
        self._complete_unassisted(1)
        self.assertFalse(self.scheduler._learning_demand_active(ARTIFACT_A))
        self.assertIsNone(self.scheduler._learning_phone_demand(
            None, None, SimpleNamespace(artifact_sha256=ARTIFACT_A), 1
        ))
        self.assertTrue(self.scheduler._learning_demand_active(ARTIFACT_B))
        self.scheduler._record_phone_route_use(_ticket(ARTIFACT_A, "assisted", phone_device_id="phone"))
        self.assertTrue(self.scheduler._learning_demand_active(ARTIFACT_A))
        self.assertEqual(
            self.scheduler._phone_route_use_by_artifact[ARTIFACT_A], (False, False, True)
        )

    def test_issued_phone_control_counts_as_a_selection(self):
        controller = self.scheduler._model_placement_controller
        controller.record_request_helper_event(
            "probe", "ASSISTANCE_DECISION", 0, {"selected_fraction_ppm": 500_000}
        )
        controller.record_request_helper_event(
            "baseline", "ASSISTANCE_DECISION", 0, {"selected_fraction_ppm": 0}
        )
        controller.record_request_helper_event(
            "malformed", "FRACTION_APPLIED", 0, {"selected_fraction_ppm": "500000"}
        )
        controller.record_request_helper_event(
            "other-kind", "PREPARATION_READY", 0, {"selected_fraction_ppm": 500_000}
        )
        for request_id in ("baseline", "malformed", "other-kind"):
            self.scheduler._record_phone_route_use(_ticket(ARTIFACT_A, request_id))
        self.assertFalse(self.scheduler._learning_demand_active(ARTIFACT_A))
        self.scheduler._record_phone_route_use(_ticket(ARTIFACT_A, "probe"))
        self.assertTrue(self.scheduler._learning_demand_active(ARTIFACT_A))

    def test_history_rolls_back_with_the_runtime_transaction(self):
        self._complete_unassisted(3)
        checkpoint = self.scheduler._runtime_transaction_checkpoint()
        self.scheduler._record_phone_route_use(_ticket(ARTIFACT_A, "assisted", phone_device_id="phone"))
        self.assertTrue(self.scheduler._learning_demand_active(ARTIFACT_A))
        self.scheduler._restore_runtime_transaction(checkpoint)
        self.assertFalse(self.scheduler._learning_demand_active(ARTIFACT_A))

    def test_decayed_cached_online_demand_is_dropped(self):
        self._complete_unassisted(3)
        self.scheduler._online_learning_phone_demand_cache[ARTIFACT_A] = _learning("demand-a", "HTP0")
        discovered = _PhoneDemandDiscovery(
            demand_rows=(), sessions=None, helper_id=None,
            route_evidence_by_artifact={ARTIFACT_A: UNUSABLE},
        )
        result = self.scheduler._cached_online_learning_phone_discovery(
            SimpleNamespace(queued_work_by_artifact={ARTIFACT_A: 5}), discovered
        )
        self.assertEqual(result.demand_rows, ())
        self.assertNotIn(ARTIFACT_A, self.scheduler._online_learning_phone_demand_cache)

    def test_releasing_model_claims_a_contested_session_domain(self):
        learning = {
            ARTIFACT_A: _learning("demand-a", "HTP0"),
            ARTIFACT_B: _learning("demand-b", "HTP1", release_priority=True),
        }
        discovered = _PhoneDemandDiscovery(
            demand_rows=(), sessions=None, helper_id=None, route_evidence_by_artifact={}
        )
        merged = UnifiedScheduler._merge_learning_phone_discovery(discovered, learning)
        self.assertEqual(merged.demand_rows, ("demand-b",))
        self.assertEqual(merged.sessions, learning[ARTIFACT_B].sessions)
        self.assertEqual(
            merged.route_evidence_by_artifact[ARTIFACT_A]["reason"],
            "PHONE_RESIDENCY_SESSION_DOMAIN_MISMATCH",
        )
        without_release = {
            **learning, ARTIFACT_B: replace(learning[ARTIFACT_B], release_priority=False),
        }
        self.assertEqual(
            UnifiedScheduler._merge_learning_phone_discovery(discovered, without_release).demand_rows,
            ("demand-a",),
        )


if __name__ == "__main__":
    unittest.main()
