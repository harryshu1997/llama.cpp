"""2026-10-08: preflight names the models that have a phone route planning accepts.

The Qwen3-32B Q8_0 two-phone arm pA1q8a passed preflight and ran desktop-only: its Pixel column quantum left no hot
phone route, so the hot phone-assisted routes were PLACEMENT_INFEASIBLE or never generated. ``phone_assistance_ready``
cannot tell, because phones are not resident at preflight (it was false in the working arms too). The rows below
follow the real reports of pA1q8a (broken) and pA1q8e / pA1q4b (working).
"""

from __future__ import annotations

import unittest

from research_dev.scheduler.adapters.preflight import PhysicalPreflightReport, PhysicalPreflightRoute

MODEL = "qwen3-32b-q8_0"


def route(route_id: str, *reasons: str, maturity: str = "SHADOW") -> PhysicalPreflightRoute:
    return PhysicalPreflightRoute(
        model_id=MODEL, route_id=route_id, executor_id="physical:hot:phone-assisted:operator_split",
        device_kinds=("cpu", "gpu", "phone"), maturity=maturity, readiness="NOT_READY", admitted=False,
        rejection_reasons=tuple(sorted(reasons)), evidence_ids=(), required_evidence=())


def report(*routes: PhysicalPreflightRoute) -> PhysicalPreflightReport:
    return PhysicalPreflightReport(catalog_id="c", energy_boundary_id="e", checks=(),
                                   desktop_baselines={MODEL: "desktop"}, phone_routes=routes)


class PhoneRoutePlannableTests(unittest.TestCase):
    def test_route_rejected_only_for_live_state_is_plannable(self) -> None:
        working = report(
            route("auto:hot:phone-assisted:operator_split:750000", "COLD_RESIDENCY_BREAK_EVEN",
                  "RESIDENCY_TRANSITION_ABSENT", "ROUTE_NOT_QUALIFIED"),
            route("auto:hot:composite", "COMPOSITE_COORDINATOR_ABSENT", "EXECUTOR_NOT_READY", "MEMORY_CAPACITY",
                  "PLACEMENT_INFEASIBLE", maturity="QUALIFIED"),
        )
        self.assertFalse(working.phone_assistance_ready)
        self.assertEqual(working.phone_route_plannable_models, (MODEL,))
        self.assertEqual(working.to_json()["phone_route_plannable_models"], [MODEL])

    def test_placement_rejections_and_unvisited_routes_are_not(self) -> None:
        broken = report(
            route("auto:hot:phone-assisted:operator_offload", "PLACEMENT_INFEASIBLE", "RESIDENCY_TRANSITION_ABSENT",
                  "ROUTE_NOT_QUALIFIED"),
            route("auto:hot:phone-assisted:operator_offload:2", "TRANSFER_PLAN_ABSENT", "ROUTE_NOT_QUALIFIED"),
            route("auto:hot:composite", "EXECUTOR_NOT_READY", "MEMORY_CAPACITY", maturity="QUALIFIED"),
            route("catalog:physical:hot:phone-assisted:operator_split:not-visited", "CANDIDATE_NOT_VISITED"),
        )
        self.assertEqual(broken.phone_route_plannable_models, ())
        self.assertEqual(broken.to_json()["phone_route_plannable_models"], [])
        self.assertEqual(broken.status, "PASS")  # the report keeps its status; the rig runner opts in to stop


if __name__ == "__main__":
    unittest.main()
