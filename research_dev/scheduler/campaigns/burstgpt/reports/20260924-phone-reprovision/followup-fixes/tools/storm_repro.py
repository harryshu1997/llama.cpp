"""Reproduce the per-token REPROVISION_RETAINED storm through the real boundary hook.

Run from a tree root: PYTHONPATH=. python3 storm_repro.py"""
from types import SimpleNamespace
from unittest.mock import patch
import collections

from research_dev.scheduler import UnifiedScheduler
from research_dev.scheduler._unified.phone_residency_ops.common import _OfflineLearningDemand
from research_dev.scheduler.tests.test_phone_reprovision_portfolio import (
    PortfolioReprovisionTests, MODEL_A, MODEL_B, SESSIONS, KNOB, demand_row, layout_with_counts, ticket,
)

case = PortfolioReprovisionTests()
a1 = ticket("a-1", MODEL_A, state="ACQUIRED", output_tokens=500)
b1 = ticket("b-1", MODEL_B, output_tokens=132)
scheduler, compiler = case.scheduler([a1, b1])
compiler.phone_residency_demand = lambda model, work: None
compiler.phone_residency_evidence_status = lambda artifact: {"reason": "PHONE_RESIDENCY_ROUTE_EVIDENCE_UNUSABLE"}
for model in (MODEL_A, MODEL_B):
    scheduler._online_learning_phone_demand_cache[model.artifact_sha256] = _OfflineLearningDemand(
        demand_row(model, 100), SESSIONS, "phone-a",
        {"reason": "PHONE_RESIDENCY_LEARNING_EXPLORATION_READY", "source_route_id": "route:" + model.model_id},
    )
case.install_ready(scheduler, layout_with_counts((demand_row(MODEL_A, 100),), {MODEL_A.artifact_sha256: 3}))
remaining = {"a-1": 500}
placement = scheduler._model_placement_controller
placement.remaining_request_decode_tokens = lambda request_id, output_tokens: remaining[request_id]
before = len(scheduler.phone_residency_events())
with patch.object(UnifiedScheduler, "_automated_compiler", return_value=compiler):
    at_us = 100_000_000
    for token in range(60):
        remaining["a-1"] = 500 - token
        scheduler._reevaluate_pending_phone_layout_at_boundary(a1, at_us)
        at_us += 420_000
events = scheduler.phone_residency_events()[before:]
print(len(events), collections.Counter(row.get("reason") for row in events if row["kind"] == "EVALUATED"))
for row in events[:3]:
    print(row.get("reason"), row.get("desktop_reprovision", {}).get("mode"), row.get("observed_at_us"))
