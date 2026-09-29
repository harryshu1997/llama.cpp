"""Option-off digest guard: run the same scenarios on a tree and print digests to compare base vs root.

(a) reprovisioning knob off: 40 decode boundaries of the storm scenario plus a knob-off portfolio
    evaluation -> digest of every phone-layout event;
(b) server_policy_coherence off: lead / follow / measured phone window / completion / next request
    through the controller and the unified ASSISTANCE_DECISION recorder -> digest of directives and
    decision payloads.
Usage (tree root as cwd): PYTHONPATH=.:gguf-py python3 option_off_digest.py"""
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

from research_dev.scheduler import UnifiedScheduler
from research_dev.scheduler._internal.adaptive_decode import AdaptiveDecodeController
from research_dev.scheduler._internal.adaptive_decode_contracts import AdaptiveDecodePolicyAck
from research_dev.scheduler._unified.adaptive_decode_control import AdaptiveDecodeControlMixin
from research_dev.scheduler._unified.phone_residency_ops.common import _OfflineLearningDemand
from research_dev.scheduler.tests import test_adaptive_coherence as coherence
from research_dev.scheduler.tests import test_phone_reprovision_portfolio as portfolio


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=repr).encode()).hexdigest()


def knob_off_events():
    case = portfolio.PortfolioReprovisionTests()
    a1 = portfolio.ticket("a-1", portfolio.MODEL_A, state="ACQUIRED", output_tokens=500)
    b1 = portfolio.ticket("b-1", portfolio.MODEL_B, output_tokens=132)
    scheduler, compiler = case.scheduler([a1, b1], knob=None)
    compiler.phone_residency_demand = lambda model, work: None
    compiler.phone_residency_evidence_status = lambda artifact: {"reason": "PHONE_RESIDENCY_ROUTE_EVIDENCE_UNUSABLE"}
    for model in (portfolio.MODEL_A, portfolio.MODEL_B):
        scheduler._online_learning_phone_demand_cache[model.artifact_sha256] = _OfflineLearningDemand(
            portfolio.demand_row(model, 100), portfolio.SESSIONS, "phone-a",
            {"reason": "PHONE_RESIDENCY_LEARNING_EXPLORATION_READY", "source_route_id": "route:" + model.model_id})
    case.install_ready(scheduler, portfolio.layout_with_counts(
        (portfolio.demand_row(portfolio.MODEL_A, 100),), {portfolio.A: 3}))
    remaining = {"a-1": 500}
    scheduler._model_placement_controller.remaining_request_decode_tokens = (
        lambda request_id, output_tokens: remaining.get(request_id, output_tokens))
    with patch.object(UnifiedScheduler, "_automated_compiler", return_value=compiler):
        at_us = 100_000_000
        for _ in range(40):
            remaining["a-1"] -= 1
            scheduler._reevaluate_pending_phone_layout_at_boundary(a1, at_us)
            at_us += 420_000
    # the knob-off portfolio test scenario
    tickets = [portfolio.ticket("a-1", portfolio.MODEL_A, state="ACQUIRED"), portfolio.ticket("b-1", portfolio.MODEL_B)]
    other, other_compiler = case.scheduler(tickets, knob=None)
    split = portfolio.layout_with_counts(
        (portfolio.demand_row(portfolio.MODEL_A, 100), portfolio.demand_row(portfolio.MODEL_B, 100)),
        {portfolio.A: 2, portfolio.B: 1})
    case.install_ready(other, split)
    case.evaluate(other, other_compiler, tickets[0], portfolio.MODEL_A, 61_000_000)
    return [dict(row) for row in scheduler.phone_residency_events()] + [
        dict(row) for row in other.phone_residency_events()]


def coherence_off_records():
    case = SimpleNamespace()
    coherence.AdaptiveCoherenceTests.setUp(case)
    controller = AdaptiveDecodeController()
    recorded = []
    owner = SimpleNamespace(_adaptive_decode=controller, _model_placement_controller=SimpleNamespace(
        record_request_helper_event=lambda *args: recorded.append(args)))
    directives = []

    def note(request_id, directive, token_index, at_us):
        directives.append(None if directive is None else directive.to_json() if hasattr(directive, "to_json")
                          else repr(directive))
        if directive is not None:
            AdaptiveDecodeControlMixin._record_assistance_decision(owner, request_id, directive, token_index, at_us)

    layout = dict(helper_layout_generation=1, helper_layout_geometry_sha256="sha256:" + "7" * 64)
    note("request-a", coherence.AdaptiveCoherenceTests.start(case, controller, "request-a", 1, 60, **layout), 1, 1_000)
    leader = coherence.AdaptiveCoherenceTests.record_baseline_window(case, controller, "request-a", 1, 3_000)
    note("request-a", leader, 3, 3_000)
    note("request-a", controller.acknowledge("request-a", AdaptiveDecodePolicyAck(
        "request-a", 1, leader.control.plan_generation, 3, 3_000, case.full.policy_hash)), 3, 3_000)
    note("request-a", coherence.AdaptiveCoherenceTests.record_baseline_window(
        case, controller, "request-a", 1, 4_000, energy_per_token=40, phone_calls=2), 5, 4_000)
    note("request-b", coherence.AdaptiveCoherenceTests.start(case, controller, "request-b", 2, 8, **layout), 1, 4_001)
    controller.complete("request-a", "COMPLETED")
    note("request-c", coherence.AdaptiveCoherenceTests.start(
        case, controller, "request-c", 3, 8, helper_layout_generation=2,
        helper_layout_geometry_sha256="sha256:" + "8" * 64), 1, 5_000)
    return {"directives": directives, "records": [list(row) for row in recorded],
            "groups": len(controller.checkpoint()[6])}


if __name__ == "__main__":
    events = knob_off_events()
    records = coherence_off_records()
    print(json.dumps({
        "knob_off_events": len(events), "knob_off_digest": digest(events),
        "coherence_off_records": len(records["records"]), "coherence_off_groups": records["groups"],
        "coherence_off_digest": digest(records),
    }, sort_keys=True))
