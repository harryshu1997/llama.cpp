"""Revalidate a PROPOSED phone layout against arrived demand before loading it."""

from types import SimpleNamespace
import unittest

from research_dev.scheduler._internal.lifecycle import UnifiedScheduleError
from research_dev.scheduler._unified.helper_preparation_ops.authorization import (
    _reject_stale_phone_layout_proposal,
)
from research_dev.scheduler._unified.phone_residency_ops.common import _PhoneQueueDemand

GEMMA = "sha256:" + "a" * 64
QWEN = "sha256:" + "b" * 64
GEOMETRY = "sha256:" + "c" * 64


def demand(*, active=None, queued=None):
    active = dict(active or {})
    queued = dict(queued or {})
    return _PhoneQueueDemand(
        active_count_by_artifact={k: 1 for k in active},
        queued_count_by_artifact={k: 1 for k in queued},
        active_remaining_tokens_by_artifact=active,
        queued_output_tokens_by_artifact=queued,
        queued_work_by_artifact={k: v for k, v in {**queued, **active}.items()},
    )


def state(generation=3, changed=("HTP2",), artifacts=("gemma", "gemma", "gemma")):
    labels = {"gemma": GEMMA, "qwen": QWEN}
    shards = tuple(
        SimpleNamespace(session_id="HTP" + str(index), artifact_sha256=labels[label])
        for index, label in enumerate(artifacts)
    )
    return SimpleNamespace(generation=generation, layout=SimpleNamespace(
        shards=shards, changed_session_ids=tuple(changed), geometry_sha256=GEOMETRY,
    ))


class StaleProposalTests(unittest.TestCase):
    def controller(self, queue, *, target_generation=3, replan_error=None, minimum=24):
        calls = []
        placement = SimpleNamespace(
            target_phone_layout=lambda: (
                None if target_generation is None
                else SimpleNamespace(generation=target_generation)
            ),
            reject_phone_layout_proposal=lambda generation, **kw: calls.append(
                ("reject", generation, kw["reason"])),
            record_request_helper_event=lambda request_id, kind, at_us, payload: calls.append(
                ("event", request_id, kind, dict(payload))),
        )

        def replan(request, manifest, observed_at_us, snapshot):
            calls.append(("replan", request.request_id, observed_at_us))
            if replan_error is not None:
                raise replan_error

        host = SimpleNamespace(
            runtime_model_manifest=lambda model_id: SimpleNamespace(model_id=model_id),
            _phone_queue_demand=lambda request, manifest: queue,
            _adaptive_envelope_minimum_remaining_tokens=minimum,
            _model_placement_controller=placement,
            _update_phone_residency_portfolio=replan,
        )
        return host, calls

    def ticket(self):
        return SimpleNamespace(
            request=SimpleNamespace(request_id="gemma-36"),
            model=SimpleNamespace(model_id="gemma", artifact_sha256=GEMMA),
        )

    def test_owner_finishing_with_no_other_demand_rejects_and_replans(self):
        # v9: Gemma's 24-layer expansion started loading at its completion while
        # only Qwen was queued.
        host, calls = self.controller(demand(active={GEMMA: 2}, queued={QWEN: 71}))
        result = _reject_stale_phone_layout_proposal(
            host, "gemma-36", self.ticket(), state(), None, 136_300_000,
        )
        self.assertEqual(result["status"], "REJECTED")
        self.assertEqual(result["reason"], "PHONE_RESIDENCY_PROPOSAL_STALE")
        self.assertEqual(result["live_decode_tokens_by_artifact"], {GEMMA: 2})
        self.assertEqual(result["minimum_remaining_tokens"], 24)
        self.assertEqual(result["replan_status"], "REQUESTED")
        self.assertEqual(calls[0], ("reject", 3, "PHONE_RESIDENCY_PROPOSAL_STALE"))
        self.assertEqual(calls[1], ("replan", "gemma-36", 136_300_000))
        self.assertEqual(calls[2][:3], ("event", "gemma-36", "REJECTED"))
        self.assertEqual(calls[2][3]["phone_layout_generation"], 3)

    def test_arrived_or_running_demand_for_the_new_shards_keeps_the_proposal(self):
        ticket = self.ticket()
        for queue in (
            demand(active={GEMMA: 2}, queued={GEMMA: 292}),   # another Gemma arrived
            demand(active={GEMMA: 150}),                       # owner still has work
            demand(active={GEMMA: 24}),                        # exactly the minimum
        ):
            host, calls = self.controller(queue)
            self.assertIsNone(_reject_stale_phone_layout_proposal(
                host, "gemma-36", ticket, state(), None, 1_000))
            self.assertEqual(calls, [])

    def test_demand_for_another_model_does_not_keep_a_gemma_expansion(self):
        host, calls = self.controller(demand(active={GEMMA: 5}, queued={QWEN: 500}))
        result = _reject_stale_phone_layout_proposal(
            host, "gemma-36", self.ticket(), state(), None, 1_000)
        self.assertEqual(result["status"], "REJECTED")
        # A layout whose new shard IS Qwen's is kept for the queued Qwen.
        host, calls = self.controller(demand(active={GEMMA: 5}, queued={QWEN: 500}))
        self.assertIsNone(_reject_stale_phone_layout_proposal(
            host, "gemma-36", self.ticket(),
            state(changed=("HTP2",), artifacts=("gemma", "gemma", "qwen")), None, 1_000))
        self.assertEqual(calls, [])

    def test_no_changed_sessions_or_other_target_generation_is_left_alone(self):
        host, calls = self.controller(demand(active={GEMMA: 2}))
        self.assertIsNone(_reject_stale_phone_layout_proposal(
            host, "gemma-36", self.ticket(), state(changed=()), None, 1_000))
        host, calls = self.controller(demand(active={GEMMA: 2}), target_generation=4)
        self.assertIsNone(_reject_stale_phone_layout_proposal(
            host, "gemma-36", self.ticket(), state(generation=3), None, 1_000))
        host, calls = self.controller(demand(active={GEMMA: 2}), target_generation=None)
        self.assertIsNone(_reject_stale_phone_layout_proposal(
            host, "gemma-36", self.ticket(), state(generation=3), None, 1_000))
        self.assertEqual(calls, [])

    def test_replan_failure_is_recorded_not_raised(self):
        host, calls = self.controller(
            demand(active={GEMMA: 2}), replan_error=UnifiedScheduleError("no phone"))
        result = _reject_stale_phone_layout_proposal(
            host, "gemma-36", self.ticket(), state(), None, 1_000)
        self.assertEqual(result["status"], "REJECTED")
        self.assertEqual(result["replan_status"], "FAILED:no phone")
        self.assertEqual([c[0] for c in calls], ["reject", "replan", "event"])


if __name__ == "__main__":
    unittest.main()
