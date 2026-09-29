"""Fresh observations confirm stable demand without another request arrival."""

from dataclasses import replace
import unittest
from unittest.mock import patch

from research_dev.scheduler._internal.model_placement_controller import (
    ModelPlacementController, ModelPlacementPolicy,
)
from research_dev.scheduler._internal.capacity import DeviceMemoryCapacity
from research_dev.scheduler.tests import test_model_placement_controller as placement
from research_dev.scheduler.tests import test_replay_determinism as replay


class PhoneConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.controller = ModelPlacementController(ModelPlacementPolicy())
        self.current = placement.phone_layout("current")
        state = self.controller.propose_phone_layout(
            self.current, workspace_bytes=1024,
            shared_compute_resource_id="phone-htp",
            shared_transport_resource_ids=("phone-usb",), observed_at_us=100,
        )
        self.controller.begin_phone_layout_transition(
            state.generation, ticket_id="prepare", transition_ids=("load",),
            ready_at_us=200, workspace_bytes=1024, observed_at_us=110,
            projection_token_sha256=placement.identity("projection"),
        )
        self.controller.complete_phone_layout_transition(
            generation=state.generation, ticket_id="prepare",
            transition_ids=("load",), geometry_sha256=self.current.geometry_sha256,
            projection_token_sha256=placement.identity("projection"), finished_at_us=200,
        )
        self.target = placement.phone_layout("target")

    def confirm(self, sample, *, target=None, decision="stable", at_us=None):
        return self.controller.confirm_phone_layout_candidate(
            (target or self.target).geometry_sha256, placement.identity(decision),
            observed_at_us=at_us or sample,
            observation_sha256=placement.identity("sample:" + str(sample)),
            sampled_at_us=sample,
        )

    def test_identical_demand_confirms_on_three_fresh_observations(self):
        for sample, expected in ((1_000_000, (False, 1)),
                                 (2_000_000, (False, 2)),
                                 (3_000_000, (True, 3))):
            self.assertEqual(self.confirm(sample), expected)
        events = self.controller.phone_layout_events()[-3:]
        self.assertEqual(len({row["snapshot_sha256"] for row in events}), 1)
        self.assertEqual(len({row["observation_sha256"] for row in events}), 3)
        self.assertEqual(self.controller.ready_phone_layout().generation, 1)
        self.assertIsNone(self.controller.target_phone_layout())

    def test_cached_out_of_order_and_fast_samples_do_not_advance(self):
        self.assertEqual(self.confirm(1_000_000), (False, 1))
        for sample in (1_000_000, 900_000, 1_100_000):
            self.assertEqual(self.confirm(sample, decision="changed", at_us=2_000_000),
                             (False, 1))
        self.assertEqual(self.controller.confirm_phone_layout_candidate(
            self.target.geometry_sha256, placement.identity("no-observation"),
            observed_at_us=2_000_000,
        ), (False, 1))
        self.assertEqual(self.confirm(2_000_000), (False, 2))

    def test_changed_target_restarts_confirmation(self):
        self.confirm(1_000_000)
        self.confirm(2_000_000)
        other = placement.phone_layout("other")
        self.assertEqual(self.confirm(2_000_000, target=other), (False, 1))
        self.assertEqual(self.confirm(3_000_000), (False, 1))

    def test_current_layout_winning_withdraws_pending_target(self):
        self.confirm(1_000_000)
        self.confirm(2_000_000, target=self.current)
        self.assertIsNone(self.controller.pending_phone_layout_candidate())
        self.assertEqual(self.confirm(3_000_000), (False, 1))

    def test_checkpoint_restores_observation_fence(self):
        self.confirm(1_000_000)
        saved = self.controller.checkpoint()
        self.confirm(2_000_000)
        self.controller.restore(saved)
        self.assertEqual(self.confirm(1_000_000, decision="new"), (False, 1))
        self.assertEqual(self.confirm(2_000_000), (False, 2))

    def test_minimum_residency_deferral_keeps_confirmed_candidate(self):
        for sample in (1_000_000, 2_000_000, 3_000_000):
            self.confirm(sample)
        arguments = dict(workspace_bytes=1024,
                         shared_compute_resource_id="phone-htp",
                         shared_transport_resource_ids=("phone-usb",))
        deferred = self.controller.propose_phone_layout(
            self.target, observed_at_us=3_000_000, **arguments,
        )
        self.assertEqual(deferred.state, "READY")
        self.assertIsNotNone(self.controller.pending_phone_layout_candidate())
        self.assertEqual(self.confirm(31_000_000), (True, 3))
        proposed = self.controller.propose_phone_layout(
            self.target, observed_at_us=31_000_000, **arguments,
        )
        self.assertEqual(proposed.state, "PROPOSED")
        self.assertIsNone(self.controller.pending_phone_layout_candidate())
        self.assertEqual(self.controller.ready_phone_layout().generation, 1)


class PhoneConfirmationReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        replay.ReplayDeterminismTests.setUpClass()

    def runtime(self):
        case = next(row for row in replay.ReplayDeterminismTests.fixture["cases"]
                    if row["case_id"] == "sparse_locality24_v8")
        driver = replay.ReplayDeterminismTests(methodName="runTest")
        scheduler = driver._new_scheduler(case)
        rows = {row["combined_request_index"]: row for row in case["requests"]}
        state, snapshot, _helper, _view = driver._bootstrap_ready_layout(
            scheduler, case, rows,
            {"proposal_request_index": 34, "proof_snapshot_index": 36},
        )
        row = rows[49]
        at_us = max(row["snapshot"]["captured_at_us"], snapshot.captured_at_us + 60_000_000)
        source = driver._retime_snapshot(snapshot, at_us, "stable-demand")
        source = self.sample(source, at_us)
        request = replace(driver._request(row), arrival_us=at_us,
                          deadline_us=at_us + 600_000_000)
        ticket = scheduler.submit_automated_request(
            request, row["model_id"], source, observed_at_us=at_us,
            selection_mode=case["selection_mode"],
        )
        self.assertIsNotNone(scheduler._model_placement_controller.pending_phone_layout_candidate())
        self.assertIsNone(scheduler._model_placement_controller.target_phone_layout())
        return scheduler, ticket, source, state

    @staticmethod
    def sample(snapshot, sample_us, *, validity="VALID", age_us=0):
        return replace(snapshot, telemetry_observations={"op15-phone": {
            "source": "synthetic-phone", "sample_timestamp_ns": sample_us * 1000,
            "age_us": age_us, "maximum_age_us": 5_000_000,
            "validity": validity, "valid": validity == "VALID",
            "failure_reason": None if validity == "VALID" else "injected " + validity,
        }})

    def poll(self, scheduler, ticket, snapshot):
        return scheduler.runtime_request_helper_preparation_envelope(
            ticket.request.request_id, expected_ticket_id=ticket.ticket_id,
            observed_at_us=snapshot.captured_at_us, snapshot=snapshot,
        )

    def test_stable_saved_demand_proposes_without_arrival_or_decode_progress(self):
        scheduler, ticket, source, state = self.runtime()
        controller = scheduler._model_placement_controller
        retained = scheduler.phone_residency_session_states()
        self.assertTrue(scheduler.runtime_background_helper_preparation_allowed(
            ticket.request.request_id, expected_ticket_id=ticket.ticket_id,
        ))
        self.assertIsNone(self.poll(scheduler, ticket, source))
        for delta in (1_000_000, 2_000_000):
            refreshed = replay.ReplayDeterminismTests._retime_snapshot(
                source, source.captured_at_us + delta, "background:" + str(delta),
            )
            self.poll(scheduler, ticket, self.sample(refreshed, refreshed.captured_at_us))
        target = controller.target_phone_layout()
        self.assertIsNotNone(target)
        self.assertEqual(len(target.layout.changed_session_ids), 1)
        self.assertEqual(target.state, "PROPOSED")
        self.assertEqual(scheduler.runtime_ticket(ticket.request.request_id), ticket)
        self.assertEqual(controller.ready_phone_layout().to_json(), state)
        self.assertEqual(scheduler.phone_residency_session_states(), retained)
        event_count = sum(row["kind"] == "PROPOSED" for row in scheduler.phone_residency_events())
        self.poll(scheduler, ticket, self.sample(refreshed, refreshed.captured_at_us))
        self.assertEqual(sum(row["kind"] == "PROPOSED" for row in scheduler.phone_residency_events()), event_count)

    def test_missing_stale_or_repeated_telemetry_cannot_confirm(self):
        scheduler, ticket, source, _state = self.runtime()
        controller = scheduler._model_placement_controller
        before = controller.pending_phone_layout_candidate()
        for status, age in (("MISSING", 0), ("STALE", 0), ("TIMED_OUT", 0),
                            ("MALFORMED", 0), ("VALID", 6_000_000), ("VALID", 0)):
            snapshot = self.sample(source, source.captured_at_us, validity=status, age_us=age)
            self.assertIsNone(self.poll(scheduler, ticket, snapshot))
            self.assertEqual(controller.pending_phone_layout_candidate(), before)
            self.assertIsNone(controller.target_phone_layout())
        for delta in (1_000_000, 2_000_000):
            self.poll(scheduler, ticket, self.sample(source, source.captured_at_us + delta))
        target = controller.target_phone_layout()
        self.assertIsNotNone(target)
        self.assertEqual(len(target.layout.changed_session_ids), 1)
        self.poll(scheduler, ticket, self.sample(source, source.captured_at_us + 3_000_000))
        self.assertEqual(controller.target_phone_layout(), target)

    def test_fresh_observation_revalidates_memory_not_only_confirmation(self):
        scheduler, ticket, source, _state = self.runtime()
        pool = source.memory.capacities["op15-ram"]
        full = replace(source, memory=replace(source.memory, capacities={
            **source.memory.capacities,
            "op15-ram": DeviceMemoryCapacity("op15-ram", pool.capacity_bytes, pool.capacity_bytes, 0),
        }))
        with patch.object(scheduler, "_phone_candidate_choice", wraps=scheduler._phone_candidate_choice) as choose:
            self.poll(scheduler, ticket, self.sample(full, source.captured_at_us + 1_000_000))
        self.assertEqual(choose.call_count, 1)
        self.assertEqual(choose.call_args.args[3].live_capacity.available_bytes, 0)
        self.assertIsNone(scheduler._model_placement_controller.target_phone_layout())


if __name__ == "__main__":
    unittest.main()
