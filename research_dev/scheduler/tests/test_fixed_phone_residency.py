"""Fixed evaluation assignments reuse the ordinary session lifecycle."""

from dataclasses import replace
import time
import unittest
from unittest.mock import patch

from research_dev.scheduler import UnifiedScheduleError
from research_dev.scheduler.config import (
    FixedPhoneResidencyConfiguration, SchedulerConfigurationError,
)
from research_dev.scheduler.adapters import CanonicalOfflinePhoneResidencyPreloader
from research_dev.scheduler._internal.capacity import DeviceMemoryCapacity
from research_dev.scheduler._unified.phone_residency import (
    _OfflineLearningDemand,
)
import test_offline_phone_residency as offline


class FixedPhoneResidencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        offline.OfflinePhoneResidencyTests.setUpClass()
        scheduler, requests, snapshot = offline.OfflinePhoneResidencyTests.replay_runtime()
        plan = scheduler.plan_offline_phone_residency(
            requests, snapshot=snapshot, observed_at_us=snapshot.captured_at_us,
        )
        cls.fixed = FixedPhoneResidencyConfiguration("synthetic-fixed", tuple(
            (row.session_id, row.artifact_sha256, row.layer_mask, row.maximum_columns)
            for row in plan.target_layout.shards
        ))

    def runtime(self):
        scheduler, requests, snapshot = offline.OfflinePhoneResidencyTests.replay_runtime()
        scheduler.configure_fixed_phone_residency(self.fixed)
        return scheduler, requests, snapshot

    def test_configuration_round_trip_and_validation(self):
        self.assertEqual(FixedPhoneResidencyConfiguration.from_json(self.fixed.to_json()), self.fixed)
        self.assertEqual(replace(self.fixed, assignments=tuple(reversed(self.fixed.assignments))), self.fixed)
        for rows in ((), (self.fixed.assignments[0],) * 2, (None,),
                     (self.fixed.assignments[0], (None, None, 0, 0))):
            with self.assertRaises(SchedulerConfigurationError):
                replace(self.fixed, assignments=rows)
        session, artifact, mask, columns = self.fixed.assignments[0]
        with self.assertRaises(SchedulerConfigurationError):
            replace(self.fixed, assignments=((session, artifact, mask, columns),
                                             ("another-session", artifact, mask, columns)))

    def test_fixed_discovery_keeps_measured_and_learning_artifacts(self):
        scheduler, requests, snapshot = self.runtime()
        assisted_artifacts = {row.artifact_sha256
                              for row in scheduler._runtime_capabilities.composite_executors
                              if row.helper_device_id is not None}
        models = tuple(row for row in scheduler._runtime_manifests.values()
                       if row.artifact_sha256 in assisted_artifacts)[:2]
        request = next(iter(requests.values()))[0]
        source = scheduler._offline_phone_discovery(
            {models[0].artifact_sha256: (models[0].model_id, request)},
            {models[0].artifact_sha256: 1}, snapshot, snapshot.captured_at_us,
        )
        self.assertEqual(len(source.demand_rows), 1)
        learning = _OfflineLearningDemand(
            demand=replace(source.demand_rows[0], manifest=models[1]),
            sessions=source.sessions, helper_id=source.helper_id,
            status={"evidence_state": "LEARNING"},
        )
        with patch.object(scheduler, "_generate_automated_candidate_set"), \
             patch.object(scheduler, "_learning_phone_demand", side_effect=(
                 lambda _candidates, _request, model, _work:
                     learning if model.artifact_sha256 == models[1].artifact_sha256 else None
             )), \
             patch.object(scheduler, "_discover_phone_residency_demand", return_value=source):
            result = scheduler._offline_phone_discovery(
                {row.artifact_sha256: (row.model_id, request) for row in models},
                {row.artifact_sha256: 1 for row in models}, snapshot, snapshot.captured_at_us,
            )
        self.assertEqual({row.manifest.artifact_sha256 for row in result.demand_rows},
                         {row.artifact_sha256 for row in models})
        self.assertEqual(result.sessions, source.sessions)

    def test_progressive_fixed_preload_holds_assignment_against_queue_policy(self):
        scheduler, requests, snapshot = self.runtime()
        plan = scheduler.plan_offline_phone_residency(
            requests, snapshot=snapshot, observed_at_us=snapshot.captured_at_us,
        )
        self.assertEqual(plan.metadata["selection_policy"], "fixed-experimental-assignment")
        self.assertTrue(scheduler._fixed_phone_layout_matches(plan.target_layout))
        backend = offline._OfflinePhoneBackend(snapshot)
        preloader = CanonicalOfflinePhoneResidencyPreloader(
            scheduler, backend,
            epoch_ns=time.monotonic_ns() - snapshot.captured_at_us * 1000,
            snapshot_provider=backend.snapshot,
        )
        result = preloader.preload(plan, lambda _stage: object(),
                                   initial_snapshot=snapshot, observed_at_us=snapshot.captured_at_us)
        self.assertEqual(result.plan.state, "READY")
        states = scheduler.phone_residency_session_states()
        events = scheduler.phone_residency_events()
        request = next(iter(requests.values()))[0]
        manifest = scheduler.runtime_model_manifest(next(iter(requests)))
        self.assertFalse(scheduler._update_phone_residency_portfolio(request, manifest, 0, snapshot))
        self.assertEqual(scheduler.phone_residency_events(), events)
        self.assertEqual(scheduler.phone_residency_session_states(), states)
        self.assertTrue(all(len(command.transition.changed_phone_session_ids) == 1
                            for command in result.commands))
        self.assertEqual(set(backend.load_count_by_session.values()), {1})
        with self.assertRaisesRegex(UnifiedScheduleError, "already frozen"):
            scheduler.configure_fixed_phone_residency(replace(self.fixed, reference_id="changed"))

    def test_fixed_preparation_shapes_do_not_contain_trace_requests(self):
        scheduler, requests, snapshot = self.runtime()
        neutral = scheduler.fixed_phone_residency_requests()
        ids = {row.request_id for rows in requests.values() for row in rows}
        self.assertFalse(ids & {row.request_id for rows in neutral.values() for row in rows})
        plan = scheduler.plan_offline_phone_residency(
            neutral, snapshot=snapshot, observed_at_us=snapshot.captured_at_us,
        )
        self.assertTrue(scheduler._fixed_phone_layout_matches(plan.target_layout))

    def test_fixed_assignment_does_not_shrink_to_fit(self):
        scheduler, requests, snapshot = self.runtime()
        pool = "op15-ram"
        snapshot = replace(snapshot, memory=replace(snapshot.memory, capacities={
            **snapshot.memory.capacities,
            pool: DeviceMemoryCapacity(pool, 10_000_000_000, 8_000_000_000, 0),
        }))
        with self.assertRaises(UnifiedScheduleError):
            scheduler.plan_offline_phone_residency(
                requests, snapshot=snapshot, observed_at_us=snapshot.captured_at_us,
            )
        self.assertIsNone(scheduler.offline_phone_residency_snapshot())

    def test_fixed_assignment_rejects_gpu_layers(self):
        scheduler, requests, snapshot = offline.OfflinePhoneResidencyTests.replay_runtime()
        model = scheduler.runtime_model_manifest(next(iter(requests)))
        session, artifact, mask, columns = self.fixed.assignments[0]
        invalid = replace(self.fixed, assignments=((session, artifact, mask | (1 << (model.block_count - 1)), columns),))
        scheduler.configure_fixed_phone_residency(invalid)
        with self.assertRaisesRegex(UnifiedScheduleError, "CPU FFNs"):
            scheduler.plan_offline_phone_residency(
                requests, snapshot=snapshot, observed_at_us=snapshot.captured_at_us,
            )


if __name__ == "__main__":
    unittest.main()
