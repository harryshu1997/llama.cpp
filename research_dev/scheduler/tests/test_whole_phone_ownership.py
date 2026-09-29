from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from research_dev.scheduler import (
    DeviceMemoryCapacity, ModelResidencyObservation, RuntimeCompositeExecutorCapability, UnifiedScheduler,
    RuntimeMemoryDemand, RuntimeResidencyEviction, RuntimeTransitionPlan, RuntimeResourceError,
)
from research_dev.scheduler._internal.runtime_resources import transition_adjusted_memory_demands
from research_dev.scheduler._internal.runtime_residency_projection import (
    RuntimeResidencyProjectionError, project_scheduler_residency,
)
from research_dev.scheduler._internal.runtime_residency_cohorts import RuntimeResidencyCohortTracker
from research_dev.scheduler._internal.phone_shards import (
    PhoneFfnResidencyDemand, generate_mixed_ffn_residency_layouts,
)
from research_dev.scheduler._unified.automated_candidates_ops.generation import (
    _prepare_route_evidence_migrations,
)
from research_dev.scheduler.adapters import materialize_whole_model_endpoint

try:
    from . import test_phone_memory_cap as memory_fixtures
    from .test_multi_session_phone import session
    from .test_automated_runtime import catalog, request
except ImportError:
    import test_phone_memory_cap as memory_fixtures
    from test_multi_session_phone import session
    from test_automated_runtime import catalog, request


class WholePhoneOwnershipTests(unittest.TestCase):
    peak = memory_fixtures.WholePhoneMemoryReuseTests.peak
    whole_candidate = memory_fixtures.WholePhoneMemoryReuseTests.whole_candidate

    def setUp(self):
        memory_fixtures.WholePhoneMemoryReuseTests.setUp(self)
        parameters = dict(self.catalog.executor_by_device["helper-c"].adapter_parameters)
        source = catalog()
        sessions = tuple(replace(
            session(index), device_id="helper-c", memory_resource_id="session:" + str(index),
            shared_compute_resource_id="compute:helper-c", shared_transport_resource_ids=("link:usb-out",),
        ) for index in range(3))
        phone = replace(source.executor_by_device["helper-c"], phone_sessions=sessions,
                        exclusive_residency_resource_id="compute:helper-c",
                        execution_resource_ids=("compute:helper-c", "link:usb-out"))
        ffn = RuntimeCompositeExecutorCapability(
            executor_id="resident-ffn-executor", endpoint="synthetic://ffn",
            backend="ffn", coordinator_device_id="accelerator-b",
            participant_device_ids=("accelerator-b", "helper-c"),
            participant_resource_ids={"accelerator-b": ("compute:accelerator-b",),
                                      "helper-c": phone.execution_resource_ids},
            route_family="operator_split", assisted_operator_kind="ffn", split_axis="column",
            split_fractions_ppm=(500_000,), layer_fractions_ppm=(), residency_states=("hot",),
            resource_ids=("compute:accelerator-b", *phone.execution_resource_ids),
            operator_plan_protocol="synthetic-plan-v1", maturity="QUALIFIED",
            evidence_ids=("synthetic-ffn",), artifact_sha256="sha256:" + "f" * 64,
            replacement_group_by_device={"helper-c": "compute:helper-c"},
        )
        pools = dict(source.placement_profile.memory_pools)
        for value in sessions:
            pools[value.memory_resource_id] = replace(pools["phone-memory"],
                                                       pool_id=value.memory_resource_id)
        source = replace(source, executors=tuple(
            phone if row.device_id == phone.device_id else row for row in source.executors
        ), placement_profile=replace(source.placement_profile, memory_pools=pools),
            composite_executors=(ffn,))
        self.catalog = materialize_whole_model_endpoint(
            source, self.model, executor_id=phone.executor_id,
            endpoint="http://127.0.0.1:18382", backend="android-llama-server-opencl",
            adapter_parameters=parameters, evidence_ids=("sha256:" + "1" * 64,),
            transition_latency_us=1000, transition_energy_uj=5000,
            transition_energy_maturity="QUALIFIED",
            request_transport_identity_sha256="sha256:" + "3" * 64,
        )
        self.scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        self.scheduler.register_runtime_capabilities(self.catalog)
        self.scheduler.register_model_manifest(self.model)
        ffn_model = replace(self.model, model_id="resident-ffn-model", artifact_sha256=ffn.artifact_sha256)
        self.scheduler.register_model_manifest(ffn_model)
        layout = max(generate_mixed_ffn_residency_layouts(
            (PhoneFfnResidencyDemand(ffn_model, 100, 8, "split-row"),), sessions,
            phone_wide_limit_bytes=768,
        ), key=lambda row: row.resident_bytes).with_session_generations({row.session_id: 3 for row in sessions})
        self.scheduler._model_placement_controller.adopt_verified_phone_layout(
            layout, workspace_bytes=32, shared_compute_resource_id="compute:helper-c",
            shared_transport_resource_ids=("link:usb-out",), verified_at_us=0,
            verification_sha256="sha256:" + "b" * 64,
        )
        self.retained = ModelResidencyObservation(
            model_id="resident-ffn-model", artifact_sha256="sha256:" + "f" * 64,
            device_id="helper-c", state="hot", resident_tensor_ids=("ffn-weight",),
            resident_bytes=768, generation=3,
            executor_id=next(row.executor_id for row in source.composite_executors
                             if "helper-c" in row.participant_device_ids),
            resident_geometry_sha256=layout.geometry_sha256,
        )
        self.snapshot = replace(self.snapshot,
            telemetry_observations={"helper-c": {
                "source": "synthetic-adb", "sample_timestamp_ns": 123,
                "age_us": 0, "maximum_age_us": 5_000_000, "validity": "VALID", "valid": True,
                "failure_reason": None,
            }},
            residency=tuple(row for row in self.snapshot.residency if row.device_id != "helper-c")
                      + (self.retained,),
            memory=replace(self.snapshot.memory, capacities={
                **self.snapshot.memory.capacities,
                "phone-memory": DeviceMemoryCapacity("phone-memory", self.peak + 8192, 768, 0),
            }))

    def test_whole_endpoint_has_separate_ownership_but_shared_compute(self):
        phone = self.catalog.executor_by_device["helper-c"]
        self.assertNotEqual(phone.exclusive_residency_resource_id, "compute:helper-c")
        self.assertIn("compute:helper-c", phone.execution_resource_ids)
        self.assertEqual(self.catalog.residency_group(self.retained.executor_id, "helper-c"),
                         "compute:helper-c")
        groups, _ = RuntimeResidencyCohortTracker._session_replacement_resources(self.catalog)
        self.assertEqual(set(groups.values()), {"compute:helper-c"})

    def test_whole_load_does_not_evict_or_credit_retained_htp(self):
        row = self.whole_candidate(variant="cold")
        self.assertTrue(row.admitted, row.rejection_reasons)
        self.assertTrue(row.plan.transitions)
        self.assertFalse(any(value.evictions for value in row.plan.transitions))
        self.assertEqual(sum(value.additional_bytes for value in row.plan.memory_demands), self.peak)
        weights = next(value for value in row.plan.memory_demands if value.kind == "model_weights")
        self.assertEqual(weights.replacement_group, "residency:whole:executor:helper-c")

    def test_shared_pool_still_rejects_insufficient_capacity(self):
        snapshot = replace(self.snapshot, memory=replace(self.snapshot.memory, capacities={
            **self.snapshot.memory.capacities,
            "phone-memory": DeviceMemoryCapacity("phone-memory", self.peak + 767, 768, 0),
        }))
        row = self.whole_candidate(snapshot, variant="cold")
        self.assertIn("MEMORY_CAPACITY", row.rejection_reasons)
        self.assertFalse(any(value.evictions for value in row.plan.transitions))

    def test_htp_eviction_keeps_its_exact_anchor_after_whole_endpoint_registration(self):
        group = "compute:helper-c"
        eviction = RuntimeResidencyEviction(
            model_id=self.retained.model_id, artifact_sha256=self.retained.artifact_sha256,
            device_id="helper-c", resident_bytes=768, generation=3,
            executor_id=self.retained.executor_id, replacement_group=group,
        )
        transition = RuntimeTransitionPlan(
            transition_id="replace-ffn", device_id="helper-c", source_state="cold", target_state="hot",
            latency_us=1000, energy_uj=5000, resource_ids=(group,), maturity="QUALIFIED",
            evictions=(eviction,),
        )
        demand = RuntimeMemoryDemand(
            demand_id="next-ffn", resource_id="phone-memory", kind="model_weights",
            required_bytes=800, resident_bytes=0, lifetime="resident", share_key="next-ffn",
            replacement_group=group, device_id="helper-c",
        )

        def adjust(value):
            return transition_adjusted_memory_demands(
                (demand,), transitions=(value,), residency=(self.retained,),
                exclusive_resource_by_device=self.catalog.exclusive_residency_resources,
            )

        self.assertEqual(adjust(transition)[0].replaceable_bytes, 768)
        whole_group = self.catalog.executor_by_device["helper-c"].exclusive_residency_resource_id
        with self.assertRaisesRegex(RuntimeResourceError, "exclusive anchor"):
            adjust(replace(transition, resource_ids=(whole_group,), resource_slots={},
                           evictions=(replace(eviction, replacement_group=whole_group),)))
        for changes in ({"generation": 4}, {"resident_bytes": 767},
                        {"artifact_sha256": self.model.artifact_sha256},
                        {"executor_id": "executor:helper-c"}):
            with self.subTest(changes=changes), self.assertRaisesRegex(RuntimeResourceError, "stale"):
                adjust(replace(transition, evictions=(replace(eviction, **changes),)))

    def test_projection_accepts_independent_resident_owners(self):
        whole = replace(self.retained, model_id=self.model.model_id,
                        artifact_sha256=self.model.artifact_sha256,
                        executor_id="executor:helper-c", generation=1,
                        resident_geometry_sha256=None, resident_bytes=self.model.tensor_bytes)
        snapshot = replace(self.snapshot, residency=self.snapshot.residency + (whole,))
        projected = project_scheduler_residency(snapshot, self.catalog, (), {})
        self.assertEqual(projected, snapshot)
        conflicting = replace(whole, model_id="other-whole", artifact_sha256="sha256:" + "d" * 64)
        with self.assertRaisesRegex(RuntimeResidencyProjectionError, "incompatible resident artifacts"):
            project_scheduler_residency(replace(snapshot, residency=snapshot.residency + (conflicting,)),
                                        self.catalog, (), {})

    def test_queued_whole_load_projection_retains_htp_residency(self):
        def select(candidates, *_args, **_kwargs):
            selected = next(row for row in candidates.candidates
                            if row.route_family == "whole_model" and row.device_ids == ("helper-c",)
                            and row.residency_variant == "cold")
            return selected, (), "SYNTHETIC_WHOLE_PROJECTION"

        with patch.object(self.scheduler, "_select_automated_candidate", side_effect=select):
            ticket = self.scheduler.submit_automated_request(
                request("queued-whole"), self.model.model_id, self.snapshot, selection_mode="calibration")
        projected = project_scheduler_residency(self.snapshot, self.catalog, (ticket,),
                                               {self.model.model_id: self.model})
        retained = next(row for row in projected.residency if row.model_id == self.retained.model_id)
        self.assertEqual(retained, self.retained)
        whole = next(row for row in projected.residency
                     if row.device_id == "helper-c" and row.model_id == self.model.model_id)
        self.assertEqual(whole.executor_id, "executor:helper-c")
        self.assertEqual(projected.memory.capacities["phone-memory"].occupied_bytes, 768 + self.peak)

    def test_evidence_recovery_visits_whole_route_without_adaptive_contract(self):
        row = self.whole_candidate(variant="cold")
        compiler = Mock()
        compiler.component_capability_identity.return_value = "component"
        compiler.rebind_legacy_transition_observations.return_value = 4
        compiler.rebind_legacy_exact_route_observations.return_value = 4
        owner = SimpleNamespace(_automated_compiler=lambda: compiler,
                                _automated_observation_sources={"source": self.catalog},
                                _runtime_capabilities=self.catalog,
                                _legacy_evidence_migration_cache=set())
        candidates = SimpleNamespace(candidates=(row,), search_metadata={})
        self.assertEqual(_prepare_route_evidence_migrations(owner, candidates, self.model), 8)
        compiler.rebind_legacy_transition_observations.assert_called_once_with(
            self.catalog, self.model, row.plan, row.binding.executor_id, latency_only=False)
        self.assertEqual(_prepare_route_evidence_migrations(owner, candidates, self.model), 0)


if __name__ == "__main__":
    unittest.main()
