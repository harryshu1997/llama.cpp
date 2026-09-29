"""Exact CPU parents and NPU helpers alongside a busy GPU control."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import threading
from types import SimpleNamespace
import unittest

from research_dev.scheduler import (
    GGUFModelManifestLoader, RuntimeRouteShapeProfile, UnifiedScheduler,
)
from research_dev.scheduler.adapters import (
    CatalogMaterializationError, RuntimeModelEndpointCapability,
    RuntimePhysicalTopology, materialize_cpu_phone_endpoints,
    materialize_desktop_control,
)
from research_dev.scheduler.config import CampaignModelConfiguration
from research_dev.scheduler.campaigns.burstgpt.catalog import _register_overlay_cpu
from research_dev.scheduler.adapters.snapshot import (
    UnifiedRuntimeSnapshotBuilder, EndpointRuntimeSample, ExecutorResidencySample,
)
from test_automated_runtime import (
    catalog_with_gpu_desktop_control, executor_state, request, runtime_snapshot, system_cost_profile,
)
from research_dev.scheduler._internal.runtime_system_cost import RuntimeProtectedWorkObservation
from test_gguf_cost import write_synthetic_gguf


class WorkConservingStartTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "model.gguf"
        write_synthetic_gguf(path, block_count=2)
        self.manifest = GGUFModelManifestLoader.load("test-model", path)
        self.source = catalog_with_gpu_desktop_control(self.manifest)
        self.hardware = RuntimePhysicalTopology(
            cpu_device_id="host-a", gpu_device_id="accelerator-b",
            phone_device_id="helper-c", cpu_resource_id="compute:host-a",
            gpu_resource_id="compute:accelerator-b", host_memory_resource_id="host-memory",
            gpu_memory_resource_id="gpu-memory", phone_memory_resource_id="phone-memory",
            functionfs_resource_id="link:usb-out",
            phone_transport_resource_ids=("link:usb-out", "link:usb-in"),
            phone_compute_resource_ids=("compute:helper-c",),
            gpu_exclusive_residency_resource_id="compute:accelerator-b",
            resource_capacities={key: row.capacity for key, row in self.source.resources.items()},
            resource_identities={key: row.identity for key, row in self.source.resources.items()},
        )
        self.model = RuntimeModelEndpointCapability(
            manifest=self.manifest, desktop_executor_id="executor:accelerator-b",
            desktop_endpoint="http://gpu.invalid:1", desktop_backend="llama-server",
            phone_executor_prefix="unused:gpu-helper", phone_endpoint="http://gpu.invalid:1",
            phone_backend="llama-server-phone", desktop_gpu_first_layer=0,
            adapter_parameters={"parallel": 1, "context_size": 64, "threads": 8},
            desktop_evidence_ids=("sha256:" + "1" * 64,),
            phone_evidence_ids=("sha256:" + "2" * 64,),
            phone_preloaded=False, phone_resident_limit_bytes=1_000_000,
            ffn_column_quantum=8, phone_runtime_control_protocol="decode-boundary-v1",
            phone_adapter_parameters={
                "ffn_resident_columns": self.manifest.feed_forward_length,
                "ffn_resident_layer_mask": (1 << self.manifest.block_count) - 1,
            },
            cpu_executor_id="model:cpu", cpu_endpoint="http://cpu.invalid:1",
            cpu_backend="llama-server-cpu", cpu_phone_executor_prefix="model:cpu-npu",
            cpu_phone_endpoint="http://cpu.invalid:1", cpu_phone_backend="llama-server-cpu-phone",
            cpu_adapter_parameters={"parallel": 1, "context_size": 64, "threads": 2},
            cpu_evidence_ids=("sha256:" + "3" * 64,),
        )

    def materialize(self, **changes):
        return materialize_cpu_phone_endpoints(
            self.source, self.model, self.hardware,
            transition_latency_us=100, transition_energy_uj=100, **changes,
        )

    def cpu_profile(self):
        return RuntimeRouteShapeProfile(
            selector_id="measured:cpu", artifact_sha256=self.manifest.artifact_sha256,
            route_family="layer_placement", device_ids=("host-a",),
            assisted_operator_kind=None, split_axis="none", split_fraction_ppm=0,
            residency_variant="hot", minimum_input_tokens=1, maximum_input_tokens=64,
            minimum_output_tokens=1, maximum_output_tokens=64, service_fixed_us=500,
            service_input_token_us=1, service_output_token_us=10, service_upper_add_us=10,
            energy_fixed_uj=1, energy_input_token_uj=0, energy_output_token_uj=1,
            energy_lower_error_ppm=0, energy_upper_error_ppm=0, sample_count=2,
            maturity="QUALIFIED", evidence_ids=self.model.cpu_evidence_ids,
            executor_id=self.model.cpu_executor_id,
        )

    def startup_fixture(self):
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(self.materialize(route_shape_profiles=(self.cpu_profile(),)))
        scheduler.register_model_manifest(self.manifest)
        shape = request("startup:parent", input_tokens=1, output_tokens=2)
        snapshot = runtime_snapshot(self.manifest, include_phone=False)
        snapshot = replace(snapshot, executors={**snapshot.executors, "model:cpu": executor_state("model:cpu")})
        snapshot = replace(snapshot, residency=tuple(
            replace(row, executor_id="executor:host-a") if row.device_id == "host-a" else row
            for row in snapshot.residency
        ))
        candidate_set = scheduler.generate_automated_candidates(shape, self.manifest.model_id, snapshot)
        parent = next(row for row in candidate_set.candidates if row.plan.baseline_executor_id == "model:cpu")
        candidate_set = scheduler._automated_compiler().generate(
            shape, self.manifest, snapshot, desktop_parent=("model:cpu", parent.plan.desktop_placement_sha256),
        )
        parent = next(row for row in candidate_set.candidates if row.binding.executor_id == "model:cpu")
        return scheduler, shape, snapshot, parent

    def test_startup_preload_keeps_exact_parent_and_does_not_publish_execution_epoch(self):
        scheduler, shape, snapshot, parent = self.startup_fixture()
        before = scheduler.runtime_residency_cohort_state()
        ticket = scheduler.submit_startup_parent_preload(
            shape, self.manifest.model_id, snapshot, executor_id="model:cpu",
            desktop_placement_sha256=parent.plan.desktop_placement_sha256,
            observed_at_us=shape.arrival_us,
        )
        self.assertEqual(ticket.decision.reason, "STARTUP_PARENT_PRELOAD")
        self.assertEqual(ticket.binding.executor_id, "model:cpu")
        self.assertEqual(ticket.execution_plan.desktop_placement_sha256, parent.plan.desktop_placement_sha256)
        self.assertEqual(ticket.execution_plan.transitions, parent.plan.transitions)
        self.assertEqual(ticket.selection_mode, "calibration")
        self.assertIsNone(ticket.phone_layout_generation)
        self.assertTrue(ticket.prepare_lease_tokens)
        self.assertTrue(all(not row.evictions for row in ticket.execution_plan.transitions))
        self.assertEqual(scheduler.runtime_residency_cohort_state()["published_model_placement_epochs"],
                         before["published_model_placement_epochs"])

    def test_startup_preload_rejects_identity_capacity_and_nonidle_without_mutation(self):
        scheduler, shape, snapshot, parent = self.startup_fixture()
        from research_dev.scheduler import UnifiedScheduleError
        options = dict(executor_id="model:cpu", desktop_placement_sha256=parent.plan.desktop_placement_sha256,
                       observed_at_us=shape.arrival_us)
        for changes in ({"executor_id": "missing"}, {"desktop_placement_sha256": "sha256:" + "f" * 64}):
            before = scheduler.runtime_controller_snapshot()
            with self.subTest(changes=changes), self.assertRaisesRegex(UnifiedScheduleError, "not generated"):
                scheduler.submit_startup_parent_preload(shape, self.manifest.model_id, snapshot,
                                                       **{**options, **changes})
            self.assertEqual(scheduler.runtime_controller_snapshot(), before)
        capacities = dict(snapshot.memory.capacities)
        capacity = capacities["host-memory"]
        capacities["host-memory"] = replace(capacity, occupied_bytes=capacity.capacity_bytes - capacity.reserve_bytes)
        crowded = replace(snapshot, memory=replace(snapshot.memory, capacities=capacities))
        before = scheduler.runtime_controller_snapshot()
        with self.assertRaises(UnifiedScheduleError):
            scheduler.submit_startup_parent_preload(shape, self.manifest.model_id, crowded, **options)
        self.assertEqual(scheduler.runtime_controller_snapshot(), before)
        scheduler.submit_startup_parent_preload(shape, self.manifest.model_id, snapshot, **options)
        with self.assertRaisesRegex(UnifiedScheduleError, "idle"):
            scheduler.submit_startup_parent_preload(replace(shape, request_id="another-startup"),
                                                   self.manifest.model_id, snapshot, **options)

    def test_startup_ready_publication_requires_live_exact_execution_proof(self):
        from research_dev.scheduler.adapters import HeterogeneousPhysicalRig, PhysicalAdapterError, interpret_runtime_ticket
        scheduler, shape, snapshot, parent = self.startup_fixture()
        ticket = scheduler.submit_startup_parent_preload(
            shape, self.manifest.model_id, snapshot, executor_id="model:cpu",
            desktop_placement_sha256=parent.plan.desktop_placement_sha256, observed_at_us=shape.arrival_us,
        )
        command = interpret_runtime_ticket(ticket)
        rig = object.__new__(HeterogeneousPhysicalRig)
        rig._lock = threading.RLock()
        rig.epoch_ns = time.monotonic_ns()
        resident = SimpleNamespace(
            server=SimpleNamespace(process=SimpleNamespace(poll=lambda: None)), generation=2,
            endpoint=command.endpoint, executor_id=command.executor_id, manifest=self.manifest,
            parameters=command.adapter_parameters, operator_plan=command.operator_plan,
        )
        rig._live_executors = {command.executor_id: resident}
        proof = {key: getattr(command, key) for key in (
            "ticket_id", "artifact_sha256", "operator_plan_sha256", "executor_id")}
        rig._execution_proofs = {command.ticket_id: proof}
        ready = rig.verify_startup_parent(command)
        self.assertEqual((ready["state"], ready["generation"]), ("READY", 2))
        for key in proof:
            rig._execution_proofs = {command.ticket_id: {**proof, key: "different"}}
            with self.subTest(key=key), self.assertRaises(PhysicalAdapterError):
                rig.verify_startup_parent(command)
        rig._execution_proofs = {command.ticket_id: proof}
        resident.generation = 0
        with self.assertRaises(PhysicalAdapterError):
            rig.verify_startup_parent(command)

    def test_cpu_parent_is_independent_and_helpers_inherit_exact_cpu_contract(self):
        catalog = self.materialize()
        self.assertEqual(catalog.desktop_control_profiles, self.source.desktop_control_profiles)
        self.assertEqual(catalog.executors, self.source.executors)
        parent = catalog.composite_executor_by_id["model:cpu"]
        self.assertEqual(parent.adapter_parameters["threads"], 2)
        self.assertEqual(parent.adapter_parameters["gpu_layers"], 0)
        self.assertEqual(parent.evidence_ids, self.model.cpu_evidence_ids)
        for row in catalog.composite_executors:
            self.assertNotIn("compute:accelerator-b", row.resource_ids)
            self.assertNotIn("accelerator-b", row.participant_device_ids)
            self.assertEqual(row.adapter_parameters["threads"], 2)
            self.assertEqual(row.adapter_parameters["requires_measured_route_profile"], 1)
            if row.executor_id != parent.executor_id:
                self.assertEqual(row.baseline_executor_id, parent.executor_id)
        for transition in catalog.transitions[len(self.source.transitions):]:
            self.assertNotIn("compute:accelerator-b", transition.resource_ids)
            self.assertNotIn("accelerator-b", transition.prepares_device_ids)

    def test_missing_cpu_qualification_does_not_inherit_gpu_route_qualification(self):
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(self.materialize())
        scheduler.register_model_manifest(self.manifest)
        candidates = scheduler.generate_automated_candidates(
            request("no-cpu-evidence"), self.manifest.model_id,
            runtime_snapshot(self.manifest, gpu_busy_until_us=80_000, gpu_free_slots=0),
        )
        cpu = [row for row in candidates.candidates if row.binding.executor_id == "model:cpu"]
        self.assertTrue(cpu)
        self.assertTrue(all(not row.admitted for row in cpu))
        self.assertTrue(all("ROUTE_NOT_QUALIFIED" in row.rejection_reasons for row in cpu))

    def test_registration_rejects_missing_parent_evidence_and_duplicates(self):
        with self.assertRaisesRegex(CatalogMaterializationError, "evidence is absent"):
            materialize_cpu_phone_endpoints(
                self.source, replace(self.model, cpu_evidence_ids=None), self.hardware,
                transition_latency_us=100, transition_energy_uj=100,
            )
        with self.assertRaisesRegex(CatalogMaterializationError, "already exists"):
            materialize_cpu_phone_endpoints(
                self.materialize(), self.model, self.hardware,
                transition_latency_us=100, transition_energy_uj=100,
            )

    def test_qualified_cpu_parent_starts_without_claiming_busy_gpu(self):
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(self.materialize(route_shape_profiles=(self.cpu_profile(),)))
        scheduler.register_model_manifest(self.manifest)
        snapshot = runtime_snapshot(self.manifest, include_phone=False,
                                    gpu_busy_until_us=80_000, gpu_free_slots=0)
        snapshot = replace(snapshot, executors={
            **snapshot.executors, "model:cpu": executor_state("model:cpu"),
        })
        ticket = scheduler.submit_automated_request(
            request("early-cpu"), self.manifest.model_id,
            snapshot,
            selection_mode="energy-aware",
        )
        self.assertEqual(ticket.binding.executor_id, "model:cpu")
        self.assertLess(ticket.decision.start_us, 80_000)
        self.assertNotIn("compute:accelerator-b", ticket.execution_plan.resource_slots)
        self.assertTrue(all("accelerator-b" not in transition.prepares_device_ids
                            for transition in ticket.execution_plan.transitions))

    def test_cpu_profile_cannot_be_rebound_from_gpu_evidence(self):
        with self.assertRaisesRegex(CatalogMaterializationError, "not bound to its parent"):
            self.materialize(route_shape_profiles=(replace(
                self.cpu_profile(), device_ids=("accelerator-b",),
            ),))

    def test_slower_ready_cpu_parent_uses_queue_aware_publication_latency(self):
        for gpu_busy_until in (0, 80_000):
            with self.subTest(gpu_busy_until=gpu_busy_until):
                catalog = self.materialize(route_shape_profiles=(replace(
                    self.cpu_profile(), service_fixed_us=20_000,
                ),))
                scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
                scheduler.register_runtime_capabilities(catalog)
                scheduler.register_model_manifest(self.manifest)
                observed = UnifiedRuntimeSnapshotBuilder(catalog).build(
                    snapshot_id="ready-parent", captured_at_us=0, valid_until_us=10_000_000,
                    memory=runtime_snapshot(self.manifest).memory,
                    executor_samples={
                        "executor:host-a": EndpointRuntimeSample("healthy", "live", 1),
                        "executor:accelerator-b": EndpointRuntimeSample(
                            "healthy", "live", int(gpu_busy_until == 0), gpu_busy_until),
                        "model:cpu": EndpointRuntimeSample("healthy", "live", 1),
                    },
                    residencies=(ExecutorResidencySample(self.manifest, "model:cpu", 7),),
                )
                shape = request("queue-aware-parent", output_tokens=32)
                candidates = scheduler.generate_automated_candidates(shape, self.manifest.model_id, observed)
                if gpu_busy_until:
                    cpu = next(row for row in candidates.candidates if row.binding.executor_id == "model:cpu")
                    self.assertGreater(cpu.cost.service_upper_us, candidates.baseline.cost.service_upper_us)
                ticket = scheduler.submit_automated_request(
                    shape, self.manifest.model_id, observed, selection_mode="energy-aware")
                if not gpu_busy_until:
                    self.assertNotEqual(ticket.binding.executor_id, "model:cpu")
                    continue
                self.assertEqual(ticket.binding.executor_id, "model:cpu")
                self.assertEqual(ticket.execution_plan.desktop_placement_sha256,
                                 cpu.plan.desktop_placement_sha256)
                self.assertEqual(ticket.execution_plan.transitions, ())
                self.assertLess(ticket.decision.start_us, gpu_busy_until)
                self.assertNotIn("compute:accelerator-b", ticket.execution_plan.resource_ids)
                resolution = ticket.cost_estimates.estimates[0].details["model_placement_resolution"]
                self.assertNotEqual(resolution["outcome"], "DESKTOP_FALLBACK_AFTER_NONCONVERGENCE")
                self.assertEqual(resolution["passes"][-1]["rejection_reasons"], [])

    def test_ready_cpu_parent_executes_alongside_acquired_large_request(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "large.gguf"
            write_synthetic_gguf(path, block_count=3)
            large = GGUFModelManifestLoader.load("protected-model", path)
        self.source = replace(
            self.source,
            resources={key: replace(row, capacity=4) if key == "compute:host-a" else row
                       for key, row in self.source.resources.items()},
            executors=tuple(replace(row, exclusive_residency_resource_id="compute:accelerator-b")
                            if row.device_id == "accelerator-b" else row
                            for row in self.source.executors),
        )
        self.hardware = replace(self.hardware, resource_capacities={
            **self.hardware.resource_capacities, "compute:host-a": 4,
        })
        self.source = materialize_desktop_control(
            self.source, large, self.hardware,
            executor_id="large:desktop", endpoint="http://large.invalid:1",
            backend=self.model.desktop_backend, gpu_first_layer=1,
            adapter_parameters=self.model.adapter_parameters,
            evidence_ids=self.model.desktop_evidence_ids,
            transition_latency_us=100, transition_energy_uj=100,
        )
        catalog = self.materialize(route_shape_profiles=(self.cpu_profile(),))
        interference = system_cost_profile(helper_interference_ppm=0)
        catalog = replace(catalog, system_cost_profiles=(replace(
            interference, interference_ppm_by_resource={
                **interference.interference_ppm_by_resource, "compute:host-a": 1,
            }),))
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce", protected_work_policy="energy-budgeted")
        scheduler.register_runtime_capabilities(catalog)
        scheduler.register_model_manifest(large)
        scheduler.register_model_manifest(self.manifest)
        initial = UnifiedRuntimeSnapshotBuilder(catalog).build(
            snapshot_id="large-ready", captured_at_us=0, valid_until_us=10_000_000,
            memory=runtime_snapshot(large).memory,
            executor_samples={
                "executor:host-a": EndpointRuntimeSample("healthy", "live", 4),
                "executor:accelerator-b": EndpointRuntimeSample("healthy", "live", 1),
                "large:desktop": EndpointRuntimeSample("healthy", "live", 1),
            },
            residencies=(ExecutorResidencySample(large, "large:desktop", 1),),
        )
        protected = scheduler.submit_automated_request(
            request("protected"), large.model_id, initial, selection_mode="desktop-baseline")
        epoch_ns = time.monotonic_ns() - protected.decision.start_us * 1000
        protected = scheduler.wait_runtime_request(
            "protected", epoch_ns)
        self.assertIn("compute:host-a", {row.resource_id for row in protected.live_leases})
        self.assertIn("compute:accelerator-b", {row.resource_id for row in protected.live_leases})
        scheduler.extend_runtime_request("protected", at_us=protected.decision.start_us,
                                         reserved_until_us=80_000)
        # The adapter's residency observation, not just endpoint health, proves READY.
        ready_snapshot = UnifiedRuntimeSnapshotBuilder(catalog).build(
            snapshot_id="initial-ready-cpu", captured_at_us=0, valid_until_us=10_000_000,
            memory=initial.memory,
            executor_samples={
                "executor:host-a": EndpointRuntimeSample("healthy", "live", 3),
                "executor:accelerator-b": EndpointRuntimeSample("healthy", "live", 0, 80_000),
                "model:cpu": EndpointRuntimeSample("healthy", "live", 1),
                "large:desktop": EndpointRuntimeSample("healthy", "live", 0, 80_000),
            },
            residencies=(ExecutorResidencySample(self.manifest, "model:cpu", 1),),
            cost_features={"protected_phase": 1},
            protected_work=RuntimeProtectedWorkObservation(
                "synthetic-protected-cpu-gpu", 80_000, 100_000, 0, 0, 4, True),
        )
        ready_snapshot = replace(ready_snapshot, residency=ready_snapshot.residency + initial.residency)
        cpu = scheduler.submit_automated_request(
            request("cpu-overlap", arrival_us=protected.decision.start_us + 1, output_tokens=32),
            self.manifest.model_id, ready_snapshot, selection_mode="energy-aware")
        self.assertEqual(cpu.binding.executor_id, "model:cpu")
        self.assertEqual(cpu.decision.reason, "BUDGETED_PROTECTED_WORK_ENERGY_SAVING")
        self.assertEqual(cpu.transition_status, "NOT_REQUIRED")
        self.assertEqual(cpu.execution_plan.transitions, ())
        cpu = scheduler.wait_runtime_request(
            "cpu-overlap", epoch_ns)
        self.assertEqual(cpu.dispatch_state, "ACQUIRED")
        self.assertEqual(scheduler.runtime_ticket("protected").dispatch_state, "ACQUIRED")
        self.assertLess(cpu.decision.start_us, 80_000)
        self.assertNotIn("compute:accelerator-b", {row.resource_id for row in cpu.live_leases})
        self.assertEqual(len(next(row for row in cpu.live_leases
                                  if row.resource_id == "compute:host-a").lanes), 1)
        cpu_lanes = {lane for row in cpu.live_leases if row.resource_id == "compute:host-a"
                     for lane in row.lanes}
        protected_lanes = {lane for row in protected.live_leases if row.resource_id == "compute:host-a"
                           for lane in row.lanes}
        self.assertFalse(cpu_lanes & protected_lanes)
        self.assertGreaterEqual(scheduler.timeline.next_available_us(
            "compute:host-a", cpu.decision.start_us, slots=4, duration_us=100), 80_000)
        # Cold loading remains capacity-wide and cannot overlap this protected CPU lease.
        for transition in catalog.transitions:
            if transition.executor_id == "model:cpu":
                self.assertEqual(transition.resource_slots["compute:host-a"], 4)
        context = scheduler.runtime_protected_work_power_context(
            (protected.ticket_id,), cpu.dispatch_receipt.observed_at_us,
            cpu.dispatch_receipt.observed_at_us + 1)
        self.assertEqual(context["reason"], "PROTECTED_POWER_OVERLAP")
        self.assertIn(cpu.ticket_id, context["overlapping_ticket_ids"])

    def test_model_configuration_registers_cpu_without_synthetic_profiles(self):
        hardware = replace(self.hardware, resource_capacities={
            **self.hardware.resource_capacities,
            self.hardware.cpu_resource_id: 4,
        })
        source = materialize_desktop_control(
            replace(
                self.source,
                desktop_control_profiles=(),
                resources={
                    **self.source.resources,
                    hardware.cpu_resource_id: replace(
                        self.source.resources[hardware.cpu_resource_id], capacity=4,
                    ),
                },
                executors=tuple(
                    replace(row, exclusive_residency_resource_id=self.hardware.gpu_resource_id)
                    if row.device_id == self.hardware.gpu_device_id
                    else replace(row, evidence_ids=self.model.phone_evidence_ids)
                    if row.device_id == self.hardware.phone_device_id else row
                    for row in self.source.executors
                ),
            ),
            self.manifest, hardware,
            executor_id="overlay:gpu", endpoint=self.model.desktop_endpoint,
            backend=self.model.desktop_backend, gpu_first_layer=0,
            adapter_parameters=self.model.adapter_parameters,
            evidence_ids=self.model.desktop_evidence_ids,
            transition_latency_us=100, transition_energy_uj=100,
        )
        value = {
            "model_key": "overlay", "model_id": self.manifest.model_id,
            "kind": "overlay", "trace_role": "small",
            "host_artifact_path": "model.gguf", "phone_artifact_path": "/phone/model.gguf",
            "endpoint_ids": {"cpu": "overlay-cpu"},
            "backend_ids": {"cpu": "llama-server-cpu"},
            "cpu_runtime_parameters": {"threads": 2, "parallel": 1, "context_size": 64},
            "cpu_evidence_ids": list(self.model.cpu_evidence_ids),
            "phone_resident_limit_bytes": 1_000_000,
        }
        config = CampaignModelConfiguration.from_json(value, Path("/models"))
        restored = CampaignModelConfiguration.from_json(config.to_json(), Path("/"))
        self.assertEqual(restored.cpu_runtime_parameters, config.cpu_runtime_parameters)
        self.assertEqual(restored.cpu_evidence_ids, config.cpu_evidence_ids)
        catalog = _register_overlay_cpu(
            source, self.manifest, restored, hardware,
            {"overlay-cpu": self.model.cpu_endpoint},
        )
        cpu_id = "physical:cpu-parent:" + self.manifest.artifact_sha256[7:23]
        cpu = catalog.composite_executor_by_id[cpu_id]
        self.assertEqual(cpu.endpoint, self.model.cpu_endpoint)
        self.assertEqual(cpu.adapter_parameters["gpu_layers"], 0)
        self.assertEqual(cpu.adapter_parameters["threads"], 2)
        self.assertEqual(cpu.evidence_ids, self.model.cpu_evidence_ids)
        self.assertEqual(catalog.route_shape_profiles, source.route_shape_profiles)
        self.assertEqual(catalog.desktop_control_profiles, source.desktop_control_profiles)
        transitions = tuple(row for row in catalog.transitions if row.executor_id == cpu_id)
        self.assertTrue(transitions)
        self.assertTrue(all(row.energy_maturity == "SHADOW" for row in transitions))
        self.assertTrue(all(
            row.resource_slots[hardware.cpu_resource_id] == 4
            for row in transitions
        ))


if __name__ == "__main__":
    unittest.main()
