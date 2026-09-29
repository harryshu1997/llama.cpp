"""Automated runtime: residency, transitions, replacement and memory projection.

Split from test_automated_runtime.py on 2026-09-13; fixtures and the base class stay there."""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
import threading
import time
from unittest import mock
from research_dev.scheduler import (
    DeviceMemoryCapacity,
    GGUFModelManifestLoader,
    ModelResidencyObservation,
    RuntimeCapabilityCatalog,
    RuntimeCompositeExecutorCapability,
    RuntimeCompositeOperatorPlacement,
    RuntimeDesktopControlProfile,
    RuntimeExecutorState,
    RuntimeExecutionReceipt,
    RuntimePlacementSnapshot,
    RuntimeResourceError,
    RuntimeRouteShapeProfile,
    RuntimeTransitionCapability,
    RuntimeTransitionReceipt,
    UnifiedScheduleError,
    UnifiedScheduler,
)
from research_dev.scheduler._internal.runtime_learning import (
    RuntimeRouteObservationStore,
    _MeasuredRouteObservation,
)
from research_dev.scheduler._internal.runtime_controller import RuntimeReplanRetryRequired
from research_dev.scheduler._internal.route_generation import candidate_set_to_runtime_costs
from research_dev.scheduler._internal.runtime_plan import RuntimeResidencyEviction
from research_dev.scheduler._internal.runtime_residency_projection import (
    RuntimeResidencyProjectionError,
    _authoritative_replacement_occupancy,
    project_scheduler_residency,
)
from research_dev.scheduler._internal.runtime_residency_cohorts import (
    RuntimeResidencyComponentIdentity,
    RuntimeResidencyCohortError,
    RuntimeResidencyCohortTracker,
)
from research_dev.scheduler._internal.types import canonical_sha256
from research_dev.scheduler.adapters import model_residency_observations

try:
    from .test_automated_runtime import (
        AutomatedRuntimeTests,
        FakeAutomatedPhysicalAdapter,
        catalog,
        catalog_with_gpu_desktop_control,
        executor_state,
        request,
        runtime_snapshot,
        write_synthetic_gguf,
    )
except ImportError:  # run from the tests directory
    from test_automated_runtime import (
        AutomatedRuntimeTests,
        FakeAutomatedPhysicalAdapter,
        catalog,
        catalog_with_gpu_desktop_control,
        executor_state,
        request,
        runtime_snapshot,
        write_synthetic_gguf,
    )


class AutomatedRuntimeResidencyTests(AutomatedRuntimeTests):
    """Automated runtime: residency, transitions, replacement and memory projection."""

    def test_projection_accepts_exact_mixed_phone_layout_rows(self) -> None:
        source = catalog()
        phone = replace(
            source.executor_by_device["helper-c"],
            exclusive_residency_resource_id="compute:helper-c",
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=tuple(
                phone if row.device_id == "helper-c" else row
                for row in source.executors
            ),
        ).to_json())
        first = GGUFModelManifestLoader.load("mixed-a", self.path)
        second_path = Path(self.directory.name) / "mixed-b.gguf"
        write_synthetic_gguf(second_path, block_count=3)
        second = GGUFModelManifestLoader.load("mixed-b", second_path)
        geometry = "sha256:" + "a" * 64
        snapshot = replace(
            runtime_snapshot(first),
            residency=tuple(
                ModelResidencyObservation(
                    model_id=manifest.model_id,
                    artifact_sha256=manifest.artifact_sha256,
                    device_id="helper-c",
                    state="hot",
                    resident_tensor_ids=tuple(
                        row.tensor_id for row in manifest.tensors
                    ),
                    resident_bytes=manifest.tensor_bytes,
                    generation=3,
                    executor_id="executor:helper-c",
                    resident_geometry_sha256=geometry,
                )
                for manifest in (first, second)
            ),
        )

        projected = project_scheduler_residency(
            snapshot,
            profile,
            (),
            {first.model_id: first, second.model_id: second},
        )

        self.assertIs(projected, snapshot)
        with self.assertRaisesRegex(
            RuntimeResidencyProjectionError,
            "incompatible resident artifacts",
        ):
            project_scheduler_residency(
                replace(
                    snapshot,
                    residency=(
                        snapshot.residency[0],
                        replace(
                            snapshot.residency[1],
                            resident_geometry_sha256=(
                                "sha256:" + "b" * 64
                            ),
                        ),
                    ),
                ),
                profile,
                (),
                {first.model_id: first, second.model_id: second},
            )

    def test_component_template_precedes_legacy_route_template(self) -> None:
        store = RuntimeRouteObservationStore()
        component_rows = (_MeasuredRouteObservation(
            receipt_id="sha256:" + "1" * 64,
            latency_us=100,
            energy_uj=200,
            evidence_ids=("component",),
            input_tokens=10,
            output_tokens=2,
            component_service_us=90,
            component_energy_uj=180,
            energy_scope="warm_execution",
        ),)
        legacy_rows = (_MeasuredRouteObservation(
            receipt_id="sha256:" + "2" * 64,
            latency_us=300,
            energy_uj=None,
            evidence_ids=("legacy",),
            input_tokens=10,
            output_tokens=2,
            component_service_us=90,
            component_energy_uj=180,
            energy_scope="route_total",
        ),)
        component_prefix = (
            "sha256:" + "a" * 64,
            "sha256:" + "b" * 64,
            "sha256:" + "c" * 64,
            "semantic",
        )
        legacy_prefix = (
            component_prefix[0],
            "sha256:" + "d" * 64,
            "sha256:" + "e" * 64,
            "semantic",
        )
        store._template_rows[(*component_prefix, "old-feature")] = (
            component_rows
        )
        store._template_rows[(*legacy_prefix, "legacy-feature")] = (
            legacy_rows
        )

        selected = store._component_rows_from_exact(
            (*component_prefix, "new-feature"),
            (),
            legacy_prefix=legacy_prefix,
        )

        self.assertIs(selected, component_rows)

    def test_cold_route_reuses_hot_profile_and_adds_transition(self) -> None:
        manifest = GGUFModelManifestLoader.load(
            "unseen-model", self.path
        )
        hot_profile = RuntimeRouteShapeProfile(
            selector_id="qualified-hot-gpu-execution",
            artifact_sha256=manifest.artifact_sha256,
            route_family="whole_model",
            device_ids=("accelerator-b",),
            assisted_operator_kind=None,
            split_axis="none",
            split_fraction_ppm=0,
            residency_variant="hot",
            minimum_input_tokens=1,
            maximum_input_tokens=1_000,
            minimum_output_tokens=1,
            maximum_output_tokens=1_000,
            service_fixed_us=2_000,
            service_input_token_us=10,
            service_output_token_us=20,
            service_upper_add_us=500,
            energy_fixed_uj=4_000,
            energy_input_token_uj=10,
            energy_output_token_uj=20,
            energy_lower_error_ppm=100_000,
            energy_upper_error_ppm=100_000,
            sample_count=8,
            maturity="QUALIFIED",
            evidence_ids=("held-out-hot-gpu-execution",),
            executor_id="executor:accelerator-b",
        )
        scheduler, manifest = self.scheduler_and_manifest(replace(
            catalog(), route_shape_profiles=(hot_profile,)
        ))
        runtime_request = request("cold-from-qualified-hot")
        candidates = scheduler.generate_automated_candidates(
            runtime_request,
            manifest.model_id,
            runtime_snapshot(
                manifest,
                resident_devices=("host-a", "helper-c"),
            ),
        )
        cold = next(
            row for row in candidates.candidates
            if row.route_family == "whole_model"
            and row.device_ids == ("accelerator-b",)
            and row.residency_variant == "cold"
        )
        transition_us = sum(
            row.latency_us for row in cold.plan.transitions
        )

        self.assertTrue(cold.plan.transitions)
        self.assertEqual(
            cold.plan.route_profile_id, hot_profile.selector_id
        )
        self.assertEqual(cold.cost.latency_evidence, "MEASURED")
        self.assertEqual(cold.cost.energy_evidence, "MEASURED")
        self.assertEqual(
            cold.cost.service_us,
            hot_profile.service_us(
                runtime_request.input_tokens,
                runtime_request.output_tokens,
                {},
            ) + transition_us,
        )
        self.assertEqual(
            cold.cost.fleet_energy_upper_uj,
            cold.cost.warm_execution_energy_upper_uj
                + cold.cost.transition_energy_upper_uj,
        )
        self.assertGreater(cold.cost.transition_energy_upper_uj, 0)

    def test_foreign_executor_residency_cannot_authorize_hot_route(
        self,
    ) -> None:
        source = catalog()
        scheduler, manifest = self.scheduler_and_manifest(source)
        snapshot = runtime_snapshot(
            manifest,
            include_phone=False,
            resident_devices=(),
        )
        snapshot = replace(
            snapshot,
            residency=(ModelResidencyObservation(
                model_id=manifest.model_id,
                artifact_sha256=manifest.artifact_sha256,
                device_id="host-a",
                state="hot",
                resident_tensor_ids=tuple(
                    tensor.tensor_id for tensor in manifest.tensors
                ),
                resident_bytes=manifest.tensor_bytes,
                generation=1,
                executor_id="executor:accelerator-b",
            ),),
        )

        candidates = scheduler.generate_automated_candidates(
            request("foreign-residency"), manifest.model_id, snapshot
        )
        host_hot = next(
            row for row in candidates.candidates
            if row.route_family == "whole_model"
            and row.device_ids == ("host-a",)
            and row.residency_variant == "hot"
        )

        self.assertFalse(host_hot.admitted)
        self.assertIn(
            "RESIDENCY_EXECUTOR_MISMATCH",
            host_hot.rejection_reasons,
        )
        self.assertNotEqual(
            candidates.baseline_route_id, host_hot.candidate_id
        )

    def test_transition_can_atomically_reserve_every_execution_slot(self) -> None:
        source = catalog()
        resources = dict(source.resources)
        resources["compute:accelerator-b"] = replace(
            resources["compute:accelerator-b"], capacity=4
        )
        transitions = tuple(
            replace(
                row,
                resource_slots={"compute:accelerator-b": 4},
            )
            if row.device_id == "accelerator-b" else row
            for row in source.transitions
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source, resources=resources, transitions=transitions
        ).to_json())
        scheduler, manifest = self.scheduler_and_manifest(profile)
        snapshot = runtime_snapshot(
            manifest, resident_devices=("host-a", "helper-c")
        )
        candidates = scheduler.generate_automated_candidates(
            request("exclusive-transition"), manifest.model_id, snapshot
        )
        row = next(
            item for item in candidates.candidates
            if item.device_ids == ("accelerator-b",)
            and item.residency_variant == "cold"
        )
        self.assertEqual(
            row.plan.resource_slots["compute:accelerator-b"], 1
        )
        preview = scheduler._automated_compiler().preview_candidate(
            row,
            request("exclusive-transition-preview"),
            snapshot,
        )
        lease = next(
            item for item in preview.plans
            if item.resource_id == "compute:accelerator-b"
        )
        self.assertEqual(len(lease.lanes), 4)
        execution = next(item for item in preview.plans
                         if item.resource_id == "compute:accelerator-b" and item is not lease)
        self.assertEqual(len(execution.lanes), 1)
        self.assertEqual(execution.start_us, lease.predicted_end_us)

    def test_future_transition_does_not_replan_unchanged_snapshot(
        self,
    ) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        cold = runtime_snapshot(manifest, resident_devices=())
        scheduler.submit_automated_request(
            request("future-owner", arrival_us=1_000),
            manifest.model_id,
            cold,
            selection_mode="desktop-baseline",
        )
        follower = scheduler.submit_automated_request(
            request("future-follower", arrival_us=1_100),
            manifest.model_id,
            cold,
            selection_mode="desktop-baseline",
        )

        changed = scheduler.observe_automated_runtime_snapshot(
            cold, observed_at_us=1_200
        )

        self.assertEqual(changed, ())
        self.assertEqual(
            scheduler.runtime_ticket(
                follower.request.request_id
            ).dispatch_state,
            "QUEUED",
        )

    def test_intervening_transition_keeps_later_reload_queued(self) -> None:
        source = catalog()
        gpu = replace(
            source.executor_by_device["accelerator-b"],
            qualified_fallback=True,
            exclusive_residency_resource_id="compute:accelerator-b",
        )
        transition = replace(
            next(
                row for row in source.transitions
                if row.device_id == "accelerator-b"
            ),
            executor_id=gpu.executor_id,
            prepares_device_ids=("accelerator-b",),
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=(gpu,),
            composite_executors=(),
            transitions=(transition,),
        ).to_json())
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        first_path = Path(self.directory.name) / "intervening-large.gguf"
        write_synthetic_gguf(
            first_path, block_count=2, sliding_window=32
        )
        first_model = scheduler.register_gguf_model(
            "queue-model-a", first_path
        )
        second_model = scheduler.register_gguf_model(
            "queue-model-b", self.path
        )
        cold = replace(
            runtime_snapshot(
                first_model,
                include_phone=False,
                resident_devices=(),
            ),
            executors={
                gpu.executor_id: executor_state(gpu.executor_id)
            },
        )
        first = scheduler.submit_automated_request(
            request("queue-a-first", arrival_us=1_000),
            first_model.model_id,
            cold,
            selection_mode="desktop-baseline",
        )
        second = scheduler.submit_automated_request(
            request("queue-b", arrival_us=1_100),
            second_model.model_id,
            cold,
            selection_mode="desktop-baseline",
        )
        third = scheduler.submit_automated_request(
            request("burstgpt-v2:88139", arrival_us=1_200),
            first_model.model_id,
            cold,
            selection_mode="desktop-baseline",
        )
        fourth = scheduler.submit_automated_request(
            request("queue-b-fourth", arrival_us=1_300),
            second_model.model_id,
            cold,
            selection_mode="desktop-baseline",
        )
        self.assertGreaterEqual(
            second.decision.start_us,
            max(row.reserved_until_us for row in first.decision.leases),
        )
        self.assertGreaterEqual(
            third.decision.start_us,
            max(row.reserved_until_us for row in second.decision.leases),
        )
        acquired = scheduler.wait_runtime_request(
            first.request.request_id,
            time.monotonic_ns() - first.decision.start_us * 1_000,
        )
        scheduler.record_automated_transition_receipts(
            first.request.request_id,
            FakeAutomatedPhysicalAdapter._transition_receipts(acquired),
        )
        observed_at_us = first.decision.start_us + 1
        reserve_bytes = 10_000
        first_reclaimable_bytes = sum(
            demand.required_bytes
            for demand in first.execution_plan.memory_demands
            if demand.resource_id == "gpu-memory"
        )
        hot = replace(
            cold,
            snapshot_id="synthetic-queue-a-hot",
            captured_at_us=observed_at_us,
            valid_until_us=20_000_000,
            memory=RuntimePlacementSnapshot(
                snapshot_id="synthetic-queue-a-hot-memory",
                captured_at_us=observed_at_us,
                valid_until_us=20_000_000,
                capacities={
                    "gpu-memory": DeviceMemoryCapacity(
                        "gpu-memory",
                        first_reclaimable_bytes + reserve_bytes + 100,
                        first_reclaimable_bytes,
                        reserve_bytes,
                    )
                },
            ),
            residency=(ModelResidencyObservation(
                model_id=first_model.model_id,
                artifact_sha256=first_model.artifact_sha256,
                device_id="accelerator-b",
                state="hot",
                resident_tensor_ids=tuple(
                    row.tensor_id for row in first_model.tensors
                ),
                resident_bytes=first_model.tensor_bytes,
                generation=1,
                executor_id=gpu.executor_id,
                reclaimable_bytes=first_reclaimable_bytes,
            ),),
        )

        changed = scheduler.observe_automated_runtime_snapshot(
            hot, observed_at_us=observed_at_us
        )

        self.assertEqual(changed, ())
        self.assertEqual(
            scheduler.runtime_ticket(third.request.request_id).dispatch_state,
            "QUEUED",
        )

        execution = scheduler.runtime_execution_ticket(
            first.request.request_id
        )
        completed_at_us = max(
            observed_at_us + 1,
            second.request.arrival_us,
        )
        first_completion = scheduler.complete_automated_request(
            first.request.request_id,
            self.execution_receipt(
                execution,
                finished_us=completed_at_us,
            ),
        )
        self.assertEqual(
            first_completion.completion_event_replans,
            (second.request.request_id,),
        )
        wake = scheduler.wait_runtime_request(
            second.request.request_id,
            time.monotonic_ns() - completed_at_us * 1_000,
        )
        self.assertEqual(wake.dispatch_state, "REPLAN_REQUIRED")
        with mock.patch.object(
            scheduler,
            "_automated_snapshot_for_request",
            wraps=scheduler._automated_snapshot_for_request,
        ) as projection:
            replanned = scheduler.replan_automated_request(
                second.request.request_id,
                observed_at_us=completed_at_us,
                reason=wake.dispatch_receipt.wake_reason,
                snapshot=replace(
                    hot,
                    captured_at_us=completed_at_us,
                    memory=replace(
                        hot.memory,
                        captured_at_us=completed_at_us,
                    ),
                ),
            )
        self.assertEqual(
            projection.call_args.kwargs.get("project_before_us"),
            second.decision.start_us,
        )
        evictions = tuple(
            eviction
            for transition in replanned.execution_plan.transitions
            for eviction in transition.evictions
        )
        self.assertTrue(evictions)
        self.assertEqual({row.generation for row in evictions}, {1})
        queued_follower = scheduler.runtime_ticket(
            third.request.request_id
        )
        self.assertEqual(queued_follower.lease_status, "CANCELLED")
        self.assertEqual(
            queued_follower.memory_reservation_status, "CANCELLED"
        )
        later_follower = scheduler.runtime_ticket(
            fourth.request.request_id
        )
        self.assertEqual(later_follower.lease_status, "CANCELLED")
        self.assertEqual(
            later_follower.memory_reservation_status, "CANCELLED"
        )
        self.assertEqual(queued_follower.dispatch_state, "QUEUED")
        self.assertEqual(later_follower.dispatch_state, "QUEUED")
        queue_states = scheduler.runtime_controller_snapshot()[
            "dispatch_queue"
        ]["entry_states"]
        self.assertEqual(
            queue_states[third.request.request_id]["state"],
            "DEFERRED_REPLAN",
        )
        self.assertEqual(
            queue_states[fourth.request.request_id]["state"],
            "DEFERRED_REPLAN",
        )
        self.assertLess(
            replanned.decision.start_us,
            third.decision.start_us,
        )

        acquired_second = scheduler.wait_runtime_request(
            second.request.request_id,
            time.monotonic_ns() - replanned.decision.start_us * 1_000,
        )
        scheduler.record_automated_transition_receipts(
            second.request.request_id,
            FakeAutomatedPhysicalAdapter._transition_receipts(
                acquired_second
            ),
        )
        second_execution = scheduler.runtime_execution_ticket(
            second.request.request_id
        )
        second_completed_at_us = max(
            replanned.decision.start_us + 2,
            third.request.arrival_us,
        )
        scheduler.complete_automated_request(
            second.request.request_id,
            self.execution_receipt(
                second_execution, finished_us=second_completed_at_us
            ),
        )
        follower = scheduler.wait_runtime_request(
            third.request.request_id,
            time.monotonic_ns() - second_completed_at_us * 1_000,
        )
        self.assertEqual(follower.dispatch_state, "REPLAN_REQUIRED")
        self.assertEqual(
            follower.dispatch_receipt.wake_reason,
            "capacity_released_early",
        )
        second_reclaimable_bytes = sum(
            demand.required_bytes
            for demand in replanned.execution_plan.memory_demands
            if demand.resource_id == "gpu-memory"
        )
        second_hot = replace(
            hot,
            snapshot_id="synthetic-queue-b-hot",
            captured_at_us=second_completed_at_us,
            memory=replace(
                hot.memory,
                snapshot_id="synthetic-queue-b-hot-memory",
                captured_at_us=second_completed_at_us,
                capacities={
                    "gpu-memory": DeviceMemoryCapacity(
                        "gpu-memory",
                        max(
                            first_reclaimable_bytes,
                            second_reclaimable_bytes,
                        ) + reserve_bytes + 100,
                        second_reclaimable_bytes,
                        reserve_bytes,
                    )
                },
            ),
            residency=(ModelResidencyObservation(
                model_id=second_model.model_id,
                artifact_sha256=second_model.artifact_sha256,
                device_id="accelerator-b",
                state="hot",
                resident_tensor_ids=tuple(
                    row.tensor_id for row in second_model.tensors
                ),
                resident_bytes=second_model.tensor_bytes,
                generation=2,
                executor_id=gpu.executor_id,
                reclaimable_bytes=second_reclaimable_bytes,
            ),),
        )
        third_replanned = scheduler.replan_automated_request(
            third.request.request_id,
            observed_at_us=second_completed_at_us,
            reason=follower.dispatch_receipt.wake_reason,
            snapshot=second_hot,
        )
        self.assertEqual(third_replanned.previous_ticket_id, third.ticket_id)
        self.assertTrue(third_replanned.execution_plan.transitions)
        self.assertGreaterEqual(
            third_replanned.decision.start_us,
            second_completed_at_us,
        )
        selected = next(
            row for row in third_replanned.cost_estimates.estimates
            if row.route_id == third_replanned.decision.route_id
        )
        resolution = selected.details["model_placement_resolution"]
        self.assertLessEqual(resolution["pass_count"], 2)
        self.assertEqual(
            resolution["passes"][-1]["live_selected_route_id"],
            third_replanned.decision.route_id,
        )
        self.assertEqual(
            resolution["passes"][-1]["rejection_reasons"], []
        )
        self.assertEqual(
            scheduler.runtime_ticket(
                fourth.request.request_id
            ).dispatch_state,
            "QUEUED",
        )

    def test_authoritative_replacement_projects_target_before_publish(
        self,
    ) -> None:
        source = catalog()
        gpu = replace(
            source.executor_by_device["accelerator-b"],
            qualified_fallback=True,
            exclusive_residency_resource_id="compute:accelerator-b",
        )
        transition = replace(
            next(
                row for row in source.transitions
                if row.device_id == "accelerator-b"
            ),
            executor_id=gpu.executor_id,
            prepares_device_ids=("accelerator-b",),
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=(gpu,),
            composite_executors=(),
            transitions=(transition,),
        ).to_json())
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        first_path = Path(self.directory.name) / "active-replacement-a.gguf"
        write_synthetic_gguf(
            first_path, block_count=2, sliding_window=32
        )
        first_model = scheduler.register_gguf_model(
            "active-replacement-a", first_path
        )
        second_model = scheduler.register_gguf_model(
            "active-replacement-b", self.path
        )
        cold = replace(
            runtime_snapshot(
                first_model,
                include_phone=False,
                resident_devices=(),
            ),
            executors={
                gpu.executor_id: executor_state(gpu.executor_id)
            },
        )
        first = scheduler.submit_automated_request(
            request("active-replacement-a-first", arrival_us=1_000),
            first_model.model_id,
            cold,
            selection_mode="desktop-baseline",
        )
        first = scheduler.wait_runtime_request(
            first.request.request_id,
            time.monotonic_ns() - first.decision.start_us * 1_000,
        )
        scheduler.record_automated_transition_receipts(
            first.request.request_id,
            FakeAutomatedPhysicalAdapter._transition_receipts(first),
        )
        first = scheduler.runtime_execution_ticket(
            first.request.request_id
        )
        first_finished_us = first.decision.start_us + 1
        scheduler.complete_automated_request(
            first.request.request_id,
            self.execution_receipt(first, finished_us=first_finished_us),
        )
        first_reclaimable_bytes = sum(
            demand.required_bytes
            for demand in first.execution_plan.memory_demands
            if demand.resource_id == "gpu-memory"
        )
        hot_first = replace(
            cold,
            snapshot_id="active-replacement-a-hot",
            captured_at_us=first_finished_us,
            memory=replace(
                cold.memory,
                snapshot_id="active-replacement-a-hot-memory",
                captured_at_us=first_finished_us,
                capacities={
                    **dict(cold.memory.capacities),
                    "gpu-memory": replace(
                        cold.memory.capacities["gpu-memory"],
                        occupied_bytes=first_reclaimable_bytes,
                    ),
                },
            ),
            residency=(ModelResidencyObservation(
                model_id=first_model.model_id,
                artifact_sha256=first_model.artifact_sha256,
                device_id="accelerator-b",
                state="hot",
                resident_tensor_ids=tuple(
                    row.tensor_id for row in first_model.tensors
                ),
                resident_bytes=first_model.tensor_bytes,
                generation=1,
                executor_id=gpu.executor_id,
                reclaimable_bytes=first_reclaimable_bytes,
            ),),
        )
        second = scheduler.submit_automated_request(
            request(
                "active-replacement-b-active",
                arrival_us=first_finished_us + 1,
            ),
            second_model.model_id,
            hot_first,
            selection_mode="desktop-baseline",
        )
        self.assertTrue(second.execution_plan.transitions[0].evictions)
        second_reclaimable_bytes = sum(
            demand.required_bytes
            for demand in second.execution_plan.memory_demands
            if demand.resource_id == "gpu-memory"
        )
        reserve_bytes = 10_000
        capacity_bytes = max(
            first_reclaimable_bytes, second_reclaimable_bytes
        ) + reserve_bytes + 100
        observed_at_us = second.decision.start_us + 2
        transition_in_flight = replace(
            cold,
            snapshot_id="active-replacement-target-not-published",
            captured_at_us=observed_at_us,
            memory=replace(
                cold.memory,
                snapshot_id=(
                    "active-replacement-target-not-published-memory"
                ),
                captured_at_us=observed_at_us,
                capacities={
                    **dict(cold.memory.capacities),
                    "gpu-memory": DeviceMemoryCapacity(
                        "gpu-memory",
                        capacity_bytes,
                        second_reclaimable_bytes,
                        reserve_bytes,
                    ),
                },
            ),
            residency=(),
        )
        later_request = request(
            "active-replacement-a-later",
            arrival_us=observed_at_us,
        )
        self.assertIn(
            second.request.request_id,
            scheduler._runtime_controller.projection_request_ids(),
        )
        pending_projection = scheduler._automated_snapshot_for_request(
            later_request,
            transition_in_flight,
            project_before_us=observed_at_us,
        )
        self.assertEqual(
            {
                row.artifact_sha256
                for row in pending_projection.residency
                if row.device_id == "accelerator-b"
            },
            {second_model.artifact_sha256},
        )
        later = scheduler.submit_automated_request(
            later_request,
            first_model.model_id,
            transition_in_flight,
            observed_at_us=observed_at_us,
            selection_mode="desktop-baseline",
        )

        self.assertEqual(later.dispatch_state, "QUEUED")
        self.assertGreaterEqual(
            later.decision.start_us,
            max(second.final_reserved_until_us.values()),
        )
        evictions = tuple(
            eviction
            for row in later.execution_plan.transitions
            for eviction in row.evictions
        )
        self.assertTrue(evictions)
        self.assertEqual(
            {row.artifact_sha256 for row in evictions},
            {second_model.artifact_sha256},
        )
        next_observed_at_us = observed_at_us + 1
        target_loading = replace(
            transition_in_flight,
            snapshot_id="active-replacement-target-loading",
            captured_at_us=next_observed_at_us,
            memory=replace(
                transition_in_flight.memory,
                snapshot_id="active-replacement-target-loading-memory",
                captured_at_us=next_observed_at_us,
            ),
        )
        repeated = scheduler.submit_automated_request(
            request(
                "active-replacement-b-repeated",
                arrival_us=next_observed_at_us,
            ),
            second_model.model_id,
            target_loading,
            observed_at_us=next_observed_at_us,
            selection_mode="desktop-baseline",
        )

        self.assertEqual(repeated.dispatch_state, "QUEUED")
        self.assertGreaterEqual(
            repeated.decision.start_us,
            max(later.final_reserved_until_us.values()),
        )
        repeated_evictions = tuple(
            eviction
            for row in repeated.execution_plan.transitions
            for eviction in row.evictions
        )
        self.assertTrue(repeated_evictions)
        self.assertEqual(
            {row.artifact_sha256 for row in repeated_evictions},
            {first_model.artifact_sha256},
        )

    def test_authoritative_replacement_does_not_reclaim_unobserved_memory(
        self,
    ) -> None:
        source = catalog()
        gpu = replace(
            source.executor_by_device["accelerator-b"],
            qualified_fallback=True,
            exclusive_residency_resource_id="compute:accelerator-b",
        )
        transition = replace(
            next(
                row for row in source.transitions
                if row.device_id == "accelerator-b"
            ),
            executor_id=gpu.executor_id,
            prepares_device_ids=("accelerator-b",),
        )
        host = replace(
            source.executor_by_device["host-a"],
            qualified_fallback=False,
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=(
                host,
                gpu,
            ),
            composite_executors=(),
            transitions=(transition,),
        ).to_json())
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        first_path = Path(self.directory.name) / "unobserved-host-a.gguf"
        write_synthetic_gguf(first_path, block_count=2, sliding_window=32)
        first_model = scheduler.register_gguf_model(
            "unobserved-host-a", first_path
        )
        second_model = scheduler.register_gguf_model(
            "unobserved-host-b", self.path
        )
        hot = replace(
            runtime_snapshot(
                first_model,
                include_phone=False,
                resident_devices=("accelerator-b",),
            ),
            executors={
                gpu.executor_id: executor_state(gpu.executor_id)
            },
        )
        hot = replace(
            hot,
            residency=(replace(
                hot.residency[0],
                executor_id=gpu.executor_id,
                reclaimable_bytes=hot.residency[0].resident_bytes,
            ),),
        )
        ticket = scheduler.submit_automated_request(
            request("unobserved-host-replacement"),
            second_model.model_id,
            hot,
            selection_mode="desktop-baseline",
        )
        ticket = scheduler.wait_runtime_request(
            ticket.request.request_id,
            time.monotonic_ns() - ticket.decision.start_us * 1_000,
        )
        host_baseline = ticket.runtime_observation.memory_occupied_bytes[
            "host-memory"
        ]
        observed_at_us = ticket.decision.start_us + 1
        in_flight = replace(
            hot,
            snapshot_id="unobserved-host-in-flight",
            captured_at_us=observed_at_us,
            memory=replace(
                hot.memory,
                snapshot_id="unobserved-host-in-flight-memory",
                captured_at_us=observed_at_us,
            ),
        )
        host_allocation_bytes = 100
        host_eviction = RuntimeResidencyEviction(
            model_id=first_model.model_id,
            artifact_sha256=first_model.artifact_sha256,
            device_id="host-a",
            resident_bytes=host_baseline + 1,
            generation=1,
            executor_id="executor:host-a",
            reclaimable_bytes=host_baseline + 1,
        )
        gpu_eviction = ticket.execution_plan.transitions[0].evictions[0]

        projected = _authoritative_replacement_occupancy(
            in_flight,
            profile,
            ticket,
            "compute:accelerator-b",
            ticket.decision.start_us,
            max(ticket.final_reserved_until_us.values()),
            (
                ("accelerator-b", 100, 100),
                ("host-a", host_allocation_bytes, host_allocation_bytes),
            ),
            (gpu_eviction, host_eviction),
        )

        self.assertEqual(
            projected["host-memory"],
            host_baseline + host_allocation_bytes,
        )

    def test_authoritative_full_domain_reflects_inflight_target(self) -> None:
        source = catalog()
        gpu = replace(
            source.executor_by_device["accelerator-b"],
            qualified_fallback=True,
            exclusive_residency_resource_id="compute:accelerator-b",
        )
        transition = replace(
            next(
                row for row in source.transitions
                if row.device_id == "accelerator-b"
            ),
            executor_id=gpu.executor_id,
            prepares_device_ids=("accelerator-b",),
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=(gpu,),
            composite_executors=(),
            transitions=(transition,),
        ).to_json())
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        previous_path = Path(self.directory.name) / "full-domain-old.gguf"
        write_synthetic_gguf(previous_path, block_count=2, sliding_window=32)
        previous = scheduler.register_gguf_model(
            "full-domain-old", previous_path
        )
        target = scheduler.register_gguf_model("full-domain-new", self.path)
        hot = replace(
            runtime_snapshot(
                previous,
                include_phone=False,
                resident_devices=("accelerator-b",),
            ),
            executors={
                gpu.executor_id: executor_state(gpu.executor_id)
            },
        )
        hot = replace(
            hot,
            residency=(replace(
                hot.residency[0],
                executor_id=gpu.executor_id,
                reclaimable_bytes=hot.residency[0].resident_bytes,
            ),),
        )
        ticket = scheduler.submit_automated_request(
            request("full-domain-inflight"),
            target.model_id,
            hot,
            selection_mode="desktop-baseline",
        )
        weight = next(
            demand for demand in ticket.execution_plan.memory_demands
            if demand.demand_id == "weights:accelerator-b"
        )
        allocated_bytes = sum(
            demand.required_bytes
            for demand in ticket.execution_plan.memory_demands
            if demand.resource_id == "gpu-memory"
        )
        full_bytes = allocated_bytes + 1_024
        observed_at_us = ticket.decision.start_us + 1
        in_flight = replace(
            hot,
            snapshot_id="full-domain-inflight",
            captured_at_us=observed_at_us,
            memory=replace(
                hot.memory,
                snapshot_id="full-domain-inflight-memory",
                captured_at_us=observed_at_us,
                capacities={
                    **dict(hot.memory.capacities),
                    "gpu-memory": DeviceMemoryCapacity(
                        "gpu-memory", full_bytes, full_bytes, 0
                    ),
                },
            ),
        )

        projected = _authoritative_replacement_occupancy(
            in_flight,
            profile,
            ticket,
            "compute:accelerator-b",
            ticket.decision.start_us,
            max(ticket.final_reserved_until_us.values()),
            ((
                "accelerator-b",
                weight.required_bytes,
                allocated_bytes,
            ),),
            ticket.execution_plan.transitions[0].evictions,
        )

        self.assertEqual(projected["gpu-memory"], full_bytes)

    def test_observed_transition_invalidates_stale_downstream_projection(
        self,
    ) -> None:
        source = catalog()
        gpu = replace(
            source.executor_by_device["accelerator-b"],
            qualified_fallback=True,
            exclusive_residency_resource_id="compute:accelerator-b",
        )
        transition = replace(
            next(
                row for row in source.transitions
                if row.device_id == "accelerator-b"
            ),
            executor_id=gpu.executor_id,
            prepares_device_ids=("accelerator-b",),
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=(gpu,),
            composite_executors=(),
            transitions=(transition,),
        ).to_json())
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        first_path = Path(self.directory.name) / "downstream-first.gguf"
        write_synthetic_gguf(
            first_path, block_count=2, sliding_window=32
        )
        first_model = scheduler.register_gguf_model(
            "downstream-model-a", first_path
        )
        second_model = scheduler.register_gguf_model(
            "downstream-model-b", self.path
        )
        cold = replace(
            runtime_snapshot(
                first_model,
                include_phone=False,
                resident_devices=(),
            ),
            executors={
                gpu.executor_id: executor_state(gpu.executor_id)
            },
        )
        first = scheduler.submit_automated_request(
            request("downstream-a-first", arrival_us=1_000),
            first_model.model_id,
            cold,
            selection_mode="desktop-baseline",
        )
        intervening = scheduler.submit_automated_request(
            request("downstream-b", arrival_us=1_100),
            second_model.model_id,
            cold,
            selection_mode="desktop-baseline",
        )
        stale = scheduler.submit_automated_request(
            request("downstream-a-stale", arrival_us=1_200),
            first_model.model_id,
            cold,
            selection_mode="desktop-baseline",
        )
        active = scheduler.wait_runtime_request(
            first.request.request_id,
            time.monotonic_ns() - first.decision.start_us * 1_000,
        )
        scheduler.record_automated_transition_receipts(
            first.request.request_id,
            FakeAutomatedPhysicalAdapter._transition_receipts(active),
        )
        self.assertTrue(
            scheduler._runtime_controller.require_queued_replan(
                intervening.request.request_id,
                "residency_projection_invalid",
                defer_behind_predecessors=True,
            )
        )
        observed_at_us = first.decision.start_us + 1
        resident_bytes = sum(
            demand.required_bytes
            for demand in first.execution_plan.memory_demands
            if demand.resource_id == "gpu-memory"
        )
        hot = replace(
            cold,
            snapshot_id="synthetic-downstream-a-hot",
            captured_at_us=observed_at_us,
            valid_until_us=20_000_000,
            memory=replace(
                cold.memory,
                snapshot_id="synthetic-downstream-a-hot-memory",
                captured_at_us=observed_at_us,
                valid_until_us=20_000_000,
            ),
            residency=(ModelResidencyObservation(
                model_id=first_model.model_id,
                artifact_sha256=first_model.artifact_sha256,
                device_id="accelerator-b",
                state="hot",
                resident_tensor_ids=tuple(
                    row.tensor_id for row in first_model.tensors
                ),
                resident_bytes=first_model.tensor_bytes,
                generation=1,
                executor_id=gpu.executor_id,
                reclaimable_bytes=resident_bytes,
            ),),
        )

        changed = scheduler.observe_automated_runtime_snapshot(
            hot, observed_at_us=observed_at_us
        )

        self.assertEqual(changed, (stale.request.request_id,))
        queue = scheduler.runtime_controller_snapshot()["dispatch_queue"]
        self.assertEqual(
            queue["entry_states"][stale.request.request_id]["state"],
            "DEFERRED_REPLAN",
        )
        current = scheduler.runtime_ticket(stale.request.request_id)
        self.assertEqual(current.lease_status, "CANCELLED")
        self.assertEqual(current.memory_reservation_status, "CANCELLED")

    def test_projection_rejects_changed_reclaimable_bytes(self) -> None:
        source = catalog()
        gpu = replace(
            source.executor_by_device["accelerator-b"],
            qualified_fallback=True,
            exclusive_residency_resource_id="compute:accelerator-b",
        )
        transition = replace(
            next(
                row for row in source.transitions
                if row.device_id == "accelerator-b"
            ),
            executor_id=gpu.executor_id,
            prepares_device_ids=("accelerator-b",),
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=(gpu,),
            composite_executors=(),
            transitions=(transition,),
        ).to_json())
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        first_path = Path(self.directory.name) / "projection-first.gguf"
        write_synthetic_gguf(first_path, block_count=2, sliding_window=32)
        first_model = scheduler.register_gguf_model(
            "projection-model-a", first_path
        )
        second_model = scheduler.register_gguf_model(
            "projection-model-b", self.path
        )
        hot = replace(
            runtime_snapshot(
                first_model,
                include_phone=False,
                resident_devices=("accelerator-b",),
            ),
            executors={
                gpu.executor_id: executor_state(gpu.executor_id)
            },
        )
        hot = replace(
            hot,
            residency=(replace(
                hot.residency[0],
                executor_id=gpu.executor_id,
                reclaimable_bytes=hot.residency[0].resident_bytes,
            ),),
        )
        ticket = scheduler.submit_automated_request(
            request("projection-stale-eviction"),
            second_model.model_id,
            hot,
            selection_mode="desktop-baseline",
        )
        self.assertTrue(ticket.execution_plan.transitions)
        current = hot.residency[0]
        reclaimable = (
            current.resident_bytes
            if current.reclaimable_bytes is None
            else current.reclaimable_bytes
        )
        capacity = hot.memory.capacities["gpu-memory"]
        changed = replace(
            hot,
            snapshot_id="projection-changed-reclaimable",
            memory=replace(
                hot.memory,
                snapshot_id="projection-changed-reclaimable-memory",
                capacities={
                    "gpu-memory": replace(
                        capacity,
                        occupied_bytes=capacity.occupied_bytes + 1,
                    )
                },
            ),
            residency=(replace(
                current,
                reclaimable_bytes=reclaimable + 1,
            ),),
        )

        with self.assertRaises(RuntimeResidencyProjectionError) as raised:
            project_scheduler_residency(
                changed,
                profile,
                (ticket,),
                {
                    first_model.model_id: first_model,
                    second_model.model_id: second_model,
                },
            )

        self.assertEqual(
            raised.exception.request_id, ticket.request.request_id
        )
        self.assertEqual(raised.exception.ticket_id, ticket.ticket_id)

    def test_arrival_baseline_rejected_behind_queued_replacement_replans(
        self,
    ) -> None:
        source = catalog()
        gpu = replace(
            source.executor_by_device["accelerator-b"],
            qualified_fallback=True,
            exclusive_residency_resource_id="compute:accelerator-b",
        )
        transition = replace(
            next(
                row for row in source.transitions
                if row.device_id == "accelerator-b"
            ),
            executor_id=gpu.executor_id,
            prepares_device_ids=("accelerator-b",),
            resource_slots={"compute:accelerator-b": 2},
        )
        resources = dict(source.resources)
        resources["compute:accelerator-b"] = replace(
            resources["compute:accelerator-b"], capacity=2
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=(gpu,),
            composite_executors=(),
            transitions=(transition,),
            resources=resources,
        ).to_json())
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        other_path = Path(self.directory.name) / "replacement-other.gguf"
        write_synthetic_gguf(other_path, block_count=2, sliding_window=32)
        other = scheduler.register_gguf_model(
            "replacement-other", other_path
        )
        target = scheduler.register_gguf_model(
            "replacement-target", self.path
        )
        hot = replace(
            runtime_snapshot(
                target,
                include_phone=False,
                gpu_free_slots=2,
                resident_devices=("accelerator-b",),
            ),
            executors={
                gpu.executor_id: executor_state(
                    gpu.executor_id, free_slots=2
                )
            },
        )
        scheduler.submit_automated_request(
            request("replacement-first", arrival_us=1_000, output_tokens=40),
            target.model_id,
            hot,
            selection_mode="energy-aware",
        )
        load = scheduler.submit_automated_request(
            request("replacement-load", arrival_us=1_050),
            other.model_id,
            hot,
            selection_mode="energy-aware",
        )
        self.assertTrue(load.execution_plan.transitions)
        self.assertGreater(load.decision.start_us, 1_100)

        # Without a published epoch the arrival is planned against the
        # observed residency, so the hot baseline's calendar slot overlaps
        # the queued replacement and the memory ledger rejects it.
        with mock.patch.object(
            scheduler,
            "_published_epoch_for_request",
            return_value=(None, None, "NONE"),
        ):
            arrival = scheduler.submit_automated_request(
                request(
                    "replacement-arrival",
                    arrival_us=1_100,
                    output_tokens=400,
                ),
                target.model_id,
                hot,
                selection_mode="energy-aware",
            )

        self.assertEqual(
            {
                (row.model_id, row.device_id)
                for transition in arrival.execution_plan.transitions
                for row in transition.evictions
            },
            {(other.model_id, "accelerator-b")},
        )
        self.assertGreaterEqual(
            arrival.decision.start_us,
            max(row.reserved_until_us for row in load.decision.leases),
        )

    def test_replanned_predecessor_is_not_pushed_behind_dependents(
        self,
    ) -> None:
        source = catalog()
        gpu = replace(
            source.executor_by_device["accelerator-b"],
            qualified_fallback=True,
            exclusive_residency_resource_id="compute:accelerator-b",
        )
        transition = replace(
            next(
                row for row in source.transitions
                if row.device_id == "accelerator-b"
            ),
            executor_id=gpu.executor_id,
            prepares_device_ids=("accelerator-b",),
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=(gpu,),
            composite_executors=(),
            transitions=(transition,),
        ).to_json())
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        other_path = Path(self.directory.name) / "starved-other.gguf"
        write_synthetic_gguf(other_path, block_count=2, sliding_window=32)
        other = scheduler.register_gguf_model("starved-other", other_path)
        target = scheduler.register_gguf_model("starved-target", self.path)
        executors = {gpu.executor_id: executor_state(gpu.executor_id)}
        hot = replace(
            runtime_snapshot(
                target,
                include_phone=False,
                resident_devices=("accelerator-b",),
            ),
            executors=executors,
        )
        cold = replace(
            runtime_snapshot(
                target, include_phone=False, resident_devices=()
            ),
            executors=executors,
        )
        root = scheduler.submit_automated_request(
            request("starved-root", arrival_us=1_000),
            target.model_id,
            hot,
            selection_mode="desktop-baseline",
        )
        self.assertEqual(root.execution_plan.transitions, ())
        dependents = tuple(
            scheduler.submit_automated_request(
                request("starved-dependent-" + str(index),
                        arrival_us=1_100 + index),
                other.model_id,
                hot,
                selection_mode="desktop-baseline",
            )
            for index in range(2)
        )
        controller = scheduler._runtime_controller
        causal = controller.projection_causal_predecessors()
        for dependent in dependents:
            self.assertIn(
                root.request.request_id,
                causal[dependent.request.request_id],
            )
        self.assertTrue(controller.require_queued_replan(
            root.request.request_id, "residency_observation_changed",
        ))
        wake = scheduler.wait_runtime_request(
            root.request.request_id,
            time.monotonic_ns() - root.decision.start_us * 1_000,
        )
        self.assertEqual(wake.dispatch_state, "REPLAN_REQUIRED")

        replanned = scheduler.replan_automated_request(
            root.request.request_id,
            observed_at_us=wake.dispatch_receipt.observed_at_us,
            reason=wake.dispatch_receipt.wake_reason,
            snapshot=cold,
            expected_ticket_id=wake.ticket_id,
            expected_queue_generation=(
                wake.dispatch_receipt.queue_generation
            ),
        )

        self.assertTrue(replanned.execution_plan.transitions)
        self.assertLessEqual(
            replanned.decision.start_us,
            min(row.decision.start_us for row in dependents),
        )
        self.assertEqual(
            replanned.decision.start_us,
            wake.dispatch_receipt.observed_at_us,
        )
        states = scheduler.runtime_controller_snapshot()[
            "dispatch_queue"
        ]["entry_states"]
        for dependent in dependents:
            current = scheduler.runtime_ticket(dependent.request.request_id)
            self.assertTrue(
                current.lease_status == "CANCELLED"
                or current.decision.start_us
                    >= replanned.decision.start_us
            )
            if current.lease_status == "CANCELLED":
                self.assertEqual(
                    states[dependent.request.request_id]["state"],
                    "DEFERRED_REPLAN",
                )

    def _two_lane_gpu_replacement_scheduler(
        self, prefix: str
    ) -> tuple[UnifiedScheduler, object, object, object]:
        # A load holds both GPU lanes, its execution only one, so other work
        # can share the GPU with it in the calendar but not in memory.
        source = catalog()
        gpu = replace(
            source.executor_by_device["accelerator-b"],
            qualified_fallback=True,
            exclusive_residency_resource_id="compute:accelerator-b",
        )
        transition = replace(
            next(
                row for row in source.transitions
                if row.device_id == "accelerator-b"
            ),
            executor_id=gpu.executor_id,
            prepares_device_ids=("accelerator-b",),
            resource_slots={"compute:accelerator-b": 2},
        )
        resources = dict(source.resources)
        resources["compute:accelerator-b"] = replace(
            resources["compute:accelerator-b"], capacity=2
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=(gpu,),
            composite_executors=(),
            transitions=(transition,),
            resources=resources,
        ).to_json())
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        other_path = Path(self.directory.name) / (prefix + "-other.gguf")
        write_synthetic_gguf(other_path, block_count=2, sliding_window=32)
        other = scheduler.register_gguf_model(prefix + "-other", other_path)
        target = scheduler.register_gguf_model(prefix + "-target", self.path)
        hot = replace(
            runtime_snapshot(
                target,
                include_phone=False,
                gpu_free_slots=2,
                resident_devices=("accelerator-b",),
            ),
            executors={
                gpu.executor_id: executor_state(
                    gpu.executor_id, free_slots=2
                )
            },
        )
        return scheduler, other, target, hot

    def _wake_root_into_dependent_replacement(
        self, prefix: str
    ) -> tuple[UnifiedScheduler, object, object, object, object]:
        scheduler, other, target, hot = (
            self._two_lane_gpu_replacement_scheduler(prefix)
        )
        root = scheduler.submit_automated_request(
            request(prefix + "-root", arrival_us=1_000, output_tokens=40),
            target.model_id,
            hot,
            selection_mode="energy-aware",
        )
        self.assertEqual(root.execution_plan.transitions, ())
        dependent = scheduler.submit_automated_request(
            request(prefix + "-dependent", arrival_us=1_100),
            other.model_id,
            hot,
            selection_mode="energy-aware",
        )
        self.assertTrue(dependent.execution_plan.transitions)
        controller = scheduler._runtime_controller
        self.assertIn(
            root.request.request_id,
            controller.projection_causal_predecessors()[
                dependent.request.request_id
            ],
        )
        root_end_us = max(
            row.reserved_until_us for row in root.decision.leases
        )
        self.assertGreaterEqual(dependent.decision.start_us, root_end_us)
        self.assertTrue(controller.require_queued_replan(
            root.request.request_id, "residency_observation_changed",
        ))
        # Woken late, the root's hot slot shares the GPU with the dependent's
        # queued load, which cannot dispatch before the root completes.
        late_us = root.decision.start_us + (
            root_end_us - root.decision.start_us
        ) // 2
        with mock.patch.object(
            time, "monotonic_ns", return_value=late_us * 1_000
        ):
            wake = scheduler.wait_runtime_request(root.request.request_id, 0)
        self.assertEqual(wake.dispatch_state, "REPLAN_REQUIRED")
        self.assertEqual(wake.dispatch_receipt.observed_at_us, late_us)
        return scheduler, hot, root, dependent, wake

    def _replan_woken(self, scheduler, hot, wake, *, compaction=False):
        arguments = {
            "observed_at_us": wake.dispatch_receipt.observed_at_us,
            "reason": wake.dispatch_receipt.wake_reason,
            "snapshot": hot,
            "expected_ticket_id": wake.ticket_id,
            "expected_queue_generation": (
                wake.dispatch_receipt.queue_generation
            ),
        }
        if compaction:
            return scheduler._replan_automated_request_once(
                wake.request.request_id,
                priority_compaction_active=True,
                **arguments,
            )
        return scheduler.replan_automated_request(
            wake.request.request_id, **arguments
        )

    def test_replan_defers_dependent_replacement_inside_baseline_window(
        self,
    ) -> None:
        scheduler, hot, root, dependent, wake = (
            self._wake_root_into_dependent_replacement("window")
        )

        replanned = self._replan_woken(scheduler, hot, wake)

        self.assertEqual(replanned.dispatch_state, "QUEUED")
        self.assertEqual(replanned.execution_plan.transitions, ())
        self.assertEqual(
            replanned.decision.start_us,
            wake.dispatch_receipt.observed_at_us,
        )
        dependent_id = dependent.request.request_id
        self.assertEqual(
            scheduler.runtime_ticket(dependent_id).lease_status, "CANCELLED"
        )
        self.assertIn(
            scheduler.runtime_controller_snapshot()["dispatch_queue"][
                "entry_states"
            ][dependent_id]["state"],
            {"DEFERRED_REPLAN", "REPLAN_REQUIRED"},
        )
        self.assertEqual(scheduler._runtime_memory.owner_tokens(dependent_id), ())
        with mock.patch.object(
            time,
            "monotonic_ns",
            return_value=wake.dispatch_receipt.observed_at_us * 1_000,
        ):
            dependent_wake = scheduler.wait_runtime_request(dependent_id, 0)
        self.assertEqual(dependent_wake.dispatch_state, "REPLAN_REQUIRED")
        requeued = self._replan_woken(scheduler, hot, dependent_wake)
        self.assertTrue(requeued.execution_plan.transitions)
        self.assertGreaterEqual(
            requeued.decision.start_us,
            max(row.reserved_until_us for row in replanned.decision.leases),
        )

    def test_replan_dependent_deferral_decision_log_is_deterministic(
        self,
    ) -> None:
        heads = []
        for _ in range(2):
            scheduler, hot, _, _, wake = (
                self._wake_root_into_dependent_replacement("repeat")
            )
            self._replan_woken(scheduler, hot, wake)
            heads.append(
                scheduler.runtime_decision_log()["head_record_sha256"]
            )
        self.assertEqual(heads[0], heads[1])

    def test_compaction_replan_keeps_dependents_and_fails_closed(
        self,
    ) -> None:
        scheduler, hot, root, dependent, wake = (
            self._wake_root_into_dependent_replacement("compaction")
        )
        dependent_tokens = scheduler._runtime_memory.owner_tokens(
            dependent.request.request_id
        )
        self.assertTrue(dependent_tokens)

        # Compaction followers are replanned in their own order, so the
        # baseline window cannot defer them and selection must still fail.
        with self.assertRaisesRegex(
            UnifiedScheduleError,
            "qualified desktop baseline is not available: "
            "MEMORY_REPLACEMENT_CONFLICT_CURRENT:gpu-memory",
        ):
            self._replan_woken(scheduler, hot, wake, compaction=True)

        failed = scheduler.runtime_ticket(root.request.request_id)
        self.assertEqual(failed.dispatch_state, "FAILED")
        self.assertIn(
            "MEMORY_REPLACEMENT_CONFLICT_CURRENT", failed.failure_reason
        )

    def test_replan_reprojects_baseline_behind_unprojected_replacement(
        self,
    ) -> None:
        scheduler, other, target, hot = (
            self._two_lane_gpu_replacement_scheduler("reproject")
        )
        first = scheduler.submit_automated_request(
            request("reproject-first", arrival_us=1_000, output_tokens=40),
            target.model_id,
            hot,
            selection_mode="energy-aware",
        )
        load = scheduler.submit_automated_request(
            request("reproject-load", arrival_us=1_050),
            other.model_id,
            hot,
            selection_mode="energy-aware",
        )
        self.assertTrue(load.execution_plan.transitions)
        waiting = scheduler.submit_automated_request(
            request("reproject-waiting", arrival_us=1_100, output_tokens=40),
            target.model_id,
            hot,
            selection_mode="energy-aware",
        )
        controller = scheduler._runtime_controller
        self.assertTrue(controller.require_queued_replan(
            waiting.request.request_id, "residency_observation_changed",
        ))
        wake = scheduler.wait_runtime_request(
            waiting.request.request_id,
            time.monotonic_ns() - waiting.request.arrival_us * 1_000,
        )
        self.assertEqual(wake.dispatch_state, "REPLAN_REQUIRED")

        # Without its predecessors in the planning snapshot the replan sees
        # the target still hot, so its baseline overlaps the queued load.
        with mock.patch.object(
            controller, "replan_projection_request_ids", return_value=(),
        ):
            replanned = scheduler.replan_automated_request(
                waiting.request.request_id,
                observed_at_us=wake.dispatch_receipt.observed_at_us,
                reason=wake.dispatch_receipt.wake_reason,
                snapshot=hot,
                expected_ticket_id=wake.ticket_id,
                expected_queue_generation=(
                    wake.dispatch_receipt.queue_generation
                ),
            )

        self.assertEqual(replanned.dispatch_state, "QUEUED")
        self.assertEqual(
            {
                (row.model_id, row.device_id)
                for transition in replanned.execution_plan.transitions
                for row in transition.evictions
            },
            {(other.model_id, "accelerator-b")},
        )
        self.assertGreaterEqual(
            replanned.decision.start_us,
            max(row.reserved_until_us for row in load.decision.leases),
        )
        self.assertEqual(
            scheduler.runtime_ticket(first.request.request_id).lease_status,
            "RESERVED",
        )

    def test_replan_behind_later_arrival_evicts_projected_resident(
        self,
    ) -> None:
        source = catalog()
        gpu = replace(
            source.executor_by_device["accelerator-b"],
            qualified_fallback=True,
            exclusive_residency_resource_id="compute:accelerator-b",
        )
        transition = replace(
            next(
                row for row in source.transitions
                if row.device_id == "accelerator-b"
            ),
            executor_id=gpu.executor_id,
            prepares_device_ids=("accelerator-b",),
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=(gpu,),
            composite_executors=(),
            transitions=(transition,),
        ).to_json())
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        other_path = Path(self.directory.name) / "replan-behind-other.gguf"
        write_synthetic_gguf(other_path, block_count=2, sliding_window=32)
        other = scheduler.register_gguf_model(
            "replan-behind-other", other_path
        )
        target = scheduler.register_gguf_model(
            "replan-behind-target", self.path
        )
        manifests = {other.model_id: other, target.model_id: target}
        cold = replace(
            runtime_snapshot(
                target, include_phone=False, resident_devices=()
            ),
            executors={
                gpu.executor_id: executor_state(gpu.executor_id)
            },
        )
        stale = scheduler.submit_automated_request(
            request("replan-behind-stale", arrival_us=1_000),
            target.model_id,
            cold,
            selection_mode="desktop-baseline",
        )
        self.assertTrue(stale.execution_plan.transitions)
        self.assertTrue(
            scheduler._runtime_controller.require_queued_replan(
                stale.request.request_id,
                "residency_observation_changed",
            )
        )
        wake = scheduler.wait_runtime_request(
            stale.request.request_id,
            time.monotonic_ns() - stale.decision.start_us * 1_000,
        )
        self.assertEqual(wake.dispatch_state, "REPLAN_REQUIRED")
        later = scheduler.submit_automated_request(
            request("replan-behind-later", arrival_us=1_100),
            other.model_id,
            cold,
            selection_mode="desktop-baseline",
        )
        self.assertTrue(later.execution_plan.transitions)

        replanned = scheduler.replan_automated_request(
            stale.request.request_id,
            observed_at_us=wake.dispatch_receipt.observed_at_us,
            reason=wake.dispatch_receipt.wake_reason,
            snapshot=cold,
            expected_ticket_id=wake.ticket_id,
            expected_queue_generation=(
                wake.dispatch_receipt.queue_generation
            ),
        )

        controller = scheduler._runtime_controller
        self.assertLess(
            replanned.decision.start_us, later.decision.start_us
        )
        self.assertEqual(
            tuple(
                row
                for transition in replanned.execution_plan.transitions
                for row in transition.evictions
            ),
            (),
        )
        self.assertEqual(
            scheduler.runtime_controller_snapshot()["dispatch_queue"]
                ["entry_states"][later.request.request_id]["state"],
            "DEFERRED_REPLAN",
        )
        # Causal edges, not lease start, order projected transitions: the
        # stale-by-construction pair blames whichever attempt runs second.
        with self.assertRaises(RuntimeResidencyProjectionError) as raised:
            project_scheduler_residency(
                cold, profile, (replanned, later), manifests,
            )
        self.assertEqual(
            raised.exception.request_id, later.request.request_id
        )
        with self.assertRaises(RuntimeResidencyProjectionError) as raised:
            project_scheduler_residency(
                cold, profile, (replanned, later), manifests,
                causal_predecessors={
                    stale.request.request_id: (later.request.request_id,),
                },
            )
        self.assertEqual(
            str(raised.exception),
            "projected transition lacks current exclusive eviction: "
            "accelerator-b",
        )
        self.assertEqual(
            raised.exception.request_id, stale.request.request_id
        )

        arrival = scheduler.submit_automated_request(
            request("replan-behind-arrival", arrival_us=1_200),
            other.model_id,
            cold,
            selection_mode="desktop-baseline",
        )
        self.assertEqual(arrival.dispatch_state, "QUEUED")
        self.assertEqual(
            scheduler.runtime_ticket(stale.request.request_id).ticket_id,
            replanned.ticket_id,
        )
        self.assertIn(
            stale.request.request_id,
            controller.projection_causal_predecessors()[
                arrival.request.request_id
            ],
        )

    def test_projection_rejects_missing_current_resident_eviction(
        self,
    ) -> None:
        source = catalog()
        gpu = replace(
            source.executor_by_device["accelerator-b"],
            qualified_fallback=True,
            exclusive_residency_resource_id="compute:accelerator-b",
        )
        transition = replace(
            next(
                row for row in source.transitions
                if row.device_id == "accelerator-b"
            ),
            executor_id=gpu.executor_id,
            prepares_device_ids=("accelerator-b",),
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=(gpu,),
            composite_executors=(),
            transitions=(transition,),
        ).to_json())
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        resident_path = Path(self.directory.name) / "projection-resident.gguf"
        write_synthetic_gguf(
            resident_path, block_count=2, sliding_window=32
        )
        resident = scheduler.register_gguf_model(
            "projection-resident", resident_path
        )
        target = scheduler.register_gguf_model(
            "projection-target", self.path
        )
        cold = replace(
            runtime_snapshot(
                target,
                include_phone=False,
                resident_devices=(),
            ),
            executors={
                gpu.executor_id: executor_state(gpu.executor_id)
            },
        )
        ticket = scheduler.submit_automated_request(
            request("projection-missing-eviction"),
            target.model_id,
            cold,
            selection_mode="desktop-baseline",
        )
        self.assertTrue(ticket.execution_plan.transitions)
        self.assertEqual(
            ticket.execution_plan.transitions[0].evictions, ()
        )
        capacity = cold.memory.capacities["gpu-memory"]
        hot = replace(
            cold,
            snapshot_id="projection-unexpected-resident",
            memory=replace(
                cold.memory,
                snapshot_id="projection-unexpected-resident-memory",
                capacities={
                    "gpu-memory": replace(
                        capacity,
                        occupied_bytes=resident.tensor_bytes,
                    )
                },
            ),
            residency=(ModelResidencyObservation(
                model_id=resident.model_id,
                artifact_sha256=resident.artifact_sha256,
                device_id="accelerator-b",
                state="hot",
                resident_tensor_ids=tuple(
                    row.tensor_id for row in resident.tensors
                ),
                resident_bytes=resident.tensor_bytes,
                generation=7,
                executor_id=gpu.executor_id,
                reclaimable_bytes=resident.tensor_bytes,
            ),),
        )

        with self.assertRaises(RuntimeResidencyProjectionError) as raised:
            project_scheduler_residency(
                hot,
                profile,
                (ticket,),
                {
                    resident.model_id: resident,
                    target.model_id: target,
                },
            )

        self.assertEqual(
            str(raised.exception),
            "projected transition lacks current exclusive eviction: "
            "accelerator-b",
        )
        self.assertEqual(
            raised.exception.request_id, ticket.request.request_id
        )
        self.assertEqual(raised.exception.ticket_id, ticket.ticket_id)

    def test_residency_cohort_hysteresis_requires_observed_reuse(self) -> None:
        source = catalog()
        gpu = replace(
            source.executor_by_device["accelerator-b"],
            exclusive_residency_resource_id="compute:accelerator-b",
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=tuple(
                gpu if row.device_id == gpu.device_id else row
                for row in source.executors
            ),
        ).to_json())
        _, manifest = self.scheduler_and_manifest(profile)
        tracker = RuntimeResidencyCohortTracker()
        hot = runtime_snapshot(
            manifest, resident_devices=("accelerator-b",)
        )
        component = RuntimeResidencyComponentIdentity(
            artifact_sha256=manifest.artifact_sha256,
            resident_shard_geometry_sha256=(),
            desktop_placement_sha256=None,
            transport_generation=None,
            operator_protocol="direct-executor-v1",
            session_resource_ids=(),
            executor_id=gpu.executor_id,
        )
        tracker.record_arrival("cohort-1", component, 1_000)
        no_history = tracker.holds(profile, hot, (), 1_100)
        self.assertEqual(no_history, {})
        tracker.record_arrival("cohort-2", component, 2_000)
        tracker.record_arrival("cohort-3", component, 3_000)
        predicted = tracker.holds(profile, hot, (), 3_500)
        hold = predicted["compute:accelerator-b"]

        self.assertEqual(hold.predicted_reuse_us, 4_000)
        self.assertGreater(hold.replacement_cost_us, 0)
        self.assertGreater(hold.hold_until_us, hold.predicted_reuse_us)

    def test_residency_reuse_projection_includes_current_arrival(self) -> None:
        source = catalog()
        _, manifest = self.scheduler_and_manifest(source)
        tracker = RuntimeResidencyCohortTracker()
        component = RuntimeResidencyComponentIdentity(
            artifact_sha256=manifest.artifact_sha256,
            resident_shard_geometry_sha256=(),
            desktop_placement_sha256=None,
            transport_generation=None,
            operator_protocol="direct-executor-v1",
            session_resource_ids=(),
            executor_id=(
                source.executor_by_device[
                    "accelerator-b"
                ].executor_id
            ),
        )
        tracker.record_arrival(
            "cohort-existing", component, 1_000
        )

        projection = tracker.reuse_projections(
            manifest.artifact_sha256,
            2_000,
            request_id="cohort-current",
        )[component.identity_sha256]

        self.assertEqual(projection.observed_request_count, 2)
        self.assertEqual(projection.expected_use_count, 2)
        self.assertEqual(tracker.snapshot()["active_request_count"], 1)
        existing = tracker.reuse_projections(
            manifest.artifact_sha256,
            2_000,
            request_id="cohort-existing",
        )[component.identity_sha256]
        self.assertEqual(existing.observed_request_count, 1)

    def test_residency_reuse_excludes_incompatible_components(self) -> None:
        source = catalog()
        _, manifest = self.scheduler_and_manifest(source)
        executor_id = source.executor_by_device[
            "accelerator-b"
        ].executor_id

        def component(transport: str) -> RuntimeResidencyComponentIdentity:
            return RuntimeResidencyComponentIdentity(
                artifact_sha256=manifest.artifact_sha256,
                resident_shard_geometry_sha256=(
                    canonical_sha256("synthetic-shard"),
                ),
                desktop_placement_sha256=canonical_sha256(
                    "synthetic-desktop-placement"
                ),
                transport_generation=transport,
                operator_protocol="synthetic-operator-v1",
                session_resource_ids=("memory:synthetic-session",),
                executor_id=executor_id,
            )

        first = component("transport-a")
        incompatible = component("transport-b")
        tracker = RuntimeResidencyCohortTracker()
        tracker.record_arrival("component-a-1", first, 1_000)
        tracker.record_arrival("component-b-1", incompatible, 2_000)
        tracker.record_arrival("component-a-2", first, 3_000)

        projections = tracker.reuse_projections(
            manifest.artifact_sha256,
            4_000,
            request_id="component-current",
        )
        self.assertEqual(
            projections[first.identity_sha256].observed_request_count, 3
        )
        self.assertEqual(
            projections[
                incompatible.identity_sha256
            ].observed_request_count,
            2,
        )
        with self.assertRaises(
            RuntimeResidencyCohortError,
            msg="one request cannot change compatible residency identity",
        ):
            tracker.record_arrival("component-a-1", incompatible, 5_000)

    def test_residency_component_uses_stable_endpoint_across_route_variants(
        self,
    ) -> None:
        _, manifest = self.scheduler_and_manifest(catalog())
        common = {
            "artifact_sha256": manifest.artifact_sha256,
            "resident_shard_geometry_sha256": (
                canonical_sha256("stable-endpoint-shard"),
            ),
            "desktop_placement_sha256": canonical_sha256(
                "stable-endpoint-desktop"
            ),
            "transport_generation": "stable-endpoint-transport-v1",
            "operator_protocol": "stable-endpoint-operator-v1",
            "session_resource_ids": ("memory:stable-endpoint-session",),
            "resident_endpoint": "synthetic://stable-endpoint",
        }
        split_row = RuntimeResidencyComponentIdentity(
            **common,
            executor_id="synthetic:split-row",
        )
        coalesced = RuntimeResidencyComponentIdentity(
            **common,
            executor_id="synthetic:coalesced-batch",
        )
        other_endpoint = RuntimeResidencyComponentIdentity(
            **{
                **common,
                "resident_endpoint": "synthetic://other-endpoint",
            },
            executor_id="synthetic:coalesced-batch",
        )

        self.assertEqual(
            split_row.identity_sha256, coalesced.identity_sha256
        )
        self.assertEqual(split_row, coalesced)
        self.assertNotEqual(
            split_row.identity_sha256, other_endpoint.identity_sha256
        )

    def test_transition_overrun_replans_followers_once_in_arrival_order(
        self,
    ) -> None:
        source = catalog()
        gpu = replace(
            source.executor_by_device["accelerator-b"],
            qualified_fallback=True,
            exclusive_residency_resource_id="compute:accelerator-b",
        )
        transition = replace(
            next(
                row for row in source.transitions
                if row.device_id == "accelerator-b"
            ),
            executor_id=gpu.executor_id,
            prepares_device_ids=("accelerator-b",),
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=(gpu,),
            composite_executors=(),
            transitions=(transition,),
        ).to_json())
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        other_path = Path(self.directory.name) / "overrun-other.gguf"
        write_synthetic_gguf(other_path, block_count=2, sliding_window=32)
        models = (
            scheduler.register_gguf_model("overrun-first", self.path),
            scheduler.register_gguf_model("overrun-other", other_path),
        )
        cold = replace(
            runtime_snapshot(
                models[0], include_phone=False, resident_devices=()
            ),
            executors={
                gpu.executor_id: executor_state(gpu.executor_id)
            },
        )
        tickets = tuple(
            scheduler.submit_automated_request(
                request("overrun-" + str(index), arrival_us=1_000 + index),
                models[index % len(models)].model_id,
                cold,
                selection_mode="desktop-baseline",
            )
            for index in range(8)
        )
        first = scheduler.wait_runtime_request(
            tickets[0].request.request_id,
            time.monotonic_ns() - tickets[0].decision.start_us * 1_000,
        )
        scheduler.record_automated_transition_receipts(
            first.request.request_id,
            FakeAutomatedPhysicalAdapter._transition_receipts(first),
        )
        extended_until_us = max(
            row.decision.finish_upper_us for row in tickets
        ) + 1_000
        extension = scheduler.extend_runtime_request(
            first.request.request_id,
            at_us=first.decision.start_us + 1,
            reserved_until_us=extended_until_us,
        )
        self.assertEqual(
            set(extension.cancelled_queued_tokens),
            {row.request.request_id for row in tickets[1:]},
        )
        observed_at_us = max(
            tickets[-1].request.arrival_us,
            first.decision.start_us + 2,
        )
        resident_bytes = sum(
            demand.required_bytes
            for demand in first.execution_plan.memory_demands
            if demand.resource_id == "gpu-memory"
        )
        hot = replace(
            cold,
            snapshot_id="synthetic-overrun-hot",
            captured_at_us=observed_at_us,
            valid_until_us=extended_until_us + 1_000_000,
            memory=replace(
                cold.memory,
                snapshot_id="synthetic-overrun-hot-memory",
                captured_at_us=observed_at_us,
                valid_until_us=extended_until_us + 1_000_000,
            ),
            residency=(ModelResidencyObservation(
                model_id=models[0].model_id,
                artifact_sha256=models[0].artifact_sha256,
                device_id="accelerator-b",
                state="hot",
                resident_tensor_ids=tuple(
                    row.tensor_id for row in models[0].tensors
                ),
                resident_bytes=models[0].tensor_bytes,
                generation=1,
                executor_id=gpu.executor_id,
                reclaimable_bytes=resident_bytes,
            ),),
        )
        execution = scheduler.runtime_execution_ticket(
            first.request.request_id
        )
        scheduler.complete_automated_request(
            first.request.request_id,
            self.execution_receipt(
                execution, finished_us=observed_at_us
            ),
        )
        queue_snapshot = scheduler.runtime_controller_snapshot()[
            "dispatch_queue"
        ]
        self.assertEqual(
            scheduler._runtime_controller.replan_required_requests(),
            (tickets[1].request.request_id,),
            repr(queue_snapshot),
        )

        wake_by_request = {}
        wake_errors = {}
        wake_events = {
            ticket.request.request_id: threading.Event()
            for ticket in tickets[1:]
        }

        def wait_for_replan(ticket):
            request_id = ticket.request.request_id
            try:
                wake_by_request[request_id] = (
                    scheduler.wait_runtime_request(
                        request_id,
                        time.monotonic_ns() - extended_until_us * 1_000,
                    )
                )
            except BaseException as exc:
                wake_errors[request_id] = exc
            finally:
                wake_events[request_id].set()

        waiters = []
        for ticket in reversed(tickets[1:]):
            waiter = threading.Thread(
                target=wait_for_replan,
                args=(ticket,),
                daemon=True,
            )
            waiter.start()
            waiters.append(waiter)

        replanned = []
        for index, ticket in enumerate(tickets[1:], start=1):
            request_id = ticket.request.request_id
            self.assertTrue(
                wake_events[request_id].wait(1),
                repr(scheduler.runtime_controller_snapshot()),
            )
            self.assertNotIn(request_id, wake_errors)
            wake = wake_by_request[request_id]
            self.assertEqual(wake.dispatch_state, "REPLAN_REQUIRED")
            self.assertFalse(any(
                wake_events[later.request.request_id].is_set()
                for later in tickets[index + 1:]
            ))
            current = scheduler.replan_automated_request(
                request_id,
                observed_at_us=observed_at_us,
                reason=wake.dispatch_receipt.wake_reason,
                snapshot=hot,
            )
            replanned.append(current)
            active = scheduler.wait_runtime_request(
                request_id,
                time.monotonic_ns() - current.decision.start_us * 1_000,
            )
            if active.transition_status == "PENDING":
                scheduler.record_automated_transition_receipts(
                    request_id,
                    FakeAutomatedPhysicalAdapter._transition_receipts(
                        active
                    ),
                )
            execution = scheduler.runtime_execution_ticket(request_id)
            scheduler.complete_automated_request(
                request_id,
                self.execution_receipt(
                    execution,
                    finished_us=active.decision.start_us + 1,
                ),
            )
        for waiter in waiters:
            waiter.join(1)
            self.assertFalse(waiter.is_alive())
        replanned = tuple(replanned)

        self.assertEqual(
            {row.request.request_id for row in replanned},
            {row.request.request_id for row in tickets[1:]},
        )
        self.assertTrue(all(row.attempt_index == 1 for row in replanned))
        self.assertTrue(all(
            scheduler.runtime_ticket(row.request.request_id).dispatch_state
                == "COMPLETED"
            for row in replanned
        ))

    def test_hot_exclusive_residency_route_is_a_causal_barrier(
        self,
    ) -> None:
        source = catalog()
        gpu = replace(
            source.executor_by_device["accelerator-b"],
            qualified_fallback=True,
            exclusive_residency_resource_id="compute:accelerator-b",
        )
        transition = replace(
            next(
                row for row in source.transitions
                if row.device_id == "accelerator-b"
            ),
            executor_id=gpu.executor_id,
            prepares_device_ids=("accelerator-b",),
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=(gpu,),
            composite_executors=(),
            transitions=(transition,),
        ).to_json())
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        manifest = scheduler.register_gguf_model(
            "hot-exclusive-model", self.path
        )
        snapshot = replace(
            runtime_snapshot(
                manifest,
                include_phone=False,
                resident_devices=("accelerator-b",),
            ),
            executors={
                gpu.executor_id: executor_state(gpu.executor_id)
            },
        )
        ticket = scheduler.submit_automated_request(
            request("hot-exclusive-request", arrival_us=1_000),
            manifest.model_id,
            snapshot,
            selection_mode="desktop-baseline",
        )

        self.assertFalse(ticket.execution_plan.transitions)
        self.assertTrue(
            scheduler.runtime_controller_snapshot()["dispatch_queue"]
            ["entry_states"][ticket.request.request_id]
            ["residency_transition_barrier"]
        )

    def test_hot_completion_replans_only_first_and_defers_projection_chain(
        self,
    ) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        hot = runtime_snapshot(
            manifest,
            include_phone=False,
            resident_devices=("host-a", "accelerator-b"),
        )
        first = scheduler.submit_automated_request(
            request("hot-queue-first", arrival_us=1_000),
            manifest.model_id,
            hot,
            selection_mode="desktop-baseline",
        )
        second = scheduler.submit_automated_request(
            request("hot-queue-second", arrival_us=1_100),
            manifest.model_id,
            hot,
            selection_mode="desktop-baseline",
        )
        third = scheduler.submit_automated_request(
            request("hot-queue-third", arrival_us=1_200),
            manifest.model_id,
            hot,
            selection_mode="desktop-baseline",
        )
        self.assertEqual(first.execution_plan.transitions, ())
        self.assertEqual(second.execution_plan.transitions, ())
        self.assertEqual(third.execution_plan.transitions, ())

        active = scheduler.wait_runtime_request(
            first.request.request_id,
            time.monotonic_ns() - first.decision.start_us * 1_000,
        )
        completed_at_us = max(
            active.decision.start_us + 1,
            second.request.arrival_us,
        )
        scheduler.complete_automated_request(
            first.request.request_id,
            self.execution_receipt(active, finished_us=completed_at_us),
        )
        wake = scheduler.wait_runtime_request(
            second.request.request_id,
            time.monotonic_ns() - completed_at_us * 1_000,
        )
        self.assertEqual(wake.dispatch_state, "REPLAN_REQUIRED")
        self.assertEqual(
            wake.dispatch_receipt.wake_reason,
            "capacity_released_early",
        )
        replanned = scheduler.replan_automated_request(
            second.request.request_id,
            observed_at_us=completed_at_us,
            reason=wake.dispatch_receipt.wake_reason,
            snapshot=replace(
                hot,
                snapshot_id="hot-queue-replan-runtime",
                captured_at_us=completed_at_us,
                valid_until_us=completed_at_us + 10_000_000,
                memory=replace(
                    hot.memory,
                    snapshot_id="hot-queue-replan-memory",
                    captured_at_us=completed_at_us,
                    valid_until_us=completed_at_us + 10_000_000,
                ),
            ),
            expected_ticket_id=wake.ticket_id,
            expected_queue_generation=(
                wake.dispatch_receipt.queue_generation
            ),
        )
        wake = scheduler.wait_runtime_request(
            second.request.request_id,
            time.monotonic_ns()
            - replanned.decision.start_us * 1_000,
        )
        self.assertEqual(wake.dispatch_state, "ACQUIRED")
        self.assertNotEqual(wake.ticket_id, second.ticket_id)
        self.assertEqual(wake.binding.executor_id, second.binding.executor_id)
        self.assertEqual(wake.decision.start_us, completed_at_us)
        self.assertTrue(any(
            row["event_kind"] == "ACQUIRED"
            and row["request_ids"] == [second.request.request_id]
            for row in scheduler.runtime_decision_log()["records"]
        ))

        deferred = scheduler.runtime_ticket(third.request.request_id)
        self.assertEqual(deferred.dispatch_state, "QUEUED")
        self.assertEqual(deferred.lease_status, "CANCELLED")
        self.assertEqual(deferred.ticket_id, third.ticket_id)
        self.assertEqual(deferred.attempt_index, third.attempt_index)
        self.assertGreater(deferred.decision.start_us, completed_at_us)
        self.assertEqual(
            scheduler.runtime_controller_snapshot()["dispatch_queue"][
                "entry_states"
            ][third.request.request_id]["state"],
            "DEFERRED_REPLAN",
        )
        second_completed_at_us = max(
            wake.decision.start_us + 50, third.request.arrival_us
        )
        scheduler.complete_automated_request(
            second.request.request_id,
            self.execution_receipt(
                wake, finished_us=second_completed_at_us
            ),
        )
        third_wake = scheduler.wait_runtime_request(
            third.request.request_id,
            time.monotonic_ns() - second_completed_at_us * 1_000,
        )
        self.assertEqual(third_wake.dispatch_state, "REPLAN_REQUIRED")
        third_replanned = scheduler.replan_automated_request(
            third.request.request_id,
            observed_at_us=second_completed_at_us,
            reason=third_wake.dispatch_receipt.wake_reason,
            snapshot=replace(
                hot,
                snapshot_id="hot-queue-third-runtime",
                captured_at_us=second_completed_at_us,
                valid_until_us=second_completed_at_us + 10_000_000,
                memory=replace(
                    hot.memory,
                    snapshot_id="hot-queue-third-memory",
                    captured_at_us=second_completed_at_us,
                    valid_until_us=(
                        second_completed_at_us + 10_000_000
                    ),
                ),
            ),
            expected_ticket_id=third_wake.ticket_id,
            expected_queue_generation=(
                third_wake.dispatch_receipt.queue_generation
            ),
        )
        self.assertEqual(
            third_replanned.decision.start_us, second_completed_at_us
        )

    def test_cold_replan_fills_newly_available_hot_execution_lane(
        self,
    ) -> None:
        manifest = GGUFModelManifestLoader.load(
            "unseen-model", self.path
        )
        source = catalog_with_gpu_desktop_control(manifest)
        resources = dict(source.resources)
        for resource_id in (
            "compute:accelerator-b", "link:pcie-in", "link:pcie-out"
        ):
            resources[resource_id] = replace(
                resources[resource_id], capacity=2
            )
        transitions = tuple(
            replace(
                row,
                resource_slots={"compute:accelerator-b": 2},
            )
            if row.device_id == "accelerator-b" else row
            for row in source.transitions
        )
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(replace(
            source,
            resources=resources,
            transitions=transitions,
        ))
        scheduler.register_model_manifest(manifest)
        cold = runtime_snapshot(
            manifest,
            include_phone=False,
            gpu_free_slots=2,
            resident_devices=(),
        )
        tickets = tuple(
            scheduler.submit_automated_request(
                request(
                    "cold-hot-lane-" + str(index),
                    arrival_us=1_000 + index,
                ),
                manifest.model_id,
                cold,
                selection_mode="desktop-baseline",
            )
            for index in range(3)
        )
        self.assertTrue(all(
            row.execution_plan.transitions for row in tickets
        ))

        first = scheduler.wait_runtime_request(
            tickets[0].request.request_id,
            time.monotonic_ns() - tickets[0].decision.start_us * 1_000,
        )
        scheduler.record_automated_transition_receipts(
            first.request.request_id,
            tuple(
                replace(receipt, resource_slots=transition.resource_slots)
                for receipt, transition in zip(
                    FakeAutomatedPhysicalAdapter._transition_receipts(first),
                    first.execution_plan.transitions,
                )
            ),
        )
        completed_at_us = max(
            tickets[1].decision.start_us + 5_000,
            tickets[-1].request.arrival_us,
        )
        extension = scheduler.extend_runtime_request(
            first.request.request_id,
            at_us=first.decision.start_us + 1,
            reserved_until_us=completed_at_us + 5_000,
        )
        self.assertEqual(
            set(extension.cancelled_queued_tokens),
            {row.request.request_id for row in tickets[1:]},
        )
        scheduler.complete_automated_request(
            first.request.request_id,
            self.execution_receipt(first, finished_us=completed_at_us),
        )
        second_wake = scheduler.wait_runtime_request(
            tickets[1].request.request_id,
            time.monotonic_ns() - completed_at_us * 1_000,
        )
        self.assertEqual(second_wake.dispatch_state, "REPLAN_REQUIRED")
        self.assertEqual(
            second_wake.dispatch_receipt.wake_reason,
            "capacity_released_early",
        )
        hot = runtime_snapshot(
            manifest,
            include_phone=False,
            gpu_free_slots=2,
            resident_devices=("accelerator-b",),
        )
        hot = replace(
            hot,
            snapshot_id="cold-hot-lane-runtime",
            captured_at_us=completed_at_us,
            valid_until_us=completed_at_us + 10_000_000,
            memory=replace(
                hot.memory,
                snapshot_id="cold-hot-lane-memory",
                captured_at_us=completed_at_us,
                valid_until_us=completed_at_us + 10_000_000,
            ),
        )
        second = scheduler.replan_automated_request(
            tickets[1].request.request_id,
            observed_at_us=completed_at_us,
            reason=second_wake.dispatch_receipt.wake_reason,
            snapshot=hot,
            expected_ticket_id=second_wake.ticket_id,
            expected_queue_generation=(
                second_wake.dispatch_receipt.queue_generation
            ),
        )
        third_wake = scheduler.wait_runtime_request(
            tickets[2].request.request_id,
            time.monotonic_ns() - completed_at_us * 1_000,
        )
        self.assertEqual(third_wake.dispatch_state, "ACQUIRED")
        third = third_wake
        self.assertEqual(third.attempt_index, tickets[2].attempt_index + 1)
        self.assertEqual(second.decision.start_us, completed_at_us)
        self.assertEqual(third.decision.start_us, completed_at_us)

        second_active = scheduler.wait_runtime_request(
            second.request.request_id,
            time.monotonic_ns() - completed_at_us * 1_000,
        )
        third_active = third
        self.assertEqual(second_active.dispatch_state, "ACQUIRED")
        self.assertEqual(third_active.dispatch_state, "ACQUIRED")
        self.assertEqual(
            len(scheduler.runtime_controller_snapshot()[
                "dispatch_queue"
            ]["active"]),
            2,
        )
        follower_history = tuple(
            row
            for row in scheduler._runtime_controller.checkpoint()[1]
            if row.request.request_id == tickets[2].request.request_id
        )
        self.assertEqual(
            tuple(row.ticket_id for row in follower_history),
            (tickets[2].ticket_id, third.ticket_id),
        )
        follower_replans = tuple(
            row
            for row in scheduler.runtime_decision_log()["records"]
            if row["event_kind"] == "REPLAN"
            and row["request_ids"] == [tickets[2].request.request_id]
        )
        self.assertEqual(len(follower_replans), 1)
        live_memory_tokens = tuple(
            row["token"]
            for row in scheduler.runtime_memory_state()["reservations"]
        )
        self.assertEqual(
            len(live_memory_tokens), len(set(live_memory_tokens))
        )
        live_lease_tokens = tuple(
            lease.token
            for row in (second_active, third_active)
            for lease in row.decision.leases
        )
        self.assertEqual(
            len(live_lease_tokens), len(set(live_lease_tokens))
        )

    def test_priority_compaction_memory_mismatch_rolls_back_atomically(
        self,
    ) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        hot = runtime_snapshot(
            manifest,
            include_phone=False,
            resident_devices=("host-a", "accelerator-b"),
        )
        tickets = tuple(
            scheduler.submit_automated_request(
                request(
                    "compaction-memory-mismatch-" + str(index),
                    arrival_us=1_000 + index,
                ),
                manifest.model_id,
                hot,
                selection_mode="desktop-baseline",
            )
            for index in range(3)
        )
        active = scheduler.wait_runtime_request(
            tickets[0].request.request_id,
            time.monotonic_ns() - tickets[0].decision.start_us * 1_000,
        )
        completed_at_us = max(
            active.decision.start_us + 1,
            tickets[-1].request.arrival_us,
        )
        scheduler.complete_automated_request(
            active.request.request_id,
            self.execution_receipt(active, finished_us=completed_at_us),
        )
        root = scheduler.wait_runtime_request(
            tickets[1].request.request_id,
            time.monotonic_ns() - completed_at_us * 1_000,
        )
        fresh = replace(
            hot,
            snapshot_id="compaction-memory-mismatch-runtime",
            captured_at_us=completed_at_us,
            valid_until_us=completed_at_us + 10_000_000,
            memory=replace(
                hot.memory,
                snapshot_id="compaction-memory-mismatch-memory",
                captured_at_us=completed_at_us,
                valid_until_us=completed_at_us + 10_000_000,
            ),
        )
        before_controller = scheduler.runtime_controller_snapshot()
        before_memory = scheduler.runtime_memory_state()
        before_log = scheduler.runtime_decision_log_bytes()
        before_timeline = scheduler.timeline.checkpoint()
        follower_id = tickets[2].request.request_id
        original_owner_tokens = scheduler._runtime_memory.owner_tokens

        def mismatched_owner_tokens(owner_id):
            if owner_id == follower_id:
                return ("unexpected-memory-token",)
            return original_owner_tokens(owner_id)

        with mock.patch.object(
            scheduler._runtime_controller.queue,
            "priority_compaction_followers",
            return_value=(follower_id,),
        ), mock.patch.object(
            scheduler._runtime_memory,
            "owner_tokens",
            side_effect=mismatched_owner_tokens,
        ):
            with self.assertRaisesRegex(
                RuntimeReplanRetryRequired,
                "PRIORITY_COMPACTION_ROLLED_BACK",
            ):
                scheduler.replan_automated_request(
                    root.request.request_id,
                    observed_at_us=completed_at_us,
                    reason=root.dispatch_receipt.wake_reason,
                    snapshot=fresh,
                    expected_ticket_id=root.ticket_id,
                    expected_queue_generation=(
                        root.dispatch_receipt.queue_generation
                    ),
                )

        self.assertEqual(
            scheduler.runtime_controller_snapshot(), before_controller
        )
        self.assertEqual(scheduler.runtime_memory_state(), before_memory)
        self.assertEqual(scheduler.runtime_decision_log_bytes(), before_log)
        self.assertEqual(scheduler.timeline.checkpoint(), before_timeline)

    def test_evicted_hot_weights_wake_queued_ticket_for_replan(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        other_path = Path(self.directory.name) / "other-model.gguf"
        write_synthetic_gguf(other_path, sliding_window=16)
        other = scheduler.register_gguf_model("other-model", other_path)
        hot = runtime_snapshot(manifest)
        queued = scheduler.submit_automated_request(
            request("stale-hot-residency", arrival_us=1_000),
            manifest.model_id,
            hot,
            selection_mode="desktop-baseline",
        )
        self.assertIn("accelerator-b", queued.execution_plan.device_ids)
        self.assertFalse(queued.execution_plan.transitions)

        changed_residency = tuple(
            row for row in hot.residency
            if row.device_id != "accelerator-b"
        ) + (ModelResidencyObservation(
            model_id=other.model_id,
            artifact_sha256=other.artifact_sha256,
            device_id="accelerator-b",
            state="hot",
            resident_tensor_ids=tuple(
                tensor.tensor_id for tensor in other.tensors
            ),
            resident_bytes=other.tensor_bytes,
            generation=2,
        ),)
        observed = replace(
            hot,
            snapshot_id="synthetic-runtime-after-eviction",
            captured_at_us=1_100,
            memory=replace(
                hot.memory,
                snapshot_id="synthetic-memory-after-eviction",
                captured_at_us=1_100,
            ),
            residency=changed_residency,
        )
        changed = scheduler.observe_automated_runtime_snapshot(
            observed, observed_at_us=1_100
        )
        self.assertEqual(changed, (queued.request.request_id,))
        wake = scheduler.wait_runtime_request(
            queued.request.request_id,
            time.monotonic_ns() - 1_100 * 1_000,
        )
        self.assertEqual(wake.dispatch_state, "REPLAN_REQUIRED")
        self.assertEqual(
            wake.dispatch_receipt.wake_reason,
            "residency_observation_changed",
        )

    def test_transition_can_prepare_an_unready_participant_endpoint(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        source = runtime_snapshot(
            manifest, resident_devices=("host-a",)
        )
        states = dict(source.executors)
        states["executor:accelerator-b"] = RuntimeExecutorState(
            executor_id="executor:accelerator-b",
            healthy=False,
            ready=False,
            temperature_millic=40_000,
            battery_ppm=900_000,
            free_slots=0,
            busy_until_us=0,
        )
        candidates = scheduler.generate_automated_candidates(
            request("transition-prepares-unready"),
            manifest.model_id,
            replace(source, executors=states),
        )
        gpu = next(
            row for row in candidates.candidates
            if row.route_family == "whole_model"
            and row.device_ids == ("accelerator-b",)
            and row.residency_variant == "cold"
        )
        self.assertTrue(gpu.plan.transitions)
        self.assertNotIn("EXECUTOR_UNHEALTHY", gpu.rejection_reasons)

    def test_component_normalized_receipts_qualify_an_unseen_shape_bucket(
        self,
    ) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        ))
        snapshot = runtime_snapshot(
            manifest, phone_bandwidth=8_000_000_000
        )
        observed_route = None
        for index, (input_tokens, output_tokens) in enumerate((
            (8, 2), (12, 4), (20, 8), (28, 12)
        )):
            shaped_snapshot = replace(
                snapshot,
                cost_features={
                    "actual_batch_size": input_tokens,
                    "prompt_ubatch_count": (input_tokens + 7) // 8,
                },
            )
            if observed_route is not None:
                probe_request = request(
                    "template-probe-" + str(index),
                    arrival_us=900 + index * 100_000,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                probe = scheduler.generate_automated_candidates(
                    probe_request,
                    manifest.model_id,
                    shaped_snapshot,
                )
                probe_plan = next(
                    row.plan for row in probe.candidates
                    if row.candidate_id == observed_route
                )
                information = scheduler._automated_compiler(
                ).calibration_information(
                    manifest, probe_request, probe_plan
                )
                self.assertEqual(
                    information["priority_class"],
                    1 if index == 1 else 0,
                )
            ticket = scheduler.submit_automated_request(
                request(
                    "template-learn-" + str(index),
                    arrival_us=1_000 + index * 100_000,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
                manifest.model_id,
                shaped_snapshot,
                selection_mode="calibration",
            )
            active = scheduler.wait_runtime_request(
                ticket.request.request_id,
                time.monotonic_ns() - ticket.decision.start_us * 1_000,
            )
            self.assertIn(
                observed_route,
                {None, active.decision.route_id},
            )
            observed_route = active.decision.route_id
            selected = next(
                row for row in active.cost_estimates.estimates
                if row.route_id == active.decision.route_id
            )
            component = selected.details["cost_breakdown"]
            scheduler.complete_automated_request(
                active.request.request_id,
                self.measured_execution_receipt(
                    active,
                    latency_us=component["component_service_us"],
                    fleet_energy_uj=component["component_energy_uj"],
                ),
            )

        later = scheduler.generate_automated_candidates(
            request(
                "template-unseen-bucket",
                arrival_us=1_000_000,
                input_tokens=16,
                output_tokens=6,
            ),
            manifest.model_id,
            replace(
                snapshot,
                cost_features={
                    "actual_batch_size": 16,
                    "prompt_ubatch_count": 2,
                },
            ),
        )
        learned = next(
            row for row in later.candidates
            if row.candidate_id == observed_route
        )
        self.assertTrue(
            learned.plan.route_profile_id.startswith("online-template:")
        )
        self.assertEqual(learned.maturity, "QUALIFIED")

        persisted = dict(scheduler.automated_observation_snapshot())
        exact_by_receipt = {
            observation["receipt_id"]: row["key"]
            for row in persisted["rows"]
            for observation in row["observations"]
        }
        for row in persisted["template_rows"]:
            exact_key = exact_by_receipt[
                row["observations"][0]["receipt_id"]
            ]
            row["key"][1] = exact_key[1]
            row["key"][2] = exact_key[13]
        persisted.pop("store_sha256")
        persisted["store_sha256"] = canonical_sha256(persisted)

        restored, restored_manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        ))
        restored.load_automated_observations(persisted)
        restored.generate_automated_candidates(
            request(
                "component-restored-source",
                arrival_us=2_000_000,
                input_tokens=28,
                output_tokens=12,
            ),
            restored_manifest.model_id,
            replace(
                snapshot,
                cost_features={
                    "actual_batch_size": 28,
                    "prompt_ubatch_count": 4,
                },
            ),
        )
        restored_shorter = restored.generate_automated_candidates(
            request(
                "component-restored-shorter",
                arrival_us=3_000_000,
                input_tokens=24,
                output_tokens=4,
            ),
            restored_manifest.model_id,
            snapshot,
        )
        restored_learned = next(
            row for row in restored_shorter.candidates
            if row.candidate_id == observed_route
        )
        self.assertTrue(
            restored_learned.plan.route_profile_id.startswith(
                "online-template:"
            )
        )
        self.assertEqual(restored_learned.maturity, "QUALIFIED")
        state = scheduler.automated_observation_state()
        self.assertGreaterEqual(state["qualified_templates"], 1)

    def test_component_lookup_does_not_persist_alias_templates(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        selected = scheduler.generate_automated_candidates(
            request("component-alias-source"),
            manifest.model_id,
            runtime_snapshot(manifest),
        ).baseline
        store = RuntimeRouteObservationStore()
        generation = "sha256:" + "b" * 64
        receipt = RuntimeExecutionReceipt(
            ticket_id="component-alias-ticket",
            request_id="component-alias-request",
            artifact_sha256=manifest.artifact_sha256,
            operator_plan_sha256=selected.plan.plan_sha256,
            executor_id=selected.binding.executor_id,
            endpoint=selected.binding.endpoint,
            operator_plan_protocol=(
                selected.binding.operator_plan_protocol
            ),
            participant_executor_ids=tuple(
                row.executor_id for row in selected.binding.participants
            ),
            started_us=0,
            finished_us=100,
            output_sha256="sha256:" + "c" * 64,
            status="COMPLETED",
            energy_boundary_id="synthetic-whole-fleet",
            fleet_energy_uj_by_domain={
                "energy:accelerator-b": 100,
                "energy:helper-c": 100,
                "energy:host-a": 100,
            },
            measurement_evidence_ids=("component-alias-isolated",),
            energy_attribution_kind="isolated",
        )
        self.assertTrue(store.record(
            artifact_sha256=manifest.artifact_sha256,
            capability_generation_sha256=generation,
            plan=selected.plan,
            input_tokens=12,
            output_tokens=4,
            quality_requirement="exact",
            cost_features={},
            receipt=receipt,
            energy_boundary_id="synthetic-whole-fleet",
            required_domain_ids=(
                "energy:accelerator-b",
                "energy:helper-c",
                "energy:host-a",
            ),
            component_service_us=100,
            component_energy_uj=300,
        ))
        source_key, source_rows = next(iter(store._template_rows.items()))
        target_key = (
            source_key[0],
            "sha256:" + "d" * 64,
            "sha256:" + "e" * 64,
            source_key[3],
            source_key[4],
        )
        before = tuple(store._template_rows)

        self.assertIs(
            store._component_rows_from_exact(
                target_key,
                source_rows,
                legacy_prefix=source_key[:4],
            ),
            source_rows,
        )
        self.assertEqual(tuple(store._template_rows), before)
        restored = RuntimeRouteObservationStore()
        restored.import_json(store.to_json())
        self.assertEqual(restored.state(), store.state())

    def test_repeated_component_receipts_estimate_a_shorter_decode(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        ))
        snapshot = runtime_snapshot(
            manifest, phone_bandwidth=8_000_000_000
        )
        observed_route = None
        for index in range(4):
            sample = request(
                "component-repeat-" + str(index),
                arrival_us=1_000 + index * 100_000,
                input_tokens=24,
                output_tokens=16,
            )
            ticket = scheduler.submit_automated_request(
                sample,
                manifest.model_id,
                snapshot,
                selection_mode="calibration",
            )
            active = scheduler.wait_runtime_request(
                sample.request_id,
                time.monotonic_ns() - ticket.decision.start_us * 1_000,
            )
            self.assertIn(
                observed_route,
                {None, active.decision.route_id},
            )
            observed_route = active.decision.route_id
            selected = next(
                row for row in active.cost_estimates.estimates
                if row.route_id == observed_route
            )
            component = selected.details["cost_breakdown"]
            scheduler.complete_automated_request(
                sample.request_id,
                self.measured_execution_receipt(
                    active,
                    latency_us=component["component_service_us"],
                    fleet_energy_uj=component["component_energy_uj"],
                ),
            )

        shorter = scheduler.generate_automated_candidates(
            request(
                "component-shorter-decode",
                arrival_us=1_000_000,
                input_tokens=24,
                output_tokens=4,
            ),
            manifest.model_id,
            replace(
                snapshot,
                cost_features={
                    **snapshot.cost_features,
                    "cpu_utilization_pct": 37,
                    "large_phase_id": 2,
                },
            ),
        )
        learned = next(
            row for row in shorter.candidates
            if row.candidate_id == observed_route
        )
        self.assertTrue(
            learned.plan.route_profile_id.startswith("online-template:")
        )
        self.assertEqual(learned.maturity, "QUALIFIED")

    def test_component_template_anchors_an_exact_measured_shape(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        ))
        snapshot = runtime_snapshot(
            manifest, phone_bandwidth=8_000_000_000
        )
        selected = next(
            row for row in scheduler.generate_automated_candidates(
                request(
                    "component-anchor-plan",
                    input_tokens=24,
                    output_tokens=16,
                ),
                manifest.model_id,
                snapshot,
            ).candidates
            if row.route_family == "operator_split"
            and "helper-c" in row.device_ids
        )
        store = RuntimeRouteObservationStore()
        generation = "sha256:" + "b" * 64
        domains = (
            "energy:accelerator-b",
            "energy:helper-c",
            "energy:host-a",
        )
        for index in range(4):
            receipt = RuntimeExecutionReceipt(
                ticket_id="component-anchor-ticket-" + str(index),
                request_id="component-anchor-request-" + str(index),
                artifact_sha256=manifest.artifact_sha256,
                operator_plan_sha256=selected.plan.plan_sha256,
                executor_id=selected.binding.executor_id,
                endpoint=selected.binding.endpoint,
                operator_plan_protocol=(
                    selected.binding.operator_plan_protocol
                ),
                participant_executor_ids=tuple(
                    row.executor_id for row in selected.binding.participants
                ),
                started_us=0,
                finished_us=200,
                output_sha256="sha256:" + format(index, "064x"),
                status="COMPLETED",
                energy_boundary_id="synthetic-whole-fleet",
                fleet_energy_uj_by_domain={
                    domains[0]: 666,
                    domains[1]: 666,
                    domains[2]: 668,
                },
                transfer_energy_uj_by_link={
                    resource_id.removeprefix("link:"): 1
                    for resource_id in selected.plan.resource_ids
                    if resource_id.startswith("link:")
                },
                measurement_evidence_ids=(
                    "component-anchor-isolated-" + str(index),
                ),
                energy_attribution_kind="isolated",
            )
            self.assertTrue(store.record(
                artifact_sha256=manifest.artifact_sha256,
                capability_generation_sha256=generation,
                plan=selected.plan,
                input_tokens=24,
                output_tokens=16,
                quality_requirement="exact",
                cost_features={"cpu_utilization_pct": 0},
                receipt=receipt,
                energy_boundary_id="synthetic-whole-fleet",
                required_domain_ids=domains,
                component_service_us=100,
                component_energy_uj=1_000,
            ))

        learned = store.estimate(
            artifact_sha256=manifest.artifact_sha256,
            capability_generation_sha256=generation,
            plan=selected.plan,
            input_tokens=24,
            output_tokens=16,
            quality_requirement="exact",
            cost_features={"cpu_utilization_pct": 90},
            component_service_us=50,
            component_energy_uj=500,
        )
        self.assertIsNotNone(learned)
        self.assertTrue(
            learned.profile_id.startswith("online-template:")
        )
        self.assertEqual(learned.service_us, 200)
        self.assertGreaterEqual(learned.service_upper_us, 210)
        self.assertEqual(learned.energy_uj, 2_000)
        self.assertLessEqual(learned.energy_lower_uj, 1_900)
        self.assertGreaterEqual(learned.energy_upper_uj, 2_100)

    def test_legacy_component_evidence_survives_transition_update(
        self,
    ) -> None:
        source_catalog = catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        )
        source, manifest = self.scheduler_and_manifest(source_catalog)
        snapshot = runtime_snapshot(
            manifest,
            phone_bandwidth=8_000_000_000,
            resident_devices=("host-a", "accelerator-b"),
        )
        selected = next(
            row for row in source.generate_automated_candidates(
                request(
                    "legacy-component-source",
                    input_tokens=24,
                    output_tokens=16,
                ),
                manifest.model_id,
                snapshot,
            ).candidates
            if row.route_family == "operator_split"
            and "helper-c" in row.device_ids
            and row.plan.transitions
        )
        compiler = source._automated_compiler()
        route_identity = compiler.route_capability_identity(
            selected.plan, selected.binding.executor_id
        )
        store = RuntimeRouteObservationStore()
        domains = (
            "energy:accelerator-b",
            "energy:helper-c",
            "energy:host-a",
        )
        latency_us = selected.cost.component_service_us + 100
        energy_uj = selected.cost.component_energy_uj + 100
        for index in range(4):
            receipt = RuntimeExecutionReceipt(
                ticket_id="legacy-component-ticket-" + str(index),
                request_id="legacy-component-request-" + str(index),
                artifact_sha256=manifest.artifact_sha256,
                operator_plan_sha256=selected.plan.plan_sha256,
                executor_id=selected.binding.executor_id,
                endpoint=selected.binding.endpoint,
                operator_plan_protocol=(
                    selected.binding.operator_plan_protocol
                ),
                participant_executor_ids=tuple(
                    row.executor_id for row in selected.binding.participants
                ),
                started_us=0,
                finished_us=latency_us,
                output_sha256="sha256:" + format(index, "064x"),
                status="COMPLETED",
                energy_boundary_id="synthetic-whole-fleet",
                fleet_energy_uj_by_domain={
                    domains[0]: energy_uj // 3,
                    domains[1]: energy_uj // 3,
                    domains[2]: energy_uj - 2 * (energy_uj // 3),
                },
                transfer_energy_uj_by_link={
                    resource_id.removeprefix("link:"): 1
                    for resource_id in selected.plan.resource_ids
                    if resource_id.startswith("link:")
                },
                measurement_evidence_ids=(
                    "legacy-component-isolated-" + str(index),
                ),
                energy_attribution_kind="isolated",
            )
            self.assertTrue(store.record(
                artifact_sha256=manifest.artifact_sha256,
                capability_generation_sha256=route_identity,
                plan=selected.plan,
                input_tokens=24,
                output_tokens=16,
                quality_requirement="exact",
                cost_features={},
                receipt=receipt,
                energy_boundary_id="synthetic-whole-fleet",
                required_domain_ids=domains,
                component_service_us=selected.cost.component_service_us,
                component_energy_uj=selected.cost.component_energy_uj,
            ))

        transition_ids = {
            row.transition_id for row in selected.plan.transitions
        }
        updated_catalog = replace(
            source_catalog,
            catalog_id=source_catalog.catalog_id + ":transition-update",
            executors=tuple(
                replace(
                    row,
                    exclusive_residency_resource_id="compute:helper-c",
                )
                if row.device_id == "helper-c" else row
                for row in source_catalog.executors
            ),
            composite_executors=tuple(
                replace(
                    row,
                    replacement_group_by_device={
                        **dict(row.replacement_group_by_device),
                        "helper-c": "compute:helper-c",
                    },
                )
                if row.executor_id == selected.binding.executor_id else row
                for row in source_catalog.composite_executors
            ),
            transitions=tuple(
                replace(
                    row,
                    fixed_latency_us=row.fixed_latency_us + 40_000,
                )
                if row.transition_id in transition_ids else row
                for row in source_catalog.transitions
            ),
        )
        target, target_manifest = self.scheduler_and_manifest(
            updated_catalog
        )
        legacy = store.to_json()
        legacy["template_rows"][0]["key"][2] = (
            compiler.observation_store._template_sha256(selected.plan)
        )
        legacy.pop("store_sha256")
        legacy["store_sha256"] = canonical_sha256(legacy)
        target.load_automated_observations(legacy)
        before = next(
            row for row in target.generate_automated_candidates(
                request(
                    "legacy-component-before",
                    input_tokens=24,
                    output_tokens=16,
                ),
                target_manifest.model_id,
                snapshot,
            ).candidates
            if row.candidate_id == selected.candidate_id
        )
        self.assertFalse(
            (before.plan.route_profile_id or "").startswith("online-")
        )

        self.assertEqual(
            target.rebind_legacy_automated_component_observations(
                source_catalog,
                target_manifest.model_id,
                selected.plan,
                selected.binding.executor_id,
                before.plan,
            ),
            4,
        )
        rebound = target.automated_observation_snapshot()
        self.assertTrue(all(
            observation["energy_uj"] is None
            for row in rebound["template_rows"]
            for observation in row["observations"]
        ))
        restored = RuntimeRouteObservationStore()
        restored.import_json(dict(rebound))
        after = next(
            row for row in target.generate_automated_candidates(
                request(
                    "legacy-component-after",
                    input_tokens=24,
                    output_tokens=16,
                ),
                target_manifest.model_id,
                snapshot,
            ).candidates
            if row.candidate_id == selected.candidate_id
        )
        self.assertTrue(
            after.plan.route_profile_id.startswith("online-template:")
        )
        self.assertEqual(after.maturity, "QUALIFIED")
        self.assertEqual(after.cost.latency_evidence, "MEASURED")
        self.assertNotEqual(after.cost.energy_evidence, "MEASURED")
        self.assertEqual(
            after.cost.fleet_energy_upper_uj,
            after.cost.warm_execution_energy_upper_uj
                + after.cost.transition_energy_upper_uj,
        )

    def test_transition_and_warm_execution_evidence_are_separate(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        ))
        snapshot = runtime_snapshot(
            manifest,
            phone_bandwidth=8_000_000_000,
            resident_devices=("host-a", "accelerator-b"),
        )
        selected = next(
            row for row in scheduler.generate_automated_candidates(
                request("separate-transition-warm"),
                manifest.model_id,
                snapshot,
            ).candidates
            if row.route_family == "operator_split"
            and "helper-c" in row.device_ids
            and row.plan.transitions
        )
        transition = selected.plan.transitions[0]
        participant = next(
            row for row in selected.binding.participants
            if row.device_id == transition.device_id
        )
        compiler = scheduler._automated_compiler()
        store = RuntimeRouteObservationStore()
        component = "sha256:" + "c" * 64
        capability = compiler.component_capability_identity(selected.plan)
        domains = (
            "energy:accelerator-b",
            "energy:helper-c",
            "energy:host-a",
        )
        last = None
        for index, value in enumerate((1_000, 1_020, 990, 1_010)):
            last = RuntimeTransitionReceipt(
                ticket_id="transition-ticket-" + str(index),
                request_id="transition-request-" + str(index),
                artifact_sha256=manifest.artifact_sha256,
                operator_plan_sha256=selected.plan.plan_sha256,
                transition_id=transition.transition_id,
                executor_id=participant.executor_id,
                endpoint=participant.endpoint,
                device_id=transition.device_id,
                source_state=transition.source_state,
                target_state=transition.target_state,
                resource_ids=transition.resource_ids,
                started_us=0,
                finished_us=value,
                status="COMPLETED",
                energy_boundary_id="synthetic-whole-fleet",
                fleet_energy_uj_by_domain={
                    domains[0]: value,
                    domains[1]: value,
                    domains[2]: value,
                },
                measurement_evidence_ids=(
                    "transition-isolated-" + str(index),
                ),
                energy_attribution_kind="isolated",
            )
            self.assertTrue(store.record_transition(
                artifact_sha256=manifest.artifact_sha256,
                capability_generation_sha256=capability,
                component_identity_sha256=component,
                plan=selected.plan,
                receipt=last,
                energy_boundary_id="synthetic-whole-fleet",
                required_domain_ids=domains,
            ))

        estimate = store.transition_estimate(
            artifact_sha256=manifest.artifact_sha256,
            capability_generation_sha256=capability,
            component_identity_sha256=component,
            receipt=last,
        )
        self.assertEqual(estimate.maturity, "QUALIFIED")
        warm = RuntimeExecutionReceipt(
            ticket_id="warm-ticket",
            request_id="warm-request",
            artifact_sha256=manifest.artifact_sha256,
            operator_plan_sha256=selected.plan.plan_sha256,
            executor_id=selected.binding.executor_id,
            endpoint=selected.binding.endpoint,
            operator_plan_protocol=selected.binding.operator_plan_protocol,
            participant_executor_ids=tuple(
                row.executor_id for row in selected.binding.participants
            ),
            started_us=2_000,
            finished_us=3_000,
            output_sha256="sha256:" + "d" * 64,
            status="COMPLETED",
            energy_boundary_id="synthetic-whole-fleet",
            fleet_energy_uj_by_domain={
                domains[0]: 500,
                domains[1]: 500,
                domains[2]: 500,
            },
            measurement_evidence_ids=("warm-isolated",),
            energy_attribution_kind="isolated",
            energy_scope="warm_execution",
        )
        self.assertTrue(store.record(
            artifact_sha256=manifest.artifact_sha256,
            capability_generation_sha256=capability,
            plan=selected.plan,
            input_tokens=12,
            output_tokens=4,
            quality_requirement="exact",
            cost_features={},
            receipt=warm,
            energy_boundary_id="synthetic-whole-fleet",
            required_domain_ids=domains,
            component_service_us=selected.cost.component_service_us,
            component_energy_uj=selected.cost.component_energy_uj,
        ))
        self.assertEqual(store.state()["qualified_transitions"], 1)
        exported = store.to_json()
        self.assertEqual(len(exported["transition_rows"]), 1)
        self.assertEqual(
            exported["template_rows"][0]["observations"][0]["energy_uj"],
            1_500,
        )
        restored = RuntimeRouteObservationStore()
        restored.import_json(exported)
        self.assertEqual(restored.to_json(), exported)

    def test_legacy_cold_load_evidence_rebinds_ownership_only(self) -> None:
        source_catalog = catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        )
        source, manifest = self.scheduler_and_manifest(source_catalog)
        snapshot = runtime_snapshot(
            manifest,
            phone_bandwidth=8_000_000_000,
            resident_devices=("host-a", "accelerator-b"),
        )
        runtime_request = request("legacy-transition-source")
        selected = next(
            row for row in source.generate_automated_candidates(
                runtime_request, manifest.model_id, snapshot
            ).candidates
            if row.route_family == "operator_split"
            and "helper-c" in row.device_ids
            and row.plan.transitions
        )
        transition = selected.plan.transitions[0]
        participant = next(
            row for row in selected.binding.participants
            if row.device_id == transition.device_id
        )
        compiler = source._automated_compiler()
        domains = (
            "energy:accelerator-b",
            "energy:helper-c",
            "energy:host-a",
        )
        for index, energy_uj in enumerate((600, 610, 590, 605)):
            shares = (energy_uj // 3, energy_uj // 3)
            receipt = RuntimeTransitionReceipt(
                ticket_id="legacy-transition-ticket-" + str(index),
                request_id=runtime_request.request_id,
                artifact_sha256=manifest.artifact_sha256,
                operator_plan_sha256=selected.plan.plan_sha256,
                transition_id=transition.transition_id,
                executor_id=participant.executor_id,
                endpoint=participant.endpoint,
                device_id=transition.device_id,
                source_state=transition.source_state,
                target_state=transition.target_state,
                resource_ids=transition.resource_ids,
                started_us=0,
                finished_us=1_000 + index,
                status="COMPLETED",
                energy_boundary_id="synthetic-whole-fleet",
                fleet_energy_uj_by_domain={
                    domains[0]: shares[0],
                    domains[1]: shares[1],
                    domains[2]: energy_uj - sum(shares),
                },
                measurement_evidence_ids=(
                    "legacy-transition-isolated-" + str(index),
                ),
                energy_attribution_kind="isolated",
            )
            self.assertEqual(
                compiler.record_transition_observations(
                    manifest, selected.plan, (receipt,)
                ),
                1,
            )

        updated_catalog = replace(
            source_catalog,
            catalog_id=source_catalog.catalog_id + ":ownership-update",
            executors=tuple(
                replace(
                    row,
                    exclusive_residency_resource_id="compute:helper-c",
                )
                if row.device_id == "helper-c" else row
                for row in source_catalog.executors
            ),
            composite_executors=tuple(
                replace(
                    row,
                    replacement_group_by_device={
                        **dict(row.replacement_group_by_device),
                        "helper-c": "compute:helper-c",
                    },
                )
                if row.executor_id == selected.binding.executor_id else row
                for row in source_catalog.composite_executors
            ),
        )
        target, target_manifest = self.scheduler_and_manifest(
            updated_catalog
        )
        target.load_automated_observations(
            compiler.observation_store.to_json(),
            source_catalog=source_catalog,
        )
        rebound = next(
            row for row in target.generate_automated_candidates(
                request("legacy-transition-before"),
                target_manifest.model_id,
                snapshot,
            ).candidates
            if row.candidate_id == selected.candidate_id
        )
        self.assertEqual(
            rebound.plan.transitions[0].energy_maturity, "QUALIFIED"
        )
        self.assertEqual(
            rebound.cost.fleet_energy_upper_uj,
            rebound.cost.warm_execution_energy_upper_uj
                + rebound.cost.transition_energy_upper_uj,
        )

    def test_exact_cold_evidence_does_not_require_adaptive_contracts(self) -> None:
        with mock.patch(
            "research_dev.scheduler._unified.automated_candidates_ops.generation.adaptive_probe_contracts",
            return_value={},
        ), mock.patch(
            "research_dev.scheduler._unified.automated_candidates_ops.application.adaptive_probe_contracts",
            return_value={},
        ):
            self.test_legacy_cold_load_evidence_rebinds_ownership_only()

    def test_transition_outlier_keeps_conservative_bound(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        ))
        snapshot = runtime_snapshot(
            manifest,
            phone_bandwidth=8_000_000_000,
            resident_devices=("host-a", "accelerator-b"),
        )
        selected = next(
            row for row in scheduler.generate_automated_candidates(
                request("transition-outlier"),
                manifest.model_id,
                snapshot,
            ).candidates
            if row.route_family == "operator_split"
            and "helper-c" in row.device_ids
            and row.plan.transitions
        )
        transition = selected.plan.transitions[0]
        participant = next(
            row for row in selected.binding.participants
            if row.device_id == transition.device_id
        )
        compiler = scheduler._automated_compiler()
        store = RuntimeRouteObservationStore()
        component = "sha256:" + "e" * 64
        capability = compiler.component_capability_identity(selected.plan)
        domains = (
            "energy:accelerator-b",
            "energy:helper-c",
            "energy:host-a",
        )
        last = None
        for index, value in enumerate((10_000, 1_000, 1_020, 990, 1_010, 995)):
            last = RuntimeTransitionReceipt(
                ticket_id="transition-outlier-ticket-" + str(index),
                request_id="transition-outlier-request-" + str(index),
                artifact_sha256=manifest.artifact_sha256,
                operator_plan_sha256=selected.plan.plan_sha256,
                transition_id=transition.transition_id,
                executor_id=participant.executor_id,
                endpoint=participant.endpoint,
                device_id=transition.device_id,
                source_state=transition.source_state,
                target_state=transition.target_state,
                resource_ids=transition.resource_ids,
                started_us=0,
                finished_us=value,
                status="COMPLETED",
                energy_boundary_id="synthetic-whole-fleet",
                fleet_energy_uj_by_domain={
                    domain_id: value for domain_id in domains
                },
                measurement_evidence_ids=(
                    "transition-outlier-evidence-" + str(index),
                ),
                energy_attribution_kind="isolated",
            )
            self.assertTrue(store.record_transition(
                artifact_sha256=manifest.artifact_sha256,
                capability_generation_sha256=capability,
                component_identity_sha256=component,
                plan=selected.plan,
                receipt=last,
                energy_boundary_id="synthetic-whole-fleet",
                required_domain_ids=domains,
            ))

        estimate = store.transition_estimate(
            artifact_sha256=manifest.artifact_sha256,
            capability_generation_sha256=capability,
            component_identity_sha256=component,
            receipt=last,
        )
        self.assertEqual(estimate.maturity, "QUALIFIED")
        self.assertEqual(estimate.sample_count, 6)
        self.assertEqual(estimate.latency_us, 1_005)
        self.assertEqual(estimate.latency_upper_us, 11_000)
        self.assertEqual(estimate.latency_upper_maturity, "QUALIFIED")
        self.assertEqual(estimate.latency_upper_coverage_ppm, 1_000_000)
        self.assertEqual(estimate.energy_uj, 3_015)
        self.assertEqual(estimate.energy_upper_uj, 33_000)

    def test_transition_safe_bound_is_separate_from_point_accuracy(
        self,
    ) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        snapshot = runtime_snapshot(manifest)
        selected = next(
            row for row in scheduler.generate_automated_candidates(
                request("transition-bound"), manifest.model_id, snapshot
            ).candidates
            if row.plan.transitions
        )
        transition = selected.plan.transitions[0]
        compiler = scheduler._automated_compiler()
        store = RuntimeRouteObservationStore()
        component = "sha256:" + "d" * 64
        capability = compiler.component_capability_identity(selected.plan)
        participant = next(
            row for row in selected.binding.participants
            if row.device_id == transition.device_id
        )
        domains = (
            "energy:accelerator-b",
            "energy:helper-c",
            "energy:host-a",
        )
        last = None
        for index, latency_us in enumerate(
            (20_000, 30_000, 300_000, 250_000, 19_000, 27_000)
        ):
            last = RuntimeTransitionReceipt(
                ticket_id="transition-bound-ticket-" + str(index),
                request_id="transition-bound-request-" + str(index),
                artifact_sha256=manifest.artifact_sha256,
                operator_plan_sha256=selected.plan.plan_sha256,
                transition_id=transition.transition_id,
                executor_id=participant.executor_id,
                endpoint=participant.endpoint,
                device_id=transition.device_id,
                source_state=transition.source_state,
                target_state=transition.target_state,
                resource_ids=transition.resource_ids,
                started_us=0,
                finished_us=latency_us,
                status="COMPLETED",
                energy_boundary_id="synthetic-whole-fleet",
                fleet_energy_uj_by_domain={
                    domain_id: 1_000 for domain_id in domains
                },
                measurement_evidence_ids=(
                    "transition-bound-evidence-" + str(index),
                ),
                energy_attribution_kind="isolated",
            )
            self.assertTrue(store.record_transition(
                artifact_sha256=manifest.artifact_sha256,
                capability_generation_sha256=capability,
                component_identity_sha256=component,
                plan=selected.plan,
                receipt=last,
                energy_boundary_id="synthetic-whole-fleet",
                required_domain_ids=domains,
            ))

        estimate = store.transition_estimate(
            artifact_sha256=manifest.artifact_sha256,
            capability_generation_sha256=capability,
            component_identity_sha256=component,
            receipt=last,
        )
        self.assertEqual(estimate.latency_maturity, "QUARANTINED")
        self.assertGreater(estimate.latency_mape_ppm, 200_000)
        self.assertEqual(estimate.latency_upper_maturity, "QUALIFIED")
        self.assertEqual(estimate.latency_upper_coverage_ppm, 1_000_000)
        self.assertEqual(estimate.latency_upper_us, 330_000)

    def test_quarantined_online_transition_keeps_desktop_control(self) -> None:
        scheduler, manifest = self.scheduler_with_gpu_control()
        snapshot = runtime_snapshot(
            manifest, resident_devices=("host-a",)
        )
        runtime_request = request("quarantined-desktop-transition")
        baseline = scheduler.generate_automated_candidates(
            runtime_request, manifest.model_id, snapshot
        ).baseline
        self.assertEqual(baseline.binding.executor_id, "executor:accelerator-b")
        self.assertEqual(len(baseline.plan.transitions), 1)
        transition = baseline.plan.transitions[0]
        domains = (
            "energy:accelerator-b",
            "energy:helper-c",
            "energy:host-a",
        )
        compiler = scheduler._automated_compiler()
        for index, latency_us in enumerate((1_000, 1_000, 5_000, 5_000)):
            receipt = RuntimeTransitionReceipt(
                ticket_id="quarantined-transition-" + str(index),
                request_id=runtime_request.request_id,
                artifact_sha256=manifest.artifact_sha256,
                operator_plan_sha256=baseline.plan.plan_sha256,
                transition_id=transition.transition_id,
                executor_id=baseline.binding.executor_id,
                endpoint=baseline.binding.endpoint,
                device_id=transition.device_id,
                source_state=transition.source_state,
                target_state=transition.target_state,
                resource_ids=transition.resource_ids,
                started_us=0,
                finished_us=latency_us,
                status="COMPLETED",
                energy_boundary_id="synthetic-whole-fleet",
                fleet_energy_uj_by_domain={
                    domains[0]: latency_us,
                    domains[1]: latency_us,
                    domains[2]: latency_us,
                },
                measurement_evidence_ids=(
                    "quarantined-transition-evidence-" + str(index),
                ),
                energy_attribution_kind="isolated",
            )
            self.assertEqual(
                compiler.record_transition_observations(
                    manifest, baseline.plan, (receipt,)
                ),
                1,
            )

        refreshed = scheduler.generate_automated_candidates(
            runtime_request, manifest.model_id, snapshot
        ).baseline
        self.assertEqual(refreshed.binding.executor_id, baseline.binding.executor_id)
        self.assertTrue(refreshed.admitted)
        self.assertEqual(refreshed.rejection_reasons, ())
        self.assertEqual(refreshed.plan.transitions[0].maturity, "QUALIFIED")
        self.assertEqual(
            refreshed.plan.transitions[0].energy_maturity, "QUALIFIED"
        )

    def test_component_model_accounts_for_fixed_route_overhead(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        ))
        store = RuntimeRouteObservationStore()
        generation = "sha256:" + "b" * 64
        samples = []
        selected_route = None
        for index, (input_tokens, output_tokens) in enumerate((
            (8, 2), (12, 4), (20, 8), (28, 12)
        )):
            row_request = request(
                "fixed-overhead-" + str(index),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            features = {
                "actual_batch_size": input_tokens,
                "prompt_ubatch_count": (input_tokens + 7) // 8,
            }
            candidates = scheduler.generate_automated_candidates(
                row_request,
                manifest.model_id,
                replace(
                    runtime_snapshot(
                        manifest, phone_bandwidth=8_000_000_000
                    ),
                    cost_features=features,
                ),
            )
            if selected_route is None:
                selected = next(
                    row for row in candidates.candidates
                    if row.admitted
                    and row.route_family == "operator_split"
                    and "helper-c" in row.device_ids
                )
                selected_route = selected.candidate_id
            else:
                selected = next(
                    row for row in candidates.candidates
                    if row.candidate_id == selected_route
                )
            samples.append((row_request, features, selected))

        fixed_latency_us = 10 * max(
            row.cost.component_service_us for _, _, row in samples
        )
        fixed_energy_uj = 10 * max(
            row.cost.component_energy_uj for _, _, row in samples
        )
        for index, (row_request, features, selected) in enumerate(samples):
            latency_us = (
                fixed_latency_us + 2 * selected.cost.component_service_us
            )
            fleet_energy_uj = (
                fixed_energy_uj + 3 * selected.cost.component_energy_uj
            )
            per_domain = fleet_energy_uj // 3
            energy = {
                "energy:accelerator-b": per_domain,
                "energy:helper-c": per_domain,
                "energy:host-a": fleet_energy_uj - 2 * per_domain,
            }
            receipt = RuntimeExecutionReceipt(
                ticket_id="fixed-overhead-ticket-" + str(index),
                request_id=row_request.request_id,
                artifact_sha256=manifest.artifact_sha256,
                operator_plan_sha256=selected.plan.plan_sha256,
                executor_id=selected.binding.executor_id,
                endpoint=selected.binding.endpoint,
                operator_plan_protocol=(
                    selected.binding.operator_plan_protocol
                ),
                participant_executor_ids=tuple(
                    row.executor_id for row in selected.binding.participants
                ),
                started_us=0,
                finished_us=latency_us,
                output_sha256="sha256:" + "c" * 64,
                status="COMPLETED",
                energy_boundary_id="synthetic-whole-fleet",
                fleet_energy_uj_by_domain=energy,
                transfer_energy_uj_by_link={
                    resource_id.removeprefix("link:"): 1
                    for resource_id in selected.plan.resource_ids
                    if resource_id.startswith("link:")
                },
                measurement_evidence_ids=("fixed-overhead-isolated",),
                energy_attribution_kind="isolated",
            )
            self.assertTrue(store.record(
                artifact_sha256=manifest.artifact_sha256,
                capability_generation_sha256=generation,
                plan=selected.plan,
                input_tokens=row_request.input_tokens,
                output_tokens=row_request.output_tokens,
                quality_requirement=row_request.quality_requirement,
                cost_features=features,
                receipt=receipt,
                energy_boundary_id="synthetic-whole-fleet",
                required_domain_ids=tuple(energy),
                component_service_us=selected.cost.component_service_us,
                component_energy_uj=selected.cost.component_energy_uj,
            ))

        probe_request = request(
            "fixed-overhead-probe", input_tokens=16, output_tokens=6
        )
        probe_features = {
            "actual_batch_size": 16,
            "prompt_ubatch_count": 2,
        }
        probe_candidates = scheduler.generate_automated_candidates(
            probe_request,
            manifest.model_id,
            replace(
                runtime_snapshot(
                    manifest, phone_bandwidth=8_000_000_000
                ),
                cost_features=probe_features,
            ),
        )
        probe_plan = next(
            row.plan for row in probe_candidates.candidates
            if row.candidate_id == selected_route
        )
        probe_cost = next(
            row.cost for row in probe_candidates.candidates
            if row.candidate_id == selected_route
        )
        learned = store.estimate(
            artifact_sha256=manifest.artifact_sha256,
            capability_generation_sha256=generation,
            plan=probe_plan,
            input_tokens=probe_request.input_tokens,
            output_tokens=probe_request.output_tokens,
            quality_requirement=probe_request.quality_requirement,
            cost_features=probe_features,
            component_service_us=probe_cost.component_service_us,
            component_energy_uj=probe_cost.component_energy_uj,
        )
        self.assertIsNotNone(learned)
        self.assertEqual(learned.maturity, "QUALIFIED")
        self.assertLessEqual(learned.latency_mape_ppm, 200_000)
        self.assertLessEqual(learned.energy_mape_ppm, 200_000)
        self.assertEqual(learned.latency_upper_coverage_ppm, 1_000_000)
        self.assertEqual(learned.energy_upper_coverage_ppm, 1_000_000)

    def test_cold_transition_receipts_gate_the_physical_contract(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        ))
        snapshot = runtime_snapshot(
            manifest,
            phone_bandwidth=8_000_000_000,
            resident_devices=(),
        )
        ticket = scheduler.submit_automated_request(
            request("cold-transition"), manifest.model_id, snapshot
        )
        epoch_ns = time.monotonic_ns() - ticket.decision.start_us * 1_000
        active = scheduler.wait_runtime_request(ticket.request.request_id, epoch_ns)
        self.assertEqual(active.transition_status, "PENDING")
        self.assertTrue(active.execution_plan.transitions)
        self.assertEqual(
            [
                row["event_kind"]
                for row in scheduler.runtime_decision_log()["records"]
            ],
            ["DECISION"],
        )
        with self.assertRaises(UnifiedScheduleError):
            scheduler.runtime_execution_ticket(active.request.request_id)

        receipts = self.transition_receipts(active)
        ready = scheduler.record_automated_transition_receipts(
            active.request.request_id, receipts
        )
        self.assertEqual(ready.transition_status, "COMPLETED")
        acquired = scheduler.runtime_decision_log()["records"][-1]
        self.assertEqual(acquired["event_kind"], "ACQUIRED")
        self.assertEqual(
            acquired["selected"]["transition_receipts"],
            [row.to_json() for row in receipts],
        )
        physical = scheduler.runtime_execution_ticket(
            active.request.request_id
        )
        self.assertEqual(physical.binding, active.binding)
        self.assertEqual(physical.execution_plan, active.execution_plan)
        scheduler.complete_automated_request(
            active.request.request_id, self.execution_receipt(active)
        )
        terminal = scheduler.runtime_decision_log()["records"][-1]
        self.assertEqual(
            terminal["selected"]["transition_receipts"],
            [row.to_json() for row in receipts],
        )

    def test_invalid_transition_receipt_leaves_all_state_unchanged(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        snapshot = runtime_snapshot(manifest, resident_devices=())
        ticket = scheduler.submit_automated_request(
            request("invalid-transition-receipt"), manifest.model_id, snapshot
        )
        epoch_ns = time.monotonic_ns() - ticket.decision.start_us * 1_000
        active = scheduler.wait_runtime_request(ticket.request.request_id, epoch_ns)
        receipts = self.transition_receipts(active)
        invalid = (replace(receipts[0], endpoint="synthetic://wrong"),) + receipts[1:]
        before = (
            scheduler.timeline.causal_state(),
            scheduler.runtime_memory_state(),
            scheduler.runtime_controller_snapshot(),
            scheduler.runtime_decision_log_bytes(),
        )
        with self.assertRaises(UnifiedScheduleError):
            scheduler.record_automated_transition_receipts(
                active.request.request_id, invalid
            )
        self.assertEqual(scheduler.timeline.causal_state(), before[0])
        self.assertEqual(scheduler.runtime_memory_state(), before[1])
        self.assertEqual(scheduler.runtime_controller_snapshot(), before[2])
        self.assertEqual(scheduler.runtime_decision_log_bytes(), before[3])

    def test_current_memory_conflict_rejects_route_before_selection(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        ))
        snapshot = runtime_snapshot(
            manifest, phone_bandwidth=8_000_000_000
        )
        original_preview = scheduler._runtime_memory.preview

        def preview(demands, memory_snapshot, **interval):
            if any(
                row.resource_id == "phone-memory" for row in demands
            ):
                raise RuntimeResourceError(
                    "memory capacity is insufficient: phone-memory"
                )
            return original_preview(
                demands, memory_snapshot, **interval
            )

        with mock.patch.object(
            scheduler._runtime_memory, "preview", side_effect=preview
        ):
            ticket = scheduler.submit_automated_request(
                request("memory-selection-fallback"),
                manifest.model_id,
                snapshot,
            )

        self.assertNotIn("helper-c", ticket.execution_plan.device_ids)
        memory_rejections = {
            route_id: reason
            for route_id, reason in ticket.decision.rejected
            if reason == "MEMORY_CAPACITY_CURRENT:phone-memory"
        }
        self.assertTrue(memory_rejections)
        self.assertTrue(all(
            "helper-c" in next(
                row for row in ticket.cost_estimates.estimates
                if row.route_id == route_id
            ).details["device_ids"]
            for route_id in memory_rejections
        ))

    def test_queued_exclusive_residency_replacement_is_not_deleted(
        self,
    ) -> None:
        source = catalog()
        gpu = replace(
            source.executor_by_device["accelerator-b"],
            qualified_fallback=True,
            exclusive_residency_resource_id="compute:accelerator-b",
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=(gpu,),
            transitions=tuple(
                row for row in source.transitions
                if row.device_id == "accelerator-b"
            ),
        ).to_json())
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        target = scheduler.register_gguf_model(
            "synthetic-target", self.path
        )
        other_path = Path(self.directory.name) / "resident-model.gguf"
        write_synthetic_gguf(other_path, sliding_window=64)
        resident = scheduler.register_gguf_model(
            "synthetic-resident", other_path
        )
        snapshot = runtime_snapshot(
            target,
            include_phone=False,
            gpu_busy_until_us=50_000,
            resident_devices=(),
        )
        snapshot = replace(
            snapshot,
            executors={
                "executor:accelerator-b": executor_state(
                    "executor:accelerator-b",
                    busy_until_us=50_000,
                )
            },
            residency=(ModelResidencyObservation(
                model_id=resident.model_id,
                artifact_sha256=resident.artifact_sha256,
                device_id="accelerator-b",
                state="hot",
                resident_tensor_ids=tuple(
                    row.tensor_id for row in resident.tensors
                ),
                resident_bytes=resident.tensor_bytes,
                generation=7,
            ),),
        )
        probe = scheduler.generate_automated_candidates(
            request("replacement-probe"), target.model_id, snapshot
        ).baseline
        gpu_required = sum(
            row.required_bytes for row in probe.plan.memory_demands
            if row.resource_id == "gpu-memory"
        )
        capacities = dict(snapshot.memory.capacities)
        capacities["gpu-memory"] = DeviceMemoryCapacity(
            "gpu-memory",
            gpu_required,
            resident.tensor_bytes,
            0,
        )
        constrained = replace(
            snapshot,
            memory=replace(snapshot.memory, capacities=capacities),
        )

        ticket = scheduler.submit_automated_request(
            request("queued-replacement"),
            target.model_id,
            constrained,
            selection_mode="desktop-baseline",
        )

        self.assertGreaterEqual(ticket.decision.start_us, 50_000)
        self.assertEqual(ticket.execution_plan.device_ids, ("accelerator-b",))
        self.assertEqual(len(ticket.execution_plan.transitions), 1)
        eviction = ticket.execution_plan.transitions[0].evictions
        self.assertEqual(len(eviction), 1)
        self.assertEqual(eviction[0].model_id, resident.model_id)
        self.assertEqual(eviction[0].artifact_sha256, resident.artifact_sha256)
        self.assertEqual(eviction[0].generation, 7)
        weights = next(
            row for row in ticket.execution_plan.memory_demands
            if row.demand_id == "weights:accelerator-b"
        )
        self.assertGreater(weights.replaceable_bytes, 0)

    def test_shared_replacement_group_projects_one_atomic_transition(
        self,
    ) -> None:
        source = catalog(gpu_whole_model=False)
        target = GGUFModelManifestLoader.load(
            "target-model", self.path
        )
        previous_path = Path(self.directory.name) / "previous-model.gguf"
        write_synthetic_gguf(previous_path, sliding_window=96)
        previous_manifest = GGUFModelManifestLoader.load(
            "previous-model", previous_path
        )
        shared_resource_id = "compute:accelerator-b"
        host = replace(
            source.executor_by_device["host-a"],
            exclusive_residency_resource_id=None,
            layer_fractions_ppm=(),
        )
        gpu = replace(
            source.executor_by_device["accelerator-b"],
            exclusive_residency_resource_id=shared_resource_id,
            layer_fractions_ppm=(),
        )

        def placements(manifest):
            return tuple(
                RuntimeCompositeOperatorPlacement(
                    operator_id=operator.operator_id,
                    primary_device_id=(
                        "host-a"
                        if operator.kind in {"embedding", "lm_head"}
                        else "accelerator-b"
                    ),
                    helper_device_id=None,
                    split_axis="none",
                    split_fraction_ppm=0,
                )
                for operator in manifest.operators
            )

        def coordinator(executor_id, artifact_sha256, rows):
            return RuntimeCompositeExecutorCapability(
                executor_id=executor_id,
                endpoint="synthetic://" + executor_id,
                backend="backend:desktop",
                coordinator_device_id="host-a",
                participant_device_ids=("host-a", "accelerator-b"),
                participant_resource_ids={
                    "host-a": ("compute:host-a",),
                    "accelerator-b": ("compute:accelerator-b",),
                },
                route_family="layer_placement",
                assisted_operator_kind=None,
                split_axis="none",
                split_fractions_ppm=(),
                layer_fractions_ppm=(),
                residency_states=("cold", "hot", "warm"),
                resource_ids=("compute:host-a", "compute:accelerator-b"),
                operator_plan_protocol="synthetic-composite-v1",
                maturity="QUALIFIED",
                evidence_ids=(executor_id,),
                artifact_sha256=artifact_sha256,
                operator_placements=rows,
                replacement_group_by_device={
                    "host-a": shared_resource_id,
                    "accelerator-b": shared_resource_id,
                },
            )

        target_placements = placements(target)
        target_executor = coordinator(
            "coordinator:target", target.artifact_sha256,
            target_placements,
        )
        previous_executor = coordinator(
            "coordinator:previous", previous_manifest.artifact_sha256,
            placements(previous_manifest),
        )
        transition = RuntimeTransitionCapability(
            transition_id="load:target",
            device_id="accelerator-b",
            source_state="cold",
            target_state="hot",
            fixed_latency_us=100,
            bandwidth_bytes_per_s=1_000_000_000,
            fixed_energy_uj=100,
            dynamic_pj_per_byte=100,
            resource_ids=("compute:host-a", shared_resource_id),
            maturity="QUALIFIED",
            evidence_ids=("synthetic-shared-replacement",),
            executor_id=target_executor.executor_id,
            prepares_device_ids=("host-a", "accelerator-b"),
        )
        previous_transition = replace(
            transition,
            transition_id="load:previous",
            executor_id=previous_executor.executor_id,
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=tuple(
                host if row.device_id == host.device_id else
                gpu if row.device_id == gpu.device_id else row
                for row in source.executors
            ),
            composite_executors=(target_executor, previous_executor),
            desktop_control_profiles=(
                RuntimeDesktopControlProfile(
                    profile_id="synthetic-target-control",
                    artifact_sha256=target.artifact_sha256,
                    executor_id=target_executor.executor_id,
                    operator_placements=target_placements,
                    maturity="QUALIFIED",
                    evidence_ids=("synthetic-target-control",),
                ),
                RuntimeDesktopControlProfile(
                    profile_id="synthetic-previous-control",
                    artifact_sha256=previous_manifest.artifact_sha256,
                    executor_id=previous_executor.executor_id,
                    operator_placements=placements(previous_manifest),
                    maturity="QUALIFIED",
                    evidence_ids=("synthetic-previous-control",),
                ),
            ),
            transitions=(transition, previous_transition),
        ).to_json())
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        scheduler.register_model_manifest(previous_manifest)
        scheduler.register_model_manifest(target)
        snapshot = runtime_snapshot(
            previous_manifest,
            include_phone=False,
            resident_devices=(),
        )
        states = dict(snapshot.executors)
        states[target_executor.executor_id] = RuntimeExecutorState(
            executor_id=target_executor.executor_id,
            healthy=True,
            ready=False,
            temperature_millic=40_000,
            battery_ppm=900_000,
            free_slots=0,
            busy_until_us=0,
        )
        states[previous_executor.executor_id] = executor_state(
            previous_executor.executor_id
        )
        snapshot = replace(snapshot, executors=states)
        probe = scheduler.generate_automated_candidates(
            request("shared-replacement-probe"),
            target.model_id,
            snapshot,
        ).baseline
        required_by_resource = {}
        weight_by_device = {}
        for demand in probe.plan.memory_demands:
            required_by_resource[demand.resource_id] = (
                required_by_resource.get(demand.resource_id, 0)
                + demand.required_bytes
            )
            if demand.kind == "model_weights":
                weight_by_device[demand.device_id] = demand.required_bytes
        replacement_slack = 4_096
        reclaimable = {
            "host-a": (
                required_by_resource["host-memory"] + replacement_slack
            ),
            "accelerator-b": (
                required_by_resource["gpu-memory"] + replacement_slack
            ),
        }
        self.assertGreater(
            reclaimable["host-a"],
            weight_by_device["host-a"],
        )
        previous_residency = model_residency_observations(
            profile,
            previous_manifest,
            previous_executor.executor_id,
            generation=7,
            reclaimable_bytes_by_device=reclaimable,
        )
        target_cpu_residency = ModelResidencyObservation(
            model_id=target.model_id,
            artifact_sha256=target.artifact_sha256,
            device_id="host-a",
            state="hot",
            resident_tensor_ids=tuple(
                tensor.tensor_id for tensor in target.tensors
            ),
            resident_bytes=target.tensor_bytes,
            generation=1,
            executor_id=host.executor_id,
        )
        snapshot = replace(
            snapshot,
            memory=replace(
                snapshot.memory,
                capacities={
                    "host-memory": DeviceMemoryCapacity(
                        "host-memory",
                        reclaimable["host-a"],
                        reclaimable["host-a"],
                        0,
                    ),
                    "gpu-memory": DeviceMemoryCapacity(
                        "gpu-memory",
                        reclaimable["accelerator-b"],
                        reclaimable["accelerator-b"],
                        0,
                    ),
                    "phone-memory": snapshot.memory.capacities[
                        "phone-memory"
                    ],
                },
            ),
            residency=previous_residency + (target_cpu_residency,),
        )
        queued = scheduler.submit_automated_request(
            request("shared-replacement-target"),
            target.model_id,
            snapshot,
            selection_mode="desktop-baseline",
        )
        self.assertEqual(
            {row.device_id for row in queued.execution_plan.transitions[0].evictions},
            {"host-a", "accelerator-b"},
        )

        validation = (
            "research_dev.scheduler._internal.runtime_residency_projection."
            "_validate_projected_evictions"
        )
        with mock.patch(
            validation,
            wraps=__import__(
                "research_dev.scheduler._internal."
                "runtime_residency_projection",
                fromlist=("_validate_projected_evictions",),
            )._validate_projected_evictions,
        ) as validate:
            projected = project_scheduler_residency(
                snapshot,
                profile,
                (queued,),
                {
                    previous_manifest.model_id: previous_manifest,
                    target.model_id: target,
                },
            )
        self.assertEqual(validate.call_count, 1)
        self.assertEqual(
            {
                row.device_id
                for row in projected.residency
                if row.artifact_sha256 == target.artifact_sha256
            },
            {"host-a", "accelerator-b"},
        )
        self.assertEqual(
            projected.memory.capacities["host-memory"].occupied_bytes,
            required_by_resource["host-memory"],
        )
        self.assertEqual(
            projected.memory.capacities["gpu-memory"].occupied_bytes,
            required_by_resource["gpu-memory"],
        )
        persistent_previous_cpu = ModelResidencyObservation(
            model_id=previous_manifest.model_id,
            artifact_sha256=previous_manifest.artifact_sha256,
            device_id="host-a",
            state="hot",
            resident_tensor_ids=tuple(
                tensor.tensor_id for tensor in previous_manifest.tensors
            ),
            resident_bytes=previous_manifest.tensor_bytes,
            generation=1,
            executor_id=host.executor_id,
        )
        observed_target = replace(
            projected,
            snapshot_id="synthetic-shared-replacement-observed-target",
            captured_at_us=500,
            memory=replace(
                projected.memory,
                snapshot_id=(
                    "synthetic-shared-replacement-observed-target-memory"
                ),
                captured_at_us=500,
            ),
            residency=tuple(
                row for row in projected.residency
                if row.artifact_sha256 == target.artifact_sha256
            ) + (persistent_previous_cpu,),
        )
        observed_projection = project_scheduler_residency(
            observed_target,
            profile,
            (queued,),
            {
                previous_manifest.model_id: previous_manifest,
                target.model_id: target,
            },
        )
        self.assertEqual(
            {
                row.device_id
                for row in observed_projection.residency
                if row.artifact_sha256 == target.artifact_sha256
            },
            {"host-a", "accelerator-b"},
        )
        mapped_snapshot = replace(
            snapshot,
            snapshot_id="synthetic-shared-replacement-mapped-host",
            captured_at_us=900,
            memory=replace(
                snapshot.memory,
                snapshot_id=(
                    "synthetic-shared-replacement-mapped-host-memory"
                ),
                captured_at_us=900,
                capacities={
                    **dict(snapshot.memory.capacities),
                    "host-memory": DeviceMemoryCapacity(
                        "host-memory",
                        snapshot.memory.capacities[
                            "host-memory"
                        ].capacity_bytes,
                        replacement_slack,
                        0,
                    ),
                },
            ),
        )
        with mock.patch(
            validation,
            wraps=__import__(
                "research_dev.scheduler._internal."
                "runtime_residency_projection",
                fromlist=("_validate_projected_evictions",),
            )._validate_projected_evictions,
        ) as validate_mapped:
            projected_mapped = project_scheduler_residency(
                mapped_snapshot,
                profile,
                (queued,),
                {
                    previous_manifest.model_id: previous_manifest,
                    target.model_id: target,
                },
            )
        self.assertEqual(validate_mapped.call_count, 1)
        self.assertEqual(
            projected_mapped.memory.capacities[
                "host-memory"
            ].occupied_bytes,
            replacement_slack
                + required_by_resource["host-memory"],
        )
        following = scheduler.submit_automated_request(
                request("shared-replacement-following", arrival_us=1_100),
                previous_manifest.model_id,
                mapped_snapshot,
                selection_mode="desktop-baseline",
        )
        self.assertLessEqual(following.attempt_index, 1)
        self.assertEqual(
            following.model.artifact_sha256,
            previous_manifest.artifact_sha256,
        )

    def test_completed_transition_does_not_duplicate_observed_generation(
        self,
    ) -> None:
        source = catalog()
        gpu = replace(
            source.executor_by_device["accelerator-b"],
            qualified_fallback=True,
            exclusive_residency_resource_id="compute:accelerator-b",
        )
        transition = replace(
            next(
                row for row in source.transitions
                if row.device_id == "accelerator-b"
            ),
            executor_id=gpu.executor_id,
            prepares_device_ids=("accelerator-b",),
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=(gpu,),
            transitions=(transition,),
        ).to_json())
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        first_path = Path(self.directory.name) / "observed-first.gguf"
        write_synthetic_gguf(first_path, sliding_window=64)
        first_model = scheduler.register_gguf_model(
            "observed-first", first_path
        )
        second_model = scheduler.register_gguf_model(
            "observed-second", self.path
        )
        cold = replace(
            runtime_snapshot(
                first_model,
                include_phone=False,
                resident_devices=(),
            ),
            executors={
                "executor:accelerator-b": executor_state(
                    "executor:accelerator-b"
                )
            },
        )
        first = scheduler.submit_automated_request(
            request("observed-first", arrival_us=1_000),
            first_model.model_id,
            cold,
            selection_mode="desktop-baseline",
        )
        first = scheduler.wait_runtime_request(
            first.request.request_id,
            time.monotonic_ns() - first.decision.start_us * 1_000,
        )
        first = scheduler.record_automated_transition_receipts(
            first.request.request_id,
            FakeAutomatedPhysicalAdapter._transition_receipts(first),
        )
        self.assertEqual(first.transition_status, "COMPLETED")

        actual = replace(
            cold,
            snapshot_id="synthetic-observed-transition",
            captured_at_us=1_100,
            memory=replace(
                cold.memory,
                snapshot_id="synthetic-observed-memory",
                captured_at_us=1_100,
            ),
            residency=(ModelResidencyObservation(
                model_id=first_model.model_id,
                artifact_sha256=first_model.artifact_sha256,
                device_id="accelerator-b",
                state="hot",
                resident_tensor_ids=tuple(
                    row.tensor_id for row in first_model.tensors
                ),
                resident_bytes=first_model.tensor_bytes,
                generation=7,
            ),),
        )
        second = scheduler.submit_automated_request(
            request("observed-second", arrival_us=1_100),
            second_model.model_id,
            actual,
            selection_mode="desktop-baseline",
        )

        eviction = second.execution_plan.transitions[0].evictions[0]
        self.assertEqual(eviction.artifact_sha256, first_model.artifact_sha256)
        self.assertEqual(eviction.generation, 7)

    def test_exclusive_memory_conflict_rejects_only_affected_candidate(
        self,
    ) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        ))
        snapshot = runtime_snapshot(
            manifest, phone_bandwidth=8_000_000_000
        )
        original_preview = scheduler._runtime_memory.preview

        def preview(demands, memory_snapshot, **interval):
            if any(row.resource_id == "gpu-memory" for row in demands):
                raise RuntimeResourceError(
                    "exclusive memory replacement overlaps: gpu-memory"
                )
            return original_preview(
                demands, memory_snapshot, **interval
            )

        with mock.patch.object(
            scheduler._runtime_memory, "preview", side_effect=preview
        ):
            ticket = scheduler.submit_automated_request(
                request("exclusive-memory-conflict"),
                manifest.model_id,
                snapshot,
                selection_mode="energy-aware",
            )

        self.assertNotIn(
            "accelerator-b", ticket.execution_plan.device_ids
        )
        self.assertTrue(any(
            reason == "MEMORY_REPLACEMENT_CONFLICT_CURRENT:gpu-memory"
            for _, reason in ticket.decision.rejected
        ))

    def test_residency_fixed_point_serializes_candidates_once(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        snapshot = runtime_snapshot(manifest)
        resource_id = next(iter(snapshot.memory.capacities))
        capacity = snapshot.memory.capacities[resource_id]
        projected = replace(
            snapshot,
            memory=replace(
                snapshot.memory,
                capacities={
                    **snapshot.memory.capacities,
                    resource_id: replace(
                        capacity,
                        occupied_bytes=capacity.occupied_bytes + 1,
                    ),
                },
            ),
        )
        convert = candidate_set_to_runtime_costs
        with mock.patch.object(
            scheduler,
            "_automated_snapshot_for_request",
            side_effect=(snapshot, projected, projected),
        ), mock.patch(
            "research_dev.scheduler._unified.automated_requests_ops.selection."
            "candidate_set_to_runtime_costs",
            wraps=convert,
        ) as conversion:
            ticket = scheduler.submit_automated_request(
                request("fixed-point-one-conversion"),
                manifest.model_id,
                snapshot,
            )

        self.assertEqual(conversion.call_count, 1)
        timing = scheduler.runtime_decision_timings()[-1]
        self.assertEqual(timing["residency_fixed_point_iterations"], 2)
        self.assertEqual(
            len(ticket.cost_estimates.estimates),
            len(ticket.executor_bindings),
        )

    def test_cached_memory_demands_recompute_live_residency(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        cold = scheduler.generate_automated_candidates(
            request("cached-memory-cold"),
            manifest.model_id,
            runtime_snapshot(manifest, resident_devices=()),
        )
        hot = scheduler.generate_automated_candidates(
            request("cached-memory-hot"),
            manifest.model_id,
            runtime_snapshot(
                manifest, resident_devices=("accelerator-b",)
            ),
        )

        def gpu_weights(candidates):
            route = next(
                row for row in candidates.candidates
                if row.route_family == "whole_model"
                and row.device_ids == ("accelerator-b",)
            )
            return next(
                row for row in route.plan.memory_demands
                if row.kind == "model_weights"
            )

        cold_weights = gpu_weights(cold)
        hot_weights = gpu_weights(hot)
        self.assertEqual(cold_weights.resident_bytes, 0)
        self.assertEqual(
            hot_weights.resident_bytes, hot_weights.required_bytes
        )


if __name__ == "__main__":
    unittest.main()
