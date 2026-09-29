"""Automated runtime: admission, caches, marginal cost, snapshots and determinism.

Split from test_automated_runtime.py on 2026-09-13; fixtures and the base class stay there."""

from __future__ import annotations

import unittest
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from unittest import mock
from research_dev.scheduler import (
    DeviceMemoryCapacity,
    HeterogeneousRuntimeSnapshot,
    ModelResidencyObservation,
    ResourceProfile,
    RuntimeCapabilityCatalog,
    RuntimeCompositeExecutorCapability,
    RuntimeCompositeOperatorPlacement,
    RuntimeDesktopControlProfile,
    RuntimeExecutorCapability,
    RuntimeExecutorState,
    RuntimeExecutionReceipt,
    RuntimeRouteShapeProfile,
    RuntimeTransitionCapability,
    UnifiedScheduleError,
    UnifiedScheduler,
)
from research_dev.scheduler._internal.model_manifest import ModelManifestError
from research_dev.scheduler._internal.runtime_learning import RuntimeRouteObservationStore
from research_dev.scheduler._internal.runtime_controller import RuntimeReplanRetryRequired
from research_dev.scheduler._internal.runtime_residency_projection import RuntimeResidencyProjectionError
from research_dev.scheduler._internal.types import canonical_sha256
from research_dev.scheduler.adapters import (
    EndpointRuntimeSample,
    ExecutorResidencySample,
    materialize_desktop_control,
    RuntimePhysicalTopology,
    UnifiedRuntimeSnapshotBuilder,
    model_residency_observations,
    validate_decision_candidate_coverage,
)

try:
    from .test_automated_runtime import (
        AutomatedRuntimeTests,
        FakeAutomatedPhysicalAdapter,
        capability,
        catalog,
        executor_state,
        protected_snapshot,
        request,
        runtime_snapshot,
        system_cost_profile,
        write_synthetic_gguf,
    )
except ImportError:  # run from the tests directory
    from test_automated_runtime import (
        AutomatedRuntimeTests,
        FakeAutomatedPhysicalAdapter,
        capability,
        catalog,
        executor_state,
        protected_snapshot,
        request,
        runtime_snapshot,
        system_cost_profile,
        write_synthetic_gguf,
    )


class AutomatedRuntimeAdmissionTests(AutomatedRuntimeTests):
    """Automated runtime: admission, caches, marginal cost, snapshots and determinism."""

    def test_stat_bound_manifest_cache_skips_unchanged_gguf_reparse(
        self,
    ) -> None:
        cache_path = Path(self.directory.name) / "manifest-cache.json"
        first = UnifiedScheduler.for_runtime_discovery("enforce")
        expected = first.register_gguf_model(
            "cached-model", self.path, cache_path=cache_path
        )
        self.assertTrue(cache_path.is_file())

        second = UnifiedScheduler.for_runtime_discovery("enforce")
        with mock.patch(
            "research_dev.scheduler._internal.model_manifest_cache."
            "GGUFModelManifestLoader.load",
            side_effect=AssertionError("unexpected GGUF reparse"),
        ):
            actual = second.register_gguf_model(
                "cached-model", self.path, cache_path=cache_path
            )
        self.assertEqual(actual, expected)

        with self.path.open("ab") as stream:
            stream.write(b"changed")
        third = UnifiedScheduler.for_runtime_discovery("enforce")
        with mock.patch(
            "research_dev.scheduler._internal.model_manifest_cache."
            "GGUFModelManifestLoader.load",
            side_effect=ModelManifestError("reparse required"),
        ), self.assertRaisesRegex(UnifiedScheduleError, "reparse required"):
            third.register_gguf_model(
                "cached-model", self.path, cache_path=cache_path
            )

    def test_runtime_executor_overlay_merge_preserves_structure(self) -> None:
        base = replace(
            capability("accelerator-b", "gpu"),
            endpoint="physical://accelerator-b",
            backend="gpu-probe",
            exclusive_residency_resource_id="compute:accelerator-b",
            evidence_ids=("base-evidence",),
        )
        overlay = replace(
            capability("accelerator-b", "gpu"),
            endpoint="http://127.0.0.1:19001",
            backend="cuda-http",
            exclusive_residency_resource_id=None,
            evidence_ids=("overlay-evidence",),
            adapter_parameters={"parallel": 2},
        )

        merged = base.with_runtime_overlay(overlay)

        self.assertEqual(merged.endpoint, overlay.endpoint)
        self.assertEqual(merged.backend, overlay.backend)
        self.assertEqual(
            merged.exclusive_residency_resource_id,
            "compute:accelerator-b",
        )
        self.assertEqual(
            merged.evidence_ids,
            ("base-evidence", "overlay-evidence"),
        )
        self.assertEqual(dict(merged.adapter_parameters), {"parallel": 2})
        self.assertEqual(
            RuntimeExecutorCapability.from_json(merged.to_json()), merged
        )

    def test_coordinated_offload_uses_microbatch_usb_transfers(self) -> None:
        source = catalog(
            phone_ops_per_s=10_000_000,
            phone_power_mw=500,
            phone_bandwidth=8_000_000_000,
            phone_whole_model=False,
        )
        path = Path(self.directory.name) / "forty-block-model.gguf"
        write_synthetic_gguf(path, block_count=40)
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(source)
        manifest = scheduler.register_gguf_model("forty-block-model", path)
        placements = tuple(
            RuntimeCompositeOperatorPlacement(
                operator_id=operator.operator_id,
                primary_device_id=(
                    "accelerator-b"
                    if operator.kind == "lm_head" else "host-a"
                ),
                helper_device_id=None,
                split_axis="none",
                split_fraction_ppm=0,
                assisted=False,
            )
            for operator in manifest.operators
        )
        baseline = RuntimeCompositeExecutorCapability(
            executor_id="coordinator:desktop-baseline",
            endpoint="synthetic://desktop-baseline",
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
            residency_states=("hot",),
            resource_ids=("compute:host-a", "compute:accelerator-b"),
            operator_plan_protocol="synthetic-desktop-v1",
            maturity="QUALIFIED",
            evidence_ids=("synthetic-desktop-baseline",),
            artifact_sha256=manifest.artifact_sha256,
            operator_placements=placements,
        )
        ffn_ids = tuple(
            operator.operator_id for operator in manifest.operators
            if operator.kind == "ffn"
        )
        ffn_bytes = {
            operator.operator_id: sum(
                manifest.tensor_by_id[tensor_id].nbytes
                for tensor_id in operator.tensor_ids
            )
            for operator in manifest.operators
            if operator.kind == "ffn"
        }
        helper_limit = sum(ffn_bytes[operator_id] for operator_id in ffn_ids[:5])
        offload = RuntimeCompositeExecutorCapability(
            executor_id="coordinator:phone-offload",
            endpoint="synthetic://phone-offload",
            backend="backend:phone-offload",
            coordinator_device_id="host-a",
            participant_device_ids=(
                "host-a", "accelerator-b", "helper-c"
            ),
            participant_resource_ids={
                "host-a": ("compute:host-a",),
                "accelerator-b": ("compute:accelerator-b",),
                "helper-c": ("compute:helper-c",),
            },
            route_family="operator_offload",
            assisted_operator_kind="ffn",
            split_axis="none",
            split_fractions_ppm=(),
            layer_fractions_ppm=(),
            residency_states=("hot",),
            resource_ids=(
                "compute:host-a",
                "compute:accelerator-b",
                "compute:helper-c",
            ),
            operator_plan_protocol="synthetic-offload-v1",
            maturity="SHADOW",
            evidence_ids=("synthetic-phone-offload",),
            artifact_sha256=manifest.artifact_sha256,
            baseline_executor_id=baseline.executor_id,
            helper_device_id="helper-c",
            operator_ids=ffn_ids,
            adapter_parameters={
                "ffn_column_quantum": 1,
                "ffn_weight_buffer_layout": "selected-width",
                "maximum_helper_resident_weight_bytes": helper_limit,
                "phone_device_id": "helper-c",
                "ubatch_size": 4,
            },
        )
        shadow_profile = RuntimeRouteShapeProfile(
            selector_id="synthetic-phone-offload-shadow",
            artifact_sha256=manifest.artifact_sha256,
            route_family="operator_offload",
            device_ids=("host-a", "accelerator-b", "helper-c"),
            assisted_operator_kind="ffn",
            split_axis="none",
            split_fraction_ppm=0,
            residency_variant="hot",
            minimum_input_tokens=1,
            maximum_input_tokens=64,
            minimum_output_tokens=1,
            maximum_output_tokens=64,
            service_fixed_us=1_000,
            service_input_token_us=100,
            service_output_token_us=100,
            service_upper_add_us=1_000,
            energy_fixed_uj=1_000,
            energy_input_token_uj=10,
            energy_output_token_uj=10,
            energy_lower_error_ppm=100_000,
            energy_upper_error_ppm=100_000,
            sample_count=1,
            maturity="SHADOW",
            evidence_ids=("synthetic-phone-offload-shadow",),
            executor_id=offload.executor_id,
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            composite_executors=(baseline, offload),
            desktop_control_profiles=(RuntimeDesktopControlProfile(
                profile_id="synthetic-coordinated-desktop-control",
                artifact_sha256=manifest.artifact_sha256,
                executor_id=baseline.executor_id,
                operator_placements=placements,
                maturity="QUALIFIED",
                evidence_ids=("synthetic-desktop-baseline",),
            ),),
            route_shape_profiles=(shadow_profile,),
        ).to_json())
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        scheduler.register_model_manifest(manifest)
        snapshot = runtime_snapshot(
            manifest, phone_bandwidth=8_000_000_000
        )
        snapshot = replace(
            snapshot,
            executors={
                **snapshot.executors,
                baseline.executor_id: executor_state(baseline.executor_id),
                offload.executor_id: executor_state(offload.executor_id),
            },
        )

        candidates = scheduler.generate_automated_candidates(
            request(
                "coordinated-offload-transfer",
                input_tokens=9,
                output_tokens=3,
            ),
            manifest.model_id,
            snapshot,
        )
        calibration_request = request(
            "coordinated-offload-cold-calibration",
            input_tokens=9,
            output_tokens=3,
        )
        cold_candidates = replace(
            candidates,
            candidates=tuple(
                replace(
                    item,
                    rejection_reasons=tuple(sorted({
                        *item.rejection_reasons,
                        "COLD_RESIDENCY_BREAK_EVEN",
                    })),
                )
                if item.binding.executor_id == offload.executor_id
                else item
                for item in candidates.candidates
            ),
        )
        cold_selected, _, _ = scheduler._select_automated_candidate(
            cold_candidates,
            calibration_request,
            selection_mode="calibration",
        )
        self.assertEqual(cold_selected.binding.executor_id, offload.executor_id)
        rows = tuple(
            item for item in candidates.candidates
            if item.binding.executor_id == offload.executor_id
        )
        assisted_counts = {
            sum(
                operator.device_ids == ("helper-c",)
                for operator in item.plan.operators
                if operator.operator_kind == "ffn"
            )
            for item in rows
        }
        self.assertGreater(candidates.search_metadata["rough_plan_count"], 32)
        self.assertLessEqual(
            candidates.search_metadata["evaluated_plan_count"], 32
        )
        self.assertTrue({1, 5}.issubset(assisted_counts), assisted_counts)
        self.assertEqual(
            candidates.baseline.binding.executor_id,
            baseline.executor_id,
        )
        row = next(
            item for item in rows
            if sum(
                operator.device_ids == ("helper-c",)
                for operator in item.plan.operators
                if operator.operator_kind == "ffn"
            ) == 5
        )

        self.assertNotIn("PLACEMENT_INFEASIBLE", row.rejection_reasons)
        self.assertIsNotNone(row.paired_baseline_route_id)
        parent = next(
            item for item in candidates.candidates
            if item.candidate_id == row.paired_baseline_route_id
        )
        self.assertEqual(parent.binding.executor_id, baseline.executor_id)
        self.assertEqual(
            row.plan.desktop_placement_sha256,
            parent.plan.desktop_placement_sha256,
        )
        self.assertGreater(row.cost.transfer_us, 0)
        assisted = tuple(
            item for item in row.plan.operators
            if item.operator_kind == "ffn"
            and item.device_ids == ("helper-c",)
        )
        self.assertEqual(len(assisted), 5)
        self.assertTrue(all(
            item.device_ids == ("helper-c",)
            and item.split_axis == "none"
            and item.split_fraction_ppm == 0
            for item in assisted
        ))
        ticket = scheduler.submit_automated_request(
            request(
                "coordinated-offload-calibration",
                input_tokens=9,
                output_tokens=3,
            ),
            manifest.model_id,
            snapshot,
            selection_mode="calibration",
        )
        self.assertEqual(ticket.binding.executor_id, offload.executor_id)
        self.assertEqual(
            ticket.decision.reason, "PHYSICAL_CALIBRATION_SAMPLE"
        )
        self.assertEqual(sum(
            assignment.device_ids == ("helper-c",)
            for assignment in ticket.execution_plan.operators
            if assignment.operator_kind == "ffn"
        ), 5, tuple(
            (
                sum(
                    assignment.device_ids == ("helper-c",)
                    for assignment in candidate.plan.operators
                    if assignment.operator_kind == "ffn"
                ),
                sum(
                    demand.required_bytes
                    for demand in candidate.plan.memory_demands
                    if demand.kind == "model_weights"
                    and demand.device_id == "helper-c"
                ),
                candidate.cost.finish_upper_us,
                candidate.rejection_reasons,
            )
            for candidate in rows
        ))
        selected_cost = next(
            item for item in ticket.cost_estimates.estimates
            if item.route_id == ticket.decision.route_id
        )
        self.assertEqual(selected_cost.reason, "CALIBRATION_ADMITTED")
        self.assertIn(
            "ROUTE_NOT_QUALIFIED",
            selected_cost.details[
                "calibration_original_rejection_reasons"
            ],
        )
        self.assertEqual(
            ticket.execution_plan.adapter_parameters[
                "ffn_column_quantum"
            ],
            manifest.feed_forward_length,
        )
        coverage = validate_decision_candidate_coverage(
            profile,
            scheduler.runtime_decision_log()["records"],
            {ticket.request.request_id: manifest},
        )
        self.assertEqual(len(coverage), 1)

    def test_arrivals_admit_behind_active_partial_residency_replacement(
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
        source_path = Path(self.directory.name) / "partial-source.gguf"
        write_synthetic_gguf(source_path, block_count=2, sliding_window=32)
        source_model = scheduler.register_gguf_model(
            "partial-source-model", source_path
        )
        target_model = scheduler.register_gguf_model(
            "partial-target-model", self.path
        )
        hot = replace(
            runtime_snapshot(
                source_model,
                include_phone=False,
                resident_devices=("accelerator-b",),
            ),
            executors={
                gpu.executor_id: executor_state(gpu.executor_id)
            },
        )
        resident = replace(
            hot.residency[0],
            executor_id=gpu.executor_id,
            reclaimable_bytes=hot.residency[0].resident_bytes,
        )
        hot = replace(hot, residency=(resident,))
        active = scheduler.submit_automated_request(
            request("partial-active", arrival_us=1_000),
            target_model.model_id,
            hot,
            selection_mode="desktop-baseline",
        )
        active = scheduler.wait_runtime_request(
            active.request.request_id,
            time.monotonic_ns() - active.decision.start_us * 1_000,
        )
        self.assertEqual(active.dispatch_state, "ACQUIRED")
        self.assertTrue(active.execution_plan.transitions)
        observed_at_us = active.decision.start_us + 1
        partial = replace(
            hot,
            snapshot_id="active-partial-replacement",
            captured_at_us=observed_at_us,
            valid_until_us=observed_at_us + 10_000_000,
            memory=replace(
                hot.memory,
                snapshot_id="active-partial-replacement-memory",
                captured_at_us=observed_at_us,
                valid_until_us=observed_at_us + 10_000_000,
            ),
            residency=(replace(
                resident,
                executor_id="intermediate-source-executor",
                generation=resident.generation + 1,
            ),),
        )

        followers = tuple(
            scheduler.submit_automated_request(
                request(
                    "partial-follower-" + str(index),
                    arrival_us=observed_at_us,
                ),
                target_model.model_id,
                partial,
                observed_at_us=observed_at_us,
                selection_mode="desktop-baseline",
            )
            for index in range(4)
        )

        active_end_us = max(
            row.reserved_until_us for row in active.decision.leases
        )
        self.assertTrue(all(
            follower.decision.start_us >= active_end_us
            for follower in followers
        ))
        queue = scheduler.runtime_controller_snapshot()["dispatch_queue"]
        self.assertTrue(all(
            active.request.request_id
                in queue["causal_predecessors"][follower.request.request_id]
            for follower in followers
        ))

    def test_staged_capacity_release_reprices_cold_successor_now(
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
            resource_slots={"compute:accelerator-b": 8},
        )
        resources = dict(source.resources)
        resources["compute:accelerator-b"] = replace(
            resources["compute:accelerator-b"], capacity=8
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            resources=resources,
            executors=(gpu,),
            composite_executors=(),
            transitions=(transition,),
        ).to_json())
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        other_path = Path(self.directory.name) / "released-other.gguf"
        write_synthetic_gguf(other_path, block_count=2, sliding_window=32)
        first_model = scheduler.register_gguf_model(
            "released-first", self.path
        )
        second_model = scheduler.register_gguf_model(
            "released-other", other_path
        )
        cold = replace(
            runtime_snapshot(
                first_model,
                include_phone=False,
                gpu_free_slots=8,
                resident_devices=(),
            ),
            executors={
                gpu.executor_id: executor_state(gpu.executor_id)
            },
        )
        first = scheduler.submit_automated_request(
            request("released-first", arrival_us=1_000),
            first_model.model_id,
            cold,
            selection_mode="desktop-baseline",
        )
        second = scheduler.submit_automated_request(
            request("released-second", arrival_us=1_001),
            second_model.model_id,
            cold,
            selection_mode="desktop-baseline",
        )
        self.assertTrue(
            scheduler.runtime_controller_snapshot()["dispatch_queue"]
            ["entry_states"][second.request.request_id]
            ["residency_transition_barrier"]
        )
        followers = tuple(
            scheduler.submit_automated_request(
                request(
                    "released-follower-" + str(index),
                    arrival_us=1_002 + index,
                ),
                second_model.model_id,
                cold,
                selection_mode="desktop-baseline",
            )
            for index in range(12)
        )
        active = scheduler.wait_runtime_request(
            first.request.request_id,
            time.monotonic_ns() - first.decision.start_us * 1_000,
        )
        scheduler.record_automated_transition_receipts(
            first.request.request_id,
            FakeAutomatedPhysicalAdapter._transition_receipts(active),
        )
        completed_at_us = first.decision.start_us + 2
        resident_bytes = sum(
            demand.required_bytes
            for demand in first.execution_plan.memory_demands
            if demand.resource_id == "gpu-memory"
        )
        hot = replace(
            cold,
            snapshot_id="released-successor-hot",
            captured_at_us=completed_at_us - 1,
            valid_until_us=completed_at_us + 10_000_000,
            memory=replace(
                cold.memory,
                snapshot_id="released-successor-hot-memory",
                captured_at_us=completed_at_us - 1,
                valid_until_us=completed_at_us + 10_000_000,
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
        later_hot = scheduler.submit_automated_request(
            request("released-later-hot", arrival_us=completed_at_us - 1),
            first_model.model_id,
            hot,
            observed_at_us=completed_at_us - 1,
            selection_mode="desktop-baseline",
        )
        self.assertTrue(
            scheduler.runtime_controller_snapshot()["dispatch_queue"]
            ["entry_states"][later_hot.request.request_id]
            ["residency_transition_barrier"]
        )
        predecessors = scheduler.runtime_controller_snapshot()[
            "dispatch_queue"
        ]["causal_predecessors"]
        self.assertIn(
            second.request.request_id,
            predecessors[later_hot.request.request_id],
        )

        scheduler.release_automated_runtime_capacity(
            first.request.request_id,
            completed_at_us,
            expected_ticket_id=first.ticket_id,
        )
        wake = scheduler.wait_runtime_request(
            second.request.request_id,
            time.monotonic_ns() - completed_at_us * 1_000,
        )
        self.assertEqual(wake.dispatch_state, "REPLAN_REQUIRED")
        self.assertTrue(all(
            scheduler.runtime_ticket(ticket.request.request_id).lease_status
                == "CANCELLED"
            for ticket in followers
        ))

        replanned = scheduler.replan_automated_request(
            second.request.request_id,
            observed_at_us=completed_at_us,
            reason=wake.dispatch_receipt.wake_reason,
            snapshot=hot,
            expected_ticket_id=wake.ticket_id,
            expected_queue_generation=(
                wake.dispatch_receipt.queue_generation
            ),
        )

        self.assertEqual(replanned.attempt_index, 1)
        self.assertEqual(replanned.decision.start_us, completed_at_us)
        self.assertTrue(replanned.execution_plan.transitions)
        self.assertTrue(all(
            scheduler.runtime_ticket(ticket.request.request_id).lease_status
                == "CANCELLED"
            for ticket in followers
        ))
        self.assertTrue(all(
            scheduler.runtime_controller_snapshot()["dispatch_queue"]
                ["entry_states"][ticket.request.request_id]["state"]
                == "DEFERRED_REPLAN"
            for ticket in followers
        ))
        self.assertEqual(
            scheduler.runtime_ticket(first.request.request_id).lease_status,
            "RELEASED_PENDING_RECEIPT",
        )

    def test_priority_compaction_preserves_projection_deferral(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        hot = runtime_snapshot(
            manifest,
            include_phone=False,
            resident_devices=("host-a", "accelerator-b"),
        )
        tickets = tuple(
            scheduler.submit_automated_request(
                request("compaction-deferred-" + str(index),
                        arrival_us=1_000 + index),
                manifest.model_id,
                hot,
                selection_mode="desktop-baseline",
            )
            for index in range(3)
        )
        first = scheduler.wait_runtime_request(
            tickets[0].request.request_id,
            time.monotonic_ns() - tickets[0].decision.start_us * 1_000,
        )
        observed_at_us = max(
            first.decision.start_us + 1, tickets[-1].request.arrival_us
        )
        root_id = tickets[1].request.request_id
        follower_id = tickets[2].request.request_id
        self.assertTrue(scheduler._runtime_controller.require_queued_replan(
            root_id, "capacity_released_early"
        ))
        root = scheduler.wait_runtime_request(
            root_id, time.monotonic_ns() - observed_at_us * 1_000
        )
        with mock.patch.object(
            scheduler._runtime_controller.queue,
            "priority_compaction_followers",
            return_value=(follower_id,),
        ), mock.patch.object(
            scheduler,
            "_prepare_automated_replan",
            side_effect=RuntimeResidencyProjectionError(
                "active predecessor residency changed",
                request_id=first.request.request_id,
                ticket_id=first.ticket_id,
            ),
        ):
            deferred = scheduler.replan_automated_request(
                root_id,
                observed_at_us=observed_at_us,
                reason=root.dispatch_receipt.wake_reason,
                snapshot=hot,
                expected_ticket_id=root.ticket_id,
                expected_queue_generation=root.dispatch_receipt.queue_generation,
            )
        self.assertEqual(deferred.dispatch_state, "REPLAN_REQUIRED")
        queue = scheduler.runtime_controller_snapshot()["dispatch_queue"]
        for request_id in (root_id, follower_id):
            self.assertEqual(
                queue["entry_states"][request_id]["state"], "DEFERRED_REPLAN"
            )
            self.assertEqual(
                scheduler.runtime_ticket(request_id).lease_status, "CANCELLED"
            )
            self.assertEqual(scheduler._runtime_memory.owner_tokens(request_id), ())
        self.assertEqual(scheduler.runtime_ticket(first.request.request_id), first)
        scheduler.complete_automated_request(
            first.request.request_id,
            self.execution_receipt(first, finished_us=observed_at_us),
        )
        for request_id in (root_id, follower_id):
            ready = scheduler.wait_runtime_request(
                request_id, time.monotonic_ns() - observed_at_us * 1_000
            )
            if ready.dispatch_state == "REPLAN_REQUIRED":
                ready = scheduler.replan_automated_request(
                    request_id,
                    observed_at_us=observed_at_us,
                    reason=ready.dispatch_receipt.wake_reason,
                    snapshot=hot,
                    expected_ticket_id=ready.ticket_id,
                    expected_queue_generation=ready.dispatch_receipt.queue_generation,
                )
                ready = scheduler.wait_runtime_request(
                    request_id,
                    time.monotonic_ns() - ready.decision.start_us * 1_000,
                )
            self.assertEqual(ready.dispatch_state, "ACQUIRED")
            observed_at_us = max(observed_at_us, ready.decision.start_us) + 1
            scheduler.complete_automated_request(
                request_id,
                self.execution_receipt(ready, finished_us=observed_at_us),
            )
            self.assertEqual(
                scheduler.runtime_ticket(request_id).dispatch_state, "COMPLETED"
            )

    def test_priority_compaction_rolls_back_root_and_followers_atomically(
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
                    "compaction-rollback-" + str(index),
                    arrival_us=1_000 + index,
                ),
                manifest.model_id,
                hot,
                selection_mode="desktop-baseline",
            )
            for index in range(3)
        )
        first = scheduler.wait_runtime_request(
            tickets[0].request.request_id,
            time.monotonic_ns() - tickets[0].decision.start_us * 1_000,
        )
        completed_at_us = max(
            first.decision.start_us + 1,
            tickets[-1].request.arrival_us,
        )
        scheduler.complete_automated_request(
            first.request.request_id,
            self.execution_receipt(first, finished_us=completed_at_us),
        )
        root = scheduler.wait_runtime_request(
            tickets[1].request.request_id,
            time.monotonic_ns() - completed_at_us * 1_000,
        )
        fresh = replace(
            hot,
            snapshot_id="compaction-rollback-runtime",
            captured_at_us=completed_at_us,
            valid_until_us=completed_at_us + 10_000_000,
            memory=replace(
                hot.memory,
                snapshot_id="compaction-rollback-memory",
                captured_at_us=completed_at_us,
                valid_until_us=completed_at_us + 10_000_000,
            ),
        )
        before_controller = scheduler.runtime_controller_snapshot()
        before_memory = scheduler.runtime_memory_state()
        before_log = scheduler.runtime_decision_log_bytes()
        before_timeline = scheduler.timeline.checkpoint()
        original = scheduler._replan_automated_request_once

        def inject_failure(request_id, **kwargs):
            if kwargs["reason"] == "priority_compaction_follower":
                raise RuntimeError("synthetic follower replacement failure")
            return original(request_id, **kwargs)

        with mock.patch.object(
            scheduler._runtime_controller.queue,
            "priority_compaction_followers",
            return_value=(tickets[2].request.request_id,),
        ), mock.patch.object(
            scheduler,
            "_replan_automated_request_once",
            side_effect=inject_failure,
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

    def test_snapshot_preserves_endpoint_owned_reclaimable_memory(self) -> None:
        profile = catalog()
        _, manifest = self.scheduler_and_manifest(profile)
        allocation_bytes = manifest.tensor_bytes + 4096
        snapshot = UnifiedRuntimeSnapshotBuilder(profile).build(
            snapshot_id="endpoint-allocation",
            captured_at_us=0,
            valid_until_us=10_000_000,
            memory=runtime_snapshot(manifest).memory,
            executor_samples={
                "executor:accelerator-b": EndpointRuntimeSample(
                    "healthy", "live", 1
                ),
            },
            residencies=(ExecutorResidencySample(
                manifest,
                "executor:accelerator-b",
                4,
                reclaimable_bytes_by_device={
                    "accelerator-b": allocation_bytes,
                },
            ),),
        )

        self.assertEqual(len(snapshot.residency), 1)
        observed = snapshot.residency[0]
        self.assertEqual(observed.executor_id, "executor:accelerator-b")
        self.assertEqual(observed.reclaimable_bytes, allocation_bytes)
        self.assertEqual(observed.generation, 4)

    def test_snapshot_omits_undersized_reclaimable_memory_sample(self) -> None:
        profile = catalog()
        _, manifest = self.scheduler_and_manifest(profile)
        snapshot = UnifiedRuntimeSnapshotBuilder(profile).build(
            snapshot_id="undersized-endpoint-allocation",
            captured_at_us=0,
            valid_until_us=10_000_000,
            memory=runtime_snapshot(manifest).memory,
            executor_samples={
                "executor:accelerator-b": EndpointRuntimeSample(
                    "healthy", "live", 1
                ),
            },
            residencies=(ExecutorResidencySample(
                manifest,
                "executor:accelerator-b",
                4,
                reclaimable_bytes_by_device={
                    "accelerator-b": manifest.tensor_bytes - 1,
                },
            ),),
        )

        self.assertEqual(len(snapshot.residency), 1)
        self.assertIsNone(snapshot.residency[0].reclaimable_bytes)

    def test_logical_coordinator_needs_no_interference_measurement(self) -> None:
        source = catalog(system_cost_profiles=(system_cost_profile(
            helper_interference_ppm=100_000,
        ),))
        resources = dict(source.resources)
        resources["coordinator:gpu"] = ResourceProfile(
            resource_id="coordinator:gpu",
            kind="coordinator",
            capacity=1,
            ready=True,
            identity="coordinator:gpu",
        )
        executors = tuple(
            replace(
                row,
                execution_resource_ids=(
                    *row.execution_resource_ids,
                    "coordinator:gpu",
                ),
            ) if row.device_id == "accelerator-b" else row
            for row in source.executors
        )
        scheduler, manifest = self.scheduler_and_manifest(replace(
            source, resources=resources, executors=executors
        ))
        candidates = scheduler.generate_automated_candidates(
            request("logical-coordinator-cost"),
            manifest.model_id,
            protected_snapshot(manifest),
        )
        gpu = next(
            row for row in candidates.candidates
            if row.route_family == "whole_model"
            and row.device_ids == ("accelerator-b",)
            and row.residency_variant == "hot"
        )
        self.assertNotIn(
            "MARGINAL_SYSTEM_COST_UNKNOWN", gpu.rejection_reasons
        )
        self.assertIsNotNone(gpu.marginal_system_cost)

    def test_protected_work_marginal_cost_changes_automated_selection(self) -> None:
        direct, direct_manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        ))
        direct_ticket = direct.submit_automated_request(
            request("direct-cost-only"),
            direct_manifest.model_id,
            runtime_snapshot(
                direct_manifest, phone_bandwidth=8_000_000_000
            ),
        )
        self.assertEqual(
            direct_ticket.execution_plan.device_ids, ("helper-c",)
        )

        protected, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
            system_cost_profiles=(system_cost_profile(
                helper_interference_ppm=100_000_000,
            ),),
        ))
        ticket = protected.submit_automated_request(
            request("protected-cost"),
            manifest.model_id,
            protected_snapshot(manifest),
        )
        self.assertEqual(
            ticket.execution_plan.device_ids, ("accelerator-b",)
        )
        self.assertIsNotNone(ticket.decision.marginal_system_cost)
        self.assertEqual(
            ticket.decision.system_finish_upper_us,
            ticket.decision.marginal_system_cost[
                "system_finish_upper_us"
            ],
        )
        phone = next(
            row for row in ticket.cost_estimates.estimates
            if row.details["device_ids"] == ["helper-c"]
            and row.details["route_family"] == "whole_model"
            and row.details["residency_variant"] == "hot"
        )
        self.assertGreater(
            phone.fleet_energy_upper_uj,
            ticket.decision.energy_upper_uj,
        )
        self.assertEqual(
            phone.details["marginal_system_cost"]["profile_id"],
            "synthetic-protected-work-phase-1",
        )
        marginal = ticket.decision.marginal_system_cost
        # An immediate start overlaps no protected critical path while
        # waiting, so nothing cancels and incremental equals route energy.
        self.assertEqual(marginal["protected_overlap_us"], 0)
        self.assertEqual(marginal["protected_overlap_idle_uj"], 0)
        self.assertEqual(
            marginal["route_energy_incremental_lower_uj"],
            marginal["route_energy_lower_uj"],
        )
        self.assertEqual(
            marginal["route_energy_incremental_upper_uj"],
            marginal["route_energy_upper_uj"],
        )

    def test_protected_work_without_qualified_cost_fails_to_fallback(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        ))
        candidates = scheduler.generate_automated_candidates(
            request("missing-system-cost"),
            manifest.model_id,
            protected_snapshot(manifest),
        )
        self.assertTrue(candidates.baseline.admitted)
        otherwise_ready = [
            row for row in candidates.candidates
            if not row.baseline
            and row.candidate_id
                != candidates.recovery_fallback_route_id
            and set(row.rejection_reasons).issubset({
                "MARGINAL_SYSTEM_COST_UNKNOWN"
            })
        ]
        self.assertTrue(otherwise_ready)
        self.assertTrue(all(
            "MARGINAL_SYSTEM_COST_UNKNOWN" in row.rejection_reasons
            for row in otherwise_ready
        ))
        ticket = scheduler.submit_automated_request(
            request("missing-system-cost"),
            manifest.model_id,
            protected_snapshot(manifest),
        )
        self.assertNotIn(
            "helper-c", ticket.execution_plan.device_ids
        )
        self.assertIn(
            "MARGINAL_SYSTEM_COST_UNKNOWN",
            dict(ticket.decision.rejected)[
                next(
                    row.candidate_id for row in candidates.candidates
                    if row in otherwise_ready
                )
            ],
        )

    def test_expired_protected_work_does_not_require_marginal_profile(
        self,
    ) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        ))
        snapshot = protected_snapshot(manifest)
        snapshot = replace(
            snapshot,
            protected_work=replace(
                snapshot.protected_work,
                critical_path_end_us=0,
            ),
        )

        candidates = scheduler.generate_automated_candidates(
            request("expired-protected-work"),
            manifest.model_id,
            snapshot,
        )

        self.assertTrue(all(
            "MARGINAL_SYSTEM_COST_UNKNOWN" not in row.rejection_reasons
            for row in candidates.candidates
        ))

    def test_unknown_marginal_cost_keeps_one_desktop_baseline(self) -> None:
        shadow = replace(
            system_cost_profile(helper_interference_ppm=0),
            sample_count=1,
            maturity="SHADOW",
        )
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
            system_cost_profiles=(shadow,),
        ))
        snapshot = protected_snapshot(manifest)
        capacities = dict(snapshot.memory.capacities)
        capacities["host-memory"] = DeviceMemoryCapacity(
            "host-memory",
            capacities["host-memory"].capacity_bytes,
            capacities["host-memory"].capacity_bytes,
            0,
        )
        snapshot = replace(
            snapshot,
            memory=replace(
                snapshot.memory,
                snapshot_id="synthetic-shadow-host-full-memory",
                capacities=capacities,
            ),
            residency=tuple(
                row for row in snapshot.residency
                if row.device_id != "host-a"
            ),
        )

        candidates = scheduler.generate_automated_candidates(
            request("shadow-desktop-baseline"),
            manifest.model_id,
            snapshot,
        )

        self.assertEqual(candidates.baseline.device_ids, ("accelerator-b",))
        self.assertTrue(candidates.baseline.admitted)
        self.assertIsNone(candidates.baseline.marginal_system_cost)
        self.assertTrue(any(
            "MARGINAL_SYSTEM_COST_UNKNOWN" in row.rejection_reasons
            for row in candidates.candidates
            if row.candidate_id != candidates.baseline.candidate_id
            and "helper-c" in row.device_ids
        ))

    def test_shadow_cost_keeps_physically_qualified_desktop_baseline(
        self,
    ) -> None:
        artifact_sha256 = "sha256:" + hashlib.sha256(
            self.path.read_bytes()
        ).hexdigest()
        shadow_profiles = tuple(
            RuntimeRouteShapeProfile(
                selector_id="shadow-desktop-" + device_id,
                artifact_sha256=artifact_sha256,
                route_family="whole_model",
                device_ids=(device_id,),
                assisted_operator_kind=None,
                split_axis="none",
                split_fraction_ppm=0,
                residency_variant="hot",
                minimum_input_tokens=1,
                maximum_input_tokens=1_000,
                minimum_output_tokens=1,
                maximum_output_tokens=1_000,
                service_fixed_us=service_us,
                service_input_token_us=0,
                service_output_token_us=0,
                service_upper_add_us=service_us // 2,
                energy_fixed_uj=1,
                energy_input_token_uj=0,
                energy_output_token_uj=0,
                energy_lower_error_ppm=100_000,
                energy_upper_error_ppm=100_000,
                sample_count=1,
                maturity="SHADOW",
                evidence_ids=("diagnostic-desktop-cost",),
            )
            for device_id, service_us in (
                ("host-a", 2_000),
                ("accelerator-b", 1_000),
            )
        )
        source = catalog()
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=tuple(
                replace(
                    row,
                    supports_layer_placement=False,
                    supports_operator_placement=False,
                    supports_split_coordinator=False,
                    supports_split_helper=False,
                    coordinated_route_families=(),
                    operator_plan_protocol=None,
                )
                if row.device_id in {"host-a", "accelerator-b"}
                else row
                for row in source.executors
            ),
            route_shape_profiles=shadow_profiles,
        ).to_json())
        scheduler, manifest = self.scheduler_and_manifest(profile)
        runtime = runtime_snapshot(manifest)

        candidates = scheduler.generate_automated_candidates(
            request("shadow-cost-baseline"), manifest.model_id, runtime
        )

        self.assertEqual(candidates.baseline.device_ids, ("accelerator-b",))
        self.assertEqual(candidates.baseline.maturity, "SHADOW")
        self.assertTrue(candidates.baseline.admitted)
        self.assertTrue(candidates.baseline.binding.ready)
        self.assertIsNotNone(candidates.baseline.cost.fleet_energy_uj)
        self.assertEqual(
            candidates.baseline.cost.energy_evidence, "ASSUMED"
        )
        ticket = scheduler.submit_automated_request(
            request("shadow-cost-baseline"), manifest.model_id, runtime
        )
        self.assertEqual(
            ticket.decision.route_id, candidates.baseline.candidate_id
        )
        self.assertEqual(
            ticket.decision.reason,
            "QUALIFIED_FALLBACK_NO_SAFE_ALTERNATIVE",
        )
        self.assertTrue(any(
            "helper-c" in row.details.get("device_ids", [])
            and dict(ticket.decision.rejected).get(row.route_id)
                == "BASELINE_ENERGY_EVIDENCE_NOT_QUALIFIED"
            for row in ticket.cost_estimates.estimates
        ))

    def test_quantization_block_misaligned_splits_fail_closed(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        candidates = scheduler.generate_automated_candidates(
            request("quantization-block-cuts"),
            manifest.model_id,
            runtime_snapshot(manifest),
        )
        row_splits = [
            row for row in candidates.candidates
            if row.route_family == "operator_split"
            and row.assisted_operator_kind == "ffn"
            and row.split_axis == "row"
        ]
        column_splits = [
            row for row in candidates.candidates
            if row.route_family == "operator_split"
            and row.assisted_operator_kind == "ffn"
            and row.split_axis == "column"
        ]
        self.assertTrue(row_splits)
        self.assertTrue(column_splits)
        self.assertTrue(all(
            "QUANTIZATION_BLOCK_MISALIGNED" in row.rejection_reasons
            for row in row_splits
        ))
        self.assertTrue(any(
            "QUANTIZATION_BLOCK_MISALIGNED" not in row.rejection_reasons
            for row in column_splits
        ))

        base_catalog = catalog()
        misaligned_catalog = replace(
            base_catalog,
            executors=tuple(
                replace(row, split_fractions_ppm=(300_000,))
                for row in base_catalog.executors
            ),
        )
        other_scheduler, other_manifest = self.scheduler_and_manifest(
            misaligned_catalog
        )
        other = other_scheduler.generate_automated_candidates(
            request("misaligned-hidden-width"),
            other_manifest.model_id,
            runtime_snapshot(other_manifest),
        )
        invalid_columns = [
            row for row in other.candidates
            if row.route_family == "operator_split"
            and row.assisted_operator_kind == "ffn"
            and row.split_axis == "column"
        ]
        self.assertTrue(invalid_columns)
        self.assertTrue(all(
            "QUANTIZATION_BLOCK_MISALIGNED" in row.rejection_reasons
            for row in invalid_columns
        ))

    def test_known_busy_executor_is_queued_by_scheduler(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        free = scheduler.generate_automated_candidates(
            request("free-executor"),
            manifest.model_id,
            runtime_snapshot(manifest),
        )
        candidates = scheduler.generate_automated_candidates(
            request("known-busy-executor"),
            manifest.model_id,
            runtime_snapshot(
                manifest,
                gpu_busy_until_us=100_000,
                gpu_free_slots=0,
            ),
        )
        gpu = next(
            row for row in candidates.candidates
            if row.route_family == "whole_model"
            and row.device_ids == ("accelerator-b",)
            and row.residency_variant == "hot"
        )
        self.assertNotIn("EXECUTOR_NOT_READY", gpu.rejection_reasons)
        self.assertGreaterEqual(gpu.cost.start_us, 100_000)
        self.assertGreater(gpu.cost.queue_delay_us, 0)
        free_gpu = next(
            row for row in free.candidates
            if row.route_family == "whole_model"
            and row.device_ids == ("accelerator-b",)
            and row.residency_variant == "hot"
        )
        idle_power_mw = 3 * 500
        expected_queue_energy = (
            idle_power_mw * gpu.cost.queue_delay_us + 999
        ) // 1_000
        self.assertEqual(
            gpu.cost.fleet_energy_uj
            - free_gpu.cost.fleet_energy_uj,
            expected_queue_energy,
        )

    def test_unknown_busy_executor_queues_only_for_scheduler_owned_work(self) -> None:
        profile = catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        )
        scheduler, manifest = self.scheduler_and_manifest(profile)
        first = scheduler.submit_automated_request(
            request("owned-phone-work"),
            manifest.model_id,
            runtime_snapshot(manifest, phone_bandwidth=8_000_000_000),
        )
        self.assertEqual(first.execution_plan.device_ids, ("helper-c",))

        observed = runtime_snapshot(
            manifest, phone_bandwidth=8_000_000_000
        )
        states = dict(observed.executors)
        states["executor:helper-c"] = replace(
            states["executor:helper-c"],
            free_slots=0,
            busy_until_us=0,
        )
        observed = replace(observed, executors=states)
        second_request = request("queued-phone-work", arrival_us=1_001)
        queued = scheduler.generate_automated_candidates(
            second_request,
            manifest.model_id,
            observed,
        )
        phone = next(
            row for row in queued.candidates
            if row.route_family == "whole_model"
            and row.device_ids == ("helper-c",)
            and row.residency_variant == "hot"
        )
        self.assertTrue(phone.admitted)
        self.assertNotIn(
            "EXECUTOR_CAPACITY_UNAVAILABLE", phone.rejection_reasons
        )
        self.assertGreater(phone.cost.queue_delay_us, 0)

        isolated, isolated_manifest = self.scheduler_and_manifest(profile)
        unavailable = isolated.generate_automated_candidates(
            second_request,
            isolated_manifest.model_id,
            observed,
        )
        isolated_phone = next(
            row for row in unavailable.candidates
            if row.route_family == "whole_model"
            and row.device_ids == ("helper-c",)
            and row.residency_variant == "hot"
        )
        self.assertFalse(isolated_phone.admitted)
        self.assertIn(
            "EXECUTOR_CAPACITY_UNAVAILABLE",
            isolated_phone.rejection_reasons,
        )

    def test_not_ready_executor_queues_only_for_scheduler_owned_work(self) -> None:
        profile = catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        )
        scheduler, manifest = self.scheduler_and_manifest(profile)
        first = scheduler.submit_automated_request(
            request("owned-not-ready-phone"),
            manifest.model_id,
            runtime_snapshot(manifest, phone_bandwidth=8_000_000_000),
        )
        self.assertEqual(first.execution_plan.device_ids, ("helper-c",))

        observed = runtime_snapshot(
            manifest, phone_bandwidth=8_000_000_000
        )
        states = dict(observed.executors)
        states["executor:helper-c"] = replace(
            states["executor:helper-c"],
            ready=False,
            free_slots=0,
            busy_until_us=0,
        )
        observed = replace(observed, executors=states)
        second_request = request("queued-not-ready-phone", arrival_us=1_001)
        queued = scheduler.generate_automated_candidates(
            second_request,
            manifest.model_id,
            observed,
        )
        phone = next(
            row for row in queued.candidates
            if row.route_family == "whole_model"
            and row.device_ids == ("helper-c",)
            and row.residency_variant == "hot"
        )
        self.assertTrue(phone.admitted)
        self.assertNotIn("EXECUTOR_NOT_READY", phone.rejection_reasons)
        self.assertGreater(phone.cost.queue_delay_us, 0)

        isolated, isolated_manifest = self.scheduler_and_manifest(profile)
        unavailable = isolated.generate_automated_candidates(
            second_request,
            isolated_manifest.model_id,
            observed,
        )
        isolated_phone = next(
            row for row in unavailable.candidates
            if row.route_family == "whole_model"
            and row.device_ids == ("helper-c",)
            and row.residency_variant == "hot"
        )
        self.assertFalse(isolated_phone.admitted)
        self.assertIn("EXECUTOR_NOT_READY", isolated_phone.rejection_reasons)

    def test_layer_candidates_never_split_one_gguf_layer(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        candidates = scheduler.generate_automated_candidates(
            request("layer-block-integrity"),
            manifest.model_id,
            runtime_snapshot(manifest),
        )
        layer_by_operator = {
            row.operator_id: row.layer_id for row in manifest.operators
        }
        layer_rows = [
            row for row in candidates.candidates
            if row.route_family == "layer_placement"
        ]
        self.assertTrue(layer_rows)
        for candidate in layer_rows:
            devices_by_layer = {}
            for assignment in candidate.plan.operators:
                devices_by_layer.setdefault(
                    layer_by_operator[assignment.operator_id], set()
                ).update(assignment.device_ids)
            self.assertTrue(all(
                len(devices) == 1 for devices in devices_by_layer.values()
            ))

    def test_capability_and_snapshot_json_drive_unseen_devices(self) -> None:
        source_catalog = catalog()
        source_scheduler, manifest = self.scheduler_and_manifest(source_catalog)
        del source_scheduler
        source_snapshot = runtime_snapshot(manifest)
        loaded_catalog = RuntimeCapabilityCatalog.from_json(
            source_catalog.to_json()
        )
        loaded_snapshot = HeterogeneousRuntimeSnapshot.from_json(
            source_snapshot.to_json()
        )
        self.assertEqual(loaded_catalog.to_json(), source_catalog.to_json())
        self.assertEqual(loaded_snapshot.to_json(), source_snapshot.to_json())

        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(loaded_catalog)
        scheduler.register_gguf_model(manifest.model_id, self.path)
        candidates = scheduler.generate_automated_candidates(
            request("json-discovered-device"),
            manifest.model_id,
            loaded_snapshot,
        )
        self.assertTrue(any(
            "helper-c" in row.device_ids for row in candidates.candidates
        ))

    def test_estimated_operator_profiles_remain_shadow_only(self) -> None:
        value = catalog(
            phone_ops_per_s=8_000_000_000,
            phone_power_mw=1_000,
            phone_bandwidth=8_000_000_000,
        ).to_json()
        for kernel in value["placement_profile"]["kernels"]:
            if kernel["device_id"] == "helper-c":
                kernel["status"] = "estimated"
        shadow_catalog = RuntimeCapabilityCatalog.from_json(value)
        scheduler, manifest = self.scheduler_and_manifest(shadow_catalog)
        candidates = scheduler.generate_automated_candidates(
            request("shadow-phone-profile"),
            manifest.model_id,
            runtime_snapshot(manifest, phone_bandwidth=8_000_000_000),
        )
        phone_rows = [
            row for row in candidates.candidates
            if "helper-c" in row.device_ids
        ]
        self.assertTrue(phone_rows)
        self.assertTrue(all(not row.admitted for row in phone_rows))
        self.assertTrue(all(
            "ROUTE_NOT_QUALIFIED" in row.rejection_reasons
            and row.cost.energy_evidence == "ASSUMED"
            and row.cost.fleet_energy_uj is not None
            for row in phone_rows
        ))

    def test_rejection_reports_first_missing_physical_plan(self) -> None:
        value = catalog(
            phone_ops_per_s=8_000_000_000,
            phone_power_mw=1_000,
            phone_bandwidth=8_000_000_000,
        ).to_json()
        for kernel in value["placement_profile"]["kernels"]:
            if kernel["device_id"] == "helper-c":
                kernel["status"] = "estimated"
        value["transitions"] = [
            row for row in value["transitions"]
            if row["device_id"] != "helper-c"
        ]
        scheduler, manifest = self.scheduler_and_manifest(
            RuntimeCapabilityCatalog.from_json(value)
        )
        ticket = scheduler.submit_automated_request(
            request("first-physical-rejection"),
            manifest.model_id,
            runtime_snapshot(
                manifest,
                resident_devices=("host-a", "accelerator-b"),
                phone_bandwidth=8_000_000_000,
            ),
        )
        phone_rows = [
            row for row in ticket.cost_estimates.estimates
            if "helper-c" in row.details["device_ids"]
            and row.details["route_family"] == "whole_model"
            and "RESIDENCY_TRANSITION_ABSENT"
                in row.details["rejection_reasons"]
            and "ENERGY_UNKNOWN" not in row.details["rejection_reasons"]
        ]

        self.assertTrue(phone_rows)
        self.assertTrue(all(
            row.reason == "RESIDENCY_TRANSITION_ABSENT"
            and row.details["primary_rejection_reason"]
                == "RESIDENCY_TRANSITION_ABSENT"
            for row in phone_rows
        ))

    def test_fake_adapter_executes_scheduler_selected_mixed_stream(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        ))
        adapter = FakeAutomatedPhysicalAdapter(scheduler)
        snapshots = (
            runtime_snapshot(
                manifest, phone_bandwidth=8_000_000_000
            ),
            runtime_snapshot(manifest, include_phone=False),
            replace(
                runtime_snapshot(manifest, include_phone=False),
                executors={
                    "executor:host-a": executor_state(
                        "executor:host-a"
                    )
                },
            ),
        )
        expected_devices = (
            ("helper-c",),
            ("accelerator-b",),
            ("host-a",),
        )
        tickets = []
        for index, snapshot in enumerate(snapshots):
            ticket = scheduler.submit_automated_request(
                request(
                    f"fake-stream-{index}",
                    arrival_us=1_000 + index * 100_000,
                ),
                manifest.model_id,
                snapshot,
            )
            self.assertEqual(
                ticket.execution_plan.device_ids,
                expected_devices[index],
            )
            tickets.append(ticket)
            adapter.execute(ticket.request.request_id)

        self.assertEqual(len(adapter.executed), len(tickets))
        for executed, original in zip(adapter.executed, tickets):
            self.assertEqual(executed, (
                original.binding.endpoint,
                original.binding.operator_plan_protocol,
                original.execution_plan.plan_sha256,
            ))

        records = scheduler.runtime_decision_log()["records"]
        self.assertEqual(
            [row["event_kind"] for row in records],
            [
                "DECISION", "ACQUIRED", "COMPLETED",
                "DECISION", "ACQUIRED", "COMPLETED",
                "DECISION", "ACQUIRED", "COMPLETED",
            ],
        )
        decisions = [
            row for row in records if row["event_kind"] == "DECISION"
        ]
        self.assertTrue(all(len(row["candidates"]) > 3 for row in decisions))
        self.assertTrue(all(
            all(
                "admitted" in candidate
                and "selection_status" in candidate
                for candidate in row["candidates"]
            )
            for row in decisions
        ))
        terminals = [
            row for row in records if row["event_kind"] == "COMPLETED"
        ]
        self.assertEqual(len(terminals), len(tickets))
        self.assertTrue(all(
            row["selected"]["execution_receipt"] is not None
            for row in terminals
        ))
        scheduler.validate_runtime_decision_log(
            scheduler.runtime_decision_log()
        )

    def test_synthetic_24_request_epoch_dispatch_completes(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        adapter = FakeAutomatedPhysicalAdapter(scheduler)
        for index in range(24):
            observed_at_us = 1_000 + index * 100
            snapshot = runtime_snapshot(
                manifest, include_phone=index % 4 != 2
            )
            snapshot = replace(
                snapshot,
                snapshot_id="epoch-dispatch-" + str(index),
                captured_at_us=observed_at_us,
                valid_until_us=10_000_000,
                memory=replace(
                    snapshot.memory,
                    snapshot_id=(
                        "epoch-dispatch-memory-" + str(index)
                    ),
                    captured_at_us=observed_at_us,
                    valid_until_us=10_000_000,
                ),
            )
            ticket = scheduler.submit_automated_request(
                request(
                    "late-epoch-" + str(88116 + index),
                    arrival_us=observed_at_us,
                ),
                manifest.model_id,
                snapshot,
                observed_at_us=observed_at_us,
            )
            adapter.execute(ticket.request.request_id)

        records = scheduler.runtime_decision_log()["records"]
        self.assertEqual(sum(
            row["event_kind"] == "COMPLETED" for row in records
        ), 24)
        self.assertFalse(any(
            row["event_kind"] in {"FAILED", "CANCELLED"}
            for row in records
        ))
        self.assertTrue(all(
            row["selected"].get("model_placement_resolution")
            for row in records
            if row["event_kind"] in {"DECISION", "REPLAN"}
        ))
        scheduler.validate_runtime_decision_log(
            scheduler.runtime_decision_log()
        )

    def test_current_store_round_trips_component_generation_change(
        self,
    ) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        selected = scheduler.generate_automated_candidates(
            request("component-generation-source"),
            manifest.model_id,
            runtime_snapshot(manifest),
        ).baseline
        store = RuntimeRouteObservationStore()
        generation = "sha256:" + "a" * 64
        domains = (
            "energy:accelerator-b",
            "energy:helper-c",
            "energy:host-a",
        )
        for index in range(2):
            receipt = RuntimeExecutionReceipt(
                ticket_id="component-generation-ticket-" + str(index),
                request_id="component-generation-request-" + str(index),
                artifact_sha256=manifest.artifact_sha256,
                operator_plan_sha256=selected.plan.plan_sha256,
                executor_id=selected.binding.executor_id,
                endpoint=selected.binding.endpoint,
                operator_plan_protocol=(
                    selected.binding.operator_plan_protocol
                ),
                participant_executor_ids=tuple(
                    row.executor_id
                    for row in selected.binding.participants
                ),
                started_us=0,
                finished_us=100 + index,
                output_sha256="sha256:" + format(index, "064x"),
                status="COMPLETED",
                energy_boundary_id="synthetic-whole-fleet",
                fleet_energy_uj_by_domain={
                    domains[0]: 100,
                    domains[1]: 100,
                    domains[2]: 100,
                },
                measurement_evidence_ids=(
                    "component-generation-isolated-" + str(index),
                ),
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
                required_domain_ids=domains,
                component_service_us=100,
                component_energy_uj=300,
            ))

        persisted = store.to_json()
        current = json.loads(json.dumps(persisted["template_rows"][0]))
        legacy = json.loads(json.dumps(current))
        legacy["key"][2] = persisted["rows"][0]["key"][13]
        legacy["observations"] = current["observations"][:1]
        current["observations"] = current["observations"][1:]
        persisted["template_rows"] = [legacy, current]
        persisted.pop("store_sha256")
        persisted["store_sha256"] = canonical_sha256(persisted)

        restored = RuntimeRouteObservationStore()
        restored.import_json(persisted)
        self.assertEqual(restored.state()["complete_receipts"], 2)
        self.assertEqual(restored.state()["templates"], 2)
        round_trip = RuntimeRouteObservationStore()
        round_trip.import_json(restored.to_json())
        self.assertEqual(round_trip.state(), restored.state())

    def test_platform_thermal_status_is_the_coarse_phone_gate(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        ))
        snapshot = runtime_snapshot(
            manifest, phone_bandwidth=8_000_000_000
        )
        helper = snapshot.executors["executor:helper-c"]
        qualified = replace(
            snapshot,
            executors={
                **snapshot.executors,
                "executor:helper-c": replace(
                    helper,
                    temperature_millic=95_000,
                    thermal_qualified=True,
                ),
            },
        )
        candidates = scheduler.generate_automated_candidates(
            request("platform-thermal-qualified"),
            manifest.model_id,
            qualified,
        )
        phone_rows = tuple(
            row for row in candidates.candidates
            if "helper-c" in row.device_ids
        )
        self.assertTrue(phone_rows)
        self.assertTrue(all(
            "THERMAL_LIMIT" not in row.rejection_reasons
            for row in phone_rows
        ))

        emergency = replace(
            qualified,
            executors={
                **qualified.executors,
                "executor:helper-c": replace(
                    helper,
                    temperature_millic=80_000,
                    thermal_qualified=False,
                ),
            },
        )
        rejected = scheduler.generate_automated_candidates(
            request("platform-thermal-emergency"),
            manifest.model_id,
            emergency,
        )
        self.assertTrue(all(
            "THERMAL_LIMIT" in row.rejection_reasons
            for row in rejected.candidates
            if "helper-c" in row.device_ids
        ))

    def test_registered_identity_hashes_bind_runtime_costs(self) -> None:
        profile = catalog()
        scheduler, manifest = self.scheduler_and_manifest(profile)
        snapshot = runtime_snapshot(manifest)
        ticket = scheduler.submit_automated_request(
            request("registered-identity-hashes"),
            manifest.model_id,
            snapshot,
        )

        self.assertEqual(
            ticket.cost_estimates.planning_profile_sha256,
            canonical_sha256(profile),
        )
        self.assertEqual(
            ticket.cost_estimates.model_manifest_sha256,
            canonical_sha256(manifest),
        )
        self.assertEqual(
            ticket.cost_estimates.runtime_system_snapshot_sha256,
            canonical_sha256(snapshot),
        )

    def test_observation_store_merge_is_atomic_and_deterministic(self) -> None:
        profile = catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        )

        def one_store(index: int) -> dict[str, object]:
            scheduler, manifest = self.scheduler_and_manifest(profile)
            ticket = scheduler.submit_automated_request(
                request(
                    "merge-observation-" + str(index),
                    arrival_us=1_000 + index * 10_000,
                ),
                manifest.model_id,
                runtime_snapshot(
                    manifest, phone_bandwidth=8_000_000_000
                ),
            )
            active = scheduler.wait_runtime_request(
                ticket.request.request_id,
                time.monotonic_ns() - ticket.decision.start_us * 1_000,
            )
            scheduler.complete_automated_request(
                active.request.request_id,
                self.measured_execution_receipt(
                    active,
                    latency_us=active.decision.service_us + index,
                    fleet_energy_uj=9_000 + index,
                ),
            )
            return dict(scheduler.automated_observation_snapshot())

        first = one_store(1)
        second = one_store(2)
        merged, _ = self.scheduler_and_manifest(profile)
        merged.load_automated_observations(first)
        merged.merge_automated_observations(second)
        self.assertEqual(
            merged.automated_observation_state()["complete_receipts"], 2
        )
        expected = dict(merged.automated_observation_snapshot())
        with self.assertRaisesRegex(
            UnifiedScheduleError,
            "runtime observation store receipts overlap",
        ):
            merged.merge_automated_observations(second)
        self.assertEqual(
            dict(merged.automated_observation_snapshot()), expected
        )

    def test_incomplete_fleet_measurement_does_not_qualify_online_cost(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        snapshot = runtime_snapshot(manifest)
        ticket = scheduler.submit_automated_request(
            request("partial-measurement"), manifest.model_id, snapshot
        )
        epoch_ns = time.monotonic_ns() - ticket.decision.start_us * 1_000
        active = scheduler.wait_runtime_request(
            ticket.request.request_id, epoch_ns
        )
        scheduler.complete_automated_request(
            active.request.request_id,
            self.execution_receipt(
                active,
                energy_boundary_id="synthetic-whole-fleet",
                fleet_energy_uj_by_domain={"energy:host-a": 10},
                measurement_evidence_ids=("partial-energy",),
            ),
        )
        state = scheduler.automated_observation_state()
        self.assertEqual(state["qualified_shape_buckets"], 0)
        self.assertEqual(state["incomplete_receipts"], 1)

    def test_same_artifact_executor_replacement_keeps_desktop_control(
        self,
    ) -> None:
        source = catalog(gpu_whole_model=False)
        _, manifest = self.scheduler_and_manifest()
        host = replace(
            source.executor_by_device["host-a"],
            layer_fractions_ppm=(),
        )
        gpu = replace(
            source.executor_by_device["accelerator-b"],
            exclusive_residency_resource_id="compute:accelerator-b",
            layer_fractions_ppm=(),
        )
        placements = tuple(
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
        desktop = RuntimeCompositeExecutorCapability(
            executor_id="coordinator:replacement-desktop",
            endpoint="synthetic://replacement-desktop",
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
            evidence_ids=("synthetic-replacement-desktop",),
            artifact_sha256=manifest.artifact_sha256,
            operator_placements=placements,
            replacement_group_by_device={
                "host-a": "compute:accelerator-b",
                "accelerator-b": "compute:accelerator-b",
            },
        )
        previous = replace(
            desktop,
            executor_id="coordinator:replacement-previous",
            endpoint="synthetic://replacement-previous",
            backend="backend:previous",
            evidence_ids=("synthetic-replacement-previous",),
        )
        transition = RuntimeTransitionCapability(
            transition_id="load:replacement-desktop",
            device_id="accelerator-b",
            source_state="cold",
            target_state="hot",
            fixed_latency_us=100,
            bandwidth_bytes_per_s=1_000_000_000,
            fixed_energy_uj=100,
            dynamic_pj_per_byte=100,
            resource_ids=("compute:host-a", "compute:accelerator-b"),
            maturity="QUALIFIED",
            evidence_ids=("synthetic-replacement-desktop",),
            executor_id=desktop.executor_id,
            prepares_device_ids=("host-a", "accelerator-b"),
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=tuple(
                host if row.device_id == host.device_id else
                gpu if row.device_id == gpu.device_id else row
                for row in source.executors
            ),
            composite_executors=(desktop, previous),
            desktop_control_profiles=(RuntimeDesktopControlProfile(
                profile_id="synthetic-replacement-desktop",
                artifact_sha256=manifest.artifact_sha256,
                executor_id=desktop.executor_id,
                operator_placements=placements,
                maturity="QUALIFIED",
                evidence_ids=("synthetic-replacement-desktop",),
            ),),
            transitions=source.transitions + (transition,),
        ).to_json())
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        scheduler.register_model_manifest(manifest)
        cold = runtime_snapshot(
            manifest,
            include_phone=False,
            resident_devices=(),
        )
        states = dict(cold.executors)
        states[desktop.executor_id] = RuntimeExecutorState(
            executor_id=desktop.executor_id,
            healthy=True,
            ready=False,
            temperature_millic=40_000,
            battery_ppm=900_000,
            free_slots=0,
            busy_until_us=0,
        )
        states[previous.executor_id] = executor_state(previous.executor_id)
        cold = replace(cold, executors=states)
        probe = scheduler.generate_automated_candidates(
            request("same-artifact-replacement-probe"),
            manifest.model_id,
            cold,
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
        reclaimable = {
            "accelerator-b": required_by_resource["gpu-memory"],
            "host-a": weight_by_device["host-a"],
        }
        residency = model_residency_observations(
            profile,
            manifest,
            previous.executor_id,
            generation=7,
            reclaimable_bytes_by_device=reclaimable,
        )
        capacities = dict(cold.memory.capacities)
        capacities["host-memory"] = DeviceMemoryCapacity(
            "host-memory",
            required_by_resource["host-memory"],
            weight_by_device["host-a"],
            0,
        )
        capacities["gpu-memory"] = DeviceMemoryCapacity(
            "gpu-memory",
            required_by_resource["gpu-memory"],
            required_by_resource["gpu-memory"],
            0,
        )
        constrained = replace(
            cold,
            memory=replace(cold.memory, capacities=capacities),
            residency=residency,
        )

        ticket = scheduler.submit_automated_request(
            request("same-artifact-replacement"),
            manifest.model_id,
            constrained,
            selection_mode="desktop-baseline",
        )

        self.assertEqual(ticket.binding.executor_id, desktop.executor_id)
        self.assertEqual(ticket.dispatch_state, "QUEUED")
        self.assertEqual(ticket.transition_status, "PENDING")
        self.assertEqual(len(ticket.execution_plan.transitions), 1)
        evictions = ticket.execution_plan.transitions[0].evictions
        self.assertEqual(
            {row.device_id for row in evictions},
            {"host-a", "accelerator-b"},
        )
        self.assertTrue(all(
            row.artifact_sha256 == manifest.artifact_sha256
            and row.executor_id == previous.executor_id
            and row.generation == 7
            for row in evictions
        ))
        self.assertTrue(all(
            row.replaceable_bytes > 0
            for row in ticket.execution_plan.memory_demands
            if row.kind == "model_weights"
        ))

    def test_unavoidable_slo_miss_uses_feasible_desktop_baseline(
        self,
    ) -> None:
        probe_scheduler, manifest = self.scheduler_and_manifest()
        del probe_scheduler
        source = catalog(gpu_whole_model=False)
        host = replace(
            source.executor_by_device["host-a"],
            layer_fractions_ppm=(),
        )
        gpu = replace(
            source.executor_by_device["accelerator-b"],
            layer_fractions_ppm=(),
        )
        composite = RuntimeCompositeExecutorCapability(
            executor_id="coordinator:tardy-desktop",
            endpoint="synthetic://tardy-desktop",
            backend="backend:composite",
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
            layer_fractions_ppm=(500_000,),
            residency_states=("hot",),
            resource_ids=("compute:host-a", "compute:accelerator-b"),
            operator_plan_protocol="synthetic-composite-v1",
            maturity="QUALIFIED",
            evidence_ids=("synthetic-tardy-desktop",),
        )
        route_profile = RuntimeRouteShapeProfile(
            selector_id="synthetic-tardy-desktop",
            artifact_sha256=manifest.artifact_sha256,
            route_family="layer_placement",
            device_ids=("host-a", "accelerator-b"),
            assisted_operator_kind=None,
            split_axis="none",
            split_fraction_ppm=0,
            residency_variant="hot",
            minimum_input_tokens=1,
            maximum_input_tokens=1_000,
            minimum_output_tokens=1,
            maximum_output_tokens=1_000,
            service_fixed_us=10_000_000,
            service_input_token_us=0,
            service_output_token_us=0,
            service_upper_add_us=1_000_000,
            energy_fixed_uj=1_000,
            energy_input_token_uj=0,
            energy_output_token_uj=0,
            energy_lower_error_ppm=100_000,
            energy_upper_error_ppm=100_000,
            sample_count=2,
            maturity="QUALIFIED",
            evidence_ids=("synthetic-tardy-desktop",),
            executor_id=composite.executor_id,
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=tuple(
                host if row.device_id == host.device_id else
                gpu if row.device_id == gpu.device_id else row
                for row in source.executors
            ),
            composite_executors=(composite,),
            route_shape_profiles=(route_profile,),
        ).to_json())
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        manifest = scheduler.register_gguf_model(
            manifest.model_id, self.path
        )
        snapshot = runtime_snapshot(manifest, include_phone=False)
        states = dict(snapshot.executors)
        states[composite.executor_id] = executor_state(
            composite.executor_id
        )
        snapshot = replace(snapshot, executors=states)
        probe = scheduler.generate_automated_candidates(
            request("unavoidable-tardy-probe"),
            manifest.model_id,
            snapshot,
        )
        composite_row = next(
            row for row in probe.candidates
            if row.binding.executor_id == composite.executor_id
        )
        host_required = sum(
            row.required_bytes
            for row in composite_row.plan.memory_demands
            if row.resource_id == "host-memory"
        )
        capacities = dict(snapshot.memory.capacities)
        capacities["host-memory"] = DeviceMemoryCapacity(
            "host-memory", host_required, 0, 0
        )
        constrained = replace(
            snapshot,
            memory=replace(snapshot.memory, capacities=capacities),
        )

        ticket = scheduler.submit_automated_request(
            request("unavoidable-tardy-desktop"),
            manifest.model_id,
            constrained,
            selection_mode="desktop-baseline",
        )

        self.assertEqual(ticket.binding.executor_id, composite.executor_id)
        self.assertGreater(
            ticket.decision.finish_upper_us,
            ticket.request.deadline_us,
        )
        self.assertNotIn(
            "SLO_UPPER_BOUND",
            dict(ticket.decision.rejected).values(),
        )

    def test_queued_request_reserves_memory_after_predecessor(self) -> None:
        source = catalog()
        cpu = source.executor_by_device["host-a"]
        cpu_only = RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=(cpu,),
            composite_executors=(),
            transitions=tuple(
                row for row in source.transitions
                if row.device_id == "host-a"
            ),
        ).to_json())
        scheduler, manifest = self.scheduler_and_manifest(cpu_only)
        cold = runtime_snapshot(manifest, resident_devices=())
        probe = scheduler.generate_automated_candidates(
            request("queued-memory-probe"), manifest.model_id, cold
        ).baseline
        required = sum(
            row.additional_bytes for row in probe.plan.memory_demands
        )
        capacities = dict(cold.memory.capacities)
        capacities["host-memory"] = DeviceMemoryCapacity(
            "host-memory", required, 0, 0
        )
        constrained = replace(
            cold,
            memory=replace(cold.memory, capacities=capacities),
        )

        first = scheduler.submit_automated_request(
            request("queued-memory-a", arrival_us=1_000),
            manifest.model_id,
            constrained,
        )
        second = scheduler.submit_automated_request(
            request("queued-memory-b", arrival_us=1_001),
            manifest.model_id,
            constrained,
        )

        self.assertGreaterEqual(
            second.decision.start_us,
            max(row.reserved_until_us for row in first.decision.leases),
        )
        intervals = {
            row["owner_id"]: (
                row["start_us"], row["reserved_until_us"]
            )

            for row in scheduler.runtime_memory_state()["reservations"]
        }
        self.assertEqual(intervals["queued-memory-a"][1],
                         intervals["queued-memory-b"][0])

    def test_preallocated_context_pool_owns_hot_request_memory(self) -> None:
        source = catalog()
        _, manifest = self.scheduler_and_manifest()
        resources = dict(source.resources)
        for resource_id in ("compute:host-a", "compute:accelerator-b"):
            resources[resource_id] = replace(
                resources[resource_id], capacity=4
            )
        source = replace(
            source,
            resources=resources,
            executors=tuple(
                replace(
                    row,
                    exclusive_residency_resource_id=(
                        "compute:accelerator-b"
                    ),
                )
                if row.device_id == "accelerator-b" else row
                for row in source.executors
            ),
        )
        topology = RuntimePhysicalTopology(
            cpu_device_id="host-a",
            gpu_device_id="accelerator-b",
            phone_device_id="helper-c",
            cpu_resource_id="compute:host-a",
            gpu_resource_id="compute:accelerator-b",
            host_memory_resource_id="host-memory",
            gpu_memory_resource_id="gpu-memory",
            phone_memory_resource_id="phone-memory",
            functionfs_resource_id="link:usb-out",
            phone_transport_resource_ids=(
                "link:usb-in", "link:usb-out"
            ),
            phone_compute_resource_ids=("compute:helper-c",),
            gpu_exclusive_residency_resource_id=(
                "compute:accelerator-b"
            ),
            resource_capacities={
                "compute:host-a": 4,
                "compute:accelerator-b": 4,
                "compute:helper-c": 1,
                "link:usb-in": 1,
                "link:usb-out": 1,
            },
            resource_identities={},
        )
        executor_id = "coordinator:preallocated-desktop"
        profile = materialize_desktop_control(
            source,
            manifest,
            topology,
            executor_id=executor_id,
            endpoint="synthetic://preallocated-desktop",
            backend="backend:composite",
            gpu_first_layer=0,
            adapter_parameters={
                "context_size": 20,
                "parallel": 2,
            },
            evidence_ids=("sha256:" + "7" * 64,),
            transition_latency_us=100,
            transition_energy_uj=100,
        )
        context_resource_id = profile.composite_executor_by_id[
            executor_id
        ].adapter_parameters["context_resource_id"]
        self.assertEqual(
            profile.resources[context_resource_id].capacity, 20
        )

        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        scheduler.register_model_manifest(manifest)
        snapshot = runtime_snapshot(manifest, include_phone=False)
        states = dict(snapshot.executors)
        states[executor_id] = executor_state(
            executor_id, free_slots=2
        )
        residency = tuple(
            replace(row, executor_id=executor_id)
            for row in snapshot.residency
            if row.device_id in {"host-a", "accelerator-b"}
        )
        snapshot = replace(
            snapshot, executors=states, residency=residency
        )
        first_request = request(
            "preallocated-context-a",
            input_tokens=12,
            output_tokens=4,
        )
        hot = scheduler.generate_automated_candidates(
            first_request, manifest.model_id, snapshot
        ).baseline
        self.assertEqual(
            hot.plan.resource_slots[context_resource_id], 16
        )
        self.assertTrue(all(
            row.additional_bytes == 0
            for row in hot.plan.memory_demands
            if row.kind in {"kv_cache", "workspace"}
        ))

        cold = scheduler.generate_automated_candidates(
            request("preallocated-context-cold"),
            manifest.model_id,
            replace(snapshot, residency=()),
        ).baseline
        cold_request_memory = tuple(
            row for row in cold.plan.memory_demands
            if row.kind in {"kv_cache", "workspace"}
        )
        expected_kv_bytes = (
            manifest.block_count
            * 20
            * manifest.head_count_kv
            * (manifest.key_length + manifest.value_length)
            * 2
        )
        self.assertEqual(
            sum(
                row.required_bytes
                for row in cold_request_memory
                if row.kind == "kv_cache"
            ),
            expected_kv_bytes,
        )
        self.assertTrue(cold_request_memory)
        self.assertTrue(all(
            row.additional_bytes > 0
            and row.lifetime == "resident"
            and row.share_key is not None
            for row in cold_request_memory
        ))
        other_shape = scheduler.generate_automated_candidates(
            request(
                "preallocated-context-other-shape",
                input_tokens=10,
                output_tokens=2,
            ),
            manifest.model_id,
            replace(snapshot, residency=()),
        ).baseline
        self.assertEqual(
            {
                (row.device_id, row.resource_id, row.kind): row.share_key
                for row in cold_request_memory
            },
            {
                (row.device_id, row.resource_id, row.kind): row.share_key
                for row in other_shape.plan.memory_demands
                if row.kind in {"kv_cache", "workspace"}
            },
        )
        self.assertEqual(
            {
                (row.device_id, row.kind): row.required_bytes
                for row in cold_request_memory
                if row.kind == "kv_cache"
            },
            {
                (row.device_id, row.kind): row.required_bytes
                for row in other_shape.plan.memory_demands
                if row.kind == "kv_cache"
            },
        )

        first = scheduler.submit_automated_request(
            first_request,
            manifest.model_id,
            snapshot,
            selection_mode="desktop-baseline",
        )
        busy_states = dict(snapshot.executors)
        busy_states[executor_id] = RuntimeExecutorState(
            executor_id=executor_id,
            healthy=True,
            ready=False,
            temperature_millic=40_000,
            battery_ppm=900_000,
            free_slots=0,
            busy_until_us=0,
        )
        second = scheduler.submit_automated_request(
            request(
                "preallocated-context-b",
                arrival_us=1_001,
                input_tokens=12,
                output_tokens=4,
            ),
            manifest.model_id,
            replace(snapshot, executors=busy_states),
            selection_mode="desktop-baseline",
        )
        self.assertGreaterEqual(
            second.decision.start_us,
            max(row.reserved_until_us for row in first.decision.leases),
        )

    def test_automated_decision_replay_is_byte_deterministic(self) -> None:
        outputs = []
        for _ in range(2):
            scheduler, manifest = self.scheduler_and_manifest()
            scheduler.submit_automated_request(
                request("automated-replay"),
                manifest.model_id,
                runtime_snapshot(manifest),
            )
            output = scheduler.runtime_decision_log_bytes()
            scheduler.validate_runtime_decision_log(
                scheduler.runtime_decision_log()
            )
            outputs.append(output)
        self.assertEqual(outputs[0], outputs[1])

    def test_static_operator_costs_are_cached_but_live_admission_is_not(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        snapshot = runtime_snapshot(manifest)
        scheduler.generate_automated_candidates(
            request("cache-first"), manifest.model_id, snapshot
        )
        first = dict(scheduler.automated_cost_cache_stats())
        scheduler.generate_automated_candidates(
            request("cache-second"), manifest.model_id, snapshot
        )
        second = dict(scheduler.automated_cost_cache_stats())
        self.assertGreater(first["misses"], 0)
        self.assertEqual(second["misses"], first["misses"])
        self.assertEqual(second["hits"], first["hits"] + first["misses"])

    def test_bounded_refiner_starts_from_measured_desktop_baseline(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        candidates = scheduler.generate_automated_candidates(
            request("bounded-desktop-baseline"),
            manifest.model_id,
            runtime_snapshot(manifest),
        )
        self.assertLessEqual(len(candidates.candidates), 32)
        self.assertNotIn("helper-c", candidates.baseline.device_ids)
        self.assertEqual(candidates.baseline.device_ids, ("accelerator-b",))
        self.assertEqual(
            candidates.baseline.maturity, "QUALIFIED"
        )
        self.assertEqual(
            candidates.search_metadata["evaluated_plan_count"],
            len(candidates.candidates),
        )
        self.assertLessEqual(
            candidates.search_metadata["search_budget"], 32
        )

    def test_rough_plan_cache_uses_request_shape_buckets(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        snapshot = runtime_snapshot(manifest)
        scheduler.generate_automated_candidates(
            request("bucket-first", input_tokens=11, output_tokens=3),
            manifest.model_id,
            snapshot,
        )
        first = dict(scheduler.automated_cost_cache_stats())
        scheduler.generate_automated_candidates(
            request("bucket-second", input_tokens=12, output_tokens=4),
            manifest.model_id,
            snapshot,
        )
        second = dict(scheduler.automated_cost_cache_stats())
        self.assertEqual(second["misses"], first["misses"])
        self.assertGreater(second["hits"], first["hits"])
        self.assertEqual(second["shape_bucket_entries"], 1)

    def test_cached_synthetic_refinement_is_below_ten_milliseconds(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        snapshot = runtime_snapshot(manifest)
        scheduler.generate_automated_candidates(
            request("overhead-warm"), manifest.model_id, snapshot
        )
        durations_ns = []
        for index in range(5):
            started_ns = time.perf_counter_ns()
            candidates = scheduler.generate_automated_candidates(
                request(f"overhead-{index}"), manifest.model_id, snapshot
            )
            durations_ns.append(time.perf_counter_ns() - started_ns)
            self.assertLessEqual(len(candidates.candidates), 32)
        self.assertLess(sum(durations_ns) / len(durations_ns), 10_000_000)


if __name__ == "__main__":
    unittest.main()
