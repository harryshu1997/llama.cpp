"""Automated runtime: route templates, composites, desktop controls, measured energy and learning.

Split from test_automated_runtime.py on 2026-09-13; fixtures and the base class stay there."""

from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import threading
import time
from unittest import mock
from research_dev.scheduler import (
    GGUFModelManifestLoader,
    ModelManifest,
    ModelResidencyObservation,
    ResourceProfile,
    RuntimeCapabilityCatalog,
    RuntimeCompositeExecutorCapability,
    RuntimeCompositeOperatorPlacement,
    RuntimeDesktopControlProfile,
    RuntimeExecutorState,
    RuntimeExecutionReceipt,
    RuntimeExecutionContract,
    RuntimeKernelShapeProfile,
    RuntimeResourceError,
    RuntimeRouteShapeProfile,
    RuntimeTransitionCapability,
    RuntimeTransitionReceipt,
    UnifiedScheduleError,
    UnifiedScheduler,
)
from research_dev.scheduler._internal.background_placement import placement_frontier_key
from research_dev.scheduler._internal.adaptive_decode import ADAPTIVE_OBSERVATION_STORE_SCHEMA
from research_dev.scheduler._internal.adaptive_decode_contracts import (
    AdaptiveDecodeGroupedObservation,
    AdaptiveDecodeWindowReceipt,
)
from research_dev.scheduler._internal.adaptive_decode_planning import adaptive_decode_policies
from research_dev.scheduler._internal.runtime_learning import (
    RuntimeRouteObservationStore,
    _template_cost_feature_bucket,
)
from research_dev.scheduler._internal.runtime_plan import RuntimePlanError
from research_dev.scheduler._internal.runtime_search import request_shape_bucket
from research_dev.scheduler._internal.runtime_residency_projection import (
    transition_target_is_observed,
)
from research_dev.scheduler._internal.runtime_residency_cohorts import (
    RuntimeResidencyComponentIdentity,
    RuntimeResidencyCohortTracker,
    runtime_residency_component_identity,
)
from research_dev.scheduler._internal.types import canonical_sha256
from research_dev.scheduler.adapters import (
    EndpointRuntimeSample,
    PhysicalAdapterError,
    UnifiedRuntimeSnapshotBuilder,
    model_residency_observations,
)

try:
    from .test_automated_runtime import (
        AutomatedRuntimeTests,
        catalog,
        catalog_with_gpu_desktop_control,
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
        catalog,
        catalog_with_gpu_desktop_control,
        executor_state,
        protected_snapshot,
        request,
        runtime_snapshot,
        system_cost_profile,
        write_synthetic_gguf,
    )


class AutomatedRuntimeRoutesTests(AutomatedRuntimeTests):
    """Automated runtime: route templates, composites, desktop controls, measured energy and learning."""

    def test_zero_model_demand_preserves_legacy_template_identity(
        self,
    ) -> None:
        legacy = canonical_sha256(_template_cost_feature_bucket({
            "large_model_op15": 0,
        }))
        causal_zero = canonical_sha256(_template_cost_feature_bucket({
            "active_model_input_tokens": 0,
            "active_model_output_tokens": 0,
            "active_model_requests": 0,
            "large_model_op15": 0,
        }))
        causal_active = canonical_sha256(_template_cost_feature_bucket({
            "active_model_input_tokens": 128,
            "active_model_output_tokens": 32,
            "active_model_requests": 1,
            "large_model_op15": 0,
        }))

        self.assertEqual(causal_zero, legacy)
        self.assertNotEqual(causal_active, legacy)

    def test_desktop_control_is_distinct_from_cpu_recovery(self) -> None:
        scheduler, manifest = self.scheduler_with_gpu_control()
        candidates = scheduler.generate_automated_candidates(
            request("control-separation"),
            manifest.model_id,
            runtime_snapshot(manifest),
        )

        self.assertEqual(
            candidates.baseline.binding.executor_id,
            "executor:accelerator-b",
        )
        self.assertEqual(
            candidates.recovery_fallback.binding.executor_id,
            "executor:host-a",
        )
        self.assertNotEqual(
            candidates.baseline_route_id,
            candidates.recovery_fallback_route_id,
        )
        self.assertIn("accelerator-b", candidates.baseline.device_ids)

    def test_desktop_control_queues_for_busy_gpu(self) -> None:
        scheduler, manifest = self.scheduler_with_gpu_control()
        busy_until_us = 80_000
        ticket = scheduler.submit_automated_request(
            request("queued-gpu-control"),
            manifest.model_id,
            runtime_snapshot(
                manifest,
                gpu_busy_until_us=busy_until_us,
                gpu_free_slots=0,
            ),
            selection_mode="desktop-baseline",
        )

        self.assertEqual(
            ticket.binding.executor_id, "executor:accelerator-b"
        )
        self.assertGreaterEqual(ticket.decision.start_us, busy_until_us)
        self.assertEqual(
            ticket.decision.reason, "DESKTOP_BASELINE_CONTROL"
        )

    def test_physical_control_failure_uses_cpu_recovery_once(self) -> None:
        scheduler, manifest = self.scheduler_with_gpu_control()
        snapshot = runtime_snapshot(manifest)
        control = scheduler.submit_automated_request(
            request("control-failure"),
            manifest.model_id,
            snapshot,
            selection_mode="desktop-baseline",
        )
        epoch_ns = time.monotonic_ns() - control.decision.start_us * 1_000
        control = scheduler.wait_runtime_request(
            control.request.request_id, epoch_ns
        )
        recovery = scheduler.fail_automated_request(
            control.request.request_id,
            failed_at_us=control.decision.start_us + 1,
            reason="synthetic GPU transport failure",
            snapshot=snapshot,
        )

        self.assertIsNotNone(recovery.fallback)
        self.assertEqual(
            recovery.fallback.binding.executor_id, "executor:host-a"
        )
        self.assertEqual(
            recovery.fallback.decision.reason,
            "QUALIFIED_RECOVERY_FALLBACK",
        )

    def test_cpu_energy_cannot_qualify_phone_against_gpu_control(
        self,
    ) -> None:
        manifest = GGUFModelManifestLoader.load(
            "unseen-model", self.path
        )
        source = catalog_with_gpu_desktop_control(manifest)
        gpu_shadow = RuntimeRouteShapeProfile(
            selector_id="synthetic-gpu-control-shadow-cost",
            artifact_sha256=manifest.artifact_sha256,
            route_family="whole_model",
            device_ids=("accelerator-b",),
            assisted_operator_kind=None,
            split_axis="none",
            split_fraction_ppm=0,
            residency_variant="hot",
            minimum_input_tokens=1,
            maximum_input_tokens=64,
            minimum_output_tokens=1,
            maximum_output_tokens=64,
            service_fixed_us=100,
            service_input_token_us=1,
            service_output_token_us=1,
            service_upper_add_us=10,
            energy_fixed_uj=100,
            energy_input_token_uj=1,
            energy_output_token_uj=1,
            energy_lower_error_ppm=100_000,
            energy_upper_error_ppm=100_000,
            sample_count=1,
            maturity="SHADOW",
            evidence_ids=("synthetic-gpu-shadow",),
            executor_id="executor:accelerator-b",
        )
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(replace(
            source, route_shape_profiles=(gpu_shadow,)
        ))
        scheduler.register_model_manifest(manifest)

        ticket = scheduler.submit_automated_request(
            request("cpu-evidence-is-not-control-evidence"),
            manifest.model_id,
            runtime_snapshot(manifest),
            selection_mode="energy-aware",
        )

        self.assertEqual(
            ticket.binding.executor_id, "executor:accelerator-b"
        )
        recovery = next(
            row for row in ticket.cost_estimates.estimates
            if row.details.get("recovery_fallback")
        )
        self.assertIsNotNone(recovery.fleet_energy_lower_uj)
        admitted_phone = tuple(
            row for row in ticket.cost_estimates.estimates
            if "helper-c" in row.details.get("device_ids", [])
            and row.admitted
        )
        self.assertTrue(admitted_phone)
        rejected = dict(ticket.decision.rejected)
        self.assertTrue(all(
            rejected[row.route_id]
                == "BASELINE_ENERGY_EVIDENCE_NOT_QUALIFIED"
            for row in admitted_phone
        ))

    def test_route_evidence_identity_ignores_unrelated_catalog_metadata(
        self,
    ) -> None:
        base = catalog()
        first, first_manifest = self.scheduler_and_manifest(base)
        first_candidates = first.generate_automated_candidates(
            request("route-identity-first"),
            first_manifest.model_id,
            runtime_snapshot(first_manifest),
        )
        first_plan = first_candidates.baseline.plan

        unrelated = replace(
            base,
            catalog_id="synthetic-catalog-with-another-route",
            minimum_energy_saving_ppm=20_000,
        )
        second, second_manifest = self.scheduler_and_manifest(unrelated)
        second_candidates = second.generate_automated_candidates(
            request("route-identity-second"),
            second_manifest.model_id,
            runtime_snapshot(second_manifest),
        )
        second_plan = second_candidates.baseline.plan

        self.assertNotEqual(canonical_sha256(base), canonical_sha256(unrelated))
        first_identity = first._automated_compiler(
        ).route_capability_identity(
            first_plan, first_candidates.baseline.binding.executor_id
        )
        second_identity = second._automated_compiler(
        ).route_capability_identity(
            second_plan, second_candidates.baseline.binding.executor_id
        )
        self.assertEqual(first_identity, second_identity)
        self.assertEqual(
            first._automated_compiler().route_capability_identity(first_plan),
            first_identity,
        )

    def test_route_evidence_identity_changes_with_relevant_kernel(self) -> None:
        base = catalog()
        first, first_manifest = self.scheduler_and_manifest(base)
        first_plan = first.generate_automated_candidates(
            request("route-kernel-first"),
            first_manifest.model_id,
            runtime_snapshot(first_manifest),
        ).baseline.plan
        profile_id = "kernel:accelerator-b:ffn"
        profiled = base.placement_profile.kernels[profile_id]
        changed_kernels = dict(base.placement_profile.kernels)
        changed_kernels[profile_id] = replace(
            profiled,
            kernel=replace(
                profiled.kernel,
                effective_ops_per_s=(
                    profiled.kernel.effective_ops_per_s + 1
                ),
            ),
        )
        changed = replace(
            base,
            placement_profile=replace(
                base.placement_profile, kernels=changed_kernels
            ),
        )
        second, second_manifest = self.scheduler_and_manifest(changed)
        second_plan = second.generate_automated_candidates(
            request("route-kernel-second"),
            second_manifest.model_id,
            runtime_snapshot(second_manifest),
        ).baseline.plan

        self.assertNotEqual(
            first._automated_compiler().route_capability_identity(first_plan),
            second._automated_compiler().route_capability_identity(second_plan),
        )

    def test_new_model_and_connected_phone_expand_capability_routes(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        without_phone = scheduler.generate_automated_candidates(
            request("without-phone"),
            manifest.model_id,
            runtime_snapshot(manifest, include_phone=False),
        )
        missing = [
            row for row in without_phone.candidates
            if "helper-c" in row.device_ids
        ]
        self.assertTrue(missing)
        self.assertTrue(all(not row.admitted for row in missing))
        self.assertTrue(all(
            "EXECUTOR_OBSERVATION_ABSENT" in row.rejection_reasons
            for row in missing
        ))
        self.assertTrue(without_phone.baseline.admitted)

        with_phone = scheduler.generate_automated_candidates(
            request("with-phone"),
            manifest.model_id,
            runtime_snapshot(manifest),
        )
        admitted_families = {
            row.route_family for row in with_phone.candidates if row.admitted
        }
        self.assertTrue({
            "whole_model",
            "layer_placement",
            "operator_offload",
            "operator_split",
        }.issubset(admitted_families))
        self.assertTrue(any(
            set(row.device_ids) == {"host-a", "accelerator-b", "helper-c"}
            for row in with_phone.candidates
            if row.route_family == "layer_placement" and row.admitted
        ))
        self.assertTrue({"column", "row", "tensor"}.issubset({
            row.split_axis for row in with_phone.candidates
            if row.route_family == "operator_split" and row.admitted
        }))
        assisted = {
            row.assisted_operator_kind
            for row in with_phone.candidates
            if row.route_family in {"operator_offload", "operator_split"}
            and row.admitted
        }
        self.assertTrue({
            "attention", "kv_cache", "embedding", "lm_head", "ffn"
        }.issubset(assisted))
        self.assertTrue({"hot", "warm", "cold"}.issubset({
            row.residency_variant for row in with_phone.candidates
        }))

    def test_composite_route_requires_a_physical_coordinator_contract(self) -> None:
        source = catalog()
        no_coordinators = replace(
            source,
            executors=tuple(
                replace(
                    row,
                    coordinated_route_families=(),
                    operator_plan_protocol=None,
                )
                for row in source.executors
            ),
        )
        scheduler, manifest = self.scheduler_and_manifest(no_coordinators)
        candidates = scheduler.generate_automated_candidates(
            request("missing-coordinator"),
            manifest.model_id,
            runtime_snapshot(manifest),
        )
        composite = [
            row for row in candidates.candidates if len(row.device_ids) > 1
        ]
        self.assertTrue(composite)
        self.assertTrue(all(not row.admitted for row in composite))
        self.assertTrue(all(
            "COMPOSITE_COORDINATOR_ABSENT" in row.rejection_reasons
            for row in composite
        ))
        self.assertTrue(all(row.binding.endpoint is None for row in composite))

    def test_composite_binding_uses_the_registered_coordinator_endpoint(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
            phone_whole_model=False,
            gpu_whole_model=False,
        ))
        candidates = scheduler.generate_automated_candidates(
            request("bound-coordinator"),
            manifest.model_id,
            runtime_snapshot(manifest, phone_bandwidth=8_000_000_000),
        )
        row = next(
            item for item in candidates.candidates
            if item.admitted and len(item.device_ids) > 1
        )
        coordinator = next(
            item for item in row.binding.participants
            if item.executor_id == row.binding.executor_id
        )
        self.assertEqual(row.binding.endpoint, coordinator.endpoint)
        self.assertEqual(
            row.binding.operator_plan_protocol, "synthetic-plan-v1"
        )
        self.assertNotIn("coordinator:auto:", row.binding.executor_id)
        self.assertEqual(
            row.binding.operator_plan_sha256, row.plan.plan_sha256
        )

    def test_explicit_composite_capability_binds_exact_physical_plan(self) -> None:
        source = catalog(
            phone_ops_per_s=8_000_000_000,
            phone_power_mw=500,
            phone_bandwidth=8_000_000_000,
            phone_whole_model=False,
            gpu_whole_model=False,
        )
        resources = dict(source.resources)
        resources["coordinator:split-c"] = ResourceProfile(
            resource_id="coordinator:split-c",
            kind="coordinator",
            capacity=1,
            ready=True,
            identity="coordinator:split-c",
        )
        physical = RuntimeCompositeExecutorCapability(
            executor_id="coordinator:split-c",
            endpoint="synthetic://split-c",
            backend="backend:composite",
            coordinator_device_id="host-a",
            participant_device_ids=("host-a", "helper-c"),
            participant_resource_ids={
                "host-a": (
                    "compute:host-a", "coordinator:split-c"
                ),
                "helper-c": ("compute:helper-c",),
            },
            route_family="operator_split",
            assisted_operator_kind="ffn",
            split_axis="column",
            split_fractions_ppm=(250_000, 500_000, 750_000),
            layer_fractions_ppm=(),
            residency_states=("hot",),
            resource_ids=(
                "compute:host-a",
                "compute:helper-c",
                "coordinator:split-c",
                "link:usb-out",
                "link:usb-in",
            ),
            operator_plan_protocol="synthetic-composite-v1",
            maturity="QUALIFIED",
            evidence_ids=("physical-split-evidence",),
            adapter_parameters={
                "ffn_n_embd": 32,
                "phone_device_id": "helper-c",
                "ubatch_size": 512,
            },
        )
        profile = replace(
            source,
            resources=resources,
            executors=tuple(
                replace(
                    row,
                    coordinated_route_families=(),
                    operator_plan_protocol=None,
                )
                for row in source.executors
            ),
            composite_executors=(physical,),
        )
        profile = RuntimeCapabilityCatalog.from_json(profile.to_json())
        scheduler, manifest = self.scheduler_and_manifest(profile)
        snapshot = runtime_snapshot(
            manifest,
            phone_bandwidth=8_000_000_000,
        )
        snapshot = replace(
            snapshot,
            executors={
                **snapshot.executors,
                physical.executor_id: executor_state(physical.executor_id),
            },
        )
        candidates = scheduler.generate_automated_candidates(
            request("physical-composite"), manifest.model_id, snapshot
        )
        rows = [
            row for row in candidates.candidates
            if row.binding.executor_id == physical.executor_id
        ]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertTrue(row.admitted, row.rejection_reasons)
        self.assertEqual(row.binding.endpoint, physical.endpoint)
        self.assertEqual(
            row.binding.operator_plan_protocol,
            physical.operator_plan_protocol,
        )
        self.assertEqual(
            row.binding.operator_plan_sha256, row.plan.plan_sha256
        )
        self.assertEqual(row.plan.adapter_parameters["ffn_max_tokens"], 16)
        self.assertEqual(row.plan.adapter_parameters["ubatch_size"], 16)
        self.assertEqual(
            row.binding.resource_ids,
            tuple(sorted(physical.resource_ids)),
        )
        participant_resources = {
            item.device_id: item.resource_ids
            for item in row.binding.participants
        }
        self.assertEqual(
            participant_resources,
            dict(physical.participant_resource_ids),
        )
        ticket = scheduler.submit_automated_request(
            request("selected-physical-composite"),
            manifest.model_id,
            snapshot,
        )
        self.assertEqual(
            ticket.binding.executor_id,
            physical.executor_id,
            ticket.decision,
        )
        self.assertEqual(ticket.execution_plan, row.plan)

    def test_three_device_coordinator_binds_explicit_operator_placement(self) -> None:
        source = catalog(
            phone_ops_per_s=8_000_000_000,
            phone_power_mw=500,
            phone_bandwidth=8_000_000_000,
            phone_whole_model=False,
            gpu_whole_model=False,
        )
        scheduler, manifest = self.scheduler_and_manifest(source)
        placements = tuple(
            RuntimeCompositeOperatorPlacement(
                operator_id=operator.operator_id,
                primary_device_id=(
                    "host-a"
                    if operator.kind == "ffn"
                    else "accelerator-b"
                ),
                helper_device_id=(
                    "helper-c" if operator.kind == "ffn" else None
                ),
                split_axis=(
                    "column" if operator.kind == "ffn" else "none"
                ),
                split_fraction_ppm=(
                    500_000 if operator.kind == "ffn" else 0
                ),
                assisted=operator.kind == "ffn",
            )
            for operator in manifest.operators
        )
        resources = dict(source.resources)
        resources["coordinator:three-device"] = ResourceProfile(
            resource_id="coordinator:three-device",
            kind="coordinator",
            capacity=1,
            ready=True,
            identity="coordinator:three-device",
        )
        physical = RuntimeCompositeExecutorCapability(
            executor_id="coordinator:three-device",
            endpoint="synthetic://three-device",
            backend="backend:composite",
            coordinator_device_id="host-a",
            participant_device_ids=(
                "host-a", "accelerator-b", "helper-c"
            ),
            participant_resource_ids={
                "host-a": ("compute:host-a", "coordinator:three-device"),
                "accelerator-b": ("compute:accelerator-b",),
                "helper-c": ("compute:helper-c",),
            },
            route_family="operator_split",
            assisted_operator_kind="ffn",
            split_axis="column",
            split_fractions_ppm=(500_000,),
            layer_fractions_ppm=(),
            residency_states=("hot",),
            resource_ids=(
                "compute:host-a",
                "compute:accelerator-b",
                "compute:helper-c",
                "coordinator:three-device",
                "link:pcie-out",
                "link:pcie-in",
                "link:usb-out",
                "link:usb-in",
            ),
            operator_plan_protocol="synthetic-three-device-v1",
            maturity="QUALIFIED",
            evidence_ids=("physical-three-device-evidence",),
            artifact_sha256=manifest.artifact_sha256,
            operator_placements=placements,
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            resources=resources,
            composite_executors=(physical,),
        ).to_json())
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        scheduler.register_model_manifest(
            ModelManifest.from_json(manifest.to_json())
        )
        snapshot = runtime_snapshot(
            manifest, phone_bandwidth=8_000_000_000
        )
        snapshot = replace(
            snapshot,
            executors={
                **snapshot.executors,
                physical.executor_id: executor_state(physical.executor_id),
            },
        )

        candidates = scheduler.generate_automated_candidates(
            request("three-device"), manifest.model_id, snapshot
        )
        row = next(
            item for item in candidates.candidates
            if item.binding.executor_id == physical.executor_id
        )
        self.assertTrue(row.admitted, row.rejection_reasons)
        self.assertEqual(
            row.plan.device_ids,
            ("accelerator-b", "helper-c", "host-a"),
        )
        actual = {
            item.operator_id: (
                item.device_ids,
                item.split_axis,
                item.split_fraction_ppm,
            )
            for item in row.plan.operators
        }
        expected = {
            item.operator_id: (
                tuple(sorted(filter(None, (
                    item.primary_device_id, item.helper_device_id
                )))),
                item.split_axis,
                item.split_fraction_ppm,
            )
            for item in placements
        }
        self.assertEqual(actual, expected)
        self.assertEqual(
            RuntimeCompositeExecutorCapability.from_json(physical.to_json()),
            physical,
        )

    def test_model_placement_epoch_predicts_only_compatible_reuse(
        self,
    ) -> None:
        source = catalog()
        _, manifest = self.scheduler_and_manifest(source)
        executor_id = source.executor_by_device[
            "accelerator-b"
        ].executor_id

        def component(transport: str) -> RuntimeResidencyComponentIdentity:
            return RuntimeResidencyComponentIdentity(
                artifact_sha256=manifest.artifact_sha256,
                resident_shard_geometry_sha256=(
                    canonical_sha256("epoch-shard"),
                ),
                desktop_placement_sha256=canonical_sha256(
                    "epoch-desktop-placement"
                ),
                transport_generation=transport,
                operator_protocol="epoch-operator-v1",
                session_resource_ids=("memory:epoch-session",),
                executor_id=executor_id,
            )

        compatible = component("transport-compatible")
        incompatible = component("transport-incompatible")
        tracker = RuntimeResidencyCohortTracker()
        for index, arrival_us in enumerate((1_000, 2_000, 3_000)):
            request_id = f"epoch-compatible-{index}"
            tracker.record_planning_arrival(
                request_id, (compatible,), arrival_us
            )
            tracker.record_terminal(request_id)
        for index, arrival_us in enumerate((1_000, 1_100, 1_200)):
            request_id = f"epoch-incompatible-{index}"
            tracker.record_planning_arrival(
                request_id, (incompatible,), arrival_us
            )
            tracker.record_terminal(request_id)

        epoch = tracker.model_placement_epoch(
            manifest.artifact_sha256,
            3_500,
            request_id="epoch-current",
        )

        self.assertEqual(
            epoch.reuse_projections[
                compatible.identity_sha256
            ].expected_use_count,
            2,
        )
        self.assertEqual(
            epoch.reuse_projections[
                compatible.identity_sha256
            ].horizon_us,
            500,
        )
        self.assertEqual(
            epoch.reuse_projections[
                incompatible.identity_sha256
            ].expected_use_count,
            1,
        )
        self.assertEqual(epoch.active_request_count, 0)
        self.assertEqual(epoch.virtual_queue_request_count, 0)
        self.assertEqual(
            epoch.component_opportunity_counts[
                compatible.identity_sha256
            ],
            3,
        )

    def test_model_placement_epoch_counts_queued_compatible_route_once(
        self,
    ) -> None:
        source = catalog()
        _, manifest = self.scheduler_and_manifest(source)
        executor_id = source.executor_by_device[
            "accelerator-b"
        ].executor_id

        def component(label: str) -> RuntimeResidencyComponentIdentity:
            return RuntimeResidencyComponentIdentity(
                artifact_sha256=manifest.artifact_sha256,
                resident_shard_geometry_sha256=(
                    canonical_sha256("queue-shard-" + label),
                ),
                desktop_placement_sha256=canonical_sha256(
                    "queue-desktop-placement"
                ),
                transport_generation="queue-transport-v1",
                operator_protocol="queue-operator-v1",
                session_resource_ids=("memory:queue-session",),
                executor_id=executor_id,
            )

        desktop = component("desktop")
        phone = component("phone")
        tracker = RuntimeResidencyCohortTracker()
        tracker.record_planning_arrival(
            "queued-request", (desktop, phone), 1_000
        )
        tracker.record_arrival("queued-request", desktop, 1_000)

        epoch = tracker.model_placement_epoch(
            manifest.artifact_sha256,
            1_100,
            request_id="current-request",
            queued_request_ids=("queued-request",),
        )

        self.assertEqual(epoch.virtual_queue_request_count, 1)
        self.assertEqual(
            epoch.reuse_projections[
                desktop.identity_sha256
            ].observed_request_count,
            2,
        )
        self.assertEqual(
            epoch.reuse_projections[
                phone.identity_sha256
            ].observed_request_count,
            2,
        )

    def test_scheduler_binds_model_placement_epoch_to_candidate_set(
        self,
    ) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        candidates = scheduler.generate_automated_candidates(
            request("placement-epoch-metadata"),
            manifest.model_id,
            runtime_snapshot(manifest),
        )

        self.assertEqual(
            candidates.search_metadata[
                "model_placement_epoch_generation"
            ],
            0,
        )
        self.assertEqual(
            candidates.search_metadata["virtual_queue_request_count"],
            0,
        )
        self.assertTrue(
            candidates.search_metadata[
                "model_placement_epoch_sha256"
            ].startswith("sha256:")
        )

    def test_published_model_placement_reuses_compatible_shape(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        snapshot = runtime_snapshot(manifest)
        first = scheduler.submit_automated_request(
            request(
                "published-model-shape-first",
                input_tokens=12,
                output_tokens=4,
            ),
            manifest.model_id,
            snapshot,
        )
        compiler = scheduler._automated_compiler()
        with mock.patch.object(
            compiler,
            "generate",
            side_effect=AssertionError("full placement generation ran"),
        ):
            second = scheduler.submit_automated_request(
                request(
                    "published-model-shape-second",
                    arrival_us=1_100,
                    input_tokens=65,
                    output_tokens=33,
                ),
                manifest.model_id,
                snapshot,
                observed_at_us=1_100,
            )

        metadata = next(
            row.details for row in second.cost_estimates.estimates
            if row.route_id == second.decision.route_id
        )
        self.assertTrue(metadata["route_template_cross_shape_reuse"])
        self.assertEqual(
            tuple(metadata["route_template_source_shape_bucket"]),
            request_shape_bucket(12, 4),
        )
        self.assertEqual(
            tuple(metadata["route_template_request_shape_bucket"]),
            request_shape_bucket(65, 33),
        )
        self.assertEqual(
            runtime_residency_component_identity(
                first.model.artifact_sha256,
                first.execution_plan,
                first.binding,
            ).identity_sha256,
            runtime_residency_component_identity(
                second.model.artifact_sha256,
                second.execution_plan,
                second.binding,
            ).identity_sha256,
        )
        timing = scheduler.runtime_decision_timings()[-1]
        self.assertTrue(timing["model_placement_epoch_fast_path"])
        self.assertEqual(
            timing["model_placement_epoch_invalidation_reason"],
            "MODEL_PLACEMENT_EPOCH_COMPATIBLE_SHAPE",
        )

    def test_model_epoch_accepts_compatible_cold_to_hot_route(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        cold = runtime_snapshot(manifest, resident_devices=())
        value = request("epoch-cold-hot")
        cold_candidates = scheduler.generate_automated_candidates(
            value, manifest.model_id, cold
        )
        cold_route = cold_candidates.baseline
        demand, action = scheduler._evaluate_model_placement(
            value,
            manifest,
            cold,
            value.arrival_us,
            None,
            "desktop-baseline",
        )
        epoch, _ = scheduler._propose_model_placement_epoch(
            request=value,
            manifest=manifest,
            candidate_set=cold_candidates,
            selected=cold_route,
            observed_at_us=value.arrival_us,
            selection_mode="desktop-baseline",
            invalidation_reason="MODEL_PLACEMENT_EPOCH_ABSENT",
            snapshot=cold,
            demand_snapshot=demand,
            placement_action=action,
            current_epoch=None,
        )
        hot = runtime_snapshot(manifest)
        hot_candidates = scheduler.generate_automated_candidates(
            value, manifest.model_id, hot
        )
        hot_route = hot_candidates.baseline

        self.assertNotEqual(
            cold_route.candidate_id, hot_route.candidate_id
        )
        compatibility = scheduler._model_placement_compatibility(
            epoch, hot_route, manifest, hot_candidates
        )
        self.assertTrue(compatibility.compatible)
        self.assertEqual(compatibility.rejection_reasons, ())

    def test_model_epoch_accepts_approved_fraction_change(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        snapshot = runtime_snapshot(manifest)
        first = scheduler.submit_automated_request(
            request("epoch-fraction-source"),
            manifest.model_id,
            snapshot,
        )
        epoch = next(iter(
            scheduler._runtime_residency_cohorts
            .compatible_published_model_placement_epochs(
                manifest.artifact_sha256,
                first.request.quality_requirement,
                first.selection_mode,
                scheduler._runtime_capabilities.maximum_latency_ppm,
            )
        ))
        templates = scheduler._runtime_route_template_sets[
            epoch.epoch_sha256
        ]
        source = next(
            row for row in templates.candidate_set.candidates
            if "helper-c" in row.device_ids
            and row.assisted_operator_kind == "ffn"
        )
        adaptive = RuntimeExecutionContract(
            execution_mode="adaptive-split",
            initial_split_fraction_ppm=750_000,
            allowed_adaptive_fractions_ppm=(0, 750_000),
            batch_plan="single",
            maximum_batch_size=1,
            queue_depth=1,
            phone_device_id="helper-c",
            phone_endpoint=source.binding.endpoint,
            operator_kind="ffn",
        )
        adaptive_plan = replace(
            source.plan, execution_contract=adaptive
        )
        adaptive_source = replace(
            source,
            plan=adaptive_plan,
            binding=replace(
                source.binding,
                operator_plan_sha256=adaptive_plan.plan_sha256,
            ),
        )
        adaptive_candidates = replace(
            templates.candidate_set,
            candidates=tuple(
                adaptive_source
                if row.candidate_id == source.candidate_id else row
                for row in templates.candidate_set.candidates
            ),
        )
        component = runtime_residency_component_identity(
            manifest.artifact_sha256,
            adaptive_plan,
            adaptive_source.binding,
        )
        parent_id = (
            adaptive_source.paired_baseline_route_id
            or adaptive_candidates.baseline_route_id
        )
        parent = next(
            row for row in adaptive_candidates.candidates
            if row.candidate_id == parent_id
        )
        adaptive_epoch = replace(
            epoch,
            selected_component_identity_sha256=(
                component.identity_sha256
            ),
            selected_route_id=adaptive_source.candidate_id,
            selected_executor_id=adaptive_source.binding.executor_id,
            selected_operator_plan_sha256=adaptive_plan.plan_sha256,
            selected_desktop_parent_route_id=parent.candidate_id,
            selected_desktop_parent_plan_sha256=parent.plan.plan_sha256,
            selected_desktop_parent_placement_sha256=(
                parent.plan.desktop_placement_sha256
            ),
            selected_resident_artifact_sha256s=(
                component.resident_artifact_sha256s
            ),
            selected_resident_shard_geometry_sha256s=(
                component.resident_shard_geometry_sha256
            ),
            selected_session_resource_ids=(
                component.session_resource_ids
            ),
            allowed_adaptive_fractions_ppm=(0, 750_000),
            epoch_sha256="",
        )

        compatibility = scheduler._model_placement_compatibility(
            adaptive_epoch,
            adaptive_source,
            manifest,
            adaptive_candidates,
        )
        self.assertTrue(compatibility.compatible)
        self.assertEqual(
            compatibility.component.identity_sha256,
            adaptive_epoch.selected_component_identity_sha256,
        )

    def test_learning_refresh_publishes_before_queued_replan(self) -> None:
        manifest = GGUFModelManifestLoader.load(
            "unseen-model", self.path
        )
        source = catalog(
            phone_ops_per_s=10_000_000_000,
            phone_power_mw=500,
            phone_bandwidth=8_000_000_000,
            phone_whole_model=False,
            gpu_whole_model=False,
        )
        placements = tuple(
            RuntimeCompositeOperatorPlacement(
                operator_id=operator.operator_id,
                primary_device_id=(
                    "host-a"
                    if operator.kind in {"embedding", "kv_cache"}
                    else "accelerator-b"
                ),
                helper_device_id=None,
                split_axis="none",
                split_fraction_ppm=0,
            )
            for operator in manifest.operators
        )
        desktop = RuntimeCompositeExecutorCapability(
            executor_id="coordinator:learning-refresh-desktop",
            endpoint="synthetic://learning-refresh-desktop",
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
            resource_ids=(
                "compute:host-a",
                "compute:accelerator-b",
                "link:pcie-out",
                "link:pcie-in",
            ),
            operator_plan_protocol="synthetic-desktop-v1",
            maturity="QUALIFIED",
            evidence_ids=("synthetic-learning-refresh-desktop",),
            artifact_sha256=manifest.artifact_sha256,
            operator_placements=placements,
        )
        phone = RuntimeCompositeExecutorCapability(
            executor_id="coordinator:learning-refresh-phone",
            endpoint="synthetic://learning-refresh-phone",
            backend="backend:phone",
            coordinator_device_id="host-a",
            participant_device_ids=(
                "host-a", "accelerator-b", "helper-c"
            ),
            participant_resource_ids={
                "host-a": ("compute:host-a",),
                "accelerator-b": ("compute:accelerator-b",),
                "helper-c": ("compute:helper-c",),
            },
            route_family="operator_split",
            assisted_operator_kind="ffn",
            split_axis="column",
            split_fractions_ppm=(250_000, 500_000, 750_000),
            layer_fractions_ppm=(),
            residency_states=("hot",),
            resource_ids=(
                "compute:host-a",
                "compute:accelerator-b",
                "compute:helper-c",
                "link:pcie-out",
                "link:pcie-in",
                "link:usb-out",
                "link:usb-in",
            ),
            operator_plan_protocol="synthetic-phone-v1",
            maturity="QUALIFIED",
            evidence_ids=("synthetic-learning-refresh-phone",),
            artifact_sha256=manifest.artifact_sha256,
            operator_ids=tuple(
                operator.operator_id for operator in manifest.operators
                if operator.kind == "ffn"
            ),
            baseline_executor_id=desktop.executor_id,
            helper_device_id="helper-c",
            adapter_parameters={
                "ffn_column_quantum": 1,
                "ffn_n_embd": manifest.embedding_length,
                "ffn_runtime_control_protocol": "decode-boundary-v1",
                "ffn_weight_buffer_layout": "selected-width",
                "maximum_helper_resident_weight_bytes": (
                    manifest.tensor_bytes
                ),
                "phone_device_id": "helper-c",
                "requires_measured_route_profile": 1,
                "ubatch_size": 4,
            },
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            composite_executors=(desktop, phone),
            desktop_control_profiles=(RuntimeDesktopControlProfile(
                profile_id="synthetic-learning-refresh-control",
                artifact_sha256=manifest.artifact_sha256,
                executor_id=desktop.executor_id,
                operator_placements=placements,
                maturity="QUALIFIED",
                evidence_ids=("synthetic-learning-refresh-desktop",),
            ),),
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
                desktop.executor_id: executor_state(desktop.executor_id),
                phone.executor_id: executor_state(phone.executor_id),
            },
        )
        history_request = request(
            "learning-refresh-history",
            input_tokens=33,
            output_tokens=33,
        )
        history_candidates = scheduler.generate_automated_candidates(
            history_request, manifest.model_id, snapshot
        )
        baseline_policy, policies, envelope = adaptive_decode_policies(
            history_candidates,
            manifest,
            scheduler._runtime_capabilities,
            history_request.output_tokens,
        )
        self.assertIsNotNone(envelope)
        winner = max(policies, key=lambda row: row.split_fraction_ppm)
        baseline_energy = max(
            100,
            history_candidates.baseline.cost.warm_execution_energy_uj
                // history_request.output_tokens,
        )
        groups = []
        for index in range(2):
            request_id = "learning-refresh-history-" + str(index)
            baseline_window = AdaptiveDecodeWindowReceipt(
                request_id=request_id,
                slot_id=0,
                window_index=0,
                token_start=0,
                token_end=4,
                context_length=history_request.input_tokens,
                active_batch=1,
                started_at_us=0,
                finished_at_us=4_000,
                policy=baseline_policy,
                applied_ack=None,
                fleet_energy_uj_by_domain={"fleet": baseline_energy * 4},
                latency_per_token_us=1_000,
                phone_compute_us=0,
                usb_transfer_us=0,
                rpc_us=0,
                exposed_tail_us=0,
                output_valid=True,
                evidence_ids=("learning-refresh-energy",),
                energy_boundary_id=(
                    scheduler._runtime_capabilities.placement_profile
                        .energy_boundary_id
                ),
                energy_attribution_kind="isolated",
                failure_reason=None,
                previous_record_sha256="0" * 64,
                window_role="exploration",
            )
            assisted_window = AdaptiveDecodeWindowReceipt(
                request_id=request_id,
                slot_id=0,
                window_index=1,
                token_start=4,
                token_end=8,
                context_length=history_request.input_tokens,
                active_batch=1,
                started_at_us=4_000,
                finished_at_us=7_200,
                policy=winner,
                applied_ack=None,
                fleet_energy_uj_by_domain={
                    "fleet": baseline_energy
                        * winner.split_fraction_ppm // 4_000_000
                },
                latency_per_token_us=800,
                phone_compute_us=1_000,
                usb_transfer_us=100,
                rpc_us=100,
                exposed_tail_us=100,
                output_valid=True,
                evidence_ids=("learning-refresh-energy",),
                energy_boundary_id=(
                    scheduler._runtime_capabilities.placement_profile
                        .energy_boundary_id
                ),
                energy_attribution_kind="isolated",
                failure_reason=None,
                previous_record_sha256=(
                    baseline_window.record_sha256.removeprefix("sha256:")
                ),
                window_role="exploitation",
            )
            groups.append(AdaptiveDecodeGroupedObservation(
                request_id=request_id,
                ticket_id=request_id + ":attempt:0",
                model_artifact_sha256=manifest.artifact_sha256,
                planning_profile_sha256=(
                    scheduler._runtime_capability_generation_sha256
                ),
                desktop_placement_sha256=(
                    baseline_policy.desktop_placement_sha256
                ),
                windows=(baseline_window, assisted_window),
                final_policy=winner,
                terminal_status="COMPLETED",
                state_history=(
                    "BASELINE", "PROBING", "EXPLOITING", "COMPLETED"
                ),
            ))
        first = scheduler.submit_automated_request(
            request("learning-refresh-first", arrival_us=1_000),
            manifest.model_id,
            snapshot,
            selection_mode="calibration",
        )
        second = scheduler.submit_automated_request(
            request(
                "learning-refresh-second",
                arrival_us=1_100,
                input_tokens=65,
                output_tokens=33,
            ),
            manifest.model_id,
            snapshot,
            observed_at_us=1_100,
            selection_mode="calibration",
        )
        history_body = {
            "groups": [row.to_json() for row in groups],
            "schema": ADAPTIVE_OBSERVATION_STORE_SCHEMA,
        }
        scheduler.load_adaptive_decode_observations({
            **history_body,
            "store_sha256": canonical_sha256(history_body),
        })
        epoch_ns = time.monotonic_ns() - first.decision.start_us * 1_000
        active = scheduler.wait_runtime_request(
            first.request.request_id, epoch_ns
        )
        finished_us = max(
            active.decision.start_us + 1,
            second.request.arrival_us,
        )

        def refreshed_snapshot(ticket, observed_at_us):
            return replace(
                snapshot,
                snapshot_id="learning-refresh-runtime",
                captured_at_us=observed_at_us,
                valid_until_us=observed_at_us + 10_000_000,
                memory=replace(
                    snapshot.memory,
                    snapshot_id="learning-refresh-memory",
                    captured_at_us=observed_at_us,
                    valid_until_us=observed_at_us + 10_000_000,
                ),
            )

        refresh_order = []
        refreshed_candidates = []
        apply_adaptive = scheduler._apply_adaptive_history_costs
        select_candidate = scheduler._select_automated_candidate

        def apply_before_admission(*args, **kwargs):
            refresh_order.append("adaptive_history")
            result = apply_adaptive(*args, **kwargs)
            refreshed_candidates.append(result)
            return result

        def select_after_history(*args, **kwargs):
            refresh_order.append("route_admission")
            return select_candidate(*args, **kwargs)

        with mock.patch.object(
            scheduler,
            "_apply_adaptive_history_costs",
            side_effect=apply_before_admission,
        ) as adaptive_rerank:
            with mock.patch.object(
                scheduler,
                "_select_automated_candidate",
                side_effect=select_after_history,
            ):
                completion = scheduler.complete_automated_request(
                    active.request.request_id,
                    self.measured_execution_receipt(
                        active,
                        latency_us=(
                            finished_us - active.decision.start_us
                        ),
                        fleet_energy_uj=10_000,
                    ),
                    snapshot_provider=refreshed_snapshot,
                )
                self.assertEqual(
                    completion.completion_event_replans,
                    (second.request.request_id,),
                )
                deadline = time.monotonic() + 2
                while (
                    scheduler.model_placement_epoch_stats()[
                        "background_refreshes"
                    ] < 1
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.005)
                wake = scheduler.wait_runtime_request(
                    second.request.request_id,
                    time.monotonic_ns() - finished_us * 1_000,
                )
                self.assertEqual(
                    wake.dispatch_state, "REPLAN_REQUIRED"
                )
                wake_receipt = wake.dispatch_receipt
                second = scheduler.replan_automated_request(
                    second.request.request_id,
                    observed_at_us=wake_receipt.observed_at_us,
                    reason=wake_receipt.wake_reason,
                    snapshot=refreshed_snapshot(
                        wake, wake_receipt.observed_at_us
                    ),
                    expected_ticket_id=wake.ticket_id,
                    expected_queue_generation=(
                        wake_receipt.queue_generation
                    ),
                )
                wake = scheduler.wait_runtime_request(
                    second.request.request_id,
                    time.monotonic_ns()
                    - second.decision.start_us * 1_000,
                )
                self.assertGreaterEqual(adaptive_rerank.call_count, 1)
                self.assertLess(
                    refresh_order.index("adaptive_history"),
                    refresh_order.index("route_admission"),
                )
        measured_phone = tuple(
            row
            for candidates in refreshed_candidates
            for row in candidates.candidates
            if "helper-c" in row.device_ids
            and row.residency_variant == "hot"
            and (row.residency_break_even or {}).get(
                "adaptive_history_applied"
            )
        )
        self.assertTrue(measured_phone)
        self.assertTrue(any(
            row.residency_break_even[
                "adaptive_history_requested_context_bucket"
            ] == (65).bit_length()
            and row.residency_break_even[
                "adaptive_history_evidence_context_bucket"
            ] == (33).bit_length()
            for row in measured_phone
        ))
        self.assertTrue(any(
            row.maturity == "QUALIFIED"
            and row.cost.energy_evidence == "MEASURED"
            and row.cost.latency_evidence == "MEASURED"
            and "ROUTE_NOT_QUALIFIED" not in row.rejection_reasons
            for row in measured_phone
        ))
        self.assertEqual(wake.dispatch_state, "ACQUIRED")
        stats = scheduler.model_placement_epoch_stats()
        self.assertEqual(stats["background_refreshes"], 1)
        self.assertEqual(
            stats["background_refresh_candidates"],
            len(first.cost_estimates.estimates),
        )
        self.assertEqual(stats["background_refresh_failures"], 0)
        self.assertGreater(stats["background_refresh_total_us"], 0)

    def test_learning_rerank_is_background_and_keeps_queue_live(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        snapshot = runtime_snapshot(manifest)
        first = scheduler.submit_automated_request(
            request("background-learning-first", arrival_us=1_000),
            manifest.model_id,
            snapshot,
        )
        second = scheduler.submit_automated_request(
            request(
                "background-learning-second",
                arrival_us=1_100,
                input_tokens=65,
                output_tokens=33,
            ),
            manifest.model_id,
            snapshot,
            observed_at_us=1_100,
        )
        epoch_ns = time.monotonic_ns() - first.decision.start_us * 1_000
        active = scheduler.wait_runtime_request(
            first.request.request_id, epoch_ns
        )
        finished_us = max(
            active.decision.start_us + 1,
            second.request.arrival_us,
        )

        def refreshed_snapshot(ticket, observed_at_us):
            return replace(
                snapshot,
                snapshot_id="background-learning-runtime",
                captured_at_us=observed_at_us,
                valid_until_us=observed_at_us + 10_000_000,
                memory=replace(
                    snapshot.memory,
                    snapshot_id="background-learning-memory",
                    captured_at_us=observed_at_us,
                    valid_until_us=observed_at_us + 10_000_000,
                ),
            )

        compiler = scheduler._runtime_epoch_route_compiler
        self.assertIsNotNone(compiler)
        original = compiler.rerank_route_template_set
        entered = threading.Event()
        release = threading.Event()

        def blocked_rerank(*args, **kwargs):
            entered.set()
            if not release.wait(2):
                raise AssertionError("background rerank was not released")
            return original(*args, **kwargs)

        with mock.patch.object(
            compiler,
            "rerank_route_template_set",
            side_effect=blocked_rerank,
        ), mock.patch.object(
            compiler,
            "replace_observations",
            side_effect=AssertionError("observation JSON was reparsed"),
        ):
            completion = scheduler.complete_automated_request(
                active.request.request_id,
                self.measured_execution_receipt(
                    active,
                    latency_us=(
                        finished_us - active.decision.start_us
                    ),
                    fleet_energy_uj=10_000,
                ),
                snapshot_provider=refreshed_snapshot,
            )
            self.assertEqual(
                completion.completion_event_replans,
                (second.request.request_id,),
            )
            self.assertTrue(entered.wait(1))
            with ThreadPoolExecutor(max_workers=1) as pool:
                waiting_for_replan = pool.submit(
                    scheduler.wait_runtime_request,
                    second.request.request_id,
                    time.monotonic_ns() - finished_us * 1_000,
                )
                wake = waiting_for_replan.result(2)
                self.assertEqual(
                    wake.dispatch_state, "REPLAN_REQUIRED"
                )
                wake_receipt = wake.dispatch_receipt
                replanned = scheduler.replan_automated_request(
                    second.request.request_id,
                    observed_at_us=wake_receipt.observed_at_us,
                    reason=wake_receipt.wake_reason,
                    snapshot=refreshed_snapshot(
                        wake, wake_receipt.observed_at_us
                    ),
                    expected_ticket_id=wake.ticket_id,
                    expected_queue_generation=(
                        wake_receipt.queue_generation
                    ),
                )
                waiting = pool.submit(
                    scheduler.wait_runtime_request,
                    second.request.request_id,
                    time.monotonic_ns()
                    - replanned.decision.start_us * 1_000,
                )
                wake = waiting.result(2)
                completed_before_publication = waiting.done()
                release.set()
                self.assertTrue(completed_before_publication)
        self.assertEqual(wake.dispatch_state, "ACQUIRED")
        deadline = time.monotonic() + 2
        while (
            scheduler.model_placement_epoch_stats()[
                "background_refreshes"
            ] < 1
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        self.assertEqual(
            scheduler.model_placement_epoch_stats()[
                "background_refreshes"
            ],
            1,
        )

    def test_learning_refresh_includes_existing_replan(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        snapshot = runtime_snapshot(manifest)
        first = scheduler.submit_automated_request(
            request("existing-replan-first", arrival_us=1_000),
            manifest.model_id,
            snapshot,
        )
        second = scheduler.submit_automated_request(
            request(
                "existing-replan-second",
                arrival_us=1_100,
                input_tokens=65,
                output_tokens=33,
            ),
            manifest.model_id,
            snapshot,
            observed_at_us=1_100,
        )
        epoch_ns = time.monotonic_ns() - first.decision.start_us * 1_000
        active = scheduler.wait_runtime_request(
            first.request.request_id, epoch_ns
        )
        self.assertTrue(
            scheduler._runtime_controller.require_queued_replan(
                second.request.request_id, "runtime_state_changed"
            )
        )
        finished_us = active.decision.finish_us

        def refreshed_snapshot(ticket, observed_at_us):
            return replace(
                snapshot,
                snapshot_id="existing-replan-runtime",
                captured_at_us=observed_at_us,
                valid_until_us=observed_at_us + 10_000_000,
                memory=replace(
                    snapshot.memory,
                    snapshot_id="existing-replan-memory",
                    captured_at_us=observed_at_us,
                    valid_until_us=observed_at_us + 10_000_000,
                ),
            )

        completion = scheduler.complete_automated_request(
            active.request.request_id,
            self.measured_execution_receipt(
                active,
                latency_us=finished_us - active.decision.start_us,
                fleet_energy_uj=10_000,
            ),
            snapshot_provider=refreshed_snapshot,
        )
        self.assertEqual(completion.completion_event_replans, ())
        deadline = time.monotonic() + 2
        while (
            scheduler.model_placement_epoch_stats()[
                "background_refreshes"
            ] < 1
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        wake = scheduler.wait_runtime_request(
            second.request.request_id,
            time.monotonic_ns() - finished_us * 1_000,
        )
        self.assertEqual(wake.dispatch_state, "REPLAN_REQUIRED")
        compiler = scheduler._automated_compiler()
        with mock.patch.object(
            compiler,
            "generate",
            side_effect=AssertionError("foreground route generation ran"),
        ):
            scheduler.replan_automated_request(
                second.request.request_id,
                observed_at_us=wake.dispatch_receipt.observed_at_us,
                reason=wake.dispatch_receipt.wake_reason,
                snapshot=refreshed_snapshot(
                    wake, wake.dispatch_receipt.observed_at_us
                ),
            )

        timing = scheduler.runtime_decision_timings()[-1]
        self.assertTrue(timing["model_placement_epoch_fast_path"])
        self.assertEqual(
            timing["model_placement_epoch_invalidation_reason"], "NONE"
        )
        self.assertEqual(
            scheduler.model_placement_epoch_stats()[
                "background_refreshes"
            ],
            1,
        )

    def test_explicit_composite_requires_live_coordinator_state(self) -> None:
        source = catalog()
        physical = RuntimeCompositeExecutorCapability(
            executor_id="coordinator:split-c",
            endpoint="synthetic://split-c",
            backend="backend:composite",
            coordinator_device_id="host-a",
            participant_device_ids=("host-a", "helper-c"),
            participant_resource_ids={
                "host-a": ("compute:host-a",),
                "helper-c": ("compute:helper-c",),
            },
            route_family="operator_split",
            assisted_operator_kind="ffn",
            split_axis="column",
            split_fractions_ppm=(500_000,),
            layer_fractions_ppm=(),
            residency_states=("hot",),
            resource_ids=("compute:host-a", "compute:helper-c"),
            operator_plan_protocol="synthetic-composite-v1",
            maturity="QUALIFIED",
            evidence_ids=("physical-split-evidence",),
        )
        scheduler, manifest = self.scheduler_and_manifest(replace(
            source, composite_executors=(physical,)
        ))
        candidates = scheduler.generate_automated_candidates(
            request("missing-composite-state"),
            manifest.model_id,
            runtime_snapshot(manifest),
        )
        row = next(
            item for item in candidates.candidates
            if item.binding.executor_id == physical.executor_id
        )
        self.assertFalse(row.admitted)
        self.assertIn(
            "EXECUTOR_OBSERVATION_ABSENT", row.rejection_reasons
        )

    def test_composite_transition_prepares_hot_unpublished_participant(
        self,
    ) -> None:
        source = catalog()
        composite = RuntimeCompositeExecutorCapability(
            executor_id="coordinator:desktop",
            endpoint="synthetic://desktop",
            backend="backend:composite",
            coordinator_device_id="host-a",
            participant_device_ids=("host-a", "accelerator-b"),
            participant_resource_ids={
                "accelerator-b": ("compute:accelerator-b",),
                "host-a": ("compute:host-a",),
            },
            route_family="layer_placement",
            assisted_operator_kind=None,
            split_axis="none",
            split_fractions_ppm=(),
            layer_fractions_ppm=(500_000,),
            residency_states=("cold", "hot", "warm"),
            resource_ids=("compute:accelerator-b", "compute:host-a"),
            operator_plan_protocol="synthetic-composite-v1",
            maturity="QUALIFIED",
            evidence_ids=("synthetic-desktop-transition",),
        )
        transition = RuntimeTransitionCapability(
            transition_id="publish:desktop",
            device_id="host-a",
            source_state="cold",
            target_state="hot",
            fixed_latency_us=100,
            bandwidth_bytes_per_s=1_000_000_000,
            fixed_energy_uj=100,
            dynamic_pj_per_byte=100,
            resource_ids=("compute:accelerator-b", "compute:host-a"),
            maturity="QUALIFIED",
            evidence_ids=("synthetic-desktop-transition",),
            executor_id=composite.executor_id,
            prepares_device_ids=("accelerator-b", "host-a"),
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            composite_executors=(composite,),
            transitions=(transition,),
        ).to_json())
        scheduler, manifest = self.scheduler_and_manifest(profile)
        snapshot = runtime_snapshot(
            manifest, resident_devices=("accelerator-b",)
        )
        states = dict(snapshot.executors)
        states["executor:accelerator-b"] = replace(
            states["executor:accelerator-b"],
            healthy=False,
            ready=False,
            free_slots=0,
        )
        states[composite.executor_id] = RuntimeExecutorState(
            executor_id=composite.executor_id,
            healthy=True,
            ready=True,
            temperature_millic=40_000,
            battery_ppm=900_000,
            free_slots=1,
            busy_until_us=0,
        )

        candidates = scheduler.generate_automated_candidates(
            request("composite-publication"),
            manifest.model_id,
            replace(snapshot, executors=states),
        )
        row = next(
            candidate for candidate in candidates.candidates
            if candidate.binding.executor_id == composite.executor_id
            and candidate.residency_variant == "cold"
        )

        self.assertTrue(row.plan.transitions)
        self.assertEqual(
            row.plan.transitions[0].executor_id, composite.executor_id
        )
        self.assertNotIn("EXECUTOR_UNHEALTHY", row.rejection_reasons)
        self.assertTrue(row.admitted)

    def test_ready_composite_owns_participant_endpoint_health(self) -> None:
        source = catalog()
        composite = RuntimeCompositeExecutorCapability(
            executor_id="coordinator:desktop",
            endpoint="synthetic://desktop",
            backend="backend:composite",
            coordinator_device_id="host-a",
            participant_device_ids=("host-a", "accelerator-b"),
            participant_resource_ids={
                "accelerator-b": ("compute:accelerator-b",),
                "host-a": ("compute:host-a",),
            },
            route_family="layer_placement",
            assisted_operator_kind=None,
            split_axis="none",
            split_fractions_ppm=(),
            layer_fractions_ppm=(500_000,),
            residency_states=("cold", "hot", "warm"),
            resource_ids=("compute:accelerator-b", "compute:host-a"),
            operator_plan_protocol="synthetic-composite-v1",
            maturity="QUALIFIED",
            evidence_ids=("synthetic-desktop-coordinator",),
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source, composite_executors=(composite,)
        ).to_json())
        scheduler, manifest = self.scheduler_and_manifest(profile)
        snapshot = runtime_snapshot(manifest)
        states = dict(snapshot.executors)
        states["executor:accelerator-b"] = replace(
            states["executor:accelerator-b"],
            healthy=False,
            ready=False,
            free_slots=0,
        )
        states[composite.executor_id] = RuntimeExecutorState(
            executor_id=composite.executor_id,
            healthy=True,
            ready=True,
            temperature_millic=40_000,
            battery_ppm=900_000,
            free_slots=1,
            busy_until_us=0,
        )

        candidates = scheduler.generate_automated_candidates(
            request("ready-composite"),
            manifest.model_id,
            replace(snapshot, executors=states),
        )
        row = next(
            candidate for candidate in candidates.candidates
            if candidate.binding.executor_id == composite.executor_id
            and candidate.residency_variant == "hot"
        )

        self.assertFalse(row.plan.transitions)
        self.assertNotIn("EXECUTOR_UNHEALTHY", row.rejection_reasons)
        self.assertTrue(row.admitted)

    def test_composite_transition_health_marks_participant_links_ready(
        self,
    ) -> None:
        source = catalog()
        composite = RuntimeCompositeExecutorCapability(
            executor_id="coordinator:desktop",
            endpoint="synthetic://desktop",
            backend="backend:composite",
            coordinator_device_id="host-a",
            participant_device_ids=("host-a", "accelerator-b"),
            participant_resource_ids={
                "accelerator-b": ("compute:accelerator-b",),
                "host-a": ("compute:host-a",),
            },
            route_family="layer_placement",
            assisted_operator_kind=None,
            split_axis="none",
            split_fractions_ppm=(),
            layer_fractions_ppm=(500_000,),
            residency_states=("cold", "hot", "warm"),
            resource_ids=(
                "compute:accelerator-b", "compute:host-a"
            ),
            operator_plan_protocol="synthetic-composite-v1",
            maturity="QUALIFIED",
            evidence_ids=("synthetic-desktop-transition",),
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source, composite_executors=(composite,)
        ).to_json())
        _, manifest = self.scheduler_and_manifest(profile)
        snapshot = UnifiedRuntimeSnapshotBuilder(profile).build(
            snapshot_id="composite-transition-health",
            captured_at_us=0,
            valid_until_us=10_000_000,
            memory=runtime_snapshot(manifest).memory,
            executor_samples={
                "executor:host-a": EndpointRuntimeSample(
                    "healthy", "live", 1
                ),
                "executor:accelerator-b": EndpointRuntimeSample(
                    "unavailable", "unavailable", 0
                ),
                composite.executor_id: EndpointRuntimeSample(
                    "unavailable", "unavailable", 0,
                    transition_available=True,
                ),
            },
        )
        self.assertTrue(snapshot.links["pcie-out"].ready)
        self.assertTrue(snapshot.links["pcie-in"].ready)

    def test_composite_residency_can_report_one_preloaded_participant(
        self,
    ) -> None:
        source = catalog()
        _, manifest = self.scheduler_and_manifest(source)
        operator = next(
            row for row in manifest.operators if row.kind == "ffn"
        )
        composite = RuntimeCompositeExecutorCapability(
            executor_id="coordinator:preloaded-helper",
            endpoint="synthetic://preloaded-helper",
            backend="backend:composite",
            coordinator_device_id="host-a",
            participant_device_ids=("host-a", "helper-c"),
            participant_resource_ids={
                "host-a": ("compute:host-a",),
                "helper-c": ("compute:helper-c",),
            },
            route_family="operator_split",
            assisted_operator_kind="ffn",
            split_axis="column",
            split_fractions_ppm=(500_000,),
            layer_fractions_ppm=(),
            residency_states=("cold", "hot", "warm"),
            resource_ids=("compute:host-a", "compute:helper-c"),
            operator_plan_protocol="synthetic-composite-v1",
            maturity="QUALIFIED",
            evidence_ids=("synthetic-preloaded-helper",),
            artifact_sha256=manifest.artifact_sha256,
            operator_placements=(RuntimeCompositeOperatorPlacement(
                operator_id=operator.operator_id,
                primary_device_id="host-a",
                helper_device_id="helper-c",
                split_axis="column",
                split_fraction_ppm=500_000,
                assisted=True,
            ),),
        )
        profile = replace(source, composite_executors=(composite,))
        rows = model_residency_observations(
            profile,
            manifest,
            composite.executor_id,
            generation=1,
            resident_device_ids=("helper-c",),
        )
        self.assertEqual(tuple(row.device_id for row in rows), ("helper-c",))
        self.assertGreater(rows[0].resident_bytes, 0)
        with self.assertRaisesRegex(
            PhysicalAdapterError, "not an executor participant"
        ):
            model_residency_observations(
                profile,
                manifest,
                composite.executor_id,
                generation=1,
                resident_device_ids=("missing-device",),
            )

    def test_dynamic_composite_residency_uses_exact_execution_plan(
        self,
    ) -> None:
        source = catalog()
        _, manifest = self.scheduler_and_manifest(source)
        ffn = next(row for row in manifest.operators if row.kind == "ffn")
        composite = RuntimeCompositeExecutorCapability(
            executor_id="coordinator:dynamic-residency",
            endpoint="synthetic://dynamic-residency",
            backend="backend:composite",
            coordinator_device_id="host-a",
            participant_device_ids=("host-a", "helper-c"),
            participant_resource_ids={
                "host-a": ("compute:host-a",),
                "helper-c": ("compute:helper-c",),
            },
            route_family="operator_split",
            assisted_operator_kind="ffn",
            split_axis="column",
            split_fractions_ppm=(500_000,),
            layer_fractions_ppm=(),
            residency_states=("cold", "hot", "warm"),
            resource_ids=("compute:host-a", "compute:helper-c"),
            operator_plan_protocol="synthetic-composite-v1",
            maturity="QUALIFIED",
            evidence_ids=("synthetic-dynamic-residency",),
            artifact_sha256=manifest.artifact_sha256,
            operator_ids=(ffn.operator_id,),
        )
        profile = replace(source, composite_executors=(composite,))
        operators = []
        for operator in manifest.operators:
            split = operator.operator_id == ffn.operator_id
            operators.append({
                "candidate_id": "synthetic:" + operator.operator_id,
                "device_ids": (
                    ["helper-c", "host-a"] if split else ["host-a"]
                ),
                "operator_id": operator.operator_id,
                "operator_kind": operator.kind,
                "split_axis": "column" if split else "none",
                "split_fraction_ppm": 500_000 if split else 0,
            })
        plan = {
            "memory_demands": [
                {
                    "device_id": "host-a",
                    "kind": "model_weights",
                    "required_bytes": 1000,
                },
                {
                    "device_id": "helper-c",
                    "kind": "model_weights",
                    "required_bytes": 250,
                },
            ],
            "operators": operators,
            "schema": "research-scheduler-execution-plan-v1",
        }

        rows = model_residency_observations(
            profile,
            manifest,
            composite.executor_id,
            generation=4,
            operator_plan=plan,
        )

        by_device = {row.device_id: row for row in rows}
        self.assertEqual(by_device["host-a"].resident_bytes, 1000)
        self.assertEqual(by_device["helper-c"].resident_bytes, 250)
        self.assertEqual(
            set(by_device["helper-c"].resident_tensor_ids),
            set(ffn.tensor_ids),
        )
        with self.assertRaisesRegex(
            PhysicalAdapterError,
            "runtime composite residency contract is incomplete",
        ):
            model_residency_observations(
                profile,
                manifest,
                composite.executor_id,
                generation=4,
            )

    def test_exact_shape_profile_overrides_prior_and_mismatch_stays_shadow(self) -> None:
        value = catalog().to_json()
        kernels = value["placement_profile"]["kernels"]
        prior = next(
            row for row in kernels
            if row["profile_id"] == "kernel:helper-c:ffn"
        )
        prior["status"] = "estimated"
        exact = dict(prior)
        exact.update({
            "effective_bytes_per_s": 9_000_000_000,
            "effective_ops_per_s": 9_000_000_000,
            "evidence_ids": ["exact-shape-evidence"],
            "kernel_id": "kernel:helper-c:ffn:exact-12-4",
            "profile_id": "kernel:helper-c:ffn:exact-12-4",
            "status": "measured",
        })
        kernels.append(exact)
        helper = next(
            row for row in value["executors"]
            if row["device_id"] == "helper-c"
        )
        helper["kernel_shape_profiles"] = [RuntimeKernelShapeProfile(
            selector_id="helper-ffn-exact-12-4",
            operator_kind="ffn",
            profile_id="kernel:helper-c:ffn:exact-12-4",
            minimum_input_tokens=12,
            maximum_input_tokens=12,
            minimum_output_tokens=4,
            maximum_output_tokens=4,
            minimum_compute_ops=1,
            maximum_compute_ops=2**63 - 1,
            minimum_memory_bytes=1,
            maximum_memory_bytes=2**63 - 1,
            maturity="QUALIFIED",
            evidence_ids=("exact-shape-evidence",),
        ).to_json()]
        profile = RuntimeCapabilityCatalog.from_json(value)
        scheduler, manifest = self.scheduler_and_manifest(profile)

        exact_rows = scheduler.generate_automated_candidates(
            request("exact-shape"), manifest.model_id, runtime_snapshot(manifest)
        ).candidates
        exact_phone_ffn = [
            row for row in exact_rows
            if "helper-c" in row.device_ids
            and any(
                assignment.operator_kind == "ffn"
                and "helper-c" in assignment.device_ids
                for assignment in row.plan.operators
            )
        ]
        self.assertTrue(any(
            "kernel:helper-c:ffn:exact-12-4"
            in {
                profile_id
                for assignment in row.plan.operators
                for profile_id in assignment.kernel_profile_ids
            }
            for row in exact_phone_ffn
        ))
        self.assertTrue(any(row.admitted for row in exact_phone_ffn))

        mismatch_rows = scheduler.generate_automated_candidates(
            request("shape-mismatch", input_tokens=13),
            manifest.model_id,
            runtime_snapshot(manifest),
        ).candidates
        mismatch_phone_ffn = [
            row for row in mismatch_rows
            if "helper-c" in row.device_ids
            and any(
                assignment.operator_kind == "ffn"
                and "helper-c" in assignment.device_ids
                for assignment in row.plan.operators
            )
        ]
        self.assertTrue(mismatch_phone_ffn)
        self.assertTrue(all(not row.admitted for row in mismatch_phone_ffn))
        self.assertTrue(all(
            "ROUTE_NOT_QUALIFIED" in row.rejection_reasons
            for row in mismatch_phone_ffn
        ))

    def test_measured_route_shape_overrides_operator_priors(self) -> None:
        source = catalog(
            phone_ops_per_s=8_000_000_000,
            phone_power_mw=1_000,
            phone_bandwidth=8_000_000_000,
        )
        value = source.to_json()
        for kernel in value["placement_profile"]["kernels"]:
            if kernel["device_id"] == "helper-c":
                kernel["status"] = "estimated"
        shadow = RuntimeCapabilityCatalog.from_json(value)
        route_profile = RuntimeRouteShapeProfile(
            selector_id="measured-phone-whole-12-4",
            artifact_sha256=(
                "sha256:" + hashlib.sha256(
                    self.path.read_bytes()
                ).hexdigest()
            ),
            route_family="whole_model",
            device_ids=("helper-c",),
            assisted_operator_kind=None,
            split_axis="none",
            split_fraction_ppm=0,
            residency_variant="hot",
            minimum_input_tokens=12,
            maximum_input_tokens=12,
            minimum_output_tokens=4,
            maximum_output_tokens=4,
            service_fixed_us=100,
            service_input_token_us=10,
            service_output_token_us=20,
            service_upper_add_us=30,
            energy_fixed_uj=1_000,
            energy_input_token_uj=10,
            energy_output_token_uj=20,
            energy_lower_error_ppm=100_000,
            energy_upper_error_ppm=100_000,
            sample_count=8,
            maturity="QUALIFIED",
            evidence_ids=("held-out-phone-route",),
        )
        measured = replace(
            shadow, route_shape_profiles=(route_profile,)
        )
        measured = RuntimeCapabilityCatalog.from_json(measured.to_json())
        self.assertEqual(
            measured.route_shape_profiles, (route_profile,)
        )
        scheduler, manifest = self.scheduler_and_manifest(measured)
        ticket = scheduler.submit_automated_request(
            request("measured-route-shape"),
            manifest.model_id,
            runtime_snapshot(manifest, phone_bandwidth=8_000_000_000),
        )
        self.assertEqual(ticket.execution_plan.device_ids, ("helper-c",))
        self.assertEqual(
            ticket.execution_plan.route_profile_id,
            route_profile.selector_id,
        )
        self.assertEqual(ticket.decision.service_us, 300)
        self.assertEqual(ticket.decision.energy_uj, 1_200)
        self.assertEqual(ticket.decision.energy_upper_uj, 1_320)
        estimate = next(
            row for row in ticket.cost_estimates.estimates
            if row.route_id == ticket.decision.route_id
        )
        self.assertEqual(estimate.latency_sample_count, 8)
        self.assertEqual(
            estimate.latency_profile_label,
            route_profile.selector_id,
        )

        mismatch = scheduler.generate_automated_candidates(
            request("measured-route-mismatch", input_tokens=13),
            manifest.model_id,
            runtime_snapshot(manifest, phone_bandwidth=8_000_000_000),
        )
        phone_whole = next(
            row for row in mismatch.candidates
            if row.route_family == "whole_model"
            and row.device_ids == ("helper-c",)
            and row.residency_variant == "hot"
        )
        self.assertIsNone(phone_whole.plan.route_profile_id)
        self.assertFalse(phone_whole.admitted)
        self.assertIn(
            "ROUTE_NOT_QUALIFIED", phone_whole.rejection_reasons
        )

        other_path = Path(self.directory.name) / "other-model.gguf"
        write_synthetic_gguf(other_path, sliding_window=3)
        other_manifest = scheduler.register_gguf_model(
            "other-unseen-model", other_path
        )
        other = scheduler.generate_automated_candidates(
            request("measured-route-other-model"),
            other_manifest.model_id,
            runtime_snapshot(other_manifest),
        )
        other_phone = next(
            row for row in other.candidates
            if row.route_family == "whole_model"
            and row.device_ids == ("helper-c",)
            and row.residency_variant == "hot"
        )
        self.assertIsNone(other_phone.plan.route_profile_id)
        self.assertFalse(other_phone.admitted)

    def test_route_shape_cost_uses_live_snapshot_features(self) -> None:
        value = catalog(
            phone_ops_per_s=8_000_000_000,
            phone_power_mw=1_000,
            phone_bandwidth=8_000_000_000,
        ).to_json()
        for kernel in value["placement_profile"]["kernels"]:
            if kernel["device_id"] == "helper-c":
                kernel["status"] = "estimated"
        source = RuntimeCapabilityCatalog.from_json(value)
        artifact_sha256 = "sha256:" + hashlib.sha256(
            self.path.read_bytes()
        ).hexdigest()

        def profile(phase: int) -> RuntimeRouteShapeProfile:
            return RuntimeRouteShapeProfile(
                selector_id=f"phone-route-phase-{phase}",
                artifact_sha256=artifact_sha256,
                route_family="whole_model",
                device_ids=("helper-c",),
                assisted_operator_kind=None,
                split_axis="none",
                split_fraction_ppm=0,
                residency_variant="hot",
                minimum_input_tokens=1,
                maximum_input_tokens=128,
                minimum_output_tokens=1,
                maximum_output_tokens=128,
                service_fixed_us=100,
                service_input_token_us=10,
                service_output_token_us=20,
                service_upper_add_us=30,
                energy_fixed_uj=1_000,
                energy_input_token_uj=10,
                energy_output_token_uj=20,
                energy_lower_error_ppm=100_000,
                energy_upper_error_ppm=100_000,
                sample_count=8,
                maturity="QUALIFIED",
                evidence_ids=(f"held-out-phase-{phase}",),
                feature_ranges={"contention_class": (phase, phase)},
                service_feature_coefficients_us={
                    "contention_class": 100
                },
                energy_feature_coefficients_uj={
                    "contention_class": 200
                },
            )

        profiles = (profile(1), profile(2))
        scheduler, manifest = self.scheduler_and_manifest(replace(
            source, route_shape_profiles=profiles
        ))
        observed = []
        for phase in (1, 2):
            snapshot = replace(
                runtime_snapshot(manifest),
                cost_features={"contention_class": phase},
            )
            candidates = scheduler.generate_automated_candidates(
                request(f"route-feature-{phase}"),
                manifest.model_id,
                snapshot,
            )
            row = next(
                candidate for candidate in candidates.candidates
                if candidate.route_family == "whole_model"
                and candidate.device_ids == ("helper-c",)
                and candidate.residency_variant == "hot"
            )
            self.assertTrue(row.admitted)
            self.assertEqual(
                row.plan.route_profile_id,
                f"phone-route-phase-{phase}",
            )
            observed.append((row.cost.service_us, row.cost.fleet_energy_uj))
        self.assertEqual(observed, [(400, 1_400), (500, 1_600)])

    def test_energy_latency_alpha_controls_energy_positive_route(self) -> None:
        source = catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        )
        strict, strict_manifest = self.scheduler_and_manifest(replace(
            source, maximum_latency_ppm=1_000_000
        ))
        strict_ticket = strict.submit_automated_request(
            request("strict-alpha"),
            strict_manifest.model_id,
            runtime_snapshot(
                strict_manifest, phone_bandwidth=8_000_000_000
            ),
        )
        relaxed, relaxed_manifest = self.scheduler_and_manifest(replace(
            source, maximum_latency_ppm=1_500_000
        ))
        relaxed_ticket = relaxed.submit_automated_request(
            request("relaxed-alpha"),
            relaxed_manifest.model_id,
            runtime_snapshot(
                relaxed_manifest, phone_bandwidth=8_000_000_000
            ),
        )
        moderate, moderate_manifest = self.scheduler_and_manifest(replace(
            source, maximum_latency_ppm=1_250_000
        ))
        moderate_ticket = moderate.submit_automated_request(
            request("moderate-alpha"),
            moderate_manifest.model_id,
            runtime_snapshot(
                moderate_manifest, phone_bandwidth=8_000_000_000
            ),
        )
        energy_first, energy_first_manifest = self.scheduler_and_manifest(
            replace(source, maximum_latency_ppm=1_000_000)
        )
        energy_first_ticket = energy_first.submit_automated_request(
            request("energy-first"),
            energy_first_manifest.model_id,
            runtime_snapshot(
                energy_first_manifest, phone_bandwidth=8_000_000_000
            ),
            selection_mode="energy-first",
        )

        self.assertEqual(
            strict_ticket.execution_plan.device_ids,
            ("accelerator-b",),
        )
        self.assertEqual(
            relaxed_ticket.execution_plan.device_ids,
            ("helper-c",),
        )
        self.assertEqual(
            moderate_ticket.execution_plan.device_ids,
            ("helper-c",),
        )
        self.assertEqual(
            energy_first_ticket.execution_plan.device_ids,
            ("helper-c",),
        )
        self.assertEqual(
            energy_first_ticket.decision.reason,
            "CONSERVATIVE_FLEET_ENERGY_FIRST",
        )

    def test_waiting_route_cancels_idle_energy_common_with_protected_work(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
            system_cost_profiles=(system_cost_profile(
                helper_interference_ppm=0,
            ),),
        ))
        snapshot = protected_snapshot(manifest)
        # The accelerator is busy until the protected work ends; the GPU route
        # must wait and its idle charge during that wait is common to both
        # scenarios.
        snapshot = replace(snapshot, executors={
            **snapshot.executors,
            "executor:accelerator-b": executor_state(
                "executor:accelerator-b", busy_until_us=3_000_000, free_slots=0,
            ),
        })
        ticket = scheduler.submit_automated_request(
            request("waiting-cost"), manifest.model_id, snapshot,
        )
        waiting = [
            row for row in ticket.cost_estimates.estimates
            if row.details.get("marginal_system_cost")
            and row.details["cost_breakdown"]["queue_delay_us"] > 0
        ]
        self.assertTrue(waiting)
        for row in waiting:
            marginal = row.details["marginal_system_cost"]
            queue = row.details["cost_breakdown"]["queue_delay_us"]
            self.assertGreater(marginal["protected_overlap_us"], 0)
            self.assertLessEqual(marginal["protected_overlap_us"], queue)
            self.assertLessEqual(
                marginal["protected_overlap_us"],
                snapshot.protected_work.critical_path_end_us,
            )
            self.assertGreater(marginal["protected_overlap_idle_uj"], 0)
            self.assertEqual(
                marginal["route_energy_incremental_lower_uj"],
                max(0, marginal["route_energy_lower_uj"] - marginal["protected_overlap_idle_uj"]),
            )
            self.assertEqual(
                marginal["route_energy_incremental_upper_uj"],
                max(0, marginal["route_energy_upper_uj"] - marginal["protected_overlap_idle_uj"]),
            )

    def test_slow_energy_negative_phone_loses_to_local_fallback(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=50_000_000,
            phone_power_mw=80_000,
            phone_bandwidth=100_000,
        ))
        ticket = scheduler.submit_automated_request(
            request("phone-loses"),
            manifest.model_id,
            runtime_snapshot(manifest, phone_bandwidth=100_000),
        )
        self.assertEqual(len(ticket.execution_plan.device_ids), 1)
        self.assertNotIn("helper-c", ticket.execution_plan.device_ids)
        phone_rows = [
            row for row in ticket.cost_estimates.estimates
            if "helper-c" in row.details.get("device_ids", [])
        ]
        self.assertTrue(phone_rows)
        self.assertTrue(all(
            row.route_id != ticket.decision.route_id for row in phone_rows
        ))

    def test_three_device_layer_cuts_come_from_capability_contracts(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        candidates = scheduler.generate_automated_candidates(
            request("supported-three-way-cuts"),
            manifest.model_id,
            runtime_snapshot(manifest),
        )
        three_way = [
            row for row in candidates.candidates
            if row.route_family == "layer_placement"
            and len(row.device_ids) == 3
        ]
        self.assertTrue(three_way)
        supported = {250_000, 500_000, 750_000}
        for row in three_way:
            encoded = row.candidate_id.split(":")[-3]
            cuts = {int(value) for value in encoded.split("+")}
            self.assertEqual(len(cuts), 2)
            self.assertLessEqual(cuts, supported)

    def test_additive_device_registration_affects_later_arrivals(self) -> None:
        complete_catalog = catalog()
        initial_catalog = replace(
            complete_catalog,
            executors=tuple(
                row for row in complete_catalog.executors
                if row.device_id != "helper-c"
            ),
            transitions=tuple(
                row for row in complete_catalog.transitions
                if row.device_id != "helper-c"
            ),
        )
        scheduler, manifest = self.scheduler_and_manifest(initial_catalog)
        first = scheduler.submit_automated_request(
            request("before-device-connect"),
            manifest.model_id,
            runtime_snapshot(manifest, include_phone=False),
        )
        self.assertNotIn("helper-c", first.execution_plan.device_ids)

        scheduler.register_runtime_capabilities(complete_catalog)
        later = scheduler.generate_automated_candidates(
            request("after-device-connect", arrival_us=2_000),
            manifest.model_id,
            runtime_snapshot(manifest),
        )
        self.assertTrue(any(
            "helper-c" in row.device_ids for row in later.candidates
        ))
        self.assertEqual(
            scheduler.runtime_ticket(first.request.request_id).binding,
            first.binding,
        )

    def test_selected_physical_plan_and_journal_are_hash_bound(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        ))
        ticket = scheduler.submit_automated_request(
            request("journal-plan"),
            manifest.model_id,
            runtime_snapshot(manifest, phone_bandwidth=8_000_000_000),
        )
        decision_record = scheduler.runtime_decision_log()["records"][0]
        self.assertEqual(decision_record["event_kind"], "DECISION")
        self.assertEqual(
            decision_record["selected"]["operator_plan"],
            ticket.execution_plan.to_json(),
        )
        unhashed_plan = dict(ticket.execution_plan.to_json())
        plan_sha256 = unhashed_plan.pop("plan_sha256")
        self.assertEqual(
            plan_sha256,
            "sha256:" + hashlib.sha256(json.dumps(
                unhashed_plan,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")).hexdigest(),
        )
        self.assertEqual(
            decision_record["selected"]["executor"],
            ticket.binding.to_json(),
        )
        self.assertEqual(
            len(decision_record["candidates"]),
            len(ticket.cost_estimates.estimates),
        )
        self.assertTrue(all(
            row.get("details", {}).get("route_family")
            for row in decision_record["candidates"]
        ))
        self.assertTrue(all(
            row.get("details", {}).get("operator_plan_sha256")
            and row.get("details", {}).get("resource_ids")
            and "transitions" in row.get("details", {})
            and "operator_plan" not in row.get("details", {})
            for row in decision_record["candidates"]
        ))

        epoch_ns = time.monotonic_ns() - ticket.decision.start_us * 1_000
        active = scheduler.wait_runtime_request(ticket.request.request_id, epoch_ns)
        scheduler.complete_automated_request(
            ticket.request.request_id,
            self.execution_receipt(active),
        )
        self.assertEqual(
            scheduler.runtime_memory_state()["reservations"], []
        )
        events = [
            row["event_kind"]
            for row in scheduler.runtime_decision_log()["records"]
        ]
        self.assertEqual(events, ["DECISION", "ACQUIRED", "COMPLETED"])

    def test_measured_receipts_update_later_shape_bucket_costs(self) -> None:
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
            row_request = request(
                "learn-route-" + str(index),
                arrival_us=1_000 + index * 10_000,
                input_tokens=12,
                output_tokens=4,
            )
            ticket = scheduler.submit_automated_request(
                row_request, manifest.model_id, snapshot
            )
            epoch_ns = time.monotonic_ns() - ticket.decision.start_us * 1_000
            active = scheduler.wait_runtime_request(
                ticket.request.request_id, epoch_ns
            )
            observed_route = active.decision.route_id
            scheduler.complete_automated_request(
                active.request.request_id,
                self.measured_execution_receipt(
                    active,
                    latency_us=active.decision.service_us,
                    fleet_energy_uj=9_000 + index * 10,
                ),
            )

        later = scheduler.generate_automated_candidates(
            request(
                "learn-route-later",
                arrival_us=100_000,
                input_tokens=12,
                output_tokens=4,
            ),
            manifest.model_id,
            snapshot,
        )
        learned = next(
            row for row in later.candidates
            if row.candidate_id == observed_route
        )
        self.assertTrue(learned.plan.route_profile_id.startswith("online:"))
        self.assertEqual(learned.maturity, "QUALIFIED")
        self.assertGreaterEqual(
            learned.cost.service_upper_us, learned.cost.service_us
        )
        self.assertGreaterEqual(learned.cost.fleet_energy_upper_uj, 9_030)

    def test_qualified_transition_evidence_materializes_candidate(self) -> None:
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
        runtime_request = request(
            "consume-transition-evidence",
            input_tokens=12,
            output_tokens=4,
        )
        selected = next(
            row for row in scheduler.generate_automated_candidates(
                runtime_request,
                manifest.model_id,
                snapshot,
            ).candidates
            if row.route_family == "operator_split"
            and "helper-c" in row.device_ids
            and row.plan.transitions
        )
        compiler = scheduler._automated_compiler()
        transition = selected.plan.transitions[0]
        participant = next(
            row for row in selected.binding.participants
            if row.device_id == transition.device_id
        )
        domains = (
            "energy:accelerator-b",
            "energy:helper-c",
            "energy:host-a",
        )
        for index, (latency_us, energy_uj) in enumerate(zip(
            (1_000, 1_020, 990, 1_010),
            (600, 620, 590, 610),
        )):
            shares = (energy_uj // 3, energy_uj // 3)
            transition_receipt = RuntimeTransitionReceipt(
                ticket_id="consumed-transition-ticket-" + str(index),
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
                finished_us=latency_us,
                status="COMPLETED",
                energy_boundary_id="synthetic-whole-fleet",
                fleet_energy_uj_by_domain={
                    domains[0]: shares[0],
                    domains[1]: shares[1],
                    domains[2]: energy_uj - sum(shares),
                },
                measurement_evidence_ids=(
                    "consumed-transition-" + str(index),
                ),
                energy_attribution_kind="isolated",
            )
            self.assertEqual(
                compiler.record_transition_observations(
                    manifest, selected.plan, (transition_receipt,)
                ),
                1,
            )
            warm_energy_uj = (3_000, 3_020, 2_980, 3_010)[index]
            warm_shares = (
                warm_energy_uj // 3,
                warm_energy_uj // 3,
            )
            execution_receipt = RuntimeExecutionReceipt(
                ticket_id="consumed-warm-ticket-" + str(index),
                request_id=runtime_request.request_id,
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
                finished_us=10_000 + index * 10,
                output_sha256="sha256:" + format(index + 10, "064x"),
                status="COMPLETED",
                energy_boundary_id="synthetic-whole-fleet",
                fleet_energy_uj_by_domain={
                    domains[0]: warm_shares[0],
                    domains[1]: warm_shares[1],
                    domains[2]: warm_energy_uj - sum(warm_shares),
                },
                measurement_evidence_ids=(
                    "consumed-warm-" + str(index),
                ),
                energy_attribution_kind="isolated",
                energy_scope="warm_execution",
            )
            self.assertTrue(compiler.record_execution_observation(
                runtime_request,
                manifest,
                selected.plan,
                execution_receipt,
                snapshot.cost_features,
                selected.cost.component_service_us,
                selected.cost.component_energy_uj,
            ))

        updated = next(
            row for row in scheduler.generate_automated_candidates(
                runtime_request,
                manifest.model_id,
                snapshot,
            ).candidates
            if row.candidate_id == selected.candidate_id
        )
        self.assertEqual(updated.plan.transitions[0].latency_us, 1_005)
        self.assertEqual(updated.plan.transitions[0].energy_uj, 605)
        self.assertEqual(
            updated.plan.transitions[0].energy_maturity,
            "QUALIFIED",
        )
        self.assertEqual(updated.cost.transition_energy_uj, 605)
        self.assertEqual(updated.cost.transition_energy_lower_uj, 560)
        self.assertEqual(updated.cost.transition_energy_upper_uj, 682)
        self.assertEqual(updated.cost.energy_evidence, "MEASURED")

    def test_route_energy_has_explicit_warm_transition_decomposition(
        self,
    ) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        candidates = scheduler.generate_automated_candidates(
            request("explicit-energy-decomposition"),
            manifest.model_id,
            runtime_snapshot(manifest),
        )
        known = tuple(
            row for row in candidates.candidates
            if row.cost.fleet_energy_uj is not None
        )
        self.assertTrue(known)
        for row in known:
            self.assertEqual(
                row.cost.fleet_energy_uj,
                row.cost.warm_execution_energy_uj
                    + row.cost.transition_energy_uj,
            )
            self.assertEqual(
                row.cost.fleet_energy_lower_uj,
                row.cost.warm_execution_energy_lower_uj
                    + row.cost.transition_energy_lower_uj,
            )
            self.assertEqual(
                row.cost.fleet_energy_upper_uj,
                row.cost.warm_execution_energy_upper_uj
                    + row.cost.transition_energy_upper_uj,
            )
        cold = next(
            row for row in known
            if row.residency_variant == "cold" and row.plan.transitions
        )
        self.assertGreater(cold.cost.warm_execution_energy_upper_uj, 1)
        with self.assertRaises(RuntimePlanError):
            replace(
                cold.cost,
                fleet_energy_upper_uj=(
                    cold.cost.fleet_energy_upper_uj + 1
                ),
            )

    def test_structural_route_catalog_is_reused_across_request_lengths(
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
        scheduler.generate_automated_candidates(
            request(
                "catalog-long-decode",
                input_tokens=310,
                output_tokens=341,
            ),
            manifest.model_id,
            snapshot,
        )
        after_long = scheduler.automated_cost_cache_stats()
        scheduler.generate_automated_candidates(
            request(
                "catalog-short-decode",
                input_tokens=295,
                output_tokens=133,
            ),
            manifest.model_id,
            snapshot,
        )
        after_short = scheduler.automated_cost_cache_stats()

        self.assertEqual(
            after_short["template_entries"],
            after_long["template_entries"],
        )
        self.assertLess(
            after_short["operator_assignment_entries"]
            - after_long["operator_assignment_entries"],
            after_long["operator_assignment_entries"],
        )

    def test_template_learning_separates_prefill_and_decode_costs(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        ))
        snapshot = runtime_snapshot(
            manifest, phone_bandwidth=8_000_000_000
        )
        store = RuntimeRouteObservationStore()
        generation = "sha256:" + "b" * 64
        samples = (
            (915, 292, 45_764_176, 5_454_654_499,
             138_048_372, 16_917_835_154),
            (309, 11, 11_595_140, 1_374_821_373,
             11_770_232, 2_041_231_765),
            (271, 41, 11_312_663, 1_339_945_928,
             24_312_848, 3_503_903_922),
            (271, 41, 11_312_663, 1_339_945_928,
             24_000_000, 3_400_000_000),
        )
        selected_route = None
        selected_plan = None
        for index, (
            input_tokens,
            output_tokens,
            component_service_us,
            component_energy_uj,
            latency_us,
            energy_uj,
        ) in enumerate(samples):
            candidates = scheduler.generate_automated_candidates(
                request(
                    "phase-learning-" + str(index),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
                manifest.model_id,
                snapshot,
            )
            if selected_route is None:
                selected = next(
                    row for row in candidates.candidates
                    if row.route_family == "operator_split"
                    and "helper-c" in row.device_ids
                )
                selected_route = selected.candidate_id
            else:
                selected = next(
                    row for row in candidates.candidates
                    if row.candidate_id == selected_route
                )
            selected_plan = selected.plan
            per_domain = energy_uj // 3
            receipt = RuntimeExecutionReceipt(
                ticket_id="phase-learning-ticket-" + str(index),
                request_id="phase-learning-" + str(index),
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
                    "energy:accelerator-b": per_domain,
                    "energy:helper-c": per_domain,
                    "energy:host-a": energy_uj - 2 * per_domain,
                },
                transfer_energy_uj_by_link={
                    resource_id.removeprefix("link:"): 1
                    for resource_id in selected.plan.resource_ids
                    if resource_id.startswith("link:")
                },
                measurement_evidence_ids=(
                    "phase-learning-isolated-" + str(index),
                ),
                energy_attribution_kind="isolated",
            )
            self.assertTrue(store.record(
                artifact_sha256=manifest.artifact_sha256,
                capability_generation_sha256=generation,
                plan=selected.plan,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                quality_requirement="exact",
                cost_features={},
                receipt=receipt,
                energy_boundary_id="synthetic-whole-fleet",
                required_domain_ids=(
                    "energy:accelerator-b",
                    "energy:helper-c",
                    "energy:host-a",
                ),
                component_service_us=component_service_us,
                component_energy_uj=component_energy_uj,
            ))

        learned = store.estimate(
            artifact_sha256=manifest.artifact_sha256,
            capability_generation_sha256=generation,
            plan=selected_plan,
            input_tokens=271,
            output_tokens=41,
            quality_requirement="exact",
            cost_features={},
            component_service_us=11_312_663,
            component_energy_uj=1_339_945_928,
        )
        self.assertIsNotNone(learned)
        self.assertTrue(learned.profile_id.startswith("online-template:"))
        self.assertEqual(learned.maturity, "QUALIFIED")
        self.assertLessEqual(learned.latency_mape_ppm, 200_000)
        self.assertLessEqual(learned.energy_mape_ppm, 200_000)
        self.assertEqual(learned.latency_upper_coverage_ppm, 1_000_000)
        self.assertEqual(learned.energy_upper_coverage_ppm, 1_000_000)

    def test_qualified_exact_shape_precedes_qualified_template(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        ))
        candidates = scheduler.generate_automated_candidates(
            request("exact-shape-plan", input_tokens=8, output_tokens=2),
            manifest.model_id,
            runtime_snapshot(manifest, phone_bandwidth=8_000_000_000),
        )
        selected = next(
            row for row in candidates.candidates
            if row.admitted and row.route_family == "operator_split"
        )
        store = RuntimeRouteObservationStore()
        generation = "sha256:" + "b" * 64
        domains = (
            "energy:accelerator-b",
            "energy:helper-c",
            "energy:host-a",
        )
        samples = (
            (8, 2, 100),
            (12, 4, 200),
            (8, 2, 100),
            (12, 4, 200),
            (8, 2, 100),
            (8, 2, 100),
        )
        for index, (input_tokens, output_tokens, component) in enumerate(
            samples
        ):
            observed_plan = replace(
                selected.plan,
                route_profile_id=(
                    "synthetic-profile-" + str(index % 2)
                ),
            )
            energy = component * 20
            receipt = RuntimeExecutionReceipt(
                ticket_id="exact-shape-ticket-" + str(index),
                request_id="exact-shape-request-" + str(index),
                artifact_sha256=manifest.artifact_sha256,
                operator_plan_sha256=observed_plan.plan_sha256,
                executor_id=selected.binding.executor_id,
                endpoint=selected.binding.endpoint,
                operator_plan_protocol=(
                    selected.binding.operator_plan_protocol
                ),
                participant_executor_ids=tuple(
                    row.executor_id for row in selected.binding.participants
                ),
                started_us=0,
                finished_us=component * 10,
                output_sha256="sha256:" + format(index, "064x"),
                status="COMPLETED",
                energy_boundary_id="synthetic-whole-fleet",
                fleet_energy_uj_by_domain={
                    domains[0]: energy // 3,
                    domains[1]: energy // 3,
                    domains[2]: energy - 2 * (energy // 3),
                },
                measurement_evidence_ids=(
                    "exact-shape-isolated-" + str(index),
                ),
                energy_attribution_kind="isolated",
            )
            self.assertTrue(store.record(
                artifact_sha256=manifest.artifact_sha256,
                capability_generation_sha256=generation,
                plan=observed_plan,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                quality_requirement="exact",
                cost_features={},
                receipt=receipt,
                energy_boundary_id="synthetic-whole-fleet",
                required_domain_ids=domains,
                component_service_us=component,
                component_energy_uj=component,
            ))

        learned = store.estimate(
            artifact_sha256=manifest.artifact_sha256,
            capability_generation_sha256=generation,
            plan=selected.plan,
            input_tokens=8,
            output_tokens=2,
            quality_requirement="exact",
            cost_features={},
            component_service_us=100,
            component_energy_uj=100,
        )

        self.assertIsNotNone(learned)
        self.assertEqual(learned.maturity, "QUALIFIED")
        self.assertTrue(learned.profile_id.startswith("online:"))
        self.assertFalse(learned.profile_id.startswith("online-template:"))
        self.assertGreaterEqual(store.state()["qualified_shape_buckets"], 1)

        exported = store.to_json()
        body = dict(exported)
        body.pop("store_sha256")
        body["schema"] = "runtime-route-observations-v3"
        exact_row = next(
            row for row in body["rows"]
            if len(row["observations"]) == 4
        )
        body["rows"].remove(exact_row)
        for index, observations in enumerate((
            exact_row["observations"][:2],
            exact_row["observations"][2:],
        )):
            key = list(exact_row["key"])
            key[-1] = "sha256:" + str(index + 1) * 64
            body["rows"].append({
                "key": key,
                "observations": observations,
            })
        legacy = {
            **body,
            "store_sha256": canonical_sha256(body),
        }
        migrated = RuntimeRouteObservationStore()
        migrated.import_json(legacy)
        self.assertGreaterEqual(
            migrated.state()["qualified_shape_buckets"], 1
        )
        self.assertEqual(
            migrated.to_json()["schema"],
            "runtime-route-observations-v6",
        )

    def test_measured_observations_persist_across_physical_campaigns(self) -> None:
        profile = catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        )
        scheduler, manifest = self.scheduler_and_manifest(profile)
        observed_route = None
        for index, utilization in enumerate((1, 8)):
            snapshot = replace(
                runtime_snapshot(
                    manifest, phone_bandwidth=8_000_000_000
                ),
                cost_features={"cpu_utilization_pct": utilization},
            )
            ticket = scheduler.submit_automated_request(
                request(
                    "persist-before-" + str(index),
                    arrival_us=1_000 + index * 10_000,
                ),
                manifest.model_id,
                snapshot,
            )
            active = scheduler.wait_runtime_request(
                ticket.request.request_id,
                time.monotonic_ns() - ticket.decision.start_us * 1_000,
            )
            observed_route = active.decision.route_id
            scheduler.complete_automated_request(
                active.request.request_id,
                self.measured_execution_receipt(
                    active,
                    latency_us=active.decision.service_us,
                    fleet_energy_uj=9_000 + index * 10,
                ),
            )
        persisted = dict(scheduler.automated_observation_snapshot())

        resumed, resumed_manifest = self.scheduler_and_manifest(profile)
        resumed.load_automated_observations(persisted)
        for index, utilization in enumerate((2, 9), 2):
            snapshot = replace(
                runtime_snapshot(
                    resumed_manifest, phone_bandwidth=8_000_000_000
                ),
                cost_features={"cpu_utilization_pct": utilization},
            )
            ticket = resumed.submit_automated_request(
                request(
                    "persist-after-" + str(index),
                    arrival_us=1_000 + index * 10_000,
                ),
                resumed_manifest.model_id,
                snapshot,
            )
            active = resumed.wait_runtime_request(
                ticket.request.request_id,
                time.monotonic_ns() - ticket.decision.start_us * 1_000,
            )
            self.assertEqual(active.decision.route_id, observed_route)
            resumed.complete_automated_request(
                active.request.request_id,
                self.measured_execution_receipt(
                    active,
                    latency_us=active.decision.service_us,
                    fleet_energy_uj=9_000 + index * 10,
                ),
            )

        later = resumed.generate_automated_candidates(
            request("persist-later", arrival_us=100_000),
            resumed_manifest.model_id,
            replace(
                runtime_snapshot(
                    resumed_manifest, phone_bandwidth=8_000_000_000
                ),
                cost_features={"cpu_utilization_pct": 5},
            ),
        )
        learned = next(
            row for row in later.candidates
            if row.candidate_id == observed_route
        )
        self.assertEqual(learned.maturity, "QUALIFIED")
        self.assertEqual(
            resumed.automated_observation_state()["complete_receipts"], 4
        )

        tampered = dict(persisted)
        tampered["incomplete_receipts"] = 1
        empty, _ = self.scheduler_and_manifest(profile)
        with self.assertRaisesRegex(
            UnifiedScheduleError, "runtime observation store is invalid"
        ):
            empty.load_automated_observations(tampered)

    def test_measured_observation_is_bound_to_exact_execution_plan(self) -> None:
        profile = catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        )
        scheduler, manifest = self.scheduler_and_manifest(profile)
        snapshot = runtime_snapshot(
            manifest, phone_bandwidth=8_000_000_000
        )
        ticket = scheduler.submit_automated_request(
            request("plan-bound-observation"), manifest.model_id, snapshot
        )
        receipt = self.measured_execution_receipt(
            ticket,
            latency_us=ticket.decision.service_us,
            fleet_energy_uj=9_000,
        )
        generation = "sha256:" + "a" * 64
        store = RuntimeRouteObservationStore()
        self.assertTrue(store.record(
            artifact_sha256=manifest.artifact_sha256,
            capability_generation_sha256=generation,
            plan=ticket.execution_plan,
            input_tokens=ticket.request.input_tokens,
            output_tokens=ticket.request.output_tokens,
            quality_requirement=ticket.request.quality_requirement,
            cost_features=snapshot.cost_features,
            receipt=receipt,
            energy_boundary_id=profile.placement_profile.energy_boundary_id,
            required_domain_ids=tuple(profile.placement_profile.domains),
        ))
        changed_parameters = dict(ticket.execution_plan.adapter_parameters)
        changed_parameters["transport_generation"] = "synthetic-v2"
        changed_plan = replace(
            ticket.execution_plan,
            adapter_parameters=changed_parameters,
        )
        self.assertNotEqual(
            changed_plan.plan_sha256, ticket.execution_plan.plan_sha256
        )
        self.assertIsNone(store.estimate(
            artifact_sha256=manifest.artifact_sha256,
            capability_generation_sha256=generation,
            plan=changed_plan,
            input_tokens=ticket.request.input_tokens,
            output_tokens=ticket.request.output_tokens,
            quality_requirement=ticket.request.quality_requirement,
            cost_features=snapshot.cost_features,
        ))

    def test_isolated_fleet_energy_does_not_require_link_attribution(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        snapshot = runtime_snapshot(manifest)
        ticket = scheduler.submit_automated_request(
            request("fleet-without-link-attribution"),
            manifest.model_id,
            snapshot,
        )
        active = scheduler.wait_runtime_request(
            ticket.request.request_id,
            time.monotonic_ns() - ticket.decision.start_us * 1_000,
        )
        measured = self.measured_execution_receipt(
            active,
            latency_us=active.decision.service_us,
            fleet_energy_uj=9_000,
        )
        scheduler.complete_automated_request(
            active.request.request_id,
            replace(measured, transfer_energy_uj_by_link={}),
        )
        state = scheduler.automated_observation_state()
        self.assertEqual(state["complete_receipts"], 1)
        self.assertEqual(state["incomplete_receipts"], 0)
        self.assertEqual(state["unattributed_transfer_receipts"], 1)

    def test_qualified_cold_transition_can_prepare_a_stopped_endpoint(self) -> None:
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
        states = dict(snapshot.executors)
        states["executor:helper-c"] = replace(
            states["executor:helper-c"], ready=False, free_slots=0
        )
        snapshot = replace(snapshot, executors=states)
        ticket = scheduler.submit_automated_request(
            request("cold-stopped-phone"), manifest.model_id, snapshot
        )

        self.assertEqual(ticket.execution_plan.device_ids, ("helper-c",))
        self.assertEqual(ticket.execution_plan.residency_variant, "cold")
        self.assertTrue(ticket.execution_plan.transitions)
        self.assertEqual(ticket.transition_status, "PENDING")

    def test_audit_only_placement_receives_memory_revalidation(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        ))
        snapshot = runtime_snapshot(
            manifest, phone_bandwidth=8_000_000_000
        )
        candidates = scheduler.generate_automated_candidates(
            request("audit-only-memory-revalidation"),
            manifest.model_id,
            snapshot,
        )
        marker = "MODEL_EPOCH_AUDIT_ONLY"
        marked = replace(candidates, candidates=tuple(
            replace(
                row,
                binding=replace(
                    row.binding,
                    ready=False,
                    eligibility_reasons=(marker,),
                ),
                admitted=False,
                rejection_reasons=(marker,),
            )
            if "helper-c" in row.device_ids and row.admitted else row
            for row in candidates.candidates
        ))
        marked_phone_ids = {
            row.candidate_id for row in marked.candidates
            if row.rejection_reasons == (marker,)
        }
        self.assertTrue(marked_phone_ids)
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
            rejected = scheduler._runtime_memory_rejections(
                marked, snapshot
            )

        self.assertTrue(marked_phone_ids.issubset(rejected))
        self.assertTrue(all(
            rejected[route_id]
                == "MEMORY_CAPACITY_CURRENT:phone-memory"
            for route_id in marked_phone_ids
        ))

    def test_desktop_control_transition_accepts_same_artifact_fallback(
        self,
    ) -> None:
        source = catalog()
        _, manifest = self.scheduler_and_manifest()
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
        composite = RuntimeCompositeExecutorCapability(
            executor_id="coordinator:desktop-control",
            endpoint="synthetic://desktop-control",
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
            layer_fractions_ppm=(),
            residency_states=("cold", "hot", "warm"),
            resource_ids=("compute:host-a", "compute:accelerator-b"),
            operator_plan_protocol="synthetic-composite-v1",
            maturity="QUALIFIED",
            evidence_ids=("synthetic-desktop-control",),
            artifact_sha256=manifest.artifact_sha256,
            operator_placements=placements,
        )
        transition = RuntimeTransitionCapability(
            transition_id="load:desktop-control",
            device_id="accelerator-b",
            source_state="cold",
            target_state="hot",
            fixed_latency_us=100,
            bandwidth_bytes_per_s=1_000_000_000,
            fixed_energy_uj=100,
            dynamic_pj_per_byte=100,
            resource_ids=("compute:host-a", "compute:accelerator-b"),
            maturity="QUALIFIED",
            evidence_ids=("synthetic-desktop-control",),
            executor_id=composite.executor_id,
            prepares_device_ids=("host-a", "accelerator-b"),
        )
        profile = replace(
            source,
            composite_executors=(composite,),
            desktop_control_profiles=(RuntimeDesktopControlProfile(
                profile_id="synthetic-desktop-control",
                artifact_sha256=manifest.artifact_sha256,
                executor_id=composite.executor_id,
                operator_placements=placements,
                maturity="QUALIFIED",
                evidence_ids=("synthetic-desktop-control",),
            ),),
            transitions=source.transitions + (transition,),
        )
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        scheduler.register_model_manifest(manifest)
        snapshot = runtime_snapshot(
            manifest,
            include_phone=False,
            resident_devices=(),
        )
        states = dict(snapshot.executors)
        states[composite.executor_id] = RuntimeExecutorState(
            executor_id=composite.executor_id,
            healthy=True,
            ready=False,
            temperature_millic=40_000,
            battery_ppm=900_000,
            free_slots=0,
            busy_until_us=0,
        )
        snapshot = replace(
            snapshot,
            executors=states,
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
                executor_id="executor:host-a",
            ),),
        )

        ticket = scheduler.submit_automated_request(
            request("same-artifact-fallback-residency"),
            manifest.model_id,
            snapshot,
            selection_mode="desktop-baseline",
        )

        self.assertEqual(ticket.binding.executor_id, composite.executor_id)
        self.assertEqual(ticket.transition_status, "PENDING")
        self.assertEqual(ticket.dispatch_state, "QUEUED")
        self.assertEqual(len(ticket.execution_plan.transitions), 1)
        self.assertEqual(ticket.execution_plan.transitions[0].evictions, ())
        observed = replace(
            snapshot,
            residency=model_residency_observations(
                profile,
                manifest,
                composite.executor_id,
                generation=2,
            ),
        )
        transition_plan = ticket.execution_plan.transitions[0]
        self.assertTrue(transition_target_is_observed(
            observed, ticket, transition_plan
        ))
        self.assertFalse(transition_target_is_observed(
            observed,
            ticket,
            replace(
                transition_plan,
                source_state=transition_plan.target_state,
            ),
        ))

    def test_measured_allocator_profile_expands_runtime_memory(self) -> None:
        source = catalog()
        host = source.executor_by_device["host-a"]
        cpu_only = RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=(host,),
            composite_executors=(),
            transitions=tuple(
                row for row in source.transitions
                if row.device_id == "host-a"
            ),
        ).to_json())
        unprofiled, manifest = self.scheduler_and_manifest(cpu_only)
        snapshot = runtime_snapshot(manifest, include_phone=False)
        baseline = unprofiled.generate_automated_candidates(
            request("allocator-unprofiled"), manifest.model_id, snapshot
        ).baseline
        expected_weight_bytes = next(
            row.required_bytes for row in baseline.plan.memory_demands
            if row.kind == "model_weights"
        )

        profiled_host = replace(host, adapter_parameters={
            "memory_model_weight_allocation_ppm:host-a": 1_500_000,
            "memory_workspace_minimum_bytes:host-a": 1_000_000,
        })
        profiled_catalog = RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=(profiled_host,),
            composite_executors=(),
            transitions=tuple(
                row for row in source.transitions
                if row.device_id == "host-a"
            ),
        ).to_json())
        profiled, profiled_manifest = self.scheduler_and_manifest(
            profiled_catalog
        )
        candidate = profiled.generate_automated_candidates(
            request("allocator-profiled"),
            profiled_manifest.model_id,
            runtime_snapshot(profiled_manifest, include_phone=False),
        ).baseline
        demands = {row.kind: row for row in candidate.plan.memory_demands}

        self.assertEqual(
            demands["model_weights"].required_bytes,
            (expected_weight_bytes * 3 + 1) // 2,
        )
        self.assertEqual(
            demands["workspace"].required_bytes, 1_000_000
        )

    def test_physical_failure_replans_with_immutable_identity(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        ))
        original = request("automated-fallback")
        first = scheduler.submit_automated_request(
            original,
            manifest.model_id,
            runtime_snapshot(manifest, phone_bandwidth=8_000_000_000),
        )
        epoch_ns = time.monotonic_ns() - first.decision.start_us * 1_000
        active = scheduler.wait_runtime_request(first.request.request_id, epoch_ns)
        recovery = scheduler.fail_automated_request(
            first.request.request_id,
            failed_at_us=active.decision.start_us + 1,
            reason="synthetic_transport_failure",
            snapshot=runtime_snapshot(
                manifest, phone_bandwidth=8_000_000_000
            ),
        )
        self.assertIsNotNone(recovery.fallback)
        fallback = recovery.fallback
        self.assertEqual(fallback.request, original)
        self.assertEqual(fallback.model, first.model)
        self.assertEqual(fallback.previous_ticket_id, first.ticket_id)
        self.assertNotEqual(
            fallback.decision.route_id, first.decision.route_id
        )
        self.assertTrue(
            scheduler.runtime_route_is_quarantined(first.decision.route_id)
        )
        attempt_events = [
            row["event_kind"]
            for row in scheduler.runtime_decision_log()["records"]
            if row["event_kind"] in {"DECISION", "FALLBACK"}
        ]
        self.assertEqual(attempt_events, ["DECISION", "FALLBACK"])

    def test_frontier_cache_miss_is_single_flight(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        snapshot = runtime_snapshot(manifest)
        compiler = scheduler._automated_compiler()
        original = compiler._rough_visits

        def delayed_visits(*args, **kwargs):
            time.sleep(0.03)
            return original(*args, **kwargs)

        def prepare(_index):
            return compiler.prepare_frontier(
                manifest,
                19,
                11,
                "exact",
                snapshot,
                snapshot.captured_at_us,
            )

        with mock.patch.object(
            compiler, "_rough_visits", side_effect=delayed_visits
        ) as visits:
            with ThreadPoolExecutor(max_workers=6) as executor:
                frontiers = tuple(executor.map(prepare, range(6)))

        self.assertTrue(all(row is frontiers[0] for row in frontiers))
        self.assertEqual(visits.call_count, 1)
        stats = compiler.cache_stats()
        self.assertEqual(stats["frontier_generations"], 1)
        self.assertGreaterEqual(stats["frontier_singleflight_waits"], 1)

    def test_capability_catalog_derived_indexes_are_immutable_and_cached(self) -> None:
        profile = catalog()

        self.assertIs(profile.executor_by_id, profile.executor_by_id)
        self.assertIs(profile.executor_by_device, profile.executor_by_device)
        self.assertIs(
            profile.composite_executor_by_id,
            profile.composite_executor_by_id,
        )
        self.assertIs(
            profile.fallback,
            next(row for row in profile.executors if row.qualified_fallback),
        )
        with self.assertRaises(TypeError):
            profile.executor_by_device["new-device"] = profile.fallback

    def test_background_frontier_uses_live_generation_not_snapshot_name(
        self,
    ) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        original = runtime_snapshot(manifest)
        capability_generation = canonical_sha256(catalog())
        first_key = placement_frontier_key(
            manifest=manifest,
            capability_generation_sha256=capability_generation,
            input_tokens=11,
            output_tokens=3,
            quality_requirement="exact",
            snapshot=original,
        )
        link_id = sorted(original.links)[0]
        changed = replace(
            original,
            links={
                **original.links,
                link_id: replace(
                    original.links[link_id],
                    measured_bandwidth_bytes_per_s=(
                        original.links[
                            link_id
                        ].measured_bandwidth_bytes_per_s // 2
                    ),
                ),
            },
        )
        second_key = placement_frontier_key(
            manifest=manifest,
            capability_generation_sha256=capability_generation,
            input_tokens=11,
            output_tokens=3,
            quality_requirement="exact",
            snapshot=changed,
        )

        self.assertEqual(original.snapshot_id, changed.snapshot_id)
        self.assertNotEqual(
            first_key.cost_profile_generation_sha256,
            second_key.cost_profile_generation_sha256,
        )
        scheduler.generate_automated_candidates(
            request("background-original", input_tokens=11, output_tokens=3),
            manifest.model_id,
            original,
        )
        deadline = time.monotonic() + 2
        while (
            scheduler.background_placement_stats()["entries"] < 1
            and time.monotonic() < deadline
        ):
            time.sleep(0.001)
        before = dict(scheduler.background_placement_stats())
        scheduler.generate_automated_candidates(
            request("background-changed", input_tokens=11, output_tokens=3),
            manifest.model_id,
            changed,
        )
        deadline = time.monotonic() + 2
        while (
            scheduler.background_placement_stats()["compilations"]
                <= before["compilations"]
            and time.monotonic() < deadline
        ):
            time.sleep(0.001)
        after = dict(scheduler.background_placement_stats())

        self.assertGreater(after["compilations"], before["compilations"])

    def test_background_frontier_excludes_request_live_state(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        original = runtime_snapshot(manifest)
        capability_generation = canonical_sha256(catalog())
        executor_id = sorted(original.executors)[0]
        executor = original.executors[executor_id]
        changed = replace(
            original,
            executors={
                **original.executors,
                executor_id: replace(
                    executor,
                    busy_until_us=executor.busy_until_us + 1_000_000,
                    free_slots=max(0, executor.free_slots - 1),
                    ready=not executor.ready,
                ),
            },
            cost_features={
                **original.cost_features,
                "active_request_batch_size": 7,
            },
        )

        first_key = placement_frontier_key(
            manifest=manifest,
            capability_generation_sha256=capability_generation,
            input_tokens=11,
            output_tokens=3,
            quality_requirement="exact",
            snapshot=original,
        )
        second_key = placement_frontier_key(
            manifest=manifest,
            capability_generation_sha256=capability_generation,
            input_tokens=11,
            output_tokens=3,
            quality_requirement="exact",
            snapshot=changed,
        )

        self.assertEqual(first_key, second_key)
        scheduler.generate_automated_candidates(
            request(
                "background-live-original", input_tokens=11, output_tokens=3
            ),
            manifest.model_id,
            original,
        )
        deadline = time.monotonic() + 2
        while (
            scheduler.background_placement_stats()["entries"] < 1
            and time.monotonic() < deadline
        ):
            time.sleep(0.001)
        before = dict(scheduler.background_placement_stats())
        scheduler.generate_automated_candidates(
            request(
                "background-live-changed", input_tokens=11, output_tokens=3
            ),
            manifest.model_id,
            changed,
        )
        time.sleep(0.01)

        self.assertEqual(
            scheduler.background_placement_stats()["compilations"],
            before["compilations"],
        )


if __name__ == "__main__":
    unittest.main()
