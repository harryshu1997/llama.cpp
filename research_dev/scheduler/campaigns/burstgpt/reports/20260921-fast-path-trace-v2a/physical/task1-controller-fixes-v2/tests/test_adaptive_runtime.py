#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from research_dev.scheduler import (
    GGUFModelManifestLoader,
    RuntimeCapabilityCatalog,
    RuntimeCompositeExecutorCapability,
    RuntimeCompositeOperatorPlacement,
    RuntimeDesktopControlProfile,
    RuntimeExecutionReceipt,
    RuntimeExecutionFailure,
    UnifiedScheduler,
)
from research_dev.scheduler._internal.adaptive_decode_contracts import (
    AdaptiveDecodeConfig,
    AdaptiveDecodeDirective,
    AdaptiveDecodeError,
    AdaptiveDecodeGroupedObservation,
    AdaptiveDecodePolicyAck,
    AdaptiveDecodeRawWindowObservation,
    AdaptiveDecodeWindowReceipt,
)
from research_dev.scheduler._internal.adaptive_decode import (
    ADAPTIVE_OBSERVATION_STORE_SCHEMA,
)
from research_dev.scheduler._internal.adaptive_decode_planning import (
    _policy,
    adaptive_candidate_set_for_parent,
    adaptive_decode_policies,
    adaptive_probe_contracts,
)
from research_dev.scheduler._internal.placement import TransferLink
from research_dev.scheduler._internal.route_generation import (
    AutomatedRouteCompiler,
    RouteGenerationError,
)
from research_dev.scheduler._internal.runtime_plan import (
    RuntimeTransitionPlan,
)
from research_dev.scheduler._internal.runtime_capabilities import (
    PhoneSessionResidencyObservation,
    RuntimePhoneSessionCapability,
)
from research_dev.scheduler._internal.types import canonical_sha256
from research_dev.scheduler._unified.phone_residency import (
    _OfflineLearningDemand,
    _PhoneDemandDiscovery,
)
from research_dev.scheduler.adapters import (
    interpret_runtime_ticket,
    validate_decision_candidate_coverage,
)
from research_dev.scheduler.adapters.residency import (
    physical_residency_parameters_match,
    physical_residency_supports_execution_plan,
)

try:
    from .test_automated_runtime import (
        FakeAutomatedPhysicalAdapter,
        catalog,
        executor_state,
        request,
        runtime_snapshot,
    )
    from .test_gguf_cost import write_synthetic_gguf
except ImportError:
    from test_automated_runtime import (
        FakeAutomatedPhysicalAdapter,
        catalog,
        executor_state,
        request,
        runtime_snapshot,
    )
    from test_gguf_cost import write_synthetic_gguf


class AdaptiveRuntimeIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        model_path = Path(self.directory.name) / "adaptive.gguf"
        write_synthetic_gguf(model_path, block_count=8)
        self.manifest = GGUFModelManifestLoader.load(
            "adaptive-model", model_path
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
            for operator in self.manifest.operators
        )
        self.desktop = RuntimeCompositeExecutorCapability(
            executor_id="coordinator:adaptive-desktop",
            endpoint="synthetic://adaptive-desktop",
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
            evidence_ids=("synthetic-adaptive-desktop",),
            artifact_sha256=self.manifest.artifact_sha256,
            operator_placements=placements,
        )
        ffn_ids = tuple(
            operator.operator_id for operator in self.manifest.operators
            if operator.kind == "ffn"
        )
        self.phone = RuntimeCompositeExecutorCapability(
            executor_id="coordinator:adaptive-phone",
            endpoint="synthetic://adaptive-phone",
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
            split_fractions_ppm=(250_000, 500_000),
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
            evidence_ids=("synthetic-adaptive-phone",),
            artifact_sha256=self.manifest.artifact_sha256,
            operator_ids=ffn_ids,
            baseline_executor_id=self.desktop.executor_id,
            helper_device_id="helper-c",
            adapter_parameters={
                "ffn_column_quantum": 1,
                "ffn_n_embd": self.manifest.embedding_length,
                "ffn_runtime_control_protocol": "decode-boundary-v1",
                "ffn_weight_buffer_layout": "selected-width",
                "maximum_helper_resident_weight_bytes": (
                    self.manifest.tensor_bytes
                ),
                "phone_device_id": "helper-c",
                "requires_measured_route_profile": 1,
                "ubatch_size": 4,
            },
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            composite_executors=(self.desktop, self.phone),
            desktop_control_profiles=(RuntimeDesktopControlProfile(
                profile_id="synthetic-adaptive-desktop-control",
                artifact_sha256=self.manifest.artifact_sha256,
                executor_id=self.desktop.executor_id,
                operator_placements=placements,
                maturity="QUALIFIED",
                evidence_ids=("synthetic-adaptive-desktop",),
            ),),
        ).to_json())
        self.profile = profile
        self.scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        self.scheduler.register_runtime_capabilities(profile)
        self.scheduler.register_model_manifest(self.manifest)
        snapshot = runtime_snapshot(
            self.manifest, phone_bandwidth=8_000_000_000
        )
        self.snapshot = replace(
            snapshot,
            executors={
                **snapshot.executors,
                self.desktop.executor_id: executor_state(
                    self.desktop.executor_id
                ),
                self.phone.executor_id: executor_state(
                    self.phone.executor_id
                ),
            },
        )

    def test_decode_boundary_reevaluates_uncovered_arrived_model(self):
        other_artifact = "sha256:" + "1" * 64
        current = SimpleNamespace(
            dispatch_state="ACQUIRED",
            model=SimpleNamespace(
                artifact_sha256=self.manifest.artifact_sha256,
                model_id=self.manifest.model_id,
            ),
            request=SimpleNamespace(
                request_id="ready-model",
                output_tokens=32,
            ),
        )
        uncovered = SimpleNamespace(
            dispatch_state="ACQUIRED",
            model=SimpleNamespace(
                artifact_sha256=other_artifact,
                model_id="other-model",
            ),
            request=SimpleNamespace(
                request_id="uncovered-model",
                output_tokens=64,
            ),
        )
        ready = SimpleNamespace(
            layout=SimpleNamespace(
                geometry_sha256="sha256:" + "2" * 64,
            ),
            covers_artifact=lambda artifact: (
                artifact == self.manifest.artifact_sha256
            ),
        )
        controller = self.scheduler._model_placement_controller
        with (
            mock.patch.object(
                controller,
                "pending_phone_layout_candidate",
                return_value=None,
            ),
            mock.patch.object(
                controller, "ready_phone_layout", return_value=ready
            ),
            mock.patch.object(
                controller, "target_phone_layout", return_value=None
            ),
            mock.patch.object(
                controller,
                "remaining_request_decode_tokens",
                side_effect=lambda _request_id, output_tokens: output_tokens,
            ),
            mock.patch.object(
                self.scheduler._runtime_controller,
                "current_tickets",
                return_value=(current, uncovered),
            ),
            mock.patch.object(
                self.scheduler._automated_compiler(),
                "phone_residency_evidence_status",
                return_value={
                    "normalized_benefit_uj": 10,
                    "reason": "PHONE_RESIDENCY_ROUTE_EVIDENCE_READY",
                    "source_route_id": "phone-route",
                },
            ),
            mock.patch.object(
                self.scheduler, "_update_phone_residency_portfolio"
            ) as update,
        ):
            self.scheduler._reevaluate_pending_phone_layout_at_boundary(
                current, 10_000
            )

        update.assert_called_once_with(
            current.request,
            self.manifest,
            10_000,
            None,
        )

    def test_decode_boundary_reevaluates_cached_learning_model(self):
        other_artifact = "sha256:" + "1" * 64
        current = SimpleNamespace(
            dispatch_state="ACQUIRED",
            model=SimpleNamespace(
                artifact_sha256=self.manifest.artifact_sha256,
                model_id=self.manifest.model_id,
            ),
            request=SimpleNamespace(
                request_id="ready-model",
                output_tokens=32,
            ),
        )
        uncovered = SimpleNamespace(
            dispatch_state="ACQUIRED",
            model=SimpleNamespace(
                artifact_sha256=other_artifact,
                model_id="other-model",
            ),
            request=SimpleNamespace(
                request_id="uncovered-learning-model",
                output_tokens=64,
            ),
        )
        ready = SimpleNamespace(
            layout=SimpleNamespace(
                geometry_sha256="sha256:" + "2" * 64,
            ),
            covers_artifact=lambda artifact: (
                artifact == self.manifest.artifact_sha256
            ),
        )
        self.scheduler._online_learning_phone_demand_cache[
            other_artifact
        ] = _OfflineLearningDemand(
            demand=SimpleNamespace(),
            sessions=(),
            helper_id="helper-c",
            status={},
        )
        controller = self.scheduler._model_placement_controller
        with (
            mock.patch.object(
                controller,
                "pending_phone_layout_candidate",
                return_value=None,
            ),
            mock.patch.object(
                controller, "ready_phone_layout", return_value=ready
            ),
            mock.patch.object(
                controller, "target_phone_layout", return_value=None
            ),
            mock.patch.object(
                controller,
                "remaining_request_decode_tokens",
                side_effect=lambda _request_id, output_tokens: output_tokens,
            ),
            mock.patch.object(
                self.scheduler._runtime_controller,
                "current_tickets",
                return_value=(current, uncovered),
            ),
            mock.patch.object(
                self.scheduler._automated_compiler(),
                "phone_residency_evidence_status",
                return_value={
                    "reason": "PHONE_RESIDENCY_ROUTE_EVIDENCE_UNUSABLE",
                },
            ),
            mock.patch.object(
                self.scheduler, "_update_phone_residency_portfolio"
            ) as update,
        ):
            self.scheduler._reevaluate_pending_phone_layout_at_boundary(
                current, 10_000
            )

        update.assert_called_once_with(
            current.request,
            self.manifest,
            10_000,
            None,
        )

    def test_live_learning_discovery_falls_back_to_cached_route(self):
        empty = _PhoneDemandDiscovery(
            demand_rows=(),
            sessions=None,
            helper_id=None,
            route_evidence_by_artifact={},
        )
        value = request("learning-cache-fallback")
        with (
            mock.patch.object(
                self.scheduler,
                "_phone_queue_demand",
                return_value=SimpleNamespace(
                    queued_work_by_artifact={
                        self.manifest.artifact_sha256: 32
                    },
                ),
            ),
            mock.patch.object(
                self.scheduler,
                "_discover_phone_residency_demand",
                return_value=empty,
            ),
            mock.patch.object(
                self.scheduler,
                "_online_learning_phone_discovery",
                return_value=empty,
            ) as online,
            mock.patch.object(
                self.scheduler,
                "_cached_online_learning_phone_discovery",
                return_value=empty,
            ) as cached,
            mock.patch.object(
                self.scheduler,
                "_record_phone_residency_demand_unavailable",
            ),
        ):
            self.assertFalse(self.scheduler._update_phone_residency_portfolio(
                value,
                self.manifest,
                value.arrival_us,
                self.snapshot,
            ))

        online.assert_called_once()
        cached.assert_called_once()

    def test_decode_boundary_coalesces_unchanged_pressure_bucket(self):
        other_artifact = "sha256:" + "1" * 64
        current = SimpleNamespace(
            dispatch_state="ACQUIRED",
            model=SimpleNamespace(
                artifact_sha256=self.manifest.artifact_sha256,
                model_id=self.manifest.model_id,
            ),
            request=SimpleNamespace(
                request_id="ready-model",
                output_tokens=32,
            ),
        )
        uncovered = SimpleNamespace(
            dispatch_state="ACQUIRED",
            model=SimpleNamespace(
                artifact_sha256=other_artifact,
                model_id="other-model",
            ),
            request=SimpleNamespace(
                request_id="uncovered-model",
                output_tokens=64,
            ),
        )
        ready = SimpleNamespace(
            layout=SimpleNamespace(
                geometry_sha256="sha256:" + "2" * 64,
            ),
            covers_artifact=lambda artifact: (
                artifact == self.manifest.artifact_sha256
            ),
        )
        evidence = {
            "normalized_benefit_uj": 10,
            "reason": "PHONE_RESIDENCY_ROUTE_EVIDENCE_READY",
            "source_route_id": "phone-route",
        }
        previous = {
            "kind": "EVALUATED",
            "queue_work_by_artifact": {
                self.manifest.artifact_sha256: 32,
                other_artifact: 64,
            },
            "route_evidence_by_artifact": {
                other_artifact: evidence,
            },
        }
        controller = self.scheduler._model_placement_controller
        with (
            mock.patch.object(
                controller,
                "pending_phone_layout_candidate",
                return_value=None,
            ),
            mock.patch.object(
                controller, "ready_phone_layout", return_value=ready
            ),
            mock.patch.object(
                controller, "target_phone_layout", return_value=None
            ),
            mock.patch.object(
                controller,
                "remaining_request_decode_tokens",
                side_effect=lambda _request_id, output_tokens: output_tokens,
            ),
            mock.patch.object(
                controller,
                "phone_layout_events",
                return_value=(previous,),
            ),
            mock.patch.object(
                self.scheduler._runtime_controller,
                "current_tickets",
                return_value=(current, uncovered),
            ),
            mock.patch.object(
                self.scheduler._automated_compiler(),
                "phone_residency_evidence_status",
                return_value=evidence,
            ),
            mock.patch.object(
                self.scheduler, "_update_phone_residency_portfolio"
            ) as update,
        ):
            self.scheduler._reevaluate_pending_phone_layout_at_boundary(
                current, 10_000
            )

        update.assert_not_called()

    def test_decode_boundary_reevaluates_changed_resident_work_bucket(self):
        other_artifact = "sha256:" + "1" * 64
        current = SimpleNamespace(
            dispatch_state="ACQUIRED",
            model=SimpleNamespace(
                artifact_sha256=self.manifest.artifact_sha256,
                model_id=self.manifest.model_id,
            ),
            request=SimpleNamespace(
                request_id="ready-model",
                output_tokens=32,
            ),
        )
        uncovered = SimpleNamespace(
            dispatch_state="ACQUIRED",
            model=SimpleNamespace(
                artifact_sha256=other_artifact,
                model_id="other-model",
            ),
            request=SimpleNamespace(
                request_id="uncovered-model",
                output_tokens=64,
            ),
        )
        ready = SimpleNamespace(
            layout=SimpleNamespace(
                geometry_sha256="sha256:" + "2" * 64,
            ),
            covers_artifact=lambda artifact: (
                artifact == self.manifest.artifact_sha256
            ),
        )
        evidence = {
            "normalized_benefit_uj": 10,
            "reason": "PHONE_RESIDENCY_ROUTE_EVIDENCE_READY",
            "source_route_id": "phone-route",
        }
        previous = {
            "kind": "EVALUATED",
            "queue_work_by_artifact": {
                self.manifest.artifact_sha256: 64,
                other_artifact: 64,
            },
            "route_evidence_by_artifact": {
                other_artifact: evidence,
            },
        }
        controller = self.scheduler._model_placement_controller
        with (
            mock.patch.object(
                controller,
                "pending_phone_layout_candidate",
                return_value=None,
            ),
            mock.patch.object(
                controller, "ready_phone_layout", return_value=ready
            ),
            mock.patch.object(
                controller, "target_phone_layout", return_value=None
            ),
            mock.patch.object(
                controller,
                "remaining_request_decode_tokens",
                side_effect=lambda _request_id, output_tokens: output_tokens,
            ),
            mock.patch.object(
                controller,
                "phone_layout_events",
                return_value=(previous,),
            ),
            mock.patch.object(
                self.scheduler._runtime_controller,
                "current_tickets",
                return_value=(current, uncovered),
            ),
            mock.patch.object(
                self.scheduler._automated_compiler(),
                "phone_residency_evidence_status",
                return_value=evidence,
            ),
            mock.patch.object(
                self.scheduler, "_update_phone_residency_portfolio"
            ) as update,
        ):
            self.scheduler._reevaluate_pending_phone_layout_at_boundary(
                current, 10_000
            )

        update.assert_called_once_with(
            current.request,
            self.manifest,
            10_000,
            None,
        )

    def _cpu_parent_candidates(self, *, constrain_gpu=True):
        cpu = replace(
            self.desktop,
            executor_id="coordinator:actual-cpu-parent",
            endpoint="synthetic://actual-cpu-parent",
            participant_device_ids=("host-a",),
            participant_resource_ids={"host-a": ("compute:host-a",)},
            resource_ids=("compute:host-a",),
            operator_placements=tuple(
                replace(row, primary_device_id="host-a")
                for row in self.desktop.operator_placements
            ),
        )
        helper = replace(
            self.phone,
            executor_id="coordinator:actual-cpu-helper",
            endpoint=cpu.endpoint,
            baseline_executor_id=cpu.executor_id,
            adapter_parameters={
                **self.phone.adapter_parameters,
                "maximum_helper_resident_weight_bytes": 2_000_000_000,
            },
            participant_device_ids=("host-a", "helper-c"),
            participant_resource_ids={
                "host-a": ("compute:host-a",),
                "helper-c": ("compute:helper-c",),
            },
            resource_ids=(
                "compute:host-a", "compute:helper-c",
                "link:usb-out", "link:usb-in",
            ),
        )
        phone_device = replace(
            self.profile.executor_by_device["helper-c"],
            execution_resource_ids=(
                "compute:helper-c", "link:usb-in", "link:usb-out",
            ),
            phone_sessions=(RuntimePhoneSessionCapability(
                session_id="session-a",
                device_id="helper-c",
                endpoint="session://helper-c/session-a",
                worker_identity_sha256="sha256:" + "9" * 64,
                memory_resource_id="phone-memory",
                resident_memory_limit_bytes=2_000_000_000,
                shared_compute_resource_id="compute:helper-c",
                shared_transport_resource_ids=("link:usb-in", "link:usb-out"),
                supported_layer_mask=(1 << self.manifest.block_count) - 1,
                maximum_columns=self.manifest.feed_forward_length,
                column_quantum=1,
                supported_data_types=("F16", "F32", "Q8_0"),
                batch_plans=("split-row",),
                ready=True,
                residency_state="cold",
            ),),
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            self.profile,
            composite_executors=(self.desktop, self.phone, cpu, helper),
            executors=tuple(
                phone_device if row.device_id == "helper-c" else row
                for row in self.profile.executors
            ),
        ).to_json())
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        scheduler.register_model_manifest(self.manifest)
        capacities = dict(self.snapshot.memory.capacities)
        capacities["gpu-memory"] = replace(
            capacities["gpu-memory"],
            occupied_bytes=(capacities["gpu-memory"].capacity_bytes
                            if constrain_gpu else 0),
            reserve_bytes=0,
        )
        snapshot = replace(
            self.snapshot,
            memory=replace(self.snapshot.memory, capacities=capacities),
            executors={
                **self.snapshot.executors,
                cpu.executor_id: executor_state(cpu.executor_id),
                helper.executor_id: executor_state(helper.executor_id),
            },
        )
        probe = request("actual-cpu-parent", input_tokens=8, output_tokens=128)
        parent_hash = replace(
            profile.desktop_control_profiles[0],
            executor_id=cpu.executor_id,
            operator_placements=cpu.operator_placements,
        ).placement_sha256
        values = scheduler._generate_automated_candidate_set(
            probe, self.manifest, snapshot,
            max(probe.arrival_us, snapshot.captured_at_us),
            update_phone_residency_portfolio=False,
            desktop_parent=(cpu.executor_id, parent_hash),
        )
        return scheduler, profile, snapshot, probe, values, cpu, helper

    def test_cpu_parent_helper_rebuilds_mapping_without_gpu_qualification(self):
        scheduler, profile, _, probe, values, cpu, helper = (
            self._cpu_parent_candidates()
        )
        self.assertEqual(values.baseline.binding.executor_id, cpu.executor_id)
        baseline, policies, envelope = adaptive_decode_policies(
            values, self.manifest, profile, probe.output_tokens
        )
        self.assertIsNotNone(envelope)
        self.assertTrue(policies)
        self.assertEqual(envelope.binding.executor_id, helper.executor_id)
        self.assertEqual(envelope.binding.endpoint, cpu.endpoint)
        self.assertEqual(envelope.plan.baseline_executor_id, cpu.executor_id)
        self.assertEqual(
            envelope.plan.desktop_placement_sha256,
            values.baseline.plan.desktop_placement_sha256,
        )
        self.assertNotEqual(
            baseline.desktop_placement_sha256,
            profile.desktop_control_profiles[0].placement_sha256,
        )
        parents = {row.operator_id: row for row in values.baseline.plan.operators}
        for row in envelope.plan.operators:
            if row.operator_kind != "ffn":
                self.assertEqual(row, parents[row.operator_id])
            self.assertNotIn("accelerator-b", row.device_ids)
        self.assertNotIn("compute:accelerator-b", envelope.plan.resource_ids)
        self.assertTrue(all(
            "accelerator-b" not in (row.source_device, row.target_device)
            for row in envelope.cost.transfer_costs
        ))
        self.assertIn("ROUTE_NOT_QUALIFIED", envelope.rejection_reasons)
        opportunity, = scheduler._compact_helper_opportunities(values, probe)
        self.assertEqual(opportunity.evidence_state, "LEARNING")
        self.assertEqual(opportunity.desktop_parent_placement_sha256,
                         baseline.desktop_placement_sha256)

    def test_ready_shard_bootstraps_a_new_parent_without_an_old_envelope(self):
        scheduler, profile, snapshot, probe, values, cpu, helper = (
            self._cpu_parent_candidates()
        )
        _, _, envelope = adaptive_decode_policies(
            values, self.manifest, profile, probe.output_tokens
        )
        shard, = envelope.plan.execution_contract.phone_shards
        shard = replace(shard, session_generation=2)
        ready = SimpleNamespace(
            state="READY", generation=2,
            layout=SimpleNamespace(
                geometry_sha256=envelope.plan.adapter_parameters[
                    "phone_shard_set_geometry_sha256"
                ],
                shards=(shard,),
                session_generation_by_id={shard.session_id: 2},
            ),
            covers_artifact=lambda artifact: artifact == self.manifest.artifact_sha256,
        )
        ticket = SimpleNamespace(
            ticket_id="cpu-parent:attempt:0", request=probe,
            model=SimpleNamespace(model_id=self.manifest.model_id,
                                  artifact_sha256=self.manifest.artifact_sha256),
            decision=SimpleNamespace(route_id=values.baseline.candidate_id),
            binding=values.baseline.binding, execution_plan=values.baseline.plan,
            dispatch_state="ACQUIRED",
        )
        observed = PhoneSessionResidencyObservation(
            session_id=shard.session_id, device_id="helper-c",
            executor_id=self.phone.executor_id, endpoint=shard.endpoint,
            artifact_sha256=shard.artifact_sha256,
            resident_geometry_sha256=shard.resident_geometry_sha256,
            operator_plan_sha256=shard.operator_plan_sha256,
            session_generation=2, resident_bytes=shard.resident_bytes,
        )
        snapshot = replace(
            snapshot, phone_session_residency=(observed,),
            executors={**snapshot.executors,
                       helper.executor_id: replace(snapshot.executors[helper.executor_id],
                                                   ready=False, free_slots=0)},
        )
        with mock.patch.object(scheduler._model_placement_controller,
                               "ready_phone_layout", return_value=ready):
            self.assertIsNone(scheduler._resolve_verified_ready_helper_plan(ticket, ready))
            projected = scheduler._snapshot_with_verified_ready_helper(
                ticket, ready, snapshot, None
            )
            self.assertTrue(projected.executors[helper.executor_id].ready)
            self.assertFalse(snapshot.executors[helper.executor_id].ready)
            self.assertEqual(projected.phone_session_residency, (observed,))
            for field, invalid in (
                ("artifact_sha256", "sha256:" + "1" * 64),
                ("resident_geometry_sha256", "sha256:" + "2" * 64),
                ("operator_plan_sha256", "sha256:" + "3" * 64),
                ("session_generation", 3),
                ("resident_bytes", shard.resident_bytes + 1),
            ):
                with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError, "physical session identity differs"
                ):
                    scheduler._snapshot_with_verified_ready_helper(
                        ticket, ready, replace(snapshot, phone_session_residency=(
                            replace(observed, **{field: invalid}),
                        )), None,
                    )

    def test_ready_replacement_envelope_does_not_require_its_preparation_owner(self):
        scheduler, profile, _, probe, values, _, _ = self._cpu_parent_candidates()
        _, _, candidate = adaptive_decode_policies(
            values, self.manifest, profile, probe.output_tokens
        )
        shard, = candidate.plan.execution_contract.phone_shards
        layout = SimpleNamespace(
            state="READY", generation=2,
            layout=SimpleNamespace(
                changed_session_ids=(shard.session_id,),
                geometry_sha256=candidate.plan.adapter_parameters[
                    "phone_shard_set_geometry_sha256"
                ],
            ),
        )
        scheduler._request_helper_preparation_envelopes.clear()
        scheduler._request_helper_preparations.clear()
        arguments = dict(
            artifact_sha256=self.manifest.artifact_sha256,
            desktop_parent_route_id=values.baseline.candidate_id,
            desktop_placement_sha256=values.baseline.plan.desktop_placement_sha256,
            helper_plan=candidate.plan, helper_binding=candidate.binding,
            layout=layout,
        )
        with mock.patch.object(
            scheduler, "_phone_session_replacement_authorization",
            side_effect=ValueError("no preparation owner"),
        ) as authorize:
            envelope = scheduler._build_helper_envelope(**arguments)
            self.assertEqual(envelope.preparation_changed_session_ids, ())
            self.assertIsNone(envelope.replacement_authorization)
            self.assertEqual(envelope.helper_plan, candidate.plan)
            self.assertEqual(envelope.helper_binding, candidate.binding)
            authorize.assert_not_called()
            layout.state = "PROPOSED"
            with self.assertRaisesRegex(ValueError, "no preparation owner"):
                scheduler._build_helper_envelope(**arguments)
            authorize.assert_called_once_with(layout)

    def test_selected_cpu_parent_receives_its_own_baseline_and_probe_contracts(self):
        scheduler, profile, _, probe, values, cpu, helper = self._cpu_parent_candidates(
            constrain_gpu=False
        )
        selected = values.baseline
        plan = replace(selected.plan, adapter_parameters={
            **selected.plan.adapter_parameters, "dormant_phone_ffn_runtime_v1": "{}",
        })
        selected = replace(selected, plan=plan, binding=replace(
            selected.binding, operator_plan_sha256=plan.plan_sha256,
        ))
        values = replace(values, candidates=tuple(
            selected if row.candidate_id == selected.candidate_id else row
            for row in values.candidates
        ))
        gpu_parent = next(row for row in values.candidates
                          if row.binding.executor_id == self.desktop.executor_id)
        values = adaptive_candidate_set_for_parent(values, gpu_parent)
        contracts = adaptive_probe_contracts(values, self.manifest, profile, probe.output_tokens)
        self.assertTrue(contracts[gpu_parent.candidate_id]["baseline"])
        self.assertTrue(contracts[selected.candidate_id]["baseline"])
        ticket = SimpleNamespace(
            binding=selected.binding, execution_plan=selected.plan,
            cost_estimates=SimpleNamespace(estimates=tuple(
                SimpleNamespace(details={"adaptive_decode_contract": value})
                for value in contracts.values()
            )),
        )
        baseline, policies = scheduler._ticket_adaptive_policy_records(ticket)
        self.assertEqual(baseline.executor_id, cpu.executor_id)
        self.assertEqual(baseline.desktop_placement_sha256, plan.desktop_placement_sha256)
        self.assertTrue(any(row.executor_id == helper.executor_id for row in policies))
        selected = next(row for row in values.candidates
                        if row.candidate_id == selected.candidate_id)
        scheduler._refresh_request_helper_opportunities(
            probe, values, selected, mock.sentinel.estimates,
            mock.sentinel.bound_helper,
        )
        opportunity, = scheduler._request_helper_opportunities[probe.request_id]
        self.assertEqual(opportunity.evidence_state, "LEARNING")

    def test_missing_parent_helper_is_logged_once_until_configuration_changes(self):
        from research_dev.scheduler._unified.helper_envelopes import (
            _ReadyHelperParentUnavailable,
        )

        scheduler, _, _, probe, values, _, _ = self._cpu_parent_candidates()
        helper, = scheduler._compact_helper_opportunities(values, probe)
        shard, = helper.helper_operator_plan.execution_contract.phone_shards
        ready = SimpleNamespace(
            generation=2,
            layout=SimpleNamespace(
                geometry_sha256=helper.phone_layout_geometry_sha256,
                shards=(shard,), session_generation_by_id={shard.session_id: 2},
            ),
        )
        ticket = SimpleNamespace(
            ticket_id="cpu-parent:attempt:0", request=probe,
            execution_plan=values.baseline.plan,
            binding=values.baseline.binding,
            cost_estimates=SimpleNamespace(estimates=()),
            model=SimpleNamespace(artifact_sha256=self.manifest.artifact_sha256),
        )
        with self.assertRaises(_ReadyHelperParentUnavailable) as failed:
            scheduler._ticket_adaptive_policy_records(ticket)
        error = failed.exception
        for at_us in range(62):
            if not scheduler._ready_helper_parent_rejection_unchanged(ticket, ready):
                scheduler._handle_ready_helper_materialization_error(
                    ticket, ready, error, at_us
                )
        failures = [row for row in scheduler.request_helper_events()
                    if row["kind"] == "HELPER_REMATERIALIZATION_FAILED"]
        self.assertEqual(len(failures), 1)
        ready.layout.session_generation_by_id = {shard.session_id: 3}
        self.assertFalse(scheduler._ready_helper_parent_rejection_unchanged(ticket, ready))
        ready.layout.session_generation_by_id = {shard.session_id: 2}
        ticket.ticket_id = "cpu-parent:attempt:1"
        self.assertFalse(scheduler._ready_helper_parent_rejection_unchanged(ticket, ready))

    def test_cpu_desktop_retains_dormant_startup_contract_during_telemetry_loss(self):
        self.phone = replace(self.phone, adapter_parameters={
            **self.phone.adapter_parameters,
            "ffn_activation": "F32", "ffn_transport": "tcp",
            "ffn_timeout_ms": 1000, "ffn_bridge_host": "127.0.0.1",
            "ffn_bridge_port": 12000, "bridge_allocator": "host",
            "bridge_queue_depth": 1,
        })
        scheduler, profile, _, probe, values, cpu, _ = self._cpu_parent_candidates(
            constrain_gpu=False
        )
        parent = values.baseline
        plan = replace(parent.plan, residency_variant="cold")
        parent = replace(parent, plan=plan, binding=replace(
            parent.binding, operator_plan_sha256=plan.plan_sha256,
        ))
        cpu_helper = next(row for row in values.candidates
                          if row.plan.baseline_executor_id == cpu.executor_id)
        self.assertIsNotNone(scheduler._complete_dormant_phone_ffn_parameters(
            cpu_helper.plan.adapter_parameters
        ))
        values = replace(values, candidates=tuple(
            parent if row.candidate_id == parent.candidate_id else
            replace(row, admitted=False, rejection_reasons=tuple(sorted({
                *row.rejection_reasons, "BATTERY_LIMIT",
            }))) if row.assisted_operator_kind == "ffn" else row
            for row in values.candidates
        ))
        gpu_parent = next(row for row in values.candidates
                          if row.binding.executor_id == self.desktop.executor_id)
        values = adaptive_candidate_set_for_parent(values, gpu_parent)
        parent = next(row for row in values.candidates
                      if row.candidate_id == parent.candidate_id)
        with mock.patch.object(scheduler, "_authorize_phone_helper_plan") as authorize:
            values, selected, _, _ = scheduler._desktop_with_async_phone_helper(
                values, parent, (), "CALIBRATION", probe, self.manifest, "calibration"
            )
        authorize.assert_not_called()
        self.assertEqual(selected.binding.executor_id, cpu.executor_id)
        self.assertEqual(selected.plan.desktop_placement_sha256, plan.desktop_placement_sha256)
        self.assertEqual(selected.plan.resource_ids, plan.resource_ids)
        self.assertEqual(selected.plan.execution_contract, plan.execution_contract)
        self.assertIsNone(selected.plan.helper_envelope)
        self.assertTrue(scheduler._has_dormant_phone_ffn_runtime(selected.plan))
        contracts = adaptive_probe_contracts(values, self.manifest, profile, probe.output_tokens)
        ticket = SimpleNamespace(
            binding=selected.binding, execution_plan=selected.plan,
            cost_estimates=SimpleNamespace(estimates=tuple(
                SimpleNamespace(details={"adaptive_decode_contract": value})
                for value in contracts.values()
            )),
        )
        baseline, _ = scheduler._ticket_adaptive_policy_records(ticket)
        self.assertEqual(baseline.executor_id, cpu.executor_id)
        self.assertEqual(baseline.operator_plan_sha256, selected.plan.plan_sha256)
        _, policies, envelope = adaptive_decode_policies(
            adaptive_candidate_set_for_parent(values, selected),
            self.manifest, profile, probe.output_tokens,
        )
        self.assertEqual(policies, ())
        self.assertIsNone(envelope)

    def test_missing_phone_telemetry_is_not_measured_exhaustion_or_overheating(self):
        from research_dev.scheduler._internal.runtime_cost import RuntimeMemoryDemand

        scheduler, _, snapshot, _, _, _, _ = self._cpu_parent_candidates()
        compiler = scheduler._automated_compiler()
        capacity = snapshot.memory.capacities["phone-memory"]
        state = snapshot.executors["executor:helper-c"]
        demand = RuntimeMemoryDemand(
            demand_id="test-workspace", resident_bytes=0, lifetime="request",
            resource_id="phone-memory", required_bytes=100_000_000,
            kind="workspace", device_id="helper-c",
        )
        pattern = SimpleNamespace(device_ids=("helper-c",), phone_resident_envelope=None,
                                  split_axis="none")
        snapshot = replace(snapshot, residency=(),
            memory=replace(snapshot.memory, capacities={
                **snapshot.memory.capacities,
                "phone-memory": replace(capacity, occupied_bytes=(
                    capacity.capacity_bytes - capacity.reserve_bytes
                )),
            }),
            executors={**snapshot.executors, state.executor_id: replace(
                state, battery_ppm=0, temperature_millic=100_000,
                thermal_qualified=False,
            )},
        )
        for validity in ("MISSING", "STALE", "VALID"):
            sample = replace(snapshot, telemetry_observations={"helper-c": {
                "source": "synthetic-http", "validity": validity,
                "valid": validity == "VALID", "failure_reason": validity,
                "sample_timestamp_ns": 1, "age_us": 0, "maximum_age_us": 5_000_000,
            }})
            with self.subTest(validity=validity), mock.patch.object(
                compiler, "_matching_residency", return_value=None,
            ):
                reasons = compiler._eligibility(
                    self.manifest, pattern, sample, sample.captured_at_us,
                    "cold", {"helper-c": "cold"}, (), (), "QUALIFIED", (demand,), None,
                )
            if validity == "VALID":
                self.assertIn("MEMORY_CAPACITY", reasons)
                self.assertIn("THERMAL_LIMIT", reasons)
                self.assertIn("BATTERY_LIMIT", reasons)
                self.assertNotIn("PHONE_TELEMETRY_UNAVAILABLE", reasons)
            else:
                self.assertIn("PHONE_TELEMETRY_UNAVAILABLE", reasons)
                self.assertTrue({"MEMORY_CAPACITY", "THERMAL_LIMIT", "BATTERY_LIMIT"}.isdisjoint(reasons))
                with mock.patch.object(compiler, "_matching_residency", return_value=None):
                    oversized = replace(demand, required_bytes=capacity.capacity_bytes + 1)
                    reasons = compiler._eligibility(
                        self.manifest, pattern, sample, sample.captured_at_us,
                        "cold", {"helper-c": "cold"}, (), (), "QUALIFIED", (oversized,), None,
                    )
                self.assertIn("MEMORY_CAPACITY", reasons)

    def test_decode_envelope_preserves_parent_and_uses_decode_transport(self):
        values = self.scheduler.generate_automated_candidates(
            request("adaptive-phase", input_tokens=8, output_tokens=30),
            self.manifest.model_id,
            self.snapshot,
        )
        parent = values.baseline
        _, policies, envelope = adaptive_decode_policies(
            values,
            self.manifest,
            self.profile,
            30,
        )
        self.assertIsNotNone(envelope)
        self.assertEqual(
            {row.split_fraction_ppm for row in policies},
            set(range(62_500, 1_000_001, 62_500)),
        )
        rows = tuple(
            row for row in values.candidates
            if row.paired_baseline_route_id == parent.candidate_id
            and row.assisted_operator_kind == "ffn"
        )
        self.assertTrue(rows)
        self.assertTrue(all(
            "PLACEMENT_INFEASIBLE" not in row.rejection_reasons
            for row in rows
        ))
        for row in rows:
            parameters = row.plan.adapter_parameters
            self.assertEqual(parameters["ffn_assistance_phase"], "decode")
            self.assertEqual(parameters["ffn_max_tokens"], 1)
            self.assertEqual(parameters["ubatch_size"], 4)
            self.assertEqual(
                parameters["ffn_column_quantum"],
                self.manifest.feed_forward_length // 16,
            )
            self.assertEqual(
                row.plan.desktop_placement_sha256,
                parent.plan.desktop_placement_sha256,
            )
            parent_weights = {
                demand.device_id: demand.required_bytes
                for demand in parent.plan.memory_demands
                if demand.kind == "model_weights"
            }
            assisted_weights = {
                demand.device_id: demand.required_bytes
                for demand in row.plan.memory_demands
                if demand.kind == "model_weights"
            }
            for device_id, amount in parent_weights.items():
                self.assertEqual(assisted_weights[device_id], amount)
            self.assertGreater(assisted_weights["helper-c"], 0)
        identities = {
            row.plan.adapter_parameters["resident_model_identity_sha256"]
            for row in rows
        }
        self.assertEqual(len(identities), 1)
        self.assertEqual(
            {policy.executor_id for policy in policies},
            {envelope.binding.executor_id},
        )
        matching_geometry = tuple(
            row for row in values.candidates
            if row.assisted_operator_kind == "ffn"
            and row.route_family == "operator_split"
            and row.plan.adapter_parameters.get(
                "ffn_resident_layer_mask"
            ) == envelope.plan.adapter_parameters.get(
                "ffn_resident_layer_mask"
            )
            and row.plan.adapter_parameters.get(
                "ffn_resident_columns"
            ) == envelope.plan.adapter_parameters.get(
                "ffn_resident_columns"
            )
            and not (
                set(row.rejection_reasons) - {
                    "COLD_RESIDENCY_BREAK_EVEN",
                    "ENERGY_UNKNOWN",
                    "ROUTE_NOT_QUALIFIED",
                    "SLO_UPPER_BOUND",
                }
            )
        )
        self.assertEqual(
            envelope.cost.service_upper_us,
            min(row.cost.service_upper_us for row in matching_geometry),
        )
        self.assertEqual(
            {policy.operator_plan_sha256 for policy in policies},
            {envelope.plan.plan_sha256},
        )
        resident_plan = envelope.plan.to_json()
        zero_plan = dict(resident_plan)
        zero_plan["split_fraction_ppm"] = 0
        learned_plan = dict(resident_plan)
        learned_plan["split_fraction_ppm"] = 750_000
        self.assertTrue(physical_residency_parameters_match(
            envelope.plan.adapter_parameters,
            envelope.plan.adapter_parameters,
        ))
        self.assertTrue(physical_residency_supports_execution_plan(
            resident_plan, zero_plan
        ))
        self.assertTrue(physical_residency_supports_execution_plan(
            resident_plan, learned_plan
        ))

    def test_decode_envelope_keeps_selected_transport_contract(self):
        alternate = replace(
            self.phone,
            executor_id="coordinator:adaptive-phone:a",
            backend="backend:phone:coalesced",
            maturity="SHADOW",
            evidence_ids=("synthetic-adaptive-phone-coalesced",),
            adapter_parameters={
                **dict(self.phone.adapter_parameters),
                "usb_batch_plan": "coalesced-batch",
            },
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            self.profile,
            composite_executors=(self.desktop, self.phone, alternate),
        ).to_json())
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        scheduler.register_model_manifest(self.manifest)
        snapshot = replace(
            self.snapshot,
            executors={
                **self.snapshot.executors,
                alternate.executor_id: executor_state(alternate.executor_id),
            },
        )
        values = scheduler.generate_automated_candidates(
            request("adaptive-transports", input_tokens=8, output_tokens=30),
            self.manifest.model_id,
            snapshot,
        )

        _, policies, envelope = adaptive_decode_policies(
            values,
            self.manifest,
            profile,
            30,
        )
        contracts = adaptive_probe_contracts(
            values,
            self.manifest,
            profile,
            30,
        )

        self.assertIsNotNone(envelope)
        self.assertIn(envelope.candidate_id, contracts)
        self.assertTrue(contracts[envelope.candidate_id]["adaptive_envelope"])
        self.assertEqual(
            {policy.executor_id for policy in policies},
            {envelope.binding.executor_id},
        )
        adaptive_candidates = tuple(
            row for row in values.candidates
            if row.assisted_operator_kind == "ffn"
            and row.route_family == "operator_split"
            and row.plan.execution_contract.execution_mode
                == "adaptive-split"
            and row.binding.executor_id in {
                self.phone.executor_id,
                alternate.executor_id,
            }
        )
        self.assertEqual(
            {row.binding.executor_id for row in adaptive_candidates},
            {self.phone.executor_id, alternate.executor_id},
        )
        for candidate in adaptive_candidates:
            contract = contracts[candidate.candidate_id]
            self.assertTrue(contract["adaptive_envelope"])
            probe_contracts = contract[
                "adaptive_decode_probe_contracts"
            ]
            self.assertTrue(probe_contracts)
            self.assertEqual(
                {row["executor_id"] for row in probe_contracts},
                {candidate.binding.executor_id},
            )
            self.assertEqual(
                {
                    row["operator_plan_sha256"]
                    for row in probe_contracts
                },
                {candidate.plan.plan_sha256},
            )

    def test_decode_envelope_uses_qualified_route_with_shadow_coordinator(self):
        alternate = replace(
            self.phone,
            executor_id="coordinator:adaptive-phone:qualified-route",
            backend="backend:phone:coalesced",
            maturity="SHADOW",
            evidence_ids=("synthetic-adaptive-phone-coalesced",),
            adapter_parameters={
                **dict(self.phone.adapter_parameters),
                "usb_batch_plan": "coalesced-batch",
            },
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            self.profile,
            composite_executors=(self.desktop, self.phone, alternate),
        ).to_json())
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        scheduler.register_model_manifest(self.manifest)
        snapshot = replace(
            self.snapshot,
            executors={
                **self.snapshot.executors,
                alternate.executor_id: executor_state(alternate.executor_id),
            },
        )
        value = request(
            "adaptive-qualified-route", input_tokens=8, output_tokens=30
        )
        candidates = scheduler.generate_automated_candidates(
            value, self.manifest.model_id, snapshot
        )
        qualified_route_id = None
        updated = []
        for row in candidates.candidates:
            if (
                row.binding.executor_id == alternate.executor_id
                and row.assisted_operator_kind == "ffn"
                and row.split_fraction_ppm == 500_000
            ):
                qualified_route_id = row.candidate_id
                row = replace(
                    row,
                    maturity="QUALIFIED",
                    admitted=True,
                    binding=replace(
                        row.binding,
                        ready=True,
                        eligibility_reasons=(),
                    ),
                    rejection_reasons=(),
                    cost=replace(
                        row.cost,
                        fleet_energy_lower_uj=40_000,
                        fleet_energy_uj=45_000,
                        fleet_energy_upper_uj=50_000,
                        component_energy_uj=45_000,
                        warm_execution_energy_lower_uj=40_000,
                        warm_execution_energy_uj=45_000,
                        warm_execution_energy_upper_uj=50_000,
                        energy_evidence="MEASURED",
                    ),
                )
            updated.append(row)
        self.assertIsNotNone(qualified_route_id)
        candidates = replace(candidates, candidates=tuple(updated))

        _, policies, envelope = adaptive_decode_policies(
            candidates,
            self.manifest,
            profile,
            value.output_tokens,
        )
        selected, _, _ = scheduler._select_automated_candidate(
            candidates,
            value,
            selection_mode="adaptive-decode",
        )

        self.assertIsNotNone(envelope)
        self.assertEqual(envelope.candidate_id, qualified_route_id)
        self.assertEqual(selected.candidate_id, qualified_route_id)
        self.assertEqual(
            {policy.executor_id for policy in policies},
            {alternate.executor_id},
        )

    def test_energy_aware_uses_grouped_fraction_cost_not_mixed_route_cost(self):
        value = request(
            "adaptive-grouped-energy", input_tokens=8, output_tokens=30
        )
        candidates = self.scheduler.generate_automated_candidates(
            value, self.manifest.model_id, self.snapshot
        )
        baseline_policy, policies, envelope = adaptive_decode_policies(
            candidates,
            self.manifest,
            self.profile,
            value.output_tokens,
        )
        self.assertIsNotNone(envelope)
        winner = max(policies, key=lambda row: row.split_fraction_ppm)
        baseline_energy = max(
            100,
            candidates.baseline.cost.warm_execution_energy_uj
                // value.output_tokens,
        )

        groups = []
        for index in range(2):
            request_id = f"adaptive-history-{index}"
            first = AdaptiveDecodeWindowReceipt(
                request_id=request_id,
                slot_id=0,
                window_index=0,
                token_start=0,
                token_end=4,
                context_length=value.input_tokens,
                active_batch=1,
                started_at_us=0,
                finished_at_us=4_000,
                policy=baseline_policy,
                applied_ack=None,
                fleet_energy_uj_by_domain={
                    "fleet": baseline_energy * 4
                },
                latency_per_token_us=1_000,
                phone_compute_us=0,
                usb_transfer_us=0,
                rpc_us=0,
                exposed_tail_us=0,
                output_valid=True,
                evidence_ids=("synthetic-grouped-energy",),
                energy_boundary_id=(
                    self.profile.placement_profile.energy_boundary_id
                ),
                energy_attribution_kind="isolated",
                failure_reason=None,
                previous_record_sha256="0" * 64,
                window_role="exploration",
            )
            second = AdaptiveDecodeWindowReceipt(
                request_id=request_id,
                slot_id=0,
                window_index=1,
                token_start=4,
                token_end=8,
                context_length=value.input_tokens,
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
                evidence_ids=("synthetic-grouped-energy",),
                energy_boundary_id=(
                    self.profile.placement_profile.energy_boundary_id
                ),
                energy_attribution_kind="isolated",
                failure_reason=None,
                previous_record_sha256=(
                    first.record_sha256.removeprefix("sha256:")
                ),
                window_role="exploitation",
            )
            groups.append(AdaptiveDecodeGroupedObservation(
                request_id=request_id,
                ticket_id=request_id + ":attempt:0",
                model_artifact_sha256=self.manifest.artifact_sha256,
                planning_profile_sha256=(
                    self.scheduler._runtime_capability_generation_sha256
                ),
                desktop_placement_sha256=(
                    baseline_policy.desktop_placement_sha256
                ),
                windows=(first, second),
                final_policy=winner,
                terminal_status="COMPLETED",
                state_history=(
                    "BASELINE", "PROBING", "EXPLOITING", "COMPLETED"
                ),
            ))
        body = {
            "groups": [row.to_json() for row in groups],
            "schema": ADAPTIVE_OBSERVATION_STORE_SCHEMA,
        }
        self.scheduler.load_adaptive_decode_observations({
            **body,
            "store_sha256": canonical_sha256(body),
        })
        contaminated = []
        for row in candidates.candidates:
            if row.candidate_id == envelope.candidate_id:
                reasons = (
                    "MODEL_EPOCH_AUDIT_ONLY",
                    "ROUTE_NOT_QUALIFIED",
                    "SLO_UPPER_BOUND",
                )
                row = replace(
                    row,
                    maturity="QUARANTINED",
                    admitted=False,
                    binding=replace(
                        row.binding,
                        ready=False,
                        eligibility_reasons=reasons,
                    ),
                    rejection_reasons=reasons,
                    cost=replace(
                        row.cost,
                        latency_evidence="ASSUMED",
                    ),
                )
            contaminated.append(row)
        candidates = replace(
            candidates, candidates=tuple(contaminated)
        )

        assumed_baseline = candidates.baseline
        assumed_cost = replace(
            assumed_baseline.cost,
            fleet_energy_lower_uj=1,
            fleet_energy_uj=1,
            fleet_energy_upper_uj=1,
            warm_execution_energy_lower_uj=1,
            warm_execution_energy_uj=1,
            warm_execution_energy_upper_uj=1,
            transition_energy_lower_uj=0,
            transition_energy_uj=0,
            transition_energy_upper_uj=0,
            energy_evidence="ASSUMED",
        )
        candidates = replace(
            candidates,
            candidates=tuple(
                replace(row, cost=assumed_cost)
                if row.candidate_id == candidates.baseline_route_id
                else row
                for row in candidates.candidates
            ),
        )

        refined = self.scheduler._apply_adaptive_history_costs(
            candidates, value, self.manifest
        )
        self.assertEqual(
            refined.baseline.cost.energy_evidence, "MEASURED"
        )
        self.assertEqual(refined.baseline.maturity, "QUALIFIED")
        self.assertGreater(
            refined.baseline.cost.fleet_energy_lower_uj, 1
        )
        audited_rows = []
        for row in refined.candidates:
            if row.candidate_id == envelope.candidate_id:
                reasons = tuple(sorted(set(
                    row.rejection_reasons
                    + ("MODEL_EPOCH_AUDIT_ONLY",)
                )))
                row = replace(
                    row,
                    admitted=False,
                    binding=replace(
                        row.binding,
                        ready=False,
                        eligibility_reasons=reasons,
                    ),
                    rejection_reasons=reasons,
                )
            elif row.candidate_id != refined.baseline_route_id:
                reasons = tuple(sorted(set(
                    row.rejection_reasons + ("SEARCH_COVERAGE_ONLY",)
                )))
                row = replace(
                    row,
                    admitted=False,
                    binding=replace(
                        row.binding,
                        ready=False,
                        eligibility_reasons=reasons,
                    ),
                    rejection_reasons=reasons,
                )
            audited_rows.append(row)
        audited = replace(refined, candidates=tuple(audited_rows))
        strict, _, _ = self.scheduler._select_automated_candidate(
            audited,
            value,
            excluded_route_ids=tuple(
                row.candidate_id
                for row in audited.candidates
                if row.candidate_id not in {
                    audited.baseline_route_id, envelope.candidate_id
                }
            ),
            selection_mode="energy-aware",
        )
        prospective, _, _ = (
            self.scheduler._select_model_placement_candidate(
                audited,
                value,
                selection_mode="energy-aware",
            )
        )

        self.assertEqual(strict.candidate_id, audited.baseline_route_id)
        self.assertEqual(prospective.candidate_id, envelope.candidate_id)
        self.assertFalse(next(
            row for row in audited.candidates
            if row.candidate_id == envelope.candidate_id
        ).admitted)

        demand, action = self.scheduler._evaluate_model_placement(
            value,
            self.manifest,
            self.snapshot,
            value.arrival_us,
            None,
            "energy-aware",
        )
        apply_history = self.scheduler._apply_adaptive_history_costs

        def apply_history_with_cold_rejection(*args, **kwargs):
            result = apply_history(*args, **kwargs)
            rows = []
            for row in result.candidates:
                if row.candidate_id != envelope.candidate_id:
                    rows.append(row)
                    continue
                reasons = tuple(sorted(set(
                    row.rejection_reasons
                    + ("COLD_RESIDENCY_BREAK_EVEN",)
                )))
                rows.append(replace(
                    row,
                    admitted=False,
                    binding=replace(
                        row.binding,
                        ready=False,
                        eligibility_reasons=reasons,
                    ),
                    rejection_reasons=reasons,
                ))
            return replace(result, candidates=tuple(rows))

        def authorize_cold_portfolio(candidate_set, _manifest, _request, *, snapshot, observed_at_us):
            self.assertIs(snapshot, self.snapshot)
            self.assertEqual(observed_at_us, value.arrival_us)
            rows = []
            for row in candidate_set.candidates:
                if row.candidate_id != envelope.candidate_id:
                    rows.append(row)
                    continue
                reasons = tuple(
                    reason for reason in row.rejection_reasons
                    if reason != "COLD_RESIDENCY_BREAK_EVEN"
                )
                rows.append(replace(
                    row,
                    admitted=not reasons,
                    binding=replace(
                        row.binding,
                        ready=not reasons,
                        eligibility_reasons=reasons,
                    ),
                    rejection_reasons=reasons,
                ))
            return replace(candidate_set, candidates=tuple(rows))

        with mock.patch.object(
            self.scheduler,
            "_apply_adaptive_history_costs",
            side_effect=apply_history_with_cold_rejection,
        ), mock.patch.object(
            self.scheduler,
            "_apply_phone_residency_portfolio_authorization",
            side_effect=authorize_cold_portfolio,
        ) as authorize:
            (
                published_candidates,
                published_selected,
                _,
                _,
                published_epoch,
                published_templates,
            ) = self.scheduler._materialize_model_placement_candidate(
                candidate_set=audited,
                prospective=prospective,
                request=value,
                manifest=self.manifest,
                snapshot=self.snapshot,
                observed_at_us=value.arrival_us,
                selection_mode="energy-aware",
                invalidation_reason="MODEL_PLACEMENT_EPOCH_ABSENT",
                runtime_rejections={},
                current_epoch=None,
                demand_snapshot=demand,
                placement_action=action,
                residency_holds={},
            )
        authorize.assert_called_once()
        self.assertIsNotNone(published_epoch)
        self.assertIsNotNone(published_templates)
        self.assertEqual(
            published_selected.candidate_id, envelope.candidate_id
        )
        rematerialized = next(
            row for row in published_candidates.candidates
            if row.candidate_id == published_selected.candidate_id
        )
        self.assertTrue(rematerialized.admitted)
        self.assertNotIn(
            "MODEL_EPOCH_AUDIT_ONLY",
            rematerialized.rejection_reasons,
        )

        selected = next(
            row for row in refined.candidates
            if row.candidate_id == envelope.candidate_id
        )

        self.assertEqual(selected.candidate_id, envelope.candidate_id)
        self.assertFalse(selected.admitted)
        self.assertIn(
            "MODEL_EPOCH_AUDIT_ONLY", selected.rejection_reasons
        )
        self.assertEqual(selected.cost.energy_evidence, "MEASURED")
        self.assertEqual(selected.cost.latency_evidence, "MEASURED")
        self.assertEqual(selected.maturity, "QUALIFIED")
        self.assertNotIn("SLO_UPPER_BOUND", selected.rejection_reasons)
        self.assertTrue(
            selected.residency_break_even["adaptive_history_applied"]
        )
        self.assertEqual(
            selected.residency_break_even[
                "adaptive_history_selected_fraction_ppm"
            ],
            winner.split_fraction_ppm,
        )
        self.assertLess(
            selected.residency_break_even[
                "paired_warm_energy_delta_upper_uj"
            ],
            0,
        )
        parent = refined.baseline
        parent_lower = parent.cost.fleet_energy_lower_uj
        self.assertIsNotNone(parent_lower)
        transition_lower = selected.cost.transition_energy_lower_uj
        transition = selected.cost.transition_energy_uj
        transition_upper = selected.cost.transition_energy_upper_uj
        self.assertIsNotNone(transition_lower)
        self.assertIsNotNone(transition)
        self.assertIsNotNone(transition_upper)
        warm_lower = selected.cost.warm_execution_energy_lower_uj
        warm = selected.cost.warm_execution_energy_uj
        self.assertIsNotNone(warm_lower)
        self.assertIsNotNone(warm)
        overlapping_warm_upper = max(warm, parent_lower)
        overlapping_cost = replace(
            selected.cost,
            fleet_energy_upper_uj=(
                overlapping_warm_upper + transition_upper
            ),
            warm_execution_energy_upper_uj=overlapping_warm_upper,
        )
        overlapping = replace(selected, cost=overlapping_cost)
        overlapping_set = replace(
            audited,
            candidates=tuple(
                overlapping
                if row.candidate_id == selected.candidate_id else row
                for row in audited.candidates
            ),
        )
        paired_selected, _, _ = (
            self.scheduler._select_model_placement_candidate(
                overlapping_set,
                value,
                selection_mode="energy-aware",
            )
        )
        self.assertGreater(
            overlapping.cost.fleet_energy_upper_uj,
            parent_lower
                * (1_000_000 - self.profile.minimum_energy_saving_ppm)
                // 1_000_000,
        )
        self.assertEqual(
            paired_selected.candidate_id, selected.candidate_id
        )
        paired_epoch, _ = self.scheduler._propose_model_placement_epoch(
            request=value,
            manifest=self.manifest,
            candidate_set=overlapping_set,
            selected=overlapping,
            observed_at_us=value.arrival_us,
            selection_mode="energy-aware",
            invalidation_reason="LEARNING_GENERATION_CHANGED",
            snapshot=self.snapshot,
            demand_snapshot=demand,
            placement_action=action,
            current_epoch=None,
        )
        self.assertEqual(
            paired_epoch.selected_route_id, selected.candidate_id
        )

        source = next(
            row for row in candidates.candidates
            if row.candidate_id == envelope.candidate_id
        )
        warm = (
            source.cost.warm_execution_energy_lower_uj,
            source.cost.warm_execution_energy_uj,
            source.cost.warm_execution_energy_upper_uj,
        )
        self.assertTrue(all(value is not None for value in warm))
        transition = RuntimeTransitionPlan(
            transition_id="load:adaptive-model:helper-c:cold",
            device_id="helper-c",
            source_state="cold",
            target_state="hot",
            latency_us=100,
            energy_uj=1_000,
            resource_ids=(source.plan.resource_ids[0],),
            maturity="QUALIFIED",
            energy_maturity="SHADOW",
            executor_id=source.binding.executor_id,
        )
        cold_plan = replace(
            source.plan,
            residency_variant="cold",
            transitions=(transition,),
        )
        cold_cost = replace(
            source.cost,
            fleet_energy_lower_uj=warm[0] + 500,
            fleet_energy_uj=warm[1] + 1_000,
            fleet_energy_upper_uj=warm[2] + 1_500,
            transition_energy_lower_uj=500,
            transition_energy_uj=1_000,
            transition_energy_upper_uj=1_500,
        )
        cold_reasons = ("ROUTE_NOT_QUALIFIED",)
        cold_candidate = replace(
            source,
            plan=cold_plan,
            binding=replace(
                source.binding,
                ready=False,
                eligibility_reasons=cold_reasons,
                operator_plan_sha256=cold_plan.plan_sha256,
            ),
            cost=cold_cost,
            maturity="SHADOW",
            admitted=False,
            rejection_reasons=cold_reasons,
        )
        cold_candidates = replace(
            candidates,
            candidates=tuple(
                cold_candidate
                if row.candidate_id == cold_candidate.candidate_id else row
                for row in candidates.candidates
            ),
        )
        learned_cold = self.scheduler._apply_adaptive_history_costs(
            cold_candidates, value, self.manifest
        )
        cold = next(
            row for row in learned_cold.candidates
            if row.candidate_id == cold_candidate.candidate_id
        )
        self.assertTrue(
            cold.residency_break_even["adaptive_history_applied"]
        )
        self.assertEqual(cold.cost.energy_evidence, "ASSUMED")
        self.assertEqual(cold.maturity, "SHADOW")
        self.assertIn("ROUTE_NOT_QUALIFIED", cold.rejection_reasons)

        hot_plan = replace(
            cold.plan,
            residency_variant="hot",
            transitions=(),
        )
        hot_cost = replace(
            cold.cost,
            fleet_energy_lower_uj=(
                cold.cost.warm_execution_energy_lower_uj
            ),
            fleet_energy_uj=cold.cost.warm_execution_energy_uj,
            fleet_energy_upper_uj=(
                cold.cost.warm_execution_energy_upper_uj
            ),
            transition_energy_lower_uj=0,
            transition_energy_uj=0,
            transition_energy_upper_uj=0,
        )
        hot_candidate = replace(
            cold,
            plan=hot_plan,
            binding=replace(
                cold.binding,
                operator_plan_sha256=hot_plan.plan_sha256,
            ),
            cost=hot_cost,
        )
        hot_candidates = replace(
            learned_cold,
            candidates=tuple(
                hot_candidate
                if row.candidate_id == hot_candidate.candidate_id else row
                for row in learned_cold.candidates
            ),
        )
        learned_hot = self.scheduler._apply_adaptive_history_costs(
            hot_candidates, value, self.manifest
        )
        hot = next(
            row for row in learned_hot.candidates
            if row.candidate_id == hot_candidate.candidate_id
        )
        self.assertEqual(hot.cost.energy_evidence, "MEASURED")
        self.assertEqual(hot.cost.latency_evidence, "MEASURED")
        self.assertEqual(hot.maturity, "QUALIFIED")
        self.assertTrue(hot.admitted)

        updated_profile = replace(
            self.profile,
            catalog_id=self.profile.catalog_id + ":ownership-update",
            executors=tuple(
                replace(
                    row,
                    exclusive_residency_resource_id="compute:helper-c",
                )
                if row.device_id == "helper-c" else row
                for row in self.profile.executors
            ),
            composite_executors=tuple(
                replace(
                    row,
                    replacement_group_by_device={
                        **dict(row.replacement_group_by_device),
                        "helper-c": "compute:helper-c",
                    },
                )
                if row.executor_id == self.phone.executor_id else row
                for row in self.profile.composite_executors
            ),
        )
        target = UnifiedScheduler.for_runtime_discovery("enforce")
        target.register_runtime_capabilities(updated_profile)
        target.register_model_manifest(self.manifest)
        target.load_adaptive_decode_observations(
            {
                **body,
                "store_sha256": canonical_sha256(body),
            },
            source_catalog=self.profile,
        )
        target_candidates = target.generate_automated_candidates(
            value, self.manifest.model_id, self.snapshot
        )
        target_refined = target._apply_adaptive_history_costs(
            target_candidates, value, self.manifest
        )
        rebound = tuple(
            row for row in target_refined.candidates
            if row.binding.executor_id == self.phone.executor_id
            and (row.residency_break_even or {}).get(
                "adaptive_history_applied"
            )
        )
        self.assertTrue(rebound)

    def test_decode_envelope_replaces_nondividing_catalog_quantum(self):
        phone = replace(
            self.phone,
            adapter_parameters={
                **dict(self.phone.adapter_parameters),
                "ffn_column_quantum": 3,
            },
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            self.profile,
            composite_executors=(self.desktop, phone),
        ).to_json())
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        scheduler.register_model_manifest(self.manifest)

        values = scheduler.generate_automated_candidates(
            request("adaptive-quantum", input_tokens=8, output_tokens=30),
            self.manifest.model_id,
            self.snapshot,
        )
        assisted = tuple(
            row for row in values.candidates
            if row.paired_baseline_route_id == values.baseline_route_id
            and row.assisted_operator_kind == "ffn"
        )

        self.assertTrue(assisted)
        quanta = {
            row.plan.adapter_parameters["ffn_column_quantum"]
            for row in assisted
        }
        self.assertEqual(len(quanta), 1)
        self.assertGreater(next(iter(quanta)), 3)
        self.assertTrue(all(
            (
                self.manifest.feed_forward_length
                * row.split_fraction_ppm // 1_000_000
            ) % row.plan.adapter_parameters["ffn_column_quantum"] == 0
            for row in assisted
        ))

    def test_decode_transport_capacity_covers_live_parallel_batch(self):
        phone = replace(
            self.phone,
            adapter_parameters={
                **dict(self.phone.adapter_parameters),
                "parallel": 4,
            },
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            self.profile,
            composite_executors=(self.desktop, phone),
        ).to_json())
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        scheduler.register_model_manifest(self.manifest)
        values = scheduler.generate_automated_candidates(
            request("adaptive-batch", input_tokens=8, output_tokens=30),
            self.manifest.model_id,
            self.snapshot,
        )
        assisted = tuple(
            row for row in values.candidates
            if row.paired_baseline_route_id == values.baseline_route_id
            and row.assisted_operator_kind == "ffn"
        )
        self.assertTrue(assisted)
        self.assertEqual(
            {row.plan.adapter_parameters["ffn_max_tokens"] for row in assisted},
            {4},
        )

    def test_decode_transport_separates_cost_and_ring_capacity_profiles(self):
        identity = "sha256:" + "3" * 64

        def link(
            link_id: str,
            source: str,
            target: str,
            payload: int,
            depth: int,
        ) -> TransferLink:
            return TransferLink(
                link_id=link_id,
                source_device=source,
                target_device=target,
                fixed_latency_us=10,
                bandwidth_bytes_per_s=100_000_000,
                fixed_dynamic_uj=0,
                dynamic_pj_per_byte=0,
                domain_active_power_mw={},
                status="measured",
                ready=True,
                evidence_ids=(identity,),
                minimum_payload_bytes=payload,
                maximum_payload_bytes=payload,
                queue_depth=depth,
                concurrent_streams=depth,
                allocator="devmem",
                full_duplex=True,
                transport_generation="synthetic-async-ring-v1",
                transport_profile_id=link_id + ":profile",
                usbfs_available_bytes=16_777_216,
                slot_safety_bytes=65_536,
                qualification_identity_sha256=identity,
            )

        small_h2d = link(
            "small-h2d", "host-a", "helper-c", 10_240, 1
        )
        small_d2h = link(
            "small-d2h", "helper-c", "host-a", 10_240, 1
        )
        capacity_h2d = link(
            "capacity-h2d", "host-a", "helper-c", 81_920, 4
        )
        capacity_d2h = link(
            "capacity-d2h", "helper-c", "host-a", 81_920, 4
        )
        profile = replace(
            self.profile.placement_profile,
            links=(small_h2d, small_d2h, capacity_h2d, capacity_d2h),
        )

        parameters = AutomatedRouteCompiler._transport_adapter_parameters(
            profile,
            (small_h2d.link_id, small_d2h.link_id),
            required_maximum_payload_bytes=40_960,
            slot_payload_multiplier=4,
        )

        self.assertEqual(parameters["usb_queue_depth"], 4)
        self.assertEqual(parameters["usb_concurrent_streams"], 4)
        self.assertEqual(parameters["usb_max_payload_bytes"], 40_960)
        self.assertEqual(
            parameters["usb_h2d_transport_profile_id"],
            small_h2d.transport_profile_id,
        )
        self.assertEqual(
            parameters["usb_d2h_transport_profile_id"],
            small_d2h.transport_profile_id,
        )
        self.assertEqual(
            parameters["usb_capacity_h2d_transport_profile_id"],
            capacity_h2d.transport_profile_id,
        )
        self.assertEqual(
            parameters["usb_capacity_d2h_transport_profile_id"],
            capacity_d2h.transport_profile_id,
        )

    def test_split_row_transport_uses_exact_row_qualification(self):
        identity = "sha256:" + "4" * 64

        def link(link_id: str, source: str, target: str) -> TransferLink:
            return TransferLink(
                link_id=link_id,
                source_device=source,
                target_device=target,
                fixed_latency_us=10,
                bandwidth_bytes_per_s=100_000_000,
                fixed_dynamic_uj=0,
                dynamic_pj_per_byte=0,
                domain_active_power_mw={},
                status="measured",
                ready=True,
                evidence_ids=(identity,),
                minimum_payload_bytes=10_240,
                maximum_payload_bytes=10_240,
                queue_depth=4,
                concurrent_streams=4,
                allocator="devmem",
                full_duplex=True,
                transport_generation="synthetic-async-ring-v1",
                transport_profile_id=link_id + ":profile",
                usbfs_available_bytes=16_777_216,
                slot_safety_bytes=65_536,
                qualification_identity_sha256=identity,
            )

        h2d = link("row-h2d", "host-a", "helper-c")
        d2h = link("row-d2h", "helper-c", "host-a")
        profile = replace(
            self.profile.placement_profile,
            links=(h2d, d2h),
        )
        parameters = AutomatedRouteCompiler._transport_adapter_parameters(
            profile,
            (h2d.link_id, d2h.link_id),
            required_maximum_payload_bytes=40_960,
            slot_payload_multiplier=4,
            batch_plan="split-row",
            maximum_tokens=4,
        )
        self.assertEqual(parameters["usb_max_payload_bytes"], 40_960)
        self.assertEqual(
            parameters["usb_qualified_transfer_payload_bytes"], 10_240
        )
        self.assertEqual(parameters["usb_queue_depth"], 4)
        with self.assertRaisesRegex(
            RouteGenerationError,
            "maximum payload is not qualified",
        ):
            AutomatedRouteCompiler._transport_adapter_parameters(
                profile,
                (h2d.link_id, d2h.link_id),
                required_maximum_payload_bytes=40_960,
                slot_payload_multiplier=4,
                batch_plan="coalesced-batch",
                maximum_tokens=4,
            )

    def test_qualified_desktop_placement_allows_shadow_cost_calibration(self):
        values = self.scheduler.generate_automated_candidates(
            request("adaptive-shadow", input_tokens=8, output_tokens=30),
            self.manifest.model_id,
            self.snapshot,
        )
        candidates = tuple(
            replace(row, maturity="SHADOW")
            if row.candidate_id == values.baseline_route_id else row
            for row in values.candidates
        )
        shadow_values = replace(values, candidates=candidates)

        baseline, policies, envelope = adaptive_decode_policies(
            shadow_values,
            self.manifest,
            self.profile,
            30,
        )

        self.assertEqual(
            shadow_values.baseline.maturity, "SHADOW"
        )
        self.assertTrue(baseline.baseline)
        self.assertTrue(policies)
        self.assertIsNotNone(envelope)
        self.assertEqual(
            envelope.plan.desktop_placement_sha256,
            self.profile.desktop_control_by_artifact[
                self.manifest.artifact_sha256
            ].placement_sha256,
        )

    def test_cold_break_even_rejection_remains_available_for_decode_probe(self):
        values = self.scheduler.generate_automated_candidates(
            request("adaptive-cold-probe", input_tokens=8, output_tokens=30),
            self.manifest.model_id,
            self.snapshot,
        )
        candidates = []
        changed = 0
        for row in values.candidates:
            if (
                row.paired_baseline_route_id == values.baseline_route_id
                and row.assisted_operator_kind == "ffn"
            ):
                reasons = tuple(sorted(set(
                    row.rejection_reasons
                    + ("COLD_RESIDENCY_BREAK_EVEN",)
                )))
                row = replace(
                    row,
                    admitted=False,
                    binding=replace(
                        row.binding,
                        ready=False,
                        eligibility_reasons=reasons,
                    ),
                    rejection_reasons=reasons,
                    residency_break_even={
                        "passed": False,
                        "transition_energy_uj": 100,
                    },
                )
                changed += 1
            candidates.append(row)
        self.assertGreater(changed, 0)

        _, policies, envelope = adaptive_decode_policies(
            replace(values, candidates=tuple(candidates)),
            self.manifest,
            self.profile,
            30,
        )

        self.assertTrue(policies)
        self.assertIsNotNone(envelope)
        self.assertIn(
            "COLD_RESIDENCY_BREAK_EVEN", envelope.rejection_reasons
        )

    def test_unknown_contention_cost_allows_measurement_but_not_unsafe_helpers(self):
        scheduler, profile, _, value, values, _, _ = self._cpu_parent_candidates()
        for safety_reason in (None, "TRANSPORT_PROFILE_INCOMPLETE", "MEMORY_CAPACITY"):
            with self.subTest(safety_reason=safety_reason):
                candidates = []
                for row in values.candidates:
                    if row.assisted_operator_kind == "ffn":
                        reasons = tuple(sorted({
                            *row.rejection_reasons, "MARGINAL_SYSTEM_COST_UNKNOWN",
                            *((safety_reason,) if safety_reason else ()),
                        }))
                        row = replace(row, admitted=False, rejection_reasons=reasons,
                                      binding=replace(row.binding, ready=False, eligibility_reasons=reasons))
                    candidates.append(row)
                candidate_set = replace(values, candidates=tuple(candidates))
                _, policies, envelope = adaptive_decode_policies(
                    candidate_set, self.manifest, profile, value.output_tokens)
                if safety_reason:
                    self.assertFalse(policies)
                    self.assertIsNone(envelope)
                else:
                    self.assertTrue(policies)
                    self.assertFalse(envelope.admitted)
                    self.assertIn("MARGINAL_SYSTEM_COST_UNKNOWN", envelope.rejection_reasons)
                    opportunities = scheduler._compact_helper_opportunities(candidate_set, value)
                    self.assertTrue(opportunities)
                    self.assertTrue(all(row.evidence_state == "LEARNING" for row in opportunities))

    def test_ready_helper_keeps_the_desktop_transport_batch_plan(self):
        scheduler, profile, _, value, values, _, _ = self._cpu_parent_candidates()
        _, _, envelope = adaptive_decode_policies(values, self.manifest, profile, value.output_tokens)
        ready = SimpleNamespace(layout=SimpleNamespace(
            geometry_sha256=envelope.plan.adapter_parameters["phone_shard_set_geometry_sha256"],
            shards=envelope.plan.execution_contract.phone_shards,
        ))
        candidates = []
        for row in values.candidates:
            if row.assisted_operator_kind == "ffn":
                plan = replace(row.plan, adapter_parameters={**row.plan.adapter_parameters,
                                                             "usb_batch_plan": "coalesced-batch"})
                row = replace(row, plan=plan, binding=replace(row.binding,
                                                            operator_plan_sha256=plan.plan_sha256))
            candidates.append(row)
        values = replace(values, candidates=tuple(candidates))
        for batch_plan, expected in (("coalesced-batch", True), ("split-row", False)):
            filtered = scheduler._candidate_set_for_ready_phone_layout(
                values, self.manifest.artifact_sha256, ready, usb_batch_plan=batch_plan)
            self.assertEqual(any(row.assisted_operator_kind == "ffn" for row in filtered.candidates), expected)
            self.assertEqual(filtered.baseline, values.baseline)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _active_ticket(self, request_id: str):
        value = request(request_id, input_tokens=8, output_tokens=30)
        ticket = self.scheduler.submit_automated_request(
            value,
            self.manifest.model_id,
            self.snapshot,
            selection_mode="adaptive-decode",
        )
        epoch_ns = time.monotonic_ns() - ticket.decision.start_us * 1_000
        return self.scheduler.wait_runtime_request(request_id, epoch_ns)

    def test_runtime_default_uses_configured_probe_threshold(self):
        scheduler = UnifiedScheduler.for_runtime_discovery(
            "enforce",
            adaptive_decode_config=AdaptiveDecodeConfig(
                minimum_remaining_tokens=100,
            ),
        )
        scheduler.register_runtime_capabilities(self.profile)
        scheduler.register_model_manifest(self.manifest)
        value = request(
            "configured-threshold", input_tokens=8, output_tokens=30
        )
        ticket = scheduler.submit_automated_request(
            value,
            self.manifest.model_id,
            self.snapshot,
            selection_mode="calibration",
        )
        epoch_ns = time.monotonic_ns() - ticket.decision.start_us * 1_000
        ticket = scheduler.wait_runtime_request(value.request_id, epoch_ns)

        directive = scheduler.start_adaptive_decode(
            value.request_id,
            slot_id=3,
            first_token_index=1,
            at_us=2_000,
        )

        self.assertEqual(directive.state, "EXPLOITING")
        self.assertIsNotNone(directive.control)
        self.assertEqual(
            directive.control.policy.route_id,
            ticket.decision.route_id,
        )
        self.assertEqual(ticket.binding.executor_id, self.phone.executor_id)
        self.assertEqual(
            ticket.execution_plan.execution_contract.execution_mode,
            "adaptive-split",
        )

    def test_calibration_rejects_adaptive_route_without_safe_envelope(self):
        cpu_placements = tuple(
            RuntimeCompositeOperatorPlacement(
                operator_id=operator.operator_id,
                primary_device_id="host-a",
                helper_device_id=None,
                split_axis="none",
                split_fraction_ppm=0,
            )
            for operator in self.manifest.operators
        )
        cpu_parent = replace(
            self.desktop,
            executor_id="coordinator:cpu-parent",
            endpoint="synthetic://cpu-parent",
            backend="backend:cpu",
            participant_device_ids=("host-a",),
            participant_resource_ids={
                "host-a": ("compute:host-a",),
            },
            resource_ids=("compute:host-a",),
            operator_plan_protocol="synthetic-cpu-v1",
            operator_placements=cpu_placements,
        )
        phone = replace(
            self.phone,
            participant_device_ids=("host-a", "helper-c"),
            participant_resource_ids={
                "host-a": ("compute:host-a",),
                "helper-c": ("compute:helper-c",),
            },
            resource_ids=(
                "compute:host-a",
                "compute:helper-c",
                "link:usb-out",
                "link:usb-in",
            ),
            baseline_executor_id=cpu_parent.executor_id,
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            self.profile,
            composite_executors=(self.desktop, cpu_parent, phone),
        ).to_json())
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        scheduler.register_model_manifest(self.manifest)
        snapshot = replace(
            self.snapshot,
            executors={
                **self.snapshot.executors,
                cpu_parent.executor_id: executor_state(
                    cpu_parent.executor_id
                ),
                phone.executor_id: executor_state(phone.executor_id),
            },
        )
        value = request(
            "unsafe-adaptive-parent", input_tokens=8, output_tokens=30
        )
        candidates = scheduler.generate_automated_candidates(
            value, self.manifest.model_id, snapshot
        )
        unbound = tuple(
            row for row in candidates.candidates
            if row.binding.executor_id == phone.executor_id
        )
        self.assertTrue(unbound)
        self.assertTrue(all(
            row.plan.execution_contract.execution_mode
                == "adaptive-split"
            and row.paired_baseline_route_id
                != candidates.baseline_route_id
            for row in unbound
        ))

        ticket = scheduler.submit_automated_request(
            value,
            self.manifest.model_id,
            snapshot,
            selection_mode="calibration",
        )

        self.assertEqual(
            ticket.binding.executor_id, self.desktop.executor_id
        )
        rejected = dict(ticket.decision.rejected)
        self.assertTrue(all(
            rejected[row.candidate_id]
                == "ADAPTIVE_EXECUTION_CONTRACT_ABSENT"
            for row in unbound
        ))

    def _seal_baseline_tail(self, ticket, *, reason: str = "test_tail"):
        directive = self.scheduler.start_adaptive_decode(
            ticket.request.request_id,
            slot_id=3,
            first_token_index=1,
            at_us=2_000,
            config=AdaptiveDecodeConfig(
                minimum_remaining_tokens=4,
                minimum_window_tokens=2,
                maximum_window_tokens=2,
                maximum_probe_tokens=20,
                maximum_probe_candidates=2,
                measurement_resolution_us=1,
                transition_cost_us=1,
                transition_energy_uj=1,
                warmup_windows_per_policy=0,
            ),
        )
        token_index = directive.target_token_index
        self.scheduler.seal_adaptive_decode_tail(
            ticket.request.request_id,
            slot_id=3,
            token_index=token_index,
            reason=reason,
        )
        boundary = self.scheduler.adaptive_decode_boundary(
            ticket.request.request_id,
            slot_id=3,
            token_index=token_index,
            at_us=4_000,
            terminal=True,
        ).boundary
        return self.scheduler.record_adaptive_decode_window(
            ticket.request.request_id,
            boundary,
            AdaptiveDecodeRawWindowObservation(
                fleet_energy_uj_by_domain={"fleet": 200},
                phone_compute_us=0,
                usb_transfer_us=0,
                rpc_us=0,
                exposed_tail_us=0,
                output_valid=True,
                evidence_ids=("synthetic-sealed-tail",),
                energy_boundary_id="synthetic-fleet",
                energy_attribution_kind="isolated",
            ),
        )

    def test_sealed_tail_survives_checkpoint_until_physical_completion(self):
        ticket = self._active_ticket("adaptive-sealed-tail")
        directive = self._seal_baseline_tail(
            ticket, reason="cohort_membership_changed"
        )

        self.assertEqual(directive.reason, "COHORT_MEMBERSHIP_CHANGED")
        self.assertEqual(
            self.scheduler.adaptive_decode_snapshot(
                ticket.request.request_id
            )["tail_state"],
            "SEALED",
        )
        checkpoint = self.scheduler._runtime_transaction_checkpoint()
        self.scheduler._restore_runtime_transaction(checkpoint)
        preview = self.scheduler.preview_adaptive_decode_completion(
            ticket.request.request_id
        )
        completed = self.scheduler.complete_adaptive_decode(
            ticket.request.request_id
        )

        self.assertEqual(
            completed.grouped_observation_sha256,
            preview.grouped_observation_sha256,
        )
        self.assertGreater(completed.unmeasured_tail_tokens, 0)
        with self.assertRaisesRegex(Exception, "not active"):
            self.scheduler.preview_adaptive_decode_completion(
                ticket.request.request_id
            )

    def _cohort_scheduler(self):
        phone = replace(
            self.phone,
            adapter_parameters={
                **dict(self.phone.adapter_parameters),
                "decode_cohort_formation_us": 1_000_000,
                "parallel": 4,
                "usb_concurrent_streams": 4,
                "usb_queue_depth": 4,
            },
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            self.profile,
            composite_executors=(self.desktop, phone),
        ).to_json())
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        scheduler.register_model_manifest(self.manifest)
        return scheduler

    def test_singleton_cohort_returns_to_request_owned_adaptive_path(self):
        scheduler = self._cohort_scheduler()
        ticket = scheduler.submit_automated_request(
            request("adaptive-singleton", input_tokens=8, output_tokens=30),
            self.manifest.model_id,
            self.snapshot,
            selection_mode="adaptive-decode",
        )
        self.assertEqual(ticket.decode_cohort.active_batch, 1)
        self.assertTrue(
            scheduler.runtime_decode_cohort_snapshot()["cohorts"]
        )
        epoch_ns = time.monotonic_ns() - ticket.decision.start_us * 1_000
        active = scheduler.wait_runtime_request(
            ticket.request.request_id, epoch_ns
        )

        self.assertIsNone(active.decode_cohort)
        self.assertTrue(all(
            row.owner_id == ticket.request.request_id
            for row in active.decision.leases
        ))
        self.assertEqual(
            scheduler.runtime_decode_cohort_snapshot()["cohorts"], {}
        )
        scheduler.cancel_runtime_request(
            active.request.request_id,
            active.decision.start_us,
            "synthetic_singleton_cleanup",
        )

    def test_replan_replaces_prospective_singleton_cohort_atomically(self):
        scheduler = self._cohort_scheduler()
        ticket = scheduler.submit_automated_request(
            request(
                "adaptive-singleton-replan",
                input_tokens=8,
                output_tokens=30,
            ),
            self.manifest.model_id,
            self.snapshot,
            selection_mode="adaptive-decode",
        )
        self.assertIsNotNone(ticket.decode_cohort)
        changed = scheduler._runtime_controller.invalidate_queued_attempts(
            (ticket.request.request_id,),
            "residency_projection_invalid",
            ticket.request.arrival_us,
            cancel_owner=scheduler.cancel,
            release_memory=scheduler._runtime_memory.release_owner,
        )
        self.assertEqual(changed, (ticket.request.request_id,))
        wake = scheduler._runtime_controller.wait(
            ticket.request.request_id,
            time.monotonic_ns() - ticket.decision.start_us * 1_000,
        )
        self.assertEqual(wake.dispatch_state, "REPLAN_REQUIRED")
        self.assertIsNotNone(wake.dispatch_receipt)

        replacement = scheduler.replan_automated_request(
            ticket.request.request_id,
            observed_at_us=ticket.request.arrival_us,
            reason=wake.dispatch_receipt.wake_reason,
            snapshot=self.snapshot,
            expected_ticket_id=ticket.ticket_id,
            expected_queue_generation=(
                wake.dispatch_receipt.queue_generation
            ),
        )

        self.assertEqual(replacement.attempt_index, 1)
        self.assertEqual(replacement.previous_ticket_id, ticket.ticket_id)
        self.assertIsNotNone(replacement.decode_cohort)
        cohort_state = scheduler.runtime_decode_cohort_snapshot()
        self.assertEqual(len(cohort_state["cohorts"]), 1)
        current_cohort = next(iter(cohort_state["cohorts"].values()))
        self.assertEqual(
            current_cohort["member_request_ids"],
            [ticket.request.request_id],
        )

    def test_replan_joins_and_extends_existing_decode_cohort_atomically(self):
        scheduler = self._cohort_scheduler()
        leader = scheduler.submit_automated_request(
            request("adaptive-replan-leader", input_tokens=8, output_tokens=30),
            self.manifest.model_id,
            self.snapshot,
            selection_mode="adaptive-decode",
        )
        phone_unready = replace(
            self.snapshot,
            executors={
                **self.snapshot.executors,
                self.phone.executor_id: replace(
                    self.snapshot.executors[self.phone.executor_id],
                    ready=False,
                ),
            },
        )
        follower = scheduler.submit_automated_request(
            request(
                "adaptive-replan-follower",
                arrival_us=1_001,
                input_tokens=8,
                output_tokens=60,
            ),
            self.manifest.model_id,
            phone_unready,
            selection_mode="adaptive-decode",
        )
        self.assertIsNotNone(leader.decode_cohort)
        self.assertIsNone(follower.decode_cohort)
        changed = scheduler._runtime_controller.require_queued_replan(
            follower.request.request_id,
            "capacity_released_early",
        )
        self.assertTrue(changed)
        wake = scheduler._runtime_controller.wait(
            follower.request.request_id,
            time.monotonic_ns() - follower.decision.start_us * 1_000,
        )

        replacement = scheduler.replan_automated_request(
            follower.request.request_id,
            observed_at_us=follower.request.arrival_us,
            reason=wake.dispatch_receipt.wake_reason,
            snapshot=self.snapshot,
            expected_ticket_id=follower.ticket_id,
            expected_queue_generation=wake.dispatch_receipt.queue_generation,
        )

        leader = scheduler.runtime_ticket(leader.request.request_id)
        self.assertEqual(replacement.attempt_index, 1)
        self.assertEqual(replacement.decode_cohort, leader.decode_cohort)
        self.assertEqual(replacement.decode_cohort.active_batch, 2)
        self.assertEqual(
            tuple(row.token for row in replacement.decision.leases),
            replacement.decode_cohort.shared_lease_tokens,
        )
        self.assertTrue(all(
            leader.final_reserved_until_us[token]
                == replacement.final_reserved_until_us[token]
            for token in replacement.decode_cohort.shared_lease_tokens
        ))

    def test_unequal_requests_share_and_extend_one_cohort_lease(self):
        scheduler = self._cohort_scheduler()
        short = scheduler.submit_automated_request(
            request("adaptive-short", input_tokens=8, output_tokens=30),
            self.manifest.model_id,
            self.snapshot,
            selection_mode="adaptive-decode",
        )
        long = scheduler.submit_automated_request(
            request(
                "adaptive-long",
                arrival_us=1_001,
                input_tokens=8,
                output_tokens=60,
            ),
            self.manifest.model_id,
            self.snapshot,
            selection_mode="adaptive-decode",
        )
        epoch_ns = time.monotonic_ns() - short.decision.start_us * 1_000
        short = scheduler.wait_runtime_request(
            short.request.request_id, epoch_ns
        )
        long = scheduler.wait_runtime_request(
            long.request.request_id, epoch_ns
        )

        self.assertIsNotNone(short.decode_cohort)
        self.assertEqual(short.decode_cohort, long.decode_cohort)
        self.assertEqual(short.decode_cohort.active_batch, 2)
        self.assertEqual(
            {row.token for row in short.decision.leases},
            {row.token for row in long.decision.leases},
        )
        self.assertTrue(all(
            short.final_reserved_until_us[token]
                == long.final_reserved_until_us[token]
            for token in short.final_reserved_until_us
        ))
        scheduler.cancel_runtime_request(
            short.request.request_id,
            short.decision.start_us,
            "synthetic_short_cleanup",
        )
        survivor = scheduler.runtime_ticket(long.request.request_id)
        self.assertTrue(all(
            row.owner_id == long.request.request_id
            for row in survivor.decision.leases
        ))
        cohort_state = scheduler.runtime_decode_cohort_snapshot()[
            "cohorts"
        ][long.decode_cohort.cohort_id]
        self.assertEqual(
            cohort_state["active_member_request_ids"],
            [long.request.request_id],
        )
        self.assertEqual(
            cohort_state["lease_owner_id"], long.request.request_id
        )
        scheduler.cancel_runtime_request(
            long.request.request_id,
            long.decision.start_us,
            "synthetic_long_cleanup",
        )

    def test_extended_cohort_lease_hands_back_to_shorter_survivor(self):
        scheduler = self._cohort_scheduler()
        short = scheduler.submit_automated_request(
            request("adaptive-survivor", input_tokens=8, output_tokens=30),
            self.manifest.model_id,
            self.snapshot,
            selection_mode="adaptive-decode",
        )
        long = scheduler.submit_automated_request(
            request(
                "adaptive-retiring",
                arrival_us=1_001,
                input_tokens=8,
                output_tokens=60,
            ),
            self.manifest.model_id,
            self.snapshot,
            selection_mode="adaptive-decode",
        )
        epoch_ns = time.monotonic_ns() - short.decision.start_us * 1_000
        short = scheduler.wait_runtime_request(
            short.request.request_id, epoch_ns
        )
        long = scheduler.wait_runtime_request(
            long.request.request_id, epoch_ns
        )
        original_coverage = {
            row.token: row.reserved_until_us
            for row in short.decision.leases
        }

        recovery = scheduler.fail_automated_request(
            long.request.request_id,
            failed_at_us=long.decision.start_us,
            reason="synthetic_long_follower_failure",
            snapshot=self.snapshot,
            physical_failure=RuntimeExecutionFailure(
                phase="execution",
                retry_safe=True,
                execution_started=False,
                failed_resource_ids=(),
            ),
        )

        self.assertIsNotNone(recovery.fallback)
        survivor = scheduler.runtime_ticket(short.request.request_id)
        self.assertTrue(all(
            row.owner_id == short.request.request_id
            for row in survivor.decision.leases
        ))
        self.assertTrue(any(
            row.reserved_until_us > original_coverage[row.token]
            for row in survivor.decision.leases
        ))
        self.assertEqual(
            survivor.final_reserved_until_us,
            {
                row.token: row.reserved_until_us
                for row in survivor.decision.leases
            },
        )
        scheduler.cancel_runtime_request(
            short.request.request_id,
            short.decision.start_us,
            "synthetic_survivor_cleanup",
        )
        fallback = recovery.fallback
        scheduler.cancel_runtime_request(
            fallback.request.request_id,
            fallback.decision.start_us,
            "synthetic_fallback_cleanup",
        )

    def test_cold_phone_cohort_shares_transition_identity(self):
        scheduler = self._cohort_scheduler()
        snapshot = replace(
            self.snapshot,
            residency=tuple(
                row for row in self.snapshot.residency
                if row.device_id != "helper-c"
            ),
        )
        first = scheduler.submit_automated_request(
            request("cold-cohort-a", input_tokens=8, output_tokens=30),
            self.manifest.model_id,
            snapshot,
            selection_mode="adaptive-decode",
        )
        second = scheduler.submit_automated_request(
            request(
                "cold-cohort-b",
                arrival_us=1_001,
                input_tokens=8,
                output_tokens=60,
            ),
            self.manifest.model_id,
            snapshot,
            selection_mode="adaptive-decode",
        )
        epoch_ns = time.monotonic_ns() - first.decision.start_us * 1_000
        first = scheduler.wait_runtime_request(
            first.request.request_id, epoch_ns
        )
        second = scheduler.wait_runtime_request(
            second.request.request_id, epoch_ns
        )
        commands = tuple(map(
            interpret_runtime_ticket, (first, second)
        ))

        self.assertTrue(all(row.transitions for row in commands))
        self.assertEqual(
            commands[0].transitions[0].transition,
            commands[1].transitions[0].transition,
        )
        self.assertEqual(
            commands[0].transitions[0].decode_cohort,
            commands[1].transitions[0].decode_cohort,
        )
        scheduler.cancel_runtime_request(
            first.request.request_id,
            first.decision.start_us,
            "synthetic_first_cleanup",
        )
        scheduler.cancel_runtime_request(
            second.request.request_id,
            second.decision.start_us,
            "synthetic_second_cleanup",
        )

    def test_cold_cohort_releases_preparation_before_survivor_handoff(self):
        scheduler = self._cohort_scheduler()
        snapshot = replace(
            self.snapshot,
            residency=tuple(
                row for row in self.snapshot.residency
                if row.device_id != "helper-c"
            ),
        )
        tickets = tuple(
            scheduler.submit_automated_request(
                request(
                    f"cold-phase-{index}", arrival_us=1_000 + index,
                    input_tokens=8, output_tokens=30 * (index + 1),
                ),
                self.manifest.model_id, snapshot,
                selection_mode="adaptive-decode",
            )
            for index in range(2)
        )
        epoch_ns = time.monotonic_ns() - tickets[0].decision.start_us * 1_000
        tickets = tuple(
            scheduler.wait_runtime_request(ticket.request.request_id, epoch_ns)
            for ticket in tickets
        )
        queue = scheduler._runtime_controller.queue
        for ticket in tickets:
            updated = scheduler.record_automated_transition_receipts(
                ticket.request.request_id,
                FakeAutomatedPhysicalAdapter._transition_receipts(ticket),
            )
            self.assertEqual(updated.transition_status, "COMPLETED")
            self.assertEqual(len(queue.snapshot()["active"]), 2)
            for peer in tickets:
                identity = peer.request.request_id
                self.assertFalse(queue._active_conflict(
                    queue._entries[identity], excluding_request_id=identity,
                ))
        scheduler.cancel_runtime_request(
            tickets[0].request.request_id,
            tickets[0].decision.start_us + 2,
            "synthetic_first_phase_cleanup",
        )
        survivor = scheduler.runtime_ticket(tickets[1].request.request_id)
        self.assertTrue(all(
            row.owner_id == survivor.request.request_id
            for row in survivor.live_leases
        ))
        self.assertEqual(
            queue._entries[survivor.request.request_id].decision.leases,
            survivor.live_leases,
        )
        scheduler.cancel_runtime_request(
            survivor.request.request_id, survivor.decision.start_us + 2,
            "synthetic_survivor_phase_cleanup",
        )
        self.assertEqual(queue.snapshot()["active"], {})

    def test_cohort_energy_is_ingested_once_for_total_work(self):
        scheduler = self._cohort_scheduler()
        first = scheduler.submit_automated_request(
            request("cohort-energy-a", input_tokens=8, output_tokens=30),
            self.manifest.model_id,
            self.snapshot,
            selection_mode="adaptive-decode",
        )
        second = scheduler.submit_automated_request(
            request(
                "cohort-energy-b",
                arrival_us=1_001,
                input_tokens=12,
                output_tokens=50,
            ),
            self.manifest.model_id,
            self.snapshot,
            selection_mode="adaptive-decode",
        )
        epoch_ns = time.monotonic_ns() - first.decision.start_us * 1_000
        first = scheduler.wait_runtime_request(
            first.request.request_id, epoch_ns
        )
        second = scheduler.wait_runtime_request(
            second.request.request_id, epoch_ns
        )
        cohort_id = first.decode_cohort.cohort_id
        domain_ids = tuple(
            self.profile.placement_profile.idle_charge_domains
        )
        scheduler.record_decode_cohort_measurement(
            cohort_id,
            started_at_us=min(
                first.decision.start_us, second.decision.start_us
            ),
            finished_at_us=max(
                first.decision.finish_us, second.decision.finish_us
            ),
            fleet_energy_uj_by_domain={
                domain_id: 100 for domain_id in domain_ids
            },
            transfer_energy_uj_by_link={
                resource_id.removeprefix("link:"): 1
                for resource_id in first.execution_plan.resource_ids
                if resource_id.startswith("link:")
            },
            measurement_evidence_ids=("synthetic-cohort-energy",),
            attribution_kind="isolated",
            energy_boundary_id=(
                self.profile.placement_profile.energy_boundary_id
            ),
            total_input_tokens=20,
            total_output_tokens=80,
        )

        def receipt(ticket, suffix):
            return RuntimeExecutionReceipt(
                ticket_id=ticket.ticket_id,
                request_id=ticket.request.request_id,
                artifact_sha256=ticket.model.artifact_sha256,
                operator_plan_sha256=ticket.execution_plan.plan_sha256,
                executor_id=ticket.binding.executor_id,
                endpoint=ticket.binding.endpoint,
                operator_plan_protocol=(
                    ticket.binding.operator_plan_protocol
                ),
                participant_executor_ids=tuple(
                    row.executor_id for row in ticket.binding.participants
                ),
                started_us=ticket.decision.start_us,
                finished_us=ticket.decision.finish_us,
                output_sha256="sha256:" + suffix * 64,
                status="COMPLETED",
            )

        before = scheduler.automated_observation_state()[
            "complete_receipts"
        ]
        scheduler.complete_automated_request(
            first.request.request_id, receipt(first, "1")
        )
        self.assertEqual(
            scheduler.automated_observation_state()["complete_receipts"],
            before,
        )
        scheduler.complete_automated_request(
            second.request.request_id, receipt(second, "2")
        )
        self.assertEqual(
            scheduler.automated_observation_state()["complete_receipts"],
            before + 1,
        )
        snapshot = scheduler.runtime_decode_cohort_snapshot()
        self.assertEqual(snapshot["estimator_ingested"], [cohort_id])
        exported = scheduler.automated_observation_snapshot()
        rows = [
            row
            for bucket in exported["rows"]
            for row in bucket["observations"]
        ]
        self.assertEqual(len(rows), before + 1)
        self.assertEqual(rows[-1]["input_tokens"], 20)
        self.assertEqual(rows[-1]["output_tokens"], 80)

    def test_adaptive_ticket_leases_envelope_and_recovers_to_parent(self):
        ticket = self._active_ticket("adaptive-recovery")
        self.assertEqual(ticket.binding.executor_id, self.phone.executor_id)
        self.assertEqual(
            ticket.decision.reason, "INTRA_REQUEST_CALIBRATION_ENVELOPE"
        )
        self.assertTrue({
            "compute:host-a",
            "compute:accelerator-b",
            "compute:helper-c",
            "link:usb-out",
            "link:usb-in",
        }.issubset({row.resource_id for row in ticket.decision.leases}))
        selected_estimate = next(
            row for row in ticket.cost_estimates.estimates
            if row.route_id == ticket.decision.route_id
        )
        contract = selected_estimate.details["adaptive_decode_contract"]
        self.assertGreaterEqual(
            len(contract["adaptive_decode_probe_contracts"]), 2
        )
        coverage = validate_decision_candidate_coverage(
            self.profile,
            self.scheduler.runtime_decision_log()["records"],
            {ticket.request.request_id: self.manifest},
        )
        self.assertEqual(len(coverage), 1)

        directive = self.scheduler.start_adaptive_decode(
            ticket.request.request_id,
            slot_id=3,
            first_token_index=1,
            at_us=2_000,
            config=AdaptiveDecodeConfig(
                minimum_remaining_tokens=4,
                minimum_window_tokens=2,
                maximum_window_tokens=2,
                maximum_probe_tokens=20,
                maximum_probe_candidates=2,
                measurement_resolution_us=1,
                transition_cost_us=1,
                transition_energy_uj=1,
                warmup_windows_per_policy=0,
            ),
        )
        boundary_directive = self.scheduler.adaptive_decode_boundary(
            ticket.request.request_id,
            slot_id=3,
            token_index=directive.target_token_index,
            at_us=4_000,
        )
        boundary = boundary_directive.boundary
        self.scheduler.record_adaptive_decode_window(
            ticket.request.request_id,
            boundary,
            AdaptiveDecodeRawWindowObservation(
                fleet_energy_uj_by_domain={"fleet": 200},
                phone_compute_us=0,
                usb_transfer_us=0,
                rpc_us=0,
                exposed_tail_us=0,
                output_valid=True,
                evidence_ids=("synthetic-adaptive-window",),
                energy_boundary_id="synthetic-fleet",
                energy_attribution_kind="isolated",
            ),
        )
        recovery = self.scheduler.fail_automated_request(
            ticket.request.request_id,
            failed_at_us=4_100,
            reason="synthetic_phone_probe_failure",
            snapshot=self.snapshot,
            physical_failure=RuntimeExecutionFailure(
                phase="phone_probe",
                retry_safe=True,
                execution_started=False,
                failed_resource_ids=("compute:helper-c",),
            ),
        )

        self.assertIsNotNone(recovery.fallback)
        self.assertEqual(
            recovery.fallback.binding.executor_id, self.desktop.executor_id
        )
        self.assertEqual(
            recovery.fallback.decision.reason, "PAIRED_DESKTOP_RECOVERY"
        )
        self.assertEqual(
            recovery.fallback.selection_mode, "desktop-baseline"
        )
        self.assertEqual(
            recovery.fallback.execution_plan.desktop_placement_sha256,
            ticket.execution_plan.desktop_placement_sha256,
        )
        grouped = self.scheduler.adaptive_decode_grouped_observation(
            ticket.request.request_id
        )
        self.assertEqual(grouped.terminal_status, "FAILED")
        self.assertEqual(
            grouped.terminal_reason, "synthetic_phone_probe_failure"
        )
        self.assertEqual(
            self.scheduler.adaptive_decode_observation_state()[
                "grouped_observations"
            ],
            1,
        )

    def test_adaptive_policy_rejects_stateful_operator_migration(self):
        candidate_set = self.scheduler.generate_automated_candidates(
            request("adaptive-kv", input_tokens=8, output_tokens=30),
            self.manifest.model_id,
            self.snapshot,
        )
        by_id = {
            row.candidate_id: row for row in candidate_set.candidates
        }
        candidate = next(
            row for row in candidate_set.candidates
            if row.paired_baseline_route_id == candidate_set.baseline_route_id
            and row.assisted_operator_kind == "ffn"
        )
        parent = by_id[candidate.paired_baseline_route_id]
        changed = False
        operators = []
        for row in candidate.plan.operators:
            if not changed and row.operator_kind == "kv_cache":
                row = replace(row, device_ids=("helper-c",))
                changed = True
            operators.append(row)
        self.assertTrue(changed)
        bad_plan = replace(candidate.plan, operators=tuple(operators))
        bad_candidate = replace(
            candidate,
            plan=bad_plan,
            binding=replace(
                candidate.binding,
                operator_plan_sha256=bad_plan.plan_sha256,
            ),
        )
        with self.assertRaisesRegex(
            AdaptiveDecodeError,
            "only stateless FFN",
        ):
            _policy(
                bad_candidate,
                parent,
                self.manifest,
                30,
            )

    def test_stale_dormant_helper_keeps_desktop_adaptive_session(self):
        request_id = "stale-dormant-helper"
        ticket = SimpleNamespace(
            request=SimpleNamespace(request_id=request_id),
            execution_plan=SimpleNamespace(
                desktop_placement_sha256="desktop-placement",
                helper_envelope=None,
            ),
            binding=SimpleNamespace(executor_id="desktop-executor"),
            cost_estimates=SimpleNamespace(
                estimates=(SimpleNamespace(route_id="desktop-route"),)
            ),
            decision=SimpleNamespace(route_id="desktop-route"),
            model=SimpleNamespace(artifact_sha256="model-artifact"),
        )
        baseline = mock.sentinel.baseline_policy
        candidate = mock.sentinel.stale_helper_policy
        self.scheduler._request_helper_opportunities[request_id] = (
            SimpleNamespace(
                desktop_parent_placement_sha256="desktop-placement",
                desktop_parent_route_id="desktop-route",
                helper_binding=SimpleNamespace(
                    artifact_sha256="model-artifact"
                ),
                helper_operator_plan=SimpleNamespace(
                    baseline_executor_id="desktop-executor"
                ),
                route_id="stale-helper-route",
            ),
        )

        with mock.patch.object(
            self.scheduler,
            "_materialize_late_request_helper",
            return_value=None,
        ), mock.patch.object(
            self.scheduler,
            "_has_dormant_phone_ffn_runtime",
            return_value=True,
        ), mock.patch.object(
            self.scheduler,
            "_ticket_adaptive_policy_records",
            return_value=(baseline, (candidate,)),
        ), mock.patch.object(
            self.scheduler,
            "_helper_opportunity_policies",
            return_value=None,
        ):
            actual = self.scheduler._adaptive_policies_from_ticket(ticket)

        self.assertEqual(actual, (baseline, (), None))

    def test_stale_late_helper_context_is_retried_at_next_boundary(self):
        request_id = "stale-late-helper-context"
        cached = SimpleNamespace(helper=mock.sentinel.stale_helper)
        self.scheduler._late_request_helper_contexts[request_id] = cached
        ticket = SimpleNamespace(
            request=SimpleNamespace(request_id=request_id),
            execution_plan=None,
        )

        with mock.patch.object(
            self.scheduler,
            "_validated_cached_late_request_helper",
            return_value=None,
        ):
            actual = self.scheduler._materialize_late_request_helper(ticket)

        self.assertIsNone(actual)
        self.assertNotIn(
            request_id, self.scheduler._late_request_helper_contexts
        )

    def test_ready_late_helper_uses_base_ticket_transition_cost(self):
        request_id = "ready-late-helper"
        desktop_estimate = SimpleNamespace(
            route_id="desktop-route",
            details={},
        )
        ticket = SimpleNamespace(
            request=SimpleNamespace(
                request_id=request_id,
                input_tokens=8,
                output_tokens=30,
                deadline_us=1_000_000,
            ),
            ticket_id=request_id + ":attempt:0",
            execution_plan=SimpleNamespace(
                execution_contract=SimpleNamespace(
                    execution_mode="desktop",
                ),
                helper_envelope=None,
            ),
            decision=SimpleNamespace(route_id="desktop-route"),
            cost_estimates=SimpleNamespace(estimates=(desktop_estimate,)),
            model=SimpleNamespace(artifact_sha256="model-artifact"),
            planning_profile_sha256="planning-profile",
        )
        helper = SimpleNamespace(
            route_id="late-helper-route",
            phone_layout_generation=2,
            phone_layout_geometry_sha256="layout-geometry",
            helper_plan=SimpleNamespace(
                execution_contract=SimpleNamespace(phone_device_id=None),
            ),
        )
        baseline = mock.sentinel.baseline_policy
        candidate = mock.sentinel.helper_policy
        directive = mock.sentinel.directive
        self.scheduler._late_request_helper_contexts[request_id] = (
            SimpleNamespace(
                component=SimpleNamespace(identity_sha256="component"),
                evidence_state="LEARNING",
            )
        )

        with mock.patch.object(
            self.scheduler,
            "runtime_execution_ticket",
            return_value=ticket,
        ), mock.patch.object(
            self.scheduler,
            "_has_dormant_phone_ffn_runtime",
            return_value=True,
        ), mock.patch.object(
            self.scheduler,
            "_request_helper_opportunity_for_ticket",
            return_value=None,
        ), mock.patch.object(
            self.scheduler,
            "_adaptive_policies_from_ticket",
            return_value=(baseline, (candidate,), candidate),
        ), mock.patch.object(
            self.scheduler,
            "_request_helper_envelope",
            return_value=helper,
        ), mock.patch.object(
            self.scheduler,
            "_attach_ready_request_helper",
            return_value=True,
        ), mock.patch.object(
            self.scheduler._adaptive_decode,
            "start",
            return_value=directive,
        ) as start, mock.patch.object(
            self.scheduler,
            "_track_adaptive_directive",
            return_value=directive,
        ):
            actual = self.scheduler.start_adaptive_decode(
                request_id,
                slot_id=2,
                first_token_index=1,
                at_us=2_000,
            )

        self.assertIs(actual, directive)
        self.assertEqual(
            start.call_args.kwargs["config"].transition_cost_us,
            self.scheduler._adaptive_decode_config.transition_cost_us,
        )
        self.assertEqual(
            start.call_args.kwargs["config"].transition_energy_uj,
            self.scheduler._adaptive_decode_config.transition_energy_uj,
        )
        self.assertEqual(
            start.call_args.kwargs["helper_evidence_state"], "LEARNING"
        )

    def test_ready_helper_ignores_ticket_cold_residency_estimate(self):
        desktop = SimpleNamespace(route_id="desktop-route", details={})
        old_helper = SimpleNamespace(
            route_id="helper-route",
            details={"residency_break_even": {
                "incremental_transition_latency_us": 54_949_624,
                "incremental_transition_energy_uj": 9_477_649_639,
            }},
        )
        ticket = SimpleNamespace(
            cost_estimates=SimpleNamespace(estimates=(desktop, old_helper)),
            decision=SimpleNamespace(route_id=desktop.route_id),
        )
        selected, opportunity = self.scheduler._adaptive_start_selected_estimate(
            ticket,
            old_helper.route_id,
            mock.sentinel.ready_helper,
            mock.sentinel.validated_late_context,
            mock.sentinel.dormant_opportunity,
        )
        self.assertIs(selected, desktop)
        self.assertIsNone(opportunity)
        self.assertEqual(
            self.scheduler._adaptive_start_transition_costs(selected),
            (2_000, 20_000),
        )
        self.assertEqual(
            old_helper.details["residency_break_even"]
                ["incremental_transition_latency_us"],
            54_949_624,
        )
        selected, _ = self.scheduler._adaptive_start_selected_estimate(
            ticket, old_helper.route_id, mock.sentinel.helper, None, None
        )
        self.assertIs(selected, old_helper)

    def test_unready_bound_helper_defers_nonzero_control(self) -> None:
        request_id = "unready-bound-helper"
        control = SimpleNamespace(
            policy=SimpleNamespace(split_fraction_ppm=500_000),
            slot_id=2,
        )
        directive = SimpleNamespace(
            control=control,
            target_token_index=9,
            reason="POLICY_CHANGE_REQUIRED",
        )
        deferred = mock.sentinel.deferred_directive
        ticket = SimpleNamespace(execution_plan=SimpleNamespace())
        binding = {
            "helper_envelope": {
                "phone_layout_generation": 3,
                "phone_layout_geometry_sha256": "layout-geometry",
            },
            "resident_component_identity_sha256": "component",
        }

        with mock.patch.object(
            self.scheduler,
            "runtime_execution_ticket",
            return_value=ticket,
        ), mock.patch.object(
            self.scheduler,
            "_request_helper_envelope",
            return_value=None,
        ), mock.patch.object(
            self.scheduler._model_placement_controller,
            "request_binding",
            return_value=binding,
        ), mock.patch.object(
            self.scheduler._model_placement_controller,
            "record_request_helper_event",
        ) as record_event, mock.patch.object(
            self.scheduler._model_placement_controller,
            "update_request_fraction",
        ) as update_fraction, mock.patch.object(
            self.scheduler._adaptive_decode,
            "helper_unavailable",
        ) as helper_unavailable, mock.patch.object(
            self.scheduler._adaptive_decode,
            "defer_control",
            return_value=deferred,
        ) as defer_control, mock.patch.object(self.scheduler, "_record_assistance_decision"):
            actual = self.scheduler._track_adaptive_directive(
                request_id,
                directive,
                at_us=5_000,
                token_index=9,
            )

        self.assertIs(actual, deferred)
        update_fraction.assert_not_called()
        helper_unavailable.assert_called_once_with(request_id)
        defer_control.assert_called_once_with(
            request_id,
            control,
            "PHONE_HELPER_NOT_READY",
            at_us=5_000,
        )
        self.assertEqual(
            record_event.call_args.args[1], "EXECUTION_DEFERRED"
        )

    def test_incomplete_verification_is_exported_without_request_failure(self) -> None:
        directive = AdaptiveDecodeDirective(
            state="EXPLOITING", reason="VERIFICATION_INCOMPLETE", target_token_index=12)
        snapshot = {
            "helper_layout_generation": 3,
            "helper_layout_geometry_sha256": "sha256:" + "a" * 64,
            "verification": {"outcome": "INCOMPLETE", "reason": "COMPLETE_PAIR_BUDGET",
                             "attempts": 0, "budget": None, "policy_hash": None},
        }
        with mock.patch.object(self.scheduler, "runtime_execution_ticket",
                               return_value=SimpleNamespace(execution_plan=None)), \
                mock.patch.object(self.scheduler._adaptive_decode, "snapshot", return_value=snapshot), \
                mock.patch.object(self.scheduler, "_record_assistance_decision"), \
                mock.patch.object(self.scheduler._model_placement_controller, "record_request_helper_event") as record:
            actual = self.scheduler._track_adaptive_directive(
                "request-a", directive, at_us=5_000, token_index=8)
        self.assertIs(actual, directive)
        self.assertEqual(record.call_args.args[:3], ("request-a", "VERIFICATION_INCOMPLETE", 5_000))
        self.assertEqual(record.call_args.args[3]["reason"], "COMPLETE_PAIR_BUDGET")
        self.assertEqual(record.call_args.args[3]["evidence_state"], "DIAGNOSTIC")


    def _record_window_at(self, ticket, directive, at_us, *, fleet_energy=200):
        boundary = self.scheduler.adaptive_decode_boundary(
            ticket.request.request_id,
            slot_id=3,
            token_index=directive.target_token_index,
            at_us=at_us,
        ).boundary
        return boundary, self.scheduler.record_adaptive_decode_window(
            ticket.request.request_id,
            boundary,
            AdaptiveDecodeRawWindowObservation(
                fleet_energy_uj_by_domain={"fleet": fleet_energy},
                phone_compute_us=0,
                usb_transfer_us=0,
                rpc_us=0,
                exposed_tail_us=0,
                output_valid=True,
                evidence_ids=("synthetic-adaptive-window",),
                energy_boundary_id="synthetic-fleet",
                energy_attribution_kind="isolated",
            ),
        )

    def test_window_measurement_context_names_other_acquired_server_work(self):
        ticket = self._active_ticket("adaptive-external-context")
        request_id = ticket.request.request_id
        directive = self.scheduler.start_adaptive_decode(
            request_id,
            slot_id=3,
            first_token_index=1,
            at_us=2_000,
            config=AdaptiveDecodeConfig(
                minimum_remaining_tokens=4,
                minimum_window_tokens=2,
                maximum_window_tokens=2,
                maximum_probe_tokens=20,
                maximum_probe_candidates=2,
                measurement_resolution_us=1,
                transition_cost_us=1,
                transition_energy_uj=1,
                warmup_windows_per_policy=0,
            ),
        )
        controller = self.scheduler._runtime_controller
        real_overlaps = controller.acquired_lease_overlaps
        calls = []

        def overlaps(ticket_id, resource_ids, start_us, end_us):
            calls.append((ticket_id, tuple(resource_ids), start_us, end_us))
            result = real_overlaps(ticket_id, resource_ids, start_us, end_us)
            return result if result is None else (*result, *peers)

        # Nothing else runs on the desktop during the first window.
        peers = ()
        with mock.patch.object(controller, "acquired_lease_overlaps", overlaps):
            boundary, directive = self._record_window_at(ticket, directive, 4_000)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], ticket.ticket_id)
        self.assertEqual(calls[0][2:], (boundary.started_at_us, boundary.finished_at_us))
        self.assertIn("compute:host-a", calls[0][1])
        self.assertIn("compute:accelerator-b", calls[0][1])
        self.assertNotIn("compute:helper-c", calls[0][1])
        quiet = self.scheduler._adaptive_decode.snapshot(request_id)["external_activity_sha256"]
        self.assertIsNotNone(quiet)
        self.assertEqual(quiet, canonical_sha256({"external_desktop_ticket_ids": []}))
        if directive.control is not None:
            directive = self.scheduler.acknowledge_adaptive_decode_control(
                request_id,
                AdaptiveDecodePolicyAck(
                    request_id=request_id, slot_id=3,
                    plan_generation=directive.control.plan_generation,
                    applied_token_index=directive.target_token_index
                        or self.scheduler._adaptive_decode.snapshot(request_id)["transition_start_token"],
                    applied_at_us=4_000, policy_hash=directive.control.policy.policy_hash,
                ),
            )
        # Another request acquires server leases while this request stays
        # at HTTP batch 1: the measurement context changes, execution does not.
        peers = ("cpu-peer:attempt:0",)
        events_before = len(self.scheduler.request_helper_events())
        with mock.patch.object(controller, "acquired_lease_overlaps", overlaps):
            boundary, directive = self._record_window_at(ticket, directive, 6_000)
        session = self.scheduler._adaptive_decode.checkpoint()[1][request_id]
        self.assertEqual(session.active_batch, 1)
        self.assertTrue(session.records[-1].external_activity_changed)
        self.assertFalse(session.records[-1].measurement_eligible)
        self.assertEqual(session.context_record_start, len(session.records))
        busy = self.scheduler._adaptive_decode.snapshot(request_id)["external_activity_sha256"]
        self.assertEqual(busy, canonical_sha256({"external_desktop_ticket_ids": ["cpu-peer:attempt:0"]}))
        changes = [
            row for row in self.scheduler.request_helper_events()[events_before:]
            if row["kind"] == "CONTEXT_CHANGED"
        ]
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["reason"], "EXTERNAL_DESKTOP_ACTIVITY_CHANGED")
        self.assertEqual(changes[0]["external_desktop_ticket_ids"], ["cpu-peer:attempt:0"])
        self.assertEqual(changes[0]["previous_external_activity_sha256"], quiet)
        self.assertEqual(changes[0]["external_activity_sha256"], busy)
        self.assertEqual(changes[0]["previous_active_batch"], 1)
        self.assertEqual(changes[0]["active_batch"], 1)
        self.assertFalse(changes[0]["measurement_eligible"])
        decisions = [
            row for row in self.scheduler.request_helper_events()[events_before:]
            if row["kind"] == "ASSISTANCE_DECISION"
        ]
        self.assertTrue(decisions)
        self.assertEqual(decisions[-1]["external_activity_sha256"], busy)
        # A window the ledger cannot cover keeps the current context.
        with mock.patch.object(controller, "acquired_lease_overlaps", lambda *args: None):
            if directive.control is not None:
                directive = self.scheduler.acknowledge_adaptive_decode_control(
                    request_id,
                    AdaptiveDecodePolicyAck(
                        request_id=request_id, slot_id=3,
                        plan_generation=directive.control.plan_generation,
                        applied_token_index=directive.target_token_index
                            or self.scheduler._adaptive_decode.snapshot(request_id)["transition_start_token"],
                        applied_at_us=6_000, policy_hash=directive.control.policy.policy_hash,
                    ),
                )
            start = session.context_record_start
            boundary, directive = self._record_window_at(ticket, directive, 8_000)
        session = self.scheduler._adaptive_decode.checkpoint()[1][request_id]
        self.assertEqual(session.context_record_start, start)
        self.assertIsNone(session.records[-1].external_activity_sha256)
        self.assertFalse(session.records[-1].external_activity_changed)
        self.assertEqual(
            self.scheduler._adaptive_decode.snapshot(request_id)["external_activity_sha256"], busy)

if __name__ == "__main__":
    unittest.main()
