#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
import threading
import time
import unittest
from unittest.mock import patch

from research_dev.scheduler import (
    DeviceMemoryCapacity,
    ModelManifest,
    ModelOperatorManifest,
    ModelTensorManifest,
    RuntimeExecutionContract,
    RuntimeExecutionReceipt,
    RuntimeCompositeExecutorCapability,
    RuntimeCompositeOperatorPlacement,
    RuntimeDesktopControlProfile,
    RuntimeExecutorCapability,
    RuntimeMemoryDemand,
    RuntimePhonePowerProfile,
    RuntimePhoneSessionCapability,
    RuntimePhoneShard,
    RuntimePlacementSnapshot,
    RuntimeResourceError,
    RuntimeResidencyEviction,
    RuntimeTransitionPlan,
    UnifiedScheduler,
)
from research_dev.scheduler._internal.placement import MemoryPoolProfile
from research_dev.scheduler._internal.policy import (
    LeaseDemand,
    ResourceProfile,
    ResourceTimeline,
)
from research_dev.scheduler._internal.phone_shards import (
    PhoneFfnResidencyDemand,
    PhoneFfnResidencyLayout,
    PhoneFfnSessionIdentity,
    PhoneFfnShardPlacement,
    PhoneFfnShardStorageMetadata,
    PhoneShardPlacementError,
    _generate_disjoint_ffn_shard_sets,
    generate_disjoint_ffn_shard_sets,
    generate_mixed_ffn_residency_layouts,
    select_mixed_ffn_residency_layout,
)
from research_dev.scheduler._internal.model_placement_controller import (
    ModelPlacementController,
    ModelPhoneResidencyLayout,
)
from research_dev.scheduler._internal.runtime_capabilities import (
    ModelResidencyObservation,
    PhoneSessionResidencyObservation,
    RuntimeCapabilityError,
)
from research_dev.scheduler._internal.route_generation import (
    AutomatedRouteCompiler,
    _Pattern,
    _static_executor_identity,
)
from research_dev.scheduler._internal.runtime_plan import (
    PhoneSessionReplacementAuthorization,
    RuntimeHelperExecutionEnvelope,
    RuntimePlanError,
)
from research_dev.scheduler._internal.runtime_resources import (
    RuntimeMemoryLedger,
)
from research_dev.scheduler._internal.runtime_search import (
    RoughPlacementVisit,
)
from research_dev.scheduler._internal.runtime_residency_cohorts import (
    RuntimeResidencyComponentIdentity,
    RuntimeResidencyCohortTracker,
    runtime_residency_component_identity_from_parts,
)
from research_dev.scheduler._internal.types import canonical_sha256
from research_dev.scheduler.adapters import (
    CanonicalPhysicalAdapter,
    DirectPhoneFfnSession,
    PhoneSessionDiscoveryConfiguration,
    PhysicalAdapterError,
    PhysicalExecutionCommand,
    PhysicalParticipantCommand,
    PhysicalTransitionCommand,
    RawEnergyMeasurement,
    RawExecutionObservation,
    RawTransitionObservation,
    interpret_runtime_ticket,
    model_residency_observations,
    parse_phone_session_probe,
    parse_direct_phone_ffn_terminal,
    transition_receipt_from_observation,
    validate_physical_execution_command,
)
from research_dev.scheduler.adapters.llama_server import (
    LlamaServerPhoneSessionProof,
)
from research_dev.scheduler.adapters.ticket import (
    bind_ready_helper_to_physical_command,
)

try:
    from .test_automated_runtime import (
        catalog,
        catalog_with_gpu_desktop_control,
        executor_state,
        request,
        runtime_snapshot,
    )
except ImportError:
    from test_automated_runtime import (
        catalog,
        catalog_with_gpu_desktop_control,
        executor_state,
        request,
        runtime_snapshot,
    )


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64


class _MeasuredShardBackend:
    def __init__(self, fleet_energy_uj: int) -> None:
        self.fleet_energy_uj = fleet_energy_uj
        self.execution_commands = []
        self.transition_commands = []

    def apply_transition(self, command, _payload, control_check):
        self.transition_commands.append(command)
        control_check()
        return RawTransitionObservation(
            started_us=0,
            finished_us=1,
            status="COMPLETED",
            evicted_artifact_sha256s=tuple(sorted({
                row.artifact_sha256
                for row in command.transition.evictions
            })),
        )

    def execute(self, command, _payload, control_check):
        self.execution_commands.append(command)
        validate_physical_execution_command(command)
        control_check()
        domains = (
            "energy:accelerator-b",
            "energy:helper-c",
            "energy:host-a",
        )
        base = self.fleet_energy_uj // len(domains)
        energy = {domain_id: base for domain_id in domains}
        energy[domains[-1]] += self.fleet_energy_uj - sum(
            energy.values()
        )
        transfers = {
            resource_id.removeprefix("link:"): 1
            for resource_id in command.operator_plan["resource_ids"]
            if resource_id.startswith("link:")
        }
        return RawExecutionObservation(
            started_us=command.planned_start_us,
            finished_us=command.planned_finish_us,
            output_sha256=SHA_B,
            payload={"tokens": (1, 2, 3)},
            energy=RawEnergyMeasurement(
                energy_boundary_id="synthetic-whole-fleet",
                fleet_energy_uj_by_domain=energy,
                transfer_energy_uj_by_link=transfers,
                measurement_evidence_ids=(
                    "synthetic-multi-session-isolated",
                ),
                attribution_kind="isolated",
            ),
        )


def manifest(layer_count: int = 8) -> ModelManifest:
    tensors = tuple(
        ModelTensorManifest(
            tensor_id=f"layer.{index}.ffn.weight",
            shape=(8, 8),
            quantization="F16",
            quantization_block_size=1,
            quantization_type_size=2,
            nbytes=128,
            role="ffn_weight",
            layer_index=index,
        )
        for index in range(layer_count)
    )
    operators = tuple(
        ModelOperatorManifest(
            operator_id=f"layer:{index}:ffn",
            layer_id=f"layer:{index}",
            kind="ffn",
            dependencies=(),
            tensor_ids=(tensors[index].tensor_id,),
        )
        for index in range(layer_count)
    )
    return ModelManifest(
        model_id="synthetic-session-model",
        artifact_sha256=SHA_A,
        artifact_bytes=sum(row.nbytes for row in tensors),
        architecture="synthetic",
        context_length=128,
        sliding_window=128,
        embedding_length=8,
        feed_forward_length=8,
        block_count=layer_count,
        head_count=2,
        head_count_kv=1,
        key_length=4,
        value_length=4,
        tensors=tensors,
        operators=operators,
    )


def session(index: int, *, ready: bool = True) -> RuntimePhoneSessionCapability:
    return RuntimePhoneSessionCapability(
        session_id=f"HTP{index}",
        device_id="phone-a",
        endpoint=f"session://phone-a/HTP{index}",
        worker_identity_sha256=SHA_B,
        memory_resource_id=f"phone-session-{index}",
        resident_memory_limit_bytes=256,
        shared_compute_resource_id="phone-htp",
        shared_transport_resource_ids=("phone-functionfs", "phone-usb"),
        supported_layer_mask=(1 << 8) - 1,
        maximum_columns=8,
        column_quantum=2,
        supported_data_types=("F16",),
        batch_plans=("coalesced-batch", "split-row"),
        ready=ready,
        residency_state="cold",
    )


def shard(index: int, layer_mask: int) -> RuntimePhoneShard:
    return RuntimePhoneShard(
        session_id=f"HTP{index}",
        endpoint=f"session://phone-a/HTP{index}",
        layer_mask=layer_mask,
        maximum_columns=8,
        resident_bytes=256,
        resident_geometry_sha256=(SHA_B if index == 0 else SHA_C),
        operator_plan_sha256=(SHA_C if index == 0 else SHA_B),
    )


class MultiSessionPhoneTests(unittest.TestCase):
    def test_arrival_preserves_in_progress_phone_layout(self) -> None:
        value = request("queued-during-phone-preparation")
        model = manifest()
        events = []
        preparing = SimpleNamespace(
            generation=7,
            layout=SimpleNamespace(geometry_sha256=SHA_C),
            transition_ticket_id="phone-layout-transition-7",
        )
        controller = SimpleNamespace(
            phone_preload_inflight=lambda: False,
            preparing_phone_layout=lambda: preparing,
            record_phone_layout_evaluation=(
                lambda _observed_at_us, event: events.append(event)
            ),
        )
        ticket = SimpleNamespace(
            dispatch_state="QUEUED",
            model=SimpleNamespace(artifact_sha256=model.artifact_sha256),
            request=value,
        )
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler._runtime_capabilities = object()
        scheduler._runtime_controller = SimpleNamespace(
            current_tickets=lambda: (ticket,)
        )
        scheduler._model_placement_controller = controller
        with patch.object(
            UnifiedScheduler,
            "_automated_compiler",
            return_value=object(),
        ), patch(
            "research_dev.scheduler._unified.phone_residency_ops.economics."
            "generate_mixed_ffn_residency_layouts",
            side_effect=AssertionError("portfolio was regenerated"),
        ):
            changed = scheduler._update_phone_residency_portfolio(
                value, model, 123, None
            )
        self.assertFalse(changed)
        self.assertEqual(len(events), 1)
        self.assertEqual(
            events[0]["reason"],
            "PHONE_RESIDENCY_TRANSITION_IN_PROGRESS",
        )
        self.assertEqual(events[0]["phone_layout_generation"], 7)
        self.assertEqual(
            events[0]["queue_work_by_artifact"],
            {model.artifact_sha256: value.output_tokens},
        )

    def test_partial_phone_replacement_reuses_exact_session_memory(
        self,
    ) -> None:
        artifact_a = "sha256:" + "a" * 64
        artifact_b = "sha256:" + "b" * 64

        def shard(
            session_id: str,
            artifact_sha256: str,
            resident_bytes: int,
        ) -> PhoneFfnShardPlacement:
            suffix = session_id + artifact_sha256[-1]
            return PhoneFfnShardPlacement(
                artifact_sha256=artifact_sha256,
                session_id=session_id,
                endpoint="session://phone/" + session_id,
                memory_resource_id="phone-memory:" + session_id,
                operator_ids=("blk." + session_id[-1] + ".ffn",),
                layer_mask=1 << int(session_id[-1]),
                maximum_columns=1024,
                resident_bytes=resident_bytes,
                resident_geometry_sha256=canonical_sha256({
                    "geometry": suffix,
                }),
                operator_plan_sha256=canonical_sha256({
                    "plan": suffix,
                }),
            )

        def layout(
            shards: tuple[PhoneFfnShardPlacement, ...],
            changed_session_ids: tuple[str, ...],
        ) -> PhoneFfnResidencyLayout:
            artifacts = {row.artifact_sha256 for row in shards}
            geometry = canonical_sha256(
                {
                    "artifact_sha256": next(iter(artifacts)),
                    "shards": [
                        {
                            "geometry_sha256": (
                                row.resident_geometry_sha256
                            ),
                            "session_id": row.session_id,
                        }
                        for row in shards
                    ],
                }
                if len(artifacts) == 1 else
                {
                    "shards": [
                        {
                            "artifact_sha256": row.artifact_sha256,
                            "geometry_sha256": (
                                row.resident_geometry_sha256
                            ),
                            "session_id": row.session_id,
                        }
                        for row in shards
                    ],
                }
            )
            benefit_by_artifact = {
                artifact: sum(
                    100 for row in shards
                    if row.artifact_sha256 == artifact
                )
                for artifact in artifacts
            }
            return PhoneFfnResidencyLayout(
                shards=shards,
                queued_work_by_artifact={
                    artifact: 10 for artifact in artifacts
                },
                queue_benefit_by_artifact=benefit_by_artifact,
                queue_benefit_by_session={
                    row.session_id: 100 for row in shards
                },
                queue_benefit=100 * len(shards),
                transition_cost=len(changed_session_ids),
                transition_cost_by_session={
                    session_id: 1
                    for session_id in changed_session_ids
                },
                objective=len(changed_session_ids) - 100 * len(shards),
                objective_kind="queue_energy_delta_uj",
                changed_session_ids=changed_session_ids,
                geometry_sha256=geometry,
            )

        source_layout = layout(
            tuple(shard("HTP" + str(index), artifact_a, 100)
                  for index in range(3)),
            ("HTP0", "HTP1", "HTP2"),
        ).with_session_generations({
            "HTP0": 1, "HTP1": 1, "HTP2": 1,
        })
        target_layout = layout(
            (
                source_layout.shards[0],
                source_layout.shards[1],
                shard("HTP2", artifact_b, 120),
            ),
            ("HTP2",),
        ).with_session_generations({
            "HTP0": 1, "HTP1": 1, "HTP2": 2,
        })
        source = ModelPhoneResidencyLayout(
            generation=1,
            state="READY",
            layout=source_layout,
            workspace_bytes=4,
            shared_compute_resource_id="phone-htp",
            shared_transport_resource_ids=("phone-usb",),
            proposed_at_us=0,
            minimum_resident_until_us=0,
            minimum_residency_interval_us=0,
            session_identities=source_layout.session_identities,
            ready_at_us=1,
            verification_sha256=canonical_sha256({"ready": 1}),
        )
        target = ModelPhoneResidencyLayout(
            generation=2,
            state="PROPOSED",
            layout=target_layout,
            workspace_bytes=4,
            shared_compute_resource_id="phone-htp",
            shared_transport_resource_ids=("phone-usb",),
            proposed_at_us=2,
            minimum_resident_until_us=2,
            minimum_residency_interval_us=0,
            session_identities=target_layout.session_identities,
        )
        self.assertEqual(
            UnifiedScheduler
                ._copy_on_write_preparation_yielding_resources(
                    source, target
                ),
            ("phone-htp", "phone-usb"),
        )
        runtime_shards = tuple(
            RuntimePhoneShard(
                session_id=row.session_id,
                endpoint=row.endpoint,
                layer_mask=row.layer_mask,
                maximum_columns=row.maximum_columns,
                resident_bytes=row.resident_bytes,
                resident_geometry_sha256=(
                    row.resident_geometry_sha256
                ),
                operator_plan_sha256=row.operator_plan_sha256,
                artifact_sha256=row.artifact_sha256,
                session_generation=(
                    target_layout.session_generation_by_id[row.session_id]
                ),
            )
            for row in target_layout.shards
        )
        transition = RuntimeTransitionPlan(
            transition_id="replace-phone-session",
            device_id="phone",
            source_state="cold",
            target_state="hot",
            latency_us=10,
            energy_uj=10,
            resource_ids=("phone-htp",),
            maturity="QUALIFIED",
            prepares_device_ids=("phone",),
            evictions=(RuntimeResidencyEviction(
                model_id="model-a",
                artifact_sha256=artifact_a,
                device_id="phone",
                resident_bytes=100,
                generation=1,
                executor_id="phone-executor",
                replacement_group="phone-htp",
                session_id="HTP2",
                resident_geometry_sha256=(
                    source_layout.shards[2].resident_geometry_sha256
                ),
                operator_plan_sha256=(
                    source_layout.shards[2].operator_plan_sha256
                ),
            ),),
            phone_shards=runtime_shards,
            changed_phone_session_ids=("HTP2",),
        )
        demands = (
            RuntimeMemoryDemand(
                demand_id="weights:phone",
                resource_id="phone-memory",
                kind="model_weights",
                required_bytes=320,
                resident_bytes=0,
                lifetime="resident",
                share_key=target_layout.geometry_sha256,
                replacement_group="phone-htp",
                replaceable_bytes=300,
                device_id="phone",
            ),
            *(
                RuntimeMemoryDemand(
                    demand_id="phone-session:" + row.session_id,
                    resource_id=row.memory_resource_id,
                    kind="session_residency_constraint",
                    required_bytes=row.resident_bytes,
                    resident_bytes=0,
                    lifetime="resident",
                    share_key=row.resident_geometry_sha256,
                    device_id="phone",
                )
                for row in target_layout.shards
            ),
            RuntimeMemoryDemand(
                demand_id="workspace:phone",
                resource_id="phone-memory",
                kind="workspace",
                required_bytes=4,
                resident_bytes=0,
                lifetime="resident",
                share_key=target_layout.geometry_sha256 + ":workspace",
                device_id="phone",
            ),
        )
        snapshot = SimpleNamespace(residency=(
            ModelResidencyObservation(
                model_id="model-a",
                artifact_sha256=artifact_a,
                device_id="phone",
                state="hot",
                resident_tensor_ids=("weight-a",),
                resident_bytes=300,
                generation=1,
                executor_id="phone-executor",
                resident_geometry_sha256=(
                    source_layout.geometry_sha256
                ),
            ),
        ), phone_session_residency=tuple(
            PhoneSessionResidencyObservation(
                session_id=row.session_id,
                device_id="phone",
                executor_id="phone-executor",
                endpoint=row.endpoint,
                artifact_sha256=row.artifact_sha256,
                resident_geometry_sha256=(
                    row.resident_geometry_sha256
                ),
                operator_plan_sha256=row.operator_plan_sha256,
                session_generation=1,
                resident_bytes=row.resident_bytes,
            )
            for row in source_layout.shards
        ))

        adjusted = UnifiedScheduler._partial_phone_replacement_memory_demands(
            demands,
            phone_device_id="phone",
            source=source,
            target=target,
            transitions=(transition,),
            snapshot=snapshot,
        )
        by_id = {row.demand_id: row for row in adjusted}

        self.assertEqual(by_id["weights:phone"].resident_bytes, 300)
        self.assertEqual(by_id["weights:phone"].additional_bytes, 20)
        self.assertIsNone(by_id["weights:phone"].replacement_group)
        self.assertEqual(
            by_id["phone-session:HTP0"].additional_bytes, 0
        )
        self.assertEqual(
            by_id["phone-session:HTP1"].additional_bytes, 0
        )
        self.assertEqual(
            by_id["phone-session:HTP2"].additional_bytes, 20
        )
        self.assertEqual(by_id["workspace:phone"].additional_bytes, 0)
        placement = RuntimePlacementSnapshot(
            snapshot_id="partial-phone-memory",
            captured_at_us=0,
            valid_until_us=100,
            capacities={
                "phone-memory": DeviceMemoryCapacity(
                    "phone-memory", 500, 200, 0
                ),
                **{
                    "phone-memory:HTP" + str(index): (
                        DeviceMemoryCapacity(
                            "phone-memory:HTP" + str(index),
                            120,
                            100,
                            0,
                        )
                    )
                    for index in range(3)
                },
            },
        )
        reservations = RuntimeMemoryLedger().reserve(
            "partial-phone-replacement",
            adjusted,
            placement,
            transitions=(transition,),
            residency=snapshot.residency,
            exclusive_resource_by_device={"phone": "phone-htp"},
        )
        self.assertEqual(
            {
                row.resource_id: row.reserved_bytes
                for row in reservations
            },
            {"phone-memory": 20, "phone-memory:HTP2": 20},
        )

        no_headroom = replace(
            placement,
            snapshot_id="partial-phone-memory-no-headroom",
            capacities={
                resource_id: replace(
                    capacity,
                    occupied_bytes=(
                        capacity.capacity_bytes - capacity.reserve_bytes
                    ),
                )
                for resource_id, capacity in placement.capacities.items()
            },
        )
        with self.assertRaisesRegex(
            RuntimeResourceError, "memory capacity is insufficient"
        ):
            RuntimeMemoryLedger().reserve(
                "partial-phone-replacement-no-headroom",
                adjusted,
                no_headroom,
                transitions=(transition,),
                residency=snapshot.residency,
                exclusive_resource_by_device={"phone": "phone-htp"},
            )

    def test_mixed_helpers_time_share_capacity_one_htp(self) -> None:
        timeline = ResourceTimeline({
            resource_id: ResourceProfile(
                resource_id=resource_id,
                kind="shared-phone",
                capacity=1,
                ready=True,
                identity=resource_id + "-identity",
            )
            for resource_id in ("phone-htp", "phone-usb")
        })

        def demands(owner: str) -> tuple[LeaseDemand, ...]:
            return tuple(
                LeaseDemand(
                    lease_id=owner + ":" + resource_id,
                    resource_id=resource_id,
                    slots=1,
                    start_offset_us=0,
                    duration_us=100,
                    duration_upper_us=100,
                )
                for resource_id in ("phone-htp", "phone-usb")
            )

        gemma = timeline.preview_leases(demands("gemma"), 0, 100, 100)
        timeline.commit_leases(gemma, "gemma-window")
        qwen = timeline.preview_leases(demands("qwen"), 0, 100, 100)
        self.assertEqual(gemma.start_us, 0)
        self.assertEqual(qwen.start_us, 100)
        qwen_leases = timeline.commit_leases(qwen, "qwen-window")
        self.assertEqual(
            {row.resource_id for row in qwen_leases},
            {"phone-htp", "phone-usb"},
        )

    def test_arrived_decode_work_excludes_completed_active_request(
        self,
    ) -> None:
        self.assertEqual(
            dict(UnifiedScheduler._arrived_decode_work_by_artifact(
                {SHA_A: 0, SHA_B: 7},
                {SHA_A: 0, SHA_B: 3, SHA_C: 5},
            )),
            {SHA_B: 10, SHA_C: 5},
        )

    def test_mixed_residency_generation_uses_disjoint_session_arenas(
        self,
    ) -> None:
        first = manifest(8)
        second = replace(
            manifest(8),
            model_id="synthetic-session-model-b",
            artifact_sha256=SHA_C,
        )
        sessions = tuple(session(index) for index in range(3))

        layouts = generate_mixed_ffn_residency_layouts(
            (
                PhoneFfnResidencyDemand(
                    first, 2, 8, "coalesced-batch"
                ),
                PhoneFfnResidencyDemand(
                    second, 2, 8, "coalesced-batch"
                ),
            ),
            sessions,
            phone_wide_limit_bytes=768,
        )

        assignments = {
            tuple(sorted(
                row.artifact_sha256 for row in layout.shards
            ))
            for layout in layouts
            if len(layout.shards) == 3
        }
        self.assertIn((SHA_A, SHA_A, SHA_C), assignments)
        self.assertIn((SHA_A, SHA_C, SHA_C), assignments)
        for layout in layouts:
            self.assertLessEqual(layout.resident_bytes, 768)
            self.assertEqual(
                len({row.session_id for row in layout.shards}),
                len(layout.shards),
            )

    def test_mixed_residency_reuses_static_shard_packings(self) -> None:
        model = replace(
            manifest(8),
            model_id="synthetic-cache-model",
            artifact_sha256="sha256:" + "d" * 64,
        )
        sessions = tuple(session(index) for index in range(3))
        benefits = {
            row.operator_id: 100 for row in model.operators
        }
        first_demand = PhoneFfnResidencyDemand(
            model,
            2,
            8,
            "coalesced-batch",
            benefit_by_operator=benefits,
        )
        second_demand = replace(first_demand, queued_work=5)
        target = (
            "research_dev.scheduler._internal.phone_shards."
            "_generate_disjoint_ffn_shard_sets"
        )

        with patch(
            target, wraps=_generate_disjoint_ffn_shard_sets
        ) as generate:
            first = generate_mixed_ffn_residency_layouts(
                (first_demand,),
                sessions,
                phone_wide_limit_bytes=768,
                transition_energy_uj_by_session={
                    "HTP0": 10, "HTP1": 10, "HTP2": 10,
                },
            )
            first_call_count = generate.call_count
            second = generate_mixed_ffn_residency_layouts(
                (second_demand,),
                sessions,
                phone_wide_limit_bytes=768,
                transition_energy_uj_by_session={
                    "HTP0": 20, "HTP1": 20, "HTP2": 20,
                },
            )

        self.assertGreater(first_call_count, 0)
        self.assertEqual(generate.call_count, first_call_count)
        first_by_geometry = {
            row.geometry_sha256: row for row in first
        }
        second_by_geometry = {
            row.geometry_sha256: row for row in second
        }
        self.assertEqual(set(first_by_geometry), set(second_by_geometry))
        for geometry, first_layout in first_by_geometry.items():
            second_layout = second_by_geometry[geometry]
            self.assertEqual(
                second_layout.queue_benefit * 2,
                first_layout.queue_benefit * 5,
            )
            self.assertEqual(
                second_layout.transition_cost,
                first_layout.transition_cost * 2,
            )

    def test_mixed_residency_selection_uses_queue_benefit_and_hysteresis(
        self,
    ) -> None:
        first = manifest(8)
        second = replace(
            manifest(8),
            model_id="synthetic-session-model-b",
            artifact_sha256=SHA_C,
        )
        benefits = {
            row.operator_id: 100 for row in first.operators
        }
        layouts = generate_mixed_ffn_residency_layouts(
            (
                PhoneFfnResidencyDemand(
                    first,
                    3,
                    8,
                    "coalesced-batch",
                    benefit_by_operator=benefits,
                    transition_energy_uj_by_session={
                        "HTP0": 11, "HTP1": 11, "HTP2": 11,
                    },
                ),
                PhoneFfnResidencyDemand(
                    second,
                    1,
                    8,
                    "coalesced-batch",
                    benefit_by_operator=benefits,
                    transition_energy_uj_by_session={
                        "HTP0": 22, "HTP1": 22, "HTP2": 22,
                    },
                ),
            ),
            tuple(session(index) for index in range(3)),
            phone_wide_limit_bytes=768,
            transition_energy_uj_by_session={
                "HTP0": 999,
                "HTP1": 999,
                "HTP2": 999,
            },
        )

        for layout in layouts:
            self.assertEqual(
                layout.transition_cost,
                sum(
                    11 if row.artifact_sha256 == SHA_A else 22
                    for row in layout.shards
                ),
            )

        selected, reason = select_mixed_ffn_residency_layout(
            layouts,
            current_geometry_sha256=None,
            switching_margin_uj=0,
        )
        self.assertIsNotNone(selected)
        self.assertEqual(reason, "PHONE_RESIDENCY_QUEUE_BENEFIT")
        self.assertEqual(
            {row.artifact_sha256 for row in selected.shards},
            {SHA_A},
        )
        retained, retained_reason = select_mixed_ffn_residency_layout(
            layouts,
            current_geometry_sha256=selected.geometry_sha256,
            switching_margin_uj=10**9,
        )
        self.assertEqual(retained, selected)
        self.assertEqual(retained_reason, "PHONE_RESIDENCY_HYSTERESIS")

    def test_mixed_residency_energy_uses_tokens_not_request_count(
        self,
    ) -> None:
        first = manifest(8)
        second = replace(
            manifest(8),
            model_id="synthetic-session-model-b",
            artifact_sha256=SHA_C,
        )
        first_benefits = {
            row.operator_id: 10 for row in first.operators
        }
        second_benefits = {
            row.operator_id: 20 for row in second.operators
        }
        layouts = generate_mixed_ffn_residency_layouts(
            (
                PhoneFfnResidencyDemand(
                    first,
                    7,
                    8,
                    "coalesced-batch",
                    benefit_by_operator=first_benefits,
                ),
                PhoneFfnResidencyDemand(
                    second,
                    3,
                    8,
                    "coalesced-batch",
                    benefit_by_operator=second_benefits,
                ),
            ),
            tuple(session(index) for index in range(3)),
            phone_wide_limit_bytes=768,
        )
        mixed = next(
            row for row in layouts
            if set(row.queue_benefit_by_artifact) == {SHA_A, SHA_C}
        )
        selected_by_artifact = {
            artifact: sum(
                (
                    first_benefits
                    if artifact == SHA_A else second_benefits
                )[operator_id]
                for shard in mixed.shards
                if shard.artifact_sha256 == artifact
                for operator_id in shard.operator_ids
            )
            for artifact in (SHA_A, SHA_C)
        }
        self.assertEqual(
            dict(mixed.queue_benefit_by_artifact),
            {
                SHA_A: selected_by_artifact[SHA_A] * 7,
                SHA_C: selected_by_artifact[SHA_C] * 3,
            },
        )

    def test_equal_residency_layout_uses_discovery_order_prefix(self) -> None:
        model = manifest(8)
        benefits = {
            row.operator_id: 100 for row in model.operators
        }
        layouts = generate_mixed_ffn_residency_layouts(
            (PhoneFfnResidencyDemand(
                model,
                3,
                8,
                "coalesced-batch",
                benefit_by_operator=benefits,
                transition_energy_uj_by_session={
                    "HTP0": 11, "HTP1": 11, "HTP2": 11,
                },
            ),),
            tuple(session(index) for index in range(3)),
            phone_wide_limit_bytes=512,
        )

        selected, reason = select_mixed_ffn_residency_layout(
            layouts,
            current_geometry_sha256=None,
            switching_margin_uj=0,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(reason, "PHONE_RESIDENCY_QUEUE_BENEFIT")
        self.assertEqual(
            tuple(sorted(row.session_id for row in selected.shards)),
            ("HTP0", "HTP1"),
        )

    def test_mixed_residency_retains_inactive_shards(self) -> None:
        first = manifest(8)
        second = replace(
            manifest(8),
            model_id="synthetic-session-model-b",
            artifact_sha256=SHA_C,
        )
        sessions = tuple(session(index) for index in range(3))
        current = generate_disjoint_ffn_shard_sets(
            first,
            sessions,
            phone_wide_limit_bytes=768,
            maximum_columns=8,
            batch_plan="coalesced-batch",
        )[-1]
        benefits = {
            row.operator_id: 100 for row in second.operators
        }

        layouts = generate_mixed_ffn_residency_layouts(
            (PhoneFfnResidencyDemand(
                second,
                1,
                8,
                "coalesced-batch",
                benefit_by_operator=benefits,
                transition_energy_uj_by_session={
                    "HTP0": 22, "HTP1": 22, "HTP2": 22,
                },
            ),),
            sessions,
            phone_wide_limit_bytes=768,
            current_shards=current.shards,
            transition_energy_uj_by_session={
                "HTP0": 999, "HTP1": 999, "HTP2": 999,
            },
        )

        retained = next(
            layout for layout in layouts
            if tuple(sorted(
                row.artifact_sha256 for row in layout.shards
            )) == (SHA_A, SHA_A, SHA_C)
        )
        self.assertEqual(retained.transition_cost, 22)
        self.assertEqual(len(retained.changed_session_ids), 1)
        self.assertEqual(
            dict(retained.queued_work_by_artifact), {SHA_C: 1}
        )
        self.assertEqual(
            set(retained.queue_benefit_by_artifact), {SHA_C}
        )
        self.assertEqual(
            retained.packing_value_for_artifact(SHA_A), 0
        )
        self.assertGreater(
            retained.packing_value_for_artifact(SHA_C), 0
        )
        with self.assertRaisesRegex(
            PhoneShardPlacementError,
            "phone residency artifact is absent",
        ):
            retained.packing_value_for_artifact(SHA_B)
        unchanged = set(sessions[index].session_id for index in range(3))
        unchanged.difference_update(retained.changed_session_ids)
        self.assertEqual(
            {
                row.session_id
                for row in retained.shards
                if row.artifact_sha256 == SHA_A
            },
            unchanged,
        )

    def test_v3_mixed_layout_reserves_retained_shards_globally(self) -> None:
        data = (
            Path(__file__).resolve().parents[1]
            / "campaigns" / "burstgpt" / "data"
        )
        qwen = ModelManifest.from_json(json.loads(
            (data / "QWEN_MANIFEST.json").read_text(encoding="ascii")
        ))
        gemma = ModelManifest.from_json(json.loads(
            (data / "GEMMA_MANIFEST.json").read_text(encoding="ascii")
        ))
        sessions = tuple(
            RuntimePhoneSessionCapability(
                session_id="HTP" + str(index),
                device_id="op15-phone",
                endpoint="session://op15-phone/HTP" + str(index),
                worker_identity_sha256=SHA_B,
                memory_resource_id=(
                    "op15-ram:session:HTP" + str(index)
                ),
                resident_memory_limit_bytes=3_208_646_656,
                shared_compute_resource_id="op15-htp",
                shared_transport_resource_ids=(
                    "desktop-usb-root",
                    "op15-functionfs",
                    "op15-ncm",
                ),
                supported_layer_mask=(1 << 48) - 1,
                maximum_columns=17_408,
                column_quantum=512,
                supported_data_types=("F16",),
                batch_plans=("split-row", "coalesced-batch"),
                ready=True,
                residency_state="cold",
            )
            for index in range(3)
        )
        qwen_demand = PhoneFfnResidencyDemand(
            qwen,
            36,
            17_408,
            "coalesced-batch",
            benefit_by_operator={
                row.operator_id: 1
                for row in qwen.operators if row.kind == "ffn"
            },
        )
        gemma_demand = PhoneFfnResidencyDemand(
            gemma,
            292,
            15_360,
            "coalesced-batch",
            benefit_by_operator={
                row.operator_id: 1
                for row in gemma.operators if row.kind == "ffn"
            },
        )
        current = next(
            row for row in generate_mixed_ffn_residency_layouts(
                (qwen_demand,),
                sessions,
                phone_wide_limit_bytes=9_161_588_608,
            )
            if len(row.shards) == 3
        )
        current_by_session = {
            row.session_id: row for row in current.shards
        }
        self.assertEqual(
            current_by_session["HTP1"].resident_bytes,
            3_208_642_560,
        )
        self.assertEqual(
            current_by_session["HTP2"].resident_bytes,
            3_208_642_560,
        )

        phone_limit = 8_328_056_704
        layouts = generate_mixed_ffn_residency_layouts(
            (gemma_demand,),
            sessions,
            phone_wide_limit_bytes=phone_limit,
            current_shards=current.shards,
        )
        target = next(
            row for row in layouts
            if {
                shard.session_id: shard.artifact_sha256
                for shard in row.shards
            } == {
                "HTP0": gemma.artifact_sha256,
                "HTP1": qwen.artifact_sha256,
                "HTP2": qwen.artifact_sha256,
            }
        )
        target_by_session = {
            row.session_id: row for row in target.shards
        }

        self.assertEqual(target.changed_session_ids, ("HTP0",))
        self.assertEqual(
            target_by_session["HTP0"].resident_bytes,
            1_769_472_000,
        )
        self.assertEqual(
            target_by_session["HTP1"], current_by_session["HTP1"]
        )
        self.assertEqual(
            target_by_session["HTP2"], current_by_session["HTP2"]
        )
        self.assertEqual(target.resident_bytes, 8_186_757_120)
        self.assertEqual(phone_limit - target.resident_bytes, 141_299_584)
        self.assertLessEqual(target.resident_bytes, phone_limit)
        for shard, capability in zip(target.shards, sessions):
            self.assertEqual(shard.session_id, capability.session_id)
            self.assertLessEqual(
                shard.resident_bytes,
                capability.resident_memory_limit_bytes,
            )
        for artifact in {row.artifact_sha256 for row in target.shards}:
            masks = [
                row.layer_mask for row in target.shards
                if row.artifact_sha256 == artifact
            ]
            self.assertEqual(sum(mask.bit_count() for mask in masks), (
                0 if not masks else (sum(masks)).bit_count()
            ))

        retained_qwen_operator_ids = tuple(sorted(
            operator_id
            for session_id in ("HTP1", "HTP2")
            for operator_id in current_by_session[session_id].operator_ids
        ))
        selection_qwen_demand = replace(
            qwen_demand,
            benefit_by_operator={
                operator_id: 1
                for operator_id in retained_qwen_operator_ids
            },
            allowed_operator_ids=retained_qwen_operator_ids,
        )
        selection_gemma_demand = replace(
            gemma_demand,
            transition_energy_uj_by_session={
                row.session_id: 100 for row in sessions
            },
        )
        selection_layouts = generate_mixed_ffn_residency_layouts(
            (selection_qwen_demand, selection_gemma_demand),
            sessions,
            phone_wide_limit_bytes=phone_limit,
            current_shards=current.shards,
        )
        selected, selection_reason, gains = (
            ModelPlacementController().select_phone_layout_candidate(
                (current, *selection_layouts),
                current_layout=current,
                minimum_energy_saving_ppm=0,
            )
        )
        self.assertEqual(selection_reason, (
            "PHONE_RESIDENCY_SESSION_MARGINAL_GAIN"
        ))
        self.assertEqual(selected.changed_session_ids, ("HTP0",))
        self.assertEqual(selected.shards, target.shards)
        self.assertEqual([row.session_id for row in gains], ["HTP0"])
        self.assertGreater(gains[0].current_warm_energy_saved_uj, 0)
        self.assertGreater(gains[0].gain_over_current_uj, 0)
        target = selected

        controller = ModelPlacementController()
        proposed_current = controller.propose_phone_layout(
            current,
            workspace_bytes=1,
            shared_compute_resource_id="op15-htp",
            shared_transport_resource_ids=(
                "desktop-usb-root", "op15-functionfs", "op15-ncm"
            ),
            observed_at_us=100,
        )
        controller.begin_phone_layout_transition(
            proposed_current.generation,
            ticket_id="load-v3-qqq",
            transition_ids=("load-v3-qqq",),
            ready_at_us=200,
            projection_token_sha256=canonical_sha256({"v3": "qqq"}),
            workspace_bytes=1,
            observed_at_us=110,
        )
        ready_current = controller.complete_phone_layout_transition(
            generation=proposed_current.generation,
            ticket_id="load-v3-qqq",
            transition_ids=("load-v3-qqq",),
            geometry_sha256=current.geometry_sha256,
            projection_token_sha256=canonical_sha256({"v3": "qqq"}),
            finished_at_us=200,
        )
        controller.bind_dispatched_request(
            "v3-desktop-during-replacement",
            qwen.artifact_sha256,
            "desktop-qwen",
            canonical_sha256({"v3": "desktop-qwen"}),
            0,
        )
        controller.mark_request_acquired(
            "v3-desktop-during-replacement"
        )
        desktop_binding = controller.request_binding(
            "v3-desktop-during-replacement"
        )
        unchanged_before = {
            row.session_id: row
            for row in controller.phone_session_states()
            if row.session_id in {"HTP1", "HTP2"}
        }
        failed_target = controller.propose_phone_layout(
            target,
            workspace_bytes=1,
            shared_compute_resource_id="op15-htp",
            shared_transport_resource_ids=(
                "desktop-usb-root", "op15-functionfs", "op15-ncm"
            ),
            observed_at_us=31_000_000,
        )
        self.assertEqual(
            controller.ready_phone_layout().layout.geometry_sha256,
            ready_current.layout.geometry_sha256,
        )
        event_count = len(controller.phone_layout_events())
        failed_token = canonical_sha256({"v3": "qqg-failure"})
        controller.begin_phone_layout_transition(
            failed_target.generation,
            ticket_id="replace-v3-htp0-failure",
            transition_ids=("replace-v3-htp0-failure",),
            ready_at_us=31_100_000,
            projection_token_sha256=failed_token,
            workspace_bytes=1,
            observed_at_us=31_000_100,
        )
        self.assertEqual(
            controller.ready_phone_layout().layout.geometry_sha256,
            ready_current.layout.geometry_sha256,
        )
        self.assertEqual(
            {
                row.session_id: row.state
                for row in controller.phone_session_states()
            },
            {"HTP0": "LOADING", "HTP1": "READY", "HTP2": "READY"},
        )
        self.assertEqual(
            [
                row["kind"]
                for row in controller.phone_layout_events()[event_count:]
            ],
            ["SESSION_DRAINING", "SESSION_LOADING", "PREPARING"],
        )
        self.assertEqual(
            {
                row.session_id: row
                for row in controller.phone_session_states()
                if row.session_id in {"HTP1", "HTP2"}
            },
            unchanged_before,
        )
        self.assertEqual(
            controller.request_binding("v3-desktop-during-replacement"),
            desktop_binding,
        )

        self.assertTrue(controller.fail_phone_layout_transition(
            "replace-v3-htp0-failure",
            generation=failed_target.generation,
            projection_token_sha256=failed_token,
            failed_at_us=31_050_000,
            reason="injected-v3-htp0-failure",
        ))
        restored = {
            row.session_id: row for row in controller.phone_session_states()
        }
        self.assertEqual(
            {key: row.state for key, row in restored.items()},
            {"HTP0": "READY", "HTP1": "READY", "HTP2": "READY"},
        )
        self.assertEqual(restored["HTP1"], unchanged_before["HTP1"])
        self.assertEqual(restored["HTP2"], unchanged_before["HTP2"])
        self.assertEqual(
            restored["HTP0"].resident_artifact_sha256,
            qwen.artifact_sha256,
        )
        self.assertEqual(
            controller.ready_phone_layout().layout.geometry_sha256,
            ready_current.layout.geometry_sha256,
        )

        proposed_target = controller.propose_phone_layout(
            target,
            workspace_bytes=1,
            shared_compute_resource_id="op15-htp",
            shared_transport_resource_ids=(
                "desktop-usb-root", "op15-functionfs", "op15-ncm"
            ),
            observed_at_us=62_000_000,
        )
        success_token = canonical_sha256({"v3": "qqg-success"})
        controller.begin_phone_layout_transition(
            proposed_target.generation,
            ticket_id="replace-v3-htp0-success",
            transition_ids=("replace-v3-htp0-success",),
            ready_at_us=62_100_000,
            projection_token_sha256=success_token,
            workspace_bytes=1,
            observed_at_us=62_000_100,
        )
        ready_target = controller.complete_phone_layout_transition(
            generation=proposed_target.generation,
            ticket_id="replace-v3-htp0-success",
            transition_ids=("replace-v3-htp0-success",),
            geometry_sha256=target.geometry_sha256,
            projection_token_sha256=success_token,
            finished_at_us=62_100_000,
        )
        self.assertEqual(
            ready_target.layout.geometry_sha256, target.geometry_sha256
        )
        final_states = {
            row.session_id: row for row in controller.phone_session_states()
        }
        self.assertEqual(
            final_states["HTP0"].resident_artifact_sha256,
            gemma.artifact_sha256,
        )
        self.assertEqual(final_states["HTP1"], unchanged_before["HTP1"])
        self.assertEqual(final_states["HTP2"], unchanged_before["HTP2"])
        self.assertEqual(
            controller.request_binding("v3-desktop-during-replacement"),
            desktop_binding,
        )

    def test_mixed_layout_envelope_supersedes_generic_artifact_shard(self):
        gemma = manifest()
        qwen = replace(
            manifest(),
            model_id="synthetic-qwen-session-model",
            artifact_sha256=SHA_C,
        )
        sessions = tuple(session(index) for index in range(3))
        benefits = {
            row.operator_id: 1 for row in gemma.operators
        }
        current = next(
            row for row in generate_mixed_ffn_residency_layouts(
                (PhoneFfnResidencyDemand(
                    qwen,
                    30,
                    qwen.feed_forward_length,
                    "coalesced-batch",
                    benefit_by_operator=benefits,
                ),),
                sessions,
                phone_wide_limit_bytes=768,
            )
            if len(row.shards) == 3
        )
        target = next(
            row for row in generate_mixed_ffn_residency_layouts(
                (PhoneFfnResidencyDemand(
                    gemma,
                    30,
                    gemma.feed_forward_length,
                    "coalesced-batch",
                    benefit_by_operator=benefits,
                ),),
                sessions,
                phone_wide_limit_bytes=768,
                current_shards=current.shards,
            )
            if {
                shard.session_id: shard.artifact_sha256
                for shard in row.shards
            } == {
                "HTP0": gemma.artifact_sha256,
                "HTP1": qwen.artifact_sha256,
                "HTP2": qwen.artifact_sha256,
            }
        ).with_session_generations({
            "HTP0": 2,
            "HTP1": 1,
            "HTP2": 1,
        })
        helper = SimpleNamespace(
            memory_resource_id="phone-memory",
            phone_sessions=sessions,
        )
        compiler = object.__new__(AutomatedRouteCompiler)
        compiler.catalog = SimpleNamespace(
            executor_by_device={"phone-a": helper},
            placement_profile=SimpleNamespace(
                devices={
                    "phone-a": SimpleNamespace(
                        allocation_limit_bytes=768
                    ),
                },
                memory_pools={
                    "phone-memory": MemoryPoolProfile(
                        "phone-memory", 768, 0
                    ),
                },
            ),
        )
        compiler._phone_residency_layout = target
        compiler._phone_ffn_shard_storage_by_artifact = MappingProxyType({})
        compiler._phone_ffn_shard_storage_sha256 = None
        compiler._ffn_resident_envelope_cache = {}
        compiler._cache_lock = threading.RLock()
        compiler._static_cache_evictions = 0
        coordinator = SimpleNamespace(
            executor_id="coordinator:phone",
            assisted_operator_kind="ffn",
            baseline_executor_id="coordinator:desktop",
            helper_device_id="phone-a",
            operator_ids=tuple(
                row.operator_id for row in gemma.operators
            ),
            split_fractions_ppm=(250_000, 500_000, 750_000),
            adapter_parameters={
                "ffn_column_quantum": 2,
                "ffn_max_runtime_partitions": 4,
                "ffn_runtime_control_protocol": "decode-boundary-v1",
                "ffn_weight_buffer_layout": "selected-width",
                "maximum_helper_resident_weight_bytes": 256,
                "usb_batch_plan": "coalesced-batch",
            },
        )

        envelopes = compiler._ffn_resident_envelopes(
            gemma, coordinator, gemma.request_work(8, 30)
        )
        exact = tuple(
            row for row in envelopes
            if row.geometry_sha256 == target.geometry_sha256
        )

        self.assertEqual(len(exact), 1)
        self.assertEqual(
            tuple(row.session_id for row in exact[0].shards),
            ("HTP0",),
        )
        self.assertEqual(
            tuple(row.session_id for row in exact[0].transition_shards),
            ("HTP0", "HTP1", "HTP2"),
        )
        self.assertEqual(exact[0].changed_session_ids, ("HTP0",))
        self.assertEqual(
            {
                row.session_id: row.session_generation
                for row in exact[0].transition_shards
            },
            {"HTP0": 2, "HTP1": 1, "HTP2": 1},
        )

        qwen_envelope = next(
            row for row in compiler._ffn_resident_envelopes(
                qwen, coordinator, qwen.request_work(8, 30)
            )
            if row.geometry_sha256 == target.geometry_sha256
        )
        logical_coordinator = object.__new__(
            RuntimeCompositeExecutorCapability
        )
        object.__setattr__(
            logical_coordinator, "executor_id", "logical:phone-route"
        )
        object.__setattr__(
            logical_coordinator, "helper_device_id", "phone-a"
        )
        pattern = _Pattern(
            route_key="mixed-retained-qwen",
            route_family="operator_split",
            assignments={
                row.operator_id: (
                    "desktop-a",
                    (
                        "phone-a"
                        if row.operator_id in qwen_envelope.operator_ids
                        else None
                    ),
                    (
                        250_000
                        if row.operator_id in qwen_envelope.operator_ids
                        else 0
                    ),
                )
                for row in qwen.operators
            },
            assisted_operator_kind="ffn",
            split_axis="column",
            split_fraction_ppm=250_000,
            overlap_kind="none",
            coordinator_device_id="desktop-a",
            coordinator_executor_id=logical_coordinator.executor_id,
            baseline_executor_id="desktop:parent",
            desktop_assignments={
                row.operator_id: "desktop-a" for row in qwen.operators
            },
            assistance_phase="decode",
            resident_envelope=True,
            phone_session_count=len(qwen_envelope.shards),
            phone_resident_envelope=qwen_envelope,
        )
        memory = (RuntimeMemoryDemand(
            demand_id="phone-wide-weights",
            resource_id="phone-memory",
            kind="model_weights",
            required_bytes=qwen_envelope.weight_bytes,
            resident_bytes=0,
            lifetime="resident",
            share_key="mixed-phone-weights",
            device_id="phone-a",
        ),)
        observations = tuple(
            PhoneSessionResidencyObservation(
                session_id=row.session_id,
                device_id="phone-a",
                executor_id=(
                    "physical:session-worker:" + row.session_id
                ),
                endpoint=row.endpoint,
                artifact_sha256=row.artifact_sha256,
                resident_geometry_sha256=(
                    row.resident_geometry_sha256
                ),
                operator_plan_sha256=row.operator_plan_sha256,
                session_generation=row.session_generation,
                resident_bytes=row.resident_bytes,
            )
            for row in qwen_envelope.transition_shards
        )
        exact_snapshot = replace(
            runtime_snapshot(qwen),
            phone_session_residency=observations,
        )
        demands, _envelope, error, requires_transition = (
            compiler._resident_phone_memory_demands(
                qwen,
                pattern,
                exact_snapshot,
                logical_coordinator,
                memory,
            )
        )
        phone_wide = next(
            row for row in demands if row.kind == "model_weights"
        )
        self.assertIsNone(error)
        self.assertFalse(requires_transition)
        self.assertEqual(
            phone_wide.resident_bytes, qwen_envelope.weight_bytes
        )
        self.assertEqual(
            {row.resident_bytes for row in demands
             if row.kind == "session_residency_constraint"},
            {256},
        )

        mismatches = (
            replace(observations[0], endpoint="session://wrong/HTP0"),
            replace(observations[0], artifact_sha256=SHA_B),
            replace(
                observations[0], resident_geometry_sha256=SHA_A
            ),
            replace(observations[0], operator_plan_sha256=SHA_A),
            replace(
                observations[0],
                session_generation=observations[0].session_generation + 1,
            ),
            replace(
                observations[0],
                resident_bytes=observations[0].resident_bytes - 1,
            ),
            replace(observations[0], state="LOADING"),
            replace(observations[0], device_id="phone-b"),
        )
        for mismatch in mismatches:
            with self.subTest(mismatch=mismatch.to_json()):
                mismatch_snapshot = replace(
                    exact_snapshot,
                    phone_session_residency=(
                        mismatch, *observations[1:]
                    ),
                )
                mismatch_demands, _value, _error, transition = (
                    compiler._resident_phone_memory_demands(
                        qwen,
                        pattern,
                        mismatch_snapshot,
                        logical_coordinator,
                        memory,
                    )
                )
                self.assertTrue(transition)
                self.assertEqual(
                    next(
                        row for row in mismatch_demands
                        if row.kind == "model_weights"
                    ).resident_bytes,
                    0,
                )

        cache_scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        cache_scheduler.register_runtime_capabilities(catalog())
        cache_compiler = cache_scheduler._automated_compiler()
        self.assertTrue(cache_compiler.set_phone_residency_layout(target))
        cache_key = (qwen.artifact_sha256, "shape", "coordinator")
        cache_compiler._ffn_resident_envelope_cache[cache_key] = (
            qwen_envelope,
        )
        bumped = replace(
            target,
            session_generation_by_id={
                "HTP0": 4, "HTP1": 1, "HTP2": 1,
            },
            replacement_source_identities=tuple(
                replace(row, session_generation=3)
                if row.session_id == "HTP0" else row
                for row in target.replacement_source_identities
            ),
        )
        self.assertEqual(bumped.geometry_sha256, target.geometry_sha256)
        self.assertTrue(cache_compiler.set_phone_residency_layout(bumped))
        self.assertNotIn(
            cache_key, cache_compiler._ffn_resident_envelope_cache
        )

        rough_catalog = catalog_with_gpu_desktop_control(qwen)
        rough_compiler = AutomatedRouteCompiler(
            rough_catalog, ResourceTimeline(rough_catalog.resources)
        )
        rough_compiler.set_phone_residency_layout(target)
        object.__setattr__(
            logical_coordinator,
            "baseline_executor_id",
            rough_catalog.desktop_control_by_artifact[
                qwen.artifact_sha256
            ].executor_id,
        )
        selected_pattern = SimpleNamespace(
            route_key="selected-mixed-retained-pair",
            resident_envelope=True,
            phone_resident_envelope=qwen_envelope,
        )
        alternative_pattern = SimpleNamespace(
            route_key="cheaper-hypothetical-pair",
            resident_envelope=True,
            phone_resident_envelope=replace(
                qwen_envelope, geometry_sha256=SHA_A
            ),
        )

        def rough_visit(**kwargs):
            value = kwargs["pattern"]
            selected = value is selected_pattern
            return (
                RoughPlacementVisit(
                    route_key=value.route_key,
                    residency_variant="hot",
                    rough_latency_us=(200 if selected else 100),
                    rough_energy_uj=(200 if selected else 100),
                    required_group="retained-two-session-helper",
                ),
                False,
                "retained-two-session-helper",
            )

        with patch.object(
            rough_compiler, "_rough_visit", side_effect=rough_visit
        ), patch.object(
            rough_compiler,
            "_coordinator",
            return_value=logical_coordinator,
        ):
            visits = rough_compiler._rough_visits(
                qwen,
                (alternative_pattern, selected_pattern),
                qwen.request_work(8, 30),
                rough_catalog.placement_profile,
                canonical_sha256(rough_catalog.placement_profile),
                runtime_snapshot(qwen),
                0,
            )
        mandatory = {
            row.route_key for row in visits if row.mandatory
        }
        self.assertIn(selected_pattern.route_key, mandatory)
        self.assertNotIn(alternative_pattern.route_key, mandatory)

    def test_ffn_storage_limits_session_assignments(self):
        model = manifest()
        sessions = tuple(session(index) for index in range(3))
        storage = tuple(
            PhoneFfnShardStorageMetadata(
                parent_artifact_sha256=model.artifact_sha256,
                shard_sha256=(SHA_B if index != 1 else SHA_C),
                path=f"/data/local/tmp/shards/HTP{index}.ffn.gguf",
                layer_mask=mask,
                maximum_columns=model.feed_forward_length,
                session_id=f"HTP{index}",
            )
            for index, mask in enumerate(
                (0b00000011, 0b00001100, 0b00110000)
            )
        )
        compiler = object.__new__(AutomatedRouteCompiler)
        compiler.catalog = SimpleNamespace(
            executor_by_device={
                "phone-a": SimpleNamespace(
                    memory_resource_id="phone-memory",
                    phone_sessions=sessions,
                ),
            },
            placement_profile=SimpleNamespace(
                devices={
                    "phone-a": SimpleNamespace(
                        allocation_limit_bytes=768
                    ),
                },
                memory_pools={
                    "phone-memory": MemoryPoolProfile(
                        "phone-memory", 768, 0
                    ),
                },
            ),
        )
        compiler._phone_residency_layout = None
        compiler._phone_ffn_shard_storage_by_artifact = MappingProxyType({
            model.artifact_sha256: MappingProxyType({
                row.session_id: row for row in storage
            }),
        })
        compiler._phone_ffn_shard_storage_sha256 = canonical_sha256(
            [row.to_json() for row in storage]
        )
        compiler._ffn_resident_envelope_cache = {}
        compiler._cache_lock = threading.RLock()
        compiler._static_cache_evictions = 0
        coordinator = SimpleNamespace(
            executor_id="coordinator:phone-storage",
            assisted_operator_kind="ffn",
            baseline_executor_id="coordinator:desktop",
            helper_device_id="phone-a",
            operator_ids=tuple(row.operator_id for row in model.operators),
            split_fractions_ppm=(250_000, 500_000, 750_000),
            adapter_parameters={
                "ffn_column_quantum": 2,
                "ffn_max_runtime_partitions": 4,
                "ffn_runtime_control_protocol": "decode-boundary-v1",
                "ffn_weight_buffer_layout": "selected-width",
                "maximum_helper_resident_weight_bytes": 256,
                "usb_batch_plan": "coalesced-batch",
            },
        )

        envelopes = compiler._ffn_resident_envelopes(
            model, coordinator, model.request_work(8, 30)
        )
        three_sessions = next(
            row for row in envelopes if row.session_count == 3
        )

        self.assertEqual(
            {row.session_id: row.layer_mask for row in three_sessions.shards},
            {
                "HTP0": 0b00000011,
                "HTP1": 0b00001100,
                "HTP2": 0b00110000,
            },
        )
        self.assertEqual(
            three_sessions.layer_mask
                & ~sum(row.layer_mask for row in storage),
            0,
        )

    def test_shared_phone_transition_energy_is_unioned(self) -> None:
        first = manifest(8)
        second = replace(
            manifest(8),
            model_id="synthetic-session-model-b",
            artifact_sha256=SHA_C,
        )
        sessions = tuple(session(index) for index in range(3))
        benefits = {
            row.operator_id: 1_000 for row in first.operators
        }
        transition_by_session = {
            row.session_id: 450 for row in sessions
        }
        first_demand = PhoneFfnResidencyDemand(
            first,
            10,
            8,
            "coalesced-batch",
            benefit_by_operator=benefits,
            transition_energy_uj_by_session=transition_by_session,
            transition_energy_aggregation="shared_phone_union",
        )
        initial = next(
            row for row in generate_mixed_ffn_residency_layouts(
                (first_demand,),
                sessions,
                phone_wide_limit_bytes=768,
            )
            if len(row.shards) == 3
        )
        self.assertEqual(initial.transition_cost, 450)
        self.assertEqual(
            sum(initial.transition_cost_by_session.values()), 450
        )
        self.assertEqual(
            initial.transition_cost_aggregation, "shared_phone_union"
        )

        second_demand = PhoneFfnResidencyDemand(
            second,
            10,
            8,
            "coalesced-batch",
            benefit_by_operator=benefits,
            transition_energy_uj_by_session=transition_by_session,
            transition_energy_aggregation="shared_phone_union",
        )
        mixed = next(
            row for row in generate_mixed_ffn_residency_layouts(
                (second_demand,),
                sessions,
                phone_wide_limit_bytes=768,
                current_shards=initial.shards,
            )
            if tuple(sorted(
                shard.artifact_sha256 for shard in row.shards
            )) == (SHA_A, SHA_A, SHA_C)
        )
        self.assertEqual(len(mixed.changed_session_ids), 1)
        self.assertEqual(mixed.transition_cost, 450)
        self.assertEqual(
            sum(mixed.transition_cost_by_session.values()), 450
        )

    def test_mixed_residency_does_not_compare_rough_work_with_energy(
        self,
    ) -> None:
        layouts = generate_mixed_ffn_residency_layouts(
            (PhoneFfnResidencyDemand(
                manifest(8), 2, 8, "coalesced-batch"
            ),),
            tuple(session(index) for index in range(3)),
            phone_wide_limit_bytes=768,
            transition_energy_uj_by_session={
                "HTP0": 1,
                "HTP1": 1,
                "HTP2": 1,
            },
        )

        selected, reason = select_mixed_ffn_residency_layout(
            layouts,
            current_geometry_sha256=None,
            switching_margin_uj=0,
        )

        self.assertIsNone(selected)
        self.assertEqual(reason, "PHONE_RESIDENCY_ENERGY_UNKNOWN")

    def test_phone_shard_layer_overlap_is_scoped_to_artifact(self) -> None:
        first = replace(shard(0, 0b0011), artifact_sha256=SHA_A)
        second = replace(shard(1, 0b0011), artifact_sha256=SHA_C)

        contract = RuntimeExecutionContract(
            execution_mode="adaptive-split",
            initial_split_fraction_ppm=0,
            allowed_adaptive_fractions_ppm=(0, 500_000),
            batch_plan="coalesced-batch",
            maximum_batch_size=4,
            queue_depth=4,
            phone_device_id="phone-a",
            phone_endpoint="physical://phone-a",
            operator_kind="ffn",
            phone_shards=(first, second),
        )

        self.assertEqual(len(contract.phone_shards), 2)
        with self.assertRaisesRegex(
            RuntimePlanError, "phone shard layers overlap"
        ):
            replace(
                contract,
                phone_shards=(
                    first,
                    replace(second, artifact_sha256=SHA_A),
                ),
            )

    def test_mixed_residency_identity_is_request_artifact_independent(
        self,
    ) -> None:
        shards = (
            replace(shard(0, 0b0011), artifact_sha256=SHA_A),
            replace(shard(1, 0b0011), artifact_sha256=SHA_C),
        )
        values = tuple(
            runtime_residency_component_identity_from_parts(
                artifact_sha256=artifact_sha256,
                resident_shard_geometry_sha256=(SHA_B, SHA_C),
                desktop_placement_sha256=SHA_A,
                transport_generation="synthetic-transport-v1",
                operator_protocol="synthetic-plan-v1",
                session_resource_ids=(
                    "phone-session-0", "phone-session-1"
                ),
                executor_id="synthetic-phone-plan",
                resident_endpoint="synthetic://phone-resident",
                phone_shards=shards,
            )
            for artifact_sha256 in (SHA_A, SHA_C)
        )

        self.assertEqual(
            values[0].identity_sha256, values[1].identity_sha256
        )
        self.assertEqual(values[0].resident_artifact_sha256s, (SHA_A, SHA_C))
        self.assertIsNone(values[0].desktop_placement_sha256)

    def test_mixed_component_reuse_is_scoped_per_request_artifact(
        self,
    ) -> None:
        tracker = RuntimeResidencyCohortTracker()
        mixed = RuntimeResidencyComponentIdentity(
            artifact_sha256=SHA_B,
            resident_shard_geometry_sha256=(SHA_B, SHA_C),
            desktop_placement_sha256=None,
            transport_generation="synthetic-transport-v1",
            operator_protocol="synthetic-plan-v1",
            session_resource_ids=(
                "phone-session-0", "phone-session-1"
            ),
            executor_id="synthetic-phone-plan",
            resident_artifact_sha256s=(SHA_A, SHA_C),
        )
        desktop = RuntimeResidencyComponentIdentity(
            artifact_sha256=SHA_A,
            resident_shard_geometry_sha256=(),
            desktop_placement_sha256=SHA_C,
            transport_generation=None,
            operator_protocol="synthetic-plan-v1",
            session_resource_ids=(),
            executor_id="synthetic-desktop-plan",
        )

        tracker.record_planning_arrival(
            "mixed-planning-a", (desktop, mixed), 1_000
        )
        tracker.record_arrival(
            "mixed-selected-a", mixed, 1_100, SHA_A
        )
        tracker.record_arrival(
            "mixed-selected-c", mixed, 1_200, SHA_C
        )

        self.assertIn(
            mixed.identity_sha256,
            tracker.reuse_projections(SHA_A, 1_300),
        )
        self.assertIn(
            mixed.identity_sha256,
            tracker.reuse_projections(SHA_C, 1_300),
        )

    def test_different_model_desktop_work_keeps_confirmed_phone_component(
        self,
    ) -> None:
        source = catalog()
        arena = replace(
            session(0),
            device_id="helper-c",
            endpoint="session://helper-c/HTP0",
            memory_resource_id="phone-session-0",
            shared_compute_resource_id="compute:helper-c",
            shared_transport_resource_ids=(
                "link:usb-in", "link:usb-out"
            ),
        )
        pools = dict(source.placement_profile.memory_pools)
        pools[arena.memory_resource_id] = MemoryPoolProfile(
            arena.memory_resource_id,
            arena.resident_memory_limit_bytes,
            0,
        )
        helper = replace(
            source.executor_by_device["helper-c"],
            execution_resource_ids=(
                "compute:helper-c", "link:usb-in", "link:usb-out"
            ),
            exclusive_residency_resource_id="compute:helper-c",
            phone_sessions=(arena,),
        )
        profile = replace(
            source,
            placement_profile=replace(
                source.placement_profile,
                memory_pools=MappingProxyType(pools),
            ),
            executors=tuple(
                helper if row.device_id == "helper-c" else row
                for row in source.executors
            ),
        )
        first_model = manifest()
        other_model = replace(
            first_model,
            model_id="synthetic-other-model",
            artifact_sha256=SHA_C,
        )
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        scheduler.register_model_manifest(first_model)
        scheduler.register_model_manifest(other_model)
        component = RuntimeResidencyComponentIdentity(
            artifact_sha256=first_model.artifact_sha256,
            resident_shard_geometry_sha256=(SHA_B,),
            desktop_placement_sha256=SHA_C,
            transport_generation="synthetic-transport-v1",
            operator_protocol="synthetic-plan-v1",
            session_resource_ids=(arena.memory_resource_id,),
            executor_id="synthetic-phone-plan",
            resident_endpoint="synthetic://phone-resident",
        )
        desktop = RuntimeResidencyComponentIdentity(
            artifact_sha256=other_model.artifact_sha256,
            resident_shard_geometry_sha256=(),
            desktop_placement_sha256=SHA_B,
            transport_generation=None,
            operator_protocol="synthetic-plan-v1",
            session_resource_ids=(),
            executor_id="synthetic-desktop-plan",
        )
        tracker = scheduler._runtime_residency_cohorts
        self.assertEqual(
            tracker.confirm_resident_component(component, profile),
            ("compute:helper-c",),
        )
        snapshot = runtime_snapshot(first_model)
        demand = scheduler._model_demand_snapshot(
            request("different-model-desktop"),
            other_model,
            snapshot,
            1_000,
        )

        self.assertEqual(
            demand.current_resident_component_identity_sha256,
            component.identity_sha256,
        )
        self.assertEqual(
            tracker.confirm_resident_component(desktop, profile), ()
        )
        self.assertEqual(
            tracker.confirmed_resident_components(profile, snapshot)[
                "compute:helper-c"
            ].identity_sha256,
            component.identity_sha256,
        )

    def test_session_diagnostic_reason_does_not_change_capability_identity(
        self,
    ) -> None:
        helper = catalog().executor_by_device["helper-c"]
        unavailable = replace(
            session(0, ready=False),
            device_id="helper-c",
            unavailable_reason="probe-a",
        )
        resources = (
            unavailable.shared_compute_resource_id,
            *unavailable.shared_transport_resource_ids,
        )
        first = replace(
            helper,
            execution_resource_ids=resources,
            phone_sessions=(unavailable,),
        )
        second = replace(
            first,
            phone_sessions=(replace(unavailable, unavailable_reason="probe-b"),),
        )

        self.assertNotEqual(first, second)
        self.assertEqual(
            _static_executor_identity(first),
            _static_executor_identity(second),
        )

    def test_energy_aware_packing_selects_the_highest_value_layer_set(
        self,
    ) -> None:
        base = manifest(3)
        sizes = (6, 5, 5)
        tensors = tuple(
            replace(
                row,
                shape=(1, size),
                nbytes=size,
            )
            for row, size in zip(base.tensors, sizes)
        )
        model = replace(
            base,
            artifact_bytes=sum(sizes),
            tensors=tensors,
        )
        arena = replace(
            session(0),
            resident_memory_limit_bytes=10,
        )

        rows = generate_disjoint_ffn_shard_sets(
            model,
            (arena,),
            phone_wide_limit_bytes=10,
            maximum_columns=8,
            batch_plan="coalesced-batch",
            benefit_by_operator={
                "layer:0:ffn": 12,
                "layer:1:ffn": 9,
                "layer:2:ffn": 9,
            },
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0].operator_ids,
            ("layer:1:ffn", "layer:2:ffn"),
        )
        self.assertEqual(rows[0].resident_bytes, 10)

    def test_shards_count_only_resident_ffn_matrices(self) -> None:
        base = manifest(2)
        norms = tuple(
            ModelTensorManifest(
                tensor_id=f"layer.{index}.ffn.norm",
                shape=(8,),
                quantization="F32",
                quantization_block_size=1,
                quantization_type_size=4,
                nbytes=32,
                role="ffn_weight",
                layer_index=index,
            )
            for index in range(2)
        )
        model = replace(
            base,
            artifact_bytes=base.artifact_bytes + sum(
                row.nbytes for row in norms
            ),
            tensors=base.tensors + norms,
            operators=tuple(
                replace(
                    row,
                    tensor_ids=row.tensor_ids + (norms[index].tensor_id,),
                )
                for index, row in enumerate(base.operators)
            ),
        )

        rows = generate_disjoint_ffn_shard_sets(
            model,
            (session(0),),
            phone_wide_limit_bytes=256,
            maximum_columns=8,
            batch_plan="coalesced-batch",
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].resident_bytes, 256)
        self.assertEqual(rows[0].shards[0].layer_mask, 0b11)

    def test_disjoint_shards_cover_every_discovered_session_count(self) -> None:
        probe = "\n".join(
            json.dumps({
                "allocated_bytes": 256,
                "backend": f"HTP{index}",
                "event": "resident",
                "session_index": index,
            })
            for index in range(3)
        ) + "\n" + json.dumps({
            "event": "complete",
            "resident_sessions": 3,
            "status": "PASS",
        })
        discovered = parse_phone_session_probe(
            probe,
            PhoneSessionDiscoveryConfiguration(
                device_id="phone-a",
                endpoint_prefix="session://phone-a/",
                worker_identity_sha256=SHA_B,
                shared_compute_resource_id="phone-htp",
                shared_transport_resource_ids=(
                    "phone-functionfs", "phone-usb"
                ),
                phone_wide_memory_resource_id="phone-memory",
                phone_wide_limit_bytes=768,
                supported_layer_mask=(1 << 8) - 1,
                maximum_columns=8,
                column_quantum=2,
                supported_data_types=("F16",),
                batch_plans=("coalesced-batch", "split-row"),
            ),
        )
        rows = generate_disjoint_ffn_shard_sets(
            manifest(),
            discovered,
            phone_wide_limit_bytes=768,
            maximum_columns=8,
            batch_plan="coalesced-batch",
        )

        self.assertEqual(tuple(len(row.shards) for row in rows), (1, 2, 3))
        self.assertEqual(tuple(row.resident_bytes for row in rows), (256, 512, 768))
        for session_count, row in enumerate(rows, start=1):
            self.assertEqual(
                tuple(placed.session_id for placed in row.shards),
                tuple(f"HTP{index}" for index in range(session_count)),
            )
            covered = 0
            for placed in row.shards:
                self.assertEqual(placed.resident_bytes, 256)
                self.assertEqual(covered & placed.layer_mask, 0)
                covered |= placed.layer_mask
            self.assertEqual(covered.bit_count(), 2 * len(row.shards))

    def test_unavailable_discovered_session_is_logged_and_degrades(self) -> None:
        probe = "\n".join((
            json.dumps({
                "allocated_bytes": 256,
                "backend": "HTP0",
                "event": "resident",
                "session_index": 0,
            }),
            json.dumps({
                "allocated_bytes": 256,
                "backend": "HTP1",
                "event": "resident",
                "session_index": 1,
            }),
            json.dumps({
                "backend": "HTP2",
                "event": "unavailable",
                "reason": "memory_safety_floor",
                "requested_bytes": 256,
                "session_index": 2,
            }),
            json.dumps({
                "event": "complete",
                "resident_sessions": 2,
                "status": "PASS",
            }),
        ))
        discovered = parse_phone_session_probe(
            probe,
            PhoneSessionDiscoveryConfiguration(
                device_id="phone-a",
                endpoint_prefix="session://phone-a/",
                worker_identity_sha256=SHA_B,
                shared_compute_resource_id="phone-htp",
                shared_transport_resource_ids=(
                    "phone-functionfs", "phone-usb"
                ),
                phone_wide_memory_resource_id="phone-memory",
                phone_wide_limit_bytes=768,
                supported_layer_mask=(1 << 8) - 1,
                maximum_columns=8,
                column_quantum=2,
                supported_data_types=("F16",),
                batch_plans=("coalesced-batch", "split-row"),
            ),
        )

        self.assertEqual(len(discovered), 3)
        self.assertEqual(
            tuple(row.ready for row in discovered),
            (True, True, False),
        )
        ready = generate_disjoint_ffn_shard_sets(
            manifest(),
            discovered,
            phone_wide_limit_bytes=768,
            maximum_columns=8,
            batch_plan="coalesced-batch",
        )
        diagnostic = generate_disjoint_ffn_shard_sets(
            manifest(),
            discovered,
            phone_wide_limit_bytes=768,
            maximum_columns=8,
            batch_plan="coalesced-batch",
            include_unready=True,
        )

        self.assertEqual(tuple(len(row.shards) for row in ready), (1, 2))
        self.assertEqual(
            tuple(len(row.shards) for row in diagnostic),
            (1, 2, 3),
        )
        self.assertEqual(
            diagnostic[-1].unavailable_session_ids,
            ("HTP2",),
        )

    def test_phone_sessions_share_execution_resources_not_capacity(self) -> None:
        sessions = tuple(session(index) for index in range(3))
        helper = RuntimeExecutorCapability(
            executor_id="phone-executor",
            device_id="phone-a",
            endpoint="physical://phone-a",
            backend="phone",
            execution_resource_ids=(
                "phone-functionfs", "phone-htp", "phone-usb"
            ),
            memory_resource_id="phone-memory",
            kernel_profiles={"ffn": "phone-ffn"},
            supported_quantizations=("*",),
            supports_whole_model=False,
            supports_layer_placement=False,
            supports_operator_placement=True,
            supports_kv_cache=False,
            supports_split_coordinator=False,
            supports_split_helper=True,
            split_axes=("column",),
            split_fractions_ppm=(500_000,),
            layer_fractions_ppm=(),
            residency_states=("cold", "hot", "warm"),
            maturity="QUALIFIED",
            evidence_ids=("synthetic-phone-evidence",),
            qualified_fallback=False,
            maximum_temperature_millic=90_000,
            minimum_battery_ppm=100_000,
            workspace_bytes_per_token=16,
            phone_sessions=sessions,
        )

        self.assertEqual(
            {row.shared_compute_resource_id for row in helper.phone_sessions},
            {"phone-htp"},
        )
        self.assertEqual(
            len({row.memory_resource_id for row in helper.phone_sessions}),
            3,
        )
        with self.assertRaisesRegex(
            RuntimeCapabilityError, "must share execution resources"
        ):
            replace(
                helper,
                phone_sessions=(
                    sessions[0],
                    replace(sessions[1], shared_compute_resource_id="other-htp"),
                ),
            )

    def test_multi_session_transition_binds_one_atomic_shard_set(self) -> None:
        base = RuntimeTransitionPlan(
            transition_id="load-phone-shards",
            device_id="phone-a",
            source_state="cold",
            target_state="hot",
            latency_us=10,
            energy_uj=20,
            resource_ids=("phone-functionfs", "phone-htp", "phone-usb"),
            maturity="SHADOW",
        )
        shards = (shard(0, 0b0011), shard(1, 0b1100))

        bound = AutomatedRouteCompiler._bind_phone_shards_to_transition(
            base, shards
        )

        self.assertEqual(bound.latency_us, base.latency_us)
        self.assertEqual(bound.energy_uj, base.energy_uj)
        self.assertEqual(bound.phone_shards, shards)

    def test_partial_transition_binds_exact_source_session_identity(
        self,
    ) -> None:
        base = RuntimeTransitionPlan(
            transition_id="replace-phone-shard",
            device_id="phone-a",
            source_state="warm",
            target_state="hot",
            latency_us=10,
            energy_uj=20,
            resource_ids=("phone-htp",),
            maturity="QUALIFIED",
            executor_id="phone-executor",
            evictions=(RuntimeResidencyEviction(
                model_id="model-a",
                artifact_sha256=SHA_A,
                device_id="phone-a",
                resident_bytes=768,
                generation=7,
                executor_id="phone-executor",
            ),),
        )
        source = PhoneFfnSessionIdentity(
            session_id="HTP2",
            artifact_sha256=SHA_A,
            resident_geometry_sha256=SHA_B,
            operator_plan_sha256=SHA_C,
            session_generation=4,
        )
        target_shards = (
            replace(
                shard(0, 0b0011),
                artifact_sha256=SHA_A,
                session_generation=1,
            ),
            replace(
                shard(1, 0b1100),
                artifact_sha256=SHA_A,
                session_generation=1,
            ),
            replace(
                shard(2, 0b10000),
                artifact_sha256=SHA_B,
                session_generation=5,
            ),
        )

        bound = AutomatedRouteCompiler._bind_phone_shards_to_transition(
            base,
            target_shards,
            changed_session_ids=("HTP2",),
            replacement_source_identities=(source,),
            replacement_source_resident_bytes_by_session={"HTP2": 256},
            model_id="model-b",
            target_artifact_sha256=SHA_B,
            phone_device_id="phone-a",
        )

        self.assertEqual(bound.changed_phone_session_ids, ("HTP2",))
        self.assertEqual(bound.phone_shards, target_shards)
        self.assertEqual(len(bound.evictions), 1)
        eviction = bound.evictions[0]
        self.assertEqual(eviction.session_id, "HTP2")
        self.assertEqual(eviction.artifact_sha256, SHA_A)
        self.assertEqual(eviction.resident_bytes, 256)
        self.assertEqual(eviction.generation, 4)
        self.assertEqual(eviction.resident_geometry_sha256, SHA_B)
        self.assertEqual(eviction.operator_plan_sha256, SHA_C)

    def test_reverse_transition_does_not_evict_retained_model_aggregate(self) -> None:
        retained = tuple(
            replace(shard(index, 1 << index), artifact_sha256=SHA_A,
                    resident_bytes=3_208_642_560, session_generation=1)
            for index in range(2)
        )
        replacement = replace(
            shard(2, 1 << 2), artifact_sha256=SHA_A,
            resident_bytes=3_208_642_560, session_generation=3,
        )
        selected = replacement.session_id
        source = PhoneFfnSessionIdentity(
            selected, SHA_B, SHA_C, SHA_A, 2,
        )
        retained_eviction = RuntimeResidencyEviction(
            model_id="model-a", artifact_sha256=SHA_A,
            device_id="phone-a", resident_bytes=6_417_285_120,
            generation=2, executor_id="previous-logical-executor",
        )
        source_eviction = replace(
            retained_eviction, model_id="model-b", artifact_sha256=SHA_B,
            resident_bytes=2_831_155_200,
        )
        desktop_eviction = replace(retained_eviction, device_id="desktop")
        base = RuntimeTransitionPlan(
            transition_id="reverse-one-session", device_id="phone-a",
            source_state="warm", target_state="hot", latency_us=10,
            energy_uj=20, resource_ids=("phone-htp",), maturity="QUALIFIED",
            evictions=(retained_eviction, source_eviction, desktop_eviction),
            prepares_device_ids=("desktop", "phone-a"),
        )

        authorization = PhoneSessionReplacementAuthorization.create(
            selected_session_id=selected,
            source_shards=(*retained, replace(
                replacement, artifact_sha256=SHA_B, resident_bytes=2_831_155_200,
                resident_geometry_sha256=SHA_C, operator_plan_sha256=SHA_A,
                session_generation=2,
            )),
            target_shards=(*retained, replacement),
        )

        def prepare(transition, *, ready=False):
            bound = AutomatedRouteCompiler._bind_phone_shards_to_transition(
                transition, (*retained, replacement), (selected,), (source,),
                {selected: 2_831_155_200}, "model-a", SHA_A, "phone-a",
                {SHA_B: "model-b"},
            )
            holder = SimpleNamespace(
                helper_plan=SimpleNamespace(
                    execution_contract=SimpleNamespace(phone_device_id="phone-a"),
                    transitions=(bound,),
                ),
                helper_binding=SimpleNamespace(participants=(SimpleNamespace(
                    device_id="phone-a", resource_ids=("phone-htp",),
                ),)),
                preparation_changed_session_ids=() if ready else (selected,),
                replacement_authorization=None if ready else authorization,
            )
            prepared = RuntimeHelperExecutionEnvelope.preparation_transitions.fget(holder)[0]
            self.assertIn(desktop_eviction, bound.evictions)
            return prepared

        bound = prepare(base)
        phone_evictions = bound.evictions
        self.assertEqual(bound.changed_phone_session_ids, (selected,))
        self.assertEqual(bound.phone_shards[:2], retained)
        self.assertEqual(len(phone_evictions), 1)
        self.assertEqual(phone_evictions[0].session_id, selected)
        self.assertEqual(phone_evictions[0].generation, 2)
        self.assertEqual(phone_evictions[0].resident_bytes, 2_831_155_200)
        self.assertEqual(phone_evictions[0].artifact_sha256, SHA_B)
        self.assertNotIn(desktop_eviction, bound.evictions)
        ready = prepare(base, ready=True)
        self.assertEqual(ready.changed_phone_session_ids, ())
        self.assertEqual(len(ready.evictions), 2)
        for invalid in (
            replace(retained_eviction, resident_bytes=6_417_285_121),
            replace(retained_eviction, artifact_sha256=SHA_C),
        ):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "retained phone eviction differs from the session map"
            ):
                prepare(replace(base, evictions=(source_eviction, invalid,
                                                desktop_eviction)))

    def test_failed_multi_session_memory_reservation_rolls_back_all(self) -> None:
        ledger = RuntimeMemoryLedger()
        snapshot = RuntimePlacementSnapshot(
            snapshot_id="multi-session-memory",
            captured_at_us=0,
            valid_until_us=100,
            capacities={
                "phone-memory": DeviceMemoryCapacity(
                    "phone-memory", 512, 0, 0
                ),
                "phone-session-0": DeviceMemoryCapacity(
                    "phone-session-0", 256, 0, 0
                ),
                "phone-session-1": DeviceMemoryCapacity(
                    "phone-session-1", 127, 0, 0
                ),
            },
        )
        demands = (
            RuntimeMemoryDemand(
                demand_id="weights:phone-a",
                resource_id="phone-memory",
                kind="model_weights",
                required_bytes=256,
                resident_bytes=0,
                lifetime="resident",
                share_key=SHA_A + ":phone-a",
                device_id="phone-a",
            ),
            RuntimeMemoryDemand(
                demand_id="phone-session:HTP0:weights",
                resource_id="phone-session-0",
                kind="session_residency_constraint",
                required_bytes=128,
                resident_bytes=0,
                lifetime="resident",
                share_key=SHA_A + ":HTP0",
                device_id="phone-a",
            ),
            RuntimeMemoryDemand(
                demand_id="phone-session:HTP1:weights",
                resource_id="phone-session-1",
                kind="session_residency_constraint",
                required_bytes=128,
                resident_bytes=0,
                lifetime="resident",
                share_key=SHA_A + ":HTP1",
                device_id="phone-a",
            ),
        )
        before = ledger.snapshot()

        with self.assertRaisesRegex(
            RuntimeResourceError, "memory capacity is insufficient"
        ):
            ledger.reserve(
                "multi-session-owner", demands, snapshot,
                start_us=1, reserved_until_us=20,
            )

        self.assertEqual(ledger.snapshot(), before)

    @staticmethod
    def execution_command() -> PhysicalExecutionCommand:
        shards = tuple(
            replace(row, session_generation=1)
            for row in (shard(0, 0b0011), shard(1, 0b1100))
        )
        contract = RuntimeExecutionContract(
            execution_mode="adaptive-split",
            initial_split_fraction_ppm=0,
            allowed_adaptive_fractions_ppm=(0, 500_000, 1_000_000),
            batch_plan="coalesced-batch",
            maximum_batch_size=4,
            queue_depth=4,
            phone_device_id="phone-a",
            phone_endpoint="physical://phone-a",
            operator_kind="ffn",
            phone_shards=shards,
        )
        transition = RuntimeTransitionPlan(
            transition_id="load-phone-shards",
            device_id="phone-a",
            source_state="cold",
            target_state="hot",
            latency_us=10,
            energy_uj=20,
            resource_ids=("phone-functionfs", "phone-htp", "phone-usb"),
            maturity="QUALIFIED",
            prepares_device_ids=("phone-a",),
            phone_shards=shards,
        )
        participant = PhysicalParticipantCommand(
            executor_id="phone-executor",
            device_id="phone-a",
            endpoint="physical://phone-a",
            backend="phone",
            resource_ids=("phone-functionfs", "phone-htp", "phone-usb"),
        )
        plan_hash = SHA_A
        parameters = {
            "active_request_batch_size": 1,
            "ffn_assistance_phase": "decode",
            "ffn_max_tokens": 4,
            "ffn_resident_columns": 8,
            "ffn_resident_layer_mask": 0b1111,
            "ffn_runtime_control_protocol": "decode-boundary-v1",
            "phone_device_id": "phone-a",
            "phone_session_count": 2,
            "usb_batch_plan": "coalesced-batch",
            "usb_queue_depth": 4,
        }
        transition_command = PhysicalTransitionCommand(
            ticket_id="ticket-a",
            request_id="request-a",
            artifact_sha256=SHA_A,
            route_id="route-a",
            operator_plan_sha256=plan_hash,
            participant=participant,
            transition=transition,
            execution_contract=contract,
            adapter_parameters=parameters,
            phone_layout_generation=1,
            operator_plan_protocol="synthetic-v1",
            operator_plan={},
        )
        operator_plan = {
            "assisted_operator_kind": "ffn",
            "execution_contract": contract.to_json(),
            "plan_sha256": plan_hash,
        }
        return PhysicalExecutionCommand(
            ticket_id="ticket-a",
            request_id="request-a",
            model_id="synthetic-session-model",
            artifact_sha256=SHA_A,
            route_id="route-a",
            executor_id="coordinator-a",
            endpoint="synthetic://coordinator-a",
            operator_plan_protocol="synthetic-v1",
            operator_plan_sha256=plan_hash,
            planned_start_us=1,
            planned_finish_us=10,
            planned_finish_upper_us=20,
            operator_plan=operator_plan,
            participants=(participant,),
            leases=(),
            memory_reservations=tuple(
                {
                    "demand_id": f"phone-session:{row.session_id}:weights",
                    "reserved_bytes": row.resident_bytes,
                }
                for row in shards
            ),
            transitions=(transition_command,),
            execution_contract=contract,
            adapter_parameters=parameters,
            phone_layout_generation=1,
        )

    def test_ticket_and_transition_bind_the_exact_shard_set(self) -> None:
        command = self.execution_command()
        validate_physical_execution_command(command)

        changed = replace(
            command.transitions[0].transition,
            phone_shards=(
                command.execution_contract.phone_shards[0],
                replace(
                    command.execution_contract.phone_shards[1],
                    operator_plan_sha256=SHA_C,
                ),
            ),
        )
        with self.assertRaisesRegex(
            PhysicalAdapterError, "transition differs"
        ):
            validate_physical_execution_command(replace(
                command,
                transitions=(replace(
                    command.transitions[0], transition=changed
                ),),
            ))

    def test_physical_phone_command_rejects_generation_zero(self) -> None:
        command = self.execution_command()
        zero_shards = tuple(
            replace(row, session_generation=0)
            for row in command.execution_contract.phone_shards
        )
        contract = replace(
            command.execution_contract, phone_shards=zero_shards
        )
        transition = replace(
            command.transitions[0].transition,
            phone_shards=zero_shards,
        )
        transition_command = replace(
            command.transitions[0],
            transition=transition,
            execution_contract=contract,
        )
        operator_plan = {
            **command.operator_plan,
            "execution_contract": contract.to_json(),
        }
        with self.assertRaisesRegex(
            PhysicalAdapterError, "session generation zero"
        ):
            validate_physical_execution_command(replace(
                command,
                execution_contract=contract,
                transitions=(transition_command,),
                operator_plan=operator_plan,
            ))

    def test_per_session_physical_proof_requires_calls_for_every_shard(self) -> None:
        command = self.execution_command()
        rows = []
        for index, placed in enumerate(command.execution_contract.phone_shards):
            rows.append({
                "session_id": placed.session_id,
                "endpoint_sha256": "sha256:" + hashlib.sha256(
                    placed.endpoint.encode("ascii")
                ).hexdigest(),
                "resident_geometry_sha256": placed.resident_geometry_sha256,
                "operator_plan_sha256": placed.operator_plan_sha256,
                "layer_mask": placed.layer_mask,
                "calls": index + 1,
                "rows": index + 1,
                "h2d_bytes": 16,
                "d2h_bytes": 16,
                "h2d_us": 1,
                "d2h_us": 1,
                "compute_us": 1,
                "rpc_us": 1,
            })
        terminal = parse_direct_phone_ffn_terminal((
            "MULTIPHONEFFN " + json.dumps({
                "transport": "session-router",
                "requests": 3,
                "queue_depth": 4,
                "maximum_pending_outputs": 0,
                "phone_payload_copies": 12,
                "d2h_completions": 3,
                "d2h_queue_us": 0,
                "d2h_queue_max_us": 0,
                "recoveries": 0,
                "status": 0,
                "host_sessions": 1,
                "shards": rows,
            }, sort_keys=True),
        ))
        DirectPhoneFfnSession._validate_shard_terminal(
            terminal, command.execution_contract.phone_shards
        )

        rows[1]["calls"] = 0
        rows[1]["rows"] = 0
        rows[0]["calls"] = 3
        invalid = parse_direct_phone_ffn_terminal((
            "MULTIPHONEFFN " + json.dumps({
                "transport": "session-router",
                "requests": 3,
                "queue_depth": 4,
                "maximum_pending_outputs": 0,
                "phone_payload_copies": 12,
                "d2h_completions": 3,
                "d2h_queue_us": 0,
                "d2h_queue_max_us": 0,
                "recoveries": 0,
                "status": 0,
                "host_sessions": 1,
                "shards": rows,
            }, sort_keys=True),
        ))
        with self.assertRaisesRegex(
            PhysicalAdapterError, "execution differs"
        ):
            DirectPhoneFfnSession._validate_shard_terminal(
                invalid, command.execution_contract.phone_shards
            )
        DirectPhoneFfnSession._validate_shard_terminal(
            invalid,
            command.execution_contract.phone_shards,
            require_execution=False,
        )

    def test_prepared_shard_is_not_call_required_until_execution(self) -> None:
        executed = replace(shard(0, 0b0011), artifact_sha256=SHA_A)
        dormant = replace(shard(1, 0b1100), artifact_sha256=SHA_B)
        rows = []
        for placed, calls in ((executed, 2), (dormant, 0)):
            rows.append({
                "session_id": placed.session_id,
                "endpoint_sha256": "sha256:" + hashlib.sha256(
                    placed.endpoint.encode("ascii")
                ).hexdigest(),
                "artifact_sha256": placed.artifact_sha256,
                "resident_geometry_sha256": (
                    placed.resident_geometry_sha256
                ),
                "operator_plan_sha256": placed.operator_plan_sha256,
                "layer_mask": placed.layer_mask,
                "calls": calls,
                "rows": calls,
                "h2d_bytes": calls * 16,
                "d2h_bytes": calls * 16,
                "h2d_us": calls,
                "d2h_us": calls,
                "compute_us": calls,
                "rpc_us": calls,
            })
        terminal = parse_direct_phone_ffn_terminal((
            "MULTIPHONEFFN " + json.dumps({
                "transport": "session-router",
                "requests": 2,
                "queue_depth": 4,
                "maximum_pending_outputs": 0,
                "phone_payload_copies": 8,
                "d2h_completions": 2,
                "d2h_queue_us": 0,
                "d2h_queue_max_us": 0,
                "recoveries": 0,
                "status": 0,
                "host_sessions": 1,
                "shards": rows,
            }, sort_keys=True),
        ))

        DirectPhoneFfnSession._validate_shard_terminal(
            terminal,
            (executed, dormant),
            executed_shards=(executed,),
        )
        with self.assertRaisesRegex(
            PhysicalAdapterError, "execution differs"
        ):
            DirectPhoneFfnSession._validate_shard_terminal(
                terminal,
                (executed, dormant),
                executed_shards=(executed, dormant),
            )

        session = object.__new__(DirectPhoneFfnSession)
        session._launch = SimpleNamespace(phone_shards=(executed, dormant))
        session._remote_root = "/data/local/tmp/resident"
        session._proof_shards = [executed, dormant]
        session._executed_proof_shards = []
        session._bound_ticket_ids = ["transition-ticket"]
        session.record_execution_proof(
            "execution-ticket",
            SHA_A,
            (LlamaServerPhoneSessionProof(
                session_id=executed.session_id,
                endpoint=executed.endpoint,
                artifact_sha256=SHA_A,
                resident_geometry_sha256=(
                    executed.resident_geometry_sha256
                ),
                operator_plan_sha256=executed.operator_plan_sha256,
                session_generation=executed.session_generation,
                layer_mask=executed.layer_mask,
                calls=2,
                rows=2,
                payload_bytes=32,
            ),),
        )
        self.assertEqual(session._executed_proof_shards, [executed])
        self.assertEqual(
            session._bound_ticket_ids,
            ["transition-ticket", "execution-ticket"],
        )

    def test_restored_unexecuted_epoch_needs_no_terminal_proof(self) -> None:
        retained = replace(
            shard(1, 0b1100),
            artifact_sha256=SHA_A,
            session_generation=1,
        )
        historical = replace(
            shard(0, 0b0011),
            artifact_sha256=SHA_A,
            session_generation=1,
        )
        restored = replace(historical, session_generation=3)
        rows = []
        for placed, calls in ((retained, 2), (historical, 3)):
            rows.append({
                "session_id": placed.session_id,
                "endpoint_sha256": "sha256:" + hashlib.sha256(
                    placed.endpoint.encode("ascii")
                ).hexdigest(),
                "artifact_sha256": placed.artifact_sha256,
                "resident_geometry_sha256": (
                    placed.resident_geometry_sha256
                ),
                "operator_plan_sha256": placed.operator_plan_sha256,
                "layer_mask": placed.layer_mask,
                "calls": calls,
                "rows": calls,
                "h2d_bytes": calls * 16,
                "d2h_bytes": calls * 16,
                "h2d_us": calls,
                "d2h_us": calls,
                "compute_us": calls,
                "rpc_us": calls,
                "session_generation": placed.session_generation,
            })
        terminal = parse_direct_phone_ffn_terminal((
            "MULTIPHONEFFN " + json.dumps({
                "transport": "session-router",
                "requests": 5,
                "queue_depth": 4,
                "maximum_pending_outputs": 0,
                "phone_payload_copies": 20,
                "d2h_completions": 5,
                "d2h_queue_us": 0,
                "d2h_queue_max_us": 0,
                "recoveries": 0,
                "status": 0,
                "host_sessions": 1,
                "shards": rows,
            }, sort_keys=True),
        ))

        DirectPhoneFfnSession._validate_shard_terminal(
            terminal,
            (restored, retained),
            executed_shards=(retained,),
            historical_shards=(historical,),
        )
        with self.assertRaisesRegex(
            PhysicalAdapterError, "execution proofs are incomplete"
        ):
            DirectPhoneFfnSession._validate_shard_terminal(
                terminal,
                (restored, retained),
                executed_shards=(restored, retained),
                historical_shards=(historical,),
            )

    def test_router_h2d_idle_wait_is_not_transfer_evidence(self) -> None:
        command = self.execution_command()
        rows = []
        for shard in command.execution_contract.phone_shards:
            rows.append({
                "session_id": shard.session_id,
                "endpoint_sha256": "sha256:" + hashlib.sha256(
                    shard.endpoint.encode("ascii")
                ).hexdigest(),
                "resident_geometry_sha256": (
                    shard.resident_geometry_sha256
                ),
                "operator_plan_sha256": shard.operator_plan_sha256,
                "layer_mask": shard.layer_mask,
                "calls": 1,
                "rows": 1,
                "h2d_bytes": 16,
                "d2h_bytes": 16,
                "h2d_us": 2,
                "h2d_active_us": 2,
                "idle_receive_wait_us": 1_000_000,
                "h2d_timing_scope": "payload-ready-read-v1",
                "d2h_us": 1,
                "compute_us": 1,
                "rpc_us": 1,
            })
        terminal = parse_direct_phone_ffn_terminal((
            "MULTIPHONEFFN " + json.dumps({
                "transport": "session-router",
                "requests": len(rows),
                "queue_depth": 4,
                "maximum_pending_outputs": 0,
                "phone_payload_copies": len(rows) * 4,
                "d2h_completions": len(rows),
                "d2h_queue_us": 0,
                "d2h_queue_max_us": 0,
                "recoveries": 0,
                "status": 0,
                "host_sessions": 1,
                "shards": rows,
            }, sort_keys=True),
        ))

        self.assertTrue(all(
            not row.h2d_estimator_eligible
            and row.h2d_us == 2
            and row.idle_receive_wait_us == 1_000_000
            for row in terminal.session_proofs
        ))
        legacy = replace(
            terminal.session_proofs[0],
            h2d_active_us=None,
            idle_receive_wait_us=None,
            h2d_timing_scope="legacy-idle-inclusive",
        )
        self.assertFalse(legacy.h2d_estimator_eligible)
        wait_only = replace(
            terminal.session_proofs[0],
            h2d_us=0,
            h2d_active_us=0,
            h2d_timing_scope="host-completion-required-v2",
        )
        self.assertFalse(wait_only.h2d_estimator_eligible)

    def test_contract_rejects_overlapping_shards(self) -> None:
        with self.assertRaisesRegex(RuntimePlanError, "layers overlap"):
            RuntimeExecutionContract(
                execution_mode="adaptive-split",
                initial_split_fraction_ppm=0,
                allowed_adaptive_fractions_ppm=(0, 1_000_000),
                batch_plan="coalesced-batch",
                maximum_batch_size=4,
                queue_depth=4,
                phone_device_id="phone-a",
                phone_endpoint="physical://phone-a",
                operator_kind="ffn",
                phone_shards=(shard(0, 0b0011), shard(1, 0b0010)),
            )

    def test_energy_aware_request_executes_qualified_discovered_shards(
        self,
    ) -> None:
        base_model = manifest()
        tensors = tuple(
            replace(
                row,
                shape=(4_096, 8_192),
                nbytes=4_096 * 8_192 * 2,
            )
            for row in base_model.tensors
        )
        model = replace(
            base_model,
            artifact_bytes=sum(row.nbytes for row in tensors),
            embedding_length=4_096,
            feed_forward_length=8_192,
            tensors=tensors,
        )
        layer_bytes = tensors[0].nbytes
        session_limit = 2 * layer_bytes
        cold_sessions = tuple(
            RuntimePhoneSessionCapability(
                session_id=f"HTP{index}",
                device_id="helper-c",
                endpoint=f"session://helper-c/HTP{index}",
                worker_identity_sha256=SHA_B,
                memory_resource_id=f"phone-session-{index}",
                resident_memory_limit_bytes=session_limit,
                shared_compute_resource_id="compute:helper-c",
                shared_transport_resource_ids=(
                    "link:usb-in", "link:usb-out"
                ),
                supported_layer_mask=(1 << model.block_count) - 1,
                maximum_columns=model.feed_forward_length,
                column_quantum=512,
                supported_data_types=("F16",),
                batch_plans=("coalesced-batch", "split-row"),
                ready=True,
                residency_state="cold",
            )
            for index in range(3)
        )
        sessions = cold_sessions
        source = catalog(
            phone_ops_per_s=1_000_000_000_000,
            phone_power_mw=500,
            phone_bandwidth=8_000_000_000,
            phone_whole_model=False,
            gpu_whole_model=False,
        )
        pools = dict(source.placement_profile.memory_pools)
        for row in sessions:
            pools[row.memory_resource_id] = MemoryPoolProfile(
                row.memory_resource_id,
                row.resident_memory_limit_bytes,
                0,
            )
        helper = replace(
            source.executor_by_device["helper-c"],
            execution_resource_ids=(
                "compute:helper-c", "link:usb-in", "link:usb-out"
            ),
            exclusive_residency_resource_id="compute:helper-c",
            phone_sessions=sessions,
        )
        placements = tuple(
            RuntimeCompositeOperatorPlacement(
                operator_id=row.operator_id,
                primary_device_id="accelerator-b",
                helper_device_id=None,
                split_axis="none",
                split_fraction_ppm=0,
            )
            for row in model.operators
        )
        desktop = RuntimeCompositeExecutorCapability(
            executor_id="coordinator:desktop",
            endpoint="synthetic://desktop",
            backend="backend:desktop",
            coordinator_device_id="accelerator-b",
            participant_device_ids=("accelerator-b",),
            participant_resource_ids={
                "accelerator-b": ("compute:accelerator-b",),
            },
            route_family="layer_placement",
            assisted_operator_kind=None,
            split_axis="none",
            split_fractions_ppm=(),
            layer_fractions_ppm=(),
            residency_states=("hot",),
            resource_ids=("compute:accelerator-b",),
            operator_plan_protocol="synthetic-plan-v1",
            maturity="QUALIFIED",
            evidence_ids=("synthetic-desktop",),
            artifact_sha256=model.artifact_sha256,
            operator_placements=placements,
        )
        phone = RuntimeCompositeExecutorCapability(
            executor_id="coordinator:phone",
            endpoint="synthetic://phone",
            backend="backend:phone",
            coordinator_device_id="accelerator-b",
            participant_device_ids=("accelerator-b", "helper-c"),
            participant_resource_ids={
                "accelerator-b": ("compute:accelerator-b",),
                "helper-c": (
                    "compute:helper-c", "link:usb-in", "link:usb-out"
                ),
            },
            route_family="operator_split",
            assisted_operator_kind="ffn",
            split_axis="column",
            split_fractions_ppm=(250_000, 500_000, 750_000),
            layer_fractions_ppm=(),
            residency_states=("cold", "hot", "warm"),
            resource_ids=(
                "compute:accelerator-b",
                "compute:helper-c",
                "link:usb-in",
                "link:usb-out",
            ),
            operator_plan_protocol="synthetic-plan-v1",
            maturity="QUALIFIED",
            evidence_ids=("synthetic-phone",),
            artifact_sha256=model.artifact_sha256,
            operator_ids=tuple(row.operator_id for row in model.operators),
            baseline_executor_id=desktop.executor_id,
            helper_device_id="helper-c",
            adapter_parameters={
                "ffn_activation": "geglu",
                "ffn_column_quantum": 1,
                "ffn_max_tokens": 4,
                "ffn_n_embd": model.embedding_length,
                "ffn_runtime_control_protocol": "decode-boundary-v1",
                "ffn_timeout_ms": 5000,
                "ffn_transport": "functionfs-usb",
                "ffn_weight_buffer_layout": "selected-width",
                "maximum_helper_resident_weight_bytes": (
                    session_limit
                ),
                "phone_device_id": "helper-c",
                "usb_allocator": "devmem",
                "usb_batch_plan": "coalesced-batch",
                "usb_full_duplex": 1,
                "usb_max_payload_bytes": 65536,
                "usb_product_id": 0x5678,
                "usb_queue_depth": 4,
                "usb_slot_safety_bytes": 64,
                "usb_split_h2d": 0,
                "usb_transport_generation": "synthetic-functionfs-v1",
                "usb_transport_profile_id": (
                    "synthetic-functionfs-profile-v1"
                ),
                "usb_vendor_id": 0x1234,
                "usbfs_available_bytes": 262144,
            },
            replacement_group_by_device={
                "helper-c": "compute:helper-c",
            },
        )
        profile = replace(
            source,
            placement_profile=replace(
                source.placement_profile,
                memory_pools=MappingProxyType(pools),
            ),
            executors=tuple(
                helper if row.device_id == "helper-c" else row
                for row in source.executors
            ),
            composite_executors=(desktop, phone),
            desktop_control_profiles=(RuntimeDesktopControlProfile(
                profile_id="synthetic-desktop-control",
                artifact_sha256=model.artifact_sha256,
                executor_id=desktop.executor_id,
                operator_placements=placements,
                maturity="QUALIFIED",
                evidence_ids=("synthetic-desktop",),
            ),),
        )
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        scheduler.register_model_manifest(model)
        snapshot = runtime_snapshot(
            model, phone_bandwidth=8_000_000_000
        )
        capacities = dict(snapshot.memory.capacities)
        for row in sessions:
            capacities[row.memory_resource_id] = DeviceMemoryCapacity(
                row.memory_resource_id,
                row.resident_memory_limit_bytes,
                0,
                0,
            )
        snapshot = replace(
            snapshot,
            memory=replace(snapshot.memory, capacities=capacities),
            executors={
                **snapshot.executors,
                desktop.executor_id: executor_state(desktop.executor_id),
                phone.executor_id: executor_state(phone.executor_id),
            },
        )
        value = request(
            "multi-session-adaptive", input_tokens=8, output_tokens=30
        )
        cold_helper_state = replace(
            snapshot.executors["executor:helper-c"],
            healthy=False,
            ready=False,
            free_slots=0,
        )
        cold_snapshot = replace(
            snapshot,
            executors={
                **snapshot.executors,
                "executor:helper-c": cold_helper_state,
            },
        )
        cold_demand = scheduler._model_demand_snapshot(
            value, model, cold_snapshot, value.arrival_us
        )
        self.assertEqual(
            cold_demand.available_session_ids,
            ("HTP0", "HTP1", "HTP2"),
        )
        compiler = scheduler._automated_compiler()
        first_envelopes = compiler._ffn_resident_envelopes(
            model, phone, model.request_work(8, 30)
        )
        second_envelopes = compiler._ffn_resident_envelopes(
            model, phone, model.request_work(16, 60)
        )
        self.assertEqual(
            tuple(row.session_count for row in first_envelopes),
            (1, 2, 3),
        )
        self.assertIs(second_envelopes, first_envelopes)
        compiler._ffn_resident_envelope_cache.clear()
        compiler._pattern_cache.clear()
        self.assertIsNone(compiler.phone_residency_demand(model, 30))
        candidates = scheduler.generate_automated_candidates(
            value, model.model_id, snapshot
        )
        self.assertEqual(
            compiler.phone_residency_evidence_status(
                model.artifact_sha256
            )["reason"],
            "PHONE_RESIDENCY_ROUTE_EVIDENCE_UNUSABLE",
        )
        session_counts = {
            len(row.plan.execution_contract.phone_shards)
            for row in candidates.candidates
            if row.plan.execution_contract.phone_shards
        }
        self.assertEqual(session_counts, {1, 2, 3})
        self.assertEqual({
            row.plan.adapter_parameters[
                "phone_shard_packing_value_kind"
            ]
            for row in candidates.candidates
            if row.plan.execution_contract.phone_shards
        }, {"rough_compute_ops"})

        assisted = max(
            (
                row for row in candidates.candidates
                if row.plan.execution_contract.phone_shards
            ),
            key=lambda row: len(
                row.plan.execution_contract.phone_shards
            ),
        )
        baseline = candidates.baseline

        def learned_cost(cost, lower, center, upper):
            transition_lower = cost.transition_energy_lower_uj or 0
            transition = cost.transition_energy_uj or 0
            transition_upper = cost.transition_energy_upper_uj or 0
            return replace(
                cost,
                energy_evidence="MEASURED",
                latency_evidence="MEASURED",
                fleet_energy_lower_uj=lower + transition_lower,
                fleet_energy_uj=center + transition,
                fleet_energy_upper_uj=upper + transition_upper,
                warm_execution_energy_lower_uj=lower,
                warm_execution_energy_uj=center,
                warm_execution_energy_upper_uj=upper,
            )

        measured_baseline = replace(
            baseline,
            cost=learned_cost(
                baseline.cost, 1_000_000, 1_050_000, 1_100_000
            ),
        )
        measured_assisted = replace(
            assisted,
            admitted=False,
            binding=replace(
                assisted.binding,
                ready=False,
                eligibility_reasons=(
                    "BATTERY_LIMIT",
                    "MODEL_EPOCH_AUDIT_ONLY",
                    "THERMAL_LIMIT",
                ),
            ),
            cost=learned_cost(
                assisted.cost, 600_000, 650_000, 700_000
            ),
            maturity="QUALIFIED",
            rejection_reasons=(
                "BATTERY_LIMIT",
                "MODEL_EPOCH_AUDIT_ONLY",
                "THERMAL_LIMIT",
            ),
            residency_break_even={
                **dict(assisted.residency_break_even or {}),
                "paired_energy_evidence": "MEASURED",
                "paired_transition_energy_delta_lower_uj": 30_000,
                "paired_transition_energy_delta_uj": 60_000,
                "paired_transition_energy_delta_upper_uj": 90_000,
                "paired_warm_energy_delta_lower_uj": -500_000,
                "paired_warm_energy_delta_uj": -400_000,
                "paired_warm_energy_delta_upper_uj": -300_000,
                "paired_warm_energy_delta_lower_per_decode_token_uj": (
                    -16_667
                ),
                "paired_warm_energy_delta_per_decode_token_uj": -13_334,
                "paired_warm_energy_delta_upper_per_decode_token_uj": (
                    -10_000
                ),
            },
        )
        measured_set = replace(
            candidates,
            candidates=tuple(
                measured_baseline
                if row.candidate_id == baseline.candidate_id else
                measured_assisted
                if row.candidate_id == assisted.candidate_id else row
                for row in candidates.candidates
            ),
        )
        evidence = compiler.capture_phone_residency_route_evidence(
            model, measured_set, 30
        )
        self.assertEqual(
            evidence["reason"], "PHONE_RESIDENCY_ROUTE_EVIDENCE_READY"
        )
        self.assertEqual(evidence["route_benefit_uj"], 300_000)
        self.assertEqual(evidence["normalized_benefit_uj"], 10_000)
        self.assertEqual(
            evidence["benefit_normalization_source"],
            "paired_warm_energy_delta_per_decode_token",
        )
        longer_evidence = compiler.capture_phone_residency_route_evidence(
            model, measured_set, 60
        )
        self.assertEqual(longer_evidence["route_benefit_uj"], 600_000)
        self.assertEqual(
            longer_evidence["normalized_benefit_uj"], 10_000
        )
        self.assertEqual(
            evidence["residency_pressure_unit"],
            "remaining_decode_token",
        )
        demand = compiler.phone_residency_demand(model, 30)
        self.assertIsNotNone(demand)
        self.assertEqual(
            demand[0].benefit_value_kind, "measured_net_energy_uj"
        )
        self.assertEqual(
            sum(demand[0].benefit_by_operator.values()), 10_000
        )
        self.assertEqual(
            sum(demand[0].transition_energy_uj_by_session.values()),
            90_000,
        )
        layouts = generate_mixed_ffn_residency_layouts(
            (demand[0],),
            demand[1],
            phone_wide_limit_bytes=sum(
                row.resident_memory_limit_bytes for row in demand[1]
            ),
        )
        selected_layout, selection_reason = (
            select_mixed_ffn_residency_layout(
                layouts,
                current_geometry_sha256=None,
                switching_margin_uj=0,
            )
        )
        self.assertIsNotNone(selected_layout)
        self.assertEqual(
            selection_reason, "PHONE_RESIDENCY_QUEUE_BENEFIT"
        )
        self.assertEqual(
            assisted.plan.adapter_parameters[
                "phone_shard_set_geometry_sha256"
            ],
            selected_layout.geometry_sha256,
        )
        layout_state = (
            scheduler._model_placement_controller.propose_phone_layout(
                selected_layout,
                workspace_bytes=0,
                shared_compute_resource_id=(
                    demand[1][0].shared_compute_resource_id
                ),
                shared_transport_resource_ids=(
                    demand[1][0].shared_transport_resource_ids
                ),
                observed_at_us=0,
                selection_reason=selection_reason,
                queue_work_by_artifact={
                    model.artifact_sha256: demand[0].queued_work,
                },
                queue_benefit_uj=selected_layout.queue_benefit,
                transition_cost_uj=selected_layout.transition_cost,
                switching_margin_uj=0,
            )
        )
        compiler.set_phone_residency_layout(layout_state.layout)
        authorized_plan = compiler.authorize_phone_residency_plan(
            assisted.plan,
            layout_state.layout,
            model_id=model.model_id,
            artifact_sha256=model.artifact_sha256,
        )
        self.assertEqual(
            {
                row.session_id: row.session_generation
                for row in authorized_plan.execution_contract.phone_shards
            },
            dict(layout_state.layout.session_generation_by_id),
        )
        authorized_transition = next(
            row for row in authorized_plan.transitions
            if row.phone_shards
        )
        self.assertEqual(
            authorized_transition.changed_phone_session_ids,
            layout_state.layout.changed_session_ids,
        )
        self.assertEqual(
            {
                row.session_id: row.session_generation
                for row in authorized_transition.phone_shards
            },
            dict(layout_state.layout.session_generation_by_id),
        )
        layout_state = (
            scheduler._model_placement_controller.propose_phone_layout(
                selected_layout,
                workspace_bytes=0,
                shared_compute_resource_id=(
                    demand[1][0].shared_compute_resource_id
                ),
                shared_transport_resource_ids=(
                    demand[1][0].shared_transport_resource_ids
                ),
                observed_at_us=1,
                selection_reason="PHONE_RESIDENCY_HYSTERESIS",
                queue_work_by_artifact={
                    model.artifact_sha256: demand[0].queued_work,
                },
                queue_benefit_uj=selected_layout.queue_benefit,
                transition_cost_uj=selected_layout.transition_cost,
                switching_margin_uj=0,
            )
        )
        self.assertEqual(
            layout_state.selection_reason,
            "PHONE_RESIDENCY_HYSTERESIS",
        )
        ready_candidates = (
            scheduler._candidate_set_for_ready_phone_layout(
                candidates,
                model.artifact_sha256,
                layout_state,
            )
        )
        self.assertIn(
            assisted.candidate_id,
            {row.candidate_id for row in ready_candidates.candidates},
        )
        cold_audit = replace(
            measured_assisted,
            binding=replace(
                measured_assisted.binding,
                eligibility_reasons=(
                    "COLD_RESIDENCY_BREAK_EVEN",
                    "MODEL_EPOCH_AUDIT_ONLY",
                ),
            ),
            rejection_reasons=(
                "COLD_RESIDENCY_BREAK_EVEN",
                "MODEL_EPOCH_AUDIT_ONLY",
            ),
        )
        authorized = (
            scheduler._apply_phone_residency_portfolio_authorization(
                replace(
                    measured_set,
                    candidates=tuple(
                        cold_audit
                        if row.candidate_id == assisted.candidate_id
                        else row
                        for row in measured_set.candidates
                    ),
                ),
                model,
                value,
            )
        )
        authorized_route = next(
            row for row in authorized.candidates
            if row.candidate_id == assisted.candidate_id
        )
        self.assertNotIn(
            "COLD_RESIDENCY_BREAK_EVEN",
            authorized_route.rejection_reasons,
        )
        self.assertIn(
            "MODEL_EPOCH_AUDIT_ONLY",
            authorized_route.rejection_reasons,
        )
        self.assertEqual(
            sum(selected_layout.queue_benefit_by_artifact.values()),
            selected_layout.queue_benefit,
        )
        authorization = authorized_route.residency_break_even[
            "phone_residency_portfolio_authorization"
        ]
        self.assertEqual(
            authorization["net_benefit_share_uj"],
            authorization["request_queue_benefit_share_uj"]
            - authorization["transition_share_uj"],
        )
        self.assertEqual(authorization["hysteresis_share_uj"], 0)
        self.assertEqual(
            authorization["layout_generation"], layout_state.generation
        )
        self.assertEqual(
            authorization["layout_selection_reason"],
            "PHONE_RESIDENCY_HYSTERESIS",
        )
        self.assertEqual(authorization["layout_state"], "PROPOSED")
        (
            _unavailable_candidates,
            unavailable_selected,
            unavailable_rejected,
            unavailable_reason,
        ) = scheduler._desktop_with_async_phone_helper(
            authorized,
            authorized.baseline,
            (),
            "QUALIFIED_FALLBACK_NO_SAFE_ALTERNATIVE",
            value,
            model,
            "energy-aware",
        )
        self.assertEqual(
            unavailable_reason,
            "READY_DESKTOP_HELPER_RUNTIME_UNAVAILABLE",
        )
        self.assertEqual(unavailable_selected, authorized.baseline)
        self.assertIn(
            "RESIDENT_DESKTOP_HELPER_RUNTIME_UNAVAILABLE",
            dict(unavailable_rejected).values(),
        )
        with patch.object(
            scheduler,
            "_dormant_phone_ffn_runtime_supports",
            return_value=True,
        ):
            (
                helper_candidates,
                helper_selected,
                _,
                helper_reason,
            ) = scheduler._desktop_with_async_phone_helper(
                authorized,
                authorized.baseline,
                (),
                "QUALIFIED_FALLBACK_NO_SAFE_ALTERNATIVE",
                value,
                model,
                "energy-aware",
            )
        self.assertEqual(
            helper_reason, "READY_DESKTOP_WITH_ASYNC_PHONE_HELPER"
        )
        self.assertEqual(
            helper_selected.candidate_id,
            helper_candidates.baseline.candidate_id,
        )
        self.assertIsNotNone(helper_selected.plan.helper_envelope)
        self.assertEqual(
            helper_selected.plan.helper_envelope.phone_layout_generation,
            layout_state.generation,
        )
        self.assertEqual(
            authorized_route.residency_break_even[
                "portfolio_effective_paired_energy_upper_uj"
            ],
            measured_baseline.cost.fleet_energy_lower_uj
            - authorization["net_benefit_share_uj"],
        )
        pending_layout = (
            scheduler._model_placement_controller.target_phone_layout()
        )
        self.assertIsNotNone(pending_layout)
        self.assertEqual(pending_layout.state, "PROPOSED")
        self.assertEqual(
            pending_layout.layout.geometry_sha256,
            selected_layout.geometry_sha256,
        )
        self.assertEqual(
            pending_layout.selection_reason,
            "PHONE_RESIDENCY_HYSTERESIS",
        )
        self.assertEqual(
            dict(pending_layout.queue_work_by_artifact),
            {model.artifact_sha256: demand[0].queued_work},
        )
        assumed_assisted = replace(
            measured_assisted,
            cost=replace(
                measured_assisted.cost,
                energy_evidence="ASSUMED",
            ),
            residency_break_even={
                **dict(measured_assisted.residency_break_even or {}),
                "paired_energy_evidence": "ASSUMED_4P5W",
            },
        )
        assumed_set = replace(
            measured_set,
            candidates=tuple(
                assumed_assisted
                if row.candidate_id == assisted.candidate_id else row
                for row in measured_set.candidates
            ),
        )
        compiler.catalog = replace(
            compiler.catalog,
            phone_power_profiles=(
                RuntimePhonePowerProfile.assumed_4p5w(
                    device_id="helper-c",
                    domain_id="energy:helper-c",
                    allow_assumed_for_scheduling=True,
                ),
            ),
        )
        assumed_evidence = (
            compiler.capture_phone_residency_route_evidence(
                model, assumed_set, 30
            )
        )
        self.assertEqual(
            assumed_evidence["energy_evidence"], "ASSUMED_4P5W"
        )
        source_shards = tuple(
            assumed_assisted.plan.execution_contract.phone_shards
        )
        source_shard = next(
            row for row in source_shards
            if row.layer_mask.bit_count() > 1
        )
        removed_layer = 1 << max(source_shard.layer_indices)
        subset_shards = tuple(
            replace(
                row,
                layer_mask=row.layer_mask ^ removed_layer,
                resident_geometry_sha256=SHA_A,
                operator_plan_sha256=SHA_A,
            )
            if row.session_id == source_shard.session_id else row
            for row in source_shards
        )
        subset_plan = replace(
            assumed_assisted.plan,
            execution_contract=replace(
                assumed_assisted.plan.execution_contract,
                phone_shards=subset_shards,
            ),
            transitions=tuple(
                replace(transition, phone_shards=subset_shards)
                if transition.phone_shards else transition
                for transition in assumed_assisted.plan.transitions
            ),
        )
        subset_candidate = replace(
            assumed_assisted,
            plan=subset_plan,
            binding=replace(
                assumed_assisted.binding,
                operator_plan_sha256=subset_plan.plan_sha256,
            ),
        )
        subset_proof = compiler.phone_residency_subset_evidence(
            model, subset_candidate
        )
        self.assertIsNotNone(subset_proof)
        self.assertEqual(
            subset_proof["source_layer_mask"],
            assumed_evidence["source_layer_mask"],
        )
        self.assertEqual(
            subset_proof["target_layer_mask"],
            assumed_evidence["source_layer_mask"] ^ removed_layer,
        )
        self.assertIsNone(
            compiler.phone_residency_subset_evidence(
                model,
                replace(
                    subset_candidate,
                    binding=replace(
                        subset_candidate.binding,
                        endpoint="physical://different-endpoint",
                    ),
                ),
            )
        )
        assumed_demand = compiler.phone_residency_demand(model, 30)
        self.assertIsNotNone(assumed_demand)
        self.assertEqual(
            assumed_demand[0].benefit_value_kind,
            "assumed_net_energy_uj",
        )

        two_session = next(
            row for row in candidates.candidates
            if len(row.plan.execution_contract.phone_shards) == 2
            and row.assisted_operator_kind == "ffn"
        )
        active_session_ids = {
            row.session_id
            for row in two_session.plan.execution_contract.phone_shards
        }
        foreign = replace(
            next(
                row for row in assisted.plan.execution_contract.phone_shards
                if row.session_id not in active_session_ids
            ),
            artifact_sha256=SHA_C,
        )
        mixed_shards = tuple(sorted(
            (*two_session.plan.execution_contract.phone_shards, foreign),
            key=lambda row: row.session_id,
        ))
        mixed_plan = replace(
            two_session.plan,
            execution_contract=replace(
                two_session.plan.execution_contract,
                phone_shards=mixed_shards,
            ),
            transitions=tuple(
                replace(transition, phone_shards=mixed_shards)
                if transition.phone_shards else transition
                for transition in two_session.plan.transitions
            ),
        )
        mixed_candidate = replace(
            two_session,
            plan=mixed_plan,
            binding=replace(
                two_session.binding,
                operator_plan_sha256=mixed_plan.plan_sha256,
                ready=False,
                eligibility_reasons=("MODEL_EPOCH_AUDIT_ONLY",),
            ),
            cost=learned_cost(
                two_session.cost, 400_000, 450_000, 500_000
            ),
            maturity="QUALIFIED",
            admitted=False,
            rejection_reasons=("MODEL_EPOCH_AUDIT_ONLY",),
            residency_break_even={
                **dict(two_session.residency_break_even or {}),
                "paired_energy_evidence": "MEASURED",
                "paired_transition_energy_delta_lower_uj": 30_000,
                "paired_transition_energy_delta_uj": 60_000,
                "paired_transition_energy_delta_upper_uj": 90_000,
                "paired_warm_energy_delta_lower_uj": -700_000,
                "paired_warm_energy_delta_uj": -650_000,
                "paired_warm_energy_delta_upper_uj": -600_000,
            },
        )
        mixed_set = replace(
            candidates,
            candidates=tuple(
                measured_baseline
                if row.candidate_id == baseline.candidate_id else
                mixed_candidate
                if row.candidate_id == two_session.candidate_id else row
                for row in candidates.candidates
            ),
        )
        mixed_evidence = compiler.capture_phone_residency_route_evidence(
            model, mixed_set, 30
        )
        self.assertEqual(
            mixed_evidence["source_route_id"], mixed_candidate.candidate_id
        )
        self.assertEqual(mixed_evidence["route_benefit_uj"], 600_000)
        self.assertLess(
            mixed_evidence["transition_energy_upper_uj"], 90_000
        )
        self.assertEqual(
            set(
                mixed_evidence[
                    "transition_energy_upper_uj_by_session"
                ]
            ),
            active_session_ids,
        )
        compiler.capture_phone_residency_route_evidence(
            model, assumed_set, 30
        )
        compiler.catalog = profile

        degraded_sessions = (
            sessions[0],
            sessions[1],
            replace(
                sessions[2],
                ready=False,
                unavailable_reason="health_probe_failed",
            ),
        )
        degraded_helper = replace(
            helper,
            phone_sessions=degraded_sessions,
        )
        degraded_profile = replace(
            profile,
            executors=tuple(
                degraded_helper if row.device_id == "helper-c" else row
                for row in profile.executors
            ),
        )
        degraded_scheduler = UnifiedScheduler.for_runtime_discovery(
            "enforce"
        )
        degraded_scheduler.register_runtime_capabilities(degraded_profile)
        degraded_scheduler.register_model_manifest(model)
        degraded_candidates = (
            degraded_scheduler.generate_automated_candidates(
                replace(value, request_id="multi-session-degraded"),
                model.model_id,
                snapshot,
            )
        )
        degraded_rows = tuple(
            row for row in degraded_candidates.candidates
            if row.plan.execution_contract.phone_shards
        )
        self.assertEqual(
            {
                len(row.plan.execution_contract.phone_shards)
                for row in degraded_rows
            },
            {1, 2, 3},
        )
        unavailable = next(
            row for row in degraded_rows
            if len(row.plan.execution_contract.phone_shards) == 3
        )
        self.assertIn(
            "PHONE_SESSION_NOT_READY:HTP2:HEALTH_PROBE_FAILED",
            unavailable.rejection_reasons,
        )
        self.assertEqual(
            unavailable.primary_rejection_reason,
            "PHONE_SESSION_NOT_READY:HTP2:HEALTH_PROBE_FAILED",
        )
        self.assertFalse(unavailable.admitted)
        degraded_two = next(
            row for row in degraded_rows
            if len(row.plan.execution_contract.phone_shards) == 2
            and "SEARCH_COVERAGE_ONLY" not in row.rejection_reasons
        )
        degraded_identity = degraded_two.plan.adapter_parameters[
            "resident_model_identity_sha256"
        ]
        degraded_live = model_residency_observations(
            degraded_profile,
            model,
            degraded_two.binding.executor_id,
            generation=3,
            operator_plan=degraded_two.plan.to_json(),
        )
        degraded_hot_snapshot = replace(
            snapshot,
            snapshot_id="multi-session-degraded-hot",
            residency=tuple(
                replace(
                    row,
                    resident_adapter_parameters=(
                        helper_selected.plan.adapter_parameters
                    ),
                )
                if row.device_id == "accelerator-b" else row
                for row in snapshot.residency
                if row.device_id != "helper-c"
            ) + tuple(
                row for row in degraded_live
                if row.device_id == "helper-c"
            ),
        )
        degraded_hot_candidates = (
            degraded_scheduler.generate_automated_candidates(
                replace(value, request_id="multi-session-degraded-hot"),
                model.model_id,
                degraded_hot_snapshot,
            )
        )
        degraded_hot_two = next(
            row for row in degraded_hot_candidates.candidates
            if len(row.plan.execution_contract.phone_shards) == 2
            and row.plan.adapter_parameters.get(
                "resident_model_identity_sha256"
            ) == degraded_identity
            and row.split_fraction_ppm == degraded_two.split_fraction_ppm
        )
        degraded_compiler = degraded_scheduler._automated_compiler()
        degraded_energy = max(
            1,
            degraded_hot_two.cost.component_energy_uj // 10,
        )
        degraded_latency = max(
            1,
            degraded_hot_candidates.baseline.cost.service_us // 2,
        )
        degraded_domains = tuple(sorted(
            degraded_profile.placement_profile.idle_charge_domains
        ))
        degraded_energy_by_domain = {
            domain_id: degraded_energy // len(degraded_domains)
            for domain_id in degraded_domains
        }
        degraded_energy_by_domain[degraded_domains[-1]] += (
            degraded_energy - sum(degraded_energy_by_domain.values())
        )
        for index in range(4):
            sample = request(
                "degraded-two-session-sample-" + str(index),
                input_tokens=value.input_tokens,
                output_tokens=value.output_tokens,
            )
            self.assertTrue(degraded_compiler.record_execution_observation(
                sample,
                model,
                degraded_hot_two.plan,
                RuntimeExecutionReceipt(
                    ticket_id="degraded-two-session-ticket-" + str(index),
                    request_id=sample.request_id,
                    artifact_sha256=model.artifact_sha256,
                    operator_plan_sha256=(
                        degraded_hot_two.plan.plan_sha256
                    ),
                    executor_id=degraded_hot_two.binding.executor_id,
                    endpoint=degraded_hot_two.binding.endpoint,
                    operator_plan_protocol=(
                        degraded_hot_two.binding.operator_plan_protocol
                    ),
                    participant_executor_ids=tuple(
                        row.executor_id
                        for row in degraded_hot_two.binding.participants
                    ),
                    started_us=0,
                    finished_us=degraded_latency,
                    output_sha256=SHA_C,
                    status="COMPLETED",
                    energy_boundary_id=(
                        degraded_profile.placement_profile.energy_boundary_id
                    ),
                    fleet_energy_uj_by_domain=(
                        degraded_energy_by_domain
                    ),
                    transfer_energy_uj_by_link={
                        resource_id.removeprefix("link:"): 1
                        for resource_id in (
                            degraded_hot_two.plan.resource_ids
                        )
                        if resource_id.startswith("link:")
                    },
                    measurement_evidence_ids=(
                        "degraded-two-session-isolated-" + str(index),
                    ),
                    energy_attribution_kind="isolated",
                ),
                degraded_hot_snapshot.cost_features,
                degraded_hot_two.cost.component_service_us,
                degraded_hot_two.cost.component_energy_uj,
            ))
        degraded_layout = next(
            row for row in generate_mixed_ffn_residency_layouts(
                (demand[0],),
                degraded_sessions,
                phone_wide_limit_bytes=sum(
                    row.resident_memory_limit_bytes
                    for row in degraded_sessions
                ),
            )
            if len(row.shards) == 2
        )
        degraded_state = (
            degraded_scheduler._model_placement_controller
            .propose_phone_layout(
                degraded_layout,
                workspace_bytes=0,
                shared_compute_resource_id=(
                    degraded_sessions[0].shared_compute_resource_id
                ),
                shared_transport_resource_ids=(
                    degraded_sessions[0]
                        .shared_transport_resource_ids
                ),
                observed_at_us=0,
            )
        )
        degraded_compiler.set_phone_residency_layout(
            degraded_state.layout
        )
        degraded_ticket = degraded_scheduler.submit_automated_request(
            replace(value, request_id="degraded-two-session-selected"),
            model.model_id,
            degraded_hot_snapshot,
            selection_mode="energy-aware",
        )
        self.assertEqual(
            degraded_ticket.execution_plan.execution_contract.execution_mode,
            "desktop",
        )
        self.assertIsNotNone(degraded_ticket.execution_plan.helper_envelope)
        self.assertEqual(
            len(
                degraded_ticket.execution_plan.helper_envelope
                .helper_plan.execution_contract.phone_shards
            ),
            2,
        )
        self.assertNotIn(
            "HTP2",
            {
                shard.session_id for shard in (
                    degraded_ticket.execution_plan.helper_envelope
                    .helper_plan.execution_contract.phone_shards
                )
            },
        )

        target = next(
            row for row in candidates.candidates
            if len(row.plan.execution_contract.phone_shards) == 3
        )
        cold_scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        cold_scheduler.register_runtime_capabilities(profile)
        cold_scheduler.register_model_manifest(model)
        cold_layout = (
            cold_scheduler._model_placement_controller
            .propose_phone_layout(
                selected_layout,
                workspace_bytes=0,
                shared_compute_resource_id=(
                    sessions[0].shared_compute_resource_id
                ),
                shared_transport_resource_ids=(
                    sessions[0].shared_transport_resource_ids
                ),
                observed_at_us=0,
                selection_reason="PHONE_RESIDENCY_QUEUE_BENEFIT",
                queue_work_by_artifact={
                    model.artifact_sha256: demand[0].queued_work,
                },
                queue_benefit_uj=selected_layout.queue_benefit,
                transition_cost_uj=selected_layout.transition_cost,
                switching_margin_uj=0,
            )
        )
        cold_scheduler._automated_compiler().set_phone_residency_layout(
            cold_layout.layout
        )
        helper_runtime_snapshot = replace(
            snapshot,
            residency=tuple(
                replace(
                    row,
                    resident_adapter_parameters=(
                        helper_selected.plan.adapter_parameters
                    ),
                )
                if row.device_id == "accelerator-b" else row
                for row in snapshot.residency
            ),
        )
        cold_ticket = cold_scheduler.submit_automated_request(
            replace(value, request_id="multi-session-cold-projection"),
            model.model_id,
            helper_runtime_snapshot,
            selection_mode="calibration",
        )
        helper = cold_ticket.execution_plan.helper_envelope
        self.assertIsNotNone(helper)
        self.assertEqual(
            cold_ticket.execution_plan.execution_contract.execution_mode,
            "desktop",
        )
        proposed = (
            cold_scheduler._model_placement_controller
            .target_phone_layout()
        )
        self.assertIsNotNone(proposed)
        self.assertEqual(proposed.state, "PROPOSED")
        self.assertEqual(
            helper.phone_layout_generation, proposed.generation
        )
        self.assertEqual(
            helper.phone_layout_geometry_sha256,
            proposed.layout.geometry_sha256,
        )
        self.assertEqual(
            helper.preparation_changed_session_ids,
            proposed.layout.changed_session_ids,
        )
        self.assertEqual(
            {
                row.session_id: row.session_generation
                for row in helper.helper_plan.execution_contract.phone_shards
            },
            dict(proposed.layout.session_generation_by_id),
        )
        self.assertTrue(helper.preparation_transitions)
        self.assertTrue(all(
            row.changed_phone_session_ids
                == proposed.layout.changed_session_ids
            for row in helper.preparation_transitions
        ))
        self.assertIsNone(cold_ticket.residency_projection_token)
        self.assertTrue(
            cold_scheduler._runtime_controller.require_queued_replan(
                cold_ticket.request.request_id,
                "helper-envelope-replan-test",
            )
        )
        wake = cold_scheduler.wait_runtime_request(
            cold_ticket.request.request_id,
            time.monotonic_ns(),
        )
        self.assertEqual(wake.dispatch_state, "REPLAN_REQUIRED")
        cold_ticket = cold_scheduler.replan_automated_request(
            cold_ticket.request.request_id,
            observed_at_us=value.arrival_us,
            reason="helper-envelope-replan-test",
            snapshot=helper_runtime_snapshot,
            expected_ticket_id=wake.ticket_id,
            expected_queue_generation=(
                wake.dispatch_receipt.queue_generation
            ),
        )
        replanned_helper = cold_ticket.execution_plan.helper_envelope
        self.assertIsNotNone(replanned_helper)
        self.assertEqual(
            replanned_helper.phone_layout_generation,
            proposed.generation,
        )
        self.assertEqual(
            replanned_helper.phone_layout_geometry_sha256,
            proposed.layout.geometry_sha256,
        )
        self.assertEqual(
            {
                row.session_id: row.session_generation
                for row in replanned_helper.helper_plan
                    .execution_contract.phone_shards
            },
            dict(proposed.layout.session_generation_by_id),
        )
        structural_command = interpret_runtime_ticket(cold_ticket)
        helper = (
            cold_scheduler.runtime_request_helper_preparation_envelope(
                cold_ticket.request.request_id,
                expected_ticket_id=cold_ticket.ticket_id,
                observed_at_us=value.arrival_us,
                snapshot=helper_runtime_snapshot,
            )
        )
        self.assertIsNotNone(helper)
        self.assertEqual(
            {
                row.session_id: row.session_generation
                for row in helper.helper_plan.execution_contract.phone_shards
            },
            dict(proposed.layout.session_generation_by_id),
        )
        self.assertTrue(all(
            {
                row.session_id: row.session_generation
                for row in transition.phone_shards
            } == dict(proposed.layout.session_generation_by_id)
            for transition in helper.preparation_transitions
        ))
        queued_command = bind_ready_helper_to_physical_command(
            replace(
                structural_command,
                operator_plan=MappingProxyType({
                    key: item
                    for key, item in structural_command.operator_plan.items()
                    if key != "helper_envelope"
                }),
                helper_envelope=None,
                helper_transitions=(),
            ),
            helper,
        )
        preparation = cold_scheduler.begin_request_helper_preparation(
            cold_ticket.request.request_id,
            observed_at_us=value.arrival_us,
            snapshot=helper_runtime_snapshot,
            expected_phone_layout_generation=(
                helper.phone_layout_generation
            ),
            expected_phone_layout_geometry_sha256=(
                helper.phone_layout_geometry_sha256
            ),
            expected_operator_plan_sha256=(
                helper.operator_plan_sha256
            ),
        )
        self.assertEqual(preparation["status"], "OWNER")
        self.assertEqual(
            queued_command.helper_transitions[0].ticket_id,
            preparation["preparation_ticket_id"],
        )
        acquired = cold_scheduler.wait_runtime_request(
            cold_ticket.request.request_id,
            time.monotonic_ns() - cold_ticket.decision.start_us * 1_000,
        )
        self.assertEqual(acquired.dispatch_state, "ACQUIRED")
        command = interpret_runtime_ticket(acquired)
        physical_helper_command = bind_ready_helper_to_physical_command(
            replace(
                command,
                operator_plan=MappingProxyType({
                    key: item for key, item in command.operator_plan.items()
                    if key != "helper_envelope"
                }),
                helper_envelope=None,
                helper_transitions=(),
            ),
            helper,
        )
        self.assertEqual(
            helper.preparation_ticket_id(cold_ticket.ticket_id),
            helper.preparation_ticket_id(acquired.ticket_id),
        )
        self.assertEqual(command.execution_contract.execution_mode, "desktop")
        self.assertEqual(
            len(physical_helper_command.helper_transitions),
            len(helper.preparation_transitions),
        )
        phone_device_id = (
            helper.helper_plan.execution_contract.phone_device_id
        )
        phone_resources = set(next(
            row.resource_ids for row in helper.helper_binding.participants
            if row.device_id == phone_device_id
        ))
        self.assertTrue(physical_helper_command.helper_transitions)
        for transition in physical_helper_command.helper_transitions:
            self.assertEqual(
                transition.transition.prepares_device_ids,
                (phone_device_id,),
            )
            self.assertEqual(transition.transition.device_id, phone_device_id)
            self.assertTrue(
                set(transition.transition.resource_ids) <= phone_resources
            )
            self.assertTrue(transition.helper_only)
        transitioning = (
            cold_scheduler._model_placement_controller
            .preparing_phone_layout()
        )
        self.assertIsNotNone(transitioning)
        cold_scheduler._close_request_helper_runtime(
            acquired.request.request_id,
            acquired.decision.start_us + 1,
            "REQUEST_COMPLETED",
        )
        self.assertEqual(
            cold_scheduler._model_placement_controller
                .preparing_phone_layout().generation,
            transitioning.generation,
        )
        self.assertEqual(
            cold_scheduler._request_helper_preparations[
                preparation["preparation_ticket_id"]
            ].state,
            "TRANSITIONING",
        )
        self.assertEqual(
            cold_scheduler._model_placement_controller
                .request_helper_events(acquired.request.request_id)[-1][
                    "kind"
                ],
            "PREPARATION_CONTINUES_AFTER_REQUEST",
        )
        receipts = tuple(
            transition_receipt_from_observation(
                transition,
                RawTransitionObservation(
                    started_us=acquired.decision.start_us,
                    finished_us=acquired.decision.start_us + 1,
                    status="COMPLETED",
                    evicted_artifact_sha256s=tuple(sorted({
                        row.artifact_sha256
                        for row in transition.transition.evictions
                    })),
                ),
            )
            for transition in physical_helper_command.helper_transitions
        )
        helper_rows = model_residency_observations(
            profile,
            model,
            helper.helper_binding.executor_id,
            generation=2,
            resident_device_ids=("helper-c",),
            operator_plan=helper.helper_plan.to_json(),
        )
        helper_ready_observed_at_us = acquired.decision.start_us + 2
        helper_executor_id = helper.helper_binding.executor_id
        phone_executor_id = profile.executor_by_device[
            phone_device_id
        ].executor_id
        helper_snapshot = replace(
            helper_runtime_snapshot,
            snapshot_id="multi-session-helper-ready",
            captured_at_us=helper_ready_observed_at_us,
            valid_until_us=helper_ready_observed_at_us + 2_500_000,
            executors={
                **snapshot.executors,
                acquired.binding.executor_id: replace(
                    snapshot.executors[acquired.binding.executor_id],
                    ready=False,
                    free_slots=0,
                ),
                helper_executor_id: replace(
                    snapshot.executors[helper_executor_id],
                    ready=False,
                    free_slots=0,
                ),
                phone_executor_id: replace(
                    snapshot.executors[phone_executor_id],
                    battery_ppm=0,
                    temperature_millic=100_000,
                    thermal_qualified=None,
                ),
            },
            residency=tuple(
                row for row in helper_runtime_snapshot.residency
                if row.device_id != "helper-c"
            ) + tuple(
                row for row in helper_rows
                if row.device_id == "helper-c"
            ),
        )
        projected_request_sets = []
        original_snapshot_for_request = (
            cold_scheduler._automated_snapshot_for_request
        )
        original_generate_candidate_set = (
            cold_scheduler._generate_automated_candidate_set
        )

        def capture_snapshot_projection(*args, **kwargs):
            projected_request_sets.append(
                kwargs.get("project_request_ids")
            )
            return original_snapshot_for_request(*args, **kwargs)

        def generate_with_observed_parent(*args, **kwargs):
            candidate_set = original_generate_candidate_set(
                *args, **kwargs
            )
            baseline = candidate_set.baseline
            observed_route_id = baseline.candidate_id + ":observed-hot"
            observed_plan = replace(
                baseline.plan, route_id=observed_route_id
            )
            observed_binding = replace(
                baseline.binding,
                route_id=observed_route_id,
                operator_plan_sha256=observed_plan.plan_sha256,
            )
            observed_baseline = replace(
                baseline,
                candidate_id=observed_route_id,
                plan=observed_plan,
                binding=observed_binding,
            )
            candidates = tuple(
                observed_baseline
                if row.candidate_id == baseline.candidate_id else
                replace(
                    row,
                    paired_baseline_route_id=observed_route_id,
                )
                if row.paired_baseline_route_id == baseline.candidate_id
                else row
                for row in candidate_set.candidates
            )
            metadata = dict(candidate_set.search_metadata)
            for name in (
                "visited_plan_ids", "route_template_live_route_ids"
            ):
                if name in metadata:
                    metadata[name] = tuple(
                        observed_route_id
                        if value == baseline.candidate_id else value
                        for value in metadata[name]
                    )
            return replace(
                candidate_set,
                candidates=candidates,
                baseline_route_id=observed_route_id,
                recovery_fallback_route_id=(
                    observed_route_id
                    if candidate_set.recovery_fallback_route_id
                        == baseline.candidate_id
                    else candidate_set.recovery_fallback_route_id
                ),
                search_metadata=metadata,
            )

        cold_scheduler._request_helper_preparation_envelopes[(
            acquired.request.request_id,
            acquired.ticket_id,
            helper.phone_layout_generation,
        )] = helper
        cold_scheduler._request_helper_opportunities[
            acquired.request.request_id
        ] = ()
        resident_baseline_plan = replace(
            candidates.baseline.plan,
            adapter_parameters=helper_selected.plan.adapter_parameters,
        )
        resident_baseline = replace(
            candidates.baseline,
            plan=resident_baseline_plan,
            binding=replace(
                candidates.baseline.binding,
                operator_plan_sha256=resident_baseline_plan.plan_sha256,
            ),
        )
        resident_candidates = replace(
            candidates,
            candidates=tuple(
                resident_baseline
                if row.candidate_id == candidates.baseline.candidate_id
                else row
                for row in candidates.candidates
            ),
        )
        with patch(
            "research_dev.scheduler._unified.automated_selection_ops.helpers."
            "adaptive_decode_policies",
            return_value=(None, (), None),
        ):
            (
                dormant_candidates,
                dormant_selected,
                _,
                _,
            ) = cold_scheduler._desktop_with_async_phone_helper(
                resident_candidates,
                resident_candidates.baseline,
                (),
                "QUALIFIED_FALLBACK_NO_SAFE_ALTERNATIVE",
                acquired.request,
                model,
                "energy-aware",
            )
        self.assertIsNone(dormant_selected.plan.helper_envelope)
        self.assertTrue(
            cold_scheduler._has_dormant_phone_ffn_runtime(
                dormant_selected.plan
            ),
            (
                helper.desktop_parent_route_id,
                candidates.baseline.candidate_id,
                helper.desktop_placement_sha256,
                candidates.baseline.plan.desktop_placement_sha256,
                helper.helper_plan.baseline_executor_id,
                candidates.baseline.binding.executor_id,
                helper.phone_layout_generation,
                cold_scheduler._model_placement_controller
                    .planning_phone_layout().generation,
                helper.artifact_sha256,
                model.artifact_sha256,
                helper.phone_layout_geometry_sha256,
                cold_scheduler._model_placement_controller
                    .planning_phone_layout().layout.geometry_sha256,
                helper.operator_plan_sha256,
            ),
        )
        self.assertEqual(
            dormant_candidates.baseline.plan.plan_sha256,
            dormant_selected.plan.plan_sha256,
        )
        with patch.object(
            cold_scheduler,
            "_automated_snapshot_for_request",
            side_effect=capture_snapshot_projection,
        ), patch.object(
            cold_scheduler,
            "_generate_automated_candidate_set",
            side_effect=generate_with_observed_parent,
        ):
            cold_scheduler.complete_request_helper_preparation(
                acquired.request.request_id,
                helper.preparation_ticket_id(acquired.ticket_id),
                receipts,
                snapshot=helper_snapshot,
            )
        self.assertEqual(projected_request_sets, [()])
        helper_events = (
            cold_scheduler._model_placement_controller
            .request_helper_events(acquired.request.request_id)
        )
        helper_event_kinds = tuple(
            row["kind"] for row in helper_events
        )
        self.assertIn(
            "HELPER_REMATERIALIZED", helper_event_kinds, helper_events
        )
        self.assertNotIn(
            "HELPER_REMATERIALIZATION_FAILED", helper_event_kinds
        )
        rematerialized = next(
            row for row in helper_events
            if row["kind"] == "HELPER_REMATERIALIZED"
        )
        self.assertNotEqual(
            rematerialized["generated_desktop_parent_route_id"],
            rematerialized["immutable_desktop_parent_route_id"],
        )
        self.assertEqual(
            cold_scheduler._model_placement_controller
                .request_helper_events(acquired.request.request_id)[-1][
                    "observed_at_us"
                ],
            helper_ready_observed_at_us,
        )
        cold_scheduler.start_adaptive_decode(
            acquired.request.request_id,
            slot_id=2,
            first_token_index=0,
            at_us=helper_ready_observed_at_us + 1,
        )
        attach_attempt = cold_scheduler._try_attach_ready_request_helper(
            acquired,
            slot_id=2,
            token_index=4,
            at_us=helper_ready_observed_at_us + 2,
            fraction_ppm=500_000,
        )
        self.assertTrue(attach_attempt.attached, attach_attempt)
        request_binding = (
            cold_scheduler._model_placement_controller.request_binding(
                acquired.request.request_id
            )
        )
        self.assertEqual(
            request_binding["base"]["route_id"], acquired.decision.route_id
        )
        self.assertEqual(
            request_binding["base"]["kv_cache_owner_id"], acquired.ticket_id
        )
        self.assertEqual(request_binding["base"]["server_slot_id"], 2)
        self.assertEqual(request_binding["fraction_ppm"], 500_000)
        runtime_helper = cold_scheduler.runtime_attached_helper_envelope(
            acquired.request.request_id,
            expected_ticket_id=acquired.ticket_id,
        )
        self.assertEqual(
            runtime_helper,
            cold_scheduler._late_request_helper_contexts[
                acquired.request.request_id
            ].helper,
        )
        self.assertEqual(
            runtime_helper.operator_plan_sha256,
            request_binding["helper_attachment"][
                "operator_plan_sha256"
            ],
        )
        original_horizon = min(acquired.final_reserved_until_us.values())
        helper_horizon = cold_scheduler._helper_window_lease_horizon(
            acquired.request.request_id,
            requested_fraction_ppm=500_000,
            at_us=helper_ready_observed_at_us + 2,
        )
        self.assertEqual(
            request_binding["helper_attachment"][
                "lease_reserved_until_us"
            ],
            helper_horizon,
        )
        self.assertLess(helper_horizon, original_horizon)
        self.assertEqual(
            cold_scheduler._runtime_lease_renewal_horizon(acquired),
            helper_horizon,
        )
        renewed_horizon = original_horizon + 10_000_000
        extension = cold_scheduler.extend_runtime_request(
            acquired.request.request_id,
            at_us=helper_ready_observed_at_us + 3,
            reserved_until_us=renewed_horizon,
        )
        self.assertTrue(extension.extended_leases)
        renewed_binding = (
            cold_scheduler._model_placement_controller.request_binding(
                acquired.request.request_id
            )
        )
        self.assertEqual(
            renewed_binding["helper_attachment"][
                "lease_reserved_until_us"
            ],
            max(
                helper_horizon,
                cold_scheduler._helper_window_lease_horizon(
                    acquired.request.request_id,
                    requested_fraction_ppm=500_000,
                    at_us=helper_ready_observed_at_us + 3,
                ),
            ),
        )
        release_at_us = helper_ready_observed_at_us + 4
        cold_scheduler._release_request_helper_leases(
            acquired.request.request_id,
            release_at_us,
        )
        cold_scheduler._release_request_helper_leases(
            acquired.request.request_id,
            release_at_us,
        )
        released_binding = (
            cold_scheduler._model_placement_controller.request_binding(
                acquired.request.request_id
            )
        )
        self.assertEqual(released_binding["fraction_ppm"], 0)
        self.assertEqual(
            released_binding["helper_attachment"]["lease_tokens"], []
        )
        live_rows = model_residency_observations(
            profile,
            model,
            target.binding.executor_id,
            generation=2,
            operator_plan=target.plan.to_json(),
        )
        hot_snapshot = replace(
            helper_runtime_snapshot,
            snapshot_id="multi-session-hot-runtime",
            residency=tuple(
                row for row in helper_runtime_snapshot.residency
                if row.device_id != "helper-c"
            ) + tuple(
                row for row in live_rows
                if row.device_id == "helper-c"
            ),
        )
        hot_candidates = scheduler.generate_automated_candidates(
            replace(value, request_id="multi-session-hot-reuse"),
            model.model_id,
            hot_snapshot,
        )
        resident_identity = target.plan.adapter_parameters[
            "resident_model_identity_sha256"
        ]
        different_shape = scheduler.generate_automated_candidates(
            request(
                "multi-session-different-shape",
                input_tokens=16,
                output_tokens=60,
            ),
            model.model_id,
            snapshot,
        )
        different_shape_target = next(
            row for row in different_shape.candidates
            if len(row.plan.execution_contract.phone_shards) == 3
        )
        self.assertEqual(
            different_shape_target.plan.adapter_parameters[
                "resident_model_identity_sha256"
            ],
            resident_identity,
        )
        hot_target = next(
            row for row in hot_candidates.candidates
            if len(row.plan.execution_contract.phone_shards) == 3
            and row.plan.adapter_parameters.get(
                "resident_model_identity_sha256"
            ) == resident_identity
        )
        self.assertEqual(hot_target.residency_variant, "hot")
        self.assertFalse(any(
            transition.phone_shards
            for transition in hot_target.plan.transitions
        ))
        session_demands = tuple(
            demand for demand in hot_target.plan.memory_demands
            if demand.kind == "session_residency_constraint"
        )
        self.assertEqual(len(session_demands), 3)
        self.assertTrue(all(
            demand.resident_bytes == demand.required_bytes
            for demand in session_demands
        ))
        audit_reason = "MODEL_EPOCH_AUDIT_ONLY"
        templates = compiler.compile_route_template_set(
            candidates,
            candidates.baseline,
            model,
            input_token_bucket=value.input_tokens,
            output_token_bucket=value.output_tokens,
            quality_requirement=value.quality_requirement,
            snapshot=snapshot,
        )
        revalidated = compiler.materialize_route_template_set(
            templates,
            replace(value, request_id="multi-session-audit-hot"),
            model,
            hot_snapshot,
            observed_at_us=value.arrival_us,
            residency_holds={},
            search_metadata={},
        )
        revalidated_target = next(
            row for row in revalidated.candidates
            if len(row.plan.execution_contract.phone_shards) == 3
            and row.plan.adapter_parameters.get(
                "resident_model_identity_sha256"
            ) == resident_identity
        )
        self.assertIn(audit_reason, revalidated_target.rejection_reasons)
        self.assertEqual(
            revalidated_target.paired_baseline_route_id,
            revalidated.baseline_route_id,
        )
        self.assertNotIn("MEMORY_CAPACITY", revalidated_target.rejection_reasons)
        self.assertEqual(revalidated_target.residency_variant, "hot")
        self.assertTrue(all(
            demand.resident_bytes == demand.required_bytes
            for demand in revalidated_target.plan.memory_demands
            if demand.kind in {
                "model_weights", "session_residency_constraint"
            }
            and demand.device_id == "helper-c"
        ))
        self.assertEqual(
            compiler.component_capability_identity(target.plan),
            compiler.component_capability_identity(hot_target.plan),
        )
        baseline_energy = candidates.baseline.cost.fleet_energy_lower_uj
        self.assertIsNotNone(baseline_energy)
        measured_energy = max(
            1, target.cost.component_energy_uj // 10
        )
        measured_latency = max(
            1, candidates.baseline.cost.service_us // 2
        )
        domains = tuple(sorted(
            profile.placement_profile.idle_charge_domains
        ))
        per_domain = measured_energy // len(domains)
        energy_by_domain = {
            domain_id: per_domain for domain_id in domains
        }
        energy_by_domain[domains[-1]] += (
            measured_energy - sum(energy_by_domain.values())
        )
        for index in range(4):
            sample = request(
                "multi-session-sample-" + str(index),
                input_tokens=8,
                output_tokens=30,
            )
            self.assertTrue(compiler.record_execution_observation(
                sample,
                model,
                hot_target.plan,
                RuntimeExecutionReceipt(
                    ticket_id="multi-session-ticket-" + str(index),
                    request_id=sample.request_id,
                    artifact_sha256=model.artifact_sha256,
                    operator_plan_sha256=hot_target.plan.plan_sha256,
                    executor_id=hot_target.binding.executor_id,
                    endpoint=hot_target.binding.endpoint,
                    operator_plan_protocol=(
                        hot_target.binding.operator_plan_protocol
                    ),
                    participant_executor_ids=tuple(
                        row.executor_id
                        for row in hot_target.binding.participants
                    ),
                    started_us=0,
                    finished_us=measured_latency,
                    output_sha256=SHA_C,
                    status="COMPLETED",
                    energy_boundary_id=(
                        profile.placement_profile.energy_boundary_id
                    ),
                    fleet_energy_uj_by_domain=energy_by_domain,
                    transfer_energy_uj_by_link={
                        resource_id.removeprefix("link:"): 1
                        for resource_id in hot_target.plan.resource_ids
                        if resource_id.startswith("link:")
                    },
                    measurement_evidence_ids=(
                        "multi-session-isolated-" + str(index),
                    ),
                    energy_attribution_kind="isolated",
                ),
                hot_snapshot.cost_features,
                hot_target.cost.component_service_us,
                hot_target.cost.component_energy_uj,
            ))

        learned = scheduler.generate_automated_candidates(
            value, model.model_id, snapshot
        )
        qualified = next(
            row for row in learned.candidates
            if row.candidate_id == target.candidate_id
        )
        self.assertEqual(qualified.cost.energy_evidence, "MEASURED")
        self.assertTrue(
            qualified.plan.route_profile_id.startswith("online-template:")
        )
        self.assertEqual(
            qualified.cost.fleet_energy_upper_uj,
            qualified.cost.warm_execution_energy_upper_uj
                + qualified.cost.transition_energy_upper_uj,
        )
        self.assertTrue(qualified.plan.transitions)
        shadow_transition_plan = replace(
            qualified.plan,
            transitions=tuple(
                replace(transition, energy_maturity="SHADOW")
                for transition in qualified.plan.transitions
            ),
        )
        shadow_transition_candidate = replace(
            qualified,
            plan=shadow_transition_plan,
            binding=replace(
                qualified.binding,
                operator_plan_sha256=shadow_transition_plan.plan_sha256,
            ),
        )
        compiler._executor_id_by_plan_sha256[
            shadow_transition_plan.plan_sha256
        ] = qualified.binding.executor_id
        calibration_candidates = replace(
            learned,
            candidates=tuple(
                shadow_transition_candidate
                if row.candidate_id == qualified.candidate_id else row
                for row in learned.candidates
            ),
        )
        calibration_selected, _, _ = (
            scheduler._select_automated_candidate(
                calibration_candidates,
                replace(value, request_id="multi-session-cold-calibration"),
                selection_mode="calibration",
            )
        )
        self.assertEqual(
            len(
                calibration_selected.plan.execution_contract.phone_shards
            ),
            3,
        )
        learned_hot = scheduler.generate_automated_candidates(
            replace(value, request_id="multi-session-hot-qualified"),
            model.model_id,
            hot_snapshot,
        )
        qualified_hot = next(
            row for row in learned_hot.candidates
            if len(row.plan.execution_contract.phone_shards) == 3
            and row.plan.adapter_parameters.get(
                "resident_model_identity_sha256"
            ) == resident_identity
        )
        component_estimate = compiler.observation_store.estimate(
            artifact_sha256=model.artifact_sha256,
            capability_generation_sha256=(
                compiler.route_capability_identity(hot_target.plan)
            ),
            component_capability_generation_sha256=(
                compiler.component_capability_identity(hot_target.plan)
            ),
            plan=hot_target.plan,
            input_tokens=value.input_tokens,
            output_tokens=value.output_tokens,
            quality_requirement=value.quality_requirement,
            cost_features=hot_snapshot.cost_features,
            component_service_us=hot_target.cost.component_service_us,
            component_energy_uj=hot_target.cost.component_energy_uj,
        )
        self.assertIsNotNone(component_estimate)
        self.assertEqual(component_estimate.energy_scope, "route_total")
        self.assertEqual(
            component_estimate.maturity,
            "QUALIFIED",
            component_estimate,
        )
        self.assertEqual(
            qualified_hot.maturity, "QUALIFIED", qualified_hot
        )
        audit_only = replace(
            qualified_hot,
            binding=replace(
                qualified_hot.binding,
                ready=False,
                eligibility_reasons=(audit_reason,),
            ),
            admitted=False,
            rejection_reasons=(audit_reason,),
        )
        audit_candidates = replace(
            learned_hot,
            candidates=tuple(
                audit_only
                if row.candidate_id == qualified_hot.candidate_id
                else row
                for row in learned_hot.candidates
            ),
        )
        strict, _, _ = scheduler._select_automated_candidate(
            audit_candidates,
            value,
            selection_mode="energy-aware",
        )
        prospective, _, _ = scheduler._select_model_placement_candidate(
            audit_candidates,
            value,
            selection_mode="energy-aware",
        )
        self.assertNotEqual(strict.candidate_id, qualified_hot.candidate_id)
        self.assertEqual(
            prospective.candidate_id, qualified_hot.candidate_id
        )
        published_templates = compiler.compile_route_template_set(
            audit_candidates,
            prospective,
            model,
            input_token_bucket=value.input_tokens,
            output_token_bucket=value.output_tokens,
            quality_requirement=value.quality_requirement,
            snapshot=hot_snapshot,
        )
        published_candidates = compiler.materialize_route_template_set(
            published_templates,
            value,
            model,
            hot_snapshot,
            observed_at_us=value.arrival_us,
            residency_holds={},
            search_metadata={},
        )
        published_candidates = scheduler._apply_adaptive_history_costs(
            published_candidates,
            value,
            model,
            hot_snapshot.cost_features,
        )
        published_target = next(
            row for row in published_candidates.candidates
            if row.candidate_id == qualified_hot.candidate_id
        )
        self.assertNotIn(
            audit_reason, published_target.rejection_reasons
        )
        self.assertTrue(published_target.admitted)
        ticket = scheduler.submit_automated_request(
            value,
            model.model_id,
            hot_snapshot,
            selection_mode="energy-aware",
        )
        self.assertEqual(ticket.decision.route_id, candidates.baseline_route_id)
        self.assertEqual(
            ticket.execution_plan.helper_envelope.route_id,
            qualified_hot.candidate_id,
            (qualified_hot, ticket.decision),
        )
        backend = _MeasuredShardBackend(measured_energy)
        adapter = CanonicalPhysicalAdapter(
            scheduler,
            backend,
            epoch_ns=(
                time.monotonic_ns() - ticket.decision.start_us * 1_000
            ),
            snapshot_provider=lambda _ticket, _at_us: hot_snapshot,
            lease_guard_us=50_000,
            lease_quantum_us=100_000,
        )
        result = adapter.execute(ticket, {"prompt_tokens": (1, 2)})
        command = result.command

        self.assertEqual(command.execution_contract.execution_mode, "desktop")
        self.assertIsNotNone(command.helper_envelope)
        self.assertEqual(
            len(
                command.helper_envelope.helper_plan.execution_contract
                .phone_shards
            ),
            len(sessions),
        )
        self.assertEqual(
            command.operator_plan["helper_envelope"],
            command.helper_envelope.to_json(),
        )
        self.assertEqual(
            {
                row.resource_id
                for row in scheduler.runtime_ticket(
                    value.request_id
                ).decision.leases
            },
            set(command.operator_plan["resource_ids"]),
        )
        self.assertEqual(len(backend.execution_commands), 1)
        self.assertEqual(
            backend.execution_commands[0].to_json(), command.to_json()
        )
        self.assertEqual(result.completion.status, "released")
        self.assertEqual(
            result.completion.execution_receipt.operator_plan_sha256,
            command.operator_plan_sha256,
        )

        capped = UnifiedScheduler.for_runtime_discovery(
            "enforce", maximum_phone_sessions=1
        )
        capped.register_runtime_capabilities(profile)
        capped.register_model_manifest(model)
        single_layout = generate_mixed_ffn_residency_layouts(
            (demand[0],),
            sessions[:1],
            phone_wide_limit_bytes=(
                sessions[0].resident_memory_limit_bytes
            ),
        )[0]
        capped_layout = (
            capped._model_placement_controller.propose_phone_layout(
                single_layout,
                workspace_bytes=0,
                shared_compute_resource_id=(
                    sessions[0].shared_compute_resource_id
                ),
                shared_transport_resource_ids=(
                    sessions[0].shared_transport_resource_ids
                ),
                observed_at_us=0,
            )
        )
        capped._automated_compiler().set_phone_residency_layout(
            capped_layout.layout
        )
        capped_ticket = capped.submit_automated_request(
            request(
                "single-session-adaptive",
                input_tokens=8,
                output_tokens=30,
            ),
            model.model_id,
            helper_runtime_snapshot,
            selection_mode="calibration",
        )
        self.assertEqual(
            len(
                capped_ticket.execution_plan.helper_envelope.helper_plan
                .execution_contract.phone_shards
            ),
            1,
        )
        self.assertTrue(any(
            reason == "PHONE_SESSION_LIMIT"
            for _route_id, reason in capped_ticket.decision.rejected
        ))


if __name__ == "__main__":
    unittest.main()
