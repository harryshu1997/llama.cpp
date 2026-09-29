#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import time
import unittest

from research_dev.scheduler import (
    DeviceMemoryCapacity,
    GGUFModelManifestLoader,
    HeterogeneousRuntimeSnapshot,
    ModelManifest,
    ModelResidencyObservation,
    PlacementHardwareProfile,
    Request,
    ResourceProfile,
    RuntimeCapabilityCatalog,
    RuntimeCompositeExecutorCapability,
    RuntimeCompositeOperatorPlacement,
    RuntimeDesktopControlProfile,
    RuntimeExecutorCapability,
    RuntimeExecutorState,
    RuntimeExecutionReceipt,
    RuntimeExecutionContract,
    RuntimeKernelShapeProfile,
    RuntimeLinkState,
    RuntimePhonePowerProfile,
    RuntimePlacementSnapshot,
    RuntimeProtectedWorkObservation,
    RuntimeResourceError,
    RuntimeRouteShapeProfile,
    RuntimeSystemCostProfile,
    RuntimeTransitionCapability,
    RuntimeTransitionReceipt,
    UnifiedScheduleError,
    UnifiedScheduler,
)
from research_dev.scheduler._internal.decision_log import DecisionLogError
from research_dev.scheduler._internal.background_placement import (
    placement_frontier_key,
)
from research_dev.scheduler._internal.adaptive_decode import (
    ADAPTIVE_OBSERVATION_STORE_SCHEMA,
)
from research_dev.scheduler._internal.adaptive_decode_contracts import (
    AdaptiveDecodeError,
    AdaptiveDecodeGroupedObservation,
    AdaptiveDecodeWindowReceipt,
)
from research_dev.scheduler._internal.adaptive_decode_planning import (
    adaptive_decode_policies,
)
from research_dev.scheduler._internal.model_manifest import ModelManifestError
from research_dev.scheduler._internal.runtime_learning import (
    RuntimeRouteObservationStore,
    _MeasuredRouteObservation,
    _template_cost_feature_bucket,
)
from research_dev.scheduler._internal.runtime_controller import (
    RuntimeReplanRetryRequired,
)
from research_dev.scheduler._internal.route_generation import (
    RouteGenerationError,
    candidate_set_to_runtime_costs,
)
from research_dev.scheduler._internal.runtime_plan import (
    RuntimePlanError,
    RuntimeResidencyEviction,
)
from research_dev.scheduler._internal.runtime_search import (
    request_shape_bucket,
)
from research_dev.scheduler._internal.runtime_residency_projection import (
    RuntimeResidencyProjectionError,
    _authoritative_replacement_occupancy,
    project_scheduler_residency,
    transition_target_is_observed,
)
from research_dev.scheduler._internal.runtime_residency_cohorts import (
    RuntimeResidencyComponentIdentity,
    RuntimeResidencyCohortError,
    RuntimeResidencyCohortTracker,
    runtime_residency_component_identity,
)
from research_dev.scheduler._internal.types import canonical_sha256
from research_dev.scheduler.adapters import (
    apply_assumed_phone_power_catalog,
    catalog_preloaded_residency_samples,
    apply_assumed_phone_power_profile,
    EndpointRuntimeSample,
    ExecutorResidencySample,
    live_executor_residency_sample,
    materialize_desktop_control,
    PhysicalAdapterError,
    RuntimePhysicalTopology,
    UnifiedRuntimeSnapshotBuilder,
    model_residency_observations,
    validate_decision_candidate_coverage,
)

__all__ = [
    'ADAPTIVE_OBSERVATION_STORE_SCHEMA',
    'AdaptiveDecodeError',
    'AdaptiveDecodeGroupedObservation',
    'AdaptiveDecodeWindowReceipt',
    'AutomatedRuntimeTests',
    'DecisionLogError',
    'DeviceMemoryCapacity',
    'EndpointRuntimeSample',
    'ExecutorResidencySample',
    'FakeAutomatedPhysicalAdapter',
    'GGUFModelManifestLoader',
    'HeterogeneousRuntimeSnapshot',
    'ModelManifest',
    'ModelManifestError',
    'ModelResidencyObservation',
    'PhysicalAdapterError',
    'PlacementHardwareProfile',
    'Request',
    'ResourceProfile',
    'RouteGenerationError',
    'RuntimeCapabilityCatalog',
    'RuntimeCompositeExecutorCapability',
    'RuntimeCompositeOperatorPlacement',
    'RuntimeDesktopControlProfile',
    'RuntimeExecutionContract',
    'RuntimeExecutionReceipt',
    'RuntimeExecutorCapability',
    'RuntimeExecutorState',
    'RuntimeKernelShapeProfile',
    'RuntimeLinkState',
    'RuntimePhonePowerProfile',
    'RuntimePhysicalTopology',
    'RuntimePlacementSnapshot',
    'RuntimePlanError',
    'RuntimeProtectedWorkObservation',
    'RuntimeReplanRetryRequired',
    'RuntimeResidencyCohortError',
    'RuntimeResidencyCohortTracker',
    'RuntimeResidencyComponentIdentity',
    'RuntimeResidencyEviction',
    'RuntimeResidencyProjectionError',
    'RuntimeResourceError',
    'RuntimeRouteObservationStore',
    'RuntimeRouteShapeProfile',
    'RuntimeSystemCostProfile',
    'RuntimeTransitionCapability',
    'RuntimeTransitionReceipt',
    'UnifiedRuntimeSnapshotBuilder',
    'UnifiedScheduleError',
    'UnifiedScheduler',
    '_MeasuredRouteObservation',
    '_authoritative_replacement_occupancy',
    '_template_cost_feature_bucket',
    'adaptive_decode_policies',
    'apply_assumed_phone_power_catalog',
    'apply_assumed_phone_power_profile',
    'candidate_set_to_runtime_costs',
    'canonical_sha256',
    'capability',
    'catalog',
    'catalog_preloaded_residency_samples',
    'catalog_with_gpu_desktop_control',
    'executor_state',
    'live_executor_residency_sample',
    'materialize_desktop_control',
    'model_residency_observations',
    'placement_frontier_key',
    'placement_profile',
    'project_scheduler_residency',
    'protected_snapshot',
    'request',
    'request_shape_bucket',
    'runtime_residency_component_identity',
    'runtime_snapshot',
    'system_cost_profile',
    'transition_target_is_observed',
    'validate_decision_candidate_coverage',
]

try:
    from .test_gguf_cost import write_synthetic_gguf
except ImportError:
    from test_gguf_cost import write_synthetic_gguf


OPERATOR_KINDS = (
    "attention",
    "attention_projection",
    "embedding",
    "ffn",
    "kv_cache",
    "lm_head",
)


def placement_profile(
    *,
    phone_ops_per_s: int = 2_000_000_000,
    phone_power_mw: int = 4_000,
    phone_bandwidth: int = 2_000_000_000,
    extra_phones: tuple[str, ...] = (),
) -> PlacementHardwareProfile:
    """The synthetic fleet; ``extra_phones`` adds phones shaped like helper-c."""
    devices = (
        ("host-a", "cpu", "host-memory", 30_000),
        ("accelerator-b", "gpu", "gpu-memory", 55_000),
        ("helper-c", "phone", "phone-memory", phone_power_mw),
        *(
            (device_id, "phone", device_id + "-memory", phone_power_mw)
            for device_id in extra_phones
        ),
    )
    kernels = []
    domains = []
    for device_id, _, _, active_power in devices:
        domains.append({
            "domain_id": "energy:" + device_id,
            "evidence_ids": ["domain:" + device_id],
            "idle_power_mw": 500,
            "status": "measured",
        })
        throughput = {
            "host-a": 1_000_000_000,
            "accelerator-b": 8_000_000_000,
        }.get(device_id, phone_ops_per_s)
        for kind in OPERATOR_KINDS:
            kernels.append({
                "active_power_mw": active_power,
                "device_id": device_id,
                "domain_id": "energy:" + device_id,
                "effective_bytes_per_s": throughput,
                "effective_ops_per_s": throughput,
                "evidence_ids": [f"kernel:{device_id}:{kind}"],
                "kernel_id": f"kernel:{device_id}:{kind}",
                "launch_us": 2,
                "profile_id": f"kernel:{device_id}:{kind}",
                "status": "measured",
            })
    links = []
    for source, target, name, bandwidth in (
        ("host-a", "accelerator-b", "pcie-out", 8_000_000_000),
        ("accelerator-b", "host-a", "pcie-in", 8_000_000_000),
        ("host-a", "helper-c", "usb-out", phone_bandwidth),
        ("helper-c", "host-a", "usb-in", phone_bandwidth),
        *(
            row
            for device_id in extra_phones
            for row in (
                ("host-a", device_id, f"usb-{device_id}-out", phone_bandwidth),
                (device_id, "host-a", f"usb-{device_id}-in", phone_bandwidth),
            )
        ),
    ):
        links.append({
            "bandwidth_bytes_per_s": bandwidth,
            "domain_active_power_mw": {},
            "dynamic_pj_per_byte": 50,
            "evidence_ids": ["link:" + name],
            "fixed_dynamic_uj": 10,
            "fixed_latency_us": 10,
            "link_id": name,
            "ready": True,
            "source_device": source,
            "status": "measured",
            "target_device": target,
        })
    return PlacementHardwareProfile.from_json({
        "devices": [
            {
                "allocation_limit_bytes": 2_000_000_000,
                "device_id": device_id,
                "kind": kind,
                "memory_pool_id": pool_id,
                "ready": True,
            }
            for device_id, kind, pool_id, _ in devices
        ],
        "domains": domains,
        "energy_boundary_id": "synthetic-whole-fleet",
        "idle_charge_domains": [
            "energy:" + device_id for device_id, _, _, _ in devices
        ],
        "kernels": kernels,
        "links": links,
        "memory_pools": [
            {
                "capacity_bytes": 2_000_000_000,
                "pool_id": pool_id,
                "reserved_bytes": 0,
            }
            for _, _, pool_id, _ in devices
        ],
        "profile_id": "synthetic-capability-profile",
        "schema": "s42-placement-hardware-profile-v1",
    })


def capability(
    device_id: str,
    kind: str,
    *,
    fallback: bool = False,
    whole_model: bool = True,
    coordinated_route_families: tuple[str, ...] = (
        "layer_placement",
        "operator_offload",
        "operator_split",
    ),
    kernel_shape_profiles: tuple[RuntimeKernelShapeProfile, ...] = (),
) -> RuntimeExecutorCapability:
    return RuntimeExecutorCapability(
        executor_id="executor:" + device_id,
        device_id=device_id,
        endpoint="synthetic://" + device_id,
        backend="backend:" + kind,
        execution_resource_ids=("compute:" + device_id,),
        memory_resource_id={
            "host-a": "host-memory",
            "accelerator-b": "gpu-memory",
            "helper-c": "phone-memory",
        }.get(device_id, device_id + "-memory"),
        kernel_profiles={
            op_kind: f"kernel:{device_id}:{op_kind}"
            for op_kind in OPERATOR_KINDS
        },
        supported_quantizations=("*",),
        supports_whole_model=whole_model,
        supports_layer_placement=True,
        supports_operator_placement=True,
        supports_kv_cache=True,
        supports_split_coordinator=kind != "phone",
        supports_split_helper=True,
        split_axes=("column", "row", "tensor"),
        split_fractions_ppm=(250_000, 500_000, 750_000),
        layer_fractions_ppm=(250_000, 500_000, 750_000),
        residency_states=("hot", "warm", "cold"),
        maturity="QUALIFIED",
        evidence_ids=("executor-evidence:" + device_id,),
        qualified_fallback=fallback,
        maximum_temperature_millic=90_000,
        minimum_battery_ppm=(100_000 if kind == "phone" else 0),
        workspace_bytes_per_token=32,
        coordinated_route_families=coordinated_route_families,
        operator_plan_protocol="synthetic-plan-v1",
        kernel_shape_profiles=kernel_shape_profiles,
    )


def catalog(
    *,
    phone_ops_per_s: int = 2_000_000_000,
    phone_power_mw: int = 4_000,
    phone_bandwidth: int = 2_000_000_000,
    phone_whole_model: bool = True,
    gpu_whole_model: bool = True,
    system_cost_profiles: tuple[RuntimeSystemCostProfile, ...] = (),
    extra_phones: tuple[str, ...] = (),
) -> RuntimeCapabilityCatalog:
    profile = placement_profile(
        phone_ops_per_s=phone_ops_per_s,
        phone_power_mw=phone_power_mw,
        phone_bandwidth=phone_bandwidth,
        extra_phones=extra_phones,
    )
    resource_ids = [
        "compute:host-a",
        "compute:accelerator-b",
        "compute:helper-c",
        "link:pcie-out",
        "link:pcie-in",
        "link:usb-out",
        "link:usb-in",
        *(
            resource_id
            for device_id in extra_phones
            for resource_id in (
                "compute:" + device_id,
                f"link:usb-{device_id}-out",
                f"link:usb-{device_id}-in",
            )
        ),
    ]
    resources = {
        resource_id: ResourceProfile(
            resource_id=resource_id,
            kind=("compute" if resource_id.startswith("compute:") else "link"),
            capacity=1,
            ready=True,
            identity=resource_id,
        )
        for resource_id in resource_ids
    }
    transitions = tuple(
        RuntimeTransitionCapability(
            transition_id=f"load:{device_id}",
            device_id=device_id,
            source_state="cold",
            target_state="hot",
            fixed_latency_us=100,
            bandwidth_bytes_per_s=1_000_000_000,
            fixed_energy_uj=100,
            dynamic_pj_per_byte=100,
            resource_ids=("compute:" + device_id,),
            maturity="QUALIFIED",
            evidence_ids=("transition:" + device_id,),
        )
        for device_id in ("host-a", "accelerator-b", "helper-c", *extra_phones)
    )
    return RuntimeCapabilityCatalog(
        catalog_id="synthetic-catalog",
        placement_profile=profile,
        resources=resources,
        executors=(
            capability("host-a", "cpu", fallback=True),
            capability(
                "accelerator-b", "gpu", whole_model=gpu_whole_model
            ),
            capability(
                "helper-c",
                "phone",
                whole_model=phone_whole_model,
            ),
            *(
                capability(device_id, "phone", whole_model=phone_whole_model)
                for device_id in extra_phones
            ),
        ),
        transitions=transitions,
        minimum_energy_saving_ppm=10_000,
        maximum_latency_ppm=10_000_000,
        system_cost_profiles=system_cost_profiles,
    )


def catalog_with_gpu_desktop_control(
    manifest,
    source: RuntimeCapabilityCatalog | None = None,
) -> RuntimeCapabilityCatalog:
    source = catalog() if source is None else source
    placements = tuple(
        RuntimeCompositeOperatorPlacement(
            operator_id=operator.operator_id,
            primary_device_id="accelerator-b",
            helper_device_id=None,
            split_axis="none",
            split_fraction_ppm=0,
        )
        for operator in manifest.operators
    )
    return replace(
        source,
        desktop_control_profiles=(RuntimeDesktopControlProfile(
            profile_id="synthetic-gpu-control",
            artifact_sha256=manifest.artifact_sha256,
            executor_id="executor:accelerator-b",
            operator_placements=placements,
            maturity="QUALIFIED",
            evidence_ids=("synthetic-gpu-control-evidence",),
        ),),
    )


def system_cost_profile(
    *,
    helper_interference_ppm: int,
    control_delay_us: int = 0,
    control_delay_upper_us: int = 0,
) -> RuntimeSystemCostProfile:
    return RuntimeSystemCostProfile(
        selector_id="synthetic-protected-work-phase-1",
        feature_ranges={"protected_phase": (1, 1)},
        interference_ppm_by_resource={
            "compute:host-a": 0,
            "compute:accelerator-b": 0,
            "compute:helper-c": helper_interference_ppm,
            "link:pcie-out": 0,
            "link:pcie-in": 0,
            "link:usb-out": 0,
            "link:usb-in": 0,
        },
        control_delay_us=control_delay_us,
        control_delay_upper_us=control_delay_upper_us,
        lower_error_ppm=100_000,
        upper_error_ppm=100_000,
        sample_count=4,
        maturity="QUALIFIED",
        evidence_ids=("synthetic-held-out-system-cost",),
    )


def protected_snapshot(
    manifest,
) -> HeterogeneousRuntimeSnapshot:
    return replace(
        runtime_snapshot(manifest, phone_bandwidth=8_000_000_000),
        cost_features={"protected_phase": 1},
        protected_work=RuntimeProtectedWorkObservation(
            observation_id="synthetic-protected-work",
            critical_path_end_us=4_000_000,
            phase_power_mw=250_000,
            stranded_idle_power_mw=50_000,
            causal_tail_power_mw=100_000,
            sample_count=4,
            measured=True,
        ),
    )


def executor_state(
    executor_id: str,
    *,
    busy_until_us: int = 0,
    free_slots: int = 1,
):
    return RuntimeExecutorState(
        executor_id=executor_id,
        healthy=True,
        ready=True,
        temperature_millic=40_000,
        battery_ppm=900_000,
        free_slots=free_slots,
        busy_until_us=busy_until_us,
    )


def runtime_snapshot(
    manifest,
    *,
    include_phone: bool = True,
    phone_bandwidth: int = 2_000_000_000,
    gpu_busy_until_us: int = 0,
    gpu_free_slots: int = 1,
    resident_devices: tuple[str, ...] = (
        "host-a", "accelerator-b", "helper-c"
    ),
    extra_phones: tuple[str, ...] = (),
) -> HeterogeneousRuntimeSnapshot:
    states = {
        "executor:host-a": executor_state("executor:host-a"),
        "executor:accelerator-b": executor_state(
            "executor:accelerator-b",
            busy_until_us=gpu_busy_until_us,
            free_slots=gpu_free_slots,
        ),
    }
    if include_phone:
        states["executor:helper-c"] = executor_state("executor:helper-c")
    for device_id in extra_phones:
        states["executor:" + device_id] = executor_state("executor:" + device_id)
    link_states = {
        name: RuntimeLinkState(
            link_id=name,
            ready=True,
            measured_bandwidth_bytes_per_s=(
                phone_bandwidth if name.startswith("usb") else 8_000_000_000
            ),
            busy_until_us=0,
        )
        for name in (
            "pcie-out", "pcie-in", "usb-out", "usb-in",
            *(
                f"usb-{device_id}-{direction}"
                for device_id in extra_phones for direction in ("out", "in")
            ),
        )
    }
    memory = RuntimePlacementSnapshot(
        snapshot_id="synthetic-memory",
        captured_at_us=0,
        valid_until_us=10_000_000,
        capacities={
            resource_id: DeviceMemoryCapacity(
                resource_id,
                2_000_000_000,
                1_500_000_000,
                10_000_000,
            )
            for resource_id in (
                "host-memory", "gpu-memory", "phone-memory",
                *(device_id + "-memory" for device_id in extra_phones),
            )
        },
    )
    residency = tuple(
        ModelResidencyObservation(
            model_id=manifest.model_id,
            artifact_sha256=manifest.artifact_sha256,
            device_id=device_id,
            state="hot",
            resident_tensor_ids=tuple(
                tensor.tensor_id for tensor in manifest.tensors
            ),
            resident_bytes=manifest.tensor_bytes,
            generation=1,
        )
        for device_id in resident_devices
    )
    return HeterogeneousRuntimeSnapshot(
        snapshot_id="synthetic-runtime",
        captured_at_us=0,
        valid_until_us=10_000_000,
        memory=memory,
        executors=states,
        links=link_states,
        residency=residency,
    )


def request(
    request_id: str,
    *,
    arrival_us: int = 1_000,
    input_tokens: int = 12,
    output_tokens: int = 4,
) -> Request:
    return Request(
        request_id=request_id,
        workload_id="unseen-workload",
        arrival_us=arrival_us,
        deadline_us=arrival_us + 5_000_000,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        quality_requirement="exact",
    )


class FakeAutomatedPhysicalAdapter:
    """Execute only the endpoint and plan returned by UnifiedScheduler."""

    def __init__(self, scheduler: UnifiedScheduler) -> None:
        self.scheduler = scheduler
        self.executed: list[tuple[str, str, str]] = []

    @staticmethod
    def _transition_receipts(ticket):
        participants = {
            row.device_id: row for row in ticket.binding.participants
        }
        return tuple(
            RuntimeTransitionReceipt(
                ticket_id=ticket.ticket_id,
                request_id=ticket.request.request_id,
                artifact_sha256=ticket.model.artifact_sha256,
                operator_plan_sha256=ticket.execution_plan.plan_sha256,
                transition_id=transition.transition_id,
                executor_id=participants[
                    transition.device_id
                ].executor_id,
                endpoint=participants[transition.device_id].endpoint,
                device_id=transition.device_id,
                source_state=transition.source_state,
                target_state=transition.target_state,
                resource_ids=transition.resource_ids,
                resource_slots=transition.resource_slots,
                started_us=ticket.decision.start_us,
                finished_us=ticket.decision.start_us + 1,
                status="COMPLETED",
            )
            for transition in ticket.execution_plan.transitions
        )

    def execute(self, request_id: str):
        queued = self.scheduler.runtime_ticket(request_id)
        epoch_ns = time.monotonic_ns() - queued.decision.start_us * 1_000
        ticket = self.scheduler.wait_runtime_request(request_id, epoch_ns)
        if ticket.transition_status == "PENDING":
            self.scheduler.record_automated_transition_receipts(
                request_id, self._transition_receipts(ticket)
            )
        ticket = self.scheduler.runtime_execution_ticket(request_id)
        endpoint = ticket.binding.endpoint
        protocol = ticket.binding.operator_plan_protocol
        plan_sha256 = ticket.execution_plan.plan_sha256
        self.executed.append((endpoint, protocol, plan_sha256))
        output = hashlib.sha256(
            (endpoint + protocol + plan_sha256).encode("ascii")
        ).hexdigest()
        return self.scheduler.complete_automated_request(
            request_id,
            RuntimeExecutionReceipt(
                ticket_id=ticket.ticket_id,
                request_id=request_id,
                artifact_sha256=ticket.model.artifact_sha256,
                operator_plan_sha256=plan_sha256,
                executor_id=ticket.binding.executor_id,
                endpoint=endpoint,
                operator_plan_protocol=protocol,
                participant_executor_ids=tuple(
                    row.executor_id for row in ticket.binding.participants
                ),
                started_us=ticket.decision.start_us,
                finished_us=ticket.decision.finish_us,
                output_sha256="sha256:" + output,
                status="COMPLETED",
            ),
        )


class AutomatedRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "new-model.gguf"
        write_synthetic_gguf(self.path)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def scheduler_and_manifest(self, profile=None):
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile or catalog())
        manifest = scheduler.register_gguf_model("unseen-model", self.path)
        return scheduler, manifest

    def scheduler_with_gpu_control(self):
        manifest = GGUFModelManifestLoader.load(
            "unseen-model", self.path
        )
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(
            catalog_with_gpu_desktop_control(manifest)
        )
        scheduler.register_model_manifest(manifest)
        return scheduler, manifest

    @staticmethod
    def transition_receipts(ticket, *, failed: bool = False):
        participants = {
            row.device_id: row for row in ticket.binding.participants
        }
        rows = []
        for index, transition in enumerate(ticket.execution_plan.transitions):
            participant = participants[transition.device_id]
            rows.append(RuntimeTransitionReceipt(
                ticket_id=ticket.ticket_id,
                request_id=ticket.request.request_id,
                artifact_sha256=ticket.model.artifact_sha256,
                operator_plan_sha256=ticket.execution_plan.plan_sha256,
                transition_id=transition.transition_id,
                executor_id=participant.executor_id,
                endpoint=participant.endpoint,
                device_id=transition.device_id,
                source_state=transition.source_state,
                target_state=transition.target_state,
                resource_ids=transition.resource_ids,
                started_us=ticket.decision.start_us,
                finished_us=ticket.decision.start_us + 1,
                status=("FAILED" if failed and index == 0 else "COMPLETED"),
            ))
        return tuple(rows)

    @staticmethod
    def execution_receipt(ticket, **changes):
        values = {
            "ticket_id": ticket.ticket_id,
            "request_id": ticket.request.request_id,
            "artifact_sha256": ticket.model.artifact_sha256,
            "operator_plan_sha256": ticket.execution_plan.plan_sha256,
            "executor_id": ticket.binding.executor_id,
            "endpoint": ticket.binding.endpoint,
            "operator_plan_protocol": ticket.binding.operator_plan_protocol,
            "participant_executor_ids": tuple(
                row.executor_id for row in ticket.binding.participants
            ),
            "started_us": ticket.decision.start_us,
            "finished_us": ticket.decision.finish_us,
            "output_sha256": "sha256:" + "a" * 64,
            "status": "COMPLETED",
        }
        values.update(changes)
        return RuntimeExecutionReceipt(**values)

    @staticmethod
    def measured_execution_receipt(ticket, *, latency_us, fleet_energy_uj):
        domain_ids = (
            "energy:accelerator-b",
            "energy:helper-c",
            "energy:host-a",
        )
        base = fleet_energy_uj // len(domain_ids)
        energy = {domain_id: base for domain_id in domain_ids}
        energy[domain_ids[-1]] += fleet_energy_uj - sum(energy.values())
        links = {
            resource_id.removeprefix("link:"): 1
            for resource_id in ticket.execution_plan.resource_ids
            if resource_id.startswith("link:")
        }
        return AutomatedRuntimeTests.execution_receipt(
            ticket,
            finished_us=ticket.decision.start_us + latency_us,
            energy_boundary_id="synthetic-whole-fleet",
            fleet_energy_uj_by_domain=energy,
            transfer_energy_uj_by_link=links,
            measurement_evidence_ids=(
                "synthetic-matched-energy-boundary",
            ),
            energy_attribution_kind="isolated",
        )

if __name__ == "__main__":
    unittest.main()
