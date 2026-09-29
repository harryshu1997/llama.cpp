"""Non-executing validation for a scheduler-controlled physical rig."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence
from urllib.parse import urlsplit

from .._internal.model_manifest import ModelManifest
from .._internal.policy import Request
from .._internal.runtime_capabilities import (
    HeterogeneousRuntimeSnapshot,
    RuntimeCapabilityCatalog,
)
from .contracts import PhysicalAdapterError
from .coverage import catalog_structural_device_families


PHYSICAL_PREFLIGHT_SCHEMA = "research-scheduler-physical-preflight-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _evidence_for_reasons(reasons: Sequence[str]) -> tuple[str, ...]:
    requirements = {
        "BATTERY_LIMIT": "live battery observation above executor threshold",
        "CANDIDATE_NOT_VISITED": (
            "capability-compatible route generation and a matching cost profile"
        ),
        "COMPOSITE_COORDINATOR_ABSENT": (
            "registered physical coordinator for the exact operator plan"
        ),
        "ENERGY_UNKNOWN": (
            "repeated whole-fleet energy under the catalog accounting boundary"
        ),
        "EXECUTOR_CAPACITY_UNAVAILABLE": (
            "live free-slot observation or scheduler-owned completion event"
        ),
        "EXECUTOR_NOT_READY": (
            "live endpoint publication receipt for the exact artifact"
        ),
        "EXECUTOR_OBSERVATION_ABSENT": (
            "fresh endpoint health and free-slot observation"
        ),
        "EXECUTOR_UNHEALTHY": "successful endpoint health and slot probes",
        "LINK_NOT_READY": "fresh measured transport-link readiness",
        "LINK_OBSERVATION_ABSENT": "fresh measured transport-link observation",
        "MEMORY_CAPACITY": "fresh capacity receipt with required reserve",
        "MEMORY_RESOURCE_ABSENT": "registered physical memory capacity",
        "RESIDENCY_TRANSITION_ABSENT": (
            "qualified load, publication, rollback, and restore transition"
        ),
        "ROUTE_NOT_QUALIFIED": (
            "held-out latency, exact-output, and matched whole-fleet energy evidence"
        ),
        "SLO_UPPER_BOUND": "held-out latency upper bound meeting the request SLO",
        "THERMAL_LIMIT": "fresh thermal observation below executor threshold",
    }
    result = {
        requirements.get(reason, "resolved fail-closed reason: " + reason)
        for reason in reasons
    }
    return tuple(sorted(result))


@dataclass(frozen=True)
class PhysicalPreflightModel:
    manifest: ModelManifest
    model_path: Path
    request: Request
    snapshot: HeterogeneousRuntimeSnapshot
    expected_model_alias: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.manifest, ModelManifest)
            or not isinstance(self.model_path, Path)
            or not isinstance(self.request, Request)
            or not isinstance(self.snapshot, HeterogeneousRuntimeSnapshot)
            or self.request.workload_id == ""
            or (
                self.expected_model_alias is not None
                and (
                    type(self.expected_model_alias) is not str
                    or not self.expected_model_alias
                    or not self.expected_model_alias.isascii()
                )
            )
        ):
            raise PhysicalAdapterError("physical preflight model is invalid")
        self.request.validate()


@dataclass(frozen=True)
class PhysicalPreflightCheck:
    check_id: str
    status: str
    detail: str

    def __post_init__(self) -> None:
        if (
            type(self.check_id) is not str
            or not self.check_id
            or not self.check_id.isascii()
            or self.status not in {"BLOCKED", "PASS", "WARN"}
            or type(self.detail) is not str
            or not self.detail
            or not self.detail.isascii()
        ):
            raise PhysicalAdapterError("physical preflight check is invalid")

    def to_json(self) -> dict[str, str]:
        return {
            "check_id": self.check_id,
            "detail": self.detail,
            "status": self.status,
        }


@dataclass(frozen=True)
class PhysicalPreflightRoute:
    model_id: str
    route_id: str
    executor_id: str
    device_kinds: tuple[str, ...]
    maturity: str
    readiness: str
    admitted: bool
    rejection_reasons: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    required_evidence: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "admitted": self.admitted,
            "device_kinds": list(self.device_kinds),
            "evidence_ids": list(self.evidence_ids),
            "executor_id": self.executor_id,
            "maturity": self.maturity,
            "model_id": self.model_id,
            "readiness": self.readiness,
            "rejection_reasons": list(self.rejection_reasons),
            "required_evidence": list(self.required_evidence),
            "route_id": self.route_id,
        }


@dataclass(frozen=True)
class PhysicalPreflightReport:
    catalog_id: str
    energy_boundary_id: str
    checks: tuple[PhysicalPreflightCheck, ...]
    desktop_baselines: Mapping[str, str]
    phone_routes: tuple[PhysicalPreflightRoute, ...]
    request_shapes: tuple[Mapping[str, object], ...] = ()

    @property
    def desktop_baseline_ready(self) -> bool:
        return bool(self.desktop_baselines) and not any(
            row.status == "BLOCKED" for row in self.checks
        )

    @property
    def phone_assistance_ready(self) -> bool:
        return any(
            row.admitted and row.maturity == "QUALIFIED"
            for row in self.phone_routes
        )

    @property
    def status(self) -> str:
        return "PASS" if self.desktop_baseline_ready else "BLOCKED"

    def to_json(self) -> dict[str, object]:
        unavailable = tuple(
            row for row in self.phone_routes
            if not row.admitted or row.maturity != "QUALIFIED"
        )
        return {
            "catalog_id": self.catalog_id,
            "checks": [row.to_json() for row in self.checks],
            "desktop_baseline_ready": self.desktop_baseline_ready,
            "desktop_baselines": dict(self.desktop_baselines),
            "energy_boundary_id": self.energy_boundary_id,
            "phone_assistance_ready": self.phone_assistance_ready,
            "phone_routes": [row.to_json() for row in self.phone_routes],
            "request_shapes": {
                "checked": len(self.request_shapes),
                "unsupported": [
                    dict(row) for row in self.request_shapes
                    if not row.get("supported", True)
                ],
            },
            "schema": PHYSICAL_PREFLIGHT_SCHEMA,
            "shadow_or_unavailable_phone_routes": [
                row.to_json() for row in unavailable
            ],
            "status": self.status,
        }


def _append_check(
    checks: list[PhysicalPreflightCheck],
    check_id: str,
    condition: bool,
    detail: str,
    *,
    advisory: bool = False,
) -> None:
    checks.append(PhysicalPreflightCheck(
        check_id,
        "PASS" if condition else "WARN" if advisory else "BLOCKED",
        detail,
    ))


def _check_dependency_paths(
    checks: list[PhysicalPreflightCheck],
    executable_paths: Mapping[str, Path],
    required_paths: Mapping[str, Path],
    library_directories: Mapping[str, Path],
) -> None:
    for name, path in sorted(executable_paths.items()):
        _append_check(
            checks,
            "executable:" + name,
            isinstance(path, Path) and path.is_file() and os.access(path, os.X_OK),
            "executable path " + str(path),
        )
    for name, path in sorted(required_paths.items()):
        _append_check(
            checks,
            "path:" + name,
            isinstance(path, Path) and path.is_file(),
            "required path " + str(path),
        )
    for name, path in sorted(library_directories.items()):
        _append_check(
            checks,
            "library:" + name,
            isinstance(path, Path) and path.is_dir(),
            "library directory " + str(path),
        )


def _check_model_identity(
    checks: list[PhysicalPreflightCheck],
    scheduler: object,
    catalog: RuntimeCapabilityCatalog,
    row: PhysicalPreflightModel,
) -> None:
    expected = row.manifest.artifact_sha256
    actual = (
        _sha256(row.model_path) if row.model_path.is_file() else "missing"
    )
    _append_check(
        checks,
        "model-hash:" + row.manifest.model_id,
        actual == expected,
        "GGUF artifact " + str(row.model_path) + " is " + actual,
    )
    registered = scheduler.runtime_model_manifest(row.manifest.model_id)
    _append_check(
        checks,
        "model-catalog:" + row.manifest.model_id,
        registered == row.manifest,
        "registered manifest identity is exact",
    )
    memory_resources = set(catalog.placement_profile.memory_pools)
    missing_memory = tuple(sorted(
        memory_resources - set(row.snapshot.memory.capacities)
    ))
    _append_check(
        checks,
        "memory:" + row.manifest.model_id,
        not missing_memory,
        (
            "runtime snapshot covers catalog memory resources; missing="
            + ",".join(missing_memory or ("none",))
            + "; live capacities are admission-authoritative"
        ),
    )


def _check_model_executor_coverage(
    checks: list[PhysicalPreflightCheck],
    catalog: RuntimeCapabilityCatalog,
    row: PhysicalPreflightModel,
    kind_by_device: Mapping[str, str],
) -> None:
    applicable_executors = (
        tuple(catalog.executors)
        + tuple(
            capability
            for capability in catalog.composite_executors
            if capability.artifact_sha256 in {
                None, row.manifest.artifact_sha256
            }
        )
    )
    desktop_executors = {
        capability.executor_id
        for capability in applicable_executors
        if "phone" not in {
            kind_by_device[device_id]
            for device_id in (
                capability.participant_device_ids
                if hasattr(capability, "participant_device_ids")
                else (capability.device_id,)
            )
        }
    }
    phone_executors = {
        capability.executor_id
        for capability in applicable_executors
    } - desktop_executors
    _append_check(
        checks,
        "desktop-executors:" + row.manifest.model_id,
        desktop_executors.issubset(row.snapshot.executors),
        "runtime snapshot covers every applicable desktop executor",
    )
    _append_check(
        checks,
        "phone-executors:" + row.manifest.model_id,
        phone_executors.issubset(row.snapshot.executors),
        "runtime snapshot covers every applicable phone executor",
        advisory=True,
    )


def _thermal_status_detail(state, capability) -> str:
    """Opt-in thermal status policy detail; empty unless the state carries a raw status."""
    if state is None or state.thermal_status is None:
        return ""
    return (
        ", thermal status " + str(state.thermal_status)
        + " (limit " + str(capability.maximum_thermal_status) + ")"
    )


def _check_model_phone_state(
    checks: list[PhysicalPreflightCheck],
    catalog: RuntimeCapabilityCatalog,
    row: PhysicalPreflightModel,
) -> None:
    phone_count = sum(catalog.placement_profile.devices[row.device_id].kind == "phone"
                      for row in catalog.executors)
    for capability in catalog.executors:
        if (
            catalog.placement_profile.devices[
                capability.device_id
            ].kind != "phone"
        ):
            continue
        state = row.snapshot.executors.get(capability.executor_id)
        check_suffix = row.manifest.model_id + (":" + capability.device_id if phone_count > 1 else "")
        live_ok = state is not None and state.healthy
        _append_check(
            checks,
            "phone-health:" + check_suffix,
            live_ok,
            (
                "phone executor health is "
                + ("absent" if state is None else str(state.healthy))
            ),
            advisory=True,
        )
        thermal_ok = (
            state is not None
            and state.temperature_millic
                <= capability.maximum_temperature_millic
            and (
                state.thermal_status is None
                or state.thermal_status <= capability.maximum_thermal_status
            )
        )
        _append_check(
            checks,
            "phone-thermal:" + check_suffix,
            thermal_ok,
            (
                "phone temperature millic is "
                + ("absent" if state is None else str(
                    state.temperature_millic
                ))
                + _thermal_status_detail(state, capability)
            ),
            advisory=True,
        )
        battery_ok = (
            state is not None
            and state.battery_ppm >= capability.minimum_battery_ppm
        )
        _append_check(
            checks,
            "phone-battery:" + check_suffix,
            battery_ok,
            (
                "phone battery ppm is "
                + ("absent" if state is None else str(
                    state.battery_ppm
                ))
            ),
            advisory=True,
        )


def _check_endpoint_contracts(
    checks: list[PhysicalPreflightCheck],
    catalog: RuntimeCapabilityCatalog,
) -> None:
    endpoint_contracts: dict[
        tuple[str, int], dict[str, set[str]]
    ] = {}
    endpoint_valid = True
    for capability in (*catalog.executors, *catalog.composite_executors):
        if (getattr(capability, "supports_split_helper", False)
            and not capability.supports_whole_model
            and catalog.placement_profile.devices[capability.device_id].kind == "phone"
            and capability.endpoint == "physical://" + capability.device_id):
            continue
        parsed = urlsplit(capability.endpoint)
        valid = (
            parsed.scheme == "http"
            and parsed.hostname is not None
            and parsed.port is not None
            and parsed.path in {"", "/"}
            and not parsed.query
            and not parsed.fragment
            and parsed.username is None
            and parsed.password is None
        )
        endpoint_valid = endpoint_valid and valid
        if valid:
            contract = endpoint_contracts.setdefault(
                (parsed.hostname, parsed.port),
                {"artifacts": set(), "contexts": set(), "models": set()},
            )
            artifact_sha256 = getattr(
                capability, "artifact_sha256", None
            )
            if artifact_sha256 is not None:
                contract["artifacts"].add(artifact_sha256)
            model_alias = capability.adapter_parameters.get("model_alias")
            if isinstance(model_alias, str):
                contract["models"].add(model_alias)
            context_resource_id = capability.adapter_parameters.get(
                "context_resource_id"
            )
            if isinstance(context_resource_id, str):
                contract["contexts"].add(context_resource_id)
    endpoint_valid = endpoint_valid and all(
        len(contract["artifacts"]) <= 1
        and len(contract["models"]) <= 1
        and len(contract["contexts"]) <= 1
        for contract in endpoint_contracts.values()
    )
    shared_endpoint_count = sum(
        1
        for endpoint in {
            capability.endpoint
            for capability in (
                *catalog.executors, *catalog.composite_executors
            )
        }
        if sum(
            capability.endpoint == endpoint
            for capability in (
                *catalog.executors, *catalog.composite_executors
            )
        ) > 1
    )
    _append_check(
        checks,
        "endpoint-contracts",
        endpoint_valid,
        (
            "HTTP endpoint bindings are valid and shared variants agree; "
            "shared endpoints=" + str(shared_endpoint_count)
        ),
    )
    _append_check(
        checks,
        "energy-boundary",
        bool(catalog.placement_profile.energy_boundary_id),
        "catalog has one whole-fleet energy boundary",
    )


def _check_candidate_set(
    checks: list[PhysicalPreflightCheck],
    catalog: RuntimeCapabilityCatalog,
    row: PhysicalPreflightModel,
    candidates: object,
    kind_by_device: Mapping[str, str],
) -> None:
    expected_families = set(catalog_structural_device_families(
        catalog, row.manifest
    ))
    observed_families = {
        tuple(sorted(
            kind_by_device[device_id]
            for device_id in candidate.device_ids
        ))
        for candidate in candidates.candidates
    }
    missing_families = tuple(sorted(
        expected_families - observed_families
    ))
    _append_check(
        checks,
        "candidate-families:" + row.manifest.model_id,
        not missing_families,
        "missing structural candidate families are "
        + repr(missing_families),
    )
    _append_check(
        checks,
        "candidate-costs:" + row.manifest.model_id,
        all(
            candidate.cost.service_us > 0
            and candidate.cost.service_upper_us
                >= candidate.cost.service_us
            and (
                candidate.admitted
                or bool(candidate.rejection_reasons)
            )
            for candidate in candidates.candidates
        ),
        "every visited candidate has cost and admission evidence",
    )


def _check_desktop_baseline(
    checks: list[PhysicalPreflightCheck],
    scheduler: object,
    catalog: RuntimeCapabilityCatalog,
    row: PhysicalPreflightModel,
    baseline: object,
    kind_by_device: Mapping[str, str],
    route_profile_by_id: Mapping[str, object],
    transition_by_id: Mapping[str, object],
) -> bool:
    baseline_kinds = tuple(sorted(
        kind_by_device[device_id] for device_id in baseline.device_ids
    ))
    transitions_valid = all(
        transition.transition_id in transition_by_id
        and transition.maturity == "QUALIFIED"
        for transition in baseline.plan.transitions
    )
    selected_profile = route_profile_by_id.get(
        baseline.plan.route_profile_id
    )
    profile_qualified = (
        selected_profile is not None
        and selected_profile.maturity == "QUALIFIED"
    ) or (
        baseline.plan.route_profile_id is not None
        and baseline.cost.latency_evidence == "MEASURED"
        and baseline.cost.energy_evidence == "MEASURED"
    )
    desktop_control = catalog.desktop_control_by_artifact.get(
        row.manifest.artifact_sha256
    )
    control_qualified = (
        desktop_control is not None
        and desktop_control.maturity == "QUALIFIED"
        and bool(desktop_control.evidence_ids)
        and desktop_control.executor_id == baseline.binding.executor_id
        and desktop_control.placement_sha256
            == baseline.plan.desktop_placement_sha256
    )
    maturity_qualified = (
        baseline.maturity == "QUALIFIED" or control_qualified
    )
    execution_qualified = control_qualified or (
        baseline.maturity == "QUALIFIED" and profile_qualified
    )
    capacity_source = (
        None if desktop_control is None else
        catalog.composite_executor_by_id.get(desktop_control.executor_id)
    )
    capacity_selection = (
        None
        if capacity_source is None
        or type(capacity_source.adapter_parameters.get("gpu_layers")) is not int
        else scheduler.select_live_vram_desktop_parent(
            row.manifest.model_id, row.snapshot.memory,
            preserve_placement=True,
        )
    )
    selected_parent = (
        None if capacity_selection is None else capacity_selection.selected
    )
    capacity_parent_ready = (
        True if selected_parent is None else
        selected_parent.placement_sha256 == desktop_control.placement_sha256
        and selected_parent.executor_id == desktop_control.executor_id
        and selected_parent.feasible
    )
    _append_check(
        checks,
        "desktop-parent-capacity:" + row.manifest.model_id,
        capacity_parent_ready,
        (
            "selected_gpu_layers=" + (
                "not_applicable" if selected_parent is None
                else str(selected_parent.gpu_layers)
            )
            + "; selected_placement="
            + (
                "not_applicable" if selected_parent is None
                else selected_parent.placement_sha256
            )
            + "; catalog_placement="
            + (
                "absent" if desktop_control is None
                else desktop_control.placement_sha256
            )
            + "; live_free_vram_bytes="
            + (
                "not_applicable" if selected_parent is None
                else str(selected_parent.live_free_vram_bytes)
            )
            + "; required_with_reserve_bytes="
            + (
                "not_applicable" if selected_parent is None
                else str(selected_parent.required_with_reserve_bytes)
            )
            + "; selection_sha256="
            + (
                "not_applicable" if capacity_selection is None
                else capacity_selection.selection_sha256
            )
        ),
    )
    baseline_ready = (
        baseline.admitted
        and maturity_qualified
        and capacity_parent_ready
        and "phone" not in baseline_kinds
        and baseline.binding.endpoint is not None
        and baseline.binding.operator_plan_protocol is not None
        and transitions_valid
        and execution_qualified
        and baseline.cost.fleet_energy_uj is not None
        and (
            row.expected_model_alias is None
            or baseline.plan.adapter_parameters.get("model_alias")
                == row.expected_model_alias
        )
    )
    _append_check(
        checks,
        "desktop-baseline:" + row.manifest.model_id,
        baseline_ready,
        (
            "route " + baseline.candidate_id + " uses "
            + "+".join(baseline_kinds) + "; reasons="
            + ",".join(baseline.rejection_reasons or ("none",))
            + "; maturity=" + baseline.maturity
            + "; latency_evidence="
            + baseline.cost.latency_evidence
            + "; energy_evidence=" + baseline.cost.energy_evidence
            + "; profile="
            + str(baseline.plan.route_profile_id)
            + "; profile_qualified=" + str(profile_qualified)
            + "; control_qualified=" + str(control_qualified)
            + "; transitions_qualified=" + str(transitions_valid)
        ),
    )
    _append_check(
        checks,
        "desktop-cost-evidence:" + row.manifest.model_id,
        profile_qualified,
        (
            "route cost evidence is latency="
            + baseline.cost.latency_evidence
            + ",energy=" + baseline.cost.energy_evidence
            + "; profile=" + str(baseline.plan.route_profile_id)
            + "; profile_maturity="
            + (
                "absent"
                if selected_profile is None
                else selected_profile.maturity
            )
        ),
        advisory=True,
    )
    return baseline_ready


def _visited_phone_routes(
    catalog: RuntimeCapabilityCatalog,
    row: PhysicalPreflightModel,
    candidates: object,
    kind_by_device: Mapping[str, str],
    route_profile_by_id: Mapping[str, object],
    transition_by_id: Mapping[str, object],
) -> tuple[list[PhysicalPreflightRoute], set[str]]:
    phone_routes = []
    represented_phone_executors = set()
    for candidate in candidates.candidates:
        device_kinds = tuple(sorted(
            kind_by_device[device_id]
            for device_id in candidate.device_ids
        ))
        if "phone" not in device_kinds:
            continue
        represented_phone_executors.add(candidate.binding.executor_id)
        profile = route_profile_by_id.get(
            candidate.plan.route_profile_id
        )
        executor = (
            catalog.executor_by_id.get(candidate.binding.executor_id)
            or catalog.composite_executor_by_id.get(
                candidate.binding.executor_id
            )
        )
        evidence = set(() if executor is None else executor.evidence_ids)
        if profile is not None:
            evidence.update(profile.evidence_ids)
        for transition in candidate.plan.transitions:
            evidence.update(
                transition_by_id[transition.transition_id].evidence_ids
            )
        reasons = candidate.rejection_reasons
        required_evidence = _evidence_for_reasons(reasons)
        if candidate.maturity != "QUALIFIED" and not required_evidence:
            required_evidence = _evidence_for_reasons(
                ("ROUTE_NOT_QUALIFIED",)
            )
        phone_routes.append(PhysicalPreflightRoute(
            model_id=row.manifest.model_id,
            route_id=candidate.candidate_id,
            executor_id=candidate.binding.executor_id,
            device_kinds=device_kinds,
            maturity=candidate.maturity,
            readiness=(
                "READY" if candidate.binding.ready else "NOT_READY"
            ),
            admitted=candidate.admitted,
            rejection_reasons=reasons,
            evidence_ids=tuple(sorted(evidence)),
            required_evidence=required_evidence,
        ))
    return phone_routes, represented_phone_executors


def _unvisited_phone_routes(
    catalog: RuntimeCapabilityCatalog,
    row: PhysicalPreflightModel,
    kind_by_device: Mapping[str, str],
    represented_phone_executors: set[str],
) -> list[PhysicalPreflightRoute]:
    applicable_phone_capabilities = []
    for capability in catalog.executors:
        if (
            kind_by_device[capability.device_id] == "phone"
            and capability.supports_whole_model
        ):
            applicable_phone_capabilities.append((
                capability,
                (capability.device_id,),
            ))
    for capability in catalog.composite_executors:
        if (
            "phone" in {
                kind_by_device[device_id]
                for device_id in capability.participant_device_ids
            }
            and capability.artifact_sha256 in {
                None, row.manifest.artifact_sha256
            }
        ):
            applicable_phone_capabilities.append((
                capability,
                capability.participant_device_ids,
            ))
    phone_routes = []
    for capability, device_ids in applicable_phone_capabilities:
        if capability.executor_id in represented_phone_executors:
            continue
        phone_routes.append(PhysicalPreflightRoute(
            model_id=row.manifest.model_id,
            route_id="catalog:" + capability.executor_id + ":not-visited",
            executor_id=capability.executor_id,
            device_kinds=tuple(sorted(
                kind_by_device[device_id] for device_id in device_ids
            )),
            maturity=capability.maturity,
            readiness="NOT_READY",
            admitted=False,
            rejection_reasons=("CANDIDATE_NOT_VISITED",),
            evidence_ids=tuple(sorted(capability.evidence_ids)),
            required_evidence=_evidence_for_reasons((
                "CANDIDATE_NOT_VISITED",
            )),
        ))
    return phone_routes


def run_physical_preflight(
    scheduler: object,
    catalog: RuntimeCapabilityCatalog,
    models: Sequence[PhysicalPreflightModel],
    *,
    executable_paths: Mapping[str, Path],
    required_paths: Mapping[str, Path] = MappingProxyType({}),
    library_directories: Mapping[str, Path] = MappingProxyType({}),
    rig_checks: Sequence[PhysicalPreflightCheck] = (),
    trace_requests: Sequence[tuple[Request, str]] = (),
) -> PhysicalPreflightReport:
    """Validate dependencies and scheduler plans without executing inference.

    trace_requests: every (request, model_id) the campaign will submit; each
    shape is judged against the catalog's preallocated context capacity so
    a permanently unsupported request blocks preflight before any paid run.
    """
    required_methods = (
        "generate_automated_candidates",
        "runtime_model_manifest",
        "select_live_vram_desktop_parent",
    )
    if any(not callable(getattr(scheduler, name, None)) for name in required_methods):
        raise PhysicalAdapterError("physical preflight scheduler is invalid")
    if not isinstance(catalog, RuntimeCapabilityCatalog):
        raise PhysicalAdapterError("physical preflight catalog is invalid")
    model_rows = tuple(models)
    if not model_rows or any(
        not isinstance(row, PhysicalPreflightModel) for row in model_rows
    ):
        raise PhysicalAdapterError("physical preflight models are invalid")

    checks = list(rig_checks)
    if any(
        not isinstance(row, PhysicalPreflightCheck) for row in checks
    ) or len({row.check_id for row in checks}) != len(checks):
        raise PhysicalAdapterError("physical preflight rig checks are invalid")

    _check_dependency_paths(
        checks, executable_paths, required_paths, library_directories
    )

    manifests = {row.manifest.model_id: row.manifest for row in model_rows}
    kind_by_device = {
        device.device_id: device.kind
        for device in catalog.placement_profile.devices.values()
    }
    _append_check(
        checks,
        "catalog-artifacts",
        all(
            capability.artifact_sha256 is None
            or capability.artifact_sha256 in {
                manifest.artifact_sha256 for manifest in manifests.values()
            }
            for capability in catalog.composite_executors
        ),
        "catalog composite artifacts match registered models",
    )
    for row in model_rows:
        _check_model_identity(checks, scheduler, catalog, row)
        _check_model_executor_coverage(checks, catalog, row, kind_by_device)
        _check_model_phone_state(checks, catalog, row)

    _check_endpoint_contracts(checks, catalog)

    route_profile_by_id = {
        row.selector_id: row for row in catalog.route_shape_profiles
    }
    transition_by_id = {
        row.transition_id: row for row in catalog.transitions
    }
    desktop_baselines = {}
    phone_routes = []
    for row in model_rows:
        candidates = scheduler.generate_automated_candidates(
            row.request,
            row.manifest.model_id,
            row.snapshot,
            observed_at_us=row.request.arrival_us,
        )
        baseline = candidates.baseline
        _check_candidate_set(checks, catalog, row, candidates, kind_by_device)
        baseline_ready = _check_desktop_baseline(
            checks,
            scheduler,
            catalog,
            row,
            baseline,
            kind_by_device,
            route_profile_by_id,
            transition_by_id,
        )
        if baseline_ready:
            desktop_baselines[row.manifest.model_id] = baseline.candidate_id

        visited, represented_phone_executors = _visited_phone_routes(
            catalog,
            row,
            candidates,
            kind_by_device,
            route_profile_by_id,
            transition_by_id,
        )
        phone_routes.extend(visited)
        phone_routes.extend(_unvisited_phone_routes(
            catalog, row, kind_by_device, represented_phone_executors
        ))

    request_shapes = _check_request_shapes(checks, scheduler, trace_requests)

    if len({row.check_id for row in checks}) != len(checks):
        raise PhysicalAdapterError("physical preflight check id is duplicated")
    return PhysicalPreflightReport(
        catalog_id=catalog.catalog_id,
        energy_boundary_id=catalog.placement_profile.energy_boundary_id,
        checks=tuple(sorted(checks, key=lambda row: row.check_id)),
        desktop_baselines=MappingProxyType(dict(sorted(desktop_baselines.items()))),
        phone_routes=tuple(sorted(
            phone_routes, key=lambda row: (row.model_id, row.route_id)
        )),
        request_shapes=request_shapes,
    )


def _check_request_shapes(
    checks: list[PhysicalPreflightCheck],
    scheduler: object,
    trace_requests: Sequence[tuple[Request, str]],
) -> tuple[Mapping[str, object], ...]:
    """Block when any trace request can never fit a preallocated context."""
    if not trace_requests:
        return ()
    support = getattr(scheduler, "request_shape_support", None)
    if not callable(support):
        raise PhysicalAdapterError("physical preflight scheduler is invalid")
    verdicts = []
    for request, model_id in trace_requests:
        if not isinstance(request, Request) or type(model_id) is not str:
            raise PhysicalAdapterError("physical preflight trace request is invalid")
        verdicts.append(MappingProxyType(dict(
            support(request, model_id).to_json()
        )))
    unsupported = [row for row in verdicts if not row["supported"]]
    _append_check(
        checks,
        "request-shapes",
        not unsupported,
        str(len(verdicts)) + " trace requests fit their preallocated context; "
        + ("none unsupported" if not unsupported else "unsupported: " + ", ".join(
            str(row["request_id"]) + "=" + str(row["reason"]) + "(" + str(row["tokens"]) + " tokens)"
            for row in unsupported
        )),
    )
    return tuple(verdicts)
