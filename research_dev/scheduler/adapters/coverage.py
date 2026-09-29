"""Validate per-request candidate coverage in scheduler decision journals."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Mapping, Sequence

from .._internal.model_manifest import ModelManifest
from .._internal.runtime_capabilities import RuntimeCapabilityCatalog
from .contracts import PhysicalAdapterError


@dataclass(frozen=True)
class RuntimeCandidateCoverage:
    request_id: str
    candidate_count: int
    expected_families: tuple[tuple[str, ...], ...]
    observed_families: tuple[tuple[str, ...], ...]
    unavailable_families: tuple[tuple[str, ...], ...]

    def to_json(self) -> dict[str, object]:
        return {
            "candidate_count": self.candidate_count,
            "expected_families": [
                list(row) for row in self.expected_families
            ],
            "observed_families": [
                list(row) for row in self.observed_families
            ],
            "request_id": self.request_id,
            "unavailable_families": [
                list(row) for row in self.unavailable_families
            ],
        }


def _supports_manifest(capability: object, manifest: ModelManifest) -> bool:
    operator_kinds = set(getattr(capability, "operator_kinds", ()))
    quantizations = set(getattr(capability, "supported_quantizations", ()))
    tensors = manifest.tensor_by_id
    return all(
        operator.kind in operator_kinds
        and all(
            "*" in quantizations
            or tensors[tensor_id].quantization in quantizations
            for tensor_id in operator.tensor_ids
        )
        for operator in manifest.operators
    )


def catalog_structural_device_families(
    catalog: RuntimeCapabilityCatalog,
    manifest: ModelManifest,
) -> tuple[tuple[str, ...], ...]:
    """Return device-kind families supported by registered capabilities."""
    if not isinstance(catalog, RuntimeCapabilityCatalog):
        raise PhysicalAdapterError("candidate coverage catalog is invalid")
    if not isinstance(manifest, ModelManifest):
        raise PhysicalAdapterError("candidate coverage manifest is invalid")
    kinds_by_device = {
        row.device_id: row.kind
        for row in catalog.placement_profile.devices.values()
    }
    whole_kinds = {
        kinds_by_device[row.device_id]
        for row in catalog.executors
        if row.supports_whole_model and _supports_manifest(row, manifest)
    }
    families = {(kind,) for kind in whole_kinds}
    layer_capabilities = tuple(
        row for row in catalog.executors
        if row.supports_layer_placement and _supports_manifest(row, manifest)
    )
    layer_count = len({row.layer_id for row in manifest.operators})
    for count in range(2, len(layer_capabilities) + 1):
        if layer_count < count:
            continue
        for capabilities in itertools.combinations(
            layer_capabilities, count
        ):
            kinds = tuple(sorted(
                kinds_by_device[row.device_id] for row in capabilities
            ))
            if len(set(kinds)) != count:
                continue
            common_fractions = set(capabilities[0].layer_fractions_ppm)
            for capability in capabilities[1:]:
                common_fractions.intersection_update(
                    capability.layer_fractions_ppm
                )
            if len(common_fractions) >= count - 1:
                families.add(kinds)
    operator_capabilities = tuple(
        row for row in catalog.executors
        if row.supports_operator_placement
    )
    for base, helper in itertools.permutations(operator_capabilities, 2):
        kinds = tuple(sorted((
            kinds_by_device[base.device_id],
            kinds_by_device[helper.device_id],
        )))
        if len(set(kinds)) != 2:
            continue
        if any(
            all(
                (
                    operator.kind in helper.operator_kinds
                    and all(
                        helper.supports_quantization(
                            manifest.tensor_by_id[tensor_id].quantization
                        )
                        for tensor_id in operator.tensor_ids
                    )
                ) if current.operator_id == operator.operator_id else (
                    current.kind in base.operator_kinds
                    and all(
                        base.supports_quantization(
                            manifest.tensor_by_id[tensor_id].quantization
                        )
                        for tensor_id in current.tensor_ids
                    )
                )
                for current in manifest.operators
            )
            for operator in manifest.operators
        ):
            families.add(kinds)
    for row in catalog.composite_executors:
        if row.artifact_sha256 not in {None, manifest.artifact_sha256}:
            continue
        families.add(tuple(sorted(
            kinds_by_device[device_id]
            for device_id in row.participant_device_ids
        )))
    return tuple(sorted(families))


def _candidate_devices(candidate: Mapping[str, object]) -> tuple[str, ...]:
    executor = candidate.get("executor")
    if type(executor) is dict:
        participants = executor.get("participants")
        if type(participants) is not list or not participants:
            raise PhysicalAdapterError(
                "candidate executor participants are invalid"
            )
        try:
            device_ids = tuple(sorted(
                participant["device_id"] for participant in participants
            ))
        except (KeyError, TypeError) as error:
            raise PhysicalAdapterError(
                "candidate participant device is invalid"
            ) from error
    else:
        details = candidate.get("details")
        if type(details) is not dict:
            raise PhysicalAdapterError(
                "absent candidate lacks structural details"
            )
        raw = details.get("device_ids")
        if type(raw) is not list or not raw:
            raise PhysicalAdapterError(
                "absent candidate lacks structural devices"
            )
        device_ids = tuple(sorted(raw))
    if any(type(row) is not str or not row for row in device_ids):
        raise PhysicalAdapterError("candidate device id is invalid")
    return device_ids


def validate_decision_candidate_coverage(
    catalog: RuntimeCapabilityCatalog,
    records: Sequence[Mapping[str, object]],
    manifests_by_request_id: Mapping[str, ModelManifest],
) -> tuple[RuntimeCandidateCoverage, ...]:
    """Validate candidate costs, rejection facts, and binding per arrival."""
    if not isinstance(catalog, RuntimeCapabilityCatalog):
        raise PhysicalAdapterError("candidate coverage catalog is invalid")
    manifests = dict(manifests_by_request_id)
    if not manifests or any(
        type(request_id) is not str
        or not isinstance(manifest, ModelManifest)
        for request_id, manifest in manifests.items()
    ):
        raise PhysicalAdapterError("candidate coverage models are invalid")
    decisions = [
        row for row in records
        if row.get("event_kind") == "DECISION"
        and row.get("attempt_index") == 0
    ]
    by_request = {}
    for decision in decisions:
        request_ids = decision.get("request_ids")
        if type(request_ids) is not list or len(request_ids) != 1:
            raise PhysicalAdapterError(
                "candidate coverage decision request is invalid"
            )
        request_id = request_ids[0]
        if request_id in by_request:
            raise PhysicalAdapterError(
                "candidate coverage decision is duplicated"
            )
        by_request[request_id] = decision
    if set(by_request) != set(manifests):
        raise PhysicalAdapterError(
            "candidate coverage differs from request arrivals"
        )
    kind_by_device = {
        row.device_id: row.kind
        for row in catalog.placement_profile.devices.values()
    }
    result = []
    for request_id in manifests:
        decision = by_request[request_id]
        candidates = decision.get("candidates")
        selected = decision.get("selected")
        if type(candidates) is not list or not candidates or type(selected) is not dict:
            raise PhysicalAdapterError(
                "candidate coverage decision payload is invalid"
            )
        observed: set[tuple[str, ...]] = set()
        admitted_by_family: dict[tuple[str, ...], int] = {}
        selected_rows = []
        for candidate in candidates:
            if type(candidate) is not dict:
                raise PhysicalAdapterError("candidate cost row is invalid")
            devices = _candidate_devices(candidate)
            try:
                family = tuple(sorted(
                    kind_by_device[device_id] for device_id in devices
                ))
            except KeyError as error:
                raise PhysicalAdapterError(
                    "candidate references an unknown device"
                ) from error
            observed.add(family)
            admitted = candidate.get("admitted") is True
            admitted_by_family[family] = (
                admitted_by_family.get(family, 0) + int(admitted)
            )
            reason = candidate.get("reason")
            readiness = candidate.get("readiness")
            admitted_reasons = {
                "ADMITTED",
                "ADAPTIVE_ENVELOPE_ADMITTED",
                "CALIBRATION_ADMITTED",
            }
            if (
                type(candidate.get("route_id")) is not str
                or type(candidate.get("service_us")) is not int
                or candidate["service_us"] <= 0
                or type(candidate.get("service_upper_us")) is not int
                or candidate["service_upper_us"] < candidate["service_us"]
                or type(reason) is not str
                or not reason
                or readiness not in {"ABSENT", "NOT_READY", "READY"}
                or admitted != (reason in admitted_reasons)
                or (readiness == "ABSENT" and admitted)
            ):
                raise PhysicalAdapterError(
                    "candidate cost or rejection record is invalid"
                )
            details = candidate.get("details")
            if type(details) is not dict or tuple(sorted(
                details.get("device_ids", ())
            )) != devices:
                raise PhysicalAdapterError(
                    "candidate structural details differ from binding"
                )
            if candidate.get("selection_status") == "SELECTED":
                selected_rows.append(candidate)
        expected = set(catalog_structural_device_families(
            catalog, manifests[request_id]
        ))
        if not expected.issubset(observed):
            missing = sorted(expected - observed)
            raise PhysicalAdapterError(
                "candidate families are absent for " + request_id + ": "
                + repr(missing)
            )
        if len(selected_rows) != 1:
            raise PhysicalAdapterError(
                "candidate selection record is not unique"
            )
        selected_row = selected_rows[0]
        selected_executor = selected.get("executor")
        if (
            selected_row.get("route_id") != selected.get("route_id")
            or selected_row.get("executor") != selected_executor
        ):
            raise PhysicalAdapterError(
                "selected candidate differs from selected binding"
            )
        result.append(RuntimeCandidateCoverage(
            request_id=request_id,
            candidate_count=len(candidates),
            expected_families=tuple(sorted(expected)),
            observed_families=tuple(sorted(observed)),
            unavailable_families=tuple(sorted(
                family for family in observed
                if admitted_by_family.get(family, 0) == 0
            )),
        ))
    return tuple(result)
