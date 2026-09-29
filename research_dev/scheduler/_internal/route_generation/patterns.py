"""Supported device patterns: whole, layer, operator, and coordinated placements.

Mixin of ``AutomatedRouteCompiler``; methods moved here verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from types import MappingProxyType
from typing import Mapping
from ..model_manifest import ModelManifest, ModelRequestWork
from ..plan_contracts.co_helpers import co_helper_declaration
from ..runtime_capabilities import RuntimeExecutorCapability
from ..types import canonical_sha256
from .common import (
    RouteGenerationError,
    _ceil_div,
    _Pattern,
)
from .remote_resident import REMOTE_RESIDENT_FFN_PARAMETER, parse_remote_resident_declaration


@dataclass(frozen=True)
class _CoordinatedPatternContext:
    kind: str
    assistance_phase: str
    helper: RuntimeExecutorCapability
    base_assignments: Mapping[str, tuple[str, None, int]]
    fractions: tuple[int, ...]
    ordered_ids: tuple[str, ...]
    base_subsets: tuple[frozenset[str], ...]
    resident_envelopes: tuple[object, ...]
    helper_limit: int | object
    # FFN operator -> static co-helper phone that owns its layer
    co_helper_devices: Mapping[str, str] = MappingProxyType({})


class RoutePatternMixin:
    """Supported device patterns: whole, layer, operator, and coordinated placements."""

    def _supports_model(
        self,
        capability: RuntimeExecutorCapability,
        manifest: ModelManifest,
    ) -> bool:
        return all(
            self._supports_operator(capability, manifest, operator)
            for operator in manifest.operators
        )

    @staticmethod
    def _supports_operator(
        capability: RuntimeExecutorCapability,
        manifest: ModelManifest,
        operator,
    ) -> bool:
        if operator.kind not in capability.operator_kinds:
            return False
        tensors = manifest.tensor_by_id
        return all(
            capability.supports_quantization(
                tensors[tensor_id].quantization
            )
            for tensor_id in operator.tensor_ids
        )

    def _whole_patterns(self, manifest: ModelManifest) -> list[_Pattern]:
        rows = []
        for capability in self.catalog.executors:
            if not capability.supports_whole_model or not self._supports_model(
                capability, manifest
            ):
                continue
            assignments = {
                operator.operator_id: (capability.device_id, None, 0)
                for operator in manifest.operators
            }
            rows.append(_Pattern(
                route_key=f"whole:{capability.device_id}",
                route_family="whole_model",
                assignments=assignments,
                assisted_operator_kind=None,
                split_axis="none",
                split_fraction_ppm=0,
                overlap_kind="none",
                coordinator_device_id=capability.device_id,
            ))
        return rows

    @staticmethod
    def _layer_assignment(
        manifest: ModelManifest,
        devices: tuple[str, ...],
        fractions: tuple[int, ...],
    ) -> Mapping[str, tuple[str, None, int]]:
        layer_ids = tuple(dict.fromkeys(
            operator.layer_id for operator in manifest.operators
        ))
        count = len(layer_ids)
        boundaries = [
            min(count - 1, max(1, _ceil_div(count * value, 1_000_000)))
            for value in fractions
        ]
        layer_index = {
            layer_id: index for index, layer_id in enumerate(layer_ids)
        }
        assignments = {}
        for operator in manifest.operators:
            index = layer_index[operator.layer_id]
            segment = sum(index >= boundary for boundary in boundaries)
            assignments[operator.operator_id] = (devices[segment], None, 0)
        return assignments

    @staticmethod
    def _uses_every_device(
        assignments: Mapping[str, tuple[str, None, int]],
        devices: tuple[str, ...],
    ) -> bool:
        return {
            assignment[0] for assignment in assignments.values()
        } == set(devices)

    def _layer_patterns(self, manifest: ModelManifest) -> list[_Pattern]:
        capabilities = [
            row for row in self.catalog.executors
            if row.supports_layer_placement and self._supports_model(row, manifest)
        ]
        rows = []
        for device_count in range(2, len(capabilities) + 1):
            for permutation in itertools.permutations(
                capabilities, device_count
            ):
                device_ids = tuple(row.device_id for row in permutation)
                common = set(permutation[0].layer_fractions_ppm)
                for capability in permutation[1:]:
                    common.intersection_update(
                        capability.layer_fractions_ppm
                    )
                if len(common) < device_count - 1:
                    continue
                seen_assignments = set()
                for cuts in itertools.combinations(
                    sorted(common), device_count - 1
                ):
                    assignments = self._layer_assignment(
                        manifest, device_ids, cuts
                    )
                    signature = tuple(
                        assignments[operator.operator_id][0]
                        for operator in manifest.operators
                    )
                    if (
                        signature in seen_assignments
                        or not self._uses_every_device(
                            assignments, device_ids
                        )
                    ):
                        continue
                    seen_assignments.add(signature)
                    rows.append(_Pattern(
                        route_key=(
                            "layers:" + "+".join(device_ids) + ":"
                            + "+".join(str(value) for value in cuts)
                        ),
                        route_family="layer_placement",
                        assignments=assignments,
                        assisted_operator_kind=None,
                        split_axis="none",
                        split_fraction_ppm=0,
                        overlap_kind="queue_overflow",
                        coordinator_device_id=permutation[0].device_id,
                    ))
        return rows

    def _operator_patterns(self, manifest: ModelManifest) -> list[_Pattern]:
        capabilities = [
            row for row in self.catalog.executors
            if row.supports_operator_placement
        ]
        kinds = tuple(sorted({row.kind for row in manifest.operators}))
        rows = []
        for base, helper in itertools.permutations(capabilities, 2):
            for kind in kinds:
                if kind == "kv_cache" and not helper.supports_kv_cache:
                    continue
                if not all(
                    self._supports_operator(
                        helper if operator.kind == kind else base,
                        manifest,
                        operator,
                    )
                    for operator in manifest.operators
                ):
                    continue
                assignments = {
                    operator.operator_id: (
                        helper.device_id if operator.kind == kind else base.device_id,
                        None,
                        0,
                    )
                    for operator in manifest.operators
                }
                rows.append(_Pattern(
                    route_key=f"offload:{base.device_id}+{helper.device_id}:{kind}",
                    route_family="operator_offload",
                    assignments=assignments,
                    assisted_operator_kind=kind,
                    split_axis="none",
                    split_fraction_ppm=0,
                    overlap_kind="serial_operator_offload",
                    coordinator_device_id=base.device_id,
                ))
            if (
                not base.supports_split_coordinator
                or not helper.supports_split_helper
                or not self._supports_model(base, manifest)
            ):
                continue
            axes = tuple(sorted(set(base.split_axes) & set(helper.split_axes)))
            fractions = tuple(sorted(
                set(base.split_fractions_ppm) & set(helper.split_fractions_ppm)
            ))
            for kind in ("attention_projection", "ffn", "lm_head"):
                assisted = tuple(
                    operator for operator in manifest.operators
                    if operator.kind == kind
                )
                if not assisted or not all(
                    self._supports_operator(helper, manifest, operator)
                    for operator in assisted
                ):
                    continue
                for axis, fraction in itertools.product(axes, fractions):
                    assignments = {
                        operator.operator_id: (
                            base.device_id,
                            helper.device_id if operator.kind == kind else None,
                            fraction if operator.kind == kind else 0,
                        )
                        for operator in manifest.operators
                    }
                    rows.append(_Pattern(
                        route_key=(
                            f"split:{base.device_id}+{helper.device_id}:"
                            f"{kind}:{axis}:{fraction}"
                        ),
                        route_family="operator_split",
                        assignments=assignments,
                        assisted_operator_kind=kind,
                        split_axis=axis,
                        split_fraction_ppm=fraction,
                        overlap_kind="parallel_operator_branches",
                        coordinator_device_id=base.device_id,
                    ))
        return rows

    def _explicit_coordinated_pattern(
        self,
        manifest: ModelManifest,
        coordinator: object,
        operator_by_id: Mapping[str, object],
    ) -> _Pattern | None:
        placements = coordinator.operator_placements
        placement_by_id = {row.operator_id: row for row in placements}
        if set(placement_by_id) != set(operator_by_id):
            return None
        if any(
            operator_by_id[row.operator_id].kind
                != coordinator.assisted_operator_kind
            for row in placements if row.assisted
        ):
            return None
        if any(
            not self._supports_operator(
                self.catalog.executor_by_device[device_id],
                manifest,
                operator_by_id[row.operator_id],
            )
            for row in placements
            for device_id in (row.primary_device_id, row.helper_device_id)
            if device_id is not None
        ):
            return None
        assignments = {
            row.operator_id: (
                row.primary_device_id,
                row.helper_device_id,
                row.split_fraction_ppm,
            )
            for row in placements
        }
        fractions = {
            row.split_fraction_ppm
            for row in placements if row.helper_device_id is not None
        }
        if len(fractions) > 1:
            return None
        overlap_kind = (
            "serial_layer_islands"
            if coordinator.route_family == "layer_placement"
            else "serial_operator_offload"
            if coordinator.route_family == "operator_offload"
            else "parallel_operator_branches"
        )
        owner_devices = ()
        declaration = coordinator.adapter_parameters.get(REMOTE_RESIDENT_FFN_PARAMETER)
        if declaration is not None:
            group = parse_remote_resident_declaration(declaration)
            owners = tuple(self._phone_session_capability(row.session_id) for row in group.sessions)
            if any(row is None for row in owners):
                raise RouteGenerationError("remote-resident owner session is unknown")
            owner_devices = tuple(sorted({row[0] for row in owners}))
        return _Pattern(
            route_key="coordinated:" + coordinator.executor_id,
            route_family=coordinator.route_family,
            assignments=assignments,
            assisted_operator_kind=coordinator.assisted_operator_kind,
            split_axis=coordinator.split_axis,
            split_fraction_ppm=next(iter(fractions), 0),
            overlap_kind=overlap_kind,
            coordinator_device_id=coordinator.coordinator_device_id,
            coordinator_executor_id=coordinator.executor_id,
            resident_owner_device_ids=owner_devices,
        )

    def _layer_coordinated_pattern(
        self,
        manifest: ModelManifest,
        coordinator: object,
        capabilities: tuple[RuntimeExecutorCapability, ...],
    ) -> _Pattern | None:
        if not all(
            capability.supports_layer_placement
            and self._supports_model(capability, manifest)
            for capability in capabilities
        ):
            return None
        assignments = self._layer_assignment(
            manifest,
            coordinator.participant_device_ids,
            coordinator.layer_fractions_ppm,
        )
        return _Pattern(
            route_key="coordinated:" + coordinator.executor_id,
            route_family=coordinator.route_family,
            assignments=assignments,
            assisted_operator_kind=None,
            split_axis="none",
            split_fraction_ppm=0,
            overlap_kind="queue_overflow",
            coordinator_device_id=coordinator.coordinator_device_id,
            coordinator_executor_id=coordinator.executor_id,
        )

    def _coordinated_pattern_context(
        self,
        manifest: ModelManifest,
        coordinator: object,
        capabilities: tuple[RuntimeExecutorCapability, ...],
        operator_by_id: Mapping[str, object],
        request_work: ModelRequestWork | None,
    ) -> _CoordinatedPatternContext | None:
        kind = coordinator.assisted_operator_kind
        assert kind is not None
        protocol = coordinator.adapter_parameters.get(
            "ffn_runtime_control_protocol"
        )
        if protocol is None:
            assistance_phase = "all"
        elif (
            protocol == "decode-boundary-v1"
            and kind == "ffn"
            and coordinator.baseline_executor_id is not None
        ):
            assistance_phase = "decode"
        else:
            raise RouteGenerationError(
                "runtime operator control protocol is unsupported"
            )
        selected_ids = (
            frozenset(coordinator.operator_ids)
            if coordinator.operator_ids else
            frozenset(
                row.operator_id for row in manifest.operators
                if row.kind == kind
            )
        )
        co_helper_devices = self._co_helper_devices(
            manifest, coordinator, operator_by_id
        )
        if co_helper_devices is None:
            return None
        selected_ids = selected_ids - set(co_helper_devices)
        if (
            not selected_ids
            or set(selected_ids) - set(operator_by_id)
            or any(operator_by_id[value].kind != kind for value in selected_ids)
        ):
            return None
        if coordinator.baseline_executor_id is not None:
            baseline = self.catalog.composite_executor_by_id[
                coordinator.baseline_executor_id
            ]
            baseline_by_id = {
                row.operator_id: row for row in baseline.operator_placements
            }
            helper = self.catalog.executor_by_device[
                coordinator.helper_device_id
            ]
            if set(baseline_by_id) != set(operator_by_id) or any(
                row.helper_device_id is not None
                for row in baseline_by_id.values()
            ):
                return None
            base_assignments = {
                operator_id: (row.primary_device_id, None, 0)
                for operator_id, row in baseline_by_id.items()
            }
        else:
            base, helper = capabilities
            base_assignments = {
                row.operator_id: (base.device_id, None, 0)
                for row in manifest.operators
            }
        if not all(
            self._supports_operator(
                self.catalog.executor_by_device[
                    base_assignments[row.operator_id][0]
                ],
                manifest,
                row,
            )
            for row in manifest.operators
        ) or not all(
            self._supports_operator(helper, manifest, operator_by_id[value])
            for value in selected_ids
        ):
            return None
        fractions = (
            (0,)
            if coordinator.route_family == "operator_offload"
            else coordinator.split_fractions_ppm
        )
        ordered_ids = tuple(
            row.operator_id for row in manifest.operators
            if row.operator_id in selected_ids
        )
        base_subsets = [frozenset(ordered_ids)]
        if coordinator.baseline_executor_id is not None:
            base_subsets = [frozenset((value,)) for value in ordered_ids]
            for numerator in (1, 2, 3, 4):
                count = _ceil_div(len(ordered_ids) * numerator, 4)
                base_subsets.append(frozenset(ordered_ids[:count]))
            base_subsets = list(dict.fromkeys(base_subsets))
        envelopes = self._ffn_resident_envelopes(
            manifest, coordinator, request_work
        )
        if helper.phone_sessions and not envelopes:
            return None
        helper_limit = coordinator.adapter_parameters.get(
            "maximum_helper_resident_weight_bytes"
        )
        if helper_limit is None:
            helper_device = self.catalog.placement_profile.devices[
                helper.device_id
            ]
            helper_pool = self.catalog.placement_profile.memory_pools[
                helper_device.memory_pool_id
            ]
            helper_limit = min(
                helper_device.allocation_limit_bytes,
                helper_pool.capacity_bytes - helper_pool.reserved_bytes,
            )
        return _CoordinatedPatternContext(
            kind=kind,
            assistance_phase=assistance_phase,
            helper=helper,
            base_assignments=base_assignments,
            fractions=fractions,
            ordered_ids=ordered_ids,
            base_subsets=tuple(base_subsets),
            resident_envelopes=tuple(envelopes),
            helper_limit=helper_limit,
            co_helper_devices=co_helper_devices,
        )

    def _co_helper_devices(
        self,
        manifest: ModelManifest,
        coordinator: object,
        operator_by_id: Mapping[str, object],
    ) -> Mapping[str, str] | None:
        """FFN operators owned by static co-helper phones; None when one cannot serve them."""
        declaration = co_helper_declaration(coordinator.adapter_parameters)
        if declaration is None:
            return MappingProxyType({})
        if (
            coordinator.assisted_operator_kind != "ffn"
            or coordinator.baseline_executor_id is None
            or any(
                device_id not in coordinator.participant_device_ids
                or device_id in {
                    coordinator.coordinator_device_id,
                    coordinator.helper_device_id,
                }
                for device_id in declaration.device_ids
            )
        ):
            raise RouteGenerationError(
                "co-helper phones are not participants of their route"
            )
        result = {}
        covered = 0
        for operator in manifest.operators:
            prefix, separator, index = operator.layer_id.partition(":")
            if (
                operator.kind != "ffn"
                or prefix != "layer"
                or not separator
                or not index.isdecimal()
            ):
                continue
            row = declaration.helper_for_layer(int(index))
            if row is not None:
                result[operator.operator_id] = row.device_id
                covered |= 1 << int(index)
        executors = self.catalog.executor_by_device
        if covered != declaration.layer_mask or any(
            device_id not in executors
            or not self._supports_operator(
                executors[device_id], manifest, operator_by_id[operator_id]
            )
            for operator_id, device_id in result.items()
        ):
            return None
        return MappingProxyType(result)

    def _fitting_helper_prefix(
        self,
        manifest: ModelManifest,
        coordinator: object,
        context: _CoordinatedPatternContext,
        fraction: int,
        operator_by_id: Mapping[str, object],
    ) -> frozenset[str]:
        if type(context.helper_limit) is not int or context.helper_limit <= 0:
            return frozenset()
        tensor_by_id = manifest.tensor_by_id
        helper_bytes = 0
        fitting_prefix = []
        for operator_id in context.ordered_ids:
            operator_bytes = sum(
                tensor_by_id[tensor_id].nbytes
                for tensor_id in operator_by_id[operator_id].tensor_ids
            )
            required_bytes = (
                operator_bytes
                if coordinator.route_family == "operator_offload"
                else self._fraction(operator_bytes, fraction)[1]
            )
            if helper_bytes + required_bytes > context.helper_limit:
                break
            helper_bytes += required_bytes
            fitting_prefix.append(operator_id)
        return frozenset(fitting_prefix)

    def _coordinated_assisted_pattern(
        self,
        coordinator: object,
        context: _CoordinatedPatternContext,
        resident_envelope: object | None,
        fraction: int,
        assisted_ids: frozenset[str],
    ) -> _Pattern:
        assignments = dict(context.base_assignments)
        for operator_id, helper_device_id in (
            *((value, context.helper.device_id) for value in assisted_ids),
            *context.co_helper_devices.items(),
        ):
            base_device_id = context.base_assignments[operator_id][0]
            assignments[operator_id] = (
                (helper_device_id, None, 0)
                if coordinator.route_family == "operator_offload"
                else (base_device_id, helper_device_id, fraction)
            )
        route_key = "coordinated:" + coordinator.executor_id
        if fraction:
            route_key += ":" + str(fraction)
        if resident_envelope is not None and resident_envelope.session_count > 1:
            route_key += ":sessions:" + str(resident_envelope.session_count)
        if resident_envelope is not None:
            route_key += ":geometry:" + resident_envelope.geometry_sha256[7:19]
        if assisted_ids != frozenset(context.ordered_ids):
            route_key += ":ops:" + canonical_sha256(sorted(assisted_ids))[7:19]
        return _Pattern(
            route_key=route_key,
            route_family=coordinator.route_family,
            assignments=assignments,
            assisted_operator_kind=context.kind,
            split_axis=coordinator.split_axis,
            split_fraction_ppm=fraction,
            overlap_kind=(
                "serial_operator_offload"
                if coordinator.route_family == "operator_offload"
                else "parallel_operator_branches"
            ),
            coordinator_device_id=coordinator.coordinator_device_id,
            coordinator_executor_id=coordinator.executor_id,
            baseline_executor_id=coordinator.baseline_executor_id,
            desktop_assignments={
                key: value[0] for key, value in context.base_assignments.items()
            },
            assistance_phase=context.assistance_phase,
            resident_envelope=(
                resident_envelope is not None
                and assisted_ids == frozenset(resident_envelope.operator_ids)
            ),
            phone_session_count=(
                0 if resident_envelope is None
                else resident_envelope.session_count
            ),
            phone_resident_envelope=resident_envelope,
        )

    def _coordinated_assisted_patterns(
        self,
        manifest: ModelManifest,
        coordinator: object,
        context: _CoordinatedPatternContext,
        operator_by_id: Mapping[str, object],
    ) -> list[_Pattern]:
        rows = []
        envelope_options = (
            context.resident_envelopes
            if context.resident_envelopes else (None,)
        )
        for resident_envelope in envelope_options:
            if context.co_helper_devices and resident_envelope is None:
                continue
            fractions = context.fractions
            subsets = list(context.base_subsets)
            if resident_envelope is not None:
                subsets = [frozenset(resident_envelope.operator_ids)]
                if coordinator.route_family == "operator_split":
                    fractions = (max(context.fractions),)
            for fraction in fractions:
                fraction_subsets = list(subsets)
                if resident_envelope is None:
                    fitting = self._fitting_helper_prefix(
                        manifest,
                        coordinator,
                        context,
                        fraction,
                        operator_by_id,
                    )
                    if fitting:
                        fraction_subsets.append(fitting)
                rows.extend(
                    self._coordinated_assisted_pattern(
                        coordinator,
                        context,
                        resident_envelope,
                        fraction,
                        assisted_ids,
                    )
                    for assisted_ids in dict.fromkeys(fraction_subsets)
                )
        return rows

    def _coordinated_patterns(
        self,
        manifest: ModelManifest,
        request_work: ModelRequestWork | None = None,
    ) -> list[_Pattern]:
        rows = []
        operator_by_id = {
            operator.operator_id: operator for operator in manifest.operators
        }
        for coordinator in self.catalog.composite_executors:
            if (
                coordinator.artifact_sha256 is not None
                and coordinator.artifact_sha256 != manifest.artifact_sha256
            ):
                continue
            capabilities = tuple(
                self.catalog.executor_by_device[device_id]
                for device_id in coordinator.participant_device_ids
            )
            if coordinator.operator_placements:
                pattern = self._explicit_coordinated_pattern(
                    manifest, coordinator, operator_by_id
                )
                if pattern is not None:
                    rows.append(pattern)
                continue
            if coordinator.route_family == "layer_placement":
                pattern = self._layer_coordinated_pattern(
                    manifest, coordinator, capabilities
                )
                if pattern is not None:
                    rows.append(pattern)
                continue
            context = self._coordinated_pattern_context(
                manifest,
                coordinator,
                capabilities,
                operator_by_id,
                request_work,
            )
            if context is not None:
                rows.extend(self._coordinated_assisted_patterns(
                    manifest, coordinator, context, operator_by_id
                ))
        return rows

    def _patterns(
        self,
        manifest: ModelManifest,
        request_work: ModelRequestWork | None = None,
    ) -> tuple[_Pattern, ...]:
        cache_key = (
            manifest.artifact_sha256,
            self._phone_residency_generation(),
        )
        with self._cache_lock:
            cached = self._pattern_cache.get(cache_key)
            if cached is not None:
                return cached
            rows = (
                self._whole_patterns(manifest)
                + self._layer_patterns(manifest)
                + self._operator_patterns(manifest)
                + self._coordinated_patterns(manifest, request_work)
            )
            by_key = {row.route_key: row for row in rows}
            if len(by_key) != len(rows):
                raise RouteGenerationError(
                    "generated route keys are duplicated"
                )
            result = tuple(by_key[key] for key in sorted(by_key))
            self._bounded_cache_store(
                self._pattern_cache, cache_key, result, 512
            )
            self._bounded_cache_store(
                self._pattern_by_key_cache, cache_key, by_key, 512
            )
            return result

    def _patterns_by_key(
        self, manifest: ModelManifest
    ) -> Mapping[str, _Pattern]:
        cache_key = (
            manifest.artifact_sha256,
            self._phone_residency_generation(),
        )
        with self._cache_lock:
            cached = self._pattern_by_key_cache.get(cache_key)
            if cached is not None:
                return cached
            patterns = self._patterns(manifest)
            result = {row.route_key: row for row in patterns}
            self._bounded_cache_store(
                self._pattern_by_key_cache, cache_key, result, 512
            )
            return result
