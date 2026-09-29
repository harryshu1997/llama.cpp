"""FFN resident envelopes, operator assignments, execution contracts, phone shard binding.

Mixin of ``AutomatedRouteCompiler``; methods moved here verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping, Sequence
from ..model_manifest import ModelManifest, ModelRequestWork
from ..placement import ComputeStep, OperatorNode
from ..plan_contracts.co_helpers import co_helper_declaration
from ..phone_shards import (
    PhoneFfnResidencyLayout,
    PhoneFfnSessionIdentity,
    generate_disjoint_ffn_shard_sets,
    phone_sessions_with_storage_coverage,
)
from ..runtime_capabilities import RuntimeCompositeExecutorCapability
from ..runtime_cost import RuntimeMemoryDemand
from ..runtime_plan import (
    RuntimeExecutionContract,
    RuntimeExecutionPlan,
    RuntimePhoneShard,
    RuntimeResidencyEviction,
    RuntimeOperatorAssignment,
    RuntimeTransitionPlan,
)
from ..types import canonical_sha256
from .remote_resident import RouteRemoteResidentMixin
from .common import (
    RouteGenerationError,
    _FfnResidentEnvelope,
    _Pattern,
)


@dataclass(frozen=True)
class _FfnEnvelopeSettings:
    helper_id: str
    helper_limit: int
    columns: int
    column_quantum: int
    partition_count: int
    allowed_ids: frozenset[str]


class RouteEnvelopeMixin:
    """FFN resident envelopes, operator assignments, execution contracts, phone shard binding."""

    def _ffn_envelope_settings(
        self,
        manifest: ModelManifest,
        coordinator: RuntimeCompositeExecutorCapability,
    ) -> _FfnEnvelopeSettings | None:
        parameters = coordinator.adapter_parameters
        helper_id = coordinator.helper_device_id
        helper_limit = parameters.get(
            "maximum_helper_resident_weight_bytes"
        )
        minimum_quantum = parameters.get("ffn_column_quantum")
        maximum_partitions = parameters.get("ffn_max_runtime_partitions")
        if (
            coordinator.assisted_operator_kind != "ffn"
            or coordinator.baseline_executor_id is None
            or helper_id is None
            or parameters.get("ffn_runtime_control_protocol")
                != "decode-boundary-v1"
            or parameters.get("ffn_weight_buffer_layout") not in {
                "resident-superset", "selected-width"
            }
            or type(helper_limit) is not int
            or helper_limit <= 0
            or type(minimum_quantum) is not int
            or minimum_quantum <= 0
        ):
            return None
        if maximum_partitions is not None and (
            type(maximum_partitions) is not int or maximum_partitions <= 0
        ):
            raise RouteGenerationError(
                "runtime FFN partition capacity is invalid"
            )
        co_helpers = co_helper_declaration(parameters)
        if co_helpers is not None:
            minimum_quantum = max(minimum_quantum, co_helpers.column_quantum(1))
        columns = manifest.feed_forward_length
        required_columns = [columns]
        for fraction in coordinator.split_fractions_ppm:
            scaled = columns * fraction
            if scaled % 1_000_000 == 0:
                required_columns.append(scaled // 1_000_000)
        partition_limit = 16 if maximum_partitions is None else maximum_partitions
        partition_count = 0
        column_quantum = 0
        for count in range(partition_limit, 0, -1):
            if columns % count:
                continue
            quantum = columns // count
            if quantum < minimum_quantum or any(
                value % quantum for value in required_columns
            ):
                continue
            if co_helpers is not None and co_helpers.column_quantum(quantum) != quantum:
                continue
            partition_count = count
            column_quantum = quantum
            break
        if partition_count == 0:
            return None
        allowed_ids = frozenset(coordinator.operator_ids) or frozenset(
            row.operator_id for row in manifest.operators
            if row.kind == "ffn"
        )
        if co_helpers is not None:
            allowed_ids = frozenset(
                row.operator_id for row in manifest.operators
                if row.operator_id in allowed_ids
                and not any(
                    row.layer_id == f"layer:{index}"
                    for helper in co_helpers.helpers
                    for index in helper.layer_indices
                )
            )
        return _FfnEnvelopeSettings(
            helper_id=helper_id,
            helper_limit=helper_limit,
            columns=columns,
            column_quantum=column_quantum,
            partition_count=partition_count,
            allowed_ids=allowed_ids,
        )

    def _portfolio_ffn_envelope(
        self,
        manifest: ModelManifest,
        settings: _FfnEnvelopeSettings,
        sessions: Sequence[object],
    ) -> _FfnResidentEnvelope | None:
        layout = self._phone_residency_layout
        if (
            layout is None
            or not sessions
            or not any(
                row.artifact_sha256 == manifest.artifact_sha256
                for row in layout.shards
            )
        ):
            return None
        relevant_shards = tuple(
            row for row in layout.shards
            if row.artifact_sha256 == manifest.artifact_sha256
        )
        session_by_id = {row.session_id: row for row in sessions}
        if any(row.session_id not in session_by_id for row in layout.shards):
            raise RouteGenerationError(
                "phone residency layout references an absent session"
            )
        relevant_ids = frozenset(
            operator_id
            for shard in relevant_shards
            for operator_id in shard.operator_ids
        )
        operator_ids = tuple(
            row.operator_id for row in manifest.operators
            if row.operator_id in relevant_ids
        )
        if not operator_ids or set(operator_ids) - settings.allowed_ids:
            raise RouteGenerationError(
                "phone residency layout operators differ"
            )
        def runtime_shard(row: object) -> RuntimePhoneShard:
            return RuntimePhoneShard(
                session_id=row.session_id,
                endpoint=row.endpoint,
                layer_mask=row.layer_mask,
                maximum_columns=row.maximum_columns,
                resident_bytes=row.resident_bytes,
                resident_geometry_sha256=row.resident_geometry_sha256,
                operator_plan_sha256=row.operator_plan_sha256,
                artifact_sha256=row.artifact_sha256,
                session_generation=layout.session_generation_by_id.get(
                    row.session_id, 0
                ),
            )
        return _FfnResidentEnvelope(
            operator_ids=operator_ids,
            layer_mask=sum(row.layer_mask for row in relevant_shards),
            columns=settings.columns,
            weight_bytes=layout.resident_bytes,
            column_quantum=settings.column_quantum,
            partition_count=settings.partition_count,
            geometry_sha256=layout.geometry_sha256,
            packing_value=layout.packing_value_for_artifact(
                manifest.artifact_sha256
            ),
            packing_value_kind=layout.objective_kind,
            unavailable_session_ids=tuple(sorted(
                row.session_id for row in layout.shards
                if not session_by_id[row.session_id].ready
            )),
            shards=tuple(runtime_shard(row) for row in relevant_shards),
            residency_shards=tuple(runtime_shard(row) for row in layout.shards),
            changed_session_ids=layout.changed_session_ids,
            replacement_source_identities=layout.replacement_source_identities,
            replacement_source_resident_bytes_by_session=(
                layout.replacement_source_resident_bytes_by_session
            ),
        )

    @staticmethod
    def _shard_set_envelope(
        row: object, settings: _FfnEnvelopeSettings
    ) -> _FfnResidentEnvelope:
        return _FfnResidentEnvelope(
            operator_ids=row.operator_ids,
            layer_mask=sum(shard.layer_mask for shard in row.shards),
            columns=settings.columns,
            weight_bytes=row.resident_bytes,
            column_quantum=settings.column_quantum,
            partition_count=settings.partition_count,
            geometry_sha256=row.geometry_sha256,
            packing_value=row.packing_value,
            packing_value_kind=row.packing_value_kind,
            unavailable_session_ids=row.unavailable_session_ids,
            shards=tuple(
                RuntimePhoneShard(
                    artifact_sha256=shard.artifact_sha256,
                    session_id=shard.session_id,
                    endpoint=shard.endpoint,
                    layer_mask=shard.layer_mask,
                    maximum_columns=shard.maximum_columns,
                    resident_bytes=shard.resident_bytes,
                    resident_geometry_sha256=shard.resident_geometry_sha256,
                    operator_plan_sha256=shard.operator_plan_sha256,
                )
                for shard in row.shards
            ),
        )

    def _session_ffn_envelopes(
        self,
        manifest: ModelManifest,
        coordinator: RuntimeCompositeExecutorCapability,
        settings: _FfnEnvelopeSettings,
        sessions: Sequence[object],
        portfolio: _FfnResidentEnvelope | None,
    ) -> tuple[_FfnResidentEnvelope, ...]:
        helper = self.catalog.executor_by_device[settings.helper_id]
        stored_by_session = self._phone_ffn_shard_storage_by_artifact.get(
            manifest.artifact_sha256
        )
        sessions = phone_sessions_with_storage_coverage(
            sessions, stored_by_session
        )
        pool = self.catalog.placement_profile.memory_pools[
            helper.memory_resource_id
        ]
        phone_wide_limit = min(
            self.catalog.placement_profile.devices[
                settings.helper_id
            ].allocation_limit_bytes,
            pool.capacity_bytes - pool.reserved_bytes,
        )
        kwargs = {
            "phone_wide_limit_bytes": phone_wide_limit,
            "maximum_columns": settings.columns,
            "batch_plan": str(coordinator.adapter_parameters.get(
                "usb_batch_plan", "split-row"
            )),
            "allowed_operator_ids": tuple(sorted(settings.allowed_ids)),
        }
        shard_sets = generate_disjoint_ffn_shard_sets(
            manifest, sessions, **kwargs
        )
        diagnostic_sets = generate_disjoint_ffn_shard_sets(
            manifest, sessions, include_unready=True, **kwargs
        )
        represented_counts = {len(row.shards) for row in shard_sets}
        shard_sets = tuple(sorted(
            (
                *shard_sets,
                *(row for row in diagnostic_sets
                  if len(row.shards) not in represented_counts),
            ),
            key=lambda row: len(row.shards),
        ))
        results = tuple(
            self._shard_set_envelope(row, settings) for row in shard_sets
        )
        if portfolio is None:
            return results
        portfolio_shards = tuple(
            (
                shard.session_id,
                shard.artifact_sha256,
                shard.resident_geometry_sha256,
            )
            for shard in portfolio.shards
        )
        nonmatching = tuple(
            row for row in results
            if tuple(
                (
                    shard.session_id,
                    shard.artifact_sha256,
                    shard.resident_geometry_sha256,
                )
                for shard in row.shards
            ) != portfolio_shards
        )
        layout = self._phone_residency_layout
        if (
            len(portfolio.shards) != len(layout.shards)
            or len(nonmatching) == len(results)
        ):
            return tuple(sorted(
                (*nonmatching, portfolio),
                key=lambda row: (row.session_count, row.geometry_sha256),
            ))
        return results

    def _legacy_ffn_envelope(
        self,
        manifest: ModelManifest,
        settings: _FfnEnvelopeSettings,
    ) -> _FfnResidentEnvelope | None:
        tensor_by_id = manifest.tensor_by_id
        operator_ids = []
        weight_bytes = 0
        layer_mask = 0
        for operator in manifest.operators:
            if (
                operator.kind != "ffn"
                or operator.operator_id not in settings.allowed_ids
            ):
                continue
            prefix, separator, suffix = operator.layer_id.partition(":")
            if prefix != "layer" or separator != ":":
                raise RouteGenerationError(
                    "resident FFN layer identity is invalid"
                )
            try:
                layer_index = int(suffix)
            except ValueError as error:
                raise RouteGenerationError(
                    "resident FFN layer identity is invalid"
                ) from error
            if not 0 <= layer_index < 64:
                continue
            operator_bytes = sum(
                tensor_by_id[tensor_id].nbytes
                for tensor_id in operator.tensor_ids
            )
            if weight_bytes + operator_bytes > settings.helper_limit:
                break
            operator_ids.append(operator.operator_id)
            layer_mask |= 1 << layer_index
            weight_bytes += operator_bytes
        if not operator_ids:
            return None
        return _FfnResidentEnvelope(
            operator_ids=tuple(operator_ids),
            layer_mask=layer_mask,
            columns=settings.columns,
            weight_bytes=weight_bytes,
            column_quantum=settings.column_quantum,
            partition_count=settings.partition_count,
            geometry_sha256=canonical_sha256({
                "artifact_sha256": manifest.artifact_sha256,
                "column_quantum": settings.column_quantum,
                "columns": settings.columns,
                "helper_device_id": settings.helper_id,
                "operator_ids": operator_ids,
                "weight_bytes": weight_bytes,
            }),
        )

    def _ffn_resident_envelopes(
        self,
        manifest: ModelManifest,
        coordinator: RuntimeCompositeExecutorCapability,
        request_work: ModelRequestWork | None = None,
    ) -> tuple[_FfnResidentEnvelope, ...]:
        key = (
            manifest.artifact_sha256,
            coordinator.executor_id,
            self._phone_residency_generation(),
        )
        cached = self._ffn_resident_envelope_cache.get(key)
        if cached is not None:
            return cached
        settings = self._ffn_envelope_settings(manifest, coordinator)
        if settings is None:
            results = ()
        else:
            helper = self.catalog.executor_by_device[settings.helper_id]
            sessions = helper.phone_sessions
            portfolio = self._portfolio_ffn_envelope(
                manifest, settings, sessions
            )
            if sessions:
                results = self._session_ffn_envelopes(
                    manifest,
                    coordinator,
                    settings,
                    sessions,
                    portfolio,
                )
            else:
                legacy = self._legacy_ffn_envelope(manifest, settings)
                results = () if legacy is None else (legacy,)
        self._bounded_cache_store(
            self._ffn_resident_envelope_cache, key, results, 512
        )
        return results

    def _ffn_resident_envelope(
        self,
        manifest: ModelManifest,
        coordinator: RuntimeCompositeExecutorCapability,
        session_count: int = 0,
        request_work: ModelRequestWork | None = None,
    ) -> _FfnResidentEnvelope | None:
        rows = self._ffn_resident_envelopes(
            manifest, coordinator, request_work
        )
        if not rows:
            return None
        if session_count <= 0:
            return rows[0]
        return next(
            (row for row in rows if row.session_count == session_count),
            None,
        )

    def _operator_assignments(
        self,
        manifest: ModelManifest,
        nodes: Sequence[OperatorNode],
        pattern: _Pattern,
    ) -> tuple[RuntimeOperatorAssignment, ...]:
        kernel_signature = tuple(
            tuple(sorted({
                step.kernel_profile_id
                for branch in node.candidates[0].branches
                for step in branch.steps
                if isinstance(step, ComputeStep)
            }))
            for node in nodes
        )
        key = (
            manifest.artifact_sha256,
            pattern.route_key,
            kernel_signature,
        )
        cached = self._operator_assignment_cache.get(key)
        if cached is not None:
            return cached
        result = tuple(
            RuntimeOperatorAssignment(
                operator_id=operator.operator_id,
                operator_kind=operator.kind,
                candidate_id=nodes[index].candidates[0].candidate_id,
                device_ids=tuple(
                    value
                    for value in pattern.assignments[
                        operator.operator_id
                    ][0:2]
                    if value is not None
                ),
                split_axis=(
                    pattern.split_axis
                    if pattern.assignments[operator.operator_id][1]
                    is not None
                    else "none"
                ),
                split_fraction_ppm=pattern.assignments[
                    operator.operator_id
                ][2],
                kernel_profile_ids=tuple(sorted({
                    step.kernel_profile_id
                    for branch in nodes[index].candidates[0].branches
                    for step in branch.steps
                    if isinstance(step, ComputeStep)
                })),
            )
            for index, operator in enumerate(manifest.operators)
        )
        if len(self._operator_assignment_cache) >= 4_096:
            self._operator_assignment_cache.pop(next(iter(
                self._operator_assignment_cache
            )))
        self._operator_assignment_cache[key] = result
        return result

    def _execution_contract(
        self,
        manifest: ModelManifest,
        pattern: _Pattern,
        adapter_parameters: Mapping[str, int | str],
        resident_envelope: _FfnResidentEnvelope | None = None,
        *, snapshot=None,
    ) -> RuntimeExecutionContract:
        remote_resident = RouteRemoteResidentMixin._remote_resident_group(
            self, manifest, pattern
        )
        if remote_resident is not None:
            # the phone runtime parameters belong to the desktop parent itself here
            bound, _reasons = self._remote_resident_owner_status(
                manifest, remote_resident, snapshot
            )
            return RuntimeExecutionContract.desktop(
                remote_resident if bound is None else bound
            )
        phone_device_id = adapter_parameters.get("phone_device_id")
        if phone_device_id is None:
            return RuntimeExecutionContract.desktop()
        if (
            type(phone_device_id) is not str
            or phone_device_id not in self.catalog.executor_by_device
            or phone_device_id not in pattern.device_ids
        ):
            raise RouteGenerationError(
                "phone execution contract lacks its participant"
            )
        phone_endpoint = self.catalog.executor_by_device[
            phone_device_id
        ].endpoint
        operator_kind = pattern.assisted_operator_kind or "whole_model"
        adaptive = (
            operator_kind == "ffn"
            and pattern.assistance_phase == "decode"
            and adapter_parameters.get("ffn_runtime_control_protocol")
                == "decode-boundary-v1"
        )
        if adaptive:
            columns = adapter_parameters.get(
                "ffn_resident_columns",
                adapter_parameters.get("ffn_selected_columns"),
            )
            quantum = adapter_parameters.get("ffn_column_quantum")
            if (
                type(columns) is not int
                or columns <= 0
                or columns > manifest.feed_forward_length
                or type(quantum) is not int
                or quantum <= 0
                or columns % quantum
            ):
                raise RouteGenerationError(
                    "adaptive phone execution width is invalid"
                )
            fractions = (0, *tuple(
                value * 1_000_000 // manifest.feed_forward_length
                for value in range(quantum, columns + 1, quantum)
            ))
            if len(fractions) != len(set(fractions)):
                raise RouteGenerationError(
                    "adaptive phone execution fractions are ambiguous"
                )
            execution_mode = "adaptive-split"
            initial_fraction = 0
        else:
            execution_mode = "static-split"
            fractions = ()
            initial_fraction = (
                pattern.split_fraction_ppm
                if pattern.split_fraction_ppm > 0 else 1_000_000
            )
        batch_plan = adapter_parameters.get(
            "usb_batch_plan",
            "split-row" if operator_kind == "ffn" else "single",
        )
        maximum_batch_size = adapter_parameters.get(
            "ffn_max_tokens",
            adapter_parameters.get("parallel", 1),
        )
        queue_depth = adapter_parameters.get("usb_queue_depth", 1)
        if batch_plan == "single":
            maximum_batch_size = 1
        if (
            type(batch_plan) is not str
            or type(maximum_batch_size) is not int
            or type(queue_depth) is not int
        ):
            raise RouteGenerationError(
                "phone execution batch contract is invalid"
            )
        return RuntimeExecutionContract(
            execution_mode=execution_mode,
            initial_split_fraction_ppm=initial_fraction,
            allowed_adaptive_fractions_ppm=fractions,
            batch_plan=batch_plan,
            maximum_batch_size=maximum_batch_size,
            queue_depth=queue_depth,
            phone_device_id=phone_device_id,
            phone_endpoint=phone_endpoint,
            operator_kind=operator_kind,
            phone_shards=(
                () if resident_envelope is None
                else resident_envelope.shards
            ),
        )

    @staticmethod
    def _bind_phone_shards_to_transition(
        transition: RuntimeTransitionPlan,
        shards: tuple[RuntimePhoneShard, ...],
        changed_session_ids: tuple[str, ...] = (),
        replacement_source_identities: tuple[
            PhoneFfnSessionIdentity, ...
        ] = (),
        replacement_source_resident_bytes_by_session: Mapping[
            str, int
        ] = MappingProxyType({}),
        model_id: str | None = None,
        target_artifact_sha256: str | None = None,
        phone_device_id: str | None = None,
        source_model_id_by_artifact: Mapping[str, str] = MappingProxyType({}),
    ) -> RuntimeTransitionPlan:
        sources = {
            row.session_id: row
            for row in replacement_source_identities
        }
        source_bytes = dict(
            replacement_source_resident_bytes_by_session
        )
        if set(sources) != set(source_bytes):
            raise RouteGenerationError(
                "phone transition replacement source is incomplete"
            )
        target_by_session = {row.session_id: row for row in shards}
        if any(
            session_id not in changed_session_ids
            or (
                session_id in target_by_session
                and target_by_session[session_id].session_generation
                    != source.session_generation + 1
            )
            for session_id, source in sources.items()
        ):
            raise RouteGenerationError(
                "phone transition session generation differs"
            )
        evictions = list(transition.evictions)
        used_eviction_indices = set()
        exact_evictions = []
        for session_id, source in sorted(sources.items()):
            matching_indices = tuple(
                index for index, eviction in enumerate(evictions)
                if index not in used_eviction_indices
                and eviction.device_id == phone_device_id
                and eviction.artifact_sha256 == source.artifact_sha256
                and eviction.session_id is None
            )
            if len(matching_indices) == 1:
                index = matching_indices[0]
                used_eviction_indices.add(index)
                anchor = evictions[index]
                exact_evictions.append(replace(
                    anchor,
                    resident_bytes=source_bytes[session_id],
                    generation=source.session_generation,
                    reclaimable_bytes=(
                        source_bytes[session_id]
                        if anchor.reclaimable_bytes is not None else None
                    ),
                    session_id=session_id,
                    resident_geometry_sha256=(
                        source.resident_geometry_sha256
                    ),
                    operator_plan_sha256=(
                        source.operator_plan_sha256
                    ),
                ))
                continue
            if (
                matching_indices
                or model_id is None
                or target_artifact_sha256 is None
                or phone_device_id is None
                or (
                    source.artifact_sha256 != target_artifact_sha256
                    and source.artifact_sha256
                        not in source_model_id_by_artifact
                )
            ):
                raise RouteGenerationError(
                    "phone transition eviction source is not exact"
                )
            exact_evictions.append(RuntimeResidencyEviction(
                model_id=(
                    model_id
                    if source.artifact_sha256 == target_artifact_sha256
                    else source_model_id_by_artifact[
                        source.artifact_sha256
                    ]
                ),
                artifact_sha256=source.artifact_sha256,
                device_id=phone_device_id,
                resident_bytes=source_bytes[session_id],
                generation=source.session_generation,
                executor_id=transition.executor_id,
                session_id=session_id,
                resident_geometry_sha256=(
                    source.resident_geometry_sha256
                ),
                operator_plan_sha256=source.operator_plan_sha256,
            ))
        return replace(
            transition,
            phone_shards=shards,
            changed_phone_session_ids=changed_session_ids,
            evictions=tuple(
                eviction
                for index, eviction in enumerate(evictions)
                if index not in used_eviction_indices
            ) + tuple(exact_evictions),
        )

    def authorize_phone_residency_plan(
        self,
        plan: RuntimeExecutionPlan,
        layout: PhoneFfnResidencyLayout,
        *,
        model_id: str,
        artifact_sha256: str,
    ) -> RuntimeExecutionPlan:
        """Bind a structural helper plan to exact session generations."""

        if (
            not isinstance(plan, RuntimeExecutionPlan)
            or not isinstance(layout, PhoneFfnResidencyLayout)
        ):
            raise RouteGenerationError(
                "phone plan authorization input is invalid"
            )
        generation_by_id = dict(layout.session_generation_by_id)
        if set(generation_by_id) != {
            row.session_id for row in layout.shards
        }:
            raise RouteGenerationError(
                "phone plan authorization generations are incomplete"
            )
        target_shards = tuple(
            RuntimePhoneShard(
                session_id=row.session_id,
                endpoint=row.endpoint,
                layer_mask=row.layer_mask,
                maximum_columns=row.maximum_columns,
                resident_bytes=row.resident_bytes,
                resident_geometry_sha256=row.resident_geometry_sha256,
                operator_plan_sha256=row.operator_plan_sha256,
                artifact_sha256=row.artifact_sha256,
                session_generation=generation_by_id[row.session_id],
            )
            for row in layout.shards
        )
        artifact_shards = tuple(
            row for row in target_shards
            if row.artifact_sha256 == artifact_sha256
        )
        contract = plan.execution_contract
        structural_identity = lambda row: (
            row.session_id,
            row.artifact_sha256,
            row.endpoint,
            row.layer_mask,
            row.maximum_columns,
            row.resident_bytes,
            row.resident_geometry_sha256,
            row.operator_plan_sha256,
        )
        if (
            not artifact_shards
            or contract.phone_device_id is None
            or tuple(map(structural_identity, contract.phone_shards))
                != tuple(map(structural_identity, artifact_shards))
            or plan.adapter_parameters.get(
                "phone_shard_set_geometry_sha256"
            ) != layout.geometry_sha256
        ):
            raise RouteGenerationError(
                "phone plan authorization differs from its layout"
            )
        phone_transition_count = 0
        transitions = []
        replacement_sources = {
            row.session_id: row
            for row in layout.replacement_source_identities
        }
        replacement_source_bytes = dict(
            layout.replacement_source_resident_bytes_by_session
        )
        for transition in plan.transitions:
            if contract.phone_device_id not in (
                transition.prepares_device_ids
            ):
                transitions.append(transition)
                continue
            phone_transition_count += 1
            session_evictions = {
                row.session_id: row for row in transition.evictions
                if row.session_id is not None
            }
            evictions_are_exact = bool(replacement_sources) and (
                set(session_evictions) == set(replacement_sources)
                and all(
                    session_evictions[session_id].device_id
                        == contract.phone_device_id
                    and session_evictions[session_id].artifact_sha256
                        == source.artifact_sha256
                    and session_evictions[session_id].resident_bytes
                        == replacement_source_bytes[session_id]
                    and session_evictions[session_id].generation
                        == source.session_generation
                    and session_evictions[session_id]
                        .resident_geometry_sha256
                        == source.resident_geometry_sha256
                    and session_evictions[session_id].operator_plan_sha256
                        == source.operator_plan_sha256
                    for session_id, source in replacement_sources.items()
                )
                and not any(
                    row.session_id is None
                    and row.device_id == contract.phone_device_id
                    and row.artifact_sha256 in {
                        source.artifact_sha256
                        for source in replacement_sources.values()
                    }
                    for row in transition.evictions
                )
            )
            if (
                transition.phone_shards == target_shards
                and transition.changed_phone_session_ids
                    == layout.changed_session_ids
                and (
                    not replacement_sources or evictions_are_exact
                )
            ):
                transitions.append(transition)
                continue
            if any(
                row.session_id is not None
                for row in transition.evictions
            ):
                raise RouteGenerationError(
                    "phone plan authorization eviction is already bound"
                )
            transitions.append(self._bind_phone_shards_to_transition(
                transition,
                target_shards,
                layout.changed_session_ids,
                layout.replacement_source_identities,
                layout.replacement_source_resident_bytes_by_session,
                model_id,
                artifact_sha256,
                contract.phone_device_id,
                self._model_id_by_artifact,
            ))
        if phone_transition_count > 1:
            raise RouteGenerationError(
                "phone plan authorization transition is not exact"
            )
        return replace(
            plan,
            transitions=tuple(transitions),
            execution_contract=replace(
                contract, phone_shards=artifact_shards
            ),
        )

    def _cached_execution_plan(
        self,
        *,
        route_id: str,
        pattern: _Pattern,
        residency_variant: str,
        overlap_kind: str,
        assignments: tuple[RuntimeOperatorAssignment, ...],
        transitions: tuple[RuntimeTransitionPlan, ...],
        resources: tuple[str, ...],
        memory_demands: tuple[RuntimeMemoryDemand, ...],
        execution_contract: RuntimeExecutionContract,
        route_profile_id: str | None,
        resource_slots: Mapping[str, int],
        adapter_parameters: Mapping[str, int | str],
        desktop_placement_sha256: str,
    ) -> RuntimeExecutionPlan:
        transition_identity = tuple(
            (
                row.transition_id,
                row.device_id,
                row.source_state,
                row.target_state,
                row.latency_us,
                row.energy_uj,
                row.resource_ids,
                row.maturity,
                tuple(row.resource_slots.items()),
                row.evictions,
                row.executor_id,
                row.prepares_device_ids,
                row.energy_maturity,
                row.phone_shards,
                row.changed_phone_session_ids,
            )
            for row in transitions
        )
        key = (
            route_id,
            pattern.route_family,
            pattern.device_ids,
            pattern.assisted_operator_kind,
            pattern.split_axis,
            pattern.split_fraction_ppm,
            residency_variant,
            overlap_kind,
            id(assignments),
            transition_identity,
            resources,
            memory_demands,
            execution_contract,
            route_profile_id,
            tuple(sorted(resource_slots.items())),
            tuple(sorted(adapter_parameters.items())),
            pattern.baseline_executor_id,
            desktop_placement_sha256,
        )
        cached = self._execution_plan_cache.get(key)
        if cached is not None and cached.operators is assignments:
            return cached
        result = RuntimeExecutionPlan(
            route_id=route_id,
            route_family=pattern.route_family,
            device_ids=pattern.device_ids,
            assisted_operator_kind=pattern.assisted_operator_kind,
            split_axis=pattern.split_axis,
            split_fraction_ppm=pattern.split_fraction_ppm,
            residency_variant=residency_variant,
            overlap_kind=overlap_kind,
            operators=assignments,
            transitions=transitions,
            resource_ids=resources,
            memory_demands=memory_demands,
            execution_contract=execution_contract,
            route_profile_id=route_profile_id,
            resource_slots=resource_slots,
            adapter_parameters=adapter_parameters,
            baseline_executor_id=pattern.baseline_executor_id,
            desktop_placement_sha256=desktop_placement_sha256,
        )
        if len(self._execution_plan_cache) >= 4_096:
            self._execution_plan_cache.pop(next(iter(
                self._execution_plan_cache
            )))
        self._execution_plan_cache[key] = result
        return result
