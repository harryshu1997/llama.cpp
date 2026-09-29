"""Phone residency layout state, transition latency, and residency route evidence.

Mixin of ``AutomatedRouteCompiler``; methods moved here verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
from ..model_manifest import ModelManifest
from ..phone_shards import (
    PhoneFfnResidencyDemand,
    PhoneFfnResidencyLayout,
    PhoneFfnShardStorageMetadata,
)
from ..types import canonical_sha256
from ..runtime_capabilities import RuntimePhoneSessionCapability
from ..runtime_plan import AutomatedCandidateSet, AutomatedRouteCandidate
from .common import (
    RouteGenerationError,
    PHONE_RESIDENCY_EVIDENCE_NEUTRAL_REJECTIONS,
    _PhoneResidencyRouteEvidence,
)


@dataclass(frozen=True)
class _EligiblePhoneEvidence:
    normalized_benefit: int
    route_benefit: int
    operator_ids: tuple[str, ...]
    source_route_id: str
    parent_route_id: str
    benefit_value_kind: str
    energy_evidence: str
    transition_energy_upper_uj: int
    transition_energy_upper_uj_by_session: Mapping[str, int]
    transition_energy_aggregation: str
    transition_latency_upper_us: int | None
    normalization_source: str

    @property
    def rank(self) -> tuple[int, int, int, str]:
        return (
            self.normalized_benefit,
            self.route_benefit,
            len(self.operator_ids),
            self.source_route_id,
        )


class PhoneResidencyEvidenceMixin:
    """Phone residency layout state, transition latency, and residency route evidence."""

    @property
    def phone_residency_layout(self) -> PhoneFfnResidencyLayout | None:
        return self._phone_residency_layout

    def set_phone_residency_layout(
        self, layout: PhoneFfnResidencyLayout | None
    ) -> bool:
        """Publish one queue-selected structural phone portfolio."""

        if layout is not None and not isinstance(
            layout, PhoneFfnResidencyLayout
        ):
            raise RouteGenerationError(
                "phone residency layout is invalid"
            )
        with self._cache_lock:
            previous = self._phone_residency_layout
            def execution_identity(value):
                if value is None:
                    return None
                return (
                    value.geometry_sha256,
                    tuple(value.changed_session_ids),
                    tuple(sorted(value.session_generation_by_id.items())),
                    tuple(value.replacement_source_identities),
                    tuple(sorted(
                        value.replacement_source_resident_bytes_by_session
                            .items()
                    )),
                )

            if execution_identity(previous) == execution_identity(layout):
                self._phone_residency_layout = layout
                return False
            affected = {
                row.artifact_sha256
                for source in (previous, layout)
                if source is not None
                for row in source.shards
            }
            self._phone_residency_layout = layout
            self._pattern_cache = {
                key: value for key, value in self._pattern_cache.items()
                if key[0] not in affected
            }
            self._pattern_by_key_cache = {
                key: value
                for key, value in self._pattern_by_key_cache.items()
                if key[0] not in affected
            }
            self._ffn_resident_envelope_cache = {
                key: value
                for key, value in self._ffn_resident_envelope_cache.items()
                if key[0] not in affected
            }
            self._execution_plan_cache.clear()
            self._placement_cache.clear()
            self._rough_compiler.invalidate_artifacts(tuple(affected))
            return True

    def set_phone_ffn_shard_storage(
        self, records: tuple[PhoneFfnShardStorageMetadata, ...]
    ) -> bool:
        """Constrain generated session assignments to deployed shard files."""

        rows = tuple(records)
        if any(
            not isinstance(row, PhoneFfnShardStorageMetadata)
            for row in rows
        ):
            raise RouteGenerationError(
                "phone FFN shard storage metadata is invalid"
            )
        keys = tuple(
            (row.parent_artifact_sha256, row.session_id) for row in rows
        )
        if len(keys) != len(set(keys)):
            raise RouteGenerationError(
                "phone FFN shard storage sessions are duplicated"
            )
        ordered = tuple(sorted(
            rows,
            key=lambda row: (
                row.parent_artifact_sha256, row.session_id
            ),
        ))
        identity = (
            None if not ordered else
            canonical_sha256([row.to_json() for row in ordered])
        )
        if identity == self._phone_ffn_shard_storage_sha256:
            return False
        previous = self._phone_ffn_shard_storage_by_artifact
        affected = {
            *previous,
            *(row.parent_artifact_sha256 for row in ordered),
        }
        grouped: dict[
            str, dict[str, PhoneFfnShardStorageMetadata]
        ] = {}
        for row in ordered:
            grouped.setdefault(row.parent_artifact_sha256, {})[
                row.session_id
            ] = row
        self._phone_ffn_shard_storage_by_artifact = MappingProxyType({
            artifact: MappingProxyType(dict(sorted(by_session.items())))
            for artifact, by_session in sorted(grouped.items())
        })
        self._phone_ffn_shard_storage_sha256 = identity
        self._pattern_cache = {
            key: value for key, value in self._pattern_cache.items()
            if key[0] not in affected
        }
        self._pattern_by_key_cache = {
            key: value for key, value in self._pattern_by_key_cache.items()
            if key[0] not in affected
        }
        self._ffn_resident_envelope_cache = {
            key: value
            for key, value in self._ffn_resident_envelope_cache.items()
            if key[0] not in affected
        }
        self._execution_plan_cache.clear()
        self._placement_cache.clear()
        self._rough_compiler.invalidate_artifacts(tuple(affected))
        return True

    def _phone_residency_generation(self) -> str:
        layout = self._phone_residency_layout
        layout_identity = "none" if layout is None else layout.geometry_sha256
        storage_identity = self._phone_ffn_shard_storage_sha256
        if storage_identity is None:
            return layout_identity
        return canonical_sha256({
            "layout": layout_identity,
            "storage": storage_identity,
        })

    def _phone_transition_latency_upper_us(
        self,
        candidate: AutomatedRouteCandidate,
        artifact_sha256: str,
        phone_device_id: str,
        resident_bytes: int,
    ) -> int | None:
        planned = tuple(
            transition
            for transition in candidate.plan.transitions
            if phone_device_id in transition.prepares_device_ids
        )
        if planned:
            latency_us = sum(row.latency_us for row in planned)
        else:
            capabilities = tuple(
                row for row in self.catalog.transitions
                if phone_device_id in row.prepares_device_ids
                and row.target_state == "hot"
                and row.maturity == "QUALIFIED"
                and row.artifact_sha256 in {None, artifact_sha256}
                and row.executor_id in {
                    None, candidate.binding.executor_id,
                }
            )
            exact_artifact = tuple(
                row for row in capabilities
                if row.artifact_sha256 == artifact_sha256
            )
            if exact_artifact:
                capabilities = exact_artifact
            exact_executor = tuple(
                row for row in capabilities
                if row.executor_id == candidate.binding.executor_id
            )
            if exact_executor:
                capabilities = exact_executor
            if not capabilities:
                return None
            latency_us = max(
                row.cost(resident_bytes)[0] for row in capabilities
            )
        return (
            latency_us * self.catalog.maximum_latency_ppm + 999_999
        ) // 1_000_000

    @staticmethod
    def _ffn_operator_by_layer(
        manifest: ModelManifest,
    ) -> dict[int, str]:
        result = {}
        for operator in manifest.operators:
            if operator.kind != "ffn":
                continue
            prefix, separator, raw_index = operator.layer_id.partition(":")
            try:
                layer_index = int(raw_index)
            except ValueError:
                continue
            if (
                prefix == "layer"
                and separator == ":"
                and 0 <= layer_index < 64
            ):
                result[layer_index] = operator.operator_id
        return result

    @staticmethod
    def _phone_candidate_energy_kind(
        candidate: AutomatedRouteCandidate,
        paired_evidence: Mapping[str, object],
    ) -> tuple[str, str] | None:
        if paired_evidence.get("paired_energy_evidence") == "ASSUMED_4P5W":
            return "assumed_net_energy_uj", "ASSUMED_4P5W"
        if candidate.cost.energy_evidence == "MEASURED":
            return "measured_net_energy_uj", "MEASURED"
        return None

    @staticmethod
    def _phone_candidate_energy_deltas(
        paired_evidence: Mapping[str, object],
    ):
        warm = tuple(
            paired_evidence.get(
                "paired_warm_energy_delta" + suffix
            )
            for suffix in ("_lower_uj", "_uj", "_upper_uj")
        )
        per_token = tuple(
            paired_evidence.get(key)
            for key in (
                "paired_warm_energy_delta_lower_per_decode_token_uj",
                "paired_warm_energy_delta_per_decode_token_uj",
                "paired_warm_energy_delta_upper_per_decode_token_uj",
            )
        )
        transition = tuple(
            paired_evidence.get(
                "paired_transition_energy_delta" + suffix
            )
            for suffix in ("_lower_uj", "_uj", "_upper_uj")
        )
        return warm, per_token, transition

    @staticmethod
    def _normalized_phone_benefit(
        route_benefit: int,
        per_token: tuple[object, ...],
        decode_tokens: int,
    ) -> tuple[int, int, str] | tuple[None, None, str]:
        if all(type(value) is int for value in per_token):
            if not per_token[0] <= per_token[1] <= per_token[2]:
                return (
                    None,
                    None,
                    "ROUTE_MARGINAL_PER_TOKEN_EVIDENCE_INVALID",
                )
            return (
                -per_token[2],
                -per_token[2] * decode_tokens,
                "paired_warm_energy_delta_per_decode_token",
            )
        if any(value is not None for value in per_token):
            return (
                None,
                None,
                "ROUTE_MARGINAL_PER_TOKEN_EVIDENCE_INCOMPLETE",
            )
        normalized, _ = divmod(route_benefit, decode_tokens)
        return normalized, route_benefit, "legacy_route_total"

    def _assumed_phone_transition_evidence(
        self,
        candidate: AutomatedRouteCandidate,
        artifact_sha256: str,
        all_shards,
        total_resident_bytes: int,
    ):
        phone_device_id = (
            candidate.plan.execution_contract.phone_device_id
        )
        phone_power = (
            None
            if phone_device_id is None
            else self.catalog.phone_power_profile_by_device.get(
                phone_device_id
            )
        )
        if (
            phone_power is None
            or not phone_power.allow_assumed_for_scheduling
        ):
            return None, "PHONE_TRANSITION_POWER_MODEL_ABSENT"
        latency = self._phone_transition_latency_upper_us(
            candidate,
            artifact_sha256,
            phone_device_id,
            total_resident_bytes,
        )
        if latency is None:
            return None, "PHONE_TRANSITION_LATENCY_EVIDENCE_ABSENT"
        energy = (
            phone_power.active_power_mw * latency + 999
        ) // 1_000
        return (
            (
                {
                    shard.session_id: energy for shard in all_shards
                },
                "shared_phone_union",
                latency,
            ),
            None,
        )

    @staticmethod
    def _measured_phone_transition_evidence(
        all_shards,
        route_transition_energy_upper_uj: int,
        total_resident_bytes: int,
    ):
        by_session = {
            shard.session_id: (
                route_transition_energy_upper_uj
                * shard.resident_bytes
                // total_resident_bytes
            )
            for shard in sorted(
                all_shards, key=lambda row: row.session_id
            )
        }
        undistributed = (
            route_transition_energy_upper_uj - sum(by_session.values())
        )
        for shard in sorted(
            all_shards, key=lambda row: row.session_id
        ):
            if undistributed <= 0:
                break
            by_session[shard.session_id] += 1
            undistributed -= 1
        if sum(by_session.values()) != route_transition_energy_upper_uj:
            raise RouteGenerationError(
                "phone residency transition apportionment differs"
            )
        return by_session, "sum", None

    def _phone_transition_evidence(
        self,
        candidate: AutomatedRouteCandidate,
        artifact_sha256: str,
        all_shards,
        shards,
        route_transition_energy_upper_uj: int,
        energy_evidence: str,
    ):
        total_bytes = sum(
            shard.resident_bytes for shard in all_shards
        )
        if total_bytes <= 0:
            return None, "RESIDENCY_ROUTE_BYTES_ABSENT"
        if energy_evidence == "ASSUMED_4P5W":
            result, reason = self._assumed_phone_transition_evidence(
                candidate,
                artifact_sha256,
                all_shards,
                total_bytes,
            )
            if reason is not None:
                return None, reason
            all_by_session, aggregation, latency = result
        else:
            all_by_session, aggregation, latency = (
                self._measured_phone_transition_evidence(
                    all_shards,
                    route_transition_energy_upper_uj,
                    total_bytes,
                )
            )
        by_session = {
            shard.session_id: all_by_session[shard.session_id]
            for shard in sorted(shards, key=lambda row: row.session_id)
        }
        upper = (
            max(by_session.values(), default=0)
            if aggregation == "shared_phone_union"
            else sum(by_session.values())
        )
        return (
            (
                upper,
                MappingProxyType(dict(sorted(by_session.items()))),
                aggregation,
                latency,
            ),
            None,
        )

    def _eligible_phone_candidate(
        self,
        candidate: AutomatedRouteCandidate,
        candidates_by_id: Mapping[str, AutomatedRouteCandidate],
        artifact_sha256: str,
        ffn_by_layer: Mapping[int, str],
        ignored_reasons: set[str],
        decode_tokens: int,
    ) -> tuple[_EligiblePhoneEvidence | None, tuple[str, ...] | None]:
        all_shards = tuple(
            candidate.plan.execution_contract.phone_shards
        )
        shards = tuple(
            shard for shard in all_shards
            if shard.artifact_sha256 in {None, artifact_sha256}
        )
        if candidate.assisted_operator_kind != "ffn" or not shards:
            return None, None
        parent = candidates_by_id.get(candidate.paired_baseline_route_id)
        if parent is None:
            return None, ("PAIRED_DESKTOP_ROUTE_ABSENT",)
        if (
            candidate.plan.desktop_placement_sha256 is None
            or candidate.plan.desktop_placement_sha256
                != parent.plan.desktop_placement_sha256
        ):
            return None, ("PAIRED_DESKTOP_PLACEMENT_MISMATCH",)
        paired_evidence = dict(candidate.residency_break_even or {})
        energy_kind = self._phone_candidate_energy_kind(
            candidate, paired_evidence
        )
        if energy_kind is None:
            return None, ("ROUTE_MARGINAL_ENERGY_EVIDENCE_ABSENT",)
        benefit_kind, energy_evidence = energy_kind
        warm, per_token, transition = (
            self._phone_candidate_energy_deltas(paired_evidence)
        )
        if (
            any(type(value) is not int for value in warm)
            or not warm[0] <= warm[1] <= warm[2]
        ):
            return None, ("ROUTE_MARGINAL_WARM_EVIDENCE_ABSENT",)
        if (
            any(type(value) is not int for value in transition)
            or not transition[0] <= transition[1] <= transition[2]
        ):
            return None, ("ROUTE_MARGINAL_TRANSITION_EVIDENCE_ABSENT",)
        allowed = set(ignored_reasons)
        if (
            candidate.cost.latency_evidence in {"CALIBRATED", "MEASURED"}
            and energy_evidence in {"MEASURED", "ASSUMED_4P5W"}
        ):
            allowed.add("ROUTE_NOT_QUALIFIED")
        allowed.update({
            "ENERGY_UNKNOWN",
            "PHONE_RESIDENCY_LAYOUT_NOT_SELECTED",
        })
        physical_reasons = tuple(sorted(
            set(candidate.rejection_reasons) - allowed
        ))
        if physical_reasons:
            return None, physical_reasons
        layer_mask = 0
        for shard in shards:
            layer_mask |= shard.layer_mask
        operator_ids = tuple(
            operator_id
            for layer_index, operator_id in sorted(ffn_by_layer.items())
            if layer_mask & (1 << layer_index)
        )
        if not operator_ids:
            return None, ("ASSISTED_FFN_LAYER_SET_ABSENT",)
        route_benefit = -warm[2]
        if route_benefit <= 0:
            return None, ("ROUTE_MARGINAL_BENEFIT_NON_POSITIVE",)
        normalized, route_benefit, normalization = (
            self._normalized_phone_benefit(
                route_benefit, per_token, decode_tokens
            )
        )
        if normalized is None:
            return None, (normalization,)
        if normalized <= 0:
            return None, (
                "ROUTE_MARGINAL_BENEFIT_BELOW_ONE_UJ_PER_TOKEN",
            )
        transition_result, reason = self._phone_transition_evidence(
            candidate,
            artifact_sha256,
            all_shards,
            shards,
            max(0, transition[2]),
            energy_evidence,
        )
        if reason is not None:
            return None, (reason,)
        transition_upper, transition_by_session, aggregation, latency = (
            transition_result
        )
        return (
            _EligiblePhoneEvidence(
                normalized_benefit=normalized,
                route_benefit=route_benefit,
                operator_ids=operator_ids,
                source_route_id=candidate.candidate_id,
                parent_route_id=parent.candidate_id,
                benefit_value_kind=benefit_kind,
                energy_evidence=energy_evidence,
                transition_energy_upper_uj=transition_upper,
                transition_energy_upper_uj_by_session=(
                    transition_by_session
                ),
                transition_energy_aggregation=aggregation,
                transition_latency_upper_us=latency,
                normalization_source=normalization,
            ),
            None,
        )

    def _phone_residency_evidence_record(
        self,
        manifest: ModelManifest,
        candidates_by_id: Mapping[str, AutomatedRouteCandidate],
        selected: _EligiblePhoneEvidence,
        decode_tokens: int,
    ) -> _PhoneResidencyRouteEvidence:
        base_value, extra = divmod(
            selected.normalized_benefit,
            len(selected.operator_ids),
        )
        operator_benefits = {
            operator_id: base_value + (index < extra)
            for index, operator_id in enumerate(selected.operator_ids)
            if base_value + (index < extra) > 0
        }
        if sum(operator_benefits.values()) != selected.normalized_benefit:
            raise RouteGenerationError(
                "phone residency route benefit apportionment differs"
            )
        source = candidates_by_id[selected.source_route_id]
        source_shards = tuple(
            shard
            for shard in source.plan.execution_contract.phone_shards
            if shard.artifact_sha256 in {
                None, manifest.artifact_sha256,
            }
        )
        return _PhoneResidencyRouteEvidence(
            artifact_sha256=manifest.artifact_sha256,
            source_route_id=selected.source_route_id,
            paired_desktop_route_id=selected.parent_route_id,
            assisted_operator_ids=selected.operator_ids,
            benefit_by_operator_per_decode_token_uj=MappingProxyType(
                dict(sorted(operator_benefits.items()))
            ),
            route_benefit_uj=selected.route_benefit,
            decode_tokens=decode_tokens,
            normalized_benefit_uj=selected.normalized_benefit,
            normalization_remainder_uj=(
                selected.route_benefit
                - selected.normalized_benefit * decode_tokens
            ),
            benefit_value_kind=selected.benefit_value_kind,
            energy_evidence=selected.energy_evidence,
            transition_energy_upper_uj=(
                selected.transition_energy_upper_uj
            ),
            transition_energy_upper_uj_by_session=(
                selected.transition_energy_upper_uj_by_session
            ),
            transition_energy_aggregation=(
                selected.transition_energy_aggregation
            ),
            source_component_capability_sha256=(
                self.component_capability_identity(
                    source.plan, source.binding.executor_id
                )
            ),
            source_desktop_placement_sha256=(
                source.plan.desktop_placement_sha256
            ),
            source_executor_id=source.binding.executor_id,
            source_endpoint=source.binding.endpoint,
            source_operator_plan_protocol=(
                source.binding.operator_plan_protocol
            ),
            source_layer_mask=sum(
                shard.layer_mask for shard in source_shards
            ),
            source_maximum_columns=max(
                shard.maximum_columns for shard in source_shards
            ),
            source_session_ids=tuple(sorted(
                shard.session_id for shard in source_shards
            )),
            source_batch_plan=(
                source.plan.execution_contract.batch_plan
            ),
            source_maximum_batch_size=(
                source.plan.execution_contract.maximum_batch_size
            ),
            source_queue_depth=(
                source.plan.execution_contract.queue_depth
            ),
        )

    @staticmethod
    def _phone_residency_ready_status(
        artifact_sha256: str,
        selected: _EligiblePhoneEvidence,
        evidence: _PhoneResidencyRouteEvidence,
        rejected: Mapping[str, tuple[str, ...]],
        decode_tokens: int,
    ) -> Mapping[str, object]:
        return MappingProxyType({
            "artifact_sha256": artifact_sha256,
            "assisted_operator_count": len(selected.operator_ids),
            "benefit_normalization": "decode_token",
            "benefit_normalization_source": (
                selected.normalization_source
            ),
            "benefit_unit": "uJ_per_decode_token",
            "residency_pressure_unit": "remaining_decode_token",
            "benefit_value_kind": selected.benefit_value_kind,
            "candidate_rejections": dict(sorted(rejected.items())),
            "decode_tokens": decode_tokens,
            "energy_evidence": selected.energy_evidence,
            "normalized_benefit_uj": selected.normalized_benefit,
            "normalization_remainder_uj": (
                evidence.normalization_remainder_uj
            ),
            "paired_desktop_route_id": selected.parent_route_id,
            "reason": "PHONE_RESIDENCY_ROUTE_EVIDENCE_READY",
            "route_benefit_uj": selected.route_benefit,
            "source_route_id": selected.source_route_id,
            "source_component_capability_sha256": (
                evidence.source_component_capability_sha256
            ),
            "source_layer_mask": evidence.source_layer_mask,
            "source_maximum_columns": evidence.source_maximum_columns,
            "source_session_ids": list(evidence.source_session_ids),
            "transition_energy_upper_uj": (
                selected.transition_energy_upper_uj
            ),
            "transition_energy_upper_uj_by_session": dict(
                selected.transition_energy_upper_uj_by_session
            ),
            "transition_energy_aggregation": (
                selected.transition_energy_aggregation
            ),
            "transition_latency_upper_us": (
                selected.transition_latency_upper_us
            ),
        })

    def capture_phone_residency_route_evidence(
        self,
        manifest: ModelManifest,
        candidate_set: AutomatedCandidateSet,
        decode_tokens: int,
    ) -> Mapping[str, object]:
        """Convert paired route evidence into per-token packing value."""

        if (
            not isinstance(manifest, ModelManifest)
            or not isinstance(candidate_set, AutomatedCandidateSet)
            or type(decode_tokens) is not int
            or decode_tokens <= 0
        ):
            raise RouteGenerationError(
                "phone residency route evidence input is invalid"
            )
        artifact = manifest.artifact_sha256
        candidates_by_id = {
            row.candidate_id: row for row in candidate_set.candidates
        }
        ignored_reasons = {
            "COLD_RESIDENCY_BREAK_EVEN",
            "MODEL_EPOCH_AUDIT_ONLY",
            *PHONE_RESIDENCY_EVIDENCE_NEUTRAL_REJECTIONS,
        }
        rejected = {}
        eligible = []
        ffn_by_layer = self._ffn_operator_by_layer(manifest)
        for candidate in candidate_set.candidates:
            row, reasons = self._eligible_phone_candidate(
                candidate,
                candidates_by_id,
                artifact,
                ffn_by_layer,
                ignored_reasons,
                decode_tokens,
            )
            if reasons is not None:
                rejected[candidate.candidate_id] = reasons
            if row is not None:
                eligible.append(row)
        if not eligible:
            reason = (
                "PHONE_RESIDENCY_ASSISTED_ROUTE_ABSENT"
                if not rejected
                else "PHONE_RESIDENCY_ROUTE_EVIDENCE_UNUSABLE"
            )
            status = MappingProxyType({
                "artifact_sha256": artifact,
                "candidate_rejections": dict(sorted(rejected.items())),
                "reason": reason,
            })
            self._phone_residency_route_evidence.pop(artifact, None)
            self._phone_residency_evidence_status[artifact] = status
            return status
        selected = max(eligible, key=lambda row: row.rank)
        evidence = self._phone_residency_evidence_record(
            manifest, candidates_by_id, selected, decode_tokens
        )
        status = self._phone_residency_ready_status(
            artifact, selected, evidence, rejected, decode_tokens
        )
        self._phone_residency_route_evidence[artifact] = evidence
        self._phone_residency_evidence_status[artifact] = status
        return status

    def phone_residency_subset_evidence(
        self,
        manifest: ModelManifest,
        candidate: AutomatedRouteCandidate,
    ) -> Mapping[str, object] | None:
        """Validate one generated FFN subset against a current source route."""

        if not isinstance(manifest, ModelManifest) or not isinstance(
            candidate, AutomatedRouteCandidate
        ):
            raise RouteGenerationError(
                "phone residency subset evidence input is invalid"
            )
        evidence = self._phone_residency_route_evidence.get(
            manifest.artifact_sha256
        )
        shards = tuple(
            shard for shard in candidate.plan.execution_contract.phone_shards
            if shard.artifact_sha256 in {
                None, manifest.artifact_sha256,
            }
        )
        if evidence is None or not shards:
            return None
        target_layer_mask = 0
        for shard in shards:
            target_layer_mask |= shard.layer_mask
        if (
            target_layer_mask == evidence.source_layer_mask
            or target_layer_mask & evidence.source_layer_mask
                != target_layer_mask
            or candidate.plan.desktop_placement_sha256
                != evidence.source_desktop_placement_sha256
            or candidate.binding.executor_id != evidence.source_executor_id
            or candidate.binding.endpoint != evidence.source_endpoint
            or candidate.binding.operator_plan_protocol
                != evidence.source_operator_plan_protocol
            or candidate.plan.execution_contract.batch_plan
                != evidence.source_batch_plan
            or candidate.plan.execution_contract.maximum_batch_size
                != evidence.source_maximum_batch_size
            or candidate.plan.execution_contract.queue_depth
                != evidence.source_queue_depth
            or max(shard.maximum_columns for shard in shards)
                != evidence.source_maximum_columns
            or tuple(sorted(shard.session_id for shard in shards))
                != evidence.source_session_ids
            or self.component_capability_identity(
                candidate.plan, candidate.binding.executor_id
            ) != evidence.source_component_capability_sha256
        ):
            return None
        operators_by_layer = {
            int(operator.layer_id.partition(":")[2]): operator.operator_id
            for operator in manifest.operators
            if operator.kind == "ffn"
            and operator.layer_id.startswith("layer:")
            and operator.layer_id.partition(":")[2].isdigit()
        }
        target_operator_ids = tuple(
            operator_id
            for layer_index, operator_id in sorted(
                operators_by_layer.items()
            )
            if target_layer_mask & (1 << layer_index)
        )
        if (
            not target_operator_ids
            or not set(target_operator_ids).issubset(
                evidence.assisted_operator_ids
            )
        ):
            return None
        return MappingProxyType({
            "energy_evidence": evidence.energy_evidence,
            "source_component_capability_sha256": (
                evidence.source_component_capability_sha256
            ),
            "source_layer_mask": evidence.source_layer_mask,
            "source_route_id": evidence.source_route_id,
            "source_session_ids": evidence.source_session_ids,
            "target_layer_mask": target_layer_mask,
            "target_operator_ids": target_operator_ids,
        })

    def phone_residency_evidence_status(
        self, artifact_sha256: str
    ) -> Mapping[str, object]:
        status = self._phone_residency_evidence_status.get(
            artifact_sha256
        )
        if status is not None:
            return status
        return MappingProxyType({
            "artifact_sha256": artifact_sha256,
            "candidate_rejections": {},
            "reason": "PHONE_RESIDENCY_ROUTE_EVIDENCE_NOT_EVALUATED",
        })

    def phone_residency_demand(
        self,
        manifest: ModelManifest,
        queued_work: int,
    ) -> tuple[
        PhoneFfnResidencyDemand,
        tuple[RuntimePhoneSessionCapability, ...],
        str,
    ] | None:
        """Build queue-weighted packing input from executable capabilities."""

        if not isinstance(manifest, ModelManifest) or (
            type(queued_work) is not int or queued_work <= 0
        ):
            raise RouteGenerationError(
                "phone residency demand input is invalid"
            )
        evidence = self._phone_residency_route_evidence.get(
            manifest.artifact_sha256
        )
        if evidence is None:
            return None
        for coordinator in sorted(
            self.catalog.composite_executors,
            key=lambda row: row.executor_id,
        ):
            helper_id = coordinator.helper_device_id
            if (
                coordinator.assisted_operator_kind != "ffn"
                or coordinator.baseline_executor_id is None
                or helper_id is None
                or coordinator.adapter_parameters.get(
                    "ffn_runtime_control_protocol"
                ) != "decode-boundary-v1"
            ):
                continue
            helper = self.catalog.executor_by_device[helper_id]
            sessions = helper.phone_sessions
            if not sessions:
                continue
            allowed_ids = tuple(sorted(
                coordinator.operator_ids or tuple(
                    row.operator_id for row in manifest.operators
                    if row.kind == "ffn"
                )
            ))
            benefits = MappingProxyType({
                operator_id: value
                for operator_id, value in (
                    evidence
                    .benefit_by_operator_per_decode_token_uj.items()
                )
                if operator_id in allowed_ids and value > 0
            })
            if not benefits:
                continue
            return (
                PhoneFfnResidencyDemand(
                    manifest=manifest,
                    queued_work=queued_work,
                    maximum_columns=manifest.feed_forward_length,
                    batch_plan=str(coordinator.adapter_parameters.get(
                        "usb_batch_plan", "split-row"
                    )),
                    benefit_by_operator=benefits,
                    benefit_value_kind=evidence.benefit_value_kind,
                    allowed_operator_ids=tuple(sorted(benefits)),
                    transition_energy_uj_by_session=(
                        evidence.transition_energy_upper_uj_by_session
                    ),
                    transition_energy_aggregation=(
                        evidence.transition_energy_aggregation
                    ),
                ),
                sessions,
                helper_id,
            )
        return None

    def _minimum_battery_ppm(self, device_id: str) -> int:
        phone_power = self.catalog.phone_power_profile_by_device.get(
            device_id
        )
        if phone_power is not None:
            return phone_power.minimum_battery_ppm
        return self.catalog.executor_by_device[
            device_id
        ].minimum_battery_ppm
