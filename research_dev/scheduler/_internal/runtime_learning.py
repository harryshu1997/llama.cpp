"""Conservative online estimates from hash-bound physical receipts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping, Sequence

from .runtime_plan import (
    RuntimeExecutionPlan,
    RuntimeExecutionReceipt,
    RuntimeTransitionPlan,
    RuntimeTransitionReceipt,
)
from .runtime_search import request_shape_bucket
from .types import canonical_sha256


class RuntimeLearningError(ValueError):
    pass


RUNTIME_OBSERVATION_STORE_SCHEMA = "runtime-route-observations-v6"
_LEGACY_RUNTIME_OBSERVATION_STORE_SCHEMAS = frozenset({
    "runtime-route-observations-v3",
    "runtime-route-observations-v4",
    "runtime-route-observations-v5",
})


def _ceil_div(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise RuntimeLearningError("invalid ceiling division")
    return (numerator + denominator - 1) // denominator


def _integer_median(values: Sequence[int]) -> int:
    if not values or any(type(value) is not int for value in values):
        raise RuntimeLearningError("invalid median values")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) // 2


def _fit_affine(
    pairs: Sequence[tuple[int, int]],
) -> tuple[int, int, int] | None:
    if len(pairs) < 2 or len({row[0] for row in pairs}) < 2:
        return None
    count = len(pairs)
    sum_x = sum(row[0] for row in pairs)
    sum_y = sum(row[1] for row in pairs)
    sum_xx = sum(row[0] * row[0] for row in pairs)
    sum_xy = sum(row[0] * row[1] for row in pairs)
    denominator = count * sum_xx - sum_x * sum_x
    slope_numerator = count * sum_xy - sum_x * sum_y
    if denominator <= 0 or slope_numerator <= 0:
        return None
    intercept_numerator = sum_y * sum_xx - sum_x * sum_xy
    return slope_numerator, intercept_numerator, denominator


def _predict_component(
    pairs: Sequence[tuple[int, int]],
    component_value: int,
) -> int:
    model = _fit_affine(pairs)
    if model is not None:
        slope, intercept, denominator = model
        numerator = slope * component_value + intercept
        if numerator > 0:
            return _ceil_div(numerator, denominator)
    ratios = tuple(
        _ceil_div(measured * 1_000_000, component)
        for component, measured in pairs
    )
    center = max(1, sum(ratios) // len(ratios))
    return max(1, _ceil_div(component_value * center, 1_000_000))


def _prediction_multipliers_ppm(
    pairs: Sequence[tuple[int, int]],
) -> tuple[int, int]:
    ratios = tuple(
        _ceil_div(
            measured * 1_000_000,
            _predict_component(pairs, component),
        )
        for component, measured in pairs
    )
    lower = max(
        1,
        min(
            900_000,
            min(ratios) * 950_000 // 1_000_000,
        ),
    )
    upper = max(
        1_100_000,
        _ceil_div(max(ratios) * 1_050_000, 1_000_000),
    )
    return lower, upper


def _fit_token_phases(
    samples: Sequence[tuple[int, int, int]],
) -> tuple[int, int, int] | None:
    """Fit positive prefill and decode terms without a free intercept."""
    if len(samples) < 2:
        return None
    sum_input_sq = sum(row[0] * row[0] for row in samples)
    sum_output_sq = sum(row[1] * row[1] for row in samples)
    sum_input_output = sum(row[0] * row[1] for row in samples)
    sum_input_value = sum(row[0] * row[2] for row in samples)
    sum_output_value = sum(row[1] * row[2] for row in samples)
    denominator = (
        sum_input_sq * sum_output_sq
        - sum_input_output * sum_input_output
    )
    input_numerator = (
        sum_input_value * sum_output_sq
        - sum_output_value * sum_input_output
    )
    output_numerator = (
        sum_output_value * sum_input_sq
        - sum_input_value * sum_input_output
    )
    if (
        denominator <= 0
        or input_numerator <= 0
        or output_numerator <= 0
    ):
        return None
    return input_numerator, output_numerator, denominator


def _predict_route_value(
    shape_samples: Sequence[tuple[int, int, int]],
    component_pairs: Sequence[tuple[int, int]],
    *,
    input_tokens: int,
    output_tokens: int,
    component_value: int,
) -> int:
    phase_model = _fit_token_phases(shape_samples)
    if phase_model is not None:
        input_numerator, output_numerator, denominator = phase_model
        numerator = (
            input_numerator * input_tokens
            + output_numerator * output_tokens
        )
        if numerator > 0:
            return max(1, _ceil_div(numerator, denominator))
    return _predict_component(component_pairs, component_value)


def _route_prediction_multipliers_ppm(
    shape_samples: Sequence[tuple[int, int, int]],
    component_pairs: Sequence[tuple[int, int]],
) -> tuple[int, int]:
    if len(shape_samples) != len(component_pairs):
        raise RuntimeLearningError(
            "runtime route prediction samples are inconsistent"
        )
    ratios = tuple(
        _ceil_div(
            measured * 1_000_000,
            _predict_route_value(
                shape_samples,
                component_pairs,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                component_value=component,
            ),
        )
        for (input_tokens, output_tokens, measured), (component, _)
        in zip(shape_samples, component_pairs)
    )
    lower = max(
        1,
        min(
            900_000,
            min(ratios) * 950_000 // 1_000_000,
        ),
    )
    upper = max(
        1_200_000,
        _ceil_div(max(ratios) * 1_050_000, 1_000_000),
    )
    return lower, upper


def _magnitude_bucket(value: int) -> int:
    if value == 0:
        return 0
    sign = -1 if value < 0 else 1
    absolute = abs(value)
    return sign * (1 << (absolute.bit_length() - 1))


def _cost_feature_bucket(
    cost_features: Mapping[str, int],
) -> Mapping[str, int]:
    bucket = {}
    for name, value in sorted(cost_features.items()):
        if (
            type(name) is not str
            or not name
            or not name.isascii()
            or type(value) is not int
        ):
            raise RuntimeLearningError("runtime cost feature is invalid")
        if name.endswith("_pct"):
            bucket[name] = value // 10 * 10
        elif name.endswith("_basis_points"):
            bucket[name] = value // 500 * 500
        elif (
            name.endswith("_id")
            or name.endswith("_class")
            or name.endswith("_phase")
            or name.endswith("_state")
        ):
            bucket[name] = value
        else:
            bucket[name] = _magnitude_bucket(value)
    return MappingProxyType(bucket)


_TEMPLATE_REQUEST_SHAPE_FEATURES = frozenset({
    "actual_batch_size",
    "cpu_utilization_pct",
    "large_phase_id",
    "prompt_ubatch_count",
})

_TEMPLATE_OPTIONAL_ZERO_FEATURES = frozenset({
    "active_model_input_tokens",
    "active_model_output_tokens",
    "active_model_requests",
})


def _template_cost_feature_bucket(
    cost_features: Mapping[str, int],
) -> Mapping[str, int]:
    return _cost_feature_bucket({
        name: value
        for name, value in cost_features.items()
        if name not in _TEMPLATE_REQUEST_SHAPE_FEATURES
        and not (
            name in _TEMPLATE_OPTIONAL_ZERO_FEATURES and value == 0
        )
    })


@dataclass(frozen=True)
class RuntimeLearnedRouteEstimate:
    profile_id: str
    sample_count: int
    service_us: int
    service_upper_us: int
    energy_uj: int | None
    energy_lower_uj: int | None
    energy_upper_uj: int | None
    maturity: str
    latency_maturity: str
    energy_maturity: str
    latency_mape_ppm: int | None
    energy_mape_ppm: int | None
    latency_upper_coverage_ppm: int | None
    energy_upper_coverage_ppm: int | None
    evidence_ids: tuple[str, ...]
    energy_scope: str


@dataclass(frozen=True)
class _MeasuredRouteObservation:
    receipt_id: str
    latency_us: int
    energy_uj: int | None
    evidence_ids: tuple[str, ...]
    input_tokens: int = 0
    output_tokens: int = 0
    component_service_us: int = 0
    component_energy_uj: int | None = None
    energy_scope: str = "route_total"


@dataclass(frozen=True)
class RuntimeLearnedTransitionEstimate:
    profile_id: str
    sample_count: int
    latency_us: int
    latency_upper_us: int
    energy_uj: int | None
    energy_lower_uj: int | None
    energy_upper_uj: int | None
    latency_maturity: str
    latency_upper_maturity: str
    latency_mape_ppm: int | None
    latency_upper_coverage_ppm: int | None
    energy_maturity: str
    maturity: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class _MeasuredTransitionObservation:
    receipt_id: str
    latency_us: int
    energy_uj: int | None
    evidence_ids: tuple[str, ...]


_REQUEST_DEPENDENT_ADAPTER_PARAMETERS = frozenset({
    "ffn_max_tokens",
    "ubatch_size",
    "usb_d2h_transport_profile_id",
    "usb_h2d_transport_profile_id",
    "usb_max_payload_bytes",
    "usb_transport_profile_id",
})


def _plan_template_sha256(plan: RuntimeExecutionPlan) -> str:
    """Hash physical placement while excluding request-shape payload choices."""

    return canonical_sha256({
        "adapter_parameters": {
            key: value
            for key, value in plan.adapter_parameters.items()
            if key not in _REQUEST_DEPENDENT_ADAPTER_PARAMETERS
        },
        "assisted_operator_kind": plan.assisted_operator_kind,
        "device_ids": list(plan.device_ids),
        "operators": [
            {
                "device_ids": list(row.device_ids),
                "operator_id": row.operator_id,
                "operator_kind": row.operator_kind,
                "split_axis": row.split_axis,
                "split_fraction_ppm": row.split_fraction_ppm,
            }
            for row in plan.operators
        ],
        "overlap_kind": plan.overlap_kind,
        "residency_variant": plan.residency_variant,
        "route_family": plan.route_family,
        "split_axis": plan.split_axis,
        "split_fraction_ppm": plan.split_fraction_ppm,
        "transitions": [
            {
                "device_id": row.device_id,
                "executor_id": row.executor_id,
                "source_state": row.source_state,
                "target_state": row.target_state,
                "transition_id": row.transition_id,
            }
            for row in plan.transitions
        ],
        "transport": {
            key: plan.adapter_parameters[key]
            for key in (
                "usb_allocator",
                "usb_concurrent_streams",
                "usb_full_duplex",
                "usb_queue_depth",
                "usb_transport_generation",
            )
            if key in plan.adapter_parameters
        },
    })


def _component_plan_template_sha256(plan: RuntimeExecutionPlan) -> str:
    """Hash executable work while excluding request and residency state."""

    return canonical_sha256({
        "adapter_parameters": {
            key: value
            for key, value in plan.adapter_parameters.items()
            if key not in _REQUEST_DEPENDENT_ADAPTER_PARAMETERS
        },
        "assisted_operator_kind": plan.assisted_operator_kind,
        "device_ids": list(plan.device_ids),
        "operators": [
            {
                "device_ids": list(row.device_ids),
                "operator_id": row.operator_id,
                "operator_kind": row.operator_kind,
                "split_axis": row.split_axis,
                "split_fraction_ppm": row.split_fraction_ppm,
            }
            for row in plan.operators
        ],
        "overlap_kind": plan.overlap_kind,
        "route_family": plan.route_family,
        "split_axis": plan.split_axis,
        "split_fraction_ppm": plan.split_fraction_ppm,
        "transport": {
            key: plan.adapter_parameters[key]
            for key in (
                "usb_allocator",
                "usb_concurrent_streams",
                "usb_full_duplex",
                "usb_queue_depth",
                "usb_transport_generation",
            )
            if key in plan.adapter_parameters
        },
    })


class RuntimeRouteObservationStore:
    """Learn one conservative route estimate per request-shape bucket."""

    def __init__(self, qualification_samples: int = 4) -> None:
        if type(qualification_samples) is not int or qualification_samples < 4:
            raise RuntimeLearningError(
                "runtime qualification requires at least four samples"
            )
        self.qualification_samples = qualification_samples
        self._rows: dict[
            tuple[object, ...], tuple[_MeasuredRouteObservation, ...]
        ] = {}
        self._template_rows: dict[
            tuple[object, ...], tuple[_MeasuredRouteObservation, ...]
        ] = {}
        self._transition_rows: dict[
            tuple[str, ...], tuple[_MeasuredTransitionObservation, ...]
        ] = {}
        self._receipt_ids: set[str] = set()
        self._template_sha256_by_plan_sha256: dict[str, str] = {}
        self._component_template_sha256_by_plan_sha256: dict[str, str] = {}
        self._template_sha256_by_identity: dict[
            tuple[object, ...],
            tuple[tuple[object, ...], str],
        ] = {}
        self._exact_estimate_cache: dict[
            tuple[object, ...],
            tuple[
                tuple[_MeasuredRouteObservation, ...],
                RuntimeLearnedRouteEstimate,
            ],
        ] = {}
        self._template_estimate_cache: dict[
            tuple[object, ...],
            tuple[
                tuple[_MeasuredRouteObservation, ...],
                RuntimeLearnedRouteEstimate | None,
            ],
        ] = {}
        self._incomplete_receipts = 0
        self._diagnostic_energy_receipts = 0
        self._unattributed_transfer_receipts = 0

    def _key(
        self,
        *,
        artifact_sha256: str,
        capability_generation_sha256: str,
        plan: RuntimeExecutionPlan,
        input_tokens: int,
        output_tokens: int,
        quality_requirement: str,
        cost_features: Mapping[str, int],
    ) -> tuple[object, ...]:
        input_bucket, output_bucket = request_shape_bucket(
            input_tokens, output_tokens
        )
        return (
            artifact_sha256,
            capability_generation_sha256,
            plan.route_id,
            plan.route_family,
            plan.device_ids,
            plan.assisted_operator_kind,
            plan.split_axis,
            plan.split_fraction_ppm,
            plan.residency_variant,
            input_bucket,
            output_bucket,
            quality_requirement,
            canonical_sha256(_cost_feature_bucket(cost_features)),
            self._template_sha256(plan),
        )

    @staticmethod
    def _profile_id(key: tuple[object, ...]) -> str:
        return "online:" + canonical_sha256(list(key))[7:]

    def _template_key(
        self,
        *,
        artifact_sha256: str,
        capability_generation_sha256: str,
        component_capability_generation_sha256: str | None = None,
        plan: RuntimeExecutionPlan,
        quality_requirement: str,
        cost_features: Mapping[str, int],
    ) -> tuple[object, ...]:
        template_sha256 = self._component_template_sha256(plan)
        return (
            artifact_sha256,
            (
                capability_generation_sha256
                if component_capability_generation_sha256 is None
                else component_capability_generation_sha256
            ),
            template_sha256,
            quality_requirement,
            canonical_sha256(_template_cost_feature_bucket(cost_features)),
        )

    def _component_template_sha256(
        self, plan: RuntimeExecutionPlan
    ) -> str:
        cached = self._component_template_sha256_by_plan_sha256.get(
            plan.plan_sha256
        )
        if cached is not None:
            return cached
        value = _component_plan_template_sha256(plan)
        if len(self._component_template_sha256_by_plan_sha256) >= 4_096:
            self._component_template_sha256_by_plan_sha256.pop(next(iter(
                self._component_template_sha256_by_plan_sha256
            )))
        self._component_template_sha256_by_plan_sha256[
            plan.plan_sha256
        ] = value
        return value

    def _template_sha256(self, plan: RuntimeExecutionPlan) -> str:
        cached = self._template_sha256_by_plan_sha256.get(plan.plan_sha256)
        if cached is not None:
            return cached
        stable_parameters = tuple(
            (key, value)
            for key, value in plan.adapter_parameters.items()
            if key not in _REQUEST_DEPENDENT_ADAPTER_PARAMETERS
        )
        transition_identity = tuple(
            (
                row.device_id,
                row.executor_id,
                row.source_state,
                row.target_state,
                row.transition_id,
            )
            for row in plan.transitions
        )
        identity = (
            id(plan.operators),
            plan.assisted_operator_kind,
            plan.device_ids,
            plan.overlap_kind,
            plan.residency_variant,
            plan.route_family,
            plan.split_axis,
            plan.split_fraction_ppm,
            stable_parameters,
            transition_identity,
        )
        identity_cached = self._template_sha256_by_identity.get(identity)
        if (
            identity_cached is not None
            and identity_cached[0] is plan.operators
        ):
            template_sha256 = identity_cached[1]
        else:
            template_sha256 = _plan_template_sha256(plan)
            if len(self._template_sha256_by_identity) >= 4_096:
                self._template_sha256_by_identity.pop(next(iter(
                    self._template_sha256_by_identity
                )))
            self._template_sha256_by_identity[identity] = (
                plan.operators,
                template_sha256,
            )
        if len(self._template_sha256_by_plan_sha256) >= 4_096:
            self._template_sha256_by_plan_sha256.pop(next(iter(
                self._template_sha256_by_plan_sha256
            )))
        self._template_sha256_by_plan_sha256[
            plan.plan_sha256
        ] = template_sha256
        return template_sha256

    def record(
        self,
        *,
        artifact_sha256: str,
        capability_generation_sha256: str,
        plan: RuntimeExecutionPlan,
        input_tokens: int,
        output_tokens: int,
        quality_requirement: str,
        cost_features: Mapping[str, int],
        receipt: RuntimeExecutionReceipt,
        energy_boundary_id: str,
        required_domain_ids: Sequence[str],
        component_service_us: int | None = None,
        component_energy_uj: int | None = None,
        component_capability_generation_sha256: str | None = None,
    ) -> bool:
        if (
            component_service_us is not None
            and (
                type(component_service_us) is not int
                or component_service_us <= 0
            )
        ):
            raise RuntimeLearningError(
                "component service prior must be positive"
            )
        if (
            component_energy_uj is not None
            and (
                type(component_energy_uj) is not int
                or component_energy_uj <= 0
            )
        ):
            raise RuntimeLearningError(
                "component energy prior must be positive"
            )
        receipt_id = canonical_sha256(receipt.to_json())
        if receipt_id in self._receipt_ids:
            raise RuntimeLearningError(
                "runtime execution receipt was already observed"
            )
        self._receipt_ids.add(receipt_id)
        required_domains = frozenset(required_domain_ids)
        required_links = frozenset(
            resource_id.removeprefix("link:")
            for resource_id in plan.resource_ids
            if resource_id.startswith("link:")
        )
        latency_complete = receipt.actual_latency_us > 0
        energy_accounting_complete = (
            receipt.energy_boundary_id == energy_boundary_id
            and required_domains.issubset(
                receipt.fleet_energy_uj_by_domain
            )
            and receipt.whole_fleet_energy_uj is not None
        )
        energy_complete = (
            energy_accounting_complete
            and receipt.energy_attribution_kind in {
                "isolated", "matched_abba", "device_domain"
            }
        )
        if not latency_complete:
            self._incomplete_receipts += 1
            return False
        if (
            energy_accounting_complete
            and not required_links.issubset(
                receipt.transfer_energy_uj_by_link
            )
        ):
            self._unattributed_transfer_receipts += 1
        if (
            receipt.energy_boundary_id is not None
            and not energy_accounting_complete
        ):
            self._incomplete_receipts += 1
        elif energy_accounting_complete and not energy_complete:
            self._diagnostic_energy_receipts += 1
        key = self._key(
            artifact_sha256=artifact_sha256,
            capability_generation_sha256=capability_generation_sha256,
            plan=plan,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            quality_requirement=quality_requirement,
            cost_features=cost_features,
        )
        observation = _MeasuredRouteObservation(
            receipt_id=receipt_id,
            latency_us=receipt.actual_latency_us,
            energy_uj=(
                receipt.whole_fleet_energy_uj if energy_complete else None
            ),
            evidence_ids=receipt.measurement_evidence_ids,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            component_service_us=(component_service_us or 0),
            component_energy_uj=component_energy_uj,
            energy_scope=receipt.energy_scope,
        )
        self._rows[key] = self._rows.get(key, ()) + (observation,)
        if component_service_us is not None:
            template_key = self._template_key(
                artifact_sha256=artifact_sha256,
                capability_generation_sha256=(
                    capability_generation_sha256
                ),
                component_capability_generation_sha256=(
                    component_capability_generation_sha256
                ),
                plan=plan,
                quality_requirement=quality_requirement,
                cost_features=cost_features,
            )
            component_observation = (
                replace(observation, energy_uj=None)
                if plan.transitions
                and receipt.energy_scope != "warm_execution"
                else observation
            )
            self._template_rows[template_key] = (
                self._template_rows.get(template_key, ())
                + (component_observation,)
            )
        return True

    def estimate(
        self,
        *,
        artifact_sha256: str,
        capability_generation_sha256: str,
        plan: RuntimeExecutionPlan,
        input_tokens: int,
        output_tokens: int,
        quality_requirement: str,
        cost_features: Mapping[str, int],
        component_service_us: int | None = None,
        component_energy_uj: int | None = None,
        component_capability_generation_sha256: str | None = None,
    ) -> RuntimeLearnedRouteEstimate | None:
        key = self._key(
            artifact_sha256=artifact_sha256,
            capability_generation_sha256=capability_generation_sha256,
            plan=plan,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            quality_requirement=quality_requirement,
            cost_features=cost_features,
        )
        rows = self._rows.get(key, ())
        exact = None
        if rows:
            exact = self._cached_exact_estimate(key, rows)
        template = None
        if component_service_us is not None:
            template_key = self._template_key(
                artifact_sha256=artifact_sha256,
                capability_generation_sha256=(
                    capability_generation_sha256
                ),
                component_capability_generation_sha256=(
                    component_capability_generation_sha256
                ),
                plan=plan,
                quality_requirement=quality_requirement,
                cost_features=cost_features,
            )
            template = self._cached_template_estimate(
                template_key,
                self._component_rows_from_exact(
                    template_key,
                    rows,
                    legacy_prefix=(
                        artifact_sha256,
                        capability_generation_sha256,
                        self._template_sha256(plan),
                        quality_requirement,
                    ),
                ),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                component_service_us=component_service_us,
                component_energy_uj=component_energy_uj,
            )
        if plan.transitions and template is not None:
            return template
        if exact is not None and exact.maturity in {
            "QUALIFIED", "QUARANTINED"
        }:
            return exact
        if template is not None and template.maturity in {
            "QUALIFIED", "QUARANTINED"
        }:
            return template
        return exact if exact is not None else template

    def exact_estimate(
        self,
        *,
        artifact_sha256: str,
        capability_generation_sha256: str,
        plan: RuntimeExecutionPlan,
        input_tokens: int,
        output_tokens: int,
        quality_requirement: str,
        cost_features: Mapping[str, int],
    ) -> RuntimeLearnedRouteEstimate | None:
        """Return only exact route-total evidence for one physical plan."""
        key = self._key(
            artifact_sha256=artifact_sha256,
            capability_generation_sha256=capability_generation_sha256,
            plan=plan,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            quality_requirement=quality_requirement,
            cost_features=cost_features,
        )
        rows = self._rows.get(key, ())
        return None if not rows else self._cached_exact_estimate(key, rows)

    @staticmethod
    def _transition_key_values(
        *,
        artifact_sha256: str,
        capability_generation_sha256: str,
        component_identity_sha256: str,
        transition_id: str,
        source_state: str,
        target_state: str,
        executor_id: str,
    ) -> tuple[str, ...]:
        values = (
            artifact_sha256,
            capability_generation_sha256,
            component_identity_sha256,
            transition_id,
            source_state,
            target_state,
            executor_id,
        )
        if any(
            type(value) is not str or not value or not value.isascii()
            for value in values
        ) or any(
            not value.startswith("sha256:") or len(value) != 71
            for value in values[:3]
        ):
            raise RuntimeLearningError(
                "runtime transition observation identity is invalid"
            )
        return values

    @classmethod
    def _transition_key(
        cls,
        *,
        artifact_sha256: str,
        capability_generation_sha256: str,
        component_identity_sha256: str,
        receipt: RuntimeTransitionReceipt,
    ) -> tuple[str, ...]:
        return cls._transition_key_values(
            artifact_sha256=artifact_sha256,
            capability_generation_sha256=capability_generation_sha256,
            component_identity_sha256=component_identity_sha256,
            transition_id=receipt.transition_id,
            source_state=receipt.source_state,
            target_state=receipt.target_state,
            executor_id=receipt.executor_id,
        )

    def record_transition(
        self,
        *,
        artifact_sha256: str,
        capability_generation_sha256: str,
        component_identity_sha256: str,
        plan: RuntimeExecutionPlan,
        receipt: RuntimeTransitionReceipt,
        energy_boundary_id: str,
        required_domain_ids: Sequence[str],
    ) -> bool:
        if not isinstance(plan, RuntimeExecutionPlan) or not isinstance(
            receipt, RuntimeTransitionReceipt
        ):
            raise RuntimeLearningError(
                "runtime transition observation is invalid"
            )
        if (
            receipt.artifact_sha256 != artifact_sha256
            or receipt.operator_plan_sha256 != plan.plan_sha256
        ):
            raise RuntimeLearningError(
                "runtime transition observation plan differs"
            )
        key = self._transition_key(
            artifact_sha256=artifact_sha256,
            capability_generation_sha256=capability_generation_sha256,
            component_identity_sha256=component_identity_sha256,
            receipt=receipt,
        )
        receipt_id = canonical_sha256(receipt.to_json())
        if receipt_id in self._receipt_ids:
            raise RuntimeLearningError(
                "runtime transition receipt was already observed"
            )
        self._receipt_ids.add(receipt_id)
        if receipt.status != "COMPLETED" or receipt.actual_latency_us <= 0:
            self._incomplete_receipts += 1
            return False
        required_domains = frozenset(required_domain_ids)
        energy_accounting_complete = (
            receipt.energy_boundary_id == energy_boundary_id
            and required_domains.issubset(
                receipt.fleet_energy_uj_by_domain
            )
            and receipt.whole_fleet_energy_uj is not None
        )
        energy_complete = (
            energy_accounting_complete
            and receipt.energy_attribution_kind in {
                "isolated", "matched_abba", "device_domain"
            }
        )
        if (
            receipt.energy_boundary_id is not None
            and not energy_accounting_complete
        ):
            self._incomplete_receipts += 1
        elif energy_accounting_complete and not energy_complete:
            self._diagnostic_energy_receipts += 1
        observation = _MeasuredTransitionObservation(
            receipt_id=receipt_id,
            latency_us=receipt.actual_latency_us,
            energy_uj=(
                receipt.whole_fleet_energy_uj if energy_complete else None
            ),
            evidence_ids=receipt.measurement_evidence_ids,
        )
        self._transition_rows[key] = (
            self._transition_rows.get(key, ()) + (observation,)
        )
        return True

    def transition_estimate(
        self,
        *,
        artifact_sha256: str,
        capability_generation_sha256: str,
        component_identity_sha256: str,
        receipt: RuntimeTransitionReceipt,
    ) -> RuntimeLearnedTransitionEstimate | None:
        key = self._transition_key(
            artifact_sha256=artifact_sha256,
            capability_generation_sha256=capability_generation_sha256,
            component_identity_sha256=component_identity_sha256,
            receipt=receipt,
        )
        rows = self._transition_rows.get(key, ())
        return self._transition_estimate(key, rows)

    def transition_estimate_for_plan(
        self,
        *,
        artifact_sha256: str,
        capability_generation_sha256: str,
        component_identity_sha256: str,
        transition: RuntimeTransitionPlan,
        executor_id: str | None = None,
    ) -> RuntimeLearnedTransitionEstimate | None:
        if not isinstance(transition, RuntimeTransitionPlan):
            raise RuntimeLearningError(
                "runtime transition estimate plan is invalid"
            )
        executor_id = (
            transition.executor_id
            if executor_id is None else executor_id
        )
        if executor_id is None:
            return None
        key = self._transition_key_values(
            artifact_sha256=artifact_sha256,
            capability_generation_sha256=capability_generation_sha256,
            component_identity_sha256=component_identity_sha256,
            transition_id=transition.transition_id,
            source_state=transition.source_state,
            target_state=transition.target_state,
            executor_id=executor_id,
        )
        return self._transition_estimate(
            key, self._transition_rows.get(key, ())
        )

    def _transition_estimate(
        self,
        key: tuple[str, ...],
        rows: Sequence[_MeasuredTransitionObservation],
    ) -> RuntimeLearnedTransitionEstimate | None:
        if not rows:
            return None
        latencies = tuple(row.latency_us for row in rows)
        energies = tuple(
            row.energy_uj for row in rows if row.energy_uj is not None
        )
        latency = max(1, _integer_median(latencies))
        latency_upper = max(
            max(latencies), _ceil_div(max(latencies) * 1_100_000, 1_000_000)
        )
        energy = (
            None if not energies else max(1, _integer_median(energies))
        )
        energy_lower = (
            None if not energies
            else max(1, min(energies) * 950_000 // 1_000_000)
        )
        energy_upper = (
            None if not energies
            else max(
                max(energies),
                _ceil_div(max(energies) * 1_100_000, 1_000_000),
            )
        )
        latency_maturity = "SHADOW"
        latency_upper_maturity = "SHADOW"
        latency_mape_ppm = None
        latency_upper_coverage_ppm = None
        if len(rows) >= self.qualification_samples:
            calibration = latencies[:-2]
            held_out = latencies[-2:]
            center = max(1, _integer_median(calibration))
            limit = _ceil_div(max(calibration) * 1_100_000, 1_000_000)
            latency_mape_ppm = sum(
                abs(value - center) * 1_000_000 // value
                for value in held_out
            ) // len(held_out)
            latency_upper_coverage_ppm = (
                sum(value <= limit for value in held_out)
                * 1_000_000 // len(held_out)
            )
            latency_maturity = (
                "QUALIFIED"
                if latency_mape_ppm <= 200_000
                and latency_upper_coverage_ppm == 1_000_000
                else "QUARANTINED"
            )
            latency_upper_maturity = (
                "QUALIFIED"
                if latency_upper_coverage_ppm == 1_000_000
                else "QUARANTINED"
            )
        energy_maturity = "SHADOW"
        if len(energies) >= self.qualification_samples:
            calibration = energies[:-2]
            held_out = energies[-2:]
            center = max(1, _integer_median(calibration))
            limit = _ceil_div(max(calibration) * 1_100_000, 1_000_000)
            mape = sum(
                abs(value - center) * 1_000_000 // value
                for value in held_out
            ) // len(held_out)
            energy_maturity = (
                "QUALIFIED"
                if mape <= 200_000
                and all(value <= limit for value in held_out)
                else "QUARANTINED"
            )
        maturity = (
            "QUARANTINED"
            if "QUARANTINED" in {
                latency_maturity, energy_maturity
            }
            else "QUALIFIED"
            if latency_maturity == energy_maturity == "QUALIFIED"
            else "SHADOW"
        )
        return RuntimeLearnedTransitionEstimate(
            profile_id=(
                "online-transition:" + canonical_sha256(list(key))[7:]
            ),
            sample_count=len(rows),
            latency_us=latency,
            latency_upper_us=latency_upper,
            energy_uj=energy,
            energy_lower_uj=energy_lower,
            energy_upper_uj=energy_upper,
            latency_maturity=latency_maturity,
            latency_upper_maturity=latency_upper_maturity,
            latency_mape_ppm=latency_mape_ppm,
            latency_upper_coverage_ppm=latency_upper_coverage_ppm,
            energy_maturity=energy_maturity,
            maturity=maturity,
            evidence_ids=tuple(sorted({
                evidence_id
                for row in rows
                for evidence_id in row.evidence_ids
            })),
        )

    def _cached_exact_estimate(
        self,
        key: tuple[object, ...],
        rows: tuple[_MeasuredRouteObservation, ...],
    ) -> RuntimeLearnedRouteEstimate:
        cached = self._exact_estimate_cache.get(key)
        if cached is not None and cached[0] is rows:
            return cached[1]
        estimate = self._estimate_exact(key, rows)
        if len(self._exact_estimate_cache) >= 4_096:
            self._exact_estimate_cache.pop(next(iter(
                self._exact_estimate_cache
            )))
        self._exact_estimate_cache[key] = (rows, estimate)
        return estimate

    def _cached_template_estimate(
        self,
        key: tuple[object, ...],
        rows: tuple[_MeasuredRouteObservation, ...],
        *,
        input_tokens: int,
        output_tokens: int,
        component_service_us: int,
        component_energy_uj: int | None,
    ) -> RuntimeLearnedRouteEstimate | None:
        cache_key = (
            key,
            input_tokens,
            output_tokens,
            component_service_us,
            component_energy_uj,
        )
        cached = self._template_estimate_cache.get(cache_key)
        if cached is not None and cached[0] is rows:
            return cached[1]
        estimate = self._estimate_template(
            key,
            rows,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            component_service_us=component_service_us,
            component_energy_uj=component_energy_uj,
        )
        if len(self._template_estimate_cache) >= 4_096:
            self._template_estimate_cache.pop(next(iter(
                self._template_estimate_cache
            )))
        self._template_estimate_cache[cache_key] = (rows, estimate)
        return estimate

    def _component_rows_from_exact(
        self,
        template_key: tuple[object, ...],
        exact_rows: tuple[_MeasuredRouteObservation, ...],
        *,
        legacy_prefix: tuple[object, ...],
    ) -> tuple[_MeasuredRouteObservation, ...]:
        rows = self._template_rows.get(template_key, ())
        if rows:
            return rows
        exact_receipts = frozenset(
            observation.receipt_id for observation in exact_rows
        )

        def compatible_sources(
            prefix: tuple[object, ...],
        ) -> dict[
            tuple[str, ...], tuple[_MeasuredRouteObservation, ...]
        ]:
            sources = (
                source_rows
                for source_key, source_rows in self._template_rows.items()
                if source_key != template_key
                and source_key[:4] == prefix
                and (
                    not exact_receipts
                    or exact_receipts.issubset({
                        observation.receipt_id
                        for observation in source_rows
                    })
                )
            )
            return {
                tuple(
                    observation.receipt_id
                    for observation in source_rows
                ): source_rows
                for source_rows in sources
            }

        component_sources = compatible_sources(template_key[:4])
        if component_sources:
            if len(component_sources) != 1:
                return ()
            return next(iter(component_sources.values()))
        legacy_sources = compatible_sources(legacy_prefix)
        if len(legacy_sources) != 1:
            return ()
        return next(iter(legacy_sources.values()))

    def calibration_information(
        self,
        *,
        artifact_sha256: str,
        capability_generation_sha256: str,
        component_capability_generation_sha256: str | None = None,
        plan: RuntimeExecutionPlan,
        input_tokens: int,
        output_tokens: int,
        quality_requirement: str,
    ) -> Mapping[str, int]:
        """Rank calibration work by qualification information gain."""
        if not isinstance(plan, RuntimeExecutionPlan):
            raise RuntimeLearningError(
                "runtime calibration plan is invalid"
            )
        current_bucket = request_shape_bucket(input_tokens, output_tokens)
        template_sha256 = _component_plan_template_sha256(plan)
        template_capability_sha256 = (
            capability_generation_sha256
            if component_capability_generation_sha256 is None
            else component_capability_generation_sha256
        )
        observations = tuple(
            observation
            for key, rows in self._template_rows.items()
            if key[0] == artifact_sha256
            and key[1] == template_capability_sha256
            and key[2] == template_sha256
            and key[3] == quality_requirement
            for observation in rows
        )
        buckets = tuple(
            request_shape_bucket(
                observation.input_tokens,
                observation.output_tokens,
            )
            for observation in observations
        )
        distinct_buckets = frozenset(buckets)
        current_bucket_samples = buckets.count(current_bucket)
        sample_count = len(observations)
        if (
            len(distinct_buckets) >= 2
            and sample_count < self.qualification_samples
        ):
            priority_class = 0
        elif sample_count and len(distinct_buckets) < 2 and (
            current_bucket not in distinct_buckets
        ):
            priority_class = 1
        elif sample_count == 0 or current_bucket_samples == 0:
            priority_class = 2
        else:
            priority_class = 3
        return MappingProxyType({
            "current_bucket_samples": current_bucket_samples,
            "distinct_shape_buckets": len(distinct_buckets),
            "priority_class": priority_class,
            "samples_needed": max(
                0, self.qualification_samples - sample_count
            ),
            "template_samples": sample_count,
        })

    def _estimate_exact(
        self,
        key: tuple[object, ...],
        rows: Sequence[_MeasuredRouteObservation],
    ) -> RuntimeLearnedRouteEstimate:
        latencies = tuple(row.latency_us for row in rows)
        energies = tuple(
            row.energy_uj for row in rows if row.energy_uj is not None
        )
        service_us = max(1, sum(latencies) // len(latencies))
        energy_uj = (
            None if not energies else max(1, sum(energies) // len(energies))
        )
        service_upper_us = max(
            max(latencies), _ceil_div(max(latencies) * 1_050_000, 1_000_000)
        )
        energy_lower_uj = (
            None if not energies
            else max(1, min(energies) * 950_000 // 1_000_000)
        )
        energy_upper_uj = (
            None if not energies
            else max(
                max(energies),
                _ceil_div(max(energies) * 1_050_000, 1_000_000),
            )
        )
        latency_maturity = "SHADOW"
        energy_maturity = "SHADOW"
        latency_mape_ppm = None
        energy_mape_ppm = None
        latency_upper_coverage_ppm = None
        energy_upper_coverage_ppm = None
        if len(rows) >= self.qualification_samples:
            calibration = rows[:-2]
            held_out = rows[-2:]
            latency_center = max(
                1,
                sum(row.latency_us for row in calibration)
                // len(calibration),
            )
            latency_limit = _ceil_div(
                max(row.latency_us for row in calibration) * 1_100_000,
                1_000_000,
            )
            latency_mape_ppm = sum(
                abs(row.latency_us - latency_center) * 1_000_000
                // row.latency_us
                for row in held_out
            ) // len(held_out)
            latency_upper_coverage_ppm = (
                sum(row.latency_us <= latency_limit for row in held_out)
                * 1_000_000 // len(held_out)
            )
            latency_maturity = (
                "QUALIFIED"
                if latency_mape_ppm <= 200_000
                and latency_upper_coverage_ppm == 1_000_000
                else "QUARANTINED"
            )
        energy_rows = tuple(row for row in rows if row.energy_uj is not None)
        if len(energy_rows) >= self.qualification_samples:
            calibration = energy_rows[:-2]
            held_out = energy_rows[-2:]
            energy_center = max(
                1,
                sum(row.energy_uj for row in calibration) // len(calibration),
            )
            energy_limit = _ceil_div(
                max(row.energy_uj for row in calibration) * 1_100_000,
                1_000_000,
            )
            energy_mape_ppm = sum(
                abs(row.energy_uj - energy_center) * 1_000_000
                // row.energy_uj
                for row in held_out
            ) // len(held_out)
            energy_upper_coverage_ppm = (
                sum(row.energy_uj <= energy_limit for row in held_out)
                * 1_000_000 // len(held_out)
            )
            energy_maturity = (
                "QUALIFIED"
                if energy_mape_ppm <= 200_000
                and energy_upper_coverage_ppm == 1_000_000
                else "QUARANTINED"
            )
        maturity = (
            "QUARANTINED"
            if "QUARANTINED" in {latency_maturity, energy_maturity}
            else "QUALIFIED"
            if latency_maturity == energy_maturity == "QUALIFIED"
            else "SHADOW"
        )
        evidence = tuple(sorted({
            evidence_id
            for row in rows
            for evidence_id in row.evidence_ids
        }))
        return RuntimeLearnedRouteEstimate(
            profile_id=self._profile_id(key),
            sample_count=len(rows),
            service_us=service_us,
            service_upper_us=service_upper_us,
            energy_uj=energy_uj,
            energy_lower_uj=energy_lower_uj,
            energy_upper_uj=energy_upper_uj,
            maturity=maturity,
            latency_maturity=latency_maturity,
            energy_maturity=energy_maturity,
            latency_mape_ppm=latency_mape_ppm,
            energy_mape_ppm=energy_mape_ppm,
            latency_upper_coverage_ppm=latency_upper_coverage_ppm,
            energy_upper_coverage_ppm=energy_upper_coverage_ppm,
            evidence_ids=evidence,
            energy_scope="route_total",
        )

    @staticmethod
    def _template_service_values(
        rows: Sequence[_MeasuredRouteObservation],
        input_tokens: int,
        output_tokens: int,
        component_service_us: int,
    ) -> tuple[int, int]:
        pairs = tuple(
            (row.component_service_us, row.latency_us) for row in rows
        )
        shapes = tuple(
            (row.input_tokens, row.output_tokens, row.latency_us)
            for row in rows
        )
        matching = tuple(
            row.latency_us for row in rows
            if row.input_tokens == input_tokens
            and row.output_tokens == output_tokens
        )
        if matching:
            value = max(1, sum(matching) // len(matching))
            upper = max(
                max(matching),
                _ceil_div(max(matching) * 1_050_000, 1_000_000),
            )
            return value, upper
        value = _predict_route_value(
            shapes,
            pairs,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            component_value=component_service_us,
        )
        _, upper_ppm = _route_prediction_multipliers_ppm(shapes, pairs)
        return value, _ceil_div(value * upper_ppm, 1_000_000)

    @staticmethod
    def _template_energy_values(
        rows: Sequence[_MeasuredRouteObservation],
        input_tokens: int,
        output_tokens: int,
        component_energy_uj: int | None,
    ) -> tuple[
        int | None,
        int | None,
        int | None,
        tuple[_MeasuredRouteObservation, ...],
    ]:
        energy_rows = tuple(
            row for row in rows
            if row.energy_uj is not None
            and row.component_energy_uj is not None
        )
        pairs = tuple(
            (row.component_energy_uj, row.energy_uj) for row in energy_rows
        )
        shapes = tuple(
            (row.input_tokens, row.output_tokens, row.energy_uj)
            for row in energy_rows
        )
        if component_energy_uj is None or not pairs:
            return None, None, None, energy_rows
        matching = tuple(
            row.energy_uj for row in energy_rows
            if row.input_tokens == input_tokens
            and row.output_tokens == output_tokens
        )
        if matching:
            value = max(1, sum(matching) // len(matching))
            lower = max(1, min(matching) * 950_000 // 1_000_000)
            upper = max(
                max(matching),
                _ceil_div(max(matching) * 1_050_000, 1_000_000),
            )
            return value, lower, upper, energy_rows
        value = _predict_route_value(
            shapes,
            pairs,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            component_value=component_energy_uj,
        )
        lower_ppm, upper_ppm = _route_prediction_multipliers_ppm(
            shapes, pairs
        )
        lower = max(1, value * lower_ppm // 1_000_000)
        upper = max(value, _ceil_div(value * upper_ppm, 1_000_000))
        return value, lower, upper, energy_rows

    @staticmethod
    def _template_holdout_maturity(
        rows: Sequence[_MeasuredRouteObservation],
        qualification_samples: int,
        component_name: str,
        measured_name: str,
    ) -> tuple[str, int | None, int | None]:
        if len(rows) < qualification_samples:
            return "SHADOW", None, None
        calibration = rows[:-2]
        held_out = rows[-2:]
        pairs = tuple(
            (getattr(row, component_name), getattr(row, measured_name))
            for row in calibration
        )
        shapes = tuple(
            (
                row.input_tokens,
                row.output_tokens,
                getattr(row, measured_name),
            )
            for row in calibration
        )
        _, upper = _route_prediction_multipliers_ppm(shapes, pairs)
        predicted = tuple(max(
            1,
            _predict_route_value(
                shapes,
                pairs,
                input_tokens=row.input_tokens,
                output_tokens=row.output_tokens,
                component_value=getattr(row, component_name),
            ),
        ) for row in held_out)
        limits = tuple(
            max(1, _ceil_div(prediction * upper, 1_000_000))
            for prediction in predicted
        )
        mape_ppm = sum(
            abs(getattr(row, measured_name) - estimate) * 1_000_000
            // getattr(row, measured_name)
            for row, estimate in zip(held_out, predicted)
        ) // len(held_out)
        coverage_ppm = (
            sum(
                getattr(row, measured_name) <= limit
                for row, limit in zip(held_out, limits)
            ) * 1_000_000 // len(held_out)
        )
        maturity = (
            "QUALIFIED"
            if mape_ppm <= 200_000 and coverage_ppm == 1_000_000
            else "QUARANTINED"
        )
        return maturity, mape_ppm, coverage_ppm

    def _estimate_template(
        self,
        key: tuple[object, ...],
        rows: Sequence[_MeasuredRouteObservation],
        *,
        input_tokens: int,
        output_tokens: int,
        component_service_us: int,
        component_energy_uj: int | None,
    ) -> RuntimeLearnedRouteEstimate | None:
        if not rows or any(row.component_service_us <= 0 for row in rows):
            return None
        service_us, service_upper_us = self._template_service_values(
            rows, input_tokens, output_tokens, component_service_us
        )
        (
            energy_uj,
            energy_lower_uj,
            energy_upper_uj,
            energy_rows,
        ) = self._template_energy_values(
            rows, input_tokens, output_tokens, component_energy_uj
        )
        (
            latency_maturity,
            latency_mape_ppm,
            latency_upper_coverage_ppm,
        ) = self._template_holdout_maturity(
            rows,
            self.qualification_samples,
            "component_service_us",
            "latency_us",
        )
        (
            energy_maturity,
            energy_mape_ppm,
            energy_upper_coverage_ppm,
        ) = self._template_holdout_maturity(
            energy_rows,
            self.qualification_samples,
            "component_energy_uj",
            "energy_uj",
        )
        maturity = (
            "QUARANTINED"
            if "QUARANTINED" in {latency_maturity, energy_maturity}
            else "QUALIFIED"
            if latency_maturity == energy_maturity == "QUALIFIED"
            else "SHADOW"
        )
        return RuntimeLearnedRouteEstimate(
            profile_id="online-template:" + canonical_sha256(list(key))[7:],
            sample_count=len(rows),
            service_us=service_us,
            service_upper_us=service_upper_us,
            energy_uj=energy_uj,
            energy_lower_uj=energy_lower_uj,
            energy_upper_uj=energy_upper_uj,
            maturity=maturity,
            latency_maturity=latency_maturity,
            energy_maturity=energy_maturity,
            latency_mape_ppm=latency_mape_ppm,
            energy_mape_ppm=energy_mape_ppm,
            latency_upper_coverage_ppm=latency_upper_coverage_ppm,
            energy_upper_coverage_ppm=energy_upper_coverage_ppm,
            evidence_ids=tuple(sorted({
                evidence_id
                for row in rows
                for evidence_id in row.evidence_ids
            })),
            energy_scope="warm_execution",
        )

    @staticmethod
    def _key_json(key: tuple[object, ...]) -> list[object]:
        values = list(key)
        values[4] = list(values[4])
        return values

    @staticmethod
    def _key_from_json(value: object) -> tuple[object, ...]:
        if type(value) is not list or len(value) != 14:
            raise RuntimeLearningError("runtime observation key is invalid")
        devices = value[4]
        if type(devices) is not list or any(
            type(device_id) is not str
            or not device_id
            or not device_id.isascii()
            for device_id in devices
        ):
            raise RuntimeLearningError(
                "runtime observation devices are invalid"
            )
        values = list(value)
        values[4] = tuple(devices)
        if any(
            type(values[index]) is not str
            or not values[index]
            or not values[index].isascii()
            for index in (0, 1, 2, 3, 6, 8, 11, 12, 13)
        ) or any(
            type(values[index]) is not int or values[index] < 0
            for index in (7, 9, 10)
        ) or (
            values[5] is not None
            and (
                type(values[5]) is not str
                or not values[5]
                or not values[5].isascii()
            )
        ):
            raise RuntimeLearningError("runtime observation key is invalid")
        return tuple(values)

    @staticmethod
    def _template_key_from_json(value: object) -> tuple[object, ...]:
        if (
            type(value) is not list
            or len(value) != 5
            or any(
                type(item) is not str or not item or not item.isascii()
                for item in value
            )
        ):
            raise RuntimeLearningError(
                "runtime template observation key is invalid"
            )
        return tuple(value)

    @staticmethod
    def _observation_json(
        observation: _MeasuredRouteObservation,
    ) -> dict[str, object]:
        return {
            "component_energy_uj": observation.component_energy_uj,
            "component_service_us": observation.component_service_us,
            "energy_uj": observation.energy_uj,
            "energy_scope": observation.energy_scope,
            "evidence_ids": list(observation.evidence_ids),
            "input_tokens": observation.input_tokens,
            "latency_us": observation.latency_us,
            "output_tokens": observation.output_tokens,
            "receipt_id": observation.receipt_id,
        }

    @staticmethod
    def _transition_observation_json(
        observation: _MeasuredTransitionObservation,
    ) -> dict[str, object]:
        return {
            "energy_uj": observation.energy_uj,
            "evidence_ids": list(observation.evidence_ids),
            "latency_us": observation.latency_us,
            "receipt_id": observation.receipt_id,
        }

    def to_json(self) -> dict[str, object]:
        rows = []
        for key in sorted(self._rows, key=lambda row: canonical_sha256(
            self._key_json(row)
        )):
            rows.append({
                "key": self._key_json(key),
                "observations": [
                    self._observation_json(observation)
                    for observation in self._rows[key]
                ],
            })
        template_rows = []
        for key in sorted(
            self._template_rows,
            key=lambda row: canonical_sha256(list(row)),
        ):
            template_rows.append({
                "key": list(key),
                "observations": [
                    self._observation_json(observation)
                    for observation in self._template_rows[key]
                ],
            })
        transition_rows = []
        for key in sorted(
            self._transition_rows,
            key=lambda row: canonical_sha256(list(row)),
        ):
            transition_rows.append({
                "key": list(key),
                "observations": [
                    self._transition_observation_json(observation)
                    for observation in self._transition_rows[key]
                ],
            })
        body = {
            "diagnostic_energy_receipts": self._diagnostic_energy_receipts,
            "incomplete_receipts": self._incomplete_receipts,
            "qualification_samples": self.qualification_samples,
            "receipt_ids": sorted(self._receipt_ids),
            "rows": rows,
            "schema": RUNTIME_OBSERVATION_STORE_SCHEMA,
            "template_rows": template_rows,
            "transition_rows": transition_rows,
            "unattributed_transfer_receipts": (
                self._unattributed_transfer_receipts
            ),
        }
        return {**body, "store_sha256": canonical_sha256(body)}

    def _validated_import_body(
        self, value: object
    ) -> tuple[dict[str, object], list, list, list, list[str]]:
        if type(value) is not dict:
            raise RuntimeLearningError(
                "runtime observation store must be an object"
            )
        body = dict(value)
        observed_hash = body.pop("store_sha256", None)
        rows = body.get("rows")
        template_rows = body.get("template_rows", [])
        transition_rows = body.get("transition_rows", [])
        receipt_ids = body.get("receipt_ids")
        if (
            body.get("schema") not in (
                _LEGACY_RUNTIME_OBSERVATION_STORE_SCHEMAS
                | {RUNTIME_OBSERVATION_STORE_SCHEMA}
            )
            or body.get("qualification_samples") != self.qualification_samples
            or type(rows) is not list
            or type(template_rows) is not list
            or type(transition_rows) is not list
            or type(receipt_ids) is not list
            or any(
                type(receipt_id) is not str
                or not receipt_id.startswith("sha256:")
                or len(receipt_id) != 71
                for receipt_id in receipt_ids
            )
            or len(receipt_ids) != len(set(receipt_ids))
            or type(body.get("incomplete_receipts")) is not int
            or body["incomplete_receipts"] < 0
            or type(body.get("diagnostic_energy_receipts")) is not int
            or body["diagnostic_energy_receipts"] < 0
            or type(body.get("unattributed_transfer_receipts")) is not int
            or body["unattributed_transfer_receipts"] < 0
            or observed_hash != canonical_sha256(body)
        ):
            raise RuntimeLearningError("runtime observation store is invalid")
        if (
            self._rows
            or self._receipt_ids
            or self._incomplete_receipts
            or self._diagnostic_energy_receipts
            or self._unattributed_transfer_receipts
            or self._template_rows
            or self._transition_rows
        ):
            raise RuntimeLearningError(
                "runtime observation store import requires an empty store"
            )
        return body, rows, template_rows, transition_rows, receipt_ids

    @staticmethod
    def _import_route_observation(
        observation: object,
        key: tuple[object, ...],
        receipt_keys: dict[str, tuple[object, ...]],
    ) -> _MeasuredRouteObservation:
        if type(observation) is not dict:
            raise RuntimeLearningError("runtime observation is invalid")
        receipt_id = observation.get("receipt_id")
        evidence_ids = observation.get("evidence_ids")
        latency_us = observation.get("latency_us")
        energy_uj = observation.get("energy_uj")
        input_tokens = observation.get("input_tokens", 0)
        output_tokens = observation.get("output_tokens", 0)
        component_service_us = observation.get("component_service_us", 0)
        component_energy_uj = observation.get("component_energy_uj")
        energy_scope = observation.get("energy_scope", "route_total")
        if (
            type(receipt_id) is not str
            or not receipt_id.startswith("sha256:")
            or len(receipt_id) != 71
            or type(evidence_ids) is not list
            or any(
                type(evidence_id) is not str
                or not evidence_id
                or not evidence_id.isascii()
                for evidence_id in evidence_ids
            )
            or type(latency_us) is not int
            or latency_us <= 0
            or type(input_tokens) is not int
            or input_tokens < 0
            or type(output_tokens) is not int
            or output_tokens < 0
            or type(component_service_us) is not int
            or component_service_us < 0
            or (energy_uj is not None and (
                type(energy_uj) is not int or energy_uj <= 0
            ))
            or (component_energy_uj is not None and (
                type(component_energy_uj) is not int
                or component_energy_uj <= 0
            ))
            or energy_scope not in {"route_total", "warm_execution"}
            or receipt_id in receipt_keys
        ):
            raise RuntimeLearningError("runtime observation is invalid")
        receipt_keys[receipt_id] = key
        return _MeasuredRouteObservation(
            receipt_id=receipt_id,
            latency_us=latency_us,
            energy_uj=energy_uj,
            evidence_ids=tuple(evidence_ids),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            component_service_us=component_service_us,
            component_energy_uj=component_energy_uj,
            energy_scope=energy_scope,
        )

    def _import_route_rows(
        self, rows: list
    ) -> tuple[
        dict[tuple[object, ...], tuple[_MeasuredRouteObservation, ...]],
        dict[str, tuple[object, ...]],
    ]:
        imported = {}
        receipt_keys = {}
        for row in rows:
            if type(row) is not dict or type(row.get("observations")) is not list:
                raise RuntimeLearningError("runtime observation row is invalid")
            key = self._key_from_json(row.get("key"))
            if key in imported:
                raise RuntimeLearningError(
                    "runtime observation key is duplicated"
                )
            observations = tuple(
                self._import_route_observation(value, key, receipt_keys)
                for value in row["observations"]
            )
            if not observations:
                raise RuntimeLearningError("runtime observation row is empty")
            imported[key] = observations
        return imported, receipt_keys

    @staticmethod
    def _import_template_observation(
        observation: object,
        key: tuple[object, ...],
        receipt_ids: Sequence[str],
        receipt_keys: Mapping[str, tuple[object, ...]],
        template_receipts: dict[str, tuple[str, int]],
        index: int,
    ) -> _MeasuredRouteObservation:
        if type(observation) is not dict:
            raise RuntimeLearningError(
                "runtime template observation is invalid"
            )
        receipt_id = observation.get("receipt_id")
        evidence_ids = observation.get("evidence_ids")
        values = (
            observation.get("latency_us"),
            observation.get("input_tokens"),
            observation.get("output_tokens"),
            observation.get("component_service_us"),
        )
        energy_uj = observation.get("energy_uj")
        component_energy_uj = observation.get("component_energy_uj")
        energy_scope = observation.get("energy_scope", "route_total")
        if (
            type(receipt_id) is not str
            or receipt_id not in receipt_ids
            or type(evidence_ids) is not list
            or any(
                type(evidence_id) is not str
                or not evidence_id
                or not evidence_id.isascii()
                for evidence_id in evidence_ids
            )
            or any(type(item) is not int or item <= 0 for item in values)
            or (energy_uj is not None and (
                type(energy_uj) is not int or energy_uj <= 0
            ))
            or (component_energy_uj is not None and (
                type(component_energy_uj) is not int
                or component_energy_uj <= 0
            ))
            or energy_scope not in {"route_total", "warm_execution"}
        ):
            raise RuntimeLearningError(
                "runtime template observation is invalid"
            )
        exact_key = receipt_keys.get(receipt_id)
        if (
            exact_key is not None
            and exact_key[8] == "cold"
            and energy_scope != "warm_execution"
        ):
            energy_uj = None
        previous = template_receipts.get(receipt_id)
        current = (key[2], index)
        if previous is not None and previous[0] != current[0]:
            raise RuntimeLearningError(
                "runtime receipt has multiple plan templates"
            )
        template_receipts[receipt_id] = current
        return _MeasuredRouteObservation(
            receipt_id=receipt_id,
            latency_us=values[0],
            energy_uj=energy_uj,
            evidence_ids=tuple(evidence_ids),
            input_tokens=values[1],
            output_tokens=values[2],
            component_service_us=values[3],
            component_energy_uj=component_energy_uj,
            energy_scope=energy_scope,
        )

    def _import_template_rows(
        self,
        rows: list,
        receipt_ids: Sequence[str],
        receipt_keys: Mapping[str, tuple[object, ...]],
    ) -> tuple[
        dict[tuple[object, ...], tuple[_MeasuredRouteObservation, ...]],
        dict[str, tuple[str, int]],
    ]:
        imported = {}
        template_receipts = {}
        for row in rows:
            if type(row) is not dict or type(row.get("observations")) is not list:
                raise RuntimeLearningError(
                    "runtime template observation row is invalid"
                )
            key = self._template_key_from_json(row.get("key"))
            if key in imported:
                raise RuntimeLearningError(
                    "runtime template observation key is duplicated"
                )
            observations = tuple(
                self._import_template_observation(
                    value,
                    key,
                    receipt_ids,
                    receipt_keys,
                    template_receipts,
                    index,
                )
                for index, value in enumerate(row["observations"])
            )
            if not observations:
                raise RuntimeLearningError(
                    "runtime template observation row is empty"
                )
            imported[key] = observations
        return imported, template_receipts

    def _normalize_imported_route_rows(
        self,
        imported: Mapping[
            tuple[object, ...], tuple[_MeasuredRouteObservation, ...]
        ],
        template_receipts: Mapping[str, tuple[str, int]],
        legacy_schema: bool,
    ) -> dict[tuple[object, ...], tuple[_MeasuredRouteObservation, ...]]:
        normalized = {}
        for key, observations in imported.items():
            normalized_key = key
            template_sha256 = key[13]
            if legacy_schema:
                templates = {
                    template_receipts[row.receipt_id][0]
                    for row in observations
                    if row.receipt_id in template_receipts
                }
                if len(templates) > 1:
                    raise RuntimeLearningError(
                        "runtime observation row spans plan templates"
                    )
                template_sha256 = next(iter(templates), key[13])
                normalized_key = (*key[:13], template_sha256)
            target = normalized.setdefault(normalized_key, [])
            for fallback_order, observation in enumerate(observations):
                order = fallback_order
                if legacy_schema:
                    order = template_receipts.get(
                        observation.receipt_id,
                        (template_sha256, fallback_order),
                    )[1]
                target.append((order, observation))
            if legacy_schema:
                self._template_sha256_by_plan_sha256[key[13]] = template_sha256
        return {
            key: tuple(
                observation for _, observation in sorted(
                    observations,
                    key=lambda row: (row[0], row[1].receipt_id),
                )
            )
            for key, observations in normalized.items()
        }

    @staticmethod
    def _import_transition_observation(
        observation: object,
        receipt_ids: Sequence[str],
        transition_receipts: set[str],
    ) -> _MeasuredTransitionObservation:
        if type(observation) is not dict:
            raise RuntimeLearningError(
                "runtime transition observation is invalid"
            )
        receipt_id = observation.get("receipt_id")
        evidence_ids = observation.get("evidence_ids")
        latency_us = observation.get("latency_us")
        energy_uj = observation.get("energy_uj")
        if (
            type(receipt_id) is not str
            or receipt_id not in receipt_ids
            or receipt_id in transition_receipts
            or type(evidence_ids) is not list
            or any(
                type(value) is not str or not value or not value.isascii()
                for value in evidence_ids
            )
            or type(latency_us) is not int
            or latency_us <= 0
            or (energy_uj is not None and (
                type(energy_uj) is not int or energy_uj <= 0
            ))
        ):
            raise RuntimeLearningError(
                "runtime transition observation is invalid"
            )
        transition_receipts.add(receipt_id)
        return _MeasuredTransitionObservation(
            receipt_id=receipt_id,
            latency_us=latency_us,
            energy_uj=energy_uj,
            evidence_ids=tuple(evidence_ids),
        )

    def _import_transition_rows(
        self, rows: list, receipt_ids: Sequence[str]
    ) -> dict[tuple[str, ...], tuple[_MeasuredTransitionObservation, ...]]:
        imported = {}
        transition_receipts: set[str] = set()
        for row in rows:
            if (
                type(row) is not dict
                or type(row.get("key")) is not list
                or len(row["key"]) != 7
                or any(
                    type(value) is not str or not value or not value.isascii()
                    for value in row["key"]
                )
                or any(
                    not value.startswith("sha256:") or len(value) != 71
                    for value in row["key"][:3]
                )
                or type(row.get("observations")) is not list
                or not row["observations"]
            ):
                raise RuntimeLearningError(
                    "runtime transition observation row is invalid"
                )
            key = tuple(row["key"])
            if key in imported:
                raise RuntimeLearningError(
                    "runtime transition observation key is duplicated"
                )
            imported[key] = tuple(
                self._import_transition_observation(
                    value, receipt_ids, transition_receipts
                )
                for value in row["observations"]
            )
        return imported

    def import_json(self, value: object) -> None:
        (
            body,
            rows,
            template_rows,
            transition_rows,
            receipt_ids,
        ) = self._validated_import_body(value)
        imported, receipt_keys = self._import_route_rows(rows)
        imported_templates, template_receipts = self._import_template_rows(
            template_rows, receipt_ids, receipt_keys
        )
        imported = self._normalize_imported_route_rows(
            imported,
            template_receipts,
            body["schema"] in _LEGACY_RUNTIME_OBSERVATION_STORE_SCHEMAS,
        )
        imported_transitions = self._import_transition_rows(
            transition_rows, receipt_ids
        )
        if set(receipt_keys) - set(receipt_ids):
            raise RuntimeLearningError(
                "runtime observation receipt set is incomplete"
            )
        self._rows = imported
        self._template_rows = imported_templates
        self._transition_rows = imported_transitions
        self._receipt_ids = set(receipt_ids)
        self._incomplete_receipts = body["incomplete_receipts"]
        self._diagnostic_energy_receipts = body[
            "diagnostic_energy_receipts"
        ]
        self._unattributed_transfer_receipts = body[
            "unattributed_transfer_receipts"
        ]
        self._exact_estimate_cache.clear()
        self._template_estimate_cache.clear()

    def merge_json(self, value: object) -> None:
        """Atomically merge a separately validated observation store."""
        incoming = RuntimeRouteObservationStore(
            qualification_samples=self.qualification_samples
        )
        incoming.import_json(value)
        duplicates = self._receipt_ids & incoming._receipt_ids
        if duplicates:
            raise RuntimeLearningError(
                "runtime observation store receipts overlap"
            )
        rows = dict(self._rows)
        for key, observations in incoming._rows.items():
            rows[key] = rows.get(key, ()) + observations
        template_rows = dict(self._template_rows)
        for key, observations in incoming._template_rows.items():
            template_rows[key] = (
                template_rows.get(key, ()) + observations
            )
        transition_rows = dict(self._transition_rows)
        for key, observations in incoming._transition_rows.items():
            transition_rows[key] = (
                transition_rows.get(key, ()) + observations
            )
        template_hashes = dict(self._template_sha256_by_plan_sha256)
        for plan_sha256, template_sha256 in (
            incoming._template_sha256_by_plan_sha256.items()
        ):
            existing = template_hashes.get(plan_sha256)
            if existing is not None and existing != template_sha256:
                raise RuntimeLearningError(
                    "runtime observation plan template differs"
                )
            template_hashes[plan_sha256] = template_sha256
        self._rows = rows
        self._template_rows = template_rows
        self._transition_rows = transition_rows
        self._receipt_ids = self._receipt_ids | incoming._receipt_ids
        self._template_sha256_by_plan_sha256 = template_hashes
        self._incomplete_receipts += incoming._incomplete_receipts
        self._diagnostic_energy_receipts += (
            incoming._diagnostic_energy_receipts
        )
        self._unattributed_transfer_receipts += (
            incoming._unattributed_transfer_receipts
        )
        self._exact_estimate_cache.clear()
        self._template_estimate_cache.clear()

    def normalize_template_features(
        self,
        receipt_contexts: Sequence[
            tuple[str, Mapping[str, int]]
        ],
    ) -> None:
        """Atomically re-key templates using ordered raw runtime context."""
        contexts: dict[str, tuple[int, str]] = {}
        for index, row in enumerate(receipt_contexts):
            if (
                type(row) is not tuple
                or len(row) != 2
                or type(row[0]) is not str
                or row[0] in contexts
                or not isinstance(row[1], Mapping)
            ):
                raise RuntimeLearningError(
                    "runtime template context is invalid"
                )
            contexts[row[0]] = (
                index,
                canonical_sha256(
                    _template_cost_feature_bucket(row[1])
                ),
            )
        observations = {
            observation.receipt_id
            for rows in self._template_rows.values()
            for observation in rows
        }
        if observations != set(contexts):
            raise RuntimeLearningError(
                "runtime template context receipts differ"
            )
        grouped: dict[
            tuple[object, ...],
            list[tuple[int, _MeasuredRouteObservation]],
        ] = {}
        for key, rows in self._template_rows.items():
            for observation in rows:
                order, feature_sha256 = contexts[observation.receipt_id]
                target = key[:4] + (feature_sha256,)
                grouped.setdefault(target, []).append((order, observation))
        normalized = {
            key: tuple(
                observation for _, observation in sorted(
                    rows, key=lambda row: row[0]
                )
            )
            for key, rows in grouped.items()
        }
        self._template_rows = normalized
        self._template_estimate_cache.clear()

    def rebind_legacy_component_rows(
        self,
        *,
        artifact_sha256: str,
        source_capability_sha256: str,
        source_template_sha256: str,
        component_capability_sha256: str,
        component_template_sha256: str,
        latency_only: bool = False,
    ) -> int:
        """Move verified legacy route rows to a component identity."""
        values = (
            artifact_sha256,
            source_capability_sha256,
            source_template_sha256,
            component_capability_sha256,
            component_template_sha256,
        )
        if type(latency_only) is not bool or any(
            type(value) is not str
            or not value.startswith("sha256:")
            or len(value) != 71
            or any(
                character not in "0123456789abcdef"
                for character in value[7:]
            )
            for value in values
        ):
            raise RuntimeLearningError(
                "runtime component observation identity is invalid"
            )
        source_prefix = (
            artifact_sha256,
            source_capability_sha256,
            source_template_sha256,
        )
        target_prefix = (
            artifact_sha256,
            component_capability_sha256,
            component_template_sha256,
        )
        if source_prefix == target_prefix:
            return 0
        matches = tuple(
            (key, rows)
            for key, rows in self._template_rows.items()
            if key[:3] == source_prefix
        )
        if not matches:
            raise RuntimeLearningError(
                "legacy component observation rows are absent"
            )
        updated = dict(self._template_rows)
        rebound = 0
        for source_key, rows in matches:
            target_key = target_prefix + source_key[3:]
            target_rows = (
                tuple(replace(
                    row,
                    energy_uj=None,
                    component_energy_uj=None,
                ) for row in rows)
                if latency_only else rows
            )
            existing = updated.get(target_key)
            if existing is not None and existing != target_rows:
                raise RuntimeLearningError(
                    "component observation target conflicts"
                )
            updated[target_key] = target_rows
            del updated[source_key]
            rebound += len(rows)
        self._template_rows = updated
        self._template_estimate_cache.clear()
        return rebound

    def rebind_legacy_exact_route_rows(
        self,
        *,
        artifact_sha256: str,
        source_capability_sha256: str,
        target_capability_sha256: str,
        plan: RuntimeExecutionPlan,
        latency_only: bool = False,
    ) -> int:
        """Re-key exact route evidence after verified catalog migration."""
        values = (
            artifact_sha256,
            source_capability_sha256,
            target_capability_sha256,
        )
        if type(latency_only) is not bool or any(
            type(value) is not str
            or not value.startswith("sha256:")
            or len(value) != 71
            or any(
                character not in "0123456789abcdef"
                for character in value[7:]
            )
            for value in values
        ) or not isinstance(plan, RuntimeExecutionPlan):
            raise RuntimeLearningError(
                "runtime exact-route observation identity is invalid"
            )
        target_template = self._template_sha256(plan)
        matches = tuple(
            (key, rows)
            for key, rows in self._rows.items()
            if len(key) == 14
            and key[0] == artifact_sha256
            and key[1] == source_capability_sha256
            and key[2] == plan.route_id
            and key[3] == plan.route_family
            and key[4] == plan.device_ids
            and key[5] == plan.assisted_operator_kind
            and key[6] == plan.split_axis
            and key[7] == plan.split_fraction_ppm
            and key[8] == plan.residency_variant
        )
        if not matches:
            raise RuntimeLearningError(
                "legacy exact-route observation rows are absent"
            )
        updated = dict(self._rows)
        rebound = 0
        for source_key, rows in matches:
            target = list(source_key)
            target[1] = target_capability_sha256
            target[13] = target_template
            target_key = tuple(target)
            target_rows = (
                tuple(replace(
                    row,
                    energy_uj=None,
                    component_energy_uj=None,
                ) for row in rows)
                if latency_only else rows
            )
            existing = updated.get(target_key)
            if existing is not None and existing != target_rows:
                raise RuntimeLearningError(
                    "exact-route observation target conflicts"
                )
            if target_key != source_key:
                updated[target_key] = target_rows
                del updated[source_key]
            elif latency_only:
                updated[target_key] = target_rows
            rebound += len(rows)
        self._rows = updated
        self._exact_estimate_cache.clear()
        return rebound

    def rebind_legacy_transition_rows(
        self,
        *,
        artifact_sha256: str,
        source_capability_sha256: str,
        target_capability_sha256: str,
        component_identity_sha256: str,
        transitions: Sequence[RuntimeTransitionPlan],
        latency_only: bool = False,
    ) -> int:
        """Re-key transition rows after physical identity validation."""
        values = (
            artifact_sha256,
            source_capability_sha256,
            target_capability_sha256,
            component_identity_sha256,
        )
        if type(latency_only) is not bool or any(
            type(value) is not str
            or not value.startswith("sha256:")
            or len(value) != 71
            or any(
                character not in "0123456789abcdef"
                for character in value[7:]
            )
            for value in values
        ) or any(
            not isinstance(transition, RuntimeTransitionPlan)
            for transition in transitions
        ):
            raise RuntimeLearningError(
                "runtime transition rebind identity is invalid"
            )
        updated = dict(self._transition_rows)
        rebound = 0
        for transition in transitions:
            executor_id = transition.executor_id
            if executor_id is None:
                raise RuntimeLearningError(
                    "runtime transition rebind executor is absent"
                )
            source_key = self._transition_key_values(
                artifact_sha256=artifact_sha256,
                capability_generation_sha256=source_capability_sha256,
                component_identity_sha256=component_identity_sha256,
                transition_id=transition.transition_id,
                source_state=transition.source_state,
                target_state=transition.target_state,
                executor_id=executor_id,
            )
            rows = updated.get(source_key)
            if rows is None:
                raise RuntimeLearningError(
                    "legacy transition observation rows are absent"
                )
            target_key = self._transition_key_values(
                artifact_sha256=artifact_sha256,
                capability_generation_sha256=target_capability_sha256,
                component_identity_sha256=component_identity_sha256,
                transition_id=transition.transition_id,
                source_state=transition.source_state,
                target_state=transition.target_state,
                executor_id=executor_id,
            )
            target_rows = (
                tuple(replace(row, energy_uj=None) for row in rows)
                if latency_only else rows
            )
            existing = updated.get(target_key)
            if existing is not None and existing != target_rows:
                raise RuntimeLearningError(
                    "transition observation target conflicts"
                )
            if target_key != source_key:
                updated[target_key] = target_rows
                del updated[source_key]
            elif latency_only:
                updated[target_key] = target_rows
            rebound += len(rows)
        self._transition_rows = updated
        return rebound

    def checkpoint(self) -> tuple[object, ...]:
        return (
            dict(self._rows),
            dict(self._template_rows),
            dict(self._transition_rows),
            frozenset(self._receipt_ids),
            self._incomplete_receipts,
            self._diagnostic_energy_receipts,
            self._unattributed_transfer_receipts,
        )

    def restore(self, checkpoint: tuple[object, ...]) -> None:
        (
            rows,
            template_rows,
            transition_rows,
            receipt_ids,
            incomplete,
            diagnostic,
            unattributed,
        ) = checkpoint
        self._rows = dict(rows)
        self._template_rows = dict(template_rows)
        self._transition_rows = dict(transition_rows)
        self._receipt_ids = set(receipt_ids)
        self._incomplete_receipts = int(incomplete)
        self._diagnostic_energy_receipts = int(diagnostic)
        self._unattributed_transfer_receipts = int(unattributed)
        self._exact_estimate_cache.clear()
        self._template_estimate_cache.clear()

    def state(self) -> Mapping[str, int]:
        estimates = []
        for key, rows in self._rows.items():
            if len(rows) < self.qualification_samples:
                continue
            calibration = rows[:-2]
            held_out = rows[-2:]
            latency_limit = _ceil_div(
                max(row.latency_us for row in calibration) * 1_100_000,
                1_000_000,
            )
            latency_center = max(
                1,
                sum(row.latency_us for row in calibration)
                // len(calibration),
            )
            latency_mape_ppm = sum(
                abs(row.latency_us - latency_center) * 1_000_000
                // row.latency_us
                for row in held_out
            ) // len(held_out)
            energy_rows = tuple(
                row for row in rows if row.energy_uj is not None
            )
            if len(energy_rows) < self.qualification_samples:
                continue
            energy_calibration = energy_rows[:-2]
            energy_held_out = energy_rows[-2:]
            energy_limit = _ceil_div(
                max(row.energy_uj for row in energy_calibration)
                * 1_100_000,
                1_000_000,
            )
            energy_center = max(
                1,
                sum(row.energy_uj for row in energy_calibration)
                // len(energy_calibration),
            )
            energy_mape_ppm = sum(
                abs(row.energy_uj - energy_center) * 1_000_000
                // row.energy_uj
                for row in energy_held_out
            ) // len(energy_held_out)
            if (
                latency_mape_ppm <= 200_000
                and energy_mape_ppm <= 200_000
                and all(
                    row.latency_us <= latency_limit for row in held_out
                )
                and all(
                    row.energy_uj <= energy_limit
                    for row in energy_held_out
                )
            ):
                estimates.append(key)
        qualified_templates = 0
        quarantined_templates = 0
        for key, rows in self._template_rows.items():
            estimate = self._cached_template_estimate(
                key,
                rows,
                input_tokens=rows[-1].input_tokens,
                output_tokens=rows[-1].output_tokens,
                component_service_us=rows[-1].component_service_us,
                component_energy_uj=rows[-1].component_energy_uj,
            )
            if estimate is not None and estimate.maturity == "QUALIFIED":
                qualified_templates += 1
            elif estimate is not None and estimate.maturity == "QUARANTINED":
                quarantined_templates += 1
        qualified_transitions = 0
        for rows in self._transition_rows.values():
            if len(rows) < self.qualification_samples:
                continue
            calibration = rows[:-2]
            held_out = rows[-2:]
            energy_rows = tuple(
                row for row in rows if row.energy_uj is not None
            )
            if len(energy_rows) < self.qualification_samples:
                continue
            energy_calibration = energy_rows[:-2]
            energy_held_out = energy_rows[-2:]
            latency_center = max(1, _integer_median(tuple(
                row.latency_us for row in calibration
            )))
            energy_center = max(1, _integer_median(tuple(
                row.energy_uj for row in energy_calibration
            )))
            if (
                all(
                    row.latency_us <= _ceil_div(
                        max(item.latency_us for item in calibration)
                        * 1_100_000,
                        1_000_000,
                    )
                    and abs(row.latency_us - latency_center) * 1_000_000
                        // row.latency_us <= 200_000
                    for row in held_out
                )
                and all(
                    row.energy_uj <= _ceil_div(
                        max(item.energy_uj for item in energy_calibration)
                        * 1_100_000,
                        1_000_000,
                    )
                    and abs(row.energy_uj - energy_center) * 1_000_000
                        // row.energy_uj <= 200_000
                    for row in energy_held_out
                )
            ):
                qualified_transitions += 1
        return MappingProxyType({
            "complete_receipts": sum(len(rows) for rows in self._rows.values()),
            "incomplete_receipts": self._incomplete_receipts,
            "diagnostic_energy_receipts": (
                self._diagnostic_energy_receipts
            ),
            "qualified_shape_buckets": len(estimates),
            "shape_buckets": len(self._rows),
            "qualified_templates": qualified_templates,
            "quarantined_templates": quarantined_templates,
            "templates": len(self._template_rows),
            "qualified_transitions": qualified_transitions,
            "transitions": len(self._transition_rows),
            "unattributed_transfer_receipts": (
                self._unattributed_transfer_receipts
            ),
        })
