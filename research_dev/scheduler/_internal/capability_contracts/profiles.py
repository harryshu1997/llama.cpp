"""Device, executor, catalog and snapshot capability contracts: profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from ..runtime_system_cost import ROUTE_MATURITY_STATES
from .common import (
    GENERATED_ROUTE_FAMILIES,
    RESIDENCY_STATES,
    RuntimeCapabilityError,
    SPLIT_AXES,
    _integer,
    _list,
    _object,
    _text,
    _texts,
)


@dataclass(frozen=True)
class RuntimeKernelShapeProfile:
    """Bind one calibrated kernel profile to a bounded request/operator shape."""

    selector_id: str
    operator_kind: str
    profile_id: str
    minimum_input_tokens: int
    maximum_input_tokens: int
    minimum_output_tokens: int
    maximum_output_tokens: int
    minimum_compute_ops: int
    maximum_compute_ops: int
    minimum_memory_bytes: int
    maximum_memory_bytes: int
    maturity: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("selector_id", "operator_kind", "profile_id"):
            _text(f"kernel shape {name}", getattr(self, name))
        for prefix in (
            "input_tokens",
            "output_tokens",
            "compute_ops",
            "memory_bytes",
        ):
            minimum = _integer(
                f"kernel shape minimum {prefix}",
                getattr(self, "minimum_" + prefix),
                1,
            )
            maximum = _integer(
                f"kernel shape maximum {prefix}",
                getattr(self, "maximum_" + prefix),
                1,
            )
            if maximum < minimum:
                raise RuntimeCapabilityError(
                    f"kernel shape {prefix} range is empty"
                )
        if self.maturity not in ROUTE_MATURITY_STATES:
            raise RuntimeCapabilityError("kernel shape maturity is invalid")
        object.__setattr__(
            self,
            "evidence_ids",
            _texts("kernel shape evidence id", self.evidence_ids),
        )

    def matches(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        compute_ops: int,
        memory_bytes: int,
    ) -> bool:
        return (
            self.minimum_input_tokens <= input_tokens <= self.maximum_input_tokens
            and self.minimum_output_tokens
            <= output_tokens
            <= self.maximum_output_tokens
            and self.minimum_compute_ops <= compute_ops <= self.maximum_compute_ops
            and self.minimum_memory_bytes
            <= memory_bytes
            <= self.maximum_memory_bytes
        )

    @property
    def specificity(self) -> tuple[int, int, int, int, str]:
        return (
            self.maximum_input_tokens - self.minimum_input_tokens,
            self.maximum_output_tokens - self.minimum_output_tokens,
            self.maximum_compute_ops - self.minimum_compute_ops,
            self.maximum_memory_bytes - self.minimum_memory_bytes,
            self.selector_id,
        )

    def to_json(self) -> dict[str, object]:
        result = {
            "evidence_ids": list(self.evidence_ids),
            "maturity": self.maturity,
            "maximum_compute_ops": self.maximum_compute_ops,
            "maximum_input_tokens": self.maximum_input_tokens,
            "maximum_memory_bytes": self.maximum_memory_bytes,
            "maximum_output_tokens": self.maximum_output_tokens,
            "minimum_compute_ops": self.minimum_compute_ops,
            "minimum_input_tokens": self.minimum_input_tokens,
            "minimum_memory_bytes": self.minimum_memory_bytes,
            "minimum_output_tokens": self.minimum_output_tokens,
            "operator_kind": self.operator_kind,
            "profile_id": self.profile_id,
            "selector_id": self.selector_id,
        }
        return result

    @classmethod
    def from_json(cls, value: object) -> "RuntimeKernelShapeProfile":
        row = _object("runtime kernel shape profile", value)
        return cls(
            selector_id=row.get("selector_id"),
            operator_kind=row.get("operator_kind"),
            profile_id=row.get("profile_id"),
            minimum_input_tokens=row.get("minimum_input_tokens"),
            maximum_input_tokens=row.get("maximum_input_tokens"),
            minimum_output_tokens=row.get("minimum_output_tokens"),
            maximum_output_tokens=row.get("maximum_output_tokens"),
            minimum_compute_ops=row.get("minimum_compute_ops"),
            maximum_compute_ops=row.get("maximum_compute_ops"),
            minimum_memory_bytes=row.get("minimum_memory_bytes"),
            maximum_memory_bytes=row.get("maximum_memory_bytes"),
            maturity=row.get("maturity"),
            evidence_ids=tuple(_list(
                "runtime kernel shape evidence", row.get("evidence_ids")
            )),
        )


@dataclass(frozen=True)
class RuntimeRouteShapeProfile:
    """Bind a held-out end-to-end cost to one physical route shape."""

    selector_id: str
    artifact_sha256: str
    route_family: str
    device_ids: tuple[str, ...]
    assisted_operator_kind: str | None
    split_axis: str
    split_fraction_ppm: int
    residency_variant: str
    minimum_input_tokens: int
    maximum_input_tokens: int
    minimum_output_tokens: int
    maximum_output_tokens: int
    service_fixed_us: int
    service_input_token_us: int
    service_output_token_us: int
    service_upper_add_us: int
    energy_fixed_uj: int
    energy_input_token_uj: int
    energy_output_token_uj: int
    energy_lower_error_ppm: int
    energy_upper_error_ppm: int
    sample_count: int
    maturity: str
    evidence_ids: tuple[str, ...]
    feature_ranges: Mapping[str, tuple[int, int]] = field(
        default_factory=dict
    )
    service_feature_coefficients_us: Mapping[str, int] = field(
        default_factory=dict
    )
    energy_feature_coefficients_uj: Mapping[str, int] = field(
        default_factory=dict
    )
    executor_id: str | None = None

    def __post_init__(self) -> None:
        _text("route shape selector id", self.selector_id)
        digest = _text("route shape artifact hash", self.artifact_sha256)
        if (
            not digest.startswith("sha256:")
            or len(digest) != 71
            or any(value not in "0123456789abcdef" for value in digest[7:])
        ):
            raise RuntimeCapabilityError(
                "route shape artifact hash must be SHA-256"
            )
        if self.route_family not in GENERATED_ROUTE_FAMILIES:
            raise RuntimeCapabilityError(
                "route shape family is unsupported"
            )
        devices = _texts("route shape device id", self.device_ids)
        if self.executor_id is not None:
            _text("route shape executor id", self.executor_id)
        if self.assisted_operator_kind is not None:
            _text(
                "route shape assisted operator",
                self.assisted_operator_kind,
            )
        if self.split_axis != "none" and self.split_axis not in SPLIT_AXES:
            raise RuntimeCapabilityError("route shape split axis is invalid")
        split = _integer(
            "route shape split fraction", self.split_fraction_ppm
        )
        if self.route_family == "operator_split":
            if self.split_axis == "none" or not 0 < split < 1_000_000:
                raise RuntimeCapabilityError(
                    "split route shape requires a proper fraction"
                )
        elif self.split_axis != "none" or split:
            raise RuntimeCapabilityError(
                "unsplit route shape carries a split"
            )
        if self.residency_variant not in RESIDENCY_STATES:
            raise RuntimeCapabilityError(
                "route shape residency state is invalid"
            )
        for prefix in ("input_tokens", "output_tokens"):
            minimum = _integer(
                f"route shape minimum {prefix}",
                getattr(self, "minimum_" + prefix),
                1,
            )
            maximum = _integer(
                f"route shape maximum {prefix}",
                getattr(self, "maximum_" + prefix),
                1,
            )
            if maximum < minimum:
                raise RuntimeCapabilityError(
                    f"route shape {prefix} range is empty"
                )
        for name in (
            "service_fixed_us",
            "service_input_token_us",
            "service_output_token_us",
            "service_upper_add_us",
            "energy_fixed_uj",
            "energy_input_token_uj",
            "energy_output_token_uj",
            "energy_lower_error_ppm",
            "energy_upper_error_ppm",
        ):
            value = _integer(
                "route shape " + name.replace("_", " "),
                getattr(self, name),
            )
            if name.endswith("error_ppm") and value >= 1_000_000:
                raise RuntimeCapabilityError(
                    "route shape energy error must be below one"
                )
        samples = _integer(
            "route shape sample count", self.sample_count, 1
        )
        if self.maturity not in ROUTE_MATURITY_STATES:
            raise RuntimeCapabilityError("route shape maturity is invalid")
        if self.maturity == "QUALIFIED" and samples < 2:
            raise RuntimeCapabilityError(
                "qualified route shape requires repeated samples"
            )
        evidence = _texts(
            "route shape evidence id", self.evidence_ids
        )
        ranges = {}
        for name, bounds in self.feature_ranges.items():
            feature = _text("route shape feature", name)
            if (
                type(bounds) not in {tuple, list}
                or len(bounds) != 2
            ):
                raise RuntimeCapabilityError(
                    "route shape feature range is invalid"
                )
            minimum = _integer(
                "route shape feature minimum", bounds[0]
            )
            maximum = _integer(
                "route shape feature maximum", bounds[1]
            )
            if maximum < minimum:
                raise RuntimeCapabilityError(
                    "route shape feature range is empty"
                )
            ranges[feature] = (minimum, maximum)

        def coefficients(name: str, values: Mapping[str, int]):
            rows = {
                _text(f"route shape {name} feature", key): _integer(
                    f"route shape {name} coefficient", value
                )
                for key, value in values.items()
            }
            if set(rows) - set(ranges):
                raise RuntimeCapabilityError(
                    f"route shape {name} coefficient has no feature range"
                )
            return MappingProxyType(dict(sorted(rows.items())))

        object.__setattr__(self, "device_ids", tuple(sorted(devices)))
        object.__setattr__(self, "evidence_ids", evidence)
        object.__setattr__(
            self,
            "feature_ranges",
            MappingProxyType(dict(sorted(ranges.items()))),
        )
        object.__setattr__(
            self,
            "service_feature_coefficients_us",
            coefficients(
                "service", self.service_feature_coefficients_us
            ),
        )
        object.__setattr__(
            self,
            "energy_feature_coefficients_uj",
            coefficients(
                "energy", self.energy_feature_coefficients_uj
            ),
        )

    @property
    def specificity(self) -> tuple[int, int, int, int, str]:
        return (
            -len(self.feature_ranges),
            sum(maximum - minimum for minimum, maximum in (
                self.feature_ranges.values()
            )),
            self.maximum_input_tokens - self.minimum_input_tokens,
            self.maximum_output_tokens - self.minimum_output_tokens,
            self.selector_id,
        )

    def matches(
        self,
        *,
        artifact_sha256: str,
        route_family: str,
        device_ids: tuple[str, ...],
        assisted_operator_kind: str | None,
        split_axis: str,
        split_fraction_ppm: int,
        residency_variant: str,
        input_tokens: int,
        output_tokens: int,
        cost_features: Mapping[str, int],
        executor_id: str | None = None,
    ) -> bool:
        return (
            self.artifact_sha256 == artifact_sha256
            and self.route_family == route_family
            and self.device_ids == tuple(sorted(device_ids))
            and self.assisted_operator_kind == assisted_operator_kind
            and self.split_axis == split_axis
            and self.split_fraction_ppm == split_fraction_ppm
            and self.residency_variant == residency_variant
            and (
                self.executor_id is None
                or self.executor_id == executor_id
            )
            and self.minimum_input_tokens
            <= input_tokens
            <= self.maximum_input_tokens
            and self.minimum_output_tokens
            <= output_tokens
            <= self.maximum_output_tokens
            and all(
                feature in cost_features
                and minimum <= cost_features[feature] <= maximum
                for feature, (minimum, maximum) in (
                    self.feature_ranges.items()
                )
            )
        )

    def service_us(
        self,
        input_tokens: int,
        output_tokens: int,
        cost_features: Mapping[str, int],
    ) -> int:
        return max(
            1,
            self.service_fixed_us
            + self.service_input_token_us * input_tokens
            + self.service_output_token_us * output_tokens
            + sum(
                coefficient * cost_features[feature]
                for feature, coefficient in (
                    self.service_feature_coefficients_us.items()
                )
            ),
        )

    def energy_uj(
        self,
        input_tokens: int,
        output_tokens: int,
        cost_features: Mapping[str, int],
    ) -> int:
        return max(
            1,
            self.energy_fixed_uj
            + self.energy_input_token_uj * input_tokens
            + self.energy_output_token_uj * output_tokens
            + sum(
                coefficient * cost_features[feature]
                for feature, coefficient in (
                    self.energy_feature_coefficients_uj.items()
                )
            ),
        )

    def to_json(self) -> dict[str, object]:
        result = {
            "assisted_operator_kind": self.assisted_operator_kind,
            "artifact_sha256": self.artifact_sha256,
            "device_ids": list(self.device_ids),
            "energy_fixed_uj": self.energy_fixed_uj,
            "energy_input_token_uj": self.energy_input_token_uj,
            "energy_lower_error_ppm": self.energy_lower_error_ppm,
            "energy_output_token_uj": self.energy_output_token_uj,
            "energy_upper_error_ppm": self.energy_upper_error_ppm,
            "evidence_ids": list(self.evidence_ids),
            "energy_feature_coefficients_uj": dict(
                self.energy_feature_coefficients_uj
            ),
            "feature_ranges": {
                key: list(value)
                for key, value in self.feature_ranges.items()
            },
            "maturity": self.maturity,
            "maximum_input_tokens": self.maximum_input_tokens,
            "maximum_output_tokens": self.maximum_output_tokens,
            "minimum_input_tokens": self.minimum_input_tokens,
            "minimum_output_tokens": self.minimum_output_tokens,
            "residency_variant": self.residency_variant,
            "route_family": self.route_family,
            "sample_count": self.sample_count,
            "selector_id": self.selector_id,
            "service_fixed_us": self.service_fixed_us,
            "service_input_token_us": self.service_input_token_us,
            "service_output_token_us": self.service_output_token_us,
            "service_feature_coefficients_us": dict(
                self.service_feature_coefficients_us
            ),
            "service_upper_add_us": self.service_upper_add_us,
            "split_axis": self.split_axis,
            "split_fraction_ppm": self.split_fraction_ppm,
        }
        if self.executor_id is not None:
            result["executor_id"] = self.executor_id
        return result

    @classmethod
    def from_json(cls, value: object) -> "RuntimeRouteShapeProfile":
        row = _object("runtime route shape profile", value)
        return cls(
            selector_id=row.get("selector_id"),
            artifact_sha256=row.get("artifact_sha256"),
            route_family=row.get("route_family"),
            device_ids=tuple(_list(
                "runtime route shape devices", row.get("device_ids")
            )),
            assisted_operator_kind=row.get("assisted_operator_kind"),
            split_axis=row.get("split_axis"),
            split_fraction_ppm=row.get("split_fraction_ppm"),
            residency_variant=row.get("residency_variant"),
            minimum_input_tokens=row.get("minimum_input_tokens"),
            maximum_input_tokens=row.get("maximum_input_tokens"),
            minimum_output_tokens=row.get("minimum_output_tokens"),
            maximum_output_tokens=row.get("maximum_output_tokens"),
            service_fixed_us=row.get("service_fixed_us"),
            service_input_token_us=row.get("service_input_token_us"),
            service_output_token_us=row.get("service_output_token_us"),
            service_upper_add_us=row.get("service_upper_add_us"),
            energy_fixed_uj=row.get("energy_fixed_uj"),
            energy_input_token_uj=row.get("energy_input_token_uj"),
            energy_output_token_uj=row.get("energy_output_token_uj"),
            energy_lower_error_ppm=row.get("energy_lower_error_ppm"),
            energy_upper_error_ppm=row.get("energy_upper_error_ppm"),
            sample_count=row.get("sample_count"),
            maturity=row.get("maturity"),
            evidence_ids=tuple(_list(
                "runtime route shape evidence", row.get("evidence_ids")
            )),
            feature_ranges={
                key: tuple(value)
                for key, value in _object(
                    "runtime route shape feature ranges",
                    row.get("feature_ranges", {}),
                ).items()
            },
            service_feature_coefficients_us=dict(_object(
                "runtime route shape service coefficients",
                row.get("service_feature_coefficients_us", {}),
            )),
            energy_feature_coefficients_uj=dict(_object(
                "runtime route shape energy coefficients",
                row.get("energy_feature_coefficients_uj", {}),
            )),
            executor_id=row.get("executor_id"),
        )
