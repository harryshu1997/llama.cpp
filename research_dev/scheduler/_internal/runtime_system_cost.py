"""Protected-work interference and controller-delay cost contracts."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence


ROUTE_MATURITY_STATES = frozenset({
    "PRIOR_ONLY",
    "CALIBRATION_PENDING",
    "SHADOW",
    "QUALIFIED",
    "QUARANTINED",
})


class RuntimeSystemCostError(ValueError):
    pass


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise RuntimeSystemCostError(
            f"{name} must be non-empty ASCII text"
        )
    return value


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RuntimeSystemCostError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def _boolean(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise RuntimeSystemCostError(f"{name} must be bool")
    return value


def _object(name: str, value: object) -> Mapping[str, object]:
    if type(value) is not dict:
        raise RuntimeSystemCostError(f"{name} must be an object")
    return value


def _list(name: str, value: object) -> list[object]:
    if type(value) is not list:
        raise RuntimeSystemCostError(f"{name} must be a list")
    return value


def calculate_marginal_system_cost(
    *,
    critical_path_end_us: int,
    phase_power_mw: int,
    stranded_idle_power_mw: int,
    causal_tail_power_mw: int,
    interference_ppm: int,
    lower_error_ppm: int,
    upper_error_ppm: int,
    service_us: int,
    service_upper_us: int,
    finish_us: int,
    finish_upper_us: int,
) -> dict[str, int]:
    """Return conservative causal critical-path time and energy."""
    for name, value in (
        ("critical path end", critical_path_end_us),
        ("phase power", phase_power_mw),
        ("stranded idle power", stranded_idle_power_mw),
        ("causal tail power", causal_tail_power_mw),
        ("interference", interference_ppm),
        ("lower error", lower_error_ppm),
        ("upper error", upper_error_ppm),
        ("service", service_us),
        ("service upper", service_upper_us),
        ("finish", finish_us),
        ("finish upper", finish_upper_us),
    ):
        _integer("marginal system " + name, value)
    if lower_error_ppm > 1_000_000 or upper_error_ppm > 1_000_000:
        raise RuntimeSystemCostError(
            "marginal system error must not exceed 100 percent"
        )
    if service_upper_us < service_us or finish_upper_us < finish_us:
        raise RuntimeSystemCostError("marginal system bounds are invalid")

    upper_ppm = (
        interference_ppm * (1_000_000 + upper_error_ppm) + 999_999
    ) // 1_000_000
    lower_ppm = (
        interference_ppm * (1_000_000 - lower_error_ppm)
    ) // 1_000_000
    interference_us = (
        service_us * interference_ppm + 999_999
    ) // 1_000_000
    interference_lower_us = (
        service_us * lower_ppm + 999_999
    ) // 1_000_000
    interference_upper_us = (
        service_upper_us * upper_ppm + 999_999
    ) // 1_000_000
    route_tail_us = max(0, finish_us - critical_path_end_us)
    route_tail_upper_us = max(
        0, finish_upper_us - critical_path_end_us
    )
    critical_path_extension_us = max(interference_us, route_tail_us)
    critical_path_extension_upper_us = max(
        interference_upper_us, route_tail_upper_us
    )
    causal_tail_us = max(0, route_tail_us - interference_us)

    def energy(power_mw: int, duration_us: int) -> int:
        return (power_mw * duration_us + 999) // 1000

    phase_interference_uj = energy(phase_power_mw, interference_us)
    causal_tail_uj = energy(causal_tail_power_mw, causal_tail_us)
    stranded_idle_uj = energy(stranded_idle_power_mw, causal_tail_us)
    total_uj = phase_interference_uj + causal_tail_uj + stranded_idle_uj
    lower_power_ppm = 1_000_000 - lower_error_ppm
    upper_power_ppm = 1_000_000 + upper_error_ppm
    lower_phase_power_mw = (
        phase_power_mw * lower_power_ppm // 1_000_000
    )
    lower_tail_power_mw = (
        (causal_tail_power_mw + stranded_idle_power_mw)
        * lower_power_ppm
        // 1_000_000
    )
    upper_phase_power_mw = (
        phase_power_mw * upper_power_ppm + 999_999
    ) // 1_000_000
    upper_tail_power_mw = (
        (causal_tail_power_mw + stranded_idle_power_mw)
        * upper_power_ppm
        + 999_999
    ) // 1_000_000

    def bounded_energy(
        phase_mw: int,
        tail_mw: int,
        interference_duration_us: int,
        route_tail_duration_us: int,
    ) -> int:
        return energy(phase_mw, interference_duration_us) + energy(
            tail_mw,
            max(0, route_tail_duration_us - interference_duration_us),
        )

    lower_candidates = (
        interference_lower_us,
        interference_upper_us,
        min(
            interference_upper_us,
            max(interference_lower_us, route_tail_us),
        ),
    )
    upper_candidates = (
        interference_lower_us,
        interference_upper_us,
        min(
            interference_upper_us,
            max(interference_lower_us, route_tail_upper_us),
        ),
    )
    lower_uj = min(
        bounded_energy(
            lower_phase_power_mw,
            lower_tail_power_mw,
            duration_us,
            route_tail_us,
        )
        for duration_us in lower_candidates
    )
    upper_uj = max(
        bounded_energy(
            upper_phase_power_mw,
            upper_tail_power_mw,
            duration_us,
            route_tail_upper_us,
        )
        for duration_us in upper_candidates
    )
    return {
        "causal_tail_uj": causal_tail_uj,
        "causal_tail_us": causal_tail_us,
        "critical_path_end_us": critical_path_end_us,
        "critical_path_extension_us": critical_path_extension_us,
        "critical_path_extension_upper_us": (
            critical_path_extension_upper_us
        ),
        "interference_us": interference_us,
        "interference_upper_us": interference_upper_us,
        "lower_uj": lower_uj,
        "phase_interference_uj": phase_interference_uj,
        "stranded_idle_uj": stranded_idle_uj,
        "total_uj": total_uj,
        "upper_uj": upper_uj,
    }


@dataclass(frozen=True)
class RuntimeProtectedWorkObservation:
    observation_id: str
    critical_path_end_us: int
    phase_power_mw: int
    stranded_idle_power_mw: int
    causal_tail_power_mw: int
    sample_count: int
    measured: bool

    def __post_init__(self) -> None:
        _text("protected work observation id", self.observation_id)
        for name in (
            "critical_path_end_us",
            "phase_power_mw",
            "stranded_idle_power_mw",
            "causal_tail_power_mw",
        ):
            _integer(
                "protected work " + name.replace("_", " "),
                getattr(self, name),
            )
        _integer("protected work sample count", self.sample_count, 1)
        _boolean("protected work measured", self.measured)

    def to_json(self) -> dict[str, object]:
        return {
            "causal_tail_power_mw": self.causal_tail_power_mw,
            "critical_path_end_us": self.critical_path_end_us,
            "measured": self.measured,
            "observation_id": self.observation_id,
            "phase_power_mw": self.phase_power_mw,
            "sample_count": self.sample_count,
            "stranded_idle_power_mw": self.stranded_idle_power_mw,
        }

    @classmethod
    def from_json(cls, value: object) -> "RuntimeProtectedWorkObservation":
        row = _object("protected work observation", value)
        return cls(
            observation_id=row.get("observation_id"),
            critical_path_end_us=row.get("critical_path_end_us"),
            phase_power_mw=row.get("phase_power_mw"),
            stranded_idle_power_mw=row.get("stranded_idle_power_mw"),
            causal_tail_power_mw=row.get("causal_tail_power_mw"),
            sample_count=row.get("sample_count"),
            measured=row.get("measured"),
        )


@dataclass(frozen=True)
class RuntimeSystemCostProfile:
    selector_id: str
    feature_ranges: Mapping[str, tuple[int, int]]
    interference_ppm_by_resource: Mapping[str, int]
    control_delay_us: int
    control_delay_upper_us: int
    lower_error_ppm: int
    upper_error_ppm: int
    sample_count: int
    maturity: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _text("system cost selector id", self.selector_id)
        ranges = {}
        for name, bounds in self.feature_ranges.items():
            feature = _text("system cost feature", name)
            if type(bounds) not in {tuple, list} or len(bounds) != 2:
                raise RuntimeSystemCostError(
                    "system cost feature range is invalid"
                )
            minimum = _integer("system cost feature minimum", bounds[0])
            maximum = _integer("system cost feature maximum", bounds[1])
            if maximum < minimum:
                raise RuntimeSystemCostError(
                    "system cost feature range is empty"
                )
            ranges[feature] = (minimum, maximum)
        interference = {
            _text("system cost resource", key): _integer(
                "system cost interference", value
            )
            for key, value in self.interference_ppm_by_resource.items()
        }
        if not interference:
            raise RuntimeSystemCostError(
                "system cost requires resource interference"
            )
        delay = _integer("system cost control delay", self.control_delay_us)
        delay_upper = _integer(
            "system cost control delay upper", self.control_delay_upper_us
        )
        if delay_upper < delay:
            raise RuntimeSystemCostError(
                "system cost control delay bound is invalid"
            )
        for name in ("lower_error_ppm", "upper_error_ppm"):
            value = _integer(
                "system cost " + name.replace("_", " "),
                getattr(self, name),
            )
            if value > 1_000_000:
                raise RuntimeSystemCostError(
                    "system cost error must not exceed 100 percent"
                )
        samples = _integer("system cost sample count", self.sample_count, 1)
        if self.maturity not in ROUTE_MATURITY_STATES:
            raise RuntimeSystemCostError("system cost maturity is invalid")
        if self.maturity == "QUALIFIED" and samples < 2:
            raise RuntimeSystemCostError(
                "qualified system cost requires repeated samples"
            )
        evidence = tuple(
            _text("system cost evidence id", value)
            for value in self.evidence_ids
        )
        if not evidence or len(evidence) != len(set(evidence)):
            raise RuntimeSystemCostError(
                "system cost evidence ids are invalid"
            )
        object.__setattr__(
            self,
            "feature_ranges",
            MappingProxyType(dict(sorted(ranges.items()))),
        )
        object.__setattr__(
            self,
            "interference_ppm_by_resource",
            MappingProxyType(dict(sorted(interference.items()))),
        )
        object.__setattr__(self, "evidence_ids", tuple(sorted(evidence)))

    @property
    def specificity(self) -> tuple[int, int, str]:
        return (
            -len(self.feature_ranges),
            sum(
                maximum - minimum
                for minimum, maximum in self.feature_ranges.values()
            ),
            self.selector_id,
        )

    def matches(self, cost_features: Mapping[str, int]) -> bool:
        return all(
            name in cost_features and minimum <= cost_features[name] <= maximum
            for name, (minimum, maximum) in self.feature_ranges.items()
        )

    def missing_resources(
        self, resource_ids: Sequence[str]
    ) -> tuple[str, ...]:
        return tuple(sorted(
            set(resource_ids) - set(self.interference_ppm_by_resource)
        ))

    def route_cost(
        self,
        resource_ids: Sequence[str],
        observation: RuntimeProtectedWorkObservation,
        *,
        service_us: int,
        service_upper_us: int,
        finish_us: int,
        finish_upper_us: int,
    ) -> dict[str, int | str | bool | list[str]]:
        if not isinstance(observation, RuntimeProtectedWorkObservation):
            raise RuntimeSystemCostError(
                "protected work observation is invalid"
            )
        resources = tuple(
            _text("system cost route resource", value)
            for value in resource_ids
        )
        missing = self.missing_resources(resources)
        if missing:
            raise RuntimeSystemCostError(
                "system cost lacks route resources: " + ",".join(missing)
            )
        interference_ppm = sum(
            self.interference_ppm_by_resource[value]
            for value in set(resources)
        )
        result: dict[str, int | str | bool | list[str]] = dict(
            calculate_marginal_system_cost(
                critical_path_end_us=observation.critical_path_end_us,
                phase_power_mw=observation.phase_power_mw,
                stranded_idle_power_mw=(
                    observation.stranded_idle_power_mw
                ),
                causal_tail_power_mw=observation.causal_tail_power_mw,
                interference_ppm=interference_ppm,
                lower_error_ppm=self.lower_error_ppm,
                upper_error_ppm=self.upper_error_ppm,
                service_us=service_us,
                service_upper_us=service_upper_us,
                finish_us=finish_us,
                finish_upper_us=finish_upper_us,
            )
        )
        result.update({
            "control_delay_upper_us": self.control_delay_upper_us,
            "control_delay_us": self.control_delay_us,
            "evidence_ids": list(self.evidence_ids),
            "interference_ppm": interference_ppm,
            "measured": observation.measured,
            "observation_id": observation.observation_id,
            "profile_id": self.selector_id,
            "resource_ids": sorted(set(resources)),
            "sample_count": min(self.sample_count, observation.sample_count),
            "system_finish_upper_us": (
                observation.critical_path_end_us
                + int(result["critical_path_extension_upper_us"])
            ),
        })
        return result

    def to_json(self) -> dict[str, object]:
        return {
            "control_delay_upper_us": self.control_delay_upper_us,
            "control_delay_us": self.control_delay_us,
            "evidence_ids": list(self.evidence_ids),
            "feature_ranges": {
                key: list(value) for key, value in self.feature_ranges.items()
            },
            "interference_ppm_by_resource": dict(
                self.interference_ppm_by_resource
            ),
            "lower_error_ppm": self.lower_error_ppm,
            "maturity": self.maturity,
            "sample_count": self.sample_count,
            "selector_id": self.selector_id,
            "upper_error_ppm": self.upper_error_ppm,
        }

    @classmethod
    def from_json(cls, value: object) -> "RuntimeSystemCostProfile":
        row = _object("runtime system cost profile", value)
        return cls(
            selector_id=row.get("selector_id"),
            feature_ranges={
                key: tuple(bounds)
                for key, bounds in _object(
                    "runtime system cost feature ranges",
                    row.get("feature_ranges"),
                ).items()
            },
            interference_ppm_by_resource=dict(_object(
                "runtime system cost interference",
                row.get("interference_ppm_by_resource"),
            )),
            control_delay_us=row.get("control_delay_us"),
            control_delay_upper_us=row.get("control_delay_upper_us"),
            lower_error_ppm=row.get("lower_error_ppm"),
            upper_error_ppm=row.get("upper_error_ppm"),
            sample_count=row.get("sample_count"),
            maturity=row.get("maturity"),
            evidence_ids=tuple(_list(
                "runtime system cost evidence", row.get("evidence_ids")
            )),
        )
