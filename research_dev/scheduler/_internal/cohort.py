"""Admission for exact measured workload cohorts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .runtime_gates import (
    RequestSemantics,
    RouteRuntimeContract,
    RuntimeGateReceipt,
    RuntimeSnapshot,
    evaluate_runtime_gate,
)
from .types import (
    AccountingKind,
    AccountingScope,
    CandidateSet,
    PlacementGranularity,
    RouteAlternative,
    RouteMaturity,
)


__all__ = [
    "COHORT_POLICY_MODES",
    "CohortDecision",
    "CohortPolicy",
    "CohortScheduleError",
    "CohortPlanner",
    "cohort_decision_to_json",
]


COHORT_POLICY_MODES = {"control", "enforce", "shadow"}


class CohortScheduleError(ValueError):
    pass


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CohortScheduleError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class CohortPolicy:
    energy_saving_ppm: int = 100_000
    latency_regression_ppm: int = 0
    max_exposed_join_wait_ppm: int = 50_000
    minimum_samples: int = 3

    def __post_init__(self) -> None:
        _integer("cohort energy_saving_ppm", self.energy_saving_ppm)
        _integer("cohort latency_regression_ppm", self.latency_regression_ppm)
        _integer(
            "cohort max_exposed_join_wait_ppm",
            self.max_exposed_join_wait_ppm,
        )
        _integer("cohort minimum_samples", self.minimum_samples, 1)
        if self.energy_saving_ppm > 1_000_000:
            raise CohortScheduleError("cohort energy saving exceeds 100 percent")


@dataclass(frozen=True)
class CohortDecision:
    profile_id: str
    unit_id: str
    work_set_hash: str
    mode: str
    route_id: str
    fallback_route_id: str | None
    reason: str
    latency_mean_us: int
    latency_upper_us: int
    energy_mean_uj: int | None
    energy_upper_uj: int | None
    runtime_gate: RuntimeGateReceipt | None
    rejected: tuple[tuple[str, str], ...]


class CohortPlanner:
    def __init__(self, policy: CohortPolicy | None = None) -> None:
        self.policy = policy or CohortPolicy()

    def _static_rejection(
        self,
        route: RouteAlternative,
        baseline: RouteAlternative,
        mode: str,
    ) -> str | None:
        if route.maturity not in {RouteMaturity.MEASURED, RouteMaturity.STABLE}:
            return "ROUTE_NOT_MEASURED"
        if not route.placement_verified:
            return "PLACEMENT_UNVERIFIED"
        if not route.latency_us.measured:
            return "LATENCY_NOT_MEASURED"
        if route.latency_us.sample_count < self.policy.minimum_samples:
            return "LATENCY_SAMPLE_COUNT"
        baseline_lower = baseline.latency_us.lower
        if baseline_lower is None:
            return "BASELINE_LATENCY_LOWER_MISSING"
        latency_limit = (
            baseline_lower * (1_000_000 + self.policy.latency_regression_ppm)
        ) // 1_000_000
        if route.latency_us.upper > latency_limit:
            return "LATENCY_LIMIT"

        needs_overlap = (
            route.placement_granularity != PlacementGranularity.TASK
            and len(route.resources) > 1
        )
        if needs_overlap:
            if route.overlap.status != "measured":
                return "OVERLAP_NOT_MEASURED"
            if (
                route.overlap.upper_join_wait_ppm is None
                or route.overlap.upper_join_wait_ppm
                > self.policy.max_exposed_join_wait_ppm
            ):
                return "OVERLAP_LIMIT"

        if mode != "enforce":
            return None
        baseline_energy = baseline.energy_uj
        route_energy = route.energy_uj
        if baseline_energy is None or route_energy is None:
            return "ENERGY_NOT_MEASURED"
        if not baseline_energy.measured or not route_energy.measured:
            return "ENERGY_NOT_MEASURED"
        if (
            baseline_energy.sample_count < self.policy.minimum_samples
            or route_energy.sample_count < self.policy.minimum_samples
        ):
            return "ENERGY_SAMPLE_COUNT"
        if baseline_energy.lower is None:
            return "BASELINE_ENERGY_LOWER_MISSING"
        if any(
            component.accounting.scope != AccountingScope.COHORT
            or component.accounting.kind
            != AccountingKind.NON_ADDITIVE_COHORT_TOTAL
            for component in (*baseline.energy, *route.energy)
        ):
            return "ENERGY_ACCOUNTING_SCOPE"
        threshold = (
            baseline_energy.lower
            * (1_000_000 - self.policy.energy_saving_ppm)
        ) // 1_000_000
        if route_energy.upper > threshold:
            return "ENERGY_MARGIN"
        return None

    @staticmethod
    def _runtime_gate(
        route: RouteAlternative,
        contracts: Mapping[str, RouteRuntimeContract],
        snapshot: RuntimeSnapshot | None,
        now_us: int | None,
        semantics: RequestSemantics,
        required: bool,
    ) -> RuntimeGateReceipt | None:
        contract = contracts.get(route.route_id)
        if contract is None:
            if required:
                return RuntimeGateReceipt(
                    admitted=False,
                    reason="RUNTIME_CONTRACT_MISSING",
                    snapshot_id=None,
                    snapshot_generation=None,
                    epoch_key=None,
                    checked_resources=(),
                )
            return None
        if not required and snapshot is None:
            return None
        return evaluate_runtime_gate(contract, semantics, snapshot, now_us)

    def schedule(
        self,
        candidates: CandidateSet,
        mode: str,
        *,
        runtime_contracts: Mapping[str, RouteRuntimeContract] | None = None,
        runtime_snapshot: RuntimeSnapshot | None = None,
        runtime_now_us: int | None = None,
        semantics: RequestSemantics | None = None,
        require_runtime_gates: bool = False,
    ) -> CohortDecision:
        if mode not in COHORT_POLICY_MODES:
            raise CohortScheduleError("unknown cohort policy mode")
        if candidates.unit.kind.value != "cohort":
            raise CohortScheduleError("cohort scheduler requires a cohort unit")
        baseline = next(route for route in candidates.routes if route.baseline)
        contracts = runtime_contracts or {}
        request_semantics = semantics or RequestSemantics()

        baseline_runtime = self._runtime_gate(
            baseline,
            contracts,
            runtime_snapshot,
            runtime_now_us,
            request_semantics,
            require_runtime_gates,
        )
        if baseline_runtime is not None and not baseline_runtime.admitted:
            raise CohortScheduleError(
                f"baseline runtime gate failed: {baseline_runtime.reason}"
            )

        rejected: list[tuple[str, str]] = []
        admitted: list[tuple[RouteAlternative, RuntimeGateReceipt | None]] = []
        for route in candidates.routes:
            if route.baseline:
                continue
            reason = self._static_rejection(route, baseline, mode)
            if reason is not None:
                rejected.append((route.route_id, reason))
                continue
            runtime_gate = self._runtime_gate(
                route,
                contracts,
                runtime_snapshot,
                runtime_now_us,
                request_semantics,
                require_runtime_gates,
            )
            if runtime_gate is not None and not runtime_gate.admitted:
                rejected.append((route.route_id, runtime_gate.reason))
                continue
            admitted.append((route, runtime_gate))

        if mode == "control":
            selected = baseline
            selected_runtime = baseline_runtime
            reason = "CONTROL_BASELINE"
        elif mode == "shadow" and admitted:
            selected, selected_runtime = min(
                admitted,
                key=lambda item: (
                    item[0].latency_us.upper,
                    item[0].route_id,
                ),
            )
            reason = "SHADOW_FASTEST_MEASURED_COHORT"
        elif mode == "enforce" and admitted:
            selected, selected_runtime = min(
                admitted,
                key=lambda item: (
                    item[0].energy_uj.upper,
                    item[0].latency_us.upper,
                    item[0].route_id,
                ),
            )
            reason = "VERIFIED_COHORT_ENERGY_SAVING"
        else:
            selected = baseline
            selected_runtime = baseline_runtime
            reason = "FAIL_CLOSED_BASELINE"

        energy = selected.energy_uj
        return CohortDecision(
            profile_id=candidates.profile_id,
            unit_id=candidates.unit.unit_id,
            work_set_hash=candidates.unit.work_set_hash,
            mode=mode,
            route_id=selected.route_id,
            fallback_route_id=(
                None if selected.baseline else baseline.route_id
            ),
            reason=reason,
            latency_mean_us=selected.latency_us.mean,
            latency_upper_us=selected.latency_us.upper,
            energy_mean_uj=None if energy is None else energy.mean,
            energy_upper_uj=None if energy is None else energy.upper,
            runtime_gate=selected_runtime,
            rejected=tuple(sorted(rejected)),
        )


def cohort_decision_to_json(decision: CohortDecision) -> dict[str, object]:
    runtime = decision.runtime_gate
    return {
        "energy_mean_uj": decision.energy_mean_uj,
        "energy_upper_uj": decision.energy_upper_uj,
        "fallback_route_id": decision.fallback_route_id,
        "latency_mean_us": decision.latency_mean_us,
        "latency_upper_us": decision.latency_upper_us,
        "mode": decision.mode,
        "profile_id": decision.profile_id,
        "reason": decision.reason,
        "rejected": [
            {"reason": reason, "route_id": route_id}
            for route_id, reason in decision.rejected
        ],
        "route_id": decision.route_id,
        "runtime_gate": (
            None
            if runtime is None
            else {
                "admitted": runtime.admitted,
                "checked_resources": list(runtime.checked_resources),
                "epoch_key": runtime.epoch_key,
                "reason": runtime.reason,
                "snapshot_generation": runtime.snapshot_generation,
                "snapshot_id": runtime.snapshot_id,
            }
        ),
        "unit_id": decision.unit_id,
        "work_set_hash": decision.work_set_hash,
    }
