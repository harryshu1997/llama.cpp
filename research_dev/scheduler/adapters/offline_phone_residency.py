"""Execute scheduler-authorized phone residency before request admission."""

from __future__ import annotations

from dataclasses import dataclass
import time
from types import MappingProxyType
from typing import Callable, Mapping, Sequence, TYPE_CHECKING

from .._internal.offline_phone_residency import (
    OfflinePhoneResidencyPlan,
    OfflinePhoneResidencyStage,
)
from .._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from .._internal.lifecycle import PhoneTelemetryUnavailable
from .._internal.runtime_plan import RuntimeTransitionReceipt
from .contracts import (
    PhysicalAdapterError,
    PhysicalExecutionBackend,
    RawTransitionObservation,
)
from .receipts import transition_receipt_from_observation
from .ticket import (
    PhysicalTransitionCommand,
    interpret_offline_phone_residency_stage,
)

if TYPE_CHECKING:
    from ..scheduler import UnifiedScheduler


OfflinePhoneSnapshotProvider = Callable[
    [OfflinePhoneResidencyStage, int], HeterogeneousRuntimeSnapshot
]
OfflinePhonePayloadProvider = Callable[[OfflinePhoneResidencyStage], object]


@dataclass(frozen=True)
class OfflinePhoneResidencyResult:
    plan: OfflinePhoneResidencyPlan
    commands: tuple[PhysicalTransitionCommand, ...]
    receipts: tuple[RuntimeTransitionReceipt, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "commands": [row.to_json() for row in self.commands],
            "plan": self.plan.to_json(),
            "receipts": [row.to_json() for row in self.receipts],
            "schema": "research-offline-phone-residency-result-v1",
        }


def _failure_detail(error: BaseException) -> str:
    value = str(error).encode(
        "ascii", errors="backslashreplace"
    ).decode("ascii")
    return (" ".join(value.split()) or type(error).__name__)[:512]


def _rollback_applied_commands(
    backend: PhysicalExecutionBackend,
    commands: Sequence[PhysicalTransitionCommand],
    error: BaseException,
) -> tuple[tuple[str, ...], Mapping[str, int]]:
    rollback = getattr(backend, "rollback_transition", None)
    unavailable = set()
    restored_generations = {}
    for command in reversed(tuple(commands)):
        changed = tuple(command.transition.changed_phone_session_ids)
        if not callable(rollback):
            unavailable.update(changed)
            error.add_note(
                "offline phone transition rollback is unavailable: "
                + command.transition.transition_id
            )
            continue
        try:
            receipt = rollback(command)
        except BaseException as rollback_error:
            unavailable.update(changed)
            error.add_note(
                "offline phone transition rollback failed: "
                + _failure_detail(rollback_error)
            )
            continue
        if not isinstance(receipt, Mapping) or not receipt.get(
            "physical_change"
        ):
            continue
        restored_empty = set(receipt.get(
            "restored_empty_session_ids", ()
        ))
        if restored_empty == set(changed):
            continue
        raw_generations = receipt.get(
            "restored_session_generations", {}
        )
        if not isinstance(raw_generations, Mapping) or any(
            type(raw_generations.get(session_id)) is not int
            or int(raw_generations[session_id]) < 1
            for session_id in changed
        ):
            unavailable.update(changed)
            error.add_note(
                "offline phone rollback lacks physical epochs: "
                + command.transition.transition_id
            )
            continue
        restored_generations.update({
            session_id: int(raw_generations[session_id])
            for session_id in changed
        })
    for session_id in unavailable:
        restored_generations.pop(session_id, None)
    return (
        tuple(sorted(unavailable)),
        MappingProxyType(dict(sorted(restored_generations.items()))),
    )


class CanonicalOfflinePhoneResidencyPreloader:
    """Drive scheduler stages through physical load and verification."""

    @staticmethod
    def _with_observation_refresh(
        operation, *, snapshot, snapshot_provider,
        refresh_observation, epoch_ns, observation_timeout_s=15,
    ) -> OfflinePhoneResidencyPlan:
        """Retry a deferred offline plan without owning execution resources."""
        deadline = time.monotonic() + observation_timeout_s
        while True:
            try:
                return operation(snapshot)
            except PhoneTelemetryUnavailable:
                refresh_observation()
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
                snapshot = snapshot_provider(max(
                    snapshot.captured_at_us,
                    (time.monotonic_ns() - epoch_ns) // 1000,
                ))

    @staticmethod
    def plan_with_observation_refresh(scheduler, requests_by_model, **options):
        return CanonicalOfflinePhoneResidencyPreloader._with_observation_refresh(
            lambda snapshot: scheduler.plan_offline_phone_residency(
                requests_by_model, snapshot=snapshot,
                observed_at_us=snapshot.captured_at_us,
            ), **options,
        )

    @staticmethod
    def next_with_observation_refresh(scheduler, plan_id, **options):
        return CanonicalOfflinePhoneResidencyPreloader._with_observation_refresh(
            lambda snapshot: scheduler.next_offline_phone_residency_stage(
                plan_id, snapshot=snapshot, observed_at_us=snapshot.captured_at_us,
            ), **options,
        )

    def __init__(
        self,
        scheduler: "UnifiedScheduler",
        backend: PhysicalExecutionBackend,
        *,
        epoch_ns: int,
        snapshot_provider: OfflinePhoneSnapshotProvider,
        preparation_wait_timeout_s: float = 300,
    ) -> None:
        required = (
            "begin_offline_phone_residency_stage",
            "check_offline_phone_residency_stage",
            "complete_offline_phone_residency_stage",
            "fail_offline_phone_residency_stage",
            "next_offline_phone_residency_stage",
            "offline_phone_residency_stage",
        )
        if any(
            not callable(getattr(scheduler, name, None))
            for name in required
        ):
            raise PhysicalAdapterError(
                "offline phone scheduler interface is invalid"
            )
        if not callable(getattr(backend, "apply_transition", None)):
            raise PhysicalAdapterError(
                "offline phone physical backend is invalid"
            )
        if type(epoch_ns) is not int or epoch_ns < 0:
            raise PhysicalAdapterError(
                "offline phone physical epoch is invalid"
            )
        if not callable(snapshot_provider):
            raise PhysicalAdapterError(
                "offline phone snapshot provider is invalid"
            )
        if preparation_wait_timeout_s <= 0:
            raise PhysicalAdapterError(
                "offline phone preparation wait timeout is invalid"
            )
        self._scheduler = scheduler
        self._backend = backend
        self._epoch_ns = epoch_ns
        self._snapshot_provider = snapshot_provider
        self._preparation_wait_timeout_s = preparation_wait_timeout_s

    def _now_us(self, minimum_us: int = 0) -> int:
        return max(
            minimum_us,
            (time.monotonic_ns() - self._epoch_ns) // 1000,
        )

    def execute_next_stage(
        self,
        plan_id: str,
        payload: object,
        *,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
    ) -> OfflinePhoneResidencyResult:
        deadline = time.monotonic() + self._preparation_wait_timeout_s
        observed_at_us = max(observed_at_us, snapshot.captured_at_us)
        while True:
            decision = self._scheduler.begin_offline_phone_residency_stage(
                plan_id,
                snapshot=snapshot,
                observed_at_us=observed_at_us,
            )
            status = decision.get("status")
            if status == "OWNER":
                break
            if status != "DEFERRED" or time.monotonic() >= deadline:
                raise PhysicalAdapterError(
                    "offline phone residency stage did not acquire: "
                    + str(status)
                )
            time.sleep(0.05)
            proposed = self._scheduler.offline_phone_residency_stage(
                plan_id
            )
            if proposed is None:
                raise PhysicalAdapterError(
                    "offline phone residency proposed stage is absent"
                )
            observed_at_us = self._now_us(observed_at_us)
            snapshot = self._snapshot_provider(
                proposed, observed_at_us
            )
            observed_at_us = max(
                observed_at_us, snapshot.captured_at_us
            )
        stage = self._scheduler.offline_phone_residency_stage(plan_id)
        if stage is None or stage.state != "LOADING":
            raise PhysicalAdapterError(
                "offline phone residency loading stage is absent"
            )
        commands = interpret_offline_phone_residency_stage(stage)
        receipts = []
        applied = []

        def control_check() -> None:
            self._scheduler.check_offline_phone_residency_stage(
                plan_id,
                observed_at_us=self._now_us(observed_at_us),
            )

        try:
            for command in commands:
                observation = self._backend.apply_transition(
                    command, payload, control_check
                )
                if not isinstance(observation, RawTransitionObservation):
                    raise PhysicalAdapterError(
                        "offline phone backend returned an invalid transition"
                    )
                receipt = transition_receipt_from_observation(
                    command, observation
                )
                if receipt.status != "COMPLETED":
                    raise PhysicalAdapterError(
                        "offline phone transition failed"
                    )
                receipts.append(receipt)
                applied.append(command)
            if not receipts:
                raise PhysicalAdapterError(
                    "offline phone transition command is absent"
                )
            finished_at_us = max(row.finished_us for row in receipts)
            completion_snapshot = self._snapshot_provider(
                stage, finished_at_us
            )
            plan = self._scheduler.complete_offline_phone_residency_stage(
                plan_id,
                tuple(receipts),
                snapshot=completion_snapshot,
            )
            return OfflinePhoneResidencyResult(
                plan=plan,
                commands=commands,
                receipts=tuple(receipts),
            )
        except BaseException as error:
            unavailable, restored_generations = _rollback_applied_commands(
                self._backend, applied, error
            )
            failure_options = {}
            if unavailable:
                failure_options["unavailable_session_ids"] = unavailable
            if restored_generations:
                failure_options["restored_session_generations"] = (
                    restored_generations
                )
            try:
                self._scheduler.fail_offline_phone_residency_stage(
                    plan_id,
                    failed_at_us=self._now_us(observed_at_us),
                    reason=(
                        "physical_offline_phone_preparation_failed:"
                        + _failure_detail(error)
                    ),
                    **failure_options,
                )
            except BaseException as scheduler_error:
                error.add_note(
                    "offline phone scheduler rollback failed: "
                    + _failure_detail(scheduler_error)
                )
            raise PhysicalAdapterError(
                "physical offline phone preparation failed"
            ) from error

    def preload(
        self,
        plan: OfflinePhoneResidencyPlan,
        payload_provider: OfflinePhonePayloadProvider,
        *,
        initial_snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
    ) -> OfflinePhoneResidencyResult:
        if not isinstance(plan, OfflinePhoneResidencyPlan):
            raise PhysicalAdapterError(
                "offline phone residency plan is invalid"
            )
        if not callable(payload_provider):
            raise PhysicalAdapterError(
                "offline phone payload provider is invalid"
            )
        commands = []
        receipts = []
        current = plan
        snapshot = initial_snapshot
        next_observed_at_us = observed_at_us
        while current.state not in {"READY", "FAILED"}:
            stage = self._scheduler.offline_phone_residency_stage(
                current.plan_id
            )
            if stage is None:
                raise PhysicalAdapterError(
                    "offline phone residency plan has no pending stage"
                )
            result = self.execute_next_stage(
                current.plan_id,
                payload_provider(stage),
                snapshot=snapshot,
                observed_at_us=next_observed_at_us,
            )
            commands.extend(result.commands)
            receipts.extend(result.receipts)
            current = result.plan
            if current.state == "READY":
                break
            completed = current.current_stage
            assert completed is not None
            next_observed_at_us = max(
                completed.verified_at_us or 0,
                self._now_us(next_observed_at_us),
            )
            snapshot = self._snapshot_provider(
                completed, next_observed_at_us
            )
            current = self.next_with_observation_refresh(
                self._scheduler, current.plan_id, snapshot=snapshot,
                snapshot_provider=lambda at_us: self._snapshot_provider(completed, at_us),
                refresh_observation=lambda: None,
                epoch_ns=self._epoch_ns,
            )
        return OfflinePhoneResidencyResult(
            plan=current,
            commands=tuple(commands),
            receipts=tuple(receipts),
        )
