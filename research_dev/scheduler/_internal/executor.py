"""Hash-bound execution plans for scheduler-selected backends."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .capacity import LayerPlacementContract
from .cohort import CohortDecision, cohort_decision_to_json
from .offload import BackendArtifact, OperatorOffloadContract
from .runtime_gates import RuntimeGateReceipt
from .types import canonical_sha256


__all__ = [
    "EXECUTION_PLAN_SCHEMA",
    "ExecutionPlan",
    "ExecutionPlanError",
    "build_execution_plan",
    "load_execution_plan",
    "write_execution_plan",
]


EXECUTION_PLAN_SCHEMA = "research-scheduler-execution-plan-v1"


class ExecutionPlanError(ValueError):
    pass


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise ExecutionPlanError(f"{name} must be a non-empty string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ExecutionPlanError(f"{name} must be ASCII") from exc
    return value


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ExecutionPlanError(f"{name} must be an integer >= {minimum}")
    return value


def _sha256(name: str, value: object) -> str:
    result = _text(name, value).removeprefix("sha256:")
    if len(result) != 64 or any(ch not in "0123456789abcdef" for ch in result):
        raise ExecutionPlanError(f"{name} must be a lowercase SHA-256")
    return "sha256:" + result


def _scalar_mapping(
    name: str, value: Mapping[str, bool | int | str]
) -> Mapping[str, bool | int | str]:
    result: dict[str, bool | int | str] = {}
    for key, item in value.items():
        clean_key = _text(f"{name} key", key)
        if type(item) is str:
            result[clean_key] = _text(f"{name} {clean_key}", item)
        elif type(item) is int:
            result[clean_key] = item
        elif type(item) is bool:
            result[clean_key] = item
        else:
            raise ExecutionPlanError(
                f"{name} {clean_key} must be bool, int, or string"
            )
    return MappingProxyType(dict(sorted(result.items())))


def _integer_mapping(name: str, value: Mapping[str, int]) -> Mapping[str, int]:
    result = {
        _text(f"{name} key", key): _integer(f"{name} {key}", item)
        for key, item in value.items()
    }
    return MappingProxyType(dict(sorted(result.items())))


@dataclass(frozen=True)
class ExecutionPlan:
    plan_id: str
    profile_id: str
    epoch_key: str
    decision: CohortDecision
    execution_mode: str
    trace_sha256: str
    request_count: int
    input_tokens: int
    output_tokens: int
    model_hashes: Mapping[str, str]
    artifacts: tuple[BackendArtifact, ...]
    runtime_bindings: Mapping[str, bool | int | str]
    expected_work: Mapping[str, int]
    offload: OperatorOffloadContract | None
    layer_placement: LayerPlacementContract | None
    evidence_ids: tuple[str, ...]
    admission_phase: str
    plan_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "plan_id",
            "profile_id",
            "execution_mode",
            "admission_phase",
        ):
            _text(name, getattr(self, name))
        object.__setattr__(
            self, "epoch_key", _sha256("execution epoch_key", self.epoch_key)
        )
        object.__setattr__(
            self,
            "trace_sha256",
            _sha256("execution trace_sha256", self.trace_sha256),
        )
        object.__setattr__(
            self,
            "plan_sha256",
            _sha256("execution plan_sha256", self.plan_sha256),
        )
        _integer("execution request_count", self.request_count, 1)
        _integer("execution input_tokens", self.input_tokens, 1)
        _integer("execution output_tokens", self.output_tokens, 1)
        if not isinstance(self.decision, CohortDecision):
            raise ExecutionPlanError("execution decision is invalid")
        if self.decision.profile_id != self.profile_id:
            raise ExecutionPlanError("execution profile and decision differ")

        models = {
            _text("model role", role): _sha256(f"model hash {role}", digest)
            for role, digest in self.model_hashes.items()
        }
        if not models:
            raise ExecutionPlanError("execution models must not be empty")
        object.__setattr__(
            self, "model_hashes", MappingProxyType(dict(sorted(models.items())))
        )

        artifacts = tuple(self.artifacts)
        if (
            not artifacts
            or any(not isinstance(item, BackendArtifact) for item in artifacts)
            or len({item.role for item in artifacts}) != len(artifacts)
        ):
            raise ExecutionPlanError("execution artifacts are invalid")
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(
            self,
            "runtime_bindings",
            _scalar_mapping("runtime binding", self.runtime_bindings),
        )
        object.__setattr__(
            self,
            "expected_work",
            _integer_mapping("expected work", self.expected_work),
        )
        evidence = tuple(_text("evidence id", item) for item in self.evidence_ids)
        if not evidence or len(evidence) != len(set(evidence)):
            raise ExecutionPlanError("execution evidence must be non-empty and unique")
        object.__setattr__(self, "evidence_ids", evidence)
        if self.offload is not None:
            if not isinstance(self.offload, OperatorOffloadContract):
                raise ExecutionPlanError("execution offload contract is invalid")
            if self.offload.route_id != self.decision.route_id:
                raise ExecutionPlanError("offload route and decision differ")
        if self.layer_placement is not None:
            if not isinstance(self.layer_placement, LayerPlacementContract):
                raise ExecutionPlanError("execution layer placement is invalid")
            if self.layer_placement.route_id != self.decision.route_id:
                raise ExecutionPlanError(
                    "layer placement route and decision differ"
                )
            if (
                self.layer_placement.selected.model_sha256
                not in self.model_hashes.values()
            ):
                raise ExecutionPlanError(
                    "layer placement model is absent from execution models"
                )
            if self.offload is not None and not set(
                self.offload.split.layer_ids
            ).issubset(self.layer_placement.cpu_layer_ids):
                raise ExecutionPlanError(
                    "phone offload includes a non-CPU-resident layer"
                )

        expected_hash = canonical_sha256(self._without_hash())
        if self.plan_sha256 != expected_hash:
            raise ExecutionPlanError("execution plan hash mismatch")

    def _without_hash(self) -> dict[str, object]:
        value: dict[str, object] = {
            "admission_phase": self.admission_phase,
            "artifacts": [item.to_json() for item in self.artifacts],
            "decision": cohort_decision_to_json(self.decision),
            "epoch_key": self.epoch_key,
            "evidence_ids": list(self.evidence_ids),
            "execution_mode": self.execution_mode,
            "expected_work": dict(self.expected_work),
            "input_tokens": self.input_tokens,
            "model_hashes": dict(self.model_hashes),
            "offload": None if self.offload is None else self.offload.to_json(),
            "output_tokens": self.output_tokens,
            "plan_id": self.plan_id,
            "profile_id": self.profile_id,
            "request_count": self.request_count,
            "runtime_bindings": dict(self.runtime_bindings),
            "schema": EXECUTION_PLAN_SCHEMA,
            "trace_sha256": self.trace_sha256,
        }
        if self.layer_placement is not None:
            value["layer_placement"] = self.layer_placement.to_json()
        return value

    def to_json(self) -> dict[str, object]:
        return {**self._without_hash(), "plan_sha256": self.plan_sha256}

    def artifact(self, role: str) -> BackendArtifact:
        _text("artifact role", role)
        try:
            return next(item for item in self.artifacts if item.role == role)
        except StopIteration as exc:
            raise ExecutionPlanError(f"execution artifact is missing: {role}") from exc

    def validate_runtime_bindings(
        self, observed: Mapping[str, bool | int | str]
    ) -> None:
        for key, value in observed.items():
            if key not in self.runtime_bindings:
                raise ExecutionPlanError(f"runtime binding is undeclared: {key}")
            if self.runtime_bindings[key] != value:
                raise ExecutionPlanError(f"runtime binding differs: {key}")

    def validate_artifact(
        self, role: str, path: str, sha256: str
    ) -> None:
        expected = self.artifact(role)
        if expected.path != path:
            raise ExecutionPlanError(f"artifact path differs: {role}")
        if expected.sha256 != _sha256("observed artifact sha256", sha256):
            raise ExecutionPlanError(f"artifact hash differs: {role}")


def _receipt_from_json(value: object) -> RuntimeGateReceipt | None:
    if value is None:
        return None
    if type(value) is not dict:
        raise ExecutionPlanError("runtime gate must be an object or null")
    checked = value.get("checked_resources")
    if type(checked) is not list:
        raise ExecutionPlanError("runtime checked_resources must be a list")
    admitted = value.get("admitted")
    if type(admitted) is not bool:
        raise ExecutionPlanError("runtime admitted must be bool")
    generation = value.get("snapshot_generation")
    if generation is not None:
        _integer("runtime snapshot_generation", generation)
    epoch_key = value.get("epoch_key")
    if epoch_key is not None:
        epoch_key = _sha256("runtime epoch_key", epoch_key)
    snapshot_id = value.get("snapshot_id")
    if snapshot_id is not None:
        snapshot_id = _text("runtime snapshot_id", snapshot_id)
    return RuntimeGateReceipt(
        admitted=admitted,
        reason=_text("runtime reason", value.get("reason")),
        snapshot_id=snapshot_id,
        snapshot_generation=generation,
        epoch_key=epoch_key,
        checked_resources=tuple(
            _text("runtime checked resource", item) for item in checked
        ),
    )


def _decision_from_json(value: object) -> CohortDecision:
    if type(value) is not dict:
        raise ExecutionPlanError("execution decision must be an object")
    rejected = value.get("rejected")
    if type(rejected) is not list:
        raise ExecutionPlanError("decision rejected must be a list")
    rejected_rows: list[tuple[str, str]] = []
    for row in rejected:
        if type(row) is not dict:
            raise ExecutionPlanError("decision rejection must be an object")
        rejected_rows.append((
            _text("rejected route_id", row.get("route_id")),
            _text("rejected reason", row.get("reason")),
        ))
    energy_mean = value.get("energy_mean_uj")
    energy_upper = value.get("energy_upper_uj")
    if energy_mean is not None:
        _integer("decision energy_mean_uj", energy_mean, 1)
    if energy_upper is not None:
        _integer("decision energy_upper_uj", energy_upper, 1)
    fallback = value.get("fallback_route_id")
    if fallback is not None:
        fallback = _text("decision fallback_route_id", fallback)
    return CohortDecision(
        profile_id=_text("decision profile_id", value.get("profile_id")),
        unit_id=_text("decision unit_id", value.get("unit_id")),
        work_set_hash=_sha256(
            "decision work_set_hash", value.get("work_set_hash")
        ),
        mode=_text("decision mode", value.get("mode")),
        route_id=_text("decision route_id", value.get("route_id")),
        fallback_route_id=fallback,
        reason=_text("decision reason", value.get("reason")),
        latency_mean_us=_integer(
            "decision latency_mean_us", value.get("latency_mean_us"), 1
        ),
        latency_upper_us=_integer(
            "decision latency_upper_us", value.get("latency_upper_us"), 1
        ),
        energy_mean_uj=energy_mean,
        energy_upper_uj=energy_upper,
        runtime_gate=_receipt_from_json(value.get("runtime_gate")),
        rejected=tuple(rejected_rows),
    )


def build_execution_plan(
    *,
    plan_id: str,
    epoch_key: str,
    decision: CohortDecision,
    execution_mode: str,
    trace_sha256: str,
    request_count: int,
    input_tokens: int,
    output_tokens: int,
    model_hashes: Mapping[str, str],
    artifacts: Sequence[BackendArtifact],
    runtime_bindings: Mapping[str, bool | int | str],
    expected_work: Mapping[str, int],
    offload: OperatorOffloadContract | None,
    evidence_ids: Sequence[str],
    layer_placement: LayerPlacementContract | None = None,
    admission_phase: str = "static_selected_runtime_gate_deferred",
) -> ExecutionPlan:
    values = {
        "plan_id": plan_id,
        "profile_id": decision.profile_id,
        "epoch_key": _sha256("execution epoch_key", epoch_key),
        "decision": decision,
        "execution_mode": execution_mode,
        "trace_sha256": _sha256("execution trace_sha256", trace_sha256),
        "request_count": request_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model_hashes": {
            role: _sha256(f"model hash {role}", digest)
            for role, digest in model_hashes.items()
        },
        "artifacts": tuple(artifacts),
        "runtime_bindings": runtime_bindings,
        "expected_work": expected_work,
        "offload": offload,
        "layer_placement": layer_placement,
        "evidence_ids": tuple(evidence_ids),
        "admission_phase": admission_phase,
    }
    provisional = object.__new__(ExecutionPlan)
    for key, value in values.items():
        object.__setattr__(provisional, key, value)
    object.__setattr__(provisional, "plan_sha256", "sha256:" + "0" * 64)
    plan_hash = canonical_sha256(provisional._without_hash())
    return ExecutionPlan(**values, plan_sha256=plan_hash)


def load_execution_plan(path: Path) -> ExecutionPlan:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExecutionPlanError(f"cannot read execution plan: {exc}") from exc
    if type(value) is not dict or value.get("schema") != EXECUTION_PLAN_SCHEMA:
        raise ExecutionPlanError("execution plan schema mismatch")
    raw_models = value.get("model_hashes")
    raw_artifacts = value.get("artifacts")
    raw_runtime = value.get("runtime_bindings")
    raw_work = value.get("expected_work")
    raw_evidence = value.get("evidence_ids")
    if not all(
        type(item) is expected
        for item, expected in (
            (raw_models, dict),
            (raw_artifacts, list),
            (raw_runtime, dict),
            (raw_work, dict),
            (raw_evidence, list),
        )
    ):
        raise ExecutionPlanError("execution plan collection field")
    return ExecutionPlan(
        plan_id=value.get("plan_id"),
        profile_id=value.get("profile_id"),
        epoch_key=value.get("epoch_key"),
        decision=_decision_from_json(value.get("decision")),
        execution_mode=value.get("execution_mode"),
        trace_sha256=value.get("trace_sha256"),
        request_count=value.get("request_count"),
        input_tokens=value.get("input_tokens"),
        output_tokens=value.get("output_tokens"),
        model_hashes=raw_models,
        artifacts=tuple(BackendArtifact.from_json(item) for item in raw_artifacts),
        runtime_bindings=raw_runtime,
        expected_work=raw_work,
        offload=(
            None
            if value.get("offload") is None
            else OperatorOffloadContract.from_json(value.get("offload"))
        ),
        layer_placement=(
            None
            if value.get("layer_placement") is None
            else LayerPlacementContract.from_json(value.get("layer_placement"))
        ),
        evidence_ids=tuple(raw_evidence),
        admission_phase=value.get("admission_phase"),
        plan_sha256=value.get("plan_sha256"),
    )


def write_execution_plan(path: Path, plan: ExecutionPlan) -> None:
    if path.exists():
        raise ExecutionPlanError(f"execution plan already exists: {path}")
    path.write_text(
        json.dumps(plan.to_json(), ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="ascii",
    )
