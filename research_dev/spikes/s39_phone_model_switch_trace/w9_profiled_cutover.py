#!/usr/bin/env python3
"""W9 profiled cutover policy and durable publication ledger."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import phone_cuda_delta_probe as w6


CONTRACT_SCHEMA = "s39-profiled-cutover-contract-v1"
LEDGER_SCHEMA = "s39-profiled-cutover-ledger-record-v1"
ZERO_SHA256 = "0" * 64
HEX64 = re.compile(r"[0-9a-f]{64}")


class W9Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise W9Error(message)


def is_int(value: object) -> bool:
    return type(value) is int


def exact_keys(
    value: object,
    expected: set[str],
    field: str,
) -> dict[str, Any]:
    require(type(value) is dict, f"{field}: expected object")
    require(set(value) == expected, f"{field}: unexpected keys")
    return value


def checked_digest(value: object, field: str) -> str:
    require(
        type(value) is str and HEX64.fullmatch(value) is not None,
        f"{field}: invalid SHA-256",
    )
    return value


def checked_nonnegative(value: object, field: str) -> int:
    require(is_int(value) and value >= 0, f"{field}: expected nonnegative integer")
    return value


def checked_positive(value: object, field: str) -> int:
    require(is_int(value) and value > 0, f"{field}: expected positive integer")
    return value


@dataclass(frozen=True)
class CutoverContract:
    raw_sha256: str
    w8_contract_sha256: str
    base_contract_sha256: str
    delta_contract_sha256: str
    physical_gate_sha256: str
    w8_manifest_sha256: str
    w8_treatment_report_sha256: str
    batch: int
    output_tokens: int
    preexisting_committed_tokens: int
    max_inflight_tokens: int
    min_cuda_continuation_tokens: int
    cutover_margin_us: int
    phone_batch_estimate_us: int
    max_cuda_ready_us: int
    max_launch_delay_us: int
    paired_launch_delta_us: int
    phone_thermal_range_millic: int
    idle_samples: int
    idle_span_us: int
    cuda_replay_us: int
    predicted_commit_us: int
    phone_extra_us: tuple[int, ...]
    delta_ingest_us: tuple[int, ...]
    cuda_tokens_us: tuple[int, ...]
    candidate_k: tuple[int, ...]
    pair_ordinals: tuple[str, ...]
    completion_ratio_num: int
    completion_ratio_den: int
    next_token_ratio_num: int
    next_token_ratio_den: int
    source_sha256: dict[str, str]
    reference_source_bindings: tuple[dict[str, object], ...]
    cuda_uuid: str
    cuda_name: str
    op15_serial: str
    op15_wifi: str
    op12_serial: str
    op12_wifi: str


@dataclass(frozen=True)
class CutoverDecision:
    k_extra: int
    k_max: int
    phone_tokens_at_f0: int
    inflight_present: bool
    inflight_elapsed_us: int
    predicted_inflight_remaining_us: int
    candidates: tuple[dict[str, int | bool], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "candidates": [dict(item) for item in self.candidates],
            "inflight_elapsed_us": self.inflight_elapsed_us,
            "inflight_present": self.inflight_present,
            "k_extra": self.k_extra,
            "k_max": self.k_max,
            "phone_tokens_at_f0": self.phone_tokens_at_f0,
            "predicted_inflight_remaining_us": (
                self.predicted_inflight_remaining_us
            ),
        }


def _integer_vector(value: object, field: str) -> tuple[int, ...]:
    require(type(value) is list and bool(value), f"{field}: expected nonempty list")
    result = tuple(
        checked_nonnegative(item, f"{field}.{index}")
        for index, item in enumerate(value)
    )
    require(
        all(left <= right for left, right in zip(result, result[1:])),
        f"{field}: vector is not monotone",
    )
    return result


def load_contract(path: Path) -> CutoverContract:
    value, raw = w6.read_canonical(path, "w9_contract")
    root = exact_keys(
        value,
        {
            "dependencies",
            "devices",
            "execution",
            "pair_ordinals",
            "performance_gates",
            "policy",
            "reference_profile",
            "requirements",
            "scheduler_eligible_on_pass",
            "schema",
            "scope",
            "source_sha256",
            "status",
        },
        "w9_contract",
    )
    require(
        root["schema"] == CONTRACT_SCHEMA
        and root["status"] == "FROZEN_BEFORE_ACQUISITION"
        and root["scope"] == "MECHANICS_ONLY"
        and root["scheduler_eligible_on_pass"] is False,
        "w9_contract: labels",
    )
    dependencies = exact_keys(
        root["dependencies"],
        {
            "base_contract_sha256",
            "delta_contract_sha256",
            "physical_gate_sha256",
            "w8_contract_sha256",
            "w8_manifest_sha256",
            "w8_treatment_report_sha256",
        },
        "w9_contract.dependencies",
    )
    for name, digest in dependencies.items():
        checked_digest(digest, f"w9_contract.dependencies.{name}")
    devices = exact_keys(
        root["devices"],
        {"cuda", "op12", "op15"},
        "w9_contract.devices",
    )
    cuda_device = exact_keys(
        devices["cuda"],
        {"name", "uuid"},
        "w9_contract.devices.cuda",
    )
    require(
        type(cuda_device["uuid"]) is str
        and cuda_device["uuid"].startswith("GPU-")
        and type(cuda_device["name"]) is str
        and cuda_device["name"],
        "w9_contract: CUDA device",
    )
    phone_devices = {}
    for name, expected_layers in (("op15", [0, 30]), ("op12", [30, 48])):
        device = exact_keys(
            devices[name],
            {"layers", "serial", "wifi"},
            f"w9_contract.devices.{name}",
        )
        require(
            device["layers"] == expected_layers
            and type(device["serial"]) is str
            and device["serial"]
            and type(device["wifi"]) is str
            and device["wifi"],
            f"w9_contract: {name} device",
        )
        phone_devices[name] = device

    execution = exact_keys(
        root["execution"],
        {
            "batch",
            "idle_samples",
            "idle_span_us",
            "max_cuda_ready_us",
            "max_inflight_tokens",
            "max_launch_delay_us",
            "min_cuda_continuation_tokens",
            "output_tokens",
            "paired_launch_delta_us",
            "phone_thermal_range_millic",
            "preexisting_committed_tokens",
        },
        "w9_contract.execution",
    )
    positive_execution = {
        name: checked_positive(value, f"w9_contract.execution.{name}")
        for name, value in execution.items()
    }
    require(
        positive_execution["batch"] == 8
        and positive_execution["output_tokens"] == 13
        and positive_execution["preexisting_committed_tokens"] == 2
        and positive_execution["max_inflight_tokens"] == 1
        and positive_execution["min_cuda_continuation_tokens"] == 1
        and positive_execution["idle_samples"] >= 5
        and positive_execution["idle_span_us"] >= 1_000_000,
        "w9_contract: frozen execution geometry",
    )

    policy = exact_keys(
        root["policy"],
        {
            "candidate_k",
            "cuda_replay_us",
            "cuda_tokens_us",
            "cutover_margin_us",
            "delta_ingest_us",
            "phone_batch_estimate_us",
            "phone_extra_us",
            "predicted_commit_us",
        },
        "w9_contract.policy",
    )
    candidate = _integer_vector(policy["candidate_k"], "w9_contract.candidate_k")
    require(
        candidate == tuple(range(len(candidate))),
        "w9_contract: candidate domain is not contiguous",
    )
    phone_extra = _integer_vector(
        policy["phone_extra_us"],
        "w9_contract.phone_extra_us",
    )
    delta_ingest = _integer_vector(
        policy["delta_ingest_us"],
        "w9_contract.delta_ingest_us",
    )
    cuda_tokens = _integer_vector(
        policy["cuda_tokens_us"],
        "w9_contract.cuda_tokens_us",
    )
    require(
        len(phone_extra) == len(candidate)
        and len(delta_ingest)
        >= positive_execution["max_inflight_tokens"] + len(candidate)
        and len(cuda_tokens) > positive_execution["output_tokens"],
        "w9_contract: predictor domain",
    )
    require(
        phone_extra[0] == 0
        and delta_ingest[0] == 0
        and cuda_tokens[0] == 0,
        "w9_contract: cumulative predictor origin",
    )
    scalar_policy = {
        name: checked_nonnegative(policy[name], f"w9_contract.policy.{name}")
        for name in (
            "cuda_replay_us",
            "cutover_margin_us",
            "phone_batch_estimate_us",
            "predicted_commit_us",
        )
    }
    require(
        scalar_policy["cuda_replay_us"] > scalar_policy["cutover_margin_us"] > 0
        and scalar_policy["phone_batch_estimate_us"] > 0,
        "w9_contract: policy timing",
    )

    gates = exact_keys(
        root["performance_gates"],
        {
            "completion_ratio_den",
            "completion_ratio_num",
            "next_token_ratio_den",
            "next_token_ratio_num",
        },
        "w9_contract.performance_gates",
    )
    gate_values = {
        name: checked_positive(value, f"w9_contract.performance_gates.{name}")
        for name, value in gates.items()
    }
    require(
        gate_values
        == {
            "completion_ratio_den": 4,
            "completion_ratio_num": 5,
            "next_token_ratio_den": 4,
            "next_token_ratio_num": 3,
        },
        "w9_contract: performance gates",
    )

    ordinals = root["pair_ordinals"]
    require(
        ordinals == ["P1", "P2", "P3", "P4"],
        "w9_contract: pair ordinals",
    )
    requirements = root["requirements"]
    require(
        type(requirements) is list
        and len(requirements) == len(set(requirements))
        and all(type(item) is str and item for item in requirements),
        "w9_contract: requirements",
    )
    sources = root["source_sha256"]
    require(type(sources) is dict and bool(sources), "w9_contract: source map")
    checked_sources = {}
    for name, digest in sources.items():
        require(
            type(name) is str
            and name
            and "/" not in name
            and name not in (".", ".."),
            "w9_contract: source name",
        )
        checked_sources[name] = checked_digest(
            digest,
            f"w9_contract.source_sha256.{name}",
        )

    contract = CutoverContract(
        raw_sha256=w6.sha256(raw),
        w8_contract_sha256=dependencies["w8_contract_sha256"],
        base_contract_sha256=dependencies["base_contract_sha256"],
        delta_contract_sha256=dependencies["delta_contract_sha256"],
        physical_gate_sha256=dependencies["physical_gate_sha256"],
        w8_manifest_sha256=dependencies["w8_manifest_sha256"],
        w8_treatment_report_sha256=dependencies[
            "w8_treatment_report_sha256"
        ],
        batch=positive_execution["batch"],
        output_tokens=positive_execution["output_tokens"],
        preexisting_committed_tokens=positive_execution[
            "preexisting_committed_tokens"
        ],
        max_inflight_tokens=positive_execution["max_inflight_tokens"],
        min_cuda_continuation_tokens=positive_execution[
            "min_cuda_continuation_tokens"
        ],
        cutover_margin_us=scalar_policy["cutover_margin_us"],
        phone_batch_estimate_us=scalar_policy["phone_batch_estimate_us"],
        max_cuda_ready_us=positive_execution["max_cuda_ready_us"],
        max_launch_delay_us=positive_execution["max_launch_delay_us"],
        paired_launch_delta_us=positive_execution["paired_launch_delta_us"],
        phone_thermal_range_millic=positive_execution[
            "phone_thermal_range_millic"
        ],
        idle_samples=positive_execution["idle_samples"],
        idle_span_us=positive_execution["idle_span_us"],
        cuda_replay_us=scalar_policy["cuda_replay_us"],
        predicted_commit_us=scalar_policy["predicted_commit_us"],
        phone_extra_us=phone_extra,
        delta_ingest_us=delta_ingest,
        cuda_tokens_us=cuda_tokens,
        candidate_k=candidate,
        pair_ordinals=tuple(ordinals),
        completion_ratio_num=gate_values["completion_ratio_num"],
        completion_ratio_den=gate_values["completion_ratio_den"],
        next_token_ratio_num=gate_values["next_token_ratio_num"],
        next_token_ratio_den=gate_values["next_token_ratio_den"],
        source_sha256=checked_sources,
        reference_source_bindings=(),
        cuda_uuid=cuda_device["uuid"],
        cuda_name=cuda_device["name"],
        op15_serial=phone_devices["op15"]["serial"],
        op15_wifi=phone_devices["op15"]["wifi"],
        op12_serial=phone_devices["op12"]["serial"],
        op12_wifi=phone_devices["op12"]["wifi"],
    )
    reference = exact_keys(
        root["reference_profile"],
        {
            "expected_k_extra",
            "inflight_elapsed_us",
            "inflight_present",
            "phone_tokens_at_f0",
            "source_bindings",
        },
        "w9_contract.reference_profile",
    )
    require(
        reference["inflight_present"] is True
        and checked_nonnegative(
            reference["inflight_elapsed_us"],
            "w9_contract.reference_profile.inflight_elapsed_us",
        )
        < contract.phone_batch_estimate_us
        and checked_nonnegative(
            reference["phone_tokens_at_f0"],
            "w9_contract.reference_profile.phone_tokens_at_f0",
        )
        < contract.output_tokens
        and reference["expected_k_extra"] == 0,
        "w9_contract: reference profile",
    )
    bindings = reference["source_bindings"]
    require(type(bindings) is list and bool(bindings), "w9_contract: bindings")
    for index, binding in enumerate(bindings):
        item = exact_keys(
            binding,
            {
                "artifact_sha256",
                "field",
                "rounding",
                "source_kind",
                "value_us",
            },
            f"w9_contract.reference_profile.source_bindings.{index}",
        )
        artifact_sha256 = checked_digest(
            item["artifact_sha256"],
            f"binding.{index}.artifact",
        )
        require(
            type(item["field"]) is str
            and item["field"]
            and item["rounding"] in {"CEIL", "CONSERVATIVE_BOUND"}
            and item["source_kind"] in {
                "PROSPECTIVE_BOUND",
                "W8_ARTIFACT",
            }
            and (
                (
                    item["source_kind"] == "W8_ARTIFACT"
                    and artifact_sha256
                    == dependencies["w8_treatment_report_sha256"]
                )
                or (
                    item["source_kind"] == "PROSPECTIVE_BOUND"
                    and artifact_sha256 == ZERO_SHA256
                )
            )
            and checked_nonnegative(item["value_us"], f"binding.{index}.value")
            >= 0,
            f"w9_contract: invalid binding {index}",
        )
    expected_bindings = (
        (
            "W8_ARTIFACT",
            "metrics.cuda_replay.elapsed_us",
            "CEIL",
            contract.cuda_replay_us,
        ),
        (
            "W8_ARTIFACT",
            (
                "metrics.phone_service.batch_timeline[2]."
                "(ended_ns - started_ns) / 1000"
            ),
            "CEIL",
            contract.phone_batch_estimate_us,
        ),
        (
            "W8_ARTIFACT",
            "metrics.cuda_delta.elapsed_us / 2",
            "CONSERVATIVE_BOUND",
            contract.delta_ingest_us[1],
        ),
        (
            "W8_ARTIFACT",
            "metrics.cuda_continuation.elapsed_us / 8",
            "CONSERVATIVE_BOUND",
            contract.cuda_tokens_us[1],
        ),
        (
            "PROSPECTIVE_BOUND",
            "predicted_commit_us",
            "CONSERVATIVE_BOUND",
            contract.predicted_commit_us,
        ),
    )
    require(
        tuple(
            (
                item["source_kind"],
                item["field"],
                item["rounding"],
                item["value_us"],
            )
            for item in bindings
        )
        == expected_bindings,
        "w9_contract: predictor binding set",
    )
    object.__setattr__(
        contract,
        "reference_source_bindings",
        tuple(dict(item) for item in bindings),
    )
    decision = select_cutover(
        contract,
        phone_tokens_at_f0=reference["phone_tokens_at_f0"],
        inflight_present=True,
        inflight_elapsed_us=reference["inflight_elapsed_us"],
    )
    require(
        decision.k_extra == reference["expected_k_extra"],
        "w9_contract: reference decision mismatch",
    )
    return contract


def select_cutover(
    contract: CutoverContract,
    *,
    phone_tokens_at_f0: int,
    inflight_present: bool,
    inflight_elapsed_us: int,
) -> CutoverDecision:
    require(
        is_int(phone_tokens_at_f0)
        and 0 <= phone_tokens_at_f0 < contract.output_tokens,
        "cutover: invalid F0 token count",
    )
    require(type(inflight_present) is bool, "cutover: invalid in-flight flag")
    require(
        is_int(inflight_elapsed_us) and inflight_elapsed_us >= 0,
        "cutover: invalid in-flight elapsed time",
    )
    require(
        inflight_present or inflight_elapsed_us == 0,
        "cutover: absent batch has elapsed time",
    )
    remaining = (
        max(0, contract.phone_batch_estimate_us - inflight_elapsed_us)
        if inflight_present
        else 0
    )
    k_max = max(
        0,
        contract.output_tokens
        - phone_tokens_at_f0
        - contract.max_inflight_tokens
        - contract.min_cuda_continuation_tokens,
    )
    require(
        k_max < len(contract.candidate_k),
        "cutover: K_max exceeds frozen candidate domain",
    )
    candidates = []
    selected = 0
    baseline_completion = None
    for k in contract.candidate_k[: k_max + 1]:
        predicted_delta = contract.max_inflight_tokens + k
        cuda_remaining_tokens = (
            contract.output_tokens - phone_tokens_at_f0 - predicted_delta
        )
        require(
            cuda_remaining_tokens >= contract.min_cuda_continuation_tokens,
            "cutover: CUDA continuation budget exhausted",
        )
        phone_side = remaining + contract.phone_extra_us[k]
        cutover = (
            max(contract.cuda_replay_us, phone_side)
            + contract.delta_ingest_us[predicted_delta]
            + contract.predicted_commit_us
        )
        completion = cutover + contract.cuda_tokens_us[cuda_remaining_tokens]
        if baseline_completion is None:
            baseline_completion = completion
        feasible = (
            k == 0
            or (
                phone_side
                <= contract.cuda_replay_us - contract.cutover_margin_us
                and completion <= baseline_completion
            )
        )
        if feasible:
            selected = k
        candidates.append({
            "completion_us": completion,
            "cuda_remaining_tokens": cuda_remaining_tokens,
            "cutover_us": cutover,
            "feasible": feasible,
            "k": k,
            "phone_side_us": phone_side,
            "predicted_delta_tokens": predicted_delta,
        })
    return CutoverDecision(
        selected,
        k_max,
        phone_tokens_at_f0,
        inflight_present,
        inflight_elapsed_us,
        remaining,
        tuple(candidates),
    )


@dataclass(frozen=True)
class LedgerRecord:
    name: str
    sha256: str
    value: dict[str, object]


class PublicationLedger:
    def __init__(
        self,
        path: Path,
        *,
        transaction_id: str,
        run_id: str,
        request_ids: Sequence[int],
    ) -> None:
        checked_digest(transaction_id, "ledger.transaction_id")
        checked_digest(run_id, "ledger.run_id")
        require(
            bool(request_ids)
            and len(request_ids) == len(set(request_ids))
            and all(is_int(item) and item >= 0 for item in request_ids),
            "ledger: invalid request IDs",
        )
        path.mkdir(mode=0o700, parents=False, exist_ok=False)
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        self.path = path
        self.transaction_id = transaction_id
        self.run_id = run_id
        self.request_ids = tuple(request_ids)
        self.records: list[LedgerRecord] = []

    def append(
        self,
        event: str,
        payload: dict[str, object],
        *,
        event_ns: int | None = None,
    ) -> tuple[LedgerRecord, int]:
        require(type(event) is str and event, "ledger: invalid event")
        require(type(payload) is dict, "ledger: invalid payload")
        if event_ns is None:
            event_ns = time.monotonic_ns()
        require(is_int(event_ns) and event_ns > 0, "ledger: invalid timestamp")
        index = len(self.records)
        previous = self.records[-1].sha256 if self.records else ZERO_SHA256
        value = {
            "event": event,
            "event_ns": event_ns,
            "payload": payload,
            "previous_record_sha256": previous,
            "record_index": index,
            "request_ids": list(self.request_ids),
            "run_id": self.run_id,
            "schema": LEDGER_SCHEMA,
            "transaction_id": self.transaction_id,
        }
        raw = w6.canonical(value)
        name = f"{index:06d}.json"
        target = self.path / name
        with target.open("xb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        directory_fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        durable_ns = time.monotonic_ns()
        record = LedgerRecord(name, w6.sha256(raw), value)
        self.records.append(record)
        return record, durable_ns


def load_ledger(
    path: Path,
    *,
    transaction_id: str,
    run_id: str,
    request_ids: Sequence[int],
) -> list[LedgerRecord]:
    checked_digest(transaction_id, "ledger.transaction_id")
    checked_digest(run_id, "ledger.run_id")
    require(path.is_dir() and not path.is_symlink(), "ledger: invalid directory")
    children = sorted(path.iterdir(), key=lambda item: item.name)
    require(bool(children), "ledger: empty")
    require(
        [item.name for item in children]
        == [f"{index:06d}.json" for index in range(len(children))],
        "ledger: file sequence",
    )
    previous = ZERO_SHA256
    result = []
    keys = {
        "event",
        "event_ns",
        "payload",
        "previous_record_sha256",
        "record_index",
        "request_ids",
        "run_id",
        "schema",
        "transaction_id",
    }
    last_ns = 0
    for index, child in enumerate(children):
        require(child.is_file() and not child.is_symlink(), "ledger: unsafe file")
        value, raw = w6.read_canonical(child, f"ledger.{child.name}")
        exact_keys(value, keys, f"ledger.{child.name}")
        require(
            value["schema"] == LEDGER_SCHEMA
            and value["record_index"] == index
            and value["previous_record_sha256"] == previous
            and value["transaction_id"] == transaction_id
            and value["run_id"] == run_id
            and value["request_ids"] == list(request_ids)
            and type(value["event"]) is str
            and bool(value["event"])
            and type(value["payload"]) is dict
            and is_int(value["event_ns"])
            and value["event_ns"] >= last_ns,
            f"ledger.{child.name}: invalid record",
        )
        digest = w6.sha256(raw)
        result.append(LedgerRecord(child.name, digest, value))
        previous = digest
        last_ns = value["event_ns"]
    return result


def ledger_summary(records: Sequence[LedgerRecord]) -> dict[str, object]:
    require(bool(records), "ledger: no records")
    return {
        "event_count": len(records),
        "final_record_sha256": records[-1].sha256,
        "first_event": records[0].value["event"],
        "last_event": records[-1].value["event"],
    }


def transaction_id(
    contract: CutoverContract,
    run_id: str,
    request_ids: Sequence[int],
) -> str:
    checked_digest(run_id, "transaction.run_id")
    return w6.sha256(w6.canonical({
        "contract_sha256": contract.raw_sha256,
        "request_ids": list(request_ids),
        "run_id": run_id,
    }))


def write_atomic(path: Path, value: dict[str, object]) -> None:
    raw = w6.canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def read_json(path: Path, field: str) -> tuple[dict[str, object], bytes]:
    return w6.read_canonical(path, field)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--phone-tokens-at-f0", type=int, required=True)
    parser.add_argument("--inflight-elapsed-us", type=int, default=0)
    parser.add_argument("--inflight-present", action="store_true")
    args = parser.parse_args()
    try:
        decision = select_cutover(
            load_contract(args.contract),
            phone_tokens_at_f0=args.phone_tokens_at_f0,
            inflight_present=args.inflight_present,
            inflight_elapsed_us=args.inflight_elapsed_us,
        )
        print(json.dumps(decision.as_dict(), sort_keys=True, separators=(",", ":")))
    except (KeyError, OSError, TypeError, ValueError, W9Error, w6.DeltaError) as exc:
        print(json.dumps({"error": str(exc), "status": "W9_POLICY_ERROR"}))
        raise SystemExit(2)
