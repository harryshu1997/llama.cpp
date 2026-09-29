"""Construction and validation of physically executable split contracts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

from .._internal.runtime_controller import RuntimeRequestTicket
from .contracts import PhysicalAdapterError


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


@dataclass(frozen=True)
class StaticSplitContract:
    fraction_ppm: int
    operator_ids: tuple[str, ...]
    protocol: str

    def __post_init__(self) -> None:
        if not 0 < self.fraction_ppm < 1_000_000:
            raise PhysicalAdapterError("physical split fraction is invalid")
        if (
            not self.operator_ids
            or len(self.operator_ids) != len(set(self.operator_ids))
            or any(
                type(value) is not str or not value or not value.isascii()
                for value in self.operator_ids
            )
        ):
            raise PhysicalAdapterError("physical split operators are invalid")
        if (
            type(self.protocol) is not str
            or not self.protocol
            or not self.protocol.isascii()
        ):
            raise PhysicalAdapterError("physical split protocol is invalid")
        object.__setattr__(self, "operator_ids", tuple(sorted(self.operator_ids)))


def compile_static_column_split_contract(
    manifest: Mapping[str, object],
    policy: Mapping[str, object],
    *,
    protocol_namespace: str,
) -> StaticSplitContract:
    """Compile one measured static endpoint contract without choosing a route."""
    geometry = manifest.get("geometry")
    buckets = policy.get("compiled_buckets")
    if (
        type(protocol_namespace) is not str
        or not protocol_namespace
        or not protocol_namespace.isascii()
        or type(geometry) is not dict
        or type(geometry.get("n_ff")) is not int
        or geometry["n_ff"] <= 0
        or type(geometry.get("resident_layer_ids")) is not list
        or not geometry["resident_layer_ids"]
        or type(buckets) is not list
        or not buckets
    ):
        raise PhysicalAdapterError("physical split contract is invalid")
    phone_columns = {
        row.get("phone_columns") for row in buckets if type(row) is dict
    }
    if len(phone_columns) != 1 or not all(
        type(value) is int for value in phone_columns
    ):
        raise PhysicalAdapterError(
            "physical split policy is not one request-scoped fraction"
        )
    columns = next(iter(phone_columns))
    n_ff = geometry["n_ff"]
    if not 0 < columns < n_ff or columns * 1_000_000 % n_ff:
        raise PhysicalAdapterError(
            "physical split fraction cannot be represented exactly"
        )
    layers = tuple(geometry["resident_layer_ids"])
    if (
        any(type(value) is not int or value < 0 for value in layers)
        or len(layers) != len(set(layers))
    ):
        raise PhysicalAdapterError("physical split layers are invalid")
    policy_text = policy.get("policy_text")
    if type(policy_text) is not str or not policy_text.isascii():
        raise PhysicalAdapterError("physical split policy text is invalid")
    protocol = protocol_namespace + ":" + hashlib.sha256(_canonical({
        "compiled_buckets": buckets,
        "policy_text": policy_text,
    })).hexdigest()
    return StaticSplitContract(
        fraction_ppm=columns * 1_000_000 // n_ff,
        operator_ids=tuple(f"layer:{value}:ffn" for value in layers),
        protocol=protocol,
    )


def static_split_contract_values(
    manifest: Mapping[str, object],
    policy: Mapping[str, object],
    *,
    protocol_namespace: str,
) -> tuple[int, tuple[str, ...], str]:
    value = compile_static_column_split_contract(
        manifest, policy, protocol_namespace=protocol_namespace
    )
    return value.fraction_ppm, value.operator_ids, value.protocol


def validate_static_split_capability(
    capability: object,
    *,
    endpoint: str,
    artifact_sha256: str,
    manifest: Mapping[str, object],
    policy: Mapping[str, object],
    protocol_namespace: str,
) -> None:
    contract = compile_static_column_split_contract(
        manifest, policy, protocol_namespace=protocol_namespace
    )
    if not (
        getattr(capability, "endpoint", None) == endpoint
        and getattr(capability, "artifact_sha256", None) == artifact_sha256
        and getattr(capability, "route_family", None) == "operator_split"
        and getattr(capability, "assisted_operator_kind", None) == "ffn"
        and getattr(capability, "split_axis", None) == "column"
        and getattr(capability, "split_fractions_ppm", None)
            == (contract.fraction_ppm,)
        and getattr(capability, "operator_ids", None)
            == contract.operator_ids
        and getattr(capability, "operator_plan_protocol", None)
            == contract.protocol
        and getattr(capability, "maturity", None) == "QUALIFIED"
    ):
        raise PhysicalAdapterError(
            "split capability differs from the physical endpoint"
        )


def validate_ticket_split_contract(
    ticket: RuntimeRequestTicket, capability: object
) -> None:
    """Prove the static endpoint exactly implements the selected ticket."""
    if not isinstance(ticket, RuntimeRequestTicket):
        raise PhysicalAdapterError("physical split ticket is invalid")
    plan = ticket.execution_plan
    if plan is None:
        raise PhysicalAdapterError("physical split execution plan is absent")
    fraction = getattr(capability, "split_fractions_ppm", (None,))[0]
    selected = frozenset(getattr(capability, "operator_ids", ()))
    assignments = {row.operator_id: row for row in plan.operators}
    participants = tuple(sorted(
        getattr(capability, "participant_device_ids", ())
    ))
    coordinator = getattr(capability, "coordinator_device_id", None)
    if not (
        ticket.binding.executor_id == getattr(capability, "executor_id", None)
        and ticket.binding.endpoint == getattr(capability, "endpoint", None)
        and ticket.binding.operator_plan_protocol
            == getattr(capability, "operator_plan_protocol", None)
        and ticket.binding.operator_plan_sha256 == plan.plan_sha256
        and plan.route_family == getattr(capability, "route_family", None)
        and plan.assisted_operator_kind
            == getattr(capability, "assisted_operator_kind", None)
        and plan.split_axis == getattr(capability, "split_axis", None)
        and plan.split_fraction_ppm == fraction
        and selected
        and selected.issubset(assignments)
        and all(
            assignment.device_ids == participants
            and assignment.split_axis == plan.split_axis
            and assignment.split_fraction_ppm == fraction
            for operator_id, assignment in assignments.items()
            if operator_id in selected
        )
        and all(
            assignment.device_ids == (coordinator,)
            and assignment.split_axis == "none"
            and assignment.split_fraction_ppm == 0
            for operator_id, assignment in assignments.items()
            if operator_id not in selected
        )
    ):
        raise PhysicalAdapterError(
            "scheduler split plan differs from the physical split contract"
        )


def build_static_split_prewarm(
    rows: Sequence[Mapping[str, object]],
    policy: Mapping[str, object],
    *,
    event_id: str,
    request_index: int,
) -> tuple[dict[str, object], dict[str, int]]:
    """Build deterministic prewarm input for an already selected endpoint."""
    buckets = policy.get("compiled_buckets")
    if type(buckets) is not list or not buckets:
        raise PhysicalAdapterError("compiled split buckets are invalid")
    phone_buckets = [
        bucket
        for bucket in buckets
        if type(bucket) is dict
        and type(bucket.get("max_tokens")) is int
        and bucket["max_tokens"] > 0
        and type(bucket.get("phone_columns")) is int
        and bucket["phone_columns"] > 0
    ]
    if not phone_buckets or not rows:
        raise PhysicalAdapterError("split prewarm shape is unavailable")
    bucket = max(phone_buckets, key=lambda value: value["max_tokens"])
    tokens = bucket["max_tokens"]
    source = max(rows, key=lambda row: len(row.get("prompt_tokens", ())))
    prompt = source.get("prompt_tokens")
    if (
        type(prompt) is not list
        or not prompt
        or any(type(token) is not int for token in prompt)
    ):
        raise PhysicalAdapterError("split prewarm prompt is invalid")
    repeated = (prompt * ((tokens + len(prompt) - 1) // len(prompt)))[:tokens]
    row = dict(source)
    row.update({
        "event_id": event_id,
        "input_tokens": tokens,
        "output_tokens": 2,
        "overlay_request_index": request_index,
        "prompt_tokens": repeated,
    })
    return row, {
        "input_tokens": tokens,
        "phone_columns": bucket["phone_columns"],
    }
