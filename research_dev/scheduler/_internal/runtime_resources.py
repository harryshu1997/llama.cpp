"""Atomic request-memory reservations for the shared runtime lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
import threading
from types import MappingProxyType
from typing import Mapping, Sequence

from .runtime_capabilities import ModelResidencyObservation
from .runtime_cost import RuntimeMemoryDemand
from .runtime_placement import RuntimePlacementSnapshot
from .runtime_plan import RuntimeTransitionPlan
from .types import canonical_sha256
from .policy import LeaseDemand


RUNTIME_MEMORY_LEDGER_SCHEMA = "research-scheduler-memory-ledger-v1"
_MAX_RESERVATION_TIME_US = 2**63 - 1


def runtime_phase_lease_demands(
    route_id: str,
    resource_slots: Mapping[str, int],
    transitions: Sequence[RuntimeTransitionPlan],
    service_us: int,
    service_upper_us: int,
) -> tuple[LeaseDemand, ...]:
    """Reserve transitions in adapter order, followed by execution."""
    prepare_us = 0
    demands = []
    for row in transitions:
        if row.latency_us == 0:
            continue
        demands.extend(LeaseDemand(
            lease_id=f"{route_id}:prepare:{row.transition_id}:{resource_id}",
            resource_id=resource_id,
            slots=slots,
            start_offset_us=prepare_us,
            duration_us=row.latency_us,
            duration_upper_us=row.latency_us,
        ) for resource_id, slots in sorted(row.resource_slots.items()))
        prepare_us += row.latency_us
    return tuple(demands) + tuple(
        LeaseDemand(
            lease_id=f"{route_id}:{resource_id}",
            resource_id=resource_id,
            slots=slots,
            start_offset_us=prepare_us,
            duration_us=max(1, service_us - prepare_us),
            duration_upper_us=max(1, service_upper_us - prepare_us),
        )
        for resource_id, slots in sorted(resource_slots.items())
    )


class RuntimeResourceError(ValueError):
    pass


REMOTE_RESIDENT_OMISSION_PROOF_SCHEMA = "scheduler-remote-resident-omission-v1"
REMOTE_RESIDENT_ACCOUNTING_SCHEMA = "research-scheduler-remote-resident-accounting-v1"
_REMOTE_RESIDENT_PROOF_PREFIX = "S41SERVERFFN remote_resident "


def _nonnegative(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise RuntimeResourceError(name + " must be a non-negative integer")
    return value


def _nonnegative_map(name: str, values: Mapping[str, object]) -> Mapping[str, int]:
    if not isinstance(values, Mapping):
        raise RuntimeResourceError(name + " must be a mapping")
    result = {}
    for key in sorted(values):
        if type(key) is not str or not key:
            raise RuntimeResourceError(name + " has an invalid key")
        result[key] = _nonnegative(name + "[" + key + "]", values[key])
    return MappingProxyType(result)


@dataclass(frozen=True)
class RuntimeRemoteResidentOmissionProof:
    """The reduced desktop server's own allocation record for its omitted FFN weights.

    Emitted once after the model loads and the warm-up decode ran through the phone owner:
    ``S41SERVERFFN remote_resident mask=<int> omitted_bytes=<int> unmapped_bytes=<int>
    warmup=<validated|skipped>``. The scheduler credits reclaimed memory only against this
    record, never against the plan alone.
    """

    layer_mask: int
    omitted_bytes: int
    unmapped_bytes: int
    warmup: str

    def __post_init__(self) -> None:
        if _nonnegative("omission proof layer mask", self.layer_mask) == 0:
            raise RuntimeResourceError("omission proof layer mask is empty")
        if _nonnegative("omission proof omitted bytes", self.omitted_bytes) == 0:
            raise RuntimeResourceError("omission proof omitted bytes are empty")
        _nonnegative("omission proof unmapped bytes", self.unmapped_bytes)
        if self.unmapped_bytes > self.omitted_bytes:
            raise RuntimeResourceError("omission proof unmapped bytes exceed omitted bytes")
        if self.warmup not in {"validated", "skipped"}:
            raise RuntimeResourceError("omission proof warm-up state is invalid")

    @classmethod
    def parse_line(cls, line: str) -> "RuntimeRemoteResidentOmissionProof | None":
        """Return the proof carried by one server log line, or None for other lines."""
        if type(line) is not str:
            return None
        # Template diagnostics can leave a color reset before the next stderr record.
        line = re.sub(r"^(?:\x1b\[[0-9;]*m)+", "", line)
        if not line.startswith(_REMOTE_RESIDENT_PROOF_PREFIX):
            return None
        fields = {}
        for item in line[len(_REMOTE_RESIDENT_PROOF_PREFIX):].split():
            name, separator, value = item.partition("=")
            if not separator or name in fields:
                raise RuntimeResourceError("omission proof line is malformed")
            fields[name] = value
        try:
            return cls(
                layer_mask=int(fields["mask"]),
                omitted_bytes=int(fields["omitted_bytes"]),
                unmapped_bytes=int(fields["unmapped_bytes"]),
                warmup=fields["warmup"],
            )
        except (KeyError, ValueError) as error:
            raise RuntimeResourceError("omission proof line is malformed") from error

    def to_json(self) -> dict[str, object]:
        return {
            "layer_mask": self.layer_mask,
            "omitted_bytes": self.omitted_bytes,
            "schema": REMOTE_RESIDENT_OMISSION_PROOF_SCHEMA,
            "unmapped_bytes": self.unmapped_bytes,
            "warmup": self.warmup,
        }

    @classmethod
    def from_json(cls, value: object) -> "RuntimeRemoteResidentOmissionProof":
        if not isinstance(value, Mapping) or value.get("schema") != (
            REMOTE_RESIDENT_OMISSION_PROOF_SCHEMA
        ):
            raise RuntimeResourceError("omission proof record is invalid")
        return cls(
            layer_mask=value.get("layer_mask"),
            omitted_bytes=value.get("omitted_bytes"),
            unmapped_bytes=value.get("unmapped_bytes"),
            warmup=value.get("warmup"),
        )


HOST_SHARE_RELEASE_PROOF_SCHEMA = "scheduler-host-share-release-proof-v1"
_HOST_SHARE_RELEASE_PROOF_PREFIX = "S41SERVERFFN dormant_host_share "


@dataclass(frozen=True)
class RuntimeHostShareReleaseProof:
    """The desktop server's own record of a decode-only relocation phase change.

    With ``S41_SERVER_FFN_DORMANT_HOST_SHARE=1`` the server keeps every FFN weight mapped but
    releases the pages of the phone-executed column suffix ``[host_columns, n_ff)`` of the split
    layers while every processing slot decodes (``phase=decode ... released_bytes=<int>
    ranges=<int>``) and populates them again before any prompt processing (``phase=local ...
    restored_bytes=<int>``). A launch may defer population until local execution faults pages in.
    Host memory is credited only against a ``decode`` record and the
    credit ends at the next ``local`` record; the plan alone never earns it.
    """

    phase: str
    layer_mask: int
    host_columns: int
    released_bytes: int
    ranges: int
    elapsed_us: int

    def __post_init__(self) -> None:
        if self.phase not in {"decode", "local"}:
            raise RuntimeResourceError("host share release proof phase is invalid")
        if _nonnegative("host share release proof layer mask", self.layer_mask) == 0:
            raise RuntimeResourceError("host share release proof layer mask is empty")
        _nonnegative("host share release proof host columns", self.host_columns)
        _nonnegative("host share release proof released bytes", self.released_bytes)
        _nonnegative("host share release proof ranges", self.ranges)
        _nonnegative("host share release proof elapsed", self.elapsed_us)

    @classmethod
    def parse_line(cls, line: str) -> "RuntimeHostShareReleaseProof | None":
        """Return the proof carried by one server log line, or None for other lines."""
        if type(line) is not str:
            return None
        start = line.find(_HOST_SHARE_RELEASE_PROOF_PREFIX)
        if start < 0:
            return None
        fields = {}
        for item in line[start + len(_HOST_SHARE_RELEASE_PROOF_PREFIX):].split():
            name, separator, value = item.partition("=")
            if not separator or name in fields:
                raise RuntimeResourceError("host share release proof line is malformed")
            fields[name] = value
        try:
            phase = fields["phase"]
            return cls(
                phase=phase,
                layer_mask=int(fields["layer_mask"]),
                host_columns=int(fields["host_columns"]),
                released_bytes=int(fields["released_bytes" if phase == "decode" else "restored_bytes"]),
                ranges=int(fields.get("ranges", 0)),
                elapsed_us=int(fields["elapsed_us"]),
            )
        except (KeyError, ValueError) as error:
            raise RuntimeResourceError("host share release proof line is malformed") from error

    def to_json(self) -> dict[str, object]:
        return {
            "elapsed_us": self.elapsed_us,
            "host_columns": self.host_columns,
            "layer_mask": self.layer_mask,
            "phase": self.phase,
            "ranges": self.ranges,
            "released_bytes": self.released_bytes,
            "schema": HOST_SHARE_RELEASE_PROOF_SCHEMA,
        }

    @classmethod
    def from_json(cls, value: object) -> "RuntimeHostShareReleaseProof":
        if not isinstance(value, Mapping) or value.get("schema") != HOST_SHARE_RELEASE_PROOF_SCHEMA:
            raise RuntimeResourceError("host share release proof record is invalid")
        return cls(
            phase=value.get("phase"),
            layer_mask=value.get("layer_mask"),
            host_columns=value.get("host_columns"),
            released_bytes=value.get("released_bytes"),
            ranges=value.get("ranges"),
            elapsed_us=value.get("elapsed_us"),
        )


def host_share_release_lower_bound_bytes(
    manifest,
    layer_mask: int,
    host_columns: int,
    page_size: int = 4096,
) -> int:
    """Planning lower bound of the host bytes a decode-only relocation releases.

    Mirrors ``llama_model::ffn_host_share_release``: the phone suffix of gate/up is one
    contiguous row range per tensor, the suffix of ``down`` is one range per output row, and
    every range is aligned inward to whole pages. File offsets are not in the manifest, so each
    range is charged the worst alignment (one page fewer than its length allows). The proof
    record carries the exact figure; this bound is for admission before the proof exists.
    """
    layer_mask = _nonnegative("host share layer mask", layer_mask)
    host_columns = _nonnegative("host share host columns", host_columns)
    if page_size <= 0:
        raise RuntimeResourceError("host share page size is invalid")
    total = 0
    tensor_by_id = manifest.tensor_by_id
    for layer in range(manifest.block_count):
        if not (layer_mask >> layer) & 1:
            continue
        for suffix in ("ffn_gate", "ffn_up", "ffn_down"):
            tensor = tensor_by_id.get(f"blk.{layer}.{suffix}.weight")
            if tensor is None:
                raise RuntimeResourceError(f"host share tensor is absent from the manifest: blk.{layer}.{suffix}")
            if tensor.quantization_block_size != 1:
                raise RuntimeResourceError("host share release supports dense element types only")
            element = tensor.quantization_type_size
            if suffix == "ffn_down":
                n_ff, rows = tensor.shape[0], tensor.shape[1]
                if host_columns > n_ff:
                    raise RuntimeResourceError("host share host columns exceed the FFN width")
                length = (n_ff - host_columns) * element
                pages = max(0, length // page_size - 1)
                total += pages * page_size * rows
            else:
                n_embd, n_ff = tensor.shape[0], tensor.shape[1]
                if host_columns > n_ff:
                    raise RuntimeResourceError("host share host columns exceed the FFN width")
                length = (n_ff - host_columns) * n_embd * element
                pages = max(0, length // page_size - 1)
                total += pages * page_size
    return total


@dataclass(frozen=True)
class RuntimeRemoteResidentAccounting:
    """Memory truth for one reduced desktop parent and its recovery contract.

    ``reclaimed_bytes`` is credited only when the omission proof matches the plan. The
    reclaimed memory is split, never shared, between ``fallback_reserve_bytes`` (kept free so
    the full-weight route stays feasible) and ``kv_capacity_gain_bytes`` (usable for KV).
    """

    artifact_sha256: str
    desktop_pool_id: str
    layer_mask: int
    desktop_weights_full_bytes: int
    desktop_weights_allocated_bytes: int
    omitted_bytes_planned: int
    omitted_bytes_verified: int
    verified: bool
    phone_weights_bytes_by_session: Mapping[str, int]
    phone_workspace_bytes: int
    kv_bytes_by_pool: Mapping[str, int]
    transition_peak_bytes_by_pool: Mapping[str, int]
    recovery_required_bytes_by_pool: Mapping[str, int]
    recovery_capacity_bytes_by_pool: Mapping[str, int]
    recovery_feasible: bool
    fallback_mode: str
    fallback_reserve_bytes: int
    reclaimed_bytes: int
    kv_capacity_gain_bytes: int
    proof: RuntimeRemoteResidentOmissionProof | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "desktop_pool_id": self.desktop_pool_id,
            "desktop_weights_allocated_bytes": self.desktop_weights_allocated_bytes,
            "desktop_weights_full_bytes": self.desktop_weights_full_bytes,
            "fallback_mode": self.fallback_mode,
            "fallback_reserve_bytes": self.fallback_reserve_bytes,
            "kv_bytes_by_pool": dict(self.kv_bytes_by_pool),
            "kv_capacity_gain_bytes": self.kv_capacity_gain_bytes,
            "layer_mask": self.layer_mask,
            "omitted_bytes_planned": self.omitted_bytes_planned,
            "omitted_bytes_verified": self.omitted_bytes_verified,
            "phone_weights_bytes_by_session": dict(self.phone_weights_bytes_by_session),
            "phone_workspace_bytes": self.phone_workspace_bytes,
            "proof": None if self.proof is None else self.proof.to_json(),
            "reclaimed_bytes": self.reclaimed_bytes,
            "recovery_capacity_bytes_by_pool": dict(self.recovery_capacity_bytes_by_pool),
            "recovery_feasible": self.recovery_feasible,
            "recovery_required_bytes_by_pool": dict(self.recovery_required_bytes_by_pool),
            "schema": REMOTE_RESIDENT_ACCOUNTING_SCHEMA,
            "transition_peak_bytes_by_pool": dict(self.transition_peak_bytes_by_pool),
            "verified": self.verified,
        }


def remote_resident_accounting(
    *,
    artifact_sha256: str,
    desktop_pool_id: str,
    layer_mask: int,
    desktop_weights_full_bytes: int,
    omitted_bytes_planned: int,
    proof: RuntimeRemoteResidentOmissionProof | None,
    phone_weights_bytes_by_session: Mapping[str, int],
    phone_workspace_bytes: int,
    kv_bytes_by_pool: Mapping[str, int],
    transition_peak_bytes_by_pool: Mapping[str, int],
    live_available_bytes_by_pool: Mapping[str, int],
    reduced_allocation_bytes_by_pool: Mapping[str, int],
    recovery_required_bytes_by_pool: Mapping[str, int],
    fallback_mode: str = "teardown",
) -> RuntimeRemoteResidentAccounting:
    """Account a reduced desktop parent: credit after proof, recovery before admission.

    ``fallback_mode`` is ``"teardown"`` when the full-weight route replaces the reduced parent
    (its allocation is released first) or ``"alongside"`` when it must fit next to it.
    """
    artifact = _text("remote-resident artifact", artifact_sha256)
    pool = _text("remote-resident desktop pool", desktop_pool_id)
    if _nonnegative("remote-resident layer mask", layer_mask) == 0:
        raise RuntimeResourceError("remote-resident layer mask is empty")
    full = _nonnegative("remote-resident full desktop weights", desktop_weights_full_bytes)
    planned = _nonnegative("remote-resident planned omission", omitted_bytes_planned)
    if planned == 0 or planned > full:
        raise RuntimeResourceError("remote-resident planned omission exceeds the desktop weights")
    if fallback_mode not in {"teardown", "alongside"}:
        raise RuntimeResourceError("remote-resident fallback mode is invalid")
    if proof is not None and not isinstance(proof, RuntimeRemoteResidentOmissionProof):
        raise RuntimeResourceError("remote-resident proof is invalid")
    sessions = _nonnegative_map("phone weights by session", phone_weights_bytes_by_session)
    if not sessions:
        raise RuntimeResourceError("remote-resident accounting lacks phone sessions")
    workspace = _nonnegative("phone workspace bytes", phone_workspace_bytes)
    kv = _nonnegative_map("kv bytes by pool", kv_bytes_by_pool)
    peaks = _nonnegative_map("transition peak bytes by pool", transition_peak_bytes_by_pool)
    live = _nonnegative_map("live available bytes by pool", live_available_bytes_by_pool)
    reduced = _nonnegative_map("reduced allocation bytes by pool", reduced_allocation_bytes_by_pool)
    required = _nonnegative_map("recovery required bytes by pool", recovery_required_bytes_by_pool)
    if pool not in live or pool not in required:
        raise RuntimeResourceError("remote-resident desktop pool lacks live or recovery figures")

    verified = (
        proof is not None
        and proof.layer_mask == layer_mask
        and proof.omitted_bytes == planned
        and proof.warmup == "validated"
    )
    omitted_verified = planned if verified else 0
    reclaimed = omitted_verified
    capacity = {}
    for name in sorted(set(required) | set(live)):
        released = 0 if fallback_mode == "alongside" else reduced.get(name, 0)
        capacity[name] = live.get(name, 0) + released
    recovery_feasible = all(capacity.get(name, 0) >= required[name] for name in required)
    if fallback_mode == "teardown":
        # the reduced parent releases everything it holds, so memory spent on KV returns too
        reserve = 0
    else:
        shortfall = max(0, required[pool] - (live[pool] - reclaimed))
        reserve = min(reclaimed, shortfall)
    return RuntimeRemoteResidentAccounting(
        artifact_sha256=artifact,
        desktop_pool_id=pool,
        layer_mask=layer_mask,
        desktop_weights_full_bytes=full,
        desktop_weights_allocated_bytes=full - omitted_verified,
        omitted_bytes_planned=planned,
        omitted_bytes_verified=omitted_verified,
        verified=verified,
        phone_weights_bytes_by_session=sessions,
        phone_workspace_bytes=workspace,
        kv_bytes_by_pool=kv,
        transition_peak_bytes_by_pool=peaks,
        recovery_required_bytes_by_pool=required,
        recovery_capacity_bytes_by_pool=MappingProxyType(capacity),
        recovery_feasible=recovery_feasible,
        fallback_mode=fallback_mode,
        fallback_reserve_bytes=reserve,
        reclaimed_bytes=reclaimed,
        kv_capacity_gain_bytes=reclaimed - reserve,
        proof=proof,
    )


def runtime_preparation_windows(ticket, *, active_until_us=None):
    """Retime only the active transition and its dependent phases."""
    receipts = {row.transition_id: row for row in ticket.transition_receipts}
    windows = {}
    cursor = ticket.decision.start_us
    active = True
    for transition in ticket.execution_plan.transitions:
        leases = tuple(
            row for row in ticket.decision.leases
            if row.lease_id == (
                f"{ticket.decision.route_id}:prepare:"
                f"{transition.transition_id}:{row.resource_id}"
            )
        )
        receipt = receipts.get(transition.transition_id)
        if receipt is not None:
            end = receipt.finished_us
            if end < cursor:
                raise RuntimeResourceError("preparation receipt precedes its phase")
        else:
            end = max(
                cursor + transition.latency_us,
                *(ticket.final_reserved_until_us[row.token] for row in leases),
            ) if leases else cursor + transition.latency_us
            if active and active_until_us is not None:
                end = max(end, active_until_us)
            active = False
        windows.update({row.token: (cursor, end) for row in leases})
        cursor = end
    for lease in ticket.decision.leases:
        if lease.token not in ticket.prepare_lease_tokens:
            duration = lease.reserved_until_us - lease.start_us
            windows[lease.token] = (
                cursor, max(ticket.final_reserved_until_us[lease.token], cursor + duration),
            )
    return windows


def _transition_reclaims(
    transitions: Sequence[RuntimeTransitionPlan],
    residency: Sequence[ModelResidencyObservation],
    exclusive_resource_by_device: Mapping[str | tuple[str, str], str],
) -> tuple[
    Mapping[tuple[str, str], int],
    Mapping[tuple[str, str], int],
]:
    transition_rows = tuple(transitions)
    residency_rows = tuple(residency)
    if any(not isinstance(row, RuntimeTransitionPlan) for row in transition_rows):
        raise RuntimeResourceError("memory transition is invalid")
    if any(
        not isinstance(row, ModelResidencyObservation)
        for row in residency_rows
    ):
        raise RuntimeResourceError("memory residency observation is invalid")
    exclusive = {
        (
            tuple(_text("exclusive residency owner", value) for value in device_id)
            if isinstance(device_id, tuple) and len(device_id) == 2
            else _text("exclusive residency device", device_id)
        ): _text(
            "exclusive residency resource", resource_id
        )
        for device_id, resource_id in exclusive_resource_by_device.items()
    }

    def anchor_group(eviction):
        return exclusive.get(
            (eviction.device_id, eviction.executor_id),
            exclusive.get(eviction.device_id),
        )
    observed = {
        (
            row.model_id,
            row.artifact_sha256,
            row.device_id,
            row.resident_bytes,
            row.generation,
            row.executor_id,
            (
                row.resident_bytes
                if row.reclaimable_bytes is None
                else row.reclaimable_bytes
            ),
        )
        for row in residency_rows
        if row.state in {"hot", "warm"}
    }
    resident_reclaimed: dict[tuple[str, str], int] = {}
    total_reclaimed: dict[tuple[str, str], int] = {}
    eviction_keys = set()
    for transition in transition_rows:
        transition_evictions = []
        for eviction in transition.evictions:
            if eviction.session_id is not None:
                eviction_key = (
                    eviction.device_id,
                    eviction.session_id,
                    eviction.artifact_sha256,
                    eviction.resident_geometry_sha256,
                    eviction.generation,
                )
                if eviction_key in eviction_keys:
                    raise RuntimeResourceError(
                        "transition eviction is duplicated: "
                        + eviction.device_id
                    )
                eviction_keys.add(eviction_key)
                continue
            reclaimable_bytes = (
                eviction.resident_bytes
                if eviction.reclaimable_bytes is None
                else eviction.reclaimable_bytes
            )
            eviction_key = (
                eviction.model_id,
                eviction.artifact_sha256,
                eviction.device_id,
                eviction.resident_bytes,
                eviction.generation,
                eviction.executor_id,
                reclaimable_bytes,
            )
            if eviction_key not in observed:
                raise RuntimeResourceError(
                    "transition eviction is stale: " + eviction.device_id
                )
            if eviction_key in eviction_keys:
                raise RuntimeResourceError(
                    "transition eviction is duplicated: "
                    + eviction.device_id
                )
            eviction_keys.add(eviction_key)
            replacement_group = (
                anchor_group(eviction)
                if eviction.replacement_group is None
                else eviction.replacement_group
            )
            if replacement_group is None:
                raise RuntimeResourceError(
                    "transition eviction resource is not exclusive: "
                    + eviction.device_id
                )
            if replacement_group not in transition.resource_ids:
                raise RuntimeResourceError(
                    "transition eviction resource is not leased: "
                    + replacement_group
                )
            transition_evictions.append((
                eviction,
                replacement_group,
                reclaimable_bytes,
            ))

        anchors = {
            (
                eviction.model_id,
                eviction.artifact_sha256,
                eviction.generation,
                eviction.executor_id,
                replacement_group,
            )
            for eviction, replacement_group, _ in transition_evictions
            if anchor_group(eviction) == replacement_group
        }
        for eviction, replacement_group, reclaimable_bytes in (
            transition_evictions
        ):
            if anchor_group(eviction) != replacement_group:
                anchor = (
                    eviction.model_id,
                    eviction.artifact_sha256,
                    eviction.generation,
                    eviction.executor_id,
                    replacement_group,
                )
                if eviction.executor_id is None or anchor not in anchors:
                    raise RuntimeResourceError(
                        "associated eviction lacks an exact exclusive anchor: "
                        + eviction.device_id
                    )
            key = (eviction.device_id, replacement_group)
            resident_reclaimed[key] = (
                resident_reclaimed.get(key, 0) + eviction.resident_bytes
            )
            total_reclaimed[key] = (
                total_reclaimed.get(key, 0) + reclaimable_bytes
            )
    return (
        MappingProxyType(dict(sorted(resident_reclaimed.items()))),
        MappingProxyType(dict(sorted(total_reclaimed.items()))),
    )


def transition_adjusted_memory_demands(
    demands: Sequence[RuntimeMemoryDemand],
    *,
    transitions: Sequence[RuntimeTransitionPlan],
    residency: Sequence[ModelResidencyObservation],
    exclusive_resource_by_device: Mapping[str | tuple[str, str], str],
) -> tuple[RuntimeMemoryDemand, ...]:
    """Bind replacement credit to exact transition eviction rows."""

    rows = tuple(demands)
    if any(not isinstance(row, RuntimeMemoryDemand) for row in rows):
        raise RuntimeResourceError("memory demand is invalid")
    transition_rows = tuple(transitions)
    if not transition_rows:
        if not any(row.replaceable_bytes for row in rows):
            return rows
        return tuple(replace(row, replaceable_bytes=0) for row in rows)
    reclaimed, _ = _transition_reclaims(
        transition_rows, residency, exclusive_resource_by_device
    )

    adjusted = []
    remaining_reclaimed = dict(reclaimed)
    for demand in rows:
        base = replace(demand, replaceable_bytes=0)
        if demand.replacement_group is None:
            adjusted.append(base)
            continue
        if demand.device_id is None:
            if demand.replaceable_bytes:
                raise RuntimeResourceError(
                    "replacement memory device is absent"
                )
            adjusted.append(base)
            continue
        key = (demand.device_id, demand.replacement_group)
        available = remaining_reclaimed.get(key, 0)
        replacement = min(
            base.required_bytes - base.resident_bytes,
            available,
        )
        adjusted.append(replace(
            base,
            replaceable_bytes=replacement,
        ))
        remaining_reclaimed[key] = available - replacement
    return tuple(adjusted)


def preview_runtime_memory(
    demands: Sequence[RuntimeMemoryDemand],
    snapshot: RuntimePlacementSnapshot,
    *,
    transitions: Sequence[RuntimeTransitionPlan] = (),
    residency: Sequence[ModelResidencyObservation] = (),
    exclusive_resource_by_device: Mapping[str | tuple[str, str], str] = MappingProxyType({}),
    reserved_by_resource: Mapping[str, int] = MappingProxyType({}),
    shared_by_resource_key: Mapping[tuple[str, str], int] = MappingProxyType({}),
    replacement_share_keys_by_resource_group: Mapping[
        tuple[str, str], tuple[str, ...]
    ] = MappingProxyType({}),
    enforce_live_capacity: bool = True,
) -> Mapping[str, int]:
    """Preview current capacity using the same transition proof as commit."""

    if not isinstance(snapshot, RuntimePlacementSnapshot):
        raise RuntimeResourceError("memory capacity snapshot is invalid")
    if type(enforce_live_capacity) is not bool:
        raise RuntimeResourceError("memory capacity mode is invalid")
    rows = tuple(demands)
    expected = transition_adjusted_memory_demands(
        rows,
        transitions=transitions,
        residency=residency,
        exclusive_resource_by_device=exclusive_resource_by_device,
    )
    if rows != expected:
        raise RuntimeResourceError(
            "memory replacement credit differs from transition plan"
        )
    if transitions:
        _, total_reclaimed = _transition_reclaims(
            transitions, residency, exclusive_resource_by_device
        )
    else:
        total_reclaimed = {}
    extra_reclaimed_by_resource: dict[str, int] = {}
    credited_by_key: dict[tuple[str, str], int] = {}
    resource_by_key: dict[tuple[str, str], str] = {}
    for demand in rows:
        if (
            demand.device_id is None
            or demand.replacement_group is None
        ):
            continue
        key = (demand.device_id, demand.replacement_group)
        previous_resource = resource_by_key.setdefault(
            key, demand.resource_id
        )
        if previous_resource != demand.resource_id:
            raise RuntimeResourceError(
                "replacement device uses multiple memory resources: "
                + demand.device_id
            )
        credited_by_key[key] = (
            credited_by_key.get(key, 0) + demand.replaceable_bytes
        )
    for key, resource_id in resource_by_key.items():
        extra = total_reclaimed.get(key, 0) - credited_by_key.get(key, 0)
        if extra > 0:
            extra_reclaimed_by_resource[resource_id] = (
                extra_reclaimed_by_resource.get(resource_id, 0)
                + extra
            )

    total_required: dict[str, int] = {}
    required: dict[str, int] = {}
    required_shared: dict[tuple[str, str], int] = {}
    reflected_shared: dict[tuple[str, str], int] = {}
    for demand in rows:
        total_required[demand.resource_id] = (
            total_required.get(demand.resource_id, 0)
            + demand.required_bytes
        )
        if demand.share_key is not None and demand.resident_bytes:
            key = (demand.resource_id, demand.share_key)
            reflected_shared[key] = max(
                reflected_shared.get(key, 0),
                min(
                    demand.resident_bytes,
                    shared_by_resource_key.get(key, 0),
                ),
            )
        if not demand.additional_bytes:
            continue
        if demand.share_key is not None:
            key = (demand.resource_id, demand.share_key)
            required_shared[key] = max(
                required_shared.get(key, 0), demand.additional_bytes
            )
        else:
            required[demand.resource_id] = (
                required.get(demand.resource_id, 0)
                + demand.additional_bytes
            )

    for demand in rows:
        if demand.replacement_group is None or demand.share_key is None:
            continue
        key = (demand.resource_id, demand.replacement_group)
        conflicting = set(
            replacement_share_keys_by_resource_group.get(key, ())
        ) - {demand.share_key}
        if conflicting:
            raise RuntimeResourceError(
                "exclusive memory replacement overlaps: "
                + demand.resource_id
            )

    for (resource_id, share_key), amount in required_shared.items():
        additional = max(
            0,
            amount - shared_by_resource_key.get(
                (resource_id, share_key), 0
            ),
        )
        required[resource_id] = required.get(resource_id, 0) + additional

    reflected_by_resource: dict[str, int] = {}
    for (resource_id, _), amount in reflected_shared.items():
        reflected_by_resource[resource_id] = (
            reflected_by_resource.get(resource_id, 0) + amount
        )

    resource_ids = set(total_required) | set(required)
    for resource_id in resource_ids:
        capacity = snapshot.capacities.get(resource_id)
        if capacity is None:
            raise RuntimeResourceError(
                f"memory resource is absent: {resource_id}"
            )
        if total_required.get(resource_id, 0) > (
            capacity.capacity_bytes - capacity.reserve_bytes
        ) or (enforce_live_capacity and (
            max(
                0,
                reserved_by_resource.get(resource_id, 0)
                - reflected_by_resource.get(resource_id, 0),
            )
            + required.get(resource_id, 0)
            > capacity.available_bytes
            + extra_reclaimed_by_resource.get(resource_id, 0)
        )):
            raise RuntimeResourceError(
                f"memory capacity is insufficient: {resource_id}"
            )
    return MappingProxyType(dict(sorted(required.items())))


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise RuntimeResourceError(f"{name} must be non-empty ASCII text")
    return value


@dataclass(frozen=True)
class RuntimeResidencyProjectionToken:
    """Bind projected memory to one exact scheduler-owned transition."""

    predecessor_ticket_id: str
    predecessor_request_id: str
    transition_id: str
    target_geometry_sha256: str
    target_shards_sha256: str
    layout_generation: int
    ready_at_us: int
    resource_ids: tuple[str, ...]
    memory_delta_bytes_by_resource: Mapping[str, int]
    token_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "predecessor_ticket_id",
            "predecessor_request_id",
            "transition_id",
        ):
            _text("residency projection " + name, getattr(self, name))
        for name in (
            "target_geometry_sha256",
            "target_shards_sha256",
        ):
            value = _text(
                "residency projection " + name, getattr(self, name)
            )
            if not value.startswith("sha256:") or len(value) != 71:
                raise RuntimeResourceError(
                    "residency projection hash is invalid"
                )
        if (
            type(self.layout_generation) is not int
            or self.layout_generation < 1
            or type(self.ready_at_us) is not int
            or self.ready_at_us < 0
        ):
            raise RuntimeResourceError(
                "residency projection generation is invalid"
            )
        resources = tuple(sorted(
            _text("residency projection resource", value)
            for value in self.resource_ids
        ))
        if not resources or len(resources) != len(set(resources)):
            raise RuntimeResourceError(
                "residency projection resources are invalid"
            )
        memory_delta = dict(self.memory_delta_bytes_by_resource)
        if (
            not memory_delta
            or any(
                _text("residency projection memory resource", resource_id)
                != resource_id
                or type(value) is not int
                for resource_id, value in memory_delta.items()
            )
        ):
            raise RuntimeResourceError(
                "residency projection memory delta is invalid"
            )
        memory_delta = dict(sorted(memory_delta.items()))
        body = {
            "layout_generation": self.layout_generation,
            "memory_delta_bytes_by_resource": memory_delta,
            "predecessor_request_id": self.predecessor_request_id,
            "predecessor_ticket_id": self.predecessor_ticket_id,
            "ready_at_us": self.ready_at_us,
            "resource_ids": list(resources),
            "schema": "research-scheduler-residency-projection-token-v1",
            "target_geometry_sha256": self.target_geometry_sha256,
            "target_shards_sha256": self.target_shards_sha256,
            "transition_id": self.transition_id,
        }
        expected = canonical_sha256(body)
        if self.token_sha256 and self.token_sha256 != expected:
            raise RuntimeResourceError(
                "residency projection token hash differs"
            )
        object.__setattr__(self, "resource_ids", resources)
        object.__setattr__(
            self,
            "memory_delta_bytes_by_resource",
            MappingProxyType(memory_delta),
        )
        object.__setattr__(self, "token_sha256", expected)

    def to_json(self) -> dict[str, object]:
        return {
            "layout_generation": self.layout_generation,
            "memory_delta_bytes_by_resource": dict(
                self.memory_delta_bytes_by_resource
            ),
            "predecessor_request_id": self.predecessor_request_id,
            "predecessor_ticket_id": self.predecessor_ticket_id,
            "ready_at_us": self.ready_at_us,
            "resource_ids": list(self.resource_ids),
            "schema": "research-scheduler-residency-projection-token-v1",
            "target_geometry_sha256": self.target_geometry_sha256,
            "target_shards_sha256": self.target_shards_sha256,
            "token_sha256": self.token_sha256,
            "transition_id": self.transition_id,
        }


@dataclass(frozen=True)
class RuntimeMemoryReservation:
    token: str
    owner_id: str
    demand_id: str
    resource_id: str
    kind: str
    reserved_bytes: int
    lifetime: str
    replaced_bytes: int = 0
    share_key: str | None = None
    replacement_group: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "token",
            "owner_id",
            "demand_id",
            "resource_id",
            "kind",
            "lifetime",
        ):
            _text(f"memory reservation {name}", getattr(self, name))
        if (
            type(self.reserved_bytes) is not int
            or self.reserved_bytes < 0
            or type(self.replaced_bytes) is not int
            or self.replaced_bytes < 0
            or self.reserved_bytes + self.replaced_bytes <= 0
        ):
            raise RuntimeResourceError("memory reservation bytes are invalid")
        if self.share_key is not None:
            _text("memory reservation share key", self.share_key)
            if self.lifetime != "resident":
                raise RuntimeResourceError(
                    "only resident reservations can be shared"
                )
        if self.replacement_group is not None:
            _text(
                "memory reservation replacement group",
                self.replacement_group,
            )
            if self.share_key is None:
                raise RuntimeResourceError(
                    "replacement reservation requires a share key"
                )
        if self.replaced_bytes and self.replacement_group is None:
            raise RuntimeResourceError(
                "replaced reservation bytes require a replacement group"
            )

    def to_json(self) -> dict[str, int | str]:
        result = {
            "demand_id": self.demand_id,
            "kind": self.kind,
            "lifetime": self.lifetime,
            "owner_id": self.owner_id,
            "reserved_bytes": self.reserved_bytes,
            "resource_id": self.resource_id,
            "token": self.token,
        }
        if self.replaced_bytes:
            result["replaced_bytes"] = self.replaced_bytes
        if self.share_key is not None:
            result["share_key"] = self.share_key
        if self.replacement_group is not None:
            result["replacement_group"] = self.replacement_group
        return result


@dataclass(frozen=True)
class RuntimeMemoryCheckpoint:
    reservations: Mapping[str, RuntimeMemoryReservation]
    intervals: Mapping[str, tuple[int, int]]
    next_token: int


class RuntimeMemoryLedger:
    """Reserve all memory demands together against one live capacity view."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._reservations: dict[str, RuntimeMemoryReservation] = {}
        self._intervals: dict[str, tuple[int, int]] = {}
        self._next_token = 1
        self._fail_after_mutations: int | None = None

    def checkpoint(self) -> RuntimeMemoryCheckpoint:
        with self._lock:
            return RuntimeMemoryCheckpoint(
                reservations=dict(self._reservations),
                intervals=dict(self._intervals),
                next_token=self._next_token,
            )

    def restore(self, checkpoint: RuntimeMemoryCheckpoint) -> None:
        if not isinstance(checkpoint, RuntimeMemoryCheckpoint):
            raise RuntimeResourceError("memory checkpoint is invalid")
        reservations = dict(checkpoint.reservations)
        intervals = dict(checkpoint.intervals)
        if (
            set(reservations) != set(intervals)
            or any(
                type(value) is not tuple
                or len(value) != 2
                or type(value[0]) is not int
                or type(value[1]) is not int
                or value[0] < 0
                or value[1] <= value[0]
                for value in intervals.values()
            )
        ):
            raise RuntimeResourceError("memory checkpoint is invalid")
        with self._lock:
            self._reservations = reservations
            self._intervals = intervals
            self._next_token = checkpoint.next_token

    def fail_after_mutations_for_test(self, count: int | None) -> None:
        if count is not None and (type(count) is not int or count <= 0):
            raise RuntimeResourceError("fault mutation count must be positive")
        with self._lock:
            self._fail_after_mutations = count

    def _maybe_fail(self, mutations: int) -> None:
        if self._fail_after_mutations == mutations:
            self._fail_after_mutations = None
            raise RuntimeResourceError("injected memory reservation failure")

    @staticmethod
    def _reservation_interval(
        start_us: int,
        reserved_until_us: int | None,
    ) -> tuple[int, int]:
        end_us = (
            _MAX_RESERVATION_TIME_US
            if reserved_until_us is None else reserved_until_us
        )
        if (
            type(start_us) is not int
            or start_us < 0
            or type(end_us) is not int
            or end_us <= start_us
        ):
            raise RuntimeResourceError(
                "memory reservation interval is invalid"
            )
        return start_us, end_us

    def _overlaps(self, token: str, start_us: int, end_us: int) -> bool:
        existing_start_us, existing_end_us = self._intervals[token]
        return existing_start_us < end_us and start_us < existing_end_us

    def _reserved_by_resource(
        self,
        start_us: int,
        end_us: int,
        exclude_owner_id: str | None = None,
    ) -> dict[str, int]:
        result: dict[str, int] = {}
        shared: dict[tuple[str, str], int] = {}
        for token, reservation in self._reservations.items():
            if reservation.owner_id == exclude_owner_id:
                continue
            if not self._overlaps(token, start_us, end_us):
                continue
            if reservation.share_key is not None:
                key = (reservation.resource_id, reservation.share_key)
                shared[key] = max(
                    shared.get(key, 0), reservation.reserved_bytes
                )
                continue
            result[reservation.resource_id] = (
                result.get(reservation.resource_id, 0)
                + reservation.reserved_bytes
            )
        for (resource_id, _), amount in shared.items():
            result[resource_id] = result.get(resource_id, 0) + amount
        return result

    def _shared_by_resource_key(
        self,
        start_us: int,
        end_us: int,
        exclude_owner_id: str | None = None,
    ) -> dict[tuple[str, str], int]:
        result: dict[tuple[str, str], int] = {}
        for token, reservation in self._reservations.items():
            if reservation.owner_id == exclude_owner_id:
                continue
            if not self._overlaps(token, start_us, end_us):
                continue
            if reservation.share_key is None:
                continue
            key = (reservation.resource_id, reservation.share_key)
            result[key] = max(result.get(key, 0), reservation.reserved_bytes)
        return result

    def _replacement_share_keys_by_resource_group(
        self,
        start_us: int,
        end_us: int,
        exclude_owner_id: str | None = None,
    ) -> dict[tuple[str, str], tuple[str, ...]]:
        result: dict[tuple[str, str], set[str]] = {}
        for token, reservation in self._reservations.items():
            if reservation.owner_id == exclude_owner_id:
                continue
            if not self._overlaps(token, start_us, end_us):
                continue
            if reservation.replacement_group is None:
                continue
            key = (
                reservation.resource_id,
                reservation.replacement_group,
            )
            assert reservation.share_key is not None
            result.setdefault(key, set()).add(reservation.share_key)
        return {
            key: tuple(sorted(values)) for key, values in result.items()
        }

    def preview(
        self,
        demands: Sequence[RuntimeMemoryDemand],
        snapshot: RuntimePlacementSnapshot,
        *,
        start_us: int = 0,
        reserved_until_us: int | None = None,
        transitions: Sequence[RuntimeTransitionPlan] = (),
        residency: Sequence[ModelResidencyObservation] = (),
        exclusive_resource_by_device: Mapping[str | tuple[str, str], str] = MappingProxyType({}),
        exclude_owner_id: str | None = None,
        enforce_live_capacity: bool = True,
    ) -> Mapping[str, int]:
        rows = tuple(demands)
        if exclude_owner_id is not None:
            exclude_owner_id = _text(
                "memory preview excluded owner", exclude_owner_id
            )
        start_us, end_us = self._reservation_interval(
            start_us, reserved_until_us
        )
        with self._lock:
            reserved = self._reserved_by_resource(
                start_us, end_us, exclude_owner_id
            )
            existing_shared = self._shared_by_resource_key(
                start_us, end_us, exclude_owner_id
            )
            replacement_share_keys = (
                self._replacement_share_keys_by_resource_group(
                    start_us, end_us, exclude_owner_id
                )
            )
            return preview_runtime_memory(
                rows,
                snapshot,
                transitions=transitions,
                residency=residency,
                exclusive_resource_by_device=(
                    exclusive_resource_by_device
                ),
                reserved_by_resource=reserved,
                shared_by_resource_key=existing_shared,
                replacement_share_keys_by_resource_group=(
                    replacement_share_keys
                ),
                enforce_live_capacity=enforce_live_capacity,
            )

    def reserve(
        self,
        owner_id: str,
        demands: Sequence[RuntimeMemoryDemand],
        snapshot: RuntimePlacementSnapshot,
        *,
        start_us: int = 0,
        reserved_until_us: int | None = None,
        transitions: Sequence[RuntimeTransitionPlan] = (),
        residency: Sequence[ModelResidencyObservation] = (),
        exclusive_resource_by_device: Mapping[str | tuple[str, str], str] = MappingProxyType({}),
    ) -> tuple[RuntimeMemoryReservation, ...]:
        owner_id = _text("memory reservation owner", owner_id)
        rows = tuple(demands)
        start_us, end_us = self._reservation_interval(
            start_us, reserved_until_us
        )
        self.preview(
            rows,
            snapshot,
            start_us=start_us,
            reserved_until_us=end_us,
            transitions=transitions,
            residency=residency,
            exclusive_resource_by_device=exclusive_resource_by_device,
        )
        reservations = []
        added_tokens = []
        starting_token = self._next_token
        mutations = 0
        try:
            with self._lock:
                for demand in sorted(rows, key=lambda row: row.demand_id):
                    if (
                        demand.additional_bytes == 0
                        and demand.replaceable_bytes == 0
                    ):
                        continue
                    token = f"memory-{self._next_token}"
                    self._next_token += 1
                    reservation = RuntimeMemoryReservation(
                        token=token,
                        owner_id=owner_id,
                        demand_id=demand.demand_id,
                        resource_id=demand.resource_id,
                        kind=demand.kind,
                        reserved_bytes=demand.additional_bytes,
                        replaced_bytes=demand.replaceable_bytes,
                        lifetime=demand.lifetime,
                        share_key=demand.share_key,
                        replacement_group=demand.replacement_group,
                    )
                    self._reservations[token] = reservation
                    self._intervals[token] = (start_us, end_us)
                    reservations.append(reservation)
                    added_tokens.append(token)
                    mutations += 1
                    self._maybe_fail(mutations)
            return tuple(reservations)
        except BaseException:
            with self._lock:
                for token in added_tokens:
                    self._reservations.pop(token, None)
                    self._intervals.pop(token, None)
                self._next_token = starting_token
            raise

    def extend_owner(
        self, owner_id: str, reserved_until_us: int
    ) -> Mapping[str, int]:
        """Extend one owner's memory coverage without partial mutation."""
        owner_id = _text("memory extension owner", owner_id)
        if type(reserved_until_us) is not int or reserved_until_us <= 0:
            raise RuntimeResourceError(
                "memory extension end must be positive"
            )
        with self._lock:
            tokens = tuple(sorted(
                token for token, reservation in self._reservations.items()
                if reservation.owner_id == owner_id
            ))
            if not tokens:
                return MappingProxyType({})
            previous = {
                token: self._intervals[token][1] for token in tokens
            }
            updates = []
            for token in tokens:
                reservation = self._reservations[token]
                start_us, old_end_us = self._intervals[token]
                if reserved_until_us <= old_end_us:
                    continue
                for other_token, other in self._reservations.items():
                    if other.owner_id == owner_id:
                        continue
                    other_start_us, other_end_us = self._intervals[
                        other_token
                    ]
                    if (
                        reservation.resource_id != other.resource_id
                        or other_start_us >= reserved_until_us
                        or old_end_us >= other_end_us
                        or other_start_us < old_end_us
                        or (
                            reservation.share_key is not None
                            and reservation.share_key == other.share_key
                        )
                    ):
                        continue
                    raise RuntimeResourceError(
                        "memory extension overlaps a queued owner: "
                        + reservation.resource_id
                    )
                updates.append((token, start_us))
            for token, start_us in updates:
                self._intervals[token] = (start_us, reserved_until_us)
            return MappingProxyType(previous)

    def release_owner(self, owner_id: str) -> tuple[str, ...]:
        owner_id = _text("memory release owner", owner_id)
        released = []
        removed: dict[
            str, tuple[RuntimeMemoryReservation, tuple[int, int]]
        ] = {}
        mutations = 0
        try:
            with self._lock:
                tokens = sorted(
                    token
                    for token, reservation in self._reservations.items()
                    if reservation.owner_id == owner_id
                )
                for token in tokens:
                    removed[token] = (
                        self._reservations[token], self._intervals[token]
                    )
                    del self._reservations[token]
                    del self._intervals[token]
                    released.append(token)
                    mutations += 1
                    self._maybe_fail(mutations)
            return tuple(released)
        except BaseException:
            with self._lock:
                for token, (reservation, interval) in removed.items():
                    self._reservations[token] = reservation
                    self._intervals[token] = interval
            raise

    def owner_tokens(self, owner_id: str) -> tuple[str, ...]:
        """Return the live reservation tokens owned by one request."""
        owner_id = _text("memory reservation owner", owner_id)
        with self._lock:
            return tuple(sorted(
                token
                for token, reservation in self._reservations.items()
                if reservation.owner_id == owner_id
            ))

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            by_resource: dict[str, int] = {}
            for reservation in self._reservations.values():
                by_resource[reservation.resource_id] = (
                    by_resource.get(reservation.resource_id, 0)
                    + reservation.reserved_bytes
                )
            return {
                "by_resource_bytes": dict(sorted(by_resource.items())),
                "reservations": [
                    {
                        **self._reservations[token].to_json(),
                        "reserved_until_us": self._intervals[token][1],
                        "start_us": self._intervals[token][0],
                    }
                    for token in sorted(self._reservations)
                ],
                "schema": RUNTIME_MEMORY_LEDGER_SCHEMA,
            }
