"""Phone FFN configuration, physical events and receipts: receipts."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from types import MappingProxyType
from typing import Mapping

from ..._internal.runtime_plan import RuntimePhoneShard
from ..bridge import AndroidUsbRestorationReceipt, FunctionFsUsbObservation
from ..contracts import PhysicalAdapterError
from ..llama_server import PhoneFfnExecutionContract
from ..phone_transport import PhoneTransportContract
from .events import DirectPhoneFfnTerminalReceipt, PhoneResidencyCallEvent, PhoneResidencyPhaseEvent


@dataclass(frozen=True)
class PhoneFfnWeightSource:
    """The exact file and stored slice used by one physical session."""

    session_id: str
    session_generation: int
    weight_source: str
    source_path: str
    source_sha256: str
    index_sha256: str | None
    parent_artifact_sha256: str
    stored_layer_mask: int
    executed_layer_mask: int
    stored_columns: int
    active_columns: int
    bytes_loaded: int
    load_started_epoch_us: int | None = None
    load_finished_epoch_us: int | None = None
    verified_epoch_us: int | None = None
    ready_epoch_us: int | None = None

    def __post_init__(self) -> None:
        hashes = (
            self.source_sha256,
            self.parent_artifact_sha256,
            *((self.index_sha256,) if self.index_sha256 is not None else ()),
        )
        timestamps = (
            self.load_started_epoch_us,
            self.load_finished_epoch_us,
            self.verified_epoch_us,
            self.ready_epoch_us,
        )
        if (
            type(self.session_id) is not str
            or not self.session_id
            or not self.session_id.isascii()
            or type(self.session_generation) is not int
            or self.session_generation < 1
            or self.weight_source not in {"ffn_shard", "full_gguf"}
            or not re.fullmatch(r"/[A-Za-z0-9._:/-]+", self.source_path)
            or any(
                not isinstance(value, str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
                for value in hashes
            )
            or self.stored_layer_mask <= 0
            or self.executed_layer_mask <= 0
            or self.executed_layer_mask & ~self.stored_layer_mask
            or self.stored_columns <= 0
            or not 0 < self.active_columns <= self.stored_columns
            or self.bytes_loaded <= 0
            or any(
                value is not None
                and (type(value) is not int or value < 0)
                for value in timestamps
            )
            or (
                self.weight_source == "ffn_shard"
                and self.index_sha256 is None
            )
            or (
                self.weight_source == "full_gguf"
                and self.index_sha256 is not None
            )
        ):
            raise PhysicalAdapterError("phone FFN weight source is invalid")

    @staticmethod
    def _layers(mask: int) -> list[int]:
        return [index for index in range(64) if mask & (1 << index)]

    def to_json(self) -> dict[str, object]:
        return {
            "active_column_width": self.active_columns,
            "bytes_loaded": self.bytes_loaded,
            "executed_layer_indices": self._layers(
                self.executed_layer_mask
            ),
            "executed_layer_mask": f"{self.executed_layer_mask:016x}",
            "index_sha256": self.index_sha256,
            "load_finished_epoch_us": self.load_finished_epoch_us,
            "load_started_epoch_us": self.load_started_epoch_us,
            "parent_artifact_sha256": self.parent_artifact_sha256,
            "ready_epoch_us": self.ready_epoch_us,
            "schema": "s42-phone-ffn-weight-source-v1",
            "session_generation": self.session_generation,
            "session_id": self.session_id,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "stored_column_width": self.stored_columns,
            "stored_layer_indices": self._layers(self.stored_layer_mask),
            "stored_layer_mask": f"{self.stored_layer_mask:016x}",
            "verified_epoch_us": self.verified_epoch_us,
            "weight_source": self.weight_source,
        }


@dataclass(frozen=True)
class DirectPhoneFfnLaunchReceipt:
    ticket_id: str
    artifact_sha256: str
    session_id: str
    execution: PhoneFfnExecutionContract
    transport: PhoneTransportContract
    usb: FunctionFsUsbObservation
    remote_hashes: Mapping[str, str]
    usb_close_sha256: str
    phone_kernel_release: str
    diagnostic_interface: str | None = None
    phone_shards: tuple[RuntimePhoneShard, ...] = ()
    shard_manifest_sha256: str | None = None
    weight_sources: tuple[PhoneFfnWeightSource, ...] = ()
    cpu_affinity: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "remote_hashes",
            MappingProxyType(dict(sorted(self.remote_hashes.items()))),
        )
        if (
            not self.artifact_sha256.startswith("sha256:")
            or len(self.artifact_sha256) != 71
            or not self.usb_close_sha256.startswith("sha256:")
            or len(self.usb_close_sha256) != 71
            or type(self.phone_kernel_release) is not str
            or not self.phone_kernel_release
            or not self.phone_kernel_release.isascii()
            or (self.cpu_affinity is not None and (
                type(self.cpu_affinity) is not str
                or re.fullmatch(r"[0-9a-f]{1,16}", self.cpu_affinity) is None
                or int(self.cpu_affinity, 16) == 0
            ))
            or (
                self.diagnostic_interface is not None
                and (
                    type(self.diagnostic_interface) is not str
                    or not self.diagnostic_interface
                    or not self.diagnostic_interface.isascii()
                )
            )
            or any(
                not isinstance(row, RuntimePhoneShard)
                for row in self.phone_shards
            )
            or bool(self.phone_shards) != bool(
                self.shard_manifest_sha256
            )
            or (
                self.shard_manifest_sha256 is not None
                and (
                    not self.shard_manifest_sha256.startswith("sha256:")
                    or len(self.shard_manifest_sha256) != 71
                )
            )
            or any(
                not isinstance(row, PhoneFfnWeightSource)
                for row in self.weight_sources
            )
            or (
                self.phone_shards
                and {
                    (row.session_id, row.session_generation)
                    for row in self.phone_shards
                } != {
                    (row.session_id, row.session_generation)
                    for row in self.weight_sources
                }
            )
        ):
            raise PhysicalAdapterError("direct phone launch identity is invalid")

    def to_json(self) -> dict[str, object]:
        result = {
            "allocator": self.transport.allocator,
            "artifact_sha256": self.artifact_sha256,
            "columns": self.execution.columns,
            "diagnostic_interface": self.diagnostic_interface,
            "full_duplex": self.transport.full_duplex,
            "layer_mask": self.execution.layer_mask,
            "layers": self.execution.layers,
            "max_payload_bytes": self.transport.max_payload_bytes,
            "max_tokens": self.execution.max_tokens,
            "n_embd": self.execution.n_embd,
            "phone_kernel_release": self.phone_kernel_release,
            "queue_depth": self.transport.queue_depth,
            "qualification_identity_sha256": (
                self.transport.qualification_identity_sha256
            ),
            "remote_hashes": dict(self.remote_hashes),
            "session_id": self.session_id,
            "ticket_id": self.ticket_id,
            "transport_generation": self.transport.generation,
            "transport_profile_id": self.transport.profile_id,
            "usb": self.usb.to_json(),
            "usb_close_sha256": self.usb_close_sha256,
        }
        if self.phone_shards:
            result["phone_shards"] = [
                row.to_json() for row in self.phone_shards
            ]
            result["shard_manifest_sha256"] = (
                self.shard_manifest_sha256
            )
            result["weight_sources"] = [
                row.to_json() for row in self.weight_sources
            ]
        if self.cpu_affinity is not None:
            result["cpu_affinity"] = self.cpu_affinity
        return result


@dataclass(frozen=True)
class DirectPhoneFfnReconfigurationReceipt:
    ticket_id: str
    previous_manifest_sha256: str
    target_manifest_sha256: str
    changed_session_id: str
    load_count: int
    previous_load_count: int
    column_quantum: int
    previous_column_quantum: int | None
    max_tokens: int
    previous_max_tokens: int
    previous_shard: RuntimePhoneShard | None
    target_shard: RuntimePhoneShard | None
    target_shards: tuple[RuntimePhoneShard, ...]
    target_weight_source: PhoneFfnWeightSource | None = None

    def __post_init__(self) -> None:
        if (
            type(self.ticket_id) is not str
            or not self.ticket_id
            or not self.ticket_id.isascii()
            or not self.previous_manifest_sha256.startswith("sha256:")
            or len(self.previous_manifest_sha256) != 71
            or not self.target_manifest_sha256.startswith("sha256:")
            or len(self.target_manifest_sha256) != 71
            or type(self.changed_session_id) is not str
            or not self.changed_session_id
            or not self.changed_session_id.isascii()
            or type(self.load_count) is not int
            or self.load_count < 0
            or type(self.previous_load_count) is not int
            or self.previous_load_count < 0
            or type(self.column_quantum) is not int
            or self.column_quantum <= 0
            or (
                self.previous_column_quantum is not None
                and (
                    type(self.previous_column_quantum) is not int
                    or self.previous_column_quantum <= 0
                )
            )
            or type(self.max_tokens) is not int
            or self.max_tokens <= 0
            or type(self.previous_max_tokens) is not int
            or self.previous_max_tokens <= 0
            or (
                self.previous_shard is not None
                and not isinstance(self.previous_shard, RuntimePhoneShard)
            )
            or (
                self.previous_shard is not None
                and self.previous_column_quantum is None
            )
            or (
                self.previous_shard is not None
                and self.previous_column_quantum is not None
                and self.previous_shard.maximum_columns
                    % self.previous_column_quantum
            )
            or (
                self.previous_shard is None
                and self.previous_column_quantum is not None
            )
            or (
                self.target_shard is not None
                and not isinstance(self.target_shard, RuntimePhoneShard)
            )
            or (
                self.target_shard is not None
                and self.target_shard.maximum_columns
                    % self.column_quantum
            )
            or (
                self.previous_shard is None
                and self.target_shard is None
            )
            or not self.target_shards
            or len({
                row.session_id for row in self.target_shards
            }) != len(self.target_shards)
            or any(
                not isinstance(row, RuntimePhoneShard)
                for row in self.target_shards
            )
            or (
                self.target_shard is not None
                and self.target_shard not in self.target_shards
            )
            or (
                self.target_shard is None
                and self.changed_session_id in {
                    row.session_id for row in self.target_shards
                }
            )
            or (
                self.previous_shard is not None
                and self.previous_shard.session_id
                    != self.changed_session_id
            )
            or (
                self.target_shard is not None
                and self.target_shard.session_id
                    != self.changed_session_id
            )
            or (
                self.target_weight_source is not None
                and (
                    self.target_shard is None
                    or self.target_weight_source.session_id
                        != self.changed_session_id
                    or self.target_weight_source.session_generation
                        != self.target_shard.session_generation
                )
            )
        ):
            raise PhysicalAdapterError(
                "direct phone reconfiguration receipt is invalid"
            )

    def to_json(self) -> dict[str, object]:
        return {
            "changed_session_id": self.changed_session_id,
            "column_quantum": self.column_quantum,
            "kind": "partial_phone_residency_reconfiguration",
            "load_count": self.load_count,
            "max_tokens": self.max_tokens,
            "previous_load_count": self.previous_load_count,
            "previous_column_quantum": self.previous_column_quantum,
            "previous_max_tokens": self.previous_max_tokens,
            "previous_manifest_sha256": self.previous_manifest_sha256,
            "previous_shard": (
                None
                if self.previous_shard is None
                else self.previous_shard.to_json()
            ),
            "schema": "research-scheduler-phone-session-reconfigure-v2",
            "target_manifest_sha256": self.target_manifest_sha256,
            "target_shard": (
                None
                if self.target_shard is None
                else self.target_shard.to_json()
            ),
            "target_shards": [
                row.to_json() for row in self.target_shards
            ],
            "target_weight_source": (
                None
                if self.target_weight_source is None
                else self.target_weight_source.to_json()
            ),
            "ticket_id": self.ticket_id,
        }


@dataclass(frozen=True)
class DirectPhoneFfnCloseReceipt:
    launch: DirectPhoneFfnLaunchReceipt
    terminal: DirectPhoneFfnTerminalReceipt
    restoration: AndroidUsbRestorationReceipt
    bound_ticket_ids: tuple[str, ...] = ()
    historical_execution_proofs: tuple[Mapping[str, object], ...] = ()
    residency_phase_events: tuple[PhoneResidencyPhaseEvent, ...] = ()
    residency_call_events: tuple[PhoneResidencyCallEvent, ...] = ()

    def __post_init__(self) -> None:
        ticket_ids = self.bound_ticket_ids or (self.launch.ticket_id,)
        if (
            ticket_ids[0] != self.launch.ticket_id
            or len(ticket_ids) != len(set(ticket_ids))
            or any(
                type(ticket_id) is not str
                or not ticket_id
                or not ticket_id.isascii()
                for ticket_id in ticket_ids
            )
            or any(
                not isinstance(row, PhoneResidencyPhaseEvent)
                for row in self.residency_phase_events
            )
            or any(
                not isinstance(row, PhoneResidencyCallEvent)
                for row in self.residency_call_events
            )
        ):
            raise PhysicalAdapterError(
                "direct phone bound ticket identity is invalid"
            )
        object.__setattr__(self, "bound_ticket_ids", ticket_ids)

    def to_json(self) -> dict[str, object]:
        result = {
            "bound_ticket_ids": list(self.bound_ticket_ids),
            "launch": self.launch.to_json(),
            "restoration": self.restoration.to_json(),
            "terminal": self.terminal.to_json(),
        }
        if self.historical_execution_proofs:
            result["historical_execution_proofs"] = [
                dict(row) for row in self.historical_execution_proofs
            ]
        if self.residency_phase_events:
            result["residency_phase_events"] = [
                row.to_json() for row in self.residency_phase_events
            ]
        if self.residency_call_events:
            result["residency_call_events"] = [
                row.to_json() for row in self.residency_call_events
            ]
        return result


@dataclass(frozen=True)
class DirectPhoneFfnPreflightReceipt:
    remote_hashes: Mapping[str, str]
    usb_close_sha256: str
    restoration: AndroidUsbRestorationReceipt
    phone_kernel_release: str
    ffn_shard_indexes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "remote_hashes",
            MappingProxyType(dict(sorted(self.remote_hashes.items()))),
        )
        object.__setattr__(
            self,
            "ffn_shard_indexes",
            MappingProxyType(dict(sorted(self.ffn_shard_indexes.items()))),
        )
        if (
            not self.usb_close_sha256.startswith("sha256:")
            or len(self.usb_close_sha256) != 71
            or type(self.phone_kernel_release) is not str
            or not self.phone_kernel_release
            or not self.phone_kernel_release.isascii()
            or any(
                re.fullmatch(r"sha256:[0-9a-f]{64}", artifact) is None
                or re.fullmatch(r"sha256:[0-9a-f]{64}", index) is None
                for artifact, index in self.ffn_shard_indexes.items()
            )
        ):
            raise PhysicalAdapterError(
                "direct phone preflight close hash is invalid"
            )

    def to_json(self) -> dict[str, object]:
        return {
            "remote_hashes": dict(self.remote_hashes),
            "ffn_shard_indexes": dict(self.ffn_shard_indexes),
            "phone_kernel_release": self.phone_kernel_release,
            "restoration": self.restoration.to_json(),
            "usb_close_sha256": self.usb_close_sha256,
        }
